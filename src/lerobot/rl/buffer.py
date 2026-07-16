#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import hashlib
import json
import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as F_vision
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, REWARD
from lerobot.utils.transition import Transition

logger = logging.getLogger(__name__)

IMAGE_STORAGE_DTYPE_BFLOAT16 = "bfloat16"
IMAGE_STORAGE_DTYPE_UINT8 = "uint8"
CACHE_SCHEMA_VERSION = 2


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


def done_list_ids_from_subtask_indices(
    subtask_indices: torch.Tensor,  # (T,)
    episode_ends: torch.Tensor,  # (T,) bool
    cap: int,
) -> torch.Tensor:
    """Per-frame done-list: the subtasks completed so far in the episode, oldest → newest.

    Append-on-switch with consecutive dedup: when subtask_index changes A→B at
    frame t, A is done from frame t onward. A subtask that recurs later appends
    again (this is execution history, not a set). Frames with index -1
    (unannotated) neither append nor become current. The list resets after
    episode-end frames and keeps the newest `cap` entries, padded with -1.

    Returns (T, cap) long.
    """
    indices = [int(i) for i in subtask_indices.tolist()]
    ends = episode_ends.tolist()
    out = torch.full((len(indices), cap), -1, dtype=torch.long)
    completed: list[int] = []
    prev = -1
    for t, cur in enumerate(indices):
        if cur != -1:
            if prev != -1 and cur != prev:
                completed.append(prev)
            prev = cur
        tail = completed[-cap:]
        if tail:
            out[t, : len(tail)] = torch.tensor(tail, dtype=torch.long)
        if ends[t]:
            completed = []
            prev = -1
    return out


def assemble_history_windows(
    entries: Sequence[dict[str, torch.Tensor]],
    history_offsets: dict[str, list[int]],
    current_state: dict[str, torch.Tensor],
    action_dim: int,
) -> dict[str, torch.Tensor]:
    """Actor-side mirror of ReplayBuffer._gather_history.

    entries are the completed (state, action) steps of the current episode,
    oldest → newest, each value shaped (1, ...); the frame at lookback distance
    k is entries[-k]. Slots reaching past the oldest entry repeat it and are
    flagged True in the pad mask — same clamp + mask semantics as the
    learner-side gather. With no entries at all, state slots repeat the current
    state and action slots are zeros (padded either way).

    Returns {"history.{key}": (1, T_h, ...), "history.{key}_is_pad": (1, T_h) bool}.
    """
    reach = len(entries)
    windows = {}
    for key, offsets in history_offsets.items():
        slots = []
        for k in offsets:  # oldest → newest (offsets normalized descending)
            distance = min(k, reach)
            if distance > 0:
                value = entries[-distance][key]
            elif key == ACTION:
                value = torch.zeros(1, action_dim)
            else:
                value = current_state[key]
            slots.append(value)
        windows[f"history.{key}"] = torch.stack(slots, dim=1)
        windows[f"history.{key}_is_pad"] = torch.tensor([[k > reach for k in offsets]])
    return windows


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = True,
        reward_normalization_constant: float = 1.0,
        terminal_failure_reward: float = -1.0,
        image_storage_dtype: str = IMAGE_STORAGE_DTYPE_BFLOAT16,
        image_storage_size: tuple[int, int] | None = None,
        history_offsets: dict[str, list[int]] | None = None,
    ):
        """
        Replay buffer for storing transitions.
        It will allocate tensors on the specified device, when the first transition is added.
        NOTE: If you encounter memory issues, you can try to use the `optimize_memory` flag to save memory or
        and use the `storage_device` flag to store the buffer on a different device.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (list[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Callable | None): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
            image_storage_dtype: Dtype used for image observations in storage. "uint8" stores raw
                camera-like bytes and converts float [0, 1] inputs to [0, 255]; "bfloat16" preserves the
                previous normalized-float storage behavior.
            image_storage_size: Optional (height, width) resize applied before image storage. None keeps
                the input resolution.
            history_offsets: Optional map of state key → lookback distances in buffer steps
                (e.g. {"observation.state": [30, 60, 90, 120]}). When set, sample() emits
                "history.{key}" windows (oldest → newest) plus "history.{key}_is_pad" masks.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory
        self.reward_normalization_constant = reward_normalization_constant
        self.terminal_failure_reward = terminal_failure_reward
        self.image_storage_dtype = self._normalize_image_storage_dtype(image_storage_dtype)
        self.image_storage_size = self._normalize_image_storage_size(image_storage_size)
        # Image/depth rows kept every N-th frame (memmap caches only; see from_cache).
        # Low-dim arrays stay dense, so action chunks / dones / history are unaffected.
        self.image_stride = 1
        self.history_offsets = self._normalize_history_offsets(history_offsets)
        self._lock = threading.Lock()

        # Track episode boundaries for memory optimization
        self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq


    @staticmethod
    def _normalize_image_storage_dtype(dtype: str) -> str:
        normalized = str(dtype).lower()
        if normalized in {"bf16", "torch.bfloat16"}:
            normalized = IMAGE_STORAGE_DTYPE_BFLOAT16
        if normalized not in {IMAGE_STORAGE_DTYPE_BFLOAT16, IMAGE_STORAGE_DTYPE_UINT8}:
            raise ValueError(
                f"Unsupported image_storage_dtype={dtype!r}. "
                f"Expected '{IMAGE_STORAGE_DTYPE_UINT8}' or '{IMAGE_STORAGE_DTYPE_BFLOAT16}'."
            )
        return normalized

    @staticmethod
    def _normalize_image_storage_size(size: tuple[int, int] | Sequence[int] | None) -> tuple[int, int] | None:
        if size is None:
            return None
        if len(size) != 2:
            raise ValueError(f"image_storage_size must be (height, width) or None, got {size!r}")
        height, width = int(size[0]), int(size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"image_storage_size values must be positive, got {size!r}")
        return height, width

    @staticmethod
    def _normalize_history_offsets(
        history_offsets: dict[str, list[int]] | None,
    ) -> dict[str, list[int]] | None:
        if not history_offsets:
            return None
        normalized = {}
        for key, offsets in history_offsets.items():
            offsets = sorted({int(o) for o in offsets}, reverse=True)  # oldest → newest
            if not offsets or offsets[-1] <= 0:
                raise ValueError(f"history_offsets[{key!r}] must be positive lookback distances, got {offsets}")
            normalized[key] = offsets
        return normalized

    @staticmethod
    def _is_image_key(key: str) -> bool:
        return key.startswith(OBS_IMAGE) or "images" in key

    def _prepare_image_for_storage(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_storage_size is not None and tensor.shape[-2:] != self.image_storage_size:
            tensor = F_vision.resize(tensor, self.image_storage_size)

        if self.image_storage_dtype == IMAGE_STORAGE_DTYPE_UINT8:
            if tensor.dtype == torch.uint8:
                return tensor.to(device=self.storage_device)
            if tensor.is_floating_point():
                # LeRobot images are normally normalized to [0, 1]. Actor paths can occasionally
                # hand us byte-scale floats, so preserve those instead of multiplying twice.
                max_value = float(tensor.detach().max().item()) if tensor.numel() else 1.0
                if max_value <= 1.0:
                    tensor = tensor * 255.0
                return tensor.round().clamp(0, 255).to(dtype=torch.uint8, device=self.storage_device)
            return tensor.clamp(0, 255).to(dtype=torch.uint8, device=self.storage_device)

        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        return tensor.to(dtype=torch.bfloat16, device=self.storage_device)

    def _prepare_tensor_for_storage(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if self._is_image_key(key):
            return self._prepare_image_for_storage(tensor)
        return tensor.to(dtype=torch.bfloat16, device=self.storage_device)

    def _prepare_state_for_storage(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: self._prepare_tensor_for_storage(key, value) for key, value in state.items()}

    def _prepare_action_for_storage(self, action: torch.Tensor) -> torch.Tensor:
        return action.to(dtype=torch.bfloat16, device=self.storage_device)

    def _prepare_complementary_info_for_storage(
        self, complementary_info: dict[str, torch.Tensor] | None
    ) -> dict[str, torch.Tensor | float | int] | None:
        if complementary_info is None:
            return None
        prepared = {}
        for key, value in complementary_info.items():
            if isinstance(value, torch.Tensor):
                # Depth rides through complementary_info but must stay lossless: bf16 has only
                # 8 mantissa bits and mangles raw uint16 depth. Keep its native dtype, mirroring
                # the offline memmap cache (which keeps depth raw uint16, no bf16 view).
                dtype = value.dtype if key.startswith("depth.") else torch.bfloat16
                prepared[key] = value.to(dtype=dtype, device=self.storage_device)
            else:
                prepared[key] = value
        return prepared

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.empty(
                (self.capacity, *shape), dtype=state[key].dtype, device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), dtype=torch.bfloat16, device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), dtype=torch.bfloat16, device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.empty((self.capacity, *shape), dtype=state[key].dtype, device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # Pre-allocate tensors for each key in complementary_info
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    # Depth keeps its native (uint16) dtype — lossless, see
                    # _prepare_complementary_info_for_storage. Everything else stays bf16.
                    dtype = value.dtype if key.startswith("depth.") else torch.bfloat16
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), dtype=dtype, device=self.storage_device
                    )
                elif isinstance(value, (int | float)):
                    # Handle scalar values similar to reward
                    self.complementary_info[key] = torch.empty((self.capacity,), dtype=torch.bfloat16, device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def __len__(self):
        return self.size

    def reset(self) -> None:
        """Clear the buffer without deallocating pre-allocated tensors."""
        with self._lock:
            self.position = 0
            self.size = 0

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor] | None,
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Saves a transition, ensuring tensors are stored on the designated storage device."""
        with self._lock:
            state = self._prepare_state_for_storage(state)
            action = self._prepare_action_for_storage(action)
            complementary_info = self._prepare_complementary_info_for_storage(complementary_info)
            if not self.optimize_memory:
                if next_state is None:
                    raise ValueError("next_state must be provided when optimize_memory=False")
                next_state = self._prepare_state_for_storage(next_state)

            # Initialize storage if this is the first transition
            if not self.initialized:
                self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

            # Store the transition in pre-allocated tensors
            for key in self.states:
                self.states[key][self.position].copy_(state[key].squeeze(dim=0))

                if not self.optimize_memory:
                    # Only store next_states if not optimizing memory
                    self.next_states[key][self.position].copy_(next_state[key].squeeze(dim=0))

            self.actions[self.position].copy_(action.squeeze(0))
            self.rewards[self.position] = reward
            self.dones[self.position] = done
            self.truncateds[self.position] = truncated

            # Handle complementary_info if provided and storage is initialized
            if complementary_info is not None and self.has_complementary_info:
                for key in self.complementary_info_keys:
                    if key in complementary_info:
                        value = complementary_info[key]
                        if isinstance(value, torch.Tensor):
                            self.complementary_info[key][self.position].copy_(value.squeeze(dim=0))
                        elif isinstance(value, (int | float)):
                            self.complementary_info[key][self.position] = value

            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def materialize_done_lists(self, cap: int) -> None:
        """Derive complementary_info["done_list_ids"] (capacity, cap) from the per-frame
        subtask_index column. No-op when the buffer carries no subtask annotations.
        Call AFTER any subtask vocabulary remapping — the stored ids are final.
        """
        if not self.has_complementary_info or "subtask_index" not in self.complementary_info:
            return
        n = self.size
        subtask_indices = self.complementary_info["subtask_index"][:n].reshape(n)
        episode_ends = (self.dones[:n] | self.truncateds[:n]).reshape(n)
        done_ids = torch.full((self.capacity, cap), -1, dtype=torch.long, device=self.storage_device)
        done_ids[:n] = done_list_ids_from_subtask_indices(subtask_indices, episode_ends, cap)
        self.complementary_info["done_list_ids"] = done_ids
        if "done_list_ids" not in self.complementary_info_keys:
            self.complementary_info_keys.append("done_list_ids")

    def materialize_metadata(
        self,
        quality: int,
        mistake: bool,
        speed_bucket_steps: int,
    ) -> None:
        """π0.7-style episode metadata as complementary columns (offline buffers).

        quality/mistake are dataset-level defaults (curated demos: 5 / False);
        speed is the per-episode length bucketed by `speed_bucket_steps`,
        broadcast to every frame of the episode.
        """
        n = self.size
        ends = (self.dones[:n] | self.truncateds[:n]).reshape(n).tolist()
        speed = torch.full((self.capacity,), -1.0, dtype=torch.bfloat16, device=self.storage_device)
        episode_start = 0
        for t, end in enumerate(ends):
            if end or t == n - 1:
                length = t - episode_start + 1
                speed[episode_start : t + 1] = float(length // speed_bucket_steps)
                episode_start = t + 1
        self.complementary_info["metadata_quality"] = torch.full(
            (self.capacity,), float(quality), dtype=torch.bfloat16, device=self.storage_device
        )
        self.complementary_info["metadata_mistake"] = torch.full(
            (self.capacity,), float(mistake), dtype=torch.bfloat16, device=self.storage_device
        )
        self.complementary_info["metadata_speed"] = speed
        for key in ("metadata_quality", "metadata_mistake", "metadata_speed"):
            if key not in self.complementary_info_keys:
                self.complementary_info_keys.append(key)
        self.has_complementary_info = True

    def _gather_history(self, idx: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Assemble lookback windows for the sampled indices.

        Mirror of the next_state index math in sample(): history at distance k is
        states[idx - k]. A slot is invalid when the lookback crosses an episode
        boundary (done/truncated between idx-k and idx-1) or reaches past the
        oldest stored frame (index 0 while filling; the write head once the
        circular buffer is full). Invalid slots repeat the earliest valid frame
        of the current episode and are flagged True in the pad mask.

        The key "action" gathers from self.actions: every fill path stores one
        executed action per frame, so actions[idx - k] is what the robot did k
        steps ago. Keys starting with "depth." gather from complementary_info,
        where per-frame raw depth lives (uint16, preserved).

        Returns ({key: (B, T_h, ...)}, {key: (B, T_h) bool}), windows oldest → newest.
        """
        device = idx.device
        max_back = max(offsets[0] for offsets in self.history_offsets.values())
        back = torch.arange(1, max_back + 1, device=device)  # (M,)
        back_idx = (idx.unsqueeze(1) - back) % self.capacity  # (B, M)
        boundary = self.dones[back_idx] | self.truncateds[back_idx]
        crossed = torch.cummax(boundary.int(), dim=1).values.bool()  # (B, M)

        # Oldest stored frame: index 0 while filling, the write head once the circular buffer is full.
        oldest_distance = idx if self.size < self.capacity else (idx - self.position) % self.capacity
        invalid = crossed | (back.unsqueeze(0) > oldest_distance.unsqueeze(1))  # (B, M)
        reach = (~invalid).int().cumprod(dim=1).sum(dim=1)  # (B,) farthest usable lookback

        history = {}
        pad = {}
        for key, offsets in self.history_offsets.items():
            offs = torch.tensor(offsets, device=device)  # (T_h,) oldest → newest
            dist = torch.minimum(offs.unsqueeze(0), reach.unsqueeze(1))  # (B, T_h)
            gather_idx = (idx.unsqueeze(1) - dist) % self.capacity
            if key == ACTION:
                source = self.actions
            elif key.startswith("depth."):
                source = self.complementary_info[key]
            else:
                source = self.states[key]
            history[key] = source[gather_idx]
            pad[key] = offs.unsqueeze(0) > reach.unsqueeze(1)
        return history, pad

    def sample(self, batch_size: int, action_chunk_size: int = 50) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        with self._lock:
            batch_size = min(batch_size, self.size)
            high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size

            stride = self.image_stride
            if stride > 1 and action_chunk_size % stride != 0:
                raise ValueError(
                    f"action_chunk_size={action_chunk_size} must be a multiple of "
                    f"image_stride={stride} so idx + chunk lands on a stored image row."
                )

            valid_indices = []
            collected_count = 0
            max_retries = 200  # Safety hatch to prevent infinite loops
            attempts = 0

            # This while loop is to ensure frames are valid
            while collected_count < batch_size:
                if attempts >= max_retries:
                    raise RuntimeError(
                        "Failed to sample enough valid chunks. Is action_chunk_size larger than your episodes?"
                    )
                attempts += 1

                # Only sample enough to fill the remaining needed batch size (times a safety factor)
                remaining = batch_size - collected_count
                if stride > 1:
                    # Images/depth exist only every stride-th frame: draw image-aligned
                    # starts so observations are exact (image row = idx // stride).
                    idx = stride * torch.randint(
                        low=0, high=(high + stride - 1) // stride, size=(4 * remaining,), device=self.storage_device
                    )
                else:
                    idx = torch.randint(low=0, high=high, size=(4 * remaining,), device=self.storage_device)

                if len(self.actions.shape) == 2 and action_chunk_size > 1:
                    # Use action_chunk_size - 1 if you don't want to check the final step's done flag
                    check_length = action_chunk_size - 1
                    chunk_indices = (
                        idx.unsqueeze(1) + torch.arange(check_length, device=self.storage_device)
                    ) % self.capacity

                    chunk_dones = self.dones[chunk_indices]
                    invalid_mask = chunk_dones.any(dim=1)
                    idx = idx[~invalid_mask]

                if len(idx) > 0:
                    valid_indices.append(idx)
                    collected_count += len(idx)

            # Concatenate all collected indices and slice exactly to batch_size
            idx = torch.cat(valid_indices)[:batch_size]

            # Identify image keys that need augmentation
            image_keys = [k for k in self.states if k.startswith(OBS_IMAGE)] if self.use_drq else []

            # Create batched state and next_state
            batch_state = {}
            batch_next_state = {}

            # First pass: load all state tensors to target device.
            # Image tensors hold one row per stride-th frame, so their row index is
            # idx // stride (exact: sampled idx and idx + chunk are stride-aligned).
            for key in self.states:
                row = idx // stride if self._is_image_key(key) else idx
                state_arr = self.states[key][row]
                batch_state[key] = state_arr.to(self.device)

                if not self.optimize_memory:
                    # Standard approach - load next_states directly
                    batch_next_state[key] = self.next_states[key][row].to(self.device)
                else:
                    # Memory-optimized approach - get next_state from the next index
                    next_idx = (idx + action_chunk_size) % self.capacity
                    next_row = next_idx // stride if self._is_image_key(key) else next_idx
                    batch_next_state[key] = self.states[key][next_row].to(self.device)

            # Short-term memory: lookback windows for the configured keys
            if self.history_offsets is not None:
                history, pad = self._gather_history(idx)
                for key in history:
                    batch_state[f"history.{key}"] = history[key].to(self.device)
                    batch_state[f"history.{key}_is_pad"] = pad[key].to(self.device)

            # Sample actions - handle both pre-chunked and single actions
            # Check if actions are already chunked (offline buffer: shape (N, chunk_size, action_dim))
            # or single actions that need chunking (online buffer: shape (N, action_dim))
            if len(self.actions.shape) == 3:
                batch_actions = self.actions[idx].to(self.device)
            elif action_chunk_size == 1:
                batch_actions = self.actions[idx].to(self.device)
            else:  # len(self.actions.shape) == 2 and action_chunk_size > 1
                chunk_indices = (
                    idx.unsqueeze(1) + torch.arange(action_chunk_size, device=self.storage_device)
                ) % self.capacity
                batch_actions = self.actions[chunk_indices].to(self.device)

            # Handle rewards and dones with lookahead awareness
            if len(self.actions.shape) == 2 and action_chunk_size > 1:
                lookahead_window = (
                    idx.unsqueeze(1) + torch.arange(action_chunk_size, device=self.storage_device)
                ) % self.capacity
                lookahead_window = lookahead_window + action_chunk_size
                lookahead_window = torch.clamp(lookahead_window, max=self.size - 1)

                batch_rewards = self.rewards[lookahead_window].max(dim=1)[0].to(self.device)
                batch_dones = self.dones[lookahead_window].any(dim=1).float().to(self.device)
            else:
                batch_rewards = self.rewards[idx].to(self.device)
                batch_dones = self.dones[idx].to(self.device).float()

            batch_truncateds = self.truncateds[idx].to(self.device).float()

            # Sample complementary_info if available
            batch_complementary_info = None
            if self.has_complementary_info:
                batch_complementary_info = {}
                # Depth for the critic target V(s'): next_state is derived at sample time
                # (optimize_memory), so its depth is the same column at the same next_idx.
                # Emitted as next_depth.{key} so online/offline batches concatenate symmetrically.
                depth_next_idx = (
                    (idx + action_chunk_size) % self.capacity if self.optimize_memory else None
                )
                for key in self.complementary_info_keys:
                    row = idx // stride if key.startswith("depth.") else idx
                    batch_complementary_info[key] = self.complementary_info[key][row].to(self.device)
                    if depth_next_idx is not None and key.startswith("depth."):
                        batch_complementary_info[f"next_{key}"] = self.complementary_info[key][
                            depth_next_idx // stride
                        ].to(self.device)

        # Image augmentation operates only on local batch_state/batch_next_state
        # tensors -- safe to do outside the lock.
        if self.use_drq and image_keys:
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            all_images_tensor = torch.cat(all_images, dim=0)
            if not all_images_tensor.is_floating_point():
                all_images_tensor = all_images_tensor.to(torch.float32) / 255.0
            augmented_images = self.image_augmentation_function(all_images_tensor)

            for i, key in enumerate(image_keys):
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # Apply reward transformation outside the lock (works on local batch tensors)
        new_rewards = torch.full_like(batch_rewards, -1.0)
        # Success case: Done and Reward=1 -> 0
        success_mask = (batch_dones > 0.5) & (batch_rewards > 0.5)
        new_rewards[success_mask] = 0.0
        # Failure case: Done and Reward=0 -> terminal_failure_reward
        failure_mask = (batch_dones > 0.5) & (batch_rewards < 0.5)
        new_rewards[failure_mask] = self.terminal_failure_reward
        batch_rewards = new_rewards / self.reward_normalization_constant

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
        action_chunk_size: int = 50,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)
            action_chunk_size (int): Number of future actions to sample (default: 50)

        Yields:
            BatchTransition: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size, action_chunk_size=action_chunk_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size, action_chunk_size=action_chunk_size)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2, action_chunk_size: int = 50):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.
            action_chunk_size (int): Number of future actions to sample.

        Yields:
            BatchTransition: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size, action_chunk_size=action_chunk_size)
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2, action_chunk_size: int = 50):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            queue_size (int): Number of initial batches to prefetch
            action_chunk_size (int): Number of future actions to sample

        Yields:
            BatchTransition: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, action_chunk_size=action_chunk_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_cache(
        cls,
        cache_dir: str | Path,
        device: str = "cuda:0",
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        reward_normalization_constant: float = 1.0,
        terminal_failure_reward: float = -1.0,
        inject_complementary_info: dict | None = None,
        history_offsets: dict[str, list[int]] | None = None,
    ) -> "ReplayBuffer":
        """
        Load a ReplayBuffer from pre-decoded memmap cache files.

        Image data stays memory-mapped (OS pages in on demand), while
        non-image data is loaded into RAM (small enough to fit easily).
        """
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {cache_dir}")

        with open(meta_path) as f:
            meta = json.load(f)

        num_transitions = meta["num_transitions"]
        state_keys = meta["state_keys"]
        image_keys = meta["image_keys"]
        non_image_state_keys = meta["non_image_state_keys"]
        image_storage_dtype = meta.get("image_storage_dtype", IMAGE_STORAGE_DTYPE_BFLOAT16)
        image_storage_size = meta.get("image_storage_size", meta.get("image_size", [224, 224]))
        image_stride = int(meta.get("image_stride", 1))
        # Builder writes image/depth rows for frames 0, stride, 2*stride, ...
        image_rows = (num_transitions + image_stride - 1) // image_stride

        if image_stride > 1 and history_offsets:
            strided_keys = [
                k for k in history_offsets if cls._is_image_key(k) or k.startswith("depth.")
            ]
            if strided_keys:
                raise ValueError(
                    f"history_offsets on image/depth keys {strided_keys} are unsupported with "
                    f"image_stride={image_stride}: lookback rows are not stored."
                )

        logger.info(
            f"Loading buffer cache from {cache_dir} ({num_transitions} transitions, image_stride={image_stride})"
        )

        replay_buffer = cls(
            capacity=num_transitions,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device="cpu",
            optimize_memory=True,
            reward_normalization_constant=reward_normalization_constant,
            terminal_failure_reward=terminal_failure_reward,
            image_storage_dtype=image_storage_dtype,
            image_storage_size=image_storage_size,
            history_offsets=history_offsets,
        )

        def _sanitize(key: str) -> str:
            return key.replace("/", "_")

        def _load_memmap(
            key: str, shape: tuple, as_torch_dtype: torch.dtype | None = None, rows: int | None = None
        ) -> torch.Tensor:
            safe_key = _sanitize(key)
            bin_path = cache_dir / f"{safe_key}.bin"
            dtype_str = meta["dtypes"][safe_key]
            np_dtype = np.dtype(dtype_str)
            count = num_transitions if rows is None else rows
            full_shape = tuple([count] + meta["shapes"][safe_key]) if meta["shapes"][safe_key] else (count,)
            mm = np.memmap(str(bin_path), dtype=np_dtype, mode="c", shape=full_shape)
            t = torch.from_numpy(mm)
            if as_torch_dtype is not None and np_dtype == np.uint16:
                t = t.view(as_torch_dtype)
            return t

        def _load_small(key: str, clone: bool = True, as_torch_dtype: torch.dtype | None = None) -> torch.Tensor:
            t = _load_memmap(key, (), as_torch_dtype=as_torch_dtype)
            return t.clone() if clone else t

        def _bf16_view_if_uint16(key: str) -> torch.dtype | None:
            return torch.bfloat16 if np.dtype(meta["dtypes"][_sanitize(key)]) == np.uint16 else None

        # Image keys: keep as memmap-backed tensors. uint8 caches load as uint8; old bf16
        # caches are stored as uint16 and viewed back as torch.bfloat16.
        replay_buffer.states = {}
        for key in image_keys:
            replay_buffer.states[key] = _load_memmap(
                key, (), as_torch_dtype=_bf16_view_if_uint16(key), rows=image_rows
            )
            logger.info(f"  {key}: memmap {replay_buffer.states[key].shape} {replay_buffer.states[key].dtype}")

        # Non-image state: small, clone into RAM
        for key in non_image_state_keys:
            replay_buffer.states[key] = _load_small(key, clone=True, as_torch_dtype=torch.bfloat16)
            logger.info(f"  {key}: RAM {replay_buffer.states[key].shape}")

        # optimize_memory=True: next_states is just a reference
        replay_buffer.next_states = replay_buffer.states

        # Actions, rewards, dones -- small, load into RAM
        replay_buffer.actions = _load_small("actions", clone=True, as_torch_dtype=torch.bfloat16)
        replay_buffer.rewards = _load_small("rewards", clone=True, as_torch_dtype=torch.bfloat16)
        replay_buffer.dones = _load_small("dones", clone=True).to(torch.bool)
        replay_buffer.truncateds = _load_small("truncateds", clone=True).to(torch.bool)
        replay_buffer.episode_ends = _load_small("episode_ends", clone=True).to(torch.bool)

        # Complementary info
        comp_keys = meta.get("complementary_info_keys", [])
        has_golden = meta.get("inject_golden", False)

        # Caller-side override: if inject_complementary_info explicitly sets
        # is_golden, honor it over the cache's baked-in flag.
        caller_wants_golden: bool | None = None
        if inject_complementary_info is not None and "is_golden" in inject_complementary_info:
            caller_wants_golden = bool(inject_complementary_info["is_golden"])

        all_comp_keys = []
        for k in comp_keys:
            if k == "is_golden" and caller_wants_golden is False:
                continue
            safe = _sanitize(f"complementary_info.{k}")
            if (cache_dir / f"{safe}.bin").exists():
                all_comp_keys.append(k)
        load_golden = has_golden if caller_wants_golden is None else caller_wants_golden
        if load_golden and "is_golden" not in all_comp_keys:
            safe = _sanitize("complementary_info.is_golden")
            if (cache_dir / f"{safe}.bin").exists():
                all_comp_keys.append("is_golden")
            elif caller_wants_golden is True:
                logger.warning(
                    "treat_main_dataset_as_golden=True but cache has no is_golden column; "
                    "proceeding without golden bypass. Rebuild the cache with --inject-golden to enable."
                )

        replay_buffer.has_complementary_info = len(all_comp_keys) > 0
        replay_buffer.complementary_info_keys = list(all_comp_keys)
        replay_buffer.complementary_info = {}

        for k in all_comp_keys:
            full_key = f"complementary_info.{k}"
            replay_buffer.complementary_info[k] = _load_small(
                full_key, clone=True, as_torch_dtype=torch.bfloat16
            )
            logger.info(f"  complementary_info.{k}: {replay_buffer.complementary_info[k].shape}")

        # Depth sidecar: surfaced through complementary_info but kept RAW uint16 (no bf16 view) and
        # memmap-backed (large, like images). The point-map builder consumes raw metric depth downstream.
        for k in meta.get("depth_keys", []):
            full_key = f"depth.{k}"
            if not (cache_dir / f"{_sanitize(full_key)}.bin").exists():
                continue
            replay_buffer.complementary_info[full_key] = _load_memmap(
                full_key, (), as_torch_dtype=None, rows=image_rows
            )
            replay_buffer.complementary_info_keys.append(full_key)
            replay_buffer.has_complementary_info = True
            logger.info(
                f"  depth {full_key}: memmap {replay_buffer.complementary_info[full_key].shape} "
                f"{replay_buffer.complementary_info[full_key].dtype}"
            )

        replay_buffer.image_stride = image_stride
        replay_buffer.size = num_transitions
        replay_buffer.position = num_transitions % replay_buffer.capacity
        replay_buffer.initialized = True

        logger.info(f"Buffer loaded from cache: {num_transitions} transitions, memmap images, RAM non-image data")
        return replay_buffer

    @staticmethod
    def _dataset_fingerprint(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
        image_storage_dtype: str = IMAGE_STORAGE_DTYPE_BFLOAT16,
        image_storage_size: tuple[int, int] | Sequence[int] | None = (224, 224),
        image_stride: int = 1,
    ) -> str:
        """Hash dataset identity plus replay-cache storage semantics."""
        storage_size = ReplayBuffer._normalize_image_storage_size(image_storage_size)
        key_payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "dataset_root": str(dataset.root),
            "total_frames": int(dataset.meta.total_frames),
            "total_episodes": int(dataset.meta.total_episodes),
            "state_keys": sorted(str(k) for k in state_keys) if state_keys is not None else None,
            "image_storage_dtype": ReplayBuffer._normalize_image_storage_dtype(image_storage_dtype),
            "image_storage_size": list(storage_size) if storage_size is not None else None,
        }
        # Only fingerprinted when != 1 so dense caches built before this field keep their hash.
        if image_stride != 1:
            key_payload["image_stride"] = int(image_stride)
        key = json.dumps(key_payload, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @staticmethod
    def _legacy_dataset_fingerprint(dataset: LeRobotDataset) -> str:
        key = f"{dataset.root}|{dataset.meta.total_frames}|{dataset.meta.total_episodes}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    @classmethod
    def find_cache(
        cls,
        dataset: LeRobotDataset,
        cache_dir: str | Path,
        state_keys: Sequence[str] | None = None,
        image_storage_dtype: str = IMAGE_STORAGE_DTYPE_BFLOAT16,
        image_storage_size: tuple[int, int] | Sequence[int] | None = (224, 224),
        image_stride: int = 1,
    ) -> Path | None:
        """Check if a valid cache exists for this dataset and image storage spec."""
        cache_dir = Path(cache_dir)
        fingerprint = cls._dataset_fingerprint(
            dataset,
            state_keys=state_keys,
            image_storage_dtype=image_storage_dtype,
            image_storage_size=image_storage_size,
            image_stride=image_stride,
        )
        candidates = [cache_dir / fingerprint]

        # Backward-compatible lookup for old 224/bf16 caches produced before storage settings
        # were part of the fingerprint.
        storage_size = cls._normalize_image_storage_size(image_storage_size)
        if (
            cls._normalize_image_storage_dtype(image_storage_dtype) == IMAGE_STORAGE_DTYPE_BFLOAT16
            and storage_size == (224, 224)
        ):
            candidates.append(cache_dir / cls._legacy_dataset_fingerprint(dataset))

        for candidate in candidates:
            meta_path = candidate / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("fingerprint") not in {fingerprint, cls._legacy_dataset_fingerprint(dataset)}:
                continue
            if meta.get("num_transitions", 0) == 0:
                continue
            meta_dtype = meta.get("image_storage_dtype", IMAGE_STORAGE_DTYPE_BFLOAT16)
            meta_size = meta.get("image_storage_size", meta.get("image_size", [224, 224]))
            if cls._normalize_image_storage_dtype(meta_dtype) != cls._normalize_image_storage_dtype(image_storage_dtype):
                continue
            if cls._normalize_image_storage_size(meta_size) != storage_size:
                continue
            if int(meta.get("image_stride", 1)) != int(image_stride):
                continue
            return candidate
        return None

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        reward_normalization_constant: float = 1.0,
        terminal_failure_reward: float = -1.0,
        inject_complementary_info: dict | None = None,
        cache_dir: str | Path | None = None,
        image_storage_dtype: str = IMAGE_STORAGE_DTYPE_BFLOAT16,
        image_storage_size: tuple[int, int] | None = (224, 224),
        image_stride: int = 1,
        history_offsets: dict[str, list[int]] | None = None,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        # Check for memmap cache before doing the expensive video decode
        if cache_dir is not None:
            cached = cls.find_cache(
                lerobot_dataset,
                cache_dir,
                state_keys=state_keys,
                image_storage_dtype=image_storage_dtype,
                image_storage_size=image_storage_size,
                image_stride=image_stride,
            )
            if cached is not None:
                logger.info(f"Found memmap cache at {cached}, loading from disk...")
                return cls.from_cache(
                    cache_dir=cached,
                    device=device,
                    image_augmentation_function=image_augmentation_function,
                    use_drq=use_drq,
                    reward_normalization_constant=reward_normalization_constant,
                    terminal_failure_reward=terminal_failure_reward,
                    inject_complementary_info=inject_complementary_info,
                    history_offsets=history_offsets,
                )
            else:
                logger.info(f"No valid cache found in {cache_dir}, falling back to video decode")

        if image_stride != 1:
            # The in-RAM decode path below stores every frame; at the dataset sizes that
            # motivate a stride, silently falling back would OOM. Fail loudly instead.
            raise RuntimeError(
                f"image_stride={image_stride} requires a memmap cache and none matched under "
                f"{cache_dir!r}. Build one: lerobot_memmap_buffer_cache.py --image-stride {image_stride}"
            )

        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
            reward_normalization_constant=reward_normalization_constant,
            terminal_failure_reward=terminal_failure_reward,
            image_storage_dtype=image_storage_dtype,
            image_storage_size=image_storage_size,
            history_offsets=history_offsets,
        )

        # Process dataset transitions one at a time to save memory
        transition_generator = cls._lerobotdataset_to_transitions_generator(
            dataset=lerobot_dataset,
            state_keys=state_keys,
            inject_complementary_info=inject_complementary_info,
        )

        # Get first transition for initialization
        first_transition = next(transition_generator, None)

        if first_transition is not None:
            replay_buffer.add(
                state=first_transition["state"],
                action=first_transition[ACTION],
                reward=first_transition["reward"],
                next_state=first_transition["next_state"],
                done=first_transition["done"],
                truncated=False,
                complementary_info=first_transition.get("complementary_info", None),
            )

        # Process remaining transitions one at a time. Storage dtype/resize policy is centralized
        # in ReplayBuffer.add(), so online and offline data use the same conversion path.
        for _i, data in enumerate(transition_generator):
            replay_buffer.add(
                state=data["state"],
                action=data[ACTION],
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
                complementary_info=data.get("complementary_info", None),
            )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object.
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0]
        act_info = infer_feature_info(t=sample_action, name=ACTION)
        features[ACTION] = act_info

        # Add "reward" and "done"
        features[REWARD] = {"dtype": "float32", "shape": (1,)}
        features[DONE] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = infer_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Add complementary_info keys if available
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0]
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = infer_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.writer.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames

        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict[ACTION] = self.actions[actual_idx].cpu()
            frame_dict[REWARD] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict[DONE] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()
            frame_dict["task"] = task_name

            # Add complementary_info if available
            if self.has_complementary_info:
                for key in self.complementary_info_keys:
                    val = self.complementary_info[key][actual_idx]
                    # Convert tensors to CPU
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        frame_dict[f"complementary_info.{key}"] = val.cpu()
                    # Non-tensor values can be used directly
                    else:
                        frame_dict[f"complementary_info.{key}"] = val

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()

        # Save any remaining frames in the buffer
        if lerobot_dataset.has_pending_frames():
            lerobot_dataset.save_episode()

        lerobot_dataset.writer.stop_image_writer()
        lerobot_dataset.finalize()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions_generator(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
        inject_complementary_info: dict | None = None,
    ):
        """
        Generator version that yields RL transitions one at a time to save memory.

        Args:
            dataset (LeRobotDataset): The dataset to convert.
            state_keys (Sequence[str] | None): The dataset keys to include in 'state' and 'next_state'.

        Yields:
            Transition: One transition at a time.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        num_frames = len(dataset)
        if num_frames == 0:
            return

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = DONE in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        if "subtask_index" in sample:
            complementary_info_keys.append("subtask_index")

        if inject_complementary_info:
            for k in inject_complementary_info:
                if k not in complementary_info_keys:
                    complementary_info_keys.append(k)

        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        import multiprocessing

        from torch.utils.data import DataLoader

        num_workers = min(4, multiprocessing.cpu_count() or 1)
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        def process_sample(current_sample, next_sample, is_last):
            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample[ACTION]

            # CRITICAL FIX: Handle pre-chunked actions from dataset
            # If the dataset was loaded with delta_indices (e.g., [0, 1, ..., 49]),
            # actions will be shape [50, 6] instead of [6]
            # We only want the FIRST action to keep buffer storage simple and consistent
            if action.ndim == 2:  # Shape is [chunk_size, action_dim]
                action = action[0]  # Extract first timestep only  → shape [action_dim]

            action = action.unsqueeze(0)  # Add batch dimension → shape [1, action_dim]

            # ----- 3) Determine done flag -----
            if has_done_key:
                done = bool(current_sample[DONE].item())
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if is_last or next_sample["episode_index"] != current_sample["episode_index"]:
                    done = True

            # Reward is inferred from done if not present
            if REWARD in current_sample:
                reward = current_sample[REWARD].item()
            else:
                reward = 1.0 if done else 0.0

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and not is_last:
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----

            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    if key.startswith("complementary_info."):
                        clean_key = key[len("complementary_info."):]
                    else:
                        clean_key = key

                    if inject_complementary_info is not None and clean_key in inject_complementary_info:
                        val = inject_complementary_info[clean_key]
                        if isinstance(val, bool):
                            val = torch.tensor(val, dtype=torch.bool)
                        elif isinstance(val, float):
                            val = torch.tensor(val, dtype=torch.float32)
                        elif hasattr(val, "dtype"): # Already a tensor
                            pass
                        else:
                            val = torch.tensor(val)
                    else:
                        val = current_sample[key]

                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct and yield the Transition -----
            return Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )

        iterator = iter(loader)
        try:
            prev_sample = next(iterator)
        except StopIteration:
            return

        for current_sample in tqdm(iterator, total=num_frames - 1):
            yield process_sample(prev_sample, current_sample, is_last=False)
            prev_sample = current_sample

        yield process_sample(prev_sample, None, is_last=True)


# Utility function to guess shapes/dtypes from a tensor
def infer_feature_info(t, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """

    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition,
    right_batch_transition: BatchTransition,
    action_dim: int = 6,
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.
        action_dim (int): Active (unpadded) action dimension. The left batch's action
            tensor is sliced to this width before concatenation so it matches the right
            batch, which is assumed to already be at action_dim.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION][:, :, :action_dim], right_batch_transition[ACTION]],
        dim=0,
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # If both are None, nothing to do
    if left_info is None and right_info is None:
        return left_batch_transitions

    # Ensure left_batch_transitions has a complementary_info dict
    if left_info is None:
        left_info = {}
        left_batch_transitions["complementary_info"] = left_info

    # If right_info is None, treat as empty dict for key iteration
    if right_info is None:
        right_info = {}

    # Calculate batch sizes to determine padding size
    # We use the 'reward' field as reference for batch size of each part
    # Note: left_batch_transitions['reward'] is ALREADY concatenated, so we need to derive sizes differently.
    # The 'state' concatenation loop assumes keys match, but 'action' concatenation line 932
    # uses slicing on left and right.
    # We can infer the size of 'right' from right_batch_transition['reward'].shape[0]
    # And total size from left_batch_transitions['reward'].shape[0]
    total_len = left_batch_transitions["reward"].shape[0]
    right_len = right_batch_transition["reward"].shape[0]
    left_len = total_len - right_len

    # Get union of all keys
    all_keys = set(left_info.keys()) | set(right_info.keys())

    for key in all_keys:
        left_val = left_info.get(key)
        right_val = right_info.get(key)

        # 1. Present in both
        if left_val is not None and right_val is not None:
             left_info[key] = torch.cat([left_val, right_val], dim=0)

        # 2. Present only in Right (Missing in Left) -> Pad Left
        elif left_val is None:
             # Create padding for left
             # right_val shape: (right_len, ...)
             # padding shape: (left_len, ...) + right_val.shape[1:]
             shape = (left_len,) + right_val.shape[1:]
             padding = torch.zeros(shape, dtype=right_val.dtype, device=right_val.device)
             left_info[key] = torch.cat([padding, right_val], dim=0)

        # 3. Present only in Left (Missing in Right) -> Pad Right
        elif right_val is None:
             # Create padding for right
             # left_val represents the ORIGINAL left value (before any concatenation)
             # Wait! 'left_info' is a mutable reference to the dictionary in 'left_batch_transitions'.
             # Since we haven't modified 'left_info[key]' yet in this loop, 'left_val' IS the original tensor from the left batch.
             shape = (right_len,) + left_val.shape[1:]
             padding = torch.zeros(shape, dtype=left_val.dtype, device=left_val.device)
             left_info[key] = torch.cat([left_val, padding], dim=0)

    return left_batch_transitions
