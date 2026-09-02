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

"""Source-native actor replay over the diverse corpus (integration plan phase B).

``ReplayBuffer`` is a frame-indexed store: it holds one action per frame and builds a
chunk by walking forward from the sampled index. The diverse corpus is not that. Its
actor view is a list of 5 Hz *anchors*, each already carrying a complete one-second
future at 30 Hz, and the frames between two anchors are not transitions of that view.
Reconstructing a chunk from adjacent anchor rows would fabricate a trajectory that was
never recorded, and 5 of every 6 recorded actions would never be a target.

So this buffer samples anchors directly and preserves the stored ``(30, D)`` chunk end
to end. It emits the same ``BatchTransition`` the offline collection iterator consumes,
which is what lets a ReBot ``ReplayBuffer`` and this object be mixed by
``concatenate_variable_dim_batch_transitions`` without either knowing about the other.

What this buffer does NOT do, deliberately:

* it does not encode actions. Anchor encoding is ``AnchorEncodeStep``'s job, applied to
  the whole chunk inside the preprocessor once the current state and the native-width
  masks are both present. The buffer's contract is absolute actions plus honest masks.
* it does not serve a critic. ``next_state`` mirrors ``state`` so batches concatenate;
  no TD target may be computed from it. See the plan's deferred critic phase.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.datasets.diverse_actor_selection import (
    CANONICAL_CAMERA_ROLES,
    MAX_ACTION_DIM,
    DiverseActorSelection,
    packed_history_slots,
    PACKED_CURRENT_SLOT,
)
from lerobot.datasets.diverse_prompt import (
    UNKNOWN_QUALITY,
    episode_task,
    quality_provenance_id,
    retention_reason_id,
    should_render_quality,
)
from lerobot.datasets.embodiment import embodiment_index
from lerobot.rl.buffer import BatchTransition
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.depth_gripper_events import DEPTH_GRIPPER_EVENT_TARGET_KEYS

logger = logging.getLogger(__name__)

# Stable integer ids for the corpus sources, carried on every batch for reproducibility
# and for the realized-mixture telemetry. APPEND-ONLY, like the action layouts.
SOURCE_IDS: dict[str, int] = {
    "droid": 0,
    "droid_success": 1,
    "fmb": 2,
    "robochallenge": 3,
    "ur7e": 4,
    "rebot": 5,
}

DEPTH_INVALID_Z16 = (0, 65_535)


# ── Resize geometry ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResizeGeometry:
    """One deterministic aspect-preserving resize into a fixed frame, with padding.

    The source is scaled by a single factor (so nothing is stretched) and centred in the
    target frame; the leftover band is padding, not image. ``valid_box`` is exactly the
    region that came from the sensor, and it is the compact form of the pixel-valid mask
    -- ``pixel_valid_mask`` expands it. Depth back-projection reads the same numbers
    through ``transform_intrinsics``, so the geometry the pixels went through and the
    geometry the intrinsics went through cannot drift apart.
    """

    source_height: int
    source_width: int
    height: int
    width: int
    scale: float
    scaled_height: int
    scaled_width: int
    top: int
    left: int

    @classmethod
    def fit(cls, source: tuple[int, int], target: tuple[int, int]) -> "ResizeGeometry":
        source_height, source_width = int(source[0]), int(source[1])
        height, width = int(target[0]), int(target[1])
        scale = min(height / source_height, width / source_width)
        scaled_height = max(1, int(round(source_height * scale)))
        scaled_width = max(1, int(round(source_width * scale)))
        return cls(
            source_height=source_height,
            source_width=source_width,
            height=height,
            width=width,
            scale=scale,
            scaled_height=scaled_height,
            scaled_width=scaled_width,
            top=(height - scaled_height) // 2,
            left=(width - scaled_width) // 2,
        )

    @property
    def valid_box(self) -> tuple[int, int, int, int]:
        """(top, left, height, width) of the sensor region inside the target frame."""
        return (self.top, self.left, self.scaled_height, self.scaled_width)

    @property
    def is_identity(self) -> bool:
        return (self.source_height, self.source_width) == (self.height, self.width)

    def pixel_valid_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[self.top : self.top + self.scaled_height, self.left : self.left + self.scaled_width] = True
        return mask

    def transform_intrinsics(
        self, intrinsics: Sequence[float]
    ) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) carried through this exact resize and pad.

        A uniform scale multiplies both focal lengths and both principal-point
        coordinates; the centring pad then shifts the principal point only.
        """
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        return (
            fx * self.scale,
            fy * self.scale,
            cx * self.scale + self.left,
            cy * self.scale + self.top,
        )


def _resize_frames(frames: np.ndarray, geometry: ResizeGeometry, *, nearest: bool) -> np.ndarray:
    """Resize (N, H, W[, C]) into the target frame, zero-padding the leftover band."""
    if geometry.is_identity:
        return np.ascontiguousarray(frames)

    array = torch.from_numpy(np.ascontiguousarray(frames))
    channels_last = array.ndim == 4
    if channels_last:  # (N, H, W, C) -> (N, C, H, W)
        array = array.permute(0, 3, 1, 2)
    else:  # (N, H, W) -> (N, 1, H, W)
        array = array.unsqueeze(1)

    source_dtype = array.dtype
    mode = "nearest" if nearest else "bilinear"
    resized = torch.nn.functional.interpolate(
        array.to(torch.float32),
        size=(geometry.scaled_height, geometry.scaled_width),
        mode=mode,
        **({} if nearest else {"align_corners": False, "antialias": True}),
    )
    if source_dtype == torch.uint8:
        resized = resized.round().clamp(0, 255)
    elif not source_dtype.is_floating_point:
        resized = resized.round()
    resized = resized.to(source_dtype)

    canvas = torch.zeros(
        (resized.shape[0], resized.shape[1], geometry.height, geometry.width), dtype=source_dtype
    )
    canvas[
        :,
        :,
        geometry.top : geometry.top + geometry.scaled_height,
        geometry.left : geometry.left + geometry.scaled_width,
    ] = resized
    if channels_last:
        canvas = canvas.permute(0, 2, 3, 1)
    else:
        canvas = canvas.squeeze(1)
    return canvas.numpy()


# ── Sample specification ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiverseSampleSpec:
    """What one diverse training sample contains. Also the cache fingerprint input."""

    history_ages_s: tuple[float, ...] = (6.0, 4.0, 2.0)
    action_horizon: int = 30
    action_rate_hz: float = 30.0
    max_width: int = MAX_ACTION_DIM
    camera_roles: tuple[str, ...] = CANONICAL_CAMERA_ROLES
    image_size: tuple[int, int] = (480, 640)
    depth_size: tuple[int, int] = (480, 640)
    depth_role: str = "wrist_0"
    load_images: bool = True
    load_depth: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1.")
        if self.depth_role not in self.camera_roles:
            raise ValueError(f"depth_role {self.depth_role!r} is not one of {self.camera_roles}.")
        packed_history_slots(list(self.history_ages_s))  # raises on an unpacked instant

    @property
    def history_slots(self) -> list[int]:
        return packed_history_slots(list(self.history_ages_s))

    @property
    def num_history(self) -> int:
        return len(self.history_ages_s)

    def image_key(self, role: str) -> str:
        return f"{OBS_IMAGES}.{role}"

    def depth_key(self) -> str:
        return f"depth.{self.depth_role}.depth"

    def fingerprint(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "history_ages_s": list(self.history_ages_s),
            "action_horizon": self.action_horizon,
            "action_rate_hz": self.action_rate_hz,
            "max_width": self.max_width,
            "camera_roles": list(self.camera_roles),
            "image_size": list(self.image_size),
            "depth_size": list(self.depth_size),
            "depth_role": self.depth_role,
            "load_images": self.load_images,
            "load_depth": self.load_depth,
        }


# ── Buffer ───────────────────────────────────────────────────────────────────


@dataclass
class _RowIdentity:
    source_id: int
    episode_position: int
    anchor_index: int
    action_layout_id: int
    embodiment_index: int
    task_index: int
    subtask_index: int
    quality: float
    quality_is_valid: bool
    quality_provenance_id: int
    retention_reason_id: int
    mistake: bool


class DiverseActorBuffer:
    """Anchor-indexed replay exposing the offline collection iterator's interface."""

    def __init__(
        self,
        selection: DiverseActorSelection,
        spec: DiverseSampleSpec | None = None,
        *,
        device: str = "cpu",
        seed: int = 0,
        sampler=None,
        cache=None,
        task_indices: dict[str, int] | None = None,
        subtask_indices: dict[str, int] | None = None,
        render_automatic_quality: bool = False,
    ) -> None:
        self.selection = selection
        self.spec = spec or DiverseSampleSpec()
        self.device = device
        self.cache = cache
        self.rows = selection.rows
        self.size = len(self.rows)
        self._rng = np.random.default_rng(seed)
        self._sampler = sampler

        episode_position = {episode_id: i for i, episode_id in enumerate(selection.episode_ids)}
        self.task_indices = task_indices or {}
        self.subtask_indices = subtask_indices or {}
        self.render_automatic_quality = bool(render_automatic_quality)

        self._identity = []
        for row in self.rows:
            record = selection.episode_records[str(row["episode_id"])]
            task_text = episode_task(record, row["source"])[0]
            provenance = row.get("quality_provenance")
            # An automatically derived "5" is not a reviewed five. Unless the run asks
            # for it, the quality integer is withheld from the prompt (and the clause
            # is simply omitted) while the value stays auditable in the provenance id.
            renders = should_render_quality(provenance, render_automatic=self.render_automatic_quality)
            quality = row.get("quality")
            self._identity.append(
                _RowIdentity(
                    source_id=SOURCE_IDS[row["source"]],
                    episode_position=episode_position[row["episode_id"]],
                    anchor_index=int(row["anchor_index"]),
                    action_layout_id=int(row["action_layout_id"]),
                    embodiment_index=embodiment_index(row["embodiment"]),
                    task_index=self.task_indices.get(task_text, -1),
                    subtask_index=self.subtask_indices.get(str(row["subtask"]), -1),
                    quality=float(quality) if (renders and quality is not None) else UNKNOWN_QUALITY,
                    quality_is_valid=bool(renders and quality is not None),
                    quality_provenance_id=quality_provenance_id(provenance),
                    retention_reason_id=retention_reason_id(row.get("retention_reason")),
                    mistake=bool(row["mistake"]),
                )
            )
        self._geometry: dict[tuple[int, int, bool], ResizeGeometry] = {}

    def __len__(self) -> int:
        return self.size

    # -- geometry cache ---------------------------------------------------------

    def geometry_for(self, source_size: tuple[int, int], *, depth: bool = False) -> ResizeGeometry:
        key = (int(source_size[0]), int(source_size[1]), depth)
        geometry = self._geometry.get(key)
        if geometry is None:
            target = self.spec.depth_size if depth else self.spec.image_size
            geometry = self._geometry[key] = ResizeGeometry.fit(source_size, target)
        return geometry

    # -- sampling ---------------------------------------------------------------

    def draw_indices(self, batch_size: int) -> np.ndarray:
        """Row indices for one batch. Phase I replaces this with the hierarchical draw."""
        if self._sampler is not None:
            return np.asarray(self._sampler(batch_size), dtype=np.int64)
        return self._rng.integers(0, self.size, size=batch_size, dtype=np.int64)

    def sample(self, batch_size: int, action_chunk_size: int | None = None) -> BatchTransition:
        if action_chunk_size is not None and int(action_chunk_size) != self.spec.action_horizon:
            raise ValueError(
                f"action_chunk_size={action_chunk_size} but the diverse corpus packs "
                f"{self.spec.action_horizon}-point chunks. The stored chunk is never trimmed or "
                "extended here -- change policy.chunk_size and rebuild the views instead."
            )
        return self.collate(self.draw_indices(batch_size))

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = False,
        queue_size: int = 2,  # noqa: ARG002 - interface parity with ReplayBuffer
        action_chunk_size: int | None = None,
    ):
        if async_prefetch and self.cache is None:
            raise NotImplementedError(
                "Without a cache this buffer decodes video on the sampling thread at "
                "~1.7 s/sample; prefetching it would only move the stall. Build the cache "
                "(lerobot.rl.data_sources.diverse_actor_cache) or run async_prefetch=False."
            )
        while True:
            yield self.sample(batch_size, action_chunk_size=action_chunk_size)

    # -- one sample -------------------------------------------------------------

    def load_sample(self, row_index: int) -> dict[str, Any]:
        """One anchor, source-native, with the packed window reduced to what training uses.

        Returned arrays keep their native width; padding to ``max_width`` happens in
        ``collate`` so the native-width masks are derived from the real widths. With a
        cache attached this reads the same dict off disk instead of decoding video --
        the two paths are asserted equal by the phase C equivalence test.
        """
        if self.cache is not None:
            return self.cache.sample(row_index)
        spec = self.spec
        row = self.rows[row_index]
        sample = self.selection.corpus.actor_sample(row, cameras=spec.load_images)
        slots = spec.history_slots

        states = np.asarray(sample["observation.state"], dtype=np.float32)
        timestamps = np.asarray(sample["observation.timestamps"], dtype=np.float64)
        actions = np.asarray(sample["action"], dtype=np.float32)
        if actions.shape != (spec.action_horizon, states.shape[1]):
            raise ValueError(
                f"{row['episode_id']} anchor {row['anchor_s']}s: action {actions.shape} is not "
                f"the packed ({spec.action_horizon}, {states.shape[1]}) chunk."
            )

        out: dict[str, Any] = {
            "row_index": row_index,
            "state": states[PACKED_CURRENT_SLOT],
            "history_state": states[slots],
            "action": actions,
            "timestamp": timestamps[PACKED_CURRENT_SLOT],
            "history_timestamps": timestamps[slots],
            "native_width": int(states.shape[1]),
            "images": {},
            "image_geometry": {},
            # Presence is a property of what was recorded, so it is reported even when
            # pixels are not loaded -- an audit must not read "no cameras" off a
            # low-dimensional pass.
            "camera_roles": list(row["camera_roles"]),
        }

        if spec.load_images:
            frames_by_camera = sample["observation.images"]
            for role, camera in row["camera_roles"].items():
                if role not in spec.camera_roles:
                    raise KeyError(f"role {role!r} is not one of the model's roles {spec.camera_roles}.")
                frames = np.asarray(frames_by_camera[camera])
                geometry = self.geometry_for(frames.shape[1:3])
                wanted = _resize_frames(frames[[*slots, PACKED_CURRENT_SLOT]], geometry, nearest=False)
                # (T_h + 1, 3, H, W), current last -- the layout the cache stores and the
                # model consumes, so neither path pays for a transpose at sample time.
                out["images"][role] = np.ascontiguousarray(wanted.transpose(0, 3, 1, 2))
                out["image_geometry"][role] = geometry

        if spec.load_depth and row["has_depth"]:
            # Read depth straight from the episode rather than from actor_sample, which
            # only attaches it alongside a full RGB decode. Depth is a memmapped array;
            # coupling it to video decoding would make a depth-only check cost minutes.
            episode = self.selection.corpus.fmb.episode(str(row["episode_id"]))
            packed = np.asarray(row["history_frames"], dtype=np.int64)
            wanted = [*slots, PACKED_CURRENT_SLOT]
            raw, valid = episode.wrist_depth(packed[wanted])
            geometry = self.geometry_for(raw.shape[1:3], depth=True)
            wanted = list(range(len(wanted)))
            # Nearest neighbour: an interpolated depth pixel is a range reading that no
            # sensor produced, and averaging across an invalid pixel invents geometry.
            out["depth"] = _resize_frames(raw[wanted], geometry, nearest=True)
            out["depth_valid"] = _resize_frames(
                valid[wanted].astype(np.uint8), geometry, nearest=True
            ).astype(bool) & geometry.pixel_valid_mask()[None]
            out["depth_geometry"] = geometry
            depth_meta = self.selection.episode_records[str(row["episode_id"])]["depth"]
            out["depth_units_mm_per_level"] = float(depth_meta["depth_units_mm_per_level"])
            out["depth_scale_provenance"] = str(depth_meta["depth_scale_provenance"])
            from lerobot.rl.data_sources.diverse_actor_cache import training_depth_intrinsics

            out["depth_intrinsics"] = training_depth_intrinsics(
                (int(raw.shape[1]), int(raw.shape[2])), spec.depth_size
            )
            targets = episode.depth_gripper_event_targets(int(row["anchor_frame"]))
            out["depth_event_targets"] = np.asarray(
                [float(targets[key]) for key in DEPTH_GRIPPER_EVENT_TARGET_KEYS], dtype=np.float32
            )
        return out

    # -- collation --------------------------------------------------------------

    def collate(self, row_indices: Sequence[int]) -> BatchTransition:
        spec = self.spec
        width = spec.max_width
        horizon = spec.action_horizon
        history = spec.num_history
        batch = len(row_indices)
        samples = [self.load_sample(int(index)) for index in row_indices]

        state = torch.zeros((batch, width), dtype=torch.float32)
        history_state = torch.zeros((batch, history, width), dtype=torch.float32)
        action = torch.zeros((batch, horizon, width), dtype=torch.float32)
        # True marks padding, matching concatenate_variable_dim_batch_transitions.
        state_is_pad = torch.ones((batch, width), dtype=torch.bool)
        action_is_pad = torch.ones((batch, width), dtype=torch.bool)

        images: dict[str, torch.Tensor] = {}
        history_images: dict[str, torch.Tensor] = {}
        image_valid_box = torch.zeros((batch, len(spec.camera_roles), 4), dtype=torch.int16)
        camera_present = torch.zeros((batch, len(spec.camera_roles)), dtype=torch.bool)
        if spec.load_images:
            height, image_width = spec.image_size
            for role in spec.camera_roles:
                images[role] = torch.zeros((batch, 3, height, image_width), dtype=torch.uint8)
                history_images[role] = torch.zeros(
                    (batch, history, 3, height, image_width), dtype=torch.uint8
                )

        depth_height, depth_width = spec.depth_size
        depth = torch.zeros((batch, depth_height, depth_width), dtype=torch.uint16)
        depth_history = torch.zeros((batch, history, depth_height, depth_width), dtype=torch.uint16)
        depth_valid = torch.zeros((batch, depth_height, depth_width), dtype=torch.bool)
        depth_history_valid = torch.zeros((batch, history, depth_height, depth_width), dtype=torch.bool)
        depth_present = torch.zeros((batch,), dtype=torch.bool)
        depth_history_present = torch.zeros((batch, history), dtype=torch.bool)
        depth_units = torch.zeros((batch,), dtype=torch.float32)
        depth_intrinsics = torch.zeros((batch, 4), dtype=torch.float32)
        depth_event_targets = torch.zeros((batch, 2), dtype=torch.float32)

        identity_fields = (
            "source_id",
            "episode_position",
            "anchor_index",
            "action_layout_id",
            "embodiment_index",
            "task_index",
            "subtask_index",
            "quality_provenance_id",
            "retention_reason_id",
        )
        identity = {name: torch.zeros((batch,), dtype=torch.long) for name in identity_fields}
        metadata_quality = torch.full((batch,), UNKNOWN_QUALITY, dtype=torch.float32)
        metadata_quality_is_valid = torch.zeros((batch,), dtype=torch.bool)
        metadata_mistake = torch.zeros((batch,), dtype=torch.float32)
        row_index_column = torch.zeros((batch,), dtype=torch.long)

        for position, sample in enumerate(samples):
            native = sample["native_width"]
            if native > width:
                raise ValueError(f"native width {native} exceeds the model layout width {width}.")
            state[position, :native] = torch.from_numpy(sample["state"])
            history_state[position, :, :native] = torch.from_numpy(sample["history_state"])
            action[position, :, :native] = torch.from_numpy(sample["action"])
            state_is_pad[position, :native] = False
            action_is_pad[position, :native] = False

            for role_index, role in enumerate(spec.camera_roles):
                camera_present[position, role_index] = role in sample["camera_roles"]
                frames = sample["images"].get(role)
                if frames is None:
                    continue
                image_valid_box[position, role_index] = torch.tensor(
                    sample["image_geometry"][role].valid_box, dtype=torch.int16
                )
                stack = torch.from_numpy(np.ascontiguousarray(frames))
                history_images[role][position] = stack[:history]
                images[role][position] = stack[history]

            if "depth" in sample:
                frames = torch.from_numpy(np.ascontiguousarray(sample["depth"].astype(np.uint16)))
                valid = torch.from_numpy(np.ascontiguousarray(sample["depth_valid"]))
                depth_history[position] = frames[:history]
                depth[position] = frames[history]
                depth_history_valid[position] = valid[:history]
                depth_valid[position] = valid[history]
                depth_present[position] = True
                depth_history_present[position] = True
                depth_units[position] = sample["depth_units_mm_per_level"]
                depth_intrinsics[position] = torch.tensor(
                    sample["depth_intrinsics"], dtype=torch.float32
                )
                depth_event_targets[position] = torch.as_tensor(
                    sample["depth_event_targets"], dtype=torch.float32
                )

            entry = self._identity[sample["row_index"]]
            for name in identity_fields:
                identity[name][position] = getattr(entry, name)
            metadata_quality[position] = entry.quality
            metadata_quality_is_valid[position] = entry.quality_is_valid
            metadata_mistake[position] = float(entry.mistake)
            row_index_column[position] = sample["row_index"]

        state_dict: dict[str, torch.Tensor] = {OBS_STATE: state.to(self.device)}
        state_dict[f"history.{OBS_STATE}"] = history_state.to(self.device)
        state_dict[f"history.{OBS_STATE}_is_pad"] = torch.zeros(
            (batch, history), dtype=torch.bool, device=self.device
        )
        for role_index, role in enumerate(spec.camera_roles):
            if not spec.load_images:
                break
            key = spec.image_key(role)
            state_dict[key] = images[role].to(self.device)
            state_dict[f"history.{key}"] = history_images[role].to(self.device)
            # A history slot is padded exactly when its camera is absent: the anchor
            # stride starts at 6 s, so no slot ever reaches past an episode start.
            state_dict[f"history.{key}_is_pad"] = (
                ~camera_present[:, role_index, None].expand(batch, history)
            ).contiguous().to(self.device)

        complementary: dict[str, Any] = {
            "state_dim_is_pad": state_is_pad.to(self.device),
            "action_dim_is_pad": action_is_pad.to(self.device),
            "camera_is_present": camera_present.to(self.device),
            "history.camera_is_present": camera_present[:, :, None]
            .expand(batch, len(spec.camera_roles), history)
            .contiguous()
            .to(self.device),
            "image_valid_box": image_valid_box.to(self.device),
            # Per-key presence beside the matrix: the prompt step looks presence up by
            # image key, because the batch's role order and the policy's image_keys order
            # are configured in different places.
            **{
                f"camera_is_present.{spec.image_key(role)}": camera_present[:, index].to(self.device)
                for index, role in enumerate(spec.camera_roles)
            },
            "embodiment_index": identity["embodiment_index"].to(self.device),
            "action_layout_id": identity["action_layout_id"].to(self.device),
            "source_id": identity["source_id"].to(self.device),
            "episode_position": identity["episode_position"].to(self.device),
            "anchor_index": identity["anchor_index"].to(self.device),
            "diverse_row_index": row_index_column.to(self.device),
            "metadata_quality": metadata_quality.to(self.device),
            "metadata_quality_is_valid": metadata_quality_is_valid.to(self.device),
            "metadata_mistake": metadata_mistake.to(self.device),
            "task_index": identity["task_index"].to(self.device),
            "subtask_index": identity["subtask_index"].to(self.device),
            "quality_provenance_id": identity["quality_provenance_id"].to(self.device),
            "retention_reason_id": identity["retention_reason_id"].to(self.device),
        }
        if spec.load_depth:
            depth_key = spec.depth_key()
            # Current depth rides complementary_info, where the trainer's depth injector
            # looks for it. Depth HISTORY rides state, because that is where ReBot's
            # ReplayBuffer.sample() puts its history and where batch_to_transition picks
            # "history.*" up for the pipeline. The two halves have to agree on both.
            complementary[depth_key] = depth.to(self.device)
            state_dict[f"history.{depth_key}"] = depth_history.to(self.device)
            state_dict[f"history.{depth_key}_is_pad"] = (
                ~depth_history_present
            ).contiguous().to(self.device)
            complementary[f"{depth_key}_valid"] = depth_valid.to(self.device)
            complementary[f"history.{depth_key}_valid"] = depth_history_valid.to(self.device)
            complementary[f"{depth_key}_is_present"] = depth_present.to(self.device)
            complementary[f"history.{depth_key}_is_present"] = depth_history_present.to(self.device)
            complementary[f"{depth_key}_units_mm_per_level"] = depth_units.to(self.device)
            # (fx, fy, cx, cy) already carried through this run's resize geometry, so the
            # back-projection and the pixels cannot disagree about where the image moved.
            complementary[f"depth.{spec.depth_role}.intrinsics"] = depth_intrinsics.to(self.device)
            # Real labels where depth exists; zeros elsewhere, which the presence mask
            # excludes from the event loss and from its denominator.
            for index, key in enumerate(DEPTH_GRIPPER_EVENT_TARGET_KEYS):
                complementary[key] = depth_event_targets[:, index].to(self.device)

        zeros = torch.zeros((batch,), dtype=torch.float32, device=self.device)
        return BatchTransition(
            state=state_dict,
            action=action.to(self.device),
            reward=zeros,
            # Actor-only: nothing reads next_state. It mirrors state so a mixed batch
            # concatenates key for key; a critic must not be pointed at this buffer.
            next_state=dict(state_dict),
            done=zeros.clone(),
            truncated=zeros.clone(),
            complementary_info=complementary,
        )
