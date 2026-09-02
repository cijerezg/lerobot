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

"""Bring the ReBot replay onto the canonical camera roles (integration plan phases C/D).

The two halves of the 50/50 mixture are concatenated key by key, so they must agree on
the camera vocabulary. ReBot's cache spells its views ``top`` and ``wrist``; the corpus
sources spell theirs six different ways. Everything is renamed onto
``external_0 / external_1 / wrist_0``.

ReBot records no second external view. That role is therefore *absent*, and absent is
not black: the batch carries ``camera_is_present`` so the model can drop the slot's
tokens rather than learn to read a black frame. The zero tensor exists only because a
concatenated batch needs one tensor per key.

Probing before rebuilding matters here: the ReBot caches are 140 GB across four sources.
``probe_rebot_cache`` checks the three things reuse actually depends on -- that
``image_stride`` can address the -6 s lookback exactly, that the RGB and depth columns
the run needs are present, and that the storage provenance is recorded -- so a rebuild
happens only when one of them fails.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.diverse_actor_selection import CAMERA_ROLE_MAP, CANONICAL_CAMERA_ROLES
from lerobot.rl.data_sources.diverse_actor_buffer import SOURCE_IDS
from lerobot.utils.constants import ACTION, OBS_IMAGES

logger = logging.getLogger(__name__)


class RebotCacheProbeError(RuntimeError):
    """The ReBot cache cannot serve this run's observation contract."""


@dataclass
class RebotCacheReport:
    cache_dir: Path
    num_transitions: int
    image_stride: int
    image_keys: list[str]
    depth_keys: list[str]
    image_shape: list[int] | None
    depth_shape: list[int] | None
    image_storage_dtype: str
    image_storage_size: list[int] | None
    history_reach: dict[int, float] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        lines = [
            f"ReBot cache {self.cache_dir.name}: {self.num_transitions} transitions, "
            f"image_stride={self.image_stride}, images {self.image_shape} "
            f"{self.image_storage_dtype}, depth {self.depth_shape}",
        ]
        for offset, fraction in sorted(self.history_reach.items()):
            lines.append(f"  lookback {offset} frames reachable on {fraction:.1%} of frames")
        for problem in self.problems:
            lines.append(f"  PROBLEM: {problem}")
        return "\n".join(lines)


def probe_rebot_cache(
    cache_dir: str | Path,
    *,
    history_offsets_frames: list[int],
    depth_role: str | None = "wrist",
    measure_reach: bool = True,
) -> RebotCacheReport:
    """Can this cache serve the mixed run's observations? Report, do not raise."""
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise RebotCacheProbeError(f"{cache_dir} holds no metadata.json.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    stride = int(metadata.get("image_stride", 1))
    image_keys = list(metadata.get("image_keys") or [])
    depth_keys = list(metadata.get("depth_keys") or [])
    shapes = metadata.get("shapes") or {}
    report = RebotCacheReport(
        cache_dir=cache_dir,
        num_transitions=int(metadata.get("num_transitions", 0)),
        image_stride=stride,
        image_keys=image_keys,
        depth_keys=depth_keys,
        image_shape=shapes.get(image_keys[0]) if image_keys else None,
        depth_shape=shapes.get(f"depth.{depth_keys[0]}") if depth_keys else None,
        image_storage_dtype=str(metadata.get("image_storage_dtype", "")),
        image_storage_size=metadata.get("image_storage_size"),
    )

    if report.num_transitions <= 0:
        report.problems.append("cache declares zero transitions")
    # -6 s addressability: image rows exist only every stride-th frame, so a lookback
    # that is not a multiple of the stride has no stored row to read.
    misaligned = [offset for offset in history_offsets_frames if offset % stride != 0]
    if misaligned:
        report.problems.append(
            f"history offsets {misaligned} are not multiples of image_stride={stride}; "
            "those image rows were never written"
        )
    expected_cameras = set(CAMERA_ROLE_MAP["rebot"])
    present_cameras = {key.rsplit(".", 1)[-1] for key in image_keys}
    missing = sorted(expected_cameras - present_cameras)
    if missing:
        report.problems.append(f"missing RGB columns for {missing}")
    if depth_role is not None and f"{depth_role}.depth" not in depth_keys:
        report.problems.append(f"missing depth column depth.{depth_role}.depth (has {depth_keys})")
    for field_name in ("image_storage_dtype", "image_stride", "fingerprint", "num_transitions"):
        if field_name not in metadata:
            report.problems.append(f"missing provenance field {field_name!r}")

    if measure_reach and report.num_transitions > 0 and history_offsets_frames:
        dones_path = cache_dir / "dones.bin"
        if dones_path.is_file():
            dtype = np.dtype((metadata.get("dtypes") or {}).get("dones", "bool"))
            dones = np.asarray(
                np.memmap(dones_path, dtype=dtype, mode="r", shape=(report.num_transitions,))
            ).astype(bool)
            index = np.arange(report.num_transitions)
            starts = np.concatenate([[True], dones[:-1]])
            reach = index - np.maximum.accumulate(np.where(starts, index, -1))
            for offset in history_offsets_frames:
                report.history_reach[int(offset)] = float((reach >= offset).mean())
    return report


# ── Role alignment ───────────────────────────────────────────────────────────


def rebot_role_renames(image_keys: list[str], depth_keys: list[str]) -> dict[str, str]:
    """Cache key -> canonical-role key, for both RGB and depth."""
    mapping = CAMERA_ROLE_MAP["rebot"]
    renames: dict[str, str] = {}
    for key in image_keys:
        camera = key.rsplit(".", 1)[-1]
        role = mapping.get(camera)
        if role is None:
            raise RebotCacheProbeError(
                f"ReBot camera {camera!r} is unmapped. Every recorded camera must land in a "
                "canonical role or be removed deliberately."
            )
        renames[key] = f"{OBS_IMAGES}.{role}"
    for key in depth_keys:
        camera = key.split(".", 1)[0]
        role = mapping.get(camera)
        if role is not None:
            renames[f"depth.{key}"] = f"depth.{role}.depth"
    return renames


def rebot_state_keys(state_keys) -> list[str]:
    """The key names a ReBot cache was fingerprinted under, from the run's canonical ones.

    Renaming the cameras onto canonical roles changes the policy's `input_features`, and
    `ReplayBuffer._dataset_fingerprint` hashes those key names -- so a 140 GB cache that
    is otherwise perfectly usable stops being findable. The payload did not change, only
    what this run calls it, so the lookup asks under the old names. Roles ReBot never
    recorded (external_1) are dropped rather than looked up.
    """
    inverse = {
        f"{OBS_IMAGES}.{role}": f"{OBS_IMAGES}.{camera}"
        for camera, role in CAMERA_ROLE_MAP["rebot"].items()
    }
    resolved: list[str] = []
    for key in state_keys:
        if str(key).startswith(f"{OBS_IMAGES}."):
            mapped = inverse.get(str(key))
            if mapped is not None:
                resolved.append(mapped)
        else:
            resolved.append(str(key))
    return resolved


def align_rebot_buffer(buffer, *, depth_role: str = "wrist") -> dict[str, str]:
    """Rename a loaded ReBot ``ReplayBuffer``'s columns onto the canonical roles, in place.

    Renaming the storage (rather than the sampled batch) is what lets the buffer's own
    history gather run with canonical ``history_offsets``: ``_gather_history`` indexes
    ``self.states[key]`` directly.
    """
    image_keys = [key for key in buffer.states if key.startswith(f"{OBS_IMAGES}.")]
    depth_keys = [key for key in getattr(buffer, "complementary_info", {}) if key.startswith("depth.")]
    renames = rebot_role_renames(image_keys, [key[len("depth.") :] for key in depth_keys])

    for old, new in renames.items():
        if old.startswith("depth."):
            if old in buffer.complementary_info and old != new:
                buffer.complementary_info[new] = buffer.complementary_info.pop(old)
                buffer.complementary_info_keys = [
                    new if key == old else key for key in buffer.complementary_info_keys
                ]
        elif old in buffer.states and old != new:
            buffer.states[new] = buffer.states.pop(old)
    # optimize_memory makes next_states the same object; refresh the reference either way.
    buffer.next_states = buffer.states
    buffer.state_keys = [renames.get(key, key) for key in buffer.state_keys]

    # The run's history_offsets name every canonical role, including ones ReBot never
    # recorded. _gather_history indexes storage directly, so an offset on an absent
    # column is a KeyError at the first sample rather than an absent slot. Prune them
    # here; RoleAlignedBuffer.decorate supplies the absent role as a padded placeholder.
    if getattr(buffer, "history_offsets", None):
        servable = set(buffer.states) | {ACTION} | set(getattr(buffer, "complementary_info", {}))
        pruned = {key: offsets for key, offsets in buffer.history_offsets.items() if key in servable}
        dropped = sorted(set(buffer.history_offsets) - set(pruned))
        if dropped:
            logger.info("[RoleAdapter] no ReBot column for history keys %s; skipped", dropped)
        buffer.history_offsets = pruned or None
    logger.info("[RoleAdapter] ReBot columns renamed: %s", renames)
    if f"depth.{depth_role}.depth" in buffer.complementary_info:
        pass  # already canonical
    return renames


class RoleAlignedBuffer:
    """Wrap a ReBot buffer so its batches carry the full canonical role set.

    Adds the roles ReBot never recorded as *absent* slots (zeros plus a False presence
    flag), and the identity columns the diverse side supplies, so the two batches
    concatenate key for key.
    """

    def __init__(
        self,
        buffer,
        *,
        camera_roles: tuple[str, ...] = CANONICAL_CAMERA_ROLES,
        action_layout_id: int,
        source_id: int = SOURCE_IDS["rebot"],
        image_size: tuple[int, int] = (480, 640),
        depth_role: str = "wrist_0",
        depth_intrinsics: tuple[float, float, float, float] | None = None,
        align: bool = True,
    ) -> None:
        self.buffer = buffer
        self.camera_roles = tuple(camera_roles)
        self.action_layout_id = int(action_layout_id)
        self.source_id = int(source_id)
        self.image_size = tuple(image_size)
        self.depth_role = depth_role
        # ReBot's own camera calibration. A mixed batch back-projects two cameras, so the
        # row for these samples has to travel with them.
        self.depth_intrinsics = depth_intrinsics
        if align:
            align_rebot_buffer(buffer)
        self.present_roles = tuple(
            role for role in self.camera_roles if f"{OBS_IMAGES}.{role}" in buffer.states
        )
        self.absent_roles = tuple(role for role in self.camera_roles if role not in self.present_roles)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def size(self) -> int:
        return self.buffer.size

    def sample(self, batch_size: int, action_chunk_size: int = 30):
        batch = self.buffer.sample(batch_size, action_chunk_size=action_chunk_size)
        return self.decorate(batch)

    def decorate(self, batch) -> Any:
        size = int(batch["reward"].shape[0])
        device = batch[ACTION].device
        height, width = self.image_size
        history_slots = None
        for role in self.present_roles:
            key = f"history.{OBS_IMAGES}.{role}"
            if key in batch["state"]:
                history_slots = int(batch["state"][key].shape[1])
                break

        for role in self.absent_roles:
            key = f"{OBS_IMAGES}.{role}"
            zeros = torch.zeros((size, 3, height, width), dtype=torch.uint8, device=device)
            batch["state"][key] = zeros
            batch["next_state"][key] = zeros
            if history_slots:
                batch["state"][f"history.{key}"] = torch.zeros(
                    (size, history_slots, 3, height, width), dtype=torch.uint8, device=device
                )
                batch["state"][f"history.{key}_is_pad"] = torch.ones(
                    (size, history_slots), dtype=torch.bool, device=device
                )

        info = batch.get("complementary_info")
        if info is None:
            info = {}
            batch["complementary_info"] = info
        present = torch.tensor(
            [role in self.present_roles for role in self.camera_roles], device=device
        )
        info["camera_is_present"] = present[None].expand(size, -1).contiguous()
        for index, role in enumerate(self.camera_roles):
            info[f"camera_is_present.{OBS_IMAGES}.{role}"] = present[index].expand(size).contiguous()
        if history_slots:
            info["history.camera_is_present"] = (
                present[None, :, None].expand(size, len(self.camera_roles), history_slots).contiguous()
            )
        # ReBot images are stored at the model frame with no letterboxing, so the whole
        # frame is sensor: the valid box is the frame itself for present roles, empty
        # for absent ones.
        box = torch.zeros((size, len(self.camera_roles), 4), dtype=torch.int16, device=device)
        for index, role in enumerate(self.camera_roles):
            if role in self.present_roles:
                box[:, index] = torch.tensor([0, 0, height, width], dtype=torch.int16, device=device)
        info["image_valid_box"] = box
        info["action_layout_id"] = torch.full((size,), self.action_layout_id, dtype=torch.long, device=device)
        info["source_id"] = torch.full((size,), self.source_id, dtype=torch.long, device=device)
        # -1, not 0: the diverse half numbers its episodes from zero, and padding these
        # rows with a real-looking index would make "episode 0" look enormous in the
        # mixture telemetry.
        info["episode_position"] = torch.full((size,), -1, dtype=torch.long, device=device)
        quality = info.get("metadata_quality")
        if quality is not None:
            # ReBot's quality is human-reviewed where it exists; -1 is its unknown sentinel.
            info["metadata_quality_is_valid"] = torch.as_tensor(quality).reshape(-1) >= 0

        depth_key = f"depth.{self.depth_role}.depth"
        if depth_key in info:
            info[f"{depth_key}_is_present"] = torch.ones((size,), dtype=torch.bool, device=device)
            if self.depth_intrinsics is not None:
                info[f"depth.{self.depth_role}.intrinsics"] = (
                    torch.tensor(self.depth_intrinsics, dtype=torch.float32, device=device)
                    .reshape(1, 4)
                    .expand(size, -1)
                    .contiguous()
                )
            history_depth = info.get(f"history.{depth_key}")
            if history_depth is not None:
                info[f"history.{depth_key}_is_present"] = torch.ones(
                    (size, int(history_depth.shape[1])), dtype=torch.bool, device=device
                )
        return batch

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
        action_chunk_size: int = 30,
    ):
        inner = self.buffer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
            action_chunk_size=action_chunk_size,
        )
        for batch in inner:
            yield self.decorate(batch)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.buffer, name)
