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

"""Memmap cache for the diverse actor view (integration plan phase C).

Decoding the corpus live costs ~1.7 s per sample, so training needs a cache. The naive
cache would materialize four frames per anchor per camera -- which is exactly the 7x
duplication the source-native rebuild was done to escape (DIVERSE_ROBOT_DATASET.md
section 4.3). It would also be wrong twice over: adjacent anchors reference the same
instants, and an anchor's own history frames sit on the same 0.2 s grid as its
neighbours' current frames.

So frames go into per-camera **banks**, one row per distinct (episode, camera, frame),
and each anchor stores four *indices* into that bank. Measured on the corpus, a
robochallenge episode references 600 distinct frames where per-anchor materialization
would write 2,224 -- a 3.7x reduction, and the same frame is decoded once.

Building reads each video linearly instead of seeking per anchor: 8.8 s for one
episode's 600 referenced frames versus 2.2 s for four scattered ones.

The cache is written to a temporary directory and renamed into place only after it
validates, so an interrupted build can never be found by ``find_cache``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.diverse_actor_selection import (
    ACTION_LAYOUTS,
    PACKED_CURRENT_SLOT,
    CAMERA_ROLE_MAP,
    DiverseActorSelection,
    EXPECTED_ANCHORS,
    EXPECTED_EPISODES,
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.datasets.embodiment import embodiment_index
from lerobot.utils.depth_gripper_events import DEPTH_GRIPPER_EVENT_TARGET_KEYS
from lerobot.rl.data_sources.diverse_actor_buffer import (
    SOURCE_IDS,
    DiverseSampleSpec,
    ResizeGeometry,
    _resize_frames,
)

logger = logging.getLogger(__name__)

# 2: added the depth_event_targets anchor column.
CACHE_SCHEMA_VERSION = 2
ABSENT = -1

# ── Depth intrinsics assumption (plan section 1) ─────────────────────────────
# FMB did not retain its cameras' calibration, only that the wrist sensor is a D405 --
# the same conclusion that licensed the 0.1 mm/level scale. So the base intrinsics are
# the locally measured D405 factory calibration at 640x480, and the chain to a training
# pixel is spelled out here rather than left implicit in a constant:
#
#   1. D405 factory calibration, 640x480 rectified (zero distortion).
#   2. FMB stores 256x256. Its aspect ratio is not 4:3, so this is a non-uniform resize
#      of the sensor frame -- fx and fy scale by different factors. This step is an
#      ASSUMPTION about how FMB produced its images, not a measurement.
#   3. The training resize (ResizeGeometry): one uniform scale plus a centring pad.
#
# Every number and the assumption string are written into cache metadata, so a run can
# be audited against them and a future measurement can invalidate exactly this step.
D405_BASE_INTRINSICS = (394.9832, 394.9832, 322.5604, 238.6966)
D405_BASE_SIZE = (480, 640)
FMB_DEPTH_INTRINSICS_ASSUMPTION = (
    "d405_factory_calibration_at_480x640_then_non_uniform_resize_to_fmb_256x256"
)


def source_depth_intrinsics(
    source_size: tuple[int, int],
    *,
    base_intrinsics: tuple[float, float, float, float] = D405_BASE_INTRINSICS,
    base_size: tuple[int, int] = D405_BASE_SIZE,
) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) for the stored depth image, before the training resize."""
    fx, fy, cx, cy = base_intrinsics
    scale_y = source_size[0] / base_size[0]
    scale_x = source_size[1] / base_size[1]
    return (fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)


def training_depth_intrinsics(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) a back-projection must use on a cached depth frame."""
    return ResizeGeometry.fit(source_size, target_size).transform_intrinsics(
        source_depth_intrinsics(source_size)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_fingerprint(root: str | Path) -> dict[str, str]:
    """Content hashes of everything the selection reads to decide what exists."""
    root = Path(root)
    files = {
        "common_episodes": root / "corpus" / "episodes.jsonl",
        "common_actor_view": root / "corpus" / "actor_anchors_5hz.jsonl",
        "fmb_episodes": root / "fmb" / "episodes.jsonl",
        "fmb_actor_view": root / "fmb" / "actor_anchors_5hz.jsonl",
    }
    return {name: _sha256_file(path) for name, path in files.items()}


def cache_fingerprint(root: str | Path, spec: DiverseSampleSpec, *, anchors: int, episodes: int) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "corpus": corpus_fingerprint(root),
        "selection": {
            "rule": "actor_anchors(split=None, retained_only=True)",
            "anchors": int(anchors),
            "episodes": int(episodes),
        },
        "spec": spec.fingerprint(),
        "camera_map": {source: dict(mapping) for source, mapping in sorted(CAMERA_ROLE_MAP.items())},
        "action_layouts": [
            [layout.index, layout.name, layout.source, layout.embodiment, layout.dim]
            for layout in ACTION_LAYOUTS
        ],
        "depth": {
            "units_mm_per_level": 0.1,
            "scale_provenance": "user_authorized_same_D405_model_assumption",
            "intrinsics_assumption": FMB_DEPTH_INTRINSICS_ASSUMPTION,
            "base_intrinsics": list(D405_BASE_INTRINSICS),
            "base_size": list(D405_BASE_SIZE),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ── Layout ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Column:
    name: str
    shape: tuple[int, ...]
    dtype: str


def _anchor_columns(spec: DiverseSampleSpec) -> list[_Column]:
    width = spec.max_width
    slots = spec.num_history + 1  # history slots plus the current observation
    roles = len(spec.camera_roles)
    return [
        _Column("state", (width,), "float32"),
        _Column("history_state", (spec.num_history, width), "float32"),
        _Column("action", (spec.action_horizon, width), "float32"),
        _Column("native_width", (), "int8"),
        _Column("timestamps", (slots,), "float64"),
        # [source_id, episode_position, anchor_index, action_layout_id, embodiment_index]
        _Column("identity", (5,), "int32"),
        _Column("metadata", (2,), "float32"),  # [quality, mistake]
        _Column("camera_present", (roles,), "bool"),
        _Column("image_valid_box", (roles, 4), "int16"),
        _Column("frame_slot", (roles, slots), "int32"),
        _Column("depth_slot", (slots,), "int32"),
        _Column("depth_units", (), "float32"),
        # [close, open] future gripper-event targets; FMB records them, everything else
        # has no depth and is excluded from that loss by its presence mask.
        _Column("depth_event_targets", (2,), "float32"),
    ]


def _open_memmap(path: Path, dtype: str, shape: tuple[int, ...], mode: str) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(str(path), dtype=np.dtype(dtype), mode=mode, shape=shape)


# ── Plan (pass one: no decoding) ─────────────────────────────────────────────


@dataclass
class _EpisodePlan:
    episode_id: str
    row_indices: list[int]
    # The distinct episode frames any anchor references, ascending. Every present role
    # and the depth stream index into this one list, so a frame's bank row is
    # offset + position_in_referenced.
    referenced: np.ndarray
    roles: tuple[str, ...]
    offsets: dict[str, int]
    has_depth: bool
    depth_offset: int

    @property
    def lookup(self) -> dict[int, int]:
        return {int(frame): position for position, frame in enumerate(self.referenced)}


def plan_cache(selection: DiverseActorSelection, spec: DiverseSampleSpec) -> tuple[list[_EpisodePlan], dict[str, int], int]:
    """Which frames each episode contributes to each bank, and where they land.

    Deterministic: episodes in sorted id order, frames in ascending order inside an
    episode. Bank offsets follow from that, so two builds of the same corpus produce
    byte-identical indices.
    """
    slots = [*spec.history_slots, PACKED_CURRENT_SLOT]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(selection.rows):
        grouped[str(row["episode_id"])].append(index)

    plans: list[_EpisodePlan] = []
    bank_sizes: dict[str, int] = dict.fromkeys(spec.camera_roles, 0)
    depth_size = 0
    for episode_id in sorted(grouped):
        row_indices = grouped[episode_id]
        rows = [selection.rows[index] for index in row_indices]
        referenced = np.unique(
            np.asarray([[row["history_frames"][slot] for slot in slots] for row in rows], dtype=np.int64)
        )
        roles = tuple(role for role in spec.camera_roles if role in rows[0]["camera_roles"])
        offsets: dict[str, int] = {}
        for role in roles:
            offsets[role] = bank_sizes[role]
            bank_sizes[role] += len(referenced)
        has_depth = bool(rows[0]["has_depth"]) and spec.load_depth
        depth_offset = depth_size
        depth_size += len(referenced) if has_depth else 0
        plans.append(
            _EpisodePlan(
                episode_id=episode_id,
                row_indices=row_indices,
                referenced=referenced,
                roles=roles,
                offsets=offsets,
                has_depth=has_depth,
                depth_offset=depth_offset,
            )
        )
    return plans, bank_sizes, depth_size


# ── Build ────────────────────────────────────────────────────────────────────


_WORKER: dict[str, Any] = {}


def _worker_init(root: str, spec: DiverseSampleSpec, cache_dir: str, bank_sizes: dict[str, int], depth_size: int) -> None:
    selection = select_actor_anchors(open_federated_corpus(root))
    _WORKER["selection"] = selection
    _WORKER["spec"] = spec
    _WORKER["cache_dir"] = Path(cache_dir)
    _WORKER["bank_sizes"] = bank_sizes
    _WORKER["depth_size"] = depth_size
    _WORKER["plans"] = {plan.episode_id: plan for plan in plan_cache(selection, spec)[0]}


def _write_episode(episode_id: str) -> tuple[str, int]:
    return _write_episode_with(
        _WORKER["selection"],
        _WORKER["spec"],
        _WORKER["cache_dir"],
        _WORKER["plans"][episode_id],
        _WORKER["bank_sizes"],
        _WORKER["depth_size"],
    )


def _write_episode_with(
    selection: DiverseActorSelection,
    spec: DiverseSampleSpec,
    cache_dir: Path,
    plan: _EpisodePlan,
    bank_sizes: dict[str, int],
    depth_size: int,
) -> tuple[str, int]:
    """Decode one episode's referenced frames once and fill its anchor rows.

    Every write is to a disjoint slice -- anchors by row index, bank rows by the
    offsets pass one assigned -- so workers never contend.
    """
    rows = [selection.rows[index] for index in plan.row_indices]
    row = rows[0]
    episode_position = selection.episode_ids.index(plan.episode_id)
    corpus = selection.corpus.fmb if row["corpus_key"] == "fmb" else selection.corpus.common
    episode = corpus.episode(plan.episode_id)
    slots = [*spec.history_slots, PACKED_CURRENT_SLOT]
    total_anchors = len(selection.rows)

    columns = {
        column.name: _open_memmap(
            cache_dir / "anchors" / f"{column.name}.bin",
            column.dtype,
            (total_anchors, *column.shape),
            mode="r+",
        )
        for column in _anchor_columns(spec)
    }

    # -- low-dimensional --------------------------------------------------------
    state_all = np.asarray(episode.state, dtype=np.float32)
    timestamps_all = np.asarray(episode.timestamps, dtype=np.float64)
    for position, index in enumerate(plan.row_indices):
        anchor_row = rows[position]
        sample = selection.corpus.actor_sample(anchor_row, cameras=False)
        native = int(state_all.shape[1])
        packed_state = np.asarray(sample["observation.state"], dtype=np.float32)
        columns["state"][index, :native] = packed_state[PACKED_CURRENT_SLOT]
        columns["history_state"][index, :, :native] = packed_state[spec.history_slots]
        columns["action"][index, :, :native] = np.asarray(sample["action"], dtype=np.float32)
        columns["native_width"][index] = native
        columns["timestamps"][index] = np.asarray(sample["observation.timestamps"], dtype=np.float64)[slots]
        columns["identity"][index] = (
            SOURCE_IDS[anchor_row["source"]],
            0,  # filled below from the selection-wide episode order
            int(anchor_row["anchor_index"]),
            int(anchor_row["action_layout_id"]),
            embodiment_index(anchor_row["embodiment"]),
        )
        quality = anchor_row.get("quality")
        columns["metadata"][index] = (
            -1.0 if quality is None else float(quality),
            float(bool(anchor_row["mistake"])),
        )
        columns["frame_slot"][index] = ABSENT
        columns["depth_slot"][index] = ABSENT
        columns["camera_present"][index] = False
        columns["image_valid_box"][index] = 0
        columns["depth_units"][index] = 0.0

    for index in plan.row_indices:
        columns["identity"][index, 1] = episode_position

    # -- frame banks ------------------------------------------------------------
    height, width = spec.image_size
    frame_lookup = plan.lookup
    for role_index, role in enumerate(spec.camera_roles):
        if role not in plan.roles:
            continue
        camera = row["camera_roles"][role]
        wanted = plan.referenced
        decoded = np.asarray(episode.frames(camera, wanted))
        geometry = ResizeGeometry.fit(decoded.shape[1:3], spec.image_size)
        resized = _resize_frames(decoded, geometry, nearest=False)
        bank = _open_memmap(
            cache_dir / "banks" / f"rgb_{role}.bin", "uint8", (bank_sizes[role], 3, height, width), mode="r+"
        )
        offset = plan.offsets[role]
        bank[offset : offset + len(wanted)] = np.ascontiguousarray(resized.transpose(0, 3, 1, 2))
        bank.flush()
        del bank
        for index, anchor_row in zip(plan.row_indices, rows, strict=True):
            columns["camera_present"][index, role_index] = True
            columns["image_valid_box"][index, role_index] = geometry.valid_box
            columns["frame_slot"][index, role_index] = [
                offset + frame_lookup[int(anchor_row["history_frames"][slot])] for slot in slots
            ]

    # -- depth bank -------------------------------------------------------------
    if plan.has_depth:
        depth_height, depth_width = spec.depth_size
        raw, valid = episode.wrist_depth(plan.referenced)
        geometry = ResizeGeometry.fit(raw.shape[1:3], spec.depth_size)
        resized = _resize_frames(np.asarray(raw), geometry, nearest=True)
        resized_valid = _resize_frames(
            np.asarray(valid).astype(np.uint8), geometry, nearest=True
        ).astype(bool) & geometry.pixel_valid_mask()[None]
        depth_bank = _open_memmap(
            cache_dir / "banks" / "depth.bin", "uint16", (depth_size, depth_height, depth_width), mode="r+"
        )
        valid_bank = _open_memmap(
            cache_dir / "banks" / "depth_valid.bin", "bool", (depth_size, depth_height, depth_width), mode="r+"
        )
        offset = plan.depth_offset
        depth_bank[offset : offset + len(plan.referenced)] = resized
        valid_bank[offset : offset + len(plan.referenced)] = resized_valid
        depth_bank.flush()
        valid_bank.flush()
        del depth_bank, valid_bank
        units = float(selection.episode_records[plan.episode_id]["depth"]["depth_units_mm_per_level"])
        for index, anchor_row in zip(plan.row_indices, rows, strict=True):
            columns["depth_slot"][index] = [
                offset + frame_lookup[int(anchor_row["history_frames"][slot])] for slot in slots
            ]
            columns["depth_units"][index] = units
            targets = episode.depth_gripper_event_targets(int(anchor_row["anchor_frame"]))
            columns["depth_event_targets"][index] = [
                float(targets[key]) for key in DEPTH_GRIPPER_EVENT_TARGET_KEYS
            ]

    for column in columns.values():
        column.flush()
    return plan.episode_id, len(plan.row_indices)


def build_cache(
    root: str | Path,
    cache_dir: str | Path,
    spec: DiverseSampleSpec | None = None,
    *,
    selection: DiverseActorSelection | None = None,
    episodes: list[str] | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> Path:
    """Build the diverse cache atomically and return its directory.

    ``episodes`` restricts the build to a subset -- for tests and for the cached/uncached
    equivalence check. A subset build is marked ``partial`` in its metadata and is
    refused by ``open_cache(require_complete=True)``, so a test cache can never be
    mistaken for a training one.
    """
    root = Path(root)
    spec = spec or DiverseSampleSpec()
    selection = selection or select_actor_anchors(open_federated_corpus(root))
    fingerprint = cache_fingerprint(
        root, spec, anchors=len(selection.rows), episodes=len(selection.episode_ids)
    )
    cache_dir = Path(cache_dir)
    final = cache_dir / fingerprint
    if final.exists():
        if not overwrite:
            logger.info("Diverse cache already present at %s", final)
            return final
        shutil.rmtree(final)

    plans, bank_sizes, depth_size = plan_cache(selection, spec)
    if episodes is not None:
        wanted = set(episodes)
        plans = [plan for plan in plans if plan.episode_id in wanted]

    staging = cache_dir / f".building-{fingerprint}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "anchors").mkdir(parents=True)
    (staging / "banks").mkdir(parents=True)

    total_anchors = len(selection.rows)
    for column in _anchor_columns(spec):
        array = _open_memmap(
            staging / "anchors" / f"{column.name}.bin", column.dtype, (total_anchors, *column.shape), "w+"
        )
        # Anchors from unbuilt episodes must not read as valid rows.
        array[...] = ABSENT if column.name in {"frame_slot", "depth_slot", "native_width"} else 0
        array.flush()
        del array
    height, width = spec.image_size
    for role in spec.camera_roles:
        array = _open_memmap(
            staging / "banks" / f"rgb_{role}.bin", "uint8", (bank_sizes[role], 3, height, width), "w+"
        )
        array.flush()
        del array
    if spec.load_depth:
        depth_height, depth_width = spec.depth_size
        for name, dtype in (("depth", "uint16"), ("depth_valid", "bool")):
            array = _open_memmap(
                staging / "banks" / f"{name}.bin", dtype, (depth_size, depth_height, depth_width), "w+"
            )
            array.flush()
            del array

    logger.info(
        "Building diverse cache %s: %d episodes, %d anchors, banks %s, depth rows %d",
        fingerprint,
        len(plans),
        total_anchors,
        bank_sizes,
        depth_size,
    )
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(str(root), spec, str(staging), bank_sizes, depth_size),
        ) as pool:
            for done, (episode_id, count) in enumerate(
                pool.map(_write_episode, [plan.episode_id for plan in plans]), start=1
            ):
                logger.info("[%d/%d] %s (%d anchors)", done, len(plans), episode_id, count)
    else:
        for done, plan in enumerate(plans, start=1):
            _write_episode_with(selection, spec, staging, plan, bank_sizes, depth_size)
            logger.info("[%d/%d] %s (%d anchors)", done, len(plans), plan.episode_id, len(plan.row_indices))

    built_rows = sorted(index for plan in plans for index in plan.row_indices)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "corpus_root": str(root),
        "corpus": corpus_fingerprint(root),
        "spec": spec.fingerprint(),
        "camera_map": {source: dict(mapping) for source, mapping in sorted(CAMERA_ROLE_MAP.items())},
        "selection": {
            "rule": "actor_anchors(split=None, retained_only=True)",
            "anchors": total_anchors,
            "episodes": len(selection.episode_ids),
            "expected_anchors": EXPECTED_ANCHORS,
            "expected_episodes": EXPECTED_EPISODES,
        },
        "partial": episodes is not None,
        "built_episodes": [plan.episode_id for plan in plans],
        "built_rows": len(built_rows),
        "bank_sizes": bank_sizes,
        "depth_rows": depth_size,
        "columns": {
            column.name: {"shape": list(column.shape), "dtype": column.dtype}
            for column in _anchor_columns(spec)
        },
        "resize_geometry": {
            f"{height}x{width}": ResizeGeometry.fit((height, width), spec.image_size).valid_box
            for height, width in sorted({(720, 1280), (480, 640), (256, 256)})
        },
        "depth_scale_provenance": "user_authorized_same_D405_model_assumption",
        "depth_units_mm_per_level": 0.1,
        "depth_intrinsics_assumption": FMB_DEPTH_INTRINSICS_ASSUMPTION,
        "depth_base_intrinsics": list(D405_BASE_INTRINSICS),
        "depth_base_size": list(D405_BASE_SIZE),
        "depth_source_size": [256, 256],
        "depth_training_intrinsics": list(training_depth_intrinsics((256, 256), spec.depth_size)),
        "action_layouts": [
            {"index": layout.index, "name": layout.name, "source": layout.source,
             "embodiment": layout.embodiment, "dim": layout.dim}
            for layout in ACTION_LAYOUTS
        ],
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Validate before publishing: a cache that only half-wrote must not be findable.
    check = _open_memmap(
        staging / "anchors" / "native_width.bin", "int8", (total_anchors,), "r"
    )
    missing = [index for index in built_rows if int(check[index]) <= 0]
    del check
    if missing:
        raise RuntimeError(f"{len(missing)} built anchors have no native width; cache not published.")

    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    logger.info("Diverse cache published at %s", final)
    return final


# ── Read ─────────────────────────────────────────────────────────────────────


class DiverseFrameCache:
    """Read side of the cache: RAM anchor tables, memmapped frame banks."""

    def __init__(self, cache_dir: str | Path, spec: DiverseSampleSpec) -> None:
        self.cache_dir = Path(cache_dir)
        self.metadata = json.loads((self.cache_dir / "metadata.json").read_text(encoding="utf-8"))
        self.spec = spec
        total = int(self.metadata["selection"]["anchors"])
        self.total_anchors = total
        self.partial = bool(self.metadata.get("partial", False))
        self.built_rows = set()

        self.columns: dict[str, np.ndarray] = {}
        for column in _anchor_columns(spec):
            array = _open_memmap(
                self.cache_dir / "anchors" / f"{column.name}.bin",
                column.dtype,
                (total, *column.shape),
                "r",
            )
            # Anchor tables are small; keeping them in RAM removes page faults from the
            # sampling path. The banks stay memmapped because they are not.
            self.columns[column.name] = np.array(array)
            del array

        self.banks: dict[str, np.ndarray] = {}
        height, width = spec.image_size
        for role in spec.camera_roles:
            rows = int(self.metadata["bank_sizes"][role])
            self.banks[role] = _open_memmap(
                self.cache_dir / "banks" / f"rgb_{role}.bin", "uint8", (rows, 3, height, width), "r"
            )
        self.depth_bank = None
        self.depth_valid_bank = None
        depth_rows = int(self.metadata.get("depth_rows", 0))
        if spec.load_depth and depth_rows:
            depth_height, depth_width = spec.depth_size
            self.depth_bank = _open_memmap(
                self.cache_dir / "banks" / "depth.bin", "uint16", (depth_rows, depth_height, depth_width), "r"
            )
            self.depth_valid_bank = _open_memmap(
                self.cache_dir / "banks" / "depth_valid.bin",
                "bool",
                (depth_rows, depth_height, depth_width),
                "r",
            )

    @property
    def built_row_indices(self) -> np.ndarray:
        """Anchor rows this cache actually holds (all of them unless partial)."""
        return np.nonzero(self.columns["native_width"] > 0)[0]

    def sample(self, row_index: int) -> dict[str, Any]:
        """The same dict ``DiverseActorBuffer.load_sample`` returns, read from disk."""
        spec = self.spec
        native = int(self.columns["native_width"][row_index])
        if native <= 0:
            raise KeyError(
                f"anchor row {row_index} is not in this cache "
                f"({'partial build' if self.partial else 'corrupt cache'})."
            )
        history = spec.num_history
        out: dict[str, Any] = {
            "row_index": int(row_index),
            "state": self.columns["state"][row_index, :native],
            "history_state": self.columns["history_state"][row_index, :, :native],
            "action": self.columns["action"][row_index, :, :native],
            "timestamp": self.columns["timestamps"][row_index, -1],
            "history_timestamps": self.columns["timestamps"][row_index, :history],
            "native_width": native,
            "images": {},
            "image_geometry": {},
            "camera_roles": [],
        }
        for role_index, role in enumerate(spec.camera_roles):
            if not bool(self.columns["camera_present"][row_index, role_index]):
                continue
            out["camera_roles"].append(role)
            if not spec.load_images:
                continue
            slots = self.columns["frame_slot"][row_index, role_index]
            # (T_h + 1, 3, H, W), current last -- the same layout the uncached path emits.
            out["images"][role] = np.asarray(self.banks[role][slots])
            top, left, box_height, box_width = (
                int(value) for value in self.columns["image_valid_box"][row_index, role_index]
            )
            out["image_geometry"][role] = _BoxGeometry(top, left, box_height, box_width, *spec.image_size)

        depth_slots = self.columns["depth_slot"][row_index]
        if spec.load_depth and self.depth_bank is not None and int(depth_slots[0]) != ABSENT:
            out["depth"] = np.asarray(self.depth_bank[depth_slots])
            out["depth_valid"] = np.asarray(self.depth_valid_bank[depth_slots])
            out["depth_units_mm_per_level"] = float(self.columns["depth_units"][row_index])
            out["depth_scale_provenance"] = self.metadata["depth_scale_provenance"]
            out["depth_intrinsics"] = tuple(self.metadata["depth_training_intrinsics"])
            out["depth_event_targets"] = self.columns["depth_event_targets"][row_index]
        return out


@dataclass(frozen=True)
class _BoxGeometry:
    """The valid box read back from the cache, in the shape collation expects."""

    top: int
    left: int
    scaled_height: int
    scaled_width: int
    height: int
    width: int

    @property
    def valid_box(self) -> tuple[int, int, int, int]:
        return (self.top, self.left, self.scaled_height, self.scaled_width)


def find_cache(
    root: str | Path,
    cache_dir: str | Path,
    spec: DiverseSampleSpec,
    *,
    anchors: int,
    episodes: int,
    require_complete: bool = True,
) -> Path | None:
    """The cache directory for this corpus and spec, or None."""
    fingerprint = cache_fingerprint(root, spec, anchors=anchors, episodes=episodes)
    candidate = Path(cache_dir) / fingerprint
    metadata_path = candidate / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != fingerprint:
        return None
    if require_complete and metadata.get("partial", False):
        return None
    return candidate


def resolve_cache(
    root: str | Path,
    cache_dir: str | Path | None,
    spec: DiverseSampleSpec,
    selection: DiverseActorSelection,
    *,
    cache_policy: str = "fallback",
) -> DiverseFrameCache | None:
    """The cache this run should use, honouring ``cache_policy``.

    ``require`` is not "a directory exists": a cache whose fingerprint does not match
    this corpus, this sample contract and this camera map is the wrong cache, and a
    partial one is a test artifact. Both are misses, and under ``require`` a miss is a
    startup error rather than 28 hours of video decode nobody asked for.
    """
    if cache_policy not in {"require", "fallback", "off"}:
        raise ValueError(f"cache_policy must be 'require', 'fallback' or 'off', got {cache_policy!r}.")
    if cache_policy == "off":
        return None
    found = (
        None
        if cache_dir is None
        else find_cache(
            root,
            cache_dir,
            spec,
            anchors=len(selection.rows),
            episodes=len(selection.episode_ids),
            require_complete=True,
        )
    )
    if found is None:
        if cache_policy == "require":
            raise FileNotFoundError(
                f"No complete diverse replay cache for {root} under {cache_dir!r} matching this "
                f"sample contract (fingerprint "
                f"{cache_fingerprint(root, spec, anchors=len(selection.rows), episodes=len(selection.episode_ids))}). "
                "Build it: python -m lerobot.rl.data_sources.diverse_actor_cache "
                f"--root {root} --cache-dir {cache_dir} --workers 8"
            )
        logger.warning(
            "No diverse cache under %s; falling back to live video decode (~1.7 s/sample).", cache_dir
        )
        return None
    logger.info("Diverse cache found at %s", found)
    return DiverseFrameCache(found, spec)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the diverse actor memmap cache.")
    parser.add_argument("--root", default="outputs/diverse_robot_dataset")
    parser.add_argument("--cache-dir", default="outputs/buffer_cache-diverse")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=0, help="Build only the first N episodes (a partial cache)."
    )
    args = parser.parse_args()

    selection = select_actor_anchors(open_federated_corpus(args.root))
    spec = DiverseSampleSpec()
    episodes = None
    if args.episodes:
        plans, _, _ = plan_cache(selection, spec)
        episodes = [plan.episode_id for plan in plans[: args.episodes]]
    path = build_cache(
        args.root,
        args.cache_dir,
        spec,
        selection=selection,
        episodes=episodes,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()
