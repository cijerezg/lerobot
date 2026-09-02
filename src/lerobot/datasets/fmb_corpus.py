#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""FMB source-native views and a no-copy federation with the diverse corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.diverse_corpus import (
    FUTURE_FPS,
    FUTURE_POINTS,
    HISTORY_OFFSETS_S,
    DiverseCorpus,
    time_stratified_indices,
)
from lerobot.datasets.diverse_pilot import sample_action_chunk
from lerobot.utils.depth_gripper_events import DEPTH_GRIPPER_EVENT_TARGET_KEYS

# Last point of the future chunk, matching build_corpus.FUTURE_END_S.
FUTURE_END_S = (FUTURE_POINTS - 1) / FUTURE_FPS


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@dataclass(frozen=True)
class FMBCorpusEpisode:
    """One converted FMB episode with source-native continuous arrays."""

    root: Path
    record: dict[str, Any]

    @property
    def native_rate_hz(self) -> float:
        return float(self.record["nominal_fps"])

    @property
    def cameras(self) -> tuple[str, ...]:
        return ("side_1", "side_2", "wrist_1")

    def array(self, name: str) -> np.ndarray:
        return np.load(self.root / name, mmap_mode="r")

    @property
    def timestamps(self) -> np.ndarray:
        return self.array("timestamp_s.npy")

    @property
    def state(self) -> np.ndarray:
        """Measured Franka joints then gripper -- the same 8 slots DROID records.

        FMB also stores ``tcp_pose.npy`` (xyz + unit quaternion), and the corpus used
        to read state from it. It does not any more: the recorded ``actions.npy`` is a
        *normalized* [-1, 1] Cartesian delta whose scale factor appears nowhere in the
        release, so it can be neither compared against an absolute pose nor converted
        back to one. The joints, on the other hand, are measured directly and share
        DROID's convention (verified: FMB's per-joint range sits inside DROID's, with
        matching signs on the elbow and wrist), so FMB reduces to a second Franka
        rather than a coordinate system of its own.
        """
        return np.concatenate([self.array("q.npy"), self.array("gripper_pose.npy")[:, None]], axis=1)

    @property
    def actions(self) -> np.ndarray:
        """The measured joint trajectory, copied from state -- see `state`.

        Sampling this over the chunk's future offsets yields an absolute joint-position
        target, the same ``copy_state`` exception RoboChallenge takes. FMB's own
        commanded actions are unrecoverable and are not used.
        """
        return self.state

    def frames(self, camera: str, indices: list[int] | np.ndarray) -> np.ndarray:
        if camera not in self.cameras:
            raise KeyError(f"no retained FMB RGB camera {camera!r}")
        return np.asarray(self.array(f"{camera}_rgb.npy")[np.asarray(indices)])

    def wrist_depth(self, indices: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(self.array("wrist_1_depth_z16.npy")[np.asarray(indices)])
        return raw, (raw != 0) & (raw != 65_535)

    def depth_gripper_event_targets(self, index: int) -> dict[str, np.float32]:
        targets = {}
        for key in DEPTH_GRIPPER_EVENT_TARGET_KEYS:
            values = self.array(f"{key}.npy")
            if values.shape != (int(self.record["frame_count"]),) or values.dtype != np.float32:
                raise ValueError(f"Invalid FMB {key} array in {self.root}")
            value = np.float32(values[index])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid FMB {key} value at timestep {index} in {self.root}")
            targets[key] = value
        return targets


class FMBCorpus:
    """Actor and critic indexes over one non-duplicated FMB production store."""

    def __init__(self, root: str | Path, *, actor_view: str = "actor_anchors_5hz.jsonl") -> None:
        self.root = Path(root)
        self.episodes_index = _read_jsonl(self.root / "episodes.jsonl")
        self._records = {str(row["episode_id"]): row for row in self.episodes_index}
        self._actor_view_name = actor_view
        self._actor_rows: list[dict[str, Any]] | None = None
        self._critic_rows: list[dict[str, Any]] | None = None

    @lru_cache(maxsize=64)  # noqa: B019 - bounded by the production episode count
    def episode(self, episode_id: str) -> FMBCorpusEpisode:
        return FMBCorpusEpisode(self.root / "episodes" / episode_id, self._records[episode_id])

    def _interval_at(self, episode_id: str, timestep: int) -> dict[str, Any]:
        return next(
            item
            for item in self._records[episode_id]["primitive_intervals"]
            if int(item["start_timestep"]) <= timestep < int(item["end_timestep_exclusive"])
        )

    def _actor_row(self, source_row: dict[str, Any], *, anchor_index: int = 0) -> dict[str, Any]:
        """Normalize one FMB anchor onto the common corpus actor-row contract.

        Every field the common builder writes is supplied here, derived from FMB's own
        record rather than asserted, so a federated training loop can read one key set
        across both stores. Fields whose FMB meaning genuinely differs keep an
        FMB-specific value rather than borrowing the common vocabulary.
        """
        row = dict(source_row)
        episode_id = str(row["episode_id"])
        anchor = int(row["anchor_timestep"])
        anchor_s = float(row["anchor_s_nominal"])
        record = self._records[episode_id]
        fps = float(record["nominal_fps"])
        intervals = record["primitive_intervals"]
        interval_index, interval = next(
            (position, item)
            for position, item in enumerate(intervals)
            if int(item["start_timestep"]) <= anchor < int(item["end_timestep_exclusive"])
        )
        mistakes = interval.get("mistake_events", [])
        recoveries = interval.get("recovery_events", [])

        def _spans(events: list[dict[str, Any]]) -> bool:
            return any(
                int(event["start_timestep"]) <= anchor < int(event["end_timestep_exclusive"])
                for event in events
            )

        in_mistake = row.get("mistake", _spans(mistakes))
        history_frames = row.get(
            "history_frames",
            [int(round((anchor_s + offset) * fps)) for offset in HISTORY_OFFSETS_S],
        )
        # FMB has no source timestamps; frames sit on the derived nominal grid, so the
        # error between a requested history offset and the frame actually used is exact.
        timing_error = max(
            abs(float(frame) / fps - (anchor_s + float(offset)))
            for frame, offset in zip(history_frames, HISTORY_OFFSETS_S, strict=True)
        )
        interpolation = row.get("interpolation_provenance") or {}
        future_points = int(interpolation.get("target_points", FUTURE_POINTS))
        future_rate_hz = float(interpolation.get("target_rate_hz", FUTURE_FPS))
        start_s = float(interval["start_s_nominal"])
        end_s = float(interval["end_s_nominal_exclusive"])
        # Nominal-grid fields are optional: derive them when a store predates them.
        future_end_s = float(row.get("future_end_s_nominal", anchor_s + FUTURE_END_S))

        row.update(
            corpus_key="fmb",
            source="fmb",
            component="single_object_manipulation",
            embodiment="Franka",
            anchor_s=anchor_s,
            anchor_index=anchor_index,
            anchor_frame=anchor,
            history_frames=history_frames,
            max_observation_timing_error_s=timing_error,
            native_rate_hz=fps,
            future_points=future_points,
            future_rate_hz=future_rate_hz,
            future_end_s=future_end_s,
            future_inside_subtask=bool(
                row.get("primitive_conditioned_future_supported", future_end_s <= end_s + 1e-9)
            ),
            subtask_interval_index=interval_index,
            time_since_subtask_start_s=anchor_s - start_s,
            time_to_subtask_end_s=end_s - anchor_s,
            subtask=row.get("subtask", interval["normalized_description"]),
            quality=row.get("quality", interval.get("quality")),
            quality_provenance=interval.get("quality_provenance"),
            mistake=in_mistake,
            retention_reason=(
                "informative_mistake"
                if in_mistake
                else "recovery"
                if _spans(recoveries)
                else "useful_motion"
            ),
            retained=row.get("retained", True),
            rejection_reasons=row.get("rejection_reasons", []),
            action_source="copy_state",
            views_copy_source_arrays=False,
        )
        return row

    def actor_anchors(
        self, *, split: str | None = None, source: str | None = None, retained_only: bool = True
    ) -> list[dict[str, Any]]:
        if self._actor_rows is None:
            counters: dict[str, int] = {}
            built = []
            for row in _read_jsonl(self.root / self._actor_view_name):
                episode_id = str(row["episode_id"])
                index = counters.get(episode_id, 0)
                counters[episode_id] = index + 1
                built.append(self._actor_row(row, anchor_index=index))
            self._actor_rows = built
        rows = self._actor_rows
        if retained_only:
            rows = [row for row in rows if row["retained"]]
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if source is not None:
            rows = [row for row in rows if row["source"] == source]
        return rows

    def _critic_row(self, source_row: dict[str, Any]) -> dict[str, Any]:
        """Normalize one FMB primitive interval onto the common critic-row contract."""
        row = dict(source_row)
        record = self._records[str(row["episode_id"])]
        fps = float(record["nominal_fps"])
        frame_count = int(record["frame_count"])
        start_timestep = int(row["start_timestep"])
        end_timestep = int(row["end_timestep_exclusive"])
        row.update(
            corpus_key="fmb",
            source="fmb",
            component="single_object_manipulation",
            embodiment="Franka",
            duration_s=float(row["duration_s_nominal"]),
            start_s=float(row["start_s_nominal"]),
            end_s=float(row["end_s_nominal_exclusive"]),
            end_s_exclusive=float(row["end_s_nominal_exclusive"]),
            native_rate_hz=fps,
            action_source="copy_state",
            # FMB boundaries are source-native primitive runs, not reviewed subtask
            # transitions, so they keep their own vocabulary rather than borrowing the
            # common corpus's "subtask_transition".
            start_boundary=("episode_start" if start_timestep == 0 else "primitive_transition"),
            end_boundary=("episode_end" if end_timestep >= frame_count else "primitive_transition"),
            end_boundary_provenance=row.get("boundary_provenance"),
            quality_values=[row["quality"]] if row.get("quality") is not None else [],
            pause_events=row.get("pause_events", []),
            episode_outcome="unknown",
            outcome_provenance="unknown",
            views_copy_source_arrays=False,
        )
        return row

    def critic_intervals(
        self, *, split: str | None = None, source: str | None = None, eligible_only: bool = True
    ) -> list[dict[str, Any]]:
        if self._critic_rows is None:
            self._critic_rows = [
                self._critic_row(row) for row in _read_jsonl(self.root / "critic_intervals.jsonl")
            ]
        rows = self._critic_rows
        if eligible_only:
            rows = [row for row in rows if row["critic_eligible"]]
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if source is not None:
            rows = [row for row in rows if row["source"] == source]
        return rows

    def actor_sample(self, row: dict[str, Any], *, cameras: bool = True) -> dict[str, Any]:
        episode = self.episode(str(row["episode_id"]))
        history = np.asarray(row["history_frames"], dtype=np.int64)
        chunk = sample_action_chunk(
            episode.timestamps,
            episode.actions,
            float(row["anchor_s"]),
            native_rate_hz=episode.native_rate_hz,
        )
        sample = {
            "episode_id": row["episode_id"],
            "split": row["split"],
            "source": "fmb",
            "embodiment": "Franka",
            "anchor_s": float(row["anchor_s"]),
            "observation.state": episode.state[history],
            "observation.state_semantics": ["measured_franka_joint_positions", "gripper_pose"],
            "observation.timestamps": episode.timestamps[history],
            "action": chunk.values,
            "action.timestamps": chunk.target_timestamps,
            "action.interpolated_mask": chunk.interpolated_mask,
            "action.source_indices": chunk.source_indices,
            "action.source_timestamps": chunk.source_timestamps,
            "action_source": row["action_source"],
            "annotation.subtask": row["subtask"],
            "annotation.quality": row["quality"],
            "annotation.mistake": row["mistake"],
            "views_copy_source_arrays": False,
            **episode.depth_gripper_event_targets(int(row["anchor_timestep"])),
        }
        if cameras:
            sample["observation.images"] = {
                camera: episode.frames(camera, history) for camera in episode.cameras
            }
            depth, valid = episode.wrist_depth(history)
            sample["observation.depth.wrist_1_z16"] = depth
            sample["observation.depth.wrist_1_valid_mask"] = valid
            sample["observation.depth.units_mm_per_level"] = 0.1
            sample["observation.depth.scale_provenance"] = "user_authorized_same_D405_model_assumption"
        return sample

    def critic_sample(
        self,
        row: dict[str, Any],
        *,
        action_points: int | None = None,
        observation_frames: int = 0,
    ) -> dict[str, Any]:
        episode = self.episode(str(row["episode_id"]))
        start, end = int(row["start_timestep"]), int(row["end_timestep_exclusive"])
        timestamps = episode.timestamps[start:end]
        actions = episode.actions[start:end]
        sample = {
            "episode_id": row["episode_id"],
            "split": row["split"],
            "source": "fmb",
            "embodiment": "Franka",
            "subtask": row["normalized_description"],
            "classification": row["classification"],
            "quality": row["quality"],
            "quality_provenance": row["quality_provenance"],
            "subtask_outcome": row["subtask_outcome"],
            "episode_outcome": "unknown",
            "action": actions,
            "action.timestamps": timestamps,
            "observation.state": episode.state[start:end],
            "observation.state_semantics": ["measured_franka_joint_positions", "gripper_pose"],
            "native_action_samples": end - start,
            "duration_s": float(row["duration_s"]),
            "pause_events": row["pause_events"],
            "mistake_events": row["mistake_events"],
            "recovery_events": row["recovery_events"],
            "views_copy_source_arrays": False,
        }
        if action_points:
            indices, mask = time_stratified_indices(
                timestamps,
                actions,
                action_points,
                event_times=[],
                interval_start_s=float(row["start_s"]),
            )
            padded_actions = np.zeros((action_points, actions.shape[1]), dtype=actions.dtype)
            padded_times = np.zeros(action_points, dtype=timestamps.dtype)
            padded_actions[: len(indices)], padded_times[: len(indices)] = (
                actions[indices],
                timestamps[indices],
            )
            sample.update(
                {
                    "action.subsampled": padded_actions,
                    "action.subsampled_timestamps": padded_times,
                    "action.subsampled_indices": indices,
                    "action.subsampled_mask": mask,
                }
            )
        if observation_frames:
            frame_indices = np.unique(
                np.linspace(start, end - 1, observation_frames).round().astype(np.int64)
            )
            sample["observation.images"] = {
                camera: episode.frames(camera, frame_indices) for camera in episode.cameras
            }
            depth, valid = episode.wrist_depth(frame_indices)
            sample["observation.depth.wrist_1_z16"] = depth
            sample["observation.depth.wrist_1_valid_mask"] = valid
            sample["observation.image_frames"] = frame_indices
        return sample


class FederatedDiverseCorpus:
    """One logical corpus backed by common and FMB stores without copying either."""

    def __init__(
        self,
        common_root: str | Path,
        fmb_root: str | Path,
        *,
        actor_view: str = "actor_anchors_5hz.jsonl",
    ) -> None:
        self.common = DiverseCorpus(common_root, actor_view=actor_view)
        self.fmb = FMBCorpus(fmb_root, actor_view=actor_view)
        common_ids = {str(row["episode_id"]) for row in self.common.episodes_index}
        fmb_ids = {str(row["episode_id"]) for row in self.fmb.episodes_index}
        collisions = common_ids & fmb_ids
        if collisions:
            raise ValueError(f"episode IDs collide across stores: {sorted(collisions)[:3]}")

    def actor_anchors(
        self, *, split: str | None = None, source: str | None = None, retained_only: bool = True
    ) -> list[dict[str, Any]]:
        rows = [
            {**row, "corpus_key": "common"}
            for row in self.common.actor_anchors(split=split, retained_only=retained_only)
        ]
        rows.extend(self.fmb.actor_anchors(split=split, retained_only=retained_only))
        return rows if source is None else [row for row in rows if row["source"] == source]

    def critic_intervals(
        self, *, split: str | None = None, source: str | None = None, eligible_only: bool = True
    ) -> list[dict[str, Any]]:
        rows = [
            {**row, "corpus_key": "common"}
            for row in self.common.critic_intervals(split=split, eligible_only=eligible_only)
        ]
        rows.extend(self.fmb.critic_intervals(split=split, eligible_only=eligible_only))
        return rows if source is None else [row for row in rows if row["source"] == source]

    def actor_sample(self, row: dict[str, Any], *, cameras: bool = True) -> dict[str, Any]:
        corpus = self.fmb if row["corpus_key"] == "fmb" else self.common
        return corpus.actor_sample(row, cameras=cameras)

    def critic_sample(
        self,
        row: dict[str, Any],
        *,
        action_points: int | None = None,
        observation_frames: int = 0,
    ) -> dict[str, Any]:
        corpus = self.fmb if row["corpus_key"] == "fmb" else self.common
        return corpus.critic_sample(row, action_points=action_points, observation_frames=observation_frames)
