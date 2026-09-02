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

"""Read the diverse-robot source-native corpus as an actor view or a critic view.

The corpus stores each accepted episode once: the continuous native timeline, the
per-camera video at the native rate, the reviewed annotations, and the provenance.
`actor_anchors_*.jsonl` and `critic_intervals.jsonl` are indexes into that timeline,
not copies of it, so the two views never disagree about what was recorded.

The actor view returns the established packed sample - seven observations one second
apart and thirty future actions at 30 Hz. The critic view returns a whole reviewed
subtask: every native action from its start through its end, with subsampling left to
the caller so a fixed-size critic input never destroys the stored sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.diverse_pilot import sample_action_chunk

HISTORY_OFFSETS_S = np.asarray([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
FUTURE_POINTS = 30
FUTURE_FPS = 30.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@dataclass(frozen=True)
class CorpusEpisode:
    """One accepted episode: continuous native arrays plus its reviewed record."""

    root: Path
    record: dict[str, Any]

    @property
    def episode_id(self) -> str:
        return str(self.record["episode_id"])

    @property
    def native_rate_hz(self) -> float:
        return float(self.record["native_rate_hz"])

    @property
    def cameras(self) -> list[str]:
        return [camera["name"] for camera in self.record["cameras"]]

    def array(self, name: str) -> np.ndarray:
        return np.load(self.root / f"{name}.npy")

    @property
    def timestamps(self) -> np.ndarray:
        return self.array("timestamp_s")

    @property
    def state(self) -> np.ndarray:
        return self.array("state")

    @property
    def action(self) -> np.ndarray:
        return self.array("action")

    def video_path(self, camera: str) -> Path:
        for entry in self.record["cameras"]:
            if entry["name"] == camera:
                return self.root / entry["path"]
        raise KeyError(f"{self.episode_id} has no camera {camera!r}")

    def frames(self, camera: str, frame_indices: list[int] | np.ndarray) -> np.ndarray:
        """Decode the requested frames as uint8 [N, H, W, 3]."""
        from lerobot.datasets.video_utils import decode_video_frames

        fps = self.native_rate_hz
        timestamps = [float(index) / fps for index in np.asarray(frame_indices).reshape(-1)]
        decoded = decode_video_frames(
            self.video_path(camera),
            timestamps,
            tolerance_s=0.51 / fps,
            backend="pyav",
            return_uint8=True,
        )
        return decoded.permute(0, 2, 3, 1).numpy()


class DiverseCorpus:
    """Index over the corpus directory and its two views."""

    def __init__(self, root: str | Path, *, actor_view: str = "actor_anchors_5hz.jsonl") -> None:
        self.root = Path(root)
        self.episodes_index = _read_jsonl(self.root / "episodes.jsonl")
        self._actor_view_name = actor_view
        self._actor_rows: list[dict[str, Any]] | None = None
        self._critic_rows: list[dict[str, Any]] | None = None

    @lru_cache(maxsize=64)  # noqa: B019 - bounded by the corpus episode count
    def episode(self, episode_id: str) -> CorpusEpisode:
        root = self.root / "episodes" / episode_id
        record = json.loads((root / "episode.json").read_text(encoding="utf-8"))
        return CorpusEpisode(root=root, record=record)

    def actor_anchors(
        self, *, split: str | None = None, source: str | None = None, retained_only: bool = True
    ) -> list[dict[str, Any]]:
        if self._actor_rows is None:
            self._actor_rows = _read_jsonl(self.root / self._actor_view_name)
        rows = self._actor_rows
        if retained_only:
            rows = [row for row in rows if row["retained"]]
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if source is not None:
            rows = [row for row in rows if row["source"] == source]
        return rows

    def critic_intervals(
        self, *, split: str | None = None, source: str | None = None, eligible_only: bool = True
    ) -> list[dict[str, Any]]:
        if self._critic_rows is None:
            self._critic_rows = _read_jsonl(self.root / "critic_intervals.jsonl")
        rows = self._critic_rows
        if eligible_only:
            rows = [row for row in rows if row["critic_eligible"]]
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if source is not None:
            rows = [row for row in rows if row["source"] == source]
        return rows

    # -- actor view ------------------------------------------------------------------

    def actor_sample(self, row: dict[str, Any], *, cameras: bool = True) -> dict[str, Any]:
        """Seven observations at one-second spacing and a 30 Hz one-second future."""
        episode = self.episode(row["episode_id"])
        timestamps = episode.timestamps
        state = episode.state
        action = episode.action
        history_frames = np.asarray(row["history_frames"], dtype=np.int64)
        # Share the packed view's chunk rule so a 30 Hz source keeps its exact native
        # samples instead of being interpolated onto its own grid.
        chunk = sample_action_chunk(
            timestamps,
            action,
            float(row["anchor_s"]),
            native_rate_hz=episode.native_rate_hz,
        )
        sample = {
            "episode_id": row["episode_id"],
            "split": row["split"],
            "source": row["source"],
            "embodiment": row["embodiment"],
            "anchor_s": float(row["anchor_s"]),
            "observation.state": state[history_frames],
            "observation.timestamps": timestamps[history_frames],
            "action": chunk.values,
            "action.timestamps": chunk.target_timestamps,
            "action.interpolated_mask": chunk.interpolated_mask,
            "action.source_indices": chunk.source_indices,
            "action.source_timestamps": chunk.source_timestamps,
            "action_source": row["action_source"],
            "annotation.subtask": row["subtask"],
            "annotation.quality": row["quality"],
            "annotation.mistake": row["mistake"],
        }
        if cameras:
            sample["observation.images"] = {
                camera: episode.frames(camera, history_frames) for camera in episode.cameras
            }
        return sample

    # -- critic view -----------------------------------------------------------------

    def critic_sample(
        self,
        row: dict[str, Any],
        *,
        action_points: int | None = None,
        observation_frames: int = 0,
    ) -> dict[str, Any]:
        """One reviewed subtask with every native action it contains.

        `action_points` subsamples for a fixed-size critic input. The stored sequence is
        never truncated: subsampling happens here, at load time, and always returns the
        first and last action plus a validity mask.
        """
        episode = self.episode(row["episode_id"])
        start = int(row["start_timestep"])
        end = int(row["end_timestep_exclusive"])
        timestamps = episode.timestamps[start:end]
        actions = episode.action[start:end]
        states = episode.state[start:end]
        sample = {
            "episode_id": row["episode_id"],
            "split": row["split"],
            "source": row["source"],
            "embodiment": row["embodiment"],
            "subtask": row["normalized_description"],
            "classification": row["classification"],
            "quality": row["quality"],
            "quality_provenance": row["quality_provenance"],
            "subtask_outcome": row["subtask_outcome"],
            "episode_outcome": row["episode_outcome"],
            "action": actions,
            "action.timestamps": timestamps,
            "observation.state": states,
            "native_action_samples": int(end - start),
            "duration_s": float(row["duration_s"]),
            "pause_events": row["pause_events"],
            "mistake_events": row["mistake_events"],
            "recovery_events": row["recovery_events"],
        }
        if action_points:
            indices, mask = time_stratified_indices(
                timestamps,
                actions,
                action_points,
                event_times=[
                    float(event["start_s"])
                    for event in row["mistake_events"] + row["recovery_events"]
                ],
                interval_start_s=float(row["start_s"]),
            )
            padded_actions = np.zeros((action_points, actions.shape[1]), dtype=actions.dtype)
            padded_times = np.zeros(action_points, dtype=timestamps.dtype)
            padded_actions[: len(indices)] = actions[indices]
            padded_times[: len(indices)] = timestamps[indices]
            sample["action.subsampled"] = padded_actions
            sample["action.subsampled_timestamps"] = padded_times
            sample["action.subsampled_indices"] = indices
            sample["action.subsampled_mask"] = mask
        if observation_frames:
            frame_indices = np.unique(
                np.linspace(start, end - 1, observation_frames).round().astype(np.int64)
            )
            sample["observation.images"] = {
                camera: episode.frames(camera, frame_indices) for camera in episode.cameras
            }
            sample["observation.image_frames"] = frame_indices
        return sample


def time_stratified_indices(
    timestamps: np.ndarray,
    actions: np.ndarray,
    count: int,
    *,
    event_times: list[float] | None = None,
    interval_start_s: float = 0.0,
    gripper_index: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose `count` action indices without crossing the subtask boundary.

    The first and last valid action are always retained. Gripper transitions and reviewed
    mistake or recovery events are preferred over plain uniform coverage, because those
    are the points where a critic's judgement actually changes.
    """
    total = len(timestamps)
    mask = np.zeros(count, dtype=bool)
    if total <= count:
        indices = np.arange(total, dtype=np.int64)
        mask[:total] = True
        return indices, mask

    required = {0, total - 1}
    gripper = actions[:, gripper_index]
    transitions = np.nonzero(np.abs(np.diff(gripper)) > 0.5 * (np.ptp(gripper) or 1.0))[0]
    for index in transitions:
        required.add(int(index))
    for event_time in event_times or []:
        # Event times and corpus timestamps are both episode seconds.
        required.add(int(np.argmin(np.abs(timestamps - event_time))))
    required = set(sorted(required)[:count])

    remaining = count - len(required)
    if remaining > 0:
        strata = np.linspace(timestamps[0], timestamps[-1], remaining + 2)[1:-1]
        for target in strata:
            position = int(np.argmin(np.abs(timestamps - target)))
            while position in required and position < total - 1:
                position += 1
            required.add(position)
    indices = np.asarray(sorted(required)[:count], dtype=np.int64)
    mask[: len(indices)] = True
    return indices, mask
