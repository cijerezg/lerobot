#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Critical-section annotations loader and per-timestep loss-weight provider.

The annotations are produced by data-studio and saved as
``meta/critical_sections.json`` next to a LeRobot dataset.  Each annotated
episode lists one or more time intervals (in episode-relative seconds) that
should receive an elevated training loss weight.  We do **not** slice
trajectories: training keeps full action chunks, and only timesteps whose
absolute timestamp falls inside any annotated section get the heavier weight.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CriticalSection:
    """A single annotated critical interval inside an episode."""

    start: float
    end: float
    weight: float


class CriticalSectionsProvider:
    """Loads ``meta/critical_sections.json`` and produces per-timestep weights.

    Parameters
    ----------
    sections:
        Mapping from ``episode_index`` to list of :class:`CriticalSection`.
    default_weight:
        Weight applied to a timestep when it falls into a section that did
        not specify its own ``weight``.
    """

    def __init__(
        self,
        sections: dict[int, list[CriticalSection]],
        default_weight: float = 5.0,
    ):
        self._sections = sections
        self._default_weight = float(default_weight)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        default_weight: float = 5.0,
    ) -> "CriticalSectionsProvider":
        """Load and parse ``critical_sections.json`` from disk.

        Tolerates the data-studio JSON shape:

        ``{"episodes": {"<ep_id>": {"sections": [{"start", "end", "weight"?}, ...]}}}``

        Episodes without a ``sections`` list (or whose list is empty) are
        treated as not annotated.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Critical-sections file not found: {path}")

        with path.open("r") as f:
            raw: dict[str, Any] = json.load(f)

        sections = parse_critical_sections(raw, default_weight=default_weight)
        return cls(sections, default_weight=default_weight)

    @property
    def default_weight(self) -> float:
        return self._default_weight

    def annotated_episode_indices(self) -> list[int]:
        """Return the sorted list of episode indices that have at least one section."""
        return sorted(ep for ep, secs in self._sections.items() if len(secs) > 0)

    def num_annotated_episodes(self) -> int:
        return len(self.annotated_episode_indices())

    def sections_for_episode(self, episode_index: int) -> list[CriticalSection]:
        return list(self._sections.get(int(episode_index), []))

    def compute_timestep_weights(
        self,
        batch: dict[str, Any],
        chunk_size: int,
        fps: float,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Compute ``(B, T)`` per-timestep loss weights for a training batch.

        For sample ``b`` with episode ``e`` and base timestamp ``ts``, timestep
        ``t`` of the action chunk corresponds to the absolute time
        ``ts + t / fps``.  The returned weight is the maximum critical-section
        weight that covers that absolute time, falling back to ``1.0``.
        """
        if "episode_index" not in batch or "timestamp" not in batch:
            raise KeyError(
                "compute_timestep_weights expects `episode_index` and `timestamp` in the batch."
            )

        episode_indices = _to_int_array(batch["episode_index"])
        timestamps = _to_float_array(batch["timestamp"])

        if episode_indices.shape[0] != timestamps.shape[0]:
            raise ValueError(
                "episode_index and timestamp must have the same batch size, "
                f"got {episode_indices.shape[0]} and {timestamps.shape[0]}."
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")

        batch_size = episode_indices.shape[0]
        weights = np.ones((batch_size, chunk_size), dtype=np.float32)
        # Half-frame tolerance keeps things stable when the dataset timestamps
        # were quantized at ``fps`` (so a section that ends exactly on a frame
        # boundary still includes that frame instead of falling through the
        # floating-point crack).
        eps = 0.5 / float(fps)
        step_offsets = np.arange(chunk_size, dtype=np.float64) / float(fps)

        for b in range(batch_size):
            ep_sections = self._sections.get(int(episode_indices[b]))
            if not ep_sections:
                continue
            abs_times = float(timestamps[b]) + step_offsets
            for section in ep_sections:
                covered = (abs_times >= section.start - eps) & (abs_times <= section.end + eps)
                if not np.any(covered):
                    continue
                # Spans may overlap -> take the maximum weight for each timestep
                np.maximum(weights[b], np.where(covered, section.weight, 1.0), out=weights[b])

        return torch.as_tensor(weights, dtype=torch.float32, device=device)


def parse_critical_sections(
    raw: dict[str, Any],
    default_weight: float = 5.0,
) -> dict[int, list[CriticalSection]]:
    """Parse a critical-sections JSON document into ``{episode_index: [section, ...]}``.

    Tolerantly accepts:
    - top-level ``episodes`` dict whose keys are stringified ints
    - per-episode ``sections`` list
    - per-section optional ``weight`` (falls back to ``default_weight``)
    - section bounds that are out of order (we swap them and warn)
    """
    episodes = raw.get("episodes")
    if not isinstance(episodes, dict):
        raise ValueError(
            "critical_sections.json must contain a top-level `episodes` mapping."
        )

    parsed: dict[int, list[CriticalSection]] = {}
    for ep_key, ep_value in episodes.items():
        try:
            ep_idx = int(ep_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Episode keys must be integer-valued, got {ep_key!r}."
            ) from exc

        if not isinstance(ep_value, dict):
            raise ValueError(
                f"Episode {ep_idx} entry must be an object, got {type(ep_value).__name__}."
            )
        sections_raw = ep_value.get("sections", [])
        if not isinstance(sections_raw, list):
            raise ValueError(
                f"Episode {ep_idx} `sections` must be a list, got {type(sections_raw).__name__}."
            )

        sections: list[CriticalSection] = []
        for i, sec in enumerate(sections_raw):
            if not isinstance(sec, dict):
                raise ValueError(
                    f"Episode {ep_idx} section #{i} must be an object, got {type(sec).__name__}."
                )
            if "start" not in sec or "end" not in sec:
                raise ValueError(
                    f"Episode {ep_idx} section #{i} is missing `start`/`end`."
                )
            start = float(sec["start"])
            end = float(sec["end"])
            if end < start:
                logging.warning(
                    "critical_sections.json: episode %d section #%d has end (%.4f) < start (%.4f); swapping.",
                    ep_idx,
                    i,
                    end,
                    start,
                )
                start, end = end, start
            weight = float(sec.get("weight", default_weight))
            sections.append(CriticalSection(start=start, end=end, weight=weight))

        parsed[ep_idx] = sections

    return parsed


def _to_int_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.int64).reshape(-1)
    if isinstance(value, np.ndarray):
        return value.astype(np.int64).reshape(-1)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _to_float_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).reshape(-1)
    return np.asarray(value, dtype=np.float64).reshape(-1)
