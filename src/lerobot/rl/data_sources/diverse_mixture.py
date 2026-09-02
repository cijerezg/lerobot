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

"""Hierarchical sampling for the mixed ReBot + diverse run (integration plan phase I).

Two decisions, made at different levels and for different reasons.

**Outer, 50/50 by group.** Row counts do not define the objective. ReBot contributes
181,398 frames and the diverse corpus 60,728 anchors; a row-proportional mixture would
give the established distribution three quarters of every gradient by arithmetic rather
than by intent. Half and half gives the two distributions equal expected gradient mass,
which is the thing actually being chosen.

**Inner, sqrt(episodes) by source.** Inside the diverse half, row-proportional sampling
would hand RoboChallenge 67% of the diverse gradient (40,706 of 60,728 anchors) purely
because it was cheapest to acquire, while source-uniform sampling would show the four
UR7e episodes as often as RoboChallenge's two hundred and overfit them. The square root
sits between: RoboChallenge 35%, FMB 25%, each DROID 18%, UR7e 5%.

Then an episode uniformly inside the source, and an anchor uniformly inside the episode
-- so a long episode does not outvote a short one within its own source either.

Every level is configurable and the realized proportions are logged, because the point of
choosing a mixture is being able to see what you got.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from lerobot.datasets.diverse_actor_selection import DiverseActorSelection
from lerobot.rl.buffer import concatenate_variable_dim_batch_transitions
from lerobot.rl.offline_dataset_utils import _weighted_batch_sizes

logger = logging.getLogger(__name__)

GROUP_WEIGHT_RULES = ("sqrt_episodes", "episodes", "anchors", "uniform")


# ── Inner sampler ────────────────────────────────────────────────────────────


class HierarchicalAnchorSampler:
    """group -> episode -> anchor, each level drawn on its own.

    Callable as ``sampler(batch_size) -> row indices``, which is what
    ``DiverseActorBuffer`` takes.
    """

    def __init__(
        self,
        selection: DiverseActorSelection,
        *,
        group_of: Callable[[dict], str] | None = None,
        group_weight: str = "sqrt_episodes",
        group_weight_overrides: dict[str, float] | None = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if group_weight not in GROUP_WEIGHT_RULES:
            raise ValueError(f"group_weight must be one of {GROUP_WEIGHT_RULES}, got {group_weight!r}.")
        self.selection = selection
        self.group_of = group_of or (lambda row: str(row["source"]))
        self.group_weight = group_weight

        episodes: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(selection.rows):
            episodes[self.group_of(row)][str(row["episode_id"])].append(index)

        self.groups = sorted(episodes)
        self.episode_rows: dict[str, list[np.ndarray]] = {
            group: [np.asarray(rows, dtype=np.int64) for _, rows in sorted(episodes[group].items())]
            for group in self.groups
        }
        self.episode_ids: dict[str, list[str]] = {
            group: sorted(episodes[group]) for group in self.groups
        }

        weights = np.asarray([self._weight(group) for group in self.groups], dtype=np.float64)
        for name, value in (group_weight_overrides or {}).items():
            if name not in self.groups:
                raise KeyError(f"group weight override for unknown group {name!r}; have {self.groups}.")
            weights[self.groups.index(name)] = float(value)
        if weights.min() <= 0:
            raise ValueError(f"group weights must be positive, got {dict(zip(self.groups, weights))}.")
        self.probabilities = weights / weights.sum()

        # Distinct, reproducible streams per rank: two ranks drawing the same anchors
        # would halve the effective batch without saying so.
        self._rng = np.random.default_rng(
            np.random.SeedSequence(seed).spawn(max(world_size, 1))[rank]
        )
        logger.info(
            "[DiverseMixture] inner p(group) [%s]: %s",
            group_weight,
            {group: round(float(p), 4) for group, p in zip(self.groups, self.probabilities)},
        )

    def _weight(self, group: str) -> float:
        episodes = len(self.episode_rows[group])
        if self.group_weight == "sqrt_episodes":
            return math.sqrt(episodes)
        if self.group_weight == "episodes":
            return float(episodes)
        if self.group_weight == "anchors":
            return float(sum(len(rows) for rows in self.episode_rows[group]))
        return 1.0

    def target_proportions(self) -> dict[str, float]:
        return {group: float(p) for group, p in zip(self.groups, self.probabilities)}

    def __call__(self, batch_size: int) -> np.ndarray:
        group_indices = self._rng.choice(len(self.groups), size=batch_size, p=self.probabilities)
        out = np.empty(batch_size, dtype=np.int64)
        for position, group_index in enumerate(group_indices):
            episodes = self.episode_rows[self.groups[group_index]]
            rows = episodes[self._rng.integers(len(episodes))]
            out[position] = rows[self._rng.integers(len(rows))]
        return out


# ── Outer mixture ────────────────────────────────────────────────────────────


@dataclass
class MixtureGroup:
    """One half of the outer split: its buffers, their inner weights, its outer weight."""

    name: str
    buffers: list[Any]
    weight: float = 1.0
    inner_weights: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.buffers:
            raise ValueError(f"mixture group {self.name!r} has no buffers.")
        if self.inner_weights is None:
            self.inner_weights = [1.0] * len(self.buffers)
        if len(self.inner_weights) != len(self.buffers):
            raise ValueError(
                f"group {self.name!r} has {len(self.buffers)} buffers and "
                f"{len(self.inner_weights)} inner weights."
            )


def allocate_group_quotas(batch_size: int, groups: list[MixtureGroup]) -> list[list[int]]:
    """Per-buffer batch sizes: groups first, then buffers inside each group.

    Odd batches are resolved by the same largest-remainder rule the collection loader
    already uses, with ties broken by declaration order -- so with two equal groups and
    an odd batch the extra sample deterministically goes to the first group. Declare
    ReBot first and the established half is the one that never shrinks.
    """
    outer = _weighted_batch_sizes(batch_size, [group.weight for group in groups])
    return [
        _weighted_batch_sizes(quota, group.inner_weights)
        for quota, group in zip(outer, groups, strict=True)
    ]


def observed(iterator, telemetry: "MixtureTelemetry", *, depth_key: str = "depth.wrist_0.depth"):
    """Pass batches through, recording what the mixture actually produced."""
    for batch in iterator:
        telemetry.observe(batch, depth_key=depth_key)
        yield batch


def make_hierarchical_offline_iterator(
    groups: list[MixtureGroup],
    batch_size: int,
    *,
    async_prefetch: bool = False,
    queue_size: int = 2,
    action_chunk_size: int = 30,
):
    """Fixed-size batches drawn group-first, then buffer-by-weight inside each group."""
    quotas = allocate_group_quotas(batch_size, groups)
    for group, per_buffer in zip(groups, quotas, strict=True):
        logger.info(
            "[DiverseMixture] %s: outer weight %.3f, %d/%d samples per batch %s",
            group.name,
            group.weight,
            sum(per_buffer),
            batch_size,
            per_buffer,
        )
    iterators = [
        buffer.get_iterator(
            batch_size=size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
            action_chunk_size=action_chunk_size,
        )
        for group, per_buffer in zip(groups, quotas, strict=True)
        for buffer, size in zip(group.buffers, per_buffer, strict=True)
    ]
    while True:
        batch = next(iterators[0])
        for iterator in iterators[1:]:
            batch = concatenate_variable_dim_batch_transitions(batch, next(iterator))
        yield batch


# ── Telemetry ────────────────────────────────────────────────────────────────


@dataclass
class MixtureTelemetry:
    """Realized proportions of what the sampler actually produced."""

    samples: int = 0
    source: Counter = field(default_factory=Counter)
    embodiment: Counter = field(default_factory=Counter)
    layout: Counter = field(default_factory=Counter)
    episode: Counter = field(default_factory=Counter)
    camera_count: Counter = field(default_factory=Counter)
    depth: Counter = field(default_factory=Counter)
    quality_valid: Counter = field(default_factory=Counter)

    def observe(self, batch: dict, *, depth_key: str = "depth.wrist_0.depth") -> None:
        info = batch.get("complementary_info") or {}

        def column(name: str):
            value = info.get(name)
            return None if value is None else torch.as_tensor(value).detach().cpu()

        size = int(torch.as_tensor(batch["reward"]).shape[0])
        self.samples += size
        for name, counter in (
            ("source_id", self.source),
            ("embodiment_index", self.embodiment),
            ("action_layout_id", self.layout),
            ("episode_position", self.episode),
        ):
            values = column(name)
            if values is not None:
                # Negative ids are "this half does not have one" (concatenation pads a
                # missing column, and the ReBot adapter marks its rows explicitly).
                counter.update(int(v) for v in values.reshape(-1).tolist() if int(v) >= 0)
        cameras = column("camera_is_present")
        if cameras is not None:
            self.camera_count.update(int(v) for v in cameras.sum(dim=1).tolist())
        depth = column(f"{depth_key}_is_present")
        if depth is not None:
            self.depth.update(bool(v) for v in depth.reshape(-1).tolist())
        quality = column("metadata_quality_is_valid")
        if quality is not None:
            self.quality_valid.update(bool(v) for v in quality.reshape(-1).tolist())

    def proportions(self, counter: Counter) -> dict[Any, float]:
        if not self.samples:
            return {}
        return {key: value / self.samples for key, value in sorted(counter.items(), key=str)}

    def describe(self, source_names: dict[int, str] | None = None) -> str:
        lines = [f"Realized mixture over {self.samples} samples:"]
        sources = self.proportions(self.source)
        if source_names:
            sources = {source_names.get(key, key): value for key, value in sources.items()}
        lines.append(f"  source          {({k: round(v, 4) for k, v in sources.items()})}")
        lines.append(f"  action layout   {({k: round(v, 4) for k, v in self.proportions(self.layout).items()})}")
        lines.append(f"  embodiment      {({k: round(v, 4) for k, v in self.proportions(self.embodiment).items()})}")
        lines.append(f"  cameras present {({k: round(v, 4) for k, v in self.proportions(self.camera_count).items()})}")
        lines.append(f"  depth present   {({k: round(v, 4) for k, v in self.proportions(self.depth).items()})}")
        lines.append(f"  quality valid   {({k: round(v, 4) for k, v in self.proportions(self.quality_valid).items()})}")
        lines.append(f"  distinct episodes seen {len(self.episode)}")
        return "\n".join(lines)
