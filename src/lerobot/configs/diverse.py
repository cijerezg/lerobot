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

"""The `diverse:` config block (integration plan phase J).

Kept free of torch and of the RL package so `lerobot.configs.train` stays importable
without pulling the replay machinery in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StartWeightsConfig:
    """Per-start draw weights inside the ReBot pool (2026-10-16, migration/sampling_density_2026-10-16).

    Every eligible start has weight 1 (move, return, the transit to the object). Around a commit moment
    the weight rises to a peak and the start is drawn that many times more often than a move start:
    grasp = peak_ratio * precision_scale[precision - 1] at every gripper close, release = the same at the
    gripper open, every other skill (push, pour, close, stand, ...) = skill_ratio over the whole segment.
    Group shares stay the config's rebot_group_weights; inside a group the views split by their
    start-weight mass. The per-start rule only decides which starts of a group are drawn.
    """

    enabled: bool = False
    # Draws of the peak frame of a precision-5 grasp per draw of a move frame.
    peak_ratio: float = 5.0
    # Fraction of peak_ratio by precision label 1..5, for grasps and releases.
    precision_scale: tuple[float, float, float, float, float] = (0.3, 0.4, 0.6, 0.8, 1.0)
    # Flat ratio over push / pour / close / stand / plug / fold ... segments (their precision labels
    # score slack at the commit, not how precise the motion is).
    skill_ratio: float = 3.0
    # Seconds before / after the commit over which the peak rises / decays.
    grasp_window: tuple[float, float] = (8.0, 2.0)
    release_window: tuple[float, float] = (3.0, 1.0)
    # Gaussian edges (sigma = half the window) instead of rectangles.
    smooth: bool = True

    def validate(self) -> None:
        if self.peak_ratio < 1 or self.skill_ratio < 1:
            raise ValueError("rebot_start_weights ratios must be >= 1 (1 = a move frame).")
        if len(self.precision_scale) != 5 or any(not 0 < v <= 1 for v in self.precision_scale):
            raise ValueError("rebot_start_weights.precision_scale needs five values in (0, 1].")
        if any(v < 0 for v in (*self.grasp_window, *self.release_window)):
            raise ValueError("rebot_start_weights windows must be nonnegative seconds.")


@dataclass
class DiverseCollectionConfig:
    """Mix the source-native diverse corpus against the LeRobot (ReBot) sources.

    Disabled by default: an existing single-corpus config behaves exactly as before.
    """

    enabled: bool = False
    root: str = "outputs/diverse_robot_dataset"
    # None falls back to the run's buffer_cache_dir.
    cache_dir: str | None = None
    # Outer split between the two groups. 1.0 / 1.0 is the plan's 50/50: equal expected
    # gradient mass, rather than the 3:1 that raw row counts would produce.
    weight: float = 1.0
    rebot_weight: float = 1.0
    # Optional named ReBot buckets; activates sampling without a per-source floor.
    # A weight of 0 keeps the group's episodes in the cache but never samples them.
    rebot_group_weights: dict[str, float] = field(default_factory=dict)
    # Stratified start coverage inside each ReBot group view: eligible starts (every
    # image_stride-th frame) are cut into cells of this many starts; every pass draws one
    # start per cell. 1 = plain iid draws.
    # Applied after replay-cache loading; original source-root/episode frame ranges.
    rebot_exclusions_manifest: str | None = None
    rebot_start_cell: int = 1
    # Per-start draw weights around commit moments inside each ReBot group; replaces iid / cell
    # draws. rebot_group_weights still sets the group shares.
    rebot_start_weights: StartWeightsConfig = field(default_factory=StartWeightsConfig)
    # Inner rule across the five corpus sources: sqrt_episodes | episodes | anchors | uniform.
    group_weight: str = "sqrt_episodes"
    group_weight_overrides: dict[str, float] = field(default_factory=dict)
    # RoboChallenge quality is source-derived, not reviewed (DIVERSE_ROBOT_DATASET.md §6).
    # False withholds the integer so no quality clause renders for those samples.
    render_automatic_quality: bool = False
    # Which registered action layout the ReBot roots are.
    rebot_layout: str = "rebot_b601_joint7_commanded"
    # ReBot's depth camera calibration (fx, fy, cx, cy). A mixed batch back-projects two
    # cameras, so each half must carry its own row; None keeps the pointmap config's.
    rebot_depth_intrinsics: tuple[float, float, float, float] | None = None
    # Probe the ReBot caches for -6 s addressability and the required columns at startup.
    probe_rebot_caches: bool = True

    def validate(self) -> None:
        if self.weight <= 0 or self.rebot_weight <= 0:
            raise ValueError("diverse.weight and diverse.rebot_weight must be positive.")
        if any(not name or weight < 0 for name, weight in self.rebot_group_weights.items()):
            raise ValueError("ReBot group names must be nonempty and weights nonnegative (0 excludes the group).")
        if self.rebot_start_cell < 1:
            raise ValueError("diverse.rebot_start_cell must be >= 1.")
        self.rebot_start_weights.validate()
        if self.rebot_start_weights.enabled and self.rebot_start_cell != 1:
            raise ValueError("diverse.rebot_start_weights and rebot_start_cell > 1 are exclusive.")
        allowed = ("sqrt_episodes", "episodes", "anchors", "uniform")
        if self.group_weight not in allowed:
            raise ValueError(f"diverse.group_weight must be one of {allowed}, got {self.group_weight!r}.")
