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
        allowed = ("sqrt_episodes", "episodes", "anchors", "uniform")
        if self.group_weight not in allowed:
            raise ValueError(f"diverse.group_weight must be one of {allowed}, got {self.group_weight!r}.")
