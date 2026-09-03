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

"""Mapping between LeRobot joint positions and reBot 102 leader raw servo angles.

See ``pi07_wiki/leader_102hd_actuation.md`` §4.3. ``raw_to_position`` is the map
``RebotArm102Leader.get_action`` applies when reading the arm; ``position_to_raw`` is its
inverse, used to drive the arm. Both take their per-joint constants (``joint_ranges``,
``joint_directions``) from ``RebotArm102LeaderConfig``.

The clamp happens before the division so a policy commanding past the leader's mechanical
range is bounded rather than wrapped.

``unwrap_raw`` centres its window on ``range / direction``. ``get_action`` currently uses
``range * sign``, which for the gripper (direction -6) centres on 135 deg instead of 22.5.
The two agree over the gripper's real raw travel of ~45 deg; they diverge once the raw
origin shifts. Fixing the driver belongs to track A/B stage 3.
"""


def clamp_position(position: float, joint_range: list[int]) -> float:
    range_min, range_max = joint_range
    return max(float(range_min), min(float(range_max), position))


def position_to_raw(position: float, joint_range: list[int], direction: int) -> float:
    """LeRobot joint position (follower convention, degrees) -> leader raw servo angle."""
    return clamp_position(position, joint_range) / direction


def unwrap_raw(raw_angle: float, joint_range: list[int], direction: int) -> float:
    """Subtract whole turns, bringing a multi-turn angle into the joint's own +-180 window."""
    center = (joint_range[0] / direction + joint_range[1] / direction) / 2.0
    return raw_angle - round((raw_angle - center) / 360.0) * 360.0


def raw_to_position(raw_angle: float, joint_range: list[int], direction: int) -> float:
    """Leader raw servo angle -> LeRobot joint position (follower convention, degrees)."""
    return clamp_position(unwrap_raw(raw_angle, joint_range, direction) * direction, joint_range)
