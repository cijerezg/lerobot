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

"""Track C stage C1: the leader mapping algebra of leader_102hd_actuation.md §4.3.

No hardware. ``RebotArm102Leader._round_to_valid_range`` is a static method, so the
driver's own arithmetic is exercised without constructing the class or opening a port.
"""

import pytest

from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader import RebotArm102LeaderConfig
from lerobot.teleoperators.rebot_102_leader.mapping import (
    clamp_position,
    position_to_raw,
    raw_to_position,
)
from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import RebotArm102Leader

CONFIG = RebotArm102LeaderConfig(port="/dev/null")
JOINTS = list(CONFIG.joint_ids)


def constants(motor_name):
    return CONFIG.joint_ranges[motor_name], CONFIG.joint_directions[motor_name]


def get_action_position(raw_angle: float, joint_range: list[int], direction: int) -> float:
    """The arithmetic of ``RebotArm102Leader.get_action``, verbatim."""
    range_min, range_max = joint_range
    sign = 1.0 if direction >= 0 else -1.0
    unwrapped, _ = RebotArm102Leader._round_to_valid_range(raw_angle, range_min * sign, range_max * sign)
    position = unwrapped * direction
    return max(float(range_min), min(float(range_max), position))


def sample_positions(joint_range: list[int], n: int = 11) -> list[float]:
    range_min, range_max = joint_range
    return [range_min + (range_max - range_min) * i / (n - 1) for i in range(n)]


@pytest.mark.parametrize("motor_name", JOINTS)
def test_round_trip_position_to_raw_and_back(motor_name):
    """position -> raw -> position is the identity across each joint's whole range."""
    joint_range, direction = constants(motor_name)
    for position in sample_positions(joint_range):
        raw = position_to_raw(position, joint_range, direction)
        assert raw_to_position(raw, joint_range, direction) == pytest.approx(position, abs=1e-9)


@pytest.mark.parametrize("motor_name", JOINTS)
def test_round_trip_agrees_with_get_action(motor_name):
    """The raw angles we command read back through ``get_action`` as the positions we asked for."""
    joint_range, direction = constants(motor_name)
    for position in sample_positions(joint_range):
        raw = position_to_raw(position, joint_range, direction)
        assert get_action_position(raw, joint_range, direction) == pytest.approx(position, abs=1e-9)


@pytest.mark.parametrize("motor_name", JOINTS)
def test_raw_to_position_matches_get_action_over_real_travel(motor_name):
    """Both unwrap windows agree over the raw travel each joint physically has."""
    joint_range, direction = constants(motor_name)
    raw_endpoints = sorted(position_to_raw(p, joint_range, direction) for p in joint_range)
    for i in range(21):
        raw = raw_endpoints[0] + (raw_endpoints[1] - raw_endpoints[0]) * i / 20
        assert raw_to_position(raw, joint_range, direction) == pytest.approx(
            get_action_position(raw, joint_range, direction), abs=1e-9
        )


@pytest.mark.parametrize("motor_name", JOINTS)
def test_positions_out_of_range_are_clamped_not_wrapped(motor_name):
    """A policy commanding past the mechanical range is bounded (§4.3)."""
    joint_range, direction = constants(motor_name)
    range_min, range_max = joint_range
    for position in (range_min - 500.0, range_max + 500.0):
        raw = position_to_raw(position, joint_range, direction)
        assert raw == pytest.approx(clamp_position(position, joint_range) / direction, abs=1e-9)
        assert range_min <= raw_to_position(raw, joint_range, direction) <= range_max


def test_gripper_open_maps_to_45_degrees_of_raw_travel():
    """§4.3's hand check: fully open (-270) is ~45 deg of raw travel, which is what the arm has."""
    joint_range, direction = constants("gripper")
    assert position_to_raw(-270.0, joint_range, direction) == pytest.approx(45.0)
    assert position_to_raw(0.0, joint_range, direction) == pytest.approx(0.0)


def test_gripper_unwrap_window_diverges_from_driver_once_the_origin_shifts():
    """Pins the §4.3 prerequisite bug: ``range * sign`` centres the gripper window on 135, not 22.5.

    Track C does not fix it (that is track A/B stage 3); this records where the two disagree so
    the fix has a test to flip. At raw 210 the correct window unwraps to -150 (gripper closed),
    the driver's leaves it at 210 (gripper fully open) -- opposite ends of the travel.
    """
    joint_range, direction = constants("gripper")
    assert raw_to_position(210.0, joint_range, direction) == pytest.approx(0.0)
    assert get_action_position(210.0, joint_range, direction) == pytest.approx(-270.0)
