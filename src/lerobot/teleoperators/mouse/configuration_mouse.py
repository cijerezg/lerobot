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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("mouse")
@dataclass
class MouseTeleopConfig(TeleoperatorConfig):
    """Mouse / trackpad teleoperation for human intervention during rollouts.

    Translation-only: pointer motion drives end-effector x/y, the wheel drives z, and
    the wrist orientation is left at whatever pose the arm held when the intervention
    started (the EE pipeline is sent a zero rotation delta).
    """

    use_gripper: bool = True

    # evdev device path, e.g. "/dev/input/event4". None auto-detects the first device
    # advertising REL_X. Requires membership of the `input` group.
    device_path: str | None = None

    # Counts of pointer motion that map to a full-scale (1.0) delta on one axis. Larger
    # values mean slower, finer control. Tune this in simulation before going to hardware.
    counts_per_unit: float = 200.0
    wheel_counts_per_unit: float = 3.0

    # Per-axis sign, applied after normalization. Depends on how the operator's view is
    # oriented relative to the robot base frame; set these by watching the sim.
    invert_x: bool = False
    invert_y: bool = True
    invert_z: bool = False

    # Hard cap on the magnitude of a single normalized delta. EEBoundsAndSafety raises
    # (it does not clamp) when a commanded step exceeds max_ee_step_m, so a fast flick
    # must never be allowed to reach it.
    max_delta: float = 0.6

    # Take exclusive ownership of the device (EVIOCGRAB) while intervening, so pointer
    # motion and clicks never reach the desktop underneath.
    grab_device: bool = True

    # Terminal keys for episode events, matching the SOLeader convention
    # (1 = success, 0 = terminate/failure, 2 = start episode).
    enable_terminal_keys: bool = True
