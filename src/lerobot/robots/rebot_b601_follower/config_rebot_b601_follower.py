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

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class RebotB601FollowerConfig:
    """Base configuration class for the Seeed Studio reBot B601-DM follower arm.

    The B601-DM is a 6-DOF arm plus gripper driven by Damiao CAN motors. Motor
    communication goes through the ``motorbridge`` package.
    """

    # Communication port. For ``can_adapter="damiao"`` this is the Damiao serial
    # bridge device (e.g. "/dev/ttyACM0"); for ``can_adapter="socketcan"`` it is
    # the CAN channel name (e.g. "can0").
    port: str

    # CAN adapter type:
    #   "damiao"    - Damiao dedicated serial bridge (default)
    #   "socketcan" - SocketCAN based adapters (PCAN, slcan, embedded controllers, ...)
    can_adapter: str = "damiao"

    # Baud rate for the Damiao serial bridge (only used when can_adapter="damiao").
    dm_serial_baud: int = 921600

    # disconnect() parks before it releases torque: freeze (present position re-sent
    # as the target, cancelling any in-flight sweep), descend to park_pose linearly at
    # park_deg_per_s, then release only if every joint is within park_tolerance_deg.
    # The elbow runs at park_elbow_deg_per_s and lands before the other joints, and wrist_flex
    # goes to park + park_wrist_bend_deg (an L) on the elbow's clock, then straightens over the
    # rest of the descent (the manual park in tidy-up-v1 2026-11-25, which is easy on the cable).
    # Without the bend the leading elbow puts the tip on the table from low poses, and so does
    # holding the other joints while the elbow starts (sim check 2026-11-25,
    # outputs/analysis/park_stages_2026-11-25).
    # Otherwise torque stays ON and the arm holds; run park_rebot.py to retry.
    # The default park pose is the calibration zero (the sit-down position, gripper closed).
    park_pose: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
    )
    park_deg_per_s: float = 15.0
    park_elbow_deg_per_s: float = 30.0
    park_wrist_bend_deg: float = -27.0
    park_gripper_deg_per_s: float = 120.0  # gripper closes on its own rate (motor cap 180 deg/s)
    park_tolerance_deg: float = 10.0

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety purposes (in degrees). Set to a positive scalar to apply the
    # same value to all motors, or to a dict mapping motor names to per-motor values.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps motor names to their (send_can_id, recv_can_id) pair.
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )

    # Target velocity for joints running in POS_VEL mode, in degrees/s. A scalar is
    # applied to every joint; a list provides one value per joint (in motor order).
    # Joints are capped low enough (~0.5 m/s at the tip) to leave reaction time if
    # the leader is dropped. The gripper gets a higher cap: it travels its full
    # 270 degree range on every open/close, and its force is already limited by
    # gripper_torque_ratio.
    pos_vel_velocity: float | list[float] = field(default_factory=lambda: [50.0] * 6 + [180.0])

    # Position-loop gain (Damiao KP_APR) per motor name, written at connect and NOT stored in the
    # motor: a power cycle brings back the motor's own value (54 on every motor, 2026-11-13).
    # The joint's response rate is KP_APR / gearbox ratio: 54 gives 1.35 /s (0.76 s time constant)
    # on the 40:1 joints and 5.4 /s on the 10:1 wrist joints. The default raises the 40:1 joints
    # to 216 so they respond like the wrist (0.21 s, measured 2026-11-13); data recorded before
    # that date has the 54 response. Motors not listed keep their own value; None writes nothing.
    position_kp: dict[str, float] | None = field(
        default_factory=lambda: {"shoulder_pan": 216.0, "shoulder_lift": 216.0, "elbow_flex": 216.0}
    )

    # Torque/current ratio for the gripper's FORCE_POS mode, in range [0, 1].
    gripper_torque_ratio: float = 0.1

    # Soft joint limits (degrees). These are clipped against on every action.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-145.0, 145.0),
            "shoulder_lift": (-170.0, 1.0),
            "elbow_flex": (-200.0, 1.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )


def park_ramp(present: dict[str, float], config, fps: float) -> list[dict[str, float]]:
    """Joint targets (deg) of the park descent, one per control tick; the last is config.park_pose.

    Every joint goes straight to park. The elbow has its own, faster clock; wrist_flex rides it
    to the bend, then straightens over the remaining ticks. The bend is no larger than the
    elbow's travel, so a park from the park pose does not move the wrist.
    """
    target = config.park_pose
    # The gripper has its own rate: its 270 deg travel would otherwise pace the arm's descent.
    rate = dict.fromkeys(target, config.park_deg_per_s)
    rate["gripper"] = config.park_gripper_deg_per_s
    elbow_gap = abs(target["elbow_flex"] - present["elbow_flex"])
    bend = target["wrist_flex"] + math.copysign(min(abs(config.park_wrist_bend_deg), elbow_gap), config.park_wrist_bend_deg)
    elbow_steps = max(1, math.ceil(fps * max(elbow_gap, abs(bend - present["wrist_flex"])) / config.park_elbow_deg_per_s))
    straighten_steps = math.ceil(fps * abs(target["wrist_flex"] - bend) / config.park_deg_per_s)
    others = [name for name in target if name not in ("elbow_flex", "wrist_flex")]
    seconds = max(abs(target[name] - present[name]) / rate[name] for name in others)
    steps = max(math.ceil(fps), math.ceil(fps * seconds), elbow_steps + straighten_steps)
    ramp = []
    for step in range(1, steps + 1):
        tick = {name: present[name] + (target[name] - present[name]) * step / steps for name in others}
        t = min(1.0, step / elbow_steps)
        tick["elbow_flex"] = present["elbow_flex"] + (target["elbow_flex"] - present["elbow_flex"]) * t
        if step <= elbow_steps:
            tick["wrist_flex"] = present["wrist_flex"] + (bend - present["wrist_flex"]) * t
        else:
            tick["wrist_flex"] = bend + (target["wrist_flex"] - bend) * (step - elbow_steps) / (steps - elbow_steps)
        ramp.append(tick)
    return ramp


@RobotConfig.register_subclass("rebot_b601_follower")
@dataclass
class RebotB601FollowerRobotConfig(RobotConfig, RebotB601FollowerConfig):
    """Registered configuration for the reBot B601-DM follower robot."""

    pass
