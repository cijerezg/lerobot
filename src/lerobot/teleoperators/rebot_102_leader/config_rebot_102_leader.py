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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class RebotArm102LeaderConfig:
    """Base configuration class for the Seeed Studio StarArm102 / reBot Arm 102 leader.

    The reBot Arm 102 is a 7-joint (incl. gripper) leader arm driven by FashionStar
    UART smart servos. Servo communication goes through ``motorbridge-smart-servo``.
    """

    # USB-to-UART device the leader arm is connected to (e.g. "/dev/ttyUSB0").
    port: str

    baudrate: int = 1_000_000

    # One monitor request per control tick; never retry within this budget.
    monitor_read_timeout_s: float = 0.020
    # Cached poses tolerate isolated misses, but not a disconnected leader.
    monitor_stale_timeout_s: float = 2.0

    # The original 102LD is encoder-only. Only the 102HD exposes position
    # feedback/shadowing so existing LD configurations remain read-only.
    # Kept as str because the repository's pinned draccus cannot decode Literal
    # fields from YAML; __post_init__ retains strict value validation.
    variant: str = "102LD"

    # Hardware-verified 102HD sync-control settings. Commands are refreshed by
    # the rollout loop; the 100 ms interval is the vendor's smooth-control path.
    feedback_interval_ms: int = 100
    feedback_acceleration_ms: int = 50
    feedback_deceleration_ms: int = 50
    feedback_power: int = 4095
    feedback_joint_powers: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_lift": 12000,
            "elbow_flex": 12000,  # 2026-09-05: was 8000; elbow lagged a policy-driven follower by 22 raw deg
        }
    )

    # A target further than this (raw servo degrees) from the last transmitted one trips
    # a fault and unloads the arm; it is never clipped and executed. Measured 2026-09-03
    # over the four training roots (46 episodes, 181k steps): no demo step exceeds 6.1 raw
    # deg and p99.9 is 1.1-2.7 per joint, so 8 (240 deg/s at 30 Hz) is only reached by a
    # discontinuity. The initial gap is closed by the runtime's linear approach, not here.
    feedback_max_raw_step_deg: float = 8.0
    # 2026-09-06: the ceiling is per command, but a stalled leader read (834 ms, twice in one
    # episode) freezes the loop while the follower keeps closing its command lead (up to 19 raw
    # deg observed), so the next observed pose is a legitimate 12 deg away. The ceiling therefore
    # scales with the time since the last command, one ceiling per control tick, capped so a real
    # discontinuity after a stall is still rejected.
    feedback_step_period_s: float = 1 / 30
    feedback_max_step_stretch: float = 4.0

    # A stale actor must not leave the leader rigid. Tracking/current faults are
    # sustained thresholds, chosen above all values in the verified trajectories.
    # 2026-09-05: raw error raised 20 -> 40. With the follower policy-driven the leader
    # only shadows it, and the 102HD elbow lags a follower moving at the rate ceiling
    # (tripped at 22.2 raw deg / 0.77 s); a lagging shadow is not a fault worth an unload.
    # Command watchdog is separate from the age of the last complete monitor sample.
    feedback_watchdog_timeout_s: float = 2.0
    feedback_max_raw_error_deg: float = 40.0
    feedback_error_timeout_s: float = 0.75
    feedback_max_current_ma: int = 1500
    feedback_current_timeout_s: float = 0.5

    # Preserve the established HIL controls: 5 toggles handover, 1 success,
    # 0 terminate/failure, and 2 starts an episode.
    enable_keyboard_handover: bool = True

    # Servo id of each joint on the UART bus.
    joint_ids: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": 0,
            "shoulder_lift": 1,
            "elbow_flex": 2,
            "wrist_flex": 3,
            "wrist_yaw": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
    )

    # Per-joint sign applied to raw servo angles so the leader matches the follower
    # convention. The gripper additionally carries a scale (e.g. -6) to widen its
    # range to the reBot B601 follower's gripper travel.
    joint_directions: dict[str, int] = field(
        default_factory=lambda: {
            "shoulder_pan": -1,
            "shoulder_lift": -1,
            "elbow_flex": 1,
            "wrist_flex": 1,
            "wrist_yaw": 1,
            "wrist_roll": -1,
            "gripper": -6,
        }
    )

    # Per-joint [min, max] output range in degrees. Matches the reBot B601 follower
    # joint limits so leader actions can drive the follower key-for-key.
    joint_ranges: dict[str, list[int]] = field(
        default_factory=lambda: {
            "shoulder_pan": [-150, 150],
            "shoulder_lift": [-170, 1],
            "elbow_flex": [-200, 1],
            "wrist_flex": [-80, 90],
            "wrist_yaw": [-90, 90],
            "wrist_roll": [-90, 90],
            "gripper": [-270, 0],
        }
    )

    def __post_init__(self) -> None:
        if self.variant not in {"102LD", "102HD"}:
            raise ValueError(f"variant must be '102LD' or '102HD', got {self.variant!r}")


@TeleoperatorConfig.register_subclass("rebot_102_leader")
@dataclass
class RebotArm102LeaderTeleopConfig(TeleoperatorConfig, RebotArm102LeaderConfig):
    """Registered configuration for the reBot Arm 102 leader teleoperator."""

    pass
