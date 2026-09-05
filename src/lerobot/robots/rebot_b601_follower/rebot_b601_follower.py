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

import logging
import math
import time
from functools import cached_property
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _motorbridge_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_follower import RebotB601FollowerRobotConfig

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
    from motorbridge.damiao_registers import RID_MST_ID
    from motorbridge.errors import CallError
else:
    MotorBridgeController = None
    MotorBridgeMode = None

logger = logging.getLogger(__name__)

# Joint controlled in FORCE_POS mode; every other joint runs in POS_VEL mode.
GRIPPER_MOTOR = "gripper"
# Per-joint Damiao motor models for the B601-DM (passed to motorbridge).
MOTOR_MODELS = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}
_ENSURE_MODE_RETRIES = 9
_SETTLE_SEC = 0.01
_ZERO_SETTLE_SEC = 0.1
_PARK_FPS = 30.0
_PARK_SETTLE_MAX_SEC = 5.0  # arrival is re-checked every _PARK_SETTLE_CHECK_SEC up to this
_PARK_SETTLE_CHECK_SEC = 0.5
_PING_TIMEOUT_MS = 20  # synchronous register read; get_state() only returns the cached frame
_ENABLE_CONFIRM_ROUNDS = 5  # feedback refreshes after enable_all before the status check
_STATUS_ENABLED = 1  # Damiao feedback ERR nibble: 0 disabled, 1 enabled, 8..E faults


class RebotB601Follower(Robot):
    """Seeed Studio reBot B601-DM follower arm (6-DOF + gripper, Damiao CAN motors).

    Motor communication is handled by the ``motorbridge`` package over a CAN bus,
    reached either through a Damiao serial bridge or a SocketCAN adapter.
    """

    config_class = RebotB601FollowerRobotConfig
    name = "rebot_b601_follower"

    def __init__(self, config: RebotB601FollowerRobotConfig):
        require_package("motorbridge", extra="rebot")
        super().__init__(config)
        self.config = config
        self.bus: MotorBridgeController | None = None
        self.motors: dict = {}
        self._torque_enabled = False
        self._check_index = 0  # get_observation() pings one motor per call, round-robin
        self.motor_names = list(config.motor_can_ids.keys())
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus is not None and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self} on {self.config.port} (adapter={self.config.can_adapter})...")
        if self.config.can_adapter == "damiao":
            self.bus = MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        elif self.config.can_adapter == "socketcan":
            self.bus = MotorBridgeController(channel=self.config.port)
        else:
            raise ValueError(
                f"Unsupported can_adapter '{self.config.can_adapter}'. Use 'damiao' or 'socketcan'."
            )

        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.motors[motor_name] = self.bus.add_damiao_motor(send_id, recv_id, MOTOR_MODELS[motor_name])
        for motor_name in self.motor_names:
            self._check_motor(motor_name)

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_all()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot B601 to its ZERO POSITION and close the gripper.\n"
            "See the B601 manual for the zero pose (the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(_ZERO_SETTLE_SEC)
        logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, (send_id, _recv_id) in self.config.motor_can_ids.items():
            range_min, range_max = self.config.joint_limits[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.enable_all()
        self._torque_enabled = True
        for motor_name, motor in self.motors.items():
            target_mode = (
                MotorBridgeMode.FORCE_POS if motor_name == GRIPPER_MOTOR else MotorBridgeMode.POS_VEL
            )
            for attempt in range(_ENSURE_MODE_RETRIES + 1):
                try:
                    motor.ensure_mode(target_mode)
                    break
                except Exception:
                    if attempt == _ENSURE_MODE_RETRIES:
                        raise
                    time.sleep(_SETTLE_SEC)
            logger.debug(f"{motor_name} mode set to {target_mode}")
        for _ in range(_ENABLE_CONFIRM_ROUNDS):
            self._present_pos()
            time.sleep(_SETTLE_SEC)
        self._check_status()

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging)."""
        self.bus.disable_all()
        self._torque_enabled = False
        logger.info(f"{self} torque disabled.")

    def _check_motor(self, motor_name: str) -> None:
        """Raise if the motor does not answer a synchronous register read. A motor that
        drops off the chain keeps returning its cached state, so this is the only way to
        tell a frozen joint from a still one."""
        try:
            self.motors[motor_name].damiao_get_param_u32(RID_MST_ID, timeout_ms=_PING_TIMEOUT_MS)
        except CallError as error:
            raise RuntimeError(f"{self} {motor_name} stopped answering on CAN ({error}). Stopping.") from error

    def _check_status(self) -> None:
        """Raise if a motor's last feedback frame says anything but enabled. A motor that
        rebooted (brownout) or latched a fault keeps answering CAN with fresh positions
        while ignoring every position command, which the ping cannot see."""
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None and state.status_code != _STATUS_ENABLED:
                raise RuntimeError(
                    f"{self} {motor_name} reports status 0x{state.status_code:X}, not enabled "
                    f"(0 = disabled, 8..E = Damiao fault). Stopping."
                )

    def _present_pos(self) -> dict[str, float]:
        """Read present joint positions in degrees."""
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("CAN bus poll feedback failed.")

        present_pos = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            present_pos[motor_name] = math.degrees(state.pos) if state is not None else 0.0
        return present_pos

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        self._check_motor(self.motor_names[self._check_index])
        self._check_index = (self._check_index + 1) % len(self.motor_names)
        obs_dict = {f"{motor}.pos": pos for motor, pos in self._present_pos().items()}
        self._check_status()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            # Depth is emitted at runtime (for the PNG16 sidecar writer) but is not a
            # dataset feature, so it's dropped by build_dataset_frame during recording.
            if getattr(self.config.cameras[cam_key], "use_depth", False):
                obs_dict[f"{cam_key}.depth"] = cam.read_latest_depth()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the arm to a target joint configuration.

        Positions are expressed in degrees. The relative action magnitude may be
        clipped depending on `max_relative_target`, so the action actually sent is
        always returned.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Clip against soft joint limits.
        for motor_name in list(goal_pos):
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped = max(min_limit, min(max_limit, goal_pos[motor_name]))
                if clipped != goal_pos[motor_name]:
                    logger.debug(f"Clipped {motor_name} from {goal_pos[motor_name]:.2f} to {clipped:.2f}")
                goal_pos[motor_name] = clipped

        # Tolerate 6-DOF leaders that have no wrist_yaw joint by holding it at zero.
        # This is intentional: it lets a 6-DOF leader such as the SO-100 / SO-101
        # (so100_leader / so101_leader) teleoperate this 7-DOF follower — the missing
        # wrist_yaw command is simply treated as 0.0 instead of raising.
        if "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

        # Cap relative target when too far from the present position.
        if self.config.max_relative_target is not None:
            present_pos = self._present_pos()
            goal_present_pos = {key: (g, present_pos.get(key, g)) for key, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        for motor_name, position_deg in goal_pos.items():
            motor = self.motors.get(motor_name)
            if motor is None:
                continue
            idx = self.motor_names.index(motor_name)
            vel_deg_s = (
                self.config.pos_vel_velocity[idx]
                if isinstance(self.config.pos_vel_velocity, list)
                else self.config.pos_vel_velocity
            )
            pos_rad = math.radians(position_deg)
            vel_rad = math.radians(vel_deg_s)
            if motor_name == GRIPPER_MOTOR:
                motor.send_force_pos(pos_rad, vel_rad, self.config.gripper_torque_ratio)
            else:
                motor.send_pos_vel(pos_rad, vel_rad)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def park(self, on_step=None) -> None:
        """Freeze, descend to config.park_pose at config.park_deg_per_s, verify arrival.

        `on_step`, when given, receives each ramp target ({joint}.pos -> deg) right after it
        is sent, so a mirrored device (the actuated leader) rides the same descent.
        Raises RuntimeError, with torque still on, if any joint ends farther than
        config.park_tolerance_deg from the park pose.
        """
        present = self._present_pos()
        self.send_action({f"{name}.pos": pos for name, pos in present.items()})
        target = self.config.park_pose
        gap = max(abs(target[name] - present[name]) for name in target)
        steps = max(int(_PARK_FPS), math.ceil(_PARK_FPS * gap / self.config.park_deg_per_s))
        logger.info(f"{self} parking: largest gap {gap:.1f} deg, {steps / _PARK_FPS:.1f} s")
        for step in range(1, steps + 1):
            t = step / steps
            step_target = {f"{name}.pos": present[name] + (target[name] - present[name]) * t for name in target}
            self.send_action(step_target)
            if on_step is not None:
                on_step(step_target)
            time.sleep(1.0 / _PARK_FPS)
        # Settle: the motors lag the ramp (elbow 13 deg after a 134 deg descent on 2026-09-04,
        # closed on its own later), so re-check arrival every _PARK_SETTLE_CHECK_SEC up to
        # _PARK_SETTLE_MAX_SEC. A mirrored leader keeps getting its target meanwhile (its
        # feedback watchdog is 0.5 s).
        for check in range(1, int(_PARK_SETTLE_MAX_SEC / _PARK_SETTLE_CHECK_SEC) + 1):
            for _ in range(int(_PARK_FPS * _PARK_SETTLE_CHECK_SEC)):
                if on_step is not None:
                    on_step(step_target)
                time.sleep(1.0 / _PARK_FPS)
            present = self._present_pos()
            error = {name: abs(present[name] - target[name]) for name in target}
            worst = max(error, key=error.get)
            if error[worst] <= self.config.park_tolerance_deg:
                break
        if error[worst] > self.config.park_tolerance_deg:
            raise RuntimeError(
                f"{self} did not reach the park pose: {worst} is {error[worst]:.1f} deg off "
                f"(tolerance {self.config.park_tolerance_deg}) after {_PARK_SETTLE_MAX_SEC:.0f} s; present "
                + ", ".join(f"{name}={present[name]:.1f}" for name in target)
                + ". Torque stays ON; support the arm and rerun park."
            )
        logger.info(
            f"{self} parked after {check * _PARK_SETTLE_CHECK_SEC:.1f} s settle "
            f"(worst joint {worst}, {error[worst]:.1f} deg off)."
        )

    @check_if_not_connected
    def disconnect(self) -> None:
        release = not self._torque_enabled
        if self._torque_enabled:
            try:
                self.park()
                release = True
            except Exception:
                logger.error(f"{self} park failed; leaving torque ON. Support the arm and run park_rebot.py.", exc_info=True)
        for motor in self.motors.values():
            if release:
                motor.disable()
                motor.clear_error()
            motor.close()

        self.bus.close()
        self.bus = None
        self.motors = {}

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
