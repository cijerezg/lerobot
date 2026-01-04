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

import logging
import time
from queue import Queue
from threading import Lock

from pynput import keyboard

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so101_leader import SO101LeaderConfig

logger = logging.getLogger(__name__)


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.is_intervening = False
        self.is_success = False
        self.terminate_episode = False
        self.start_episode = False
        self.listener = None
        self.event_queue = Queue()
        self.bus_lock = Lock()

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        with self.bus_lock:
            self.bus.connect()
            if not self.is_calibrated and calibrate:
                logger.info(
                    "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
                )
                self.calibrate()

            self.configure()
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
        
        logger.info(f"{self} connected.")

    def _on_press(self, key):
        try:
            if hasattr(key, 'char'):
                if key.char == '5':
                    self.is_intervening = not self.is_intervening
                    logger.info(f"Intervention state toggled: {self.is_intervening}")
                    
                    with self.bus_lock:
                        if self.is_intervening:
                            # User is controlling: disable torque to allow movement
                            self.bus.disable_torque()
                            logger.info("Torque disabled for manual control.")
                        else:
                            # Leader follows follower: enable torque
                            self.bus.enable_torque()
                            logger.info("Torque enabled for feedback following.")
                
                elif key.char == '1':
                    self.is_success = True
                    logger.info("Success triggered manually.")
                elif key.char == '0':
                    self.terminate_episode = True
                    logger.info("Failure/Termination triggered manually.")
                elif key.char == '2':
                    self.start_episode = True
                    logger.info("Start Episode triggered manually.")

        except AttributeError:
            pass
        except Exception as e:
            logger.error(f"Error in keyboard listener: {e}")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        with self.bus_lock:
            for motor in reversed(self.bus.motors):
                input(f"Connect the controller board to the '{motor}' motor only and press enter.")
                self.bus.setup_motor(motor)
                print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        with self.bus_lock:
            action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        # logger.info(f"Leader Action: {action}")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_teleop_events(self) -> dict[str, bool]:
        events = {
            TeleopEvents.IS_INTERVENTION: self.is_intervening,
            TeleopEvents.TERMINATE_EPISODE: self.terminate_episode,
            TeleopEvents.SUCCESS: self.is_success,
            TeleopEvents.START_EPISODE: self.start_episode,
            TeleopEvents.RERECORD_EPISODE: False,
        }
        
        # Reset triggers
        if self.is_success:
            self.is_success = False
        if self.terminate_episode:
            self.terminate_episode = False
        if self.start_episode:
            self.start_episode = False
            
        return events

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Move the leader arm to the specified joint positions.
        """
        if self.is_intervening:
            # Do not send feedback if user is intervening to keep torque disabled
            return

        # Ensure we are writing to the correct register (Goal_Position)
        # We need to map the feedback keys (e.g. "shoulder_pan.pos") to motor names ("shoulder_pan")
        # and values.
        
        # Convert feedback dictionary to dict of goal positions
        goal_positions = {}
        for motor in self.bus.motors:
            if f"{motor}.pos" in feedback:
                goal_positions[motor] = feedback[f"{motor}.pos"]
            else:
                logger.warning(f"Missing feedback for motor {motor}")
                return

        # logger.info(f"Sending Feedback Goal Positions: {goal_positions}")

        with self.bus_lock:
            if self.is_intervening:
                return
            self.bus.sync_write("Goal_Position", goal_positions)

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        if self.listener:
            self.listener.stop()
            self.listener = None

        with self.bus_lock:
            self.bus.disconnect()
        
        logger.info(f"{self} disconnected.")
