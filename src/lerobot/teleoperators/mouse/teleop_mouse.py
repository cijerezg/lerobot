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
import select
import sys
import termios
import threading
import tty
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..gamepad.teleop_gamepad import GripperAction
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_mouse import MouseTeleopConfig

logger = logging.getLogger(__name__)


class MouseTeleop(Teleoperator):
    """Mouse / trackpad teleoperation for human intervention during policy rollouts.

    Emits the same action features as `GamepadTeleop` (`delta_x/y/z` plus a discrete
    gripper command), so it drops straight into the existing end-effector pipeline
    (`EEReferenceAndDelta` -> `EEBoundsAndSafety` -> `GripperVelocityToJoint` ->
    `InverseKinematicsRLStep`) with no changes to that pipeline.

    Controls:
        right click     toggle intervention on/off (grabs/releases the device)
        pointer motion  end-effector x / y
        wheel           end-effector z
        left click      toggle gripper open/closed
        1 / 0 / 2       success / terminate / start episode (terminal keys)

    Wrist orientation is not commanded: the pipeline receives a zero rotation delta,
    so the IK target keeps the orientation the arm already had.
    """

    config_class = MouseTeleopConfig
    name = "mouse"

    def __init__(self, config: MouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.device = None
        self._grabbed = False
        self._running = False
        self._reader_thread = None
        self._lock = threading.Lock()

        # Motion counts accumulated by the reader thread, drained by get_action().
        self._dx = 0
        self._dy = 0
        self._dwheel = 0

        self.is_intervening = False
        self.gripper_closed = False
        self.is_success = False
        self.terminate_episode = False
        self.start_episode = False

        self._terminal_thread = None
        self._terminal_fd = None
        self._terminal_old_settings = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.device is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _find_device(self):
        import evdev

        if self.config.device_path is not None:
            return evdev.InputDevice(self.config.device_path)

        paths = evdev.list_devices()
        if not paths:
            raise RuntimeError(
                "No readable /dev/input devices. Add yourself to the `input` group "
                "(`sudo usermod -aG input $USER`) and log out and back in."
            )
        for path in paths:
            dev = evdev.InputDevice(path)
            if evdev.ecodes.REL_X in dev.capabilities().get(evdev.ecodes.EV_REL, []):
                return dev
            dev.close()
        raise RuntimeError(f"No pointer device advertising REL_X among {paths}.")

    def connect(self, calibrate: bool = True) -> None:
        self.device = self._find_device()
        logger.info(f"{self} using pointer device {self.device.path} ({self.device.name})")

        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True, name="mouse_teleop")
        self._reader_thread.start()

        if self.config.enable_terminal_keys:
            self._start_terminal_listener()

        logger.info(
            "Mouse teleop ready. Right click = toggle intervention, pointer = x/y, "
            "wheel = z, left click = toggle gripper."
        )

    def _read_loop(self) -> None:
        import evdev

        while self._running:
            r, _, _ = select.select([self.device.fd], [], [], 0.1)
            if not r:
                continue
            try:
                events = list(self.device.read())
            except (OSError, BlockingIOError):
                continue

            for event in events:
                if event.type == evdev.ecodes.EV_REL:
                    with self._lock:
                        if event.code == evdev.ecodes.REL_X:
                            self._dx += event.value
                        elif event.code == evdev.ecodes.REL_Y:
                            self._dy += event.value
                        elif event.code in (evdev.ecodes.REL_WHEEL, evdev.ecodes.REL_HWHEEL):
                            self._dwheel += event.value
                elif event.type == evdev.ecodes.EV_KEY and event.value == 1:
                    if event.code == evdev.ecodes.BTN_RIGHT:
                        self._toggle_intervention()
                    elif event.code == evdev.ecodes.BTN_LEFT:
                        self.gripper_closed = not self.gripper_closed
                        logger.info(f"Gripper toggled: {'closed' if self.gripper_closed else 'open'}")

    def _toggle_intervention(self) -> None:
        self.is_intervening = not self.is_intervening
        with self._lock:
            self._dx = self._dy = self._dwheel = 0

        if self.config.grab_device:
            try:
                if self.is_intervening and not self._grabbed:
                    self.device.grab()
                    self._grabbed = True
                elif not self.is_intervening and self._grabbed:
                    self.device.ungrab()
                    self._grabbed = False
            except OSError as e:
                logger.warning(f"Could not change device grab state: {e}")

        logger.info(f"Intervention state toggled: {self.is_intervening}")

    def _handle_key_char(self, char: str) -> None:
        if char == "1":
            self.is_success = True
            logger.info("Success triggered manually.")
        elif char == "0":
            self.terminate_episode = True
            logger.info("Failure/Termination triggered manually.")
        elif char == "2":
            self.start_episode = True
            logger.info("Start Episode triggered manually.")

    def _start_terminal_listener(self) -> None:
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY; episode keys (1/0/2) disabled.")
            return

        def _loop() -> None:
            fd = sys.stdin.fileno()
            try:
                self._terminal_old_settings = termios.tcgetattr(fd)
            except termios.error as e:
                logger.warning(f"Could not put TTY into cbreak mode ({e}); episode keys disabled.")
                return
            self._terminal_fd = fd
            try:
                tty.setcbreak(fd)
                while self._running:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        self._handle_key_char(sys.stdin.read(1))
            finally:
                self._restore_terminal_settings()

        self._terminal_thread = threading.Thread(target=_loop, daemon=True, name="mouse_teleop_keys")
        self._terminal_thread.start()

    def _restore_terminal_settings(self) -> None:
        if self._terminal_fd is None or self._terminal_old_settings is None:
            return
        try:
            termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_old_settings)
        except Exception as e:
            logger.warning(f"Could not restore terminal settings: {e}")
        self._terminal_fd = None
        self._terminal_old_settings = None

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._lock:
            dx, dy, dwheel = self._dx, self._dy, self._dwheel
            self._dx = self._dy = self._dwheel = 0

        if not self.is_intervening:
            dx = dy = dwheel = 0

        delta = np.array(
            [
                dx / self.config.counts_per_unit,
                dy / self.config.counts_per_unit,
                dwheel / self.config.wheel_counts_per_unit,
            ],
            dtype=np.float32,
        )
        signs = np.array(
            [
                -1.0 if self.config.invert_x else 1.0,
                -1.0 if self.config.invert_y else 1.0,
                -1.0 if self.config.invert_z else 1.0,
            ],
            dtype=np.float32,
        )
        delta = np.clip(delta * signs, -self.config.max_delta, self.config.max_delta)

        action_dict = {
            "delta_x": float(delta[0]),
            "delta_y": float(delta[1]),
            "delta_z": float(delta[2]),
        }
        if self.config.use_gripper:
            action_dict["gripper"] = float(
                GripperAction.CLOSE.value if self.gripper_closed else GripperAction.OPEN.value
            )
        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        events = {
            TeleopEvents.IS_INTERVENTION: self.is_intervening,
            TeleopEvents.TERMINATE_EPISODE: self.terminate_episode,
            TeleopEvents.SUCCESS: self.is_success,
            TeleopEvents.START_EPISODE: self.start_episode,
            TeleopEvents.RERECORD_EPISODE: False,
        }
        self.is_success = False
        self.terminate_episode = False
        self.start_episode = False
        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self._running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._terminal_thread is not None:
            self._terminal_thread.join(timeout=1.0)
            self._terminal_thread = None
        self._restore_terminal_settings()
        if self.device is not None:
            if self._grabbed:
                try:
                    self.device.ungrab()
                except OSError:
                    pass
                self._grabbed = False
            self.device.close()
            self.device = None
