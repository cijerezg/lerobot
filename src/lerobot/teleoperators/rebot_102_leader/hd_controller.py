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

"""Hardware-tested FashionStar sync controller for the Star Arm 102HD.

This module is imported lazily by :mod:`rebot_102_leader` only for the HD
variant. The encoder-only 102LD therefore keeps its existing dependency and
read path.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import select
import struct
import sys
import termios
import threading
import time
import traceback
import tty
from typing import TYPE_CHECKING

import fashionstar_uart_sdk as uservo
import serial

from ..utils import TeleopEvents, TeleopFeedbackError
from .mapping import clamp_position, position_to_raw, raw_to_position

if TYPE_CHECKING:
    from .config_rebot_102_leader import RebotArm102LeaderConfig

logger = logging.getLogger(__name__)

STOP_UNLOAD = 0x10
SYNC_MULTITURN_BY_INTERVAL = 14


class LeaderFeedbackError(TeleopFeedbackError):
    """Latched feedback fault that leaves the HD leader unloaded."""


class RebotArm102HDController:
    """Own the 102HD UART port, feedback state, watchdog, and handover keys."""

    def __init__(self, config: RebotArm102LeaderConfig):
        self.config = config
        self.motor_names = list(config.joint_ids)
        self.ids = [config.joint_ids[name] for name in self.motor_names]

        self.uart: serial.Serial | None = None
        self.ctrl: uservo.UartServoManager | None = None
        self.io_lock = threading.RLock()

        self.is_intervening = False
        self.is_success = False
        self.terminate_episode = False
        self.start_episode = False

        self._feedback_enabled = False
        self._feedback_fault: str | None = None
        self._last_feedback_time: float | None = None
        self._last_raw_positions: dict[str, float] = {}
        self._last_sent_raw: dict[str, float] | None = None
        self._currents_ma: dict[str, int] = {}
        self._raw_error_since: dict[str, float] = {}
        self._current_since: dict[str, float] = {}

        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self.listener = None
        self._terminal_listener_running = False
        self._terminal_thread: threading.Thread | None = None
        self._terminal_settings_lock = threading.Lock()
        self._terminal_fd: int | None = None
        self._terminal_old_settings = None
        self._terminal_settings_restored = True

        self._validate_config()

    @property
    def is_connected(self) -> bool:
        return self.uart is not None and self.ctrl is not None

    @property
    def feedback_enabled(self) -> bool:
        with self.io_lock:
            return self._feedback_enabled

    @property
    def feedback_fault(self) -> str | None:
        with self.io_lock:
            return self._feedback_fault

    def _validate_config(self) -> None:
        if self.config.feedback_interval_ms < (
            self.config.feedback_acceleration_ms + self.config.feedback_deceleration_ms
        ):
            raise ValueError(
                "feedback_interval_ms must be at least acceleration_ms + deceleration_ms"
            )
        if self.config.feedback_max_raw_step_deg <= 0:
            raise ValueError("feedback_max_raw_step_deg must be positive")
        for name, value in {
            "feedback_watchdog_timeout_s": self.config.feedback_watchdog_timeout_s,
            "feedback_max_raw_error_deg": self.config.feedback_max_raw_error_deg,
            "feedback_error_timeout_s": self.config.feedback_error_timeout_s,
            "feedback_max_current_ma": self.config.feedback_max_current_ma,
            "feedback_current_timeout_s": self.config.feedback_current_timeout_s,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        powers = dict.fromkeys(self.motor_names, self.config.feedback_power)
        powers.update(self.config.feedback_joint_powers)
        unknown = set(powers) - set(self.motor_names)
        if unknown:
            raise ValueError(f"unknown feedback_joint_powers joints: {sorted(unknown)}")
        if any(not 0 <= value <= 65535 for value in powers.values()):
            raise ValueError("feedback powers must fit the protocol's unsigned 16-bit field")

    def connect(self) -> None:
        with self.io_lock:
            if self.is_connected:
                return
            uart = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=1,
                bytesize=8,
                timeout=0,
            )
            ctrl = uservo.UartServoManager(uart, srv_num=8)
            try:
                for name, servo_id in self.config.joint_ids.items():
                    if not ctrl.ping(servo_id):
                        raise RuntimeError(f"Servo not found for {name} (id={servo_id}).")
                self.uart = uart
                self.ctrl = ctrl
                self._unload_locked()
            except Exception:
                with contextlib.suppress(Exception):
                    ctrl.stop_on_control_mode(0xFF, STOP_UNLOAD, 0x00)
                with contextlib.suppress(Exception):
                    uart.close()
                raise

        self._start_watchdog()

    def configure(self) -> None:
        with self.io_lock:
            self._require_connected()
            self._unload_locked()
            self.ctrl.reset_multi_turn_angle(0xFF)
            time.sleep(0.1)
            self._last_raw_positions = self._read_raw_locked()
        if self.config.enable_keyboard_handover:
            self._start_key_listener()

    def set_origin_points(self) -> None:
        with self.io_lock:
            self._require_connected()
            self._unload_locked()
            for servo_id in self.ids:
                self.ctrl.set_origin_point(servo_id)
                time.sleep(0.01)

    def read_positions(self) -> dict[str, float]:
        with self.io_lock:
            raw = self._read_raw_locked()
            self._last_raw_positions = raw
            self._evaluate_feedback_health_locked(raw, time.monotonic())
            return {
                name: raw_to_position(
                    raw[name], self.config.joint_ranges[name], self.config.joint_directions[name]
                )
                for name in self.motor_names
            }

    def _read_raw_locked(self) -> dict[str, float]:
        self._require_connected()
        result = self.ctrl.send_sync_servo_monitor(self.ids, realtime=True)
        raw: dict[str, float] = {}
        missing: list[str] = []
        currents: dict[str, int] = {}
        for name in self.motor_names:
            servo = result[self.config.joint_ids[name]]
            angle = servo.angle_monitor
            if angle is None:
                missing.append(name)
                continue
            raw[name] = float(angle)
            if servo.current is not None:
                currents[name] = int(servo.current)
        if missing:
            self._trip_fault_locked(f"no monitor reply from {', '.join(missing)}")
        self._currents_ma = currents
        return raw

    def send_positions(self, feedback: dict[str, float]) -> None:
        with self.io_lock:
            self._require_connected()
            if self.is_intervening:
                return
            if self._feedback_fault is not None:
                raise LeaderFeedbackError(
                    f"leader feedback is faulted and unloaded: {self._feedback_fault}; "
                    "call enable_torque() explicitly after resolving the cause"
                )
            if not self._feedback_enabled:
                raise RuntimeError(
                    "leader feedback torque is disabled; call enable_torque() before send_feedback()"
                )

            requested: dict[str, float] = {}
            for name in self.motor_names:
                key = f"{name}.pos"
                if key not in feedback:
                    reason = f"missing feedback position {key}"
                    self._trip_fault_locked(reason, raise_error=False)
                    raise ValueError(reason)
                value = float(feedback[key])
                if not math.isfinite(value):
                    reason = f"feedback position {key} is not finite: {value}"
                    self._trip_fault_locked(reason, raise_error=False)
                    raise ValueError(reason)
                clamped_position = clamp_position(value, self.config.joint_ranges[name])
                if clamped_position != value:
                    logger.warning(
                        "Leader feedback target %s=%.3f is outside %s; clamped to %.3f",
                        key,
                        value,
                        self.config.joint_ranges[name],
                        clamped_position,
                    )
                requested[name] = position_to_raw(
                    clamped_position,
                    self.config.joint_ranges[name],
                    self.config.joint_directions[name],
                )

            base = self._last_sent_raw
            ceiling = self.config.feedback_max_raw_step_deg
            for name in self.motor_names:
                jump = requested[name] - base[name]
                if abs(jump) > ceiling:
                    self._trip_fault_locked(
                        f"{name} asked to move {jump:+.1f} raw deg in one step, ceiling {ceiling:.1f}"
                    )
            try:
                self._send_raw_locked(requested)
            except Exception as error:
                reason = f"leader feedback command failed: {error}"
                self._trip_fault_locked(reason, raise_error=False)
                raise LeaderFeedbackError(reason) from error
            self._last_sent_raw = requested
            self._last_feedback_time = time.monotonic()

    def enable_torque(self) -> None:
        with self.io_lock:
            self._require_connected()
            self.is_intervening = False
            self._feedback_fault = None
            self._enable_feedback_locked(send_hold=True)

    def _enable_feedback_locked(self, send_hold: bool) -> None:
        raw = self._read_raw_locked()
        self._last_raw_positions = raw
        self._last_sent_raw = raw
        self._raw_error_since.clear()
        self._current_since.clear()
        self._feedback_enabled = True
        self._last_feedback_time = time.monotonic()
        if send_hold:
            try:
                self._send_raw_locked(raw)
            except Exception as error:
                reason = f"leader hold command failed while enabling torque: {error}"
                self._trip_fault_locked(reason, raise_error=False)
                raise LeaderFeedbackError(reason) from error

    def disable_torque(self) -> None:
        with self.io_lock:
            if self.is_connected:
                self._unload_locked()

    def _unload_locked(self) -> None:
        try:
            if self.ctrl is not None:
                self.ctrl.stop_on_control_mode(0xFF, STOP_UNLOAD, 0x00)
        finally:
            self._feedback_enabled = False
            self._last_feedback_time = None
            self._last_sent_raw = None
            self._raw_error_since.clear()
            self._current_since.clear()

    def _send_raw_locked(self, raw: dict[str, float]) -> None:
        powers = dict.fromkeys(self.motor_names, self.config.feedback_power)
        powers.update(self.config.feedback_joint_powers)
        payload = [
            struct.pack(
                "<BlLHHH",
                self.config.joint_ids[name],
                int(raw[name] * 10),
                self.config.feedback_interval_ms,
                self.config.feedback_acceleration_ms,
                self.config.feedback_deceleration_ms,
                powers[name],
            )
            for name in self.motor_names
        ]
        self.ctrl.send_sync_multiturnanglebyinterval(
            SYNC_MULTITURN_BY_INTERVAL, len(payload), payload
        )

    def _evaluate_feedback_health_locked(self, raw: dict[str, float], now: float) -> None:
        if not self._feedback_enabled or self._last_sent_raw is None or self.is_intervening:
            self._raw_error_since.clear()
            self._current_since.clear()
            return

        for name in self.motor_names:
            error = abs(self._last_sent_raw[name] - raw[name])
            if error > self.config.feedback_max_raw_error_deg:
                self._raw_error_since.setdefault(name, now)
                if now - self._raw_error_since[name] >= self.config.feedback_error_timeout_s:
                    self._trip_fault_locked(
                        f"{name} remained {error:.1f} raw deg from its target for "
                        f"{now - self._raw_error_since[name]:.2f}s"
                    )
            else:
                self._raw_error_since.pop(name, None)

            current = self._currents_ma.get(name, 0)
            current_magnitude = abs(current)
            if current_magnitude > self.config.feedback_max_current_ma:
                self._current_since.setdefault(name, now)
                if now - self._current_since[name] >= self.config.feedback_current_timeout_s:
                    self._trip_fault_locked(
                        f"{name} remained at {current_magnitude} mA, above "
                        f"{self.config.feedback_max_current_ma} mA, "
                        f"for {now - self._current_since[name]:.2f}s"
                    )
            else:
                self._current_since.pop(name, None)

    def _trip_fault_locked(self, reason: str, *, raise_error: bool = True) -> None:
        self._feedback_fault = reason
        try:
            self._unload_locked()
        except Exception:
            logger.error("Leader unload command failed while handling fault:\n%s", traceback.format_exc())
        logger.error("Leader feedback fault; unload requested: %s", reason)
        if raise_error:
            raise LeaderFeedbackError(reason)

    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()

        def loop() -> None:
            period = min(0.05, self.config.feedback_watchdog_timeout_s / 4)
            while not self._watchdog_stop.wait(period):
                with self.io_lock:
                    if not self.is_connected or not self._feedback_enabled:
                        continue
                    now = time.monotonic()
                    if (
                        self._last_feedback_time is not None
                        and now - self._last_feedback_time > self.config.feedback_watchdog_timeout_s
                    ):
                        self._trip_fault_locked(
                            f"no feedback command for {now - self._last_feedback_time:.3f}s",
                            raise_error=False,
                        )

        self._watchdog_thread = threading.Thread(
            target=loop, daemon=True, name="rebot_102hd_feedback_watchdog"
        )
        self._watchdog_thread.start()

    def _handle_key_char(self, char: str) -> None:
        if char == "5":
            with self.io_lock:
                self.is_intervening = not self.is_intervening
                if self.is_intervening:
                    self._unload_locked()
                    logger.info("Intervention enabled: leader torque unloaded for manual control.")
                else:
                    self._feedback_fault = None
                    self._enable_feedback_locked(send_hold=True)
                    logger.info("Intervention ended: leader feedback following resumed.")
        elif char == "1":
            self.is_success = True
            logger.info("Success triggered manually.")
        elif char == "0":
            self.terminate_episode = True
            logger.info("Failure/termination triggered manually.")
        elif char == "2":
            self.start_episode = True
            logger.info("Start episode triggered manually.")

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
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

    def _on_press(self, key) -> None:
        char = getattr(key, "char", None)
        if char is not None:
            self._handle_key_char(char)

    def _start_key_listener(self) -> None:
        if self.listener is not None or self._terminal_thread is not None:
            return
        force_terminal = os.environ.get("LEROBOT_TELEOP_TERMINAL_KEYS", "").lower() in (
            "1",
            "true",
            "yes",
        )
        pynput_started = False
        if not force_terminal:
            try:
                from pynput import keyboard

                self.listener = keyboard.Listener(on_press=self._on_press)
                self.listener.start()
                pynput_started = True
                logger.info("Started pynput global keyboard listener for leader handover.")
            except Exception as error:
                logger.warning(
                    "pynput keyboard listener unavailable (%s: %s); using terminal input",
                    error.__class__.__name__,
                    error,
                )
        if not pynput_started:
            self._start_terminal_listener()

    def _start_terminal_listener(self) -> None:
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY; keyboard-5 handover is unavailable")
            return
        self._terminal_listener_running = True

        def loop() -> None:
            fd = sys.stdin.fileno()
            try:
                old_settings = termios.tcgetattr(fd)
            except termios.error as error:
                logger.warning("Could not enable terminal handover keys: %s", error)
                return
            try:
                with self._terminal_settings_lock:
                    self._terminal_fd = fd
                    self._terminal_old_settings = old_settings
                    self._terminal_settings_restored = False
                tty.setcbreak(fd)
                logger.info(
                    "Leader keys active: 5=toggle intervention, 1=success, "
                    "0=terminate, 2=start episode"
                )
                while self._terminal_listener_running:
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable:
                        char = sys.stdin.read(1)
                        if char:
                            self._handle_key_char(char)
            finally:
                self._restore_terminal_settings()

        self._terminal_thread = threading.Thread(
            target=loop, daemon=True, name="rebot_102hd_terminal_keys"
        )
        self._terminal_thread.start()

    def _restore_terminal_settings(self) -> None:
        with self._terminal_settings_lock:
            if self._terminal_settings_restored:
                return
            fd = self._terminal_fd
            old_settings = self._terminal_old_settings
            self._terminal_settings_restored = True
            self._terminal_fd = None
            self._terminal_old_settings = None
        if fd is not None and old_settings is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _stop_background_threads(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._terminal_listener_running = False
        self._restore_terminal_settings()
        if self._terminal_thread is not None:
            self._terminal_thread.join(timeout=1.0)
            self._terminal_thread = None
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None

    def close(self) -> None:
        self._stop_background_threads()
        with self.io_lock:
            try:
                if self.ctrl is not None:
                    with contextlib.suppress(Exception):
                        self._unload_locked()
                if self.uart is not None:
                    with contextlib.suppress(Exception):
                        self.uart.close()
            finally:
                self._feedback_enabled = False
                self.ctrl = None
                self.uart = None

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("102HD leader is not connected")
