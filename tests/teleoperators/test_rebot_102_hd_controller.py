from __future__ import annotations

import struct
import time

import pytest

from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader import RebotArm102LeaderConfig
from lerobot.teleoperators.rebot_102_leader.hd_controller import (
    STOP_UNLOAD,
    SYNC_MULTITURN_BY_INTERVAL,
    LeaderFeedbackError,
    RebotArm102HDController,
)


class _FakeServo:
    def __init__(self, angle: float = 0.0, current: int = 30) -> None:
        self.angle_monitor = angle
        self.current = current


class _FakeUART:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeCtrl:
    def __init__(self, config: RebotArm102LeaderConfig) -> None:
        self.config = config
        self.servos = {servo_id: _FakeServo() for servo_id in config.joint_ids.values()}
        self.sync_commands: list[tuple[int, int, list[bytes]]] = []
        self.stop_commands: list[tuple[int, int, int]] = []
        self.fail_send = False

    def send_sync_servo_monitor(self, ids, realtime=True):
        del realtime
        return {servo_id: self.servos[servo_id] for servo_id in ids}

    def send_sync_multiturnanglebyinterval(self, command, count, payload):
        if self.fail_send:
            raise OSError("UART write failed")
        self.sync_commands.append((command, count, payload))

    def stop_on_control_mode(self, servo_id, method, power):
        self.stop_commands.append((servo_id, method, power))


def _controller(**overrides) -> tuple[RebotArm102HDController, _FakeCtrl, _FakeUART]:
    config = RebotArm102LeaderConfig(
        port="/dev/null",
        variant="102HD",
        enable_keyboard_handover=False,
        **overrides,
    )
    controller = RebotArm102HDController(config)
    ctrl = _FakeCtrl(config)
    uart = _FakeUART()
    controller.ctrl = ctrl
    controller.uart = uart
    controller._last_raw_positions = dict.fromkeys(config.joint_ids, 0.0)
    return controller, ctrl, uart


def _feedback(config: RebotArm102LeaderConfig, **positions: float) -> dict[str, float]:
    return {f"{name}.pos": positions.get(name, 0.0) for name in config.joint_ids}


def _decoded_payload(controller: RebotArm102HDController, ctrl: _FakeCtrl):
    command, count, payload = ctrl.sync_commands[-1]
    assert command == SYNC_MULTITURN_BY_INTERVAL
    assert count == len(controller.motor_names)
    return {
        name: struct.unpack("<BlLHHH", packed)
        for name, packed in zip(controller.motor_names, payload, strict=True)
    }


def test_feedback_requires_explicit_torque_enable() -> None:
    controller, ctrl, _ = _controller()

    with pytest.raises(RuntimeError, match="enable_torque"):
        controller.send_positions(_feedback(controller.config))

    assert ctrl.sync_commands == []


def test_feedback_uses_sync_packet_mapping_step_limit_and_power_overrides() -> None:
    controller, ctrl, _ = _controller(feedback_max_raw_step_deg=5.0)
    controller.enable_torque()
    ctrl.sync_commands.clear()  # discard the explicit hold command

    controller.send_positions(
        _feedback(controller.config, shoulder_pan=100.0, shoulder_lift=-100.0, gripper=-270.0)
    )

    decoded = _decoded_payload(controller, ctrl)
    pan = decoded["shoulder_pan"]
    lift = decoded["shoulder_lift"]
    gripper = decoded["gripper"]
    assert pan[0:3] == (controller.config.joint_ids["shoulder_pan"], -50, 100)
    assert lift[0:3] == (controller.config.joint_ids["shoulder_lift"], 50, 100)
    assert gripper[0:3] == (controller.config.joint_ids["gripper"], 50, 100)
    assert pan[3:5] == (50, 50)
    assert pan[5] == controller.config.feedback_power
    assert lift[5] == controller.config.feedback_joint_powers["shoulder_lift"]


def test_feedback_clamps_to_declared_joint_ranges_before_mapping() -> None:
    controller, ctrl, _ = _controller(feedback_max_raw_step_deg=1000.0)
    controller.enable_torque()
    ctrl.sync_commands.clear()

    controller.send_positions(
        _feedback(controller.config, shoulder_pan=999.0, gripper=-999.0)
    )

    decoded = _decoded_payload(controller, ctrl)
    assert decoded["shoulder_pan"][1] == -1500
    assert decoded["gripper"][1] == 450


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_feedback_rejects_non_finite_targets_without_sending(bad_value: float) -> None:
    controller, ctrl, _ = _controller()
    controller.enable_torque()
    ctrl.sync_commands.clear()

    with pytest.raises(ValueError, match="not finite"):
        controller.send_positions(_feedback(controller.config, wrist_roll=bad_value))

    assert ctrl.sync_commands == []
    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)


def test_feedback_rejects_incomplete_joint_dictionary_without_sending() -> None:
    controller, ctrl, _ = _controller()
    controller.enable_torque()
    ctrl.sync_commands.clear()

    with pytest.raises(ValueError, match="missing feedback position"):
        controller.send_positions({"shoulder_pan.pos": 0.0})

    assert ctrl.sync_commands == []
    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None


def test_uart_send_failure_latches_fault_and_immediately_requests_unload() -> None:
    controller, ctrl, _ = _controller()
    controller.enable_torque()
    ctrl.sync_commands.clear()
    ctrl.fail_send = True

    with pytest.raises(LeaderFeedbackError, match="UART write failed"):
        controller.send_positions(_feedback(controller.config, wrist_roll=10.0))

    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)


def test_intervention_suppresses_feedback_without_reenabling_torque() -> None:
    controller, ctrl, _ = _controller()
    controller.is_intervening = True

    controller.send_positions(_feedback(controller.config, shoulder_pan=20.0))

    assert ctrl.sync_commands == []
    assert controller.feedback_enabled is False


def test_sustained_tracking_error_latches_fault_and_unloads() -> None:
    controller, ctrl, _ = _controller(
        feedback_max_raw_error_deg=10.0,
        feedback_error_timeout_s=0.2,
    )
    controller._feedback_enabled = True
    controller._last_sent_raw = dict.fromkeys(controller.motor_names, 0.0)
    raw = dict(controller._last_sent_raw)
    raw["shoulder_pan"] = 20.0

    controller._evaluate_feedback_health_locked(raw, now=1.0)
    with pytest.raises(LeaderFeedbackError, match="shoulder_pan"):
        controller._evaluate_feedback_health_locked(raw, now=1.21)

    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)


@pytest.mark.parametrize("current_ma", [1600, -1600])
def test_sustained_current_magnitude_latches_fault_and_unloads(current_ma: int) -> None:
    controller, ctrl, _ = _controller(
        feedback_max_current_ma=1500,
        feedback_current_timeout_s=0.2,
    )
    controller._feedback_enabled = True
    controller._last_sent_raw = dict.fromkeys(controller.motor_names, 0.0)
    controller._currents_ma = {"shoulder_pan": current_ma}
    raw = dict(controller._last_sent_raw)

    controller._evaluate_feedback_health_locked(raw, now=1.0)
    with pytest.raises(LeaderFeedbackError, match="shoulder_pan"):
        controller._evaluate_feedback_health_locked(raw, now=1.21)

    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)


def test_watchdog_unloads_after_feedback_stalls() -> None:
    controller, ctrl, uart = _controller(feedback_watchdog_timeout_s=0.05)
    controller._feedback_enabled = True
    controller._last_feedback_time = time.monotonic() - 1.0
    controller._start_watchdog()
    deadline = time.monotonic() + 0.5
    while controller.feedback_enabled and time.monotonic() < deadline:
        time.sleep(0.005)
    controller.close()

    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)
    assert uart.closed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feedback_max_raw_step_deg", 0.0),
        ("feedback_watchdog_timeout_s", 0.0),
        ("feedback_max_raw_error_deg", 0.0),
        ("feedback_error_timeout_s", 0.0),
        ("feedback_max_current_ma", 0),
        ("feedback_current_timeout_s", 0.0),
    ],
)
def test_invalid_safety_thresholds_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        RebotArm102HDController(
            RebotArm102LeaderConfig(
                port="/dev/null",
                variant="102HD",
                enable_keyboard_handover=False,
                **{field: value},
            )
        )
