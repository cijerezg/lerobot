from __future__ import annotations

import math
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
        self.power = 360
        self.voltage = 12000
        self.temp = 31.5
        self.status = 0


class _FakeUART:
    def __init__(self, ctrl) -> None:
        self.ctrl = ctrl
        self.closed = False
        self.requests = 0
        self.buffer = bytearray()

    def reset_input_buffer(self):
        self.buffer.clear()

    def write(self, data):
        self.requests += 1
        for servo_id, servo in self.ctrl.servos.items():
            if servo.angle_monitor is None:
                continue
            ratio = math.exp((1 / (servo.temp + 273.15) - 1 / 298.15) * 3435)
            temp_raw = round(4096 * ratio / (1 + ratio))
            params = struct.pack(
                "<BHHHHBih", servo_id, servo.voltage, servo.current, servo.power,
                temp_raw, servo.status, round(servo.angle_monitor * 10), 0,
            )
            packet = b"\x05\x1c" + bytes([22, len(params)]) + params
            self.buffer.extend(packet + bytes([sum(packet) % 256]))
        return len(data)

    @property
    def in_waiting(self):
        return len(self.buffer)

    def read(self, count):
        out = bytes(self.buffer[:count])
        del self.buffer[:count]
        return out

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
        raise AssertionError("The SDK's blocking monitor reader must not be used")

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
    uart = _FakeUART(ctrl)
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


def test_feedback_uses_sync_packet_mapping_and_power_overrides() -> None:
    controller, ctrl, _ = _controller()
    controller.enable_torque()
    ctrl.sync_commands.clear()  # discard the explicit hold command

    controller.send_positions(
        _feedback(controller.config, shoulder_pan=5.0, shoulder_lift=-5.0, gripper=-30.0)
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


def test_feedback_step_at_ceiling_is_sent_exactly() -> None:
    controller, ctrl, _ = _controller(feedback_max_raw_step_deg=8.0)
    controller.enable_torque()
    ctrl.sync_commands.clear()

    controller.send_positions(_feedback(controller.config, shoulder_pan=8.0))

    assert _decoded_payload(controller, ctrl)["shoulder_pan"][1] == -80
    assert controller.feedback_enabled is True


def test_feedback_step_beyond_ceiling_trips_fault_and_unloads_without_sending() -> None:
    controller, ctrl, _ = _controller(feedback_max_raw_step_deg=8.0)
    controller.enable_torque()
    ctrl.sync_commands.clear()

    with pytest.raises(LeaderFeedbackError, match="shoulder_pan asked to move -8.1 raw deg"):
        controller.send_positions(_feedback(controller.config, shoulder_pan=8.1))

    assert ctrl.sync_commands == []
    assert controller.feedback_enabled is False
    assert controller.feedback_fault is not None
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)
    with pytest.raises(LeaderFeedbackError, match="faulted and unloaded"):
        controller.send_positions(_feedback(controller.config))


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
        controller.send_positions(_feedback(controller.config, wrist_roll=5.0))

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
        ("monitor_read_timeout_s", 0.0),
        ("monitor_read_timeout_s", float("nan")),
        ("monitor_stale_timeout_s", 0.0),
        ("monitor_stale_timeout_s", float("inf")),
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


def test_monitor_miss_keeps_entire_previous_sample_and_retries_next_tick() -> None:
    controller, ctrl, uart = _controller()
    previous = controller.read_positions()
    last_good_time = controller._last_monitor_time
    ctrl.servos[0].angle_monitor = 12.0
    ctrl.servos[6].angle_monitor = None
    requests = uart.requests
    stale = controller.read_positions()
    assert uart.requests == requests + 1
    assert stale == previous  # no mixture of old and fresh joint positions
    assert controller.last_read_fresh is False
    assert controller._last_monitor_time == last_good_time
    assert ctrl.stop_commands == []

    ctrl.servos[6].angle_monitor = 1.0
    recovered = controller.read_positions()
    assert uart.requests == requests + 2
    assert recovered["shoulder_pan"] == -12.0
    assert controller.last_read_fresh is True
    assert controller.feedback_fault is None


def test_monitor_silence_faults_on_sample_age_not_retry_count() -> None:
    controller, ctrl, _ = _controller()
    controller.read_positions()
    for servo in ctrl.servos.values():
        servo.angle_monitor = None
    for _ in range(4):
        controller.read_positions()
    assert controller.feedback_fault is None
    controller._last_monitor_time = time.monotonic() - controller.config.monitor_stale_timeout_s
    with pytest.raises(LeaderFeedbackError, match="no fresh leader monitor sample"):
        controller.read_positions()
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)
    # Recovery requires explicit enable; a subsequent healthy reply cannot clear a fault.
    ctrl.servos[0].angle_monitor = 0.0
    with pytest.raises(LeaderFeedbackError, match="faulted and unloaded"):
        controller.read_positions()


def test_monitor_without_initial_sample_cannot_return_a_fabricated_pose() -> None:
    controller, ctrl, _ = _controller()
    for servo in ctrl.servos.values():
        servo.angle_monitor = None
    with pytest.raises(LeaderFeedbackError, match="no fresh leader monitor sample"):
        controller.read_positions()


def test_torque_enable_requires_a_fresh_sample_even_with_a_cache() -> None:
    controller, ctrl, _ = _controller()
    controller.read_positions()
    ctrl.servos[0].angle_monitor = None
    with pytest.raises(LeaderFeedbackError, match="no fresh leader monitor sample"):
        controller.enable_torque()
    assert not controller.feedback_enabled
    assert not ctrl.sync_commands


def test_intervention_release_waits_for_fresh_regular_read_without_extra_request() -> None:
    controller, ctrl, uart = _controller()
    controller.enable_torque()
    controller._handle_key_char("5")
    assert controller.is_intervening
    assert not controller.feedback_enabled
    requests = uart.requests
    controller._handle_key_char("5")
    assert uart.requests == requests  # key listener does not read the bus
    ctrl.servos[0].angle_monitor = None
    controller.read_positions()
    assert controller.is_intervening
    assert not controller.feedback_enabled
    ctrl.servos[0].angle_monitor = 5.0
    controller.read_positions()
    assert uart.requests == requests + 2
    assert not controller.is_intervening
    assert controller.feedback_enabled
    assert _decoded_payload(controller, ctrl)["shoulder_pan"][1] == 50


def test_cached_measurements_do_not_evaluate_or_erase_existing_health_timers() -> None:
    controller, ctrl, _ = _controller()
    controller.enable_torque()
    controller.read_positions()
    controller._raw_error_since["shoulder_pan"] = time.monotonic() - 10
    controller._current_since["shoulder_pan"] = time.monotonic() - 10
    ctrl.servos[6].angle_monitor = None
    controller.read_positions()
    assert "shoulder_pan" in controller._raw_error_since
    assert "shoulder_pan" in controller._current_since
    assert not controller.feedback_fault
    ctrl.servos[6].angle_monitor = 0.0
    controller.read_positions()  # a healthy fresh sample clears the timers
    assert not controller._raw_error_since
    assert not controller._current_since


def test_trace_records_reads_and_tracking_fault_dumps_the_last_second(tmp_path, caplog) -> None:
    controller, ctrl, _ = _controller(
        feedback_max_raw_error_deg=10.0,
        feedback_error_timeout_s=0.2,
    )
    controller.start_trace(tmp_path / "leader_trace.csv")
    controller._feedback_enabled = True
    controller._last_sent_raw = dict.fromkeys(controller.motor_names, 0.0)
    elbow = ctrl.servos[controller.config.joint_ids["elbow_flex"]]
    elbow.angle_monitor = 20.0
    elbow.current = 250
    elbow.status = 0x04  # stall error bit

    caplog.set_level("DEBUG")
    controller.read_positions()
    time.sleep(0.25)
    with pytest.raises(LeaderFeedbackError, match="elbow_flex remained 20.0"):
        controller.read_positions()

    lines = (tmp_path / "leader_trace.csv").read_text().splitlines()
    header = lines[0].split(",")
    assert header[:3] == ["time", "torqued", "intervening"]
    assert header[3:10] == [
        "shoulder_pan_target", "shoulder_pan_raw", "shoulder_pan_ma", "shoulder_pan_mv",
        "shoulder_pan_mw", "shoulder_pan_c", "shoulder_pan_status",
    ]
    assert len(lines) == 3
    last = dict(zip(header, lines[-1].split(","), strict=True))
    assert last["torqued"] == "1"
    assert (last["elbow_flex_target"], last["elbow_flex_raw"]) == ("0.0", "20.0")
    assert (last["elbow_flex_ma"], last["elbow_flex_mv"], last["elbow_flex_mw"], last["elbow_flex_c"]) == (
        "250", "12000", "360", "31.5"
    )
    assert last["elbow_flex_status"] == "4"

    messages = [record.getMessage() for record in caplog.records]
    assert "Leader elbow_flex status: stall" in messages
    assert sum("status: stall" in m for m in messages) == 1  # once per transition, not per read
    dump = next(m for m in messages if m.startswith("Leader elbow_flex, last 1.0 s"))
    rows = dump.splitlines()[1:]
    assert len(rows) == 2
    assert rows[-1].endswith("| stall")
    assert "|     0.0 |    20.0 |  -20.0 |  250 | 12000 |   360 |  31.5 |" in rows[-1]

    controller.close()


def test_feedback_ceiling_stretches_with_time_since_the_last_command() -> None:
    controller, ctrl, _ = _controller()
    controller.enable_torque()
    ctrl.sync_commands.clear()

    # Three ticks since the last command: 3 x 8 = 24 raw deg allowed, 20 is sent.
    controller._last_feedback_time = time.monotonic() - 3 * controller.config.feedback_step_period_s
    controller.send_positions(_feedback(controller.config, shoulder_pan=-20.0))
    assert _decoded_payload(controller, ctrl)["shoulder_pan"][1] == 200

    # A full second since the last command: capped at 4 x 8 = 32, so 33 more trips.
    controller._last_feedback_time = time.monotonic() - 1.0
    with pytest.raises(LeaderFeedbackError, match=r"shoulder_pan asked to move \+33.0 raw deg .*ceiling 32.0"):
        controller.send_positions(_feedback(controller.config, shoulder_pan=-53.0))
    assert controller.feedback_enabled is False


def test_monitor_io_failure_faults_instead_of_reusing_cache():
    controller, ctrl, uart = _controller()
    controller.read_positions()
    def failed_write(data):
        raise OSError("disconnected")
    uart.write = failed_write
    with pytest.raises(LeaderFeedbackError, match="leader monitor I/O failed: disconnected"):
        controller.read_positions()
    assert ctrl.stop_commands[-1] == (0xFF, STOP_UNLOAD, 0x00)


def test_stale_age_must_cover_the_transaction_budget():
    with pytest.raises(ValueError, match="monitor_stale_timeout_s must be at least"):
        _controller(monitor_read_timeout_s=0.020, monitor_stale_timeout_s=0.010)
