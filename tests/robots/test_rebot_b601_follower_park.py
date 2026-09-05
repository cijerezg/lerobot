import math
from types import SimpleNamespace

import pytest

from lerobot.robots.rebot_b601_follower import rebot_b601_follower as module
from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig


class _FakeMotor:
    def __init__(self, pos_deg: float, moves: bool) -> None:
        self.pos_deg = pos_deg
        self.moves = moves
        self.sent: list[float] = []
        self.calls: list[str] = []

    def request_feedback(self) -> None:
        pass

    status = 1  # Damiao ERR nibble: enabled

    def get_state(self):
        return SimpleNamespace(pos=math.radians(self.pos_deg), status_code=self.status)

    def ensure_mode(self, mode) -> None:
        self.calls.append("ensure_mode")

    def _target(self, pos_rad: float) -> None:
        self.sent.append(math.degrees(pos_rad))
        if self.moves:
            self.pos_deg = math.degrees(pos_rad)

    def send_pos_vel(self, pos, vlim) -> None:
        self._target(pos)

    def send_force_pos(self, pos, vlim, ratio) -> None:
        self._target(pos)

    def disable(self) -> None:
        self.calls.append("disable")

    def clear_error(self) -> None:
        self.calls.append("clear_error")

    def close(self) -> None:
        self.calls.append("close")


class _FakeBus:
    def __init__(self) -> None:
        self.closed = False

    def enable_all(self) -> None:
        pass

    def poll_feedback_once(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


START = {
    "shoulder_pan": 40.0, "shoulder_lift": -90.0, "elbow_flex": -120.0, "wrist_flex": 30.0,
    "wrist_yaw": 10.0, "wrist_roll": -20.0, "gripper": -200.0,
}


def _robot(monkeypatch, moves: bool, torque: bool = True):
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    robot = module.RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null", id="park_test"))
    robot.bus = _FakeBus()
    robot.motors = {name: _FakeMotor(START[name], moves) for name in robot.motor_names}
    robot._torque_enabled = torque
    return robot


def test_park_freezes_then_descends_within_rate_and_arrives(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    robot.park()
    fps, rate = module._PARK_FPS, robot.config.park_deg_per_s
    for name, motor in robot.motors.items():
        assert motor.sent[0] == pytest.approx(START[name])
        steps = [b - a for a, b in zip(motor.sent, motor.sent[1:])]
        assert max(abs(s) for s in steps) <= rate / fps + 1e-6
        assert motor.sent[-1] == pytest.approx(robot.config.park_pose[name])


def test_disconnect_releases_only_after_arrival(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    robot.disconnect()
    for motor in robot.motors.values():
        assert motor.calls == ["disable", "clear_error", "close"]
    assert robot.bus is None


def test_disconnect_keeps_torque_when_arm_does_not_arrive(monkeypatch):
    robot = _robot(monkeypatch, moves=False)
    robot.disconnect()
    for motor in robot.motors.values():
        assert motor.calls == ["close"]


def test_disconnect_with_torque_already_off_does_not_park(monkeypatch):
    robot = _robot(monkeypatch, moves=True, torque=False)
    robot.disconnect()
    for motor in robot.motors.values():
        assert motor.sent == []
        assert motor.calls == ["disable", "clear_error", "close"]


def test_park_on_step_receives_every_ramp_target(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    seen: list[dict[str, float]] = []
    robot.park(on_step=seen.append)
    # one call per ramp tick after the freeze command (not a ramp target), plus the settle
    # ticks of the single arrival check an instantly-arriving arm needs
    settle_ticks = int(module._PARK_FPS * module._PARK_SETTLE_CHECK_SEC)
    assert len(seen) == len(robot.motors["shoulder_pan"].sent) - 1 + settle_ticks
    assert set(seen[0]) == {f"{name}.pos" for name in robot.motor_names}
    assert all(abs(seen[-1][f"{name}.pos"] - robot.config.park_pose[name]) < 1e-9 for name in robot.motor_names)
    ramp = len(seen) - settle_ticks
    assert all(abs(seen[i][f"{name}.pos"] - robot.motors[name].sent[i + 1]) < 1e-9
               for i in range(ramp) for name in robot.motor_names)
    assert all(seen[i] == seen[-1] for i in range(ramp, len(seen)))


class _PingMotor(_FakeMotor):
    """A motor that answers the liveness register read unless it is in `dead`."""

    dead: set[str] = set()

    def __init__(self, name: str) -> None:
        super().__init__(START[name], moves=True)
        self.name = name
        self.pings = 0

    def damiao_get_param_u32(self, rid, timeout_ms):
        self.pings += 1
        if self.name in self.dead:
            raise module.CallError("get_register_u32 failed: timeout")
        return 1


def test_get_observation_pings_one_motor_per_call_round_robin(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    _PingMotor.dead = set()
    robot.motors = {name: _PingMotor(name) for name in robot.motor_names}
    for _ in range(2 * len(robot.motor_names)):
        robot.get_observation()
    assert [m.pings for m in robot.motors.values()] == [2] * len(robot.motor_names)


def test_get_observation_names_the_motor_that_stops_answering(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    _PingMotor.dead = set()
    robot.motors = {name: _PingMotor(name) for name in robot.motor_names}
    for _ in range(3):
        robot.get_observation()  # pan, lift, elbow answered
    _PingMotor.dead = {"wrist_yaw", "wrist_roll", "gripper"}
    robot.get_observation()  # wrist_flex still answers
    with pytest.raises(RuntimeError, match="wrist_yaw stopped answering"):
        robot.get_observation()


def test_connect_names_the_silent_motor(monkeypatch):
    class _Bus(_FakeBus):
        def add_damiao_motor(self, send_id, recv_id, model):
            return _PingMotor(next(n for n, (s, _r) in cfg.motor_can_ids.items() if s == send_id))

    cfg = RebotB601FollowerRobotConfig(port="/dev/null", id="park_test")
    monkeypatch.setattr(module, "MotorBridgeController", SimpleNamespace(from_dm_serial=lambda **_: _Bus()))
    _PingMotor.dead = {"shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"}
    robot = module.RebotB601Follower(cfg)
    with pytest.raises(RuntimeError, match="shoulder_lift stopped answering"):
        robot.connect(calibrate=False)


def test_get_observation_kills_on_a_motor_that_is_not_enabled(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    _PingMotor.dead = set()
    robot.motors = {name: _PingMotor(name) for name in robot.motor_names}
    robot.get_observation()
    robot.motors["gripper"].status = 0  # rebooted: answers CAN, ignores commands
    with pytest.raises(RuntimeError, match="gripper reports status 0x0"):
        robot.get_observation()
    robot.motors["gripper"].status = 9  # undervoltage fault
    with pytest.raises(RuntimeError, match="gripper reports status 0x9"):
        robot.get_observation()


def test_configure_confirms_every_motor_enabled(monkeypatch):
    robot = _robot(monkeypatch, moves=True)
    robot.configure()
    assert all("ensure_mode" in m.calls for m in robot.motors.values())
    robot.motors["wrist_yaw"].status = 0
    with pytest.raises(RuntimeError, match="wrist_yaw reports status 0x0"):
        robot.configure()
