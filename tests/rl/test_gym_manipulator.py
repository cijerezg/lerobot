from __future__ import annotations

import numpy as np
import pytest

from lerobot.rl.gym_manipulator import RobotEnv, make_robot_env


class _FakeBus:
    motors = {"joint": object()}


class _FakeRobot:
    def __init__(self) -> None:
        self.bus = _FakeBus()
        self.sent_actions: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float]) -> None:
        self.sent_actions.append(action)

    def get_observation(self) -> dict[str, float]:
        return {"joint.pos": 3.0}


def _make_env(*, send_actions_to_robot: bool) -> RobotEnv:
    env = RobotEnv.__new__(RobotEnv)
    env.robot = _FakeRobot()
    env._joint_names = ["joint"]
    env._image_keys = []
    env.display_cameras = False
    env.current_step = 0
    env.send_actions_to_robot = send_actions_to_robot
    env._last_requested_joint_targets = None
    return env


@pytest.mark.parametrize("send_actions_to_robot", [True, False])
def test_step_can_suppress_robot_action_without_suppressing_observation(
    send_actions_to_robot: bool,
) -> None:
    env = _make_env(send_actions_to_robot=send_actions_to_robot)

    observation, _, _, _, _ = env.step(np.asarray([9.0], dtype=np.float32))

    expected = [{"joint.pos": np.float32(9.0)}] if send_actions_to_robot else []
    assert env.robot.sent_actions == expected
    assert env.get_last_requested_joint_targets() == {"joint.pos": np.float32(9.0)}
    assert observation["agent_pos"].tolist() == [3.0]
    assert observation["joint.pos"] == 3.0
    assert env.get_raw_joint_positions() == {"joint.pos": 3.0}


def test_safety_mode_suppresses_configured_follower_reset_motion(monkeypatch) -> None:
    env = _make_env(send_actions_to_robot=False)
    env.reset_pose = [9.0]
    env.reset_time_s = 0.0
    reset_calls = []
    monkeypatch.setattr(
        "lerobot.rl.gym_manipulator.reset_follower_position",
        lambda robot, pose: reset_calls.append((robot, pose)),
    )

    observation, _ = env.reset()

    assert reset_calls == []
    assert observation["agent_pos"].tolist() == [3.0]


def test_environment_setup_failure_disconnects_already_connected_teleop(monkeypatch) -> None:
    class _FakeTeleop:
        is_connected = False

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

    teleop = _FakeTeleop()
    cfg = type(
        "Cfg",
        (),
        {
            "name": "real_robot",
            "robot": object(),
            "teleop": object(),
            "processor": type(
                "Processor",
                (),
                {"gripper": None, "observation": None, "reset": None},
            )(),
        },
    )()
    monkeypatch.setattr("lerobot.rl.gym_manipulator.make_robot_from_config", lambda config: object())
    monkeypatch.setattr(
        "lerobot.rl.gym_manipulator.make_teleoperator_from_config", lambda config: teleop
    )
    monkeypatch.setattr(
        "lerobot.rl.gym_manipulator.RobotEnv",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("follower setup failed")),
    )

    with pytest.raises(RuntimeError, match="follower setup failed"):
        make_robot_env(cfg)

    assert teleop.is_connected is False
