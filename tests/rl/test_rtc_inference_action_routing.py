from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot.rl.rtc_actor_runtime import (
    _close_robot_hardware,
    _raw_joint_action,
    _teleop_supports_feedback,
    _validate_inference_action_routing,
)


class _FeedbackTeleop:
    def __init__(self, keys: set[str]) -> None:
        self.feedback_features = dict.fromkeys(keys, float)

    def enable_torque(self) -> None:
        pass

    def disable_torque(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        del feedback


def _env(*motors: str):
    return SimpleNamespace(
        robot=SimpleNamespace(bus=SimpleNamespace(motors={name: object() for name in motors})),
        get_last_requested_joint_targets=lambda: None,
    )


def test_safety_routing_accepts_exact_actuated_leader_joint_contract() -> None:
    env = _env("shoulder_pan", "gripper")
    teleop = _FeedbackTeleop({"shoulder_pan.pos", "gripper.pos"})

    assert _teleop_supports_feedback(teleop) is True
    _validate_inference_action_routing(env, teleop)


@pytest.mark.parametrize("teleop", [None, SimpleNamespace(feedback_features={})])
def test_safety_routing_rejects_non_actuated_teleoperator(teleop) -> None:
    with pytest.raises(RuntimeError, match="actuated teleoperator"):
        _validate_inference_action_routing(_env("joint"), teleop)


def test_safety_routing_rejects_feedback_features_without_sender() -> None:
    teleop = SimpleNamespace(
        feedback_features={"joint.pos": float},
        enable_torque=lambda: None,
        disable_torque=lambda: None,
    )

    with pytest.raises(RuntimeError, match="actuated teleoperator"):
        _validate_inference_action_routing(_env("joint"), teleop)


def test_safety_routing_rejects_joint_key_mismatch_with_diagnostics() -> None:
    teleop = _FeedbackTeleop({"shoulder_pan.pos", "extra.pos"})

    with pytest.raises(RuntimeError, match="missing=.*gripper.pos.*extra=.*extra.pos"):
        _validate_inference_action_routing(_env("shoulder_pan", "gripper"), teleop)


def test_queue_fallback_preserves_all_seven_rebot_joint_positions_in_order() -> None:
    raw = {
        "shoulder_pan.pos": 1.0,
        "shoulder_lift.pos": 2.0,
        "elbow_flex.pos": 3.0,
        "wrist_flex.pos": 4.0,
        "wrist_yaw.pos": 5.0,
        "wrist_roll.pos": 6.0,
        "gripper.pos": 7.0,
    }
    env = SimpleNamespace(get_raw_joint_positions=lambda: raw)

    assert _raw_joint_action(env, action_dim=8, device="cpu").tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        0.0,
    ]


def test_hardware_cleanup_unloads_leader_before_closing_follower() -> None:
    calls = []
    teleop = SimpleNamespace(
        is_connected=True,
        disconnect=lambda: calls.append("leader_disconnect"),
    )
    env = SimpleNamespace(close=lambda: calls.append("follower_close"))

    _close_robot_hardware(env, teleop, "TEST")

    assert calls == ["leader_disconnect", "follower_close"]


def test_hardware_cleanup_still_closes_follower_if_leader_disconnect_fails() -> None:
    calls = []

    def fail_disconnect() -> None:
        calls.append("leader_disconnect")
        raise OSError("UART close failed")

    teleop = SimpleNamespace(is_connected=True, disconnect=fail_disconnect)
    env = SimpleNamespace(close=lambda: calls.append("follower_close"))

    _close_robot_hardware(env, teleop, "TEST")

    assert calls == ["leader_disconnect", "follower_close"]
