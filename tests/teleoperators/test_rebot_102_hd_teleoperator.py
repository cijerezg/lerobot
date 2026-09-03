from __future__ import annotations

import draccus
import pytest

from lerobot.teleoperators.rebot_102_leader.config_rebot_102_leader import (
    RebotArm102LeaderTeleopConfig,
)
from lerobot.teleoperators.rebot_102_leader.rebot_102_leader import RebotArm102Leader
from lerobot.teleoperators.utils import TeleopEvents


class _FakeHDController:
    instances: list[_FakeHDController] = []
    fail_configure = False

    def __init__(self, config) -> None:
        self.config = config
        self.connected = False
        self.calls: list[object] = []
        self.feedback = None
        type(self).instances.append(self)

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.calls.append("connect")
        self.connected = True

    def configure(self) -> None:
        self.calls.append("configure")
        if self.fail_configure:
            raise RuntimeError("configure failed")

    def set_origin_points(self) -> None:
        self.calls.append("set_origin_points")

    def read_positions(self):
        self.calls.append("read_positions")
        return {name: float(index) for index, name in enumerate(self.config.joint_ids)}

    def send_positions(self, feedback) -> None:
        self.calls.append("send_positions")
        self.feedback = feedback

    def enable_torque(self) -> None:
        self.calls.append("enable_torque")

    def disable_torque(self) -> None:
        self.calls.append("disable_torque")

    def get_teleop_events(self):
        self.calls.append("get_teleop_events")
        return {TeleopEvents.START_EPISODE: True}

    def close(self) -> None:
        self.calls.append("close")
        self.connected = False


def _config(tmp_path) -> RebotArm102LeaderTeleopConfig:
    return RebotArm102LeaderTeleopConfig(
        port="/dev/null",
        variant="102HD",
        id="test_hd",
        calibration_dir=tmp_path,
        enable_keyboard_handover=False,
    )


def test_leader_variant_rejects_typos(tmp_path) -> None:
    with pytest.raises(ValueError, match="102LD.*102HD"):
        RebotArm102LeaderTeleopConfig(
            port="/dev/null",
            variant="HD",
            id="bad_variant",
            calibration_dir=tmp_path,
        )


def test_explicit_hd_variant_decodes_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "leader.yaml"
    config_path.write_text("port: /dev/null\nvariant: 102HD\nid: yaml_hd\n")

    config = draccus.parse(RebotArm102LeaderTeleopConfig, config_path, args=[])

    assert config.variant == "102HD"


def test_hd_wrapper_exposes_and_delegates_full_feedback_contract(monkeypatch, tmp_path) -> None:
    import lerobot.teleoperators.rebot_102_leader.hd_controller as hd_module

    _FakeHDController.instances.clear()
    _FakeHDController.fail_configure = False
    monkeypatch.setattr(hd_module, "RebotArm102HDController", _FakeHDController)
    monkeypatch.setattr(
        "lerobot.teleoperators.rebot_102_leader.rebot_102_leader.require_package",
        lambda *args, **kwargs: None,
    )
    teleop = RebotArm102Leader(_config(tmp_path))

    assert teleop.feedback_features == teleop.action_features
    teleop.connect(calibrate=False)
    controller = _FakeHDController.instances[-1]
    assert controller.calls == ["connect", "configure"]
    assert teleop.is_connected is True

    teleop.enable_torque()
    feedback = dict.fromkeys(teleop.feedback_features, 1.0)
    teleop.send_feedback(feedback)
    assert controller.feedback is feedback
    assert set(teleop.get_action()) == set(teleop.action_features)
    assert teleop.get_teleop_events()[TeleopEvents.START_EPISODE] is True
    teleop.disable_torque()
    teleop.disconnect()

    assert controller.calls[-2:] == ["disable_torque", "close"]
    assert teleop.is_connected is False


def test_hd_connect_failure_closes_controller(monkeypatch, tmp_path) -> None:
    import lerobot.teleoperators.rebot_102_leader.hd_controller as hd_module

    _FakeHDController.instances.clear()
    _FakeHDController.fail_configure = True
    monkeypatch.setattr(hd_module, "RebotArm102HDController", _FakeHDController)
    monkeypatch.setattr(
        "lerobot.teleoperators.rebot_102_leader.rebot_102_leader.require_package",
        lambda *args, **kwargs: None,
    )
    teleop = RebotArm102Leader(_config(tmp_path))

    try:
        try:
            teleop.connect(calibrate=False)
        except RuntimeError as error:
            assert str(error) == "configure failed"
        else:
            raise AssertionError("connect should have failed")
        controller = _FakeHDController.instances[-1]
        assert controller.calls[-1] == "close"
        assert teleop.is_connected is False
    finally:
        _FakeHDController.fail_configure = False
