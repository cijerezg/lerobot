"""Home hotkey integration, without hardware or dataset I/O."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.common import control_utils
from lerobot.scripts import lerobot_record as recording


@pytest.fixture
def setup_recording(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(recording.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(recording, "precise_sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(recording, "build_dataset_frame", lambda features, values, prefix: {prefix: dict(values)})
    depth = MagicMock()
    monkeypatch.setattr(recording, "write_depth", depth)
    monkeypatch.setattr(recording, "log_rerun_data", MagicMock())
    robot = MagicMock(spec=recording.rebot_b601_follower.RebotB601Follower)
    robot.config = SimpleNamespace(park_pose={"joint": 0.0}, park_deg_per_s=20.0)
    robot.action_features = {"joint.pos": float}
    robot.name = "rebot_b601_follower"
    pose = {"joint.pos": 20.0}
    robot.get_observation.side_effect = lambda: dict(pose)

    def send(action):
        pose.update(action)
        return dict(action)

    robot.send_action.side_effect = send
    dataset = SimpleNamespace(fps=30, features={}, frames=[], writer=None)
    dataset.add_frame = dataset.frames.append
    events = dict(exit_early=False, stop_recording=False, rerecord_episode=False, return_home=True)
    teleop = MagicMock(spec=recording.rebot_102_leader.RebotArm102Leader)
    teleop.config = SimpleNamespace(variant="102HD")
    teleop.get_action.return_value = {"joint.pos": 20.0}
    calls = []
    teleop.enable_torque.side_effect = lambda: calls.append("enable")
    teleop.send_feedback.side_effect = lambda action: calls.append(("feedback", dict(action)))
    teleop.disable_torque.side_effect = lambda: calls.append("release")

    def park(on_step, *, fps):
        assert fps == dataset.fps
        for position in (10.0, 0.0, 0.0):
            action = {"joint.pos": position}
            robot.send_action(action)
            on_step(action)
        calls.append("arrived")

    robot.park.side_effect = park
    return SimpleNamespace(robot=robot, dataset=dataset, events=events, teleop=teleop, calls=calls, depth=depth)


def run_loop(s, **kwargs):
    options = dict(
        robot=s.robot, events=s.events, fps=s.dataset.fps,
        teleop_action_processor=lambda pair: pair[0],
        robot_action_processor=lambda pair: pair[0],
        robot_observation_processor=lambda obs: obs,
        dataset=s.dataset, teleop=s.teleop, control_time_s=0.1,
        single_task="transfer pills", display_data=True, depth_stride=3,
    )
    options.update(kwargs)
    return recording.record_loop(**options)


def test_home_records_return_then_releases_leader_after_arrival(setup_recording):
    s = setup_recording
    assert run_loop(s) == {"joint.pos": 0.0}
    assert s.calls[0] == "enable"
    assert s.calls[-2:] == ["arrived", "release"]
    assert s.calls[-3] == ("feedback", {"joint.pos": 0.0})
    assert [frame["action"]["joint.pos"] for frame in s.dataset.frames[-3:]] == [10, 0, 0]
    assert all(frame["task"] == "transfer pills" for frame in s.dataset.frames)
    assert s.depth.call_count == len(s.dataset.frames)
    assert all(call.args[2] == 3 for call in s.depth.call_args_list)
    assert not s.events["return_home"]
    # Initial leader alignment only: teleop never drives the follower during park.
    s.teleop.get_action.assert_called_once()


def test_encoder_only_leader_does_not_receive_feedback(setup_recording):
    s = setup_recording
    s.teleop.config.variant = "102LD"
    assert run_loop(s) == {"joint.pos": 0.0}
    assert s.calls == ["arrived"]
    assert len(s.dataset.frames) == 3
    s.teleop.enable_torque.assert_not_called()


def test_reset_holds_follower_without_reading_leader(setup_recording):
    s = setup_recording
    run_loop(s, dataset=None, hold_position={"joint.pos": 0.0})
    s.teleop.get_action.assert_not_called()
    s.robot.park.assert_not_called()
    assert s.robot.send_action.call_count > 0
    assert all(call.args[0] == {"joint.pos": 0.0} for call in s.robot.send_action.call_args_list)
    assert s.dataset.frames == []
    assert not s.events["return_home"]


@pytest.mark.parametrize("event", ["stop_recording", "rerecord_episode"])
def test_existing_hotkeys_interrupt_park_and_release_leader(setup_recording, event):
    s = setup_recording

    def interrupted_park(on_step, *, fps):
        s.events[event] = True
        s.events["exit_early"] = True
        on_step({"joint.pos": 10.0})
        pytest.fail("Parking must stop after the interrupt")

    s.robot.park.side_effect = interrupted_park
    assert run_loop(s) == {"joint.pos": 20.0}
    assert s.calls[-1] == "release"
    assert s.events[event]
    assert not s.events["exit_early"]
    assert not s.events["return_home"]
    s.robot.send_action.assert_called_with({"joint.pos": 20.0})


def test_failed_arrival_propagates_and_releases_leader(setup_recording):
    s = setup_recording
    s.robot.park.side_effect = RuntimeError("did not reach the park pose")
    with pytest.raises(RuntimeError, match="did not reach"):
        run_loop(s)
    assert s.calls[-1] == "release"
    assert "arrived" not in s.calls
    assert not s.events["return_home"]


def test_leader_fault_still_parks_follower(setup_recording):
    s = setup_recording
    s.teleop.send_feedback.side_effect = recording.TeleopFeedbackError("unloaded")
    assert run_loop(s) == {"joint.pos": 0.0}
    assert s.calls[-2:] == ["arrived", "release"]
    s.teleop.send_feedback.assert_called_once()


def test_keyboard_home_is_opt_in_and_does_not_exit_early(monkeypatch):
    callbacks = []
    keys = SimpleNamespace(right=object(), left=object(), esc=object())
    keyboard = SimpleNamespace(Key=keys, Listener=lambda on_press: callbacks.append(on_press) or MagicMock())
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=keyboard))
    monkeypatch.setattr(control_utils, "is_headless", lambda: False)
    _, disabled = control_utils.init_keyboard_listener()
    callbacks[-1](SimpleNamespace(char="h"))
    assert not disabled["return_home"]
    _, enabled = control_utils.init_keyboard_listener(enable_return_home=True)
    callbacks[-1](SimpleNamespace(char="H"))
    assert enabled["return_home"]
    assert not enabled["exit_early"]
    callbacks[-1](keys.esc)
    assert enabled["stop_recording"] and enabled["exit_early"]


def test_precision_scales_arm_not_gripper_then_resyncs_leader(setup_recording):
    s = setup_recording
    s.events.update(return_home=False, frozen=False, precision=False)
    s.teleop.config.precision_scale = 0.25
    s.robot.action_features = {"joint.pos": float, "gripper.pos": float}
    reads = []

    def get_action():
        reads.append(None)
        if len(reads) == 2:
            s.events["precision"] = True  # P: this read is the anchor
        if len(reads) == 3:
            s.events["precision"] = False  # P mid-tick: this tick is still scaled
        if len(reads) == 1 or len(reads) == 2:
            return {"joint.pos": 20.0, "gripper.pos": 0.0}
        # The leader moved 8 deg and closed the gripper; the resync then drives it onto the follower.
        return {"joint.pos": 22.0 if "release" in s.calls else 28.0, "gripper.pos": -40.0}

    s.teleop.get_action.side_effect = get_action
    run_loop(s, control_time_s=4.5 / 30)
    assert [frame["action"] for frame in s.dataset.frames] == [
        {"joint.pos": 20.0, "gripper.pos": 0.0},
        {"joint.pos": 20.0, "gripper.pos": 0.0},
        {"joint.pos": 22.0, "gripper.pos": -40.0},
        {"joint.pos": 22.0, "gripper.pos": -40.0},
        {"joint.pos": 22.0, "gripper.pos": -40.0},
    ]
    assert max(call.args[0]["joint.pos"] for call in s.robot.send_action.call_args_list) == 22.0
    feedback = [call[1] for call in s.calls if isinstance(call, tuple)]
    assert s.calls[0] == "enable" and s.calls[-1] == "release"
    assert len(feedback) == 15 and feedback[0]["joint.pos"] == pytest.approx(27.6)
    assert feedback[-1] == {"joint.pos": 22.0, "gripper.pos": -40.0}
    assert not s.events["precision"]


@pytest.mark.parametrize("event", ["exit_early", "frozen", "timeout"])
def test_precision_resyncs_before_the_leader_drives_the_follower_again(setup_recording, event):
    s = setup_recording
    s.events.update(return_home=False, frozen=False, precision=True)
    s.teleop.config.precision_scale = 0.25
    reads = []

    def get_action():
        reads.append(None)
        if len(reads) == 2 and event != "timeout":
            s.events[event] = True
        if "release" in s.calls:
            s.events["frozen"] = False  # ends the freeze hold
            return {"joint.pos": 21.0}
        return {"joint.pos": 20.0 if len(reads) == 1 else 24.0}

    s.teleop.get_action.side_effect = get_action
    run_loop(s, control_time_s=1.5 / 30 if event == "timeout" else 0.3)
    assert max(call.args[0]["joint.pos"] for call in s.robot.send_action.call_args_list) == 21.0
    feedback = [call[1] for call in s.calls if isinstance(call, tuple)]
    assert feedback[14] == {"joint.pos": 21.0} and "release" in s.calls
    assert not s.events["precision"]


def test_precision_key_is_opt_in(monkeypatch):
    callbacks = []
    keys = SimpleNamespace(right=object(), left=object(), esc=object())
    keyboard = SimpleNamespace(Key=keys, Listener=lambda on_press: callbacks.append(on_press) or MagicMock())
    monkeypatch.setitem(sys.modules, "pynput", SimpleNamespace(keyboard=keyboard))
    monkeypatch.setattr(control_utils, "is_headless", lambda: False)
    _, disabled = control_utils.init_keyboard_listener()
    callbacks[-1](SimpleNamespace(char="p"))
    assert not disabled["precision"]
    _, enabled = control_utils.init_keyboard_listener(enable_precision=True)
    callbacks[-1](SimpleNamespace(char="P"))
    assert enabled["precision"]
    callbacks[-1](SimpleNamespace(char="p"))
    assert not enabled["precision"]
