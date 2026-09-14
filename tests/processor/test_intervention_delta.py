import torch

from lerobot.processor.hil_processor import (
    TELEOP_ACTION_KEY,
    TELEOP_ACTION_IS_FRESH,
    AddTeleopActionAsComplimentaryDataStep,
    InterventionActionProcessorStep,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey


def _transition(leader, follower, intervening):
    return {
        TransitionKey.ACTION: torch.zeros(3),
        TransitionKey.OBSERVATION: follower,
        TransitionKey.COMPLEMENTARY_DATA: {TELEOP_ACTION_KEY: leader},
        TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: intervening},
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.REWARD: 0.0,
    }


def test_joint_intervention_is_delta_with_absolute_gripper():
    step = InterventionActionProcessorStep()
    follower = {"shoulder_lift.pos": -50.0, "elbow_flex.pos": -100.0, "gripper.pos": -10.0}
    # The shadowing leader trails the follower by 20 deg on the elbow at the keypress.
    leader = {"shoulder_lift.pos": -45.0, "elbow_flex.pos": -80.0, "gripper.pos": -40.0}

    # First intervening step: the arm stays put, the gripper takes the leader's value.
    out = step(_transition(leader, follower, True))
    assert out[TransitionKey.ACTION].tolist() == [-50.0, -100.0, -40.0]

    # Moving the leader moves the follower by the same displacement.
    moved = {"shoulder_lift.pos": -35.0, "elbow_flex.pos": -90.0, "gripper.pos": -40.0}
    out = step(_transition(moved, follower, True))
    assert out[TransitionKey.ACTION].tolist() == [-40.0, -110.0, -40.0]

    # Ending the intervention clears the offset; the next one captures a fresh gap.
    step(_transition(moved, follower, False))
    follower2 = {"shoulder_lift.pos": 0.0, "elbow_flex.pos": 0.0, "gripper.pos": 0.0}
    out = step(_transition(moved, follower2, True))
    assert out[TransitionKey.ACTION].tolist() == [0.0, 0.0, -40.0]


def test_stale_handover_holds_all_joints_and_defers_delta_anchor():
    step = InterventionActionProcessorStep()
    follower = {"shoulder_lift.pos": -50.0, "elbow_flex.pos": -100.0, "gripper.pos": -10.0}
    stale = dict.fromkeys(follower, 0.0)
    def tick(leader, observation, fresh):
        transition = _transition(leader, observation, True)
        transition[TransitionKey.ACTION] = torch.full((3,), 999.0)
        transition[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_IS_FRESH] = fresh
        return step(transition)[TransitionKey.ACTION].tolist()

    assert tick(stale, follower, False) == [-50.0, -100.0, -10.0]
    assert step._offset is None
    # Repeated misses hold the same target even if measured position drifts.
    drifted = {k: v + 1 for k, v in follower.items()}
    assert tick(stale, drifted, False) == [-50.0, -100.0, -10.0]
    fresh = {"shoulder_lift.pos": -30.0, "elbow_flex.pos": -80.0, "gripper.pos": -40.0}
    assert tick(fresh, drifted, True) == [-50.0, -100.0, -40.0]
    moved = {"shoulder_lift.pos": -25.0, "elbow_flex.pos": -90.0, "gripper.pos": -30.0}
    assert tick(moved, drifted, True) == [-45.0, -110.0, -30.0]
    assert tick(stale, drifted, False) == [-45.0, -110.0, -30.0]
    step.reset()
    assert tick(stale, drifted, False) == list(drifted.values())


def test_stale_leader_does_not_change_autonomous_policy_action():
    step = InterventionActionProcessorStep()
    transition = _transition({"joint.pos": 99.0}, {"joint.pos": 0.0}, False)
    transition[TransitionKey.ACTION] = torch.tensor([12.0])
    transition[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_IS_FRESH] = False
    assert step(transition)[TransitionKey.ACTION].tolist() == [12.0]


def test_teleop_action_processor_propagates_freshness_and_defaults_other_devices_to_fresh():
    from types import SimpleNamespace
    teleop = SimpleNamespace(get_action=lambda: {"joint.pos": 1.0}, action_is_fresh=False)
    result = AddTeleopActionAsComplimentaryDataStep(teleop).complementary_data({})
    assert result[TELEOP_ACTION_IS_FRESH] is False
    del teleop.action_is_fresh
    result = AddTeleopActionAsComplimentaryDataStep(teleop).complementary_data({})
    assert result[TELEOP_ACTION_IS_FRESH] is True
