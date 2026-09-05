import torch

from lerobot.processor.hil_processor import TELEOP_ACTION_KEY, InterventionActionProcessorStep
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
