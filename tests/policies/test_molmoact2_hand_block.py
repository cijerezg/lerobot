"""Hand block (docs/ee_mixture_loss/TODO.md Phase 2): pose, deltas, state, masks, and the
unnormalize-then-FK path, on the frozen kinematics assets."""

import torch

from lerobot.model.fk import get_chain
from lerobot.policies.molmoact2.action_layout import JOINT_SLOTS, native_joint_widths
from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY, AnchorEncodeStep
from lerobot.policies.molmoact2.hand_block import (
    HAND_ACTION_SLOTS,
    HAND_ACTION_WIDTH,
    HAND_STATE_SLOTS,
    HAND_STATE_WIDTH,
    MolmoAct2HandBlockProcessorStep,
    hand_delta,
    hand_pose,
    hand_pose_from_normalized_joints,
    hand_state,
)
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2UnifiedLayoutProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE

REBOT, DROID, FMB, MOLMOACT = 6, 0, 2, 7


def _rebot_state():
    return torch.tensor([[10.0, -60.0, -90.0, 30.0, 5.0, -20.0, -135.0]])


def _panda_state():
    return torch.tensor([[0.02, 0.07, 0.17, -2.34, 0.01, 2.41, -0.60, 0.0]])


def test_hand_pose_matches_the_chains_and_marks_rows_without_a_chain():
    rebot = torch.cat([_rebot_state(), torch.zeros(1, 1)], dim=-1)  # padded to 8
    vectors = torch.cat([rebot, _panda_state(), _panda_state()], dim=0)
    p, r, g, valid = hand_pose(torch.tensor([REBOT, DROID, MOLMOACT]), vectors)
    assert valid.tolist() == [True, True, False]
    pr, rr, gr = get_chain("rebot_b601_follower").fk(_rebot_state().double())
    assert torch.allclose(p[0].double(), pr[0]) and torch.allclose(r[0].double(), rr[0])
    assert torch.allclose(g[0].double(), gr[0])
    assert torch.equal(p[2], torch.zeros(3)) and torch.equal(r[2], torch.eye(3))


def test_hand_delta_is_zero_at_the_anchor_and_inverts_a_known_rotation():
    p, r, g, _ = hand_pose(torch.tensor([REBOT]), _rebot_state())
    assert torch.allclose(hand_delta(p, r, g, p, r, g), torch.zeros(1, 7))
    # rotate the hand by 0.3 rad about base z: the rotation vector delta is (0, 0, 0.3)
    rz = torch.tensor(
        [
            [
                [torch.cos(torch.tensor(0.3)), -torch.sin(torch.tensor(0.3)), 0.0],
                [torch.sin(torch.tensor(0.3)), torch.cos(torch.tensor(0.3)), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    d = hand_delta(p, r, g, p + 0.1, rz @ r, g + 0.01)
    assert torch.allclose(d, torch.tensor([[0.1, 0.1, 0.1, 0.0, 0.0, 0.3, 0.01]]), atol=1e-6)
    assert hand_state(p, r, g).shape == (1, 10)
    assert torch.allclose(hand_state(p, r, g)[:, 3:6], r[:, :, 0])


def test_processor_step_fills_hand_slots_and_masks_after_anchor_encoding():
    state = torch.cat([_rebot_state(), _panda_state()[:, :7]], dim=0)  # both 7 wide here
    state[1] = _panda_state()[0, :7]
    chunk = torch.stack([state + 0.0, state + 2.0, state + 4.0], dim=1)  # (2, 3, 7)
    layout = MolmoAct2UnifiedLayoutProcessorStep(
        state_dim=HAND_STATE_WIDTH,
        action_dim=HAND_ACTION_WIDTH,
        native_action_dims=[8, 8, 8, 7, 7, 7, 7, 7, 7],
        stats_index_key="action_layout_id",
    )
    transition = create_transition(
        observation={OBS_STATE: state},
        action=chunk,
        complementary_data={"action_layout_id": torch.tensor([REBOT, MOLMOACT])},
    )
    out = MolmoAct2HandBlockProcessorStep()(AnchorEncodeStep("anchor")(layout(transition)))
    action = out[TransitionKey.ACTION]
    obs_state = out[TransitionKey.OBSERVATION][OBS_STATE]
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    assert action.shape == (2, 3, HAND_ACTION_WIDTH) and obs_state.shape == (2, HAND_STATE_WIDTH)
    # joint block untouched: anchor-encoded as before
    assert torch.allclose(action[:, :, :7], torch.tensor([0.0, 2.0, 4.0])[None, :, None].expand(2, 3, 7))
    # ReBot row: hand delta zero at step 0 (the chunk's first step equals the anchor), hand state = fk
    assert torch.allclose(action[0, 0, HAND_ACTION_SLOTS], torch.zeros(7), atol=1e-6)
    p, r, g, _ = hand_pose(torch.tensor([REBOT]), _rebot_state())
    assert torch.allclose(obs_state[0, HAND_STATE_SLOTS], hand_state(p, r, g)[0], atol=1e-6)
    assert comp["action_dim_is_pad"][0].tolist() == [False] * 7 + [True] + [False] * 7
    assert comp["state_dim_is_pad"][0].tolist() == [False] * 7 + [True] + [False] * 10
    # MolmoAct row: no chain, hand block is padding and zero
    assert comp["action_dim_is_pad"][1].tolist() == [False] * 7 + [True] * 8
    assert torch.equal(action[1, :, HAND_ACTION_SLOTS], torch.zeros(3, 7))
    assert native_joint_widths(comp["action_dim_is_pad"]).tolist() == [7, 7]
    assert comp[ANCHOR_KEY].shape[-1] == HAND_ACTION_WIDTH


def test_unnormalize_then_fk_reproduces_the_hand_targets_on_demo_joints():
    anchor = torch.cat([_rebot_state(), torch.zeros(1, 1)], dim=-1).repeat(2, 1)
    absolute = anchor[:, None, :] + torch.tensor([0.0, 3.0, 6.0])[None, :, None]
    delta = absolute - anchor[:, None, :]
    q01 = torch.full((2, 3, JOINT_SLOTS), -10.0)
    q99 = torch.full((2, 3, JOINT_SLOTS), 10.0)
    normalized = (2.0 * (delta - q01) / (q99 - q01) - 1.0).requires_grad_(True)
    ids = torch.tensor([REBOT, REBOT])
    p, r, g, valid = hand_pose_from_normalized_joints(normalized, q01, q99, anchor, ids)
    pt, rt, gt, _ = hand_pose(ids, absolute)
    assert valid.all() and torch.allclose(p, pt, atol=1e-5) and torch.allclose(r, rt, atol=1e-5)
    assert torch.allclose(g, gt, atol=1e-6)
    p.sum().backward()
    assert normalized.grad is not None and torch.isfinite(normalized.grad).all()
    assert normalized.grad[..., :6].abs().sum() > 0


def test_native_joint_widths_ignores_hand_slots():
    mask = torch.tensor([[False] * 7 + [True] + [False] * 7, [False] * 8 + [True] * 7])
    assert native_joint_widths(mask).tolist() == [7, 8]
    assert native_joint_widths(torch.tensor([[False] * 6 + [True, True]])).tolist() == [6]
