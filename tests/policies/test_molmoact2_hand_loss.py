"""EE mixture loss, Phase 3 (docs/ee_mixture_loss/TODO.md): block-mean flow loss, FK term,
config widening, the FK batch keys and the inference monitor. No checkpoint needed."""

import math

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.action_layout import JOINT_SLOTS
from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY, AnchorDecodeStep, AnchorEncodeStep
from lerobot.policies.molmoact2.configuration_molmoact2 import (
    HAND_BLOCK_ACTION_WIDTH,
    HAND_BLOCK_STATE_WIDTH,
    HandBlockConfig,
    MolmoAct2Config,
)
from lerobot.policies.molmoact2.hand_block import (
    HAND_ACTION_SLOTS,
    HAND_ACTION_WIDTH,
    HAND_STATE_WIDTH,
    MolmoAct2HandBlockProcessorStep,
    MolmoAct2HandMonitorProcessorStep,
    hand_delta,
    hand_monitor_gap,
    hand_pose,
    rotation_exp,
)
from lerobot.policies.molmoact2.hand_loss import (
    HAND_FK_KEYS,
    block_mean_flow_loss,
    hand_fk_term,
    implied_clean_sample,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (
    MolmoAct2RestoreActionLayoutProcessorStep,
    MolmoAct2UnifiedLayoutProcessorStep,
)
from lerobot.processor.converters import create_transition
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

REBOT, MOLMOACT = 6, 7
HORIZON = 3


def _rebot_state():
    return torch.tensor([[10.0, -60.0, -90.0, 30.0, 5.0, -20.0, -135.0]])


def _mask(hand: bool, joints: int = 7) -> list[bool]:
    return [False] * joints + [True] * (JOINT_SLOTS - joints) + [not hand] * 7


# ── block means ──────────────────────────────────────────────────────────────


def test_block_mean_flow_loss_weighs_blocks_as_configured():
    loss = torch.zeros(2, 1, 1, HAND_ACTION_WIDTH)
    loss[:, :, :, :7] = 2.0  # joints
    loss[:, :, :, 7] = 99.0  # the hole at slot 7: never counted
    loss[:, :, :, 8:11] = 3.0  # position
    loss[:, :, :, 11:14] = 6.0  # rotation
    loss[:, :, :, 14] = 9.0  # aperture
    pad = torch.tensor([_mask(hand=True), _mask(hand=False)])
    reduced, means = block_mean_flow_loss(loss, pad, joint_weight=0.1)
    # hand row: 0.1 * joint mean + mean of the three hand block means
    assert reduced[0].item() == pytest.approx(0.1 * 2.0 + (3.0 + 6.0 + 9.0) / 3, rel=1e-5)
    # row without a hand block: its joint block keeps weight 1 and nothing else
    assert reduced[1].item() == pytest.approx(2.0, rel=1e-5)
    assert means["joint"].item() == pytest.approx(2.0)
    assert means["position"].item() == pytest.approx(3.0)
    assert means["rotation"].item() == pytest.approx(6.0)
    assert means["aperture"].item() == pytest.approx(9.0)


def test_block_mean_flow_loss_gives_padding_zero_gradient():
    prediction = torch.zeros(1, 2, HORIZON, HAND_ACTION_WIDTH, requires_grad=True)
    target = torch.ones_like(prediction)
    loss = (prediction - target) ** 2
    reduced, _ = block_mean_flow_loss(loss, torch.tensor([_mask(hand=True)]), joint_weight=1.0)
    reduced.mean().backward()
    assert prediction.grad[..., 7].abs().max() == 0
    assert prediction.grad[..., :7].abs().min() > 0
    assert prediction.grad[..., 8:].abs().min() > 0


# ── the FK term ──────────────────────────────────────────────────────────────


def _fk_batch(normalized_delta: torch.Tensor):
    """A two-row batch: a ReBot row whose demo joints are anchor + 3 deg per step, and a
    MolmoAct row with no chain. Stats: q01/q99 = -/+ 10 so normalized = delta / 10."""
    anchor = torch.cat([_rebot_state(), torch.zeros(1, 1)], dim=-1).repeat(2, 1)
    q01 = torch.full((2, HORIZON, JOINT_SLOTS), -10.0)
    q99 = torch.full((2, HORIZON, JOINT_SLOTS), 10.0)
    absolute = anchor[:, None, :] + normalized_delta * 10.0
    pt, rt, gt, _ = hand_pose(torch.tensor([REBOT, REBOT]), absolute)
    return {
        "hand_fk_valid": torch.tensor([True, False]),
        "hand_fk_layout_id": torch.tensor([REBOT, MOLMOACT]),
        "hand_fk_anchor": anchor,
        "hand_fk_q01": q01,
        "hand_fk_q99": q99,
        "hand_fk_target_p": pt,
        "hand_fk_target_r": rt,
        "hand_fk_target_g": gt,
        "hand_fk_block_scale": torch.tensor([[0.05, 0.2, 0.01]]).repeat(HORIZON, 1),
    }


def test_fk_term_is_zero_at_the_demo_joints_and_grows_away_from_them():
    demo = torch.zeros(2, 1, HORIZON, JOINT_SLOTS)
    demo[..., :6] = torch.tensor([0.0, 0.3, 0.6])[None, None, :, None]  # 0, 3, 6 deg per joint
    batch = _fk_batch(demo[:, 0])
    at_target, means = hand_fk_term(demo, batch)
    assert torch.allclose(at_target, torch.zeros(2), atol=1e-8)
    assert set(means) == {"fk_position", "fk_rotation", "fk_aperture", "fk"}
    off = demo.clone()
    off[..., 0] += 0.5  # 5 deg on the shoulder pan: a real hand displacement
    away, _ = hand_fk_term(off, batch)
    assert away[0] > 1e-3
    assert away[1] == 0  # no chain, no term


def test_fk_term_backpropagates_into_the_joint_block_only_of_rows_with_a_chain():
    demo = torch.zeros(2, 2, HORIZON, HAND_ACTION_WIDTH)
    x_hat = (demo + 0.2).requires_grad_(True)
    batch = _fk_batch(torch.zeros(2, HORIZON, JOINT_SLOTS))
    value, _ = hand_fk_term(x_hat, batch, action_horizon_is_pad=torch.tensor([[False, False, True]] * 2))
    value.sum().backward()
    grad = x_hat.grad
    assert torch.isfinite(grad).all()
    assert grad[0, :, :2, :6].abs().sum() > 0  # valid steps, real joints
    assert grad[0, :, 2].abs().sum() == 0  # padded step
    assert grad[0, ..., 7:].abs().sum() == 0  # slot 7 and the hand slots: not the FK term's
    assert grad[1].abs().sum() == 0  # the MolmoAct row


def test_implied_clean_sample_recovers_the_target_from_the_true_velocity():
    noise = torch.randn(2, 3, HORIZON, 8)
    target = torch.randn(2, 3, HORIZON, 8)
    t = torch.rand(2, 3)
    xt = (1 - t[..., None, None]) * noise + t[..., None, None] * target
    assert torch.allclose(implied_clean_sample(xt, target - noise, t), target, atol=1e-6)


# ── config ───────────────────────────────────────────────────────────────────


def _config(**kwargs):
    return MolmoAct2Config(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        **kwargs,
    )


def test_hand_block_widens_the_features_and_remembers_the_joint_widths():
    cfg = _config(hand_block=True)
    assert cfg.output_features[ACTION].shape == (HAND_BLOCK_ACTION_WIDTH,)
    assert cfg.input_features[OBS_STATE].shape == (HAND_BLOCK_STATE_WIDTH,)
    assert (cfg.hand_joint_action_dim, cfg.hand_joint_state_dim) == (8, 8)
    # idempotent: a saved config reloads already widened
    again = MolmoAct2Config(
        input_features=dict(cfg.input_features),
        output_features=dict(cfg.output_features),
        hand_block=True,
        hand_joint_action_dim=7,
        hand_joint_state_dim=7,
    )
    assert again.output_features[ACTION].shape == (HAND_BLOCK_ACTION_WIDTH,)
    assert again.hand_joint_action_dim == 7
    off = _config()
    assert off.output_features[ACTION].shape == (8,) and off.hand_joint_action_dim is None


def test_hand_settings_are_refused_without_the_flag_and_validated_with_it():
    with pytest.raises(ValueError, match="hand_block is on"):
        _config(hand=HandBlockConfig(joint_weight=0.1))
    with pytest.raises(ValueError, match="fast_layout"):
        HandBlockConfig(fast_layout="hand_last")
    with pytest.raises(ValueError, match="joint_weight"):
        HandBlockConfig(joint_weight=-1.0)
    cfg = _config(hand_block=True, hand=HandBlockConfig(joint_weight=0.1, fast_layout="hand_first"))
    assert cfg.hand.joint_weight == 0.1


# ── processor: FK keys and the hand_state switch ─────────────────────────────


def _layout_step():
    return MolmoAct2UnifiedLayoutProcessorStep(
        state_dim=HAND_STATE_WIDTH,
        action_dim=HAND_ACTION_WIDTH,
        native_action_dims=[8, 8, 8, 7, 7, 7, 7, 7, 7],
        stats_index_key="action_layout_id",
    )


def _two_row_transition():
    state = torch.cat([_rebot_state(), _rebot_state() + 1.0], dim=0)
    chunk = torch.stack([state + 0.0, state + 2.0, state + 4.0], dim=1)
    return create_transition(
        observation={OBS_STATE: state},
        action=chunk,
        complementary_data={"action_layout_id": torch.tensor([REBOT, MOLMOACT])},
    )


def test_hand_step_ships_the_fk_keys_when_it_holds_the_stats():
    q01 = torch.full((9, HORIZON, JOINT_SLOTS), -10.0)
    q99 = torch.full((9, HORIZON, JOINT_SLOTS), 10.0)
    scale = torch.tensor([[0.05, 0.2, 0.01]]).repeat(HORIZON, 1)
    step = MolmoAct2HandBlockProcessorStep(joint_q01=q01, joint_q99=q99, block_scale=scale)
    out = step(AnchorEncodeStep("anchor")(_layout_step()(_two_row_transition())))
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    assert all(key in comp for key in HAND_FK_KEYS)
    assert comp["hand_fk_valid"].tolist() == [True, False]
    assert comp["hand_fk_q01"].shape == (2, HORIZON, JOINT_SLOTS)
    assert comp["hand_fk_target_r"].shape == (2, HORIZON, 3, 3)
    assert torch.equal(comp["hand_fk_anchor"][0, :7], _rebot_state()[0])
    # the FK term reproduces the step's own hand targets: zero at the demo joints
    normalized = (out[TransitionKey.ACTION][..., :JOINT_SLOTS] + 10.0) / 20.0 * 2.0 - 1.0
    value, _ = hand_fk_term(normalized[:, None], comp)
    assert torch.allclose(value, torch.zeros(2), atol=1e-8)
    # the stats ride state_dict, not the json config
    assert set(step.state_dict()) == {"joint_q01", "joint_q99", "block_scale"}
    assert "joint_q01" not in step.get_config()
    fresh = MolmoAct2HandBlockProcessorStep()
    assert not fresh.emits_fk_keys
    fresh.load_state_dict(step.state_dict())
    assert fresh.emits_fk_keys


def test_hand_step_without_stats_ships_no_fk_keys():
    out = MolmoAct2HandBlockProcessorStep()(AnchorEncodeStep("anchor")(_layout_step()(_two_row_transition())))
    assert not any(key in out[TransitionKey.COMPLEMENTARY_DATA] for key in HAND_FK_KEYS)


def test_hand_state_off_leaves_the_state_hand_slots_as_padding():
    out = MolmoAct2HandBlockProcessorStep(hand_state=False)(
        AnchorEncodeStep("anchor")(_layout_step()(_two_row_transition()))
    )
    state = out[TransitionKey.OBSERVATION][OBS_STATE]
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    assert torch.equal(state[:, JOINT_SLOTS:], torch.zeros(2, 10))
    assert comp["state_dim_is_pad"][0].tolist() == [False] * 7 + [True] * 11
    # the action hand block is still there: only the state side is the ablation
    assert comp["action_dim_is_pad"][0].tolist() == [False] * 7 + [True] + [False] * 7
    assert out[TransitionKey.ACTION][0, 1, HAND_ACTION_SLOTS].abs().sum() > 0


# ── inference: the monitor and the joint path ────────────────────────────────


def test_rotation_exp_inverts_the_log_map():
    from lerobot.model.fk.chain import rotation_log

    v = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [1.0, 0.5, -0.25]], dtype=torch.float64)
    assert torch.allclose(rotation_log(rotation_exp(v)), v, atol=1e-10)


def test_monitor_gap_is_zero_when_the_hand_block_matches_the_joints():
    anchor = torch.cat([_rebot_state(), torch.zeros(1, 1)], dim=-1)
    delta = torch.zeros(1, HORIZON, HAND_ACTION_WIDTH)
    delta[..., :6] = torch.tensor([0.0, 3.0, 6.0])[None, :, None]
    absolute = anchor[:, None, :] + delta[..., :JOINT_SLOTS]
    p0, r0, g0, _ = hand_pose(torch.tensor([REBOT]), anchor)
    pt, rt, gt, _ = hand_pose(torch.tensor([REBOT]), absolute)
    delta[..., HAND_ACTION_SLOTS] = hand_delta(p0[:, None], r0[:, None], g0[:, None], pt, rt, gt)
    gap = hand_monitor_gap(torch.tensor([REBOT]), anchor, delta)
    assert gap["valid"].tolist() == [True]
    assert gap["position_mm"].item() == pytest.approx(0.0, abs=1e-6)
    assert gap["rotation_deg"].item() == pytest.approx(0.0, abs=1e-6)
    assert gap["aperture_mm"].item() == pytest.approx(0.0, abs=1e-6)
    # 1 cm off in x, 0.1 rad about z, 5 mm of aperture
    delta[..., 8] += 0.01
    delta[..., 13] += 0.1
    delta[..., 14] += 0.005
    gap = hand_monitor_gap(torch.tensor([REBOT]), anchor, delta)
    assert gap["position_mm"].item() == pytest.approx(10.0, abs=1e-4)
    # rotation vectors do not add exactly (steps 1 and 2 already rotate), hence the tolerance
    assert gap["rotation_deg"].item() == pytest.approx(math.degrees(0.1), abs=0.01)
    assert gap["aperture_mm"].item() == pytest.approx(5.0, abs=1e-4)
    nan = hand_monitor_gap(torch.tensor([MOLMOACT]), anchor, delta)
    assert not nan["valid"].item() and math.isnan(nan["position_mm"].item())


def test_monitor_step_leaves_the_executed_joints_as_the_joint_only_path():
    """With the hand block on, the output pipeline is monitor -> anchor decode -> restore; the
    joints that come out equal what the joint-only pipeline (the flag off) decodes."""
    anchor = torch.cat([_rebot_state(), torch.zeros(1, 1)], dim=-1)
    chunk = torch.randn(1, HORIZON, HAND_ACTION_WIDTH)
    comp = {ANCHOR_KEY: anchor.clone(), "action_layout_id": torch.tensor([REBOT])}
    monitor = MolmoAct2HandMonitorProcessorStep(log_every=0)
    with_hand = MolmoAct2RestoreActionLayoutProcessorStep(native_action_dim=7)(
        AnchorDecodeStep("anchor")(monitor(create_transition(action=chunk, complementary_data=comp)))
    )
    joints_only = MolmoAct2RestoreActionLayoutProcessorStep(native_action_dim=7)(
        AnchorDecodeStep("anchor")(
            create_transition(action=chunk[..., :JOINT_SLOTS], complementary_data={ANCHOR_KEY: anchor[:, :8]})
        )
    )
    assert torch.equal(with_hand[TransitionKey.ACTION], joints_only[TransitionKey.ACTION])
    assert with_hand[TransitionKey.ACTION].shape == (1, HORIZON, 7)
    assert monitor.last is not None and monitor.last["valid"].tolist() == [True]
    assert with_hand[TransitionKey.COMPLEMENTARY_DATA]["hand_monitor"] is monitor.last
