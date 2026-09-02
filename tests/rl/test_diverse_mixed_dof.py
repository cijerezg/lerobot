"""Mixed 7D/8D widths through layout, encoding, loss and gradients (plan phase E).

Gate E: a mixed batch must give the Franka's eighth dimension a real target and a real
gradient, and every padded dimension exactly zero loss and zero gradient.

The full-model version of this runs in the phase J smoke test. These tests exercise the
same production code paths -- the layout step, the anchor encoder, and the policy's own
masking helpers -- on tensors small enough to check by hand.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="molmoact2 modules import policy deps")

from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY, AnchorEncodeStep  # noqa: E402
from lerobot.policies.molmoact2.modeling_molmoact2 import (  # noqa: E402
    MolmoAct2Policy,
    _action_band_auxiliary_loss,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2UnifiedLayoutProcessorStep,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

HORIZON = 4


def _mixed_batch():
    """Row 0 is an 8D Franka, row 1 a 7D arm padded to width 8."""
    state = torch.zeros(2, 8)
    state[0] = torch.arange(1, 9, dtype=torch.float32)
    state[1, :7] = torch.arange(1, 8, dtype=torch.float32)
    action = torch.zeros(2, HORIZON, 8)
    action[0] = torch.arange(1, 9, dtype=torch.float32) + 0.5
    action[1, :, :7] = torch.arange(1, 8, dtype=torch.float32) + 0.25
    is_pad = torch.zeros(2, 8, dtype=torch.bool)
    is_pad[1, 7] = True
    return state, action, is_pad


# ── Layout ───────────────────────────────────────────────────────────────────


def test_layout_pads_without_reordering_and_keeps_separate_masks() -> None:
    state, action, is_pad = _mixed_batch()
    step = MolmoAct2UnifiedLayoutProcessorStep(state_dim=8, action_dim=8)
    out = step(
        create_transition(
            observation={OBS_STATE: state[:, :8]},
            action=action,
            complementary_data={"action_dim_is_pad": is_pad, "state_dim_is_pad": is_pad},
        )
    )
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    # Source order survives: the gripper stays where its robot records it.
    assert torch.equal(out[TransitionKey.OBSERVATION][OBS_STATE][1, :7], state[1, :7])
    assert complementary["action_dim_is_pad"][1, 7]
    assert not complementary["action_dim_is_pad"][0].any()


def test_state_and_action_widths_are_tracked_independently() -> None:
    """Their native widths need not match, so one mask cannot stand for both."""
    state, action, action_is_pad = _mixed_batch()
    state_is_pad = torch.zeros(2, 8, dtype=torch.bool)  # both rows record 8 state values
    step = MolmoAct2UnifiedLayoutProcessorStep(state_dim=8, action_dim=8)
    out = step(
        create_transition(
            observation={OBS_STATE: state},
            action=action,
            complementary_data={
                "action_dim_is_pad": action_is_pad,
                "state_dim_is_pad": state_is_pad,
            },
        )
    )
    complementary = out[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["action_dim_is_pad"][1, 7]
    assert not complementary["state_dim_is_pad"][1, 7]


def test_anchor_encoding_leaves_padded_dimensions_at_zero() -> None:
    state, action, is_pad = _mixed_batch()
    out = AnchorEncodeStep(encoding="anchor")(
        create_transition(observation={OBS_STATE: state}, action=action)
    )
    encoded = out[TransitionKey.ACTION]
    assert torch.allclose(encoded[0], action[0] - state[0][None])
    assert encoded[1, :, 7].abs().max() == 0
    assert torch.equal(out[TransitionKey.COMPLEMENTARY_DATA][ANCHOR_KEY], state)


# ── Gate E: loss and gradient ────────────────────────────────────────────────


def _masked_flow_loss(prediction: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor):
    """The production reduction: zero the padded dims, then average over valid ones."""
    loss = torch.nn.functional.mse_loss(prediction, target, reduction="none")
    reduced = MolmoAct2Policy._apply_action_dim_padding_mask(loss, is_pad)
    return reduced.reshape(prediction.shape[0], -1).mean(dim=1)


def test_padded_dimensions_receive_exactly_zero_gradient() -> None:
    _, action, is_pad = _mixed_batch()
    prediction = torch.zeros(2, HORIZON, 8, requires_grad=True)
    _masked_flow_loss(prediction, action, is_pad).sum().backward()
    grad = prediction.grad
    # Row 1 is 7D: its eighth dimension must be untouched by the loss.
    assert grad[1, :, 7].abs().max() == 0
    # ... while its real dimensions are supervised.
    assert grad[1, :, :7].abs().min() > 0


def test_the_frankas_eighth_dimension_is_really_supervised() -> None:
    """The failure this gate exists for: an 8D robot silently trained as a 7D one."""
    _, action, is_pad = _mixed_batch()
    prediction = torch.zeros(2, HORIZON, 8, requires_grad=True)
    _masked_flow_loss(prediction, action, is_pad).sum().backward()
    assert prediction.grad[0, :, 7].abs().min() > 0


def test_a_seven_dimensional_row_scores_the_same_alone_or_beside_an_eight(
) -> None:
    """Averaging must use each sample's own valid count, not the batch width."""
    _, action, is_pad = _mixed_batch()
    prediction = torch.zeros(2, HORIZON, 8)
    together = _masked_flow_loss(prediction, action, is_pad)
    alone = _masked_flow_loss(prediction[1:], action[1:], is_pad[1:])
    assert torch.allclose(together[1], alone[0])


def test_padding_cannot_change_a_seven_dimensional_loss() -> None:
    _, action, is_pad = _mixed_batch()
    prediction = torch.zeros(2, HORIZON, 8)
    baseline = _masked_flow_loss(prediction, action, is_pad)
    noisy = action.clone()
    noisy[1, :, 7] = 1e6  # garbage in the padded slot
    assert torch.allclose(_masked_flow_loss(prediction, noisy, is_pad), baseline)


def test_the_band_auxiliary_also_excludes_padding() -> None:
    _, action, is_pad = _mixed_batch()

    class _Config:
        band_spec = "dc=0;k1=1;detail=2-3"
        band_powers = (1.0, 1.0, 1.0)
        gamma = 0.0
        weight = 1.0

    residual = torch.zeros(2, 1, HORIZON, 8, requires_grad=True)
    loss, _, _ = _action_band_auxiliary_loss(
        residual + action.unsqueeze(1), _Config(), action_dim_is_pad=is_pad
    )
    loss.sum().backward()
    assert residual.grad[1, :, :, 7].abs().max() == 0
    assert residual.grad[0, :, :, 7].abs().max() > 0


def test_flow_target_and_noise_are_zeroed_on_padded_dimensions() -> None:
    _, action, is_pad = _mixed_batch()
    masked = MolmoAct2Policy._mask_action_dim_tensor(action, is_pad)
    assert masked[1, :, 7].abs().max() == 0
    assert torch.equal(masked[0], action[0])


def test_a_mask_of_the_wrong_width_is_refused() -> None:
    _, action, _ = _mixed_batch()
    with pytest.raises(ValueError, match="does not match target width"):
        MolmoAct2Policy._action_dim_valid_mask(action, torch.zeros(2, 7, dtype=torch.bool))


# ── Per-layout statistics ────────────────────────────────────────────────────


def _two_row_normalizer(stats_index_key: str):
    """Two stats rows with deliberately different scales, so a mis-gather is visible."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        MolmoAct2MaskedNormalizerProcessorStep,
    )

    stats = {
        OBS_STATE: {
            "q01": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            "q99": torch.tensor([[1.0, 1.0], [10.0, 10.0]]),
            "mask": torch.ones(2, 2, dtype=torch.bool),
        }
    }
    return MolmoAct2MaskedNormalizerProcessorStep(
        features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats=stats,
        embodiment_names=["row_a", "row_b"],
        stats_index_key=stats_index_key,
    )


def test_stats_rows_are_gathered_by_action_layout_when_the_artifact_says_so() -> None:
    step = _two_row_normalizer("action_layout_id")
    out = step(
        create_transition(
            observation={OBS_STATE: torch.tensor([[1.0, 1.0], [1.0, 1.0]])},
            complementary_data={
                "action_layout_id": torch.tensor([0, 1]),
                # Same robot under two conventions: the embodiment column is identical,
                # so a gather that read it would give both rows the same scale.
                "embodiment_index": torch.tensor([0, 0]),
            },
        )
    )
    normalized = out[TransitionKey.OBSERVATION][OBS_STATE]
    assert not torch.allclose(normalized[0], normalized[1])


def test_the_default_key_is_still_the_embodiment_index() -> None:
    step = _two_row_normalizer("embodiment_index")
    out = step(
        create_transition(
            observation={OBS_STATE: torch.tensor([[1.0, 1.0], [1.0, 1.0]])},
            complementary_data={"embodiment_index": torch.tensor([0, 1])},
        )
    )
    normalized = out[TransitionKey.OBSERVATION][OBS_STATE]
    assert not torch.allclose(normalized[0], normalized[1])


def test_a_missing_index_column_is_an_error_not_a_default_row() -> None:
    step = _two_row_normalizer("action_layout_id")
    with pytest.raises(ValueError, match="action_layout_id"):
        step(
            create_transition(
                observation={OBS_STATE: torch.tensor([[1.0, 1.0]])},
                complementary_data={"embodiment_index": torch.tensor([0])},
            )
        )
