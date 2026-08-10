from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.molmoact2.configuration_molmoact2 import (
    DiscreteActionAuxiliaryLossConfig,
)
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    _build_fast_auxiliary_token_tables,
    _cumulative_ordinal_ce_from_log_masses,
    _fast_action_auxiliary_components,
    _grouped_logsumexp,
)


def test_discrete_auxiliary_config_has_master_switch_and_validates_weights() -> None:
    disabled = DiscreteActionAuxiliaryLossConfig()
    assert not disabled.enabled

    with pytest.raises(ValueError, match="at least one positive weight"):
        DiscreteActionAuxiliaryLossConfig(enabled=True)
    with pytest.raises(ValueError, match="ordinal_weight must be non-negative"):
        DiscreteActionAuxiliaryLossConfig(ordinal_weight=-1.0)

    enabled = DiscreteActionAuxiliaryLossConfig(enabled=True, ordinal_weight=0.25)
    assert enabled.enabled
    assert enabled.ordinal_weight == 0.25


def test_grouped_logsumexp_matches_direct_computation_and_has_finite_gradients() -> None:
    logits = torch.tensor([[0.2, -0.4, 1.1, -2.0]], requires_grad=True)
    # Groups 0 and 1 are coefficient values; group 2 is the non-reaching drop bin.
    groups = torch.tensor([[0, 1, 0, 2]])

    actual = _grouped_logsumexp(logits, groups, num_groups=2)
    expected = torch.stack(
        (
            torch.logsumexp(logits[0, [0, 2]], dim=0),
            logits[0, 1],
        )
    )[None, :]
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3] == 0


def test_cumulative_ordinal_ce_matches_worked_example() -> None:
    # Support [-1, 0, 1, 2, 3], q(0)=5/7 and q(1)=2/7, target coefficient 0.
    probabilities = torch.tensor([[0.0, 5.0 / 7.0, 2.0 / 7.0, 0.0, 0.0]])
    log_masses = torch.where(
        probabilities > 0,
        probabilities.log(),
        torch.full_like(probabilities, -torch.inf),
    )

    loss = _cumulative_ordinal_ce_from_log_masses(log_masses, torch.tensor([1]))
    expected = -math.log(5.0 / 7.0) / 4.0
    torch.testing.assert_close(loss, torch.tensor([expected]))


def test_cumulative_ordinal_ce_is_finite_for_extreme_wrong_logits() -> None:
    logits = torch.tensor([[-1000.0, 1000.0, 0.0]], requires_grad=True)
    log_masses = _grouped_logsumexp(
        logits,
        torch.tensor([[0, 1, 2]]),
        num_groups=3,
    )

    loss = _cumulative_ordinal_ce_from_log_masses(log_masses, torch.tensor([0])).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() <= 1


def test_fast_auxiliary_path_and_shape_use_conditional_coefficient_means() -> None:
    # One ground-truth token represents two coefficients [0, 0] (T=2, D=1).
    # Candidate token values by offset:
    #   v0 -> [0, 0], length 2, probability 1/2 (ground truth)
    #   v1 -> [1],    length 1, probability 1/4 (cannot reach offset 1)
    #   v2 -> [-1,1], length 2, probability 1/4
    # At offset 0 the conditional mean is 0. At offset 1 it is 1/3, because v1
    # is conditioned away. Hence path MSE=1/18 and shape MSE=2/9 at scale 1.
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.25]])).requires_grad_()
    token_lengths = torch.tensor([2, 1, 2])
    # Real bins are [-1, 0, 1] -> [0, 1, 2]; 3 is the non-reaching sentinel.
    value_bins_by_offset = torch.tensor(
        [
            [1, 2, 0],
            [1, 3, 2],
        ]
    )

    components = _fast_action_auxiliary_components(
        logits,
        torch.tensor([0]),
        torch.tensor([0]),
        token_lengths=token_lengths,
        value_bins_by_offset=value_bins_by_offset,
        coefficient_values=torch.tensor([-1.0, 0.0, 1.0]),
        coefficient_min=-1,
        horizon=2,
        scale=1.0,
        batch_size=1,
        target_actions=torch.zeros(1, 2, 1),
        hold_actions=torch.tensor([[[1.0], [-1.0]]]),
    )

    torch.testing.assert_close(components["path_mse"], torch.tensor([1.0 / 18.0]))
    torch.testing.assert_close(components["shape_mse"], torch.tensor([2.0 / 9.0]))
    torch.testing.assert_close(components["path_relative"], torch.tensor([1.0 / 18.0]))
    torch.testing.assert_close(components["shape_relative"], torch.tensor([1.0 / 18.0]))
    assert torch.isfinite(components["ordinal_ce"]).all()

    sum(value.sum() for value in components.values()).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_fast_auxiliary_table_builder_aligns_lm_and_bpe_ids() -> None:
    class FakeBPE:
        vocab_size = 3

        @staticmethod
        def decode(token_ids: list[int]) -> str:
            pieces = {
                0: chr(1),
                1: chr(2) + chr(1),
                2: chr(0),
            }
            return "".join(pieces[token_id] for token_id in token_ids)

    tokenizer = SimpleNamespace(bpe_tokenizer=FakeBPE(), min_token=-1, scale=10)
    tables = _build_fast_auxiliary_token_tables(
        tokenizer,
        {103: 2, 101: 0, 102: 1},
        max_slots=2,
    )

    assert tables["action_lm_ids"].tolist() == [101, 102, 103]
    assert tables["token_lengths"].tolist() == [1, 2, 1]
    assert tables["coefficient_min"] == -1
    assert tables["coefficient_values"].tolist() == [-1.0, 0.0, 1.0]
    # BPE id 0 says value 0 at offset 0 and does not reach offset 1.
    assert tables["value_bins_by_offset"][:, 0].tolist() == [1, 3]
