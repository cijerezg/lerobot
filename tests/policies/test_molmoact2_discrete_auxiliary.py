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

    with pytest.raises(ValueError, match="positive ordinal_weight"):
        DiscreteActionAuxiliaryLossConfig(enabled=True)
    with pytest.raises(ValueError, match="ordinal_weight must be non-negative"):
        DiscreteActionAuxiliaryLossConfig(ordinal_weight=-1.0)

    enabled = DiscreteActionAuxiliaryLossConfig(enabled=True, ordinal_weight=0.25)
    assert enabled.enabled
    assert enabled.ordinal_weight == 0.25
    assert enabled.path_weight == enabled.shape_weight == 0

    with pytest.raises(ValueError, match="non-empty band_spec"):
        DiscreteActionAuxiliaryLossConfig(enabled=True, ordinal_weight=0.25, band_spec="")

    legacy = DiscreteActionAuxiliaryLossConfig(enabled=True, path_weight=0.5)
    assert not legacy.enabled


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


def test_fast_auxiliary_balances_bands_and_excludes_untrusted_tail() -> None:
    # The ground-truth token represents [0, 0, 0, 0] (T=4, D=1). The other
    # candidate is wrong at k=0,1,3 but correct at k=2. Its k=3 error must not
    # affect the loss because the band spec stops at k=2.
    logits = torch.log(torch.tensor([[0.75, 0.25]])).requires_grad_()
    token_lengths = torch.tensor([4, 4])
    # Real bins are [0, 1]; 2 is the non-reaching sentinel.
    value_bins_by_offset = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 1],
        ]
    )

    components = _fast_action_auxiliary_components(
        logits,
        torch.tensor([0]),
        torch.tensor([0]),
        token_lengths=token_lengths,
        value_bins_by_offset=value_bins_by_offset,
        num_bins=2,
        horizon=4,
        batch_size=1,
        band_spec="dc=0;wide=1-2",
    )

    wrong = torch.tensor([-math.log(0.75)])
    torch.testing.assert_close(components["band_dc_ordinal_ce"], wrong)
    torch.testing.assert_close(components["band_wide_ordinal_ce"], wrong / 2)
    torch.testing.assert_close(components["ordinal_ce"], 3 * wrong / 4)
    assert set(components) == {"ordinal_ce", "band_dc_ordinal_ce", "band_wide_ordinal_ce"}

    components["ordinal_ce"].sum().backward()
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
    assert tables["num_bins"] == 3
    # BPE id 0 says value 0 at offset 0 and does not reach offset 1.
    assert tables["value_bins_by_offset"][:, 0].tolist() == [1, 3]
