from __future__ import annotations

import torch
from torch.nn import functional

from lerobot.policies.molmoact2.configuration_molmoact2 import DepthGripperEventLossConfig
from lerobot.policies.molmoact2.modeling_molmoact2 import _depth_gripper_event_loss


def test_masked_loss_is_valid_only_mean_with_configured_weight() -> None:
    logits = torch.tensor([[0.0, 1.0], [20.0, -20.0], [-1.0, 0.5]], requires_grad=True)
    close = torch.tensor([0.0, 1.0, 0.25])
    open_ = torch.tensor([1.0, 0.0, 0.75])
    valid = torch.tensor([True, False, True])

    loss_by_example, head_bce = _depth_gripper_event_loss(
        logits, close, open_, valid, weight=0.2, reduction="none"
    )

    targets = torch.stack([close, open_], dim=-1)
    raw = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    expected_heads = raw[valid].mean(dim=0)
    torch.testing.assert_close(head_bce, expected_heads)
    torch.testing.assert_close(loss_by_example.mean(), 0.2 * expected_heads.mean())
    assert loss_by_example[1] == 0


def test_all_masked_loss_stays_graph_connected_and_zero() -> None:
    logits = torch.randn(3, 2, requires_grad=True)
    loss, head_bce = _depth_gripper_event_loss(
        logits,
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.bool),
        weight=0.2,
        reduction="mean",
    )

    assert loss.item() == 0
    assert head_bce.tolist() == [0.0, 0.0]
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_two_independent_logits_represent_no_nearby_event() -> None:
    targets = torch.zeros(1)
    valid = torch.ones(1, dtype=torch.bool)
    neutral, _ = _depth_gripper_event_loss(
        torch.zeros(1, 2), targets, targets, valid, weight=0.2, reduction="mean"
    )
    neither, _ = _depth_gripper_event_loss(
        torch.full((1, 2), -8.0), targets, targets, valid, weight=0.2, reduction="mean"
    )

    assert neither < neutral


def test_linear_head_is_default_and_mlp_width_is_validated() -> None:
    assert DepthGripperEventLossConfig(enabled=True, weight=0.2).hidden_dim is None
    try:
        DepthGripperEventLossConfig(enabled=True, weight=0.2, hidden_dim=0)
    except ValueError as exc:
        assert "hidden_dim" in str(exc)
    else:
        raise AssertionError("hidden_dim=0 should be rejected")
