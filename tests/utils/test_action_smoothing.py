from types import SimpleNamespace

import torch

from lerobot.utils.action_smoothing import bound_action_chunk


def test_no_limits_is_identity():
    actions = torch.randn(30, 3) * 100
    assert torch.equal(bound_action_chunk(actions, torch.zeros(3)), actions)


def test_excursion_bound_is_relative_to_anchor():
    anchor = torch.tensor([10.0, -20.0])
    actions = anchor + torch.tensor([[100.0, -100.0]] * 5)
    out = bound_action_chunk(actions, anchor, delta_limits=[5.0, 7.0])
    assert torch.allclose(out - anchor, torch.tensor([[5.0, -7.0]] * 5))


def test_absolute_clamp():
    actions = torch.tensor([[-500.0, 500.0]] * 3)
    out = bound_action_chunk(actions, torch.zeros(2), clamp_limits=[[-10.0, 10.0], [-1.0, 1.0]])
    assert torch.equal(out, torch.tensor([[-10.0, 1.0]] * 3))


def test_rate_limit_seeds_from_anchor_and_tracks():
    anchor = torch.tensor([0.0])
    out = bound_action_chunk(torch.full((10, 1), 7.0), anchor, step_limits=[2.0])
    assert out[:, 0].tolist() == [2.0, 4.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]


def test_anchor_outside_workspace_walks_to_edge():
    anchor = torch.tensor([20.0])
    out = bound_action_chunk(
        torch.full((6, 1), 20.0), anchor, clamp_limits=[[-10.0, 10.0]], step_limits=[3.0]
    )
    assert out[:, 0].tolist() == [17.0, 14.0, 11.0, 10.0, 10.0, 10.0]


def test_all_three_bounds_hold_together():
    torch.manual_seed(0)
    anchor = torch.zeros(4)
    actions = torch.randn(30, 4) * 200
    out = bound_action_chunk(
        actions, anchor,
        delta_limits=[50.0] * 4, clamp_limits=[[-20.0, 30.0]] * 4, step_limits=[1.5] * 4,
    )
    steps = torch.diff(out, dim=0, prepend=anchor[None])
    assert steps.abs().max() <= 1.5 + 1e-6
    assert (out >= -20.0).all() and (out <= 30.0).all()
    assert (out - anchor).abs().max() <= 50.0 + 1e-6


def test_zero_limit_pins_padding_column():
    anchor = torch.tensor([5.0, 0.0])
    out = bound_action_chunk(torch.tensor([[9.0, 123.0]] * 3), anchor, delta_limits=[100.0, 0.0])
    assert out[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert out[:, 0].tolist() == [9.0, 9.0, 9.0]


def test_bound_policy_actions_passthrough_and_anchor_from_state():
    from lerobot.rl.inference_utils import bound_policy_actions

    actions = torch.tensor([[100.0, 0.0]] * 12)
    obs = {"observation.state": torch.tensor([[10.0, 0.0]])}
    assert torch.equal(bound_policy_actions(actions, obs, SimpleNamespace(config=SimpleNamespace())), actions)
    policy = SimpleNamespace(config=SimpleNamespace(action_delta_limits=[5.0, 5.0]))
    assert bound_policy_actions(actions, obs, policy)[:, 0].tolist() == [15.0] * 12
