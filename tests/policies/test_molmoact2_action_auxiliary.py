from types import SimpleNamespace

import torch

from lerobot.policies.molmoact2.modeling_molmoact2 import (
    _action_trajectory_components_with_padding,
    _thresholded_action_auxiliary_loss,
)
from lerobot.utils.action_metrics import terminal_direction_loss, trajectory_error_components


def test_training_components_are_the_probe_metrics_without_padding():
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    hold = torch.zeros_like(target)
    prediction = torch.stack(
        [
            target + 0.25,
            torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]]),
        ],
        dim=1,
    )

    actual = _action_trajectory_components_with_padding(prediction, target, hold)
    per_flow = [
        trajectory_error_components(prediction[:, flow], target, hold)
        for flow in range(prediction.shape[1])
    ]
    for key in actual:
        expected = torch.stack([item[key] for item in per_flow]).mean(dim=0)
        torch.testing.assert_close(actual[key], expected)


def test_training_components_ignore_padded_action_dimensions():
    target = torch.tensor([[[0.0, 0.0, 50.0], [1.0, 0.0, 50.0]]])
    hold = torch.zeros_like(target)
    prediction = target.clone().unsqueeze(1)
    prediction[..., 2] = -1000.0

    components = _action_trajectory_components_with_padding(
        prediction,
        target,
        hold,
        action_dim_is_pad=torch.tensor([[False, False, True]]),
    )

    torch.testing.assert_close(components["path_mse"], torch.zeros(1))
    torch.testing.assert_close(components["shape_mse"], torch.zeros(1))
    torch.testing.assert_close(components["terminal_mse"], torch.zeros(1))
    torch.testing.assert_close(components["path_relative"], torch.zeros(1))
    torch.testing.assert_close(components["shape_relative"], torch.zeros(1))
    torch.testing.assert_close(components["terminal_relative"], torch.zeros(1))
    torch.testing.assert_close(components["terminal_direction_loss"], torch.zeros(1))


def test_each_auxiliary_threshold_gates_only_its_own_metric():
    components = {
        "path_mse": torch.tensor([40.0, 60.0]),
        "shape_mse": torch.tensor([30.0, 30.0]),
        "terminal_mse": torch.tensor([40.0, 40.0]),
        "path_relative": torch.tensor([0.4, 0.6]),
        "shape_relative": torch.tensor([3.0, 3.0]),
        "terminal_relative": torch.tensor([4.0, 4.0]),
        "terminal_direction_loss": torch.tensor([0.2, 0.8]),
    }
    config = SimpleNamespace(
        path_weight=2.0,
        path_threshold=0.5,
        shape_weight=0.0,
        shape_threshold=0.1,
        terminal_weight=0.0,
        terminal_threshold=None,
        direction_weight=3.0,
        direction_threshold=0.5,
    )

    loss, active = _thresholded_action_auxiliary_loss(components, config)

    torch.testing.assert_close(loss, torch.tensor([0.0, 3.6]))
    assert active["path_relative_active"].tolist() == [False, True]
    assert active["terminal_direction_loss_active"].tolist() == [False, True]


def test_relative_components_are_invariant_to_joint_motion_scale():
    target = torch.tensor([[[0.0], [1.0], [3.0]]])
    hold = torch.zeros_like(target)
    prediction = (0.5 * target).unsqueeze(1)

    base = _action_trajectory_components_with_padding(prediction, target, hold)
    scaled = _action_trajectory_components_with_padding(
        4.0 * prediction,
        4.0 * target,
        4.0 * hold,
    )

    for key in ("path_relative", "shape_relative", "terminal_relative"):
        torch.testing.assert_close(base[key], scaled[key])
    for key in ("path_mse", "shape_mse", "terminal_mse"):
        torch.testing.assert_close(scaled[key], 16.0 * base[key])


def test_direction_loss_at_zero_prediction_has_a_gradient():
    prediction = torch.zeros(1, 2, requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    hold = torch.zeros_like(target)

    loss = terminal_direction_loss(prediction, target, hold)
    loss.backward()

    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
