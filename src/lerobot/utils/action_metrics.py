"""Shared action-trajectory metrics used by probes and training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


ACTION_HOLD_KEY = "action_hold"
TRAJECTORY_ERROR_KEYS = (
    "path_mse",
    "shape_mse",
    "terminal_mse",
    "terminal_direction_loss",
)


def terminal_direction_loss(
    prediction_final: torch.Tensor,
    target_final: torch.Tensor,
    hold_final: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return ``1 - cosine_similarity`` for final displacement from ``hold_final``.

    The result is 0 for aligned displacements, 1 for perpendicular displacement
    (or a zero prediction), and 2 for opposite displacement. It is undefined
    (NaN) when the target has no displacement from the hold position.
    """
    prediction_displacement = prediction_final - hold_final
    target_displacement = target_final - hold_final
    direction_loss = 1.0 - F.cosine_similarity(
        prediction_displacement,
        target_displacement,
        dim=-1,
        eps=eps,
    )
    target_norm = torch.linalg.vector_norm(target_displacement, dim=-1)
    return torch.where(
        target_norm <= eps,
        torch.full_like(direction_loss, torch.nan),
        direction_loss.clamp(0.0, 2.0),
    )


def trajectory_error_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hold: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return complementary errors between action trajectories.

    All returned metrics are lower-is-better and operate in the tensors' action
    units. The direction term is dimensionless.
    """
    if prediction.shape != target.shape or prediction.shape != hold.shape:
        raise ValueError(
            "prediction, target, and hold must have identical shapes; got "
            f"{prediction.shape}, {target.shape}, and {hold.shape}"
        )
    if prediction.ndim < 2:
        raise ValueError(f"expected tensors ending in [time, action], got {prediction.shape}")

    squared_error = (prediction - target).square()
    path_mse = squared_error.mean(dim=(-2, -1))
    terminal_mse = squared_error[..., -1, :].mean(dim=-1)

    if prediction.shape[-2] > 1:
        prediction_delta = torch.diff(prediction, dim=-2)
        target_delta = torch.diff(target, dim=-2)
        shape_mse = (prediction_delta - target_delta).square().mean(dim=(-2, -1))
    else:
        shape_mse = torch.zeros_like(path_mse)

    return {
        "path_mse": path_mse,
        "shape_mse": shape_mse,
        "terminal_mse": terminal_mse,
        "terminal_direction_loss": terminal_direction_loss(
            prediction[..., -1, :],
            target[..., -1, :],
            hold[..., -1, :],
            eps=eps,
        ),
    }
