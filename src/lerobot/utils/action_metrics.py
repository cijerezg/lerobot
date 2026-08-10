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
TRAJECTORY_RELATIVE_KEYS = (
    "path_relative",
    "shape_relative",
    "terminal_relative",
)

# Lower bound on the hold predictor's error, per term, in the same squared action units.
#
# The relative terms are the raw errors under a per-chunk weight $w = 1/d$, and $d$ is a
# function of the data alone, so the loss they define is the raw loss reweighted. That
# makes this floor a weight ceiling: without one, the smallest path denominator in the
# rebot corpus (1.97e-6) would carry 12000x the gradient weight of a typical chunk.
#
# A floor, rather than an added epsilon, leaves every chunk above it exactly unbiased and
# caps only the degenerate tail — an added constant biases all of them, including the
# small deliberate motions these metrics exist to score honestly.
#
# Each value is the corpus median of its own denominator over 10, so no chunk may carry
# more than 10x the weight of a typical one. The three differ by three decades because
# the quantities do: shape is measured on per-step increments, ~1/T of the excursion the
# other two see. At this setting the floor binds on 2.65% of chunks for path, 0.35% for
# shape, 2.95% for terminal. Regenerate whenever dataset.sources changes:
#   python lerobot/src/lerobot/scripts/compute_trajectory_scale_floors.py \
#     --encoding anchor --chunk-size 30 --stats outputs/stats/action_stats_anchor_rebot-annot-v2.pt \
#     --root <every root listed in dataset.sources>
DEFAULT_SCALE_FLOORS = {
    "path": 2.4e-3,
    "shape": 2.6e-6,
    "terminal": 3.0e-3,
}


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


def _squared_errors(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Path, shape, and terminal squared error, each a mean over its own elements."""
    squared_error = (prediction - target).square()
    path = squared_error.mean(dim=(-2, -1))
    terminal = squared_error[..., -1, :].mean(dim=-1)
    if prediction.shape[-2] > 1:
        delta_error = torch.diff(prediction, dim=-2) - torch.diff(target, dim=-2)
        shape = delta_error.square().mean(dim=(-2, -1))
    else:
        shape = torch.zeros_like(path)
    return path, shape, terminal


def trajectory_error_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hold: torch.Tensor,
    *,
    eps: float = 1e-6,
    scale_floors: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    r"""Return complementary errors between action trajectories, raw and relative.

    The ``_mse`` terms are lower-is-better in the tensors' squared action units, and the
    direction term is dimensionless.

    Absolute error is not comparable across chunks: it scales with how far the
    demonstration travels, so a chunk the arm barely moves through scores well even when
    its shape is entirely wrong, while a long sweep with the right shape and a small
    offset scores worse. The ``_relative`` terms divide each error by the same error
    committed by ``hold`` — the chunk that repeats the measured state and never moves —
    which is the demonstration's own scale measured in the matching quantity: total
    excursion for path, per-step motion for shape, final displacement for terminal. They
    read on one scale for every chunk:

        0    exact
        1    no better than freezing the arm for the whole chunk
        > 1  worse than freezing it

    Each denominator is floored at its ``scale_floors`` entry, which bounds the ratio a
    barely-moving chunk can reach and leaves every chunk above the floor untouched.
    """
    scale_floors = DEFAULT_SCALE_FLOORS if scale_floors is None else scale_floors
    if prediction.shape != target.shape or prediction.shape != hold.shape:
        raise ValueError(
            "prediction, target, and hold must have identical shapes; got "
            f"{prediction.shape}, {target.shape}, and {hold.shape}"
        )
    if prediction.ndim < 2:
        raise ValueError(f"expected tensors ending in [time, action], got {prediction.shape}")

    path_mse, shape_mse, terminal_mse = _squared_errors(prediction, target)
    hold_path, hold_shape, hold_terminal = _squared_errors(hold, target)

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
        "path_relative": path_mse / hold_path.clamp_min(scale_floors["path"]),
        "shape_relative": shape_mse / hold_shape.clamp_min(scale_floors["shape"]),
        "terminal_relative": terminal_mse / hold_terminal.clamp_min(scale_floors["terminal"]),
    }
