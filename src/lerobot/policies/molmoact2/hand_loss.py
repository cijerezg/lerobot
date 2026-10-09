"""The EE mixture loss terms on the flow expert (docs/ee_mixture_loss/TODO.md Phase 3).

Pure functions on tensors so they can be tested without a checkpoint. The policy
(``modeling_molmoact2._compute_flow_matching_loss_joint_per_layer``) calls them when
``MolmoAct2Config.hand_block`` is on.

Two terms:

- :func:`block_mean_flow_loss` replaces the per-element mean over the valid action slots by
  block means: the joint block (slots 0..7, prefix-valid), and the hand blocks position
  (8..10), rotation (11..13) and aperture (14). Every block sits at the same chance level
  because the hand columns are normalized to the joints' q01/q99 band
  (``scripts/compute_diverse_stats.py --hand``), so a weight of 1 per term means what it says.
  The hand term is the mean of its three block means; the joint block carries
  ``joint_weight`` (lambda) on rows that have a hand block and 1 on rows that do not.
- :func:`hand_fk_term` is the FK term: the fingertip pose of the implied clean joint sample
  ``x_hat = x_tau + (1 - tau) v_hat`` (unnormalized, anchor added, through the frozen
  kinematics chain) against the demonstrated fingertip pose, as three block means in the hand
  columns' normalized units. Rotations are compared as matrices, ``0.5 ||R_hat - R||_F^2``,
  which is the squared angle for small rotations and has no log map. Rows without a chain
  contribute nothing. The natural ``(1 - tau)^2`` weight of the clean-sample error is kept.

Batch keys the FK term reads, written by ``hand_block.MolmoAct2HandBlockProcessorStep``:
``hand_fk_valid`` (B,) bool, ``hand_fk_layout_id`` (B,), ``hand_fk_anchor`` (B, 8) raw joints,
``hand_fk_q01`` / ``hand_fk_q99`` (B, T, 8) the rows' joint-block quantiles,
``hand_fk_target_p`` (B, T, 3), ``hand_fk_target_r`` (B, T, 3, 3), ``hand_fk_target_g`` (B, T),
``hand_fk_block_scale`` (T, 3) the half-bands of the position, rotation and aperture columns.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.molmoact2.action_layout import JOINT_SLOTS
from lerobot.policies.molmoact2.hand_block import (
    HAND_ACTION_WIDTH,
    HAND_DELTA_BLOCKS,
    hand_pose_from_normalized_joints,
)

HAND_FK_KEYS = (
    "hand_fk_valid",
    "hand_fk_layout_id",
    "hand_fk_anchor",
    "hand_fk_q01",
    "hand_fk_q99",
    "hand_fk_target_p",
    "hand_fk_target_r",
    "hand_fk_target_g",
    "hand_fk_block_scale",
)
HAND_BLOCK_NAMES = ("position", "rotation", "aperture")


def _hand_block_slices() -> list[slice]:
    return [slice(JOINT_SLOTS + lo, JOINT_SLOTS + hi) for lo, hi in HAND_DELTA_BLOCKS]


def block_mean_flow_loss(
    loss: Tensor, action_dim_is_pad: Tensor, *, joint_weight: float
) -> tuple[Tensor, dict[str, Tensor]]:
    """Reduce the per-element flow loss over the action width as block means.

    ``loss`` (B, F, T, D >= 15) squared velocity errors, ``action_dim_is_pad`` (B, D) bool with
    the joint block prefix-valid in slots 0..7 and the hand slots 8..14 valid as a whole or not.
    Returns the per-(B, F, T) loss and detached block means for logging (``joint`` over every
    row, the hand blocks over the rows that carry a hand block).
    """
    if loss.shape[-1] < HAND_ACTION_WIDTH:
        raise ValueError(
            f"hand block loss needs an action width >= {HAND_ACTION_WIDTH}, got {loss.shape[-1]}."
        )
    pad = torch.as_tensor(action_dim_is_pad, device=loss.device, dtype=torch.bool)
    if pad.ndim != 2 or pad.shape[0] != loss.shape[0] or pad.shape[-1] != loss.shape[-1]:
        raise ValueError(f"action_dim_is_pad {tuple(pad.shape)} does not match the loss {tuple(loss.shape)}.")
    valid = (~pad).to(loss.dtype)[:, None, None, :]  # (B, 1, 1, D)
    joint_valid = valid[..., :JOINT_SLOTS]
    joint = (loss[..., :JOINT_SLOTS] * joint_valid).sum(-1) / joint_valid.sum(-1).clamp_min(1.0)
    hand_rows = ~pad[:, JOINT_SLOTS]  # (B,)
    blocks = [loss[..., sl].mean(-1) for sl in _hand_block_slices()]  # each (B, F, T)
    hand = torch.stack(blocks, dim=0).mean(0)
    row_hand = hand_rows[:, None, None]
    weight = torch.where(row_hand, torch.full_like(joint, float(joint_weight)), torch.ones_like(joint))
    reduced = weight * joint + torch.where(row_hand, hand, torch.zeros_like(hand))

    means: dict[str, Tensor] = {"joint": joint.detach().float().mean()}
    if bool(hand_rows.any()):
        for name, block in zip(HAND_BLOCK_NAMES, blocks, strict=True):
            means[name] = block[hand_rows].detach().float().mean()
        means["hand"] = hand[hand_rows].detach().float().mean()
    return reduced, means


def hand_fk_term(
    joints_hat: Tensor,
    batch: dict[str, Tensor],
    *,
    action_horizon_is_pad: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The FK term per sample.

    ``joints_hat`` (B, F, T, >= 8): the implied clean sample's joint block in normalized
    anchor-encoded units, with gradient. Returns ``(B,)`` with zeros for rows without a chain,
    and detached block means over the rows that have one.
    """
    missing = [key for key in HAND_FK_KEYS if batch.get(key) is None]
    if missing:
        raise KeyError(f"hand FK term needs batch keys {missing}; is the hand step in the preprocessor?")
    batch_size, num_flow, horizon = (
        int(joints_hat.shape[0]),
        int(joints_hat.shape[1]),
        int(joints_hat.shape[2]),
    )
    device = joints_hat.device
    valid = torch.as_tensor(batch["hand_fk_valid"], device=device, dtype=torch.bool).reshape(-1)
    out = torch.zeros(batch_size, device=device, dtype=torch.float32)
    rows = torch.nonzero(valid).reshape(-1)
    if int(rows.numel()) == 0:
        return out, {}

    x = joints_hat[rows, ..., :JOINT_SLOTS].float().reshape(-1, horizon, JOINT_SLOTS)  # (b F, T, 8)
    repeat = lambda t: torch.as_tensor(t, device=device)[rows].repeat_interleave(num_flow, dim=0)  # noqa: E731
    q01 = repeat(batch["hand_fk_q01"]).float()
    q99 = repeat(batch["hand_fk_q99"]).float()
    anchor = repeat(batch["hand_fk_anchor"]).float()
    layout = repeat(batch["hand_fk_layout_id"]).reshape(-1)
    p, r, g, chain_valid = hand_pose_from_normalized_joints(x, q01, q99, anchor, layout)
    if not bool(chain_valid.all()):
        raise ValueError("hand_fk_valid marks a row whose layout has no kinematic chain.")
    target_p = repeat(batch["hand_fk_target_p"]).float()
    target_r = repeat(batch["hand_fk_target_r"]).float()
    target_g = repeat(batch["hand_fk_target_g"]).float()
    scale = torch.as_tensor(batch["hand_fk_block_scale"], device=device).float()  # (T, 3)
    if tuple(scale.shape) != (horizon, 3):
        raise ValueError(f"hand_fk_block_scale must be (T={horizon}, 3), got {tuple(scale.shape)}.")
    s_p, s_r, s_g = scale[:, 0], scale[:, 1], scale[:, 2]  # each (T,)
    position = ((p - target_p) ** 2).sum(-1) / (3.0 * s_p**2)
    rotation = 0.5 * ((r - target_r) ** 2).sum((-1, -2)) / (3.0 * s_r**2)
    aperture = (g - target_g) ** 2 / s_g**2
    per_step = (position + rotation + aperture) / 3.0  # (b F, T)

    if action_horizon_is_pad is not None:
        step_valid = (~torch.as_tensor(action_horizon_is_pad, device=device, dtype=torch.bool))[rows]
        step_valid = step_valid.repeat_interleave(num_flow, dim=0).float()
    else:
        step_valid = torch.ones_like(per_step)
    per_sample = (per_step * step_valid).sum(-1) / step_valid.sum(-1).clamp_min(1.0)  # (b F,)
    out[rows] = per_sample.reshape(-1, num_flow).mean(-1)

    def _mean(block: Tensor) -> Tensor:
        return ((block * step_valid).sum() / step_valid.sum().clamp_min(1.0)).detach().float()

    means = {
        "fk_position": _mean(position),
        "fk_rotation": _mean(rotation),
        "fk_aperture": _mean(aperture),
        "fk": _mean(per_step),
    }
    return out, means


def implied_clean_sample(xt: Tensor, pred_velocity: Tensor, timesteps: Tensor) -> Tensor:
    """``x_hat = x_tau + (1 - tau) v_hat`` with ``xt``/``pred_velocity`` (B, F, T, D) and
    ``timesteps`` (B, F); the flow runs from noise at tau = 0 to the clean chunk at tau = 1."""
    t = timesteps.to(xt.dtype).view(xt.shape[0], xt.shape[1], 1, 1)
    return xt + (1.0 - t) * pred_velocity
