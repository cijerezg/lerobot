"""Hand block for the EE mixture loss: fingertip pose from the joints, appended to the vectors.

Spec: ``docs/ee_mixture_loss/TODO.md``. The hand frame is the fingertip midpoint with common
axes (z approach, y closing) in the robot's base frame, from the frozen kinematics assets
(``lerobot.model.fk``). Nothing is stored offline: the pose is a pure function of the joints and
the row's ``action_layout_id``, so it is computed where the joints are, in the processor.

Vectors, when the block is on (``MolmoAct2Config.hand_block``):

- action chunk, 15 wide: joints in slots 0..7 (anchor-encoded as today), hand delta in 8..14 =
  ``(p_t - p_0, log(R_t R_0^T), g_t - g_0)``, the same "action minus anchor" sense as the joints
  (the TODO writes ``p_0 - p_t``; the code's joint encoding is ``action - anchor``, so the hand
  block follows the code). Position m, rotation vector rad, aperture m.
- state, 18 wide: joints in 0..7, hand state in 8..17 = ``(p_0, R_0[:, 0], R_0[:, 1], g_0)``.
- masks: the joint block keeps its prefix-valid padding inside slots 0..7; the hand block is
  valid as a whole for rows whose layout has a kinematic chain, padding otherwise (MolmoAct and
  the robots without an asset yet).

Normalization uses the same per-row artifact as the joints; the hand columns are identical in
every row (global per-step stats, one scale per block), see ``scripts/compute_diverse_stats.py --hand``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.model.fk import SerialChain, get_chain
from lerobot.model.fk.chain import rotation_log
from lerobot.policies.molmoact2.action_layout import JOINT_SLOTS
from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

HAND_DELTA_DIM = 7  # dp 3, drot 3, dg 1
HAND_STATE_DIM = 10  # p 3, two rotation columns 6, g 1
HAND_ACTION_WIDTH = JOINT_SLOTS + HAND_DELTA_DIM  # 15
HAND_STATE_WIDTH = JOINT_SLOTS + HAND_STATE_DIM  # 18
HAND_ACTION_SLOTS = slice(JOINT_SLOTS, HAND_ACTION_WIDTH)
HAND_STATE_SLOTS = slice(JOINT_SLOTS, HAND_STATE_WIDTH)
# Block boundaries inside the hand delta / hand state, for the one-scale-per-block stats.
HAND_DELTA_BLOCKS = ((0, 3), (3, 6), (6, 7))
HAND_STATE_BLOCKS = ((0, 3), (3, 9), (9, 10))

# action_layout_id (datasets/diverse_actor_selection.ACTION_LAYOUTS index) -> kinematics asset.
# Layouts not listed have no hand block: 3 ARX5, 4 UR5, 5 UR7e, 8 YAM (no asset yet) and
# 7 MolmoAct (pose only; its Euler convention is not established, see TODO.md Phase 1).
LAYOUT_CHAIN_KEYS: dict[int, str] = {
    0: "droid",
    1: "droid_success",
    2: "fmb",
    6: "rebot_b601_follower",
}


@cache
def layout_chain(layout_id: int) -> SerialChain | None:
    key = LAYOUT_CHAIN_KEYS.get(int(layout_id))
    return None if key is None else get_chain(key)


def layout_has_hand(layout_id: int) -> bool:
    return int(layout_id) in LAYOUT_CHAIN_KEYS


def hand_pose(layout_ids: Tensor, vectors: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fingertip pose of joint vectors, per row's layout.

    ``layout_ids`` (B,), ``vectors`` (B, ..., D >= JOINT_SLOTS) in source order (joints then
    gripper, raw corpus units). Returns ``p`` (B, ..., 3), ``R`` (B, ..., 3, 3), ``g`` (B, ...)
    in ``vectors.dtype`` and ``valid`` (B,) bool. Rows without a chain get zeros, identity and
    ``valid = False``.
    """
    vectors = torch.as_tensor(vectors)
    layout_ids = torch.as_tensor(layout_ids).reshape(-1).to(torch.long)
    if int(layout_ids.shape[0]) != int(vectors.shape[0]):
        raise ValueError(f"layout_ids {tuple(layout_ids.shape)} do not match vectors {tuple(vectors.shape)}.")
    lead = vectors.shape[:-1]
    p = torch.zeros((*lead, 3), dtype=vectors.dtype, device=vectors.device)
    r = torch.eye(3, dtype=vectors.dtype, device=vectors.device).expand(*lead, 3, 3).clone()
    g = torch.zeros(lead, dtype=vectors.dtype, device=vectors.device)
    valid = torch.zeros(vectors.shape[0], dtype=torch.bool, device=vectors.device)
    for layout_id in torch.unique(layout_ids).tolist():
        chain = layout_chain(layout_id)
        if chain is None:
            continue
        rows = torch.nonzero(layout_ids == layout_id).reshape(-1)
        sub = vectors[rows]  # (n, ..., D)
        inner = sub.shape[:-1]
        q = sub[..., : chain.dof + 1].reshape(-1, chain.dof + 1).to(torch.float64)
        pr, rr, gr = chain.fk(q)
        p[rows] = pr.reshape(*inner, 3).to(p)
        r[rows] = rr.reshape(*inner, 3, 3).to(r)
        g[rows] = gr.reshape(*inner).to(g)
        valid[rows] = True
    return p, r, g, valid


def hand_delta(p0: Tensor, r0: Tensor, g0: Tensor, p: Tensor, r: Tensor, g: Tensor) -> Tensor:
    """``(p - p0, log(R R0^T), g - g0)`` over any leading shape; the anchor broadcasts."""
    lead = torch.broadcast_shapes(p.shape[:-1], p0.shape[:-1])
    rel = (r @ r0.transpose(-1, -2)).expand(*lead, 3, 3).reshape(-1, 3, 3)
    drot = rotation_log(rel.to(torch.float64)).to(p).reshape(*lead, 3)
    return torch.cat([(p - p0).expand(*lead, 3), drot, (g - g0).expand(*lead)[..., None]], dim=-1)


def hand_state(p: Tensor, r: Tensor, g: Tensor) -> Tensor:
    """``(p, R[:, 0], R[:, 1], g)``: position, the hand x and y axes in the base frame, aperture."""
    return torch.cat([p, r[..., :, 0], r[..., :, 1], g[..., None]], dim=-1)


def rotation_exp(v: Tensor) -> Tensor:
    """Rodrigues: rotation vectors (..., 3) to matrices (..., 3, 3); the inverse of ``rotation_log``."""
    v = torch.as_tensor(v)
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = v / theta
    kx = torch.zeros((*v.shape[:-1], 3, 3), dtype=v.dtype, device=v.device)
    kx[..., 0, 1], kx[..., 0, 2] = -k[..., 2], k[..., 1]
    kx[..., 1, 0], kx[..., 1, 2] = k[..., 2], -k[..., 0]
    kx[..., 2, 0], kx[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return eye + s * kx + (1 - c) * (kx @ kx)


def hand_monitor_gap(layout_ids: Tensor, anchor: Tensor, action: Tensor) -> dict[str, Tensor]:
    """Joint-vs-hand agreement of one unnormalized chunk, the inference monitor.

    ``anchor`` (B, >= 8) raw joints, ``action`` (B, T, 15) raw anchor deltas. The hand the
    decoded joints produce, ``FK(anchor + joint delta)`` relative to ``FK(anchor)``, against the
    hand block the model wrote. Returns per-row means over the chunk: ``position_mm``,
    ``rotation_deg``, ``aperture_mm`` and ``valid`` (False where the layout has no chain; the
    numbers are NaN there).
    """
    action = torch.as_tensor(action)
    anchor = torch.as_tensor(anchor).to(action)
    joints = anchor[:, None, :JOINT_SLOTS] + action[..., :JOINT_SLOTS]
    p0, r0, g0, valid = hand_pose(layout_ids, anchor[:, :JOINT_SLOTS])
    pt, rt, gt, _ = hand_pose(layout_ids, joints)
    fk_delta = hand_delta(p0[:, None], r0[:, None], g0[:, None], pt, rt, gt)  # (B, T, 7)
    pred = action[..., HAND_ACTION_SLOTS]
    position = (fk_delta[..., :3] - pred[..., :3]).norm(dim=-1).mean(-1) * 1000.0
    relative = rotation_exp(fk_delta[..., 3:6].double()) @ rotation_exp(pred[..., 3:6].double()).transpose(
        -1, -2
    )
    angle = rotation_log(relative.reshape(-1, 3, 3)).norm(dim=-1).reshape(pred.shape[:-1])
    rotation = angle.to(action).mean(-1) * (180.0 / math.pi)
    aperture = (fk_delta[..., 6] - pred[..., 6]).abs().mean(-1) * 1000.0
    nan = torch.full_like(position, float("nan"))
    return {
        "position_mm": torch.where(valid, position, nan),
        "rotation_deg": torch.where(valid, rotation, nan),
        "aperture_mm": torch.where(valid, aperture, nan),
        "valid": valid,
    }


def unnormalize_joint_block(normalized: Tensor, q01: Tensor, q99: Tensor) -> Tensor:
    """Inverse of the QUANTILES normalizer on the joint block: ``(x + 1) (q99 - q01) / 2 + q01``."""
    return (normalized + 1.0) * (q99 - q01) / 2.0 + q01


def hand_pose_from_normalized_joints(
    normalized_action: Tensor, q01: Tensor, q99: Tensor, anchor: Tensor, layout_ids: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The unnormalize-then-FK path of the FK term.

    ``normalized_action`` (B, T, W >= JOINT_SLOTS) in normalized anchor-encoded space, ``q01`` /
    ``q99`` the rows' per-step stats broadcastable to it, ``anchor`` (B, >= JOINT_SLOTS) the raw
    anchor state. Returns the fingertip pose of the absolute joints ``anchor + delta`` per step.
    Differentiable in ``normalized_action``.
    """
    joints = JOINT_SLOTS
    delta = unnormalize_joint_block(normalized_action[..., :joints], q01[..., :joints], q99[..., :joints])
    absolute = anchor[:, None, :joints].to(delta) + delta
    return hand_pose(layout_ids, absolute)


@ProcessorStepRegistry.register(name="molmoact2_hand_block")
@dataclass
class MolmoAct2HandBlockProcessorStep(ProcessorStep):
    """Fill the hand block of the state, the state history and the anchor-encoded action chunk.

    Insert after ``AnchorEncodeStep`` (encoding "anchor") and before the normalizer. Expects the
    unified-layout step to have padded the state to 18 and the action to 15, and the row's
    ``action_layout_id`` in the complementary data.
    """

    stats_index_key: str = "action_layout_id"
    # Hand pose in the state (slots 8..17). Off (ablation C): the slots stay zero and padding,
    # and the pack step renders the joint state tokens only (prompt_state_width).
    hand_state: bool = True
    # The FK term's inputs (hand_loss.py), gathered per row and shipped as hand_fk_* keys:
    # the artifact's joint-block q01/q99 ([E, T, 8], so the model can unnormalize the joint
    # block of its own sample) and the half-bands of the hand columns ([T, 3]: position,
    # rotation, aperture) that put the FK residual in the hand block's normalized units.
    # None = no FK keys (inference, or a run without the FK term). Tensors ride state_dict.
    joint_q01: Tensor | None = None
    joint_q99: Tensor | None = None
    block_scale: Tensor | None = None

    def get_config(self) -> dict[str, Any]:
        return {"stats_index_key": self.stats_index_key, "hand_state": bool(self.hand_state)}

    def state_dict(self) -> dict[str, Tensor]:
        return {
            name: torch.as_tensor(getattr(self, name)).detach().cpu().clone()
            for name in ("joint_q01", "joint_q99", "block_scale")
            if getattr(self, name) is not None
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for name in ("joint_q01", "joint_q99", "block_scale"):
            if name in state:
                setattr(self, name, torch.as_tensor(state[name]).detach().cpu().clone())

    @property
    def emits_fk_keys(self) -> bool:
        return self.joint_q01 is not None and self.joint_q99 is not None and self.block_scale is not None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        if not isinstance(observation, dict) or OBS_STATE not in observation:
            return transition
        layout_ids = complementary.get(self.stats_index_key)
        if layout_ids is None:
            raise ValueError(f"hand block needs complementary_data[{self.stats_index_key!r}] on every row.")
        state = torch.as_tensor(observation[OBS_STATE])
        if state.ndim == 1:
            state = state[None]
        if int(state.shape[-1]) != HAND_STATE_WIDTH:
            raise ValueError(
                f"hand block expects the state padded to {HAND_STATE_WIDTH}, got {state.shape[-1]}."
            )
        layout_ids = torch.as_tensor(layout_ids).reshape(-1).to(torch.long)

        p0, r0, g0, valid = hand_pose(layout_ids, state)
        state = state.clone()
        state_mask = torch.as_tensor(complementary["state_dim_is_pad"], dtype=torch.bool).clone()
        if self.hand_state:
            state[:, HAND_STATE_SLOTS] = hand_state(p0, r0, g0)
            state_mask[:, HAND_STATE_SLOTS] = ~valid[:, None]
        else:
            state[:, HAND_STATE_SLOTS] = 0.0
            state_mask[:, HAND_STATE_SLOTS] = True
        observation = observation.copy()
        observation[OBS_STATE] = state
        complementary["state_dim_is_pad"] = state_mask

        history_key = f"history.{OBS_STATE}"
        if history_key in complementary and self.hand_state:
            history = torch.as_tensor(complementary[history_key]).clone()  # (B, H, 18)
            ph, rh, gh, _ = hand_pose(layout_ids, history)
            history[..., HAND_STATE_SLOTS] = hand_state(ph, rh, gh)
            complementary[history_key] = history

        if action is not None:
            anchor = complementary.get(ANCHOR_KEY)
            if anchor is None:
                raise ValueError(
                    "hand block runs after AnchorEncodeStep (encoding 'anchor'); no anchor found."
                )
            action = torch.as_tensor(action).clone()  # (B, T, 15), joint block anchor-encoded
            if int(action.shape[-1]) != HAND_ACTION_WIDTH:
                raise ValueError(
                    f"hand block expects the action padded to {HAND_ACTION_WIDTH}, got {action.shape[-1]}."
                )
            anchor = torch.as_tensor(anchor).to(action)
            absolute = action[..., :JOINT_SLOTS] + anchor[:, None, :JOINT_SLOTS]
            pt, rt, gt, _ = hand_pose(layout_ids, absolute)
            action[..., HAND_ACTION_SLOTS] = hand_delta(p0[:, None], r0[:, None], g0[:, None], pt, rt, gt).to(
                action
            )
            action_mask = torch.as_tensor(complementary["action_dim_is_pad"], dtype=torch.bool).clone()
            action_mask[:, HAND_ACTION_SLOTS] = ~valid[:, None]
            transition[TransitionKey.ACTION] = action
            complementary["action_dim_is_pad"] = action_mask
            if self.emits_fk_keys:
                complementary.update(
                    self._fk_keys(layout_ids, valid, anchor, pt, rt, gt, horizon=int(action.shape[1]))
                )
        elif "action_dim_is_pad" in complementary:
            action_mask = torch.as_tensor(complementary["action_dim_is_pad"], dtype=torch.bool).clone()
            action_mask[:, HAND_ACTION_SLOTS] = ~valid[:, None]
            complementary["action_dim_is_pad"] = action_mask

        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def _fk_keys(
        self,
        layout_ids: Tensor,
        valid: Tensor,
        anchor: Tensor,
        pt: Tensor,
        rt: Tensor,
        gt: Tensor,
        *,
        horizon: int,
    ) -> dict[str, Tensor]:
        """The FK term's batch keys (hand_loss.HAND_FK_KEYS), gathered for this batch's rows."""
        q01 = torch.as_tensor(self.joint_q01)
        q99 = torch.as_tensor(self.joint_q99)
        scale = torch.as_tensor(self.block_scale)
        if int(q01.shape[1]) != horizon or int(scale.shape[0]) != horizon:
            raise ValueError(
                f"hand FK stats hold a {int(q01.shape[1])}-step chunk, the batch a {horizon}-step one."
            )
        rows = layout_ids.clamp(0, int(q01.shape[0]) - 1)
        return {
            "hand_fk_valid": valid.clone(),
            "hand_fk_layout_id": layout_ids.clone(),
            "hand_fk_anchor": anchor[:, :JOINT_SLOTS].detach().to(torch.float32).clone(),
            "hand_fk_q01": q01[rows].to(torch.float32),
            "hand_fk_q99": q99[rows].to(torch.float32),
            "hand_fk_target_p": pt.detach().to(torch.float32),
            "hand_fk_target_r": rt.detach().to(torch.float32),
            "hand_fk_target_g": gt.detach().to(torch.float32),
            "hand_fk_block_scale": scale.to(torch.float32).clone(),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="molmoact2_hand_monitor")
@dataclass
class MolmoAct2HandMonitorProcessorStep(ProcessorStep):
    """Inference: read the hand block of the generated chunk as a free monitor, then drop it.

    Insert after the unnormalizer and before ``AnchorDecodeStep``. The action arrives
    ``(B, T, 15)`` in raw anchor-delta units, the anchor (the raw state prefix, >= 8 wide) and
    the row's layout id ride the complementary data. The gap between the hand the decoded
    joints produce and the hand the model wrote goes to ``complementary["hand_monitor"]`` and
    to :attr:`last`; the action leaves 8 wide (the joint block) and so does the anchor, so the
    anchor decode and the width restore downstream see exactly what they see with the hand
    block off. The hand block is never executed.
    """

    stats_index_key: str = "action_layout_id"
    # Log the batch-mean gap through ``logging`` every n calls; 0 never logs.
    log_every: int = 20
    last: dict[str, Tensor] | None = field(default=None, repr=False, compare=False)
    _calls: int = field(default=0, repr=False, compare=False)

    def get_config(self) -> dict[str, Any]:
        return {"stats_index_key": self.stats_index_key, "log_every": int(self.log_every)}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        transition = transition.copy()
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        action = torch.as_tensor(action)
        squeeze = action.ndim == 2
        if squeeze:
            action = action[None]
        if int(action.shape[-1]) != HAND_ACTION_WIDTH:
            raise ValueError(
                f"hand monitor expects a {HAND_ACTION_WIDTH}-wide chunk, got {action.shape[-1]}."
            )
        anchor = complementary.get(ANCHOR_KEY)
        if anchor is None:
            raise ValueError("hand monitor runs on the anchor path; no anchor in the payload.")
        anchor = torch.as_tensor(anchor)
        if anchor.ndim == 1:
            anchor = anchor[None]
        layout_ids = complementary.get(self.stats_index_key)
        if layout_ids is None:
            raise ValueError(f"hand monitor needs complementary_data[{self.stats_index_key!r}].")
        layout_ids = torch.as_tensor(layout_ids).reshape(-1).to(torch.long)
        if int(layout_ids.numel()) == 1 and int(action.shape[0]) > 1:
            layout_ids = layout_ids.expand(int(action.shape[0]))
        gap = hand_monitor_gap(layout_ids.cpu(), anchor.detach().cpu(), action.detach().cpu())
        self.last = gap
        complementary["hand_monitor"] = gap
        self._calls += 1
        if self.log_every > 0 and self._calls % self.log_every == 0 and bool(gap["valid"].any()):
            logging.getLogger(__name__).info(
                "[hand_monitor] joints vs hand: %.1f mm, %.2f deg, %.1f mm aperture",
                float(gap["position_mm"][gap["valid"]].mean()),
                float(gap["rotation_deg"][gap["valid"]].mean()),
                float(gap["aperture_mm"][gap["valid"]].mean()),
            )
        joints = action[..., :JOINT_SLOTS]
        complementary[ANCHOR_KEY] = anchor[..., :JOINT_SLOTS]
        transition[TransitionKey.ACTION] = joints[0] if squeeze else joints
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
