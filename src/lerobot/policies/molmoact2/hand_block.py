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

from dataclasses import dataclass
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

    def get_config(self) -> dict[str, Any]:
        return {"stats_index_key": self.stats_index_key}

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
        state[:, HAND_STATE_SLOTS] = hand_state(p0, r0, g0)
        state_mask = torch.as_tensor(complementary["state_dim_is_pad"], dtype=torch.bool).clone()
        state_mask[:, HAND_STATE_SLOTS] = ~valid[:, None]
        observation = observation.copy()
        observation[OBS_STATE] = state
        complementary["state_dim_is_pad"] = state_mask

        history_key = f"history.{OBS_STATE}"
        if history_key in complementary:
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
        elif "action_dim_is_pad" in complementary:
            action_mask = torch.as_tensor(complementary["action_dim_is_pad"], dtype=torch.bool).clone()
            action_mask[:, HAND_ACTION_SLOTS] = ~valid[:, None]
            complementary["action_dim_is_pad"] = action_mask

        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
