"""Depth stream blocks + wrist-cam token helpers.

CRITIC-ONLY as of the prefix migration. The ACTOR no longer co-evolves depth: the
point-map encoder's tokens go through a separate copy of Molmo's visual blocks,
pooler, and projector into DEPTH_TOKEN positions in the VLM prefix.
The joint-softmax read (``join_depth_columns``, ``depth_attention_mass``), the learned
per-layer bias ``b_l``, ``slice_wrist_cam_kv`` and the ``DepthStream`` aggregate that
owned them are all deleted — as is the alpha gate they replaced.

What remains is used by two callers:

  - ``DepthStreamBlock`` (+ ``StreamAttention``/``StreamMLP``) — the critic
    (rl_molmoact2.py) still stacks these over its own encoder's tokens, attending its
    own wrist-cam embeds, and appends the final state for its value queries. Per block
    (pre-norm; width d_d, ``stream_num_heads`` heads):

        t <- t + SelfAttn(LN(t))
        t <- t + CrossAttn(LN(t); K=k_wrist, V=v_wrist)
        t <- t + MLP(LN(t))

    ``cross_on=False`` kills the wrist bridge for a row: the K/V arrive by direct
    gather, which bypasses attention masks, so RGB dropout has to be applied here
    explicitly or the dropped rows would still see the wrist through this path.

  - ``wrist_cam_token_indices`` / ``gather_kv_at_indices`` / ``mask_camera_patch_span``
    — the processor's RGB-dropout mask and the depth probe.

Fresh float32.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_pointmap import DepthPointmapConfig


def wrist_cam_token_indices(
    input_ids: Tensor, *, image_patch_id: int, num_images: int, cam_index: int
) -> Tensor:
    """Per-row sequence positions of the depth camera's image-patch tokens.

    The VLM lays out image-patch tokens (``input_ids == image_patch_id``) as
    ``num_images`` equal-length contiguous runs, one per camera in ``image_keys``
    order. ``cam_index`` selects the run belonging to the depth camera. The indices
    are recovered per-row (robust to left-padding / variable task-text length); the
    only assumption is that every row carries the same number of image-patch tokens.

    Returns ``(B, T_cam)`` integer column indices.
    """
    is_img = input_ids == image_patch_id  # (B, T)
    counts = is_img.sum(dim=1)  # (B,)
    total = int(counts[0].item())
    if not bool((counts == total).all().item()):
        raise ValueError(f"Rows carry unequal image-patch token counts: {counts.tolist()}.")
    if total % num_images:
        raise ValueError(f"{total} image-patch tokens not divisible by num_images={num_images}.")
    per_image = total // num_images
    cols = is_img.nonzero(as_tuple=False)[:, 1].view(input_ids.shape[0], total)
    return cols[:, cam_index * per_image : (cam_index + 1) * per_image]  # (B, T_cam)


def mask_camera_patch_span(
    attention_mask: Tensor,
    input_ids: Tensor,
    *,
    image_patch_id: int,
    num_images: int,
    cam_index: int,
    rows: Tensor | None = None,
) -> Tensor:
    """Return ``attention_mask`` with one camera's image-patch span disabled.

    ``rows`` selects the batch rows to mask; ``None`` masks every row. This is the
    single production implementation shared by training-time RGB dropout and the
    depth-modality probe, so the probe's RGB− condition exercises the same mask
    semantics (including the model-side bridge kill derived from this mask).
    """
    masked = attention_mask.clone()
    sel = wrist_cam_token_indices(
        input_ids,
        image_patch_id=image_patch_id,
        num_images=num_images,
        cam_index=cam_index,
    )
    if rows is None:
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    else:
        rows = rows.to(device=input_ids.device, dtype=torch.long)
    masked[rows[:, None], sel[rows]] = 0
    return masked


def gather_kv_at_indices(key: Tensor, value: Tensor, sel: Tensor) -> tuple[Tensor, Tensor]:
    """Gather one layer's flat ``(B, T, d_vlm)`` K/V at the ``(B, T_cam)`` positions ``sel``."""
    idx = sel.unsqueeze(-1).expand(-1, -1, key.shape[-1])  # (B, T_cam, d_vlm)
    return torch.gather(key, 1, idx), torch.gather(value, 1, idx)


class StreamAttention(nn.Module):
    """Multi-head attention with independent query / key / value input widths.

    Self-attention passes the same tensor as query, key, and value (``d_kv == d_q``);
    cross-attention passes depth tokens as the query and the wrist-cam K/V (width
    ``d_kv``) as key and value inputs.
    """

    def __init__(self, *, d_q: int, d_kv: int, num_heads: int) -> None:
        super().__init__()
        if d_q % num_heads:
            raise ValueError(f"d_q {d_q} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.head_dim = d_q // num_heads
        self.q_proj = nn.Linear(d_q, d_q)
        self.k_proj = nn.Linear(d_kv, d_q)
        self.v_proj = nn.Linear(d_kv, d_q)
        self.out_proj = nn.Linear(d_q, d_q)

    def _heads(self, x: Tensor, proj: nn.Linear) -> Tensor:
        b, t, _ = x.shape
        return proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)

    def forward(self, x_q: Tensor, x_k: Tensor, x_v: Tensor) -> Tensor:
        b, tq, _ = x_q.shape
        q = self._heads(x_q, self.q_proj)
        k = self._heads(x_k, self.k_proj)
        v = self._heads(x_v, self.v_proj)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, Tq, Dh)
        out = out.transpose(1, 2).reshape(b, tq, self.num_heads * self.head_dim)
        return self.out_proj(out)


class StreamMLP(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DepthStreamBlock(nn.Module):
    """One co-evolution block: depth self-attn + depth→wrist-cam cross-attn + MLP.

    ``cross_on`` (B,) bool kills the wrist bridge per row: the RGB-dropout
    attention mask (processor-side) doesn't reach this direct K/V gather, so
    dropped samples must zero the cross-attn residual here — forward contribution
    and gradient both vanish (×0), and the row's stream runs depth-only
    (self-attn + MLP), the correct "no RGB" semantics."""

    def __init__(self, *, d_d: int, d_vlm: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_d)
        self.self_attn = StreamAttention(d_q=d_d, d_kv=d_d, num_heads=num_heads)
        self.norm_cross = nn.LayerNorm(d_d)
        self.cross_attn = StreamAttention(d_q=d_d, d_kv=d_vlm, num_heads=num_heads)
        self.norm_mlp = nn.LayerNorm(d_d)
        self.mlp = StreamMLP(d_d, mlp_ratio)

    def forward(
        self, t: Tensor, wrist_k: Tensor, wrist_v: Tensor, cross_on: Tensor | None = None
    ) -> Tensor:
        h = self.norm_self(t)
        t = t + self.self_attn(h, h, h)
        cross = self.cross_attn(self.norm_cross(t), wrist_k, wrist_v)
        if cross_on is not None:
            cross = cross * cross_on.to(cross.dtype).view(-1, 1, 1)
        t = t + cross
        t = t + self.mlp(self.norm_mlp(t))
        return t


