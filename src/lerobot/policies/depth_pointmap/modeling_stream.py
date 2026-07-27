"""Co-evolving depth stream + action-read bridge (depth_pointmap_design.md Part B).

The point-map encoder (modeling_pointmap.py) tokenizes one depth frame into N tokens
of width ``d_d`` (the stream width). This module then lets those tokens **co-evolve**
through M light transformer blocks so the action expert can read depth features *as
they exist at each of its layers*, rather than one frozen single-shot encoding.

Per block ℓ (pre-norm; width d_d, ``stream_num_heads`` heads):

    t ← t + SelfAttn(LN(t))                      # depth tokens mix among themselves
    t ← t + CrossAttn(LN(t); K=k_wristℓ, V=v_wristℓ)   # attend the wrist-cam KV at layer ℓ
    t ← t + MLP(LN(t))

The wrist-cam K/V are the VLM's own per-layer cached keys/values, sliced to the
wrist-camera token span (model-side) and handed in as ``(B, T_w, d_vlm)`` tensors;
each block projects them d_vlm→d_d internally. The depth stream attends **self +
wrist-cam only** (design §3.2) and has no action dependence, so it is a pure function
of the observation — computed once per observation and reused across all flow-matching
denoising steps.

Action-read bridge (JOINT softmax, decided 2026-07-26 — depth_redesign_options.md
§5.2; replaces the α-gated additive read whose scalar gate never trained). For
action layer ℓ the action expert runs ONE softmax over context and depth columns:

    out = SDPA(q, [K_ctx ; K_dℓ], [V_ctx ; V_dℓ], mask=[ctx mask ; b_ℓ])

where (K_dℓ, V_dℓ) are this module's per-layer depth state projected into the action
expert's head space (``read_kv``) and ``b_ℓ = depth_bias[ℓ]`` is a learned per-layer
score bias on the depth columns (init −2: initial depth mass ×e⁻² without zeroing
content gradients — additive inside the softmax, unlike the retired multiplicative
gate). Abstention needs no sink column: the context keys are the natural sink. The
SDPA itself lives in the action expert; this module owns the projections and
``depth_bias``. Fresh float32.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_pointmap import DepthPointmapConfig


def join_depth_columns(
    attn_mask: Tensor | None,
    *,
    k: Tensor,
    v: Tensor,
    depth_kv: tuple[Tensor, Tensor],
    depth_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Extend a cross-attention key set with depth columns for the joint softmax
    (depth_redesign_options.md §5.2).

    k, v:       (B, T_ctx, H, Dh) context keys/values.
    depth_kv:   (K_d, V_d), each (B, N, H, Dh) — the layer's depth state in action
                head space (already k-normed by the caller, like the context keys).
    attn_mask:  (B, 1, 1, T_ctx) additive context mask (0 valid / finfo.min pad),
                or None.
    depth_bias: () scalar — this layer's learned score bias b_ℓ on the depth
                columns; kept in the graph (the returned mask carries its grad).

    Returns (k_joint, v_joint, mask_joint) with T_ctx + N columns, mask
    (B or 1, 1, 1, T_ctx + N).
    """
    k_d, v_d = depth_kv
    n_d = k_d.shape[1]
    ctx_cols = k.new_zeros(1, 1, 1, k.shape[1]) if attn_mask is None else attn_mask
    bias_cols = depth_bias.to(ctx_cols.dtype).reshape(1, 1, 1, 1).expand(*ctx_cols.shape[:-1], n_d)
    mask_joint = torch.cat([ctx_cols, bias_cols], dim=-1)
    return torch.cat([k, k_d], dim=1), torch.cat([v, v_d], dim=1), mask_joint


def depth_attention_mass(
    q: Tensor, k_joint: Tensor, mask_joint: Tensor, *, num_depth: int
) -> Tensor:
    """Mean softmax mass on the depth columns — the influence telemetry that
    replaces the retired gate metric. q: (B, Tq, H, Dh) post-q_norm queries,
    k_joint/mask_joint from join_depth_columns. Detached, eager; call on probe
    steps only."""
    with torch.no_grad():
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k_joint.float())
        scores = scores / math.sqrt(q.shape[-1]) + mask_joint.float()
        weights = scores.softmax(dim=-1)
        return weights[..., -num_depth:].sum(dim=-1).mean()


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


def slice_wrist_cam_kv(
    kv_states: list[tuple[Tensor, Tensor]],
    *,
    input_ids: Tensor,
    image_patch_id: int,
    num_images: int,
    cam_index: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Slice the depth camera's image-token K/V out of every layer's prefix KV (design §3.2).

    kv_states: length-L list of (key, value), each ``(B, T, d_vlm)``.
    Returns (wrist_keys, wrist_values), length-L lists of ``(B, T_cam, d_vlm)``.
    """
    sel = wrist_cam_token_indices(
        input_ids, image_patch_id=image_patch_id, num_images=num_images, cam_index=cam_index
    )
    wrist_keys, wrist_values = [], []
    for key, value in kv_states:
        wk, wv = gather_kv_at_indices(key, value, sel)
        wrist_keys.append(wk)
        wrist_values.append(wv)
    return wrist_keys, wrist_values


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


class DepthStream(nn.Module):
    """M co-evolving depth blocks + the per-layer action-read bridge (depth_bias, K/V proj).

    ``d_act`` is the action expert width (= ``num_action_heads · action_head_dim``);
    the read projections map the depth state (width d_d) into the action expert's head
    space so the joint softmax can attend depth columns next to the context columns.
    """

    def __init__(
        self,
        config: DepthPointmapConfig,
        *,
        d_vlm: int,
        num_action_heads: int,
        action_head_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.num_action_heads = num_action_heads
        self.action_head_dim = action_head_dim
        d_d = config.stream_width
        d_act = num_action_heads * action_head_dim

        self.blocks = nn.ModuleList(
            DepthStreamBlock(
                d_d=d_d,
                d_vlm=d_vlm,
                num_heads=config.stream_num_heads,
                mlp_ratio=config.stream_mlp_ratio,
            )
            for _ in range(num_layers)
        )
        # Shared d_d → d_act read projections (one set, reused at every layer, mirroring
        # the action expert's single shared context_k_proj / context_v_proj).
        self.read_k_proj = nn.Linear(d_d, d_act)
        self.read_v_proj = nn.Linear(d_d, d_act)

        # Per-layer learned score bias b_ℓ on the depth columns of the joint softmax
        # (depth_redesign_options.md §5.2). Init −2: initial depth mass ×e⁻² ≈ 0.135 —
        # a soft start that, being additive INSIDE the softmax, does not zero the
        # content gradients (the failure mode of the retired multiplicative α gate).
        self.depth_bias = nn.Parameter(torch.full((num_layers,), -2.0))

    def forward(
        self,
        init_tokens: Tensor,
        wrist_keys: list[Tensor],
        wrist_values: list[Tensor],
        cross_on: Tensor | None = None,
    ) -> list[Tensor]:
        """Co-evolve the depth tokens through the M blocks.

        init_tokens: (B, N, d_d) from the point-map encoder.
        wrist_keys / wrist_values: length-M lists, each (B, T_w, d_vlm) — the VLM's
        per-layer cached K / V sliced to the wrist-camera token span.
        cross_on: (B,) bool or None — per-row wrist-bridge switch (False for
        RGB-dropped samples; see DepthStreamBlock).

        Returns a length-M list of depth states (B, N, d_d); state ℓ (the output of
        block ℓ) is what the action expert's layer ℓ reads.
        """
        if len(wrist_keys) != self.num_layers or len(wrist_values) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} wrist-cam KV layers, got "
                f"{len(wrist_keys)} keys / {len(wrist_values)} values."
            )
        t = init_tokens
        states = []
        for block, wk, wv in zip(self.blocks, wrist_keys, wrist_values, strict=True):
            t = block(t, wk, wv, cross_on=cross_on)
            states.append(t)
        return states

    def read_kv(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Project a depth state (B, N, d_d) into action head space for the read.

        Returns (k, v), each (B, N, num_action_heads, action_head_dim) — the same
        head layout the action expert's context keys/values use. The caller applies
        the block's cross-attn k_norm to the keys (to match the context keys).
        """
        b, n, _ = state.shape
        k = self.read_k_proj(state).view(b, n, self.num_action_heads, self.action_head_dim)
        v = self.read_v_proj(state).view(b, n, self.num_action_heads, self.action_head_dim)
        return k, v
