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

Action-read bridge (the revised §3.3 read, decided 2026-06-14 — *additive*, not a
single joint softmax, because a joint softmax is not bit-identical at gate 0). For
action layer ℓ the action expert computes

    out = SDPA(q, K_ctx, V_ctx)  +  tanh(α_ℓ) · SDPA(q, [K_dℓ, k_⋆], [V_dℓ, 0])

where (K_dℓ, V_dℓ) are this module's per-layer depth state projected into the action
expert's head space (``read_kv``), k_⋆ is a zero sink key whose per-(layer,head) logit
``sink_logit[ℓ,h]`` is injected as an additive attention bias (absolute abstaining),
and α_ℓ is the per-layer zero-init gate (``tanh(0)=0`` ⇒ the depth term is bitwise
zero at init ⇒ frozen-policy bit-identity). The SDPA itself lives in the action
expert; this module owns the projections, α, and the sink logits. Fresh float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_pointmap import DepthPointmapConfig


def gated_depth_read(q: Tensor, depth_kv: tuple[Tensor, Tensor], sink_logit: Tensor) -> Tensor:
    """Additive depth read via one SDPA over [depth tokens, zero sink] (MoT §3.3, revised).

    A single static-shape SDPA (compile-friendly — regains CUDA graphs vs the A.3 manual
    einsum; SDPA applies its own 1/√d scale). The sink is one extra key column with a zero
    key (raw score 0), a zero value (contributes nothing), and a learned per-head logit
    injected as an additive attention bias — so when no depth token scores above the sink
    bar, attention parks on the sink and the read → 0 (absolute abstaining).

    q:          (B, Tq, H, Dh) post-q_norm action queries.
    depth_kv:   (K_d, V_d), each (B, N, H, Dh) — the layer's depth state in action head space.
    sink_logit: (H,) learned per-head sink logit.
    Returns the read content (B, Tq, H, Dh); the caller scales by tanh(α_ℓ) and adds it to
    the context read, so at α=0 the term is bitwise zero (frozen-policy bit-identity).
    """
    k_d, v_d = depth_kv
    b, n, h, dh = k_d.shape
    zero = k_d.new_zeros(b, 1, h, dh)
    k = torch.cat([k_d, zero], dim=1)  # (B, N+1, H, Dh)
    v = torch.cat([v_d, zero], dim=1)
    bias = k_d.new_zeros(h, n + 1)
    bias[:, n] = sink_logit.to(k_d.dtype)  # depth columns: 0; sink column: per-head logit
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=bias.view(1, h, 1, n + 1)
    )  # (B, H, Tq, Dh)
    return out.transpose(1, 2).contiguous()


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
    """One co-evolution block: depth self-attn + depth→wrist-cam cross-attn + MLP."""

    def __init__(self, *, d_d: int, d_vlm: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_d)
        self.self_attn = StreamAttention(d_q=d_d, d_kv=d_d, num_heads=num_heads)
        self.norm_cross = nn.LayerNorm(d_d)
        self.cross_attn = StreamAttention(d_q=d_d, d_kv=d_vlm, num_heads=num_heads)
        self.norm_mlp = nn.LayerNorm(d_d)
        self.mlp = StreamMLP(d_d, mlp_ratio)

    def forward(self, t: Tensor, wrist_k: Tensor, wrist_v: Tensor) -> Tensor:
        h = self.norm_self(t)
        t = t + self.self_attn(h, h, h)
        t = t + self.cross_attn(self.norm_cross(t), wrist_k, wrist_v)
        t = t + self.mlp(self.norm_mlp(t))
        return t


class DepthStream(nn.Module):
    """M co-evolving depth blocks + the per-layer action-read bridge (α, sink, K/V proj).

    ``d_act`` is the action expert width (= ``num_action_heads · action_head_dim``);
    the read projections map the depth state (width d_d) into the action expert's head
    space so the gated additive read can SDPA action queries against depth keys/values.
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

        # Per-layer zero-init gate α_ℓ (tanh(0)=0 ⇒ bit-identity) and per-(layer,head)
        # sink logit (zero ⇒ sink starts neutral; the read adds it as a bias column).
        self.gate = nn.Parameter(torch.zeros(num_layers))
        self.sink_logit = nn.Parameter(torch.zeros(num_layers, num_action_heads))

    def gate_value(self) -> Tensor:
        return torch.tanh(self.gate)  # (num_layers,)

    def forward(self, init_tokens: Tensor, wrist_keys: list[Tensor], wrist_values: list[Tensor]) -> list[Tensor]:
        """Co-evolve the depth tokens through the M blocks.

        init_tokens: (B, N, d_d) from the point-map encoder.
        wrist_keys / wrist_values: length-M lists, each (B, T_w, d_vlm) — the VLM's
        per-layer cached K / V sliced to the wrist-camera token span.

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
            t = block(t, wk, wv)
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
