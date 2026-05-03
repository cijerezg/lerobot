#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained modeling for tinypi05v2.

This module is the single-file rewrite of the tinypi05 policy. It intentionally
avoids importing from `transformers`, `pistar06`, or `pi05` so the model layer
can be read top-to-bottom without bouncing between repos.

The module exposes three user-facing classes:

- `PaliGemmaWithExpertV2`: composite holding the SigLIP vision tower, the
  multi-modal projector, the Gemma-1-style language model (prefix), and the
  Gemma-1-style action expert (suffix), all built from scratch.
- `TinyPI05V2Pytorch`: flow-matching core wrapping `PaliGemmaWithExpertV2` and
  the action / time projections.
- `TinyPI05V2Policy`: `PreTrainedPolicy` front-end with `from_pretrained` that
  loads the existing tinypi05 `model.safetensors` checkpoint (ignoring the
  unused SigLIP MAP `head.*` keys).

Parameter names mirror the checkpoint exactly so the state dict loads with
`strict=True` after filtering the unused `vision_model.head.*` keys.
"""

from __future__ import annotations

import builtins
import inspect
import logging
import math
from collections import deque
from pathlib import Path
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.tinypi05v2.configuration_tinypi05v2 import TinyPI05V2Config
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    overlap_end: int | None
    num_flow_matching_steps: int | None


def _get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Pick a dtype that actually works on a given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize-and-pad mirroring the existing `resize_with_pad_torch` (exact copy).

    Accepts either `[*, H, W, C]` or `[*, C, H, W]` tensors; output has the same
    channel ordering as the input.
    """
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + rem_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=constant_value)

    if channels_last:
        padded = padded.permute(0, 2, 3, 1)
    return padded


def _pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Build 2D attention masks from padding and autoregressive masks (exact copy)."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
    """Turn a boolean `[B, N, N]` mask into an additive `[B, 1, N, N]` mask."""
    return torch.where(att_2d_masks[:, None, :, :], 0.0, OPENPI_ATTENTION_MASK_VALUE)


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> torch.Tensor:
    """Sine/cosine positional embedding for scalar positions (exact copy)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha_t, beta_t).sample((bsize,))


# ═══════════════════════════════════════════════════════════════════════════════
# SigLIP vision tower (self-contained, no transformers)
# ═══════════════════════════════════════════════════════════════════════════════


class _SiglipPatchEmbeddings(nn.Module):
    """SigLIP patch + positional embeddings (exact parameter names from the checkpoint)."""

    def __init__(self, hidden_size: int, image_size: int, patch_size: int, num_channels: int = 3):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.patch_embedding.weight.dtype
        x = self.patch_embedding(pixel_values.to(weight_dtype))
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding(self.position_ids)
        return x


class _SiglipMHA(nn.Module):
    """Plain multi-head self attention with the `{q,k,v,out}_proj` layout SigLIP uses."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        # tinypi05 delegates SigLIP to Transformers; the self-contained v2 tower
        # calls SDPA directly so CUDA can use fused attention kernels.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.out_proj(out)


class _SiglipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _SiglipEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.self_attn = _SiglipMHA(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = _SiglipMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _SiglipUnusedHead(nn.Module):
    """Placeholder parameters for the MAP head (`head.*`) present in the checkpoint.

    The tinypi05 checkpoint keeps SigLIP's multi-head attention pooling head even
    though the model only consumes `last_hidden_state`; including these as
    parameters lets us call `load_state_dict(strict=True)` without surgery.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # nn.MultiheadAttention's default state_dict uses the exact key names the
        # SigLIP checkpoint provides: `in_proj_weight`, `in_proj_bias`,
        # `out_proj.weight`, `out_proj.bias`.
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, bias=True, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = _SiglipMLP(hidden_size, intermediate_size)


class _SiglipVisionModelV2(nn.Module):
    """Inner SigLIP module living at `vision_tower.vision_model`."""

    def __init__(
        self,
        hidden_size: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.embeddings = _SiglipPatchEmbeddings(hidden_size, image_size, patch_size, 3)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [_SiglipEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.head = _SiglipUnusedHead(hidden_size, intermediate_size, num_heads)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


class _SiglipVisionTowerV2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vision_model = _SiglipVisionModelV2(**kwargs)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-modal projector
# ═══════════════════════════════════════════════════════════════════════════════


class _MultiModalProjectorV2(nn.Module):
    """PaliGemma multi-modal projector: `linear(vision_hidden -> text_hidden)`."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Gemma primitives (RMSNorm, AdaRMS, RoPE, attention, MLP, decoder layer)
# ═══════════════════════════════════════════════════════════════════════════════


class _GemmaRMSNormV2(nn.Module):
    """Gemma-1-style RMSNorm: `x / rms(x) * (1 + weight)`.

    `forward(x, cond=None)` returns `(out, None)` for a uniform call signature
    with the adaptive variant below. The `cond` argument is ignored here.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)
        normed = normed * (1.0 + self.weight.float())
        return normed.type_as(x).to(dtype), None


class _AdaRMSNormV2(nn.Module):
    """Adaptive RMSNorm with scale/shift/gate produced from a conditioning tensor.

    Matches the semantics of `PiGemmaRMSNorm` from `pistar06` (which tinypi05
    checkpoints are trained against):

        normed = rms(x) * (1 + scale) + shift, gate returned separately.

    `dense` produces `[scale | shift | gate]` along the last dim.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cond is None:
            raise ValueError("AdaRMSNormV2 requires a conditioning tensor")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        dtype = x.dtype
        normed = self._norm(x)
        modulation = self.dense(cond.to(dtype=self.dense.weight.dtype))
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq, head_dim)


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    # Gemma uses grouped-query attention. PyTorch 2.6's SDPA enable_gqa path
    # avoids this materialization but is slower on the current tinypi05v2
    # benchmark, so v2 keeps the tinypi05 eager layout and lets SDPA fuse the
    # attention math after K/V expansion.
    key_states = _repeat_kv(key, num_key_value_groups)
    value_states = _repeat_kv(value, num_key_value_groups)
    if query.device.type == "cuda":
        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]].to(dtype=query.dtype)
        # tinypi05 performs explicit matmul/softmax/matmul through Transformers'
        # eager attention. v2 keeps the same repeated-K/V tensor shape but calls
        # SDPA directly so CUDA can choose a fused attention kernel.
        attn_output = F.scaled_dot_product_attention(
            query,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scaling,
        )
    else:
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


class _GemmaRotaryV2(nn.Module):
    """Gemma rotary embeddings; precomputes `inv_freq`, returns `(cos, sin)`."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, reference: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = reference.device.type if isinstance(reference.device.type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.to(reference.device) @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(reference.dtype), sin.to(reference.dtype)


class _GemmaAttentionV2(nn.Module):
    """Gemma-style self-attention projections (no attention computation here)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)


class _GemmaMLPV2(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class _GemmaDecoderLayerV2(nn.Module):
    """Decoder layer with either plain RMSNorm or AdaRMSNorm; used by prefix and expert."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        *,
        use_adarms: bool,
        cond_dim: int | None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_adarms = use_adarms
        self.self_attn = _GemmaAttentionV2(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = _GemmaMLPV2(hidden_size, intermediate_size)
        if use_adarms:
            if cond_dim is None:
                raise ValueError("AdaRMSNormV2 requires cond_dim when use_adarms=True")
            self.input_layernorm = _AdaRMSNormV2(hidden_size, cond_dim)
            self.post_attention_layernorm = _AdaRMSNormV2(hidden_size, cond_dim)
        else:
            self.input_layernorm = _GemmaRMSNormV2(hidden_size)
            self.post_attention_layernorm = _GemmaRMSNormV2(hidden_size)


class _GemmaModelV2(nn.Module):
    """Gemma-1-style model used for prefix-only forward with a simple KV cache.

    The model is intentionally minimal — the joint-attention forward is done by
    `_compute_joint_layer`; this class is only used when `inputs_embeds[1] is
    None` (i.e. to populate the prefix KV cache during inference).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        vocab_size: int,
        *,
        use_adarms: bool,
        cond_dim: int | None,
        has_embed_tokens: bool,
    ):
        super().__init__()
        if has_embed_tokens:
            # Plain Embedding — sqrt(hidden_size) scaling is applied externally.
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                _GemmaDecoderLayerV2(
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_size,
                    use_adarms=use_adarms,
                    cond_dim=cond_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = (
            _AdaRMSNormV2(hidden_size, cond_dim) if use_adarms else _GemmaRMSNormV2(hidden_size)
        )
        self.rotary_emb = _GemmaRotaryV2(head_dim)
        self.gradient_checkpointing = False


# ═══════════════════════════════════════════════════════════════════════════════
# Joint-attention kernel
# ═══════════════════════════════════════════════════════════════════════════════


def _layernorm_call(
    layernorm: nn.Module, x: torch.Tensor, cond: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch to the right `forward` signature for plain vs adaptive RMSNorm."""
    return layernorm(x, cond=cond)


def _compute_joint_layer(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    adarms_cond: list[torch.Tensor | None],
    prefix_model: _GemmaModelV2,
    suffix_model: _GemmaModelV2,
    rotary_emb: _GemmaRotaryV2,
) -> list[torch.Tensor]:
    """Joint-attention kernel spanning the prefix (VLM) and suffix (expert) streams.

    This is a port of `pistar06.modeling_pistar06.compute_layer_complete` with
    all `transformers.models.gemma` symbols replaced by the inlined ones above.
    The math is identical so RNG-deterministic outputs stay bit-equivalent.
    """
    models = [prefix_model, suffix_model]
    query_states: list[torch.Tensor] = []
    key_states: list[torch.Tensor] = []
    value_states: list[torch.Tensor] = []
    gates: list[torch.Tensor | None] = []

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = _layernorm_call(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        head_dim = layer.self_attn.head_dim
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(q)
        key_states.append(k)
        value_states.append(v)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    reference = torch.empty(1, device=query_states.device, dtype=query_states.dtype)
    cos, sin = rotary_emb(reference, position_ids)
    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

    prefix_layer_attn = prefix_model.layers[layer_idx].self_attn
    scaling = prefix_layer_attn.scaling
    num_key_value_groups = prefix_layer_attn.num_key_value_groups
    att_output = _eager_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
        num_key_value_groups,
    )
    batch_size = query_states.shape[0]
    head_dim = prefix_layer_attn.head_dim
    num_heads = prefix_layer_attn.num_heads
    att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)

    outputs_embeds: list[torch.Tensor] = []
    start = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end = start + hidden_states.shape[1]
        slice_ = att_output[:, start:end]
        if slice_.dtype != layer.self_attn.o_proj.weight.dtype:
            slice_ = slice_.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(slice_)
        out = _gated_residual(hidden_states, out, gates[i])

        residual = out
        out, gate = _layernorm_call(layer.post_attention_layernorm, out, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)
        out = layer.mlp(out)
        out = _gated_residual(residual, out, gate)
        outputs_embeds.append(out)
        start = end
    return outputs_embeds


# ═══════════════════════════════════════════════════════════════════════════════
# PaliGemma + action expert composite
# ═══════════════════════════════════════════════════════════════════════════════


class _PaliGemmaInnerV2(nn.Module):
    """Holds `vision_tower`, `multi_modal_projector`, and `language_model`.

    The nesting (`paligemma.model.{vision_tower, multi_modal_projector,
    language_model}`) mirrors the tinypi05 checkpoint paths exactly.
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        vision_intermediate_size: int,
        vision_num_hidden_layers: int,
        vision_num_attention_heads: int,
        vision_patch_size: int,
        image_size: int,
        text_hidden_size: int,
        text_num_layers: int,
        text_num_heads: int,
        text_num_kv_heads: int,
        text_head_dim: int,
        text_intermediate_size: int,
        text_vocab_size: int,
    ):
        super().__init__()
        self.vision_tower = _SiglipVisionTowerV2(
            hidden_size=vision_hidden_size,
            image_size=image_size,
            patch_size=vision_patch_size,
            num_layers=vision_num_hidden_layers,
            num_heads=vision_num_attention_heads,
            intermediate_size=vision_intermediate_size,
        )
        self.multi_modal_projector = _MultiModalProjectorV2(vision_hidden_size, text_hidden_size)
        self.language_model = _GemmaModelV2(
            hidden_size=text_hidden_size,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            num_kv_heads=text_num_kv_heads,
            head_dim=text_head_dim,
            intermediate_size=text_intermediate_size,
            vocab_size=text_vocab_size,
            use_adarms=False,
            cond_dim=None,
            has_embed_tokens=True,
        )


class _PaliGemmaDirectV2(nn.Module):
    """Outer wrapper so state dict keys start with `paligemma.model.*`."""

    def __init__(self, **inner_kwargs):
        super().__init__()
        self.model = _PaliGemmaInnerV2(**inner_kwargs)


class _ActionExpertDirectV2(nn.Module):
    """Outer wrapper so state dict keys start with `gemma_expert.model.*`."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        cond_dim: int,
    ):
        super().__init__()
        self.model = _GemmaModelV2(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=0,
            use_adarms=True,
            cond_dim=cond_dim,
            has_embed_tokens=False,
        )


class PaliGemmaWithExpertV2(nn.Module):
    """Self-contained PaliGemma + Gemma action expert.

    This class owns exactly the parameters the tinypi05 checkpoint contains
    (modulo the unused `vision_model.head.*` MAP head, kept for strict loading).
    The `.paligemma.model.{vision_tower, multi_modal_projector, language_model}`
    and `.gemma_expert.model.*` attribute paths are preserved so the saved
    state_dict loads without remapping.
    """

    def __init__(
        self,
        *,
        vlm_width: int,
        vlm_depth: int,
        vlm_mlp_dim: int,
        vlm_num_heads: int,
        vlm_num_kv_heads: int,
        vlm_head_dim: int,
        expert_width: int,
        expert_depth: int,
        expert_mlp_dim: int,
        expert_num_heads: int,
        expert_num_kv_heads: int,
        expert_head_dim: int,
        vision_hidden_size: int,
        vision_intermediate_size: int,
        vision_num_hidden_layers: int,
        vision_num_attention_heads: int,
        vision_patch_size: int,
        vocab_size: int,
        image_size: int,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        super().__init__()
        if vlm_depth != expert_depth:
            raise ValueError("tinypi05v2 requires vlm_depth == expert_depth for the joint kernel")
        if vlm_num_heads != expert_num_heads:
            raise ValueError("tinypi05v2 requires vlm_num_heads == expert_num_heads for the joint kernel")

        self.precision = precision
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.text_hidden_size = vlm_width

        self.paligemma = _PaliGemmaDirectV2(
            vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            vision_patch_size=vision_patch_size,
            image_size=image_size,
            text_hidden_size=vlm_width,
            text_num_layers=vlm_depth,
            text_num_heads=vlm_num_heads,
            text_num_kv_heads=vlm_num_kv_heads,
            text_head_dim=vlm_head_dim,
            text_intermediate_size=vlm_mlp_dim,
            text_vocab_size=vocab_size,
        )
        self.gemma_expert = _ActionExpertDirectV2(
            hidden_size=expert_width,
            num_layers=expert_depth,
            num_heads=expert_num_heads,
            num_kv_heads=expert_num_kv_heads,
            head_dim=expert_head_dim,
            intermediate_size=expert_mlp_dim,
            cond_dim=expert_width,
        )

        # Independent rotary buffer shared by both streams (same head_dim on both).
        # The joint kernel uses this for position embeddings across concatenated
        # query/key tensors.
        self._apply_dtype_policy(precision)
        self._set_requires_grad()

    def _apply_dtype_policy(self, precision: Literal["bfloat16", "float32"]) -> None:
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        self.to(dtype=torch.bfloat16)
        # Parameters that must remain float32 (vision tower, projector, RMSNorms,
        # and the final model.norm). Mirrors the existing `pistar06`
        # `params_to_keep_float32` policy so per-parameter dtypes match the
        # tinypi05 checkpoint exactly.
        params_to_keep_float32 = (
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()
        return self

    # ── Embedding helpers ────────────────────────────────────────────────

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run vision tower + projector, then scale by sqrt(text_hidden_size).

        Matches `PaliGemmaWithExpertModel.embed_image` from pistar06 bit-for-bit:
        casts the input to float32 for the vision pass, applies the scale, and
        casts back to the caller's dtype.
        """
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        hidden = self.paligemma.model.vision_tower(image)
        features = self.paligemma.model.multi_modal_projector(hidden)
        features = features * (self.text_hidden_size**0.5)
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Plain `embed_tokens` lookup; sqrt scaling is applied externally."""
        return self.paligemma.model.language_model.embed_tokens(tokens)

    # ── Forward ──────────────────────────────────────────────────────────

    def _prefix_only_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
        use_cache: bool,
        adarms_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Run the prefix (VLM) Gemma stack alone; optionally populate the KV cache.

        The cache is a plain `list[(k, v)]` — one entry per layer — so the
        suffix/joint path can read it without a `transformers.DynamicCache`.
        """
        model = self.paligemma.model.language_model
        hidden = inputs_embeds
        if len(model.layers) > 0 and model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)

        reference = torch.empty(1, device=hidden.device, dtype=hidden.dtype)
        cos, sin = model.rotary_emb(reference, position_ids)

        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None
        for layer in model.layers:
            residual = hidden
            hidden, _ = _layernorm_call(layer.input_layernorm, hidden, adarms_cond)
            input_shape = hidden.shape[:-1]
            head_dim = layer.self_attn.head_dim
            hidden_shape = (*input_shape, -1, head_dim)
            q = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            if use_cache:
                new_cache.append((k, v))

            att_output = _eager_attention(
                q, k, v, attention_mask, layer.self_attn.scaling, layer.self_attn.num_key_value_groups
            )
            att_output = att_output.reshape(att_output.shape[0], att_output.shape[1], -1)
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(att_output)
            out = _gated_residual(residual, out, None)

            residual = out
            out, _ = _layernorm_call(layer.post_attention_layernorm, out, adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out = out.to(torch.bfloat16)
            out = layer.mlp(out)
            hidden = _gated_residual(residual, out, None)

        hidden, _ = _layernorm_call(model.norm, hidden, adarms_cond)
        return hidden, new_cache

    def _suffix_with_cache_forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the suffix (action expert) against a prefix KV cache (denoise step)."""
        prefix_model = self.paligemma.model.language_model
        suffix_model = self.gemma_expert.model

        hidden = inputs_embeds
        if (
            len(suffix_model.layers) > 0
            and suffix_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden = hidden.to(torch.bfloat16)

        reference = torch.empty(1, device=hidden.device, dtype=hidden.dtype)
        cos, sin = suffix_model.rotary_emb(reference, position_ids)

        num_layers = len(suffix_model.layers)
        for layer_idx in range(num_layers):
            layer = suffix_model.layers[layer_idx]
            residual = hidden
            hidden, gate = _layernorm_call(layer.input_layernorm, hidden, adarms_cond)
            head_dim = layer.self_attn.head_dim
            input_shape = hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, head_dim)
            q = layer.self_attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            # Concatenate prefix-cached K/V (already rotated) with the new
            # suffix K/V. Cache tensors are stored as (B, num_kv_heads, Nprefix,
            # head_dim). This reproduces the behaviour of the transformers
            # DynamicCache path without depending on it.
            prefix_k, prefix_v = past_key_values[layer_idx]
            k_full = torch.cat([prefix_k.to(k.dtype), k], dim=2)
            v_full = torch.cat([prefix_v.to(v.dtype), v], dim=2)

            att_output = _eager_attention(
                q,
                k_full,
                v_full,
                attention_mask,
                layer.self_attn.scaling,
                layer.self_attn.num_key_value_groups,
            )
            att_output = att_output.reshape(att_output.shape[0], att_output.shape[1], -1)
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(att_output)
            out = _gated_residual(residual, out, gate)

            residual2 = out
            out, gate2 = _layernorm_call(layer.post_attention_layernorm, out, adarms_cond)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out = out.to(torch.bfloat16)
            out = layer.mlp(out)
            hidden = _gated_residual(residual2, out, gate2)
            # Avoid stale binding warning from earlier prefix_model reference.
            _ = prefix_model

        hidden, _ = _layernorm_call(suffix_model.norm, hidden, adarms_cond)
        return hidden

    def forward(
        self,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Three branches: prefix-only / suffix-only (with cache) / joint forward."""
        if adarms_cond is None:
            adarms_cond = [None, None]

        if inputs_embeds[1] is None:
            prefix_out, prefix_cache = self._prefix_only_forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            return [prefix_out, None], prefix_cache

        if inputs_embeds[0] is None:
            if past_key_values is None:
                raise ValueError("suffix-only forward requires a pre-populated prefix KV cache")
            suffix_out = self._suffix_with_cache_forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                adarms_cond=adarms_cond[1],
            )
            return [None, suffix_out], None

        # Joint forward (training): prefix and suffix processed side-by-side at
        # every layer via the joint-attention kernel.
        prefix_model = self.paligemma.model.language_model
        suffix_model = self.gemma_expert.model

        current_embeds = list(inputs_embeds)
        if prefix_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            current_embeds = [e.to(torch.bfloat16) if e is not None else None for e in current_embeds]

        num_layers = len(prefix_model.layers)
        for layer_idx in range(num_layers):
            current_embeds = _compute_joint_layer(
                layer_idx,
                current_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                prefix_model=prefix_model,
                suffix_model=suffix_model,
                rotary_emb=prefix_model.rotary_emb,
            )

        prefix_out, _ = _layernorm_call(prefix_model.norm, current_embeds[0], adarms_cond[0])
        suffix_out, _ = _layernorm_call(suffix_model.norm, current_embeds[1], adarms_cond[1])
        return [prefix_out, suffix_out], None


# ═══════════════════════════════════════════════════════════════════════════════
# TinyPI05V2 PyTorch core (flow-matching)
# ═══════════════════════════════════════════════════════════════════════════════


class TinyPI05V2Pytorch(nn.Module):
    """Flow-matching PyTorch core. Faithful 1:1 port of `PiStar06Pytorch`."""

    def __init__(self, config: TinyPI05V2Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        arch = config.resolved_architecture()
        if arch.vlm_depth != arch.expert_depth:
            raise ValueError(
                "tinypi05v2 requires vlm_depth == expert_depth because the joint-attention "
                "kernel steps prefix and suffix layers in lockstep."
            )

        # Vocab size for the language embedding lookup. Matches the checkpoint
        # (Gemma-3-270m: 262144). We never hit the PaliGemma 257152 default
        # here because `pretrained_language_embeddings` is always set for the
        # tinypi05 checkpoints this module targets.
        if config.pretrained_language_embeddings is None:
            # Keep the PaliGemma default so unrelated callers still work.
            vocab_size = 257152
        else:
            vocab_size = 262144

        vision_hidden_size = arch.vision_hidden_size
        vision_intermediate_size = arch.vision_intermediate_size
        vision_num_hidden_layers = arch.vision_num_hidden_layers
        vision_num_attention_heads = arch.vision_num_attention_heads
        vision_patch_size = arch.vision_patch_size
        if config.pretrained_vision_model is not None:
            if config.pretrained_vision_model != "google/siglip-base-patch16-224":
                raise ValueError(
                    "tinypi05v2 can inline only google/siglip-base-patch16-224; "
                    f"got pretrained_vision_model={config.pretrained_vision_model!r}."
                )
            vision_hidden_size = 768
            vision_intermediate_size = 3072
            vision_num_hidden_layers = 12
            vision_num_attention_heads = 12
            vision_patch_size = 16

        self.paligemma_with_expert = PaliGemmaWithExpertV2(
            vlm_width=arch.vlm_width,
            vlm_depth=arch.vlm_depth,
            vlm_mlp_dim=arch.vlm_mlp_dim,
            vlm_num_heads=arch.vlm_num_heads,
            vlm_num_kv_heads=arch.vlm_num_kv_heads,
            vlm_head_dim=arch.vlm_head_dim,
            expert_width=arch.expert_width,
            expert_depth=arch.expert_depth,
            expert_mlp_dim=arch.expert_mlp_dim,
            expert_num_heads=arch.expert_num_heads,
            expert_num_kv_heads=arch.expert_num_kv_heads,
            expert_head_dim=arch.expert_head_dim,
            vision_hidden_size=vision_hidden_size,
            vision_intermediate_size=vision_intermediate_size,
            vision_num_hidden_layers=vision_num_hidden_layers,
            vision_num_attention_heads=vision_num_attention_heads,
            vision_patch_size=vision_patch_size,
            vocab_size=vocab_size,
            image_size=config.image_resolution[0],
            precision=config.dtype,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        # Action projections + time MLP — always float32 (matches checkpoint).
        self.action_in_proj = nn.Linear(config.max_action_dim, arch.expert_width)
        self.action_out_proj = nn.Linear(arch.expert_width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(arch.expert_width, arch.expert_width)
        self.time_mlp_out = nn.Linear(arch.expert_width, arch.expert_width)

        self.gradient_checkpointing_enabled = False

    # ── Gradient checkpointing toggles ───────────────────────────────────

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # ── RNG helpers ──────────────────────────────────────────────────────

    def sample_noise(self, shape, device) -> torch.Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device) -> torch.Tensor:
        time_beta = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # ── Prefix / suffix embed ────────────────────────────────────────────

    def embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with the embed_tokens lookup."""
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=True):
            vision_params = list(
                self.paligemma_with_expert.paligemma.model.vision_tower.parameters()
            )
            vision_frozen = not any(p.requires_grad for p in vision_params)
            if vision_frozen:
                with torch.no_grad():
                    img_emb = self.paligemma_with_expert.embed_image(img)
            else:
                img_emb = self._apply_checkpoint(
                    self.paligemma_with_expert.embed_image, img
                )
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens_inner):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens_inner)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs_cat = torch.cat(embs, dim=1)
        pad_masks_cat = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks_cat.device)
        att_masks_t = att_masks_t[None, :].expand(pad_masks_cat.shape[0], -1)
        return embs_cat, pad_masks_cat, att_masks_t

    def _embed_suffix_embs_and_cond(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)

        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.to(dtype=model_dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        def time_mlp_func(time_emb_inner):
            x = self.time_mlp_in(time_emb_inner)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        return action_time_emb, adarms_cond

    def embed_suffix(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed noisy actions and the flow-matching timestep."""
        action_time_emb, adarms_cond = self._embed_suffix_embs_and_cond(noisy_actions, timestep)

        bsize, seq_len = action_time_emb.shape[:2]
        pad = torch.ones(bsize, seq_len, dtype=torch.bool, device=timestep.device)
        att = [1] + ([0] * (self.config.chunk_size - 1))
        att_t = torch.tensor(att, dtype=action_time_emb.dtype, device=action_time_emb.device)
        att_t = att_t[None, :].expand(bsize, -1)
        return action_time_emb, pad, att_t, adarms_cond

    # ── Forward / sample / denoise ───────────────────────────────────────

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        prefix_layer = self.paligemma_with_expert.paligemma.model.language_model.layers[0]
        if prefix_layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_4d = _prepare_attention_masks_4d(att_2d)

        def forward_func(prefix_embs, suffix_embs, att_2d_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]

        def action_out_proj_func(suffix_out_inner):
            return self.action_out_proj(suffix_out_inner.to(dtype=self.action_out_proj.weight.dtype))

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t.float(), v_t.float(), reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_4d = _prepare_attention_masks_4d(prefix_att_2d)
        if prefix_att_2d_4d.device.type == "cuda":
            # v2 feeds additive masks straight to SDPA; pre-casting once avoids
            # the per-layer dtype conversions the eager tinypi05 path pays.
            prefix_attention_dtype = (
                self.paligemma_with_expert.paligemma.model.language_model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
            )
            prefix_att_2d_4d = prefix_att_2d_4d.to(dtype=prefix_attention_dtype)

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # The tinypi05 baseline rebuilds suffix masks/positions inside every
        # denoise_step call. In v2 they are invariant across flow steps, so keep
        # one precomputed copy and pass it through the denoise loop.
        suffix_pad_masks = torch.ones(
            bsize, self.config.chunk_size, dtype=torch.bool, device=device
        )
        suffix_att_masks = torch.zeros(
            bsize, self.config.chunk_size, dtype=prefix_embs.dtype, device=device
        )
        suffix_att_masks[:, 0] = 1
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        denoise_attention_mask = _prepare_attention_masks_4d(
            torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        )
        if denoise_attention_mask.device.type == "cuda":
            # Match the attention dtype before entering the per-step/per-layer
            # loop; PyTorch 2.6 SDPA requires additive masks to match query dtype.
            denoise_attention_dtype = (
                self.paligemma_with_expert.gemma_expert.model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
            )
            denoise_attention_mask = denoise_attention_mask.to(dtype=denoise_attention_dtype)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        denoise_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    attention_mask=denoise_attention_mask,
                    position_ids=denoise_position_ids,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                overlap_end = kwargs.get("overlap_end", kwargs.get("execution_horizon"))
                num_flow_matching_steps = kwargs.get(
                    "num_flow_matching_steps",
                    getattr(self.config, "num_inference_steps", None),
                )

                denoise_kwargs: dict = {
                    "x_t": x_t,
                    "prev_chunk_left_over": prev_chunk_left_over,
                    "inference_delay": inference_delay,
                    "time": time,
                    "original_denoise_step_partial": denoise_step_partial_call,
                    "execution_horizon": overlap_end,
                }
                try:
                    sig = inspect.signature(self.rtc_processor.denoise_step)
                    if "num_flow_matching_steps" in sig.parameters:
                        denoise_kwargs["num_flow_matching_steps"] = num_flow_matching_steps
                except (TypeError, ValueError):
                    pass

                v_t = self.rtc_processor.denoise_step(**denoise_kwargs)
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Tensor:
        suffix_embs, adarms_cond = self._embed_suffix_embs_and_cond(x_t, timestep)

        if attention_mask is None or position_ids is None:
            # Direct callers, including parity tests, can still use denoise_step
            # standalone. sample_actions passes precomputed values on the hot path.
            suffix_len = suffix_embs.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            suffix_pad_masks = torch.ones(
                batch_size, suffix_len, dtype=torch.bool, device=timestep.device
            )
            suffix_att_masks = torch.zeros(
                batch_size, suffix_len, dtype=suffix_embs.dtype, device=suffix_embs.device
            )
            suffix_att_masks[:, 0] = 1
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
            suffix_att_2d = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            attention_mask = _prepare_attention_masks_4d(full_att_2d)

        # The suffix path only reads the prefix KV cache. Keeping the list
        # by reference avoids a full per-step tensor deepcopy.
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs[1][:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out).float()


# ═══════════════════════════════════════════════════════════════════════════════
# TinyPI05V2 Policy
# ═══════════════════════════════════════════════════════════════════════════════


def _format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


class TinyPI05V2Policy(PreTrainedPolicy):
    """Self-contained TinyPI0.5 policy — same behavior as tinypi05, no transformers dep."""

    config_class = TinyPI05V2Config
    name = "tinypi05v2"

    def __init__(self, config: TinyPI05V2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = TinyPI05V2Pytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    # ── Pretrained loading ───────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model = cls(config, **kwargs)

        # Resolve a path to `model.safetensors`. Accepts either a local
        # directory (common: `.../pretrained_model/`) or a HuggingFace repo id.
        local_path = Path(pretrained_name_or_path)
        candidate = local_path / "model.safetensors"
        if candidate.is_file():
            safetensors_path = str(candidate)
        else:
            from huggingface_hub import hf_hub_download

            safetensors_path = hf_hub_download(
                repo_id=str(pretrained_name_or_path),
                filename="model.safetensors",
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
            )

        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path)

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            logging.warning(
                "TinyPI05V2 load_state_dict: missing=%d unexpected=%d (first 5 each: missing=%s, unexpected=%s)",
                len(missing),
                len(unexpected),
                missing[:5],
                unexpected[:5],
            )
        return model

    # ── PreTrainedPolicy API ─────────────────────────────────────────────

    def reset(self) -> None:
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def init_rtc_processor(self) -> None:
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Resize/normalize images to SigLIP's expected `[-1, 1]` range."""
        images: list[Tensor] = []
        img_masks: list[Tensor] = []
        device = next(self.parameters()).device

        present = [k for k in self.config.image_features if k in batch]
        missing = [k for k in self.config.image_features if k not in batch]
        if len(present) == 0:
            raise ValueError(
                f"All image features are missing from the batch. (batch={batch.keys()}, "
                f"image_features={self.config.image_features})"
            )

        for key in present:
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            channels_first = img.shape[1] == 3
            if channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = _resize_with_pad(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))

        for _ in range(len(missing)):
            img = torch.ones_like(images[-1]) * -1
            mask = torch.zeros_like(img_masks[-1])
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        return _pad_vector(batch[ACTION], self.config.max_action_dim)

    def get_optim_params(self) -> list[dict]:
        vision_prefix = "model.paligemma_with_expert.paligemma.model.vision_tower"
        embed_prefix = (
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens"
        )

        use_vision_lr = (
            self.config.optimizer_lr_vision is not None
            and not self.config.freeze_vision_encoder
            and not self.config.train_expert_only
        )
        use_embed_lr = (
            self.config.optimizer_lr_language_embeddings is not None
            and self.config.pretrained_language_embeddings is not None
            and not self.config.train_expert_only
        )

        if not use_vision_lr and not use_embed_lr:
            return list(self.parameters())

        vision_params: list[nn.Parameter] = []
        embed_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if use_embed_lr and name.startswith(embed_prefix):
                embed_params.append(param)
            elif use_vision_lr and name.startswith(vision_prefix):
                vision_params.append(param)
            else:
                other_params.append(param)
        groups: list[dict] = [{"params": other_params}]
        if use_vision_lr and vision_params:
            groups.append({"params": vision_params, "lr": self.config.optimizer_lr_vision})
        if use_embed_lr and embed_params:
            groups.append(
                {"params": embed_params, "lr": self.config.optimizer_lr_language_embeddings}
            )
        return groups

    def _get_default_peft_targets(self) -> dict[str, any]:
        common_projections = "action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
        target_modules = (
            rf"(model\.paligemma_with_expert\.gemma_expert\..*\.self_attn\.(q|v)_proj|"
            rf"model\.({common_projections}))"
        )
        return {"target_modules": target_modules, "modules_to_save": []}

    # ── Training / inference forward ─────────────────────────────────────

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        timestep_loss_weights: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict: dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if timestep_loss_weights is not None:
            if reduction == "none":
                raise ValueError(
                    "timestep_loss_weights cannot be combined with reduction='none'."
                )
            if timestep_loss_weights.dim() != 2:
                raise ValueError(
                    f"timestep_loss_weights must be 2-D; got {tuple(timestep_loss_weights.shape)}"
                )
            weights = timestep_loss_weights.to(dtype=losses.dtype, device=losses.device)
            weights_expanded = weights.unsqueeze(-1).expand_as(losses)
            denom = weights_expanded.sum().clamp_min(1e-8)
            loss = (losses * weights_expanded).sum() / denom
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]
