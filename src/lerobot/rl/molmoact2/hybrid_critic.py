"""Critic-owned fusion trunk for the MolmoAct2 hybrid value critic."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class MolmoAct2Critic(nn.Module):
    """Distributional V(s) head over detached policy modality features.

    The policy owns RGB-temporal, depth-history, state-history, and text
    encoders. Their assembled prefix tokens are detached before this module.
    This critic owns only projection, multimodal fusion, and value prediction.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_value_bins = int(config.num_value_bins)
        self.num_critic_blocks = int(config.critic_llm_depth)
        self.input_hidden_size = int(config.critic_input_hidden_size)
        self.hidden_size = int(config.critic_hidden_size)
        self.max_tokens = int(config.critic_max_tokens)

        self.input_projection = nn.Linear(self.input_hidden_size, self.hidden_size)
        # Position zero is reserved for the critic's value token.
        self.position_embeddings = nn.Parameter(torch.empty(1, self.max_tokens + 1, self.hidden_size))
        self.value_token = nn.Parameter(torch.empty(1, 1, self.hidden_size))

        feedforward_size = int(round(self.hidden_size * float(config.critic_mlp_ratio)))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(config.critic_num_attention_heads),
            dim_feedforward=feedforward_size,
            dropout=float(config.critic_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_blocks = nn.ModuleList(copy.deepcopy(layer) for _ in range(self.num_critic_blocks))
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.distribution_head = nn.Linear(self.hidden_size, self.num_value_bins)

        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.value_token, std=0.02)
        nn.init.normal_(self.input_projection.weight, std=0.02)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.distribution_head.weight, std=0.02)
        nn.init.zeros_(self.distribution_head.bias)

        bin_centers = torch.linspace(
            float(config.value_support_min),
            float(config.value_support_max),
            self.num_value_bins,
        )
        self.register_buffer("bin_centers", bin_centers, persistent=False)
        bin_width = (float(config.value_support_max) - float(config.value_support_min)) / (
            self.num_value_bins - 1
        )
        self.hl_gauss_sigma = float(config.hl_gauss_sigma_ratio) * bin_width

    def forward(self, encoder_tokens: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """Fuse detached encoder tokens and predict a categorical value distribution."""
        if encoder_tokens.ndim != 3 or encoder_tokens.shape[-1] != self.input_hidden_size:
            raise ValueError(
                "critic expected encoder tokens shaped "
                f"[B, T, {self.input_hidden_size}], got {tuple(encoder_tokens.shape)}."
            )
        if attention_mask.shape != encoder_tokens.shape[:2]:
            raise ValueError(
                f"critic attention mask {tuple(attention_mask.shape)} does not match "
                f"tokens {tuple(encoder_tokens.shape[:2])}."
            )

        batch_size, seq_len, _ = encoder_tokens.shape
        if seq_len > self.max_tokens:
            raise ValueError(
                f"critic received {seq_len} tokens, exceeding critic_max_tokens={self.max_tokens}."
            )

        dtype = self.input_projection.weight.dtype
        hidden_states = self.input_projection(encoder_tokens.detach().to(dtype=dtype))
        value_token = self.value_token.expand(batch_size, -1, -1).to(dtype=dtype)
        hidden_states = torch.cat([value_token, hidden_states], dim=1)
        hidden_states = hidden_states + self.position_embeddings[:, : seq_len + 1].to(dtype=dtype)

        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        full_mask = torch.cat(
            [
                torch.ones(batch_size, 1, device=hidden_states.device, dtype=torch.bool),
                attention_mask,
            ],
            dim=1,
        )
        padding_mask = ~full_mask
        for block in self.fusion_blocks:
            hidden_states = block(hidden_states, src_key_padding_mask=padding_mask)

        value_state = self.final_norm(hidden_states[:, 0])
        logits = self.distribution_head(value_state)
        probs = functional.softmax(logits, dim=-1)
        value = self.value_from_probs(probs)
        return {"logits": logits, "probs": probs, "value": value}

    def value_from_probs(self, probs: Tensor) -> Tensor:
        """Expected value under an already normalized distribution."""
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        return (probs * bin_centers).sum(dim=-1, keepdim=True)

    def value_from_logits(self, logits: Tensor) -> Tensor:
        return self.value_from_probs(functional.softmax(logits, dim=-1))

    def hl_gauss_target(self, target_v: Tensor) -> Tensor:
        """HL-Gauss target distribution over value bins."""
        if target_v.ndim == 2:
            target_v = target_v.squeeze(-1)
        target_v = target_v.to(device=self.bin_centers.device, dtype=self.bin_centers.dtype)
        internal_edges = 0.5 * (self.bin_centers[:-1] + self.bin_centers[1:])
        z = (internal_edges.unsqueeze(0) - target_v.unsqueeze(-1)) / (self.hl_gauss_sigma * (2.0**0.5))
        cdf_internal = 0.5 * (1.0 + torch.erf(z))
        zeros = torch.zeros_like(cdf_internal[:, :1])
        ones = torch.ones_like(cdf_internal[:, :1])
        cdf_full = torch.cat([zeros, cdf_internal, ones], dim=-1)
        return cdf_full[:, 1:] - cdf_full[:, :-1]

    def one_hot_target(self, target_v: Tensor) -> Tensor:
        """Nearest-bin one-hot target for exact terminal values."""
        if target_v.ndim == 2:
            target_v = target_v.squeeze(-1)
        target_v = target_v.to(device=self.bin_centers.device, dtype=self.bin_centers.dtype)
        idx = torch.argmin(torch.abs(self.bin_centers.unsqueeze(0) - target_v.unsqueeze(-1)), dim=-1)
        return functional.one_hot(idx, num_classes=self.num_value_bins).to(self.bin_centers.dtype)
