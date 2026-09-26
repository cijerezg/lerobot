"""MolmoAct2 distributional critic: own frozen encoder + fusion transformer at native width."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class CriticEncoder(nn.Module):
    """Frozen copies of the policy backbone's token embedding and vision backbone.

    Emits the prefix tokens the LLM would read: text embeddings with the image features
    added on the <im_patch> positions (MolmoAct2 build_input_embeddings, minus the
    embedding dropout, which is 0 in this backbone). Copied at init and never trained,
    so the critic's input is a fixed function of the observation and prompt whatever
    the actor learns, and a critic checkpoint carries everything it needs.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.wte = copy.deepcopy(backbone.transformer.wte)
        self.vision_backbone = copy.deepcopy(backbone.vision_backbone)
        self.image_patch_id = int(backbone.config.image_patch_id)
        for param in self.parameters():
            param.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return int(self.wte.embedding.shape[-1])

    def forward(self, input_ids: Tensor, images: Tensor | None, token_pooling: Tensor | None) -> Tensor:
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        tokens = self.wte(input_ids)
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling).to(tokens.device)
            is_image_patch = input_ids.view(-1) == self.image_patch_id
            tokens.view(-1, tokens.shape[-1])[is_image_patch] += image_features
        return tokens


class CriticFusion(nn.Module):
    """Pre-norm transformer over the encoder tokens, read out through a learned value token.

    Runs at the encoder's width (no input projection), so no information is dropped
    before fusion. The target network is a copy of this module alone.
    """

    def __init__(self, config: Any, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_tokens = int(config.critic_max_tokens)

        # Position zero is reserved for the value token.
        self.position_embeddings = nn.Parameter(torch.empty(1, self.max_tokens + 1, self.hidden_size))
        self.value_token = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(config.critic_num_attention_heads),
            dim_feedforward=int(round(self.hidden_size * float(config.critic_mlp_ratio))),
            dropout=float(config.critic_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.ModuleList(copy.deepcopy(layer) for _ in range(int(config.critic_llm_depth)))
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.distribution_head = nn.Linear(self.hidden_size, int(config.num_value_bins))

        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.value_token, std=0.02)
        nn.init.normal_(self.distribution_head.weight, std=0.02)
        nn.init.zeros_(self.distribution_head.bias)

    def forward(self, tokens: Tensor, attention_mask: Tensor) -> Tensor:
        """Logits over the value bins, [B, num_value_bins]."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"critic expected tokens shaped [B, T, {self.hidden_size}], got {tuple(tokens.shape)}.")
        if attention_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"critic attention mask {tuple(attention_mask.shape)} does not match tokens {tuple(tokens.shape[:2])}."
            )
        batch_size, seq_len, _ = tokens.shape
        if seq_len > self.max_tokens:
            raise ValueError(f"critic received {seq_len} tokens, exceeding critic_max_tokens={self.max_tokens}.")

        dtype = self.value_token.dtype
        hidden_states = tokens.detach().to(dtype=dtype)
        value_token = self.value_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([value_token, hidden_states], dim=1)
        hidden_states = hidden_states + self.position_embeddings[:, : seq_len + 1]

        attention_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        keep = torch.cat([torch.ones(batch_size, 1, device=hidden_states.device, dtype=torch.bool), attention_mask], dim=1)
        padding_mask = ~keep
        for block in self.blocks:
            hidden_states = block(hidden_states, src_key_padding_mask=padding_mask)
        return self.distribution_head(self.final_norm(hidden_states[:, 0]))


class MolmoAct2Critic(nn.Module):
    """Distributional V(s): CriticEncoder (frozen) → CriticFusion → HL-Gauss value bins.

    ``forward(..., fusion=critic_target)`` runs the frozen target copy of the fusion stack
    on the same encoder tokens; the encoder is frozen, so critic and target share it.
    """

    def __init__(self, config: Any, backbone: nn.Module) -> None:
        super().__init__()
        self.num_value_bins = int(config.num_value_bins)
        self.encoder = CriticEncoder(backbone)
        self.fusion = CriticFusion(config, self.encoder.hidden_size)

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

    def forward(
        self,
        input_ids: Tensor,
        images: Tensor | None,
        token_pooling: Tensor | None,
        attention_mask: Tensor,
        fusion: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        fusion = self.fusion if fusion is None else fusion
        with torch.no_grad():
            tokens = self.encoder(input_ids, images, token_pooling)
        logits = fusion(tokens, attention_mask)
        probs = functional.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "value": self.value_from_probs(probs)}

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
