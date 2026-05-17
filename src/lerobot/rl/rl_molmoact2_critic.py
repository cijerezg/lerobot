"""
MolmoAct2Critic — distributional value critic following the PI05 recipe.

Architecture:
  - Full deepcopy of the actor's vision_backbone (ViT + adapter).  The adapter
    projects ViT features to text_hidden_size (2560) and splices them into the
    token sequence at image-patch positions.
  - First `critic_llm_depth` text-transformer blocks deepcopied from the actor.
  - Learnable value_queries [1, num_value_bins, D] — one query per bin.
  - Shared bin_logit_head Linear(D → 1).

Forward signature:
    forward(inputs_embeds, attention_mask) → {logits, probs, value}

Where inputs_embeds are built by embed_observation():
    - Text token embeddings from actor's wte (detached).
    - Vision features from the critic's own vision_backbone spliced in at
      image-patch positions — same logic as the actor's build_input_embeddings.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MolmoAct2Critic(nn.Module):
    """
    Distributional value critic for MolmoAct2RLPolicy.

    Instantiated by MolmoAct2RLPolicy.init_critic() which deepcopies the
    relevant parts of the loaded HF backbone.  At construction time the
    backbone components are None; call initialize_weights_from_backbone()
    before any forward pass.
    """

    TEXT_HIDDEN_SIZE = 2560  # molmoact2_text hidden_size

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_value_bins: int = int(config.num_value_bins)
        self.num_critic_blocks: int = int(config.critic_llm_depth)
        self.compute_dtype = torch.bfloat16 if config.model_dtype == "bfloat16" else torch.float32

        D = self.TEXT_HIDDEN_SIZE

        # Learnable distributional value queries (one per bin).
        # BERT-style small init — many queries, collapse risk is real at std=1.
        self.value_queries = nn.Parameter(torch.randn(1, self.num_value_bins, D) * 0.02)
        self.value_queries.register_hook(lambda g: g.contiguous())

        # Shared per-query logit head: D → 1 scalar per query → num_value_bins logits.
        self.bin_logit_head = nn.Linear(D, 1)

        # Value support (registered as non-persistent buffers so they move with .to()).
        bin_centers = torch.linspace(
            float(config.value_support_min),
            float(config.value_support_max),
            self.num_value_bins,
        )
        self.register_buffer("bin_centers", bin_centers, persistent=False)
        bin_width = (config.value_support_max - config.value_support_min) / (self.num_value_bins - 1)
        self.hl_gauss_sigma: float = float(config.hl_gauss_sigma_ratio) * float(bin_width)

        # Backbone components — populated by initialize_weights_from_backbone().
        self.vision_backbone: nn.Module | None = None
        self.transformer_blocks: nn.ModuleList | None = None
        self.rotary_emb: nn.Module | None = None
        self.ln_f: nn.Module | None = None

    # ── Weight initialisation ─────────────────────────────────────────────────

    def initialize_weights_from_backbone(self, backbone: Any) -> None:
        """
        Deepcopy the relevant actor-backbone components into the critic.

        backbone: the object returned by policy._backbone()
            i.e. policy._hf_model().model
        """
        transformer = backbone.transformer

        # Full vision backbone (ViT + adapter) — same depth as actor, no truncation.
        self.vision_backbone = copy.deepcopy(backbone.vision_backbone)

        # First N text-transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [copy.deepcopy(transformer.blocks[i]) for i in range(self.num_critic_blocks)]
        )

        # Rotary embeddings and final norm.
        self.rotary_emb = copy.deepcopy(transformer.rotary_emb)
        self.ln_f = copy.deepcopy(transformer.ln_f)

    # ── Observation embedding ─────────────────────────────────────────────────

    def embed_observation(
        self,
        backbone: Any,
        input_ids: Tensor,
        pixel_values: Tensor | None,
        image_token_pooling: Tensor | None = None,
        image_grids: Any = None,
        image_num_crops: Any = None,
    ) -> Tensor:
        """
        Build [B, seq_len, D] embeddings for the critic's transformer.

        - Text part: actor's wte (word-token embeddings), detached.
        - Vision part: critic's own vision_backbone, spliced into the text
          sequence at image-patch positions (same logic as the actor's
          build_input_embeddings monkeypatch).
        """
        # Text embeddings — detached; the critic does not back-prop into actor wte.
        input_ids_safe = input_ids * (input_ids != -1).to(input_ids.dtype)
        text_embeds: Tensor = backbone.transformer.wte(input_ids_safe).detach().clone()

        if pixel_values is None or self.vision_backbone is None:
            return text_embeds

        # Image preprocessing (actor's merge_visual_inputs; no grad needed here).
        merge_visual = getattr(backbone, "merge_visual_inputs", None)
        if not callable(merge_visual):
            return text_embeds

        with torch.no_grad():
            images, token_pooling = merge_visual(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                pixel_values_videos=None,
                video_token_pooling=None,
                video_grids=None,
            )

        if images is None:
            return text_embeds

        # Critic's own vision features — gradients flow through here.
        vision_features: Tensor = self.vision_backbone(images, token_pooling).to(text_embeds.device)

        # Splice vision into text at image-patch token positions.
        image_patch_id = int(backbone.config.image_patch_id)
        is_image_patch = input_ids.reshape(-1) == image_patch_id
        flat = text_embeds.reshape(-1, text_embeds.shape[-1]).clone()
        flat[is_image_patch] = flat[is_image_patch] + vision_features.to(flat)
        return flat.reshape_as(text_embeds)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """
        Args:
            inputs_embeds:  [B, seq_len, D]
            attention_mask: [B, seq_len]  (bool; True = valid token)
        Returns:
            logits: [B, num_value_bins]
            probs:  [B, num_value_bins]
            value:  [B, 1]
        """
        assert self.transformer_blocks is not None, "Call initialize_weights_from_backbone first."

        B, seq_len, D = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Append value query tokens.
        queries = self.value_queries.expand(B, -1, -1).to(dtype=dtype)
        hidden_states = torch.cat([inputs_embeds, queries], dim=1)

        # Build bidirectional 4-D attention bias.
        attn_bool = attention_mask.to(torch.bool)
        query_mask = torch.ones(B, self.num_value_bins, dtype=torch.bool, device=device)
        full_mask = torch.cat([attn_bool, query_mask], dim=1)  # [B, full_len]
        full_len = full_mask.shape[1]

        # [B, 1, full_len, full_len] — valid → 0.0, invalid → -inf
        bias = full_mask[:, None, None, :].expand(B, 1, full_len, full_len)
        neg_inf = torch.finfo(dtype).min
        attn_bias = torch.where(
            bias,
            torch.zeros(1, dtype=dtype, device=device),
            torch.full((1,), neg_inf, dtype=dtype, device=device),
        )

        # Position IDs (packed; value queries follow the last valid token).
        position_ids = torch.cumsum(full_mask.long(), dim=1) - 1
        cache_position = torch.arange(full_len, device=device)

        # Rotary embeddings.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Transformer blocks — fully bidirectional (no causal mask for critic).
        for block in self.transformer_blocks:
            out = block(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attn_bias,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden_states = out[0]

        hidden_states = self.ln_f(hidden_states)

        # Extract query outputs and map to logits.
        queries_out = hidden_states[:, seq_len:]  # [B, num_value_bins, D]
        logits = self.bin_logit_head(queries_out.to(self.compute_dtype)).squeeze(-1)  # [B, num_bins]

        probs = F.softmax(logits, dim=-1)
        value = self.value_from_probs(probs)  # [B, 1]

        return {"logits": logits, "probs": probs, "value": value}

    # ── Value / target helpers ─────────────────────────────────────────────────

    def value_from_probs(self, probs: Tensor) -> Tensor:
        """Expected value E[V] = sum_i p_i * c_i.  Returns [B, 1]."""
        bin_centers = self.bin_centers.to(dtype=probs.dtype)
        return (probs * bin_centers).sum(dim=-1, keepdim=True)

    def value_from_logits(self, logits: Tensor) -> Tensor:
        return self.value_from_probs(F.softmax(logits, dim=-1))

    def hl_gauss_target(self, target_v: Tensor) -> Tensor:
        """
        HL-Gauss soft target distribution for a scalar TD target.

        target_v: [B, 1]  (clamped to [value_support_min, value_support_max])
        Returns:  [B, num_value_bins]  (soft probability distribution)
        """
        bin_centers = self.bin_centers.to(dtype=target_v.dtype)  # [num_bins]
        sigma = self.hl_gauss_sigma

        target_v_clamped = target_v.clamp(
            float(bin_centers[0].item()), float(bin_centers[-1].item())
        )  # [B, 1]

        # Gaussian cdf evaluated at bin edges.
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else torch.ones(1)
        half = bin_width / 2.0
        lower = bin_centers - half  # [num_bins]
        upper = bin_centers + half  # [num_bins]

        import math
        sqrt2 = math.sqrt(2.0)
        cdf_upper = 0.5 * (1.0 + torch.erf((upper - target_v_clamped) / (sigma * sqrt2)))
        cdf_lower = 0.5 * (1.0 + torch.erf((lower - target_v_clamped) / (sigma * sqrt2)))
        probs = (cdf_upper - cdf_lower).clamp(min=0.0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
