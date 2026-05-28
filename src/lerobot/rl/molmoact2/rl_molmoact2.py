"""
MolmoAct2 RL policy and config.

MolmoAct2RLConfig  extends MolmoAct2Config with RL training fields.
MolmoAct2RLPolicy  extends MolmoAct2Policy with a distributional critic head.

Both are registered with type "molmoact2_rl" so:
  - draccus/PreTrainedConfig can parse YAML policy blocks
  - factory.get_policy_class("molmoact2_rl") finds MolmoAct2RLPolicy via the
    naming-convention fallback in _get_policy_cls_from_policy_name
  - Trainer.for_config() routes to MolmoAct2Trainer
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.rl.shared_config import ActorLearnerConfig, ConcurrencyConfig


# ── Config ─────────────────────────────────────────────────────────────────


@PreTrainedConfig.register_subclass("molmoact2_rl")
@dataclass
class MolmoAct2RLConfig(MolmoAct2Config):
    """
    MolmoAct2 config extended with fields required by the RL training infra.

    All RL-specific fields live here so MolmoAct2Config stays upstream-clean.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    task: str = ""

    # ── Training loop ──────────────────────────────────────────────────────
    offline_steps: int = 10_000
    gradient_accumulation_steps: int = 1

    # ── Replay buffer ──────────────────────────────────────────────────────
    storage_device: str = "cpu"
    offline_buffer_capacity: int = 100_000
    image_storage_dtype: str = "uint8"
    image_storage_size: tuple[int, int] | None = None
    reward_normalization_constant: float = 1.0
    terminal_failure_reward: float = -10.0
    async_prefetch: bool = False

    # ── Actor/learner concurrency (compatibility stubs) ─────────────────────
    shared_encoder: bool = False
    num_discrete_actions: int | None = None
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = False

    # ── Online training ────────────────────────────────────────────────────
    online_steps: int = 1_000_000
    online_buffer_capacity: int = 100_000
    online_step_before_learning: int = 100   # transitions before first gradient step
    actor_device: str | None = None
    learner_device: str | None = None
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # ── Distributional critic ──────────────────────────────────────────────
    critic_llm_depth: int = 12
    num_value_bins: int = 101
    value_support_min: float = -2.0
    value_support_max: float = 0.0
    hl_gauss_sigma_ratio: float = 5.0
    critic_lr: float = 1e-4
    critic_target_update_weight: float = 0.005
    critic_target_update_every: int = 4
    discount: float = 0.97
    advantage_scaling: float = 0.2
    utd_ratio: int = 1
    critic_warmup_steps: int = 0
    policy_update_freq: int = 1

    # ── Pretrained merge ──────────────────────────────────────────────────
    # Periodic convex pull toward pretrained weights. alpha == 0 or every_n_steps == 0 disables.
    pretrained_merge_alpha: float = 0.0
    pretrained_merge_every_n_steps: int = 0
    pretrained_merge_targets: list[str] = field(default_factory=lambda: ["policy", "critic"])

    # ── Inference ─────────────────────────────────────────────────────────
    # Constant advantage value injected as prompt conditioning at inference.
    # Threaded through build_inference_batch into complementary_data["advantage"];
    # the processor bins it into a "negative"/"positive" label clause.
    # Set to null/None to drop the clause entirely — match this to training:
    # use null with skip_critic=True (actor saw no clause), keep 1.0 with
    # skip_critic=False (actor was trained on advantage-conditioned prompts).
    inference_advantage: float | None = 1.0

    # ── Advantage binning ─────────────────────────────────────────────────
    # Per-batch top-K positive labeling: the top advantage_top_k_fraction of
    # samples (ranked by squashed advantage) receive the "positive" label, the
    # rest "negative". The threshold pool excludes is_golden and is_intervention
    # samples — those are forced positive by override and would otherwise
    # saturate the quantile. All-override batches use threshold=-inf (everyone
    # positive).
    advantage_top_k_fraction: float = 0.3
    torch_compile: bool = False

    # ── Action encoding ───────────────────────────────────────────────────
    # "absolute" (default) - network predicts a_t directly.
    # "anchor"             - network predicts d_t = a_t - s_0.
    # "delta"              - network predicts step-deltas.
    action_encoding: str = "absolute"

    # Path to precomputed encoded-action stats (.pt file with normalizer stats).
    # Required when action_encoding is "anchor" or "delta"; ignored for absolute.
    action_encoding_stats_path: str | None = None

    # Per-joint [min, max] limits in degrees, applied after inference reconstruction.
    # None disables clamping.
    action_clamp_limits: list[list[float]] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.action_encoding not in {"absolute", "anchor", "delta"}:
            raise ValueError(
                f"Unsupported action_encoding={self.action_encoding!r}. "
                "Expected one of {'absolute', 'anchor', 'delta'}."
            )
        if self.action_encoding in {"anchor", "delta"}:
            stats_path = self.action_encoding_stats_path
            if not stats_path or not os.path.exists(os.path.expanduser(stats_path)):
                raise ValueError(
                    f"action_encoding={self.action_encoding!r} requires an existing "
                    f"action_encoding_stats_path, got {stats_path!r}."
                )

        if self.action_clamp_limits is not None:
            action_dim = None
            action_feature = (
                self.output_features.get("action") if isinstance(self.output_features, dict) else None
            )
            if action_feature is not None:
                shape = getattr(action_feature, "shape", None)
                if shape is None and isinstance(action_feature, dict):
                    shape = action_feature.get("shape")
                if shape:
                    action_dim = int(shape[0])

            if action_dim is not None and len(self.action_clamp_limits) != action_dim:
                raise ValueError(
                    f"action_clamp_limits must have {action_dim} [min, max] pairs, "
                    f"got {len(self.action_clamp_limits)}."
                )
            for idx, limits in enumerate(self.action_clamp_limits):
                if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                    raise ValueError(
                        f"action_clamp_limits[{idx}] must be a [min, max] pair, got {limits!r}."
                    )
                if float(limits[0]) > float(limits[1]):
                    raise ValueError(
                        f"action_clamp_limits[{idx}] min must be <= max, got {limits!r}."
                    )


# ── Critic ─────────────────────────────────────────────────────────────────


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
        self.compute_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

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
        """PI05-style HL-Gauss target distribution over value bins."""
        if target_v.ndim == 2:
            target_v = target_v.squeeze(-1)
        target_v = target_v.to(dtype=self.bin_centers.dtype)

        internal_edges = 0.5 * (self.bin_centers[:-1] + self.bin_centers[1:])
        z = (internal_edges.unsqueeze(0) - target_v.unsqueeze(-1)) / (
            self.hl_gauss_sigma * (2.0**0.5)
        )
        cdf_internal = 0.5 * (1.0 + torch.erf(z))

        zeros = torch.zeros_like(cdf_internal[:, :1])
        ones = torch.ones_like(cdf_internal[:, :1])
        cdf_full = torch.cat([zeros, cdf_internal, ones], dim=-1)
        return cdf_full[:, 1:] - cdf_full[:, :-1]

    def one_hot_target(self, target_v: Tensor) -> Tensor:
        """Nearest-bin one-hot target for exact terminal values."""
        if target_v.ndim == 2:
            target_v = target_v.squeeze(-1)
        target_v = target_v.to(dtype=self.bin_centers.dtype)
        idx = torch.argmin(torch.abs(self.bin_centers.unsqueeze(0) - target_v.unsqueeze(-1)), dim=-1)
        return F.one_hot(idx, num_classes=self.bin_centers.shape[0]).to(dtype=self.bin_centers.dtype)



# ── Policy ─────────────────────────────────────────────────────────────────


class MolmoAct2RLPolicy(MolmoAct2Policy):
    """
    MolmoAct2 policy for RL training.

    Phase 2: actor-only — identical to MolmoAct2Policy.
    Phase 3: adds distributional value critic (MolmoAct2Critic).
    """

    # config type attribute used by PreTrainedPolicy.from_pretrained
    name = "molmoact2_rl"

    # ── Critic lifecycle ──────────────────────────────────────────────────────

    def init_critic(self) -> None:
        """
        Instantiate and initialise the distributional critic + its frozen target.

        Called by the trainer only when skip_critic=False; lazy to avoid 2×
        memory overhead during actor-only runs.
        """
        device = self.config.device
        dtype = torch.bfloat16 if getattr(self.config, "dtype", "bfloat16") == "bfloat16" else torch.float32

        self.critic: MolmoAct2Critic = MolmoAct2Critic(self.config)
        backbone = self._backbone()
        self.critic.initialize_weights_from_backbone(backbone)
        self.critic = self.critic.to(device=device, dtype=dtype)

        self.critic_target: MolmoAct2Critic = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    # ── Critic forward ────────────────────────────────────────────────────────

    def _forward_critic_impl(
        self,
        critic_module,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Shared forward path for critic and critic_target.

        batch must contain the keys produced by the standard MolmoAct2
        preprocessor: input_ids, pixel_values, attention_mask (optional),
        and optionally image_token_pooling / image_grids / image_num_crops.
        """
        backbone = self._backbone()
        input_ids = batch.get("input_ids")
        pixel_values = batch.get("pixel_values")
        attention_mask = batch.get("attention_mask")

        inputs_embeds = critic_module.embed_observation(
            backbone,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=batch.get("image_token_pooling"),
            image_grids=batch.get("image_grids"),
            image_num_crops=batch.get("image_num_crops"),
        )

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = (input_ids != -1)
            else:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device
                )

        return critic_module(inputs_embeds, attention_mask.to(torch.bool))

    def forward_critic(self, batch: dict) -> dict[str, torch.Tensor]:
        """V(s) with gradient — used for critic updates."""
        return self._forward_critic_impl(self.critic, batch)

    def forward_critic_target(self, batch: dict) -> dict[str, torch.Tensor]:
        """V(s') with frozen target network — used inside torch.no_grad()."""
        return self._forward_critic_impl(self.critic_target, batch)
