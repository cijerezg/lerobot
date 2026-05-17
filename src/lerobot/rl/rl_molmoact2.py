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
from dataclasses import dataclass, field

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.rl.rl_pi05 import ActorLearnerConfig  # shared actor↔learner connection config


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
    weights_push_interval: float = 180.0     # seconds between learner→actor weight pushes
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)

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


# ── Policy ─────────────────────────────────────────────────────────────────


class MolmoAct2RLPolicy(MolmoAct2Policy):
    """
    MolmoAct2 policy for RL training.

    Phase 2: BC-only — identical to MolmoAct2Policy.
    Phase 3: adds distributional value critic (MolmoAct2Critic).
    """

    # config type attribute used by PreTrainedPolicy.from_pretrained
    name = "molmoact2_rl"

    # ── Critic lifecycle ──────────────────────────────────────────────────────

    def init_critic(self) -> None:
        """
        Instantiate and initialise the distributional critic + its frozen target.

        Called by the trainer only when skip_critic=False; lazy to avoid 2×
        memory overhead during BC-only runs.
        """
        from lerobot.rl.rl_molmoact2_critic import MolmoAct2Critic

        device = self.config.device
        dtype = torch.bfloat16 if getattr(self.config, "model_dtype", "bfloat16") == "bfloat16" else torch.float32

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
