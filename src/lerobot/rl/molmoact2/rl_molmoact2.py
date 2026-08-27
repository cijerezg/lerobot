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

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.rl.molmoact2.hybrid_critic import MolmoAct2Critic
from lerobot.rl.shared_config import ActorLearnerConfig, ConcurrencyConfig, MemoryConfig

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

    # ── Memory (short-term observation history) ─────────────────────────────
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # ── Subtask generation (two-prompt, string-level) ────────────────────────
    subtask_max_new_tokens: int = 0  # 0 = generation disabled
    subtask_regeneration_interval: float = 1.0  # seconds between regenerations
    subtask_loss_weight: float = 0.0  # CE weight on generation answers; 0 = no subtask training

    # ── Operator subtask console (eval only) ─────────────────────────────────
    # keyboard key -> subtask string. Non-empty REPLACES generation at rollout:
    # the operator latches the current step live. Strings must appear verbatim in
    # the checkpoint's subtask vocabulary; the first entry is the episode default.
    eval_subtasks: dict[str, str] = field(default_factory=dict)

    # ── Replay buffer ──────────────────────────────────────────────────────
    storage_device: str = "cpu"
    offline_buffer_capacity: int = 100_000
    image_storage_dtype: str = "uint8"
    image_storage_size: tuple[int, int] | None = None
    reward_normalization_constant: float = 1.0
    terminal_failure_reward: float = -10.0
    critic_reward_mode: str = "episode"  # episode | subtask
    critic_mistake_penalty: float = 0.0  # raw, one-time cost at each mistake-span entry
    async_prefetch: bool = False

    # ── Actor/learner concurrency (compatibility stubs) ─────────────────────
    shared_encoder: bool = False
    num_discrete_actions: int | None = None
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = False

    # ── Online training ────────────────────────────────────────────────────
    online_steps: int = 1_000_000
    online_buffer_capacity: int = 100_000
    online_step_before_learning: int = 100  # transitions before first gradient step
    actor_device: str | None = None
    learner_device: str | None = None
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # ── Distributional critic ──────────────────────────────────────────────
    # Hybrid critic: policy modality encoders supply detached 2560-wide prefix
    # tokens; a compact critic-owned transformer performs multimodal fusion.
    # critic_llm_depth keeps its old config name for checkpoint/YAML compatibility,
    # but now means the number of critic fusion blocks (not copied LLM blocks).
    critic_llm_depth: int = 4
    critic_input_hidden_size: int = 2560
    critic_hidden_size: int = 768
    critic_num_attention_heads: int = 12
    critic_mlp_ratio: float = 4.0
    critic_dropout: float = 0.0
    critic_max_tokens: int = 2048
    num_value_bins: int = 101
    value_support_min: float = -2.0
    value_support_max: float = 0.0
    hl_gauss_sigma_ratio: float = 5.0
    critic_lr: float = 1e-4
    # Actor depth keeps a separate optimizer object only so pretrained merging cannot
    # touch parameters absent from the base checkpoint. None means exactly optimizer_lr.
    depth_lr: float | None = None
    critic_target_update_weight: float = 0.005
    critic_target_update_every: int = 4
    discount: float = 0.97
    utd_ratio: int = 1
    critic_warmup_steps: int = 0
    policy_update_freq: int = 1

    # ── Pretrained merge ──────────────────────────────────────────────────
    # Periodic convex pull toward pretrained weights. alpha == 0 or every_n_steps == 0 disables.
    pretrained_merge_alpha: float = 0.0
    pretrained_merge_every_n_steps: int = 0
    pretrained_merge_targets: list[str] = field(default_factory=lambda: ["policy", "critic"])

    # ── Inference ─────────────────────────────────────────────────────────
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
                    raise ValueError(f"action_clamp_limits[{idx}] must be a [min, max] pair, got {limits!r}.")
                if float(limits[0]) > float(limits[1]):
                    raise ValueError(f"action_clamp_limits[{idx}] min must be <= max, got {limits!r}.")

        if self.critic_llm_depth < 1:
            raise ValueError("critic_llm_depth must be >= 1.")
        if self.critic_hidden_size < 1:
            raise ValueError("critic_hidden_size must be >= 1.")
        if self.critic_num_attention_heads < 1:
            raise ValueError("critic_num_attention_heads must be >= 1.")
        if self.critic_hidden_size % self.critic_num_attention_heads != 0:
            raise ValueError(
                "critic_hidden_size must be divisible by critic_num_attention_heads, got "
                f"{self.critic_hidden_size} and {self.critic_num_attention_heads}."
            )
        if self.critic_mlp_ratio <= 0:
            raise ValueError("critic_mlp_ratio must be > 0.")
        if not 0 <= self.critic_dropout < 1:
            raise ValueError("critic_dropout must be in [0, 1).")
        if self.critic_max_tokens < 1:
            raise ValueError("critic_max_tokens must be >= 1.")
        if self.num_value_bins < 2:
            raise ValueError("num_value_bins must be >= 2.")

        # Depth history rides the pointmap path: the memory block is the single source
        # of truth for the window shape, synced here so both encoder sites (policy +
        # critic) see it.
        if (
            self.pointmap_config is not None
            and f"depth.{self.pointmap_config.depth_key}.depth" in self.memory.history_keys
        ):
            self.pointmap_config.history_num_samples = self.memory.history_num_samples
            self.pointmap_config.history_window_seconds = self.memory.history_window_seconds


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

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        # init_critic() is lazy, so critic.* keys in the checkpoint would be
        # silently dropped as "unexpected" during the default load.  Pre-init
        # the sub-module here so every key lands correctly.
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as _sf:
            file_keys = set(_sf.keys())
        has_critic = any(k.startswith("critic.") for k in file_keys)
        if has_critic and not hasattr(model, "critic"):
            model.init_critic()
        return super()._load_as_safetensor(model, model_file, map_location, strict)

    def init_critic(self) -> None:
        """
        Instantiate and initialise the distributional critic + its frozen target.

        Called by the trainer only when skip_critic=False; lazy to avoid 2×
        memory overhead during actor-only runs.
        """
        device = self.config.device
        dtype = torch.bfloat16 if getattr(self.config, "dtype", "bfloat16") == "bfloat16" else torch.float32

        self.critic: MolmoAct2Critic = MolmoAct2Critic(self.config)
        self.critic = self.critic.to(device=device, dtype=dtype)

        self.critic_target: MolmoAct2Critic = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_target.eval()

    # ── Critic forward ────────────────────────────────────────────────────────

    def _forward_critic_impl(
        self,
        critic_module,
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Shared forward path for critic and critic_target.

        The policy's normal input builder is the shared encoder boundary. It
        assembles RGB-temporal, depth-history, state-history, and language tokens
        exactly as it does for the actor, but under no_grad. The critic sees only
        detached tokens and cannot update the policy encoders.
        """
        with torch.no_grad():
            model_inputs = self._model_inputs(batch)
            encoder_tokens, _, _, _ = self._prepare_joint_training_backbone_inputs(model_inputs)

        attention_mask = model_inputs.get("attention_mask")
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            input_ids = model_inputs.get("input_ids")
            attention_mask = (
                input_ids != -1
                if input_ids is not None
                else torch.ones(
                    encoder_tokens.shape[:2],
                    dtype=torch.bool,
                    device=encoder_tokens.device,
                )
            )
        return critic_module(encoder_tokens.detach(), attention_mask.to(torch.bool))

    def forward_critic(self, batch: dict) -> dict[str, torch.Tensor]:
        """V(s) with gradient — used for critic updates."""
        return self._forward_critic_impl(self.critic, batch)

    def forward_critic_target(self, batch: dict) -> dict[str, torch.Tensor]:
        """V(s') with frozen target network — used inside torch.no_grad()."""
        self.critic_target.eval()
        return self._forward_critic_impl(self.critic_target, batch)
