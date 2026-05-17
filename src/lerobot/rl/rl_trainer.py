from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Trainer(ABC):
    """
    Abstract strategy for value-based robot policy training.

    All supported models share: backbone (VLM/WAM) + action expert (flow matching).
    One concrete subclass per model family; lives in rl_<model>.py alongside the
    policy and config classes. Generic training scripts (rl_offline.py, rl_learner.py,
    rl_inference_async.py) only call methods on this interface.

    Usage:
        trainer = Trainer.for_config(cfg)
        preprocessor, postprocessor = trainer.make_processors(cfg, dataset)
        policy = trainer.make_policy(cfg)
        trainer.freeze_model(policy, cfg)
        optimizers = build_optimizers(trainer.get_optimizer_groups(policy, cfg), cfg)
    """

    # ── Setup ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:
        """
        Build (preprocessor, postprocessor) for this model.

        PI05:      make_pi05_full_processors_with_upgrade — loads norm stats from
                   checkpoint JSON, handles anchor/delta encoding.
        MolmoAct2: make_molmoact2_pre_post_processors — loads from HF norm_stats.json
                   via norm_tag; no anchor/delta complexity.
        """

    @abstractmethod
    def make_policy(self, cfg) -> nn.Module:
        """
        Build the full RL policy: backbone + action expert + critic + target critic.

        PI05:      PI05RLPolicy(cfg.policy)
        MolmoAct2: MolmoAct2RLPolicy(cfg.policy)
        """

    @abstractmethod
    def freeze_model(self, policy: nn.Module, cfg) -> None:
        """
        Apply per-layer freeze schedule from cfg.policy.trainable_params.

        Both models share the same conceptual structure (layer indices):
          PI05:      SigLIP layers + Gemma layers + projector
          MolmoAct2: Molmo vision_backbone layers + transformer blocks
        The critic's shared layers have their own freeze config under critic_*.
        """

    @abstractmethod
    def get_optimizer_groups(self, policy: nn.Module, cfg) -> list[dict]:
        """
        Return per-component parameter groups for Adam/AdamW construction.

        Each dict has keys: "params", "lr", "name".
        PI05:      [actor_params, critic_params]
        MolmoAct2: [backbone_params (trainable subset), action_expert_params, critic_params]
        """

    # ── Critic ────────────────────────────────────────────────────────────────

    @abstractmethod
    def update_critic(
        self,
        policy: nn.Module,
        optimizers: dict[str, torch.optim.Optimizer],
        online_iter,
        offline_iter,
        device: str,
        cfg,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Full critic update: sample → forward_critic → loss → backward → step.

        Handles gradient accumulation, grad clipping, and Polyak target-network update.

        Returns at minimum:
            loss_critic, loss_critic_ce, loss_critic_mse, critic_grad_norm,
            critic_value_mean, td_error_mean, target_value_mean,
            critic_histogram_from_critic (np.ndarray), target_value_histogram (np.ndarray)
        """

    @abstractmethod
    def compute_advantage(
        self,
        policy: nn.Module,
        batch: dict,
        observations: dict,
        next_observations: dict,
        rewards: torch.Tensor,
        done: torch.Tensor,
        cfg,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute TD advantage and squash it.

        raw_advantage      = reward + discount * V_target(s') - V(s)
        squashed_advantage = tanh(raw_advantage / advantage_scaling)

        Returns:
            raw_advantage [B,1], squashed_advantage [B,1], value_info dict
        """

    # ── Actor ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def build_training_batch(
        self,
        raw_batch: dict,
        observations: dict,
        actions: torch.Tensor,
        advantage: torch.Tensor,
        preprocessor,
        dataset,
        cfg,
    ) -> dict[str, Any]:
        """
        Assemble the model-specific batch for the actor forward pass.

        *** SUBTASK ISOLATION POINT ***
        All model-specific batch enrichment lives here.

        PI05:
            1. hydrate_subtasks(raw_batch["complementary_info"]["subtask_index"], dataset)
            2. preprocess_batch_for_pi05(observations, subtask_strings, advantage, preprocessor)
            → {OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_SUBTASK_TOKENS, ACTION_TOKENS, actions, advantage}

        MolmoAct2:
            1. preprocessor({**observations, "task": cfg.policy.task})
            2. pack action targets + advantage
            → {input_ids, pixel_values, attention_mask, actions, advantage}
            NO subtask lookup.
        """

    @abstractmethod
    def actor_forward(
        self,
        policy: nn.Module,
        batch: dict,
        advantage: torch.Tensor,
        cfg,
    ) -> dict[str, Any]:
        """
        Actor forward pass + advantage-weighted loss computation.

        PI05:      policy.forward(batch, model="actor")
                   loss = w_flow*loss_flow + w_action_ce*loss_action_ce + w_subtask_ce*loss_subtask_ce
        MolmoAct2: policy.forward(batch)
                   loss = w_flow*loss_flow [+ w_discrete*loss_discrete_ce]

        Returns at minimum: loss_actor (scalar), loss_flow (scalar)
        """

    @abstractmethod
    def update_actor(
        self,
        policy: nn.Module,
        optimizers: dict[str, torch.optim.Optimizer],
        online_iter,
        offline_iter,
        preprocessor,
        dataset,
        device: str,
        cfg,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Full actor update:
            sample → compute_advantage → build_training_batch → actor_forward → backward → step

        Returns metrics dict for logging.
        """

    @abstractmethod
    def update_target_networks(self, policy: nn.Module) -> None:
        """
        Polyak-update critic_target ← τ*critic + (1-τ)*critic_target.
        τ = cfg.policy.critic_target_update_weight
        Called once per update step after both critic and actor updates.
        """

    # ── Inference ─────────────────────────────────────────────────────────────

    @abstractmethod
    def build_inference_batch(
        self,
        observation: dict,
        task_str: str,
        cfg,
        **context,
    ) -> dict[str, Any]:
        """
        Assemble the tokenised batch for a single inference step (no grad, on device).

        PI05:
            - task + subtask="" + inference_advantage into complementary_data
            - anchor/delta alignment of prev_actions for RTC inpainting
            - preprocessor → full token sequence
        MolmoAct2:
            - obs["task"] = task_str
            - preprocessor → {input_ids, pixel_values, attention_mask, ...}
        """

    # ── Online loop ───────────────────────────────────────────────────────────

    @abstractmethod
    def critic_value_for_logging(
        self,
        policy: nn.Module,
        transition: dict,
        device: str,
        cfg,
    ) -> float | None:
        """
        Compute V(s) for a single transition (no grad).
        Used to overlay critic values on episode videos.
        Return None if not yet implemented for this model.
        """

    @abstractmethod
    def push_weights(self, policy: nn.Module, parameters_queue) -> None:
        """
        Serialize trainable parameters (requires_grad=True only) and push to actor queue.
        PI05:      policy.actor.named_parameters()
        MolmoAct2: policy.model.named_parameters()
        """

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_metrics(
        self,
        training_infos: dict,
        step: int,
        wandb_logger,
        _policy: nn.Module,  # available for subclasses that log weight stats / histograms
    ) -> None:
        """
        Log scalar metrics to W&B. Override to add histograms, videos, etc.
        Default: logs all scalar (int/float) values in training_infos.
        """
        scalars = {k: v for k, v in training_infos.items() if isinstance(v, (int, float))}
        if wandb_logger is not None:
            wandb_logger.log({"step": step, **scalars})

    # ── Registry ──────────────────────────────────────────────────────────────

    @staticmethod
    def for_config(cfg) -> "Trainer":
        """
        Instantiate the correct Trainer from cfg.policy.type.
        Imports are deferred to avoid loading heavy optional dependencies.
        """
        policy_type = getattr(cfg.policy, "type", None)
        if policy_type == "pi05_rl":
            from lerobot.rl.rl_pi05_trainer import PI05Trainer
            return PI05Trainer()
        if policy_type == "molmoact2_rl":
            from lerobot.rl.rl_molmoact2_trainer import MolmoAct2Trainer  # noqa: F401 — registers MolmoAct2RLConfig
            import lerobot.rl.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
            return MolmoAct2Trainer()
        raise ValueError(
            f"No Trainer registered for policy type {policy_type!r}. "
            f"Add a new rl_<model>.py and register it in Trainer.for_config()."
        )
