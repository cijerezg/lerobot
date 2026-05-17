"""
PI05Trainer — concrete Trainer implementation for PI05RLPolicy.

Thin wrapper over the existing pi05_train_utils.py functions.
All behaviour is identical to the original offline_learner_pi05.py / learner_pi05.py.
Original files remain unchanged and can be used as reference.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.pi05_train_utils import (
    _update_critic,
    _update_actor,
    _compute_advantage_with_interventions,
    _prepare_actor_batch,
    _construct_actor_forward_batch,
    _prepare_batch,
    make_pi05_full_processors_with_upgrade,
    hydrate_subtasks,
    log_pi05_training_metrics,
)
from lerobot.rl.utils import preprocess_batch_for_pi05
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.transport.utils import state_to_bytes
from lerobot.utils.constants import ACTION


# Depth constants for SigLIP-400M / Gemma-2B used by PI05
_VISION_TOWER_DEPTH = 27
_CRITIC_VISION_TOWER_DEPTH = 13
_LANGUAGE_MODEL_DEPTH = 18


class PI05Trainer(Trainer):
    """
    Trainer for PI05RLPolicy.

    Delegates all heavy lifting to the existing pi05_train_utils functions so
    that the generic training scripts (rl_offline.py, rl_learner_vla.py) produce
    identical behaviour to offline_learner_pi05.py / learner_pi05.py.
    """

    # ── Setup ─────────────────────────────────────────────────────────────────

    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:
        return make_pi05_full_processors_with_upgrade(cfg, dataset, is_main_process)

    def make_policy(self, cfg) -> nn.Module:
        from lerobot.policies.factory import make_policy
        return make_policy(cfg=cfg.policy, env_cfg=cfg.env)

    def freeze_model(self, policy: nn.Module, cfg) -> None:
        """
        Layer-granular freeze following cfg.policy.trainable_params.
        Mirrors the inline freeze block in offline_learner_pi05.py.
        """
        tp = cfg.policy.trainable_params
        critic_depth = cfg.policy.critic_llm_depth

        lm_layers = (
            list(range(tp.language_from_layer, _LANGUAGE_MODEL_DEPTH))
            if tp.language_from_layer is not None else []
        )
        vt_layers = (
            list(range(tp.vision_encoder_from_layer.vision_tower, _VISION_TOWER_DEPTH))
            if tp.vision_encoder_from_layer.vision_tower is not None else []
        )
        cr_layers = (
            list(range(tp.critic_language_from_layer, critic_depth))
            if tp.critic_language_from_layer is not None else []
        )
        cr_vt_layers = (
            list(range(tp.critic_vision_encoder_from_layer.vision_tower, _CRITIC_VISION_TOWER_DEPTH))
            if tp.critic_vision_encoder_from_layer.vision_tower is not None else []
        )

        for name, param in policy.named_parameters():
            param.requires_grad = (
                # Action expert — always trainable
                "action_in_proj" in name or
                "action_out_proj" in name or
                "time_mlp_in" in name or
                "time_mlp_out" in name or
                "gemma_expert" in name or
                # Actor vision encoder
                (tp.vision_encoder_from_layer.multi_modal_projector
                 and "paligemma" in name and "multi_modal_project" in name) or
                ("paligemma" in name and "vision_tower" in name
                 and any(f".{i}." in name for i in vt_layers)) or
                # Actor language model
                ("language_model" in name and any(f".{i}." in name for i in lm_layers)) or
                ("language_model.norm" in name and bool(lm_layers)) or
                # Critic value head + queries — always trainable
                "critic.norm" in name or
                "critic.value_head" in name or
                "critic.value_queries" in name or
                "critic.bin_logit_head" in name or
                ("critic.layers" in name and any(f".{i}." in name for i in cr_layers)) or
                # Critic vision encoder
                (tp.critic_vision_encoder_from_layer.multi_modal_projector
                 and name.startswith("critic.") and "multi_modal_project" in name) or
                (name.startswith("critic.vision_tower")
                 and any(f".{i}." in name for i in cr_vt_layers))
            )

    def get_optimizer_groups(self, policy: nn.Module, cfg) -> list[dict]:
        return [
            {
                "params": [p for p in policy.actor.parameters() if p.requires_grad],
                "lr": cfg.policy.actor_lr,
                "name": "actor",
            },
            {
                "params": [p for p in policy.critic_ensemble.parameters() if p.requires_grad],
                "lr": cfg.policy.critic_lr,
                "name": "critic",
            },
        ]

    # ── Critic ────────────────────────────────────────────────────────────────

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
        return _update_critic(
            policy=policy,
            optimizers=optimizers,
            online_iterator=online_iter,
            offline_iterator=offline_iter,
            device=device,
            cfg=cfg,
            dataset_repo_id=kwargs.get("dataset_repo_id"),
            gradient_accumulation_steps=cfg.policy.gradient_accumulation_steps,
            clip_grad_norm_value=cfg.policy.grad_clip_norm,
            cast_to_bf16_fn=kwargs.get("cast_to_bf16_fn"),
            use_amp=kwargs.get("use_amp", False),
            scaler=kwargs.get("scaler"),
        )

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
        raw_adv, raw_adv_flat, current_v, target_v = _compute_advantage_with_interventions(
            policy=policy,
            batch=batch,
            observations=observations,
            next_observations=next_observations,
            actions=batch.get(ACTION),
            rewards=rewards,
            done=done,
            cfg=cfg,
        )
        squashed = torch.tanh(raw_adv / cfg.policy.advantage_scaling)
        value_info = {"current_v": current_v, "target_v": target_v, "raw_adv_flat": raw_adv_flat}
        return raw_adv, squashed, value_info

    # ── Actor ─────────────────────────────────────────────────────────────────

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
        [SUBTASK LIVES HERE]
        Hydrates subtask strings from dataset metadata, re-tokenizes, and assembles
        the full actor forward batch including subtask CE targets.
        """
        processed, subtask_indices = _prepare_actor_batch(
            batch=raw_batch,
            observations=observations,
            actions=actions,
            raw_advantage_flat=advantage.view(-1),
            cfg=cfg,
            dataset=dataset,
            preprocessor=preprocessor,
        )
        forward_batch = _construct_actor_forward_batch(
            processed_batch=processed,
            observations=observations,
            actions=actions,
            rewards=raw_batch.get("reward"),
            done=raw_batch.get("done"),
            raw_advantage=advantage,
            cfg=cfg,
            subtask_indices=subtask_indices,
            device=str(actions.device),
            cast_to_bf16_fn=None,
        )
        return forward_batch

    def actor_forward(
        self,
        policy: nn.Module,
        batch: dict,
        advantage: torch.Tensor,
        cfg,
    ) -> dict[str, Any]:
        squashed = torch.tanh(advantage / cfg.policy.advantage_scaling)
        external_metrics = {
            "advantage": squashed,
            "critic_values": batch.get("_current_v"),
            "target_values": batch.get("_target_v"),
            "rewards": batch.get("reward"),
        }
        output = policy.forward(batch, model="actor", external_metrics=external_metrics)
        return {
            "loss_actor": output["loss_actor"],
            "loss_flow": output["flow_mse_loss"],
            "loss_action_ce": output.get("action_ce_loss", torch.tensor(0.0)),
            "loss_subtask_ce": output.get("subtask_ce_loss", torch.tensor(0.0)),
        }

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
        return _update_actor(
            policy=policy,
            optimizers=optimizers,
            online_iterator=online_iter,
            offline_iterator=offline_iter,
            device=device,
            cfg=cfg,
            dataset_repo_id=kwargs.get("dataset_repo_id"),
            gradient_accumulation_steps=cfg.policy.gradient_accumulation_steps,
            policy_update_freq=cfg.policy.policy_update_freq,
            clip_grad_norm_value=cfg.policy.grad_clip_norm,
            dataset=dataset,
            cast_to_bf16_fn=kwargs.get("cast_to_bf16_fn"),
            use_amp=kwargs.get("use_amp", False),
            scaler=kwargs.get("scaler"),
            preprocessor=preprocessor,
        )

    def update_target_networks(self, policy: nn.Module) -> None:
        tau = policy.config.critic_target_update_weight
        for p, p_tgt in zip(policy.critic.parameters(), policy.critic_target.parameters()):
            p_tgt.data.mul_(1 - tau).add_(p.data, alpha=tau)

    # ── Inference ─────────────────────────────────────────────────────────────

    def build_inference_batch(
        self,
        observation: dict,
        task_str: str,
        cfg,
        **context,
    ) -> dict[str, Any]:
        """
        Assembles the inference batch for PI05.
        context kwargs: preprocessor (required), prev_actions, robot_type,
                        subtask_override, inference_advantage.
        """
        from lerobot.types import TransitionKey
        preprocessor = context["preprocessor"]
        batch = {**observation}
        batch[TransitionKey.COMPLEMENTARY_DATA] = {
            "task": task_str,
            "subtask": context.get("subtask_override", ""),
            "advantage": context.get("inference_advantage", cfg.policy.inference_advantage),
            "robot_type": context.get("robot_type", "so101"),
        }
        if "prev_actions" in context:
            batch[TransitionKey.COMPLEMENTARY_DATA]["prev_actions"] = context["prev_actions"]
        with torch.no_grad():
            return preprocessor(batch)

    # ── Online loop ───────────────────────────────────────────────────────────

    def critic_value_for_logging(
        self,
        policy: nn.Module,
        transition: dict,
        device: str,
        cfg,
    ) -> float | None:
        observations = transition.get("state", {})
        next_observations = transition.get("next_state", {})
        rewards = transition.get("reward", torch.zeros(1))
        done = transition.get("done", torch.zeros(1))
        if not observations:
            return None
        batch = preprocess_batch_for_pi05(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
            actions=None,
            rewards=rewards,
            done=done,
            task=cfg.policy.task,
        )
        batch = move_transition_to_device(batch, device)
        with torch.no_grad():
            out = policy.forward(batch, model="critic_value")
        return float(out["value"].mean().item())

    def push_weights(self, policy: nn.Module, parameters_queue) -> None:
        trainable = {
            name: param
            for name, param in policy.actor.named_parameters()
            if param.requires_grad
        }
        state_bytes = state_to_bytes({"policy": move_state_dict_to_device(trainable, "cpu")})
        parameters_queue.put(state_bytes)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_metrics(
        self,
        training_infos: dict,
        step: int,
        wandb_logger,
        _policy: nn.Module,
    ) -> None:
        log_pi05_training_metrics(
            training_infos=training_infos,
            optimization_step=step,
            wandb_logger=wandb_logger,
            policy=_policy,
            is_main_process=True,
        )
