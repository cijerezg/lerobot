"""
MolmoAct2Trainer — concrete Trainer for MolmoAct2RLPolicy.

BC mode  (skip_critic=True):  update_actor only; flow-matching loss, no advantage weighting.
RL mode  (skip_critic=False): update_critic (HL-Gauss distributional TD) + advantage-weighted actor.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.rl.rl_trainer import Trainer
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import move_transition_to_device


class MolmoAct2Trainer(Trainer):
    """Trainer for MolmoAct2RLPolicy. Supports both BC (skip_critic=True) and RL modes."""

    # ── Setup ─────────────────────────────────────────────────────────────────

    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:  # noqa: ARG002
        """
        Build (preprocessor, postprocessor) using make_molmoact2_pre_post_processors.

        The standard MolmoAct2 preprocessor is already safe for batched replay-buffer
        inputs: AddBatchDimensionProcessorStep only adds a dim to 1-D / 3-D tensors,
        so batched images [B,C,H,W] and states [B,D] pass through unchanged.
        """
        from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors

        dataset_stats = None
        if dataset is not None and hasattr(dataset, "stats"):
            dataset_stats = dataset.stats
        dataset_meta = getattr(dataset, "meta", None) if dataset is not None else None

        result = make_molmoact2_pre_post_processors(
            config=cfg.policy,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
        self._preprocessor = result[0]  # cache for critic_value_for_logging
        return result

    def make_policy(self, cfg) -> nn.Module:
        from lerobot.policies.factory import make_policy
        return make_policy(cfg=cfg.policy, env_cfg=cfg.env)

    def freeze_model(self, policy: nn.Module, cfg) -> None:
        """
        Actor freeze is already applied during _load_hf_model:
          - train_action_expert_only → _freeze_non_action_expert_parameters()
          - enable_lora_vlm → LoRA adapters (non-LoRA params frozen)
          - freeze_embedding → _freeze_input_embeddings()

        For RL mode (skip_critic=False): the critic was deepcopied from the
        (possibly frozen) actor backbone, so its parameters inherit
        requires_grad=False.  We unfreeze all critic params here so they
        can learn value estimates.  critic_target stays frozen — init_critic()
        already set that.
        """
        skip_critic: bool = bool(getattr(cfg, "skip_critic", True))

        if not skip_critic and hasattr(policy, "critic"):
            critic_net: nn.Module = getattr(policy, "critic")
            for p in critic_net.parameters():
                p.requires_grad_(True)

            # Share frozen actor layers with critic_target to save VRAM.
            # Only valid for params that neither the critic nor actor will update.
            if hasattr(policy, "critic_target"):
                critic_target: nn.Module = getattr(policy, "critic_target")
                for p, p_tgt in zip(critic_net.parameters(), critic_target.parameters()):
                    if not p.requires_grad:
                        p_tgt.data = p.data

        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in policy.parameters())
        logging.info(
            f"[MolmoAct2Trainer] Trainable: {trainable:,} / {total:,} params "
            f"({100 * trainable / max(total, 1):.1f}%)"
        )

    def get_optimizer_groups(self, policy: nn.Module, cfg) -> list[dict]:
        """
        Actor group: all policy params with requires_grad (excludes critic).
        Critic group: all critic params (added only when skip_critic=False).
        """
        skip_critic = getattr(cfg, "skip_critic", True)

        if skip_critic or not hasattr(policy, "critic"):
            trainable = [p for p in policy.parameters() if p.requires_grad]
            return [{"name": "policy", "params": trainable, "lr": cfg.policy.optimizer_lr}]

        critic_net: nn.Module = getattr(policy, "critic")
        critic_param_ids = {id(p) for p in critic_net.parameters()}
        actor_params = [
            p for p in policy.parameters()
            if p.requires_grad and id(p) not in critic_param_ids
        ]
        critic_params = list(critic_net.parameters())
        return [
            {"name": "policy", "params": actor_params, "lr": cfg.policy.optimizer_lr},
            {"name": "critic", "params": critic_params, "lr": cfg.policy.critic_lr},
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
        """
        One critic update step (with gradient accumulation).

        TD target: r + γ * V_target(s') * (1 − done)
        Loss: cross-entropy between critic logits and HL-Gauss soft target.
        """
        from lerobot.rl.buffer import concatenate_batch_transitions

        preprocessor = kwargs["preprocessor"]
        grad_accum = int(getattr(cfg.policy, "gradient_accumulation_steps", 1))
        clip_norm = float(getattr(cfg.policy, "optimizer_grad_clip_norm", 1.0))
        action_dim = self._action_dim(cfg)
        discount = float(getattr(cfg.policy, "discount", 0.97))

        optimizers["critic"].zero_grad()

        accum_ce = 0.0
        last_critic_out: dict[str, torch.Tensor] | None = None
        last_td_target: torch.Tensor | None = None
        last_soft_target: torch.Tensor | None = None

        for _ in range(grad_accum):
            raw = next(online_iter)
            if offline_iter is not None:
                raw_off = next(offline_iter)
                raw = concatenate_batch_transitions(raw, raw_off, action_dim=action_dim)
            raw = move_transition_to_device(raw, device)

            observations = raw.get("state", {})
            next_observations = raw.get("next_state", {})
            rewards = raw["reward"]
            done = raw["done"]
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards)
            if not isinstance(done, torch.Tensor):
                done = torch.tensor(done)
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(-1)
            if done.dim() == 1:
                done = done.unsqueeze(-1)

            # Preprocess observations for current and next states.
            curr_batch = preprocessor({**observations, "task": cfg.policy.task})
            next_batch = preprocessor({**next_observations, "task": cfg.policy.task})

            _fwd_critic = getattr(policy, "forward_critic")
            _fwd_target = getattr(policy, "forward_critic_target")

            # V(s') from frozen target — no grad.
            with torch.no_grad():
                target_out = _fwd_target(next_batch)
                v_next = target_out["value"].to(rewards.dtype)
                td_target = rewards + discount * v_next * (1.0 - done.float())
                critic_net: nn.Module = getattr(policy, "critic")
                soft_target = critic_net.hl_gauss_target(td_target)  # type: ignore[attr-defined]

            # V(s) with grad.
            critic_out = _fwd_critic(curr_batch)
            logits = critic_out["logits"]  # [B, num_bins]

            loss_ce = -(soft_target * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
            (loss_ce / grad_accum).backward()

            accum_ce += loss_ce.item() / grad_accum
            last_critic_out = critic_out
            last_td_target = td_target
            last_soft_target = soft_target

        critic_net2: nn.Module = getattr(policy, "critic")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(critic_net2.parameters()), clip_norm
        ).item()
        optimizers["critic"].step()

        # Logging metrics (using last accumulation step).
        with torch.no_grad():
            v_curr = last_critic_out["value"].float() if last_critic_out else torch.zeros(1)
            td_err = (last_td_target - v_curr).abs().mean().item() if last_td_target is not None else 0.0

        return {
            "loss_critic": accum_ce,
            "loss_critic_ce": accum_ce,
            "loss_critic_mse": td_err,
            "critic_grad_norm": grad_norm,
            "critic_value_mean": v_curr.mean().item(),
            "td_error_mean": td_err,
            "target_value_mean": last_td_target.mean().item() if last_td_target is not None else 0.0,
            "critic_histogram_from_critic": (
                last_critic_out["probs"].detach().float().cpu().numpy()
                if last_critic_out else None
            ),
            "target_value_histogram": (
                last_soft_target.detach().float().cpu().numpy()
                if last_soft_target is not None else None
            ),
        }

    def compute_advantage(
        self,
        policy: nn.Module,
        batch: dict,  # noqa: ARG002 — raw batch unused; obs/next_obs extracted by caller
        observations: dict,
        next_observations: dict,
        rewards: torch.Tensor,
        done: torch.Tensor,
        cfg,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute TD advantage.

        raw_advantage      = r + γ * V_target(s') - V(s)
        squashed_advantage = tanh(raw_advantage / advantage_scaling)

        kwargs must contain:
            preprocessor: the MolmoAct2 PolicyProcessorPipeline
        """
        preprocessor = kwargs["preprocessor"]
        discount = float(getattr(cfg.policy, "discount", 0.97))
        adv_scale = float(getattr(cfg.policy, "advantage_scaling", 0.2))

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        curr_batch = preprocessor({**observations, "task": cfg.policy.task})
        next_batch = preprocessor({**next_observations, "task": cfg.policy.task})

        _fwd_critic = getattr(policy, "forward_critic")
        _fwd_target = getattr(policy, "forward_critic_target")

        with torch.no_grad():
            target_out = _fwd_target(next_batch)
            v_next = target_out["value"].to(rewards.dtype)
            curr_out = _fwd_critic(curr_batch)
            v_curr = curr_out["value"].to(rewards.dtype)

        raw_adv = rewards + discount * v_next * (1.0 - done.float()) - v_curr
        squashed_adv = torch.tanh(raw_adv / adv_scale)

        value_info = {
            "advantage_raw_mean": raw_adv.mean().item(),
            "advantage_squashed_mean": squashed_adv.mean().item(),
            "v_curr_mean": v_curr.mean().item(),
            "v_next_mean": v_next.mean().item(),
        }
        return raw_adv, squashed_adv, value_info

    # ── Actor ─────────────────────────────────────────────────────────────────

    def build_training_batch(
        self,
        raw_batch: dict,  # noqa: ARG002
        observations: dict,
        actions: torch.Tensor,
        advantage: torch.Tensor,  # noqa: ARG002
        preprocessor,
        dataset,  # noqa: ARG002
        cfg,
    ) -> dict[str, Any]:
        """
        Build the MolmoAct2 forward batch.

        Flat dict with image/state observation keys + action + task string.
        The preprocessor (PolicyProcessorPipeline) converts this via:
          batch_to_transition → (steps) → transition_to_batch
        producing input_ids, pixel_values, attention_mask etc. for forward().
        """
        action_dim = self._action_dim(cfg)
        pre_input: dict[str, Any] = {
            **observations,
            "action": actions[..., :action_dim],
            "task": cfg.policy.task,
        }
        return preprocessor(pre_input)

    def actor_forward(
        self,
        policy: nn.Module,
        batch: dict,
        advantage: torch.Tensor,  # noqa: ARG002 — weighting handled in update_actor
        cfg,  # noqa: ARG002
    ) -> dict[str, Any]:
        """BC forward pass — advantage weighting is applied by update_actor, not here."""
        loss, metrics = policy.forward(batch)
        return {
            "loss_actor": loss.item(),
            "loss_flow": metrics.get("action_flow_loss", 0.0),
            "loss_discrete_ce": metrics.get("discrete_ce_loss", 0.0),
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
        """
        Actor update: sample → [compute_advantage] → preprocess → forward → backward → step.

        BC mode (skip_critic=True):  advantage=0, weight=1.
        RL mode (skip_critic=False): advantage-weighted loss via exp(squashed_advantage.mean()).
        Handles gradient accumulation.  Only the "policy" optimizer is touched here.
        """
        from lerobot.rl.buffer import concatenate_batch_transitions

        skip_critic = getattr(cfg, "skip_critic", True)
        grad_accum = int(getattr(cfg.policy, "gradient_accumulation_steps", 1))
        clip_norm = float(getattr(cfg.policy, "optimizer_grad_clip_norm", 1.0))
        action_dim = self._action_dim(cfg)

        policy_opt = optimizers.get("policy") or next(iter(optimizers.values()))
        policy_opt.zero_grad()

        accum: dict[str, float] = {
            "loss_actor": 0.0,
            "loss_flow": 0.0,
            "loss_discrete_ce": 0.0,
            "advantage_squashed_mean": 0.0,
        }

        # Identify actor params for grad clipping (excludes critic).
        critic_param_ids: set[int] = set()
        if hasattr(policy, "critic"):
            critic_net: nn.Module = getattr(policy, "critic")
            critic_param_ids = {id(p) for p in critic_net.parameters()}
        actor_params = [
            p for p in policy.parameters()
            if p.requires_grad and id(p) not in critic_param_ids
        ]

        for _ in range(grad_accum):
            raw = next(online_iter)
            if offline_iter is not None:
                raw_off = next(offline_iter)
                raw = concatenate_batch_transitions(raw, raw_off, action_dim=action_dim)
            raw = move_transition_to_device(raw, device)

            observations = raw.get("state", {})
            actions = raw[ACTION][..., :action_dim]

            # Advantage weighting (RL mode only).
            adv_weight = torch.tensor(1.0, device=device)
            if not skip_critic:
                next_observations = raw.get("next_state", {})
                rewards = raw["reward"]
                done = raw["done"]
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards)
                if not isinstance(done, torch.Tensor):
                    done = torch.tensor(done)
                _, squashed_adv, _ = self.compute_advantage(
                    policy,
                    batch=raw,  # type: ignore[arg-type]
                    observations=observations,
                    next_observations=next_observations,
                    rewards=rewards,
                    done=done,
                    cfg=cfg,
                    preprocessor=preprocessor,
                )
                adv_weight = torch.exp(squashed_adv.mean().to(device))
                accum["advantage_squashed_mean"] += squashed_adv.mean().item() / grad_accum

            fwd_batch = self.build_training_batch(
                raw_batch=raw,  # type: ignore[arg-type]
                observations=observations,
                actions=actions,
                advantage=adv_weight,
                preprocessor=preprocessor,
                dataset=dataset,
                cfg=cfg,
            )

            loss, metrics = policy.forward(fwd_batch)
            weighted_loss = loss * adv_weight
            (weighted_loss / grad_accum).backward()

            accum["loss_actor"] += weighted_loss.item() / grad_accum
            accum["loss_flow"] += metrics.get("action_flow_loss", 0.0) / grad_accum
            accum["loss_discrete_ce"] += metrics.get("discrete_ce_loss", 0.0) / grad_accum

        torch.nn.utils.clip_grad_norm_(actor_params, clip_norm)
        policy_opt.step()

        return accum

    def update_target_networks(self, policy: nn.Module) -> None:
        """Polyak update: critic_target ← τ*critic + (1-τ)*critic_target."""
        if not hasattr(policy, "critic"):
            return
        critic_net: nn.Module = getattr(policy, "critic")
        critic_target: nn.Module = getattr(policy, "critic_target")
        tau = float(getattr(policy.config, "critic_target_update_weight", 0.005))
        with torch.no_grad():
            for p, p_tgt in zip(critic_net.parameters(), critic_target.parameters()):
                p_tgt.data.lerp_(p.data, tau)

    # ── Inference ─────────────────────────────────────────────────────────────

    def build_inference_batch(
        self,
        observation: dict,
        task_str: str,
        cfg,  # noqa: ARG002
        **context,
    ) -> dict[str, Any]:
        """
        Assemble the preprocessed inference batch for MolmoAct2.

        observation: flat dict with image/state keys (no batch dim).
        The preprocessor's AddBatchDimensionProcessorStep adds the batch dim.
        """
        preprocessor = context["preprocessor"]
        pre_input: dict[str, Any] = {**observation, "task": task_str}
        with torch.no_grad():
            return preprocessor(pre_input)

    # ── Online loop ───────────────────────────────────────────────────────────

    def critic_value_for_logging(
        self,
        policy: nn.Module,
        transition: dict,
        device: str,
        cfg,
    ) -> float | None:
        if not hasattr(policy, "critic"):
            return None
        preprocessor = getattr(self, "_preprocessor", None)
        if preprocessor is None:
            return None
        try:
            observations = transition.get("state", {})
            batch = preprocessor({**observations, "task": cfg.policy.task})
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _fwd_critic = getattr(policy, "forward_critic")
            with torch.no_grad():
                out = _fwd_critic(batch)
            return float(out["value"].mean().item())
        except Exception:
            return None

    def push_weights(self, policy: nn.Module, parameters_queue) -> None:
        """Push trainable parameters to actor queue."""
        from lerobot.transport.utils import state_to_bytes
        from lerobot.utils.transition import move_state_dict_to_device

        trainable = {
            name: param
            for name, param in policy.named_parameters()
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
        if step == 0:
            return
        scalars = {k: v for k, v in training_infos.items() if isinstance(v, (int, float))}
        logging.info(
            f"[MolmoAct2Trainer] step={step}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in scalars.items())
        )
        if wandb_logger is not None:
            wandb_logger.log({"step": step, **scalars})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _action_dim(self, cfg) -> int:
        """Resolve the active action dimension from config output_features."""
        action_feat = getattr(cfg.policy, "output_features", {}).get(ACTION)
        if action_feat is not None and action_feat.shape:
            return int(action_feat.shape[0])
        return 6  # SO-101 fallback
