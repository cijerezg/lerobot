"""
PI05Trainer — concrete Trainer for PI05RLPolicy.

Critic:  distributional TD with HL-Gauss soft targets.
Actor:   flow matching + subtask CE auxiliary loss, conditioned on squashed advantage.
Special: intervention and golden-dataset transitions override the computed advantage with 1.0.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.rl_trainer import TrainableParamsConfig as TrainableParamsConfig  # re-exported for callers
from lerobot.rl.buffer import concatenate_batch_transitions
from lerobot.rl.utils import preprocess_batch_for_pi05
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport.utils import state_to_bytes
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device


# SigLIP-400M / Gemma-2B depths used by PI05.
_VISION_TOWER_DEPTH = 27
_CRITIC_VISION_TOWER_DEPTH = 13
_LANGUAGE_MODEL_DEPTH = 18


class PI05Trainer(Trainer):
    """
    Trainer for PI05RLPolicy.

    Actor-only mode     (skip_critic=True):  update_actor only.
    Critic-trained mode (skip_critic=False): update_critic (distributional HL-Gauss TD)
                                             + actor conditioned on advantage.
    """

    _HISTOGRAM_CLIP_RANGES = {
        "critic_histogram_from_critic": (-2.0, 0.1),
        "critic_histogram": (-2.0, 0.1),
        "target_value_histogram": (-2.0, 0.1),
        "flow_loss_raw": (0.0, 0.01),
        "loss_critic_raw": (0.0, 0.005),
    }

    # ── Setup ─────────────────────────────────────────────────────────────────

    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:
        """
        Build PI05 processors.

        Stat loading priority:
          1. use_dataset_stats=True or base-model checkpoint → dataset.meta.stats
          2. Checkpoint preprocessor JSON → normalizer state file
          3. Fallback to dataset.meta.stats
          4. anchor/delta action encoding stats always override the ACTION key.
        """
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor_overrides = {
            "pi05_full_prepare_state_tokenizer_processor_step": {
                "advantage_scaling": cfg.policy.advantage_scaling,
            }
        }

        dataset_stats = self._resolve_dataset_stats(cfg, dataset, is_main_process)

        action_encoding = getattr(cfg.policy, "action_encoding", "absolute")
        if action_encoding in ("anchor", "delta"):
            stats_path = getattr(cfg.policy, "action_encoding_stats_path", None)
            if not stats_path or not os.path.exists(stats_path):
                raise ValueError(
                    f"action_encoding={action_encoding!r} requires action_encoding_stats_path, "
                    f"got {stats_path!r}."
                )
            if is_main_process:
                logging.info(f"Loading {action_encoding} action stats from {stats_path}")
            enc_stats = torch.load(stats_path, map_location="cpu")
            if dataset_stats is None:
                dataset_stats = {}
            dataset_stats[ACTION] = enc_stats

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            dataset_stats=dataset_stats,
            preprocessor_overrides=preprocessor_overrides,
        )
        return preprocessor, postprocessor

    def _resolve_dataset_stats(self, cfg, dataset, is_main_process: bool) -> dict | None:
        """Determine which normalizer stats to use, in priority order."""
        use_dataset_stats = getattr(cfg.policy, "use_dataset_stats", False)
        base_path = getattr(cfg.policy, "base_path", None) or ""

        if use_dataset_stats:
            if is_main_process:
                logging.info("PI05 stats source: dataset stats (use_dataset_stats=True).")
            return dataset.meta.stats if dataset is not None else None

        if "pi05_base" in base_path:
            if is_main_process:
                logging.info(f"PI05 stats source: dataset stats (base model '{base_path}').")
            return dataset.meta.stats if dataset is not None else None

        if base_path:
            stats = self._load_stats_from_checkpoint(base_path, is_main_process)
            if stats is not None:
                return stats
            if is_main_process:
                logging.info("PI05 stats source: dataset stats (checkpoint fallback).")
            if dataset is not None:
                return dataset.meta.stats
            if is_main_process:
                logging.warning("PI05 stats: no checkpoint stats and no dataset provided — stats will be None.")

        return None

    @staticmethod
    def _load_stats_from_checkpoint(checkpoint_path: str, is_main_process: bool) -> dict | None:
        """Load normalizer stats from a checkpoint's preprocessor JSON."""
        try:
            from safetensors.torch import load_file

            config_path = Path(checkpoint_path) / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            if is_main_process:
                logging.info(f"PI05 stats source: checkpoint ({config_path})")
            if not config_path.exists():
                if is_main_process:
                    logging.warning(f"Preprocessor config not found at {config_path}.")
                return None
            with open(config_path) as f:
                proc_config = json.load(f)
            for step in proc_config.get("steps", []):
                if step.get("registry_name") == "normalizer_processor":
                    state_file = step.get("state_file")
                    if state_file:
                        state_path = Path(checkpoint_path) / state_file
                        if state_path.exists():
                            raw = load_file(str(state_path))
                            if is_main_process:
                                logging.info(f"PI05 stats loaded from {state_path}.")
                            return PI05Trainer._unflatten_stats(raw)
            if is_main_process:
                logging.warning("No normalizer_processor step with a state file found in checkpoint.")
        except Exception as exc:
            if is_main_process:
                logging.warning(f"Failed to load stats from checkpoint: {exc}")
        return None

    def make_policy(self, cfg) -> nn.Module:
        from lerobot.policies.factory import make_policy
        return make_policy(cfg=cfg.policy, env_cfg=cfg.env)

    def make_actor_policy(self, cfg) -> nn.Module:
        # Online actors receive only actor-runtime weights; the critic path is learner-only.
        if hasattr(cfg.policy, "use_separate_critic"):
            cfg.policy.use_separate_critic = False
        return self.make_policy(cfg)

    def freeze_model(self, policy: nn.Module, cfg) -> None:
        """Layer-granular freeze following cfg.policy.trainable_params."""
        tp = cfg.policy.trainable_params
        critic_depth = cfg.policy.critic_llm_depth

        vision_on = tp.vision_from_layer is not None
        language_on = tp.language_from_layer is not None
        critic_vision_on = tp.critic_vision_from_layer is not None

        vt_layers = list(range(tp.vision_from_layer, _VISION_TOWER_DEPTH)) if vision_on else []
        lm_layers = list(range(tp.language_from_layer, _LANGUAGE_MODEL_DEPTH)) if language_on else []
        cr_layers = (
            list(range(tp.critic_language_from_layer, critic_depth))
            if tp.critic_language_from_layer is not None else []
        )
        cr_vt_layers = (
            list(range(tp.critic_vision_from_layer, _CRITIC_VISION_TOWER_DEPTH))
            if critic_vision_on else []
        )

        for name, param in policy.named_parameters():
            param.requires_grad = (
                # Action expert — always trainable.
                "action_in_proj" in name
                or "action_out_proj" in name
                or "time_mlp_in" in name
                or "time_mlp_out" in name
                or "gemma_expert" in name
                # Actor vision: ViT layers >= N, plus multi-modal projector.
                or (vision_on and "paligemma" in name and "multi_modal_project" in name)
                or (
                    "paligemma" in name
                    and "vision_tower" in name
                    and any(f".{i}." in name for i in vt_layers)
                )
                # Actor language: blocks >= N, plus final norm.
                or ("language_model" in name and any(f".{i}." in name for i in lm_layers))
                or ("language_model.norm" in name and language_on)
                # Critic heads — always trainable.
                or "critic.norm" in name
                or "critic.value_head" in name
                or "critic.value_queries" in name
                or "critic.bin_logit_head" in name
                # Critic language blocks >= N.
                or ("critic.layers" in name and any(f".{i}." in name for i in cr_layers))
                # Critic vision: ViT layers >= N, plus projector.
                or (critic_vision_on and name.startswith("critic.") and "multi_modal_project" in name)
                or (
                    name.startswith("critic.vision_tower")
                    and any(f".{i}." in name for i in cr_vt_layers)
                )
            )

    def get_optimizer_groups(self, policy: nn.Module, cfg) -> list[dict]:
        actor: nn.Module = getattr(policy, "actor")
        critic_ensemble: nn.Module = getattr(policy, "critic_ensemble")
        return [
            {
                "params": [p for p in actor.parameters() if p.requires_grad],
                "lr": cfg.policy.optimizer_lr,
                "name": "policy",
            },
            {
                "params": [p for p in critic_ensemble.parameters() if p.requires_grad],
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
        """Critic update with gradient accumulation. Distributional TD (HL-Gauss)."""
        grad_accum = int(cfg.policy.gradient_accumulation_steps)
        clip_norm = float(cfg.policy.optimizer_grad_clip_norm)
        cast_fn = kwargs.get("cast_to_bf16_fn")

        accum_loss = 0.0
        accum_ce = 0.0
        accum_mse = 0.0
        critic_values_list: list[torch.Tensor] = []
        td_error_list: list[torch.Tensor] = []
        target_values_list: list[torch.Tensor] = []
        loss_raw_list: list[torch.Tensor] = []

        optimizers["critic"].zero_grad(set_to_none=True)

        for _ in range(grad_accum):
            batch, actions, rewards, observations, next_observations, done = self._prepare_batch(
                online_iter, offline_iter, device, cfg, cast_fn
            )
            fwd = preprocess_batch_for_pi05(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                done=done,
                task=cfg.policy.task,
            )
            out = policy.forward(fwd, model="critic")
            (out["loss_critic"] / grad_accum).backward()

            accum_loss += out["loss_critic"].detach().item()
            accum_ce += _scalar(out.get("loss_critic_ce"))
            accum_mse += _scalar(out.get("loss_critic_mse"))
            critic_values_list.append(out["critic_values"].detach())
            td_error_list.append(out["td_error"].detach())
            target_values_list.append(out["target_values"].detach())
            if "loss_critic_raw" in out:
                loss_raw_list.append(out["loss_critic_raw"].detach())

        critic_ensemble: nn.Module = getattr(policy, "critic_ensemble")
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_ensemble.parameters(), clip_norm
        ).item()
        optimizers["critic"].step()
        optimizers["critic"].zero_grad(set_to_none=True)

        all_cv = torch.cat(critic_values_list, dim=0)
        all_td = torch.cat(td_error_list, dim=0)
        all_tv = torch.cat(target_values_list, dim=0)

        infos: dict[str, Any] = {
            "loss_critic": accum_loss / grad_accum,
            "loss_critic_ce": accum_ce / grad_accum,
            "loss_critic_mse": accum_mse / grad_accum,
            "critic_grad_norm": critic_grad_norm,
            "td_error_mean": all_td.mean().item(),
            "td_error_std": all_td.std().item() if all_td.numel() > 1 else 0.0,
            "critic_value_mean": all_cv.mean().item(),
            "critic_value_std": all_cv.std().item() if all_cv.numel() > 1 else 0.0,
            "target_value_mean_critic": all_tv.mean().item(),
            "target_value_std": all_tv.std().item() if all_tv.numel() > 1 else 0.0,
            "critic_histogram_from_critic": all_cv.float().cpu().numpy(),
            "target_value_histogram": all_tv.float().cpu().numpy(),
        }
        if loss_raw_list:
            infos["loss_critic_raw"] = torch.cat(loss_raw_list, dim=0).float().cpu().numpy()
        return infos

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
        TD advantage with intervention and golden-dataset overrides.

        Intervention transitions and golden-dataset transitions receive
        raw_advantage = 1.0 regardless of critic output, bypassing the critic pass
        entirely when all transitions in the batch are golden.
        """
        comp_data = batch.get(TransitionKey.COMPLEMENTARY_DATA) or batch.get("complementary_info", {})
        is_golden_mask = None

        if "is_golden" in comp_data:
            is_golden = comp_data["is_golden"]
            is_golden_mask = (is_golden > 0.5)
            if is_golden_mask.shape != rewards.shape:
                is_golden_mask = is_golden_mask.view(rewards.shape)
            if is_golden_mask.all():
                raw_adv = torch.ones_like(rewards).view(-1, 1)
                squashed = torch.tanh(raw_adv / cfg.policy.advantage_scaling)
                zeros = torch.zeros_like(raw_adv)
                return raw_adv, squashed, {"current_v": zeros, "target_v": zeros, "raw_adv_flat": raw_adv.view(-1)}

        fwd = preprocess_batch_for_pi05(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
            actions=batch[ACTION],
            rewards=rewards,
            done=done,
            task=cfg.policy.task,
        )
        with torch.no_grad():
            critic_out = policy.forward(fwd, model="critic")
            target_v = critic_out["target_values"]
            current_v = critic_out["critic_values"]
            raw_adv = target_v - current_v

            if is_golden_mask is not None:
                if is_golden_mask.shape != raw_adv.shape:
                    is_golden_mask = is_golden_mask.view(raw_adv.shape)
                raw_adv[is_golden_mask] = 1.0

            intervention_key = TeleopEvents.IS_INTERVENTION.value
            if intervention_key in comp_data:
                mask = (comp_data[intervention_key] > 0.5)
                if mask.shape != raw_adv.shape:
                    mask = mask.view(raw_adv.shape)
                raw_adv[mask] = 1.0

        squashed = torch.tanh(raw_adv / cfg.policy.advantage_scaling)
        value_info = {"current_v": current_v, "target_v": target_v, "raw_adv_flat": raw_adv.view(-1)}
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
        Build the PI05 actor training batch.

        Steps:
          1. Apply anchor/delta action encoding if configured.
          2. Hydrate subtask strings from dataset metadata.
          3. Preprocess through the PI05 preprocessor (tokenization + normalization).
          4. For online transitions that carried exact subtask tokens, override the
             preprocessor's tokenization with the actor's original tokens so the flow
             loss sees the exact same conditioning the actor used.
          5. Zero subtask loss masks for online (no-subtask, index == -1) transitions.
          6. Assemble the forward batch consumed by PI05RLPolicy.forward(model="actor").
        """
        action_encoding = getattr(cfg.policy, "action_encoding", "absolute")
        if action_encoding in ("anchor", "delta"):
            if OBS_STATE in observations:
                anchor_state = observations[OBS_STATE]
                if action_encoding == "anchor":
                    actions = actions - anchor_state[:, None, :]
                else:
                    d_0 = actions[:, 0, :] - anchor_state
                    d_rest = torch.diff(actions, dim=1) if actions.shape[1] > 1 else None
                    actions = torch.cat([d_0.unsqueeze(1), d_rest], dim=1) if d_rest is not None else d_0.unsqueeze(1)
            else:
                logging.warning(f"action_encoding={action_encoding!r} but {OBS_STATE} not in observations.")

        comp_data = raw_batch.get(TransitionKey.COMPLEMENTARY_DATA) or raw_batch.get("complementary_info")
        subtasks = [""] * actions.shape[0]
        subtask_indices = None
        if comp_data is not None and "subtask_index" in comp_data:
            subtask_indices = comp_data["subtask_index"]
            subtasks = self._hydrate_subtasks(subtask_indices, dataset)

        batch_for_proc = {
            **observations,
            ACTION: actions,
            TransitionKey.COMPLEMENTARY_DATA: {
                "subtask": subtasks,
                "task": [cfg.policy.task] * actions.shape[0],
                "advantage": advantage.view(-1),
            },
        }
        with torch.no_grad():
            processed = preprocessor(batch_for_proc)

        # Online transitions carry exact tokens the actor used — bypass preprocessor tokenization
        # so the flow loss is computed against the identical conditioning context.
        if comp_data is not None and "subtask_tokens" in comp_data:
            ref_dev = processed[OBS_LANGUAGE_TOKENS].device
            ref_dtype = processed[OBS_LANGUAGE_ATTENTION_MASK].dtype
            processed[OBS_LANGUAGE_SUBTASK_TOKENS] = comp_data["subtask_tokens"].to(
                device=ref_dev, dtype=torch.long
            )
            processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = comp_data["subtask_masks"].to(
                device=ref_dev, dtype=ref_dtype
            )

        # Suppress subtask CE loss for online transitions (subtask_index == -1).
        if subtask_indices is not None and OBS_LANGUAGE_SUBTASK_ATTENTION_MASK in processed:
            online_mask = (subtask_indices.view(-1) == -1)
            if online_mask.any():
                processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK][online_mask] = False

        # Assemble the dict that PI05RLPolicy.forward(model="actor") expects.
        forward_batch: dict[str, Any] = {
            ACTION: processed[ACTION],
            "reward": raw_batch.get("reward"),
            "done": raw_batch.get("done"),
            "next.done": raw_batch.get("done"),
            "task": [cfg.policy.task] * actions.shape[0],
            "advantage": advantage,
            "state": {**observations},
            "next_state": {},
        }

        # Language tokens — available both flat and inside state for PI05's forward.
        for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
            if key in processed:
                forward_batch[key] = processed[key]
                forward_batch["state"][key] = processed[key]

        if "critic_tokens" in processed:
            forward_batch["critic_tokens"] = processed["critic_tokens"]
            forward_batch["critic_pad_mask"] = processed["critic_pad_mask"]

        if OBS_LANGUAGE_SUBTASK_TOKENS in processed:
            for key in (OBS_LANGUAGE_SUBTASK_TOKENS, OBS_LANGUAGE_SUBTASK_ATTENTION_MASK):
                forward_batch[key] = processed[key]
                forward_batch["state"][key] = processed[key]

        comp_out = processed.get(TransitionKey.COMPLEMENTARY_DATA, {})
        token_src = comp_out if ACTION_TOKENS in comp_out else processed
        if ACTION_TOKENS in token_src:
            forward_batch[ACTION_TOKENS] = token_src[ACTION_TOKENS]
            forward_batch[ACTION_TOKEN_MASK] = token_src[ACTION_TOKEN_MASK]

        return forward_batch

    def actor_forward(
        self,
        policy: nn.Module,
        batch: dict,
        advantage: torch.Tensor,
        cfg,
    ) -> dict[str, Any]:
        """
        Actor forward pass. Returns the full output dict from PI05RLPolicy.forward
        so update_actor can access raw flow diagnostics without a second forward call.
        """
        squashed = torch.tanh(advantage / cfg.policy.advantage_scaling)
        external_metrics = {
            "advantage": squashed,
            "critic_values": batch.get("_current_v"),
            "target_values": batch.get("_target_v"),
            "rewards": batch.get("reward"),
        }
        return policy.forward(batch, model="actor", external_metrics=external_metrics)

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
        Actor update with gradient accumulation and policy_update_freq repetitions.

        Each repetition samples a fresh batch; metrics are returned from the last
        repetition only (standard behaviour matching the old _update_actor).
        """
        grad_accum = int(cfg.policy.gradient_accumulation_steps)
        policy_update_freq = int(cfg.policy.policy_update_freq)
        clip_norm = float(cfg.policy.optimizer_grad_clip_norm)
        cast_fn = kwargs.get("cast_to_bf16_fn")

        infos: dict[str, Any] = {}

        for _ in range(policy_update_freq):
            optimizers["policy"].zero_grad(set_to_none=True)

            accum_loss = 0.0
            accum_flow = 0.0
            accum_action_ce = 0.0
            accum_subtask_ce = 0.0
            accum_flow_first_10: float | None = None
            accum_flow_last_10: float | None = None
            accum_flow_time_low = 0.0
            accum_flow_time_high = 0.0
            count_flow_low = 0
            count_flow_high = 0
            advantage_list: list[torch.Tensor] = []
            critic_values_list: list[torch.Tensor] = []
            target_values_list: list[torch.Tensor] = []
            reward_list: list[torch.Tensor] = []

            for _ in range(grad_accum):
                batch, actions, rewards, observations, next_observations, done = self._prepare_batch(
                    online_iter, offline_iter, device, cfg, cast_fn
                )

                raw_adv, squashed_adv, value_info = self.compute_advantage(
                    policy=policy,
                    batch=batch,
                    observations=observations,
                    next_observations=next_observations,
                    rewards=rewards,
                    done=done,
                    cfg=cfg,
                )

                fwd_batch = self.build_training_batch(
                    raw_batch=batch,
                    observations=observations,
                    actions=actions,
                    advantage=raw_adv,
                    preprocessor=preprocessor,
                    dataset=dataset,
                    cfg=cfg,
                )
                # Thread value info into the batch so actor_forward can build external_metrics.
                fwd_batch["_current_v"] = value_info["current_v"]
                fwd_batch["_target_v"] = value_info["target_v"]
                fwd_batch["reward"] = rewards

                actor_out = self.actor_forward(policy, fwd_batch, raw_adv, cfg)

                loss = actor_out["loss_actor"]
                (loss.mean() / grad_accum).backward()

                accum_loss += loss.mean().item()
                accum_flow += _scalar(actor_out.get("flow_mse_loss"))
                accum_action_ce += _scalar(actor_out.get("action_ce_loss"))
                accum_subtask_ce += _scalar(actor_out.get("subtask_ce_loss"))

                # Fine-grained flow diagnostics (time-sliced and noise-level).
                flow_raw = actor_out.get("flow_loss_raw")
                flow_time = actor_out.get("flow_time")
                if isinstance(flow_raw, torch.Tensor) and isinstance(flow_time, torch.Tensor):
                    T = flow_raw.shape[1]
                    edge = min(10, T)
                    chunk_first = flow_raw[:, :edge, :].mean().item()
                    chunk_last = flow_raw[:, -edge:, :].mean().item()
                    accum_flow_first_10 = chunk_first if accum_flow_first_10 is None else accum_flow_first_10 + chunk_first
                    accum_flow_last_10 = chunk_last if accum_flow_last_10 is None else accum_flow_last_10 + chunk_last

                    mask_low = flow_time < 0.3
                    mask_high = flow_time > 0.7
                    if mask_low.any():
                        accum_flow_time_low += flow_raw[mask_low].mean().item() * mask_low.sum().item()
                        count_flow_low += mask_low.sum().item()
                    if mask_high.any():
                        accum_flow_time_high += flow_raw[mask_high].mean().item() * mask_high.sum().item()
                        count_flow_high += mask_high.sum().item()

                advantage_list.append(squashed_adv.detach().view(-1))
                critic_values_list.append(value_info["current_v"].detach().view(-1))
                target_values_list.append(value_info["target_v"].detach().view(-1))
                reward_list.append(rewards.detach().view(-1))

            actor_mod: nn.Module = getattr(policy, "actor")
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor_mod.parameters(), clip_norm
            ).item()
            optimizers["policy"].step()
            optimizers["policy"].zero_grad(set_to_none=True)

            all_adv = torch.cat(advantage_list)
            all_cv = torch.cat(critic_values_list)
            all_tv = torch.cat(target_values_list)
            all_r = torch.cat(reward_list)

            infos = {
                "loss_actor": accum_loss / grad_accum,
                "flow_mse_loss": accum_flow / grad_accum,
                "action_ce_loss": accum_action_ce / grad_accum,
                "subtask_ce_loss": accum_subtask_ce / grad_accum,
                "actor_grad_norm": actor_grad_norm,
                "advantage_mean": all_adv.mean().item(),
                "advantage_std": all_adv.std().item() if all_adv.numel() > 1 else 0.0,
                "advantage_positive_label_fraction": (all_adv > 0.25).float().mean().item(),
                "advantage_histogram": all_adv.float().cpu().numpy(),
                "critic_value_mean_actor": all_cv.mean().item(),
                "critic_histogram": all_cv.float().cpu().numpy(),
                "target_value_mean": all_tv.mean().item(),
                "reward_mean": all_r.mean().item(),
            }
            if accum_flow_first_10 is not None and accum_flow_last_10 is not None:
                infos["flow_loss_time/mean_first_10"] = accum_flow_first_10 / grad_accum
                infos["flow_loss_time/mean_last_10"] = accum_flow_last_10 / grad_accum
            if count_flow_low > 0:
                infos["flow_loss_noise/mean_low_noise_lt_0.3"] = accum_flow_time_low / count_flow_low
            if count_flow_high > 0:
                infos["flow_loss_noise/mean_high_noise_gt_0.7"] = accum_flow_time_high / count_flow_high

        return infos

    def update_target_networks(self, policy: nn.Module) -> None:
        """Polyak update: critic_target ← τ*critic + (1-τ)*critic_target."""
        self._target_update_call_counter = getattr(self, "_target_update_call_counter", 0) + 1
        policy_config = getattr(policy, "config")
        every = int(getattr(policy_config, "critic_target_update_every", 1))
        if every > 1 and self._target_update_call_counter % every != 0:
            return
        tau: float = float(getattr(policy_config, "critic_target_update_weight"))
        critic: nn.Module = getattr(policy, "critic")
        critic_target: nn.Module = getattr(policy, "critic_target")
        with torch.no_grad():
            for p, p_tgt in zip(critic.parameters(), critic_target.parameters()):
                if not p.requires_grad:
                    continue
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
        Assemble the inference batch for PI05.
        context kwargs: preprocessor (required), prev_actions, robot_type,
                        subtask_override, inference_advantage.
        """
        preprocessor = context["preprocessor"]
        batch = {**observation}
        batch["robot_type"] = context.get("robot_type", "so101")
        advantage = context.get("inference_advantage", cfg.policy.inference_advantage)
        if not isinstance(advantage, torch.Tensor):
            advantage = torch.tensor([[advantage]], dtype=torch.float32)
        batch[TransitionKey.COMPLEMENTARY_DATA] = {
            "task": [task_str],
            "subtask": [context.get("subtask_override", "")],
            "advantage": advantage,
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
        if not observations or ACTION not in transition:
            return None

        observations = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in observations.items()
        }
        next_observations = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in next_observations.items()
        }
        actions = transition[ACTION].to(device)
        if actions.ndim == 1:
            actions = actions.view(1, 1, -1)
        elif actions.ndim == 2:
            actions = actions.unsqueeze(0)
        rewards = torch.as_tensor(transition.get("reward", 0.0), device=device).view(1)
        done = torch.as_tensor(transition.get("done", False), device=device).view(1)

        batch = preprocess_batch_for_pi05(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            done=done,
            task=cfg.policy.task,
        )
        batch = move_transition_to_device(batch, device)  # type: ignore[arg-type]
        with torch.no_grad():
            out = policy.forward(batch, model="critic_value")
        return float(out["value"].mean().item())

    def push_weights(self, policy: nn.Module, parameters_queue) -> None:
        """Push trainable actor parameters to the actor queue."""
        actor: nn.Module = getattr(policy, "actor")
        trainable = {
            name: param
            for name, param in actor.named_parameters()
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
        """
        Log scalar and histogram metrics to W&B.

        The base Trainer.log_metrics handles the generic dispatch; this override
        just adds a console summary of the most important scalars first.
        """
        console_keys = ("loss_actor", "flow_mse_loss", "actor_grad_norm", "loss_critic")
        console_scalars = {
            k: training_infos[k]
            for k in console_keys
            if isinstance(training_infos.get(k), (int, float))
        }
        if console_scalars and step > 0:
            logging.info(
                f"[PI05Trainer] step={step}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in console_scalars.items())
            )
        super().log_metrics(training_infos, step, wandb_logger, _policy)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare_batch(
        self,
        online_iter,
        offline_iter,
        device: str,
        cfg,
        cast_fn,
    ) -> tuple:
        """Sample one batch (online + optional offline mix), move to device."""
        action_dim: int = cfg.policy.action_dim
        batch = next(online_iter)

        if offline_iter is not None:
            batch_off = next(offline_iter)
            batch_off[ACTION] = batch_off[ACTION][..., :action_dim]
            batch = concatenate_batch_transitions(batch, batch_off, action_dim=action_dim)

        batch = move_transition_to_device(batch, device)
        batch[ACTION] = batch[ACTION][..., :action_dim]

        if cast_fn is not None:
            batch = cast_fn(batch)

        actions = batch[ACTION]
        rewards = batch["reward"]
        done = batch["done"]
        observations = batch.get("state") or {
            k: v for k, v in batch.items() if k.startswith("observation.")
        }
        next_observations = batch.get("next_state") or {
            k: v for k, v in batch.items() if k.startswith("next.observation.")
        }
        return batch, actions, rewards, observations, next_observations, done

    @staticmethod
    def _hydrate_subtasks(indices, dataset) -> list[str]:
        """Convert subtask indices to description strings via dataset metadata."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        has_meta = (
            dataset is not None
            and hasattr(dataset, "meta")
            and hasattr(dataset.meta, "subtasks")
        )
        subtasks_df = dataset.meta.subtasks if has_meta else None
        names = []

        for i in indices:
            i = int(i)
            name = ""
            if i >= 0 and subtasks_df is not None:
                try:
                    if hasattr(subtasks_df, "columns") and "subtask_index" in subtasks_df.columns:
                        rows = subtasks_df[subtasks_df["subtask_index"] == i]
                        if not rows.empty:
                            name = rows.iloc[0].name
                    elif hasattr(subtasks_df, "index") and i in subtasks_df.index:
                        name = subtasks_df.iloc[i].name
                    elif hasattr(subtasks_df, "__getitem__"):
                        name = subtasks_df[i]
                except Exception:
                    pass
            names.append(name)

        return names

    @staticmethod
    def _unflatten_stats(stats: dict) -> dict:
        """Convert flat stat keys (e.g. 'action.mean') to nested dicts."""
        out: dict = {}
        for key, value in stats.items():
            if "." in key:
                prefix, suffix = key.rsplit(".", 1)
                out.setdefault(prefix, {})[suffix] = value
            else:
                out[key] = value
        return out


# ── Module-level helpers ───────────────────────────────────────────────────────

def _scalar(v) -> float:
    """Safely extract a scalar from a tensor or return the value as-is."""
    if isinstance(v, torch.Tensor):
        return v.detach().float().mean().item()
    return float(v) if v is not None else 0.0
