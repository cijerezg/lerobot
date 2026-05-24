"""
MolmoAct2Trainer — concrete Trainer for MolmoAct2RLPolicy.

Actor-only mode     (skip_critic=True):  update_actor only; no critic updates.
Critic-trained mode (skip_critic=False): update_critic (HL-Gauss distributional TD)
                                         + actor with advantage binned into the prompt.
"""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.molmoact2.frame_so101 import stats_v3_to_v21
from lerobot.rl.rl_trainer import Trainer
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    PRETRAINED_MODEL_DIR,
)
from lerobot.utils.transition import move_transition_to_device


# Depths for the SO100/SO101 MolmoAct2 checkpoint. Adjust if you point this
# trainer at a different variant.
_MOLMOACT2_VIT_DEPTH = 25        # image_vit.transformer.resblocks.{0..24}
_MOLMOACT2_LANGUAGE_DEPTH = 36   # transformer.blocks.{0..35}


def _layer_idx_after(name: str, marker: str) -> int:
    """Parse the integer layer index immediately after `marker` in `name`."""
    return int(name.split(marker)[1].split(".")[0])


def _dataset_stats(dataset: Any | None) -> dict[str, dict[str, Any]] | None:
    if dataset is None:
        return None
    stats = getattr(dataset, "stats", None)
    if stats is None:
        meta = getattr(dataset, "meta", None)
        if meta is not None:
            stats = getattr(meta, "stats", None)
    # Dataset stats live in v3.0 joint frame; the MolmoAct2 normalizer sees
    # tensors after SO101V3ToV21Step, so the stats must be converted to match.
    return stats_v3_to_v21(stats)


def _has_saved_processors(path: Path) -> bool:
    preprocessor_path = path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    return preprocessor_path.exists() and postprocessor_path.exists()


def _saved_processor_path(cfg: Any) -> Path | None:
    candidates: list[Path] = []

    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    if pretrained_path is not None:
        candidates.append(Path(pretrained_path).expanduser())

    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser()
        candidates.extend([checkpoint_path, checkpoint_path / PRETRAINED_MODEL_DIR])

    if getattr(cfg, "output_dir", None):
        candidates.append(
            Path(cfg.output_dir).expanduser()
            / CHECKPOINTS_DIR
            / LAST_CHECKPOINT_LINK
            / PRETRAINED_MODEL_DIR
        )

    for path in candidates:
        if path.is_dir() and _has_saved_processors(path):
            return path
    return None


_EXPECTED_FRAME_CONVERSION = "so101_v3_to_v21"
_STATS_METADATA_KEYS = ("encoding", "frame_conversion", "chunk_size")


def _validate_and_strip_anchor_stats_metadata(
    stats: dict[str, Any], stats_path: str, cfg_policy: Any
) -> dict[str, Any]:
    """Pop metadata stamped by compute_delta_stats.py, validating against the policy config.

    MolmoAct2 always trains on v2.1-frame stats, so frame_conversion must be
    ``so101_v3_to_v21``. Raises with a clear message on any mismatch so a wrong
    stats file fails before training rather than silently mis-aligning.
    """
    meta = {k: stats.pop(k, None) for k in _STATS_METADATA_KEYS}
    expected_encoding = getattr(cfg_policy, "action_encoding", "absolute")
    expected_chunk = getattr(cfg_policy, "chunk_size", None)

    if meta["encoding"] is None and meta["frame_conversion"] is None:
        raise ValueError(
            f"Stats file {stats_path} has no encoding/frame_conversion metadata. "
            f"It was likely produced by an older compute_delta_stats.py; re-run with "
            f"--encoding {expected_encoding} --frame-conversion {_EXPECTED_FRAME_CONVERSION}."
        )

    if meta["encoding"] != expected_encoding:
        raise ValueError(
            f"Stats file {stats_path} has encoding={meta['encoding']!r} but "
            f"cfg.policy.action_encoding={expected_encoding!r}."
        )
    if meta["frame_conversion"] != _EXPECTED_FRAME_CONVERSION:
        raise ValueError(
            f"Stats file {stats_path} has frame_conversion={meta['frame_conversion']!r} but "
            f"MolmoAct2 requires {_EXPECTED_FRAME_CONVERSION!r}. Re-run compute_delta_stats.py "
            f"with --frame-conversion {_EXPECTED_FRAME_CONVERSION}."
        )
    if meta["chunk_size"] is not None and expected_chunk is not None and meta["chunk_size"] != expected_chunk:
        raise ValueError(
            f"Stats file {stats_path} has chunk_size={meta['chunk_size']} but "
            f"cfg.policy.chunk_size={expected_chunk}."
        )
    return stats


def _override_action_stats(processors: tuple, action_stats: dict[str, Any]) -> tuple:
    for pipeline in processors:
        for step in getattr(pipeline, "steps", []):
            stats = getattr(step, "stats", None)
            if not isinstance(stats, dict):
                continue
            updated_stats = deepcopy(stats)
            updated_action_stats = deepcopy(action_stats)
            existing_action_stats = stats.get(ACTION)
            if (
                isinstance(updated_action_stats, dict)
                and isinstance(existing_action_stats, dict)
                and "mask" not in updated_action_stats
                and "mask" in existing_action_stats
            ):
                updated_action_stats["mask"] = existing_action_stats["mask"]
            updated_stats[ACTION] = updated_action_stats
            step.stats = updated_stats
            to_fn = getattr(step, "to", None)
            if callable(to_fn):
                to_fn(device=getattr(step, "device", None), dtype=getattr(step, "dtype", None))
    return processors


class MolmoAct2Trainer(Trainer):
    """Trainer for MolmoAct2RLPolicy. Supports actor-only and critic-trained modes."""

    # Value histograms live inside the distributional critic's support; clip to
    # the configured default so a single outlier doesn't flatten the bin range.
    _HISTOGRAM_CLIP_RANGES = {
        "critic_value_histogram_from_critic": (-2.0, 0.1),
        "critic_value_histogram_from_actor": (-2.0, 0.1),
        "target_value_histogram": (-2.0, 0.1),
        "target_value_histogram_actor": (-2.0, 0.1),
        "v_next_histogram_actor": (-2.0, 0.1),
        "flow_loss_per_sample_histogram": (0.0, 0.01),
        "loss_critic_histogram_flat": (0.0, 0.005),
    }

    # ── Setup ─────────────────────────────────────────────────────────────────

    def make_processors(self, cfg, dataset=None, is_main_process: bool = True) -> tuple:  # noqa: ARG002
        """
        Build MolmoAct2 processors with stable normalization stats.

        Dataset stats initialize base/original MolmoAct2 runs. Once a LeRobot
        checkpoint exists, its saved processor stats carry the normalization
        contract forward. Anchor/delta action encoding stats are explicit and
        always override ACTION stats.

        The standard MolmoAct2 preprocessor is already safe for batched replay-buffer
        inputs: AddBatchDimensionProcessorStep only adds a dim to 1-D / 3-D tensors,
        so batched images [B,C,H,W] and states [B,D] pass through unchanged.
        """
        import os

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors

        dataset_stats = _dataset_stats(dataset)
        dataset_meta = getattr(dataset, "meta", None) if dataset is not None else None

        action_stats_override = None
        action_encoding = getattr(cfg.policy, "action_encoding", "absolute")
        if action_encoding in ("anchor", "delta"):
            stats_path = getattr(cfg.policy, "action_encoding_stats_path", None)
            if not stats_path or not os.path.exists(os.path.expanduser(stats_path)):
                raise ValueError(
                    f"action_encoding={action_encoding} but action_encoding_stats_path "
                    f"{stats_path!r} is invalid or does not exist!"
                )
            if is_main_process:
                logging.info(f"Loading {action_encoding} action stats from {stats_path}")
            action_stats_override = torch.load(os.path.expanduser(stats_path), map_location="cpu")
            action_stats_override = _validate_and_strip_anchor_stats_metadata(
                action_stats_override, stats_path, cfg.policy
            )

        # Saved-processor specs were serialized before AnchorEncodeStep existed; for
        # anchor/delta encoding we must rebuild from scratch so the step is present.
        # Absolute encoding keeps the old fast-path.
        saved_processor_path = (
            None if action_encoding in ("anchor", "delta") else _saved_processor_path(cfg)
        )
        if saved_processor_path is not None:
            if is_main_process:
                logging.info(f"MolmoAct2 stats source: saved checkpoint processors ({saved_processor_path})")
            result = make_pre_post_processors(cfg.policy, pretrained_path=str(saved_processor_path))
            if action_stats_override is not None:
                if is_main_process:
                    logging.info(f"MolmoAct2 ACTION stats override: {action_encoding} stats")
                result = _override_action_stats(result, action_stats_override)
        else:
            if is_main_process:
                if dataset_stats is not None:
                    logging.info("MolmoAct2 stats source: dataset stats")
                elif str(getattr(cfg.policy, "norm_tag", "") or "").strip():
                    logging.info(f"MolmoAct2 stats source: norm_tag={cfg.policy.norm_tag!r}")
                else:
                    logging.info("MolmoAct2 stats source: none")
            result = make_molmoact2_pre_post_processors(
                config=cfg.policy,
                dataset_stats=dataset_stats,
                dataset_meta=dataset_meta,
                action_stats_override=action_stats_override,
            )
        self._preprocessor = result[0]  # cache for critic_value_for_logging
        return result

    def make_policy(self, cfg) -> nn.Module:
        from lerobot.policies.factory import make_policy
        return make_policy(cfg=cfg.policy, env_cfg=cfg.env)

    def freeze_model(self, policy: nn.Module, cfg) -> None:
        """
        Two paths:

        1. `cfg.policy.trainable_params is None` (default) → keep the coarse
           freeze applied during `_load_hf_model`:
             - train_action_expert_only → _freeze_non_action_expert_parameters()
             - enable_lora_vlm → LoRA adapters (non-LoRA params frozen)
             - freeze_embedding → _freeze_input_embeddings()

        2. `cfg.policy.trainable_params` is set → authoritative per-name freeze
           following TrainableParamsConfig. Action expert always trains;
           embeddings governed by `freeze_embedding`; everything else gated
           by the schedule (unknown params default to frozen).

        Critic-trained mode (skip_critic=False): critic params are then
        unfrozen — either fully (path 1) or per the critic_* schedule (path 2).
        critic_target stays frozen.
        """
        skip_critic: bool = bool(getattr(cfg, "skip_critic", True))
        tp = getattr(cfg.policy, "trainable_params", None)
        freeze_embedding: bool = bool(getattr(cfg.policy, "freeze_embedding", True))

        if tp is not None:
            self._apply_actor_freeze(policy, tp, freeze_embedding=freeze_embedding)

        if not skip_critic and hasattr(policy, "critic"):
            critic_net: nn.Module = getattr(policy, "critic")
            if tp is not None:
                self._apply_critic_freeze(critic_net, tp, cfg)
            else:
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

    @staticmethod
    def _apply_actor_freeze(policy: nn.Module, tp, *, freeze_embedding: bool) -> None:
        vision_on = tp.vision_from_layer is not None
        language_on = tp.language_from_layer is not None
        vt_layers = (
            set(range(tp.vision_from_layer, _MOLMOACT2_VIT_DEPTH)) if vision_on else set()
        )
        lm_layers = (
            set(range(tp.language_from_layer, _MOLMOACT2_LANGUAGE_DEPTH)) if language_on else set()
        )

        for name, param in policy.named_parameters():
            if name.startswith("critic.") or name.startswith("critic_target."):
                continue  # critic handled separately

            if ".action_expert." in name:
                param.requires_grad = True
            elif ".transformer.wte" in name:
                param.requires_grad = not freeze_embedding
            elif ".lm_head" in name or name.endswith("lm_head.weight"):
                param.requires_grad = language_on
            elif ".transformer.blocks." in name:
                param.requires_grad = _layer_idx_after(name, ".transformer.blocks.") in lm_layers
            elif ".transformer.ln_f" in name:
                param.requires_grad = language_on
            elif ".image_vit.transformer.resblocks." in name:
                param.requires_grad = _layer_idx_after(name, ".resblocks.") in vt_layers
            elif ".image_vit.patch_embedding" in name or ".image_vit.positional_embedding" in name:
                param.requires_grad = vision_on
            elif ".image_pooling_2d" in name or ".image_projector" in name:
                param.requires_grad = vision_on
            else:
                # Unknown actor param — freeze. Safer than accidentally training
                # something the schedule didn't anticipate.
                param.requires_grad = False

    @staticmethod
    def _apply_critic_freeze(critic_net: nn.Module, tp, cfg) -> None:
        cr_vision_on = tp.critic_vision_from_layer is not None
        cr_language_on = tp.critic_language_from_layer is not None
        cr_vt_layers = (
            set(range(tp.critic_vision_from_layer, _MOLMOACT2_VIT_DEPTH))
            if cr_vision_on else set()
        )
        cr_layers = (
            set(range(tp.critic_language_from_layer, int(getattr(cfg.policy, "critic_llm_depth", 12))))
            if cr_language_on else set()
        )

        for name, param in critic_net.named_parameters():
            # Always-trainable critic heads.
            if name.startswith("value_queries") or name.startswith("bin_logit_head"):
                param.requires_grad = True
            elif "transformer_blocks." in name:
                param.requires_grad = _layer_idx_after(name, "transformer_blocks.") in cr_layers
            elif name.startswith("ln_f"):
                param.requires_grad = cr_language_on
            elif ".image_vit.transformer.resblocks." in name:
                param.requires_grad = _layer_idx_after(name, ".resblocks.") in cr_vt_layers
            elif ".image_vit.patch_embedding" in name or ".image_vit.positional_embedding" in name:
                param.requires_grad = cr_vision_on
            elif ".image_pooling_2d" in name or ".image_projector" in name:
                param.requires_grad = cr_vision_on
            else:
                param.requires_grad = False

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
        v_curr_list: list[torch.Tensor] = []
        td_target_list: list[torch.Tensor] = []
        td_error_list: list[torch.Tensor] = []
        loss_per_sample_list: list[torch.Tensor] = []

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

            # V(s') from frozen target — no grad. Match PI05 target semantics:
            # clamp scalar TD target into support, and use one-hot bins for
            # exact terminal targets instead of HL-Gauss smoothing.
            with torch.no_grad():
                target_out = _fwd_target(next_batch)
                v_next = target_out["value"].to(rewards.dtype)
                td_target_raw = rewards + discount * v_next * (1.0 - done.float())
                v_min = float(getattr(cfg.policy, "value_support_min", -2.0))
                v_max = float(getattr(cfg.policy, "value_support_max", 0.0))
                td_target = td_target_raw.clamp(min=v_min, max=v_max)
                critic_net: nn.Module = getattr(policy, "critic")
                hl_target = critic_net.hl_gauss_target(td_target).to(dtype=torch.float32)  # type: ignore[attr-defined]
                one_hot_target = critic_net.one_hot_target(td_target).to(dtype=torch.float32)  # type: ignore[attr-defined]
                soft_target = torch.where(done.bool(), one_hot_target, hl_target)

            # V(s) with grad.
            critic_out = _fwd_critic(curr_batch)
            logits = critic_out["logits"]  # [B, num_bins]

            ce_per_sample = -(
                soft_target.to(logits.device) * F.log_softmax(logits.float(), dim=-1)
            ).sum(dim=-1)
            loss_ce = ce_per_sample.mean()
            (loss_ce / grad_accum).backward()

            accum_ce += loss_ce.item() / grad_accum

            with torch.no_grad():
                v_curr_step = critic_out["value"].float().view(-1)
                td_target_step = td_target.float().view(-1)
                v_curr_list.append(v_curr_step)
                td_target_list.append(td_target_step)
                td_error_list.append(td_target_step - v_curr_step)
                loss_per_sample_list.append(ce_per_sample.detach().float().view(-1))

        critic_net2: nn.Module = getattr(policy, "critic")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(critic_net2.parameters()), clip_norm
        ).item()
        optimizers["critic"].step()

        all_v_curr = torch.cat(v_curr_list)
        all_td_target = torch.cat(td_target_list)
        all_td_error = torch.cat(td_error_list)
        all_loss_per_sample = torch.cat(loss_per_sample_list)

        return {
            "loss_critic": accum_ce,
            "loss_critic_ce": accum_ce,
            "loss_critic_mse": all_td_error.square().mean().item(),
            "critic_grad_norm": grad_norm,
            "critic_value_mean": all_v_curr.mean().item(),
            "critic_value_std": all_v_curr.std().item() if all_v_curr.numel() > 1 else 0.0,
            "target_value_mean": all_td_target.mean().item(),
            "target_value_std": all_td_target.std().item() if all_td_target.numel() > 1 else 0.0,
            "td_error_mean": all_td_error.abs().mean().item(),
            "td_error_std": all_td_error.std().item() if all_td_error.numel() > 1 else 0.0,
            "critic_value_histogram_from_critic": all_v_curr.cpu().numpy(),
            "target_value_histogram": all_td_target.cpu().numpy(),
            "td_error_histogram": all_td_error.cpu().numpy(),
            "loss_critic_histogram_flat": all_loss_per_sample.cpu().numpy(),
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

        td_target = rewards + discount * v_next * (1.0 - done.float())
        raw_adv = td_target - v_curr
        squashed_adv = torch.tanh(raw_adv / adv_scale)

        value_info = {
            "advantage_raw_mean": raw_adv.mean().item(),
            "advantage_squashed_mean": squashed_adv.mean().item(),
            "v_curr_mean": v_curr.mean().item(),
            "v_next_mean": v_next.mean().item(),
            "critic_values": v_curr.detach().float().view(-1),
            "target_values": td_target.detach().float().view(-1),
            "v_next_values": v_next.detach().float().view(-1),
        }
        return raw_adv, squashed_adv, value_info

    # ── Actor ─────────────────────────────────────────────────────────────────

    def build_training_batch(
        self,
        raw_batch: dict,  # noqa: ARG002
        observations: dict,
        actions: torch.Tensor,
        advantage: torch.Tensor | None,
        preprocessor,
        dataset,  # noqa: ARG002
        cfg,
    ) -> dict[str, Any]:
        """
        Build the MolmoAct2 forward batch.

        Flat dict with image/state observation keys + action + task string.
        In RL mode, the raw advantage tensor is threaded via complementary_data
        so the prompt step bins it into a "negative"/"positive" clause.
        """
        from lerobot.types import TransitionKey

        action_dim = self._action_dim(cfg)
        actions = actions[..., :action_dim]

        pre_input: dict[str, Any] = {
            **observations,
            "action": actions,
            "task": cfg.policy.task,
        }
        if advantage is not None:
            pre_input[TransitionKey.COMPLEMENTARY_DATA] = {"advantage": advantage}
        return preprocessor(pre_input)

    def actor_forward(
        self,
        policy: nn.Module,
        batch: dict,
        advantage: torch.Tensor,  # noqa: ARG002 — advantage is consumed by the preprocessor as a prompt
        cfg,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Forward pass. Advantage conditioning is applied via the prompt in build_training_batch."""
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

        Actor-only mode (skip_critic=True): no critic-computed advantage clause in the prompt.
        Critic-trained mode (skip_critic=False): raw advantage is threaded into the batch and
                                                 the preprocessor inserts a label clause.
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
            "loss_discrete_z": 0.0,
            "advantage_squashed_mean": 0.0,
        }
        squashed_adv_list: list[torch.Tensor] = []
        raw_adv_list: list[torch.Tensor] = []
        actor_loss_list: list[torch.Tensor] = []
        flow_loss_per_sample_list: list[torch.Tensor] = []
        flow_loss_per_timestep_list: list[torch.Tensor] = []
        flow_timesteps_list: list[torch.Tensor] = []
        flow_loss_per_action_step_list: list[torch.Tensor] = []
        discrete_ce_loss_list: list[torch.Tensor] = []
        discrete_z_loss_list: list[torch.Tensor] = []
        reward_list: list[torch.Tensor] = []
        done_list: list[torch.Tensor] = []
        critic_values_actor_list: list[torch.Tensor] = []
        target_values_actor_list: list[torch.Tensor] = []
        v_next_values_actor_list: list[torch.Tensor] = []

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
            if "reward" in raw:
                reward_tensor = raw["reward"]
                if not isinstance(reward_tensor, torch.Tensor):
                    reward_tensor = torch.tensor(reward_tensor, device=actions.device)
                reward_list.append(reward_tensor.detach().float().view(-1))
            if "done" in raw:
                done_tensor = raw["done"]
                if not isinstance(done_tensor, torch.Tensor):
                    done_tensor = torch.tensor(done_tensor, device=actions.device)
                done_list.append(done_tensor.detach().float().view(-1))

            # Advantage as prompt conditioning (RL mode only). Raw advantage is
            # threaded into the batch; the preprocessor bins it into a label.
            raw_adv: torch.Tensor | None = None
            if not skip_critic:
                next_observations = raw.get("next_state", {})
                rewards = raw["reward"]
                done = raw["done"]
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards)
                if not isinstance(done, torch.Tensor):
                    done = torch.tensor(done)
                raw_adv, squashed_adv, value_info = self.compute_advantage(
                    policy,
                    batch=raw,  # type: ignore[arg-type]
                    observations=observations,
                    next_observations=next_observations,
                    rewards=rewards,
                    done=done,
                    cfg=cfg,
                    preprocessor=preprocessor,
                )
                accum["advantage_squashed_mean"] += squashed_adv.mean().item() / grad_accum
                squashed_adv_list.append(squashed_adv.detach().float().view(-1))
                raw_adv_list.append(raw_adv.detach().float().view(-1))
                critic_values_actor_list.append(value_info["critic_values"])
                target_values_actor_list.append(value_info["target_values"])
                v_next_values_actor_list.append(value_info["v_next_values"])

            fwd_batch = self.build_training_batch(
                raw_batch=raw,  # type: ignore[arg-type]
                observations=observations,
                actions=actions,
                advantage=raw_adv,
                preprocessor=preprocessor,
                dataset=dataset,
                cfg=cfg,
            )

            # Keep actor loss clean: advantage is only a prompt label.
            loss, metrics = policy.forward(fwd_batch, reduction="none", return_diagnostics=True)
            loss_for_backward = loss.mean() if isinstance(loss, torch.Tensor) else torch.as_tensor(loss, device=actions.device)
            (loss_for_backward / grad_accum).backward()

            accum["loss_actor"] += float(metrics.get("loss", loss_for_backward.detach().float().item())) / grad_accum
            accum["loss_flow"] += float(metrics.get("action_flow_loss", 0.0)) / grad_accum
            accum["loss_discrete_ce"] += float(metrics.get("discrete_ce_loss", 0.0)) / grad_accum
            if "discrete_z_loss" in metrics:
                accum["loss_discrete_z"] += float(metrics["discrete_z_loss"]) / grad_accum

            actor_loss_raw = metrics.get("loss_raw", loss.detach().float() if isinstance(loss, torch.Tensor) else None)
            if isinstance(actor_loss_raw, torch.Tensor):
                actor_loss_list.append(actor_loss_raw.detach().float().view(-1))
            flow_loss_per_sample = metrics.get("flow_loss_per_sample")
            if isinstance(flow_loss_per_sample, torch.Tensor):
                flow_loss_per_sample_list.append(flow_loss_per_sample.detach().float().view(-1))
            flow_loss_per_timestep = metrics.get("flow_loss_per_timestep")
            flow_timesteps = metrics.get("flow_timesteps")
            if isinstance(flow_loss_per_timestep, torch.Tensor) and isinstance(flow_timesteps, torch.Tensor):
                flow_loss_per_timestep_list.append(flow_loss_per_timestep.detach().float().reshape(-1))
                flow_timesteps_list.append(flow_timesteps.detach().float().reshape(-1))
            flow_loss_per_action_step = metrics.get("flow_loss_per_action_step")
            if isinstance(flow_loss_per_action_step, torch.Tensor):
                flow_loss_per_action_step_list.append(flow_loss_per_action_step.detach().float())
            discrete_ce_loss_raw = metrics.get("discrete_ce_loss_raw")
            if isinstance(discrete_ce_loss_raw, torch.Tensor):
                discrete_ce_loss_list.append(discrete_ce_loss_raw.detach().float().view(-1))
            discrete_z_loss_raw = metrics.get("discrete_z_loss_raw")
            if isinstance(discrete_z_loss_raw, torch.Tensor):
                discrete_z_loss_list.append(discrete_z_loss_raw.detach().float().view(-1))

        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, clip_norm).item()
        policy_opt.step()

        accum["actor_grad_norm"] = actor_grad_norm

        if actor_loss_list:
            all_actor_loss = torch.cat(actor_loss_list)
            accum["actor_loss_histogram"] = all_actor_loss.cpu().numpy()
        if flow_loss_per_sample_list:
            all_flow_per_sample = torch.cat(flow_loss_per_sample_list)
            accum["flow_loss_per_sample_mean"] = all_flow_per_sample.mean().item()
            accum["flow_loss_per_sample_std"] = all_flow_per_sample.std().item() if all_flow_per_sample.numel() > 1 else 0.0
            accum["flow_loss_per_sample_histogram"] = all_flow_per_sample.cpu().numpy()
        if flow_loss_per_timestep_list and flow_timesteps_list:
            all_flow_by_timestep = torch.cat(flow_loss_per_timestep_list)
            all_flow_timesteps = torch.cat(flow_timesteps_list)
            low_t = all_flow_timesteps < 0.3
            high_t = all_flow_timesteps > 0.7
            if bool(low_t.any()):
                low_mean = all_flow_by_timestep[low_t].mean().item()
                accum["flow_loss_timestep/mean_lt_0.3"] = low_mean
                accum["flow_loss_noise/mean_low_noise_lt_0.3"] = low_mean
            if bool(high_t.any()):
                high_mean = all_flow_by_timestep[high_t].mean().item()
                accum["flow_loss_timestep/mean_gt_0.7"] = high_mean
                accum["flow_loss_noise/mean_high_noise_gt_0.7"] = high_mean
        if flow_loss_per_action_step_list:
            all_flow_by_action_step = torch.cat(flow_loss_per_action_step_list, dim=0)
            horizon = int(all_flow_by_action_step.shape[1])
            edge = min(10, horizon)
            accum["flow_loss_time/mean_first_10"] = all_flow_by_action_step[:, :edge].mean().item()
            accum["flow_loss_time/mean_last_10"] = all_flow_by_action_step[:, -edge:].mean().item()
        if discrete_ce_loss_list:
            all_discrete_ce = torch.cat(discrete_ce_loss_list)
            accum["discrete_ce_loss_histogram_flat"] = all_discrete_ce.cpu().numpy()
        if discrete_z_loss_list:
            all_discrete_z = torch.cat(discrete_z_loss_list)
            accum["discrete_z_loss_histogram_flat"] = all_discrete_z.cpu().numpy()
        if reward_list:
            all_rewards = torch.cat(reward_list)
            accum["reward_mean"] = all_rewards.mean().item()
            accum["reward_std"] = all_rewards.std().item() if all_rewards.numel() > 1 else 0.0
            accum["reward_histogram"] = all_rewards.cpu().numpy()
        if done_list:
            all_done = torch.cat(done_list)
            accum["done_fraction"] = all_done.float().mean().item()

        if squashed_adv_list:
            all_squashed = torch.cat(squashed_adv_list)
            all_raw = torch.cat(raw_adv_list)
            accum["advantage_mean"] = all_squashed.mean().item()
            accum["advantage_std"] = (
                all_squashed.std().item() if all_squashed.numel() > 1 else 0.0
            )
            accum["advantage_positive_label_fraction"] = (
                (all_squashed > 0.25).float().mean().item()
            )
            accum["advantage_raw_mean"] = all_raw.mean().item()
            accum["advantage_raw_std"] = (
                all_raw.std().item() if all_raw.numel() > 1 else 0.0
            )
            accum["advantage_histogram"] = all_squashed.cpu().numpy()
            accum["advantage_raw_histogram"] = all_raw.cpu().numpy()
        if critic_values_actor_list:
            all_critic_actor = torch.cat(critic_values_actor_list)
            accum["critic_value_mean_actor"] = all_critic_actor.mean().item()
            accum["critic_value_std_actor"] = all_critic_actor.std().item() if all_critic_actor.numel() > 1 else 0.0
            accum["critic_value_histogram_from_actor"] = all_critic_actor.cpu().numpy()
        if target_values_actor_list:
            all_target_actor = torch.cat(target_values_actor_list)
            accum["target_value_mean_actor"] = all_target_actor.mean().item()
            accum["target_value_std_actor"] = all_target_actor.std().item() if all_target_actor.numel() > 1 else 0.0
            accum["target_value_histogram_actor"] = all_target_actor.cpu().numpy()
        if v_next_values_actor_list:
            all_v_next_actor = torch.cat(v_next_values_actor_list)
            accum["v_next_mean_actor"] = all_v_next_actor.mean().item()
            accum["v_next_std_actor"] = all_v_next_actor.std().item() if all_v_next_actor.numel() > 1 else 0.0
            accum["v_next_histogram_actor"] = all_v_next_actor.cpu().numpy()

        if accum["loss_discrete_z"] == 0.0:
            accum.pop("loss_discrete_z", None)

        return accum

    def update_target_networks(self, policy: nn.Module) -> None:
        """Polyak update: critic_target ← τ*critic + (1-τ)*critic_target."""
        if not hasattr(policy, "critic"):
            return
        self._target_update_call_counter = getattr(self, "_target_update_call_counter", 0) + 1
        every = int(getattr(policy.config, "critic_target_update_every", 1))
        if every > 1 and self._target_update_call_counter % every != 0:
            return
        critic_net: nn.Module = getattr(policy, "critic")
        critic_target: nn.Module = getattr(policy, "critic_target")
        tau = float(getattr(policy.config, "critic_target_update_weight", 0.005))
        with torch.no_grad():
            for p, p_tgt in zip(critic_net.parameters(), critic_target.parameters()):
                if not p.requires_grad:
                    continue
                p_tgt.data.lerp_(p.data, tau)

    # ── Inference ─────────────────────────────────────────────────────────────

    def build_inference_batch(
        self,
        observation: dict,
        task_str: str,
        cfg,
        **context,
    ) -> dict[str, Any]:
        """
        Assemble the preprocessed inference batch for MolmoAct2.

        observation: flat dict with image/state keys (no batch dim).
        The preprocessor's AddBatchDimensionProcessorStep adds the batch dim.
        Tensors are moved to the policy device after preprocessing so the env
        processor no longer needs a DeviceProcessorStep.
        """
        preprocessor = context["preprocessor"]
        device = getattr(cfg.policy, "device", "cpu")
        pre_input: dict[str, Any] = {**observation, "task": task_str}
        with torch.no_grad():
            batch = preprocessor(pre_input)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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
        """Push trainable actor-runtime parameters to the actor queue."""
        from lerobot.transport.utils import state_to_bytes
        from lerobot.utils.transition import move_state_dict_to_device

        trainable = {
            name: param
            for name, param in policy.named_parameters()
            if param.requires_grad
            and not name.startswith("critic.")
            and not name.startswith("critic_target.")
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
        console_keys = ("loss_flow", "actor_grad_norm", "loss_critic")
        console_scalars = {
            k: training_infos[k]
            for k in console_keys
            if isinstance(training_infos.get(k), (int, float))
        }
        if console_scalars:
            logging.info(
                f"[MolmoAct2Trainer] step={step}  "
                + "  ".join(f"{k}={v:.4f}" for k, v in console_scalars.items())
            )
        super().log_metrics(training_infos, step, wandb_logger, _policy)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _action_dim(self, cfg) -> int:
        """Resolve the active action dimension from config output_features."""
        action_feat = getattr(cfg.policy, "output_features", {}).get(ACTION)
        if action_feat is not None and action_feat.shape:
            return int(action_feat.shape[0])
        return 6  # SO-101 fallback
