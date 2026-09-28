"""Training-sampler calibration and provenance for frozen-critic AWR."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import torch

from lerobot.rl.awr import AWRCalibration, group_key, normalize_log_weights, weight_telemetry
from lerobot.utils.transition import move_transition_to_device


def calibration_provenance(cfg):
    p = cfg.policy
    checkpoint = Path(p.critic_pretrained_path or p.pretrained_path or "") / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError("Frozen AWR calibration requires an existing critic checkpoint.")
    stat = checkpoint.stat()
    # Size + nanosecond mtime identify the loaded local checkpoint without rereading
    # tens of GB. Preserve the artifact together with the pinned checkpoint.
    policy_fields = (
        "discount", "value_support_min", "value_support_max", "critic_reward_mode",
        "reward_normalization_constant", "critic_mistake_penalty", "terminal_failure_reward",
        "n_action_steps", "image_stride", "memory", "pointmap_config", "input_features",
        "embodiment", "embodiment_stats_path", "action_encoding", "task",
    )
    def plain(value):
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        return value
    result = {
        "critic_checkpoint": str(checkpoint.resolve()), "critic_size": stat.st_size,
        "critic_mtime_ns": stat.st_mtime_ns,
        "policy": {key: plain(getattr(p, key, None)) for key in policy_fields},
        "dataset": plain(cfg.dataset), "diverse": plain(getattr(cfg, "diverse", None)),
    }
    return json.loads(json.dumps(result, default=str))


def resolve_calibration_path(cfg):
    configured = getattr(cfg.policy, "advantage_calibration_path", None)
    if configured and Path(configured).is_file():
        return Path(configured)
    pretrained = getattr(cfg.policy, "pretrained_path", None)
    if pretrained:
        sibling = Path(pretrained) / "awr_calibration.json"
        if sibling.is_file():
            return sibling
    return Path(configured) if configured else Path(cfg.output_dir) / "awr_calibration.json"


def expected_training_groups(buffers, diverse_buffer, trainer, preprocessor, cfg):
    """Groups reachable by the sampler, using only low-dimensional metadata.

    Match ReplayBuffer's stride and physical-boundary rejection. Diverse missing
    successors are excluded with the same critic_skip view used in training.
    """
    names = trainer.subtask_vocabulary(preprocessor)
    groups = set()
    for buffer in buffers:
        comp = buffer.complementary_info
        high = max(0, buffer.size - 1) if buffer.optimize_memory and buffer.size < buffer.capacity else buffer.size
        indices = torch.arange(0, high, buffer.image_stride, device=buffer.storage_device)
        if buffer.actions.ndim == 2 and cfg.policy.n_action_steps > 1:
            valid = torch.ones_like(indices, dtype=torch.bool)
            for offset in range(cfg.policy.n_action_steps - 1):
                valid &= ~buffer.dones[(indices + offset) % buffer.capacity].bool()
            indices = indices[valid]
        embodiment = comp.get("embodiment_index")
        subtask = comp.get("subtask_index")
        embodiment = torch.full_like(indices, -1) if embodiment is None else embodiment[indices].reshape(-1).long()
        subtask = torch.full_like(indices, -1) if subtask is None else subtask[indices].reshape(-1).long()
        pairs = torch.stack((embodiment, subtask), dim=1).unique(dim=0).cpu().tolist()
        for robot, label in pairs:
            if label >= len(names):
                raise ValueError(f"Unknown subtask index {label} in AWR training buffer.")
            groups.add(group_key(robot, names[label] if label >= 0 else ""))
    if diverse_buffer is not None:
        for identity, skip in zip(diverse_buffer._identity, diverse_buffer._critic.skip, strict=True):
            if not skip:
                label = identity.subtask_index
                groups.add(group_key(identity.embodiment_index, names[label] if label >= 0 else ""))
    return groups


def prepare_calibration(trainer, policy, iterator, preprocessor, cfg, runtime, expected_groups):
    """Fit once on training draws, or load exactly matching frozen statistics."""
    p = cfg.policy
    if not cfg.skip_critic or not hasattr(policy, "critic") or any(x.requires_grad for x in policy.critic.parameters()):
        raise ValueError("Subtask AWR calibration requires a loaded frozen critic and skip_critic: true.")
    provenance = calibration_provenance(cfg)
    path = resolve_calibration_path(cfg)
    calibration = None
    if path.is_file():
        calibration = AWRCalibration.load(path, p.advantage_beta, p.advantage_clip, provenance)
        if not p.advantage_calibration_path and expected_groups - calibration.centers.keys():
            logging.info("[AWR] Existing calibration is incomplete; rebuilding with %d batches", p.advantage_calibration_batches)
            calibration = None
        else:
            logging.info("[AWR] Loaded frozen calibration from %s", path)
    if calibration is None:
        if p.advantage_calibration_path:
            raise FileNotFoundError(f"Requested AWR calibration does not exist: {path}")
        count = p.advantage_calibration_batches
        logging.info("[AWR] Calibrating %d training batches per rank; no optimizer steps", count)
        advantages, groups, terminals, mistakes = [], [], [], []
        total_rows = 0
        for index in range(count):
            raw = move_transition_to_device(next(iterator), runtime.device)
            a, kept = trainer._advantages(policy, raw, preprocessor, cfg)
            keys = trainer._advantage_groups(raw, preprocessor)
            mask = kept.cpu().tolist()
            advantages.extend(a[kept].cpu().tolist())
            groups.extend(key for key, keep in zip(keys, mask, strict=True) if keep)
            terminals.extend(raw["done"].reshape(-1)[kept].bool().cpu().tolist())
            # The normalized reward is strictly below its step/terminal baseline
            # exactly when the sampler applied its mistake-onset penalty.
            baseline = (raw["done"].reshape(-1).float() - 1) / p.reward_normalization_constant
            mistakes.extend((raw["reward"].reshape(-1)[kept] < baseline[kept] - 1e-6).cpu().tolist())
            total_rows += a.numel()
            if (index + 1) % 32 == 0 or index + 1 == count:
                logging.info("[AWR] calibration %d/%d batches, %d valid rows, %d groups",
                             index + 1, count, len(advantages), len(set(groups)))
        shards = runtime.gather_objects((advantages, groups, terminals, mistakes, total_rows))
        advantages = [x for shard in shards for x in shard[0]]
        groups = [x for shard in shards for x in shard[1]]
        terminals = [x for shard in shards for x in shard[2]]
        mistakes = [x for shard in shards for x in shard[3]]
        total_rows = sum(shard[4] for shard in shards)
        calibration = AWRCalibration.fit(advantages, groups, p.advantage_beta, p.advantage_clip, provenance)
        if runtime.is_main_process:
            calibration.save(path)
            log_q, z = calibration.log_weights(torch.tensor(advantages), groups)
            weights = normalize_log_weights(log_q, p.advantage_lambda)
            metrics, rows = weight_telemetry(advantages, weights, z, groups, terminals, mistakes, total_rows, p.advantage_clip)
            path.with_name("awr_calibration_metrics.json").write_text(json.dumps({"metrics": metrics, "groups": rows}, indent=2) + "\n")
            logging.info("[AWR] calibration ESS/N=%.3f, max weight=%.2f, terminal weight share=%.3f",
                         metrics["adv_weight_ess_frac"], metrics["adv_weight_max"], metrics["awr/terminal_weight_share"])
        runtime.wait_for_everyone()

    missing = expected_groups - calibration.centers.keys()
    coverage = {
        "expected_groups": len(expected_groups), "calibrated_groups": len(calibration.centers),
        "missing_groups": sorted(missing), "valid_samples": calibration.n, "shared_scale": calibration.scale,
        "minimum_group_samples": min(map(len, calibration.samples.values())),
        "groups_under_8_samples": sum(len(values) < 8 for values in calibration.samples.values()),
    }
    if runtime.is_main_process:
        path.with_name("awr_calibration_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if missing:
        raise ValueError(
            f"AWR calibration is missing {len(missing)}/{len(expected_groups)} training groups. "
            f"See {path.with_name('awr_calibration_coverage.json')}. "
            "Generate a fresh calibration with more advantage_calibration_batches before actor updates."
        )
    logging.info("[AWR] calibration coverage %d/%d groups, %d samples, shared scale %.6f, min group n=%d",
                 len(expected_groups), len(expected_groups), calibration.n, calibration.scale, coverage["minimum_group_samples"])
    cfg.policy.advantage_calibration_path = str(path.resolve())
    trainer._awr_calibration = calibration
    return calibration
