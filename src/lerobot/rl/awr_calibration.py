"""Training-sampler calibration and provenance for frozen-critic AWR."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch

from lerobot.rl.awr import AWRCalibration, group_key, normalize_log_weights, weight_telemetry
from lerobot.rl.awr_coverage import COVERAGE_SAMPLES_PER_GROUP, rebot_group_rows
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
    # Keep old artifacts compatible with the old rule, but never reuse their
    # scalar advantages when subtask boundaries now bootstrap.
    if getattr(p, "advantage_bootstrap_subtasks", False):
        result["policy"]["advantage_bootstrap_subtasks"] = True
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
        groups.update(rebot_group_rows(buffer, names, cfg.policy.n_action_steps))
    if diverse_buffer is not None:
        for identity, skip in zip(diverse_buffer._identity, diverse_buffer._critic.skip, strict=True):
            if not skip:
                label = identity.subtask_index
                groups.add(group_key(identity.embodiment_index, names[label] if label >= 0 else ""))
    return groups


def prepare_calibration(trainer, policy, iterator, preprocessor, cfg, runtime, expected_groups, *, coverage_sampler=None):
    """Fit once on training draws, or load exactly matching frozen statistics."""
    p = cfg.policy
    if not cfg.skip_critic or not hasattr(policy, "critic") or any(x.requires_grad for x in policy.critic.parameters()):
        raise ValueError("Subtask AWR calibration requires a loaded frozen critic and skip_critic: true.")
    provenance = calibration_provenance(cfg)
    path = resolve_calibration_path(cfg)
    calibration = None
    if path.is_file():
        calibration = AWRCalibration.load(path, p.advantage_beta, p.advantage_clip, provenance)
        logging.info("[AWR] Loaded saved calibration from %s (%d mixture samples, %d coverage samples)",
                     path, calibration.mixture_n, calibration.n - calibration.mixture_n)
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
    incomplete = missing | {key for key, values in calibration.coverage_samples.items()
                            if key in expected_groups and len(values) < COVERAGE_SAMPLES_PER_GROUP}
    if incomplete and coverage_sampler is not None:
        # A pinned/checkpoint calibration is an input artifact: repairs belong to
        # this run, so another run's copy is never rewritten.
        output_path = Path(cfg.output_dir) / "awr_calibration.json"
        if path.resolve() != output_path.resolve():
            path = output_path
        requests = [key for key in sorted(incomplete)
                    for _ in range(COVERAGE_SAMPLES_PER_GROUP - len(calibration.coverage_samples.get(key, [])))]
        logging.info("[AWR] Completing %d rare groups with %d conditional training draws; retaining all %d mixture samples",
                     len(incomplete), len(requests), calibration.mixture_n)
        local_requests = requests[runtime.process_index::runtime.num_processes]
        seed = int(getattr(cfg, "seed", 0) or 0) + runtime.process_index + calibration.n
        batches = iter(coverage_sampler(local_requests, seed))
        rounds = math.ceil(len(requests) / (cfg.batch_size * runtime.num_processes))
        for index in range(rounds):
            additions = defaultdict(list)
            raw = next(batches, None)
            if raw is not None:
                raw = move_transition_to_device(raw, runtime.device)
                values, kept = trainer._advantages(policy, raw, preprocessor, cfg)
                keys = trainer._advantage_groups(raw, preprocessor)
                if not bool(kept.all()) or not torch.isfinite(values).all():
                    raise ValueError("AWR coverage draws must all have valid finite critic advantages.")
                for key, value in zip(keys, values.detach().cpu().tolist(), strict=True):
                    if key not in incomplete:
                        raise ValueError(f"Unexpected group in targeted AWR coverage: {key}")
                    additions[key].append(value)
            gathered = defaultdict(list)
            for shard in runtime.gather_objects(dict(additions)):
                for key, values in shard.items():
                    gathered[key].extend(values)
            calibration = calibration.with_coverage(gathered)
            if runtime.is_main_process:
                calibration.save(path)  # Atomic save after every recovery batch.
            runtime.wait_for_everyone()
            logging.info("[AWR] coverage batch %d/%d: %d/%d groups, %d saved coverage samples",
                         index + 1, rounds, len(calibration.centers.keys() & expected_groups),
                         len(expected_groups), calibration.n - calibration.mixture_n)
        missing = expected_groups - calibration.centers.keys()
        remaining = {key for key in incomplete
                     if len(calibration.coverage_samples.get(key, [])) < COVERAGE_SAMPLES_PER_GROUP}
        if remaining:
            raise ValueError(f"AWR coverage sampler did not fill {len(remaining)} requested groups; progress was saved.")

    coverage = {
        "expected_groups": len(expected_groups), "calibrated_groups": len(calibration.centers),
        "missing_groups": sorted(missing), "valid_samples": calibration.n, "shared_scale": calibration.scale,
        "mixture_samples": calibration.mixture_n,
        "coverage_samples": calibration.n - calibration.mixture_n,
        "coverage_groups": {key: len(values) for key, values in sorted(calibration.coverage_samples.items())},
        "shared_scale_source": "original training mixture; conditional coverage draws excluded",
        "minimum_group_samples": min(map(len, calibration.samples.values())),
        "groups_under_8_samples": sum(len(values) < 8 for values in calibration.samples.values()),
    }
    if runtime.is_main_process:
        path.with_name("awr_calibration_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    if missing:
        raise ValueError(
            f"AWR calibration is missing {len(missing)}/{len(expected_groups)} training groups. "
            f"See {path.with_name('awr_calibration_coverage.json')}. "
            "Saved calibration was retained. Complete the missing groups with the training coverage sampler."
        )
    logging.info("[AWR] calibration coverage %d/%d groups, %d samples, shared scale %.6f, min group n=%d",
                 len(expected_groups), len(expected_groups), calibration.n, calibration.scale, coverage["minimum_group_samples"])
    cfg.policy.advantage_calibration_path = str(path.resolve())
    trainer._awr_calibration = calibration
    return calibration
