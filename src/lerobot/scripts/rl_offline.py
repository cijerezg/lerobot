#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generic offline pretraining loop for any RL policy registered with Trainer.

Replaces the PI05-specific offline_learner_pi05.py for new model families.
All model-specific behaviour is delegated to the Trainer subclass selected by
cfg.policy.type via Trainer.for_config(cfg).

Usage:
    python -m lerobot.rl.rl_offline --config config.yaml [overrides]

Config fields required beyond the standard TrainRLServerPipelineConfig:
    policy.offline_steps    int     number of gradient steps
    skip_critic             bool    True → actor-only; no critic updates (default False)

Compared to offline_learner_pi05.py this script:
  - Is model-agnostic (works for pi05_rl, molmoact2_rl, ...)
  - Is single-process only (no Accelerator / DDP)
  - Does not include weight anchors or per-step validation hooks
  - Defers all model-specific logic to Trainer
"""

import dataclasses
import gc
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from pprint import pformat

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import torch
from termcolor import colored
from torch import nn

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.offline_dataset_utils import (
    load_additional_offline_buffers,
    make_combined_offline_iterator,
)
from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.utils import cast_to_bf16
from lerobot.rl.weight_anchor import build_weight_anchors, apply_weight_anchors
from lerobot.common.wandb_utils import WandBLogger
from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.constants import CHECKPOINTS_DIR, TRAINING_STATE_DIR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, init_logging


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_optimizers(groups: list[dict]) -> dict[str, torch.optim.Optimizer]:
    """
    Convert the list of parameter-group dicts returned by Trainer.get_optimizer_groups
    into a dict of named Adam optimizers.

    Each group dict must have keys: "params", "lr", "name".
    """
    optimizers = {}
    for g in groups:
        optimizers[g["name"]] = torch.optim.Adam(g["params"], lr=g["lr"])
    return optimizers


def _save_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    step: int,
    total_steps: int,
    policy: nn.Module,
    preprocessor,
    postprocessor,
) -> None:
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = get_step_checkpoint_dir(output_dir, total_steps, step)

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        cfg=cfg,
        policy=policy,
        optimizer=None,
        scheduler=None,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    training_state_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "interaction_step": 0}, training_state_dir / "training_state.pt")

    update_last_checkpoint(checkpoint_dir)
    logging.info(f"[RL_OFFLINE] Checkpoint saved → {checkpoint_dir}")


def _save_anchor_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    step: int,
    total_steps: int,
    policy: nn.Module,
    preprocessor,
    postprocessor,
    suffix: str,
) -> None:
    output_dir = Path(cfg.output_dir)
    step_id = get_step_identifier(step, total_steps)
    checkpoint_dir = output_dir / CHECKPOINTS_DIR / f"{step_id}_{suffix}"

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        cfg=cfg,
        policy=policy,
        optimizer=None,
        scheduler=None,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    training_state_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "interaction_step": 0}, training_state_dir / "training_state.pt")
    logging.info(f"[RL_OFFLINE] Anchor checkpoint ({suffix}) saved → {checkpoint_dir}")


# ── Validation probes ─────────────────────────────────────────────────────────


def _build_validation_adapter(policy, preprocessor, postprocessor, device, cfg):
    """Wrap the already-loaded training policy in a ProbablePolicy adapter
    without re-loading weights.

    Mirrors :meth:`ProbablePolicy.for_config` but reuses the existing policy.
    """
    from lerobot.probes.base import _adapter_for_type
    adapter_cls = _adapter_for_type(getattr(cfg.policy, "type", None))
    return adapter_cls(policy, preprocessor, postprocessor, device, cfg)


@dataclasses.dataclass(frozen=True)
class _ValidationProbeSpec:
    flag: str
    module_path: str
    output_subdir: str
    run_attr: str = "run"


_VALIDATION_PROBES = (
    _ValidationProbeSpec("enable_actions", "lerobot.probes.actions", "actions"),
    _ValidationProbeSpec("enable_offline_inference", "lerobot.probes.offline_inference", "offline_inference"),
    _ValidationProbeSpec("enable_attention", "lerobot.probes.attention", "attention"),
    _ValidationProbeSpec("enable_critic_values_distribution", "lerobot.probes.critic", "critic"),
    _ValidationProbeSpec("enable_representations", "lerobot.probes.representations", "representations"),
    _ValidationProbeSpec("enable_spatial_memorization", "lerobot.probes.spatial_memorization_attention", "spatial_memorization_attention"),
    _ValidationProbeSpec("enable_action_drift_jacobian", "lerobot.probes.action_drift_jacobian", "action_drift_jacobian"),
    _ValidationProbeSpec(
        "enable_spatial_memorization_jacobian",
        "lerobot.probes.spatial_memorization_action_jacobian",
        "spatial_memorization_action_jacobian",
    ),
)


def _enabled_validation_flags(probe_cfg) -> set[str]:
    return {
        field.name
        for field in dataclasses.fields(probe_cfg)
        if field.name.startswith("enable_") and bool(getattr(probe_cfg, field.name))
    }


def _validate_probe_registry(probe_cfg) -> None:
    registered = {spec.flag for spec in _VALIDATION_PROBES}
    unregistered = sorted(_enabled_validation_flags(probe_cfg) - registered)
    if unregistered:
        raise ValueError(
            "Enabled validation probe flag(s) have no generic rl_offline handler: "
            + ", ".join(unregistered)
        )


def _run_validation_probes(
    policy: nn.Module,
    preprocessor,
    postprocessor,
    val_dataset,
    reference_dataset,
    device,
    cfg: TrainRLServerPipelineConfig,
    step: int,
) -> None:
    """Dispatch every probe whose ``cfg.probe_parameters.enable_*`` flag is set.

    Each probe is wrapped in try/except so one failure doesn't kill training.
    The policy is put into eval mode for the duration and restored at the end.
    """
    p = cfg.probe_parameters
    output_root = os.path.join(cfg.output_dir, "validation", f"step_{step:08d}")
    was_training = policy.training
    policy.eval()
    try:
        adapter = _build_validation_adapter(policy, preprocessor, postprocessor, device, cfg)

        _validate_probe_registry(p)

        import importlib
        for spec in _VALIDATION_PROBES:
            if not bool(getattr(p, spec.flag, False)):
                continue
            logging.info(f"[VAL step={step}] probe '{spec.output_subdir}' started")
            try:
                module = importlib.import_module(spec.module_path)
                run_probe = getattr(module, spec.run_attr)
                probe_output_dir = os.path.join(output_root, spec.output_subdir)
                if spec.output_subdir == "actions":
                    run_probe(
                        adapter,
                        reference_dataset if reference_dataset is not None else val_dataset,
                        cfg,
                        probe_output_dir,
                        eval_dataset=val_dataset,
                    )
                else:
                    run_probe(adapter, val_dataset, cfg, probe_output_dir)
            except Exception as exc:
                logging.warning(f"[VAL step={step}] probe '{spec.output_subdir}' failed: {exc}")
            else:
                logging.info(f"[VAL step={step}] probe '{spec.output_subdir}' finished successfully")

        # Drop the adapter; the policy lives on in the training scope.
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        if was_training:
            policy.train()


def _load_val_dataset(cfg: TrainRLServerPipelineConfig, fallback_dataset):
    """Return the validation dataset.

    If ``cfg.val_dataset_path`` is set, load it separately (mirrors the pi05
    val pipeline). Otherwise fall back to the training offline dataset.
    """
    val_path = getattr(cfg, "val_dataset_path", None)
    if not val_path:
        return fallback_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    logging.info(f"[VAL] Loading validation dataset from {val_path}")
    ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
    ds.delta_timestamps = None
    ds.delta_indices = None
    return ds


# ── Entry point ───────────────────────────────────────────────────────────────


def _preprocess_config_yaml(config_path: str) -> str:
    """
    Strip policy fields not valid for the active policy.type.

    Enables a single unified config YAML with both PI05 and MolmoAct2 sections:
    only the fields valid for the selected policy type survive into the
    draccus-parsed config.  Draccus rejects unknown fields, so without this
    step a shared YAML would always error when one model's fields are present
    while parsing the other model's config class.

    Returns the original path unchanged if no stripping is needed, otherwise
    returns the path to a NamedTemporaryFile with the cleaned YAML.
    """
    import yaml
    from lerobot.configs import PreTrainedConfig

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    policy_dict = raw.get("policy")
    if not isinstance(policy_dict, dict):
        return config_path

    policy_type = policy_dict.get("type")
    if not policy_type:
        return config_path

    try:
        policy_cls = PreTrainedConfig.get_choice_class(policy_type)
        valid_fields = {f.name for f in dataclasses.fields(policy_cls)} | {"type"}
        stripped = {k: v for k, v in policy_dict.items() if k in valid_fields}
        if len(stripped) == len(policy_dict):
            return config_path  # nothing to strip
        raw["policy"] = stripped
    except Exception:
        return config_path  # fall back on any error

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="rl_offline_cfg_"
    )
    yaml.dump(raw, tmp, allow_unicode=True)
    tmp.close()
    return tmp.name


def _extract_config_path_args(args: list[str]) -> tuple[str | None, list[str]]:
    """Accept both LeRobot's --config_path and the common --config spelling."""
    config_path = None
    filtered: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--config_path="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg in {"--config_path", "--config"}:
            if i + 1 >= len(args):
                raise ValueError(f"{arg} expects a path argument.")
            config_path = args[i + 1]
            i += 1
        else:
            filtered.append(arg)
        i += 1
    return config_path, filtered


@parser.wrap()
def offline_train_cli(cfg: TrainRLServerPipelineConfig):
    """CLI entry point — called after model registration and YAML pre-processing."""
    offline_train(cfg, job_name=cfg.job_name)
    logging.info("[RL_OFFLINE] Done.")


def main() -> None:
    """
    Real entry point.

    1. Register all model-specific configs so draccus can find their classes.
    2. Pre-process the YAML to strip fields that belong to the inactive model.
    3. Delegate to offline_train_cli (parser.wrap handles config parsing).
    """
    # Register before parsing so draccus can resolve policy.type.
    import lerobot.rl.pi05.rl_pi05            # noqa: F401 — registers PI05RLConfig
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    from lerobot.robots import so_follower         # noqa: F401 — registers so101_follower
    from lerobot.teleoperators import so_leader    # noqa: F401 — registers so101_leader

    cli_args = sys.argv[1:]
    config_path, remaining_args = _extract_config_path_args(cli_args)
    if config_path:
        processed = _preprocess_config_yaml(config_path)
        # Swap in normalized form so parser.wrap picks it up regardless of
        # whether the user passed --config, --config_path, or a space-separated value.
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={processed}"]

    offline_train_cli()  # type: ignore[call-arg]  # @parser.wrap rewrites signature


def offline_train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None) -> None:
    cfg.validate()
    cfg.env.task = cfg.policy.task

    if job_name is None:
        job_name = cfg.job_name
    if job_name is None:
        raise ValueError("job_name must be set in config or passed explicitly.")

    output_dir = (
        getattr(cfg, "offline_output_dir", None)
        or getattr(cfg, "output_dir_offline", None)
        or cfg.output_dir
    )
    if output_dir is None:
        raise ValueError("offline_output_dir or output_dir must be set.")
    cfg.output_dir = str(output_dir)

    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rl_offline_{job_name}.log")
    init_logging(log_file=log_file, display_pid=False)
    logging.info(f"[RL_OFFLINE] Logging → {log_file}")
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable:
        if getattr(cfg.wandb, "offline_project", None):
            cfg.wandb.project = cfg.wandb.offline_project
    wandb_logger = (
        WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None
    )
    if wandb_logger is None:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.dataset is None:
        raise ValueError("cfg.dataset is required for offline training.")

    if cfg.seed is not None:
        set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    run_offline_training(cfg=cfg, wandb_logger=wandb_logger, shutdown_event=shutdown_event)


# ── Core loop ─────────────────────────────────────────────────────────────────


def run_offline_training(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event,
) -> None:
    """
    Model-agnostic offline training loop.

    Data flow:
        offline dataset → ReplayBuffer → iterator
            → trainer.update_critic(online_iter=iterator, offline_iter=None)
            → trainer.update_target_networks(policy)          [if not skip_critic]
            → trainer.update_actor(online_iter=iterator, ...)
            → trainer.log_metrics(...)
    """
    # ── Config unpacking ──────────────────────────────────────────────────────
    device_name = getattr(cfg.policy, "learner_device", None) or cfg.policy.device
    device = get_safe_torch_device(try_device=device_name, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)

    offline_steps = cfg.policy.offline_steps
    utd_ratio = getattr(cfg.policy, "utd_ratio", 1)
    critic_warmup_steps = getattr(cfg.policy, "critic_warmup_steps", 0)
    policy_update_freq = getattr(cfg.policy, "policy_update_freq", 1)
    skip_critic = getattr(cfg, "skip_critic", False)
    gradient_accumulation_steps = getattr(cfg.policy, "gradient_accumulation_steps", 1)
    clip_grad_norm_value = getattr(cfg.policy, "optimizer_grad_clip_norm", 1.0)
    async_prefetch = getattr(cfg.policy, "async_prefetch", False)
    log_freq = cfg.log_freq
    save_freq = getattr(cfg, "offline_save_freq", None) or cfg.save_freq
    saving_checkpoint = cfg.save_checkpoint
    fps = cfg.env.fps

    cast_to_bf16_fn = cast_to_bf16 if getattr(cfg.policy, "dtype", None) == "bfloat16" else None

    logging.info(f"[RL_OFFLINE] offline_steps={offline_steps}  skip_critic={skip_critic}")

    # ── Build policy via Trainer ──────────────────────────────────────────────
    trainer = Trainer.for_config(cfg)

    original_device = cfg.policy.device
    cfg.policy.device = device_name
    policy = trainer.make_policy(cfg)
    cfg.policy.device = original_device

    assert isinstance(policy, nn.Module)
    policy.train()

    # ── Critic initialisation (before freeze so freeze_model can see it) ──────
    if not skip_critic:
        _init_critic = getattr(policy, "init_critic", None)
        if callable(_init_critic):
            _init_critic()

    # ── Freeze layers ─────────────────────────────────────────────────────────
    trainer.freeze_model(policy, cfg)

    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    logging.info(f"[RL_OFFLINE] Trainable params: {len(trainable)}")

    # ── Offline dataset & replay buffer ───────────────────────────────────────
    logging.info("[RL_OFFLINE] Loading offline dataset …")
    offline_dataset = make_dataset(cfg)
    offline_dataset.delta_timestamps = None
    offline_dataset.delta_indices = None

    # ── Preprocessors ─────────────────────────────────────────────────────────
    preprocessor, postprocessor = trainer.make_processors(
        cfg, dataset=offline_dataset, is_main_process=True
    )
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    # ── Optimizers ────────────────────────────────────────────────────────────
    optimizers = _build_optimizers(trainer.get_optimizer_groups(policy, cfg))
    weight_anchors = build_weight_anchors(
        optimizers=optimizers,
        alpha=float(getattr(cfg.policy, "anchor_alpha", 0.0)),
        every_n_steps=int(getattr(cfg.policy, "anchor_every_n_steps", 0)),
        targets=list(getattr(cfg.policy, "anchor_targets", [])),
    )

    # ── Replay buffer ─────────────────────────────────────────────────────────
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        reward_normalization_constant=cfg.policy.reward_normalization_constant,
        terminal_failure_reward=cfg.policy.terminal_failure_reward,
        inject_complementary_info={"is_golden": getattr(cfg, "treat_main_dataset_as_golden", False)},
        cache_dir=getattr(cfg, "buffer_cache_dir", None),
        image_storage_dtype=getattr(cfg.policy, "image_storage_dtype", "bfloat16"),
        image_storage_size=getattr(cfg.policy, "image_storage_size", (224, 224)),
    )
    offline_replay_buffer.dataset = offline_dataset
    additional_buffers = load_additional_offline_buffers(
        cfg=cfg,
        main_dataset=offline_dataset,
        device=device,
        storage_device=storage_device,
        is_main_process=True,
    )
    offline_buffers = [offline_replay_buffer, *additional_buffers]
    logging.info(
        f"[RL_OFFLINE] Buffer: {sum(len(b) for b in offline_buffers)} samples "
        f"({len(offline_buffers)} sources)"
    )

    # Share frozen critic-target params to save VRAM
    if not skip_critic and hasattr(policy, "critic") and hasattr(policy, "critic_target"):
        for p, p_tgt in zip(policy.critic.parameters(), policy.critic_target.parameters()):
            if not p.requires_grad:
                p_tgt.data = p.data

    # ── Training info ─────────────────────────────────────────────────────────
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    logging.info(colored("=" * 70, "yellow", attrs=["bold"]))
    logging.info(colored("RL_OFFLINE — generic offline training", "yellow", attrs=["bold"]))
    logging.info(colored("=" * 70, "yellow", attrs=["bold"]))
    logging.info(f"  policy type : {cfg.policy.type}")
    logging.info(f"  task        : {cfg.env.task}")
    logging.info(f"  steps       : {offline_steps}")
    logging.info(f"  skip_critic : {skip_critic}")
    logging.info(f"  trainable   : {format_big_number(n_trainable)} / {format_big_number(n_total)}")
    logging.info(colored("=" * 70, "yellow", attrs=["bold"]))

    # ── Iterator ─────────────────────────────────────────────────────────────
    buf_iter = make_combined_offline_iterator(
        buffers=offline_buffers,
        batch_size=cfg.batch_size,
        async_prefetch=async_prefetch,
        queue_size=2,
        action_chunk_size=cfg.policy.n_action_steps,
    )

    # ── Validation dataset (used by probes) ───────────────────────────────────
    val_dataset = _load_val_dataset(cfg, fallback_dataset=offline_dataset)
    val_freq = int(getattr(cfg, "val_freq", 0) or 0)
    val_on_start = bool(getattr(cfg, "val_on_start", False))

    if val_on_start:
        _run_validation_probes(
            policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
            val_dataset=val_dataset, reference_dataset=offline_dataset,
            device=device, cfg=cfg, step=0,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────
    optimization_step = 0

    while optimization_step < offline_steps:
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[RL_OFFLINE] Shutdown signal — exiting.")
            break

        t0 = time.time()

        if optimization_step % 10 == 0:
            print(f"[RL_OFFLINE] step {optimization_step}/{offline_steps}")

        # ── Critic updates (UTD - 1 extra rounds before the last combined step) ──
        if not skip_critic:
            for _ in range(utd_ratio - 1):
                trainer.update_critic(
                    policy=policy,
                    optimizers=optimizers,
                    online_iter=buf_iter,
                    offline_iter=None,
                    device=device,
                    cfg=cfg,
                    preprocessor=preprocessor,
                    dataset_repo_id=None,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    clip_grad_norm_value=clip_grad_norm_value,
                    cast_to_bf16_fn=cast_to_bf16_fn,
                )
                trainer.update_target_networks(policy)

        # ── Critic update (last / only round) + optional actor update ─────────
        training_infos: dict = {}

        if skip_critic:
            # Actor-only mode — no critic forward/backward in the offline loop.
            training_infos = trainer.update_actor(
                policy=policy,
                optimizers=optimizers,
                online_iter=buf_iter,
                offline_iter=None,
                preprocessor=preprocessor,
                dataset=offline_dataset,
                device=device,
                cfg=cfg,
                dataset_repo_id=None,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm_value=clip_grad_norm_value,
                cast_to_bf16_fn=cast_to_bf16_fn,
            )
        else:
            # Full RL: critic then actor (respecting critic_warmup_steps)
            critic_infos = trainer.update_critic(
                policy=policy,
                optimizers=optimizers,
                online_iter=buf_iter,
                offline_iter=None,
                device=device,
                cfg=cfg,
                preprocessor=preprocessor,
                dataset_repo_id=None,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm_value=clip_grad_norm_value,
                cast_to_bf16_fn=cast_to_bf16_fn,
            )
            trainer.update_target_networks(policy)
            training_infos.update(critic_infos)

            if (
                optimization_step >= critic_warmup_steps
                and optimization_step % policy_update_freq == 0
            ):
                actor_infos = trainer.update_actor(
                    policy=policy,
                    optimizers=optimizers,
                    online_iter=buf_iter,
                    offline_iter=None,
                    preprocessor=preprocessor,
                    dataset=offline_dataset,
                    device=device,
                    cfg=cfg,
                    dataset_repo_id=None,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    clip_grad_norm_value=clip_grad_norm_value,
                    cast_to_bf16_fn=cast_to_bf16_fn,
                )
                training_infos.update(actor_infos)

        anchor_fires = any(a.should_merge(optimization_step) for a in weight_anchors.values())
        if anchor_fires:
            _save_anchor_checkpoint(cfg, optimization_step, offline_steps, policy, preprocessor, postprocessor, "pre_anchor")
        apply_weight_anchors(weight_anchors, optimizers, optimization_step)
        if anchor_fires:
            _save_anchor_checkpoint(cfg, optimization_step, offline_steps, policy, preprocessor, postprocessor, "post_anchor")

        # ── Logging ───────────────────────────────────────────────────────────
        if optimization_step % log_freq == 0:
            training_infos["offline_buffer_size"] = sum(len(b) for b in offline_buffers)
            training_infos["Optimization step"] = optimization_step
            trainer.log_metrics(
                training_infos=training_infos,
                step=optimization_step,
                wandb_logger=wandb_logger,
                _policy=policy,
            )

        step_time = time.time() - t0
        step_hz = 1.0 / (step_time + 1e-9)
        if wandb_logger is not None:
            wandb_logger.log_dict(
                {"Optimization frequency loop [Hz]": step_hz, "Optimization step": optimization_step},
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1

        if optimization_step % log_freq == 0:
            logging.info(f"[RL_OFFLINE] step {optimization_step}/{offline_steps}  {step_hz:.2f} Hz")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if saving_checkpoint and (
            optimization_step % save_freq == 0 or optimization_step == offline_steps
        ):
            _save_checkpoint(
                cfg=cfg,
                step=optimization_step,
                total_steps=offline_steps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            gc.collect()
            torch.cuda.empty_cache()

        # ── Validation probes ─────────────────────────────────────────────────
        if val_freq > 0 and optimization_step % val_freq == 0:
            _run_validation_probes(
                policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                val_dataset=val_dataset, reference_dataset=offline_dataset,
                device=device, cfg=cfg, step=optimization_step,
            )

    logging.info(f"[RL_OFFLINE] Training complete after {optimization_step} steps.")


if __name__ == "__main__":
    main()
