#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Standalone asynchronous inference script for the ``pistar06`` policy.

This is the pistar06 counterpart of ``inference_pi05_async.py``. It runs the
policy in inference mode using RTC asynchronously with the environment and
delegates threading orchestration to :mod:`inference_pistar06_utils`.
Crucially, it strips out all ``pi05_full``-specific machinery:

- No subtask token injection or interactive subtask override (pistar06 does
  not generate subtasks).
- No critic-value forward pass / overlay video — pistar06 has no critic head.
- Reads the task string from ``cfg.env.task`` because ``PiStar06Config``
  inherits from :class:`PI05Config` and therefore has no ``task`` field of
  its own.

Usage::

    uv run python -m lerobot.rl.inference_pistar06_async \\
        --config_path config-pistar06.json

Minimal ``config-pistar06.json`` skeleton (illustrative — drop pi05_rl-only
fields like ``policy.task``, ``pi05_checkpoint``, ``inference_advantage``,
``use_separate_critic``, all ``critic_*`` / ``actor_*`` / ``online_*`` knobs,
etc.)::

    {
        "output_dir": "outputs/pistar06_inference",
        "job_name": "pistar06_pickplace",
        "seed": 42,
        "video_logging_cameras": ["top", "side"],
        "episode_logging_freq": 1,
        "episode_save_freq": 10,
        "policy": {
            "type": "pistar06",
            "pretrained_path": "outputs/pistar06_run/checkpoints/last/pretrained_model",
            "tokenizer_max_length": 64,
            "max_state_dim": 6,
            "num_inference_steps": 5,
            "n_obs_steps": 1,
            "device": "cuda",
            "dtype": "bfloat16",
            "gradient_checkpointing": false,
            "action_encoding": "absolute",
            "action_encoding_stats_path": null,
            "online_buffer_capacity": 200000,
            "rtc_config": {"enabled": true, "execution_horizon": 8},
            "normalization_mapping": { ... },
            "input_features":  { ... },
            "output_features": { ... },
            "enable_advantage_conditioning": true,
            "advantage_threshold": 0.0,
            "cfg_beta": 1.0
        },
        "env": {
            "type": "real_robot",
            "fps": 30,
            "task": "Pick up the orange cube and place it on the black X marker",
            ...
        }
    }
"""

import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from threading import Thread

import torch
from torch import nn

# Side-effect import: registers `pistar06` in the PreTrainedConfig registry
# so the JSON config loader can resolve `policy.type == "pistar06"`.
import lerobot.policies.pistar06  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.rl.gym_manipulator import make_processors, make_robot_env
from lerobot.rl.inference_pistar06_utils import (
    SharedState,
    env_interaction_worker,
    get_actions_worker,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.constants import ACTION, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def _unflatten_stats(stats: dict) -> dict:
    """Convert a flat ``{"action.p1": tensor}`` dict into ``{"action": {"p1": tensor}}``.

    Mirrors :func:`lerobot.rl.pi05_train_utils.unflatten_stats`. Re-implemented
    locally so this script doesn't pull in any pi05_full / pi05_rl modules.
    """
    unflattened: dict = {}
    for key, value in stats.items():
        if "." in key:
            prefix, suffix = key.rsplit(".", 1)
            if prefix not in unflattened:
                unflattened[prefix] = {}
            unflattened[prefix][suffix] = value
        else:
            if key not in unflattened:
                unflattened[key] = value
            elif isinstance(unflattened[key], dict) and isinstance(value, dict):
                unflattened[key].update(value)
            else:
                unflattened[key] = value
    return unflattened


def _make_pistar06_processors(cfg: TrainRLServerPipelineConfig, logger: logging.Logger):
    """Build pre/post-processors for a pistar06 inference run.

    Loads the normalizer state from ``cfg.policy.pretrained_path`` (without
    invoking ``make_pi05_full_processors_with_upgrade``, which adds
    ``advantage_scaling`` and other pi05_full-only overrides). Also honours
    ``cfg.policy.action_encoding`` + ``action_encoding_stats_path`` for
    anchor/delta encodings.
    """
    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    dataset_stats: dict | None = None

    if pretrained_path:
        checkpoint_path = Path(pretrained_path)
        config_path = checkpoint_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"

        logger.info(f"[STATS] Extracting stats from pretrained pipeline config: {config_path}")
        if not config_path.exists():
            logger.warning(f"[STATS] Pipeline config not found at {config_path}. Stats will be None.")
        else:
            from safetensors.torch import load_file

            with open(config_path, "r") as f:
                processor_config = json.load(f)

            for step_entry in processor_config.get("steps", []):
                if step_entry.get("registry_name") == "normalizer_processor":
                    state_filename = step_entry.get("state_file")
                    if not state_filename:
                        continue
                    state_path = checkpoint_path / state_filename
                    if state_path.exists():
                        dataset_stats = load_file(str(state_path))
                        logger.info(f"[STATS] Successfully loaded stats from {state_path}")
                        break

            if dataset_stats is not None:
                dataset_stats = _unflatten_stats(dataset_stats)
            else:
                logger.warning(
                    "[STATS] No 'normalizer_processor' step with a state file found in the pipeline config."
                )
    else:
        logger.warning(
            "[STATS] No pretrained_path set on cfg.policy — running with random weights and no stats."
        )

    # Optional: anchor/delta action encoding overrides the action stats with
    # those derived from the dataset's per-step joint displacement statistics.
    action_encoding = getattr(cfg.policy, "action_encoding", "absolute")
    if action_encoding in ["anchor", "delta"]:
        stats_path = getattr(cfg.policy, "action_encoding_stats_path", None)
        if stats_path and os.path.exists(stats_path):
            logger.info(f"[STATS] Loading {action_encoding} action stats from {stats_path}")
            disp_stats = torch.load(stats_path, map_location="cpu")
            if dataset_stats is None:
                dataset_stats = {}
            dataset_stats[ACTION] = disp_stats
        else:
            raise ValueError(
                f"action_encoding is {action_encoding} but action_encoding_stats_path "
                f"'{stats_path}' is invalid or does not exist!"
            )

    # PiStar06Config subclasses PI05Config, so this dispatches to
    # `make_pi05_pre_post_processors` inside `make_pre_post_processors`.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=dataset_stats,
    )
    return preprocessor, postprocessor


def _verify_loaded_weights(policy, cfg, logger):
    """Verify that trained weights were actually loaded, not a random init."""
    checkpoint_path = getattr(cfg.policy, "pretrained_path", "") or ""

    with torch.no_grad():
        probe_keys = ["action_out_proj", "gemma_expert"]
        logger.info("[WEIGHT CHECK] Parameter norms for key layers:")
        for name, param in policy.named_parameters():
            if any(k in name for k in probe_keys) and "weight" in name:
                logger.info(f"  {name}: norm={param.norm().item():.4f}  mean={param.mean().item():.8f}")
                break

    if not checkpoint_path:
        logger.error(
            "[WEIGHT CHECK] FAILED: No checkpoint was specified. "
            "Model is running with RANDOM weights!"
        )
        return

    model_file = None
    cp = Path(checkpoint_path)
    if cp.is_dir():
        for candidate in ["model.safetensors", "pytorch_model.bin"]:
            if (cp / candidate).exists():
                model_file = cp / candidate
                break
    elif cp.is_file():
        model_file = cp

    if model_file is None:
        logger.warning(f"[WEIGHT CHECK] Could not locate weight file at {checkpoint_path} for verification")
        return

    try:
        if str(model_file).endswith(".safetensors"):
            from safetensors.torch import load_file
            saved = load_file(str(model_file))
        else:
            saved = torch.load(str(model_file), map_location="cpu")

        mismatches = []
        checked = 0
        for name, param in policy.named_parameters():
            saved_key = name
            if saved_key not in saved:
                alt_key = name.replace("model.", "actor.", 1) if name.startswith("model.") else None
                if alt_key and alt_key in saved:
                    saved_key = alt_key
                else:
                    continue

            saved_norm = saved[saved_key].float().norm().item()
            loaded_norm = param.float().norm().item()
            if abs(saved_norm - loaded_norm) > 1e-2:
                mismatches.append((name, saved_norm, loaded_norm))
            checked += 1
            if checked >= 5:
                break

        if mismatches:
            logger.error("[WEIGHT CHECK] FAILED: Loaded weights do NOT match checkpoint!")
            for name, s, l in mismatches:
                logger.error(f"  {name}: checkpoint_norm={s:.4f}  loaded_norm={l:.4f}")
        else:
            logger.info(f"[WEIGHT CHECK] PASSED: {checked} parameters verified against {model_file.name}")
    except Exception as e:
        logger.warning(f"[WEIGHT CHECK] Could not verify weights: {e}")


@parser.wrap()
def async_inference_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    init_logging(display_pid=False)
    logger = logging.getLogger(__name__)

    pretrained_path = getattr(cfg.policy, "pretrained_path", None)
    logger.info(f"[CHECKPOINT CHECK] pretrained_path = {pretrained_path}")
    if not pretrained_path:
        logger.warning(
            "[CHECKPOINT CHECK] No checkpoint specified! "
            "The model will be randomly initialized. "
            "Pass --policy.pretrained_path <path> to load trained weights."
        )

    if getattr(cfg, "interactive", False):
        logger.warning(
            "[INTERACTIVE] cfg.interactive=true is ignored for pistar06: this "
            "policy does not consume subtask tokens, so there is nothing to "
            "override interactively. Continuing in non-interactive mode."
        )

    logger.info("Initializing Standalone Asynchronous PiStar06 Inference")

    set_seed(cfg.seed)

    device_name = cfg.policy.device
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = device.type  # Enforce propagation

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    def intercept_sigint(sig, frame):
        logger.info("\nCaught SIGINT! Shutting down async inference safely...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, intercept_sigint)

    if getattr(cfg, "use_rerun", False):
        import rerun as rr
        rr.init("lerobot_inference", spawn=True)

    logger.info("Instantiating policy architecture")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    _verify_loaded_weights(policy, cfg, logger)

    logger.info("Instantiating pistar06 processors")
    preprocessor, postprocessor = _make_pistar06_processors(cfg, logger)
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    policy = policy.to(device)
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        logger.error(
            "FATAL: RTC configuration is not populated or enabled in the config. "
            "Cannot run async inference on synchronous policy!"
        )
        sys.exit(1)

    logger.info("Instantiating generic online environment connection")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(
        online_env, teleop_device, cfg.env, cfg.policy.device
    )

    shared_state = SharedState()
    shared_state.running = not shutdown_event.is_set()

    shared_state.episode_logging_freq = cfg.episode_logging_freq
    shared_state.episode_save_freq = cfg.episode_save_freq
    shared_state.is_logging_episode = (
        shared_state.episode_counter % cfg.episode_logging_freq == 0
    )

    # Rollouts are recorded to disk as per-episode PNGs + MP4s by
    # `_save_episode_video` (see inference_pistar06_utils.env_interaction_worker).
    # The in-memory ReplayBuffer is *not* used for inference: it would
    # eagerly pre-allocate `capacity * frame_size` and quickly OOM the host
    # (200k * 2 cams * 224x224 bf16 ~= 60 GB), and we don't need to feed
    # rollouts back into a learner on this branch.
    shared_state.replay_buffer = None

    action_queue = ActionQueue(policy.config.rtc_config)

    logger.info("Spawning Inference worker thread.")
    inference_thread = Thread(
        target=get_actions_worker,
        args=(policy, shared_state, action_queue, cfg),
        daemon=True,
        name="get_actions_worker",
    )

    logger.info("Spawning Environment worker thread.")
    environment_thread = Thread(
        target=env_interaction_worker,
        args=(
            online_env,
            env_processor,
            action_processor,
            action_queue,
            shared_state,
            teleop_device,
            cfg,
            policy,
            policy.postprocessor,
        ),
        daemon=True,
        name="env_interaction_worker",
    )

    try:
        environment_thread.start()
        # Give environment thread 1 second to bootstrap the SharedState
        # before the inference loop goes wild.
        time.sleep(1.0)
        inference_thread.start()

        start_time = time.time()
        logger.info(
            "[MAIN] Successfully orchestrated asynchronous context. Awaiting KeyboardInterrupt to exit."
        )

        while not shutdown_event.is_set():
            time.sleep(5)

            q_size = action_queue.qsize() if action_queue is not None else 0
            teleop_stat = "ON" if shared_state.is_intervening else "OFF"

            metrics = shared_state.get_and_reset_metrics()
            inf_wait = metrics["inference_wait_time"]
            env_wait = metrics["env_wait_time"]
            env_steps = metrics["env_steps"]

            avg_env_wait = env_wait / max(1, env_steps)
            avg_env_active = metrics.get("env_active_time", 0.0) / max(1, env_steps)

            logger.info(
                f"[MAIN LOG] Queue Buffer Length: {q_size} | "
                f"Teleop Intervention: {teleop_stat} | "
                f"Runtime: {int(time.time() - start_time)}s"
            )
            logger.info(
                f"[metrics] Inference sleep time: {inf_wait:.2f}s | "
                f"Env active (camera/step) avg: {avg_env_active:.4f}s | "
                f"Env sleep avg: {avg_env_wait:.4f}s"
            )

    except Exception as e:
        logger.error(f"Error in main orchestration thread: {e}")
        logger.error(traceback.format_exc())
    finally:
        shutdown_event.set()
        shared_state.running = False

        logger.info("Executing safe un-spool connection teardowns.")
        if inference_thread.is_alive():
            inference_thread.join(timeout=3.0)
        if environment_thread.is_alive():
            environment_thread.join(timeout=3.0)

        try:
            online_env.close()
        except Exception:
            pass

        logger.info("Program terminated gracefully.")


if __name__ == "__main__":
    async_inference_cli()
