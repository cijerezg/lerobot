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
"""
Standalone Asynchronous Inference script for Pi05 policies.

This script runs the policy in inference mode using RTC asynchronously with
the environment. It delegates threading orchestration to `inference_utils.py`
and strictly avoids communicating with distributed learner servers, acting
as an isolated test bed for Pi05 robotics deployment.

When `interactive` is set to true in the config (or via CLI override), an
additional input thread is spawned so the operator can type subtask strings
into the terminal at any time, overriding the model's generated subtask
tokens until the next regeneration interval.

Usage:
    python lerobot/src/lerobot/rl/inference_pi05_async.py --config-path=config-hiserl.json
    python lerobot/src/lerobot/rl/inference_pi05_async.py --config-path=config-hiserl.json --interactive=true
"""

import logging
import os
import time
import sys
import traceback
import signal
from pathlib import Path
from threading import Thread

import torch
from torch import nn

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.rl.gym_manipulator import make_processors, make_robot_env
import lerobot.rl.rl_pi05  # Important: Register PI05RLConfig via import side-effects
from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.rl.inference_utils import SharedState, get_actions_worker, env_interaction_worker
from lerobot.rl.buffer import ReplayBuffer


def _verify_loaded_weights(policy, cfg, logger):
    """Verify that trained weights were actually loaded, not a random init."""
    checkpoint_path = getattr(cfg.policy, "pi05_checkpoint", "") or getattr(cfg.policy, "pretrained_path", "")

    with torch.no_grad():
        probe_keys = ["lm_head", "action_out_proj", "gemma_expert"]
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
    pi05_checkpoint = getattr(cfg.policy, "pi05_checkpoint", None)
    logger.info(f"[CHECKPOINT CHECK] pretrained_path = {pretrained_path}")
    logger.info(f"[CHECKPOINT CHECK] pi05_checkpoint = {pi05_checkpoint}")
    if not pretrained_path and not pi05_checkpoint:
        logger.warning(
            "[CHECKPOINT CHECK] No checkpoint specified! "
            "The model will be randomly initialized. "
            "Pass --policy.pi05_checkpoint <path> to load trained weights."
        )

    interactive = getattr(cfg, "interactive", False)
    mode_label = "Interactive " if interactive else ""
    logger.info(f"Initializing {mode_label}Standalone Asynchronous Pi05 Inference")

    # Override configurations for isolated actor running
    # cfg.policy.use_separate_critic = False  # Critic is now needed for logging
    set_seed(cfg.seed)

    device_name = getattr(cfg.policy, "actor_device", None)
    if device_name is None:
        device_name = cfg.policy.device
        
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = device.type  # Enforce propagation 

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup signal handler for graceful shutdown
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    
    # Optional override for debugging/stopping
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

    logger.info("Instantiating pi05 full processors + un-normalization upgrade hooks")
    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(
        cfg=cfg,
        dataset=None,
        is_main_process=True
    )
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    # Ensure policy runs entirely in eval
    policy = policy.to(device)
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    # Validate RTC features
    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        logger.error("FATAL: RTC configuration is not populated or enabled in the config. Cannot run async inference on synchronous policy!")
        sys.exit(1)

    logger.info("Instantiating generic online environment connection")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    # Instantiate bridging components -- use interactive shared state when requested
    if interactive:
        from lerobot.rl.inference_utils_interactive import (
            SharedStateInteractive,
            terminal_input_worker,
            get_actions_worker_interactive,
        )
        shared_state = SharedStateInteractive()
    else:
        shared_state = SharedState()

    shared_state.running = not shutdown_event.is_set()
    
    # Initialize Episode parameters
    shared_state.episode_logging_freq = cfg.episode_logging_freq
    shared_state.episode_save_freq = cfg.episode_save_freq

    shared_state.is_logging_episode = (shared_state.episode_counter % cfg.episode_logging_freq == 0)

    # Initialize ReplayBuffer for recording
    logger.info("Initializing ReplayBuffer for recording")
    state_keys = list(cfg.policy.input_features.keys())
    replay_buffer = ReplayBuffer(
        capacity=cfg.policy.online_buffer_capacity,
        device=device.type,
        state_keys=state_keys,
        storage_device="cpu", # Keep on CPU
    )
    shared_state.replay_buffer = replay_buffer

    action_queue = ActionQueue(policy.config.rtc_config)

    # Spawn daemonized Thread Wrappers
    if interactive:
        logger.info("Spawning Input worker thread.")
        input_thread = Thread(
            target=terminal_input_worker,
            args=(shared_state, policy, cfg, shutdown_event),
            daemon=True,
            name="InputThread",
        )

        logger.info("Spawning Inference worker thread (interactive).")
        inference_thread = Thread(
            target=get_actions_worker_interactive,
            args=(policy, shared_state, action_queue, cfg),
            daemon=True,
            name="get_actions_worker_interactive",
        )
    else:
        input_thread = None
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
        args=(online_env, env_processor, action_processor, action_queue, shared_state, teleop_device, cfg, policy, policy.postprocessor),
        daemon=True,
        name="env_interaction_worker"
    )

    try:
        environment_thread.start()
        # Give environment thread 1 second to bootstrap the `SharedState` before the inference loop goes wild
        time.sleep(1.0) 
        inference_thread.start()
        if input_thread is not None:
            input_thread.start()

        start_time = time.time()
        logger.info("[MAIN] Successfully orchestrated asynchronous context. Awaiting KeyboardInterrupt to exit.")

        # Polling supervisor log loop
        while not shutdown_event.is_set():
            time.sleep(5)
            
            q_size = action_queue.qsize() if action_queue is not None else 0
            teleop_stat = "ON" if shared_state.is_intervening else "OFF"
            
            metrics = shared_state.get_and_reset_metrics()
            inf_wait = metrics['inference_wait_time']
            env_wait = metrics['env_wait_time']
            env_steps = metrics['env_steps']
            
            avg_env_wait = env_wait / max(1, env_steps)
            avg_env_active = metrics.get('env_active_time', 0.0) / max(1, env_steps)
            
            logger.info(f"[MAIN LOG] Queue Buffer Length: {q_size} | Teleop Intervention: {teleop_stat} | Runtime: {int(time.time() - start_time)}s")
            logger.info(f"[metrics] Inference sleep time: {inf_wait:.2f}s | Env active (camera/step) avg: {avg_env_active:.4f}s | Env sleep avg: {avg_env_wait:.4f}s")

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

        # Teardown camera hooks and serial ports
        try:
            online_env.close()
        except Exception:
            pass

        logger.info("Program termintated gracefully.")

if __name__ == "__main__":
    async_inference_cli()
