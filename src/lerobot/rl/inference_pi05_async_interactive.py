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
Interactive Asynchronous Inference script for Pi05 policies.

Identical to inference_pi05_async.py except the operator can type a subtask
string into the terminal at any time. The typed text is tokenized and injected
into the policy's subtask token cache, taking effect on the very next action
chunk generation. The model's normal time-based cache then resumes.

Usage:
    python lerobot/src/lerobot/rl/inference_pi05_async_interactive.py --config-path=config-hiserl.json

Config note:
    subtask_regeneration_interval must be > 0 (e.g. 30). If it is 0 the model
    regenerates subtask tokens on every cycle and injected overrides are
    immediately overwritten.
"""

import logging
import time
import sys
import traceback
import signal
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
from lerobot.rl.inference_utils import env_interaction_worker  # reused unchanged
from lerobot.rl.buffer import ReplayBuffer

# Interactive-specific additions
from lerobot.rl.inference_utils_interactive import (
    SharedStateInteractive,
    terminal_input_worker,
    get_actions_worker_interactive,
)


@parser.wrap()
def async_inference_interactive_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    init_logging(display_pid=False)
    logger = logging.getLogger(__name__)
    logger.info("Initializing Interactive Asynchronous Pi05 Inference")

    set_seed(cfg.seed)

    device_name = getattr(cfg.policy, "actor_device", None)
    if device_name is None:
        device_name = cfg.policy.device

    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = device.type

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    def intercept_sigint(sig, frame):
        logger.info("\nCaught SIGINT! Shutting down interactive inference safely...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, intercept_sigint)

    if getattr(cfg, "use_rerun", False):
        import rerun as rr
        rr.init("lerobot_inference_interactive", spawn=True)

    logger.info("Instantiating policy architecture")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)

    logger.info("Instantiating pi05 full processors + un-normalization upgrade hooks")
    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(
        cfg=cfg, dataset=None, is_main_process=True
    )
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    policy = policy.to(device)
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        logger.error(
            "FATAL: RTC configuration is not populated or enabled. "
            "Cannot run async inference on a synchronous policy!"
        )
        sys.exit(1)

    logger.info("Instantiating generic online environment connection")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    # Use the interactive shared state instead of plain SharedState
    shared_state = SharedStateInteractive()
    shared_state.running = not shutdown_event.is_set()
    shared_state.episode_logging_freq = cfg.episode_logging_freq
    shared_state.episode_save_freq = cfg.episode_save_freq
    shared_state.is_logging_episode = (shared_state.episode_counter % cfg.episode_logging_freq == 0)

    logger.info("Initializing ReplayBuffer for recording")
    state_keys = list(cfg.policy.input_features.keys())
    replay_buffer = ReplayBuffer(
        capacity=cfg.policy.online_buffer_capacity,
        device=device.type,
        state_keys=state_keys,
        storage_device="cpu",
    )
    shared_state.replay_buffer = replay_buffer

    action_queue = ActionQueue(policy.config.rtc_config)

    # Three threads: input, inference (interactive), environment
    logger.info("Spawning Input worker thread.")
    input_thread = Thread(
        target=terminal_input_worker,
        args=(shared_state, policy, cfg, shutdown_event),
        daemon=True,
        name="InputThread",
    )

    logger.info("Spawning Inference worker thread.")
    inference_thread = Thread(
        target=get_actions_worker_interactive,
        args=(policy, shared_state, action_queue, cfg),
        daemon=True,
        name="get_actions_worker_interactive",
    )

    logger.info("Spawning Environment worker thread.")
    environment_thread = Thread(
        target=env_interaction_worker,
        args=(online_env, env_processor, action_processor, action_queue, shared_state, teleop_device, cfg, policy, policy.postprocessor),
        daemon=True,
        name="env_interaction_worker",
    )

    try:
        environment_thread.start()
        time.sleep(1.0)  # let env thread bootstrap SharedState before inference starts
        inference_thread.start()
        input_thread.start()

        start_time = time.time()
        logger.info(
            "[MAIN] Interactive inference running. "
            "Type a subtask in the terminal at any time. Awaiting KeyboardInterrupt to exit."
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
                f"[MAIN LOG] Queue Buffer Length: {q_size} | Teleop Intervention: {teleop_stat} | "
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
        # input_thread is daemon — Python will not wait for it

        try:
            online_env.close()
        except Exception:
            pass

        logger.info("Program terminated gracefully.")


if __name__ == "__main__":
    async_inference_interactive_cli()
