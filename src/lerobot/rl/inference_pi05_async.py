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

Usage:
    python lerobot/src/lerobot/rl/inference_pi05_async.py --config-path=config-hiserl.json
"""

import logging
import signal
import sys
import time
import traceback
from threading import Thread

import torch
from torch import nn

import lerobot.rl.rl_pi05  # noqa: F401  -- Important: registers PI05RLConfig ('pi05_rl') in the choice registry via import side-effects
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.gym_manipulator import make_processors, make_robot_env
from lerobot.rl.inference_utils import SharedState, env_interaction_worker, get_actions_worker
from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.rl.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


@parser.wrap()
def async_inference_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    init_logging(display_pid=False)
    logger = logging.getLogger(__name__)
    logger.info("Initializing Standalone Asynchronous Pi05 Inference")

    # Override configurations for isolated actor running
    # cfg.policy.use_separate_critic = False  # Critic is now needed for logging

    # Need to disable critic metrics so I can run inference on 4070 TI SUPER
    cfg.policy.use_separate_critic = False
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

    # Enable per-phase inference timing on the model. The flag is read by
    # PI05Policy.predict_action_chunk and PI05Pytorch.sample_actions, which
    # populate `policy.model._phase_timings_outer` and `policy.model._phase_timings`
    # for the inference worker to consume and log per chunk.
    if hasattr(policy, 'model'):
        policy.model._profile_inference = True

    # Validate RTC features
    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        logger.error("FATAL: RTC configuration is not populated or enabled in the config. Cannot run async inference on synchronous policy!")
        sys.exit(1)

    logger.info("Instantiating generic online environment connection")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    # Instantiate bridging components
    shared_state = SharedState()
    shared_state.running = not shutdown_event.is_set()

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
    logger.info("Spawning Inference worker thread.")
    inference_thread = Thread(
        target=get_actions_worker,
        args=(policy, shared_state, action_queue, cfg),
        daemon=True,
        name="get_actions_worker"
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

        start_time = time.time()
        logger.info("[MAIN] Successfully orchestrated asynchronous context. Awaiting KeyboardInterrupt to exit.")

        # Polling supervisor log loop.
        # Discard the warmup window (process-start through worker spawn) so the first
        # real window reflects only steady-state activity.
        shared_state.get_and_reset_metrics()
        target_fps = cfg.env.fps
        supervisor_period_s = 5
        while not shutdown_event.is_set():
            time.sleep(supervisor_period_s)

            q_size = action_queue.qsize() if action_queue is not None else 0
            teleop_stat = "ON" if shared_state.is_intervening else "OFF"

            metrics = shared_state.get_and_reset_metrics()
            inf_wait = metrics['inference_wait_time']
            env_wait = metrics['env_wait_time']
            env_steps = metrics['env_steps']
            window = metrics['window_elapsed']
            effective_hz = metrics['effective_hz']
            starvation_rate = metrics['starvation_rate']
            starved_steps = metrics['starved_steps']
            real_steps = env_steps - starved_steps
            real_hz = real_steps / window if window > 0 else 0.0
            inf_count = metrics['inference_count']
            inf_lat_avg = metrics['inference_latency_avg']
            inf_lat_max = metrics['inference_latency_max']
            inf_hz = metrics['inference_hz']

            avg_env_wait = env_wait / max(1, env_steps)
            avg_env_active = metrics.get('env_active_time', 0.0) / max(1, env_steps)

            logger.info(f"[MAIN LOG] Queue Buffer Length: {q_size} | Teleop Intervention: {teleop_stat} | Runtime: {int(time.time() - start_time)}s")
            logger.info(
                f"[control_rate] target={target_fps}Hz | effective={effective_hz:.2f}Hz "
                f"(real={real_hz:.2f}Hz, starved={starvation_rate * 100:.1f}% of {env_steps} steps over {window:.1f}s)"
            )
            logger.info(
                f"[inference] chunks={inf_count} ({inf_hz:.2f} chunks/s) | "
                f"latency avg={inf_lat_avg * 1000:.0f}ms max={inf_lat_max * 1000:.0f}ms | "
                f"sleep={inf_wait:.2f}s (>0 means queue is keeping up)"
            )
            logger.info(
                f"[env_loop] active avg={avg_env_active * 1000:.1f}ms | sleep avg={avg_env_wait * 1000:.1f}ms "
                f"(target step={1000.0 / target_fps:.1f}ms)"
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

        # Teardown camera hooks and serial ports
        try:
            online_env.close()
        except Exception:
            pass

        logger.info("Program termintated gracefully.")

if __name__ == "__main__":
    async_inference_cli()
