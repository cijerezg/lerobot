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
Actor server runner for distributed HILSerl robot policy training with Pi05.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.
"""

import logging
import os
import time
from functools import lru_cache
from queue import Empty

import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from lerobot.rl.gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
import lerobot.rl.rl_pi05  # Register PI05RLConfig
from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.policies.rtc.action_queue import ActionQueue

from lerobot.rl.actor import (
    use_threads,
    learner_service_client,
    establish_learner_connection,
    receive_policy,
    send_transitions,
    send_interactions,
    log_policy_frequency_issue,
    get_frequency_stats,
    push_transitions_to_transport_queue,
)

import math

# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue(maxsize=2)
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy_async(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    # Signal background threads to stop
    shutdown_event.set()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy_async(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment using async inference.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    set_seed(cfg.seed)
    
    device_name = getattr(cfg.policy, "actor_device", None)
    if device_name is None:
        device_name = cfg.policy.device
        
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = device.type

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Instantiating policy architecture")
    cfg.policy.use_separate_critic = False
    
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(
        cfg=cfg,
        dataset=None,
        is_main_process=True
    )
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    policy = policy.to(device)
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        logging.error("FATAL: RTC configuration is not populated or enabled in the config. Cannot run async actor!")
        return

    logging.info("Instantiating generic online environment connection")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    # Instantiate bridging components
    from lerobot.rl.actor_pi05_async_utils import (
        SharedStateActor,
        get_actions_worker_actor,
        env_interaction_worker_actor,
    )
    from threading import Thread

    shared_state = SharedStateActor()
    shared_state.running = not shutdown_event.is_set()
    
    action_queue = ActionQueue(policy.config.rtc_config)

    logging.info("Spawning Inference worker thread.")
    inference_thread = Thread(
        target=get_actions_worker_actor,
        args=(policy, shared_state, action_queue, parameters_queue, device, cfg),
        daemon=True,
        name="get_actions_worker_actor"
    )

    logging.info("Spawning Environment worker thread.")
    environment_thread = Thread(
        target=env_interaction_worker_actor,
        args=(online_env, env_processor, action_processor, action_queue, shared_state, teleop_device, transitions_queue, interactions_queue, cfg, policy.postprocessor),
        daemon=True,
        name="env_interaction_worker_actor"
    )

    try:
        environment_thread.start()
        time.sleep(1.0) 
        inference_thread.start()

        start_time = time.time()
        logging.info("[MAIN] Successfully orchestrated asynchronous actor context.")

        while not shutdown_event.is_set():
            time.sleep(5)
            
            q_size = action_queue.qsize() if action_queue is not None else 0
            teleop_stat = "ON" if shared_state.is_intervening else "OFF"
            episode_stat = "ON" if shared_state.episode_active else "OFF"
            
            metrics = shared_state.get_and_reset_metrics()
            inf_wait = metrics['inference_wait_time']
            env_wait = metrics['env_wait_time']
            env_steps = metrics['env_steps']
            
            avg_env_wait = env_wait / max(1, env_steps)
            avg_env_active = metrics.get('env_active_time', 0.0) / max(1, env_steps)
            
            inf_count = metrics.get('inference_count', 0)
            inf_lats = metrics.get('inference_latencies', [])
            if inf_lats:
                avg_lat = sum(inf_lats) / len(inf_lats)
                min_lat = min(inf_lats)
                max_lat = max(inf_lats)
                lat_str = f"avg={avg_lat:.3f}s  min={min_lat:.3f}s  max={max_lat:.3f}s"
            else:
                lat_str = "N/A"
            
            logging.debug(f"[MAIN LOG] Episode: {episode_stat} | Queue Buffer: {q_size} | Teleop: {teleop_stat} | Runtime: {int(time.time() - start_time)}s")
            
            # Only log detailed metrics when something is concerning
            action_interval = 1.0 / cfg.env.fps
            is_concerning = (
                avg_env_active > action_interval or   # env exceeding FPS target
                q_size < 5 or                          # queue dangerously low
                (inf_lats and max_lat > 1.0) or        # extreme latency spike
                (inf_count > 0 and inf_wait < 0.01)    # GPU never resting
            )
            if is_concerning:
                logging.info(f"[metrics] Inference sleep time: {inf_wait:.2f}s | Env active avg: {avg_env_active:.4f}s | Env sleep avg: {avg_env_wait:.4f}s")
                logging.info(f"[metrics] Inference count: {inf_count} (in 5s) | Chunk latency: {lat_str}")

    except Exception as e:
        logging.error(f"Error in main orchestration thread: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        shutdown_event.set()
        shared_state.running = False
        
        logging.info("Executing safe un-spool connection teardowns.")
        if inference_thread.is_alive():
            inference_thread.join(timeout=3.0)
        if environment_thread.is_alive():
            environment_thread.join(timeout=3.0)

        try:
            online_env.close()
        except Exception:
            pass

        logging.info("Actor worker threads stopped.")

if __name__ == "__main__":
    actor_cli()
