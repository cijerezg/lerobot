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
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
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
from lerobot.utils.robot_utils import busy_wait
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

    parameters_queue = Queue()
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

    act_with_policy(
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


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")


    
    set_seed(cfg.seed)
    
    # Use actor_device if specified, otherwise fallback to policy.device
    device_name = getattr(cfg.policy, "actor_device", None)
    if device_name is None:
        device_name = cfg.policy.device
        
    device = get_safe_torch_device(device_name, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    # Initialize preprocessor and postprocessor
    # Load stats directly from the pretrained checkpoint
    # This ensures we use the exact same stats the policy was trained with
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pi05_checkpoint,
    )
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)
    
    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)
    was_intervening = False

    try:
        for interaction_step in range(cfg.policy.online_steps):
            start_time = time.perf_counter()
            if shutdown_event.is_set():
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            # Build batch for preprocessor (needs all observation keys + task)
            batch_for_preprocessor = {}
            for k, v in transition[TransitionKey.OBSERVATION].items():
                if k in cfg.policy.input_features:
                    batch_for_preprocessor[k] = v
            
            # Add task and robot_type - observations are already processed tensors from env_processor
            # (already normalized [0,1], in CHW format, with batch dimension, on device)
            batch_for_preprocessor["task"] = cfg.policy.task
            batch_for_preprocessor["robot_type"] = online_env.robot.robot_type if hasattr(online_env, 'robot') else ""
            batch_for_preprocessor["advantage"] = cfg.policy.inference_advantage

            # Time policy inference and check if it meets FPS requirement
            with policy_timer:

                with torch.no_grad():
                    # Apply preprocessor if available (handles tokenization, state padding, etc.)
                    # NOTE: Don't use prepare_observation_for_inference here! That's for raw numpy arrays.
                    # env_processor has already converted observations to proper tensor format.
                    if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
                        processed_batch = policy.preprocessor(batch_for_preprocessor)
                    else:
                        processed_batch = batch_for_preprocessor
                    
                    # PI05RLPolicy.select_action handles advantage injection internally (default 1.0)
                    action = policy.select_action(processed_batch)

                    # Apply postprocessor if available (handles unnormalization)
                    if hasattr(policy, 'postprocessor') and policy.postprocessor is not None:
                        # Slice action to 6 dimensions as requested
                        if action.shape[-1] > 6:
                            action = action[..., :6]
                        
                        # Clamp to [-1, 1] to ensure safety and prevent violent movements
                        #clamp_val = 0.1 + 1.9 / (1 + math.exp(-0.001 * (interaction_step - 5000)))
                        #action = torch.clamp(action, lamp_val, clamp_val)

                        action = policy.postprocessor(action)
                        
                        
            policy_fps = policy_timer.fps_last

            log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

            # Use the new step function
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            # Extract values from processed transition
            # (Conversion to policy format happens later when creating the transition)

            # Teleop action is the action that was executed in the environment
            # It is either the action from the teleop device or the action from the policy
            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

            reward = new_transition[TransitionKey.REWARD]

            if reward > 0:
                logging.info(f"[ACTOR] Received transition with reward: {reward}")

            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)

            sum_reward_episode += float(reward)
            episode_total_steps += 1


            # Check for intervention from transition info
            intervention_info = new_transition[TransitionKey.INFO]
            is_intervening = intervention_info.get(TeleopEvents.IS_INTERVENTION, False)

            if was_intervening and not is_intervening:
                policy.reset()
            
            was_intervening = is_intervening   

            if is_intervening:
                episode_intervention = True
                episode_intervention_steps += 1
            else:
                # Send feedback to teleop device if not intervening
                # Extract raw joint positions from observation
                intervention_reset_policy = True
                feedback = {}
                for key, value in new_transition[TransitionKey.OBSERVATION].items():
                    if key.endswith(".pos"):
                        # value is likely a tensor, convert to float
                        if isinstance(value, torch.Tensor):
                            feedback[key] = value.item()
                        else:
                            feedback[key] = float(value)

                if feedback:
                    teleop_device.send_feedback(feedback)

         

            complementary_info = {
                "discrete_penalty": torch.tensor(
                    [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
                ),
                TeleopEvents.IS_INTERVENTION.value: torch.tensor([float(is_intervening)], dtype=torch.float32),
            }

            # Convert environment observations to policy-expected format
            def convert_env_obs_to_policy_format(env_obs: dict) -> dict:
                """Convert environment observation format to policy-expected format.
                
                Handles two cases:
                1. Observations already in policy format (observation.images.xxx, observation.state)
                2. Raw environment format (pixels dict or individual image keys, individual .pos keys)
                """
                policy_obs = {}
                
                # Check if observations are already in the correct format
                # Relaxed check: if we have images OR state in policy format, we try to preserve them
                has_policy_format = (
                    'observation.state' in env_obs or
                    any(k.startswith('observation.images.') for k in env_obs.keys())
                )
                
                if has_policy_format:
                    # Observations are already in policy format (partial or full), just filter the relevant keys
                    for key in env_obs.keys():
                        if key == 'observation.state' or key.startswith('observation.images.'):
                            policy_obs[key] = env_obs[key]
                
                # If we are missing state or images, try to find them in raw format
                # Images
                camera_mapping = {
                    'wrist': 'observation.images.wrist',
                    'top': 'observation.images.top',
                    'side': 'observation.images.side',
                }
                
                pixels_dict = env_obs.get('pixels', env_obs)
                
                for env_key, policy_key in camera_mapping.items():
                    if policy_key not in policy_obs: # Only look if not already found
                        if env_key in pixels_dict:
                            policy_obs[policy_key] = pixels_dict[env_key]
                
                # State
                if 'observation.state' not in policy_obs:
                     # Collect joint positions in the correct order (matching SO101 motor order)
                    joint_order = [
                        'shoulder_pan.pos',
                        'shoulder_lift.pos',
                        'elbow_flex.pos',
                        'wrist_flex.pos',
                        'wrist_roll.pos',
                        'gripper.pos',
                    ]
                    
                    joint_values = []
                    for joint_key in joint_order:
                        if joint_key in env_obs:
                            val = env_obs[joint_key]
                            # Convert to tensor if not already
                            if not isinstance(val, torch.Tensor):
                                val = torch.tensor([val], dtype=torch.float32)
                            elif val.dim() == 0:
                                val = val.unsqueeze(0)
                            joint_values.append(val)
                    
                    # Concatenate all joint positions into a single state tensor
                    if joint_values:
                        policy_obs['observation.state'] = torch.cat(joint_values, dim=0)
                
                return policy_obs
            
            # Create transition for learner (convert to policy format)
            observation = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
            next_observation = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
            
            list_transition_to_send_to_learner.append(
                Transition(
                    state=observation,
                    action=executed_action[:6],
                    reward=reward,
                    next_state=next_observation,
                    done=done,
                    truncated=truncated,
                    complementary_info=complementary_info,
                )
            )
            
            # Update transition for next iteration
            transition = new_transition

            if done or truncated:
                logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

                update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

                if len(list_transition_to_send_to_learner) > 0:
                    push_transitions_to_transport_queue(
                        transitions=list_transition_to_send_to_learner,
                        transitions_queue=transitions_queue,
                    )
                    list_transition_to_send_to_learner = []

                stats = get_frequency_stats(policy_timer)
                policy_timer.reset()

                # Calculate intervention rate
                intervention_rate = 0.0
                if episode_total_steps > 0:
                    intervention_rate = episode_intervention_steps / episode_total_steps

                # Send episodic reward to the learner
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention),
                            "Intervention rate": intervention_rate,
                            **stats,
                        }
                    )
                )

                # Reset intervention counters and environment
                sum_reward_episode = 0.0
                episode_intervention = False
                episode_intervention_steps = 0
                episode_total_steps = 0

                # Wait for '2' key on teleop device to start next episode
                logging.info("[ACTOR] Episode ended. Press '2' on the keyboard to start the next episode...")
                while not shutdown_event.is_set():
                    if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                        break
                    time.sleep(0.1)
                
                if shutdown_event.is_set():
                    break
                
                logging.info("[ACTOR] Starting next episode.")

                # Reset environment and processors
                obs, info = online_env.reset()
                env_processor.reset()
                action_processor.reset()
                policy.reset()

                # Process initial observation
                transition = create_transition(observation=obs, info=info)
                transition = env_processor(transition)

            if cfg.env.fps is not None:
                dt_time = time.perf_counter() - start_time
                busy_wait(1 / cfg.env.fps - dt_time)
    finally:
        pass


def update_policy_parameters(policy, parameters_queue: Queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        # For Pi05, the actor is the model
        if hasattr(policy, "actor"):
             policy.actor.load_state_dict(actor_state_dict)
        else:
             policy.model.load_state_dict(actor_state_dict)

if __name__ == "__main__":
    actor_cli()
