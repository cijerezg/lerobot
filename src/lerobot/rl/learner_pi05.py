# !/usr/bin/env python

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
Learner server runner for distributed HILSerl robot policy training with Pi05.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import cv2

episode_logging_freq = 5
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

import torch.nn.functional as F
import lerobot.rl.rl_pi05  # Register PI05RLConfig

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

from lerobot.rl.learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService
from lerobot.rl.learner import (
    use_threads,
    handle_resume_logic,
    start_learner,
    process_transitions,
    process_interaction_messages,
    check_nan_in_transition,
    get_observation_features,
    push_actor_policy_to_queue,
    make_optimizers_and_scheduler,
    load_training_state,
    initialize_replay_buffer,
    initialize_offline_replay_buffer,
    log_training_info,
    save_training_checkpoint,
)

import wandb
                

@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


# Core algorithm functions


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device_name = getattr(cfg.policy, "learner_device", None)
    if device_name is None:
        device_name = cfg.policy.device
    device = get_safe_torch_device(try_device=device_name, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    gradient_accumulation_steps = getattr(cfg.policy, "gradient_accumulation_steps", 1)
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch
    online_buffer_save_freq = getattr(cfg, "online_buffer_save_freq", None)

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    # Override device to ensure policy is created on the correct device
    original_device = cfg.policy.device
    cfg.policy.device = device_name
    
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    
    # Restore original device config
    cfg.policy.device = original_device

    # NOTE: Do not call policy.to(dtype=...) here!
    # The dtype is already correctly handled during PI05 initialization:
    # - PaliGemmaWithExpertModel handles bfloat16 conversion selectively
    # - action_in_proj and action_out_proj remain in float32 
    # - Calling .to(dtype=bfloat16) would override this and break the model

    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    # Freeze ALL parameters except the LAST layer of gemma_expert (for minimal testing)
    # gemma_expert has layers 0-17, so layer 17 is the last one
    logging.info("Freezing ALL parameters except last gemma_expert layer (minimal mode)...")
    for name, param in policy.named_parameters():
        # Only train layer 17 of gemma_expert (the last layer)
        param.requires_grad = (
            "gemma_expert" in name or
            ("critic" in name and "embed_tokens" not in name) or
            "log_alpha" in name or
            ("language_model" in name and any(f".{i}." in name for i in [0, 1, 2, 3, 4, 12, 13, 14, 15, 16, 17])) or
            "language_model.norm" in name or
            ("vision_tower" in name and any(f".{i}." in name for i in [18, 19, 20, 21, 22, 23, 24, 25, 26]))
        )
    
    # Log trainable parameters
    trainable_params = [n for n, p in policy.named_parameters() if p.requires_grad]
    logging.info(f"MINIMAL MODE: Trainable parameters: {len(trainable_params)}")
    logging.info(f"Trainable parameter names: {trainable_params}")  # Show what's being trainedebug

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    replay_buffer.reward_normalization_constant = cfg.policy.reward_normalization_constant
    replay_buffer.terminal_failure_reward = cfg.policy.terminal_failure_reward
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        # Inline initialize_offline_replay_buffer to allow modifying dataset before buffer creation
        if not cfg.resume:
            logging.info("make_dataset offline buffer")
            offline_dataset = make_dataset(cfg)
        else:
            logging.info("load offline dataset")
            dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
            offline_dataset = LeRobotDataset(
                repo_id=cfg.dataset.repo_id,
                root=dataset_offline_path,
            )

        # Disable delta_timestamps to avoid chunking (we want single frames for the buffer)
        offline_dataset.delta_timestamps = None
        offline_dataset.delta_indices = None

        logging.info("Convert to a offline replay buffer")
        offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
            offline_dataset,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
            capacity=cfg.policy.offline_buffer_capacity,
            reward_normalization_constant=cfg.policy.reward_normalization_constant,
            terminal_failure_reward=cfg.policy.terminal_failure_reward,
        )
        offline_replay_buffer.dataset = offline_dataset
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    # import pdb; pdb.set_trace()

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id
    

    # Create preprocessor and postprocessor for the policy
    # Load stats directly from the pretrained checkpoint
    # This ensures we use the exact same stats the policy was trained with
    from lerobot.policies.factory import make_pre_post_processors
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pi05_checkpoint,
    )
    # Store preprocessors on the policy for actor to access
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    while True:
        # Exit the training loop if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # Process all available interaction messages sent by the actor server
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # Track episodes for logging (every N episodes)
        if not hasattr(add_actor_information_and_train, "episode_counter"):
            add_actor_information_and_train.episode_counter = [0]

        # Process all available transitions to the replay buffer, send by the actor server
        process_transitions_pi05(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
            policy=policy,
            episode_counter=add_actor_information_and_train.episode_counter,
            cfg=cfg,
        )


        # Wait until the replay buffer has enough samples to start training
        if len(replay_buffer) < online_step_before_learning:
            continue

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size,
                async_prefetch=async_prefetch,
                queue_size=2,
                action_chunk_size=cfg.policy.n_action_steps,
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size,
                async_prefetch=async_prefetch,
                queue_size=2,
                action_chunk_size=cfg.policy.n_action_steps,
            )

        if cfg.policy.dtype == "bfloat16":
            # Helper function to cast tensors in a structure
            def cast_to_bf16(item):
                if isinstance(item, torch.Tensor):
                    if item.dtype == torch.float32:
                        return item.to(dtype=torch.bfloat16)
                    return item
                elif isinstance(item, dict):
                    return {k: cast_to_bf16(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [cast_to_bf16(v) for v in item]
                return item

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # Gradient accumulation for critic
            optimizers["critic"].zero_grad()
            for accum_step in range(gradient_accumulation_steps):
                # Sample from the iterators
                batch = next(online_iterator)

                # Ensure online batch actions are 6-dim (if buffer is 32-dim)
                if batch[ACTION].shape[-1] > 6:
                     batch[ACTION] = batch[ACTION][..., :6]

                if dataset_repo_id is not None:
                    batch_offline = next(offline_iterator)
                    # Slice offline actions to match online actions (6 dims)
                    batch_offline[ACTION] = batch_offline[ACTION][..., :6]
                    batch = concatenate_batch_transitions(
                        left_batch_transitions=batch, right_batch_transition=batch_offline
                    )

                # Move batch to device
                batch = move_transition_to_device(batch, device)

                if cfg.policy.dtype == "bfloat16":
                    # Manual casting for now
                    if isinstance(batch, dict):
                        batch = {k: cast_to_bf16(v) for k, v in batch.items()}
                    else:
                        new_batch_data = {}
                        for field in batch._fields:
                            val = getattr(batch, field)
                            new_batch_data[field] = cast_to_bf16(val)
                        
                        batch = type(batch)(**new_batch_data)

                actions = batch[ACTION]
                rewards = batch["reward"]
                observations = batch["state"]
                next_observations = batch["next_state"]
                done = batch["done"]
                current_batch_size = actions.shape[0]
                check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

                observation_features, next_observation_features = get_observation_features(
                    policy=policy, observations=observations, next_observations=next_observations
                )

                # Create a batch dictionary with all required elements for the forward method
                forward_batch = {
                    ACTION: actions,
                    "reward": rewards,
                    "state": observations,
                    "next_state": next_observations,
                    "done": done,
                    "observation_feature": observation_features,
                    "next_observation_feature": next_observation_features,
                    "task": [cfg.policy.task] * current_batch_size,
                    "advantage": torch.full((current_batch_size, 1), cfg.policy.inference_advantage, device=device),
                }

                # --- Preprocessing for Pi05 (Tokenization and Normalization) ---
                # Preprocess current observations
                batch_for_proc = {k: v for k, v in observations.items()}
                batch_for_proc["task"] = forward_batch["task"]
                batch_for_proc[ACTION] = actions
                batch_for_proc["advantage"] = forward_batch["advantage"]
                
                with torch.no_grad():
                    processed_batch = policy.preprocessor(batch_for_proc)
                
                # Preprocess next observations (for critic)
                next_batch_for_proc = {k: v for k, v in next_observations.items()}
                next_batch_for_proc["task"] = forward_batch["task"]
                next_batch_for_proc[ACTION] = actions  # Not used, but required by preprocessor
                next_batch_for_proc["advantage"] = forward_batch["advantage"]
                
                with torch.no_grad():
                    processed_next_batch = policy.preprocessor(next_batch_for_proc)
                
                from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
                
                # Update forward_batch with tokens and normalized actions
                # (We keep raw images in forward_batch["state"] so policy.forward can re-preprocess if needed)
                
                # Add tokens and normalized actions
                forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
                forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
                
                # Add tokens for next state (CRITICAL for critic)
                forward_batch["next_state"][OBS_LANGUAGE_TOKENS] = processed_next_batch[OBS_LANGUAGE_TOKENS]
                forward_batch["next_state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_ATTENTION_MASK]
                
                forward_batch[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
                forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
                forward_batch[ACTION] = processed_batch[ACTION]

                # --- Pi05 RL Update Logic ---
                
                # 1. Critic Update (Value Function) with scaled loss
                critic_output = policy.forward(forward_batch, model="critic")
                loss_critic = critic_output["loss_critic"] / gradient_accumulation_steps
                loss_critic.backward()
            
            # Clip and step after accumulation
            torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()
            
            # 2. Update Target Networks
            policy.update_target_networks()
            
            # ----------------------------

        # Sample for the last update in the UTD ratio
        # Gradient accumulation for critic
        optimizers["critic"].zero_grad()
        for accum_step in range(gradient_accumulation_steps):
            batch = next(online_iterator)

        # Ensure online batch actions are 6-dim (if buffer is 32-dim)
        if batch[ACTION].shape[-1] > 6:
                batch[ACTION] = batch[ACTION][..., :6]

        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            # Slice offline actions to match online actions (6 dims)
            batch_offline[ACTION] = batch_offline[ACTION][..., :6]
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        # Move batch to device
        batch = move_transition_to_device(batch, device)

        if cfg.policy.dtype == "bfloat16":
            # Manual casting for now
            if isinstance(batch, dict):
                batch = {k: cast_to_bf16(v) for k, v in batch.items()}
            else:
                new_batch_data = {}
                for field in batch._fields:
                    val = getattr(batch, field)
                    new_batch_data[field] = cast_to_bf16(val)
                
                batch = type(batch)(**new_batch_data)

        actions = batch[ACTION]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        current_batch_size = actions.shape[0]

        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            ACTION: actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "task": [cfg.policy.task] * current_batch_size,
            "advantage": torch.full((current_batch_size, 1), cfg.policy.inference_advantage, device=device),
        }

        # --- Preprocessing for Pi05 (Tokenization and Normalization) ---
        # Preprocess current observations
        batch_for_proc = {k: v for k, v in observations.items()}
        batch_for_proc["task"] = forward_batch["task"]
        batch_for_proc[ACTION] = actions
        batch_for_proc["advantage"] = forward_batch["advantage"]
        
        with torch.no_grad():
            processed_batch = policy.preprocessor(batch_for_proc)
        
        # Preprocess next observations (for critic)
        next_batch_for_proc = {k: v for k, v in next_observations.items()}
        next_batch_for_proc["task"] = forward_batch["task"]
        next_batch_for_proc[ACTION] = actions  # Not used, but required by preprocessor
        next_batch_for_proc["advantage"] = forward_batch["advantage"]
        
        with torch.no_grad():
            processed_next_batch = policy.preprocessor(next_batch_for_proc)
        
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        
        # Update forward_batch with tokens and normalized actions
        # (We keep raw images in forward_batch["state"] so policy.forward can re-preprocess if needed)
        
        # Add tokens and normalized actions
        # Add tokens and normalized actions
        # Add tokens and normalized actions
        forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Add critic tokens if present (from CriticTokenizerProcessorStep)
        if "critic_tokens" in processed_batch:
            forward_batch["critic_tokens"] = processed_batch["critic_tokens"]
            forward_batch["critic_pad_mask"] = processed_batch["critic_pad_mask"]
        
        # Add tokens for next state (CRITICAL for critic)
        forward_batch["next_state"][OBS_LANGUAGE_TOKENS] = processed_next_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["next_state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_ATTENTION_MASK]
        
        if "critic_tokens" in processed_next_batch:
            forward_batch["next_state"]["critic_tokens"] = processed_next_batch["critic_tokens"]
            forward_batch["next_state"]["critic_pad_mask"] = processed_next_batch["critic_pad_mask"]
        
        forward_batch[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
        forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
        forward_batch[ACTION] = processed_batch[ACTION]

        # --- Pi05 RL Update Logic (Last Step) ---
        
        # Initialize metric accumulators for critic
        accum_loss_critic = 0.0
        critic_values_list = []  # For histogram logging
        td_error_list = []
        target_values_list = []
        
        # Critic Update with scaled loss and metric accumulation
        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"] / gradient_accumulation_steps
        loss_critic.backward()
        
        # Accumulate metrics
        accum_loss_critic += critic_output["loss_critic"].item()
        critic_values_list.append(critic_output["critic_values"])
        td_error_list.append(critic_output["td_error"])
        target_values_list.append(critic_output["target_values"])
        
        # Clip and step after accumulation
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        # Compute aggregated metrics
        all_critic_values = torch.cat(critic_values_list, dim=0)
        all_td_errors = torch.cat(td_error_list, dim=0)
        all_target_values = torch.cat(target_values_list, dim=0)

        # Initialize training info dictionary
        training_infos = {
            "loss_critic": accum_loss_critic / gradient_accumulation_steps,
            "critic_grad_norm": critic_grad_norm,
            "td_error_mean": all_td_errors.mean().item(),
            "td_error_std": all_td_errors.std().item() if all_td_errors.numel() > 1 else 0.0,
            "critic_value_mean": all_critic_values.mean().item(),
            "critic_value_std": all_critic_values.std().item() if all_critic_values.numel() > 1 else 0.0,
            "target_value_mean_critic": all_target_values.mean().item(),
            "target_value_std": all_target_values.std().item() if all_target_values.numel() > 1 else 0.0,
        }
        
        # Store critic histogram from critic update
        training_infos["critic_histogram_from_critic"] = torch.cat(critic_values_list, dim=0)
            
        # Actor update (at specified frequency)
        # Skip actor update during critic warmup
        if optimization_step >= critic_warmup_steps and optimization_step % policy_update_freq == 0:
            # --- Pass 2: Calculate Advantage and Re-tokenize ---
            # We need to calculate the advantage using the *updated* critic
            # and then re-tokenize the batch with the correct advantage labels.
            
            # 1. Calculate Advantage (Raw)
            # We use the same batch as the last critic update step (or we could re-sample, but reusing is more efficient/consistent)
            # Note: 'batch' here is the last batch from the critic accumulation loop.
            # Ideally we should do this for ALL batches in the accumulation steps if we want to accumulate actor gradients too.
            # But usually we just sample new batches for actor update?
            # The original code sampled new batches for actor update.
            # Let's stick to the original logic of sampling new batches for actor update, 
            # BUT we need to process them twice (Pass 1 for Critic Value, Pass 2 for Actor Tokenization).
            
            for _ in range(policy_update_freq):
                # Gradient accumulation for actor
                optimizers["actor"].zero_grad()
                
                # Initialize metric lists for actor
                accum_loss_actor = 0.0
                advantage_values_list = []
                critic_values_list = []
                target_values_list = []
                reward_values_list = []
                
                for accum_step in range(gradient_accumulation_steps):
                    # Sample NEW batch for actor update
                    batch = process_transitions_pi05(
                        transition_queue, 
                        batch_size, 
                        offline_dataset, 
                        offline_batch_size, 
                        online_buffer, 
                        online_batch_size,
                        online_sampling_ratio,
                        return_torch=True,
                        device=device
                    )
                    batch[ACTION] = batch[ACTION][..., :6]
                    
                    if cast_to_bf16 is not None:
                         if isinstance(batch, dict):
                            batch = {k: cast_to_bf16(v) for k, v in batch.items()}
                         else:
                            new_batch_data = {}
                            for field in batch._fields:
                                val = getattr(batch, field)
                                new_batch_data[field] = cast_to_bf16(val)
                            batch = type(batch)(**new_batch_data)
                    
                    actions = batch[ACTION]
                    rewards = batch["reward"]
                    observations = batch["state"]
                    next_observations = batch["next_state"]
                    done = batch["done"]
                    
                    # --- Step 1: Get Critic Value (Pass 1) ---
                    # Preprocess for Critic (Task + State, no Advantage)
                    forward_batch_critic = preprocess_batch_for_pi05(
                        policy=policy,
                        observations=observations,
                        next_observations=next_observations,
                        actions=actions,
                        rewards=rewards,
                        done=done,
                        task=cfg.policy.task,
                    )
                    
                    # Compute V(s) and V(s') using Updated Critic
                    with torch.no_grad():
                        critic_out = policy.forward(forward_batch_critic, model="critic_value")
                        current_v = critic_out["critic_values"]
                        
                        # Compute V(s') manually for advantage calculation
                        # We need next_critic_tokens
                        if "critic_tokens" in forward_batch_critic["next_state"]:
                            next_critic_tokens = forward_batch_critic["next_state"]["critic_tokens"]
                            next_critic_pad_mask = forward_batch_critic["next_state"]["critic_pad_mask"]
                        else:
                            # Fallback (should not happen with updated processor)
                            next_critic_tokens = forward_batch_critic["next_state"][OBS_LANGUAGE_TOKENS]
                            next_critic_pad_mask = forward_batch_critic["next_state"][OBS_LANGUAGE_ATTENTION_MASK]
                            
                        # Embed next tokens (using actor embeddings)
                        # We need to access the embedding layer. 
                        # This is getting complicated to do outside the policy.
                        # Ideally `policy` should have a helper.
                        
                        # Let's assume we can use `policy.critic_target` directly if we prepare inputs.
                        # Or we can add a helper to `PI05RLPolicy`.
                        # But I cannot modify `rl_pi05.py` in this tool call (I am editing offline_learner).
                        
                        # Workaround: Use `policy.forward(model="critic")` but ignore loss?
                        # `policy.forward(model="critic")` returns `target_values`!
                        # It computes `target_q = reward + gamma * next_v * (1-done)`.
                        # So we can use that!
                        
                        critic_out_full = policy.forward(forward_batch_critic, model="critic")
                        target_v = critic_out_full["target_values"]
                        current_v = critic_out_full["critic_values"]
                        
                        # Calculate Raw Advantage
                        # Advantage = Target - Value
                        raw_advantage = target_v - current_v
                        
                        # Flatten for processor
                        raw_advantage_flat = raw_advantage.view(-1)
                    
                    # --- Step 2: Re-tokenize (Pass 2) ---
                    # Inject raw advantage into the batch and re-tokenize for the Actor
                    
                    batch_for_proc = {k: v for k, v in observations.items()}
                    batch_for_proc["task"] = [cfg.policy.task] * actions.shape[0]
                    batch_for_proc[ACTION] = actions
                    batch_for_proc["advantage"] = raw_advantage_flat
                    with torch.no_grad():
                        # Run preprocessor again to generate tokens conditioned on advantage
                        processed_batch = accelerator.unwrap_model(policy).preprocessor(batch_for_proc)
                        
                    # Now `processed_batch` has `OBS_LANGUAGE_TOKENS` with the correct advantage prompt!
                    
                    # --- Step 3: Actor Update ---
                    # Construct forward batch for Actor
                    forward_batch_actor = {
                        ACTION: processed_batch[ACTION],
                        "reward": rewards,
                        "state": {},
                        "next_state": {}, # Not needed for actor update usually, but good to have
                        "done": done,
                        "observation_feature": None,
                        "next_observation_feature": None,
                        "task": batch_for_proc["task"],
                        "advantage": raw_advantage, # Pass raw advantage for logging/metrics
                        "next.done": done,
                    }
                    # Copy state keys
                    for key in observations.keys():
                        forward_batch_actor["state"][key] = observations[key]
                            
                    # Copy next_state keys from CRITIC batch to ensure tokens are present
                    if "next_state" in forward_batch_critic:
                        forward_batch_actor["next_state"] = forward_batch_critic["next_state"]
                    
                    # Add NEW tokens
                    forward_batch_actor["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
                    forward_batch_actor["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
                    
                    # Also add critic tokens (from Pass 1 or Pass 2? Pass 2 also generates them)
                    # It doesn't matter for actor update as actor doesn't use critic tokens (unless we compute advantage inside, which we don't need to anymore)
                    # But `policy.forward(model="actor")` might still calculate metrics.
                    # So let's add them.
                    if "critic_tokens" in processed_batch:
                        forward_batch_actor["critic_tokens"] = processed_batch["critic_tokens"]
                        forward_batch_actor["critic_pad_mask"] = processed_batch["critic_pad_mask"]
                        
                    # Add tokens to root for convenience
                    forward_batch_actor[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
                    forward_batch_actor[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
                    
                    # Forward Actor
                    actor_output = policy.forward(forward_batch_actor, model="actor")
                    loss_actor = actor_output["loss_actor"] / gradient_accumulation_steps
                    loss_actor = loss_actor.mean()
                    loss_actor.backward()
                    
                    # Accumulate metrics
                    accum_loss_actor += actor_output["loss_actor"].mean().item()
                    advantage_values_list.append(actor_output["advantage_values"])
                    critic_values_list.append(actor_output["critic_values"])
                    target_values_list.append(actor_output["target_values"])
                    reward_values_list.append(actor_output["rewards"])
                
                # Clip and step after accumulation
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=accelerator.unwrap_model(policy).actor.parameters(),
                    max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # Compute aggregated metrics
                all_advantage_values = torch.cat(advantage_values_list, dim=0)
                all_critic_values = torch.cat(critic_values_list, dim=0)
                all_target_values = torch.cat(target_values_list, dim=0)
                all_reward_values = torch.cat(reward_values_list, dim=0)

                if all_reward_values.mean().item() > 0:
                    print(f'Nonzero reward received: {all_reward_values.mean().item()} at optimization step {optimization_step}')
                
                # Add actor info
                training_infos["loss_actor"] = accum_loss_actor / gradient_accumulation_steps
                training_infos["actor_grad_norm"] = actor_grad_norm
                training_infos["advantage_mean"] = all_advantage_values.mean().item()
                training_infos["advantage_std"] = all_advantage_values.std().item() if all_advantage_values.numel() > 1 else 0.0
                training_infos["target_value_mean"] = all_target_values.mean().item()
                training_infos["reward_mean"] = all_reward_values.mean().item()
                training_infos["critic_value_mean_actor"] = all_critic_values.mean().item()
                training_infos["critic_value_std_actor"] = all_critic_values.std().item() if all_critic_values.numel() > 1 else 0.0
                
                # Store concatenated histograms
                training_infos["advantage_histogram"] = all_advantage_values
                training_infos["critic_histogram"] = all_critic_values
                
                # No Temperature Update for Pi05 RL (Entropy not used)

        # Push policy to actors if needed
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # Update target networks
        policy.update_target_networks()
        
        # ----------------------------------------

        # Log training metrics at specified intervals
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # Log training metrics
            if wandb_logger:
                print(f"Logging to WandB at step {optimization_step}")
                # Extract histograms if they exist
                advantage_hist = training_infos.pop("advantage_histogram", None)
                critic_hist = training_infos.pop("critic_histogram", None)
                
                # Log scalar metrics
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")
                
                # Log histograms separately using wandb.Histogram
                if advantage_hist is not None:
                    wandb_logger._wandb.log({
                        "train/advantage_histogram": wandb.Histogram(advantage_hist.detach().float().cpu().numpy()),
                        "Optimization step": optimization_step
                    })
                if critic_hist is not None:
                    wandb_logger._wandb.log({
                        "train/critic_value_histogram": wandb.Histogram(critic_hist.detach().float().cpu().numpy()),
                        "Optimization step": optimization_step
                    })

        # Calculate and log optimization frequency
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        
        # Log optimization frequency
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # Save checkpoint at specified intervals
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )

        # Save online buffer at specified intervals
        if online_buffer_save_freq is not None and optimization_step % online_buffer_save_freq == 0:
            logging.info(f"[LEARNER] Saving online buffer at step {optimization_step}")
            online_buffer_dir = os.path.join(cfg.output_dir, "online_buffer")
            
            # Remove existing buffer directory to overwrite
            if os.path.exists(online_buffer_dir) and os.path.isdir(online_buffer_dir):
                shutil.rmtree(online_buffer_dir)
            
            # Save buffer as dataset
            replay_buffer.to_lerobot_dataset(
                repo_id="online_buffer",
                fps=fps,
                root=online_buffer_dir,
                task_name=cfg.policy.task,
            )
            logging.info(f"[LEARNER] Online buffer saved to {online_buffer_dir}")

        if optimization_step >= online_steps:
            logging.info("[LEARNER] Reached maximum online steps. Stopping training.")
            break

def process_transitions_pi05(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
    policy: nn.Module = None,
    episode_counter: list = None,
    cfg: any = None,
):
    """Process all available transitions from the queue with action dimension handling."""
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        # Check if this is a logging episode (every N episodes)
        is_logging_episode = False
        if episode_counter is not None:
            episode_counter[0] += 1
            if episode_counter[0] % episode_logging_freq == 0:
                is_logging_episode = True
                logging.info(f"[LEARNER] Starting logging episode {episode_counter[0]}")
                log_dir = os.path.join(cfg.output_dir, "logging_episodes", f"episode_{episode_counter[0]:06d}")
                os.makedirs(log_dir, exist_ok=True)
                critic_values = []

        for i, transition in enumerate(transition_list):
            transition = move_transition_to_device(transition=transition, device=device)

            # If logging, save images and compute critic value
            if is_logging_episode and policy is not None:
                with torch.no_grad():
                    # Save images
                    obs = transition["state"]
                    for key, val in obs.items():
                        if "image" in key and "top" in key:
                            # val is [1, C, H, W] tensor, convert to [H, W, C] numpy
                            img_np = val.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                            # Normalize to [0, 255] if needed (assuming [0, 1])
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            
                            img_path = os.path.join(log_dir, f"step_{i:03d}_{key.split('.')[-1]}.png")
                            Image.fromarray(img_np).save(img_path)
                        elif "image" in key and "side" in key:
                            # val is [1, C, H, W] tensor, convert to [H, W, C] numpy
                            img_np = val.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                            # Normalize to [0, 255] if needed (assuming [0, 1])
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255).astype(np.uint8)
                            
                            img_path = os.path.join(log_dir, f"step_{i:03d}_{key.split('.')[-1]}.png")
                            Image.fromarray(img_np).save(img_path)

                    # Compute critic value
                    forward_batch = {
                        "state": {k: v for k, v in transition["state"].items()},
                        "action": transition[ACTION],
                    }

                    # Preprocessing
                    batch_for_proc = {k: v for k, v in forward_batch["state"].items()}
                    batch_for_proc["task"] = [cfg.policy.task]
                    batch_for_proc[ACTION] = forward_batch["action"]
                    
                    processed_batch = policy.preprocessor(batch_for_proc)
                    
                    # Update forward_batch with processed features
                    for key in processed_batch.keys():
                        if key.startswith("observation."):
                            forward_batch["state"][key] = processed_batch[key]
                    
                    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
                    forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
                    forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
                    
                    critic_output = policy.forward(forward_batch, model="critic_value")
                    val = critic_output["critic_value_mean"]
                    critic_values.append(val)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            # --- Fix for Action Dimension Mismatch ---
            # Check if buffer is initialized and has a mismatch
            if replay_buffer.initialized:
                buffer_action_dim = replay_buffer.actions.shape[-1]
                incoming_action = transition[ACTION]
                incoming_dim = incoming_action.shape[-1]
                
                if incoming_dim != buffer_action_dim:
                    # logging.warning(f"Action dim mismatch: Buffer {buffer_action_dim}, Incoming {incoming_dim}. Adjusting...")
                    if incoming_dim < buffer_action_dim:
                        # Pad with zeros
                        padding = torch.zeros(
                            (*incoming_action.shape[:-1], buffer_action_dim - incoming_dim),
                            dtype=incoming_action.dtype,
                            device=incoming_action.device
                        )
                        transition[ACTION] = torch.cat([incoming_action, padding], dim=-1)
                    else:
                        # Slice
                        transition[ACTION] = incoming_action[..., :buffer_action_dim]
            # -----------------------------------------

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                TeleopEvents.IS_INTERVENTION
            ):
                # Apply same fix for offline buffer if needed
                if offline_replay_buffer.initialized:
                    buffer_action_dim = offline_replay_buffer.actions.shape[-1]
                    incoming_action = transition[ACTION]
                    incoming_dim = incoming_action.shape[-1]
                    
                    if incoming_dim != buffer_action_dim:
                        if incoming_dim < buffer_action_dim:
                            # Pad with zeros
                            padding = torch.zeros(
                                (*incoming_action.shape[:-1], buffer_action_dim - incoming_dim),
                                dtype=incoming_action.dtype,
                                device=incoming_action.device
                            )
                            transition[ACTION] = torch.cat([incoming_action, padding], dim=-1)
                        else:
                            # Slice
                            transition[ACTION] = incoming_action[..., :buffer_action_dim]
                            
                offline_replay_buffer.add(**transition)

        # After processing the episode, if it was a logging episode, save the plot
        if is_logging_episode and policy is not None:
            # Save critic values to JSON
            with open(os.path.join(log_dir, "critic_values.json"), "w") as f:
                json.dump(critic_values, f)
            
            # Plot critic values
            plt.figure(figsize=(10, 5))
            plt.plot(critic_values)
            plt.title(f"Critic Values - Episode {episode_counter[0]}")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True)
            plt.savefig(os.path.join(log_dir, "critic_plot.png"))
            plt.close()
            
            # Generate video with critic overlay
            try:
                save_video_with_critic_overlay(log_dir, critic_values)
                logging.info(f"[LEARNER] Video generated for episode {episode_counter[0]}")
            except Exception as e:
                logging.error(f"[LEARNER] Failed to generate video: {e}")
                
            logging.info(f"[LEARNER] Finished logging episode {episode_counter[0]}")


def save_video_with_critic_overlay(log_dir, critic_values, fps=10):
    """
    Generate a side-by-side video of top and side views with a critic curve overlay.
    """
    import glob
    import re

    # Find all top and side images
    top_images = sorted(glob.glob(os.path.join(log_dir, "*_top.png")))
    side_images = sorted(glob.glob(os.path.join(log_dir, "*_side.png")))

    if not top_images or not side_images:
        raise ValueError("No images found for video generation")

    # Ensure we have the same number of images and critic values
    num_frames = min(len(top_images), len(side_images), len(critic_values))
    
    # Video settings
    # Each view is 224x224, resized to 448x448. Side-by-side is 896x448.
    frame_width = 896
    frame_height = 448
    video_path = os.path.join(log_dir, "episode_video.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    # Prepare critic curve data for plotting
    # Normalize critic values for plotting (0 to frame_height)
    # Fixed scale from 0 to 5 as requested
    critic_np = np.array(critic_values[:num_frames])
    c_min, c_max = 0.0, 5.0
    critic_norm = (critic_np - c_min) / (c_max - c_min)
    critic_norm = np.clip(critic_norm, 0, 1)
    
    # Map to pixel coordinates (inverted Y for image space)
    # Restrict to lower half (frame_height // 2 to frame_height)
    lower_half_height = frame_height // 2
    margin = 20
    plot_y = (lower_half_height - 2 * margin) * (1 - critic_norm) + (frame_height // 2 + margin)
    plot_x = np.linspace(0, frame_width, num_frames)

    for i in range(num_frames):
        # Load and resize images
        top_img = cv2.imread(top_images[i])
        side_img = cv2.imread(side_images[i])
        
        top_img = cv2.resize(top_img, (448, 448))
        side_img = cv2.resize(side_img, (448, 448))
        
        # Concatenate side-by-side
        frame = np.hstack((top_img, side_img))
        
        # Create an overlay for the curve
        overlay = frame.copy()
        
        # 1. Draw full curve with low alpha (faint white)
        points = np.vstack((plot_x, plot_y)).T.astype(np.int32)
        cv2.polylines(overlay, [points], isClosed=False, color=(200, 200, 200), thickness=1)
        
        # 2. Draw progressing curve with high alpha (bright white)
        prog_points = points[:i+1]
        if len(prog_points) > 1:
            cv2.polylines(overlay, [prog_points], isClosed=False, color=(255, 255, 255), thickness=3)
            
            # Draw a vertical dashed line at current position
            curr_x, curr_y = prog_points[-1]
            cv2.line(overlay, (curr_x, frame_height), (curr_x, curr_y), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Blend overlay with original frame
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        out.write(frame)

    out.release()

    # Cleanup: Remove individual images after video generation
    for img_path in top_images + side_images:
        try:
            os.remove(img_path)
        except Exception:
            pass


if __name__ == "__main__":
    train_cli()
