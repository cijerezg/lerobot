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
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

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
            "gemma_expert" in name and 
            ("layers.17" in name) or
            "critic" in name or
            "log_alpha" in name
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

        # Process all available transitions to the replay buffer, send by the actor server
        process_transitions_pi05(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # Process all available interaction messages sent by the actor server
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
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
            }

            # --- Preprocessing for Pi05 (Tokenization and Normalization) ---
            # Preprocess current observations
            batch_for_proc = {k: v for k, v in observations.items()}
            current_batch_size = actions.shape[0]
            batch_for_proc["task"] = [cfg.policy.task] * current_batch_size
            batch_for_proc[ACTION] = actions
            
            with torch.no_grad():
                processed_batch = policy.preprocessor(batch_for_proc)
            
            # Preprocess next observations (for critic)
            next_batch_for_proc = {k: v for k, v in next_observations.items()}
            next_batch_for_proc["task"] = [cfg.policy.task] * current_batch_size
            next_batch_for_proc[ACTION] = actions  # Not used, but required by preprocessor
            
            with torch.no_grad():
                processed_next_batch = policy.preprocessor(next_batch_for_proc)
            
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
            
            # Update forward_batch with normalized observations for critic
            # Replace unnormalized images/state with normalized versions
            for key in processed_batch.keys():
                if key.startswith("observation."):
                    forward_batch["state"][key] = processed_batch[key]
            
            for key in processed_next_batch.keys():
                if key.startswith("observation."):
                    forward_batch["next_state"][key] = processed_next_batch[key]
            
            # Add tokens and normalized actions
            forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
            forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
            forward_batch[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
            forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
            forward_batch[ACTION] = processed_batch[ACTION]

            # --- Pi05 RL Update Logic ---
            
            # 1. Critic Update (Value Function)
            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()
            
            # 2. Update Target Networks
            policy.update_target_networks()
            
            # ----------------------------

        # Sample for the last update in the UTD ratio
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
        }

        # --- Preprocessing for Pi05 (Tokenization and Normalization) ---
        # Preprocess current observations
        batch_for_proc = {k: v for k, v in observations.items()}
        current_batch_size = actions.shape[0]
        batch_for_proc["task"] = [cfg.policy.task] * current_batch_size
        batch_for_proc[ACTION] = actions
        
        with torch.no_grad():
            processed_batch = policy.preprocessor(batch_for_proc)
        
        # Preprocess next observations (for critic)
        next_batch_for_proc = {k: v for k, v in next_observations.items()}
        next_batch_for_proc["task"] = [cfg.policy.task] * current_batch_size
        next_batch_for_proc[ACTION] = actions  # Not used, but required by preprocessor
        
        with torch.no_grad():
            processed_next_batch = policy.preprocessor(next_batch_for_proc)
        
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        
        # Update forward_batch with normalized observations for critic
        # Replace unnormalized images/state with normalized versions
        for key in processed_batch.keys():
            if key.startswith("observation."):
                forward_batch["state"][key] = processed_batch[key]
        
        for key in processed_next_batch.keys():
            if key.startswith("observation."):
                forward_batch["next_state"][key] = processed_next_batch[key]
        
        # Add tokens and normalized actions
        forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
        forward_batch[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
        forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
        forward_batch[ACTION] = processed_batch[ACTION]

        # --- Pi05 RL Update Logic (Last Step) ---

        
        # 1. Critic Update
        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        # Initialize training info dictionary
        training_infos = {
            "loss_critic": critic_output["loss_critic"].item(),
            "critic_grad_norm": critic_grad_norm,
            "td_error_mean": critic_output["td_error_mean"],
            "td_error_std": critic_output["td_error_std"],
            "critic_value_mean": critic_output["critic_value_mean"],
            "critic_value_std": critic_output["critic_value_std"],
            "target_value_mean_critic": critic_output["target_value_mean"],
            "target_value_std": critic_output["target_value_std"],
        }

        # 2. Actor Update (Flow Matching with Advantage)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # Add actor info to training info
                training_infos["loss_actor"] = actor_output["loss_actor"].item()
                training_infos["actor_grad_norm"] = actor_grad_norm
                
                # Add advantage and value metrics
                training_infos["advantage_mean"] = actor_output["advantage_mean"]
                training_infos["advantage_std"] = actor_output["advantage_std"]
                training_infos["target_value_mean"] = actor_output["target_value_mean"]
                training_infos["reward_mean"] = actor_output["reward_mean"]
                training_infos["critic_value_mean_actor"] = actor_output["critic_value_mean"]
                training_infos["critic_value_std_actor"] = actor_output["critic_value_std"]
                
                # Store histograms for wandb (will be logged separately)
                training_infos["advantage_histogram"] = actor_output["advantage_values"]
                training_infos["critic_histogram"] = actor_output["critic_values"]
                
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
                        "train/advantage_histogram": wandb.Histogram(advantage_hist.float().numpy()),
                        "Optimization step": optimization_step
                    })
                if critic_hist is not None:
                    wandb_logger._wandb.log({
                        "train/critic_value_histogram": wandb.Histogram(critic_hist.float().numpy()),
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
):
    """Process all available transitions from the queue with action dimension handling."""
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

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


if __name__ == "__main__":
    train_cli()
