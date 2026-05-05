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
from threading import Thread
from torch.multiprocessing import Process
import torch.multiprocessing as mp

import grpc
import torch
from termcolor import colored
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import cv2

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
from lerobot.utils.process import ProcessSignalHandler
from lerobot.common.wandb_utils import WandBLogger
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
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
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
)
from lerobot.utils.random_utils import set_seed
from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import (
    format_big_number,
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
from lerobot.rl.utils import preprocess_batch_for_pi05, cast_to_bf16, save_video_with_critic_overlay
from lerobot.rl.pi05_train_utils import (
    pi05_update_step,
    hydrate_subtasks,
    load_additional_offline_datasets,
    make_pi05_full_processors_with_upgrade,
    _update_critic,
    _update_actor,
    log_pi05_training_metrics,
)

import wandb
import gc

                

def push_actor_policy_to_queue_pi05(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue (Pi05 Optimized)")

    # Only send trainable parameters to save bandwidth for large models
    trainable_state_dict = {}
    for name, param in policy.actor.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param

    state_dicts = {"policy": move_state_dict_to_device(trainable_state_dict, device="cpu")}
    
    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
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
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    # Set to false to avoid VRAM consumption spike
    torch.backends.cudnn.benchmark = False
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
        concurrency_entity = Thread
    else:
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
    async_prefetch = cfg.policy.async_prefetch
    episode_save_freq = cfg.episode_save_freq
    skip_critic = getattr(cfg, "skip_critic", False)

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


    assert isinstance(policy, nn.Module)

    policy.train()

    # Enable gradient checkpointing if configured
    if cfg.policy.gradient_checkpointing:
        if hasattr(policy, "model") and hasattr(policy.model, "gradient_checkpointing_enable"):
            policy.model.gradient_checkpointing_enable()
        elif hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()
        else:
            logging.warning("Gradient checkpointing requested but not available on policy model")
    push_actor_policy_to_queue_pi05(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()


    # Freezing some parameters
    _VISION_TOWER_DEPTH = 27   # SigLIP-400M
    _CRITIC_VISION_TOWER_DEPTH = 13  # Critic truncates SigLIP to 13 layers (rl_pi05.py)
    _LANGUAGE_MODEL_DEPTH = 18  # Gemma 2B
    tp = cfg.policy.trainable_params
    critic_depth = cfg.policy.critic_llm_depth
    lm_layers = list(range(tp.language_from_layer, _LANGUAGE_MODEL_DEPTH)) if tp.language_from_layer is not None else []
    vt_layers  = list(range(tp.vision_encoder_from_layer.vision_tower, _VISION_TOWER_DEPTH)) if tp.vision_encoder_from_layer.vision_tower is not None else []
    cr_layers  = list(range(tp.critic_language_from_layer, critic_depth)) if tp.critic_language_from_layer is not None else []
    cr_vt_layers = list(range(tp.critic_vision_encoder_from_layer.vision_tower, _CRITIC_VISION_TOWER_DEPTH)) if tp.critic_vision_encoder_from_layer.vision_tower is not None else []

    for name, param in policy.named_parameters():
        param.requires_grad = (
            # Action expert — always on
            "action_in_proj" in name or
            "action_out_proj" in name or
            "time_mlp_in" in name or
            "time_mlp_out" in name or
            "gemma_expert" in name or
            # Actor vision encoder (scoped to paligemma to avoid matching critic's vision tower)
            (tp.vision_encoder_from_layer.multi_modal_projector and "paligemma" in name and "multi_modal_project" in name) or
            ("paligemma" in name and "vision_tower" in name and any(f".{i}." in name for i in vt_layers)) or
            # Language model
            ("language_model" in name and any(f".{i}." in name for i in lm_layers)) or
            ("language_model.norm" in name and bool(lm_layers)) or
            # Critic — norm/value_head/queries always on
            "critic.norm" in name or
            "critic.value_head" in name or
            "critic.value_queries" in name or
            ("critic.layers" in name and any(f".{i}." in name for i in cr_layers)) or
            # Critic vision encoder (scoped to critic. prefix to avoid critic_target.*)
            (tp.critic_vision_encoder_from_layer.multi_modal_projector and name.startswith("critic.") and "multi_modal_project" in name) or
            (name.startswith("critic.vision_tower") and any(f".{i}." in name for i in cr_vt_layers))
        )
    
    # Share underlying memory for frozen critic layers to save VRAM
    if hasattr(policy, "critic") and hasattr(policy, "critic_target"):
        for param, target_param in zip(policy.critic.parameters(), policy.critic_target.parameters()):
            if not param.requires_grad:
                target_param.data = param.data

    # Log trainable parameters
    trainable_params = [n for n, p in policy.named_parameters() if p.requires_grad]
    logging.info(f"MINIMAL MODE: Trainable parameters: {len(trainable_params)}")
    logging.info(f"Trainable parameter names: {trainable_params}")  # Show what's being trainedebug

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    last_save_episode = 0
    replay_buffer.reward_normalization_constant = cfg.policy.reward_normalization_constant
    replay_buffer.terminal_failure_reward = cfg.policy.terminal_failure_reward
    batch_size = cfg.batch_size
    offline_replay_buffer = None
    offline_dataset = None

    if cfg.dataset is not None:
        # Inline initialize_offline_replay_buffer to allow modifying dataset before buffer creation
        if not cfg.resume:
            logging.info("make_dataset offline buffer")
            offline_dataset = make_dataset(cfg)
        else:
            logging.info("load offline dataset")
            dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")

            episodes = cfg.dataset.episodes
            if episodes is None and cfg.dataset.max_episodes is not None:
                episodes = list(range(cfg.dataset.max_episodes))

            offline_dataset = LeRobotDataset(
                repo_id=cfg.dataset.repo_id,
                root=dataset_offline_path,
                episodes=episodes,
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
            inject_complementary_info={"is_golden": cfg.treat_main_dataset_as_golden},
            cache_dir=cfg.buffer_cache_dir,
        )
        offline_replay_buffer.dataset = offline_dataset

        # Load additional offline datasets
        load_additional_offline_datasets(
            cfg=cfg,
            offline_dataset=offline_dataset,
            offline_replay_buffer=offline_replay_buffer,
            storage_device=storage_device,
            is_main_process=True
        )
        batch_size: int = batch_size // 2  # We will sample from both replay buffer


    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id
    

    # Create preprocessor and postprocessor for the policy
    # Use the shared utility for runtime upgrade to support standard Pi05 checkpoints
    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(cfg, dataset=offline_dataset, is_main_process=True)

    # Store preprocessors on the policy for actor to access
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

    critic_warmup_steps = cfg.policy.critic_warmup_steps
    policy_update_freq = 1


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


        time_for_one_optimization_step = time.time()

        if not skip_critic:
            for _ in range(utd_ratio - 1):
                _update_critic(
                    policy=policy,
                    optimizers=optimizers,
                    online_iterator=online_iterator,
                    offline_iterator=offline_iterator,
                    device=device,
                    cfg=cfg,
                    dataset_repo_id=None,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    clip_grad_norm_value=clip_grad_norm_value,
                    cast_to_bf16_fn=cast_to_bf16 if cfg.policy.dtype == "bfloat16" else None,
                    use_amp=False,
                    scaler=None
                )

                # 2. Update Target Networks
                policy.update_target_networks()

                # ----------------------------

        # Sample for the last update in the UTD ratio

        if skip_critic:
            # Actor-only: skip critic forward/backward, advantage uses golden bypass
            training_infos = _update_actor(
                policy=policy,
                optimizers=optimizers,
                online_iterator=online_iterator,
                offline_iterator=offline_iterator,
                device=device,
                cfg=cfg,
                dataset_repo_id=dataset_repo_id,
                gradient_accumulation_steps=gradient_accumulation_steps,
                policy_update_freq=policy_update_freq,
                clip_grad_norm_value=clip_grad_norm_value,
                dataset=offline_dataset,
                cast_to_bf16_fn=cast_to_bf16 if cfg.policy.dtype == "bfloat16" else None,
                use_amp=False,
                scaler=None,
                preprocessor=preprocessor,
            )
        else:
            # Call shared update step function
            # This handles: data fetching, preprocessing, critic update, (optional) actor update, metrics
            training_infos = pi05_update_step(
                policy=policy,
                optimizers=optimizers,
                online_iterator=online_iterator,
                offline_iterator=offline_iterator,
                batch_size=batch_size,
                device=device,
                cfg=cfg,
                optimization_step=optimization_step,
                dataset_repo_id=dataset_repo_id,
                gradient_accumulation_steps=gradient_accumulation_steps,
                critic_warmup_steps=critic_warmup_steps,
                policy_update_freq=policy_update_freq,
                clip_grad_norm_value=clip_grad_norm_value,
                dataset=offline_dataset, # For subtask metadata
                cast_to_bf16_fn=cast_to_bf16 if cfg.policy.dtype == "bfloat16" else None,
                use_amp=False, # Learner typically uses raw backward or fp32
                scaler=None,
                preprocessor=preprocessor,
            )

        # ----------------------------------------

        # Log training metrics at specified intervals
        if optimization_step % log_freq == 0:
            
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            if wandb_logger:
                print(f"Logging to WandB at step {optimization_step}")

            log_pi05_training_metrics(
                training_infos=training_infos,
                optimization_step=optimization_step,
                wandb_logger=wandb_logger,
                policy=policy,
                is_main_process=True
            )

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

        # Save online buffer at specified intervals (based on episode count)
        current_episode = add_actor_information_and_train.episode_counter[0]
        if current_episode > 0 and current_episode >= last_save_episode + episode_save_freq:
            logging.info(f"[LEARNER] Saving online buffer at episode {current_episode}, step {optimization_step}, buffer size {len(replay_buffer)}")
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
            last_save_episode = current_episode

        if optimization_step >= online_steps:
            logging.info("[LEARNER] Reached maximum online steps. Stopping training.")
            break

        # Push policy to actors if needed
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue_pi05(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

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
            if episode_counter[0] % cfg.episode_logging_freq == 0:
                is_logging_episode = True
                logging.info(f"[LEARNER] Starting logging episode {episode_counter[0]}")
                log_dir = os.path.join(cfg.output_dir, "logging_episodes", f"episode_{episode_counter[0]:06d}")


                os.makedirs(log_dir, exist_ok=True)
                critic_values = []

        for i, transition in enumerate(transition_list):
            # Optimization: Keep transition on CPU to save GPU memory and bandwidth.
            # Only move to device if needed (e.g. for logging).
            # transition = move_transition_to_device(transition=transition, device=device)

            # If logging, save images and compute critic value
            if is_logging_episode and policy is not None:
                with torch.no_grad():
                    # Save images
                    obs = transition["state"]
                    video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])
                    
                    for key, val in obs.items():
                        if "image" in key:
                            cam_name = key.split('.')[-1]
                            if cam_name in video_logging_cameras:
                                # val is [1, C, H, W] tensor, convert to [H, W, C] numpy
                                img_np = val.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                                # Normalize to [0, 255] if needed (assuming [0, 1])
                                if img_np.max() <= 1.0:
                                    img_np = (img_np * 255).astype(np.uint8)
                                
                                img_path = os.path.join(log_dir, f"step_{i:06d}_{cam_name}.png")
                                Image.fromarray(img_np).save(img_path)


                    # Prepare inputs for preprocessor (move to device and add batch dim)
                    state_dev = {k: v.to(device) for k, v in transition["state"].items()}
                    next_state_dev = {k: v.to(device) for k, v in transition["next_state"].items()}
                    
                    # Action needs to be on device and have batch dim [1, D]
                    # We unsqueeze(0) to add the batch dimension
                    action_dev = transition[ACTION].to(device).unsqueeze(0)
                    
                    # Reward and Done need to be tensors on device with batch dim [1]
                    # .view(1) ensures they are 1D tensors of size 1
                    reward_dev = torch.tensor(transition["reward"], device=device).view(1)
                    done_dev = torch.tensor(transition["done"], device=device).view(1)
                    
                    # Use shared utility to preprocess batch (handles subtask defaults correctly)
                    forward_batch = preprocess_batch_for_pi05(
                        policy=policy,
                        observations=state_dev,
                        next_observations=next_state_dev,
                        actions=action_dev,
                        rewards=reward_dev,
                        done=done_dev,
                        task=cfg.policy.task,
                    )
                    
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
                save_video_with_critic_overlay(log_dir, critic_values, camera_names=video_logging_cameras)
                logging.info(f"[LEARNER] Video generated for episode {episode_counter[0]}")
            except Exception as e:
                logging.error(f"[LEARNER] Failed to generate video: {e}")
                
            logging.info(f"[LEARNER] Finished logging episode {episode_counter[0]}")


if __name__ == "__main__":
    train_cli()
