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
Offline pretraining script for Pi05 RL policy and critic.

This script trains the policy and critic using only the offline dataset,
without spawning actors or handling online data collection. It's designed
to create a good initialization before online RL training begins.

Usage:
    python offline_learner_pi05.py config-hiserl.json --offline-steps 10000
"""

import logging
import os
import time
from pathlib import Path
from pprint import pformat
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch import nn

import lerobot.rl.rl_pi05  # Register PI05RLConfig

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

import wandb



@dataclass
class OfflineTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    offline_output_dir: str | None = None
    offline_save_freq: int | None = None


@parser.wrap()
def offline_train_cli(cfg: OfflineTrainRLServerPipelineConfig):
    """CLI entry point for offline training."""
    # Create Accelerator for multi-GPU support
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])
    
    offline_train(cfg, job_name=cfg.job_name, accelerator=accelerator)
    
    # Properly clean up
    accelerator.wait_for_everyone()
    accelerator.end_training()
    
    if accelerator.is_main_process:
        logging.info("[OFFLINE LEARNER] Training finished")


def offline_train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None, accelerator: Accelerator | None = None):
    """
    Main offline training function.
    
    Args:
        cfg: Training configuration
        job_name: Optional job name for logging
        accelerator: Optional Accelerator instance for multi-GPU training
    """
    cfg.validate()
    
    if job_name is None:
        job_name = cfg.job_name
    
    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")
    
    # Use output_dir_offline or offline_output_dir if provided, otherwise fall back to output_dir
    output_dir = getattr(cfg, 'output_dir_offline', None) or getattr(cfg, 'offline_output_dir', None) or cfg.output_dir
    if output_dir is None:
        raise ValueError("Either output_dir_offline or output_dir must be specified in config")
    
    # Override cfg.output_dir for the rest of the script
    cfg.output_dir = output_dir
    
    # Create Accelerator if not provided
    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])
    
    # Check if main process
    is_main_process = accelerator.is_main_process
    
    # Create logs directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"offline_learner_{job_name}.log")
    
    # Initialize logging with accelerator
    init_logging(log_file=log_file, display_pid=False, accelerator=accelerator)
    
    if is_main_process:
        logging.info(f"Offline learner logging initialized, writing to {log_file}")
        logging.info(f"Output directory: {output_dir}")
        logging.info(pformat(cfg.to_dict()))
    
    # Setup WandB logging if enabled (only on main process)
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    
    # Check that offline dataset is provided
    if cfg.dataset is None:
        raise ValueError("Offline training requires a dataset. Please specify cfg.dataset.")
    
    if cfg.seed is not None:
        set_seed(seed=cfg.seed, accelerator=accelerator)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    shutdown_event = ProcessSignalHandler(True, display_pid=False).shutdown_event
    
    # Start training
    run_offline_training(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        accelerator=accelerator,
    )


def run_offline_training(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
    accelerator: Accelerator,
):
    """
    Run the offline training loop.
    
    This simplified loop only samples from the offline buffer and trains
    the policy and critic without any online data collection.
    
    Args:
        cfg: Training configuration
        wandb_logger: Optional WandB logger
        shutdown_event: Shutdown event signal
        accelerator: Accelerator instance for multi-GPU training
    """
    is_main_process = accelerator.is_main_process
    
    # Extract configuration variables
    device = accelerator.device  # Use accelerator's device
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = getattr(cfg, "offline_save_freq", None) or cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    saving_checkpoint = cfg.save_checkpoint
    async_prefetch = cfg.policy.async_prefetch
    
    # Get offline training steps from config or use default
    offline_steps = getattr(cfg.policy, "offline_steps", 10000)
    
    if is_main_process:
        logging.info(f"Offline training will run for {offline_steps} optimization steps")
        logging.info("Initializing policy")
    
    # Override device to ensure policy is created on accelerator's device
    original_device = cfg.policy.device
    cfg.policy.device = device.type
    
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    
    # Restore original device config
    cfg.policy.device = original_device
    
    assert isinstance(policy, nn.Module)
    policy.train()
    
    # Wait for all processes before continuing
    accelerator.wait_for_everyone()
    
    # Freeze parameters based on config (same as online learner)
    if is_main_process:
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
    if is_main_process:
        trainable_params = [n for n, p in policy.named_parameters() if p.requires_grad]
        logging.info(f"MINIMAL MODE: Trainable parameters: {len(trainable_params)}")
        logging.info(f"Trainable parameter names: {trainable_params}")
    
    # Create optimizers
    optimizers = make_optimizers(cfg=cfg, policy=policy)
    
    # Wait before logging training info
    accelerator.wait_for_everyone()
    
    # Log training info
    if is_main_process:
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        log_training_info(cfg=cfg, policy=policy, offline_steps=offline_steps)
    
    # Create preprocessor and postprocessor
    if is_main_process:
        logging.info("Creating preprocessors and postprocessors")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pi05_checkpoint,
    )
    # Store preprocessors on the policy
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor
    
    # Initialize offline replay buffer (only on main process first)
    if is_main_process:
        logging.info("Initializing offline replay buffer")
        offline_dataset = make_dataset(cfg)
    
    accelerator.wait_for_everyone()
    
    # Now all other processes can safely load
    if not is_main_process:
        offline_dataset = make_dataset(cfg)
    
    # Disable delta_timestamps to avoid chunking (we want single frames for the buffer)
    offline_dataset.delta_timestamps = None
    offline_dataset.delta_indices = None
    
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    offline_replay_buffer.dataset = offline_dataset

    if is_main_process:
        logging.info(f"Offline buffer initialized with {len(offline_replay_buffer)} samples")
    
    # Get batch size (no splitting needed since offline only)
    batch_size = cfg.batch_size
    
    # Initialize iterator
    offline_iterator = offline_replay_buffer.get_iterator(
        batch_size=batch_size,
        async_prefetch=async_prefetch,
        queue_size=2,
        action_chunk_size=cfg.policy.n_action_steps,
    )
    
    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizers["actor"], optimizers["critic"] = accelerator.prepare(
        policy, optimizers["actor"], optimizers["critic"]
    )
    
    # Helper function for bfloat16 casting
    if cfg.policy.dtype == "bfloat16":
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
    else:
        cast_to_bf16 = None
    
    if is_main_process:
        logging.info("Starting offline training loop")
    optimization_step = 0
    
    # Main training loop
    while optimization_step < offline_steps:
        # Exit if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[OFFLINE LEARNER] Shutdown signal received. Exiting...")
            break
        
        time_for_one_optimization_step = time.time()
        
        # UTD ratio - 1 updates (critic only)
        for _ in range(utd_ratio - 1):
            batch = next(offline_iterator)
            
            # Slice offline actions to match expected dimension (6 dims)
            batch[ACTION] = batch[ACTION][..., :6]
            
            # Move batch to device
            batch = move_transition_to_device(batch, device)
            
            # Cast to bfloat16 if needed
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
            
            # Preprocess observations
            forward_batch = preprocess_batch_for_pi05(
                policy=policy,
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                done=done,
                task=cfg.policy.task,
            )
            
            # Critic update
            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=accelerator.unwrap_model(policy).critic_ensemble.parameters(), 
                max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()
            
            # Update target networks
            if hasattr(policy, "module"):
                policy.module.update_target_networks()
            else:
                policy.update_target_networks()
        
        # Last update in UTD ratio (critic + actor)
        batch = next(offline_iterator)
        batch[ACTION] = batch[ACTION][..., :6]
        batch = move_transition_to_device(batch, device)
        
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
        
        # Preprocess observations
        forward_batch = preprocess_batch_for_pi05(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            done=done,
            task=cfg.policy.task,
        )
        
        # Critic update
        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=accelerator.unwrap_model(policy).critic_ensemble.parameters(),
            max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()
        
        # Initialize training info
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
        
        # Actor update (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=accelerator.unwrap_model(policy).actor.parameters(),
                    max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()
                
                # Add actor info
                training_infos["loss_actor"] = actor_output["loss_actor"].item()
                training_infos["actor_grad_norm"] = actor_grad_norm
                training_infos["advantage_mean"] = actor_output["advantage_mean"]
                training_infos["advantage_std"] = actor_output["advantage_std"]
                training_infos["target_value_mean"] = actor_output["target_value_mean"]
                training_infos["reward_mean"] = actor_output["reward_mean"]
                training_infos["critic_value_mean_actor"] = actor_output["critic_value_mean"]
                training_infos["critic_value_std_actor"] = actor_output["critic_value_std"]
                
                # Store histograms
                training_infos["advantage_histogram"] = actor_output["advantage_values"]
                training_infos["critic_histogram"] = actor_output["critic_values"]
        
        # Update target networks
        # Update target networks
        if hasattr(policy, "module"):
            policy.module.update_target_networks()
        else:
            policy.update_target_networks()
        
        # Log training metrics
        if optimization_step % log_freq == 0:
            training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step
            
            if wandb_logger:
                # Extract histograms
                advantage_hist = training_infos.pop("advantage_histogram", None)
                critic_hist = training_infos.pop("critic_histogram", None)
                
                # Log scalar metrics
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")
                
                # Log histograms
                if advantage_hist is not None:
                    wandb_logger._wandb.log({
                        "train/advantage_histogram": wandb.Histogram(advantage_hist.float().cpu().numpy()),
                        "Optimization step": optimization_step
                    })
                if critic_hist is not None:
                    wandb_logger._wandb.log({
                        "train/critic_value_histogram": wandb.Histogram(critic_hist.float().cpu().numpy()),
                        "Optimization step": optimization_step
                    })
        
        # Calculate optimization frequency
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)
        
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
        if optimization_step % log_freq == 0 and is_main_process:
            logging.info(f"[OFFLINE LEARNER] Optimization step: {optimization_step}/{offline_steps}")
        
        # Save checkpoint (only on main process)
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == offline_steps):
            if is_main_process:
                save_offline_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    offline_steps=offline_steps,
                    policy=accelerator.unwrap_model(policy),  # Unwrap before saving
                    optimizers=optimizers,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    fps=fps,
                    accelerator=accelerator,
                )
            
            # Wait for all processes after checkpoint saving
            accelerator.wait_for_everyone()
    
    if is_main_process:
        logging.info(f"[OFFLINE LEARNER] Training completed after {optimization_step} steps")


def preprocess_batch_for_pi05(
    policy: nn.Module,
    observations: dict,
    next_observations: dict,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    task: str,
) -> dict:
    """
    Preprocess batch for Pi05 policy (tokenization and normalization).
    
    Args:
        policy: Policy with preprocessor attached
        observations: Current observations
        next_observations: Next observations
        actions: Actions
        rewards: Rewards
        done: Done flags
        task: Task description
    
    Returns:
        Forward batch ready for policy.forward()
    """
    current_batch_size = actions.shape[0]
    
    # Preprocess current observations
    batch_for_proc = {k: v for k, v in observations.items()}
    batch_for_proc["task"] = [task] * current_batch_size
    batch_for_proc[ACTION] = actions
    
    with torch.no_grad():
        # Access preprocessor - handle potential accelerate wrapping
        preprocessor = getattr(policy, 'preprocessor', None) or getattr(policy.module, 'preprocessor', None)
        processed_batch = preprocessor(batch_for_proc)
    
    # Preprocess next observations
    next_batch_for_proc = {k: v for k, v in next_observations.items()}
    next_batch_for_proc["task"] = [task] * current_batch_size
    next_batch_for_proc[ACTION] = actions  # Required by preprocessor
    
    with torch.no_grad():
        processed_next_batch = preprocessor(next_batch_for_proc)
    
    # Build forward batch
    forward_batch = {
        ACTION: processed_batch[ACTION],
        "reward": rewards,
        "state": {},
        "next_state": {},
        "done": done,
        "observation_feature": None,
        "next_observation_feature": None,
    }
    
    # Copy preprocessed observations
    for key in processed_batch.keys():
        if key.startswith("observation."):
            forward_batch["state"][key] = processed_batch[key]
    
    for key in processed_next_batch.keys():
        if key.startswith("observation."):
            forward_batch["next_state"][key] = processed_next_batch[key]
    
    # Add tokens
    forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
    forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    forward_batch[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
    forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    
    return forward_batch


def save_offline_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    offline_steps: int,
    policy: nn.Module,
    optimizers: dict,
    preprocessor,
    postprocessor,
    fps: int,
    accelerator: Accelerator,
):
    """
    Save offline training checkpoint.
    
    CRITICAL: This function saves the preprocessor and postprocessor along with
    the policy to ensure the checkpoint can be loaded later.
    
    Note: This should only be called on the main process.
    """
    logging.info(f"[OFFLINE LEARNER] Saving checkpoint at step {optimization_step}")
    
    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, offline_steps, optimization_step)
    
    # Save checkpoint with preprocessor and postprocessor
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
        preprocessor=preprocessor,  # CRITICAL: Save preprocessor
        postprocessor=postprocessor,  # CRITICAL: Save postprocessor
    )
    
    # Save training state
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": 0}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))
    
    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)
    
    logging.info(f"[OFFLINE LEARNER] Checkpoint saved to {checkpoint_dir}")


def make_optimizers(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """Create optimizers for actor and critic."""
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in (policy.module.actor.named_parameters() if hasattr(policy, "module") else policy.actor.named_parameters())
            if p.requires_grad
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        params=(policy.module.critic_ensemble.parameters() if hasattr(policy, "module") else policy.critic_ensemble.parameters()), 
        lr=cfg.policy.critic_lr
    )
    
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
    }
    return optimizers


def log_training_info(
    cfg: TrainRLServerPipelineConfig, 
    policy: nn.Module,
    offline_steps: int,
):
    """Log information about the training process."""
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    
    logging.info(colored("="*80, "yellow", attrs=["bold"]))
    logging.info(colored("OFFLINE PRETRAINING", "yellow", attrs=["bold"]))
    logging.info(colored("="*80, "yellow", attrs=["bold"]))
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"Task: {cfg.env.task}")
    logging.info(f"Offline steps: {offline_steps}")
    logging.info(f"Dataset: {cfg.dataset.repo_id}")
    logging.info(f"Learnable params: {num_learnable_params} ({format_big_number(num_learnable_params)})")
    logging.info(f"Total params: {num_total_params} ({format_big_number(num_total_params)})")
    logging.info(colored("="*80, "yellow", attrs=["bold"]))


if __name__ == "__main__":
    offline_train_cli()
