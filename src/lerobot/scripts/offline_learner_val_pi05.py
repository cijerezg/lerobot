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
    python offline_learner_val_pi05.py config-hiserl.json --offline-steps 10000

Adds periodic validation (probe_actions, probe_representations, probe_attention)
on top of offline_learner_pi05.py. Config fields: val_dataset_path, val_split,
val_freq. See offline_val_pi05.py for full documentation.
"""

import gc
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from pathlib import Path
from pprint import pformat
from dataclasses import dataclass

import numpy as np

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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
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
from lerobot.rl.utils import preprocess_batch_for_pi05, cast_to_bf16
from lerobot.rl.pi05_train_utils import (
    pi05_update_step,
    hydrate_subtasks,
    make_pi05_full_processors_with_upgrade,
    load_additional_offline_datasets,
    _update_critic,
    log_pi05_training_metrics,
)

import wandb

from lerobot.scripts.offline_val_pi05 import (
    load_val_dataset,
    init_action_manifold,
    run_validation,
)


@dataclass
class OfflineTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    offline_output_dir: str | None = None
    offline_save_freq: int | None = None
    # Validation
    val_dataset_path: str | None = None   # path to a separate val LeRobotDataset
    val_split: float = 0.0                # fraction of main-dataset episodes to hold out
    val_freq: int = 1000                  # optimization steps between validation runs
    val_on_start: bool = False            # run validation before training starts (step 0 baseline)


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
    if cfg.wandb.enable and is_main_process:
        if cfg.wandb.offline_project:
            cfg.wandb.project = cfg.wandb.offline_project

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
        set_seed(seed=cfg.seed + accelerator.process_index, accelerator=accelerator)
    
    # Set to false to avoid benchmarking which causes a memory spike.
    torch.backends.cudnn.benchmark = False

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
    gradient_accumulation_steps = getattr(cfg.policy, "gradient_accumulation_steps", 1)
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = getattr(cfg, "offline_save_freq", None) or cfg.save_freq
    val_freq = getattr(cfg, "val_freq", 1000)
    policy_update_freq = cfg.policy.policy_update_freq
    saving_checkpoint = cfg.save_checkpoint
    async_prefetch = cfg.policy.async_prefetch
    
    offline_steps = cfg.policy.offline_steps
    
    # Critic warmup steps
    critic_warmup_steps = 0
    
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

    # Enable gradient checkpointing if configured
    if cfg.policy.gradient_checkpointing:
        if hasattr(policy, "model") and hasattr(policy.model, "gradient_checkpointing_enable"):
            policy.model.gradient_checkpointing_enable()
        elif hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()
        else:
            logging.warning("Gradient checkpointing requested but not available on policy model")
    
    # Wait for all processes before continuing
    accelerator.wait_for_everyone()
    
    
    # Freezing some parameters
    _VISION_TOWER_DEPTH = 27   # SigLIP-400M
    _LANGUAGE_MODEL_DEPTH = 18  # Gemma 2B
    tp = cfg.policy.trainable_params
    critic_depth = cfg.policy.critic_llm_depth
    lm_layers = list(range(tp.language_from_layer, _LANGUAGE_MODEL_DEPTH)) if tp.language_from_layer is not None else []
    vt_layers  = list(range(tp.vision_encoder_from_layer.vision_tower, _VISION_TOWER_DEPTH)) if tp.vision_encoder_from_layer.vision_tower is not None else []
    cr_layers  = list(range(tp.critic_language_from_layer, critic_depth)) if tp.critic_language_from_layer is not None else []

    for name, param in policy.named_parameters():
        param.requires_grad = (
            # Action expert — always on
            "action_in_proj" in name or
            "action_out_proj" in name or
            "time_mlp_in" in name or
            "time_mlp_out" in name or
            "gemma_expert" in name or
            # Vision encoder
            (tp.vision_encoder_from_layer.multi_modal_projector and "multi_modal_project" in name) or
            ("vision_tower" in name and any(f".{i}." in name for i in vt_layers)) or
            # Language model
            ("language_model" in name and any(f".{i}." in name for i in lm_layers)) or
            ("language_model.norm" in name and bool(lm_layers)) or
            # Critic — norm/value_head/queries always on
            "critic.norm" in name or
            "critic.value_head" in name or
            "critic.value_queries" in name or
            ("critic.layers" in name and any(f".{i}." in name for i in cr_layers))
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

    # Create preprocessor and postprocessor
    if is_main_process:
        logging.info("Creating preprocessors and postprocessors")
    
    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(cfg, dataset=offline_dataset, is_main_process=is_main_process)
    
    # Store preprocessors on the policy
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor
    
    buffer_cache_dir = getattr(cfg, "buffer_cache_dir", None)
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
        reward_normalization_constant=cfg.policy.reward_normalization_constant,
        terminal_failure_reward=cfg.policy.terminal_failure_reward,
        inject_complementary_info={"is_golden": True},
        cache_dir=buffer_cache_dir,
    )
    offline_replay_buffer.dataset = offline_dataset

    # Load additional offline datasets
    load_additional_offline_datasets(
        cfg=cfg,
        offline_dataset=offline_dataset,
        offline_replay_buffer=offline_replay_buffer,
        storage_device=storage_device,
        is_main_process=is_main_process
    )

    if is_main_process:
        logging.info(f"Offline buffer initialized with {len(offline_replay_buffer)} samples")

    # ── Validation dataset + action manifold (rank-0 only) ────────────────────
    val_dataset = None
    val_ep_indices = None
    manifold_cache = None
    if is_main_process:
        val_dataset, val_ep_indices = load_val_dataset(cfg, main_dataset=offline_dataset)
        if val_dataset is not None:
            manifold_cache = init_action_manifold(
                offline_dataset, None, cfg, device, output_dir=cfg.output_dir
            )
    
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

    # Share underlying GPU memory for frozen critic target layers to save VRAM.
    # Must happen AFTER accelerator.prepare, which moves params to GPU (new tensors).
    _policy = accelerator.unwrap_model(policy)
    if hasattr(_policy, "critic") and hasattr(_policy, "critic_target"):
        for param, target_param in zip(_policy.critic.parameters(), _policy.critic_target.parameters()):
            if not param.requires_grad:
                target_param.data = param.data
    
    # Helper function for bfloat16 casting
    cast_to_bf16_fn = cast_to_bf16 if cfg.policy.dtype == "bfloat16" else None
    
    # ── Optional step-0 baseline validation ──────────────────────────────────
    if is_main_process and val_dataset is not None and getattr(cfg, "val_on_start", False):
        logging.info("[OFFLINE LEARNER] Running step-0 baseline validation …")
        run_validation(
            policy=accelerator.unwrap_model(policy),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            val_dataset=val_dataset,
            val_ep_indices=val_ep_indices,
            manifold_cache=manifold_cache,
            cfg=cfg,
            step=0,
            output_dir=cfg.output_dir,
            wandb_logger=wandb_logger,
            device=device,
        )
        gc.collect()
        torch.cuda.empty_cache()

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
        if optimization_step % 10 == 0:
            print(f'optimization_step: {optimization_step}')

        # UTD ratio - 1 updates (critic only)
        for _ in range(utd_ratio - 1):
            _update_critic(
                policy=accelerator.unwrap_model(policy),
                optimizers=optimizers,
                online_iterator=offline_iterator,
                offline_iterator=None,
                device=device,
                cfg=cfg,
                dataset_repo_id=None,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm_value=clip_grad_norm_value,
                cast_to_bf16_fn=cast_to_bf16_fn,
                use_amp=False,
                scaler=None
            )
            
            # Update target networks
            if hasattr(policy, "module"):
                policy.module.update_target_networks()
            else:
                policy.update_target_networks()
        
        
        # Shared update step
        training_infos = pi05_update_step(
            policy=accelerator.unwrap_model(policy),
            optimizers=optimizers,
            online_iterator=offline_iterator,
            offline_iterator=None,
            batch_size=batch_size,
            device=device,
            cfg=cfg,
            optimization_step=optimization_step,
            dataset_repo_id=None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            critic_warmup_steps=critic_warmup_steps,
            policy_update_freq=policy_update_freq,
            clip_grad_norm_value=clip_grad_norm_value,
            dataset=offline_dataset,
            cast_to_bf16_fn=cast_to_bf16_fn,
            use_amp=False,
            scaler=None,
            preprocessor=preprocessor,
        )
        
        
        # Log training metrics
        if optimization_step % log_freq == 0:
            training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            log_pi05_training_metrics(
                training_infos=training_infos,
                optimization_step=optimization_step,
                wandb_logger=wandb_logger,
                policy=accelerator.unwrap_model(policy),
                is_main_process=is_main_process
            )

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

        # Periodic validation (rank-0 only)
        if is_main_process and val_dataset is not None and optimization_step % val_freq == 0:
            # Offload optimizer states to CPU to free VRAM for validation
            for _opt in optimizers.values():
                for _state in _opt.state.values():
                    for _k, _v in _state.items():
                        if isinstance(_v, torch.Tensor):
                            _state[_k] = _v.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            run_validation(
                policy=accelerator.unwrap_model(policy),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                val_dataset=val_dataset,
                val_ep_indices=val_ep_indices,
                manifold_cache=manifold_cache,
                cfg=cfg,
                step=optimization_step,
                output_dir=cfg.output_dir,
                wandb_logger=wandb_logger,
                device=device,
            )

            # Restore optimizer states to GPU and reclaim validation VRAM
            for _opt in optimizers.values():
                for _state in _opt.state.values():
                    for _k, _v in _state.items():
                        if isinstance(_v, torch.Tensor):
                            _state[_k] = _v.to(device)
            gc.collect()
            torch.cuda.empty_cache()

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
    
    # Create checkpoint directory (convert output_dir to Path if it's a string)
    output_dir_path = Path(cfg.output_dir) if isinstance(cfg.output_dir, str) else cfg.output_dir
    checkpoint_dir = get_step_checkpoint_dir(output_dir_path, offline_steps, optimization_step)
    
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
        params=[p for n, p in policy.actor.named_parameters() if p.requires_grad],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        params=[p for p in policy.critic_ensemble.parameters() if p.requires_grad],
        lr=cfg.policy.critic_lr,
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
