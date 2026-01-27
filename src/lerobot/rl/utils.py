import torch
import torch.nn as nn
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
)
from lerobot.processor.core import TransitionKey

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
    # Add dummy subtask for Pi05Full compatibility
    batch_for_proc[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * current_batch_size}
    # Use inference advantage for offline training
    # Handle DDP wrapping for config access
    policy_config = getattr(policy, "module", policy).config
    batch_for_proc["advantage"] = torch.full((current_batch_size, 1), policy_config.inference_advantage, device=actions.device)
    
    with torch.no_grad():
        # Access preprocessor - handle potential accelerate wrapping
        preprocessor = getattr(policy, 'preprocessor', None) or getattr(policy.module, 'preprocessor', None)
        processed_batch = preprocessor(batch_for_proc)
    
    # Preprocess next observations
    next_batch_for_proc = {k: v for k, v in next_observations.items()}
    next_batch_for_proc["task"] = [task] * current_batch_size
    next_batch_for_proc[ACTION] = actions  # Required by preprocessor
    # Add dummy subtask for Pi05Full compatibility
    next_batch_for_proc[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * current_batch_size}
    next_batch_for_proc["advantage"] = batch_for_proc["advantage"]
    
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
        "task": batch_for_proc["task"],
        "advantage": batch_for_proc["advantage"],
        "next.done": done,
    }
    
    # Copy raw observations (policy.forward will re-preprocess if needed)
    for key in observations.keys():
        forward_batch["state"][key] = observations[key]
    
    for key in next_observations.keys():
        forward_batch["next_state"][key] = next_observations[key]
    
    # Add tokens
    forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
    forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    
    # Add critic tokens if present (from CriticTokenizerProcessorStep)
    if "critic_tokens" in processed_batch:
        forward_batch["critic_tokens"] = processed_batch["critic_tokens"]
        forward_batch["critic_pad_mask"] = processed_batch["critic_pad_mask"]
        # Also add to state/next_state dicts if needed by policy (policy looks in batch root for critic tokens)
    
    # Add tokens for next state (CRITICAL for critic)
    forward_batch["next_state"][OBS_LANGUAGE_TOKENS] = processed_next_batch[OBS_LANGUAGE_TOKENS]
    forward_batch["next_state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_ATTENTION_MASK]
    
    if "critic_tokens" in processed_next_batch:
        forward_batch["next_state"]["critic_tokens"] = processed_next_batch["critic_tokens"]
        forward_batch["next_state"]["critic_pad_mask"] = processed_next_batch["critic_pad_mask"]
    
    forward_batch[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
    forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    
    return forward_batch

def monitor_advantage_impact(policy, batch, device, wandb_logger, step, cfg):
    """
    Monitors the impact of different advantage values on the policy output by observing the loss.
    If advantage affects v_t, and u_t is fixed, then MSE(u_t, v_t) should change.
    """
    policy.eval()
    advantages = [-0.32, -0.16, 0.0, 0.16, 0.32]
    
    # Select a few unique states (e.g., first 2)
    num_states_to_test = min(2, batch[ACTION].shape[0])
    
    # Fix noise and time
    # Handle DDP wrapping
    policy_model = getattr(policy, "module", policy).model
    chunk_size = policy_model.config.chunk_size
    action_dim = policy_model.config.max_action_dim
    
    with torch.no_grad():
        # Generate fixed noise and time for the subset
        noise = torch.randn(num_states_to_test, chunk_size, action_dim, device=device, dtype=torch.bfloat16)
        time = torch.full((num_states_to_test,), 0.5, device=device, dtype=torch.bfloat16)
        
        # Preprocess images once for the subset
        subset_batch = {
            "observation.images.wrist": batch["state"]["observation.images.wrist"][:num_states_to_test],
            "observation.images.top": batch["state"]["observation.images.top"][:num_states_to_test],
            "observation.images.side": batch["state"]["observation.images.side"][:num_states_to_test],
            "observation.state": batch["state"]["observation.state"][:num_states_to_test],
        }
        # Use policy._preprocess_images which handles the dict -> tensor conversion
        # Handle DDP wrapping
        policy_unwrapped = getattr(policy, "module", policy)
        images, img_masks = policy_unwrapped._preprocess_images(subset_batch)
        
        subset_actions = batch[ACTION][:num_states_to_test]
        
        losses = []
        for adv_val in advantages:
            # Prepare batch with specific advantage
            batch_for_proc = {k: v for k, v in subset_batch.items()}
            batch_for_proc["task"] = [cfg.policy.task] * num_states_to_test
            batch_for_proc[ACTION] = subset_actions
            batch_for_proc["advantage"] = torch.full((num_states_to_test,), adv_val, device=device, dtype=torch.float32)
            
            # Tokenize
            processed = policy_unwrapped.preprocessor(batch_for_proc)
            tokens = processed[OBS_LANGUAGE_TOKENS]
            masks = processed[OBS_LANGUAGE_ATTENTION_MASK]
            normalized_actions = processed[ACTION]
            
            # Forward pass (returns loss)
            # Forward pass (returns loss)
            loss = policy_model(
                images=images, 
                img_masks=img_masks, 
                high_level_task_tokens=tokens, 
                high_level_task_masks=masks, 
                subtask_tokens=None,
                subtask_masks=None,
                action_tokens=None,
                action_masks=None,
                actions=normalized_actions,
                noise=noise, time=time
            )
            # loss is [B, chunk, dim] (reduction="none")
            # Mean over chunk and dim to get scalar per batch item
            loss_per_item = loss.mean(dim=(1, 2)) 
            losses.append(loss_per_item) # [B]
            
        # Stats
        stack_losses = torch.stack(losses) # [5, B]
        std_losses = stack_losses.std(dim=0) # [B]
        mean_std = std_losses.mean().item()
        max_std = std_losses.max().item()
        mean_loss = stack_losses.mean().item()
        
        print(f"[Advantage Monitor] Step {step}: Loss Std (Mean): {mean_std:.6f}, Loss Std (Max): {max_std:.6f}, Loss Mean: {mean_loss:.6f}")
        
        if wandb_logger:
             wandb_logger.log_dict({
                 "train/advantage_impact_loss_std_mean": mean_std,
                 "train/advantage_impact_loss_std_max": max_std,
                 "train/advantage_impact_loss_mean": mean_loss,
                 "Optimization step": step
             }, mode="train", custom_step_key="Optimization step")
             
    policy.train()
