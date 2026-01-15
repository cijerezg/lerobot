import torch
import torch.nn as nn
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
)

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
