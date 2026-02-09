import torch
import torch.nn as nn
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    ACTION_TOKENS,
    ACTION_TOKEN_MASK,
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
    subtasks = observations.get("subtask", [""] * current_batch_size)

    # Use inference advantage for offline training
    # Handle DDP wrapping for config access
    policy_config = getattr(policy, "module", policy).config
    inference_advantage = getattr(policy_config, "inference_advantage", 0.0)

    # Construct Complementary Data
    complementary_data = {
        "task": [task] * current_batch_size,
        "subtask": subtasks,
        "advantage": torch.full((current_batch_size, 1), inference_advantage, device=actions.device)
    }

    # Construct EnvTransition for current step
    batch_for_proc = {
        TransitionKey.ACTION: actions,
        **observations,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data
    }
    
    
    with torch.no_grad():
        # Access preprocessor - handle potential accelerate wrapping
        preprocessor = getattr(policy, 'preprocessor', None) or getattr(policy.module, 'preprocessor', None)
        processed_batch = preprocessor(batch_for_proc)
    
    # Preprocess next observations
    next_subtasks = next_observations.get("subtask", [""] * current_batch_size)

    next_complementary_data = {
        "task": [task] * current_batch_size,
        "subtask": next_subtasks,
        "advantage": complementary_data["advantage"]
    }

    next_batch_for_proc = {
        TransitionKey.ACTION: actions, # Required by preprocessor
        **next_observations,
        TransitionKey.COMPLEMENTARY_DATA: next_complementary_data
    }
    
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
        "task": complementary_data["task"],
        "subtask": complementary_data["subtask"],
        "advantage": complementary_data["advantage"],
        "next.done": done,
    }
    
    # Copy raw observations (policy.forward will re-preprocess if needed)
    for key in observations.keys():
        if key != "subtask":
            forward_batch["state"][key] = observations[key]
    
    for key in next_observations.keys():
        if key != "subtask":
            forward_batch["next_state"][key] = next_observations[key]
    
    # Add tokens from processor output
    if OBS_LANGUAGE_TOKENS in processed_batch:
        forward_batch["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Add subtask tokens if present
    if OBS_LANGUAGE_SUBTASK_TOKENS in processed_batch:
        forward_batch["state"][OBS_LANGUAGE_SUBTASK_TOKENS] = processed_batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        forward_batch["state"][OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]

    # Add action tokens if present in complementary data (as placed by ActionTokenizerProcessorStep)
    comp_data_out = processed_batch.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if ACTION_TOKENS in comp_data_out:
        forward_batch[ACTION_TOKENS] = comp_data_out[ACTION_TOKENS]
        forward_batch[ACTION_TOKEN_MASK] = comp_data_out[ACTION_TOKEN_MASK]
    elif ACTION_TOKENS in processed_batch: # Fallback if Flattened
        forward_batch[ACTION_TOKENS] = processed_batch[ACTION_TOKENS]
        forward_batch[ACTION_TOKEN_MASK] = processed_batch[ACTION_TOKEN_MASK]
    
    # Add critic tokens if present (from CriticTokenizerProcessorStep)
    if "critic_tokens" in processed_batch:
        forward_batch["critic_tokens"] = processed_batch["critic_tokens"]
        forward_batch["critic_pad_mask"] = processed_batch["critic_pad_mask"]
    
    # Add tokens for next state (CRITICAL for critic)
    if OBS_LANGUAGE_TOKENS in processed_next_batch:
        forward_batch["next_state"][OBS_LANGUAGE_TOKENS] = processed_next_batch[OBS_LANGUAGE_TOKENS]
        forward_batch["next_state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Add subtask tokens for next state if present
    if OBS_LANGUAGE_SUBTASK_TOKENS in processed_next_batch:
        forward_batch["next_state"][OBS_LANGUAGE_SUBTASK_TOKENS] = processed_next_batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        forward_batch["next_state"][OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = processed_next_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]
    
    if "critic_tokens" in processed_next_batch:
        forward_batch["next_state"]["critic_tokens"] = processed_next_batch["critic_tokens"]
        forward_batch["next_state"]["critic_pad_mask"] = processed_next_batch["critic_pad_mask"]
    
    forward_batch[OBS_LANGUAGE_TOKENS] = forward_batch["state"].get(OBS_LANGUAGE_TOKENS)
    forward_batch[OBS_LANGUAGE_ATTENTION_MASK] = forward_batch["state"].get(OBS_LANGUAGE_ATTENTION_MASK)
    
    return forward_batch

def cast_to_bf16(item):
    """
    Helper function to cast tensors in a structure to bfloat16.
    """
    if isinstance(item, torch.Tensor):
        if item.dtype == torch.float32:
            return item.to(dtype=torch.bfloat16)
        return item
    elif isinstance(item, dict):
        return {k: cast_to_bf16(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [cast_to_bf16(v) for v in item]
    return item
