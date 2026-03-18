import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.utils.constants import (
    ACTION, 
    OBS_LANGUAGE_TOKENS, 
    OBS_LANGUAGE_ATTENTION_MASK, 
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    ACTION_TOKENS,
    ACTION_TOKEN_MASK,
)
from lerobot.rl.utils import (
    preprocess_batch_for_pi05
)
from lerobot.rl.buffer import concatenate_batch_transitions
from lerobot.utils.transition import move_transition_to_device
from lerobot.policies.pi05_full.modeling_pi05 import PI05FullPolicy
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep
import time
import numpy as np
import logging

from lerobot.processor.core import TransitionKey


def log_sampled_actions(
    policy,
    snapshot: dict,
    optimization_step: int,
    wandb_logger,
):
    """
    This function compares ground truth action to reconsctructed actions.
    The policy learns the velocity field, instead of the actions. 
    This function is helpful for tracking if the policy is going astray.
    """
    with torch.no_grad():
        images, img_masks = policy._preprocess_images(snapshot["observations"])
        sampled = policy.model.sample_actions(
            images,
            img_masks,
            snapshot["task_tokens"],
            snapshot["task_masks"],
            None,
            None
        )  

    # Use the first chunk step, first 6 dims — the action the robot would actually execute
    sampled = sampled[:, :, :6].float().cpu()
    gt_6_normalized = snapshot["gt_actions_normalized"][:, :, :6]           # [B, 6] float32 CPU
    
    dim_means = sampled.mean(dim=0).mean(dim=0).tolist()
    dim_stds  = sampled.std(dim=0).mean(dim=0).tolist()

    vs_gt_mse = F.mse_loss(sampled, gt_6_normalized).item()

    if wandb_logger:
        import wandb  # local import to avoid hard dependency in train utils
        scalar_log = {
            "Optimization step": optimization_step,
            "sampled_action/vs_gt_mse": vs_gt_mse,
        }
        for i, (m, s) in enumerate(zip(dim_means, dim_stds)):
            scalar_log[f"sampled_action/dim{i}_mean"] = m
            scalar_log[f"sampled_action/dim{i}_std"]  = s
        wandb_logger._wandb.log(scalar_log)
        wandb_logger._wandb.log({
            "sampled_action/histogram":    wandb.Histogram(sampled.numpy().flatten()),
            "sampled_action/gt_histogram": wandb.Histogram(gt_6_normalized.numpy().flatten()),
            "Optimization step": optimization_step,
        })



def hydrate_subtasks(indices: list | torch.Tensor, dataset) -> list[str]:
    """
    Converts subtask indices to subtask description strings using dataset metadata.
    
    Args:
        indices: List or Tensor of subtask indices (can include -1 for missing/online).
        dataset: The dataset object containing metadata (meta.subtasks).
        
    Returns:
        List of subtask strings. Returns empty string "" for index -1.
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
        
    subtask_names = []
    
    # helper to check if we can lookup
    has_meta = dataset and hasattr(dataset, "meta") and hasattr(dataset.meta, "subtasks")
    subtasks_df = None
    
    if has_meta:
        subtasks_df = dataset.meta.subtasks

    for i in indices:
        i = int(i)
        name = ""
        # -1 indicates "no subtask" (e.g. online data, or unannotated frames)
        if i >= 0 and has_meta:
             try:
                 if hasattr(subtasks_df, "columns") and hasattr(subtasks_df, "index"):
                     if "subtask_index" in subtasks_df.columns:
                         if i in subtasks_df.index and subtasks_df.loc[i].get("subtask_index", -999) == i:
                              name = subtasks_df.loc[i]["subtask"]
                         else:
                              rows = subtasks_df[subtasks_df["subtask_index"] == i]
                              if not rows.empty:
                                  name = rows.iloc[0]["subtask"]
                     elif i in subtasks_df.index:
                         name = subtasks_df.loc[i]["subtask"]
                 elif hasattr(subtasks_df, "__getitem__"):
                     name = subtasks_df[i]
             except Exception as e:
                 # logging.debug(f"Failed to lookup subtask index {i}: {e}")
                 pass 
                 
        subtask_names.append(name)
    
    return subtask_names
        

def remap_subtasks_for_dataset(target_dataset, source_dataset, is_main_process=False):
    """
    Computes a remapping table for subtask indices from source_dataset to target_dataset.
    If source_dataset has subtasks not present in target_dataset, they are added to target_dataset's metadata.
    
    Args:
        target_dataset: The main dataset (destination of transitions).
        source_dataset: The dataset being merged in.
        is_main_process (bool): Whether to log details.
        
    Returns:
        remap_table (dict): Map from source_index -> target_index.
    """
    remap_table = {} # old_index -> new_index
            
    # Helper to get subtask map {index: name}
    def get_idx_to_name(ds):
        mapping = {}
        if hasattr(ds, "meta") and hasattr(ds.meta, "subtasks"):
            df = ds.meta.subtasks
            # Check if 'subtask_index' is a column
            if "subtask_index" in df.columns:
                # Iterate rows. We assume index is the subtask name (as per annotate script)
                # but we should be robust.
                for idx, row in df.iterrows():
                    # If the index is the name, use it.
                    name = idx if isinstance(idx, str) else row.get("subtask", str(idx))
                    i = int(row["subtask_index"])
                    mapping[i] = name
        return mapping

    target_idx_to_name = get_idx_to_name(target_dataset)
    source_idx_to_name = get_idx_to_name(source_dataset)
    
    # Build name -> target_index map
    target_name_to_idx = {n: i for i, n in target_idx_to_name.items()}
    
    # Next available index in target dataset
    next_new_index = max(target_idx_to_name.keys()) + 1 if target_idx_to_name else 0
    
    # Calculate remapping
    for old_idx, name in source_idx_to_name.items():
        if name in target_name_to_idx:
            # Subtask exists in target dataset, use its index
            remap_table[old_idx] = target_name_to_idx[name]
        else:
            # New subtask, assign new index and update target metadata
            new_idx = next_new_index
            remap_table[old_idx] = new_idx
            target_name_to_idx[name] = new_idx
            target_idx_to_name[new_idx] = name
            next_new_index += 1
            
            # Update target_dataset metadata in memory
            if hasattr(target_dataset, "meta") and hasattr(target_dataset.meta, "subtasks"):
                    import pandas as pd
                    new_row = pd.DataFrame([{"subtask_index": new_idx}], index=[name])
                    target_dataset.meta.subtasks = pd.concat([target_dataset.meta.subtasks, new_row])
    
    if is_main_process and remap_table:
        logging.info(f"Remapping subtasks: {remap_table}")
        
    return remap_table


def load_additional_offline_datasets(
    cfg, 
    offline_dataset, 
    offline_replay_buffer, 
    storage_device, 
    is_main_process=True
):
    """Loads additional offline datasets and adds their transitions to the offline replay buffer."""
    if not (hasattr(cfg.dataset, "additional_offline_dataset_paths") and cfg.dataset.additional_offline_dataset_paths):
        return

    import torchvision.transforms.functional as F_vision
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.rl.buffer import ReplayBuffer
    
    expected_height, expected_width = 224, 224

    for path in cfg.dataset.additional_offline_dataset_paths:
        if is_main_process:
            logging.info(f"Loading additional offline dataset from {path}")
        
        # Ensure all processes load the dataset
        additional_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=path,
        )
        additional_dataset.delta_timestamps = None
        additional_dataset.delta_indices = None
        
        generator = ReplayBuffer._lerobotdataset_to_transitions_generator(
            additional_dataset,
            state_keys=cfg.policy.input_features.keys()
        )

        if is_main_process:
            logging.info(f"Adding transitions from {path} to offline buffer...")

        # --- Subtask Remapping Logic ---
        remap_table = remap_subtasks_for_dataset(offline_dataset, additional_dataset, is_main_process)
        
        for data in generator:
            # Process data (resize, cast, move)
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        if "images" in key:
                            if tensor.shape[-2:] != (expected_height, expected_width):
                                tensor = F_vision.resize(tensor, (expected_height, expected_width))
                                tensor = tensor.clamp(0.0, 1.0)
                            v[key] = tensor.to(dtype=torch.bfloat16, device=storage_device)
                        else:
                            v[key] = tensor.to(dtype=torch.bfloat16, device=storage_device)
                elif isinstance(v, torch.Tensor):
                        data[k] = v.to(dtype=torch.bfloat16, device=storage_device)
            
            # Apply remapping to subtask indices
            comp_info = data.get("complementary_info", {})
            if comp_info and "subtask_index" in comp_info:
                old_idx = comp_info["subtask_index"]
                if isinstance(old_idx, torch.Tensor):
                    old_idx = old_idx.item()
                
                if old_idx in remap_table:
                    new_idx = remap_table[old_idx]
                    if isinstance(data.get("complementary_info", {}).get("subtask_index"), torch.Tensor):
                         comp_info["subtask_index"] = torch.tensor(new_idx, dtype=torch.int64)
                    else:
                         comp_info["subtask_index"] = new_idx
            
            offline_replay_buffer.add(
                state=data["state"],
                action=data[ACTION],
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,
                complementary_info=comp_info,
            )
        
        if is_main_process:
            logging.info(f"Finished adding transitions from {path}. Buffer size: {len(offline_replay_buffer)}")



def _prepare_batch(online_iterator, offline_iterator, dataset_repo_id, device, cast_to_bf16_fn):
    batch = next(online_iterator)

    if dataset_repo_id is not None and offline_iterator is not None:
        batch_offline = next(offline_iterator)
        batch_offline[ACTION] = batch_offline[ACTION][..., :6] # Slice offline actions
        batch = concatenate_batch_transitions(
            left_batch_transitions=batch, right_batch_transition=batch_offline
        )

    batch = move_transition_to_device(batch, device)
    batch[ACTION] = batch[ACTION][..., :6] # Ensure 6D actions
    
    if cast_to_bf16_fn:
        batch = cast_to_bf16_fn(batch)
        
    actions = batch[ACTION]
    rewards = batch["reward"]
    if "state" in batch:
        observations = batch["state"]
    else:
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}
        
    if "next_state" in batch:
        next_observations = batch["next_state"]
    else:
        next_observations = {k: v for k, v in batch.items() if k.startswith("next.observation.")}
    done = batch["done"]
    
    return batch, actions, rewards, observations, next_observations, done


def _update_critic(
    policy: PI05FullPolicy,
    optimizers: dict[str, torch.optim.Optimizer],
    online_iterator: any,
    offline_iterator: any,
    device: str,
    cfg: any,
    dataset_repo_id: str | None,
    gradient_accumulation_steps: int,
    clip_grad_norm_value: float,
    cast_to_bf16_fn,
    use_amp: bool,
    scaler
) -> dict:
    accum_loss_critic = 0.0
    critic_values_list = []
    td_error_list = []
    target_values_list = []
    loss_critic_raw_list = []
    
    optimizers["critic"].zero_grad(set_to_none=True)
    
    for _ in range(gradient_accumulation_steps):
        # Sample Batch
        batch, actions, rewards, observations, next_observations, done = _prepare_batch(
            online_iterator, offline_iterator, dataset_repo_id, device, cast_to_bf16_fn
        )
        
        # Preprocess for Critic
        
        # Note: No subtask hydration needed here — critic doesn't use subtasks.
        forward_batch_critic = preprocess_batch_for_pi05(
            policy=policy,
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            done=done,
            task=cfg.policy.task,
        )

        # Forward Critic

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                critic_output = policy.forward(forward_batch_critic, model="critic")
                loss_critic = critic_output["loss_critic"] / gradient_accumulation_steps
        else:
            critic_output = policy.forward(forward_batch_critic, model="critic")
            loss_critic = critic_output["loss_critic"] / gradient_accumulation_steps

        # Backward
        if scaler:
            scaler.scale(loss_critic).backward()
        else:
            loss_critic.backward()
            
        # Accumulate metrics
        accum_loss_critic += critic_output["loss_critic"].detach().item()
        critic_values_list.append(critic_output["critic_values"].detach())
        td_error_list.append(critic_output["td_error"].detach())
        target_values_list.append(critic_output["target_values"].detach())
        if "loss_critic_raw" in critic_output:
            loss_critic_raw_list.append(critic_output["loss_critic_raw"].detach())


    # Step Critic Optimizer
    if scaler:
        scaler.unscale_(optimizers["critic"])
        
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters=policy.critic_ensemble.parameters(),
        max_norm=clip_grad_norm_value
    ).item()
    
    if scaler:
        scaler.step(optimizers["critic"])
        scaler.update()
    else:
        optimizers["critic"].step()
        
    # Free critic gradients immediately to save memory during actor update
    optimizers["critic"].zero_grad(set_to_none=True)
        
    # Aggregate Critic Metrics
    all_critic_values = torch.cat(critic_values_list, dim=0)
    all_td_errors = torch.cat(td_error_list, dim=0)
    all_target_values = torch.cat(target_values_list, dim=0)
    
    training_infos = {
        "loss_critic": accum_loss_critic / gradient_accumulation_steps,
        "critic_grad_norm": critic_grad_norm,
        "td_error_mean": all_td_errors.mean().item(),
        "td_error_std": all_td_errors.std().item() if all_td_errors.numel() > 1 else 0.0,
        "critic_value_mean": all_critic_values.mean().item(),
        "critic_value_std": all_critic_values.std().item() if all_critic_values.numel() > 1 else 0.0,
        "target_value_mean_critic": all_target_values.mean().item(),
        "target_value_std": all_target_values.std().item() if all_target_values.numel() > 1 else 0.0,
        "critic_histogram_from_critic": all_critic_values.detach().float().cpu().numpy(),
        "target_value_histogram": all_target_values.detach().float().cpu().numpy(),
    }
    
    if loss_critic_raw_list:
        all_loss_critic_raw = torch.cat(loss_critic_raw_list, dim=0)
        training_infos["loss_critic_raw"] = all_loss_critic_raw.detach().float().cpu().numpy()

    # Store pass 1 raw pass-throughs for the actor
    # NOTE: As the previous implementation just overwrote last_batch_critic sequentially, 
    # we return the very last one. 
    return training_infos


def _compute_advantage_with_interventions(
    policy: PI05FullPolicy,
    batch: dict,
    observations: dict,
    next_observations: dict,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    cfg: any
):
    # Preprocess for Critic
    forward_batch_pass1 = preprocess_batch_for_pi05(
        policy=policy,
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        done=done,
        task=cfg.policy.task,
    )
    
    with torch.no_grad():
        critic_out_pass1 = policy.forward(forward_batch_pass1, model="critic")
        
        target_v = critic_out_pass1["target_values"]
        current_v = critic_out_pass1["critic_values"]
        
        # Calculate Raw Advantage
        raw_advantage = target_v - current_v
        
        # [NEW] Intervention Override Logic
        intervention_key = TeleopEvents.IS_INTERVENTION.value
        if "complementary_info" in batch and intervention_key in batch["complementary_info"]:
            is_intervention = batch["complementary_info"][intervention_key]
            is_intervention_mask = (is_intervention > 0.5)
            
            if is_intervention_mask.shape != raw_advantage.shape:
                if is_intervention_mask.numel() < raw_advantage.numel():
                    padding_size = raw_advantage.numel() - is_intervention_mask.numel()
                    padding = torch.zeros(padding_size, dtype=torch.bool, device=is_intervention_mask.device)
                    is_intervention_mask = torch.cat([is_intervention_mask.view(-1), padding]).view(raw_advantage.shape)
                else:
                    is_intervention_mask = is_intervention_mask.view(raw_advantage.shape)
            
            raw_advantage[is_intervention_mask] = 1.0 # Max advantage for interventions

        raw_advantage_flat = raw_advantage.view(-1)

    return raw_advantage, raw_advantage_flat, current_v, target_v


def _prepare_actor_batch(
    batch: dict,
    observations: dict,
    actions: torch.Tensor,
    raw_advantage_flat: torch.Tensor,
    cfg: any,
    dataset: any,
    preprocessor: any,
) -> tuple[dict, torch.Tensor | None]:
    batch_for_proc = {k: v for k, v in observations.items()}
    # batch_for_proc["task"] and ["advantage"] are now inside COMPLEMENTARY_DATA below
    batch_for_proc[ACTION] = actions
    
    # Inject mapped subtasks
    subtasks = [""] * actions.shape[0]
    subtask_indices = None
    
    # Check for complementary data under various keys
    comp_data = batch.get(TransitionKey.COMPLEMENTARY_DATA) or batch.get("complementary_info")
    if comp_data is not None and "subtask_index" in comp_data:
        subtask_indices = comp_data["subtask_index"]
        # Use shared hydration function
        subtasks = hydrate_subtasks(subtask_indices, dataset)
    
    batch_for_proc[TransitionKey.COMPLEMENTARY_DATA] = {
        "subtask": subtasks,
        "task": [cfg.policy.task] * actions.shape[0],
        "advantage": raw_advantage_flat
    }
    
    with torch.no_grad():
        if preprocessor is None:
            raise ValueError("preprocessor must be provided for PI05 update step")
        processed_batch = preprocessor(batch_for_proc)
        
    return processed_batch, subtask_indices


def _construct_actor_forward_batch(
    processed_batch: dict,
    observations: dict,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    done: torch.Tensor,
    raw_advantage: torch.Tensor,
    cfg: any,
    subtask_indices: list | torch.Tensor | None,
    device: str,
    cast_to_bf16_fn: any,
) -> dict:
    # --- MASKING LOGIC for Online Subtasks ---
    if subtask_indices is not None:
        if isinstance(subtask_indices, torch.Tensor):
            is_online_mask = (subtask_indices == -1)
        else:
            is_online_mask = torch.tensor([i == -1 for i in subtask_indices], device=device)
        
        if is_online_mask.any():
            if OBS_LANGUAGE_SUBTASK_ATTENTION_MASK in processed_batch:
                processed_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK][is_online_mask] = 0
        
    # Construct forward batch for Actor
    forward_batch_actor = {
        ACTION: processed_batch[ACTION],
        "reward": rewards,
        "state": observations.copy(), # Should contain images/etc
        "next_state": {},
        "done": done,
        "task": [cfg.policy.task] * actions.shape[0], # Using the task we injected earlier
        "advantage": raw_advantage,
        "next.done": done,
    }
    
    # Update tokens in state
    forward_batch_actor["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
    forward_batch_actor["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    
    # Add critic tokens if available (from pass 1)
    if "critic_tokens" in processed_batch:
        forward_batch_actor["critic_tokens"] = processed_batch["critic_tokens"]
        forward_batch_actor["critic_pad_mask"] = processed_batch["critic_pad_mask"]
    
    # Flatten convenient tokens
    forward_batch_actor[OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
    forward_batch_actor[OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Add subtask tokens if present
    if OBS_LANGUAGE_SUBTASK_TOKENS in processed_batch:
        forward_batch_actor["state"][OBS_LANGUAGE_SUBTASK_TOKENS] = processed_batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        forward_batch_actor["state"][OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]
        
        # Also convenient flat access?
        forward_batch_actor[OBS_LANGUAGE_SUBTASK_TOKENS] = processed_batch[OBS_LANGUAGE_SUBTASK_TOKENS]
        forward_batch_actor[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK]

    # Add action tokens if present in complementary data
    comp_data_out = processed_batch.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if ACTION_TOKENS in comp_data_out:
        forward_batch_actor[ACTION_TOKENS] = comp_data_out[ACTION_TOKENS]
        forward_batch_actor[ACTION_TOKEN_MASK] = comp_data_out[ACTION_TOKEN_MASK]
    elif ACTION_TOKENS in processed_batch:
        forward_batch_actor[ACTION_TOKENS] = processed_batch[ACTION_TOKENS]
        forward_batch_actor[ACTION_TOKEN_MASK] = processed_batch[ACTION_TOKEN_MASK]

    if cast_to_bf16_fn:
         forward_batch_actor = cast_to_bf16_fn(forward_batch_actor)
         
    return forward_batch_actor


def _update_actor(
    policy: PI05FullPolicy,
    optimizers: dict[str, torch.optim.Optimizer],
    online_iterator: any,
    offline_iterator: any,
    device: str,
    cfg: any,
    dataset_repo_id: str | None,
    gradient_accumulation_steps: int,
    policy_update_freq: int,
    clip_grad_norm_value: float,
    dataset=None,
    cast_to_bf16_fn=None,
    use_amp: bool = False,
    scaler=None,
    preprocessor=None,
) -> dict:
    actor_infos = {}
    
    for _ in range(policy_update_freq):
        optimizers["actor"].zero_grad(set_to_none=True)
        
        accum_loss_actor = 0.0
        accum_flow_loss = 0.0
        accum_action_loss = 0.0
        accum_subtask_loss = 0.0
        accum_flow_first_10 = None
        accum_flow_last_10 = None
        accum_flow_time_low = 0.0
        accum_flow_time_high = 0.0
        count_flow_time_low = 0
        count_flow_time_high = 0
        advantage_values_list = []
        critic_values_actor_list = []
        target_values_actor_list = []
        reward_values_list = []

        
        for accum_step in range(gradient_accumulation_steps):
            # Sample NEW batch for actor update
            batch, actions, rewards, observations, next_observations, done = _prepare_batch(
                online_iterator, offline_iterator, dataset_repo_id, device, cast_to_bf16_fn
            )
            
            # --- Step 1: Calculate Advantage (Pass 1) ---
            raw_advantage, raw_advantage_flat, current_v, target_v = _compute_advantage_with_interventions(
                policy=policy,
                batch=batch,
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                done=done,
                cfg=cfg
            )

            # --- Step 2: Re-tokenize ---
            processed_batch, subtask_indices = _prepare_actor_batch(
                batch=batch,
                observations=observations,
                actions=actions,
                raw_advantage_flat=raw_advantage_flat,
                cfg=cfg,
                dataset=dataset,
                preprocessor=preprocessor
            )
            
            # --- Step 3: Actor Update (Pass 3) ---
            forward_batch_actor = _construct_actor_forward_batch(
                processed_batch=processed_batch,
                observations=observations,
                actions=actions,
                rewards=rewards,
                done=done,
                raw_advantage=raw_advantage,
                cfg=cfg,
                subtask_indices=subtask_indices,
                device=device,
                cast_to_bf16_fn=cast_to_bf16_fn
            )
            
            # External Metrics for Efficiency
            # We calculate squashed advantage here to log consistent with what the model *would* report
            # But we pass the RAW advantage and values to the model to skip re-computation
            squashed_advantage = torch.tanh(raw_advantage / cfg.policy.advantage_scaling)
            
            external_metrics = {
                "advantage": squashed_advantage,
                "critic_values": current_v,
                "target_values": target_v,
                "rewards": rewards
            }

            # Forward Actor
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    actor_output = policy.forward(forward_batch_actor, model="actor", external_metrics=external_metrics)
                    loss_actor = actor_output["loss_actor"] / gradient_accumulation_steps
            else:
                actor_output = policy.forward(forward_batch_actor, model="actor", external_metrics=external_metrics)
                loss_actor = actor_output["loss_actor"] / gradient_accumulation_steps
            
            loss_actor_mean = loss_actor.mean()
            
            accum_loss_actor += actor_output["loss_actor"].mean().item()
            accum_flow_loss += actor_output["flow_mse_loss"].mean().item()
            
            if hasattr(loss_actor_mean, "backward"):
                 if scaler:
                     scaler.scale(loss_actor_mean).backward()
                 else:
                     loss_actor_mean.backward()

            if "flow_loss_raw" in actor_output:
                raw_loss = actor_output["flow_loss_raw"]
                flow_time = actor_output["flow_time"]
                flow_loss_raw_last = raw_loss
                
                T_dim = raw_loss.shape[1]
                # First 10 and last 10
                chunk_first = raw_loss[:, :10, :].mean().item() if T_dim >= 10 else raw_loss.mean().item()
                chunk_last = raw_loss[:, -10:, :].mean().item() if T_dim >= 10 else raw_loss.mean().item()
                
                if accum_flow_first_10 is None:
                    accum_flow_first_10 = chunk_first
                    accum_flow_last_10 = chunk_last
                else:
                    accum_flow_first_10 += chunk_first
                    accum_flow_last_10 += chunk_last
                    
                # Time noise conditions
                # raw_loss: [B, T, D], flow_time: [B]
                # We want to average the loss for batches where flow_time < 0.3 or > 0.7
                mask_low = flow_time < 0.3
                mask_high = flow_time > 0.7
                
                if mask_low.any():
                    accum_flow_time_low += raw_loss[mask_low].mean().item() * mask_low.sum().item()
                    count_flow_time_low += mask_low.sum().item()
                if mask_high.any():
                    accum_flow_time_high += raw_loss[mask_high].mean().item() * mask_high.sum().item()
                    count_flow_time_high += mask_high.sum().item()
                    
            if "action_ce_loss" in actor_output:
                accum_action_loss += actor_output["action_ce_loss"].mean().item()
            if "subtask_ce_loss" in actor_output:
                accum_subtask_loss += actor_output["subtask_ce_loss"].mean().item()
            advantage_values_list.append(actor_output["advantage_values"].detach())
            critic_values_actor_list.append(actor_output["critic_values"].detach())
            target_values_actor_list.append(actor_output["target_values"].detach())
            reward_values_list.append(actor_output["rewards"].detach())

        # Step Actor Optimizer
        if scaler:
            scaler.unscale_(optimizers["actor"])
        
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.actor.parameters(),
            max_norm=clip_grad_norm_value
        ).item()
        
        if scaler:
            scaler.step(optimizers["actor"])
            scaler.update()
        else:
            optimizers["actor"].step()
            
        # Free actor gradients immediately to save memory for next steps
        optimizers["actor"].zero_grad(set_to_none=True)
            
        # Aggregate Actor Metrics
        all_advantage_values = torch.cat(advantage_values_list, dim=0)
        all_critic_values_actor = torch.cat(critic_values_actor_list, dim=0)
        all_target_values_actor = torch.cat(target_values_actor_list, dim=0)
        all_reward_values = torch.cat(reward_values_list, dim=0)
        training_infos = {}
        
        training_infos["loss_actor"] = accum_loss_actor / gradient_accumulation_steps
        training_infos["flow_mse_loss"] = accum_flow_loss / gradient_accumulation_steps
        training_infos["action_ce_loss"] = accum_action_loss / gradient_accumulation_steps
        training_infos["subtask_ce_loss"] = accum_subtask_loss / gradient_accumulation_steps
        training_infos["actor_grad_norm"] = actor_grad_norm
        training_infos["advantage_mean"] = all_advantage_values.mean().item()
        training_infos["advantage_std"] = all_advantage_values.std().item() if all_advantage_values.numel() > 1 else 0.0
        training_infos["target_value_mean"] = all_target_values_actor.mean().item()
        training_infos["reward_mean"] = all_reward_values.mean().item()
        training_infos["critic_value_mean_actor"] = all_critic_values_actor.mean().item()
        
        training_infos["critic_histogram"] = all_critic_values_actor.detach().float().cpu().numpy() # This overwrites pass 1 critic hist

        if accum_flow_first_10 is not None:
            training_infos["flow_loss_time/mean_first_10"] = accum_flow_first_10 / gradient_accumulation_steps
            training_infos["flow_loss_time/mean_last_10"] = accum_flow_last_10 / gradient_accumulation_steps
            training_infos["flow_loss_raw"] = flow_loss_raw_last.detach().float().cpu().numpy()

        if count_flow_time_low > 0:
            training_infos["flow_loss_noise/mean_low_noise_lt_0.3"] = accum_flow_time_low / count_flow_time_low
        if count_flow_time_high > 0:
            training_infos["flow_loss_noise/mean_high_noise_gt_0.7"] = accum_flow_time_high / count_flow_time_high

        # Snapshot for action-reconstruction logging (consumed by callers via .pop)
        #training_infos["_action_log_snapshot"] = {
        #"observations": observations,
        #"task_tokens": forward_batch_actor[OBS_LANGUAGE_TOKENS],
        #"task_masks": forward_batch_actor[OBS_LANGUAGE_ATTENTION_MASK],
        #"subtask_tokens": forward_batch_actor.get(OBS_LANGUAGE_SUBTASK_TOKENS),
        #"subtask_masks": forward_batch_actor.get(OBS_LANGUAGE_SUBTASK_ATTENTION_MASK),
        #"gt_actions": actions.float().cpu(),
        #"gt_actions_normalized": forward_batch_actor[ACTION].float().cpu(),
        #}

    return training_infos


def pi05_update_step(
    policy: PI05FullPolicy,
    optimizers: dict[str, torch.optim.Optimizer],
    online_iterator: any,
    offline_iterator: any,
    batch_size: int,
    device: str,
    cfg: any,
    optimization_step: int,
    dataset_repo_id: str | None,
    # Learner-specific params
    gradient_accumulation_steps: int = 1,
    critic_warmup_steps: int = 0,
    policy_update_freq: int = 1,
    clip_grad_norm_value: float = 1.0,
    dataset=None, # For subtask metadata
    cast_to_bf16_fn=None,
    use_amp: bool = False,
    scaler=None,
    preprocessor=None,
):
    """
    Performs one optimization step for PI05 (Learner Critic & Actor).
    
    Returns:
        training_infos (dict): Dictionary of aggregated metrics for logging.
    """
    
    # -------------------------------------------------------------------------
    # 1. Critic Update Loop
    # -------------------------------------------------------------------------
    training_infos = _update_critic(
        policy=policy,
        optimizers=optimizers,
        online_iterator=online_iterator,
        offline_iterator=offline_iterator,
        device=device,
        cfg=cfg,
        dataset_repo_id=dataset_repo_id,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad_norm_value=clip_grad_norm_value,
        cast_to_bf16_fn=cast_to_bf16_fn,
        use_amp=use_amp,
        scaler=scaler
    )

    # -------------------------------------------------------------------------
    # 2. Actor Update Loop (Conditional)
    # -------------------------------------------------------------------------
    
    if optimization_step >= critic_warmup_steps and optimization_step % policy_update_freq == 0:
        actor_infos = _update_actor(
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
            dataset=dataset,
            cast_to_bf16_fn=cast_to_bf16_fn,
            use_amp=use_amp,
            scaler=scaler,
            preprocessor=preprocessor
        )


    training_infos = training_infos | actor_infos

    # -------------------------------------------------------------------------
    # 3. Target Network Update & Merge Metrics
    # -------------------------------------------------------------------------
    policy.update_target_networks()
    
    return training_infos

def log_pi05_training_metrics(
    training_infos: dict,
    optimization_step: int,
    wandb_logger,
    policy,
    is_main_process: bool = True
):
    if optimization_step == 0:
        return

    import numpy as np
    import torch
    import wandb

    # Pop snapshot before wandb scalar logging (tensors must not be sent as scalars)
    # action_log_snapshot = training_infos.pop("_action_log_snapshot", None)

    if wandb_logger:
        # Extract histograms if they exist
        advantage_hist = training_infos.pop("advantage_histogram", None)
        critic_hist = training_infos.pop("critic_histogram", None)
        target_value_hist = training_infos.pop("target_value_histogram", None)
        critic_hist_from_critic = training_infos.pop("critic_histogram_from_critic", None)
        actor_loss_hist = training_infos.pop("actor_loss_histogram", None)
        flow_loss_raw = training_infos.pop("flow_loss_raw", None)
        loss_critic_raw = training_infos.pop("loss_critic_raw", None)
        
        # Log scalar metrics
        wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")
        
        # Log histograms separately using wandb.Histogram
        if advantage_hist is not None:
            if isinstance(advantage_hist, torch.Tensor):
                advantage_hist = advantage_hist.detach().float().cpu().numpy()
            wandb_logger._wandb.log({
                "train/advantage_histogram": wandb.Histogram(advantage_hist),
                "Optimization step": optimization_step
            })
        if critic_hist is not None:
            critic_vals = np.clip(critic_hist, -1, .5)
            wandb_logger._wandb.log({
                "train/critic_value_histogram": wandb.Histogram(critic_vals),
                "Optimization step": optimization_step
            })
        
        if target_value_hist is not None:
             target_vals = np.clip(target_value_hist, -1, .5)
             wandb_logger._wandb.log({
                "train/target_value_histogram": wandb.Histogram(target_vals),
                "Optimization step": optimization_step
            })
        
        # Log critic histogram from critic update
        if critic_hist_from_critic is not None:
            critic_vals_from_critic = np.clip(critic_hist_from_critic, -1, .5)
            wandb_logger._wandb.log({
                "train/critic_value_histogram_from_critic": wandb.Histogram(critic_vals_from_critic),
                "Optimization step": optimization_step
            })

        if flow_loss_raw is not None:
            wandb_logger._wandb.log({
                "train/flow_loss_histogram_flat": wandb.Histogram(np.clip(flow_loss_raw.flatten(), 0, 0.04)),
                "Optimization step": optimization_step
            })

        if loss_critic_raw is not None:
            wandb_logger._wandb.log({
                "train/loss_critic_histogram_flat": wandb.Histogram(np.clip(loss_critic_raw.flatten(), 0, .02)),
                "Optimization step": optimization_step
            })

    # --- Action reconstruction logging ---
    # if action_log_snapshot is not None and is_main_process:
    #     # from lerobot.rl.pi05_train_utils import log_sampled_actions
    #     # log_sampled_actions(
    #     #     policy=policy,
    #     #     snapshot=action_log_snapshot,
    #     #     optimization_step=optimization_step,
    #     #     wandb_logger=wandb_logger,
    #     # )

def make_pi05_full_processors_with_upgrade(cfg, dataset=None, is_main_process=True):
    """
    Creates pre/post processors for PI05 RL.
    
    Handles the "Runtime Upgrade" strategy:
    1. Loads stats from a pretrained checkpoint (which might be standard PI05 config).
    2. detailed logging
    3. Creates a NEW pipeline using the current PI05FullConfig (which supports advantage scaling).
    """
    
    preprocessor_overrides = {
        "pi05_full_prepare_state_tokenizer_processor_step": {
             "advantage_scaling": cfg.policy.advantage_scaling
        }
    }
    
    dataset_stats = None
    
    # 1. Check if we should force using dataset stats
    use_dataset_stats = getattr(cfg.policy, "use_dataset_stats", False)
    
    if use_dataset_stats:
        if is_main_process:
            logging.info("Config requests using dataset stats (use_dataset_stats=True).")
        if dataset is not None:
            dataset_stats = dataset.meta.stats
        else:
            if is_main_process:
                logging.warning("use_dataset_stats is True but no dataset provided! Stats will be None.")

    # 2. Check if we are loading the base model (which implies we should use dataset stats)
    elif "pi05_base" in cfg.policy.pi05_checkpoint:
        if is_main_process:
            logging.info(f"Loading base model '{cfg.policy.pi05_checkpoint}'. Using dataset stats.")
        if dataset is not None:
            dataset_stats = dataset.meta.stats
        else:
            if is_main_process:
                logging.warning("Loading base model but no dataset provided! Stats will be None.")

    # 3. Otherwise, try to load stats from the checkpoint
    elif cfg.policy.pi05_checkpoint:
        try:
            if is_main_process:
                logging.info(f"Loading pretrained pipeline from {cfg.policy.pi05_checkpoint} to extract stats...")
            
            # Load existing pipeline just to grab stats
            # We specifically look for "policy_preprocessor.json"
            temp_pipeline = PolicyProcessorPipeline.from_pretrained(
                cfg.policy.pi05_checkpoint,
                config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            )
            for step in temp_pipeline.steps:
                if isinstance(step, NormalizerProcessorStep):
                    dataset_stats = step.stats
                    if is_main_process:
                        logging.info("Successfully extracted dataset_stats from NormalizerProcessorStep.")
                    break
            
            if dataset_stats is None and is_main_process:
                logging.warning("No NormalizerProcessorStep found in pretrained pipeline! Stats will be None.")
                
        except Exception as e:
            if is_main_process:
                logging.warning(f"Failed to load pretrained pipeline for stats extraction: {e}")
                logging.warning("Proceeding with dataset stats as fallback.")
            
            if dataset is not None:
                dataset_stats = dataset.meta.stats
            else:
                 if is_main_process:
                     logging.warning("Fallback to dataset failed: no dataset provided! Stats will be None.")

    # Create fresh pipeline with PI05FullConfig structure + Extracted Stats
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        # pretrained_path=None, # Explicitly NOT loading structure from checkpoint
        dataset_stats=dataset_stats,
        preprocessor_overrides=preprocessor_overrides,
    )
    
    return preprocessor, postprocessor
