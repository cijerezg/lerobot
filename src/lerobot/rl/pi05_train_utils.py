import torch
import torch.nn as nn
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
    preprocess_batch_for_pi05, 
    cast_to_bf16, 
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
    start_critic = time.time()
    
    accum_loss_critic = 0.0
    critic_values_list = []
    td_error_list = []
    target_values_list = []
    
    optimizers["critic"].zero_grad()
    
    # We will keep the LAST batch of the accumulation loop to reuse for the Actor update (Pass 2)
    last_batch_critic = None
    last_forward_batch_critic = None
    
    for _ in range(gradient_accumulation_steps):
        # Sample Batch
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
        
        # Preprocess for Critic
        
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
        
        last_batch_critic = batch
        last_forward_batch_critic = forward_batch_critic

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
    }
    training_infos["critic_histogram_from_critic"] = all_critic_values.detach().float().cpu().numpy()


    # -------------------------------------------------------------------------
    # 2. Actor Update Loop (Conditional)
    # -------------------------------------------------------------------------
    
    if optimization_step >= critic_warmup_steps and optimization_step % policy_update_freq == 0:
        
        for _ in range(policy_update_freq):
            optimizers["actor"].zero_grad()
            
            accum_loss_actor = 0.0
            accum_flow_loss = 0.0
            accum_action_loss = 0.0
            accum_subtask_loss = 0.0
            advantage_values_list = []
            critic_values_actor_list = []
            target_values_actor_list = []
            reward_values_list = []

            
            for accum_step in range(gradient_accumulation_steps):
                # Sample NEW batch for actor update
                batch = next(online_iterator)

                if dataset_repo_id is not None and offline_iterator is not None:
                    batch_offline = next(offline_iterator)
                    batch_offline[ACTION] = batch_offline[ACTION][..., :6]
                    batch = concatenate_batch_transitions(
                        left_batch_transitions=batch, right_batch_transition=batch_offline
                    )
                
                batch = move_transition_to_device(batch, device)
                batch[ACTION] = batch[ACTION][..., :6]
                
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
                
                # --- Step 1: Calculate Advantage (Pass 1) ---
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
                    # Get target values from critic forward pass
                    # We can use forward(model="critic") to get values, but we need V(s) and TargetV explicitly
                    # Actually, forward(model="critic") returns target_values and critic_values!
                    # Note: Using "critic" mode computes gradients for critic loss, but here we are in no_grad
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

                # --- Step 2: Re-tokenize (Pass 2) ---
                batch_for_proc = {k: v for k, v in observations.items()}
                # batch_for_proc["task"] and ["advantage"] are now inside COMPLEMENTARY_DATA below
                batch_for_proc[ACTION] = actions
                
                # Inject mapped subtasks
                subtasks = [""] * actions.shape[0]
                # Check for complementary data under various keys

                comp_data = batch.get(TransitionKey.COMPLEMENTARY_DATA) or batch.get("complementary_info")
                if comp_data is not None:
                    if comp_data is not None and "subtask_index" in comp_data:
                        indices = comp_data["subtask_index"].tolist()
                        if dataset and hasattr(dataset, "meta") and hasattr(dataset.meta, "subtasks"):
                            subtasks_meta = dataset.meta.subtasks
                            # Check if it's a DataFrame (basic check without importing pandas if possible, or assume it behaves like one if it has columns)
                            if hasattr(subtasks_meta, "columns") and hasattr(subtasks_meta, "index"):
                                # Ensure "subtask" is a column by resetting index
                                subtasks_df = subtasks_meta.reset_index()
                                # If "subtask_index" is a column, set it as index for lookup
                                if "subtask_index" in subtasks_df.columns:
                                    subtasks_df = subtasks_df.set_index("subtask_index")
                                subtasks = [subtasks_df.loc[int(i)]["subtask"] if int(i) in subtasks_df.index else "" for i in indices]
                            else:
                                subtasks = [subtasks_meta[int(i)] for i in indices]
                        elif dataset and hasattr(dataset, "subtasks"):
                            subtasks = [dataset.subtasks[int(i)] for i in indices]
                
                batch_for_proc[TransitionKey.COMPLEMENTARY_DATA] = {
                    "subtask": subtasks,
                    "task": [cfg.policy.task] * actions.shape[0],
                    "advantage": raw_advantage_flat
                }
                

                with torch.no_grad():
                    if preprocessor is None:
                        raise ValueError("preprocessor must be provided for PI05 update step")
                    processed_batch = preprocessor(batch_for_proc)
                    
                # --- Step 3: Actor Update (Pass 3) ---
                # Construct forward batch for Actor
                forward_batch_actor = {
                    ACTION: processed_batch[ACTION],
                    "reward": rewards,
                    "state": observations.copy(), # Should contain images/etc
                    "next_state": critic_out_pass1.get("next_state", {}), # Re-use if available or empty
                    "done": done,
                    "task": batch_for_proc[TransitionKey.COMPLEMENTARY_DATA]["task"],
                    "advantage": raw_advantage,
                    "next.done": done,
                }
                
                # Update tokens in state
                forward_batch_actor["state"][OBS_LANGUAGE_TOKENS] = processed_batch[OBS_LANGUAGE_TOKENS]
                forward_batch_actor["state"][OBS_LANGUAGE_ATTENTION_MASK] = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
                
                # Add critic tokens if available (from pass 1)
                # Note: processed_batch MIGHT have critic tokens if preprocessor generates them
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
                
                if hasattr(loss_actor_mean, "backward"):
                     if scaler:
                         scaler.scale(loss_actor_mean).backward()
                     else:
                         loss_actor_mean.backward()

                # Accumulate Actor Metrics
                accum_loss_actor += actor_output["loss_actor"].mean().item()
                if "flow_mse_loss" in actor_output:
                    accum_flow_loss += actor_output["flow_mse_loss"].mean().item()
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
                
            # Aggregate Actor Metrics
            all_advantage_values = torch.cat(advantage_values_list, dim=0)
            all_critic_values_actor = torch.cat(critic_values_actor_list, dim=0)
            all_target_values_actor = torch.cat(target_values_actor_list, dim=0)
            all_reward_values = torch.cat(reward_values_list, dim=0)
            
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
            
            training_infos["critic_histogram"] = all_critic_values.detach().float().cpu().numpy() # This overwrites pass 1 critic hist
                
    # Update target networks
    policy.update_target_networks()
    
    return training_infos

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
    elif cfg.policy.pi05_checkpoint == "lerobot/pi05":
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
