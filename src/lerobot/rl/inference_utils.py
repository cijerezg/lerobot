import logging
import math
import threading
import time
import copy
import traceback
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.rl.utils import save_video_with_critic_overlay

logger = logging.getLogger(__name__)

class SharedState:
    """Thread-safe state manager for passing observations from the environment 
    thread to the background inference thread without race conditions.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_obs = None
        self.is_intervening = False
        self.policy_reset_requested = False
        self.running = True
        self.env_wait_time = 0.0
        self.env_steps = 0
        self.inference_wait_time = 0.0
        
        # Recording and Logging
        self.episode_counter = 0
        self.is_logging_episode = False
        self.replay_buffer = None
        self.critic_values = []
        self.subtask_texts = []
        self.current_subtask_text = ""
        self.episode_active = False

    def add_env_wait_time(self, wait_time: float):
        with self.lock:
            self.env_wait_time += wait_time
            self.env_steps += 1

    def add_inference_wait_time(self, wait_time: float):
        with self.lock:
            self.inference_wait_time += wait_time

    def get_and_reset_metrics(self):
        with self.lock:
            metrics = {
                'env_wait_time': self.env_wait_time,
                'env_steps': self.env_steps,
                'inference_wait_time': self.inference_wait_time,
                'env_active_time': getattr(self, 'env_active_time_total', 0.0)
            }
            self.env_wait_time = 0.0
            self.env_steps = 0
            self.inference_wait_time = 0.0
            self.env_active_time_total = 0.0
            return metrics

    def update_observation(self, obs: dict, is_intervening: bool):
        with self.lock:
            # We assume obs is already on device/preprocessed by env_processor
            # We don't deepcopy tensors here to save overhead, but we create 
            # a new dict referencing them. The tensors themselves shouldn't be 
            # modified in-place by the environment loop.
            self.latest_obs = {k: v for k, v in obs.items()}
            self.is_intervening = is_intervening

    def get_latest_observation(self):
        with self.lock:
            if self.latest_obs is None:
                return None
            return {k: v for k, v in self.latest_obs.items()}

    def set_intervention(self, status: bool):
        with self.lock:
            self.is_intervening = status

    def check_and_clear_reset(self) -> bool:
        with self.lock:
            if self.policy_reset_requested:
                self.policy_reset_requested = False
                return True
            return False

    def request_reset(self):
        with self.lock:
            self.policy_reset_requested = True


def convert_env_obs_to_policy_format(env_obs: dict) -> dict:
    """Convert environment observation format to policy-expected format.
    Handles partial conversions to avoid breaking downstream Pi05 preprocessors.
    """
    policy_obs = {}
    
    has_policy_format = (
        'observation.state' in env_obs or
        any(k.startswith('observation.images.') for k in env_obs.keys())
    )

    if has_policy_format:
        for key in env_obs.keys():
            if key == 'observation.state' or key.startswith('observation.images.'):
                policy_obs[key] = env_obs[key]
    
    camera_mapping = {
        'wrist': 'observation.images.wrist',
        'top': 'observation.images.top',
        'side': 'observation.images.side',
    }
    
    pixels_dict = env_obs.get('pixels', env_obs)
    for env_key, policy_key in camera_mapping.items():
        if policy_key not in policy_obs:
            if env_key in pixels_dict:
                policy_obs[policy_key] = pixels_dict[env_key]
    
    if 'observation.state' not in policy_obs:
        joint_order = [
            'shoulder_pan.pos',
            'shoulder_lift.pos',
            'elbow_flex.pos',
            'wrist_flex.pos',
            'wrist_roll.pos',
            'gripper.pos',
        ]
        joint_values = []
        for joint_key in joint_order:
            if joint_key in env_obs:
                val = env_obs[joint_key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor([val], dtype=torch.float32)
                elif val.dim() == 0:
                    val = val.unsqueeze(0)
                joint_values.append(val)
        
        if joint_values:
            policy_obs['observation.state'] = torch.cat(joint_values, dim=0)
    
    return policy_obs


def get_actions_worker(policy, shared_state: SharedState, action_queue, cfg):
    """
    Background inference thread wrapper.
    Continually generates actions via predict_action_chunk using RTC.
    """
    try:
        logger.info("[GET_ACTIONS] Starting background inference thread")
        latency_tracker = LatencyTracker()
        last_subtask_text = None
        inference_step = 0
        
        # In this stripped down standalone mode, we assume policy properties are set
        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk = 1.0 / cfg.env.fps

        # Used for complementary data injections
        task_str = cfg.policy.task
        advantage_val = torch.tensor([[cfg.policy.inference_advantage]], device=torch.device('cpu'), dtype=torch.float32)

        while shared_state.running:
            # 1. Reset check
            if shared_state.check_and_clear_reset():
                policy.reset()
                last_subtask_text = None
                continue
                
            if not shared_state.episode_active:
                time.sleep(0.01)
                continue
            
            # 2. Check if we actually need a new chunk (use p95 for less pessimistic threshold)
            # To avoid over-inferencing and queue saturation, only infer if we're getting close
            # In a real async inference loop we fetch if queue <= execution_horizon + delay
            current_delay = math.ceil(latency_tracker.p95() / time_per_chunk)
            if not action_queue.empty() and action_queue.qsize() > execution_horizon + current_delay:
                wait_start = time.perf_counter()
                time.sleep(0.01)
                shared_state.add_inference_wait_time(time.perf_counter() - wait_start)
                continue

            # 3. Fetch latest environment observation
            latest_obs = shared_state.get_latest_observation()
            if latest_obs is None:
                time.sleep(0.01)
                continue
            
            # 4. Filter features & format
            batch_for_preprocessor = {}
            for k, v in latest_obs.items():
                if k in cfg.policy.input_features:
                    batch_for_preprocessor[k] = v

            # 5. Inject specific complementary data needed by Pi05 preprocessor
            batch_for_preprocessor["robot_type"] = cfg.env.robot.type if hasattr(cfg.env, 'robot') else ""
            batch_for_preprocessor['complementary_data'] = {
                'task': [task_str],
                'subtask': [""],
                'advantage': advantage_val
            }

            # Critic generation moved to env worker

            current_time = time.perf_counter()
            
            # 6. Execute model inference using preprocessor and postprocessor
            with torch.no_grad():
                if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
                    processed_batch = policy.preprocessor(batch_for_preprocessor)
                else:
                    processed_batch = batch_for_preprocessor

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                # Using p95 instead of max: avoids a single latency spike biasing the model
                # to predict too far ahead. See actor_pi05_async_utils.py for full rationale.
                inference_delay = math.ceil(latency_tracker.p95() / time_per_chunk)

                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon
                )

                # --- Subtask Token Decoding ---
                inference_step += 1
                try:
                    cached_tokens = getattr(policy, '_cached_subtask_tokens', None)
                    cached_masks = getattr(policy, '_cached_subtask_masks', None)
                    if cached_tokens is not None and cached_masks is not None:
                        tokenizer = policy.model._paligemma_tokenizer
                        valid_tokens = cached_tokens[0][cached_masks[0]]
                        subtask_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                        with shared_state.lock:
                            shared_state.current_subtask_text = subtask_text
                        if subtask_text != last_subtask_text:
                            logger.info(
                                f"[SUBTASK] inference_step={inference_step} | "
                                f"subtask: \"{subtask_text}\""
                            )
                            last_subtask_text = subtask_text
                except Exception as e:
                    logger.debug(f"[SUBTASK] Could not decode subtask tokens: {e}")

                original_actions = actions_chunk.squeeze(0).clone()
                
                # --- [NEW] Global Chunk Unnormalization & Absolute Action Reconstruction ---
                unnormalized_actions = policy.postprocessor(original_actions) if hasattr(policy, 'postprocessor') and policy.postprocessor is not None else original_actions.clone()
                
                action_encoding = getattr(policy.config, "action_encoding", "absolute")
                anchor_now = None
                if action_encoding in ["anchor", "delta"]:
                    from lerobot.utils.constants import OBS_STATE
                    if OBS_STATE in latest_obs:
                        anchor_now = latest_obs[OBS_STATE]
                
                if anchor_now is not None and action_encoding in ["anchor", "delta"]:
                    # processed_actions = unnorm(d_norm) + s_now
                    # Fix broadcasting bug: anchor_now might have a batch dimension [1, 6] from env_processor
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
                    
                    if action_encoding == "anchor":
                        processed_actions = unnormalized_actions + anchor_sq.to(unnormalized_actions.device)[None, :]
                    elif action_encoding == "delta":
                        processed_actions = torch.cumsum(unnormalized_actions, dim=0) + anchor_sq.to(unnormalized_actions.device)[None, :]
                else:
                    processed_actions = unnormalized_actions
                
                # --- Apply Centered Moving Average (Window Size 3) ---
                # The policy actually outputs a full chunk at once here before passing
                # to the queue, so implementing a centered moving average is trivial.
                if processed_actions.shape[0] >= 3:
                    # Pad the start and end to maintain sequence length
                    padded = torch.cat([processed_actions[0:1], processed_actions, processed_actions[-1:]], dim=0)
                    # Compute mean over the window of 3
                    smoothed = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
                    processed_actions = smoothed
                # -----------------------------------------------------
                
                # Plot the chunks to inspect for native jerkiness with visual context
                if not hasattr(policy, '_chunk_plot_counter'):
                    policy._chunk_plot_counter = 0

                if False:
                    try:
                        import os
                        import numpy as np
                        
                        acts_np = original_actions.cpu().to(torch.float32).numpy()
                        n_joints = min(6, acts_np.shape[1])
                        
                        os.makedirs("outputs/plots", exist_ok=True)
                        os.makedirs("outputs/plots/trajectories", exist_ok=True)
                        os.makedirs("outputs/plots/images", exist_ok=True)
                        
                        # Save trajectories
                        npy_path = f"outputs/plots/trajectories/chunk_{policy._chunk_plot_counter:04d}.npy"
                        np.save(npy_path, acts_np[:, :n_joints])

                        # Save side camera observation
                        if 'observation.images.side' in batch_for_preprocessor:
                            img_side = batch_for_preprocessor['observation.images.side'].squeeze(0).cpu().to(torch.float32).numpy()
                            # Convert CHW to HWC
                            if img_side.shape[0] == 3 or img_side.shape[0] == 1:
                                img_side = img_side.transpose(1, 2, 0)
                            
                            img_path = f"outputs/plots/images/side_{policy._chunk_plot_counter:04d}.npy"
                            np.save(img_path, img_side)
                        
                        if policy._chunk_plot_counter % 50 == 0:
                            logger.info(f"Saved action chunk and side camera array to index {policy._chunk_plot_counter:04d}")
                            
                    except Exception as e:
                        logger.error(f"Failed to save chunk array: {e}")
                
                policy._chunk_plot_counter += 1

                policy._chunk_plot_counter += 1

            # Track literal latency
            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)

            # --- JERK-FREE START FIX ---
            # If the robot queue was starved (e.g., first chunk), it executed fewer actions than `new_delay`.
            # Discarding `new_delay` actions without the robot having moved causes the first movement to jerk.
            # We constrain the delay discarded by how many actions the environment actually consumed.
            current_index = action_queue.get_action_index()
            actions_consumed = max(0, current_index - action_index_before)
            effective_delay = min(new_delay, actions_consumed)

            # Merge to the queue
            action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=effective_delay,  # Use the effective delay to avoid throwing away unsent actions
                action_index_before_inference=action_index_before,
                anchor_state=anchor_now if getattr(cfg.policy, "action_encoding", "absolute") in ["anchor", "delta"] else None
            )

        logger.info("[GET_ACTIONS] Inference thread shutting down smoothly.")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())


def env_interaction_worker(
    online_env, 
    env_processor, 
    action_processor, 
    action_queue, 
    shared_state: SharedState, 
    teleop_device, 
    cfg,
    policy=None,
    postprocessor=None
):
    """
    Main environment thread loop.
    Strictly adheres to config FPS. Syncs teleop overrides.
    """
    from lerobot.rl.gym_manipulator import step_env_and_process_transition
    
    try:
        logger.info("[ENV] Starting environment interaction thread")
        action_interval = 1.0 / cfg.env.fps
        was_intervening = False
        
        # Wait for initial episode start
        logger.info("[ENV] Waiting for '2' on the teleop device to start episode...")
        while shared_state.running and not shared_state.episode_active:
            if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                shared_state.episode_active = True
                break
            time.sleep(0.1)

        # Extract initial state observation immediately to bootstrap the shared state
        # (Assuming the env was reset right before spawning threads)
        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()
        
        from lerobot.rl.gym_manipulator import create_transition
        transition = create_transition(observation=obs, info=info)
        transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
        transition = env_processor(transition)
        
        # Push initial to shared state
        policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(policy_fmt_obs, False)

        interaction_step = 0
        video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])

        while shared_state.running:
            # 1. Episode boundary check
            if not shared_state.episode_active:
                logger.info("[ENV] Episode ended. Press '2' on the keyboard to start the next episode...")
                while shared_state.running and not shared_state.episode_active:
                    if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                        shared_state.episode_active = True
                        break
                    time.sleep(0.1)

                if not shared_state.running:
                    break
                
                logger.info("[ENV] Starting next episode.")
                
                obs, info = online_env.reset()
                env_processor.reset()
                action_processor.reset()
                
                with action_queue.lock:
                    action_queue.queue = None
                    action_queue.original_queue = None
                    action_queue.last_index = 0
                
                shared_state.request_reset()
                was_intervening = False

                transition = create_transition(observation=obs, info=info)
                transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
                transition = env_processor(transition)
                
                policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(policy_fmt_obs, False)

            start_time = time.perf_counter()
            
            # --- TELEOP AND STATE OVERRIDES ---
            # If we were intervening but teleop stopped, we trigger a policy reset
            if was_intervening and not shared_state.is_intervening:
                logger.info("[ENV] Teleop disengaged, soliciting policy/queue reset")
                shared_state.request_reset()
                # Dump stale chunks manually because action_queue has no .clear()
                with action_queue.lock:
                    action_queue.queue = None
                    action_queue.original_queue = None
                    action_queue.last_index = 0
            
            was_intervening = shared_state.is_intervening

            # --- ACTION PREPARATION ---
            if was_intervening:
                if hasattr(online_env, 'get_raw_joint_positions'):
                    raw_joints = online_env.get_raw_joint_positions()
                    joint_order = [
                        'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                        'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
                    ]
                    action = torch.tensor([float(raw_joints.get(k, 0.0)) for k in joint_order], dtype=torch.float32, device=cfg.policy.device)
                else:
                    action = torch.zeros(6, dtype=torch.float32, device=cfg.policy.device)
            else:
                action = action_queue.get()
                if action is not None:
                    # Slice to strictly 6 DoF constraints for Pi05 execution
                    if action.shape[-1] > 6:
                        action = action[..., :6]
                else:
                    # Queue starvation
                    if hasattr(online_env, 'get_raw_joint_positions'):
                        raw_joints = online_env.get_raw_joint_positions()
                        joint_order = [
                            'shoulder_pan.pos',
                            'shoulder_lift.pos',
                            'elbow_flex.pos',
                            'wrist_flex.pos',
                            'wrist_roll.pos',
                            'gripper.pos',
                        ]
                        action = torch.tensor([float(raw_joints.get(k, 0.0)) for k in joint_order], dtype=torch.float32, device=cfg.policy.device)
                    else:
                        action = torch.zeros(6, dtype=torch.float32, device=cfg.policy.device)
                    logger.warning("[ENV] Action queue starved. Executing null ops.")
            
            # Save current transition before stepping for next_state later
            current_transition_data = copy.deepcopy(transition)

            # --- ENVIRONMENT STEP ---
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            
            # --- RECORDING TO BUFFER ---
            if shared_state.replay_buffer is not None:
                # Prepare data for buffer.add()
                # ReplayBuffer.add expects dicts of tensors with batch dimension
                state_dict = convert_env_obs_to_policy_format(current_transition_data[TransitionKey.OBSERVATION])
                next_state_dict = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
                
                # Ensure action is a tensor
                if not isinstance(action, torch.Tensor):
                    action_tensor = torch.tensor(action, dtype=torch.float32, device='cpu')
                else:
                    action_tensor = action.detach().cpu()
                
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)

                # Format complementary info identical to the actor to preserve teleop data in dataset
                intervention_info = current_transition_data.get(TransitionKey.INFO, {})
                current_is_intervening = intervention_info.get(TeleopEvents.IS_INTERVENTION, False)
                complementary_info = {
                    "discrete_penalty": torch.tensor(
                        [current_transition_data.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("discrete_penalty", 0.0)]
                    ),
                    TeleopEvents.IS_INTERVENTION.value: torch.tensor([float(current_is_intervening)], dtype=torch.float32),
                    "subtask_index": torch.tensor([-1], dtype=torch.long),
                }

                shared_state.replay_buffer.add(
                    state=state_dict,
                    action=action_tensor,
                    reward=float(new_transition[TransitionKey.REWARD]),
                    next_state=next_state_dict,
                    done=bool(new_transition[TransitionKey.DONE]),
                    truncated=bool(new_transition[TransitionKey.TRUNCATED]),
                    complementary_info=complementary_info,
                )

            # --- STATE PROPAGATION ---
            next_policy_fmt_obs = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])

            # --- LOGGING IMAGES ---
            if shared_state.is_logging_episode:
                # Save images as PNG based on config
                log_dir = os.path.join(cfg.output_dir, "logging_episodes", f"episode_{shared_state.episode_counter:06d}")
                os.makedirs(log_dir, exist_ok=True)
                
                for key, val in next_policy_fmt_obs.items():
                    if "image" in key:
                        cam_name = key.split('.')[-1]
                        if cam_name in video_logging_cameras:
                            # val is (1, C, H, W) or (C, H, W)
                            img_tensor = val[0] if val.ndim == 4 else val
                            
                            if img_tensor.dtype == torch.uint8:
                                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                            else:
                                v_max = img_tensor.max().item()
                                if v_max <= 5.0:
                                    img_np = (img_tensor.float().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                                else:
                                    img_np = img_tensor.float().cpu().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
                                    
                            # Save required frames for video
                            img_path = os.path.join(log_dir, f"step_{interaction_step:06d}_{cam_name}.png")
                            Image.fromarray(img_np).save(img_path)

                # Save subtask text for this step
                with shared_state.lock:
                    shared_state.subtask_texts.append(shared_state.current_subtask_text)

                # --- NEW: CRITIC VALUES SYNCHRONOUS LOGGING ---
                if policy is not None:
                    with torch.no_grad():
                        # Prepare exactly as get_actions_worker did
                        batch_for_preprocessor = {}
                        for k, v in next_policy_fmt_obs.items():
                            if k in cfg.policy.input_features:
                                batch_for_preprocessor[k] = v

                        # Inject complementary data
                        batch_for_preprocessor["robot_type"] = cfg.env.robot.type if hasattr(cfg.env, 'robot') else ""
                        task_str = cfg.policy.task
                        adv_val = torch.tensor([[cfg.policy.inference_advantage]], device=torch.device('cpu'), dtype=torch.float32)
                        
                        batch_for_preprocessor['complementary_data'] = {
                            'task': [task_str],
                            'subtask': [""],
                            'advantage': adv_val
                        }

                        if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
                            processed_batch = policy.preprocessor(batch_for_preprocessor)
                        else:
                            processed_batch = batch_for_preprocessor
                        
                        critic_output = policy.forward(processed_batch, model="critic_value")
                        if "critic_values" in critic_output:
                            critic_val = critic_output["critic_values"].mean().item()
                        elif "critic_value_mean" in critic_output:
                            critic_val = critic_output["critic_value_mean"].mean().item()
                        else:
                            logger.error("[ENV] Could not find critic values in output")
                            critic_val = 0.0
                        
                        with shared_state.lock:
                            shared_state.critic_values.append(critic_val)

            # Handle episode boundary
            if new_transition[TransitionKey.DONE] or new_transition[TransitionKey.TRUNCATED]:
                logger.info(f"[ENV] Episode {shared_state.episode_counter} finished.")
                
                # 1. Finalize logging for this episode
                if shared_state.is_logging_episode:
                    log_dir = os.path.join(cfg.output_dir, "logging_episodes", f"episode_{shared_state.episode_counter:06d}")
                    # Save critic values
                    with open(os.path.join(log_dir, "critic_values.json"), "w") as f:
                        json.dump(shared_state.critic_values, f)
                    
                    # Plot critic values
                    if len(shared_state.critic_values) > 0:
                        plt.figure(figsize=(10, 5))
                        plt.plot(shared_state.critic_values)
                        plt.title(f"Critic Values - Episode {shared_state.episode_counter}")
                        plt.xlabel("Step")
                        plt.ylabel("Min Q-Value")
                        plt.grid(True)
                        plt.savefig(os.path.join(log_dir, "critic_plot.png"))
                        plt.close()
                    
                    # Generate video with critic overlay
                    try:
                        save_video_with_critic_overlay(log_dir, shared_state.critic_values, camera_names=video_logging_cameras, subtask_texts=shared_state.subtask_texts)
                        logger.info(f"[ENV] Video generated for episode {shared_state.episode_counter}")
                    except Exception as e:
                        logger.error(f"[ENV] Failed to generate video: {e}")
                
                # 2. Increment counter and check next logging/save status
                shared_state.episode_counter += 1
                
                # Check if we should save the dataset
                episode_save_freq = getattr(cfg, "episode_save_freq", 10)
                if shared_state.episode_counter % episode_save_freq == 0:
                    logger.info(f"[ENV] Saving inference dataset at episode {shared_state.episode_counter}...")
                    dataset_root = os.path.join(cfg.output_dir, "inference_dataset")
                    import shutil
                    if os.path.exists(dataset_root):
                        shutil.rmtree(dataset_root)
                    # Important: bfloat16 to uint8 conversion is handled by LeRobotDataset if we pass right data?
                    # Actually ReplayBuffer.to_lerobot_dataset uses guess_feature_info.
                    # I'll let it use the default for now but monitor.
                    try:
                        shared_state.replay_buffer.to_lerobot_dataset(
                            repo_id="inference_recorded",
                            fps=cfg.env.fps,
                            root=dataset_root,
                            task_name=cfg.policy.task,
                        )
                    except Exception as e:
                        logger.error(f"[ENV] Failed to save lerobot dataset: {e}")
                
                # Check if next episode should be logged
                episode_logging_freq = getattr(cfg, "episode_logging_freq", 4)
                shared_state.is_logging_episode = (shared_state.episode_counter % episode_logging_freq == 0)
                shared_state.critic_values = []
                shared_state.subtask_texts = []
                
                shared_state.episode_active = False

            # Update teleop interventions
            intervention_info = new_transition[TransitionKey.INFO]
            is_intervening = intervention_info.get(TeleopEvents.IS_INTERVENTION, False)
            shared_state.set_intervention(is_intervening)

            # --- FEEDBACK LOOP ---
            if not is_intervening:
                feedback = {}
                for key, value in new_transition[TransitionKey.OBSERVATION].items():
                    if key.endswith(".pos"):
                        if isinstance(value, torch.Tensor):
                            feedback[key] = value.item()
                        else:
                            feedback[key] = float(value)
                if feedback:
                    teleop_device.send_feedback(feedback)

            # (State propagation moved up for logging)
            
            if getattr(cfg, "use_rerun", False):
                import rerun as rr
                rr.set_time_sequence("step", interaction_step)
                
                for key, val in next_policy_fmt_obs.items():
                    if "image" in key:
                        val_np = val[0].cpu().numpy() if val.ndim == 4 else val.cpu().numpy()
                        image_np = val_np.transpose(1, 2, 0)
                        rr.log(f"world/cameras/{key}", rr.Image(image_np))
                
                if hasattr(online_env, 'get_raw_joint_positions'):
                    joints = online_env.get_raw_joint_positions()
                    if joints:
                        for j_name, j_val in joints.items():
                            rr.log(f"world/robot_joints/{j_name}", rr.Scalars(float(j_val)))

            shared_state.update_observation(next_policy_fmt_obs, is_intervening)
            transition = new_transition

            # Calculate tight strict Hz sleep
            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - dt_s)
            
            # --- DEBUG LOGGING ---
            # Storing active time so the user can see exactly how long the environment took 
            if not hasattr(shared_state, 'env_active_time_total'):
                shared_state.env_active_time_total = 0.0
            shared_state.env_active_time_total += dt_s
            
            shared_state.add_env_wait_time(sleep_time)
            
            interaction_step += 1
            precise_sleep(sleep_time)

        logger.info("[ENV] Environmental loop complete.")
    except Exception as e:
        logger.error(f"[ENV] Fatal exception: {e}")
        logger.error(traceback.format_exc())

