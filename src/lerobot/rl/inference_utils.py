import logging
import math
import threading
import time
import copy
import traceback

import torch
from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.policies.rtc.latency_tracker import LatencyTracker

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
                continue
            
            # 2. Check if we actually need a new chunk
            # To avoid over-inferencing and queue saturation, only infer if we're getting close
            # In a real async inference loop we fetch if queue <= execution_horizon + delay
            current_delay = math.ceil(latency_tracker.max() / time_per_chunk)
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

            current_time = time.perf_counter()
            
            # 6. Execute model inference using preprocessor and postprocessor
            with torch.no_grad():
                if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
                    processed_batch = policy.preprocessor(batch_for_preprocessor)
                else:
                    processed_batch = batch_for_preprocessor

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                inference_delay = math.ceil(latency_tracker.max() / time_per_chunk)

                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon
                )

                original_actions = actions_chunk.squeeze(0).clone()
                
                # --- Apply Centered Moving Average (Window Size 3) ---
                # The policy actually outputs a full chunk at once here before passing
                # to the queue, so implementing a centered moving average is trivial.
                if original_actions.shape[0] >= 3:
                    # Pad the start and end to maintain sequence length
                    padded = torch.cat([original_actions[0:1], original_actions, original_actions[-1:]], dim=0)
                    # Compute mean over the window of 3
                    smoothed = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
                    original_actions = smoothed
                # -----------------------------------------------------
                
                # Plot the chunks to inspect for native jerkiness with visual context
                if not hasattr(policy, '_chunk_plot_counter'):
                    policy._chunk_plot_counter = 0

                if True:
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

                # In actor_pi05.py, actions are pushed raw into the queue and only postprocessed later.
                # So we simply merge original_actions twice (or use it for both original and processed args) 
                processed_actions = original_actions.clone()

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
                action_index_before_inference=action_index_before
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

        while shared_state.running:
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
                    
                    if postprocessor is not None:
                        action = postprocessor(action)
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

            # --- ENVIRONMENT STEP ---
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            
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

            # --- STATE PROPAGATION ---
            next_policy_fmt_obs = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
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
            
            time.sleep(sleep_time)

        logger.info("[ENV] Environmental loop complete.")
    except Exception as e:
        logger.error(f"[ENV] Fatal exception: {e}")
        logger.error(traceback.format_exc())
