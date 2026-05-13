import logging
import math
import threading
import time
import traceback
import torch

from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.transport.utils import bytes_to_state_dict
from lerobot.utils.transition import move_state_dict_to_device
from lerobot.rl.inference_utils import convert_env_obs_to_policy_format, apply_butterworth_filter

logger = logging.getLogger(__name__)

class SharedStateActor:
    """Thread-safe state manager for the async Actor loops."""
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_obs = None
        self.is_intervening = False
        self.episode_active = False
        self.policy_reset_requested = False
        self.update_parameters_requested = False
        self.running = True
        self.env_wait_time = 0.0
        self.env_steps = 0
        self.inference_wait_time = 0.0
        self.inference_count = 0
        self.inference_latencies = []
        self.current_step = 0
        self.cached_subtask_tokens = None  # [max_decoding_steps] long, CPU
        self.cached_subtask_masks = None   # [max_decoding_steps] bool, CPU
        self.params_loaded_event = threading.Event()
        self.params_loaded_event.set()  # Initially set so startup doesn't block if no params arrive

    def add_env_wait_time(self, wait_time: float):
        with self.lock:
            self.env_wait_time += wait_time
            self.env_steps += 1

    def add_inference_wait_time(self, wait_time: float):
        with self.lock:
            self.inference_wait_time += wait_time

    def add_inference_latency(self, latency: float):
        with self.lock:
            self.inference_count += 1
            self.inference_latencies.append(latency)

    def get_and_reset_metrics(self):
        with self.lock:
            inf_lats = list(self.inference_latencies)
            metrics = {
                'env_wait_time': self.env_wait_time,
                'env_steps': self.env_steps,
                'inference_wait_time': self.inference_wait_time,
                'env_active_time': getattr(self, 'env_active_time_total', 0.0),
                'inference_count': self.inference_count,
                'inference_latencies': inf_lats,
            }
            self.env_wait_time = 0.0
            self.env_steps = 0
            self.inference_wait_time = 0.0
            self.env_active_time_total = 0.0
            self.inference_count = 0
            self.inference_latencies = []
            return metrics

    def update_observation(self, obs: dict, is_intervening: bool):
        with self.lock:
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

    def set_episode_active(self, status: bool):
        with self.lock:
            self.episode_active = status

    def request_parameter_update(self):
        with self.lock:
            self.update_parameters_requested = True
        self.params_loaded_event.clear()

    def check_and_clear_parameter_update(self) -> bool:
        with self.lock:
            if self.update_parameters_requested:
                self.update_parameters_requested = False
                return True
            return False

    def update_subtask_cache(self, tokens: torch.Tensor, masks: torch.Tensor):
        with self.lock:
            self.cached_subtask_tokens = tokens.clone()
            self.cached_subtask_masks = masks.clone()

def pull_new_policy_weights(policy, parameters_queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logger.info("[ACTOR_ASYNC] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        # For Pi05, the actor is the model
        if hasattr(policy, "actor"):
             policy.actor.load_state_dict(actor_state_dict, strict=False)
        else:
             policy.model.load_state_dict(actor_state_dict, strict=False)


def align_prev_actions(
    prev_actions: torch.Tensor,
    anchor_old: torch.Tensor,
    anchor_now: torch.Tensor,
    action_encoding: str,
    chunk_size: int,
    postprocessor,
    normalizer,
) -> torch.Tensor:
    """Re-align leftover chunk actions when the anchor state changes between chunks.

    For anchor encoding (d_t = a_t - s_0), every action in the leftover references s_0
    and must be corrected when s_0 changes: d_t_new = d_t_old + (s_0_old - s_0_new).

    For delta encoding (d_0 = a_0 - s_0, d_t = a_t - a_{t-1} for t > 0), only d_0
    references s_0. Leftovers that begin at offset > 0 contain only consecutive diffs
    and need no correction at all.

    Per-timestep normalization stats have shape [chunk_size, action_dim]. The leftover
    starts at index offset = chunk_size - n_left in the original chunk, so we right-align
    it in a padded buffer before calling postprocessor (unnorm) to ensure each position
    uses the correct per-timestep stats. After the correction we left-align for renorm,
    because the model receives prev_chunk_left_over left-aligned (positions 0..n_left-1).

    Args:
        prev_actions: Leftover normalized actions from the previous chunk, [n_left, action_dim].
        anchor_old: s_0 that was used when the previous chunk was generated.
        anchor_now: Current observation state (new s_0).
        action_encoding: "anchor" or "delta".
        chunk_size: Full chunk size (used to derive offset).
        postprocessor: Callable that unnormalizes a [chunk_size, action_dim] tensor.
        normalizer: NormalizerProcessorStep used to renormalize after correction.

    Returns:
        Re-aligned normalized prev_actions with shape [n_left, action_dim].
    """
    n_left = prev_actions.shape[0]
    action_dim = prev_actions.shape[1]
    offset = chunk_size - n_left  # position of prev_actions[0] in the original chunk

    if action_encoding == "delta" and offset > 0:
        # prev_actions[0] is d_{offset} = a_{offset} - a_{offset-1}, a consecutive diff
        # that does not reference s_0 at all.  No alignment needed.
        return prev_actions

    # Right-align in a padded buffer so postprocessor applies the correct per-timestep
    # stats: stats[offset+i] is used to unnorm prev_actions[i].
    right_padded = torch.zeros(chunk_size, action_dim, device=prev_actions.device, dtype=prev_actions.dtype)
    right_padded[offset:] = prev_actions
    d_abs = postprocessor(right_padded)

    dev = d_abs.device
    delta_s = anchor_old.squeeze(0).to(dev) - anchor_now.squeeze(0).to(dev)
    if action_encoding == "anchor":
        # All positions reference s_0; shift every leftover element.
        d_abs[offset:] += delta_s
    else:
        # delta with offset == 0: only d_0 = a_0 - s_0 references the anchor.
        d_abs[0] += delta_s

    # Left-align for renorm: the model receives prev_chunk_left_over at positions 0..n_left-1.
    left_padded = torch.zeros(chunk_size, action_dim, device=dev, dtype=d_abs.dtype)
    left_padded[:n_left] = d_abs[offset:]
    return normalizer._normalize_action(left_padded, inverse=False)[:n_left]


def get_actions_worker_actor(policy, shared_state: SharedStateActor, action_queue, parameters_queue, device, cfg):
    """
    Background inference thread wrapper for the Actor.
    Continuously pulls inputs and runs predictions, but respects episode boundaries.
    """
    try:
        logger.info("[GET_ACTIONS] Starting async actor inference thread")
        latency_tracker = LatencyTracker()
        
        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk = 1.0 / cfg.env.fps

        task_str = cfg.policy.task
        advantage_val = torch.tensor([[cfg.policy.inference_advantage]], device=torch.device('cpu'), dtype=torch.float32)

        while shared_state.running:
            # 1. Update Weights
            if shared_state.check_and_clear_parameter_update():
                pull_new_policy_weights(policy, parameters_queue, device)
                shared_state.params_loaded_event.set()

            # 2. Halt if episode is not active
            if not shared_state.episode_active:
                time.sleep(0.01)
                continue

            # 3. Check for teleop reset
            if shared_state.check_and_clear_reset():
                policy.reset()
                continue
            
            # 3.5. Halt inference if human is intervening
            if shared_state.is_intervening:
                time.sleep(0.01)
                continue

            # 4. Check if we need a new chunk (use p95 for less pessimistic threshold)
            current_delay = math.ceil(latency_tracker.p95() / time_per_chunk)
            if not action_queue.empty() and action_queue.qsize() > execution_horizon + current_delay:
                wait_start = time.perf_counter()
                time.sleep(0.01)
                shared_state.add_inference_wait_time(time.perf_counter() - wait_start)
                continue

            # 5. Fetch latest environment observation
            latest_obs = shared_state.get_latest_observation()
            if latest_obs is None:
                time.sleep(0.01)
                continue
            
            # 6. Filter features & format
            batch_for_preprocessor = {}
            for k, v in latest_obs.items():
                if k in cfg.policy.input_features:
                    batch_for_preprocessor[k] = v

            # 7. Inject specific complementary data needed
            batch_for_preprocessor["robot_type"] = cfg.env.robot.type if hasattr(cfg.env, 'robot') else ""
            batch_for_preprocessor['complementary_data'] = {
                'task': [task_str],
                'subtask': [""],
                'advantage': advantage_val
            }

            current_time = time.perf_counter()
            
            # 8. Execute model inference using preprocessor and postprocessor
            with torch.no_grad():
                t_preproc_start = time.perf_counter()
                if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
                    processed_batch = policy.preprocessor(batch_for_preprocessor)
                else:
                    processed_batch = batch_for_preprocessor
                t_preproc_end = time.perf_counter()

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                # --- Anchor/Delta Alignment for RTC impainting ---
                action_encoding = getattr(policy.config, "action_encoding", "absolute")
                anchor_now = None
                if action_encoding in ["anchor", "delta"]:
                    from lerobot.utils.constants import OBS_STATE
                    if OBS_STATE in latest_obs:
                        anchor_now = latest_obs[OBS_STATE]
                        if prev_actions is not None and action_queue.anchor_state is not None:
                            anchor_old = action_queue.anchor_state
                            from lerobot.processor import NormalizerProcessorStep
                            normalizer = next(s for s in policy.preprocessor.steps if isinstance(s, NormalizerProcessorStep))
                            logger.debug(f"[RTC] Alignment Offset (delta_s): {(anchor_old - anchor_now).norm().item():.3f}")
                            prev_actions = align_prev_actions(
                                prev_actions=prev_actions,
                                anchor_old=anchor_old,
                                anchor_now=anchor_now,
                                action_encoding=action_encoding,
                                chunk_size=policy.config.chunk_size,
                                postprocessor=policy.postprocessor,
                                normalizer=normalizer,
                            )

                # Using p95 instead of max: max() is overly conservative
                inference_delay = math.ceil(latency_tracker.p95() / time_per_chunk)

                torch.cuda.synchronize()
                t_gpu_start = time.perf_counter()
                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon
                )
                torch.cuda.synchronize()
                t_gpu_end = time.perf_counter()

                if policy._cached_subtask_tokens is not None:
                    shared_state.update_subtask_cache(
                        policy._cached_subtask_tokens[0].cpu(),
                        policy._cached_subtask_masks[0].cpu(),
                    )

                # actions_chunk is [1, T, D] normalized
                original_actions = actions_chunk.squeeze(0).clone()
                
                # --- [NEW] Absolute Action Reconstruction ---
                if anchor_now is not None and action_encoding in ["anchor", "delta"]:
                    # processed_actions = unnorm(d_norm) + s_now
                    d_abs = policy.postprocessor(original_actions)
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
                    
                    if action_encoding == "anchor":
                        processed_actions = d_abs + anchor_sq.to(d_abs.device)[None, :]
                    elif action_encoding == "delta":
                        processed_actions = torch.cumsum(d_abs, dim=0) + anchor_sq.to(d_abs.device)[None, :]
                else:
                    processed_actions = original_actions.clone()
                
                # --- Zero-phase Butterworth low-pass filter ---
                processed_actions = apply_butterworth_filter(processed_actions)
                
            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)
            shared_state.add_inference_latency(new_latency)
            
            dt_preproc = t_preproc_end - t_preproc_start
            dt_gpu = t_gpu_end - t_gpu_start
            dt_post = new_latency - dt_preproc - dt_gpu
            logger.debug(
                f"[INFERENCE] chunk latency: {new_latency:.3f}s  "
                f"[preproc={dt_preproc:.3f}s  gpu={dt_gpu:.3f}s  post={dt_post:.3f}s]  "
                f"delay={new_delay}  model_delay={inference_delay}  qsize={action_queue.qsize()}"
            )

            # --- JERK-FREE START FIX ---
            current_index = action_queue.get_action_index()
            actions_consumed = max(0, current_index - action_index_before)
            effective_delay = min(new_delay, actions_consumed)

            # Prevent race condition: if env thread requested a reset while we were inferencing,
            # this chunk was generated with an old GRU state. Drop it and restart the loop!
            if shared_state.policy_reset_requested:
                continue

            # Merge to the queue
            action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=effective_delay, 
                action_index_before_inference=action_index_before,
                anchor_state=anchor_now if getattr(cfg.policy, "action_encoding", "absolute") in ["anchor", "delta"] else None
            )

        logger.info("[GET_ACTIONS] Inference thread shutting down smoothly.")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())

def env_interaction_worker_actor(
    online_env, 
    env_processor, 
    action_processor, 
    action_queue, 
    shared_state: SharedStateActor, 
    teleop_device, 
    transitions_queue,
    interactions_queue,
    cfg,
    postprocessor=None
):
    """
    Main environment thread loop for the Actor.
    Manages episodes and gathers offline transitions.
    """
    from lerobot.rl.gym_manipulator import step_env_and_process_transition, create_transition
    from lerobot.utils.transition import Transition, move_transition_to_device
    from lerobot.transport.utils import python_object_to_bytes
    from lerobot.rl.actor import push_transitions_to_transport_queue
    from lerobot.rl.inference_utils import convert_env_obs_to_policy_format
    
    try:
        logger.info("[ENV] Starting async environment interaction thread")
        action_interval = 1.0 / cfg.env.fps
        was_intervening = False
        
        sum_reward_episode = 0.0
        episode_intervention_steps = 0
        episode_total_steps = 0
        list_transition_to_send_to_learner = []

        interaction_step = 0

        # Wait for initial episode start
        logger.info("[ACTOR] Waiting for '2' on the teleop device to start episode...")
        while shared_state.running:
            if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                break
            time.sleep(0.1)

        # Initial bootstrap
        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()

        transition = create_transition(observation=obs, info=info)
        transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
        transition = env_processor(transition)

        policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(policy_fmt_obs, False)
        shared_state.request_parameter_update()

        # Wait for params to load before starting the episode
        logger.info("[ACTOR] Loading new params, please wait before episode starts...")
        while shared_state.running:
            if shared_state.params_loaded_event.wait(timeout=0.5):
                break
        logger.info("[ACTOR] Params loaded. Starting episode.")
        shared_state.set_episode_active(True)

        while shared_state.running:
            # 1. Episode boundary check
            if not shared_state.episode_active:
                logger.info("[ACTOR] Episode ended. Press '2' on the keyboard to start the next episode...")
                while shared_state.running:
                    if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                        break
                    time.sleep(0.1)

                if not shared_state.running:
                    break

                logger.info("[ACTOR] Starting next episode.")

                if getattr(cfg, "use_rerun", False):
                    import rerun as rr
                    rr.log("/", rr.Clear(recursive=True))

                obs, info = online_env.reset()
                env_processor.reset()
                action_processor.reset()

                with action_queue.lock:
                    action_queue.queue = None
                    action_queue.original_queue = None
                    action_queue.last_index = 0

                shared_state.request_reset()
                shared_state.request_parameter_update()
                was_intervening = False

                transition = create_transition(observation=obs, info=info)
                transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
                transition = env_processor(transition)

                policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(policy_fmt_obs, False)

                # Wait for params to load before starting the episode
                logger.info("[ACTOR] Loading new params, please wait before episode starts...")
                while shared_state.running:
                    if shared_state.params_loaded_event.wait(timeout=0.5):
                        break
                logger.info("[ACTOR] Params loaded. Starting episode.")
                shared_state.set_episode_active(True)

            start_time = time.perf_counter()
            
            # --- TELEOP AND STATE OVERRIDES ---
            if was_intervening != shared_state.is_intervening:
                logger.info(f"[ENV] Teleop state changed (intervening: {shared_state.is_intervening}), soliciting policy/queue reset")
                shared_state.request_reset()
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
                    action_dim = cfg.policy.action_dim
                    if action.shape[-1] > action_dim:
                        action = action[..., :action_dim]
                    # IF recursive deltas are used, the queue ALREADY contains unnormalized absolute actions.
                    # Otherwise, we unnormalize here as usual.
                    if postprocessor is not None and getattr(cfg.policy, "action_encoding", "absolute") == "absolute":
                        action = postprocessor(action)
                else:
                    if hasattr(online_env, 'get_raw_joint_positions'):
                        raw_joints = online_env.get_raw_joint_positions()
                        joint_order = [
                            'shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
                            'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos',
                        ]
                        action = torch.tensor([float(raw_joints.get(k, 0.0)) for k in joint_order], dtype=torch.float32, device=cfg.policy.device)
                    else:
                        action = torch.zeros(6, dtype=torch.float32, device=cfg.policy.device)

            # --- ENVIRONMENT STEP ---
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            if TransitionKey.COMPLEMENTARY_DATA not in new_transition:
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = {}
            if "subtask" not in new_transition[TransitionKey.COMPLEMENTARY_DATA]:
                 new_transition[TransitionKey.COMPLEMENTARY_DATA]["subtask"] = [""] * (len(new_transition[TransitionKey.OBSERVATION]) if isinstance(new_transition[TransitionKey.OBSERVATION], list) else 1)
            
            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]
            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)
            
            sum_reward_episode += float(reward)
            episode_total_steps += 1

            if reward > 0:
                logger.info(f"[ACTOR] Received transition with reward: {reward}")

            intervention_info = new_transition[TransitionKey.INFO]
            is_intervening = intervention_info.get(TeleopEvents.IS_INTERVENTION, False)
            shared_state.set_intervention(is_intervening)

            # --- DEBUG: Track episode end signal propagation ---
            info_success = intervention_info.get(TeleopEvents.SUCCESS, False)
            info_terminate = intervention_info.get(TeleopEvents.TERMINATE_EPISODE, False)
            if done or truncated or info_success or info_terminate:
                logger.info(
                    f"[ACTOR EPISODE_END_DEBUG] step={episode_total_steps} | "
                    f"done={done} truncated={truncated} | "
                    f"info.SUCCESS={info_success} info.TERMINATE={info_terminate} | "
                    f"reward={reward}"
                )

            if is_intervening:
                episode_intervention_steps += 1
            else:
                feedback = {}
                for key, value in new_transition[TransitionKey.OBSERVATION].items():
                    if key.endswith(".pos"):
                        if isinstance(value, torch.Tensor):
                            feedback[key] = value.item()
                        else:
                            feedback[key] = float(value)
                if feedback:
                    teleop_device.send_feedback(feedback)

            # --- TRANSITION FORMATTING ---
            with shared_state.lock:
                cached_tokens = shared_state.cached_subtask_tokens
                cached_masks = shared_state.cached_subtask_masks
                if cached_tokens is not None:
                    subtask_tokens_for_transition = cached_tokens.clone()
                    subtask_masks_for_transition = cached_masks.clone()
                else:
                    max_len = cfg.policy.max_decoding_steps
                    subtask_tokens_for_transition = torch.zeros(max_len, dtype=torch.long)
                    subtask_masks_for_transition = torch.zeros(max_len, dtype=torch.bool)

            complementary_info = {
                "discrete_penalty": torch.tensor(
                    [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
                ),
                TeleopEvents.IS_INTERVENTION.value: torch.tensor([float(is_intervening)], dtype=torch.float32),
                "subtask_index": torch.tensor([-1], dtype=torch.long),
                "subtask_tokens": subtask_tokens_for_transition,
                "subtask_masks": subtask_masks_for_transition,
            }

            observation = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
            next_observation = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])

            transition_to_send = Transition(
                state=observation,
                action=executed_action[:cfg.policy.action_dim],
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            
            transition_to_send = move_transition_to_device(transition_to_send, "cpu")
            list_transition_to_send_to_learner.append(transition_to_send)

            next_policy_fmt_obs = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])

            if getattr(cfg, "use_rerun", False):
                import rerun as rr
                rr.set_time_sequence("step", interaction_step)
                
                # Log the pre-processed low-res tensor images from observation
                for key, val in next_policy_fmt_obs.items():
                    if "image" in key:
                        val_np = val[0].cpu().numpy() if val.ndim == 4 else val.cpu().numpy()
                        image_np = val_np.transpose(1, 2, 0)
                        rr.log(f"world/cameras/{key}", rr.Image(image_np))
                
                # Keep tracking actual joint state so we understand physical robot state
                if hasattr(online_env, 'get_raw_joint_positions'):
                    joints = online_env.get_raw_joint_positions()
                    if joints:
                        for j_name, j_val in joints.items():
                            rr.log(f"world/robot_joints/{j_name}", rr.Scalars(float(j_val)))

            shared_state.update_observation(next_policy_fmt_obs, is_intervening)
            transition = new_transition
            
            interaction_step += 1
            shared_state.current_step = interaction_step

            # --- EPISODE END HANDLING ---
            if done or truncated:
                logger.info(
                    f"[ACTOR] Global step {interaction_step}: Episode ended. "
                    f"reward={sum_reward_episode} | done={done} truncated={truncated} | "
                    f"info.SUCCESS={info_success} info.TERMINATE={info_terminate}"
                )
                shared_state.set_episode_active(False)

                if len(list_transition_to_send_to_learner) > 0:
                    push_transitions_to_transport_queue(
                        transitions=list_transition_to_send_to_learner,
                        transitions_queue=transitions_queue,
                    )
                    list_transition_to_send_to_learner = []

                intervention_rate = 0.0
                if episode_total_steps > 0:
                    intervention_rate = episode_intervention_steps / episode_total_steps

                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention_steps > 0),
                            "Intervention rate": intervention_rate,
                        }
                    )
                )

                sum_reward_episode = 0.0
                episode_intervention_steps = 0
                episode_total_steps = 0

            # Calculate tight strict Hz sleep
            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - dt_s)
            
            if not hasattr(shared_state, 'env_active_time_total'):
                shared_state.env_active_time_total = 0.0
            shared_state.env_active_time_total += dt_s
            shared_state.add_env_wait_time(sleep_time)
            
            from lerobot.utils.robot_utils import precise_sleep
            precise_sleep(sleep_time)

        logger.info("[ENV] Environmental loop complete.")
    except Exception as e:
        logger.error(f"[ENV] Fatal exception: {e}")
        logger.error(traceback.format_exc())
