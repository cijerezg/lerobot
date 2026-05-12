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
from tqdm import tqdm
from scipy.signal import butter, filtfilt

import torch
from lerobot.utils.robot_utils import precise_sleep
from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.rl.utils import save_video_with_critic_overlay

logger = logging.getLogger(__name__)

_BUTTER_B, _BUTTER_A = butter(N=2, Wn=0.2, btype='low')


def apply_butterworth_filter(actions: torch.Tensor) -> torch.Tensor:
    """Zero-phase low-pass Butterworth filter along the time axis of an [T, D]
    action chunk. Returns input unchanged when T is too short for filtfilt's
    default padlen (3 * max(len(a), len(b)) = 9)."""
    if actions.shape[0] <= 9:
        return actions
    arr = actions.detach().to(torch.float32).cpu().numpy()
    smoothed = filtfilt(_BUTTER_B, _BUTTER_A, arr, axis=0)
    return torch.as_tensor(smoothed.copy(), dtype=actions.dtype, device=actions.device)

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


# ---------------------------------------------------------------------------
# Interactive shared state — extends SharedState with a pending subtask slot
# ---------------------------------------------------------------------------

class SharedStateInteractive(SharedState):
    """SharedState + a one-shot subtask override slot for interactive inference."""

    def __init__(self):
        super().__init__()
        self._pending_tokens = None   # torch.Tensor [max_len] long, or None
        self._pending_masks  = None   # torch.Tensor [max_len] bool, or None
        self._pending_text   = None   # str, for logging only

    def set_pending_override(self, tokens, masks, text: str):
        """Called by the terminal input thread."""
        with self.lock:
            self._pending_tokens = tokens
            self._pending_masks  = masks
            self._pending_text   = text

    def pop_pending_override(self):
        """Called by the inference worker. Returns (tokens, masks, text) and clears, or None."""
        with self.lock:
            if self._pending_tokens is None:
                return None
            result = (self._pending_tokens, self._pending_masks, self._pending_text)
            self._pending_tokens = None
            self._pending_masks  = None
            self._pending_text   = None
            return result

    def clear_pending_override(self):
        """Called at episode boundary so a stale override doesn't survive a reset."""
        with self.lock:
            self._pending_tokens = None
            self._pending_masks  = None
            self._pending_text   = None


# ---------------------------------------------------------------------------
# Terminal input thread (interactive mode only)
# ---------------------------------------------------------------------------

def terminal_input_worker(shared_state: SharedStateInteractive, policy, cfg, shutdown_event):
    """
    Daemon thread. Blocks on input(), tokenizes the text, and queues the override
    into shared_state. Log output from other threads will interleave with the
    prompt — that is expected.
    """
    tokenizer = policy.model._paligemma_tokenizer
    max_len   = cfg.policy.tokenizer_max_length

    interval = getattr(cfg.policy, "subtask_regeneration_interval", -1)
    if interval <= 0:
        logger.warning(
            "[INTERACTIVE] subtask_regeneration_interval is <= 0. "
            "Injected subtasks will be IMMEDIATELY overwritten by generate_subtask_tokens "
            "on the very next inference cycle. Set it to a positive value (e.g. 30) in your config."
        )

    print("[INTERACTIVE] Type a subtask and press Enter at any time. Ctrl+C to stop.")

    while not shutdown_event.is_set():
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            continue

        encoding = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoding["input_ids"][0]              # [max_len] long
        masks  = encoding["attention_mask"][0].bool()  # [max_len] bool

        shared_state.set_pending_override(tokens, masks, text)
        print(f"[INTERACTIVE] Queued: '{text}' — will be used at next action generation")


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
    Background inference thread. Continually generates action chunks via RTC.

    Works with both SharedState (plain) and SharedStateInteractive. When the
    shared_state has a pop_pending_override method, subtask overrides typed in
    the terminal are injected into the policy's token cache before each call to
    predict_action_chunk.
    """
    try:
        logger.info("[GET_ACTIONS] Starting background inference thread")
        latency_tracker  = LatencyTracker()
        last_subtask_text = None
        inference_step    = 0

        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk    = 1.0 / cfg.env.fps
        task_str          = cfg.policy.task
        advantage_val     = torch.tensor([[cfg.policy.inference_advantage]], device=torch.device('cpu'), dtype=torch.float32)
        device            = next(policy.parameters()).device

        # Resolve interactive helpers once — None when running in non-interactive mode
        pop_override   = getattr(shared_state, 'pop_pending_override',   None)
        clear_override = getattr(shared_state, 'clear_pending_override', None)

        while shared_state.running:
            # 1. Reset check
            if shared_state.check_and_clear_reset():
                policy.reset()
                if clear_override is not None:
                    clear_override()
                last_subtask_text = None
                continue

            if not shared_state.episode_active:
                time.sleep(0.01)
                continue

            # 2. Queue saturation check (use p95 for less pessimistic threshold)
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
            batch_for_preprocessor = {k: v for k, v in latest_obs.items() if k in cfg.policy.input_features}

            # 5. Complementary data required by Pi05 preprocessor
            batch_for_preprocessor["robot_type"] = cfg.env.robot.type if hasattr(cfg.env, 'robot') else ""
            batch_for_preprocessor['complementary_data'] = {
                'task': [task_str],
                'subtask': [""],
                'advantage': advantage_val,
            }

            current_time = time.perf_counter()

            with torch.no_grad():
                if hasattr(policy, 'preprocessor') and policy.preprocessor is not None:
                    processed_batch = policy.preprocessor(batch_for_preprocessor)
                else:
                    processed_batch = batch_for_preprocessor

                action_index_before = action_queue.get_action_index()
                prev_actions        = action_queue.get_left_over()

                # --- Anchor/Delta alignment for RTC inpainting ---
                action_encoding = getattr(policy.config, "action_encoding", "absolute")
                anchor_now = None
                if action_encoding in ["anchor", "delta"]:
                    from lerobot.utils.constants import OBS_STATE
                    if OBS_STATE in latest_obs:
                        anchor_now = latest_obs[OBS_STATE]
                        if prev_actions is not None and action_queue.anchor_state is not None:
                            anchor_old = action_queue.anchor_state
                            from lerobot.processor import NormalizerProcessorStep
                            from lerobot.rl.actor_pi05_async_utils import align_prev_actions
                            normalizer = next(
                                s for s in policy.preprocessor.steps
                                if isinstance(s, NormalizerProcessorStep)
                            )
                            logger.debug(
                                f"[RTC] Alignment offset norm: {(anchor_old - anchor_now).norm().item():.4f}"
                            )
                            prev_actions = align_prev_actions(
                                prev_actions=prev_actions,
                                anchor_old=anchor_old,
                                anchor_now=anchor_now,
                                action_encoding=action_encoding,
                                chunk_size=policy.config.chunk_size,
                                postprocessor=policy.postprocessor,
                                normalizer=normalizer,
                            )

                # --- Subtask override injection (interactive mode only) ---
                if pop_override is not None:
                    override = pop_override()
                    if override is not None:
                        tokens, masks, text = override
                        interval = getattr(policy.config, "subtask_regeneration_interval", -1)
                        if interval <= 0:
                            logger.warning(
                                f"[INTERACTIVE] Injecting subtask '{text}' but "
                                "subtask_regeneration_interval <= 0 — it will be overwritten immediately."
                            )
                        policy._cached_subtask_tokens = tokens.unsqueeze(0).to(device)
                        policy._cached_subtask_masks  = masks.unsqueeze(0).to(device)
                        policy._last_subtask_time     = time.time()
                        logger.info(f"[INTERACTIVE] Injecting subtask override: '{text}'")

                # p95 latency avoids a single spike biasing the model to look too far ahead
                inference_delay = math.ceil(latency_tracker.p95() / time_per_chunk)

                _t_pac0 = time.perf_counter()
                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon,
                )
                _t_pac1 = time.perf_counter()
                print(f"[TIMING] predict_action_chunk={(_t_pac1-_t_pac0)*1000:.1f}ms", flush=True)

                # --- Subtask token decoding (for logging) ---
                inference_step += 1
                try:
                    cached_tokens = getattr(policy, '_cached_subtask_tokens', None)
                    cached_masks  = getattr(policy, '_cached_subtask_masks',  None)
                    if cached_tokens is not None and cached_masks is not None:
                        tokenizer    = policy.model._paligemma_tokenizer
                        valid_tokens = cached_tokens[0][cached_masks[0]]
                        subtask_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                        with shared_state.lock:
                            shared_state.current_subtask_text = subtask_text
                        if subtask_text != last_subtask_text:
                            logger.info(f"[SUBTASK] inference_step={inference_step} | subtask: \"{subtask_text}\"")
                            last_subtask_text = subtask_text
                except Exception as e:
                    logger.debug(f"[SUBTASK] Could not decode subtask tokens: {e}")

                original_actions = actions_chunk.squeeze(0).clone()

                # --- Unnormalization & absolute action reconstruction ---
                unnormalized_actions = (
                    policy.postprocessor(original_actions)
                    if hasattr(policy, 'postprocessor') and policy.postprocessor is not None
                    else original_actions.clone()
                )

                if anchor_now is not None and action_encoding in ["anchor", "delta"]:
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
                    if action_encoding == "anchor":
                        processed_actions = unnormalized_actions + anchor_sq.to(unnormalized_actions.device)[None, :]
                    elif action_encoding == "delta":
                        processed_actions = torch.cumsum(unnormalized_actions, dim=0) + anchor_sq.to(unnormalized_actions.device)[None, :]
                else:
                    processed_actions = unnormalized_actions

                # --- Zero-phase Butterworth low-pass filter ---
                processed_actions = apply_butterworth_filter(processed_actions)

                if not hasattr(policy, '_chunk_plot_counter'):
                    policy._chunk_plot_counter = 0
                policy._chunk_plot_counter += 1

            # Track latency
            new_latency = time.perf_counter() - current_time
            new_delay   = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)

            # Constrain discarded delay by how many actions the env actually consumed
            # (avoids jerky start when the queue was starved on the first chunk)
            current_index    = action_queue.get_action_index()
            actions_consumed = max(0, current_index - action_index_before)
            effective_delay  = min(new_delay, actions_consumed)

            action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=effective_delay,
                action_index_before_inference=action_index_before,
                anchor_state=anchor_now if action_encoding in ["anchor", "delta"] else None,
            )

        logger.info("[GET_ACTIONS] Inference thread shutting down smoothly.")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())


def _finalize_episode_log(
    episode_log_buffer,
    policy,
    cfg,
    log_dir,
    episode_counter,
    video_logging_cameras,
    critic_batch_size=40,
    critic_subsample=1,
):
    """
    Process a buffered episode's frames after the episode ends.
    Called during the natural pause between episodes — no env loop impact.
    Runs critic inference, saves PNGs, and generates the overlay video,
    all behind tqdm progress bars.

    critic_subsample: stride at which the critic forward is run relative to
    video frames. The video uses every saved frame; only the V(s) curve has
    fewer samples. Defaults to 1 (legacy 1:1 behavior).
    """
    if not episode_log_buffer:
        return

    os.makedirs(log_dir, exist_ok=True)
    critic_subsample = max(1, int(critic_subsample))
    n_steps = len(episode_log_buffer)
    subtask_texts = [frame['subtask_text'] for frame in episode_log_buffer]

    # --- 1. Save PNGs ---
    logger.info(f"[ENV] Saving {n_steps} frames for episode {episode_counter}...")
    for step_idx, frame in enumerate(tqdm(episode_log_buffer, desc="Saving frames", unit="frame")):
        for key, val in frame['obs'].items():
            if "image" in key:
                cam_name = key.split('.')[-1]
                if cam_name in video_logging_cameras:
                    img_tensor = val[0] if val.ndim == 4 else val
                    if img_tensor.dtype == torch.uint8:
                        img_np = img_tensor.numpy().transpose(1, 2, 0)
                    else:
                        v_max = img_tensor.max().item()
                        # Heuristic: if max value is small, assume [0,1] normalized float; otherwise assume [0,255]
                        if v_max <= 5.0:
                            img_np = (img_tensor.float().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            img_np = img_tensor.float().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
                    img_path = os.path.join(log_dir, f"step_{step_idx:06d}_{cam_name}.png")
                    Image.fromarray(img_np).save(img_path)

    # --- 2. Batched critic inference ---
    critic_values = []
    if policy is not None:
        task_str = cfg.policy.task
        adv_val = torch.tensor([[cfg.policy.inference_advantage]], device=torch.device('cpu'), dtype=torch.float32)
        robot_type = cfg.env.robot.type if hasattr(cfg.env, 'robot') else ""

        critic_indices = list(range(0, n_steps, critic_subsample))
        n_critic = len(critic_indices)
        n_batches = (n_critic + critic_batch_size - 1) // critic_batch_size
        logger.info(
            f"[ENV] Running critic inference: {n_critic}/{n_steps} steps "
            f"(subsample={critic_subsample}) in {n_batches} batches (batch_size={critic_batch_size})..."
        )
        with torch.no_grad():
            for batch_start in tqdm(range(0, n_critic, critic_batch_size), desc="Critic inference", unit="batch"):
                for ci in critic_indices[batch_start:batch_start + critic_batch_size]:
                    frame = episode_log_buffer[ci]
                    batch_for_preprocessor = {
                        k: v for k, v in frame['obs'].items()
                        if k in cfg.policy.input_features
                    }
                    batch_for_preprocessor["robot_type"] = robot_type
                    batch_for_preprocessor['complementary_data'] = {
                        'task': [task_str],
                        'subtask': [""],
                        'advantage': adv_val,
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
                        logger.error("[ENV] Could not find critic values in critic output")
                        critic_val = 0.0
                    critic_values.append(critic_val)

    # --- 3. Save critic JSON + plot ---
    with open(os.path.join(log_dir, "critic_values.json"), "w") as f:
        json.dump(critic_values, f)

    if critic_values:
        plt.figure(figsize=(10, 5))
        plt.plot(critic_values)
        plt.title(f"Critic Values - Episode {episode_counter}")
        plt.xlabel("Step")
        plt.ylabel("Min Q-Value")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "critic_plot.png"))
        plt.close()

    # --- 4. Generate video ---
    try:
        save_video_with_critic_overlay(
            log_dir, critic_values,
            camera_names=video_logging_cameras,
            fps=cfg.env.fps,
            subtask_texts=subtask_texts,
            subsample=critic_subsample,
        )
        logger.info(f"[ENV] Video generated for episode {episode_counter}")
    except Exception as e:
        logger.error(f"[ENV] Failed to generate video: {e}")


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
        
        # Wait for initial episode start, then reset and populate obs BEFORE signalling
        # episode_active=True — so the inference thread never spins on latest_obs=None
        # during the robot reset (which can take several seconds).
        logger.info("[ENV] Waiting for '2' on the teleop device to start episode...")
        _episode_requested = False
        while shared_state.running and not _episode_requested:
            if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                _episode_requested = True
            time.sleep(0.1)

        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()

        from lerobot.rl.gym_manipulator import create_transition
        transition = create_transition(observation=obs, info=info)
        transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
        transition = env_processor(transition)

        policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(policy_fmt_obs, False)
        shared_state.episode_active = True  # signal inference only after obs is ready

        interaction_step = 0
        video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])
        episode_log_buffer = []

        while shared_state.running:
            # 1. Episode boundary check
            if not shared_state.episode_active:
                logger.info("[ENV] Episode ended. Press '2' on the keyboard to start the next episode...")
                new_episode_requested = False
                while shared_state.running and not new_episode_requested:
                    if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                        new_episode_requested = True
                        break
                    time.sleep(0.1)

                if not shared_state.running:
                    break

                logger.info("[ENV] Starting next episode.")

                obs, info = online_env.reset()

                # Smoothly bring the leader arm to neutral alongside the follower.
                # reset_follower_position only needs .bus with sync_read/sync_write,
                # which SOLeader satisfies via duck typing.
                reset_pose = (
                    cfg.env.processor.reset.fixed_reset_joint_positions
                    if getattr(cfg.env, "processor", None) is not None
                    and getattr(cfg.env.processor, "reset", None) is not None
                    and getattr(cfg.env.processor.reset, "fixed_reset_joint_positions", None) is not None
                    else None
                )
                if teleop_device is not None and reset_pose is not None:
                    import numpy as _np
                    from lerobot.rl.gym_manipulator import reset_follower_position
                    reset_follower_position(teleop_device, _np.array(reset_pose, dtype=_np.float32))

                env_processor.reset()
                action_processor.reset()

                with action_queue.lock:
                    action_queue.queue = None
                    action_queue.original_queue = None
                    action_queue.last_index = 0

                shared_state.request_reset()
                was_intervening = False
                episode_log_buffer = []

                transition = create_transition(observation=obs, info=info)
                transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
                transition = env_processor(transition)

                policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(policy_fmt_obs, False)
                # Set episode_active only after the full reset sequence (robot move, queue
                # clear, obs update) so the inference thread never sees episode_active=True
                # while the shared observation still points to the failed-episode position.
                shared_state.episode_active = True

            start_time = time.perf_counter()
            if not hasattr(shared_state, '_env_step_count'):
                shared_state._env_step_count = 0
            shared_state._env_step_count += 1
            _env_do_print = (shared_state._env_step_count % 30 == 1)

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
            _t_step0 = time.perf_counter() if _env_do_print else 0.0
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
            # No deep copy needed: step_env_and_process_transition never mutates transition in-place
            current_transition_data = transition

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

            # --- LOGGING BUFFER ---
            # Cheaply snapshot the current obs + subtask text into a list.
            # Expensive work (critic inference, PNG saving, video) is deferred
            # to _finalize_episode_log(), which runs during the pause between episodes.
            if shared_state.is_logging_episode:
                with shared_state.lock:
                    current_subtask = shared_state.current_subtask_text
                episode_log_buffer.append({
                    'obs': {k: v.detach().cpu().clone() for k, v in next_policy_fmt_obs.items()},
                    'subtask_text': current_subtask,
                })

            # Handle episode boundary
            if new_transition[TransitionKey.DONE] or new_transition[TransitionKey.TRUNCATED]:
                logger.info(f"[ENV] Episode {shared_state.episode_counter} finished.")

                # 1. Finalize logging — runs during the natural pause before the next episode.
                # Inference thread is already idle (episode_active=False), so no env impact.
                if shared_state.is_logging_episode:
                    log_dir = os.path.join(cfg.output_dir, "logging_episodes", f"episode_{shared_state.episode_counter:06d}")
                    _finalize_episode_log(
                        episode_log_buffer=episode_log_buffer,
                        policy=policy,
                        cfg=cfg,
                        log_dir=log_dir,
                        episode_counter=shared_state.episode_counter,
                        video_logging_cameras=video_logging_cameras,
                        critic_batch_size=10,
                    )
                    episode_log_buffer = []

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

