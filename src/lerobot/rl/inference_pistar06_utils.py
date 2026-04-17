"""Async inference worker utilities for the `pistar06` policy.

This is a lean clone of `inference_utils.py` adapted for the
advantage-conditioned ``pistar06`` model (a.k.a. PiStar06). Compared with the
``pi05_full`` flow, this module:

- Has no notion of a ``current_subtask_text`` (pistar06 doesn't generate or
  consume subtask tokens).
- Does not re-inject hardcoded subtask tokens before each chunk.
- Does not run a critic-value forward pass for video overlays — pistar06 has
  no built-in critic. Instead, we just stitch the cached PNG frames into a
  plain video.
- Reads the task string from ``cfg.env.task`` (``PiStar06Config`` does not
  define a ``task`` field).
- Drops the ``advantage`` entry from ``complementary_data`` — pistar06 injects
  the advantage text snippet itself inside ``predict_action_chunk``.

All RTC anchor/delta alignment, smoothing and queue-management logic is kept
verbatim because it is model-agnostic.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import TransitionKey
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

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
                "env_wait_time": self.env_wait_time,
                "env_steps": self.env_steps,
                "inference_wait_time": self.inference_wait_time,
                "env_active_time": getattr(self, "env_active_time_total", 0.0),
            }
            self.env_wait_time = 0.0
            self.env_steps = 0
            self.inference_wait_time = 0.0
            self.env_active_time_total = 0.0
            return metrics

    def update_observation(self, obs: dict, is_intervening: bool):
        with self.lock:
            # We assume obs is already on device/preprocessed by env_processor.
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
    policy_obs: dict = {}

    has_policy_format = (
        "observation.state" in env_obs
        or any(k.startswith("observation.images.") for k in env_obs.keys())
    )

    if has_policy_format:
        for key in env_obs.keys():
            if key == "observation.state" or key.startswith("observation.images."):
                policy_obs[key] = env_obs[key]

    camera_mapping = {
        "wrist": "observation.images.wrist",
        "top": "observation.images.top",
        "side": "observation.images.side",
    }

    pixels_dict = env_obs.get("pixels", env_obs)
    for env_key, policy_key in camera_mapping.items():
        if policy_key not in policy_obs:
            if env_key in pixels_dict:
                policy_obs[policy_key] = pixels_dict[env_key]

    if "observation.state" not in policy_obs:
        joint_order = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
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
            policy_obs["observation.state"] = torch.cat(joint_values, dim=0)

    return policy_obs


def get_actions_worker(policy, shared_state: SharedState, action_queue, cfg):
    """Background inference thread wrapper for pistar06.

    Continually generates actions via ``predict_action_chunk`` using RTC.
    """
    try:
        logger.info("[GET_ACTIONS] Starting background inference thread")
        latency_tracker = LatencyTracker()

        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk = 1.0 / cfg.env.fps

        # `task` lives on the env config for pistar06 — the policy config has no
        # `task` field (PiStar06Config -> PI05Config).
        task_str = cfg.env.task

        while shared_state.running:
            # 1. Reset check
            if shared_state.check_and_clear_reset():
                policy.reset()
                continue

            if not shared_state.episode_active:
                time.sleep(0.01)
                continue

            # 2. Check if we actually need a new chunk (use p95 for less pessimistic threshold).
            # To avoid over-inferencing and queue saturation, only infer if we're getting close.
            # In a real async inference loop we fetch if queue <= execution_horizon + delay.
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

            # 5. Inject only the language/task complementary data needed by the
            # standard pi05 preprocessor pipeline. pistar06 handles its own
            # advantage-text injection inside `predict_action_chunk`.
            batch_for_preprocessor["robot_type"] = (
                cfg.env.robot.type if hasattr(cfg.env, "robot") else ""
            )
            batch_for_preprocessor["complementary_data"] = {
                "task": [task_str],
            }

            current_time = time.perf_counter()

            # 6. Execute model inference using preprocessor and postprocessor
            with torch.no_grad():
                if hasattr(policy, "preprocessor") and policy.preprocessor is not None:
                    processed_batch = policy.preprocessor(batch_for_preprocessor)
                else:
                    processed_batch = batch_for_preprocessor

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()

                # --- Anchor/Delta Alignment for RTC inpainting ---
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

                # Using p95 instead of max: avoids a single latency spike biasing
                # the model to predict too far ahead.
                inference_delay = math.ceil(latency_tracker.p95() / time_per_chunk)

                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon,
                )

                original_actions = actions_chunk.squeeze(0).clone()

                # --- Global chunk un-normalization & absolute action reconstruction ---
                unnormalized_actions = (
                    policy.postprocessor(original_actions)
                    if hasattr(policy, "postprocessor") and policy.postprocessor is not None
                    else original_actions.clone()
                )

                if anchor_now is not None and action_encoding in ["anchor", "delta"]:
                    # Fix broadcasting: anchor_now might have a leading batch dim
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now

                    if action_encoding == "anchor":
                        processed_actions = unnormalized_actions + anchor_sq.to(
                            unnormalized_actions.device
                        )[None, :]
                    elif action_encoding == "delta":
                        processed_actions = torch.cumsum(unnormalized_actions, dim=0) + anchor_sq.to(
                            unnormalized_actions.device
                        )[None, :]
                else:
                    processed_actions = unnormalized_actions

                # --- Centred moving average smoothing (window=5) ---
                if processed_actions.shape[0] >= 5:
                    padded = torch.cat(
                        [processed_actions[0:1]] * 2
                        + [processed_actions]
                        + [processed_actions[-1:]] * 2,
                        dim=0,
                    )
                    smoothed = (
                        padded[:-4]
                        + padded[1:-3]
                        + padded[2:-2]
                        + padded[3:-1]
                        + padded[4:]
                    ) / 5.0
                    processed_actions = smoothed

                if not hasattr(policy, "_chunk_plot_counter"):
                    policy._chunk_plot_counter = 0
                policy._chunk_plot_counter += 1

            # Track literal latency
            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)

            # --- JERK-FREE START FIX ---
            # If the robot queue was starved (e.g., first chunk), it executed fewer actions
            # than `new_delay`. Discarding `new_delay` actions without the robot having moved
            # causes the first movement to jerk. We constrain the delay discarded by how many
            # actions the environment actually consumed.
            current_index = action_queue.get_action_index()
            actions_consumed = max(0, current_index - action_index_before)
            effective_delay = min(new_delay, actions_consumed)

            action_queue.merge(
                original_actions=original_actions,
                processed_actions=processed_actions,
                real_delay=effective_delay,
                action_index_before_inference=action_index_before,
                anchor_state=anchor_now
                if getattr(cfg.policy, "action_encoding", "absolute") in ["anchor", "delta"]
                else None,
            )

        logger.info("[GET_ACTIONS] Inference thread shutting down smoothly.")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Fatal exception: {e}")
        logger.error(traceback.format_exc())


def _save_episode_video(
    episode_log_buffer,
    log_dir: str,
    episode_counter: int,
    video_logging_cameras,
    fps: int,
):
    """Stitch cached PNG frames into a single mp4 per camera.

    Slim replacement for ``save_video_with_critic_overlay`` — there is no
    critic value to overlay because pistar06 has no critic head.
    """
    if not episode_log_buffer:
        return

    os.makedirs(log_dir, exist_ok=True)
    n_steps = len(episode_log_buffer)

    # 1. Save PNGs (one per cam per step) so the directory layout matches the
    # pi05_full pipeline's logging output for downstream tooling.
    logger.info(f"[ENV] Saving {n_steps} frames for episode {episode_counter}...")
    cam_to_frames: dict[str, list[np.ndarray]] = {cam: [] for cam in video_logging_cameras}
    for step_idx, frame in enumerate(
        tqdm(episode_log_buffer, desc="Saving frames", unit="frame")
    ):
        for key, val in frame["obs"].items():
            if "image" not in key:
                continue
            cam_name = key.split(".")[-1]
            if cam_name not in video_logging_cameras:
                continue

            img_tensor = val[0] if val.ndim == 4 else val
            if img_tensor.dtype == torch.uint8:
                img_np = img_tensor.numpy().transpose(1, 2, 0)
            else:
                v_max = img_tensor.max().item()
                # Heuristic: if max value is small, assume [0,1] normalized float;
                # otherwise assume [0,255].
                if v_max <= 5.0:
                    img_np = (
                        img_tensor.float().numpy().transpose(1, 2, 0) * 255.0
                    ).clip(0, 255).astype(np.uint8)
                else:
                    img_np = (
                        img_tensor.float().numpy().transpose(1, 2, 0)
                    ).clip(0, 255).astype(np.uint8)
            img_path = os.path.join(log_dir, f"step_{step_idx:06d}_{cam_name}.png")
            Image.fromarray(img_np).save(img_path)
            cam_to_frames[cam_name].append(img_np)

    # 2. Encode one mp4 per camera using OpenCV (BGR).
    for cam_name, frames in cam_to_frames.items():
        if not frames:
            continue
        height, width = frames[0].shape[:2]
        video_path = os.path.join(log_dir, f"episode_{episode_counter:06d}_{cam_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
        try:
            for img_np in frames:
                writer.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        logger.info(f"[ENV] Wrote {video_path} ({len(frames)} frames @ {fps} fps)")


def env_interaction_worker(
    online_env,
    env_processor,
    action_processor,
    action_queue,
    shared_state: SharedState,
    teleop_device,
    cfg,
    policy=None,
    postprocessor=None,
):
    """Main environment thread loop for pistar06.

    Strictly adheres to config FPS. Syncs teleop overrides. Does not write any
    subtask metadata into the per-step log buffer because pistar06 doesn't
    expose subtask state.
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

        # Extract initial state observation immediately to bootstrap shared state
        # (assuming the env was reset right before spawning threads).
        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()

        from lerobot.rl.gym_manipulator import create_transition
        transition = create_transition(observation=obs, info=info)
        # Some env processors expect a `subtask` field even though pistar06 doesn't use it;
        # keep it as an empty placeholder so the existing pipeline keeps running.
        transition[TransitionKey.COMPLEMENTARY_DATA] = {
            "subtask": [""] * (len(obs) if isinstance(obs, list) else 1)
        }
        transition = env_processor(transition)

        policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(policy_fmt_obs, False)

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
                transition[TransitionKey.COMPLEMENTARY_DATA] = {
                    "subtask": [""] * (len(obs) if isinstance(obs, list) else 1)
                }
                transition = env_processor(transition)

                policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(policy_fmt_obs, False)
                # Set episode_active only after the full reset sequence (robot move,
                # queue clear, obs update) so the inference thread never sees
                # episode_active=True while the shared observation still points to
                # the failed-episode position.
                shared_state.episode_active = True

            start_time = time.perf_counter()

            # --- TELEOP AND STATE OVERRIDES ---
            # If we were intervening but teleop stopped, we trigger a policy reset
            if was_intervening and not shared_state.is_intervening:
                logger.info("[ENV] Teleop disengaged, soliciting policy/queue reset")
                shared_state.request_reset()
                with action_queue.lock:
                    action_queue.queue = None
                    action_queue.original_queue = None
                    action_queue.last_index = 0

            was_intervening = shared_state.is_intervening

            # --- ACTION PREPARATION ---
            if was_intervening:
                if hasattr(online_env, "get_raw_joint_positions"):
                    raw_joints = online_env.get_raw_joint_positions()
                    joint_order = [
                        "shoulder_pan.pos",
                        "shoulder_lift.pos",
                        "elbow_flex.pos",
                        "wrist_flex.pos",
                        "wrist_roll.pos",
                        "gripper.pos",
                    ]
                    action = torch.tensor(
                        [float(raw_joints.get(k, 0.0)) for k in joint_order],
                        dtype=torch.float32,
                        device=cfg.policy.device,
                    )
                else:
                    action = torch.zeros(6, dtype=torch.float32, device=cfg.policy.device)
            else:
                action = action_queue.get()
                if action is not None:
                    if action.shape[-1] > 6:
                        action = action[..., :6]
                else:
                    if hasattr(online_env, "get_raw_joint_positions"):
                        raw_joints = online_env.get_raw_joint_positions()
                        joint_order = [
                            "shoulder_pan.pos",
                            "shoulder_lift.pos",
                            "elbow_flex.pos",
                            "wrist_flex.pos",
                            "wrist_roll.pos",
                            "gripper.pos",
                        ]
                        action = torch.tensor(
                            [float(raw_joints.get(k, 0.0)) for k in joint_order],
                            dtype=torch.float32,
                            device=cfg.policy.device,
                        )
                    else:
                        action = torch.zeros(6, dtype=torch.float32, device=cfg.policy.device)
                    logger.warning("[ENV] Action queue starved. Executing null ops.")

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
                state_dict = convert_env_obs_to_policy_format(
                    current_transition_data[TransitionKey.OBSERVATION]
                )
                next_state_dict = convert_env_obs_to_policy_format(
                    new_transition[TransitionKey.OBSERVATION]
                )

                if not isinstance(action, torch.Tensor):
                    action_tensor = torch.tensor(action, dtype=torch.float32, device="cpu")
                else:
                    action_tensor = action.detach().cpu()

                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)

                intervention_info = current_transition_data.get(TransitionKey.INFO, {})
                current_is_intervening = intervention_info.get(
                    TeleopEvents.IS_INTERVENTION, False
                )
                complementary_info = {
                    "discrete_penalty": torch.tensor(
                        [
                            current_transition_data.get(
                                TransitionKey.COMPLEMENTARY_DATA, {}
                            ).get("discrete_penalty", 0.0)
                        ]
                    ),
                    TeleopEvents.IS_INTERVENTION.value: torch.tensor(
                        [float(current_is_intervening)], dtype=torch.float32
                    ),
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
            next_policy_fmt_obs = convert_env_obs_to_policy_format(
                new_transition[TransitionKey.OBSERVATION]
            )

            # --- LOGGING BUFFER ---
            # Cheaply snapshot the current obs into a list. Expensive work
            # (PNG saving + video) is deferred to ``_save_episode_video``,
            # which runs during the pause between episodes.
            if shared_state.is_logging_episode:
                episode_log_buffer.append(
                    {"obs": {k: v.detach().cpu().clone() for k, v in next_policy_fmt_obs.items()}}
                )

            # Handle episode boundary
            if new_transition[TransitionKey.DONE] or new_transition[TransitionKey.TRUNCATED]:
                logger.info(f"[ENV] Episode {shared_state.episode_counter} finished.")

                if shared_state.is_logging_episode:
                    log_dir = os.path.join(
                        cfg.output_dir,
                        "logging_episodes",
                        f"episode_{shared_state.episode_counter:06d}",
                    )
                    _save_episode_video(
                        episode_log_buffer=episode_log_buffer,
                        log_dir=log_dir,
                        episode_counter=shared_state.episode_counter,
                        video_logging_cameras=video_logging_cameras,
                        fps=cfg.env.fps,
                    )
                    episode_log_buffer = []

                shared_state.episode_counter += 1

                # Periodic dataset flush is only meaningful when a replay
                # buffer is attached. The pistar06 inference flow records
                # rollouts as per-episode PNGs/MP4s (see _save_episode_video)
                # rather than streaming through a ReplayBuffer.
                if shared_state.replay_buffer is not None:
                    episode_save_freq = getattr(cfg, "episode_save_freq", 10)
                    if shared_state.episode_counter % episode_save_freq == 0:
                        logger.info(
                            f"[ENV] Saving inference dataset at episode {shared_state.episode_counter}..."
                        )
                        dataset_root = os.path.join(cfg.output_dir, "inference_dataset")
                        import shutil
                        if os.path.exists(dataset_root):
                            shutil.rmtree(dataset_root)
                        try:
                            shared_state.replay_buffer.to_lerobot_dataset(
                                repo_id="inference_recorded",
                                fps=cfg.env.fps,
                                root=dataset_root,
                                task_name=cfg.env.task,
                            )
                        except Exception as e:
                            logger.error(f"[ENV] Failed to save lerobot dataset: {e}")

                episode_logging_freq = getattr(cfg, "episode_logging_freq", 4)
                shared_state.is_logging_episode = (
                    shared_state.episode_counter % episode_logging_freq == 0
                )

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

            if getattr(cfg, "use_rerun", False):
                import rerun as rr
                rr.set_time_sequence("step", interaction_step)

                for key, val in next_policy_fmt_obs.items():
                    if "image" in key:
                        val_np = val[0].cpu().numpy() if val.ndim == 4 else val.cpu().numpy()
                        image_np = val_np.transpose(1, 2, 0)
                        rr.log(f"world/cameras/{key}", rr.Image(image_np))

                if hasattr(online_env, "get_raw_joint_positions"):
                    joints = online_env.get_raw_joint_positions()
                    if joints:
                        for j_name, j_val in joints.items():
                            rr.log(f"world/robot_joints/{j_name}", rr.Scalars(float(j_val)))

            shared_state.update_observation(next_policy_fmt_obs, is_intervening)
            transition = new_transition

            # Calculate tight strict Hz sleep
            dt_s = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - dt_s)

            if not hasattr(shared_state, "env_active_time_total"):
                shared_state.env_active_time_total = 0.0
            shared_state.env_active_time_total += dt_s

            shared_state.add_env_wait_time(sleep_time)

            interaction_step += 1
            precise_sleep(sleep_time)

        logger.info("[ENV] Environmental loop complete.")
    except Exception as e:
        logger.error(f"[ENV] Fatal exception: {e}")
        logger.error(traceback.format_exc())
