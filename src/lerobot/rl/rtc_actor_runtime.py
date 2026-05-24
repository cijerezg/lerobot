from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
import traceback
import threading
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import TransitionKey
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.actor import push_transitions_to_transport_queue
from lerobot.rl.gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.rl.inference_utils import convert_env_obs_to_policy_format, apply_butterworth_filter
from lerobot.rl.utils import save_video_with_critic_overlay
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.rl.rl_trainer import Trainer
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport.utils import bytes_to_state_dict, python_object_to_bytes
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.constants import ACTION
from lerobot.utils.transition import Transition, move_state_dict_to_device, move_transition_to_device

logger = logging.getLogger(__name__)



def _action_dim(cfg) -> int:
    if hasattr(cfg.policy, "action_dim"):
        return int(cfg.policy.action_dim)
    action_feat = getattr(cfg.policy, "output_features", {}).get("action")
    if action_feat is not None and getattr(action_feat, "shape", None):
        return int(action_feat.shape[0])
    return int(next(iter(cfg.policy.output_features.values())).shape[0])


def _raw_joint_action(online_env, action_dim: int, device) -> torch.Tensor:
    if hasattr(online_env, "get_raw_joint_positions"):
        raw_joints = online_env.get_raw_joint_positions()
        joint_order = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ]
        vals = [float(raw_joints.get(k, 0.0)) for k in joint_order[:action_dim]]
        if len(vals) < action_dim:
            vals.extend([0.0] * (action_dim - len(vals)))
        return torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.zeros(action_dim, dtype=torch.float32, device=device)


class RTCSharedState:
    """Thread-safe state manager for the RTC actor runtime."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_obs: dict | None = None
        self.is_intervening = False
        self.episode_active = False
        self.policy_reset_requested = False
        self.update_parameters_requested = False
        self.running = True
        self.env_wait_time = 0.0
        self.env_steps = 0
        self.env_active_time_total = 0.0
        self.env_action_get_time = 0.0
        self.env_step_time = 0.0
        self.env_robot_step_time = 0.0
        self.env_obs_proc_time = 0.0
        self.env_action_proc_time = 0.0
        self.env_post_step_time = 0.0
        self.env_move_cpu_time = 0.0
        self.env_rerun_time = 0.0
        self.queue_starvation_count = 0
        self.inference_wait_time = 0.0
        self.inference_count = 0
        self.inference_latencies: list[float] = []
        self.inference_preprocess_time = 0.0
        self.inference_model_time = 0.0
        self.inference_postprocess_time = 0.0
        self.current_step = 0
        self.cached_subtask_tokens: torch.Tensor | None = None
        self.cached_subtask_masks: torch.Tensor | None = None
        self.current_subtask_text = ""
        self.episode_counter = 0
        self.is_logging_episode = False
        self.replay_buffer: ReplayBuffer | None = None
        self.params_loaded_event = threading.Event()
        self.params_loaded_event.set()

    def add_env_wait_time(self, wait_time: float) -> None:
        with self.lock:
            self.env_wait_time += wait_time
            self.env_steps += 1

    def add_inference_wait_time(self, wait_time: float) -> None:
        with self.lock:
            self.inference_wait_time += wait_time

    def add_inference_latency(self, latency: float) -> None:
        with self.lock:
            self.inference_count += 1
            self.inference_latencies.append(latency)

    def add_inference_breakdown(self, preprocess: float, model: float, postprocess: float) -> None:
        with self.lock:
            self.inference_preprocess_time += preprocess
            self.inference_model_time += model
            self.inference_postprocess_time += postprocess

    def add_env_step_breakdown(self, action_get: float, step: float) -> None:
        with self.lock:
            self.env_action_get_time += action_get
            self.env_step_time += step

    def add_env_step_detail(self, robot_step: float, obs_proc: float, action_proc: float = 0.0) -> None:
        with self.lock:
            self.env_robot_step_time += robot_step
            self.env_obs_proc_time += obs_proc
            self.env_action_proc_time += action_proc

    def add_env_post_step_detail(self, post_step: float, move_cpu: float, rerun: float) -> None:
        with self.lock:
            self.env_post_step_time += post_step
            self.env_move_cpu_time += move_cpu
            self.env_rerun_time += rerun

    def add_queue_starvation(self) -> None:
        with self.lock:
            self.queue_starvation_count += 1

    def get_and_reset_metrics(self) -> dict:
        with self.lock:
            metrics = {
                "env_wait_time": self.env_wait_time,
                "env_steps": self.env_steps,
                "inference_wait_time": self.inference_wait_time,
                "env_active_time": self.env_active_time_total,
                "inference_count": self.inference_count,
                "inference_latencies": list(self.inference_latencies),
                "inference_preprocess_time": self.inference_preprocess_time,
                "inference_model_time": self.inference_model_time,
                "inference_postprocess_time": self.inference_postprocess_time,
                "env_action_get_time": self.env_action_get_time,
                "env_step_time": self.env_step_time,
                "env_robot_step_time": self.env_robot_step_time,
                "env_obs_proc_time": self.env_obs_proc_time,
                "env_action_proc_time": self.env_action_proc_time,
                "env_post_step_time": self.env_post_step_time,
                "env_move_cpu_time": self.env_move_cpu_time,
                "env_rerun_time": self.env_rerun_time,
                "queue_starvation_count": self.queue_starvation_count,
            }
            self.env_wait_time = 0.0
            self.env_steps = 0
            self.env_active_time_total = 0.0
            self.env_action_get_time = 0.0
            self.env_step_time = 0.0
            self.env_robot_step_time = 0.0
            self.env_obs_proc_time = 0.0
            self.env_action_proc_time = 0.0
            self.env_post_step_time = 0.0
            self.env_move_cpu_time = 0.0
            self.env_rerun_time = 0.0
            self.queue_starvation_count = 0
            self.inference_wait_time = 0.0
            self.inference_count = 0
            self.inference_latencies = []
            self.inference_preprocess_time = 0.0
            self.inference_model_time = 0.0
            self.inference_postprocess_time = 0.0
            return metrics

    def update_observation(self, obs: dict, is_intervening: bool) -> None:
        with self.lock:
            self.latest_obs = dict(obs)
            self.is_intervening = is_intervening

    def get_latest_observation(self) -> dict | None:
        with self.lock:
            return dict(self.latest_obs) if self.latest_obs is not None else None

    def set_intervention(self, status: bool) -> None:
        with self.lock:
            self.is_intervening = status

    def set_episode_active(self, status: bool) -> None:
        with self.lock:
            self.episode_active = status

    def request_reset(self) -> None:
        with self.lock:
            self.policy_reset_requested = True

    def check_and_clear_reset(self) -> bool:
        with self.lock:
            if self.policy_reset_requested:
                self.policy_reset_requested = False
                return True
            return False

    def request_parameter_update(self) -> None:
        with self.lock:
            self.update_parameters_requested = True
        self.params_loaded_event.clear()

    def check_and_clear_parameter_update(self) -> bool:
        with self.lock:
            if self.update_parameters_requested:
                self.update_parameters_requested = False
                return True
            return False

    def update_subtask_cache(self, tokens: torch.Tensor, masks: torch.Tensor) -> None:
        with self.lock:
            self.cached_subtask_tokens = tokens.clone()
            self.cached_subtask_masks = masks.clone()


def pull_new_policy_weights(policy: nn.Module, parameters_queue, device: torch.device) -> None:
    if parameters_queue is None:
        return
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is None:
        return
    logger.info("[RTC_ACTOR] Loading new parameters from Learner.")
    state_dicts = bytes_to_state_dict(bytes_state_dict)
    actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
    if hasattr(policy, "actor"):
        policy.actor.load_state_dict(actor_state_dict, strict=False)
    else:
        policy.load_state_dict(actor_state_dict, strict=False)
    logger.info("[RTC_ACTOR] Parameters loaded.")


def align_prev_actions(
    prev_actions: torch.Tensor,
    anchor_old: torch.Tensor,
    anchor_now: torch.Tensor,
    action_encoding: str,
    chunk_size: int,
    postprocessor,
    normalizer,
) -> torch.Tensor:
    """Re-align leftover normalized actions when anchor state changes."""
    n_left = prev_actions.shape[0]
    action_dim = prev_actions.shape[1]
    offset = chunk_size - n_left

    if action_encoding == "delta" and offset > 0:
        return prev_actions

    right_padded = torch.zeros(chunk_size, action_dim, device=prev_actions.device, dtype=prev_actions.dtype)
    right_padded[offset:] = prev_actions
    d_abs = postprocessor(right_padded)

    dev = d_abs.device
    delta_s = anchor_old.squeeze(0).to(dev) - anchor_now.squeeze(0).to(dev)
    if action_encoding == "anchor":
        d_abs[offset:] += delta_s
    else:
        d_abs[0] += delta_s

    left_padded = torch.zeros(chunk_size, action_dim, device=dev, dtype=d_abs.dtype)
    left_padded[:n_left] = d_abs[offset:]
    return normalizer._normalize_action(left_padded, inverse=False)[:n_left]


def _maybe_align_prev_actions(
    *,
    prev_actions: torch.Tensor | None,
    action_queue: ActionQueue,
    latest_obs: dict,
    policy,
    postprocessor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    action_encoding = getattr(policy.config, "action_encoding", "absolute")
    anchor_now = None
    if action_encoding not in {"anchor", "delta"}:
        return prev_actions, anchor_now

    from lerobot.utils.constants import OBS_STATE

    if OBS_STATE not in latest_obs:
        return prev_actions, anchor_now

    anchor_now = latest_obs[OBS_STATE]
    if prev_actions is None or action_queue.anchor_state is None:
        return prev_actions, anchor_now

    try:
        from lerobot.processor import NormalizerProcessorStep
        normalizer = next(
            step for step in policy.preprocessor.steps
            if isinstance(step, NormalizerProcessorStep)
        )
    except Exception:
        logger.warning("[RTC] Could not find action normalizer; skipping leftover alignment.")
        return prev_actions, anchor_now

    anchor_old = action_queue.anchor_state
    logger.debug("[RTC] Alignment offset: %.3f", (anchor_old.to(anchor_now.device) - anchor_now).norm().item())
    return align_prev_actions(
        prev_actions=prev_actions,
        anchor_old=anchor_old,
        anchor_now=anchor_now,
        action_encoding=action_encoding,
        chunk_size=policy.config.chunk_size,
        postprocessor=postprocessor,
        normalizer=normalizer,
    ), anchor_now


def rtc_inference_worker(
    policy: nn.Module,
    trainer: Trainer,
    preprocessor,
    postprocessor,
    shared_state: RTCSharedState,
    action_queue: ActionQueue,
    parameters_queue,
    device: torch.device,
    cfg,
) -> None:
    """Background inference worker using RTC ActionQueue semantics."""
    try:
        logger.info("[RTC_INFERENCE] Thread started.")
        if getattr(cfg.policy, "torch_compile", False):
            _warmup_compiled_policy(policy, trainer, preprocessor, cfg, device)
        latency_tracker = LatencyTracker()
        inference_step = 0
        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk = 1.0 / cfg.env.fps
        task_str = cfg.policy.task
        action_dim = _action_dim(cfg)

        while shared_state.running:
            if shared_state.check_and_clear_parameter_update():
                if parameters_queue is not None:
                    pull_new_policy_weights(policy, parameters_queue, device)
                shared_state.params_loaded_event.set()

            if not shared_state.episode_active:
                time.sleep(0.01)
                continue

            if shared_state.check_and_clear_reset():
                if hasattr(policy, "reset"):
                    policy.reset()
                continue

            if shared_state.is_intervening:
                time.sleep(0.01)
                continue

            current_delay = math.ceil(latency_tracker.p95() / time_per_chunk)
            if not action_queue.empty() and action_queue.qsize() > execution_horizon + current_delay:
                wait_start = time.perf_counter()
                time.sleep(0.01)
                shared_state.add_inference_wait_time(time.perf_counter() - wait_start)
                continue

            latest_obs = shared_state.get_latest_observation()
            if latest_obs is None:
                time.sleep(0.01)
                continue

            obs_filtered = {k: v for k, v in latest_obs.items() if k in cfg.policy.input_features}
            robot_type = cfg.env.robot.type if hasattr(cfg.env, "robot") else ""

            current_time = time.perf_counter()
            with torch.no_grad():
                t_preproc_start = time.perf_counter()
                processed_batch = trainer.build_inference_batch(
                    obs_filtered,
                    task_str,
                    cfg,
                    preprocessor=preprocessor,
                    robot_type=robot_type,
                )
                t_preproc_end = time.perf_counter()

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                prev_actions, anchor_now = _maybe_align_prev_actions(
                    prev_actions=prev_actions,
                    action_queue=action_queue,
                    latest_obs=latest_obs,
                    policy=policy,
                    postprocessor=postprocessor,
                )

                inference_delay = math.ceil(latency_tracker.p95() / time_per_chunk)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                torch.compiler.cudagraph_mark_step_begin()
                t_gpu_start = time.perf_counter()
                actions_chunk = policy.predict_action_chunk(
                    processed_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=prev_actions,
                    execution_horizon=execution_horizon,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_gpu_end = time.perf_counter()

                cached_tokens = getattr(policy, "_cached_subtask_tokens", None)
                cached_masks = getattr(policy, "_cached_subtask_masks", None)
                if cached_tokens is not None and cached_masks is not None:
                    shared_state.update_subtask_cache(cached_tokens[0].cpu(), cached_masks[0].cpu())

                original_actions = actions_chunk.squeeze(0).clone()[..., :action_dim]
                action_encoding = getattr(policy.config, "action_encoding", "absolute")

                # Keep original_actions normalized for RTC leftovers. The queue's
                # processed_actions are robot-space actions, matching the PI05
                # reference path before filtering/clamping and env execution.
                unnormalized_actions = (
                    postprocessor(original_actions)
                    if postprocessor is not None
                    else original_actions.clone()
                )

                if anchor_now is not None and action_encoding in {"anchor", "delta"}:
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
                    if action_encoding == "anchor":
                        processed_actions = unnormalized_actions + anchor_sq.to(unnormalized_actions.device)[None, :]
                    else:
                        processed_actions = torch.cumsum(unnormalized_actions, dim=0) + anchor_sq.to(unnormalized_actions.device)[None, :]
                else:
                    processed_actions = unnormalized_actions

                processed_actions = apply_butterworth_filter(processed_actions)

                clamp_limits = getattr(policy.config, "action_clamp_limits", None)
                if clamp_limits is not None:
                    limits = torch.tensor(clamp_limits, dtype=processed_actions.dtype, device=processed_actions.device)
                    exceeded = (processed_actions < limits[:, 0]) | (processed_actions > limits[:, 1])
                    if exceeded.any():
                        joints = exceeded.any(dim=0).nonzero(as_tuple=True)[0].tolist()
                        raw_min = processed_actions[:, joints].min(dim=0).values.tolist()
                        raw_max = processed_actions[:, joints].max(dim=0).values.tolist()
                        logger.warning(
                            "[CLAMP] Action exceeded limits on joints %s — raw range min=%s max=%s. Clamping.",
                            joints,
                            [f"{v:.1f}" for v in raw_min],
                            [f"{v:.1f}" for v in raw_max],
                        )
                    processed_actions = torch.clamp(processed_actions, min=limits[:, 0], max=limits[:, 1])

            new_latency = time.perf_counter() - current_time
            new_delay = math.ceil(new_latency / time_per_chunk)
            latency_tracker.add(new_latency)
            shared_state.add_inference_latency(new_latency)

            dt_preproc = t_preproc_end - t_preproc_start
            dt_gpu = t_gpu_end - t_gpu_start
            dt_post = new_latency - dt_preproc - dt_gpu
            shared_state.add_inference_breakdown(dt_preproc, dt_gpu, dt_post)
            inference_step += 1
            logger.debug(
                "[RTC_INFERENCE] chunk latency %.3fs [pre=%.3f gpu=%.3f post=%.3f] delay=%d model_delay=%d qsize=%d",
                new_latency, dt_preproc, dt_gpu, dt_post, new_delay, inference_delay, action_queue.qsize(),
            )

            current_index = action_queue.get_action_index()
            actions_consumed = max(0, current_index - action_index_before)
            effective_delay = min(new_delay, actions_consumed)

            if shared_state.policy_reset_requested:
                continue

            action_queue.merge(
                original_actions=original_actions.detach().cpu(),
                processed_actions=processed_actions.detach().cpu(),
                real_delay=effective_delay,
                action_index_before_inference=action_index_before,
                anchor_state=anchor_now.detach().cpu() if anchor_now is not None else None,
            )

        logger.info("[RTC_INFERENCE] Thread shut down.")
    except Exception:
        logger.error("[RTC_INFERENCE] Fatal:\n%s", traceback.format_exc())


def rtc_env_worker(
    online_env,
    env_processor,
    action_processor,
    action_queue: ActionQueue,
    shared_state: RTCSharedState,
    teleop_device,
    transitions_queue,
    interactions_queue,
    cfg,
    postprocessor=None,
    *,
    standalone: bool = False,
    policy: nn.Module | None = None,
    trainer: Trainer | None = None,
) -> None:
    """Environment interaction worker copied from the tested PI05 RTC path."""
    _ = postprocessor  # queued actions are already postprocessed in rtc_inference_worker
    try:
        logger.info("[RTC_ENV] Thread started.")
        action_interval = 1.0 / cfg.env.fps
        action_dim = _action_dim(cfg)
        device = cfg.policy.device

        was_intervening = False
        sum_reward_episode = 0.0
        episode_intervention_steps = 0
        episode_total_steps = 0
        transitions_to_send: list[Transition] = []
        interaction_step = 0
        video_logging_cameras = list(getattr(cfg, "video_logging_cameras", ["top", "side"]))
        episode_log_buffer: list[dict] = []
        episode_logging_freq = int(getattr(cfg, "episode_logging_freq", 0) or 0)
        shared_state.is_logging_episode = (episode_logging_freq > 0 and shared_state.episode_counter % episode_logging_freq == 0)

        logger.info("[ACTOR] Waiting for '2' on the teleop device to start episode...")
        while shared_state.running:
            if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                break
            time.sleep(0.1)

        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()

        transition = create_transition(observation=obs, info=info)
        transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
        transition = env_processor(transition)

        policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(policy_fmt_obs, False)
        if not standalone:
            shared_state.request_parameter_update()

            logger.info("[ACTOR] Loading new params, please wait before episode starts...")
            while shared_state.running:
                if shared_state.params_loaded_event.wait(timeout=0.5):
                    break
            logger.info("[ACTOR] Params loaded. Starting episode.")
        shared_state.set_episode_active(True)

        while shared_state.running:
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
                action_queue.clear()
                shared_state.request_reset()
                if not standalone:
                    shared_state.request_parameter_update()
                was_intervening = False
                episode_log_buffer = []

                transition = create_transition(observation=obs, info=info)
                transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
                transition = env_processor(transition)
                policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(policy_fmt_obs, False)

                if not standalone:
                    logger.info("[ACTOR] Loading new params, please wait before episode starts...")
                    while shared_state.running:
                        if shared_state.params_loaded_event.wait(timeout=0.5):
                            break
                    logger.info("[ACTOR] Params loaded. Starting episode.")
                shared_state.set_episode_active(True)

            start_time = time.perf_counter()

            if was_intervening != shared_state.is_intervening:
                logger.info("[RTC_ENV] Teleop state changed (intervening=%s), resetting policy/queue", shared_state.is_intervening)
                shared_state.request_reset()
                action_queue.clear()
            was_intervening = shared_state.is_intervening

            _t_action_start = time.perf_counter()
            if was_intervening:
                action = _raw_joint_action(online_env, action_dim, device)
            else:
                action = action_queue.get()
                if action is not None:
                    action = action[..., :action_dim].to(device)
                else:
                    shared_state.add_queue_starvation()
                    action = _raw_joint_action(online_env, action_dim, device)
            _t_action_end = time.perf_counter()

            _t_step_start = time.perf_counter()
            _step_timings: dict = {}
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
                timings=_step_timings,
            )
            _t_step_end = time.perf_counter()
            shared_state.add_env_step_detail(
                robot_step=_step_timings.get("robot_step", 0.0),
                obs_proc=_step_timings.get("obs_proc", 0.0),
                action_proc=_step_timings.get("action_proc", 0.0),
            )
            _t_post_step_start = time.perf_counter()

            if TransitionKey.COMPLEMENTARY_DATA not in new_transition:
                new_transition[TransitionKey.COMPLEMENTARY_DATA] = {}
            if "subtask" not in new_transition[TransitionKey.COMPLEMENTARY_DATA]:
                new_transition[TransitionKey.COMPLEMENTARY_DATA]["subtask"] = [""] * (
                    len(new_transition[TransitionKey.OBSERVATION])
                    if isinstance(new_transition[TransitionKey.OBSERVATION], list)
                    else 1
                )

            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA].get("teleop_action", action)
            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)

            sum_reward_episode += float(reward)
            episode_total_steps += 1
            if reward > 0:
                logger.info("[ACTOR] Received transition with reward: %s", reward)

            intervention_info = new_transition[TransitionKey.INFO]
            is_intervening = intervention_info.get(TeleopEvents.IS_INTERVENTION, False)
            shared_state.set_intervention(is_intervening)

            info_success = intervention_info.get(TeleopEvents.SUCCESS, False)
            info_terminate = intervention_info.get(TeleopEvents.TERMINATE_EPISODE, False)
            if done or truncated or info_success or info_terminate:
                logger.info(
                    "[ACTOR EPISODE_END_DEBUG] step=%s | done=%s truncated=%s | success=%s terminate=%s | reward=%s",
                    episode_total_steps, done, truncated, info_success, info_terminate, reward,
                )

            if is_intervening:
                episode_intervention_steps += 1
            else:
                feedback = {}
                for key, value in new_transition[TransitionKey.OBSERVATION].items():
                    if key.endswith(".pos"):
                        feedback[key] = value.item() if isinstance(value, torch.Tensor) else float(value)
                if feedback:
                    teleop_device.send_feedback(feedback)

            with shared_state.lock:
                cached_tokens = shared_state.cached_subtask_tokens
                cached_masks = shared_state.cached_subtask_masks
                if cached_tokens is not None and cached_masks is not None:
                    subtask_tokens = cached_tokens.clone()
                    subtask_masks = cached_masks.clone()
                else:
                    max_len = int(getattr(cfg.policy, "max_decoding_steps", 0))
                    subtask_tokens = torch.zeros(max_len, dtype=torch.long)
                    subtask_masks = torch.zeros(max_len, dtype=torch.bool)

            complementary_info = {
                "discrete_penalty": torch.tensor([
                    new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                ]),
                TeleopEvents.IS_INTERVENTION.value: torch.tensor([float(is_intervening)], dtype=torch.float32),
                "subtask_index": torch.tensor([-1], dtype=torch.long),
                "subtask_tokens": subtask_tokens,
                "subtask_masks": subtask_masks,
            }

            observation = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
            next_observation = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
            transition_to_send = Transition(
                state=observation,
                action=executed_action[..., :action_dim],
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            _t_move_cpu_start = time.perf_counter()
            transition_cpu = move_transition_to_device(transition_to_send, "cpu")
            _t_move_cpu_end = time.perf_counter()
            transitions_to_send.append(transition_cpu)

            if standalone and shared_state.replay_buffer is not None:
                _add_transition_to_replay_buffer(shared_state.replay_buffer, transition_cpu, action_dim)

            next_policy_fmt_obs = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])
            if standalone and shared_state.is_logging_episode:
                episode_log_buffer.append({
                    "obs": {k: v.detach().cpu().clone() for k, v in next_policy_fmt_obs.items()},
                    "action": transition_cpu[ACTION].detach().cpu().clone() if isinstance(transition_cpu.get(ACTION), torch.Tensor) else transition_cpu.get(ACTION),
                    "reward": float(reward),
                    "done": bool(done),
                    "subtask_text": "",
                })
            _t_rerun_start = time.perf_counter()
            if getattr(cfg, "use_rerun", False):
                import rerun as rr
                rr.set_time_sequence("step", interaction_step)
                for key, val in next_policy_fmt_obs.items():
                    if "image" in key:
                        val_np = val[0].cpu().numpy() if val.ndim == 4 else val.cpu().numpy()
                        rr.log(f"world/cameras/{key}", rr.Image(val_np.transpose(1, 2, 0)))
                if hasattr(online_env, "get_raw_joint_positions"):
                    joints = online_env.get_raw_joint_positions()
                    for j_name, j_val in joints.items():
                        rr.log(f"world/robot_joints/{j_name}", rr.Scalars(float(j_val)))
            _t_rerun_end = time.perf_counter()

            shared_state.update_observation(next_policy_fmt_obs, is_intervening)
            transition = new_transition
            interaction_step += 1
            shared_state.current_step = interaction_step

            if done or truncated:
                logger.info(
                    "[ACTOR] Global step %s: Episode ended. reward=%s | done=%s truncated=%s | success=%s terminate=%s",
                    interaction_step, sum_reward_episode, done, truncated, info_success, info_terminate,
                )
                shared_state.set_episode_active(False)

                if standalone:
                    if shared_state.is_logging_episode:
                        log_dir = os.path.join(
                            cfg.output_dir,
                            "logging_episodes",
                            f"episode_{shared_state.episode_counter:06d}",
                        )
                        _finalize_rtc_inference_log(
                            episode_log_buffer=episode_log_buffer,
                            trainer=trainer,
                            policy=policy,
                            cfg=cfg,
                            log_dir=log_dir,
                            episode_counter=shared_state.episode_counter,
                            video_logging_cameras=video_logging_cameras,
                        )
                        episode_log_buffer = []

                    shared_state.episode_counter += 1
                    _maybe_save_inference_dataset(shared_state, cfg)
                    shared_state.is_logging_episode = (
                        episode_logging_freq > 0
                        and shared_state.episode_counter % episode_logging_freq == 0
                    )
                    transitions_to_send = []
                else:
                    if transitions_to_send:
                        push_transitions_to_transport_queue(transitions_to_send, transitions_queue)
                        transitions_to_send = []

                    intervention_rate = episode_intervention_steps / max(episode_total_steps, 1)
                    interactions_queue.put(
                        python_object_to_bytes({
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention_steps > 0),
                            "Intervention rate": intervention_rate,
                        })
                    )

                sum_reward_episode = 0.0
                episode_intervention_steps = 0
                episode_total_steps = 0

            _t_post_step_end = time.perf_counter()
            shared_state.add_env_post_step_detail(
                post_step=_t_post_step_end - _t_post_step_start,
                move_cpu=_t_move_cpu_end - _t_move_cpu_start,
                rerun=_t_rerun_end - _t_rerun_start,
            )

            dt_s = time.perf_counter() - start_time
            shared_state.env_active_time_total += dt_s
            shared_state.add_env_step_breakdown(
                action_get=_t_action_end - _t_action_start,
                step=_t_step_end - _t_step_start,
            )
            shared_state.add_env_wait_time(max(0.0, action_interval - dt_s))
            precise_sleep(max(0.0, action_interval - dt_s))

        logger.info("[RTC_ENV] Thread shut down.")
    except Exception as exc:
        logger.error("[RTC_ENV] Fatal exception: %s", exc)
        logger.error(traceback.format_exc())



def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().flatten()[0].item())
    return bool(value)


def _add_transition_to_replay_buffer(replay_buffer: ReplayBuffer, transition: Transition, action_dim: int) -> None:
    action = transition[ACTION]
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.float32)
    action = action.detach().cpu()[..., :action_dim]
    if action.ndim == 1:
        action = action.unsqueeze(0)
    replay_buffer.add(
        state=transition["state"],
        action=action,
        reward=float(transition["reward"]),
        next_state=transition["next_state"],
        done=_as_bool(transition["done"]),
        truncated=_as_bool(transition.get("truncated", False)),
        complementary_info=transition.get("complementary_info", {}),
    )


def _save_log_image(value: torch.Tensor, path: str) -> None:
    img = value.detach().cpu()
    while img.ndim > 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    if img.ndim not in (2, 3):
        return
    img_np = img.numpy()
    if img_np.dtype != np.uint8:
        if np.nanmax(img_np) <= 5.0:
            img_np = img_np * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np[..., 0]
    Image.fromarray(img_np).save(path)


def _finalize_rtc_inference_log(
    *,
    episode_log_buffer: list[dict],
    trainer: Trainer | None,
    policy: nn.Module | None,
    cfg,
    log_dir: str,
    episode_counter: int,
    video_logging_cameras: list[str],
) -> None:
    if not episode_log_buffer:
        return

    os.makedirs(log_dir, exist_ok=True)
    critic_values: list[float] = []
    critic_subsample = max(1, int(getattr(cfg, "critic_subsample", 1) or 1))

    logger.info("[RTC_INFERENCE] Saving %d frames for episode %d", len(episode_log_buffer), episode_counter)
    for step_idx, frame in enumerate(episode_log_buffer):
        for key, value in frame["obs"].items():
            if "image" not in key or not isinstance(value, torch.Tensor):
                continue
            camera_name = key.split(".")[-1]
            if camera_name not in video_logging_cameras:
                continue
            _save_log_image(value, os.path.join(log_dir, f"step_{step_idx:06d}_{camera_name}.png"))

    if trainer is not None and policy is not None:
        logger.info("[RTC_INFERENCE] Running local critic values for episode %d", episode_counter)
        with torch.no_grad():
            for step_idx in range(0, len(episode_log_buffer), critic_subsample):
                frame = episode_log_buffer[step_idx]
                action = frame.get("action")
                if not isinstance(action, torch.Tensor):
                    action_dim = _action_dim(cfg)
                    action = torch.zeros(action_dim, dtype=torch.float32)
                transition = Transition(
                    state=frame["obs"],
                    action=action,
                    reward=float(frame.get("reward", 0.0)),
                    next_state=frame["obs"],
                    done=bool(frame.get("done", False)),
                    truncated=False,
                    complementary_info={},
                )
                try:
                    value = trainer.critic_value_for_logging(
                        policy=policy,
                        transition=transition,
                        device=str(cfg.policy.device),
                        cfg=cfg,
                    )
                except Exception as exc:
                    logger.warning("[RTC_INFERENCE] Critic logging failed at step %d: %s", step_idx, exc)
                    value = None
                if value is not None:
                    critic_values.append(float(value))

    with open(os.path.join(log_dir, "critic_values.json"), "w") as f:
        json.dump(critic_values, f)

    if critic_values:
        plt.figure(figsize=(10, 5))
        plt.plot(critic_values)
        plt.title(f"Critic Values - Episode {episode_counter}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "critic_plot.png"))
        plt.close()

    try:
        save_video_with_critic_overlay(
            log_dir,
            critic_values,
            camera_names=video_logging_cameras,
            fps=cfg.env.fps,
            subtask_texts=[frame.get("subtask_text", "") for frame in episode_log_buffer],
            subsample=critic_subsample,
        )
        logger.info("[RTC_INFERENCE] Video generated for episode %d", episode_counter)
    except Exception as exc:
        logger.error("[RTC_INFERENCE] Failed to generate video: %s", exc)


def _maybe_save_inference_dataset(shared_state: RTCSharedState, cfg) -> None:
    if shared_state.replay_buffer is None:
        return
    episode_save_freq = int(getattr(cfg, "episode_save_freq", 0) or 0)
    if episode_save_freq <= 0 or shared_state.episode_counter % episode_save_freq != 0:
        return
    dataset_root = os.path.join(cfg.output_dir, "inference_dataset")
    logger.info("[RTC_INFERENCE] Saving inference dataset at episode %d", shared_state.episode_counter)
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    try:
        shared_state.replay_buffer.to_lerobot_dataset(
            repo_id="inference_recorded",
            fps=cfg.env.fps,
            root=dataset_root,
            task_name=cfg.policy.task,
        )
    except Exception as exc:
        logger.error("[RTC_INFERENCE] Failed to save inference dataset: %s", exc)


def _warmup_compiled_policy(policy, trainer, preprocessor, cfg, device, n_calls: int = 3) -> None:
    logger.info("[RTC_INFERENCE] Warming up compiled policy (%d calls) — please wait...", n_calls)
    task_str = cfg.policy.task
    execution_horizon = policy.config.rtc_config.execution_horizon
    dummy_obs = {
        key: torch.zeros(tuple(feat.shape), dtype=torch.float32)
        for key, feat in cfg.policy.input_features.items()
    }
    try:
        dummy_batch = trainer.build_inference_batch(dummy_obs, task_str, cfg, preprocessor=preprocessor)
        with torch.no_grad():
            for i in range(n_calls):
                torch.compiler.cudagraph_mark_step_begin()
                policy.predict_action_chunk(
                    dummy_batch,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                    execution_horizon=execution_horizon,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                logger.info("[RTC_INFERENCE] Warmup %d/%d complete.", i + 1, n_calls)
        if hasattr(policy, "reset"):
            policy.reset()
        logger.info("[RTC_INFERENCE] Warmup done. Policy is ready.")
    except Exception as e:
        logger.warning("[RTC_INFERENCE] Warmup failed (%s); first inference call will be slow.", e)


def act_with_policy_rtc_inference(
    cfg,
    trainer: Trainer,
    shutdown_event,
) -> None:
    """Run standalone generic inference with the shared RTC ActionQueue runtime."""
    set_seed(cfg.seed)
    device_name = getattr(cfg.policy, "actor_device", None) or cfg.policy.device
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = str(device)

    if getattr(cfg.policy, "rtc_config", None) is None:
        cfg.policy.rtc_config = RTCConfig()
    cfg.policy.rtc_config.enabled = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if getattr(cfg, "use_rerun", False):
        import rerun as rr
        rr.init("lerobot_inference", spawn=True)

    logger.info("[RTC_INFERENCE] Building policy and processors...")
    policy = trainer.make_policy(cfg)
    should_init_critic = not bool(getattr(cfg, "skip_critic", False)) and int(getattr(cfg, "episode_logging_freq", 0) or 0) > 0
    if should_init_critic and not hasattr(policy, "critic"):
        init_critic = getattr(policy, "init_critic", None)
        if callable(init_critic):
            init_critic()
    policy = policy.to(device).eval()
    preprocessor, postprocessor = trainer.make_processors(cfg)
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    if getattr(cfg.policy, "torch_compile", False):
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        try:
            action_expert = policy._action_expert()
            action_expert.forward_with_context = torch.compile(
                getattr(action_expert, "forward_with_context"),
                mode="reduce-overhead",
                fullgraph=False,
            )
            logger.info("[RTC_INFERENCE] torch.compile applied to action expert.")
        except Exception as e:
            logger.warning("[RTC_INFERENCE] Could not compile action expert: %s", e)
    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        raise RuntimeError("RTC inference requires policy.config.rtc_config.enabled=True")

    logger.info("[RTC_INFERENCE] Setting up environment...")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(
        online_env, teleop_device, cfg.env, cfg.policy.device
    )

    shared = RTCSharedState()
    shared.running = not shutdown_event.is_set()
    shared.replay_buffer = ReplayBuffer(
        capacity=cfg.policy.online_buffer_capacity,
        device=str(device),
        state_keys=cfg.policy.input_features.keys(),
        storage_device="cpu",
    )
    action_queue = ActionQueue(policy.config.rtc_config)

    inf_thread = Thread(
        target=rtc_inference_worker,
        args=(policy, trainer, preprocessor, postprocessor, shared, action_queue, None, device, cfg),
        daemon=True,
        name="rtc_inference",
    )
    env_thread = Thread(
        target=rtc_env_worker,
        args=(
            online_env, env_processor, action_processor, action_queue, shared,
            teleop_device, None, None, cfg, postprocessor,
        ),
        kwargs={"standalone": True, "policy": policy, "trainer": trainer},
        daemon=True,
        name="rtc_env",
    )

    try:
        env_thread.start()
        time.sleep(1.0)
        inf_thread.start()

        start_time = time.time()
        logger.info("[RTC_INFERENCE] Threads running. Supervisor loop active.")
        while not shutdown_event.is_set():
            time.sleep(20)

            q_size = action_queue.qsize()
            teleop_stat = "ON" if shared.is_intervening else "OFF"
            metrics = shared.get_and_reset_metrics()
            env_steps = max(1, metrics["env_steps"])
            inf_count = max(1, metrics["inference_count"])
            avg_env_active = metrics["env_active_time"] / env_steps
            avg_env_wait = metrics["env_wait_time"] / env_steps
            avg_action_get = metrics["env_action_get_time"] / env_steps
            avg_env_step = metrics["env_step_time"] / env_steps
            avg_robot_step = metrics["env_robot_step_time"] / env_steps
            avg_obs_proc = metrics["env_obs_proc_time"] / env_steps
            avg_action_proc = metrics["env_action_proc_time"] / env_steps
            avg_post_step = metrics["env_post_step_time"] / env_steps
            avg_move_cpu = metrics["env_move_cpu_time"] / env_steps
            avg_rerun = metrics["env_rerun_time"] / env_steps
            avg_pre = metrics["inference_preprocess_time"] / inf_count
            avg_model = metrics["inference_model_time"] / inf_count
            avg_post = metrics["inference_postprocess_time"] / inf_count
            inf_lats = metrics["inference_latencies"]
            lat_str = f"avg={sum(inf_lats)/len(inf_lats):.3f}s max={max(inf_lats):.3f}s" if inf_lats else "N/A"

            logger.info(
                "[MAIN LOG] Queue Buffer Length: %s | Teleop Intervention: %s | Runtime: %ss",
                q_size, teleop_stat, int(time.time() - start_time),
            )
            logger.info(
                "[metrics/inference] cycles=%d | sleep=%.2fs | preprocess=%.1fms | model=%.1fms | post=%.1fms | total=%s",
                metrics["inference_count"], metrics["inference_wait_time"],
                avg_pre * 1000, avg_model * 1000, avg_post * 1000, lat_str,
            )
            logger.info(
                "[metrics/env] steps=%d | action_get=%.1fms | env_step=%.1fms (action_proc=%.1fms robot_step=%.1fms obs_proc=%.1fms) | active=%.1fms | sleep=%.1fms | starved=%d | episode=%s",
                metrics["env_steps"], avg_action_get * 1000, avg_env_step * 1000,
                avg_action_proc * 1000, avg_robot_step * 1000, avg_obs_proc * 1000,
                avg_env_active * 1000, avg_env_wait * 1000, metrics["queue_starvation_count"],
                "ACTIVE" if shared.episode_active else "IDLE",
            )
            logger.info(
                "[metrics/env_post] post_step=%.1fms (move_cpu=%.1fms rerun=%.1fms other=%.1fms)",
                avg_post_step * 1000, avg_move_cpu * 1000, avg_rerun * 1000,
                (avg_post_step - avg_move_cpu - avg_rerun) * 1000,
            )
    except Exception:
        logger.error("[RTC_INFERENCE] Orchestrator error:\n%s", traceback.format_exc())
    finally:
        shutdown_event.set()
        shared.running = False
        for thread in (inf_thread, env_thread):
            thread.join(timeout=5.0)
        try:
            online_env.close()
        except Exception:
            pass
        logger.info("[RTC_INFERENCE] Shutdown complete.")

def act_with_policy_rtc(
    cfg,
    trainer: Trainer,
    shutdown_event,
    parameters_queue,
    transitions_queue,
    interactions_queue,
) -> None:
    """Run the generic online actor using the tested RTC ActionQueue runtime."""
    set_seed(cfg.seed)
    device_name = getattr(cfg.policy, "actor_device", None) or cfg.policy.device
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = str(device)

    if getattr(cfg.policy, "rtc_config", None) is None:
        cfg.policy.rtc_config = RTCConfig()
    cfg.policy.rtc_config.enabled = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("[RTC_ACTOR] Building actor policy and processors...")
    policy = trainer.make_actor_policy(cfg).to(device).eval()
    preprocessor, postprocessor = trainer.make_processors(cfg)
    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor

    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        raise RuntimeError("RTC runtime requires policy.config.rtc_config.enabled=True")

    logger.info("[RTC_ACTOR] Setting up environment...")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(
        online_env, teleop_device, cfg.env, cfg.policy.device
    )

    shared = RTCSharedState()
    shared.running = not shutdown_event.is_set()
    action_queue = ActionQueue(policy.config.rtc_config)

    inf_thread = Thread(
        target=rtc_inference_worker,
        args=(policy, trainer, preprocessor, postprocessor, shared, action_queue, parameters_queue, device, cfg),
        daemon=True,
        name="rtc_inference",
    )
    env_thread = Thread(
        target=rtc_env_worker,
        args=(
            online_env, env_processor, action_processor, action_queue, shared,
            teleop_device, transitions_queue, interactions_queue, cfg, postprocessor,
        ),
        daemon=True,
        name="rtc_env",
    )

    try:
        env_thread.start()
        time.sleep(1.0)
        inf_thread.start()

        logger.info("[RTC_ACTOR] Supervisor loop active.")
        t_start = time.time()
        while not shutdown_event.is_set():
            time.sleep(5)
            q_size = action_queue.qsize()
            teleop_stat = "ON" if shared.is_intervening else "OFF"
            episode_stat = "ON" if shared.episode_active else "OFF"
            metrics = shared.get_and_reset_metrics()
            env_steps = max(1, metrics["env_steps"])
            inf_lats = metrics.get("inference_latencies", [])
            if inf_lats:
                lat_str = f"avg={sum(inf_lats)/len(inf_lats):.3f}s min={min(inf_lats):.3f}s max={max(inf_lats):.3f}s"
            else:
                lat_str = "N/A"
            logger.info(
                "[RTC_ACTOR] runtime=%ss q=%s teleop=%s episode=%s env_steps=%s avg_env_active=%.4fs chunk_latency=%s",
                int(time.time() - t_start), q_size, teleop_stat, episode_stat,
                metrics["env_steps"], metrics["env_active_time"] / env_steps, lat_str,
            )
    except Exception:
        logger.error("[RTC_ACTOR] Orchestrator error:\n%s", traceback.format_exc())
    finally:
        shutdown_event.set()
        shared.running = False
        for t in (inf_thread, env_thread):
            t.join(timeout=5.0)
        try:
            online_env.close()
        except Exception:
            pass
        logger.info("[RTC_ACTOR] Shutdown complete.")
