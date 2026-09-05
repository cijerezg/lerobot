from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue
from pathlib import Path
import threading
import time
import traceback
from collections import deque
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
from lerobot.rl.actor import push_transitions_to_transport_queue
from lerobot.rl.buffer import ReplayBuffer, assemble_history_windows
from lerobot.rl.online_recorder import OnlineEpisodeRecorder
from lerobot.rl.gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.rl.inference_utils import bound_policy_actions, convert_env_obs_to_policy_format
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.rl.rl_trainer import Trainer
from lerobot.rl.subtask_console import make_subtask_console
from lerobot.rl.utils import save_video_with_critic_overlay
from lerobot.rollout.inference.rtc import _normalize_prev_actions_length
from lerobot.teleoperators.utils import TeleopEvents, TeleopFeedbackError
from lerobot.transport.utils import bytes_to_state_dict, python_object_to_bytes
from lerobot.utils.action_smoothing import apply_butterworth_filter
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition, move_state_dict_to_device, move_transition_to_device

logger = logging.getLogger(__name__)


def _has_anchor_decode(postprocessor) -> bool:
    """True when the postprocessor contains an AnchorDecodeStep (molmoact2 anchor/delta path)."""
    return any(
        getattr(step, "_registry_name", None) == "anchor_decode"
        for step in getattr(postprocessor, "steps", [])
    )


def _action_dim(cfg) -> int:
    if hasattr(cfg.policy, "action_dim"):
        return int(cfg.policy.action_dim)
    action_feat = getattr(cfg.policy, "output_features", {}).get("action")
    if action_feat is not None and getattr(action_feat, "shape", None):
        return int(action_feat.shape[0])
    return int(next(iter(cfg.policy.output_features.values())).shape[0])


_JOINT_ORDER = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_yaw.pos", "wrist_roll.pos", "gripper.pos",
]


def _raw_joint_action(online_env, action_dim: int, device) -> torch.Tensor:
    if hasattr(online_env, "get_raw_joint_positions"):
        raw_joints = online_env.get_raw_joint_positions()
        vals = [float(raw_joints.get(k, 0.0)) for k in _JOINT_ORDER[:action_dim]]
        if len(vals) < action_dim:
            vals.extend([0.0] * (action_dim - len(vals)))
        return torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.zeros(action_dim, dtype=torch.float32, device=device)


def _wait_for_first_chunk(action_queue, shared_state, fps: float, timeout_s: float = 5.0) -> None:
    """Hold the executor until the inference thread has landed a chunk. Otherwise an
    episode start, or a return from intervention, opens with one chunk latency of
    starvation steps (11 of the first 92 steps on 2026-09-04), each snapping the target
    to the raw pose."""
    deadline = time.perf_counter() + timeout_s
    while shared_state.running and action_queue.empty() and time.perf_counter() < deadline:
        time.sleep(1.0 / fps)
    if action_queue.empty():
        logger.warning("[RTC_ENV] No chunk within %.1fs; stepping with the starvation fallback.", timeout_s)


def _teleop_supports_feedback(teleop_device) -> bool:
    """Whether a teleoperator can safely receive position targets."""
    return (
        teleop_device is not None
        and bool(getattr(teleop_device, "feedback_features", {}))
        and callable(getattr(teleop_device, "send_feedback", None))
        and callable(getattr(teleop_device, "enable_torque", None))
        and callable(getattr(teleop_device, "disable_torque", None))
    )


def _validate_inference_action_routing(online_env, teleop_device) -> None:
    """Fail before episode startup unless follower actions can be routed to the leader exactly."""
    if not _teleop_supports_feedback(teleop_device):
        raise RuntimeError(
            "inference_send_actions_to_robot=false requires an actuated teleoperator with "
            "feedback_features, enable_torque(), and disable_torque()"
        )
    if not hasattr(online_env, "get_last_requested_joint_targets"):
        raise RuntimeError(
            "inference_send_actions_to_robot=false requires the real-robot RobotEnv action target path"
        )

    expected_keys = set(online_env.robot.action_features)
    feedback_keys = set(teleop_device.feedback_features)
    if feedback_keys != expected_keys:
        raise RuntimeError(
            "leader feedback joints do not exactly match follower action joints: "
            f"missing={sorted(expected_keys - feedback_keys)}, "
            f"extra={sorted(feedback_keys - expected_keys)}"
        )


def _close_robot_hardware(online_env, teleop_device, log_prefix: str) -> None:
    """Best-effort shutdown with the actuated teleoperator unloaded first."""
    try:
        if teleop_device is not None and getattr(teleop_device, "is_connected", False):
            teleop_device.disconnect()
    except Exception:
        logger.error("[%s] Teleoperator disconnect failed:\n%s", log_prefix, traceback.format_exc())
    try:
        online_env.close()
    except Exception:
        logger.error("[%s] Robot environment close failed:\n%s", log_prefix, traceback.format_exc())


def _ramp_leader(teleop_device, target: dict[str, float], fps: float, deg_per_s: float = 20.0) -> None:
    """Move the actuated leader onto `target` ({joint}.pos -> follower degrees).

    Linear, one send_feedback per control tick, so the gap never reaches the driver's
    per-step ceiling (a fault, not a clip). Call after enable_torque(). Approaches the
    follower's pose before an episode; descends to the park pose after it in safety mode.
    """
    target = {k: float(v) for k, v in target.items()}
    current = {k: float(v) for k, v in teleop_device.get_action().items() if k in target}
    gap = max(abs(target[k] - current[k]) for k in current)
    steps = max(int(fps), math.ceil(fps * gap / deg_per_s))
    logger.info("[RTC_ENV] Leader ramp: largest gap %.1f deg, %.1f s", gap, steps / fps)
    for step in range(1, steps + 1):
        t = step / steps
        teleop_device.send_feedback({k: current[k] + (target[k] - current[k]) * t for k in current})
        precise_sleep(1.0 / fps)


def _return_to_rest(online_env, teleop_device, fps: float, leader_torqued: bool) -> None:
    """Both arms to the follower's park pose; the leader is released there.

    Follower: `park()` (freeze, descent at park_deg_per_s, arrival check); torque stays on,
    so it holds the rest pose until the next episode. The leader rides the same ramp
    through park's per-tick hook, which also keeps its feedback watchdog fed. In safety
    mode (follower not commanded) only the leader ramps. A leader that faults during the
    descent stays where the fault unloaded it; the follower still parks.
    """
    robot = online_env.robot
    leader_alive = leader_torqued

    def feed_leader(target: dict[str, float]) -> None:
        nonlocal leader_alive
        if not leader_alive:
            return
        try:
            teleop_device.send_feedback(target)
        except TeleopFeedbackError as error:
            leader_alive = False
            logger.error("[RTC_ENV] Leader dropped out of the descent: %s", error)

    if online_env.send_actions_to_robot:
        try:
            robot.park(on_step=feed_leader)
        except RuntimeError:
            logger.error("[RTC_ENV] Follower did not reach the rest pose; torque stays on:\n%s", traceback.format_exc())
    elif leader_alive:
        park_pose = {f"{name}.pos": pos for name, pos in robot.config.park_pose.items()}
        try:
            _ramp_leader(teleop_device, park_pose, fps)
        except TeleopFeedbackError as error:
            logger.error("[RTC_ENV] Leader dropped out of the descent: %s", error)
    if leader_torqued:
        teleop_device.disable_torque()


def _obs_with_depth(policy_obs: dict, env_obs: dict, cfg) -> dict:
    """Carry raw {cam}.depth from the env observation into the inference observation under the
    canonical observation.depth.{cam} key (convert_env_obs_to_policy_format drops depth; values
    stay raw — the point-map builder consumes metric depth). Gated on the POLICY's pointmap_config,
    not the camera's use_depth. Returns a shallow copy when depth is added so the caller's dict
    (also used for transition storage / episode logs) stays depth-free.
    """
    if getattr(cfg.policy, "pointmap_config", None) is None:
        return policy_obs
    depth = {
        f"observation.depth.{key[: -len('.depth')]}": val
        for key, val in env_obs.items()
        if isinstance(key, str) and key.endswith(".depth")
    }
    return {**policy_obs, **depth} if depth else policy_obs


def _rig_history_offsets(history_offsets: dict[str, list[int]] | None, cfg) -> dict[str, list[int]] | None:
    """history_keys name every canonical role; keep the ones this rig serves (the prune
    align_rebot_buffer applies to the replay). The policy side pads the absent roles."""
    if history_offsets is None:
        return None
    env = cfg.env
    served = {env.features_map.get(key, key) for key in env.features} | {ACTION}
    for cam, camera in getattr(env.robot, "cameras", {}).items():
        if getattr(camera, "use_depth", False):
            image_key = f"{OBS_IMAGES}.{cam}"
            served.add(f"depth.{env.features_map.get(image_key, image_key).rsplit('.', 1)[-1]}.depth")
    kept = {key: offsets for key, offsets in history_offsets.items() if key in served}
    dropped = sorted(set(history_offsets) - set(kept))
    if dropped:
        logger.info("[RTC] no rig column for history keys %s; skipped", dropped)
    return kept or None


class RTCSharedState:
    """Thread-safe state manager for the RTC actor runtime."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_obs: dict | None = None
        self.is_intervening = False
        self.episode_active = False
        self.history_offsets: dict[str, list[int]] | None = None
        self.history_entries: deque | None = None
        # Current subtask: decoded by the high-level query, or latched by the
        # operator console. default_subtask is what an episode reset falls back to
        # (None until the console installs its first binding).
        self.current_subtask_name: str | None = None
        self.current_subtask_index: int = -1
        self.default_subtask: tuple[str | None, int] = (None, -1)
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
        self.params_loaded_event = threading.Event()
        self.params_loaded_event.set()
        self.policy_ready_event = threading.Event()

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

    def configure_history(self, history_offsets: dict[str, list[int]] | None) -> None:
        """Enable short-term memory: the env worker pushes one completed (state, action)
        entry per control step; the deque holds exactly the largest lookback distance."""
        self.history_offsets = history_offsets
        if history_offsets is not None:
            max_back = max(offsets[0] for offsets in history_offsets.values())
            self.history_entries = deque(maxlen=max_back)

    def update_subtask(self, name: str, index: int) -> None:
        with self.lock:
            self.current_subtask_name = name
            self.current_subtask_index = index

    def subtask_snapshot(self) -> tuple[str | None, int]:
        with self.lock:
            return self.current_subtask_name, self.current_subtask_index

    def set_default_subtask(self, name: str, index: int) -> None:
        with self.lock:
            self.default_subtask = (name, index)
            self.current_subtask_name = name
            self.current_subtask_index = index

    def clear_subtask_state(self) -> None:
        with self.lock:
            self.current_subtask_name, self.current_subtask_index = self.default_subtask

    def push_history(self, entry: dict) -> None:
        with self.lock:
            self.history_entries.append(entry)

    def clear_history(self) -> None:
        with self.lock:
            if self.history_entries is not None:
                self.history_entries.clear()

    def history_snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.history_entries)


def _rerun_log_worker(q) -> None:
    """Daemon thread: drains the rerun queue and logs via log_rerun_data — same entity naming
    as teleop/record (observation.{cam}, observation.{cam}.depth, observation/action.{joint}.pos)
    so the viewer layout matches what teleop shows. Does not block the env loop."""
    import rerun as rr

    from lerobot.utils.visualization_utils import log_rerun_data

    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break
        step, observation, action = item
        try:
            rr.set_time_sequence("step", step)
            log_rerun_data(observation=observation, action=action)
        except Exception:
            logger.exception("[RERUN] logging failed for step %s (worker keeps running)", step)


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
    normalizer,
) -> torch.Tensor:
    """Re-align leftover normalized actions when anchor state changes.

    Operates entirely in the model's encoded-delta frame: uses the normalizer for
    both directions (inverse to unnormalize, forward to renormalize). anchor_old
    and anchor_now must be in the same frame as the normalizer stats.
    """
    n_left = prev_actions.shape[0]
    action_dim = prev_actions.shape[1]
    offset = chunk_size - n_left

    if action_encoding == "delta" and offset > 0:
        return prev_actions

    right_padded = torch.zeros(chunk_size, action_dim, device=prev_actions.device, dtype=prev_actions.dtype)
    right_padded[offset:] = prev_actions
    # Batched [1, T, D] through the normalizer: per-row stats read the batch axis from
    # dim 0, and a bare [T, D] chunk would be gathered as T samples.
    d_abs = normalizer._normalize_action(right_padded[None], inverse=True)[0]

    dev = d_abs.device
    delta_s = anchor_old.squeeze(0).to(dev) - anchor_now.squeeze(0).to(dev)
    if action_encoding == "anchor":
        d_abs[offset:] += delta_s
    else:
        d_abs[0] += delta_s

    left_padded = torch.zeros(chunk_size, action_dim, device=dev, dtype=d_abs.dtype)
    left_padded[:n_left] = d_abs[offset:]
    return normalizer._normalize_action(left_padded[None], inverse=False)[0][:n_left]


def _resolve_prev_actions_and_anchor(
    *,
    prev_actions: torch.Tensor | None,
    action_queue: ActionQueue,
    latest_obs: dict,
    policy,
    stats_rows: dict | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (prev_actions, anchor_now) ready for the next RTC inference call.

    Aligns prev_actions to the current anchor when the encoded delta references
    a stale s_0. For molmoact2 (pipeline ends in v2.1) both anchors are mapped
    into v2.1 before being subtracted, so delta_s matches the normalizer frame.
    Returns (prev_actions, None) for absolute encoding or when OBS_STATE is missing.
    """
    action_encoding = getattr(policy.config, "action_encoding", "absolute")
    anchor_now = None
    if action_encoding not in {"anchor", "delta"}:
        return prev_actions, anchor_now

    from lerobot.utils.constants import OBS_STATE

    if OBS_STATE not in latest_obs:
        return prev_actions, anchor_now

    # The chunk is the policy's padded width, the state is the rig's own; zero-fill the
    # pad columns, what the unified layout step padded the training anchor with.
    anchor_now = latest_obs[OBS_STATE]
    width = int(policy.config.output_features[ACTION].shape[-1])
    anchor_now = torch.nn.functional.pad(anchor_now, (0, width - anchor_now.shape[-1]))
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
    logger.debug(
        "[RTC] Alignment offset: %.3f",
        (anchor_old.to(anchor_now.device) - anchor_now).norm().item(),
    )
    # Per-row stats: the leftover is unnormalized and renormalized with the row the
    # batch was normalized with (stats_rows = the batch's index columns).
    context = getattr(normalizer, "_with_embodiment_indices", None)
    with (
        context({TransitionKey.COMPLEMENTARY_DATA: stats_rows})
        if context is not None and stats_rows
        else contextlib.nullcontext()
    ):
        aligned = align_prev_actions(
            prev_actions=prev_actions,
            anchor_old=anchor_old,
            anchor_now=anchor_now,
            action_encoding=action_encoding,
            chunk_size=policy.config.chunk_size,
            normalizer=normalizer,
        )
    return aligned, anchor_now


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
    post_inference_hook=None,
) -> None:
    """Background inference worker using RTC ActionQueue semantics.

    ``post_inference_hook(latest_obs)`` (optional) is called once per cycle, in
    this thread, after the action chunk is merged. It runs in-thread on purpose:
    some policies' attention/probe capture flips process-global state and patches
    modules the live forward also uses, so it must not overlap inference.
    """
    try:
        logger.info("[RTC_INFERENCE] Thread started.")
        _warmup_policy(policy, trainer, preprocessor, cfg, device, shared_state)
        shared_state.policy_ready_event.set()
        latency_tracker = LatencyTracker()
        inference_step = 0
        execution_horizon = policy.config.rtc_config.execution_horizon
        time_per_chunk = 1.0 / cfg.env.fps
        task_str = cfg.policy.task
        action_dim = _action_dim(cfg)
        # Operator-fed subtasks REPLACE generation: the console latches the current
        # step into shared_state, so the clause is still rendered but never decoded.
        operator_subtasks = bool(getattr(cfg.policy, "eval_subtasks", None))
        subtask_enabled = operator_subtasks or int(getattr(cfg.policy, "subtask_max_new_tokens", 0)) > 0
        subtask_interval = float(getattr(cfg.policy, "subtask_regeneration_interval", 1.0))
        last_subtask_time: float | None = None
        # Metadata steering at inference = prompt the best behavior (π0.7: quality 5,
        # no mistakes; speed omitted — the clause renders partially).
        memory_cfg = getattr(cfg.policy, "memory", None)
        inference_metadata = (
            {"quality": 5, "mistake": False}
            if memory_cfg is not None and memory_cfg.metadata_enabled
            else None
        )

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
                last_subtask_time = None
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

            # observation.depth.* is not an input_feature; it is only present when the env worker
            # injected it (pointmap_config set), so carrying it through keeps inference depth-aware.
            obs_filtered = {
                k: v
                for k, v in latest_obs.items()
                if k in cfg.policy.input_features or k.startswith("observation.depth.")
            }
            if shared_state.history_offsets is not None:
                # Expose current depth under the buffer-canonical key so the
                # episode-start fallback (empty deque) can repeat it.
                current = dict(latest_obs)
                for k, v in latest_obs.items():
                    if k.startswith("observation.depth."):
                        current[f"depth.{k.removeprefix('observation.depth.')}.depth"] = v.unsqueeze(0)
                obs_filtered.update(
                    assemble_history_windows(
                        shared_state.history_snapshot(),
                        shared_state.history_offsets,
                        current,
                        action_dim,
                    )
                )
            robot_type = cfg.env.robot.type if hasattr(cfg.env, "robot") else ""

            current_time = time.perf_counter()
            with torch.no_grad():
                # Subtask generation at cadence (two-prompt design). Runs in THIS
                # thread — generation is not safe against a concurrent worker.
                if subtask_enabled and not operator_subtasks and (
                    last_subtask_time is None
                    or (current_time - last_subtask_time) >= subtask_interval
                ):
                    raw_text, subtask_name, subtask_index = trainer.generate_subtask_text(
                        policy, obs_filtered, task_str, cfg, preprocessor=preprocessor,
                    )
                    shared_state.update_subtask(subtask_name, subtask_index)
                    last_subtask_time = current_time
                    if subtask_index < 0:
                        logger.warning("[RTC_INFERENCE] Subtask snap missed vocab: %r", raw_text)

                current_subtask, _ = shared_state.subtask_snapshot()
                t_preproc_start = time.perf_counter()
                processed_batch = trainer.build_inference_batch(
                    obs_filtered,
                    task_str,
                    cfg,
                    preprocessor=preprocessor,
                    robot_type=robot_type,
                    subtask=current_subtask if subtask_enabled else None,
                    metadata=inference_metadata,
                )
                t_preproc_end = time.perf_counter()

                action_index_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                prev_actions, anchor_now = _resolve_prev_actions_and_anchor(
                    prev_actions=prev_actions,
                    action_queue=action_queue,
                    latest_obs=latest_obs,
                    policy=policy,
                    stats_rows={
                        k: processed_batch[k]
                        for k in ("embodiment_index", "action_layout_id")
                        if k in processed_batch
                    },
                )
                if prev_actions is not None:
                    prev_actions = _normalize_prev_actions_length(prev_actions, target_steps=execution_horizon)

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
                if (
                    postprocessor is not None
                    and anchor_now is not None
                    and action_encoding in {"anchor", "delta"}
                    and _has_anchor_decode(postprocessor)
                ):
                    # molmoact2: thread the anchor through the postprocessor so
                    # AnchorDecodeStep reconstructs the absolute action for the robot.
                    from lerobot.policies.molmoact2.anchor_encoding import (
                        ANCHOR_KEY,
                        EMBODIMENT_INDEX_KEY,
                    )
                    anchor_sq = anchor_now.squeeze(0) if anchor_now.dim() > 1 else anchor_now
                    # Batched [1, T, D]: per-row stats read the batch axis from dim 0, and a
                    # bare [T, D] chunk would be gathered as T samples.
                    payload = {
                        ACTION: original_actions.unsqueeze(0),
                        ANCHOR_KEY: anchor_sq.to(original_actions.device),
                    }
                    # Per-embodiment stats: the unnormalizer must gather the same row the
                    # preprocessor used, else the robot receives another robot's scale.
                    if EMBODIMENT_INDEX_KEY in processed_batch:
                        payload[EMBODIMENT_INDEX_KEY] = processed_batch[EMBODIMENT_INDEX_KEY]
                    # A layout-keyed stats artifact selects its row by action_layout_id instead;
                    # the adapter carries the nested complementary dict through verbatim.
                    if "action_layout_id" in processed_batch:
                        payload[TransitionKey.COMPLEMENTARY_DATA] = {
                            "action_layout_id": processed_batch["action_layout_id"]
                        }
                    processed_actions = postprocessor(payload).squeeze(0)
                else:
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

                processed_actions = bound_policy_actions(processed_actions, latest_obs, policy)

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

            if post_inference_hook is not None:
                # Must never kill the action loop — the hook self-throttles and
                # swallows its own errors, but guard here too.
                try:
                    post_inference_hook(latest_obs)
                except Exception:
                    logger.warning("[RTC_INFERENCE] post_inference_hook failed:\n%s", traceback.format_exc())

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
    rerun_queue=None,
    recorder: OnlineEpisodeRecorder | None = None,
) -> None:
    """Environment interaction worker copied from the tested PI05 RTC path."""
    _ = postprocessor  # queued actions are already postprocessed in rtc_inference_worker
    leader_torqued = False  # our own enable/release bookkeeping; the leader unloads itself on faults
    try:
        logger.info("[RTC_ENV] Thread started.")
        action_interval = 1.0 / cfg.env.fps
        action_dim = _action_dim(cfg)
        teleop_feedback_supported = _teleop_supports_feedback(teleop_device)

        was_intervening = False
        sum_reward_episode = 0.0
        episode_intervention_steps = 0
        episode_total_steps = 0
        transitions_to_send: list[Transition] = []
        interaction_step = 0
        video_logging_cameras = list(getattr(cfg, "video_logging_cameras", ["top", "side"]))
        episode_log_buffer: list[dict] = []
        last_action: torch.Tensor | None = None
        episode_logging_freq = int(getattr(cfg, "episode_logging_freq", 0) or 0)
        shared_state.is_logging_episode = (episode_logging_freq > 0 and shared_state.episode_counter % episode_logging_freq == 0)

        while shared_state.running and not shared_state.policy_ready_event.wait(timeout=0.1):
            pass
        if not shared_state.running:
            return
        logger.info("[ACTOR] Press '2' to start episode.")
        while shared_state.running:
            if teleop_device.get_teleop_events().get(TeleopEvents.START_EPISODE, False):
                break
            time.sleep(0.1)
        if not shared_state.running:
            return

        obs, info = online_env.reset()
        env_processor.reset()
        action_processor.reset()

        raw_obs = obs
        transition = create_transition(observation=obs, info=info)
        transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
        transition = env_processor(transition)

        policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
        shared_state.update_observation(
            _obs_with_depth(policy_fmt_obs, transition[TransitionKey.OBSERVATION], cfg), False
        )
        if not standalone:
            shared_state.request_parameter_update()

            logger.info("[ACTOR] Loading new params, please wait before episode starts...")
            while shared_state.running:
                if shared_state.params_loaded_event.wait(timeout=0.5):
                    break
            logger.info("[ACTOR] Params loaded. Starting episode.")
        # Same ordering as the in-loop episode start: active before the approach so the
        # first chunk computes during it, then hold until it has landed.
        shared_state.set_episode_active(True)
        if teleop_feedback_supported:
            teleop_device.enable_torque()
            leader_torqued = True
            _ramp_leader(teleop_device, online_env.get_raw_joint_positions(), cfg.env.fps)
        _wait_for_first_chunk(action_queue, shared_state, cfg.env.fps)

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
                shared_state.clear_history()
                shared_state.clear_subtask_state()
                shared_state.request_reset()
                if not standalone:
                    shared_state.request_parameter_update()
                was_intervening = False
                last_action = None
                episode_log_buffer = []

                raw_obs = obs
                transition = create_transition(observation=obs, info=info)
                transition[TransitionKey.COMPLEMENTARY_DATA] = {"subtask": [""] * (len(obs) if isinstance(obs, list) else 1)}
                transition = env_processor(transition)
                policy_fmt_obs = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
                shared_state.update_observation(
                    _obs_with_depth(policy_fmt_obs, transition[TransitionKey.OBSERVATION], cfg), False
                )

                if not standalone:
                    logger.info("[ACTOR] Loading new params, please wait before episode starts...")
                    while shared_state.running:
                        if shared_state.params_loaded_event.wait(timeout=0.5):
                            break
                    logger.info("[ACTOR] Params loaded. Starting episode.")
                # Active BEFORE the leader approach: the observation is already published and
                # the follower does not move during the approach, so the first chunk computes
                # during that second instead of after it.
                shared_state.set_episode_active(True)
                if teleop_feedback_supported:
                    teleop_device.enable_torque()
                    leader_torqued = True
                    _ramp_leader(teleop_device, online_env.get_raw_joint_positions(), cfg.env.fps)
                _wait_for_first_chunk(action_queue, shared_state, cfg.env.fps)

            start_time = time.perf_counter()

            if was_intervening != shared_state.is_intervening:
                logger.info("[RTC_ENV] Teleop state changed (intervening=%s), resetting policy/queue", shared_state.is_intervening)
                shared_state.request_reset()
                action_queue.clear()
                if was_intervening and teleop_feedback_supported:
                    _ramp_leader(teleop_device, online_env.get_raw_joint_positions(), cfg.env.fps)
                if was_intervening:
                    _wait_for_first_chunk(action_queue, shared_state, cfg.env.fps)
            was_intervening = shared_state.is_intervening

            _t_action_start = time.perf_counter()
            if was_intervening:
                action = _raw_joint_action(online_env, action_dim, "cpu")
            else:
                action = action_queue.get()
                if action is not None:
                    action = action[..., :action_dim]
                else:
                    shared_state.add_queue_starvation()
                    # Hold the last executed target rather than snapping to the raw pose: with
                    # a frozen follower (safety mode) the raw pose is the episode-start pose,
                    # and the snap is a >8 raw-deg step that trips the leader's ceiling.
                    action = last_action if last_action is not None else _raw_joint_action(online_env, action_dim, "cpu")
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
            # A teleop override is the rig's own width; the policy's chunk (and the buffer's
            # action column) is the padded layout width. Zero-fill the pad columns.
            executed_action = torch.nn.functional.pad(executed_action, (0, action_dim - executed_action.shape[-1]))
            last_action = executed_action
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
                if not getattr(online_env, "send_actions_to_robot", True):
                    requested_targets = online_env.get_last_requested_joint_targets()
                    feedback = {
                        key: value.item() if isinstance(value, torch.Tensor) else float(value)
                        for key, value in (requested_targets or {}).items()
                    }
                else:
                    feedback = {}
                    for key, value in new_transition[TransitionKey.OBSERVATION].items():
                        if key.endswith(".pos"):
                            feedback[key] = value.item() if isinstance(value, torch.Tensor) else float(value)
                if feedback and teleop_feedback_supported:
                    try:
                        teleop_device.send_feedback(feedback)
                    except TeleopFeedbackError as error:
                        if getattr(online_env, "send_actions_to_robot", True):
                            raise
                        logger.error("[RTC_ENV] Leader stopped and unloaded; ending the episode: %s", error)
                        truncated = True

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

            # The generated subtask rides the buffer's canonical column so online
            # transitions look exactly like annotated offline frames to the learner.
            subtask_name_now, subtask_idx_now = shared_state.subtask_snapshot()
            complementary_info = {
                "discrete_penalty": torch.tensor([
                    new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                ]),
                TeleopEvents.IS_INTERVENTION.value: torch.tensor([float(is_intervening)], dtype=torch.float32),
                "subtask_index": torch.tensor([subtask_idx_now], dtype=torch.long),
            }
            # Only carry subtask tensors when the policy actually generates them
            # (pi05: max_decoding_steps>0). Empty (0,) tensors would later trip
            # compute_episode_stats during buffer→dataset serialization.
            if subtask_tokens.numel() > 0:
                complementary_info["subtask_tokens"] = subtask_tokens
                complementary_info["subtask_masks"] = subtask_masks

            # Carry raw CURRENT-state depth into the buffer under the offline cache's key
            # (depth.{cam}.depth) so online/offline batches concatenate without padding and the
            # trainer/sampler treat both identically (incl. next_depth.* for the critic target).
            # Pulled from the unfiltered transition so depth stays aligned with `state`; gated on
            # the policy's pointmap_config, not the camera's use_depth (mirrors rl/actor.py).
            if getattr(cfg.policy, "pointmap_config", None) is not None:
                for key, val in transition[TransitionKey.OBSERVATION].items():
                    if isinstance(key, str) and key.endswith(".depth"):
                        complementary_info[f"depth.{key}"] = val

            observation = convert_env_obs_to_policy_format(transition[TransitionKey.OBSERVATION])
            next_observation = convert_env_obs_to_policy_format(new_transition[TransitionKey.OBSERVATION])

            # Short-term memory: one completed (state, action) entry per control step,
            # the same (state, action, depth) this frame gets in the learner buffer.
            # Depth uses the buffer's canonical complementary key (depth.{cam}.depth),
            # pulled unbatched from the unfiltered transition like the block above.
            if shared_state.history_offsets is not None:
                entry = {}
                for k in shared_state.history_offsets:
                    if k == ACTION:
                        entry[k] = executed_action[..., :action_dim].reshape(1, -1)
                    elif k.startswith("depth."):
                        entry[k] = transition[TransitionKey.OBSERVATION][k.removeprefix("depth.")].unsqueeze(0)
                    else:
                        entry[k] = observation[k]
                shared_state.push_history(entry)
            # Images go on-wire as uint8 to match the offline buffer's uint8 storage
            # (so online/offline batches concatenate) and to shrink the gRPC payload 4x.
            state_send = {k: v.mul(255).clamp(0, 255).to(torch.uint8) if "image" in k else v for k, v in observation.items()}
            next_state_send = {k: v.mul(255).clamp(0, 255).to(torch.uint8) if "image" in k else v for k, v in next_observation.items()}
            transition_to_send = Transition(
                state=state_send,
                action=executed_action[..., :action_dim],
                reward=reward,
                next_state=next_state_send,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            _t_move_cpu_start = time.perf_counter()
            transition_cpu = move_transition_to_device(transition_to_send, "cpu")
            _t_move_cpu_end = time.perf_counter()
            if not standalone:
                transitions_to_send.append(transition_cpu)

            if recorder is not None:
                recorder.add_frame(
                    raw_obs, executed_action,
                    is_intervention=is_intervening, subtask_index=subtask_idx_now, subtask=subtask_name_now or "",
                )
                raw_obs = online_env.get_raw_observation()

            next_policy_fmt_obs = next_observation
            if standalone and shared_state.is_logging_episode:
                episode_log_buffer.append({
                    "obs": {
                        k: (transition_cpu["next_state"][k].clone() if "image" in k
                            else v.detach().cpu().clone())
                        for k, v in next_policy_fmt_obs.items()
                    },
                    "action": transition_cpu[ACTION].detach().cpu().clone() if isinstance(transition_cpu.get(ACTION), torch.Tensor) else transition_cpu.get(ACTION),
                    "reward": float(reward),
                    "done": bool(done),
                    "subtask_text": "",
                })
            _t_rerun_start = time.perf_counter()
            if getattr(cfg, "use_rerun", False) and rerun_queue is not None:
                # Bare robot keys (top, top.depth, {joint}.pos) — log_rerun_data namespaces them to
                # the same observation.*/action.* entities teleop/record show, so the viewer layout
                # matches teleop. Actions are the executed v3-frame joint targets.
                obs_log: dict = {}
                for key, val in next_policy_fmt_obs.items():
                    if "image" in key:
                        val_np = val[0].cpu().numpy() if val.ndim == 4 else val.cpu().numpy()
                        obs_log[key.removeprefix("observation.images.")] = val_np.transpose(1, 2, 0)
                for key, val in new_transition[TransitionKey.OBSERVATION].items():
                    if isinstance(key, str) and key.endswith(".depth"):
                        obs_log[key] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
                if hasattr(online_env, "get_raw_joint_positions"):
                    raw = online_env.get_raw_joint_positions()
                    obs_log.update({j_name: float(j_val) for j_name, j_val in raw.items()})
                act_np = executed_action.detach().float().cpu().numpy().reshape(-1)[:action_dim]
                act_log = {name: float(v) for name, v in zip(_JOINT_ORDER, act_np, strict=False)}
                with contextlib.suppress(queue.Full):
                    rerun_queue.put_nowait((interaction_step, obs_log, act_log))
            _t_rerun_end = time.perf_counter()

            shared_state.update_observation(
                _obs_with_depth(next_policy_fmt_obs, new_transition[TransitionKey.OBSERVATION], cfg),
                is_intervening,
            )
            transition = new_transition
            interaction_step += 1
            shared_state.current_step = interaction_step

            if done or truncated:
                logger.info(
                    "[ACTOR] Global step %s: Episode ended. reward=%s | done=%s truncated=%s | success=%s terminate=%s",
                    interaction_step, sum_reward_episode, done, truncated, info_success, info_terminate,
                )
                shared_state.set_episode_active(False)
                _return_to_rest(online_env, teleop_device, cfg.env.fps, leader_torqued)
                leader_torqued = False
                if recorder is not None:
                    recorder.save_episode()

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
    finally:
        if shared_state.episode_active:
            try:
                _return_to_rest(online_env, teleop_device, cfg.env.fps, leader_torqued)
            except Exception:
                logger.error("[RTC_ENV] Return to rest failed:\n%s", traceback.format_exc())
            shared_state.set_episode_active(False)
        if recorder is not None:
            recorder.finalize()




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
                # episode_log_buffer stores images as uint8; critic preprocessor
                # expects float32 in [0, 1], so convert image keys on the fly.
                obs_f32 = {
                    k: (v.float() / 255.0
                        if "image" in k and isinstance(v, torch.Tensor) and v.dtype == torch.uint8
                        else v)
                    for k, v in frame["obs"].items()
                }
                transition = Transition(
                    state=obs_f32,
                    action=action,
                    reward=float(frame.get("reward", 0.0)),
                    next_state=obs_f32,
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



def _warmup_observation(cfg, history_offsets: dict[str, list[int]] | None, action_dim: int) -> dict:
    """A deployment-shaped observation: this rig's cameras and state under their canonical
    names (roles the rig lacks stay absent), raw depth where the policy reads it, and the
    history windows of an episode start. The prompt, presence masks and history stack then
    match the live batch, so the kernels warmed are the ones the episode runs."""
    env = cfg.env
    obs = {}
    for key, feature in env.features.items():
        name = env.features_map.get(key, key)
        if name in cfg.policy.input_features:
            obs[name] = torch.zeros(1, *feature.shape, dtype=torch.float32)
    if getattr(cfg.policy, "pointmap_config", None) is not None:
        for cam, camera in getattr(env.robot, "cameras", {}).items():
            if getattr(camera, "use_depth", False):
                image_key = f"{OBS_IMAGES}.{cam}"
                role = env.features_map.get(image_key, image_key).rsplit(".", 1)[-1]
                obs[f"observation.depth.{role}"] = torch.zeros(camera.height, camera.width, dtype=torch.uint16)
    if history_offsets is not None:
        current = dict(obs)
        for key, value in obs.items():
            if key.startswith("observation.depth."):
                current[f"depth.{key.removeprefix('observation.depth.')}.depth"] = value.unsqueeze(0)
        obs.update(assemble_history_windows([], history_offsets, current, action_dim))
    return obs


def _warmup_policy(policy, trainer, preprocessor, cfg, device, shared_state: RTCSharedState, n_calls: int = 2) -> None:
    """First forwards on a deployment-shaped batch before the operator can start an
    episode: lazy CUDA init, cudnn autotune and (when enabled) compile/graph capture land
    here instead of in the first chunk, where they starve the queue. A failure here is the
    failure the first chunk would have hit, so it raises before any episode starts."""
    logger.info("[RTC_INFERENCE] Warming up policy (%d calls) - please wait...", n_calls + 1)
    execution_horizon = policy.config.rtc_config.execution_horizon
    action_dim = _action_dim(cfg)
    memory_cfg = getattr(cfg.policy, "memory", None)
    subtask, _ = shared_state.subtask_snapshot()
    batch = trainer.build_inference_batch(
        _warmup_observation(cfg, shared_state.history_offsets, action_dim),
        cfg.policy.task,
        cfg,
        preprocessor=preprocessor,
        robot_type=cfg.env.robot.type if hasattr(cfg.env, "robot") else "",
        subtask=subtask,
        metadata={"quality": 5, "mistake": False} if memory_cfg is not None and memory_cfg.metadata_enabled else None,
    )
    dummy_prev = torch.zeros(execution_horizon, action_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        # No-prefix calls (first chunk of an episode), then one with a leftover prefix.
        for i, prev in enumerate([None] * n_calls + [dummy_prev]):
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch.compiler.cudagraph_mark_step_begin()
            policy.predict_action_chunk(
                batch, inference_delay=0, prev_chunk_left_over=prev, execution_horizon=execution_horizon
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            logger.info("[RTC_INFERENCE] Warmup call %d/%d complete.", i + 1, n_calls + 1)
    if hasattr(policy, "reset"):
        policy.reset()
    logger.info("[RTC_INFERENCE] Warmup done. Policy is ready.")


def act_with_policy_rtc_inference(
    cfg,
    trainer: Trainer,
    shutdown_event,
    post_inference_hook_factory=None,
) -> None:
    """Run standalone generic inference with the shared RTC ActionQueue runtime.

    ``post_inference_hook_factory(policy, preprocessor, postprocessor, device, cfg)``
    (optional) is called once after the policy + processors are built; it returns
    a ``hook(latest_obs)`` callable (or None) run in the inference thread each
    cycle. Used by inference_w_attn to stream live attention maps.
    """
    set_seed(cfg.seed)
    device_name = getattr(cfg.policy, "actor_device", None) or cfg.policy.device
    device = get_safe_torch_device(device_name, log=True)
    cfg.policy.device = str(device)

    if getattr(cfg.policy, "rtc_config", None) is None:
        cfg.policy.rtc_config = RTCConfig()
    cfg.policy.rtc_config.enabled = True

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    rerun_queue = None
    if getattr(cfg, "use_rerun", False):
        import rerun as rr
        # Fresh app id (was "lerobot_inference"): the viewer caches blueprints per app id, and the
        # old id's blueprint predates the teleop-style entity naming (would render black panels).
        rr.init("lerobot_rtc_inference", spawn=True)
        rerun_queue = queue.Queue(maxsize=2)
        Thread(target=_rerun_log_worker, args=(rerun_queue,), daemon=True, name="rerun_log").start()

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
                action_expert.forward_with_context,
                mode="reduce-overhead",
                fullgraph=False,
            )
            logger.info("[RTC_INFERENCE] torch.compile applied to action expert.")
        except Exception as e:
            logger.warning("[RTC_INFERENCE] Could not compile action expert: %s", e)
    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        raise RuntimeError("RTC inference requires policy.config.rtc_config.enabled=True")

    logger.info("[RTC_INFERENCE] Setting up environment...")
    send_actions_to_robot = bool(getattr(cfg, "inference_send_actions_to_robot", True))
    if not send_actions_to_robot:
        logger.warning(
            "[RTC_INFERENCE] SAFETY MODE: follower actions are disabled; "
            "observations are still read and requested actions are routed to teleoperator feedback."
        )
    online_env, teleop_device = make_robot_env(
        cfg=cfg.env,
        send_actions_to_robot=send_actions_to_robot,
    )
    try:
        if not send_actions_to_robot:
            _validate_inference_action_routing(online_env, teleop_device)
        env_processor, action_processor = make_processors(
            online_env, teleop_device, cfg.env, cfg.policy.device
        )

        shared = RTCSharedState()
        shared.running = not shutdown_event.is_set()
        memory_cfg = getattr(cfg.policy, "memory", None)
        history_offsets = _rig_history_offsets(
            ReplayBuffer._normalize_history_offsets(
                memory_cfg.history_offsets(cfg.env.fps) if memory_cfg is not None else None
            ),
            cfg,
        )
        shared.configure_history(history_offsets)
        recorder = OnlineEpisodeRecorder(
            online_env.robot,
            root=Path(cfg.output_dir) / "inference_dataset",
            fps=cfg.env.fps,
            task=cfg.policy.task,
            # Depth PNGs land on the cache's read grid; a policy without a stride gets every frame.
            depth_stride=getattr(cfg.policy, "image_stride", 1),
        )
        action_queue = ActionQueue(policy.config.rtc_config)

        # Operator subtask console: free-text bindings, indexed against the checkpoint
        # vocabulary where they match (else -1) for the buffer's subtask_index column.
        subtask_console = make_subtask_console(cfg, trainer, preprocessor, shared)
        if subtask_console is not None:
            shared.set_default_subtask(*subtask_console.initial)

        post_inference_hook = None
        if post_inference_hook_factory is not None:
            post_inference_hook = post_inference_hook_factory(
                policy, preprocessor, postprocessor, device, cfg
            )

        inf_thread = Thread(
            target=rtc_inference_worker,
            args=(
                policy,
                trainer,
                preprocessor,
                postprocessor,
                shared,
                action_queue,
                None,
                device,
                cfg,
                post_inference_hook,
            ),
            daemon=True,
            name="rtc_inference",
        )
        env_thread = Thread(
            target=rtc_env_worker,
            args=(
                online_env,
                env_processor,
                action_processor,
                action_queue,
                shared,
                teleop_device,
                None,
                None,
                cfg,
                postprocessor,
            ),
            kwargs={
                "standalone": True,
                "policy": policy,
                "trainer": trainer,
                "rerun_queue": rerun_queue,
                "recorder": recorder,
            },
            daemon=True,
            name="rtc_env",
        )
    except Exception:
        _close_robot_hardware(online_env, teleop_device, "RTC_INFERENCE_SETUP")
        raise

    try:
        if subtask_console is not None:
            subtask_console.start()
        env_thread.start()
        time.sleep(1.0)
        inf_thread.start()

        start_time = time.time()
        next_metrics_time = time.monotonic() + 20.0
        logger.info("[RTC_INFERENCE] Threads running. Supervisor loop active.")
        while not shutdown_event.is_set():
            if shutdown_event.wait(1.0):
                break
            if not env_thread.is_alive():
                raise RuntimeError("RTC environment worker exited unexpectedly")
            if not inf_thread.is_alive():
                raise RuntimeError("RTC inference worker exited unexpectedly")
            if time.monotonic() < next_metrics_time:
                continue
            next_metrics_time += 20.0

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
                "[metrics/env] steps=%d | action_get=%.1fms | env_step=%.1fms (action_proc=%.1fms robot_step=%.1fms obs_proc=%.1fms) | active=%.1fms | sleep=%.1fms | starved=%d | post_step=%.1fms | episode=%s",
                metrics["env_steps"], avg_action_get * 1000, avg_env_step * 1000,
                avg_action_proc * 1000, avg_robot_step * 1000, avg_obs_proc * 1000,
                avg_env_active * 1000, avg_env_wait * 1000, metrics["queue_starvation_count"],
                avg_post_step * 1000, "ACTIVE" if shared.episode_active else "IDLE",
            )
    except Exception:
        logger.error("[RTC_INFERENCE] Orchestrator error:\n%s", traceback.format_exc())
    finally:
        shutdown_event.set()
        shared.running = False
        if subtask_console is not None:
            subtask_console.stop()
        inf_thread.join(timeout=5.0)
        while env_thread.is_alive():
            # The env worker may be returning the arms to rest or saving the episode; the
            # hardware close below must not race it. Ctrl-\ twice forces an exit.
            env_thread.join(timeout=10.0)
            if env_thread.is_alive():
                logger.info("[RTC_INFERENCE] Waiting for the env worker (return to rest / episode save)...")
        _close_robot_hardware(online_env, teleop_device, "RTC_INFERENCE")
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

    if getattr(cfg.policy, "torch_compile", False):
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        try:
            action_expert = policy._action_expert()
            action_expert.forward_with_context = torch.compile(
                action_expert.forward_with_context,
                mode="reduce-overhead",
                fullgraph=False,
            )
            logger.info("[RTC_ACTOR] torch.compile applied to action expert.")
        except Exception as e:
            logger.warning("[RTC_ACTOR] Could not compile action expert: %s", e)

    if getattr(policy.config, "rtc_config", None) is None or not policy.config.rtc_config.enabled:
        raise RuntimeError("RTC runtime requires policy.config.rtc_config.enabled=True")

    logger.info("[RTC_ACTOR] Setting up environment...")
    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(
        online_env, teleop_device, cfg.env, cfg.policy.device
    )

    shared = RTCSharedState()
    shared.running = not shutdown_event.is_set()
    memory_cfg = getattr(cfg.policy, "memory", None)
    shared.configure_history(
        _rig_history_offsets(
            ReplayBuffer._normalize_history_offsets(
                memory_cfg.history_offsets(cfg.env.fps) if memory_cfg is not None else None
            ),
            cfg,
        )
    )
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
            if shutdown_event.wait(5):
                break
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
            if teleop_device is not None and getattr(teleop_device, "is_connected", False):
                teleop_device.disconnect()
        except Exception:
            logger.warning("[RTC_ACTOR] Teleop disconnect failed:\n%s", traceback.format_exc())
        with contextlib.suppress(Exception):
            online_env.close()
        logger.info("[RTC_ACTOR] Shutdown complete.")
