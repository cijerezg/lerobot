#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.teleoperators.utils import TeleopEvents

from .config_squint_so101 import SquintSO101RobotConfig

logger = logging.getLogger(__name__)


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)


def infer_squint_env_id(task: str | None, fallback: str | None = None) -> str:
    """Map a dataset task string to the nearest Squint SO101 environment."""
    if fallback:
        return fallback
    task_l = (task or "").lower()
    item = "Can" if "can" in task_l else "Cube"
    if "stack" in task_l:
        verb = "Stack"
    elif "lift" in task_l:
        verb = "Lift"
    elif "reach" in task_l:
        verb = "Reach"
    else:
        verb = "Place"
    if verb == "Place" and item == "Cube" and any(token in task_l for token in ("black x", "marker", "white background")):
        return "SO101PlaceCubeMarker-v1"
    return f"SO101{verb}{item}-v1"


def read_dataset_task(dataset_root: str | None) -> str | None:
    """Read the first task instruction from a LeRobot v3 dataset root."""
    if not dataset_root:
        return None
    tasks_path = Path(dataset_root) / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return None
    try:
        import pandas as pd

        tasks = pd.read_parquet(tasks_path)
        if tasks.empty:
            return None
        # Current LeRobot v3 tasks.parquet stores task text in the index and task_index as a column.
        if tasks.index.dtype == object:
            return str(tasks.index[0])
        for column in ("task", "instruction", "task_name"):
            if column in tasks:
                return str(tasks[column].iloc[0])
    except Exception as exc:
        logger.warning("Failed to read dataset task from %s: %s", tasks_path, exc)
    return None


def read_dataset_action_range(dataset_root: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Read demonstrated action min/max ranges from a LeRobot v3 dataset root."""
    if not dataset_root:
        return None
    stats_path = Path(dataset_root) / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        action_stats = stats.get("action", {})
        low = np.asarray(action_stats.get("min"), dtype=np.float32)
        high = np.asarray(action_stats.get("max"), dtype=np.float32)
        if low.shape == (len(ACTION_KEYS),) and high.shape == (len(ACTION_KEYS),):
            # Avoid degenerate ranges; fall back per-joint if a dataset stat is malformed.
            default_low, default_high = _default_lerobot_unit_range()
            valid = np.isfinite(low) & np.isfinite(high) & ((high - low) > 1e-3)
            low = np.where(valid, low, default_low)
            high = np.where(valid, high, default_high)
            return low, high
    except Exception as exc:
        logger.warning("Failed to read dataset action stats from %s: %s", stats_path, exc)
    return None


def read_dataset_initial_state(dataset_root: str | None) -> np.ndarray | None:
    """Read the first demonstrated robot state from a LeRobot v3 dataset root."""
    if not dataset_root:
        return None
    data_root = Path(dataset_root) / "data"
    if not data_root.exists():
        return None
    try:
        import pandas as pd

        data_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
        if not data_files:
            return None
        data = pd.read_parquet(data_files[0], columns=["observation.state"])
        if data.empty:
            return None
        state = np.asarray(data["observation.state"].iloc[0], dtype=np.float32)
        if state.shape == (len(ACTION_KEYS),) and np.isfinite(state).all():
            return state
    except Exception as exc:
        logger.warning("Failed to read dataset initial state from %s: %s", data_root, exc)
    return None


def read_dataset_episode_actions(
    dataset_root: str | None,
    episode_index: int | None,
    stride: int = 1,
) -> np.ndarray | None:
    """Read demonstrated actions for one LeRobot v3 episode."""
    if not dataset_root or episode_index is None:
        return None
    data_root = Path(dataset_root) / "data"
    if not data_root.exists():
        return None
    try:
        import pandas as pd

        stride = max(1, int(stride))
        chunks: list[np.ndarray] = []
        for data_file in sorted(data_root.glob("chunk-*/file-*.parquet")):
            data = pd.read_parquet(data_file, columns=["episode_index", "frame_index", "action"])
            episode = data[data["episode_index"] == int(episode_index)].sort_values("frame_index")
            if episode.empty:
                continue
            actions = np.stack(episode["action"].to_numpy()).astype(np.float32)
            chunks.append(actions)
        if not chunks:
            return None
        actions = np.concatenate(chunks, axis=0)[::stride]
        if actions.ndim == 2 and actions.shape[1] == len(ACTION_KEYS) and np.isfinite(actions).all():
            return actions
    except Exception as exc:
        logger.warning("Failed to read dataset episode %s actions from %s: %s", episode_index, data_root, exc)
    return None


def read_dataset_episode_initial_state(dataset_root: str | None, episode_index: int | None) -> np.ndarray | None:
    """Read the first demonstrated robot state for one LeRobot v3 episode."""
    if not dataset_root or episode_index is None:
        return None
    data_root = Path(dataset_root) / "data"
    if not data_root.exists():
        return None
    try:
        import pandas as pd

        for data_file in sorted(data_root.glob("chunk-*/file-*.parquet")):
            data = pd.read_parquet(data_file, columns=["episode_index", "frame_index", "observation.state"])
            episode = data[data["episode_index"] == int(episode_index)].sort_values("frame_index")
            if episode.empty:
                continue
            state = np.asarray(episode["observation.state"].iloc[0], dtype=np.float32)
            if state.shape == (len(ACTION_KEYS),) and np.isfinite(state).all():
                return state
    except Exception as exc:
        logger.warning("Failed to read dataset episode %s initial state from %s: %s", episode_index, data_root, exc)
    return None


def _default_lerobot_unit_range() -> tuple[np.ndarray, np.ndarray]:
    low = np.array([-100.0] * 5 + [0.0], dtype=np.float32)
    high = np.array([100.0] * 5 + [100.0], dtype=np.float32)
    return low, high


def _identity_follow_calibration() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ones(len(ACTION_KEYS), dtype=np.float32),
        np.zeros(len(ACTION_KEYS), dtype=np.float32),
    )


def _load_follow_calibration(path: str | None) -> tuple[np.ndarray, np.ndarray, Path | None]:
    scale, offset = _identity_follow_calibration()
    if not path:
        return scale, offset, None

    calibration_path = Path(path).expanduser()
    if not calibration_path.exists():
        logger.warning("Squint follow calibration file does not exist: %s", calibration_path)
        return scale, offset, calibration_path

    try:
        with calibration_path.open() as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load Squint follow calibration from %s: %s", calibration_path, exc)
        return scale, offset, calibration_path

    if not isinstance(payload, dict):
        logger.warning("Ignoring malformed Squint follow calibration in %s", calibration_path)
        return scale, offset, calibration_path

    scale_payload = payload.get("scale", {})
    offset_payload = payload.get("offset", {})
    if not isinstance(scale_payload, dict):
        scale_payload = {}
    if not isinstance(offset_payload, dict):
        offset_payload = {}

    for index, key in enumerate(ACTION_KEYS):
        try:
            scale[index] = float(scale_payload.get(key, scale[index]))
            offset[index] = float(offset_payload.get(key, offset[index]))
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed calibration value for %s in %s", key, calibration_path)

    invalid_scale = ~np.isfinite(scale) | (np.abs(scale) < 1e-6)
    if invalid_scale.any():
        bad_keys = [key for key, invalid in zip(ACTION_KEYS, invalid_scale, strict=True) if invalid]
        logger.warning("Resetting invalid Squint follow calibration scale values for: %s", ", ".join(bad_keys))
        scale[invalid_scale] = 1.0

    offset = np.where(np.isfinite(offset), offset, 0.0).astype(np.float32)
    return scale.astype(np.float32), offset, calibration_path


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim >= 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] > 3:
        array = array[..., :3]
    return array


def _as_plain_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().flatten()[0].item())
    array = np.asarray(value)
    if array.shape:
        return bool(array.flatten()[0].item())
    return bool(value)


def _scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().flatten()[0].item())
    array = np.asarray(value)
    if array.shape:
        return float(array.flatten()[0].item())
    return float(value)


class SquintSO101Robot(Robot):
    config_class = SquintSO101RobotConfig
    name = "squint_so101"

    def __init__(self, config: SquintSO101RobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._env = None
        self._latest_obs: Any = None
        self._latest_info: dict[str, Any] = {}
        self._episode_index = -1
        self._episode_step = 0
        self._pending_start_event = False
        self._pending_terminal_event: str | None = None
        self._episode_had_success = False
        self._last_reward = 0.0
        self._last_sent_action_units: np.ndarray | None = None
        self._qpos_low = np.array([-np.pi] * 5 + [0.0], dtype=np.float32)
        self._qpos_high = np.array([np.pi] * 5 + [np.pi / 2], dtype=np.float32)
        self._unit_low, self._unit_high = _default_lerobot_unit_range()
        self._follow_calibration_scale, self._follow_calibration_offset, self._follow_calibration_path = (
            _load_follow_calibration(config.follow_calibration_path)
        )
        action_range = read_dataset_action_range(config.dataset_root)
        if action_range is not None:
            self._unit_low, self._unit_high = action_range
        self._initial_state_units = (
            read_dataset_initial_state(config.dataset_root) if config.use_dataset_initial_state else None
        )
        self._bootstrap_dataset_episodes = list(config.bootstrap_dataset_episodes or [])
        self._bootstrap_dataset_episode_interval = max(1, int(config.bootstrap_dataset_episode_interval))
        bootstrap_episode_ids = set(self._bootstrap_dataset_episodes)
        if config.bootstrap_dataset_episode is not None:
            bootstrap_episode_ids.add(int(config.bootstrap_dataset_episode))
        self._bootstrap_actions_by_episode: dict[int, np.ndarray] = {}
        self._bootstrap_initial_states_by_episode: dict[int, np.ndarray] = {}
        for episode in sorted(bootstrap_episode_ids):
            actions = read_dataset_episode_actions(
                config.dataset_root,
                episode,
                config.bootstrap_dataset_action_stride,
            )
            if actions is not None:
                self._bootstrap_actions_by_episode[episode] = actions
            if config.use_dataset_initial_state:
                state = read_dataset_episode_initial_state(config.dataset_root, episode)
                if state is not None:
                    self._bootstrap_initial_states_by_episode[episode] = state
        self._active_bootstrap_actions: np.ndarray | None = None
        self._bootstrap_active = False
        self._video_frames: list[np.ndarray] = []
        self._video_dir = Path(config.video_dir)
        self._last_timing: dict[str, dict[str, float]] = {}
        self._camera_pose_payload: dict[str, Any] | None = None

        dataset_task = read_dataset_task(config.dataset_root)
        self.task_instruction = config.task or dataset_task
        self.env_id = infer_squint_env_id(self.task_instruction, config.env_id)

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **{key: float for key in ACTION_KEYS},
            self.config.top_camera_name: (self.config.camera_height, self.config.camera_width, 3),
            self.config.side_camera_name: (self.config.camera_height, self.config.camera_width, 3),
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_KEYS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return
        self._import_squint()
        self._create_env()
        self._connected = True
        self._reset_episode(seed=self.config.seed)
        logger.info(
            "Connected Squint SO101 simulator env_id=%s task=%r dataset_root=%s action_unit_range=(%s, %s)",
            self.env_id,
            self.task_instruction,
            self.config.dataset_root,
            self._unit_low.tolist(),
            self._unit_high.tolist(),
        )
        if self._follow_calibration_path is not None:
            logger.info(
                "Squint follow calibration: path=%s scale=%s offset=%s",
                self._follow_calibration_path,
                self._follow_calibration_scale.tolist(),
                self._follow_calibration_offset.tolist(),
            )

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        total_start = time.perf_counter()
        timing: dict[str, float] = {}

        tick = time.perf_counter()
        qpos = self._get_qpos()
        state = self._qpos_to_lerobot_units(qpos)
        timing["qpos_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        unwrapped = self._env.unwrapped
        if hasattr(unwrapped, "_update_gripper_contact_markers"):
            unwrapped._update_gripper_contact_markers()
        top_raw = self._render_rgb()
        timing["top_render_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        top_image = self._resize_image(top_raw)
        timing["top_resize_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        side_raw = self._sensor_rgb(self._latest_obs)
        timing["side_extract_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        side_image = self._resize_image(side_raw)
        timing["side_resize_ms"] = (time.perf_counter() - tick) * 1000.0

        obs: dict[str, Any] = {key: float(value) for key, value in zip(ACTION_KEYS, state, strict=True)}
        obs[self.config.top_camera_name] = top_image
        obs[self.config.side_camera_name] = side_image
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        self._last_timing["get_observation"] = timing
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_connected()
        total_start = time.perf_counter()
        timing: dict[str, float] = {}

        tick = time.perf_counter()
        target_units = np.array([float(action[key]) for key in ACTION_KEYS], dtype=np.float32)
        if self._bootstrap_active and self._active_bootstrap_actions is not None:
            if self._episode_step < len(self._active_bootstrap_actions):
                target_units = self._active_bootstrap_actions[self._episode_step].astype(np.float32)
            else:
                self._bootstrap_active = False
        timing["target_units_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        target_qpos = self._lerobot_units_to_qpos(target_units)
        applied_target_units = self._qpos_to_lerobot_units(target_qpos)
        sim_action = self._target_qpos_to_controller_action(target_qpos)
        if self.config.action_clip is not None:
            sim_action = np.clip(sim_action, -self.config.action_clip, self.config.action_clip)
        sim_action = self._fit_action_space(sim_action.astype(np.float32))
        self._last_sent_action_units = applied_target_units.astype(np.float32).copy()
        timing["action_transform_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        obs, reward, terminated, truncated, info = self._env.step(sim_action)
        timing["env_step_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        self._latest_obs = obs
        self._latest_info = dict(info or {})
        self._last_reward = _scalar_float(reward)
        self._episode_step += 1
        timing["bookkeeping_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        self._append_video_frame()
        timing["append_video_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        step_success = self._episode_success(info)
        self._episode_had_success = self._episode_had_success or step_success
        terminal = _scalar_bool(terminated) or _scalar_bool(truncated) or step_success
        if self.config.max_episode_steps is not None and self._episode_step >= self.config.max_episode_steps:
            terminal = True
        if terminal and self._pending_terminal_event is None:
            success = self._episode_had_success or step_success
            self._pending_terminal_event = "success" if success else "failure"
            self._flush_video(success=success)
            if self.config.reset_after_terminal:
                reset_seed = self.config.seed if self.config.reset_seed_on_terminal else None
                self._reset_episode(seed=reset_seed)
        timing["terminal_ms"] = (time.perf_counter() - tick) * 1000.0

        sent_units = self._last_sent_action_units
        timing["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        self._last_timing["send_action"] = timing
        return {key: float(value) for key, value in zip(ACTION_KEYS, sent_units, strict=True)}

    def set_joint_positions(self, action: RobotAction) -> RobotAction:
        """Directly set simulator joint positions from LeRobot action units."""
        self._require_connected()
        target_units = np.array([float(action[key]) for key in ACTION_KEYS], dtype=np.float32)
        target_qpos = self._lerobot_units_to_qpos(target_units)
        applied_units = self._qpos_to_lerobot_units(target_qpos)
        qpos_tensor = torch.as_tensor(
            target_qpos, dtype=torch.float32, device=self._env.unwrapped.device
        ).view(1, -1)
        robot = self._env.unwrapped.agent.robot
        robot.set_qpos(qpos_tensor)
        if hasattr(robot, "set_qvel"):
            robot.set_qvel(torch.zeros_like(qpos_tensor))
        controller = getattr(self._env.unwrapped.agent, "controller", None)
        if hasattr(controller, "_target_qpos"):
            controller._target_qpos = qpos_tensor.clone()
        if hasattr(controller, "_start_qpos"):
            controller._start_qpos = qpos_tensor.clone()
        self._last_sent_action_units = applied_units.astype(np.float32).copy()
        try:
            self._latest_obs = self._env.unwrapped.get_obs(self._latest_info)
        except Exception as exc:
            logger.debug("Failed to refresh Squint observation after direct joint set: %s", exc)
        return {key: float(value) for key, value in zip(ACTION_KEYS, applied_units, strict=True)}

    def disconnect(self) -> None:
        if self._env is not None:
            try:
                self._flush_video(success=False)
                self._env.close()
            except Exception as exc:
                logger.debug("Squint simulator close failed: %s", exc)
        self._env = None
        self._connected = False

    def get_rlt_events(self) -> dict[str, bool]:
        """Return one-shot episode events consumed by RobotClientDrtc."""
        if self._pending_terminal_event is not None:
            event = self._pending_terminal_event
            self._pending_terminal_event = None
            if event == "success":
                return {TeleopEvents.SUCCESS.value: True}
            return {
                TeleopEvents.TERMINATE_EPISODE.value: True,
                TeleopEvents.FAILURE.value: True,
            }
        if self._pending_start_event:
            self._pending_start_event = False
            return {TeleopEvents.START_EPISODE.value: True}
        return {}

    def _import_squint(self) -> None:
        os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
        try:
            import mani_skill.envs  # noqa: F401
            import lerobot.robots.squint_so101.sim.envs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Squint SO101 simulation requires ManiSkill dependencies. "
                "Install this environment with the squint_so101 extra "
                "(notably `mani_skill_nightly`, `dacite`, and `transforms3d`)."
            ) from exc

    def _create_env(self) -> None:
        import gymnasium as gym

        domain_randomization_config: dict[str, Any] = {
            "apply_overlay": bool(self.config.white_x_background),
        }
        if self.config.white_x_background:
            domain_randomization_config.update(
                rgb_overlay_path=str(self._ensure_white_x_background()),
                randomize_lighting=False,
            )

        kwargs: dict[str, Any] = {
            "obs_mode": self.config.obs_mode,
            "render_mode": self.config.render_mode,
            "sensor_configs": {"width": self.config.sensor_width, "height": self.config.sensor_height},
            "human_render_camera_configs": {
                "width": self.config.sensor_width,
                "height": self.config.sensor_height,
            },
            "num_envs": 1,
            "domain_randomization": self.config.domain_randomization,
            "reconfiguration_freq": None,
            "control_mode": self.config.control_mode,
            "show_gripper_contact_markers": self.config.show_gripper_contact_markers,
            "use_marker_grasp_assist": self.config.use_marker_grasp_assist,
        }
        if self.config.marker_xy_offset is not None:
            kwargs["marker_xy_offset"] = self.config.marker_xy_offset
        kwargs["marker_yaw_degrees"] = self.config.marker_yaw_degrees
        kwargs["domain_randomization_config"] = domain_randomization_config
        self._env = gym.make(self.env_id, **kwargs)
        self._refresh_qpos_limits()
        self._apply_camera_pose_config(apply_render_camera=False)

    def _bootstrap_dataset_episode_for_sim_episode(self, sim_episode_index: int) -> int | None:
        if self._bootstrap_dataset_episodes:
            if sim_episode_index % self._bootstrap_dataset_episode_interval != 0:
                return None
            schedule_index = sim_episode_index // self._bootstrap_dataset_episode_interval
            if schedule_index < len(self._bootstrap_dataset_episodes):
                return self._bootstrap_dataset_episodes[schedule_index]
            return None
        if self.config.bootstrap_dataset_episode is not None and sim_episode_index == 0:
            return int(self.config.bootstrap_dataset_episode)
        return None

    def _reset_episode(self, seed: int | None) -> None:
        next_episode_index = self._episode_index + 1
        bootstrap_dataset_episode = self._bootstrap_dataset_episode_for_sim_episode(next_episode_index)
        obs, info = self._env.reset(seed=seed)
        obs = self._apply_dataset_initial_state(obs, info, bootstrap_dataset_episode)
        self._latest_obs = obs
        self._latest_info = dict(info or {})
        self._apply_camera_pose_config()
        if self.config.camera_pose_path:
            try:
                self._latest_obs = self._env.unwrapped.get_obs(self._latest_info)
            except Exception as exc:
                logger.debug("Failed to refresh Squint observation after camera pose apply: %s", exc)
        self._episode_index = next_episode_index
        self._episode_step = 0
        self._video_frames = []
        self._pending_start_event = True
        self._episode_had_success = False
        self._active_bootstrap_dataset_episode = bootstrap_dataset_episode
        self._active_bootstrap_actions = (
            self._bootstrap_actions_by_episode.get(bootstrap_dataset_episode)
            if bootstrap_dataset_episode is not None
            else None
        )
        self._bootstrap_active = self._active_bootstrap_actions is not None
        if self._bootstrap_active:
            logger.info(
                "Using dataset action bootstrap for simulator episode %s from dataset episode %s (%s actions)",
                self._episode_index,
                self._active_bootstrap_dataset_episode,
                len(self._active_bootstrap_actions),
            )
        self._append_video_frame()

    def _load_camera_pose_payload(self) -> dict[str, Any] | None:
        path_value = (self.config.camera_pose_path or "").strip()
        if not path_value:
            return None
        if self._camera_pose_payload is not None:
            return self._camera_pose_payload

        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Squint camera pose file not found: {path}")
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not isinstance(payload.get("cameras"), dict):
            raise ValueError(f"Squint camera pose file must contain a 'cameras' mapping: {path}")
        self._camera_pose_payload = payload
        return payload

    def _camera_pose_for(self, payload: dict[str, Any], canonical_name: str) -> dict[str, Any] | None:
        cameras = payload.get("cameras") or {}
        if not isinstance(cameras, dict):
            return None
        if canonical_name == "top":
            names = (self.config.top_camera_name, "top", "render", "render_camera")
        else:
            names = (self.config.side_camera_name, "side", "base", "base_camera")
        for name in names:
            pose = cameras.get(name)
            if isinstance(pose, dict):
                return pose
        return None

    @staticmethod
    def _camera_eye_target(pose: dict[str, Any], camera_name: str) -> tuple[list[float], list[float]]:
        try:
            eye = [float(value) for value in pose["eye"]]
            target = [float(value) for value in pose["target"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Camera pose for {camera_name!r} must contain numeric eye and target lists") from exc
        if len(eye) != 3 or len(target) != 3:
            raise ValueError(f"Camera pose for {camera_name!r} must contain 3D eye and target lists")
        return eye, target

    def _apply_camera_pose_config(self, *, apply_render_camera: bool = True) -> None:
        payload = self._load_camera_pose_payload()
        if payload is None or self._env is None:
            return

        unwrapped = self._env.unwrapped
        applied_state: dict[str, dict[str, list[float]]] = {}

        side_pose = self._camera_pose_for(payload, "side")
        if side_pose is not None:
            eye, target = self._camera_eye_target(side_pose, "side")
            if hasattr(unwrapped, "base_camera_settings"):
                unwrapped.base_camera_settings = {"pos": eye, "target": target}
            if hasattr(unwrapped, "camera_mount") and hasattr(unwrapped, "sample_camera_poses"):
                unwrapped.camera_mount.set_pose(unwrapped.sample_camera_poses(n=unwrapped.num_envs))
                if getattr(unwrapped, "gpu_sim_enabled", False):
                    unwrapped.scene._gpu_apply_all()
            applied_state["side"] = {"eye": eye, "target": target}

        top_pose = self._camera_pose_for(payload, "top")
        if top_pose is not None:
            eye, target = self._camera_eye_target(top_pose, "top")
            if apply_render_camera:
                try:
                    from mani_skill.utils import sapien_utils

                    if "render_camera" not in unwrapped.scene.human_render_cameras:
                        self._render_rgb()
                    camera = unwrapped.scene.human_render_cameras.get("render_camera")
                    if camera is not None:
                        camera.camera.set_local_pose(sapien_utils.look_at(eye, target).sp)
                except Exception as exc:
                    logger.warning("Failed to apply Squint top camera pose: %s", exc)
            applied_state["top"] = {"eye": eye, "target": target}

        if applied_state:
            unwrapped._teleop_camera_defaults = json.loads(json.dumps(applied_state))
            unwrapped._teleop_camera_state = json.loads(json.dumps(applied_state))

    def _apply_dataset_initial_state(
        self,
        obs: Any,
        info: dict[str, Any],
        bootstrap_dataset_episode: int | None = None,
    ) -> Any:
        state_units = (
            self._bootstrap_initial_states_by_episode.get(bootstrap_dataset_episode)
            if bootstrap_dataset_episode is not None
            else None
        )
        if state_units is None:
            state_units = self._initial_state_units
        if state_units is None:
            return obs
        try:
            qpos = self._lerobot_units_to_qpos(state_units)
            qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=self._env.unwrapped.device).view(1, -1)
            robot = self._env.unwrapped.agent.robot
            robot.set_qpos(qpos_tensor)
            if hasattr(robot, "set_qvel"):
                robot.set_qvel(torch.zeros_like(qpos_tensor))
            return self._env.unwrapped.get_obs(info)
        except Exception as exc:
            logger.warning("Failed to apply dataset initial state to Squint simulator: %s", exc)
            return obs

    def _ensure_white_x_background(self) -> Path:
        bg_dir = self._video_dir / "assets"
        bg_dir.mkdir(parents=True, exist_ok=True)
        bg_path = bg_dir / "white_x_background.png"
        if bg_path.exists():
            return bg_path
        image = np.full((self.config.sensor_height, self.config.sensor_width, 3), 255, dtype=np.uint8)
        thickness = max(2, self.config.sensor_width // 40)
        cv2.line(image, (0, 0), (self.config.sensor_width - 1, self.config.sensor_height - 1), (0, 0, 0), thickness)
        cv2.line(image, (self.config.sensor_width - 1, 0), (0, self.config.sensor_height - 1), (0, 0, 0), thickness)
        cv2.imwrite(str(bg_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return bg_path

    def _refresh_qpos_limits(self) -> None:
        try:
            qlimits = _as_plain_numpy(self._env.unwrapped.agent.robot.get_qlimits())
            if qlimits.ndim == 3:
                qlimits = qlimits[0]
            self._qpos_low = qlimits[:, 0].astype(np.float32)
            self._qpos_high = qlimits[:, 1].astype(np.float32)
            self._warn_if_external_range_exceeds_sim_limits()
        except Exception as exc:
            logger.warning("Could not read Squint SO101 qlimits; using fallback limits: %s", exc)

    def _warn_if_external_range_exceeds_sim_limits(self) -> None:
        if self._follow_calibration_path is None:
            return
        reachable_low = self._qpos_to_lerobot_units(self._qpos_low)
        reachable_high = self._qpos_to_lerobot_units(self._qpos_high)
        below = self._unit_low < (reachable_low - 1e-3)
        above = self._unit_high > (reachable_high + 1e-3)
        if not (below.any() or above.any()):
            return

        details: list[str] = []
        for index, key in enumerate(ACTION_KEYS):
            if below[index] or above[index]:
                details.append(
                    f"{key}: configured=[{self._unit_low[index]:.2f}, {self._unit_high[index]:.2f}] "
                    f"reachable=[{reachable_low[index]:.2f}, {reachable_high[index]:.2f}]"
                )
        logger.warning(
            "Squint follow calibration makes part of the configured action range unreachable; "
            "commands will be clipped for %s",
            "; ".join(details),
        )

    def _get_qpos(self) -> np.ndarray:
        qpos = _as_plain_numpy(self._env.unwrapped.agent.robot.get_qpos()).astype(np.float32)
        return qpos.reshape(-1)[: len(ACTION_KEYS)]

    def _qpos_to_lerobot_units(self, qpos: np.ndarray) -> np.ndarray:
        return self._sim_units_to_external_units(np.rad2deg(qpos).astype(np.float32))

    def _lerobot_units_to_qpos(self, values: np.ndarray) -> np.ndarray:
        sim_units = self._external_units_to_sim_units(values)
        qpos = np.deg2rad(sim_units).astype(np.float32)
        return np.clip(qpos, self._qpos_low, self._qpos_high).astype(np.float32)

    def _external_units_to_sim_units(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values * self._follow_calibration_scale + self._follow_calibration_offset).astype(np.float32)

    def _sim_units_to_external_units(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - self._follow_calibration_offset) / self._follow_calibration_scale).astype(np.float32)

    def _get_controller_target_qpos(self) -> np.ndarray | None:
        controller = getattr(self._env.unwrapped.agent, "controller", None)
        target_qpos = getattr(controller, "_target_qpos", None)
        if target_qpos is None:
            return None
        target_qpos = _as_plain_numpy(target_qpos).astype(np.float32)
        return target_qpos.reshape(-1)[: len(ACTION_KEYS)]

    def _qpos_action_to_normalized_controller_action(self, action: np.ndarray) -> np.ndarray:
        controller = getattr(self._env.unwrapped.agent, "controller", None)
        if controller is None or not getattr(controller, "_normalize_action", False):
            return action
        low = getattr(controller, "action_space_low", None)
        high = getattr(controller, "action_space_high", None)
        if low is None or high is None:
            return action
        low = _as_plain_numpy(low).astype(np.float32).reshape(-1)[: action.size]
        high = _as_plain_numpy(high).astype(np.float32).reshape(-1)[: action.size]
        span = high - low
        valid = np.abs(span) > 1e-6
        normalized = np.zeros_like(action, dtype=np.float32)
        normalized[valid] = 2.0 * (action[valid] - low[valid]) / span[valid] - 1.0
        return normalized.astype(np.float32)

    def _target_qpos_to_controller_action(self, target_qpos: np.ndarray) -> np.ndarray:
        if "delta" in self.config.control_mode:
            reference_qpos = self._get_controller_target_qpos()
            if reference_qpos is None:
                reference_qpos = self._get_qpos()
            action = target_qpos - reference_qpos
        else:
            action = target_qpos
        return self._qpos_action_to_normalized_controller_action(action)

    def _fit_action_space(self, action: np.ndarray) -> np.ndarray:
        shape = self._env.action_space.shape
        if shape == action.shape:
            out = action
        else:
            out = np.zeros(shape, dtype=np.float32)
            flat = out.reshape(-1)
            flat[: action.size] = action.reshape(-1)[: flat.size]
        low = np.asarray(self._env.action_space.low, dtype=np.float32)
        high = np.asarray(self._env.action_space.high, dtype=np.float32)
        finite = np.isfinite(low) & np.isfinite(high)
        if finite.any():
            out = np.where(finite, np.clip(out, low, high), out)
        return out.astype(np.float32)

    def _episode_success(self, info: dict[str, Any]) -> bool:
        if isinstance(info, dict) and "success" in info:
            return _scalar_bool(info["success"])
        return self._last_reward >= self.config.success_reward_threshold

    def _current_images(self) -> tuple[np.ndarray, np.ndarray]:
        top = self._resize_image(self._render_rgb())
        side = self._resize_image(self._sensor_rgb(self._latest_obs))
        return top, side

    def _sensor_rgb(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            if "sensor_data" in obs and isinstance(obs["sensor_data"], dict):
                for camera_obs in obs["sensor_data"].values():
                    if isinstance(camera_obs, dict) and "rgb" in camera_obs:
                        return _as_numpy(camera_obs["rgb"])
            if "rgb" in obs:
                rgb = _as_numpy(obs["rgb"])
                if rgb.ndim == 3 and rgb.shape[-1] > 3:
                    return rgb[..., :3]
                return rgb
            for value in obs.values():
                if isinstance(value, dict) and "rgb" in value:
                    return _as_numpy(value["rgb"])
        return self._render_rgb()

    def _render_rgb(self) -> np.ndarray:
        try:
            return _as_numpy(self._env.render())
        except Exception:
            if hasattr(self._env.unwrapped, "render_all"):
                return _as_numpy(self._env.unwrapped.render_all())
            raise

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (self.config.camera_height, self.config.camera_width):
            return image
        return cv2.resize(
            image,
            (self.config.camera_width, self.config.camera_height),
            interpolation=cv2.INTER_AREA,
        )

    def _append_video_frame(self) -> None:
        if self.config.video_every_episodes <= 0:
            return
        if self._episode_index % self.config.video_every_episodes != 0:
            return
        if self._episode_index >= self.config.video_max_episodes:
            return
        if len(self._video_frames) >= self.config.video_max_frames:
            return
        try:
            top, side = self._current_images()
            h = min(top.shape[0], side.shape[0])
            top = cv2.resize(top, (top.shape[1], h))
            side = cv2.resize(side, (side.shape[1], h))
            self._video_frames.append(np.concatenate([side, top], axis=1))
        except Exception as exc:
            logger.debug("Failed to append Squint video frame: %s", exc)

    def _flush_video(self, *, success: bool) -> None:
        if not self._video_frames:
            return
        if self._episode_index % max(self.config.video_every_episodes, 1) != 0:
            self._video_frames = []
            return
        if self._episode_index >= self.config.video_max_episodes:
            self._video_frames = []
            return
        self._video_dir.mkdir(parents=True, exist_ok=True)
        label = "success" if success else "failure"
        path = self._video_dir / f"episode_{self._episode_index:04d}_{label}.mp4"
        try:
            imageio.mimsave(path, self._video_frames, fps=self.config.video_fps, macro_block_size=8)
            logger.info("Saved Squint simulator episode video to %s", path)
        except Exception as exc:
            logger.warning("Failed to save Squint simulator video to %s: %s", path, exc)
        finally:
            self._video_frames = []

    def _require_connected(self) -> None:
        if not self._connected or self._env is None:
            raise RuntimeError("SquintSO101Robot is not connected")
