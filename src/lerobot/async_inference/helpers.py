# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import logging.handlers
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature

# NOTE: Configs need to be loaded for the client to be able to instantiate the policy config
from lerobot.policies import (  # noqa: F401
    ACTConfig,
    DiffusionConfig,
    PI0Config,
    PI05Config,
    SmolVLAConfig,
    VQBeTConfig,
)
from lerobot.robots.robot import Robot
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.utils import init_logging

Action = Any

# Type alias for the monotone control-loop clock used throughout async inference.
# See robot_client_drtc.py for the two-clock causality model documentation.
ControlStep = int

# observation as received from the robot
RawObservation = dict[str, Any]

# observation as those recorded in LeRobot dataset (keys are different)
LeRobotObservation = dict[str, Any]

# observation, ready for policy inference (image keys resized)
Observation = dict[str, Any]


def _validate_feature_names(features: dict[str, dict]) -> None:
    """Validate that feature names do not contain invalid characters.

    We keep this local to avoid importing `lerobot.datasets.utils` (which is heavyweight).
    """
    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:
    """Lightweight version of `lerobot.datasets.utils.hw_to_dataset_features`.

    The async inference client only needs a small subset of dataset feature logic, and importing
    the full dataset stack (datasets/pandas/pyarrow/torchvision/...) is very expensive on small
    devices like a Raspberry Pi.
    """
    features: dict[str, dict] = {}

    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }
    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        features[f"{prefix}.images.{key}"] = {
            "dtype": "video" if use_video else "image",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:
    """Lightweight version of `lerobot.datasets.utils.build_dataset_frame`."""
    frame: dict[str, np.ndarray] = {}
    for key, ft in ds_features.items():
        if not key.startswith(prefix):
            continue
        if ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
    return frame


def visualize_action_queue_size(action_queue_size: Sequence[int]) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.set_ylim(0, max(action_queue_size) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.show()


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def resize_robot_observation_image(image: Any, resize_dims: tuple[int, int, int]) -> Any:
    import torch

    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


# TODO(Steven): Consider implementing a pipeline step for this
def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    import torch

    observation = {}

    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0)
        else:
            observation[k] = v

    return observation


def prepare_image(image: Any) -> Any:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    import torch

    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> Any:
    """Extract the state from a raw observation."""
    import torch

    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> Any:
    """Extract the images from a raw observation."""
    import torch

    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """Make a lerobot observation from a raw observation."""
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    import torch

    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. Greps all observation.images.<> keys
    image_keys = list(filter(is_image_key, lerobot_obs))
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    # Turns the image features to (C, H, W) with H, W matching the policy image features.
    # This reduces the resolution of the images
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Get a logger using the standardized logging setup from utils.py.

    Args:
        name: Logger name (e.g., 'policy_server', 'robot_client')
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # Initialize the standardized logging
    init_logging(log_file=log_file, display_pid=False)

    # Return a named logger
    return logging.getLogger(name)


# -----------------------------------------------------------------------------
# Timed data containers
#
# DRTC uses two distinct logical clocks:
#   - control_step (t): monotone per control-loop tick; LWW key for SPSC mailboxes.
#   - action_step  (j): execution index incremented when an action is executed.
#
# For backwards compatibility with the legacy async inference path
# (`policy_server.py` / `robot_client.py` and their tests), we still accept the
# old `timestep=` kwarg in `__init__`, expose `.timestep` as a read-only alias,
# and provide a `get_timestep()` method that returns `control_step`.
# -----------------------------------------------------------------------------


class TimedData:
    """A data object with timestamp and control_step information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        control_step: The control-loop tick t when this data was created.
        timestep: Deprecated alias for ``control_step``; accepted for back-compat
            with the legacy async inference code paths and tests.
    """

    timestamp: float
    control_step: int

    def __init__(
        self,
        timestamp: float = 0.0,
        control_step: int = 0,
        *,
        timestep: int | None = None,
    ):
        self.timestamp = timestamp
        if timestep is not None:
            self.control_step = int(timestep)
        else:
            self.control_step = int(control_step)

    @property
    def timestep(self) -> int:
        """Back-compat alias for ``control_step``."""
        return self.control_step

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_control_step(self) -> int:
        return self.control_step

    def get_timestep(self) -> int:
        """Back-compat alias for ``get_control_step``."""
        return self.control_step


class TimedAction(TimedData):
    """A timed action with both control_step (t) and action_step (j).

    control_step comes from TimedData (identifies the observation/chunk).
    action_step is the execution index j = chunk_start_step + i.
    """

    action: Action
    action_step: int

    def __init__(
        self,
        timestamp: float = 0.0,
        control_step: int = 0,
        action: Action = None,
        action_step: int = 0,
        *,
        timestep: int | None = None,
    ):
        super().__init__(timestamp=timestamp, control_step=control_step, timestep=timestep)
        self.action = action
        self.action_step = int(action_step)

    def get_action_step(self) -> int:
        return self.action_step

    def get_action(self):
        return self.action


class TimedObservation(TimedData):
    """A timed observation carrying both control_step (t) and chunk_start_step (n_k).

    control_step comes from TimedData (monotone LWW key).
    chunk_start_step is the action step at which the resulting chunk should start.
    server_received_ts is set by the server when the observation is received (Unix seconds).
    """

    observation: RawObservation
    chunk_start_step: int
    must_go: bool
    server_received_ts: float

    def __init__(
        self,
        timestamp: float = 0.0,
        control_step: int = 0,
        observation: RawObservation = None,
        chunk_start_step: int = 0,
        must_go: bool = False,
        server_received_ts: float = 0.0,
        *,
        timestep: int | None = None,
    ):
        super().__init__(timestamp=timestamp, control_step=control_step, timestep=timestep)
        self.observation = observation
        self.chunk_start_step = int(chunk_start_step)
        self.must_go = bool(must_go)
        self.server_received_ts = float(server_received_ts)

    def get_observation(self):
        return self.observation


@dataclass
class FPSTracker:
    """Utility class to track FPS metrics over time."""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Calculate average FPS vs target"""
        self.total_obs_count += 1

        # Initialize first observation time
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # Calculate overall average FPS (since start)
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """Reset the FPS tracker state"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    # Client-driven RTC configuration (optional; server may ignore if policy doesn't support RTC)
    rtc_enabled: bool = False
    rtc_max_guidance_weight: float | None = None  # None = use num_flow_matching_steps (Alex Soare opt)
    rtc_prefix_attention_schedule: str = "linear"
    rtc_sigma_d: float = 1.0  # Prior variance (0.2 = stronger guidance, 1.0 = original RTC)
    rtc_full_trajectory_alignment: bool = False  # Skip gradient for faster/smoother transitions
    # Denoising steps override (Alex Soare: Beta should scale with n)
    num_flow_matching_steps: int | None = None  # None = use policy default (e.g., 10 for PI0/SmolVLA)
    # Spike injection (client-driven, for experiments)
    # List of dicts: [{"start_s": 5.0, "delay_ms": 2000}, ...]
    spikes: list[dict] = field(default_factory=list)
    # Diagnostics: when True, the server also enables verbose diagnostic output
    diagnostics_verbose: bool = False
    # pi05_rl: client-side override for the scalar advantage value injected into
    # `complementary_data["advantage"]` by the server's pi05_full preprocessor.
    # `None` means "use the value baked into the loaded policy's config"
    # (`policy.config.inference_advantage`). Set to e.g. 0.0 to force the
    # "negative" prompt label as a diagnostic A/B, or 1.0 to force "positive".
    inference_advantage: float | None = None
    # PI05/PI05-RL: optional override for subtask-token cache refresh interval
    # in seconds. None = use loaded policy config; 0 = regenerate every chunk.
    subtask_regeneration_interval: float | None = None
    # PI05/PI05-RL: when False, skip runtime subtask generation and condition
    # action sampling on the main task prompt only.
    subtask_generation_enabled: bool = True
    # PI05-RLT: optional lightweight RLT modules on top of a frozen PI0.5 VLA.
    rlt_enabled: bool = False
    rlt_embedding_checkpoint: str | None = None
    rlt_head_checkpoint: str | None = None
    rlt_chunk_size: int = 10
    # None => use the policy default (2048 for pi05_rlt; vlm_width for tinypi05_rlt).
    rlt_token_dim: int | None = None
    rlt_actor_hidden_dims: list[int] | None = None
    rlt_critic_hidden_dims: list[int] | None = None
    rlt_actor_residual_scale: float = 0.25
    # Paper-aligned actor: "gaussian" => mu_theta(x, ã) directly; "residual" =>
    # legacy ã + scale*tanh(MLP(...)) head. Default "gaussian".
    rlt_actor_mode: str = "gaussian"
    # Fixed exploration std for online data collection. 0 => deterministic mean.
    rlt_action_std: float = 0.05
    rlt_num_critics: int = 1
    rlt_bc_beta: float = 1.0
    rlt_bc_action_weights: list[float] | None = None
    rlt_jerk_beta: float = 0.0
    rlt_reference_dropout_p: float = 0.5
    rlt_online_collection_enabled: bool = False
    rlt_online_training_enabled: bool = False
    rlt_warmup_episodes: int = 1
    rlt_warmup_transitions: int = 128
    rlt_replay_capacity: int = 10000
    rlt_batch_size: int = 64
    rlt_utd_ratio: int = 1
    rlt_train_freq_s: float = 1.0
    rlt_save_freq_steps: int = 500
    rlt_output_dir: str = "outputs/rlt_online"
    rlt_demo_buffer_path: str | None = None
    rlt_online_buffer_path: str | None = None
    rlt_online_buffer_save_freq_transitions: int = 100
    rlt_persist_buffer_on_shutdown: bool = True
    rlt_actor_lr: float = 3e-4
    rlt_critic_lr: float = 3e-4
    rlt_discount: float = 0.99
    rlt_target_update_tau: float = 0.005
    rlt_execute_after_train_steps: int = 1000000
    rlt_context_cache_size: int = 256
    rlt_transition_queue_size: int = 256
    rlt_grad_clip_norm: float | None = None
    rlt_q_abs_max: float | None = None
    rlt_action_deviation_abs_max: float | None = None
    rlt_loss_abs_max: float | None = None
    rlt_safety_patience: int = 3

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Back-compat for pickles created before RTC/spike fields existed."""
        self.__dict__.update(state)
        self.__dict__.setdefault("rtc_enabled", False)
        self.__dict__.setdefault("rtc_max_guidance_weight", None)  # Default to auto (Alex Soare opt)
        self.__dict__.setdefault("rtc_prefix_attention_schedule", "linear")
        self.__dict__.setdefault("rtc_sigma_d", 1.0)
        self.__dict__.setdefault("rtc_full_trajectory_alignment", False)
        self.__dict__.setdefault("num_flow_matching_steps", None)  # Default to policy config
        # Spike injection defaults (new format)
        self.__dict__.setdefault("spikes", [])
        self.__dict__.setdefault("diagnostics_verbose", False)
        self.__dict__.setdefault("inference_advantage", None)
        self.__dict__.setdefault("subtask_regeneration_interval", None)
        self.__dict__.setdefault("subtask_generation_enabled", True)
        self.__dict__.setdefault("rlt_enabled", False)
        self.__dict__.setdefault("rlt_embedding_checkpoint", None)
        self.__dict__.setdefault("rlt_head_checkpoint", None)
        self.__dict__.setdefault("rlt_chunk_size", 10)
        self.__dict__.setdefault("rlt_token_dim", None)
        self.__dict__.setdefault("rlt_actor_hidden_dims", None)
        self.__dict__.setdefault("rlt_critic_hidden_dims", None)
        self.__dict__.setdefault("rlt_actor_residual_scale", 0.25)
        self.__dict__.setdefault("rlt_actor_mode", "gaussian")
        self.__dict__.setdefault("rlt_action_std", 0.05)
        self.__dict__.setdefault("rlt_num_critics", 1)
        self.__dict__.setdefault("rlt_bc_beta", 1.0)
        self.__dict__.setdefault("rlt_bc_action_weights", None)
        self.__dict__.setdefault("rlt_jerk_beta", 0.0)
        self.__dict__.setdefault("rlt_reference_dropout_p", 0.5)
        self.__dict__.setdefault("rlt_online_collection_enabled", False)
        self.__dict__.setdefault("rlt_online_training_enabled", False)
        self.__dict__.setdefault("rlt_warmup_episodes", 1)
        self.__dict__.setdefault("rlt_warmup_transitions", 128)
        self.__dict__.setdefault("rlt_replay_capacity", 10000)
        self.__dict__.setdefault("rlt_batch_size", 64)
        self.__dict__.setdefault("rlt_utd_ratio", 1)
        self.__dict__.setdefault("rlt_train_freq_s", 1.0)
        self.__dict__.setdefault("rlt_save_freq_steps", 500)
        self.__dict__.setdefault("rlt_output_dir", "outputs/rlt_online")
        self.__dict__.setdefault("rlt_demo_buffer_path", None)
        self.__dict__.setdefault("rlt_online_buffer_path", None)
        self.__dict__.setdefault("rlt_online_buffer_save_freq_transitions", 100)
        self.__dict__.setdefault("rlt_persist_buffer_on_shutdown", True)
        self.__dict__.setdefault("rlt_actor_lr", 3e-4)
        self.__dict__.setdefault("rlt_critic_lr", 3e-4)
        self.__dict__.setdefault("rlt_discount", 0.99)
        self.__dict__.setdefault("rlt_target_update_tau", 0.005)
        self.__dict__.setdefault("rlt_execute_after_train_steps", 1000000)
        self.__dict__.setdefault("rlt_context_cache_size", 256)
        self.__dict__.setdefault("rlt_transition_queue_size", 256)
        self.__dict__.setdefault("rlt_grad_clip_norm", None)
        self.__dict__.setdefault("rlt_q_abs_max", None)
        self.__dict__.setdefault("rlt_action_deviation_abs_max", None)
        self.__dict__.setdefault("rlt_loss_abs_max", None)
        self.__dict__.setdefault("rlt_safety_patience", 3)


def _compare_observation_states(obs1_state: Any, obs2_state: Any, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    import torch

    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """Check if two observations are similar, under a tolerance threshold. Measures distance between
    observations as the difference in joint-space between the two observations.

    NOTE(fracapuano): This is a very simple check, and it is enough for the current use case.
    An immediate next step is to use (fast) perceptual difference metrics comparing some camera views,
    to surpass this joint-space similarity check.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
