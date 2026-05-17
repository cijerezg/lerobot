#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("squint_so101")
@dataclass
class SquintSO101RobotConfig(RobotConfig):
    """SO101 simulator backed by the vendored Squint ManiSkill task suite."""

    id: str | None = "squint_so101"
    # Kept for compatibility with existing experiment configs. The simulator is
    # vendored under lerobot.robots.squint_so101.sim and no longer reads this.
    squint_root: str = ""
    env_id: str | None = None
    dataset_root: str | None = None
    dataset_repo_id: str | None = None
    task: str | None = None
    control_mode: str = "pd_joint_target_delta_pos"
    obs_mode: str = "rgb+segmentation"
    render_mode: str = "rgb_array"
    domain_randomization: bool = False
    seed: int = 0
    sensor_width: int = 224
    sensor_height: int = 224
    camera_width: int = 800
    camera_height: int = 600
    camera_pose_path: str = ""
    top_camera_name: str = "top"
    side_camera_name: str = "side"
    video_dir: str = "outputs/rlt_squint_videos"
    video_fps: int = 20
    video_every_episodes: int = 1
    video_max_episodes: int = 20
    video_max_frames: int = 300
    white_x_background: bool = True
    max_episode_steps: int | None = None
    success_reward_threshold: float = 0.5
    action_clip: float | None = None
    reset_after_terminal: bool = True
    reset_seed_on_terminal: bool = False
    use_dataset_initial_state: bool = True
    show_gripper_contact_markers: bool = False
    use_marker_grasp_assist: bool = True
    follow_calibration_path: str = ""
    bootstrap_dataset_episode: int | None = None
    bootstrap_dataset_episodes: list[int] | None = None
    bootstrap_dataset_episode_interval: int = 1
    bootstrap_dataset_action_stride: int = 1
    marker_xy_offset: list[float] | None = None
    marker_yaw_degrees: float = 0.0
