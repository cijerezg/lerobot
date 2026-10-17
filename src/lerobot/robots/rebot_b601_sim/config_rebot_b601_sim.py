from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig

# The follower's defaults are the single source for the park pose and the soft limits.
_FOLLOWER = RebotB601FollowerConfig(port="")


@CameraConfig.register_subclass("mujoco")
@dataclass
class MujocoCameraConfig(CameraConfig):
    """One rendered view: ``camera`` is the MJCF ``<camera name=...>``; ``use_depth`` adds ``{cam}.depth``.
    ``background`` (image path, this camera's size) composites the render SIMPLER-style: the real image
    supplies everything attached to the world body, the sim supplies every other body (robot, free objects)."""

    camera: str
    use_depth: bool = False
    background: str | None = None
    background_feather_px: int = 2
    # Per-channel affine colour correction of the render (real = gain * sim + offset, fitted on real frames), applied
    # before the composite so the real background stays untouched.
    color_gain: list[float] | None = None
    color_offset: list[float] | None = None


@RobotConfig.register_subclass("rebot_b601_sim")
@dataclass
class RebotB601SimRobotConfig(RobotConfig):
    """MuJoCo stand-in for the reBot B601-DM follower (lerobot/robots/rebot_b601_sim)."""

    # MJCF scene that includes the DM robot and names the cameras below
    # (migration/real2sim_2026-10-17/work/scene_nail/scene_nail.xml).
    scene: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # One send_action advances 1/fps s of physics in `substeps` steps (probes/action_trace_sim defaults).
    fps: int = 30
    substeps: int = 17
    gravcomp: bool = True
    # Per-joint slew of the commanded target in deg/s, the follower's POS_VEL velocity
    # (config_rl.yaml: [40]*6 + [180]). None drives the position actuators with the raw target,
    # the executor action_trace_sim measured against (sim lag about half the real lag).
    pos_vel_velocity: list[float] | None = None
    # Episodes to run before the robot stops the process (SIGINT, the operator's Ctrl-C); None = unbounded.
    episodes: int | None = None
    park_pose: dict[str, float] = field(default_factory=lambda: dict(_FOLLOWER.park_pose))
    park_deg_per_s: float = _FOLLOWER.park_deg_per_s
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(_FOLLOWER.joint_limits))
