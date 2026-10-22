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
    # A relative path resolves against the scene file's directory (scenes/<layout>/background_top.png).
    background: str | None = None
    background_feather_px: int = 2
    # Per-channel affine colour correction of the render (real = gain * sim + offset, fitted on real frames), applied
    # before the composite so the real background stays untouched.
    color_gain: list[float] | None = None
    color_offset: list[float] | None = None
    # Composite: when set, only these free bodies (the task objects, inpainted out of the background) and the robot are
    # always rendered; every other free body (clutter) keeps the real background pixels until it moves (> 3 mm or 1 deg
    # from its settled start pose). None = every free body is rendered.
    render_always: list[str] | None = None
    # Composite: static (world-body) geoms rendered instead of the real background pixels, by geom name, e.g. the table
    # (`table`, `table_top`): the sim table is then one texture in this view too, with no real clutter left in it.
    render_static: list[str] | None = None
    # Per-channel gain on the render_static pixels after color_gain/offset (the table texture is authored in the wrist
    # camera's colours; the top camera sees the wood lighter and less saturated).
    render_static_gain: list[float] | None = None


@RobotConfig.register_subclass("rebot_b601_sim")
@dataclass
class RebotB601SimRobotConfig(RobotConfig):
    """MuJoCo stand-in for the reBot B601-DM follower (lerobot/robots/rebot_b601_sim)."""

    # A layout name under robots/rebot_b601_sim/scenes/<name>/scene.xml (the tasks.py registry; README there), or a
    # path to such a file (it includes the DM robot and names the cameras below).
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
    # Subtask cursor and staged reward from the MuJoCo state (tasks.py scorer of the scene) instead of the operator console.
    auto_subtasks: bool = False
    park_pose: dict[str, float] = field(default_factory=lambda: dict(_FOLLOWER.park_pose))
    park_deg_per_s: float = _FOLLOWER.park_deg_per_s
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(_FOLLOWER.joint_limits))
