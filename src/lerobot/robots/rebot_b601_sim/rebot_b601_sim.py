"""MuJoCo stand-in for the reBot B601-DM follower: the recreated scene behind the same Robot seam.

``RobotEnv`` only calls ``get_observation()`` / ``send_action()``; the RTC runtime adds ``park()`` at
the episode end. One ``send_action`` advances ``1/fps`` s of physics through the position actuators
(the executor of ``probes/action_trace_sim.execute``). ``get_observation`` renders every configured
camera from the MJCF camera it names, plus ``{cam}.depth`` (uint16, 0.1 mm per level, D405 range)
where ``use_depth`` is set: the keys and dtypes the real follower emits. State and action are
degrees in the follower's motor order; ``to_qpos`` / ``from_qpos`` map them onto the 8-slot MJCF
configuration with the constants of ``action_trace_sim``.
"""

import logging
import math
import os
import signal
import threading
from functools import cached_property
from pathlib import Path

import cv2
import mujoco
import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_rebot_b601_sim import RebotB601SimRobotConfig
from .nail_task import NailTaskScorer

logger = logging.getLogger(__name__)

# Set by park() after the configured number of episodes; sim_autostart stops starting episodes.
STOP_EPISODES = threading.Event()

ARM_DOF = 6
GRIPPER_IDX = 6
NQ_ROBOT = 8
# Dataset gripper units: degrees in [-238.8, 0], 0 = closed; the MJCF drives two coupled slides in
# [0, 0.05] m, 0 = closed (probes/action_trace_sim.py).
GRIPPER_DEG_OPEN = -238.8
GRIPPER_SLIDE_OPEN = 0.05
ARM_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")
# Depth sidecar units and the real sidecar's effective range (migration/real2sim_2026-10-17 scene_render.py).
DEPTH_UNITS_M = 1e-4
DEPTH_MIN_M = 0.07
DEPTH_MAX_M = 2.5


def to_qpos(action_deg: np.ndarray) -> np.ndarray:
    slide = float(action_deg[GRIPPER_IDX]) / GRIPPER_DEG_OPEN * GRIPPER_SLIDE_OPEN
    slide = float(np.clip(slide, 0.0, GRIPPER_SLIDE_OPEN))
    return np.concatenate([np.deg2rad(action_deg[:ARM_DOF]), [slide, slide]])


def from_qpos(qpos: np.ndarray) -> np.ndarray:
    gripper = float(qpos[ARM_DOF]) / GRIPPER_SLIDE_OPEN * GRIPPER_DEG_OPEN
    return np.concatenate([np.rad2deg(qpos[:ARM_DOF]), [gripper]])


def depth_sidecar(depth_m: np.ndarray) -> np.ndarray:
    kept = np.where((depth_m >= DEPTH_MIN_M) & (depth_m <= DEPTH_MAX_M), depth_m, 0.0)
    return np.round(kept / DEPTH_UNITS_M).astype(np.uint16)


class RebotB601Sim(Robot):
    config_class = RebotB601SimRobotConfig
    name = "rebot_b601_sim"

    def __init__(self, config: RebotB601SimRobotConfig):
        super().__init__(config)
        self.config = config
        self.motor_names = list(config.joint_limits)
        self.cameras = dict(config.cameras)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._q0 = 0
        self._ctrl_deg = np.zeros(len(self.motor_names))
        # GL contexts are thread-bound: RobotEnv renders once at setup on the main thread, then
        # every frame from the env worker, so each thread owns its renderers.
        self._local = threading.local()
        self.episodes_done = 0
        self.subtask_scorer: NailTaskScorer | None = None
        self._backgrounds: dict[str, np.ndarray] = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (config.height, config.width, 3) for cam, config in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.model is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        spec = mujoco.MjSpec.from_file(str(self.config.scene))
        if self.config.gravcomp:
            for name in ARM_BODIES:
                spec.body(name).gravcomp = 1.0
        self.model = spec.compile()
        self.model.opt.timestep = 1.0 / (self.config.fps * self.config.substeps)
        self.data = mujoco.MjData(self.model)
        self._q0 = int(self.model.jnt_qposadr[self.model.joint("joint1").id])
        for cam, config in self.config.cameras.items():
            if config.background is not None:
                self._backgrounds[cam] = cv2.cvtColor(cv2.imread(config.background), cv2.COLOR_BGR2RGB)
        self._dynamic_geom = self.model.geom_bodyid != 0
        self._static_geom = {cam: np.isin(np.arange(self.model.ngeom), [self.model.geom(n).id for n in (c.render_static or [])]) for cam, c in self.config.cameras.items()}
        free = [b for b in range(self.model.nbody) if self.model.body_jntnum[b] > 0 and self.model.jnt_type[self.model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
        always = {self.model.body(n).id for c in self.config.cameras.values() for n in (c.render_always or [])}
        self._lazy_bodies = [b for b in free if b not in always] if always else []
        self.subtask_scorer = NailTaskScorer(self.model, self.data, lambda: float(self._present_deg()[GRIPPER_IDX])) if self.config.auto_subtasks else None
        self._reset_scene()
        logger.info(f"{self} loaded {self.config.scene} (timestep {self.model.opt.timestep * 1e3:.2f} ms).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _reset_scene(self) -> None:
        """Objects back to their MJCF poses, arm at qpos0 (the park pose), targets holding it."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self._qpos_robot()
        self._ctrl_deg = from_qpos(self._qpos_robot())
        mujoco.mj_forward(self.model, self.data)
        # let the objects settle (neighbours placed from the image overlap by a few mm and push apart) before the start
        # poses are taken: the composite renders a clutter body only once it has moved from here
        for _ in range(int(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self._start_xpos = self.data.xpos.copy(); self._start_xquat = self.data.xquat.copy()
        if self.subtask_scorer is not None:
            self.subtask_scorer.reset()

    def _qpos_robot(self) -> np.ndarray:
        return self.data.qpos[self._q0 : self._q0 + NQ_ROBOT]

    def _present_deg(self) -> np.ndarray:
        return from_qpos(self._qpos_robot())

    def _renderer(self, cam: str) -> mujoco.Renderer:
        renderers = getattr(self._local, "renderers", None)
        if renderers is None:
            renderers = self._local.renderers = {}
        if cam not in renderers:
            config = self.config.cameras[cam]
            renderers[cam] = mujoco.Renderer(self.model, config.height, config.width)
        return renderers[cam]

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict = {f"{motor}.pos": float(pos) for motor, pos in zip(self.motor_names, self._present_deg(), strict=True)}
        for cam, config in self.config.cameras.items():
            renderer = self._renderer(cam)
            renderer.update_scene(self.data, config.camera)
            obs_dict[cam] = renderer.render()
            if config.color_gain is not None:
                obs_dict[cam] = np.clip(obs_dict[cam] * np.asarray(config.color_gain) + np.asarray(config.color_offset or [0.0, 0.0, 0.0]), 0, 255).astype(np.uint8)
            if cam in self._backgrounds:
                obs_dict[cam] = self._composite(cam, renderer, config, obs_dict[cam], self._backgrounds[cam])
            if config.use_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(self.data, config.camera)
                obs_dict[f"{cam}.depth"] = depth_sidecar(renderer.render())
                renderer.disable_depth_rendering()
        return obs_dict

    def _composite(self, cam: str, renderer: mujoco.Renderer, config, rgb: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Sim pixels of the dynamic bodies (and the camera's render_static geoms) over the real background, mask edge feathered."""
        renderer.enable_segmentation_rendering()
        renderer.update_scene(self.data, config.camera)
        geom = renderer.render()[:, :, 0]
        renderer.disable_segmentation_rendering()
        render = self._dynamic_geom | self._static_geom[cam]
        for b in self._lazy_bodies:  # real pixels until the body leaves its start pose
            moved = np.linalg.norm(self.data.xpos[b] - self._start_xpos[b]) > 0.003 or abs(float(np.dot(self.data.xquat[b], self._start_xquat[b]))) < np.cos(np.deg2rad(1.0))
            if not moved:
                render[self.model.geom_bodyid == b] = False
        mask = ((geom >= 0) & render[np.clip(geom, 0, None)]).astype(np.uint8)
        if config.render_static_gain is not None:
            static = (geom >= 0) & self._static_geom[cam][np.clip(geom, 0, None)]
            rgb = rgb.copy(); rgb[static] = np.clip(rgb[static] * np.asarray(config.render_static_gain), 0, 255).astype(np.uint8)
        # 3x3 opening: silhouette edges of the free bodies come back labelled with robot geom ids (thin stray lines)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32)
        if config.background_feather_px > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), config.background_feather_px)
        return (mask[:, :, None] * rgb + (1.0 - mask[:, :, None]) * background).astype(np.uint8)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Set the joint targets in degrees and advance one control tick of physics."""
        goal_pos = {key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")}
        for motor_name, (min_limit, max_limit) in self.config.joint_limits.items():
            goal_pos[motor_name] = max(min_limit, min(max_limit, goal_pos[motor_name]))
        goal_deg = np.array([goal_pos[name] for name in self.motor_names])
        slew = None if self.config.pos_vel_velocity is None else np.asarray(self.config.pos_vel_velocity) * self.model.opt.timestep
        for _ in range(self.config.substeps):
            self._ctrl_deg = goal_deg if slew is None else self._ctrl_deg + np.clip(goal_deg - self._ctrl_deg, -slew, slew)
            self.data.ctrl[:] = to_qpos(self._ctrl_deg)
            mujoco.mj_step(self.model, self.data)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def park(self, on_step=None) -> None:
        """Descend to config.park_pose at config.park_deg_per_s, then put the scene back to its
        start pose. This is the sim's episode boundary: the runtime parks the follower at every
        episode end, and the next episode starts from the reset scene. `on_step` gets each ramp
        target like the follower's park. After config.episodes parks the process gets SIGINT.
        """
        present = self._present_deg()
        target = np.array([self.config.park_pose[name] for name in self.motor_names])
        gap = float(np.abs(target - present).max())
        steps = max(math.ceil(self.config.fps), math.ceil(self.config.fps * gap / self.config.park_deg_per_s))
        for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
            sent = self.send_action(dict(zip(self.action_features, present + (target - present) * t, strict=True)))
            if on_step is not None:
                on_step(sent)
        self._reset_scene()
        self.episodes_done += 1
        logger.info(f"{self} parked and reset the scene after episode {self.episodes_done}.")
        if self.config.episodes is not None and self.episodes_done >= self.config.episodes:
            logger.info(f"{self} ran its {self.config.episodes} episodes; stopping the process.")
            STOP_EPISODES.set()
            os.kill(os.getpid(), signal.SIGINT)

    @check_if_not_connected
    def disconnect(self) -> None:
        for renderer in getattr(self._local, "renderers", {}).values():
            renderer.close()
        self._local = threading.local()
        self.model = None
        self.data = None
        logger.info(f"{self} disconnected.")
