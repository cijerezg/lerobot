from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union

import dacite
import mani_skill.envs.utils.randomization as randomization
import numpy as np
import sapien
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from .base_random_env import BaseRandomEnv, DefaultRandomizationConfig
from .robot.so101 import SO101


MESH_DIR = Path(__file__).parent / "robot" / "meshes"
MOTOR_MESH_FILE = str(MESH_DIR / "xl330_motor.stl")
UNDERARM_MESH_FILE = str(MESH_DIR / "under_arm_so101_slot.stl")
MESH_SCALE = [0.001, 0.001, 0.001]

# The downloaded STLs are millimeter-scale. Poses below are in meters and match
# the SO101 URDF lower-arm frame used by the existing robot mesh assets.
MOTOR_MESH_CENTER = np.array([0.0, -0.0075, -0.0080])
MOTOR_HALF_EXTENTS = np.array([0.0100, 0.0170, 0.0145])
MOTOR_MESH_POSE = sapien.Pose(p=(-MOTOR_MESH_CENTER).tolist())
UNDERARM_MESH_POSE = sapien.Pose(p=[-0.0648499, -0.032, 0.0182], q=euler2quat(np.pi, 0, 0))

UNDERARM_TABLE_Z = 0.012
UNDERARM_ACTOR_XY = (0.3224, 0.0948)
MOTOR_SLOT_CENTER_LOCAL = np.array([-0.1224, 0.0052, 0.0187])
MOTOR_START_XY = (0.29, -0.085)

SCENE_CAMERA_POS = [0.42, -0.48, 0.30]
SCENE_CAMERA_TARGET = [0.24, 0.04, 0.06]
RENDER_CAMERA_POS = [0.42, -0.55, 0.36]
RENDER_CAMERA_TARGET = [0.22, 0.04, 0.05]


@dataclass
class PlaceMotorRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)
    motor_friction_range: Sequence[float] = (0.4, 0.8)
    motor_density_range: Sequence[float] = (700, 700)
    motor_xy_noise: Sequence[float] = (0.025, 0.025)
    underarm_xy_noise: Sequence[float] = (0.010, 0.010)
    position_tolerance: Sequence[float] = (0.018, 0.014, 0.016)
    orientation_tolerance_degrees: float = 45.0


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    diff = _quat_mul(q1, _quat_conjugate(q2))
    return 2.0 * torch.atan2(torch.linalg.norm(diff[..., 1:], dim=-1), torch.abs(diff[..., 0]))


class SceneWristCameraEnv(BaseRandomEnv):
    """Camera base with a moving SO101 wrist camera and a fixed scene camera."""

    WRIST_CAMERA_BASE_POS = (-0.0049, 0.0498, -0.0591)
    WRIST_CAMERA_BASE_ROT_RAD = (
        np.deg2rad(-90),
        np.deg2rad(91),
        np.deg2rad(-35.31),
    )
    WRIST_CAMERA_FOV = np.deg2rad(71)
    SCENE_CAMERA_FOV = np.deg2rad(60)

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config
        wrist_fov_noise = 0
        scene_fov_noise = 0
        if self.domain_randomization:
            wrist_fov_noise = config.wrist_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
            scene_fov_noise = config.third_camera_fov_noise * (2 * self._batched_episode_rng.rand() - 1)
        return [
            CameraConfig(
                "wrist_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.WRIST_CAMERA_FOV + wrist_fov_noise,
                near=0.01,
                far=100,
                mount=self.wrist_camera_mount,
            ),
            CameraConfig(
                "scene_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.SCENE_CAMERA_FOV + scene_fov_noise,
                near=0.01,
                far=100,
                mount=self.camera_mount,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(RENDER_CAMERA_POS, RENDER_CAMERA_TARGET)
        return CameraConfig("render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100)

    def sample_scene_camera_poses(self, n: int):
        config = self.domain_randomization_config
        if not self.domain_randomization:
            static_pose = sapien_utils.look_at(eye=SCENE_CAMERA_POS, target=SCENE_CAMERA_TARGET)
            pose_tensor = static_pose.raw_pose.squeeze()
            return Pose.create(pose_tensor.unsqueeze(0).expand(n, -1))

        pos = common.to_tensor(SCENE_CAMERA_POS, device=self.device)
        target = common.to_tensor(SCENE_CAMERA_TARGET, device=self.device)
        max_offset = common.to_tensor(config.third_camera_pos_noise, device=self.device)
        eyes = randomization.camera.make_camera_rectangular_prism(
            n,
            scale=max_offset,
            center=pos,
            theta=0,
            device=self.device,
        )
        return randomization.camera.noised_look_at(
            eyes,
            target=target,
            look_at_noise=config.third_camera_target_noise,
            view_axis_rot_noise=config.third_camera_rot_noise,
            device=self.device,
        )

    def _update_wrist_camera_pose(self):
        config = self.domain_randomization_config
        gripper_pose = self.agent.robot.links_map["gripper_link"].pose

        base_x, base_y, base_z = self.WRIST_CAMERA_BASE_POS
        base_roll, base_pitch, base_yaw = self.WRIST_CAMERA_BASE_ROT_RAD
        if self.domain_randomization:
            rand_vals = 2 * torch.rand(self.num_envs, 6, device=self.device) - 1
            pos_offset = config.wrist_camera_pos_noise
            rot_noise = config.wrist_camera_rot_noise
            dx = pos_offset[0] * rand_vals[:, 0]
            dy = pos_offset[1] * rand_vals[:, 1]
            dz = pos_offset[2] * rand_vals[:, 2]
            d_roll = rot_noise[0] * rand_vals[:, 3]
            d_pitch = rot_noise[1] * rand_vals[:, 4]
            d_yaw = rot_noise[2] * rand_vals[:, 5]
        else:
            dx = dy = dz = torch.zeros(self.num_envs, device=self.device)
            d_roll = d_pitch = d_yaw = torch.zeros(self.num_envs, device=self.device)

        px, py, pz = base_x + dx, base_y + dy, base_z + dz
        roll_rad = base_roll + d_roll
        pitch_rad = base_pitch + d_pitch
        yaw_rad = base_yaw + d_yaw

        cj, sj = torch.cos(pitch_rad / 2), torch.sin(pitch_rad / 2)
        ck, sk = torch.cos(yaw_rad / 2), torch.sin(yaw_rad / 2)
        ci, si = torch.cos(roll_rad / 2), torch.sin(roll_rad / 2)
        q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, sj * sk, sj * ck, cj * sk

        q = torch.stack(
            [
                q_py_w * ci - q_py_x * si,
                q_py_w * si + q_py_x * ci,
                q_py_y * ci + q_py_z * si,
                q_py_z * ci - q_py_y * si,
            ],
            dim=-1,
        )
        p = torch.stack([px, py, pz], dim=-1)
        self.wrist_camera_mount.set_pose(gripper_pose * Pose.create_from_pq(p=p, q=q))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.camera_mount.set_pose(self.sample_scene_camera_poses(n=len(env_idx)))

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()
        return obs, info

    def _before_control_step(self):
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_scene_camera_poses(n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_wrist_camera_pose()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


class PlaceMotor(SceneWristCameraEnv):
    """
    **Task Description:**
    Pick up a Feetech/XL330-sized motor and place it into the motor slot of an SO101 under-arm.

    **Success Conditions:**
    - the motor center is within the slot tolerance
    - the motor orientation is roughly aligned with the slot
    - the motor was lifted before placement
    - the motor is static and released
    """

    SUPPORTED_ROBOTS = ["so101"]
    SUPPORTED_OBS_MODES = [
        "none",
        "state",
        "state_dict",
        "rgb",
        "rgb+segmentation",
        "rgb+state",
        "rgb+segmentation+state",
        "rgb+depth+segmentation",
        "rgb+depth+segmentation+state",
    ]
    agent: SO101

    def __init__(
        self,
        *args,
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PlaceMotorRandomizationConfig, dict
        ] = PlaceMotorRandomizationConfig(),
        domain_randomization=False,
        **kwargs,
    ):
        if robot_uids != "so101":
            raise NotImplementedError("SO101PlaceMotor-v1 currently supports robot_uids='so101' only")

        self.base_z_rot = 0
        self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        self.domain_randomization_config = PlaceMotorRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PlaceMotorRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PlaceMotorRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=True
            if self.domain_randomization and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.motor_friction_range[0] + cfg.motor_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.motor_density_range[0] + cfg.motor_density_range[1]) / 2
        if self.domain_randomization:
            frictions = self._batched_episode_rng.uniform(
                low=cfg.motor_friction_range[0],
                high=cfg.motor_friction_range[1],
            )
            densities = self._batched_episode_rng.uniform(
                low=cfg.motor_density_range[0],
                high=cfg.motor_density_range[1],
            )
        self.motor_frictions = common.to_tensor(frictions, device=self.device)
        self.motor_densities = common.to_tensor(densities, device=self.device)
        self.motor_half_extents = common.to_tensor(MOTOR_HALF_EXTENTS, device=self.device).repeat(
            self.num_envs, 1
        )
        self.motor_half_sizes = self.motor_half_extents[:, 2]
        self.motor_dimensions = self.motor_half_extents
        self.position_tolerance = common.to_tensor(cfg.position_tolerance, device=self.device)
        self.orientation_tolerance = torch.tensor(
            np.deg2rad(cfg.orientation_tolerance_degrees),
            dtype=torch.float32,
            device=self.device,
        )

        motors = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_multiple_convex_collisions_from_file(
                filename=MOTOR_MESH_FILE,
                scale=MESH_SCALE,
                pose=MOTOR_MESH_POSE,
                material=material,
                density=densities[i],
            )
            builder.add_visual_from_file(
                filename=MOTOR_MESH_FILE,
                scale=MESH_SCALE,
                pose=MOTOR_MESH_POSE,
                material=sapien.render.RenderMaterial(base_color=[0.04, 0.04, 0.04, 1.0]),
            )
            builder.initial_pose = sapien.Pose(p=[0.2, 0, MOTOR_HALF_EXTENTS[2]])
            builder.set_scene_idxs([i])
            motor = builder.build(name=f"motor-{i}")
            motors.append(motor)
            self.remove_from_state_dict_registry(motor)

        self.motor = Actor.merge(motors, name="motor")
        self.add_to_state_dict_registry(self.motor)

        underarms = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_nonconvex_collision_from_file(
                filename=UNDERARM_MESH_FILE,
                scale=MESH_SCALE,
                pose=UNDERARM_MESH_POSE,
            )
            builder.add_visual_from_file(
                filename=UNDERARM_MESH_FILE,
                scale=MESH_SCALE,
                pose=UNDERARM_MESH_POSE,
                material=sapien.render.RenderMaterial(base_color=[0.92, 0.92, 0.88, 1.0]),
            )
            builder.initial_pose = sapien.Pose(p=[UNDERARM_ACTOR_XY[0], UNDERARM_ACTOR_XY[1], UNDERARM_TABLE_Z])
            builder.set_scene_idxs([i])
            underarm = builder.build_kinematic(name=f"underarm-slot-{i}")
            underarms.append(underarm)
            self.remove_from_state_dict_registry(underarm)

        self.underarm = Actor.merge(underarms, name="underarm_slot")
        self.add_to_state_dict_registry(self.underarm)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.motor)
            self.remove_object_from_greenscreen(self.underarm)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429],
            q=euler2quat(0, 0, np.pi / 2),
        )
        self._load_camera_mount()
        self._randomize_robot_color()

        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.006,
            material=sapien.render.RenderMaterial(base_color=[0.0, 1.0, 0.0, 0.5]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="motor_goal_site")
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            if not hasattr(self, "motor_lifted_once"):
                self.motor_lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.motor_lifted_once[env_idx] = False

            self.agent.robot.set_qpos(
                self.rest_qpos
                + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)))

            motor_xy = torch.tensor(MOTOR_START_XY, device=self.device, dtype=torch.float32).repeat(b, 1)
            underarm_xy = torch.tensor(UNDERARM_ACTOR_XY, device=self.device, dtype=torch.float32).repeat(b, 1)
            if self.domain_randomization:
                motor_noise = torch.tensor(
                    self.domain_randomization_config.motor_xy_noise,
                    device=self.device,
                    dtype=torch.float32,
                )
                underarm_noise = torch.tensor(
                    self.domain_randomization_config.underarm_xy_noise,
                    device=self.device,
                    dtype=torch.float32,
                )
                motor_xy += (2 * torch.rand((b, 2), device=self.device) - 1) * motor_noise
                underarm_xy += (2 * torch.rand((b, 2), device=self.device) - 1) * underarm_noise

            underarm_xyz = torch.zeros((b, 3), device=self.device)
            underarm_xyz[:, :2] = underarm_xy
            underarm_xyz[:, 2] = UNDERARM_TABLE_Z
            underarm_q = torch.tensor(euler2quat(0, 0, 0), device=self.device, dtype=torch.float32).repeat(b, 1)
            self.underarm.set_pose(Pose.create_from_pq(underarm_xyz, underarm_q))

            goal_xyz = underarm_xyz + torch.tensor(
                MOTOR_SLOT_CENTER_LOCAL,
                device=self.device,
                dtype=torch.float32,
            )
            goal_q = torch.tensor(euler2quat(0, 0, 0), device=self.device, dtype=torch.float32).repeat(b, 1)
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz, goal_q))
            if not hasattr(self, "motor_goal_q"):
                self.motor_goal_q = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            self.motor_goal_q[env_idx] = goal_q

            motor_xyz = torch.zeros((b, 3), device=self.device)
            motor_xyz[:, :2] = motor_xy
            motor_xyz[:, 2] = self.motor_half_sizes[env_idx] + 0.001
            motor_q = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.motor.set_pose(Pose.create_from_pq(motor_xyz, motor_q))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            qpos = qpos + torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_motor_grasped=info["is_motor_grasped"],
                motor_pose=self.motor.pose.raw_pose,
                underarm_pose=self.underarm.pose.raw_pose,
                goal_pose=self.goal_site.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_motor_pos=self.motor.pose.p - self.agent.tcp_pos,
                motor_to_goal_pos=self.goal_site.pose.p - self.motor.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    motor_dimensions=self.motor_dimensions,
                    motor_friction=self.motor_frictions,
                    motor_density=self.motor_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        motor_pos = self.motor.pose.p
        goal_pos = self.goal_site.pose.p
        offset = motor_pos - goal_pos
        inside_slot = torch.all(torch.abs(offset) <= self.position_tolerance, dim=-1)
        orientation_error = _quat_angle(self.motor.pose.q, self.motor_goal_q)
        orientation_aligned = orientation_error <= self.orientation_tolerance

        motor_lifted_now = motor_pos[..., 2] >= (self.motor_half_sizes + 0.018)
        if not hasattr(self, "motor_lifted_once"):
            self.motor_lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motor_lifted_once |= motor_lifted_now

        motor_vel = torch.linalg.norm(self.motor.linear_velocity, axis=-1)
        is_motor_static = motor_vel <= 2e-2
        is_motor_grasped = self.agent.is_grasping(self.motor)
        is_robot_static = self.agent.is_static()
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_motor = self.agent.is_touching(self.motor)
        robot_touching_underarm = self.agent.is_touching(self.underarm)
        success = (
            inside_slot
            & orientation_aligned
            & self.motor_lifted_once
            & is_motor_static
            & (~robot_touching_motor)
            & is_robot_static
        )
        return {
            "success": success,
            "inside_slot": inside_slot,
            "orientation_aligned": orientation_aligned,
            "orientation_error": orientation_error,
            "motor_lifted": motor_lifted_now,
            "motor_lifted_once": self.motor_lifted_once,
            "motor_vel": motor_vel,
            "is_motor_static": is_motor_static,
            "is_motor_grasped": is_motor_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_motor": robot_touching_motor,
            "robot_touching_underarm": robot_touching_underarm,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_motor_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.motor.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_motor_dist))
        reward = reaching_reward

        motor_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.motor.pose.p, axis=1)
        place_reward = 1 - torch.tanh(7.0 * motor_to_goal_dist)
        orientation_reward = 1 - torch.tanh(2.0 * info["orientation_error"])
        reward[info["is_motor_grasped"]] = (3 + place_reward + orientation_reward)[info["is_motor_grasped"]]
        reward[info["inside_slot"]] = (5 + place_reward + orientation_reward)[info["inside_slot"]]
        reward[info["success"]] = 10
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_underarm"].float()
        reward -= 1 * (~info["motor_lifted"]).float()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10


@register_env("SO101PlaceMotor-v1", max_episode_steps=120)
class PlaceMotorEnv(PlaceMotor):
    pass
