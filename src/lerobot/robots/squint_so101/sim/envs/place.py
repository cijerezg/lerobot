from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

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
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so100 import SO100
from .robot.so101 import SO101

MARKER_BASE_SIZE = (0.110, 0.070)
# Front x of base_so101_v2_convex.obj after applying the URDF collision origin.
SO101_BASE_FRONT_X = 0.06524555
MARKER_CUBE_START_DISTANCE_FROM_BASE_FRONT = 0.220
MARKER_CENTER_DISTANCE_FROM_BASE_FRONT = 0.120
MARKER_CENTER_LEFT_OF_BASE_CENTER = 0.110
PLACE_CUBE_FACE_SIZE = 0.024
MARKER_TOP_CAMERA_POS = [0.45, 1.00, 0.35]
MARKER_TOP_CAMERA_TARGET = [0.55, 0.15, 0.12]
MARKER_SIDE_CAMERA_POS = [0.25, -0.70, 0.18]
MARKER_SIDE_CAMERA_TARGET = [0.65, 0.30, 0.05]
GRIPPER_TIP_CONTACT_MARKER_RADIUS = 0.008
GRIPPER_TIP_CONTACT_MIN_FORCE = 0.02
GRIPPER_TIP_CONTACT_EXTRA_DISTANCE = 0.012
GRIPPER_TIP_CONTACT_HIDDEN_Z = -10.0


@dataclass
class PlaceRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for Place task, extending wrist camera randomization."""

    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Cube-specific randomization
    cube_half_size_range: Sequence[float] = (PLACE_CUBE_FACE_SIZE / 2, PLACE_CUBE_FACE_SIZE / 2)
    # Can-specific randomization
    can_radius_range: Sequence[float] = (0.028 / 2, 0.038 / 2)
    can_half_height_range: Sequence[float] = (0.05 / 2, 0.07 / 2)
    # Bin randomization (half sizes)
    bin_half_size_x_range: Sequence[float] = (0.07 / 2, 0.09 / 2)
    bin_half_size_y_range: Sequence[float] = (0.09 / 2, 0.11 / 2)
    bin_half_size_z_range: Sequence[float] = (0.024 / 2, 0.036 / 2)

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = False


class Place(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up an item (cube or can) and place it at a target.

    **Randomizations:**
    - the item's xy position is randomized on top of a table
    - the item's z-axis rotation is randomized
    - the target's xy position is randomized (non-overlapping with item)

    **Success Conditions:**
    - the item is in the target xy range
    - for marker targets, the item was lifted first and released on the marker
    - for bin targets, the robot is not touching the item or the bin and is static
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
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
    agent: Union[SO100, SO101]

    @property
    def _default_human_render_camera_configs(self):
        if getattr(self, "target_type", None) == "marker":
            pose = sapien_utils.look_at(MARKER_TOP_CAMERA_POS, MARKER_TOP_CAMERA_TARGET)
            return CameraConfig("render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100)
        return super()._default_human_render_camera_configs

    def __init__(
        self,
        *args,
        item_type="cube",
        robot_uids="so101",
        control_mode="pd_joint_target_delta_pos",
        target_type="bin",
        domain_randomization_config: Union[PlaceRandomizationConfig, dict] = PlaceRandomizationConfig(),
        domain_randomization=False,
        spawn_box_pos=[0.3, 0],
        spawn_box_half_size=0.2 / 2,
        marker_xy_offset: Sequence[float] | None = None,
        marker_yaw_degrees: float = 0.0,
        show_gripper_contact_markers: bool = False,
        use_marker_grasp_assist: bool = True,
        **kwargs,
    ):
        self.item_type = item_type
        if target_type not in {"bin", "marker"}:
            raise NotImplementedError(f"Unknown target_type: {target_type}")
        self.target_type = target_type

        # Robot-specific configuration
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        # Handle domain randomization config
        self.domain_randomization_config = PlaceRandomizationConfig()
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged_domain_randomization_config, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PlaceRandomizationConfig,
                data=merged_domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PlaceRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size
        if marker_xy_offset is None:
            marker_xy_offset = (
                SO101_BASE_FRONT_X + MARKER_CENTER_DISTANCE_FROM_BASE_FRONT - self.spawn_box_pos[0],
                MARKER_CENTER_LEFT_OF_BASE_CENTER,
            )
        if len(marker_xy_offset) != 2:
            raise ValueError(f"marker_xy_offset must contain exactly two values, got {marker_xy_offset}")
        self.marker_xy_offset = tuple(float(value) for value in marker_xy_offset)
        self.marker_yaw = np.deg2rad(float(marker_yaw_degrees))
        self.show_gripper_contact_markers = bool(show_gripper_contact_markers)
        self.use_marker_grasp_assist = bool(use_marker_grasp_assist)

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )
        if self.target_type == "marker" and hasattr(self, "base_camera_settings"):
            self.base_camera_settings = dict(
                pos=MARKER_SIDE_CAMERA_POS,
                target=MARKER_SIDE_CAMERA_TARGET,
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
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        if self.target_type == "marker":
            self._set_actor_visual_color(self.table_scene.table, [0.02, 0.02, 0.02, 1.0])
            self._build_dark_tabletop_overlay()

        if self.item_type not in ["cube", "can"]:
            raise NotImplementedError(f"Unknown item_type: {self.item_type}")

        # Default values
        colors = np.zeros((self.num_envs, 3))
        colors[:, 0] = 1  # Red
        if self.target_type == "marker" and self.item_type == "cube":
            colors[:, :] = [1.0, 0.62, 0.20]  # Dataset cube is tan/orange.
        cfg = self.domain_randomization_config
        frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
        densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2

        if self.item_type == "cube":
            half_sizes = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.cube_half_size_range[1]
                    + self.domain_randomization_config.cube_half_size_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_sizes = self._batched_episode_rng.uniform(
                    low=cfg.cube_half_size_range[0],
                    high=cfg.cube_half_size_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
            self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)

        elif self.item_type == "can":
            colors = np.zeros((self.num_envs, 3))
            colors[:, :] = 0
            colors[:, 2] = 1  # blue
            half_radii = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_radius_range[1]
                    + self.domain_randomization_config.can_radius_range[0]
                )
                / 2
            )
            half_heights = (
                np.ones(self.num_envs)
                * (
                    self.domain_randomization_config.can_half_height_range[1]
                    + self.domain_randomization_config.can_half_height_range[0]
                )
                / 2
            )
            if self.domain_randomization:
                half_radii = self._batched_episode_rng.uniform(
                    low=cfg.can_radius_range[0],
                    high=cfg.can_radius_range[1],
                )
                half_heights = self._batched_episode_rng.uniform(
                    low=cfg.can_half_height_range[0],
                    high=cfg.can_half_height_range[1],
                )
                if cfg.randomize_item_color:
                    colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                frictions = self._batched_episode_rng.uniform(
                    low=cfg.item_friction_range[0],
                    high=cfg.item_friction_range[1],
                )
                densities = self._batched_episode_rng.uniform(
                    low=cfg.item_density_range[0],
                    high=cfg.item_density_range[1],
                )
            self.item_half_radii = common.to_tensor(half_radii, device=self.device)
            self.item_half_heights = common.to_tensor(half_heights, device=self.device)
            self.item_half_sizes = self.item_half_heights
            self.item_dimensions = torch.stack(
                [self.item_half_radii, self.item_half_radii, self.item_half_heights], dim=-1
            )

        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        # Build items
        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )

            if self.item_type == "cube":
                builder.add_box_collision(
                    half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
                )
                builder.add_box_visual(
                    half_size=[half_sizes[i]] * 3,
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                )
                builder.initial_pose = sapien.Pose(
                    p=[0.2, 0, half_sizes[i]]
                )  # Offset to avoid collision with bin at creation

            elif self.item_type == "can":
                cylinder_pose = sapien.Pose(q=euler2quat(0, np.pi / 2, 0))
                builder.add_cylinder_collision(
                    radius=half_radii[i],
                    half_length=half_heights[i],
                    material=material,
                    density=densities[i],
                    pose=cylinder_pose,
                )
                builder.add_cylinder_visual(
                    radius=half_radii[i],
                    half_length=half_heights[i],
                    material=sapien.render.RenderMaterial(base_color=colors[i]),
                    pose=cylinder_pose,
                )
                builder.initial_pose = sapien.Pose(
                    p=[0.2, 0, half_heights[i]]
                )  # Offset to avoid collision with bin at creation

            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Build targets (per-env for domain randomization)
        target_color = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        x_color = sapien.render.RenderMaterial(base_color=[0.0, 0.0, 0.0, 1.0])
        thickness = 0.005
        self.bin_thickness = thickness

        # Default target half sizes (mid-range)
        cfg = self.domain_randomization_config
        bin_half_sizes_x = (
            np.ones(self.num_envs) * (cfg.bin_half_size_x_range[0] + cfg.bin_half_size_x_range[1]) / 2
        )
        bin_half_sizes_y = (
            np.ones(self.num_envs) * (cfg.bin_half_size_y_range[0] + cfg.bin_half_size_y_range[1]) / 2
        )
        bin_half_sizes_z = (
            np.ones(self.num_envs) * (cfg.bin_half_size_z_range[0] + cfg.bin_half_size_z_range[1]) / 2
        )

        if self.domain_randomization:
            bin_half_sizes_x = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_x_range[0], high=cfg.bin_half_size_x_range[1]
            )
            bin_half_sizes_y = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_y_range[0], high=cfg.bin_half_size_y_range[1]
            )
            bin_half_sizes_z = self._batched_episode_rng.uniform(
                low=cfg.bin_half_size_z_range[0], high=cfg.bin_half_size_z_range[1]
            )

        if self.target_type == "marker":
            bin_half_sizes_x = np.ones(self.num_envs) * (MARKER_BASE_SIZE[0] / 2)
            bin_half_sizes_y = np.ones(self.num_envs) * (MARKER_BASE_SIZE[1] / 2)
            bin_half_sizes_z = np.ones(self.num_envs) * (thickness / 2)

        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions = torch.stack(
            [self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1
        )

        targets = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()

            target_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
            target_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=target_center_pose, half_size=target_center_half_size)
            builder.add_box_visual(
                pose=target_center_pose, half_size=target_center_half_size, material=target_color
            )

            if self.target_type == "bin":
                # Bin walls
                for j in [-1, 1]:
                    # Y walls
                    y = j * target_center_half_size[1]
                    wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                    wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                    builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                    builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=target_color)
                    # X walls
                    x = j * target_center_half_size[0]
                    wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                    wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                    builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                    builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=target_color)
            else:
                bar_half_length = np.linalg.norm(bin_half_size[:2]) * 0.95
                bar_half_width = max(0.004, min(bin_half_size[0], bin_half_size[1]) * 0.07)
                bar_half_height = 0.0008
                bar_pose_z = thickness + bar_half_height
                diagonal_angle = np.arctan2(bin_half_size[1], bin_half_size[0])
                for angle in (diagonal_angle, -diagonal_angle):
                    builder.add_box_visual(
                        pose=sapien.Pose(
                            p=[0.0, 0.0, bar_pose_z],
                            q=euler2quat(0, 0, angle),
                        ),
                        half_size=[bar_half_length, bar_half_width, bar_half_height],
                        material=x_color,
                    )

            initial_z = 0.0 if self.target_type == "marker" else bin_half_size[2]
            initial_pose_kwargs = {"p": [-0.2, 0, initial_z]}
            if self.target_type == "marker":
                initial_pose_kwargs["q"] = euler2quat(0, 0, self.marker_yaw)
            builder.initial_pose = sapien.Pose(**initial_pose_kwargs)  # Offset to avoid collision with item at creation
            builder.set_scene_idxs([i])
            if self.target_type == "marker":
                bin_actor = builder.build_kinematic(name=f"bin-{i}")
            else:
                bin_actor = builder.build(name=f"bin-{i}")
            targets.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(targets, name="bin")
        self.add_to_state_dict_registry(self.bin)

        if self.target_type == "marker":
            self.bin_radius = torch.minimum(self.bin_half_sizes_x, self.bin_half_sizes_y)
        else:
            self.bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], dim=-1)

        # Set up greenscreening - keep robot, item, and target visible
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bin)

        # Convert rest_qpos to tensor
        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        # Table pose
        self.table_pose = Pose.create_from_pq(p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))

        # Build camera mount
        self._load_camera_mount()

        # Randomize robot color
        self._randomize_robot_color()

        # Goal site
        goal_builder = self.scene.create_actor_builder()
        if self.target_type == "bin":
            goal_builder.add_sphere_visual(
                radius=0.01,
                material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
            )
        else:
            goal_builder.add_sphere_visual(
                radius=0.0001,
                material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0]),
            )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)
        self.left_tip_contact_marker = None
        self.right_tip_contact_marker = None
        if self.show_gripper_contact_markers:
            self._build_gripper_contact_markers()

    def _set_actor_visual_color(self, actor: Actor, color: Sequence[float]) -> None:
        for obj in actor._objs:
            entity = getattr(obj, "entity", obj)
            render_body_component: RenderBodyComponent = entity.find_component_by_type(RenderBodyComponent)
            if render_body_component is None:
                continue
            for render_shape in render_body_component.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color(color)

    def _build_dark_tabletop_overlay(self) -> None:
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            pose=sapien.Pose([0.0, 0.0, 0.0]),
            half_size=[2.0, 2.0, 0.0002],
            material=sapien.render.RenderMaterial(base_color=[0.02, 0.02, 0.02, 1.0]),
        )
        builder.initial_pose = sapien.Pose(p=[0.32, 0.0, 0.0002])
        self.tabletop_overlay = builder.build_kinematic(name="dataset-dark-tabletop")
        self.remove_from_state_dict_registry(self.tabletop_overlay)

    def _build_gripper_contact_markers(self) -> None:
        material = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0])
        marker_sets: dict[str, list[Any]] = {"left": [], "right": []}
        for scene_idx in range(self.num_envs):
            for side, markers in marker_sets.items():
                builder = self.scene.create_actor_builder()
                builder.add_sphere_visual(
                    radius=GRIPPER_TIP_CONTACT_MARKER_RADIUS,
                    material=material,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, GRIPPER_TIP_CONTACT_HIDDEN_Z])
                builder.set_scene_idxs([scene_idx])
                marker = builder.build_kinematic(name=f"{side}_gripper_tip_contact-{scene_idx}")
                markers.append(marker)
                self.remove_from_state_dict_registry(marker)

        self.left_tip_contact_marker = Actor.merge(marker_sets["left"], name="left_gripper_tip_contact")
        self.right_tip_contact_marker = Actor.merge(marker_sets["right"], name="right_gripper_tip_contact")
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.left_tip_contact_marker)
            self.remove_object_from_greenscreen(self.right_tip_contact_marker)

    def _combined_prong_contact_force(self, links: list[Any]) -> torch.Tensor:
        force = torch.zeros_like(self.item.pose.p)
        seen = set()
        for link in links:
            if link is None or id(link) in seen:
                continue
            seen.add(id(link))
            force = force + self.scene.get_pairwise_contact_forces(link, self.item)
        return force

    def _gripper_tip_contact_debug_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_link = getattr(self.agent, "finger1_link", None)
        right_link = getattr(self.agent, "finger2_link", None)
        left_tip = getattr(self.agent, "finger1_tip", left_link)
        right_tip = getattr(self.agent, "finger2_tip", right_link)
        if left_link is None or right_link is None or left_tip is None or right_tip is None:
            false = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return false, false

        left_force = torch.linalg.norm(self._combined_prong_contact_force([left_tip, left_link]), axis=1)
        right_force = torch.linalg.norm(self._combined_prong_contact_force([right_tip, right_link]), axis=1)
        item_radius = torch.linalg.norm(self.item_dimensions, dim=-1) + GRIPPER_TIP_CONTACT_EXTRA_DISTANCE
        left_tip_near_item = torch.linalg.norm(left_tip.pose.p - self.item.pose.p, axis=1) <= item_radius
        right_tip_near_item = torch.linalg.norm(right_tip.pose.p - self.item.pose.p, axis=1) <= item_radius

        left_contact = (left_force >= GRIPPER_TIP_CONTACT_MIN_FORCE) & left_tip_near_item
        right_contact = (right_force >= GRIPPER_TIP_CONTACT_MIN_FORCE) & right_tip_near_item
        return left_contact, right_contact

    def _update_gripper_contact_markers(self) -> None:
        if not self.show_gripper_contact_markers:
            return
        if self.left_tip_contact_marker is None or self.right_tip_contact_marker is None:
            return

        left_tip = getattr(self.agent, "finger1_tip", None)
        right_tip = getattr(self.agent, "finger2_tip", None)
        if left_tip is None or right_tip is None:
            return

        left_contact, right_contact = self._gripper_tip_contact_debug_state()
        left_pos = left_tip.pose.p
        right_pos = right_tip.pose.p
        hidden_pos = torch.zeros_like(left_pos)
        hidden_pos[:, 2] = GRIPPER_TIP_CONTACT_HIDDEN_Z
        self.left_tip_contact_marker.set_pose(
            Pose.create_from_pq(torch.where(left_contact[:, None], left_pos, hidden_pos))
        )
        self.right_tip_contact_marker.set_pose(
            Pose.create_from_pq(torch.where(right_contact[:, None], right_pos, hidden_pos))
        )

    def _marker_two_prong_grasp_ready(
        self, min_force: float = 0.05, max_angle: float = 115.0
    ) -> torch.Tensor:
        left_link = getattr(self.agent, "finger1_link", None)
        right_link = getattr(self.agent, "finger2_link", None)
        left_tip = getattr(self.agent, "finger1_tip", left_link)
        right_tip = getattr(self.agent, "finger2_tip", right_link)
        if left_link is None or right_link is None:
            return self.agent.is_grasping(self.item, min_force=min_force, max_angle=max_angle)

        left_force_vec = self._combined_prong_contact_force([left_link, left_tip])
        right_force_vec = self._combined_prong_contact_force([right_link, right_tip])
        left_force = torch.linalg.norm(left_force_vec, axis=1)
        right_force = torch.linalg.norm(right_force_vec, axis=1)
        has_both_contacts = (left_force >= min_force) & (right_force >= min_force)
        force_cosine = torch.sum(left_force_vec * right_force_vec, dim=-1) / (
            left_force * right_force
        ).clamp_min(1e-6)
        opposing_prong_forces = force_cosine <= -0.5

        left_direction = left_link.pose.to_transformation_matrix()[..., :3, 1]
        right_direction = -right_link.pose.to_transformation_matrix()[..., :3, 1]
        left_angle = common.compute_angle_between(left_direction, left_force_vec)
        right_angle = common.compute_angle_between(right_direction, right_force_vec)
        inward_contact = (torch.rad2deg(left_angle) <= max_angle) & (torch.rad2deg(right_angle) <= max_angle)

        return has_both_contacts & inward_contact & opposing_prong_forces

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)
            if not hasattr(self, "item_lifted_once"):
                self.item_lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.item_lifted_once[env_idx] = False
            if self.target_type == "marker":
                if not hasattr(self, "marker_grasp_active"):
                    self.marker_grasp_active = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                    self.marker_grasp_offset = torch.zeros(
                        (self.num_envs, 3), dtype=torch.float32, device=self.device
                    )
                self.marker_grasp_active[env_idx] = False
                self.marker_grasp_offset[env_idx] = 0.0

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos
                + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)))

            # Sample positions for item and target
            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            # Use placement sampler for non-overlapping positions
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size],
            ]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)

            # Item/target radius (use max for conservative placement)
            if self.item_type == "can":
                item_radius = self.item_half_radii.max().item() + 0.01
            else:
                item_radius = self.item_half_sizes.max().item() + 0.01
            bin_radius = self.bin_radius.max().item() + 0.01

            if self.target_type == "marker":
                # Place the cube 220mm forward from the SO101 base front center.
                bin_xy_offset = torch.tensor(self.marker_xy_offset, device=self.device).repeat(b, 1)
                cube_start_x = (
                    SO101_BASE_FRONT_X
                    + MARKER_CUBE_START_DISTANCE_FROM_BASE_FRONT
                    - self.spawn_box_pos[0]
                )
                item_xy_offset = torch.tensor([cube_start_x, 0.0], device=self.device).repeat(b, 1)
            else:
                item_xy_offset = sampler.sample(item_radius, 100)
                bin_xy_offset = sampler.sample(bin_radius, 100, verbose=False)

            # Set item pose
            item_xyz = torch.zeros((b, 3))
            item_xyz[:, :2] = spawn_center[env_idx, :2] + item_xy_offset
            item_xyz[:, 2] = self.item_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            # Set target pose
            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = spawn_center[env_idx, :2] + bin_xy_offset
            bin_xyz[:, 2] = 0.0 if self.target_type == "marker" else self.bin_thickness / 2
            if self.target_type == "marker":
                marker_q = torch.tensor(
                    euler2quat(0, 0, self.marker_yaw),
                    device=self.device,
                    dtype=bin_xyz.dtype,
                ).repeat(b, 1)
                self.bin.set_pose(Pose.create_from_pq(bin_xyz, marker_q))
            else:
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.bin.set_pose(Pose.create_from_pq(bin_xyz, qs))

            # Goal is above target center
            goal_xyz = bin_xyz.clone()
            if self.target_type == "marker":
                goal_xyz[:, 2] = bin_xyz[:, 2] + self.bin_thickness + self.item_half_sizes[env_idx]
            else:
                goal_xyz[:, 2] = self.bin_thickness + self.item_half_sizes[env_idx]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _after_control_step(self):
        if hasattr(super(), "_after_control_step"):
            super()._after_control_step()
        self._update_gripper_contact_markers()
        if self.target_type != "marker" or not self.use_marker_grasp_assist:
            return
        if not hasattr(self, "marker_grasp_active"):
            self.marker_grasp_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.marker_grasp_offset = torch.zeros(
                (self.num_envs, 3), dtype=torch.float32, device=self.device
            )

        qpos = self.agent.robot.get_qpos()
        gripper_qpos = qpos[:, 5]
        gripper_pos = self.agent.robot.links_map["gripper_link"].pose.p
        item_pos = self.item.pose.p

        closed = gripper_qpos <= np.deg2rad(20)
        released = gripper_qpos >= np.deg2rad(28)
        newly_grasped = (~self.marker_grasp_active) & closed & self._marker_two_prong_grasp_ready()
        if newly_grasped.any():
            self.marker_grasp_offset[newly_grasped] = item_pos[newly_grasped] - gripper_pos[newly_grasped]
            # gripper_link is above the contact patch; this calibrated offset keeps the cube between the fingers.
            self.marker_grasp_offset[newly_grasped, 2] = -0.055
        self.marker_grasp_active |= newly_grasped
        self.marker_grasp_active &= ~released

        if self.marker_grasp_active.any():
            active = self.marker_grasp_active
            new_item_pos = item_pos.clone()
            new_item_pos[active] = gripper_pos[active] + self.marker_grasp_offset[active]
            min_z = self.item_half_sizes[active] + 0.001
            new_item_pos[active, 2] = torch.maximum(new_item_pos[active, 2], min_z)
            self.item.set_pose(Pose.create_from_pq(new_item_pos, self.item.pose.q))
            if hasattr(self.item, "set_linear_velocity"):
                self.item.set_linear_velocity(torch.zeros_like(new_item_pos))
            if hasattr(self.item, "set_angular_velocity"):
                self.item.set_angular_velocity(torch.zeros_like(new_item_pos))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        # Adding joint noise for better sim2real
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
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
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_grip_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_bin_pos=self.bin.pose.p - self.agent.tcp_pos,
                item_to_bin_pos=self.bin.pose.p - self.item.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bin_dimensions=self.bin_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        if self.target_type == "marker":
            bin_pos[:, 2] = bin_pos[:, 2] + self.bin_thickness + self.item_half_sizes
        else:
            bin_pos[:, 2] = self.bin_thickness + self.item_half_sizes

        offset = item_pos - bin_pos
        offset_for_bounds = offset
        if self.target_type == "marker":
            cos_yaw = torch.cos(torch.tensor(self.marker_yaw, device=self.device, dtype=offset.dtype))
            sin_yaw = torch.sin(torch.tensor(self.marker_yaw, device=self.device, dtype=offset.dtype))
            local_x = cos_yaw * offset[:, 0] + sin_yaw * offset[:, 1]
            local_y = -sin_yaw * offset[:, 0] + cos_yaw * offset[:, 1]
            offset_for_bounds = torch.stack([local_x, local_y, offset[:, 2]], dim=-1)
        inside_x = torch.abs(offset_for_bounds[:, 0]) < self.bin_half_sizes_x
        inside_y = torch.abs(offset_for_bounds[:, 1]) < self.bin_half_sizes_y
        is_item_above_bin = inside_x & inside_y
        height_tol = torch.maximum(self.item_half_sizes, torch.full_like(self.item_half_sizes, 0.015))
        is_item_at_target_height = torch.abs(offset[:, 2]) <= height_tol

        item_lifted = self.item.pose.p[..., -1] >= (self.item_half_sizes + 1e-3)
        item_lifted_now = self.item.pose.p[..., -1] >= (self.item_half_sizes + 0.015)
        if not hasattr(self, "item_lifted_once"):
            self.item_lifted_once = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.item_lifted_once |= item_lifted_now

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_item_grasped = self.agent.is_grasping(self.item)
        if self.target_type == "marker" and self.use_marker_grasp_assist and hasattr(self, "marker_grasp_active"):
            is_item_grasped = is_item_grasped | self.marker_grasp_active
        is_robot_static = self.agent.is_static()

        # Contact checks
        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_bin = self.agent.is_touching(self.bin)
        robot_touching_item = self.agent.is_touching(self.item)

        if self.target_type == "marker":
            is_item_on_target = is_item_above_bin & is_item_at_target_height
            success = is_item_on_target & self.item_lifted_once & (~robot_touching_item) & is_item_static
        else:
            is_item_on_target = is_item_above_bin
            success = is_item_above_bin & (~robot_touching_item) & is_robot_static & (~robot_touching_bin)

        return {
            "inside_x": inside_x,
            "inside_y": inside_y,
            "is_item_at_target_height": is_item_at_target_height,
            "item_vel": item_vel,
            "item_lifted": item_lifted,
            "item_lifted_once": self.item_lifted_once,
            "is_item_static": is_item_static,
            "success": success,
            "is_item_above_bin": is_item_above_bin,
            "is_item_on_target": is_item_on_target,
            "is_item_grasped": is_item_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_bin": robot_touching_bin,
            "robot_touching_item": robot_touching_item,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reaching reward
        tcp_to_item_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.item.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_item_dist))
        reward = reaching_reward

        # Complex place reward
        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        goal_xyz = bin_pos.clone()
        if self.target_type == "marker":
            goal_xyz[..., 2] = goal_xyz[..., 2] + self.bin_thickness + self.item_half_sizes
        else:
            goal_xyz[..., 2] = self.bin_thickness + self.item_half_sizes

        # Overall distance reward
        item_to_goal_dist = torch.linalg.norm(goal_xyz - item_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal_dist)

        # XY and Z distance with far/close logic
        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        # Far: target is above bin (encourages lifting before placing)
        item_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + (self.bin_dimensions[:, 2:] * 2) + 0.03) - item_pos[..., 2:], dim=1
        )
        # Close: target is final position
        item_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - item_pos[..., 2:], dim=1)
        item_close_to_goal = item_to_goal_dist_xy <= self.bin_radius
        item_to_goal_dist_z = torch.where(
            item_close_to_goal, item_to_goal_dist_z_close, item_to_goal_dist_z_far
        )
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        # Ungrasp reward (inverted from Reach's close gripper)
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)

        # Grasped: 3 + place_reward
        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        # Above target: 3 + place_reward + gripper_openness
        is_item_dropped = (~info["robot_touching_item"]).float()
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        place_mask = info.get("is_item_on_target", info["is_item_above_bin"])
        reward[place_mask] = (4 + place_reward + is_item_dropped + gripper_openness + static_robot_reward)[
            place_mask
        ]

        # Success
        reward[info["success"]] = 9

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        if self.target_type == "bin":
            reward -= 3 * info["robot_touching_bin"].float()
        reward -= 1 * (~info["item_lifted"]).float()  # Encourage picking item fast

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101PlaceCube-v1", max_episode_steps=50)
class PlaceCube(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", **kwargs)


@register_env("SO101PlaceCubeMarker-v1", max_episode_steps=100)
class PlaceCubeMarker(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="cube", target_type="marker", **kwargs)


@register_env("SO101PlaceCan-v1", max_episode_steps=50)
class PlaceCan(Place):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type="can", **kwargs)
