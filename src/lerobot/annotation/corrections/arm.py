"""The twin's arm for the correction pass: tool-frame FK / IK, soft limits, the top camera, the chunk check.

Tool frame: tip = the point between the finger tips, approach = unit vector along the fingers (base to tips),
jaw = the line from one finger to the other. Joints are degrees in the raw arm frame, world = the arm base frame
(metres, table at z = 0). The top camera is the twin's ``ext`` camera, good to 1-3 cm on the real images.
"""

import mujoco
import numpy as np

from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import RebotB601SimRobotConfig
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import RebotB601Sim

FPS = 30
H, W = 480, 640
TWIN = "twin_l3_close_lid_p2"  # any twin scene: only its arm and top camera are used
MAX_SPEED_DEG_S = 45.0


class Gripper:
    """Tool frame from the model: approach axis (gripper_end toward the finger tips), jaw axis (left finger to right
    finger), tip = the point between the finger tips."""

    def __init__(self, model: mujoco.MjModel, limits_deg: list[tuple[float, float]]):
        self.m = model
        self.soft_lo, self.soft_hi = np.deg2rad(np.asarray(limits_deg[:6])).T  # the follower's soft limits
        self.d = mujoco.MjData(model)  # scratch data: IK never touches the running sim
        self.ge = model.body("gripper_end").id
        self.left, self.right = model.body("gripper_left").id, model.body("gripper_right").id
        self.q0 = int(model.jnt_qposadr[model.joint("joint1").id])
        self.dofs = [int(model.jnt_dofadr[model.joint(f"joint{i}").id]) for i in range(1, 7)]
        d = self.d
        mujoco.mj_resetData(model, d)
        d.qpos[self.q0 + 6 : self.q0 + 8] = 0.05
        mujoco.mj_forward(model, d)
        rg, og = d.xmat[self.ge].reshape(3, 3), d.xpos[self.ge]
        mid = (d.xpos[self.left] + d.xpos[self.right]) / 2  # the finger bodies sit behind the tips; gripper_end is at the tip end
        self.approach_local = rg.T @ (og - mid) / np.linalg.norm(og - mid)
        jaw = d.xpos[self.right] - d.xpos[self.left]
        self.jaw_local = rg.T @ jaw / np.linalg.norm(jaw)
        reach = 0.0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in (self.left, self.right) and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[k] : model.mesh_vertadr[k] + model.mesh_vertnum[k]]
                world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                reach = max(reach, float(((world - og) @ (rg @ self.approach_local)).max()))
        self.tip_len = reach

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tip, approach, jaw) in the world frame."""
        rg = data.xmat[self.ge].reshape(3, 3)
        approach = rg @ self.approach_local
        return data.xpos[self.ge] + self.tip_len * approach, approach, rg @ self.jaw_local

    def fk(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.d.qpos[self.q0 : self.q0 + 6] = np.deg2rad(q_deg[:6])
        mujoco.mj_forward(self.m, self.d)
        tip, approach, jaw = self.pose(self.d)
        return tip.copy(), approach.copy(), jaw.copy()

    def ik(self, q_deg: np.ndarray, tip: np.ndarray, approach: np.ndarray, jaw: np.ndarray, iters: int = 100) -> np.ndarray:
        """Arm joints (deg) putting the tool frame at (tip, approach, jaw), seeded from q_deg: damped least squares,
        at most 0.05 rad per joint per iteration. The jaw is symmetric, so the sign nearer the seed is taken."""
        m, d = self.m, self.d
        q = np.deg2rad(np.asarray(q_deg[:6], dtype=float))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        approach = approach / np.linalg.norm(approach)
        jaw = jaw - (jaw @ approach) * approach
        jaw = jaw / np.linalg.norm(jaw)
        d.qpos[self.q0 : self.q0 + 6] = q
        mujoco.mj_forward(m, d)
        if self.pose(d)[2] @ jaw < 0:
            jaw = -jaw
        for _ in range(iters):
            d.qpos[self.q0 : self.q0 + 6] = q
            mujoco.mj_forward(m, d)
            p, a, j = self.pose(d)
            mujoco.mj_jac(m, d, jp, jr, p, self.ge)
            e_rot = 0.5 * (np.cross(a, approach) + np.cross(j, jaw))
            e = np.concatenate([tip - p, 0.2 * e_rot])
            jac = np.vstack([jp[:, self.dofs], 0.2 * jr[:, self.dofs]])
            dq = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(6), jac.T @ e)
            dq *= min(1.0, 0.05 / max(float(np.abs(dq).max()), 1e-9))
            q = np.clip(q + dq, self.soft_lo, self.soft_hi)
        return np.rad2deg(q)


class Arm:
    """The twin's arm: ``gripper`` (FK / IK), ``limits`` (7 x 2 soft limits, deg) and ``top``, the pinhole
    calibration of the fixed top camera in the world frame (MuJoCo camera: x right, y up, looking along -z)."""

    def __init__(self):
        robot = RebotB601Sim(RebotB601SimRobotConfig(scene=TWIN, cameras={}, fps=FPS, substeps=17, gravcomp=True, pos_vel_velocity=None))
        robot.connect()
        self.limits = np.asarray(list(robot.config.joint_limits.values()), dtype=float)
        self.gripper = Gripper(robot.model, list(robot.config.joint_limits.values()))
        cid = robot.model.camera("ext").id
        f = 0.5 * H / np.tan(np.deg2rad(robot.model.cam_fovy[cid]) / 2)
        self.top = dict(f=float(f), cx=W / 2, cy=H / 2, pos=robot.data.cam_xpos[cid].copy(), mat=robot.data.cam_xmat[cid].reshape(3, 3).copy())


def project(calib: dict, point: np.ndarray) -> tuple[float, float]:
    """World point -> pixel (u, v) of the camera."""
    local = calib["mat"].T @ (np.asarray(point) - calib["pos"])
    return calib["cx"] + calib["f"] * local[0] / -local[2], calib["cy"] - calib["f"] * local[1] / -local[2]


def frame(approach: np.ndarray, jaw: np.ndarray) -> np.ndarray:
    """Rotation matrix with columns (jaw, approach x jaw, approach)."""
    approach = approach / np.linalg.norm(approach)
    jaw = jaw - (jaw @ approach) * approach
    jaw = jaw / np.linalg.norm(jaw)
    return np.stack([jaw, np.cross(approach, jaw), approach], axis=1)


def check(arm: Arm, state: np.ndarray, actions: np.ndarray) -> dict:
    """What a chunk must pass to be commanded from ``state``: inside the soft limits, no arm joint faster than
    MAX_SPEED_DEG_S (the first step counts from the state), tool tip never more than 2 cm under the table plane
    (finger tips resting on the table read about -0.01)."""
    steps = np.abs(np.diff(np.vstack([state[None], actions]), axis=0)) * FPS
    inside = bool((actions >= arm.limits[:, 0] - 1e-6).all() and (actions <= arm.limits[:, 1] + 1e-6).all())
    speed = float(steps[:, :6].max())
    lowest = float(min(arm.gripper.fk(a)[0][2] for a in actions))
    return dict(
        ok=inside and speed <= MAX_SPEED_DEG_S + 1e-6 and lowest > -0.02, inside_limits=inside,
        max_arm_speed_deg_s=speed, max_gripper_speed_deg_s=float(steps[:, 6].max()), lowest_tip_z_m=lowest,
    )
