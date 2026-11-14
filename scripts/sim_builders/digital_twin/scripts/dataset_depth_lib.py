"""Shared pieces for fusing the recorded D405 wrist depth into the sim table frame.

Frame conventions: depth pixels unproject with the recorded D405 intrinsics in OpenCV axes (x right, y down, z forward);
MuJoCo cameras look along -z with y up, so X_mj = (x, -y, -z). World = sim world = table frame (table top at z = 0).
"""
import pathlib

import av
import cv2
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

ROOT_DIR = pathlib.Path(__file__).resolve().parents[5]
SCENE = ROOT_DIR / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/clippers_only/scene.xml"
FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966   # config_rl.yaml rebot_depth_intrinsics
UNITS_M = 1e-4
W, H = 640, 480
Z_MIN, Z_MAX = 0.07, 0.45
GRIPPER_DEG_OPEN, GRIPPER_SLIDE_OPEN = -238.8, 0.05       # action_trace_sim.to_qpos
ROBOT_BODIES = ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")


def episode_states(root, ep):
    epi = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
    row = epi[epi.episode_index == ep].iloc[0]
    df = pd.read_parquet(f"{root}/data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet")
    df = df[df.episode_index == ep].sort_values("frame_index")
    return np.stack(df["observation.state"].to_numpy()).astype(np.float64), row


def depth_frames(root, ep):
    d = pathlib.Path(root) / "depth/wrist.depth" / f"episode-{ep:06d}"
    return sorted(int(p.stem.split("-")[1]) for p in d.glob("frame-*.png"))


def load_depth(root, ep, f):
    return cv2.imread(f"{root}/depth/wrist.depth/episode-{ep:06d}/frame-{f:06d}.png", cv2.IMREAD_UNCHANGED)


def iter_rgb(root, ep, row, indices, key="observation.images.wrist"):
    """Yield (frame_index, bgr) for the requested indices in video order (one decode pass)."""
    path = f"{root}/videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
    start = int(round(row[f"videos/{key}/from_timestamp"] * 30))
    want = set(int(i) + start for i in indices)
    last = max(want)
    with av.open(path) as c:
        for n, fr in enumerate(c.decode(video=0)):
            if n in want:
                yield n - start, fr.to_ndarray(format="bgr24")
            if n >= last:
                break


def to_qpos(state_deg):
    slide = np.clip(float(state_deg[6]) / GRIPPER_DEG_OPEN * GRIPPER_SLIDE_OPEN, 0.0, GRIPPER_SLIDE_OPEN)
    return np.concatenate([np.deg2rad(state_deg[:6]), [slide, slide]])


class Sim:
    """Scene model with the calibrated wrist camera; optional camera pose perturbation (metres in gripper_end, rotvec in camera frame)."""

    def __init__(self, scene=SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.cid = self.model.camera("wrist").id
        self.base_pos = self.model.cam_pos[self.cid].copy()
        self.base_quat = self.model.cam_quat[self.cid].copy()
        self.qadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint_left", "joint_right")]
        body_ids = {self.model.body(n).id for n in ROBOT_BODIES}
        self.robot_geom = np.array([int(self.model.geom_bodyid[g]) in body_ids for g in range(self.model.ngeom)])
        self.renderer = None

    def set_camera_offset(self, dpos=(0, 0, 0), drot=(0, 0, 0)):
        self.model.cam_pos[self.cid] = self.base_pos + np.asarray(dpos, float)
        q0 = self.base_quat
        r = Rot.from_quat([q0[1], q0[2], q0[3], q0[0]]) * Rot.from_rotvec(np.asarray(drot, float))
        q = r.as_quat()
        self.model.cam_quat[self.cid] = [q[3], q[0], q[1], q[2]]

    def camera_pose(self, state_deg):
        """(t, R) of the wrist camera in the world frame; R columns = MuJoCo camera axes (x right, y up, -z forward)."""
        self.data.qpos[self.qadr] = to_qpos(state_deg)
        mujoco.mj_forward(self.model, self.data)
        return self.data.cam_xpos[self.cid].copy(), self.data.cam_xmat[self.cid].reshape(3, 3).copy()

    def robot_mask(self, state_deg, dilate_px=15):
        """Pixels of the robot (fingers) in the wrist view, posed from the state, dilated."""
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=H, width=W)
            self.renderer.enable_segmentation_rendering()
        self.data.qpos[self.qadr] = to_qpos(state_deg)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="wrist")
        seg = self.renderer.render()
        gid = seg[..., 0]
        mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
        mask[mask] = self.robot_geom[gid[mask]]
        if dilate_px:
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)).astype(bool)
        return mask


def unproject(depth_u16, stride=1, z_min=Z_MIN, z_max=Z_MAX, exclude=None):
    """Camera-frame points in MuJoCo axes (N, 3) + pixel coords (N, 2) for the in-range pixels on a stride grid."""
    z = depth_u16[::stride, ::stride].astype(np.float64) * UNITS_M
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    keep = (z > z_min) & (z < z_max)
    if exclude is not None:
        keep &= ~exclude[::stride, ::stride]
    z, u, v = z[keep], u[keep], v[keep]
    x = (u - CX) / FX * z
    y = (v - CY) / FY * z
    return np.stack([x, -y, -z], -1), np.stack([u, v], -1)


def ransac_plane(pts, n_iter=200, thresh=0.004, rng=None):
    """Unit normal n with n.z > 0 (in the given frame) and d such that n.p + d = 0; inlier mask. Refined by SVD on inliers."""
    rng = rng or np.random.default_rng(0)
    best_n, best_d, best_inl = None, None, np.zeros(len(pts), bool)
    for _ in range(n_iter):
        i = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = -n @ p0
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best_inl.sum():
            best_n, best_d, best_inl = n, d, inl
    if best_n is None:
        return None, None, best_inl
    for _ in range(2):
        q = pts[best_inl]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        n = vt[2]
        d = -n @ c
        best_inl = np.abs(pts @ n + d) < thresh
        best_n, best_d = n, d
    if best_n[2] < 0:
        best_n, best_d = -best_n, -best_d
    return best_n, best_d, best_inl
