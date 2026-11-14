"""Sim side of the scene recreation: ReBot DM scene + external camera + optional object XML.
Camera convention (MuJoCo): camera frame x right, y up, looks along -z; square pixels, principal point at the image centre.
    u = W/2 + f * x / (-z),  v = H/2 - f * y / (-z),  f = (H/2) / tan(fovy/2)."""
import os, pathlib, re, numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[5]
MJCF = ROOT / "lerobot/src/lerobot/robots/rebot_b601_follower/mjcf"
SCENE = MJCF / "scene_dm.xml"; MODEL = MJCF / "rebot_b601_dm.xml"
# same constants as lerobot/src/lerobot/probes/action_trace_sim.py (to_qpos + SOFT_LIMITS_DEG), copied, not edited
ARM_DOF = 6; GRIPPER_IDX = 6; GRIPPER_DEG_OPEN = -238.8; GRIPPER_SLIDE_OPEN = 0.05
SOFT_LIMITS_DEG = {"joint1": (-145.0, 145.0), "joint2": (-170.0, 1.0), "joint3": (-200.0, 1.0), "joint4": (-80.0, 90.0), "joint5": (-90.0, 90.0), "joint6": (-90.0, 90.0)}
ROBOT_BODIES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right"]
W, H = 640, 480


def to_qpos(action_deg):
    slide = float(action_deg[GRIPPER_IDX]) / GRIPPER_DEG_OPEN * GRIPPER_SLIDE_OPEN
    slide = float(np.clip(slide, 0.0, GRIPPER_SLIDE_OPEN))
    return np.concatenate([np.deg2rad(action_deg[:ARM_DOF]), [slide, slide]])


def quat_wxyz(rotvec):
    q = Rot.from_rotvec(rotvec).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


WRIST_FOVY = 2 * np.degrees(np.arctan(240 / 394.9832))  # D405 recorded fy, 480 rows


def wrist_camera_xml(pos=(0, 0, 0), quat=(1, 0, 0, 0), fovy=None):
    fovy = WRIST_FOVY if fovy is None else fovy
    return f'<camera name="wrist" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" fovy="{fovy:.3f}"/>'


def add_wrist_camera(robot_src, pos=(0, 0, 0), quat=(1, 0, 0, 0), fovy=None):
    """Insert the wrist camera as a child of gripper_end (the D405 bracket is on the gripper body)."""
    return re.sub(r'(<body name="gripper_end"[^>]*>)', lambda m: m.group(1) + "\n                    " + wrist_camera_xml(pos, quat, fovy), robot_src, count=1)


def build_model(extra_xml="", floor_material=None, floor_size=None, floor_pos=(0, 0, 0), assets_xml="", wrist_cam=False):
    """scene_dm.xml with widened joint ranges, a camera named 'ext' (pose set later), and extra worldbody XML.
    extra_xml goes inside <worldbody>; assets_xml inside <asset>. Temp files are written beside the originals
    because the robot model references its meshes by relative path."""
    model_src = MODEL.read_text()
    for joint, (lo, hi) in SOFT_LIMITS_DEG.items():
        m = re.search(rf'<joint name="{joint}"[^>]*range="([-0-9.e]+) ([-0-9.e]+)"', model_src)
        lo = min(np.deg2rad(lo), float(m.group(1))); hi = max(np.deg2rad(hi), float(m.group(2))); widened = f'{lo:.4f} {hi:.4f}"'
        model_src = re.sub(rf'(<joint name="{joint}"[^>]*range=")[^"]*"', lambda m: m.group(1) + widened, model_src)
        model_src = re.sub(rf'(<position [^>]*name="{joint}"[^>]*ctrlrange=")[^"]*"', lambda m: m.group(1) + widened, model_src)
    if wrist_cam: model_src = add_wrist_camera(model_src)
    scene_src = SCENE.read_text()
    floor_attrs = f'name="floor" pos="{floor_pos[0]} {floor_pos[1]} {floor_pos[2]}"'
    if floor_size is not None: floor_attrs += f' size="{floor_size[0]} {floor_size[1]} 0.05"'; scene_src = scene_src.replace('size="0 0 0.05" ', "")
    if floor_material is not None: scene_src = scene_src.replace('material="groundplane"', f'material="{floor_material}"')
    scene_src = scene_src.replace('name="floor"', floor_attrs)
    scene_src = scene_src.replace("</asset>", assets_xml + "\n  </asset>")
    scene_src = scene_src.replace("</worldbody>", '    <camera name="ext" pos="0 0 1" quat="1 0 0 0" fovy="50"/>\n' + extra_xml + "\n  </worldbody>")
    tmp_model = MODEL.with_name(f"_scene_nail_model_{os.getpid()}.xml"); tmp_scene = SCENE.with_name(f"_scene_nail_scene_{os.getpid()}.xml")
    tmp_model.write_text(model_src); tmp_scene.write_text(scene_src.replace(MODEL.name, tmp_model.name))
    try:
        return mujoco.MjModel.from_xml_path(str(tmp_scene)), scene_src.replace(MODEL.name, tmp_model.name)
    finally:
        tmp_model.unlink(); tmp_scene.unlink()


def set_camera(model, pos, rotvec, fovy, name="ext"):
    cid = model.camera(name).id; model.cam_pos[cid] = pos; model.cam_quat[cid] = quat_wxyz(rotvec); model.cam_fovy[cid] = fovy


CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera (x right, y down, z forward) -> MuJoCo camera (x right, y up, looks along -z)


def cv_rot_to_mj_quat(R_cv):
    """R_cv: columns = OpenCV camera axes in the parent frame. Returns the MuJoCo camera quaternion (w, x, y, z)."""
    q = Rot.from_matrix(R_cv @ CV_TO_MJ).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


def set_body_camera(model, name, pos, R_cv, fovy=None):
    cid = model.camera(name).id; model.cam_pos[cid] = pos; model.cam_quat[cid] = cv_rot_to_mj_quat(R_cv)
    if fovy is not None: model.cam_fovy[cid] = fovy


def set_state(model, data, state_deg):
    data.qpos[:8] = to_qpos(np.asarray(state_deg, dtype=float)); mujoco.mj_forward(model, data)


def project(points_w, pos, rotvec, fovy, w=W, h=H):
    R = Rot.from_rotvec(rotvec).as_matrix(); p = (np.asarray(points_w) - pos) @ R; f = (h / 2) / np.tan(np.deg2rad(fovy) / 2)
    z = -p[..., 2]; return np.stack([w / 2 + f * p[..., 0] / z, h / 2 - f * p[..., 1] / z], -1)


def backproject_z0(uv, pos, rotvec, fovy, z0=0.0, w=W, h=H):
    """Pixel (u, v) -> world point on the plane z = z0."""
    R = Rot.from_rotvec(rotvec).as_matrix(); f = (h / 2) / np.tan(np.deg2rad(fovy) / 2); uv = np.asarray(uv, dtype=float)
    d_cam = np.stack([(uv[..., 0] - w / 2) / f, -(uv[..., 1] - h / 2) / f, -np.ones(uv.shape[:-1])], -1)
    d_w = d_cam @ R.T; t = (z0 - pos[2]) / d_w[..., 2]; return pos + t[..., None] * d_w


class Views:
    def __init__(self, model):
        self.model = model; self.r = mujoco.Renderer(model, H, W); self.opt = mujoco.MjvOption()
        robot_ids = {model.body(b).id for b in ROBOT_BODIES}
        self.robot_geoms = np.array([model.geom_bodyid[g] in robot_ids for g in range(model.ngeom)])

    def rgb(self, data, cam="ext"):
        self.r.disable_segmentation_rendering(); self.r.disable_depth_rendering(); self.r.update_scene(data, cam, self.opt); return self.r.render().copy()

    def depth(self, data, cam="ext"):
        """Planar depth along the optical axis in metres (MuJoCo depth render), same quantity as RealSense z16."""
        self.r.disable_segmentation_rendering(); self.r.enable_depth_rendering(); self.r.update_scene(data, cam, self.opt); d = self.r.render().copy(); self.r.disable_depth_rendering(); return d

    def seg(self, data, cam="ext"):
        self.r.disable_depth_rendering(); self.r.enable_segmentation_rendering(); self.r.update_scene(data, cam, self.opt); s = self.r.render().copy(); self.r.disable_segmentation_rendering(); return s

    def robot_mask(self, data, cam="ext"):
        s = self.seg(data, cam); gid = s[:, :, 0]; return (gid >= 0) & self.robot_geoms[np.clip(gid, 0, None)]

    def body_mask(self, data, body_names, cam="ext"):
        ids = {self.model.body(b).id for b in body_names}; sel = np.array([self.model.geom_bodyid[g] in ids for g in range(self.model.ngeom)])
        s = self.seg(data, cam); gid = s[:, :, 0]; return (gid >= 0) & sel[np.clip(gid, 0, None)]
