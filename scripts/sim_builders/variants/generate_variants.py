"""Placement variants of the sim eval layouts: scenes/<layout>_v<k>/ with the task objects re-placed (xy + yaw), everything
else identical to the base scene.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/variants/generate_variants.py [--layouts ep0 ep3 ...]

Deterministic: SEED fixes every sample (per-layout rng seeded with [SEED, layout index], so one layout can be regenerated
alone without changing the others). Re-running writes byte-identical files.

Sampling region (per object centre): the coverage grid cells rows 2-3 x columns B-E of lerobot/adhocs/object_coverage
(the 6 x 4 table grid in the top image, unprojected through the scene's ext camera onto the table plane; that is where
the real internal data put the clippers, the case and the kit) intersected with the 0.50 m reach disc around the base.
Accepted when: the body's AABB stays 2 cm inside the table box, its convex footprint (hull of the geoms' oriented boxes)
is 2 cm clear of every other scene body's footprint (clutter, basket, case, other task objects), the bottle keeps <= 0.45 m from the basket centre, and a top-pinch IK pose at the object
exists inside the follower's soft joint limits (config_rebot_b601_follower.joint_limits).

close_lid: the case is world-fixed (pos + quat, the lid follows as a child, the strap-loop capsules sit on the base body). The
six variants are centred on the three T071 demo poses measured by the lid agent (lerobot/scripts/sim_builders/lid_solvability/
demo_pose_summary.json: ep0, ep2, ep3, two variants each, in that order), xy jitter +-CLOSE_LID_JITTER_M and +-20 deg yaw around
the demo body yaw (lid toward the robot side as in the demos); the open lid's footprint (= the swept arc's) must stay inside
the table and clear of the robot base, and both lid side walls must pass the reach test (a pinch across the side wall is the
grasp that closes the lid, see the lid agent's README). The reach disc / coverage region do not apply to close_lid.

Outputs: scenes/<layout>_v<k>/{scene.xml,objects.json,background_top.png} (materials stay referenced through the base
folder, `<base>/materials/...` relative to assetdir=".." as the base XMLs already do), and here: checks.json,
contact_<layout>.png (deployed top-camera composite of the base + the six variants, region outline on the base tile).
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SCENES = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes"
DEMO_POSES = HERE.parent / "lid_solvability/demo_pose_summary.json"
DEMO_KEYS = ("t071_ep0", "t071_ep2", "t071_ep3")

SEED = 20261028
N_VARIANTS = 6
# the order fixes each layout's sub-seed ([SEED, index]); never reorder, append only
LAYOUT_ORDER = ("bottle_basket", "clippers_only", "ep0", "ep3", "close_lid")
# task objects: name -> yaw sampling ("full" = uniform [0, 360), ("jitter", deg) = base yaw +- deg; close_lid aims the lid at the base)
TASK_OBJECTS = {
    "bottle_basket": {"bottle": "full", "basket": ("jitter", 25.0)},
    "clippers_only": {"clipper_case": "full", "clipper_large": "full", "clipper_small": "full"},
    "ep0": {"clipper_case": "full", "clipper_large": "full", "clipper_small": "full"},
    "ep3": {"clipper_case": "full", "clipper_large": "full", "clipper_small": "full"},
    "close_lid": {"headphone_case": ("jitter", 20.0)},
}
MARGIN_M = 0.02          # AABB clearance between bodies and to the table edge
REACH_M = 0.50           # object centre to base
BASKET_REACH_M = 0.48
BOTTLE_BASKET_MAX_M = 0.45
CLOSE_LID_JITTER_M = 0.04        # +-xy around each demo pose
SETTLE_S, SETTLE_TOL_M, PEN_TOL_M = 1.0, 0.005, 0.002
FPS, SUBSTEPS = 30, 17
# the coverage grid (lerobot/adhocs/object_coverage/extract_frames.py): table corners in the 640x480 top image, 6 x 4 cells
GRID_CORNERS_PX = np.array([[128, 42], [632, 38], [635, 388], [95, 385]], float)  # TL (robot side, left), TR, BR, BL
COLS, ROWS = 6, 4
REGION_UV = ((1.0, 5.0), (1.0, 3.0))  # columns B-E, rows 2-3 in grid units
TABLE_C, TABLE_H = np.array([0.3467, 0.0244]), np.array([0.3708, 0.5053])
# follower soft limits (config_rebot_b601_follower.joint_limits), joint1..joint6 order
SOFT_LIMITS_DEG = [(-145, 145), (-170, 1), (-200, 1), (-80, 90), (-90, 90), (-90, 90)]
# deployed top-camera composite (config_rl_sim.yaml env.robot.cameras.top)
GAIN, OFF = np.array([1.222, 1.278, 1.165]), np.array([8.0, 8.5, 28.1])
STATIC, STATIC_GAIN, FEATHER = ["table", "table_top"], np.array([1.054, 1.231, 1.403]), 2


def quat_z(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def yaw_of(q):
    assert abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6, f"not a z-only quaternion: {q}"
    return 2 * np.arctan2(q[3], q[0])


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A))
    return vt[-1].reshape(3, 3)


def apply_h(H, p):
    q = H @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


class Camera:
    """The scene's ext camera: pixel <-> table plane."""

    def __init__(self, m):
        cam = m.camera("ext")
        self.pos = np.array(m.cam_pos[cam.id])
        R = np.zeros(9); mujoco.mju_quat2Mat(R, m.cam_quat[cam.id]); self.R = R.reshape(3, 3)
        self.f = 240 / np.tan(np.deg2rad(m.cam_fovy[cam.id]) / 2)

    def unproject(self, px):
        d = self.R @ np.array([(px[0] - 320) / self.f, (240 - px[1]) / self.f, -1.0])
        t = -self.pos[2] / d[2]
        return (self.pos + t * d)[:2]

    def project(self, xyz):
        p = self.R.T @ (np.asarray(xyz, float) - self.pos)
        return np.array([320 + self.f * p[0] / -p[2], 240 - self.f * p[1] / -p[2]])


class Region:
    """Grid units (u, v) <-> table xy through the top image; membership = the B-E x 2-3 block."""

    def __init__(self, cam):
        uv = np.array([[0, 0], [COLS, 0], [COLS, ROWS], [0, ROWS]], float)
        self.corners_xy = np.array([cam.unproject(p) for p in GRID_CORNERS_PX])
        self.uv_to_xy = homography(uv, self.corners_xy)
        self.xy_to_uv = np.linalg.inv(self.uv_to_xy)
        (u0, u1), (v0, v1) = REGION_UV
        self.polygon_xy = np.array([apply_h(self.uv_to_xy, p) for p in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]])
        self.box = (self.polygon_xy.min(0), self.polygon_xy.max(0))

    def contains(self, xy):
        u, v = apply_h(self.xy_to_uv, xy)
        (u0, u1), (v0, v1) = REGION_UV
        return u0 <= u <= u1 and v0 <= v <= v1 and np.linalg.norm(xy) <= REACH_M


class Scene:
    def __init__(self, layout):
        self.layout = layout
        self.dir = SCENES / layout
        self.xml = (self.dir / "scene.xml").read_text()
        self.m = mujoco.MjModel.from_xml_path(str(self.dir / "scene.xml"))
        self.m.opt.timestep = 1.0 / (FPS * SUBSTEPS)
        self.d = mujoco.MjData(self.m)
        self.q0 = int(self.m.jnt_qposadr[self.m.joint("joint1").id])
        self.bodies = re.findall(r'^    <body name="(\w+)"', self.xml, re.M)  # top-level scene bodies (the include adds the robot)
        self.tracked = [self.m.body(b).name for n in self.bodies for b in sorted(self._subtree(self.m.body(n).id))]  # + children (the lid)
        self.base_pose = {}
        for name in self.bodies:
            mt = re.search(rf'<body name="{name}" pos="([^"]*)" quat="([^"]*)"', self.xml)
            if mt is not None:
                self.base_pose[name] = (np.array(mt[1].split(), float), np.array(mt[2].split(), float))
        self.jids = [self.m.joint(f"joint{i}").id for i in range(1, 7)]
        self.dofs = [int(self.m.jnt_dofadr[j]) for j in self.jids]
        lo = np.maximum(self.m.jnt_range[self.jids, 0], np.deg2rad([a for a, _ in SOFT_LIMITS_DEG]))
        hi = np.minimum(self.m.jnt_range[self.jids, 1], np.deg2rad([b for _, b in SOFT_LIMITS_DEG]))
        self.limits = (lo, hi)
        self.robot_base = int(self.m.body_parentid[self.m.body("link1").id])  # the fixed base link (0 = none)
        self.robot_bodies = (self._subtree(self.m.body("link1").id) | {self.robot_base}) - {0}
        self.reset()
        self.approach, self.tip_len = self._finger_axis()

    def _subtree(self, root):
        return {b for b in range(self.m.nbody) if self._has_ancestor(b, root)}

    def _has_ancestor(self, b, root):
        while b != 0:
            if b == root:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.ctrl[:] = self.d.qpos[self.q0:self.q0 + 8]
        mujoco.mj_forward(self.m, self.d)

    def _finger_axis(self):
        """Finger approach axis in the gripper_end frame (unit, pointing from the wrist to the tips) and the tip distance."""
        m, d = self.m, self.d
        self.d.qpos[self.q0 + 6:self.q0 + 8] = 0.05; mujoco.mj_forward(m, d)
        ge = m.body("gripper_end").id; Rg = d.xmat[ge].reshape(3, 3); og = d.xpos[ge]
        pts, slide = [], None
        for body in ("gripper_left", "gripper_right"):
            bid = m.body(body).id
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] | m.geom_conaffinity[g]):
                    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                    pts.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - og) @ Rg)
        pts = np.vstack(pts)
        slide = (d.xpos[m.body("gripper_right").id] - d.xpos[m.body("gripper_left").id]) @ Rg
        ax = int(np.argmax(np.abs(slide)))
        ext = np.ptp(pts, axis=0); approach_ax = int(np.argmax(np.where(np.arange(3) == ax, -1, ext)))
        # the fingers extend from the wrist side to the tips at the gripper_end origin (the TCP): approach = away from their bulk
        sign = -np.sign(pts[:, approach_ax].mean())
        a = np.zeros(3); a[approach_ax] = sign
        tip_len = max(0.0, float(np.percentile(pts[:, approach_ax] * sign, 98)))
        self.reset()
        return a, tip_len

    # ---- poses ----
    def set_pose(self, name, xy, yaw):
        m, d = self.m, self.d
        bid = m.body(name).id
        z = self.base_pose[name][0][2]
        if m.body_jntnum[bid] > 0 and m.jnt_type[m.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE:
            adr = m.jnt_qposadr[m.body_jntadr[bid]]
            d.qpos[adr:adr + 3] = [xy[0], xy[1], z]; d.qpos[adr + 3:adr + 7] = quat_z(yaw)
        else:
            m.body_pos[bid] = [xy[0], xy[1], z]; m.body_quat[bid] = quat_z(yaw)
        mujoco.mj_forward(m, d)

    def aabb(self, name):
        """World AABB (lo, hi) over the geoms of the body and its children, group 5 (hidden legacy visuals) excluded."""
        m, d = self.m, self.d
        sub = self._subtree(m.body(name).id)
        los, his = [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in sub and m.geom_group[g] != 5:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                R = d.geom_xmat[g].reshape(3, 3)
                cw = d.geom_xpos[g] + R @ c; hw = np.abs(R) @ h
                los.append(cw - hw); his.append(cw + hw)
        return np.min(los, 0), np.max(his, 0)

    def top_z(self, name):
        return float(self.aabb(name)[1][2])

    def footprint(self, name, subtree=True):
        """Convex xy hull of the world corners of the geoms' oriented boxes (body + children, group 5 excluded)."""
        m, d = self.m, self.d
        sub = self._subtree(m.body(name).id) if subtree else {m.body(name).id}
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in sub and m.geom_group[g] != 5:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                R = d.geom_xmat[g].reshape(3, 3)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        pts.append((d.geom_xpos[g] + R @ (c + h * [sx, sy, 0]))[:2])
        return convex_hull(np.array(pts))

    def penetration(self, poses):
        """Worst contact distance of the moved bodies (and their children) at the current state, the arm at park."""
        m, d = self.m, self.d
        moved = set().union(*(self._subtree(m.body(n).id) for n in poses))
        mujoco.mj_forward(m, d)
        return min([float(c.dist) for c in d.contact[:d.ncon] if m.geom_bodyid[c.geom1] in moved or m.geom_bodyid[c.geom2] in moved], default=0.0)

    # ---- reach ----
    def ik_top_pinch(self, target, max_it=200, orient=True):
        """Finger tips at `target`, approach axis straight down (`orient`; else position only); ok when a pose inside the limits reaches it."""
        m, d = self.m, self.d
        ge = m.body("gripper_end").id; a_des = np.array([0.0, 0.0, -1.0])
        lo, hi = self.limits; q = slice(self.q0, self.q0 + 6)
        yaw = np.arctan2(target[1], target[0])
        starts = [np.array([0, -1.0, -1.0, 0.5, 0, 0]), np.array([yaw, -0.8, -1.5, 0.8, 0, 0]), np.array([yaw, -1.5, -0.5, 0.0, 0, 0]),
                  np.array([-yaw, -0.8, -1.5, 0.8, 0, 0]), np.array([yaw, -0.3, -2.0, 0.5, 0, 0])]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        best = None
        for s in starts:
            d.qpos[q] = np.clip(s, lo, hi); d.qpos[self.q0 + 6:self.q0 + 8] = 0.05
            for _ in range(max_it):
                mujoco.mj_forward(m, d)
                R = d.xmat[ge].reshape(3, 3); a = R @ self.approach; tip = d.xpos[ge] + self.tip_len * a
                mujoco.mj_jac(m, d, jp, jr, tip, ge)
                w = 0.05 if orient else 0.0
                e = np.concatenate([target - tip, w * np.cross(a, a_des)])
                J = np.vstack([jp[:, self.dofs], w * jr[:, self.dofs]])
                dq = np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ e)
                d.qpos[q] = np.clip(d.qpos[q] + 0.5 * dq, lo, hi)
            mujoco.mj_forward(m, d)
            R = d.xmat[ge].reshape(3, 3); a = R @ self.approach; tip = d.xpos[ge] + self.tip_len * a
            pos_err = float(np.linalg.norm(target - tip)); ang_err = float(np.degrees(np.arccos(np.clip(a @ a_des, -1, 1))))
            table_hit = any(m.geom_bodyid[c.geom1] in self.robot_bodies or m.geom_bodyid[c.geom2] in self.robot_bodies
                            for c in d.contact[:d.ncon] if c.dist < -1e-3 and "table" in (m.geom(c.geom1).name + m.geom(c.geom2).name))
            ok = pos_err < 0.005 and (ang_err < 5.0 or not orient) and not table_hit
            res = dict(ok=bool(ok), pos_err_mm=round(pos_err * 1e3, 2), angle_err_deg=round(ang_err, 2), q_deg=np.degrees(d.qpos[q]).round(1).tolist())
            if best is None or (ok and not best["ok"]) or (ok == best["ok"] and pos_err < best["pos_err_mm"] / 1e3):
                best = res
            if ok:
                break
        d.qpos[q] = self.m.qpos0[q]; d.qpos[self.q0 + 6:self.q0 + 8] = self.m.qpos0[self.q0 + 6:self.q0 + 8]; mujoco.mj_forward(m, d)
        return best


def convex_hull(pts):
    pts = np.unique(np.round(pts, 6), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) < 3:
        return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def gap(P, Q):
    """Separation of two convex xy polygons along their edge normals (the exact distance for vertex-edge closest pairs,
    a lower bound for vertex-vertex ones; negative = overlap)."""
    best = -np.inf
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n) + 1e-12
            p, q = P @ n, Q @ n
            best = max(best, q.min() - p.max(), p.min() - q.max())
    return float(best)


def separated(a, b, margin):
    """Convex footprints at least `margin` apart."""
    return gap(a, b) >= margin


def inside_table(box, margin):
    lo, hi = TABLE_C - TABLE_H + margin, TABLE_C + TABLE_H - margin
    return bool(np.all(box[0][:2] >= lo) and np.all(box[1][:2] <= hi))


def sample_layout(layout, region, scene, rng):
    """Six accepted pose sets {name: (xy, yaw)}; each variant restarts from the base poses."""
    objects = TASK_OBJECTS[layout]
    others = [b for b in scene.bodies if b not in objects]
    if layout == "close_lid":
        demos = json.loads(DEMO_POSES.read_text())
    variants, log = [], []
    for k in range(N_VARIANTS):
        tries = 0
        while True:
            tries += 1
            scene.reset()
            for name in objects:  # back to the base pose (fixed bodies keep the model's edited pos otherwise)
                p, q = scene.base_pose[name]; scene.set_pose(name, p[:2], yaw_of(q))
            poses, boxes, ok = {}, {n: scene.footprint(n) for n in others}, True
            if scene.robot_base:
                boxes["robot_base"] = scene.footprint(scene.m.body(scene.robot_base).name, subtree=False)
            for name, mode in objects.items():
                if layout == "close_lid":
                    demo = demos[DEMO_KEYS[k % len(DEMO_KEYS)]]
                    xy = np.array(demo["base_xy"], float) + rng.uniform(-CLOSE_LID_JITTER_M, CLOSE_LID_JITTER_M, 2)
                    yaw = np.deg2rad(demo["body_yaw_deg_sim_convention"] + rng.uniform(-mode[1], mode[1]))
                else:
                    lo, hi = region.box
                    xy = rng.uniform(lo, hi)
                    yaw = rng.uniform(0, 2 * np.pi) if mode == "full" else yaw_of(scene.base_pose[name][1]) + np.deg2rad(rng.uniform(-mode[1], mode[1]))
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                reach = BASKET_REACH_M if name == "basket" else REACH_M
                if layout != "close_lid" and (not region.contains(xy) or np.linalg.norm(xy) > reach):
                    ok = False; break
                scene.set_pose(name, xy, yaw)
                foot = scene.footprint(name)
                if not inside_table(scene.aabb(name), MARGIN_M) or not all(separated(foot, b, MARGIN_M) for b in boxes.values()):
                    ok = False; break
                poses[name] = (xy, yaw); boxes[name] = foot
            if not ok:
                continue
            if layout == "bottle_basket" and np.linalg.norm(poses["bottle"][0] - poses["basket"][0]) > BOTTLE_BASKET_MAX_M:
                continue
            if scene.penetration(poses) < -1e-3:  # the parked arm, the table or a neighbour runs into a moved body
                continue
            ik = {}
            for name in objects:
                xy = poses[name][0]
                if name == "basket":  # the bottle is dropped in from a side pinch: position reach over the centre, 5 cm above the rim
                    ik[name] = scene.ik_top_pinch(np.array([xy[0], xy[1], scene.top_z("basket") + 0.05]), orient=False)
                elif name == "headphone_case":  # the grasp that closes the lid: a pinch across either side wall of the open lid
                    for wall in ("headphone_lid_wall_xp", "headphone_lid_wall_xn"):
                        g = scene.m.geom(wall).id
                        top = scene.d.geom_xpos[g] + np.abs(scene.d.geom_xmat[g].reshape(3, 3)) @ scene.m.geom_aabb[g, 3:]
                        ik[wall] = scene.ik_top_pinch(np.array([scene.d.geom_xpos[g][0], scene.d.geom_xpos[g][1], top[2]]))
                else:
                    ik[name] = scene.ik_top_pinch(np.array([xy[0], xy[1], scene.top_z(name)]))
            if not all(r["ok"] for r in ik.values()):
                continue
            variants.append(poses); log.append(dict(tries=tries, ik=ik))
            break
    return variants, log


def write_variant(layout, k, poses, scene, seed, region):
    """k is 1-based (close_lid variant k follows demo DEMO_KEYS[(k - 1) % 3])."""
    out = SCENES / f"{layout}_v{k}"
    out.mkdir(exist_ok=True)
    xml = scene.xml.replace(f'<mujoco model="scene_{layout}">', f'<mujoco model="scene_{layout}_v{k}">', 1)
    objects = {}
    for name, (xy, yaw) in poses.items():
        z = scene.base_pose[name][0][2]; q = quat_z(yaw)
        pat = rf'(<body name="{name}" pos=")[^"]*(" quat=")[^"]*(")'
        assert len(re.findall(pat, xml)) == 1, name
        xml = re.sub(pat, lambda mt: f"{mt[1]}{xy[0]:.4f} {xy[1]:.4f} {z:.4f}{mt[2]}{q[0]:.5f} 0 0 {q[3]:.5f}{mt[3]}", xml)
        free = scene.m.body_jntnum[scene.m.body(name).id] > 0
        objects[name] = dict(xy=[round(float(xy[0]), 4), round(float(xy[1]), 4)], z=z, yaw_deg=round(float(np.degrees(yaw)), 2),
                             quat_wxyz=[round(float(q[0]), 5), 0.0, 0.0, round(float(q[3]), 5)], free=bool(free),
                             radius_from_base_m=round(float(np.linalg.norm(xy)), 4))
    if layout == "close_lid":
        m = scene.m; scene.reset()
        for name, (xy, yaw) in poses.items():
            scene.set_pose(name, xy, yaw)
        lid = m.body("headphone_lid").id
        hinge = m.joint("lid_hinge").id
        hinge_world = scene.d.xanchor[hinge]; axis_world = scene.d.xaxis[hinge]
        objects["headphone_lid"] = dict(xy=[round(float(v), 4) for v in scene.d.xpos[lid][:2]], joint="lid_hinge",
                                        hinge_point_world=[round(float(v), 4) for v in hinge_world],
                                        hinge_axis_world=[round(float(v), 4) for v in axis_world], open_rad=0.0, closed_rad=3.14159)
    (out / "scene.xml").write_text(xml)
    shutil.copyfile(scene.dir / "background_top.png", out / "background_top.png")
    meta = dict(variant_of=layout, variant=k, seed=seed, generator="lerobot/scripts/sim_builders/variants/generate_variants.py",
                region=dict(grid_cells="rows 2-3 x columns B-E of lerobot/adhocs/object_coverage", polygon_xy=region.polygon_xy.round(4).tolist(),
                            reach_m=REACH_M) if layout != "close_lid" else dict(demo=DEMO_KEYS[(k - 1) % len(DEMO_KEYS)], source=str(DEMO_POSES.relative_to(WS)),
                                                                                 xy_jitter_m=CLOSE_LID_JITTER_M, yaw_jitter_deg=20.0),
                objects=objects, unchanged=[b for b in scene.bodies if b not in poses],
                assets="background_top.png copied from the base; materials and meshes referenced through the base folder (assetdir ..)")
    (out / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out


def settle_and_render(folder, scene_bodies):
    """Load a scene folder, 1 s of physics with the arm held: displacement per body, worst penetration, composite top view."""
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml")); m.opt.timestep = 1.0 / (FPS * SUBSTEPS); d = mujoco.MjData(m)
    q0 = int(m.jnt_qposadr[m.joint("joint1").id])
    mujoco.mj_resetData(m, d); d.ctrl[:] = d.qpos[q0:q0 + 8]; mujoco.mj_forward(m, d)
    ids = {n: m.body(n).id for n in scene_bodies}
    x0 = d.xpos.copy(); worst = {n: 0.0 for n in ids}
    for _ in range(int(SETTLE_S * FPS * SUBSTEPS)):
        mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            for n, i in ids.items():
                if i in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                    worst[n] = min(worst[n], float(c.dist))
    moved = {n: round(float(np.linalg.norm(d.xpos[i] - x0[i])) * 1e3, 2) for n, i in ids.items()}
    pen = {n: round(v * 1e3, 2) for n, v in worst.items()}
    # the composite at the start pose (what the policy sees at t = 0)
    mujoco.mj_resetData(m, d); d.ctrl[:] = d.qpos[q0:q0 + 8]; mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, "ext"); rgb = r.render()
    r.enable_segmentation_rendering(); r.update_scene(d, "ext"); geom = r.render()[:, :, 0]; r.disable_segmentation_rendering()
    r.close()
    rgb = np.clip(rgb * GAIN + OFF, 0, 255).astype(np.uint8)
    static_ids = [m.geom(n).id for n in STATIC]
    static = (geom >= 0) & np.isin(np.clip(geom, 0, None), static_ids)
    rgb[static] = np.clip(rgb[static] * STATIC_GAIN, 0, 255).astype(np.uint8)
    mask = ((geom >= 0) & ((m.geom_bodyid[np.clip(geom, 0, None)] != 0) | static)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32); mask = cv2.GaussianBlur(mask, (0, 0), FEATHER)
    bg = cv2.cvtColor(cv2.imread(str(folder / "background_top.png")), cv2.COLOR_BGR2RGB)
    comp = (mask[:, :, None] * rgb + (1 - mask[:, :, None]) * bg).astype(np.uint8)
    return dict(moved_mm=moved, max_moved_mm=max(moved.values()), penetration_mm=pen, worst_penetration_mm=min(pen.values())), comp


def contact_sheet(layout, tiles, region, cam, path, markers=()):
    W, H = 480, 360
    cols = 4; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * (H + 22)), (20, 20, 20))
    font = ImageFont.load_default(size=15)
    for i, (label, img, outline) in enumerate(tiles):
        im = Image.fromarray(img)
        if outline:
            dr = ImageDraw.Draw(im)
            pts = [tuple(cam.project([x, y, 0.0])) for x, y in region.polygon_xy]
            dr.polygon(pts, outline=(255, 255, 0), width=2)
            dr.ellipse([tuple(cam.project([0, 0, 0]) - 4), tuple(cam.project([0, 0, 0]) + 4)], outline=(0, 255, 255), width=2)
            for mx, my in markers:
                dr.ellipse([tuple(cam.project([mx, my, 0]) - 5), tuple(cam.project([mx, my, 0]) + 5)], outline=(255, 0, 255), width=2)
        im = im.resize((W, H))
        x, y = (i % cols) * W, (i // cols) * (H + 22)
        sheet.paste(im, (x, y + 22))
        ImageDraw.Draw(sheet).text((x + 6, y + 3), label, fill=(255, 255, 255), font=font)
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUT_ORDER))
    args = ap.parse_args()
    checks = json.loads((HERE / "checks.json").read_text()) if (HERE / "checks.json").exists() else {}
    for layout in args.layouts:
        seed = [SEED, LAYOUT_ORDER.index(layout)]
        rng = np.random.default_rng(seed)
        scene = Scene(layout)
        cam = Camera(scene.m); region = Region(cam)
        print(f"[{layout}] approach axis {scene.approach.tolist()} tip {scene.tip_len:.4f} m; region corners xy {region.polygon_xy.round(3).tolist()}")
        variants, log = sample_layout(layout, region, scene, rng)
        entries = {}
        tiles = []
        base_check, base_img = settle_and_render(scene.dir, scene.tracked)
        tiles.append((f"{layout} (base)  moved {base_check['max_moved_mm']:.1f} mm", base_img, True))
        for k, (poses, lg) in enumerate(zip(variants, log), start=1):
            out = write_variant(layout, k, poses, scene, seed, region)
            chk, img = settle_and_render(out, scene.tracked)
            task_moved = {n: chk["moved_mm"][n] for n in poses}
            if layout == "close_lid":
                task_moved["headphone_lid"] = chk["moved_mm"]["headphone_lid"]
            task_pen = {n: chk["penetration_mm"][n] for n in task_moved}
            assert max(task_moved.values()) <= SETTLE_TOL_M * 1e3, (layout, k, task_moved)
            assert min(task_pen.values()) >= -PEN_TOL_M * 1e3, (layout, k, task_pen)
            entries[f"v{k}"] = dict(poses={n: dict(xy=np.round(xy, 4).tolist(), yaw_deg=round(float(np.degrees(yaw)), 1)) for n, (xy, yaw) in poses.items()},
                                   tries=lg["tries"], ik=lg["ik"], settle=chk)
            tiles.append((f"{layout}_v{k}  moved {chk['max_moved_mm']:.1f} mm", img, False))
            print(f"  v{k}: tries {lg['tries']}, settle max {chk['max_moved_mm']:.1f} mm (task {max(task_moved.values()):.1f}), pen {chk['worst_penetration_mm']:.1f} mm (task {min(task_pen.values()):.1f}), "
                  + ", ".join(f"{n} ({xy[0]:.2f}, {xy[1]:.2f}) {np.degrees(yaw):.0f} deg" for n, (xy, yaw) in poses.items()))
        markers = [json.loads(DEMO_POSES.read_text())[key]["base_xy"] for key in DEMO_KEYS] if layout == "close_lid" else []
        contact_sheet(layout, tiles, region, cam, HERE / f"contact_{layout}.png", markers)
        checks[layout] = dict(seed=seed, region_polygon_xy=region.polygon_xy.round(4).tolist(), base_settle=base_check, variants=entries,
                              lid_demo_poses=str(DEMO_POSES.relative_to(WS)) if layout == "close_lid" else None)
    (HERE / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")


if __name__ == "__main__":
    main()
