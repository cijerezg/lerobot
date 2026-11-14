"""Twin eval suite: scenes/twin_l<L>_<config>_p<1-4>/ built from the digital twin scene (lerobot/scripts/sim_builders/digital_twin/twin/
scene_twin.xml, snapshot at build time) with the package asset copies in scenes/assets/twin/ (ENVS_REPORT.md).

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/digital_twin/envs/build_twin_envs.py [--configs ...] [--no-render] [--check-base]

Design (2026-10-30): 4 levels x 4 hand-designed base configurations (CONFIGS, the `base` poses) x 4 placements. p1 is the base
configuration itself, p2-p4 jitter every movable body of the configuration (xy +-JITTER_XY_M, yaw +-JITTER_YAW_DEG, resampled
until the checks pass). Checks = scenes/README.md placement rules: body AABB MARGIN_M inside the desk, convex footprint MARGIN_M
clear of every other body and of the robot base, no start contact < 1 mm with the parked arm, IK reach for the targets (straight-down
pinch at the object top; position-only side pinch for the tall ROAR bottle; position-only 5 cm over the basket rim; a straight-down
pinch at both kit flap edges; a pinch across both lid side walls), 1 s settle with the configuration's bodies moving < SETTLE_TOL_M.
Hand-placed poses must lie in the README region and the reach disc (basket <= BASKET_REACH_M); the full-desk levels keep the twin's
basket (0.69 m from the base, `anchor_basket`) and its contents ride along. Deterministic: SEED, per-configuration sub-seed = index
in CONFIGS; byte-identical on re-run. Outputs here: checks.json, renders/ (ext + wrist of every p1, per-config sheets, contact sheet).
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WS = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SCENES = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes"
TWIN = HERE.parent / "twin"
ASSETS = SCENES / "assets/twin"
RENDERS = HERE / "renders"
TWIN_REL = "../../../../../scripts/sim_builders/digital_twin/twin/"  # how scene_twin.xml reaches its own files from assetdir

SEED = 20261030
N_PLACEMENTS = 4
JITTER_XY_M = 0.06
JITTER_YAW_DEG = 20.0
CLUTTER_JITTER_TRIES = 20  # full desk: a clutter body is jittered only if one of these samples keeps the clearance, else it stays put
MARGIN_M = 0.02          # footprint clearance between bodies and AABB margin to the desk edge
REACH_M = 0.50           # hand-placed target centre to the base
BASKET_REACH_M = 0.48
SETTLE_S, SETTLE_TOL_M, PEN_TOL_M = 1.0, 0.005, 0.001
FPS, SUBSTEPS = 30, 17
REGION_XY = np.array([(0.151, -0.283), (0.183, 0.480), (0.565, 0.422), (0.538, -0.302)])  # scenes/README.md region (table xy)
TABLE_C, TABLE_H = np.array([0.3365, 0.0]), np.array([0.3810, 0.6096])  # the twin desk slab (room.xml room_desk_top)
SOFT_LIMITS_DEG = [(-145, 145), (-170, 1), (-200, 1), (-80, 90), (-90, 90), (-90, 90)]  # follower soft limits, joint1..6
SIDE_PINCH_Z = {"roar_bottle": 0.12}  # tall bottles are side-pinched (bottle_basket script): position-only reach at this height
BASKET_CONTENTS = ("basket_bottle", "basket_can", "basket_card", "basket_card_white")  # ride along in the basket frame
# clipper seats in the case frame: nail_task.SLOTS centres, the clippers along the slot (case y), at the height where they rest on the
# slot's convex hulls in a 1 s settle (0.023-0.024; the nominal slot floor 0.010 + half height puts the box 5-7 mm inside the hulls)
CLIPPER_SEATS = {"clipper_large": ((-0.0156, -0.0039, 0.0250), 90.0), "clipper_small": ((0.0160, -0.0049, 0.0250), 90.0)}
FLAT_Z = {"clipper_small": 0.0065}  # the twin's small clipper lies tilted in the case (z 0.0106); loose on the desk it rests at half height + 2 mm
TAPE_ROLL_2_TWIN = (0.07, -0.30, 0.0)  # the duplicate tape roll's clutter pose on the full desk (left of the robot, next to the first one)

# ---- the 16 base configurations ---------------------------------------------------------------------------------------------------
# base: body -> (x, y, yaw_deg) hand-placed; targets: reach-tested bodies in task order; bodies: the scene's top-level bodies ("all" =
# the full twin desk + tape_roll_2); groups: body -> children riding in its frame (contents, seated clippers); anchor_basket: the
# basket stays at its twin pose (outside the reach rule) and is jittered there. Every body in `base` is jittered (p2-p4).
L1 = ["basket"]
CONFIGS = {
    "twin_l1_tape_basket_right": dict(level=1, letter="a", task="tape_roll", bodies=L1 + ["tape_roll"], targets=["tape_roll"],
                                      base={"basket": (0.36, 0.22, 0.0), "tape_roll": (0.30, -0.06, 0.0)}),
    "twin_l1_tape_basket_left": dict(level=1, letter="b", task="tape_roll", bodies=L1 + ["tape_roll"], targets=["tape_roll"],
                                     base={"basket": (0.42, -0.16, 0.0), "tape_roll": (0.26, 0.14, 20.0)}),
    "twin_l1_pill_basket_far": dict(level=1, letter="c", task="pill_bottle", bodies=L1 + ["pill_bottle"], targets=["pill_bottle"],
                                    base={"basket": (0.46, 0.03, 0.0), "pill_bottle": (0.20, 0.00, 0.0)}),
    "twin_l1_roar_basket_near": dict(level=1, letter="d", task="roar_bottle", bodies=L1 + ["roar_bottle"], targets=["roar_bottle"],
                                     base={"basket": (0.30, -0.26, 0.0), "roar_bottle": (0.42, 0.16, 0.0)}),
    "twin_l2_tape_bits": dict(level=2, letter="a", task="tape_roll", bodies=L1 + ["tape_roll", "pill_bottle", "bit_1", "bit_2", "bit_3", "bit_4"], targets=["tape_roll"],
                              base={"basket": (0.42, -0.12, 0.0), "tape_roll": (0.24, 0.16, 0.0), "pill_bottle": (0.36, 0.22, 0.0),
                                    "bit_1": (0.20, 0.30, 30.0), "bit_2": (0.28, 0.33, -45.0), "bit_3": (0.34, 0.38, 80.0), "bit_4": (0.45, 0.24, 10.0)}),
    "twin_l2_pill_case": dict(level=2, letter="b", task="pill_bottle", bodies=L1 + ["pill_bottle", "clipper_case", "clipper_large", "clipper_small"], targets=["pill_bottle"],
                              groups={"clipper_case": ["clipper_large", "clipper_small"]},
                              base={"basket": (0.34, 0.26, 0.0), "pill_bottle": (0.30, -0.10, 0.0), "clipper_case": (0.42, -0.10, 180.0)}),
    "twin_l2_roar_tapes": dict(level=2, letter="c", task="roar_bottle", bodies=L1 + ["roar_bottle", "tape_roll", "tape_roll_2", "clipper_large", "clipper_small"], targets=["roar_bottle"],
                               base={"basket": (0.46, 0.08, 0.0), "roar_bottle": (0.22, -0.16, 0.0), "tape_roll": (0.22, 0.14, 0.0), "tape_roll_2": (0.36, -0.24, 0.0),
                                     "clipper_large": (0.30, 0.30, 45.0), "clipper_small": (0.22, 0.32, -30.0)}),
    "twin_l2_tape_bottles": dict(level=2, letter="d", task="tape_roll", bodies=L1 + ["tape_roll", "pill_bottle", "roar_bottle"], targets=["tape_roll"],
                                 base={"basket": (0.28, -0.26, 0.0), "tape_roll": (0.44, 0.16, 0.0), "pill_bottle": (0.22, 0.14, 0.0), "roar_bottle": (0.40, 0.34, 0.0)}),
    "twin_l3_tape": dict(level=3, letter="a", task="tape_roll", bodies="all", targets=["tape_roll"], anchor_basket=True,
                         base={"tape_roll": (0.19, 0.12, 0.0)}),
    "twin_l3_pill": dict(level=3, letter="b", task="pill_bottle", bodies="all", targets=["pill_bottle"], anchor_basket=True,
                         base={"pill_bottle": (0.21, 0.15, 0.0)}),
    "twin_l3_fold_kit": dict(level=3, letter="c", task="fold_kit", bodies="all", targets=["bits_kit"], anchor_basket=True, base={}, jitter=["bits_kit"]),
    "twin_l4_tape_cup": dict(level=4, letter="a", task=["tape_roll", "paper_cup"], bodies="all", targets=["tape_roll", "paper_cup"], anchor_basket=True,
                             base={"tape_roll": (0.19, 0.12, 0.0), "paper_cup": (0.20, 0.25, 0.0)}),
    "twin_l4_two_tapes": dict(level=4, letter="b", task=["tape_roll", "tape_roll_2"], bodies="all", targets=["tape_roll", "tape_roll_2"], anchor_basket=True,
                              base={"tape_roll": (0.19, 0.13, 0.0), "tape_roll_2": (0.37, 0.28, 0.0)}),
    "twin_l4_pill_roar": dict(level=4, letter="c", task=["pill_bottle", "roar_bottle"], bodies="all", targets=["pill_bottle", "roar_bottle"], anchor_basket=True,
                              base={"pill_bottle": (0.20, 0.12, 0.0), "roar_bottle": (0.37, 0.28, 0.0)}),
    "twin_l4_three": dict(level=4, letter="d", task=["tape_roll", "pill_bottle", "paper_cup"], bodies="all", targets=["tape_roll", "pill_bottle", "paper_cup"], anchor_basket=True,
                          base={"tape_roll": (0.19, 0.13, 0.0), "pill_bottle": (0.33, 0.28, 0.0), "paper_cup": (0.21, 0.26, 0.0)}),
    "twin_l3_close_lid": dict(level=3, letter="d", task="close_lid", bodies="all", targets=["headphone_case"], anchor_basket=True, closed_kit=True,
                              base={"headphone_case": (0.46, -0.40, 210.0)}),  # base half far left, lid toward the robot (lid walls at r ~0.45); region / reach exempt, IK only
}
# L3 (d): the kit is CLOSED (assets/bits_kit_fold/test_results.json "closed_after_10s": strip + flap angles of the settled closed stack);
# the angles are baked into the strip / flap body quaternions so qpos0 (mj_resetData) is the closed kit, with the joint ranges and the
# dead-band tendon springs shifted by the same angles (identical physics, re-parametrised joints)
CLOSED_KIT_DEG = {"bits_kit_hinge_left": 121.6, "bits_kit_hinge_left_flap": 56.2, "bits_kit_hinge_right": 147.8, "bits_kit_hinge_right_flap": 34.3}
FULL_DESK_GROUPS = {"basket": list(BASKET_CONTENTS), "clipper_case": ["clipper_small"]}  # the twin has the small clipper in the case


# ---- geometry helpers (generate_variants.py 2026-10-28) ----------------------------------------------------------------------
def quat_z(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def yaw_of(q):
    assert abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6, f"not a z-only quaternion: {q}"
    return 2 * np.arctan2(q[3], q[0])


def yaw_deg(q):
    """Yaw in degrees for a z-only quaternion, None otherwise (the twin's small clipper lies tilted)."""
    return round(float(np.degrees(yaw_of(q))), 1) if abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6 else None


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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
    best = -np.inf
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n) + 1e-12
            p, q = P @ n, Q @ n
            best = max(best, q.min() - p.max(), p.min() - q.max())
    return float(best)


def point_in_polygon(p, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def point_polygon_distance(p, poly):
    """Distance from a point to a convex polygon's boundary, 0 inside."""
    if point_in_polygon(p, poly):
        return 0.0
    best = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a; t = np.clip((p - a) @ ab / (ab @ ab + 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def inside_table(box, margin):
    lo, hi = TABLE_C - TABLE_H + margin, TABLE_C + TABLE_H - margin
    return bool(np.all(box[0][:2] >= lo) and np.all(box[1][:2] <= hi))


def region_contains(xy, reach):
    return point_in_polygon(xy, REGION_XY) and np.linalg.norm(xy) <= reach


# ---- the twin scene as text -------------------------------------------------------------------------------------------------
class TwinText:
    """scene_twin.xml split into reusable pieces: asset lines, default block, top-level bodies (+ tape_roll_2), tendons, lights."""

    def __init__(self):
        self.src = TWIN / "scene_twin.xml"
        xml = self.src.read_text()
        self.snapshot = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.src.stat().st_mtime))
        asset = re.search(r"^  <asset>\n(.*?)^  </asset>\n", xml, re.S | re.M)[1]
        asset = asset.replace(TWIN_REL + "assets/", "assets/twin/").replace(TWIN_REL + "appearance/table_top_fit.png", "assets/twin/room/table_top_fit.png")
        assert TWIN_REL not in asset, "unexpected twin-relative path"
        self.asset_lines = [ln for ln in asset.splitlines() if ln.strip()]
        self.default = re.search(r"^  <default>\n.*?^  </default>\n", xml, re.S | re.M)[0]
        self.tendon = re.search(r"^  <tendon>\n.*?^  </tendon>\n", xml, re.S | re.M)
        self.tendon = self.tendon[0] if self.tendon else ""
        self.bodies = {}
        for mt in re.finditer(r'^    <body name="(\w+)"[^\n]*\n.*?^    </body>\n', xml, re.S | re.M):
            self.bodies[mt[1]] = mt[0]
        # tape_roll_2: the tape roll body again (same meshes), body and geom names suffixed
        self.bodies["tape_roll_2"] = re.sub(r'name="([\w]+)"', r'name="\1_2"', self.bodies["tape_roll"])
        self.pose = {}
        for name, block in self.bodies.items():
            mt = re.match(r'    <body name="\w+" pos="([^"]*)" quat="([^"]*)"', block)
            self.pose[name] = (np.array(mt[1].split(), float), np.array(mt[2].split(), float))
        x, y, yaw = TAPE_ROLL_2_TWIN
        self.pose["tape_roll_2"] = (np.array([x, y, self.pose["tape_roll"][0][2]]), quat_z(np.deg2rad(yaw)))
        cam = re.search(r'    <camera name="ext"[^\n]*\n', xml)[0]
        self.camera = cam
        self.table_contact = re.search(r'    <geom name="table_contact"[^\n]*\n', xml)[0]
        app = TWIN / "appearance"
        self.visual = re.search(r"  <visual>\n.*?  </visual>\n", (app / "lights_visual.xml").read_text(), re.S)[0]
        self.statistic = re.search(r"  <statistic[^\n]*\n", (app / "lights_visual.xml").read_text())[0]
        self.lights = "".join(ln + "\n" for ln in (app / "lights.xml").read_text().splitlines() if ln.strip().startswith("<light"))
        mats = (app / "materials.xml").read_text().replace(TWIN_REL + "appearance/floor_planks.png", "assets/twin/room/floor_planks.png")
        self.floor_assets = [ln for ln in mats.splitlines() if ln.strip().startswith(("<texture", "<material"))]
        self.all_bodies = list(self.bodies)  # the full desk = every twin body + tape_roll_2

    def closed_kit(self):
        """The bits_kit body block with the closed angles baked into the strip / flap body frames, and its tendon block shifted."""
        block = self.bodies["bits_kit"]
        axis = {"bits_kit_hinge_left": 1.0, "bits_kit_hinge_left_flap": 1.0, "bits_kit_hinge_right": -1.0, "bits_kit_hinge_right_flap": -1.0}
        body_of = {"bits_kit_hinge_left": "bits_kit_left_strip", "bits_kit_hinge_left_flap": "bits_kit_left",
                   "bits_kit_hinge_right": "bits_kit_right_strip", "bits_kit_hinge_right_flap": "bits_kit_right"}
        for joint, deg in CLOSED_KIT_DEG.items():
            a = np.deg2rad(deg); q = (np.cos(a / 2), np.sin(a / 2) * axis[joint])  # rotation about the body's y axis (the hinge axis)
            block = re.sub(rf'(<body name="{body_of[joint]}" pos="[^"]*")', lambda mt: f'{mt[1]} quat="{q[0]:.6f} 0 {q[1]:.6f} 0"', block, count=1)
            def shift_range(mt, a=a):
                lo, hi = float(mt[2]), float(mt[3])
                return f'{mt[1]}range="{lo - a:.4f} {hi - a:.4f}"'
            block = re.sub(rf'(<joint name="{joint}" type="hinge" axis="[^"]*" )range="([-\d.]+) ([-\d.]+)"', shift_range, block, count=1)
            assert f'name="{joint}"' in block
        tendon = self.tendon
        for joint, deg in CLOSED_KIT_DEG.items():
            a = np.deg2rad(deg)
            tendon = re.sub(rf'springlength="([-\d.]+) ([-\d.]+)"( stiffness="[^"]*">\s*<joint joint="{joint}")',
                            lambda mt, a=a: f'springlength="{float(mt[1]) - a:.4f} {float(mt[2]) - a:.4f}"{mt[3]}', tendon, count=1)
        return block, tendon

    def body_with_pose(self, name, pos, quat):
        block = self.bodies[name]
        return re.sub(r'(^    <body name="\w+" pos=")[^"]*(" quat=")[^"]*(")',
                      lambda mt: f"{mt[1]}{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}{mt[2]}{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}{mt[3]}", block, count=1, flags=re.M)

    def scene_xml(self, model_name, bodies, poses, closed_kit=False):
        """A package scene: compiler assetdir="..", robot include, pruned assets, the room, the aligned camera, the listed bodies."""
        saved = (self.bodies["bits_kit"], self.tendon)
        if closed_kit:
            self.bodies["bits_kit"], self.tendon = self.closed_kit()
        body_text = "".join(self.body_with_pose(n, *poses.get(n, self.pose[n])) for n in bodies)
        tendon = self.tendon if "bits_kit" in bodies else ""
        self.bodies["bits_kit"], self.tendon = saved
        room = (ASSETS / "room/room.xml").read_text()
        used_mesh = set(re.findall(r'mesh="(\w+)"', body_text))
        used_mat = set(re.findall(r'material="(\w+)"', body_text + room))
        kept = []
        for ln in self.asset_lines:
            kind = re.match(r'\s*<(\w+)', ln)[1]
            name = re.search(r'name="(\w+)"', ln)
            if kind == "mesh" and name[1] not in used_mesh:
                continue
            if kind == "material" and name[1] not in used_mat:
                continue
            kept.append(ln)
        used_tex = {re.search(r'texture="(\w+)"', ln)[1] for ln in kept if re.match(r'\s*<material', ln) and 'texture="' in ln}
        kept = [ln for ln in kept if not re.match(r'\s*<texture', ln) or re.search(r'name="(\w+)"', ln) is None or re.search(r'name="(\w+)"', ln)[1] in used_tex]
        return (f'<?xml version="1.0" encoding="utf-8"?>\n<mujoco model="{model_name}">\n'
                f'  <compiler angle="radian" assetdir=".." autolimits="true" />\n'
                f'  <include file="../rebot_b601_dm_photo.xml" />\n'
                f"{self.visual}{self.statistic}"
                f"  <asset>\n" + "\n".join(kept) + "\n" + "\n".join(self.floor_assets) + "\n  </asset>\n"
                f"{self.default}"
                f"  <worldbody>\n    <include file=\"../assets/twin/room/room.xml\" />\n{self.table_contact}{self.camera}"
                f"{body_text}{self.lights}  </worldbody>\n{tendon}</mujoco>\n")


def copy_assets():
    """Package copies of the twin's own asset files (the ones scene_twin.xml references) and the room files."""
    xml = (TWIN / "scene_twin.xml").read_text()
    files = sorted(set(re.findall(re.escape(TWIN_REL) + r"assets/([\w/.\-]+)", xml)))
    copied = []
    for rel in files:
        dst = ASSETS / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = TWIN / "assets" / rel
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime or src.stat().st_size != dst.stat().st_size:
            shutil.copyfile(src, dst)
        copied.append(rel)
    room = ASSETS / "room"; room.mkdir(parents=True, exist_ok=True)
    for src, name in ((HERE.parent / "room/room.xml", "room.xml"), (TWIN / "appearance/floor_planks.png", "floor_planks.png"),
                      (TWIN / "appearance/table_top_fit.png", "table_top_fit.png"), (TWIN / "appearance/lights.xml", "lights.xml"),
                      (TWIN / "appearance/lights_visual.xml", "lights_visual.xml")):
        shutil.copyfile(src, room / name)
        copied.append("room/" + name)
    return copied


# ---- a loaded layout model: poses by qpos, footprints, IK -------------------------------------------------------------------------
class Scene:
    def __init__(self, xml_path, bodies):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.m.opt.timestep = 1.0 / (FPS * SUBSTEPS)
        self.d = mujoco.MjData(self.m)
        self.q0 = int(self.m.jnt_qposadr[self.m.joint("joint1").id])
        self.bodies = list(bodies)
        self.jids = [self.m.joint(f"joint{i}").id for i in range(1, 7)]
        self.dofs = [int(self.m.jnt_dofadr[j]) for j in self.jids]
        lo = np.maximum(self.m.jnt_range[self.jids, 0], np.deg2rad([a for a, _ in SOFT_LIMITS_DEG]))
        hi = np.minimum(self.m.jnt_range[self.jids, 1], np.deg2rad([b for _, b in SOFT_LIMITS_DEG]))
        self.limits = (lo, hi)
        self.robot_base = int(self.m.body_parentid[self.m.body("link1").id])
        self.robot_bodies = (self._subtree(self.m.body("link1").id) | {self.robot_base}) - {0}
        self.reset()
        self.approach, self.tip_len = self._finger_axis()

    def _subtree(self, root):
        out = set()
        for b in range(self.m.nbody):
            x = b
            while x != 0:
                if x == root:
                    out.add(b); break
                x = int(self.m.body_parentid[x])
        return out

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.ctrl[:] = self.d.qpos[self.q0:self.q0 + 8]
        mujoco.mj_forward(self.m, self.d)

    def _finger_axis(self):
        m, d = self.m, self.d
        d.qpos[self.q0 + 6:self.q0 + 8] = 0.05; mujoco.mj_forward(m, d)
        ge = m.body("gripper_end").id; Rg = d.xmat[ge].reshape(3, 3); og = d.xpos[ge]
        pts = []
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
        sign = -np.sign(pts[:, approach_ax].mean())
        a = np.zeros(3); a[approach_ax] = sign
        tip_len = max(0.0, float(np.percentile(pts[:, approach_ax] * sign, 98)))
        self.reset()
        return a, tip_len

    def is_free(self, name):
        bid = self.m.body(name).id
        return self.m.body_jntnum[bid] > 0 and self.m.jnt_type[self.m.body_jntadr[bid]] == mujoco.mjtJoint.mjJNT_FREE

    def set_pose(self, name, pos, quat):
        m, d = self.m, self.d
        bid = m.body(name).id
        if self.is_free(name):
            adr = m.jnt_qposadr[m.body_jntadr[bid]]
            d.qpos[adr:adr + 3] = pos; d.qpos[adr + 3:adr + 7] = quat
        else:
            m.body_pos[bid] = pos; m.body_quat[bid] = quat
        mujoco.mj_forward(m, d)

    def aabb(self, name):
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

    def penetration(self, names, groups=()):
        """Worst contact distance of the moved bodies; contacts inside one group (a parent and the bodies riding in it, e.g. the
        twin's small clipper in the case at -1.7 mm, the basket contents) are the twin's own and not counted."""
        m, d = self.m, self.d
        moved = set().union(*(self._subtree(m.body(n).id) for n in names))
        own = [set().union(*(self._subtree(m.body(n).id) for n in grp)) for grp in groups]
        mujoco.mj_forward(m, d)
        worst = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in moved or b2 in moved) and not any(b1 in g and b2 in g for g in own):
                worst = min(worst, float(c.dist))
        return worst

    def ik_top_pinch(self, target, max_it=200, orient=True):
        """Finger tips at `target`, approach axis straight down (`orient`; else position only), inside the soft limits, no table hit."""
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
            res = dict(ok=bool(ok), pos_err_mm=round(pos_err * 1e3, 2), angle_err_deg=round(ang_err, 2), q_deg=np.degrees(d.qpos[q]).round(1).tolist(),
                       target=[round(float(v), 4) for v in target])
            if best is None or (ok and not best["ok"]) or (ok == best["ok"] and pos_err < best["pos_err_mm"] / 1e3):
                best = res
            if ok:
                break
        d.qpos[q] = self.m.qpos0[q]; d.qpos[self.q0 + 6:self.q0 + 8] = self.m.qpos0[self.q0 + 6:self.q0 + 8]; mujoco.mj_forward(m, d)
        return best

    def flap_edge_targets(self):
        """Outer-edge midpoints (top) of the kit's two flap slabs, 1 cm in from the edge: the pinch that folds a flap."""
        out = {}
        for side, sign in (("left", -1.0), ("right", 1.0)):
            g = self.m.geom(f"bits_kit_{side}_slab").id
            R = self.d.geom_xmat[g].reshape(3, 3); half = self.m.geom_size[g]
            edge = self.d.geom_xpos[g] + R[:, 0] * sign * (half[0] - 0.01)
            top = self.d.geom_xpos[g][2] + (np.abs(R) @ half)[2]
            out[f"bits_kit_{side}"] = np.array([edge[0], edge[1], top])
        return out

    def lid_wall_targets(self):
        """Top midpoints of the open lid's two side walls: a pinch across either wall is the grasp that closes the lid (generate_variants)."""
        out = {}
        for wall in ("headphone_lid_wall_xp", "headphone_lid_wall_xn"):
            g = self.m.geom(wall).id
            top = self.d.geom_xpos[g] + np.abs(self.d.geom_xmat[g].reshape(3, 3)) @ self.m.geom_aabb[g, 3:]
            out[wall] = np.array([self.d.geom_xpos[g][0], self.d.geom_xpos[g][1], top[2]])
        return out

    def ik_targets(self, body, pos):
        """IK reach results for one task body at `pos` (the scene already holds the pose)."""
        if body == "basket":
            return {body: self.ik_top_pinch(np.array([pos[0], pos[1], self.top_z("basket") + 0.05]), orient=False)}
        if body == "bits_kit":
            return {flap: self.ik_top_pinch(t) for flap, t in self.flap_edge_targets().items()}
        if body == "headphone_case":
            return {wall: self.ik_top_pinch(t) for wall, t in self.lid_wall_targets().items()}
        if body in SIDE_PINCH_Z:
            return {body: self.ik_top_pinch(np.array([pos[0], pos[1], pos[2] + SIDE_PINCH_Z[body]]), orient=False)}
        return {body: self.ik_top_pinch(np.array([pos[0], pos[1], self.top_z(body)]))}


# ---- placement -------------------------------------------------------------------------------------------------------------------
def config_groups(spec, bodies):
    groups = dict(spec.get("groups", {}))
    if spec["bodies"] == "all":
        for parent, kids in FULL_DESK_GROUPS.items():
            groups.setdefault(parent, kids)
    return {p: [k for k in kids if k in bodies] for p, kids in groups.items() if p in bodies}


def group_child_poses(twin, parent, pos, quat, kids, seats=None):
    """Children riding in the parent's frame: fixed seats (case frame) or the twin's relative pose."""
    p0, q0 = twin.pose[parent]
    out = {}
    for c in kids:
        if seats and c in seats:
            (rx, ry, rzz), ryaw = seats[c]
            rel, yaw_rel = np.array([rx, ry, rzz]), np.deg2rad(ryaw)
        else:  # the twin's relative pose (the small clipper lies tilted in the case: full quaternion algebra)
            pc, qc = twin.pose[c]
            rel = rz(yaw_of(q0)).T @ (pc - p0)
            inv, qrel, qnew = np.zeros(4), np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, q0); mujoco.mju_mulQuat(qrel, inv, qc); mujoco.mju_mulQuat(qnew, quat, qrel)
            out[c] = (pos + rz(yaw_of(quat)) @ rel, qnew)
            continue
        out[c] = (pos + rz(yaw_of(quat)) @ rel, quat_z(yaw_of(quat) + yaw_rel))
    return out


def place(twin, scene, spec, bodies, rng, jitter):
    """One placement attempt: {body: (pos, quat)} for every body that leaves its twin pose, plus diagnostics. None = a hard check failed."""
    groups = config_groups(spec, bodies)
    kids = {k for ks in groups.values() for k in ks}
    seats = CLIPPER_SEATS if "clipper_case" in spec.get("groups", {}) else None
    base = dict(spec["base"])
    if spec.get("anchor_basket") and "basket" in bodies:
        p0, q0 = twin.pose["basket"]
        base["basket"] = (float(p0[0]), float(p0[1]), float(np.degrees(yaw_of(q0))))
    for b in spec.get("jitter", []):
        p0, q0 = twin.pose[b]
        base[b] = (float(p0[0]), float(p0[1]), float(np.degrees(yaw_of(q0))))
    movable = list(base)
    clutter = [b for b in bodies if b not in movable and b not in kids]
    scene.reset()
    poses, foot, log = {}, {}, {}
    # hand-placed (or anchored) bodies first, each sampled until it clears what is already placed; clutter sits at twin poses meanwhile
    for b in bodies:
        if b not in movable:
            foot[b] = scene.footprint(b)
    foot["robot_base"] = scene.footprint(scene.m.body(scene.robot_base).name, subtree=False)
    for b in movable:
        x, y, yaw = base[b]
        for k in groups.get(b, []):  # children ride inside the parent's hull
            foot.pop(k, None)
        ok = False
        for tries in range(1, 2001 if jitter else 2):
            dx, dy, dyaw = (rng.uniform(-JITTER_XY_M, JITTER_XY_M, 2).tolist() + [rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG)]) if jitter else (0.0, 0.0, 0.0)
            pos = np.array([x + dx, y + dy, FLAT_Z.get(b, twin.pose[b][0][2])]); quat = quat_z(np.deg2rad(yaw + dyaw))
            anchored = b == "basket" and spec.get("anchor_basket", False)
            # the headphone case is exempt from the region / reach rule (as close_lid in generate_variants): its base half sits far,
            # the lid toward the robot, and the pinch IK on both lid walls is the reach test
            if b in spec["targets"] and not anchored and b != "headphone_case" and not region_contains(pos[:2], REACH_M):
                continue
            if b == "basket" and not anchored and not region_contains(pos[:2], BASKET_REACH_M):
                continue
            if b not in spec["targets"] and b != "basket" and not point_in_polygon(pos[:2], REGION_XY):
                continue
            scene.set_pose(b, pos, quat)
            cps = group_child_poses(twin, b, pos, quat, groups.get(b, []), seats)
            for c, (cp, cq) in cps.items():
                scene.set_pose(c, cp, cq)
            fp = scene.footprint(b)
            others = {k: v for k, v in foot.items() if k != b}
            # the anchored basket keeps the twin's desk-edge placement (its woven mesh overhangs the margin there)
            gaps = {k: round(gap(fp, v), 4) for k, v in others.items()}
            if (not anchored and not inside_table(scene.aabb(b), MARGIN_M)) or min(gaps.values()) < MARGIN_M:
                diag = dict(inside_table=inside_table(scene.aabb(b), MARGIN_M), tight={k: g for k, g in gaps.items() if g < MARGIN_M})
                continue
            poses[b] = (pos, quat); foot[b] = fp
            for c, cq in cps.items():
                poses[c] = cq
            log[b] = dict(tries=tries, jitter=dict(dx_m=round(dx, 4), dy_m=round(dy, 4), dyaw_deg=round(dyaw, 2)) if jitter else None)
            ok = True
            break
        if not ok:
            return None, dict(failed=b, why="no clearing pose", last=diag, log=log)
    # full desk clutter: jittered only if one of a few samples keeps the clearance, else the twin pose
    if jitter and spec["bodies"] == "all":
        for b in clutter:
            if b in ("basket",) or b in kids:
                continue
            p0, q0 = twin.pose[b]
            for k in groups.get(b, []):
                foot.pop(k, None)
            for tries in range(1, CLUTTER_JITTER_TRIES + 1):
                dx, dy = rng.uniform(-JITTER_XY_M, JITTER_XY_M, 2); dyaw = rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG)
                pos = np.array([p0[0] + dx, p0[1] + dy, p0[2]]); quat = quat_z(yaw_of(q0) + np.deg2rad(dyaw))
                scene.set_pose(b, pos, quat)
                cps = group_child_poses(twin, b, pos, quat, groups.get(b, []))
                for c, (cp, cq) in cps.items():
                    scene.set_pose(c, cp, cq)
                fp = scene.footprint(b)
                if inside_table(scene.aabb(b), MARGIN_M) and all(gap(fp, v) >= MARGIN_M for k, v in foot.items() if k != b):
                    poses[b] = (pos, quat); foot[b] = fp
                    for c, cq in cps.items():
                        poses[c] = cq
                    log[b] = dict(tries=tries, jitter=dict(dx_m=round(float(dx), 4), dy_m=round(float(dy), 4), dyaw_deg=round(float(dyaw), 2)), clutter=True)
                    break
            else:
                scene.set_pose(b, p0, q0)
                for c, (cp, cq) in group_child_poses(twin, b, p0, q0, groups.get(b, [])).items():
                    scene.set_pose(c, cp, cq)
    checked = list(poses)
    pen = scene.penetration(checked, [[p] + kids for p, kids in groups.items()]) if checked else 0.0
    if pen < -PEN_TOL_M:
        return None, dict(failed="penetration", pen_mm=round(pen * 1e3, 2), log=log)
    ik = {}
    for b in spec["targets"] + (["basket"] if "basket" in bodies else []):
        ik.update(scene.ik_targets(b, poses[b][0] if b in poses else twin.pose[b][0]))
    if not all(r["ok"] for r in ik.values()):
        return None, dict(failed="ik", ik=ik, log=log)
    return poses, dict(log=log, ik=ik, pen_mm=round(pen * 1e3, 2), movable=movable, groups=groups, clutter=clutter)


def settle(folder, bodies):
    """1 s of physics with the arm held at park: displacement and worst contact distance per top-level body (children included)."""
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml")); m.opt.timestep = 1.0 / (FPS * SUBSTEPS); d = mujoco.MjData(m)
    q0 = int(m.jnt_qposadr[m.joint("joint1").id])
    mujoco.mj_resetData(m, d); d.ctrl[:] = d.qpos[q0:q0 + 8]; mujoco.mj_forward(m, d)
    ids = {n: m.body(n).id for n in bodies}
    x0 = d.xpos.copy(); worst = {n: 0.0 for n in ids}
    for _ in range(int(SETTLE_S * FPS * SUBSTEPS)):
        mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for n, i in ids.items():
                if i in (b1, b2):
                    worst[n] = min(worst[n], float(c.dist))
    moved = {n: round(float(np.linalg.norm(d.xpos[i] - x0[i])) * 1e3, 2) for n, i in ids.items()}
    pen = {n: round(v * 1e3, 2) for n, v in worst.items()}
    return dict(moved_mm=moved, max_moved_mm=max(moved.values()), contact_dist_mm=pen, worst_contact_dist_mm=min(pen.values()))


def render(folder, cams=("ext", "wrist")):
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml")); d = mujoco.MjData(m)
    q0 = int(m.jnt_qposadr[m.joint("joint1").id])
    mujoco.mj_resetData(m, d); d.ctrl[:] = d.qpos[q0:q0 + 8]
    m.opt.timestep = 1.0 / (FPS * SUBSTEPS)
    for _ in range(int(0.5 * FPS * SUBSTEPS)):  # the robot's own 0.5 s settle before the start pose
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, 480, 640)
    out = {}
    for cam in cams:
        r.update_scene(d, cam); out[cam] = r.render().copy()
    r.close()
    return out


def sheet(tiles, path, cols=4, size=(480, 360)):
    W, H = size
    rows = (len(tiles) + cols - 1) // cols
    im = Image.new("RGB", (cols * W, rows * (H + 22)), (20, 20, 20))
    font = ImageFont.load_default(size=15)
    for i, (label, img) in enumerate(tiles):
        x, y = (i % cols) * W, (i // cols) * (H + 22)
        im.paste(Image.fromarray(img).resize((W, H)), (x, y + 22))
        ImageDraw.Draw(im).text((x + 6, y + 3), label, fill=(255, 255, 255), font=font)
    im.save(path)


def role_of(n, info):
    if n in info["targets"]:
        return "target"
    if n == "basket":
        return "basket"
    if n in info["movable"]:
        return "distractor"
    for parent, kids in info["groups"].items():
        if n in kids:
            return f"rides_with_{parent}"
    return "clutter"


def pose_meta(poses, twin, bodies, info):
    objects = {}
    for n in bodies:
        pos, quat = poses.get(n, twin.pose[n])
        lg = info["log"].get(n, {})
        objects[n] = dict(xy=[round(float(pos[0]), 4), round(float(pos[1]), 4)], z=round(float(pos[2]), 4),
                          yaw_deg=round(float(np.degrees(yaw_of(quat))), 2) if abs(quat[1]) < 1e-6 and abs(quat[2]) < 1e-6 else None,
                          quat_wxyz=[round(float(v), 5) for v in quat], radius_from_base_m=round(float(np.linalg.norm(pos[:2])), 4),
                          role=role_of(n, info),
                          placed=n in poses, jittered=lg.get("jitter") is not None, jitter=lg.get("jitter"))
    return objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--check-base", action="store_true", help="only check the base configurations (p1), print the diagnostics, write nothing else")
    args = ap.parse_args()
    copied = copy_assets()
    print(f"assets: {len(copied)} files under {ASSETS.relative_to(WS)}")
    twin = TwinText()
    print(f"twin scene snapshot {twin.snapshot}: bodies {twin.all_bodies}")
    checks_path = HERE / "checks.json"
    checks = {} if args.configs == list(CONFIGS) or not checks_path.exists() else json.loads(checks_path.read_text())
    checks["_twin_snapshot"] = twin.snapshot
    checks["_rules"] = dict(seed=SEED, placements=N_PLACEMENTS, jitter_xy_m=JITTER_XY_M, jitter_yaw_deg=JITTER_YAW_DEG, margin_m=MARGIN_M, reach_m=REACH_M,
                            basket_reach_m=BASKET_REACH_M, settle_s=SETTLE_S, settle_tol_m=SETTLE_TOL_M, start_contact_tol_m=PEN_TOL_M, region_xy=REGION_XY.tolist())
    RENDERS.mkdir(exist_ok=True)
    p1_tiles = []
    for name in args.configs:
        spec = CONFIGS[name]
        bodies = list(twin.all_bodies) if spec["bodies"] == "all" else list(spec["bodies"])
        seed = [SEED, list(CONFIGS).index(name)]
        rng = np.random.default_rng(seed)
        # the model every candidate is checked in: p1's folder written first at the twin poses
        folder1 = SCENES / f"{name}_p1"; folder1.mkdir(exist_ok=True)
        closed = spec.get("closed_kit", False)
        (folder1 / "scene.xml").write_text(twin.scene_xml(f"scene_{name}_p1", bodies, {}, closed_kit=closed))
        scene = Scene(folder1 / "scene.xml", bodies)
        entries, tiles = {}, []
        for k in range(1, N_PLACEMENTS + 1):
            folder_name = f"{name}_p{k}"
            folder = SCENES / folder_name; folder.mkdir(exist_ok=True)
            attempts = 0
            while True:
                attempts += 1
                assert attempts <= 500, f"{folder_name}: no placement passed the checks in {attempts} attempts"
                poses, info = place(twin, scene, spec, bodies, rng, jitter=k > 1)
                if poses is None:
                    assert k > 1, f"{name} base configuration fails: {json.dumps(info, indent=1, default=str)}"
                    continue
                (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{folder_name}", bodies, poses, closed_kit=closed))
                chk = settle(folder, bodies)
                watched = ([b for b in poses] or spec["targets"]) + (["bits_kit"] if closed and "bits_kit" not in poses else [])
                ok_settle = max(chk["moved_mm"][b] for b in watched) <= SETTLE_TOL_M * 1e3
                if ok_settle:
                    break
                assert k > 1, f"{name} base configuration does not settle: { {b: chk['moved_mm'][b] for b in watched} }"
            info["targets"] = spec["targets"]
            meta = dict(config=name, level=spec["level"], letter=spec["letter"], placement=k, tier=f"twin_l{spec['level']}", seed=seed,
                        generator="lerobot/scripts/sim_builders/digital_twin/envs/build_twin_envs.py", twin_snapshot=twin.snapshot, source=str(twin.src.relative_to(WS)),
                        rules=checks["_rules"] | dict(jittered=k > 1), task_objects=spec["targets"], basket_anchored=bool(spec.get("anchor_basket", False)),
                        kit_closed=closed, kit_closed_angles_deg=CLOSED_KIT_DEG if closed else None,
                        objects=pose_meta(poses, twin, bodies, info), unchanged=[b for b in bodies if b not in poses],
                        ik=info["ik"], start_contact_mm=info["pen_mm"], attempts=attempts, settle=chk,
                        assets="meshes and textures under scenes/assets/twin/ and the package photoreal folders (assetdir ..); room = assets/twin/room/room.xml; no background composite")
            (folder / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
            watched_moved = {b: chk["moved_mm"][b] for b in watched}
            entries[folder_name] = dict(poses={b: dict(xy=[round(float(v), 4) for v in p[0][:2]], yaw_deg=yaw_deg(p[1])) for b, p in poses.items()},
                                        jitter={b: info["log"][b]["jitter"] for b in info["log"] if info["log"][b].get("jitter")}, attempts=attempts,
                                        ik_ok=all(r["ok"] for r in info["ik"].values()), ik=info["ik"], settle_ok=bool(ok_settle), start_contact_mm=info["pen_mm"],
                                        settle_watched_moved_mm=watched_moved, settle_all_max_mm=chk["max_moved_mm"], worst_contact_dist_mm=chk["worst_contact_dist_mm"], ngeom=int(scene.m.ngeom))
            jit = ", ".join(f"{b} ({p[0][0]:.2f}, {p[0][1]:.2f}) {np.degrees(yaw_of(p[1])):.0f}" for b, p in poses.items() if b in info["movable"])
            print(f"  {folder_name}: attempts {attempts}, ik {'ok' if entries[folder_name]['ik_ok'] else 'FAIL'}, settle max {max(watched_moved.values()):.1f} mm (all {chk['max_moved_mm']:.1f}), "
                  f"start contact {info['pen_mm']:.1f} mm, worst settle contact {chk['worst_contact_dist_mm']:.1f} mm; {jit}")
            if args.check_base:
                break
            if not args.no_render:
                imgs = render(folder, cams=("ext", "wrist") if k == 1 else ("ext",))
                tiles.append((f"{folder_name}", imgs["ext"]))
                if k == 1:
                    Image.fromarray(imgs["ext"]).save(RENDERS / f"{folder_name}_ext.png"); Image.fromarray(imgs["wrist"]).save(RENDERS / f"{folder_name}_wrist.png")
                    p1_tiles.append((f"L{spec['level']}{spec['letter']} {name}", imgs["ext"]))
        if not args.no_render and not args.check_base:
            sheet(tiles, RENDERS / f"sheet_{name}.png")
        checks[name] = dict(level=spec["level"], letter=spec["letter"], tier=f"twin_l{spec['level']}", seed=seed, bodies=bodies, targets=spec["targets"],
                            base=spec["base"], anchor_basket=bool(spec.get("anchor_basket", False)), kit_closed=closed, placements=entries)
    if p1_tiles and not args.no_render and not args.check_base:
        # one row per level, four columns, from the p1 renders on disk (so a partial run still rebuilds the whole sheet)
        p1_tiles = [(f"L{v['level']}{v['letter']} {n}", np.asarray(Image.open(RENDERS / f"{n}_p1_ext.png"))) for n, v in CONFIGS.items() if (RENDERS / f"{n}_p1_ext.png").exists()]
        rows = []
        for level in (1, 2, 3, 4):
            row = [t for t in p1_tiles if t[0].startswith(f"L{level}")]
            while len(row) < 4:
                row.append((f"L{level}{'abcd'[len(row)]} (no scene)", np.full_like(p1_tiles[0][1], 20)))
            rows += row
        sheet(rows, RENDERS / "contact_sheet_p1.png", cols=4)
    if not args.check_base:
        checks_path.write_text(json.dumps(checks, indent=1) + "\n")


if __name__ == "__main__":
    main()
