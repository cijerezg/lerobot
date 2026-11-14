"""Twin eval suite v2 (2026-11-08): four HAND-DESIGNED placements per configuration instead of p1 + three jitters.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/twin_eval_v2/build_twin_envs_v2.py --check [--configs ...]
    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/twin_eval_v2/build_twin_envs_v2.py [--configs ...] [--no-render]

Why: the 10-30 build jittered p2-p4 by xy +-6 cm / yaw +-20 deg around p1, so 48 of the 64 scenes were near copies of the 16 base
configurations (user, 2026-11-08). Here every placement is written out by hand: the target changes side (robot's left / right),
distance (near x ~0.20 / far x ~0.46), the basket moves to a different place and orientation, distractors are arranged differently
(next to the target, between target and basket, out of the way), and on the full desk the bit kit is open at its twin pose, mirrored,
or closed and moved, with the clutter re-arranged around it. Same checks as v1 (../digital_twin/envs/build_twin_envs.py,
imported here): hand-placed targets inside the README region and the 0.50 m reach disc (basket 0.48 m), every placed body 2 cm inside
the desk and 2 cm clear of every other body and the robot base, no start contact < 1 mm, IK reach (top pinch / side pinch / 5 cm over
the basket rim / both kit flap edges / both lid walls), 1 s settle < 5 mm. Scenes go to the package scenes/ folder under the SAME
names (twin_l<L>_<config>_p<1-4>), so tasks.py / run_sim_eval.sh / sim_summary need no change; v1 scenes are backed up in
assets/sim_sources/twin_eval_v2/scenes_v1_backup/. `--check` validates every placement without touching the package (models in the scratch dir).

Frame: x toward the far edge / top camera, y to the robot's right, yaw in degrees. Poses are (x, y, yaw) of the body origin.
"""
import argparse
import json
import re
import sys
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WS = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "digital_twin/envs"))
import build_twin_envs as V1  # noqa: E402  (geometry, TwinText, Scene, place, settle, render, sheet, pose_meta, copy_assets)

SCENES = V1.SCENES
# 2026-11-08 (user: objects can be farther out): the IK test is the reach rule. The 0.50 m disc, the 0.48 m basket disc and the
# data-coverage region are dropped (region = the whole desk; inside_table keeps the 2 cm desk margin). IK scan: a straight-down pinch
# at bit height reaches ~0.52 m (y +-0.46 at x 0.18, +-0.42 at x 0.30, +-0.30 at x 0.42, +-0.14 at x 0.50); a tall top pinch ~0.46 m;
# 5 cm over the basket rim reaches the whole desk.
V1.REACH_M, V1.BASKET_REACH_M = 1.0, 1.0
V1.REGION_XY = np.array([(-0.05, -0.61), (-0.05, 0.61), (0.72, 0.61), (0.72, -0.61)])
# A pinch may lean OUTWARD (away from the base) up to MAX_TILT_DEG: the real data grasps the clippers at x 0.59, past the straight-down
# reach. Min-tilt scan at bit height: 15 deg reaches x 0.54 across the middle, 30 deg x 0.62 at the centre and y +-0.50 beside the robot.
MAX_TILT_DEG = 30
_ik_targets_vertical = V1.Scene.ik_targets


def _ik_tilted(scene, target, tilt_deg, w=0.3):
    """Pinch at `target` with the approach axis tilted `tilt_deg` from straight down, leaning away from the base."""
    import mujoco
    m, d = scene.m, scene.d
    ge = m.body("gripper_end").id; lo, hi = scene.limits; q = slice(scene.q0, scene.q0 + 6)
    phi, t = np.arctan2(target[1], target[0]), np.radians(tilt_deg)
    a_des = np.array([np.cos(phi) * np.sin(t), np.sin(phi) * np.sin(t), -np.cos(t)])
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    res = dict(ok=False)
    for s in ([0, -1.0, -1.0, 0.5, 0, 0], [phi, -0.8, -1.5, 0.8, 0, 0], [phi, -1.5, -0.5, 0.0, 0, 0], [phi, -0.3, -2.0, 0.5, 0, 0], [phi, -1.2, -0.8, -0.5, 0, 0]):
        d.qpos[q] = np.clip(s, lo, hi); d.qpos[scene.q0 + 6:scene.q0 + 8] = 0.05
        for _ in range(300):
            mujoco.mj_forward(m, d); a = d.xmat[ge].reshape(3, 3) @ scene.approach; tip = d.xpos[ge] + scene.tip_len * a
            mujoco.mj_jac(m, d, jp, jr, tip, ge)
            e = np.concatenate([target - tip, w * np.cross(a, a_des)]); J = np.vstack([jp[:, scene.dofs], w * jr[:, scene.dofs]])
            d.qpos[q] = np.clip(d.qpos[q] + 0.5 * np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ e), lo, hi)
        mujoco.mj_forward(m, d); a = d.xmat[ge].reshape(3, 3) @ scene.approach; tip = d.xpos[ge] + scene.tip_len * a
        hit = any((m.geom_bodyid[c.geom1] in scene.robot_bodies or m.geom_bodyid[c.geom2] in scene.robot_bodies) and c.dist < -1e-3
                  and "table" in (m.geom(c.geom1).name + m.geom(c.geom2).name) for c in d.contact[:d.ncon])
        pos_err, ang_err = float(np.linalg.norm(target - tip)), float(np.degrees(np.arccos(np.clip(a @ a_des, -1, 1))))
        res = dict(ok=bool(pos_err < 0.005 and ang_err < 5.0 and not hit), pos_err_mm=round(pos_err * 1e3, 2), tilt_deg=tilt_deg,
                   q_deg=np.degrees(d.qpos[q]).round(1).tolist(), target=[round(float(v), 4) for v in target])
        if res["ok"]:
            break
    d.qpos[q] = m.qpos0[q]; d.qpos[scene.q0 + 6:scene.q0 + 8] = m.qpos0[scene.q0 + 6:scene.q0 + 8]; mujoco.mj_forward(m, d)
    return res


def _ik_targets(self, body, pos):
    """The v1 reach tests; a failed straight-down object pinch is retried leaning outward 15 / 30 deg (basket, kit flaps, lid walls unchanged)."""
    out = _ik_targets_vertical(self, body, pos)
    if body in ("basket", "bits_kit", "headphone_case") or body in V1.SIDE_PINCH_Z:
        return out
    for k, r in out.items():
        for tilt in range(15, MAX_TILT_DEG + 1, 15):
            if r["ok"]:
                break
            r = _ik_tilted(self, np.array(r["target"]), tilt)
        out[k] = r
    return out


V1.Scene.ik_targets = _ik_targets
RENDERS = HERE / "renders"
TWIN_KIT = (0.322, -0.055, 245.6)   # the open kit's twin pose; mirrored = yaw - 180 (bits flap on the other side)
KIT_MIRROR = (0.322, -0.055, 65.6)
CUP_TWIN = (0.200, 0.250, 0.0)

# ---- the 16 configurations x 4 hand-designed placements -------------------------------------------------------------------------
# Each placement: poses = body -> (x, y, yaw) for every body that leaves its twin pose (the basket included unless anchored),
# closed_kit (full desk only) = the bit kit closed (angles baked, as v1's twin_l3_close_lid), note = what the placement tests.
L1 = ["basket"]


def P(note, closed_kit=False, **poses):
    return dict(note=note, closed_kit=closed_kit, poses=poses)


KIT_RIGHT = (0.38, 0.14, 155.6)     # the CLOSED kit moved to the right side (free-space scan 2026-11-08: clears cup, bits box, basket)
CUP_FAR_RIGHT = (0.33, 0.28, 0.0)   # the far-right pocket beside the anchored basket; a tall top pinch still reaches here (not at x 0.38)
CONFIGS = {
    # ---- L1: basket + one object. Sides, distances, basket orientation, adjacency, basket right in front of the robot.
    "twin_l1_tape_basket_right": dict(level=1, letter="a", task="tape_roll", bodies=L1 + ["tape_roll"], targets=["tape_roll"], placements=[
        P("v1 base: tape mid-left of the basket, basket right", basket=(0.36, 0.22, 0), tape_roll=(0.30, -0.06, 0)),
        P("tape near-right, basket far-left turned 90", basket=(0.44, -0.14, 90), tape_roll=(0.20, 0.28, 0)),
        P("tape far-left, basket near-right", basket=(0.28, 0.30, 0), tape_roll=(0.46, -0.08, 0)),
        P("tape right in front of the robot, basket far behind it", basket=(0.47, 0.03, 0), tape_roll=(0.19, 0.00, 0)),
    ]),
    # 2026-11-08 bits pass (user): fewer tape-first tasks; a single bit as the target, the bit kit alone (fold only, bits never go in the kit)
    "twin_l1_bit_single": dict(level=1, letter="b", task="bit_1", text="Put the bit in the basket", bodies=L1 + ["bit_1"], targets=["bit_1"], placements=[
        P("bit far-right at 30, basket out on the left side", basket=(0.26, -0.42, 90), bit_1=(0.54, 0.26, 30)),
        P("bit far-left across the gripper (90), basket far-right corner", basket=(0.54, 0.36, 0), bit_1=(0.58, -0.20, 90)),
        P("bit at the far end straight ahead, basket out on the right side", basket=(0.26, 0.40, 90), bit_1=(0.62, 0.02, 0)),
        P("bit wide right beside the robot at -45, basket far-left corner", basket=(0.54, -0.36, 0), bit_1=(0.26, 0.48, -45)),
    ]),
    "twin_l1_pill_basket_far": dict(level=1, letter="c", task="pill_bottle", bodies=L1 + ["pill_bottle"], targets=["pill_bottle"], placements=[
        P("pill near-right, basket far-left", basket=(0.42, -0.18, 0), pill_bottle=(0.22, 0.28, 0)),
        P("basket near-left, pill adjacent to its right wall", basket=(0.30, -0.26, 0), pill_bottle=(0.24, -0.06, 0)),
        P("basket far-right turned 90, pill far-left", basket=(0.36, 0.30, 90), pill_bottle=(0.42, -0.12, 0)),
        P("basket far-right turned 30, pill near-left", basket=(0.40, 0.24, 30), pill_bottle=(0.22, -0.24, 0)),
    ]),
    "twin_l1_kit_alone": dict(level=1, letter="d", task="fold_kit", text="Close the bit kit", bodies=["bits_kit"], targets=["bits_kit"], placements=[
        P("kit alone at its twin pose", bits_kit=TWIN_KIT),
        P("kit mirrored (bits flap on the right), centre-right", bits_kit=(0.32, 0.08, 65.6)),
        P("kit turned -30 on the left side", bits_kit=(0.32, -0.12, 215.6)),
        P("kit turned +60 on the right side", bits_kit=(0.34, 0.14, 305.6)),
    ]),
    # ---- L2: target + distractors. Distractors around the target, in the path, adjacent to the target, out of the way.
    "twin_l2_bit_clutter": dict(level=2, letter="a", task="bit_1", text="Put the bit in the basket", bodies=L1 + ["bit_1", "tape_roll", "pill_bottle", "clipper_case", "clipper_large", "clipper_small"],
                                targets=["bit_1"], groups={"clipper_case": ["clipper_large", "clipper_small"]}, placements=[
        P("basket far-left corner, bit far-right, case at the far end, tape centre, pill near-left", clipper_case=(0.62, 0.08, 90), basket=(0.54, -0.36, 0), bit_1=(0.54, 0.26, 30),
          tape_roll=(0.38, 0.02, 0), pill_bottle=(0.22, -0.34, 0)),
        P("basket out on the right side, bit far-left, tape at the far end, pill in the carry path, case near-left", basket=(0.26, 0.40, 90), bit_1=(0.58, -0.20, 90), tape_roll=(0.62, 0.04, 0),
          pill_bottle=(0.42, 0.18, 0), clipper_case=(0.24, -0.38, 0)),
        P("basket out on the left side, bit at the far end in the clutter: tape and pill beside it, case in front of it", clipper_case=(0.40, 0.04, 90), basket=(0.30, -0.42, 90), bit_1=(0.62, 0.02, 0),
          tape_roll=(0.58, -0.18, 0), pill_bottle=(0.62, 0.18, 0)),
        P("basket far-right corner, bit wide right beside the robot with the pill next to it, tape far-left, case centre", basket=(0.54, 0.36, 0), bit_1=(0.26, 0.48, -45),
          pill_bottle=(0.22, 0.36, 0), tape_roll=(0.58, -0.24, 0), clipper_case=(0.40, -0.08, 0)),
    ]),
    # the clipper case is listed first where it leaves the twin pose before the basket: its two clippers ride in it
    "twin_l2_pill_case": dict(level=2, letter="b", task="pill_bottle", bodies=L1 + ["pill_bottle", "clipper_case", "clipper_large", "clipper_small"], targets=["pill_bottle"],
                              groups={"clipper_case": ["clipper_large", "clipper_small"]}, placements=[
        P("basket right, pill near-left, case far-left", clipper_case=(0.44, -0.14, 180), basket=(0.36, 0.26, 0), pill_bottle=(0.22, -0.22, 0)),
        P("basket far-left, case near-right turned 90, pill near centre", clipper_case=(0.22, 0.30, 90), basket=(0.44, -0.14, 0), pill_bottle=(0.22, 0.08, 0)),
        P("basket far centre, pill far-right, case left turned 90: pill and case on opposite sides", clipper_case=(0.30, -0.26, 90), basket=(0.46, 0.02, 0), pill_bottle=(0.30, 0.32, 0)),
        P("basket right turned 90, pill near-left, case between pill and basket", basket=(0.36, 0.30, 90), pill_bottle=(0.20, -0.22, 0), clipper_case=(0.30, -0.02, 0)),
    ]),
    "twin_l2_roar_tapes": dict(level=2, letter="c", task="roar_bottle", bodies=L1 + ["roar_bottle", "tape_roll", "tape_roll_2", "clipper_large", "clipper_small"], targets=["roar_bottle"], placements=[
        P("basket right, bottle near-left, tapes far-left and far centre, clippers near centre", basket=(0.34, 0.30, 0), roar_bottle=(0.20, -0.22, 0), tape_roll=(0.44, -0.22, 0),
          tape_roll_2=(0.40, 0.06, 0), clipper_large=(0.22, 0.04, 45), clipper_small=(0.30, -0.04, -30)),
        P("basket near-left, bottle far-right, tapes near-right and far centre-left, clippers centre", basket=(0.28, -0.26, 0), roar_bottle=(0.40, 0.28, 0), tape_roll=(0.18, 0.30, 0),
          tape_roll_2=(0.46, -0.02, 0), clipper_large=(0.30, 0.08, 90), clipper_small=(0.36, 0.16, 0)),
        P("basket far-left, bottle near-right, tapes near-left and far-right, clippers centre and right", basket=(0.44, -0.16, 0), roar_bottle=(0.22, 0.26, 0), tape_roll=(0.20, -0.10, 0),
          tape_roll_2=(0.44, 0.16, 0), clipper_large=(0.32, 0.08, 45), clipper_small=(0.30, 0.34, -30)),
        P("basket right turned 90, bottle far-left with a tape beside it, other tape near centre, clippers near-left and far centre", basket=(0.33, 0.32, 90), roar_bottle=(0.46, -0.14, 0),
          tape_roll=(0.42, -0.26, 0), tape_roll_2=(0.20, -0.02, 0), clipper_large=(0.20, -0.24, 45), clipper_small=(0.46, 0.06, -30)),
    ]),
    "twin_l2_tape_bottles": dict(level=2, letter="d", task="tape_roll", bodies=L1 + ["tape_roll", "pill_bottle", "roar_bottle"], targets=["tape_roll"], placements=[
        P("v1 base: basket near-left, tape far-right, bottles right", basket=(0.28, -0.26, 0), tape_roll=(0.44, 0.16, 0), pill_bottle=(0.22, 0.14, 0), roar_bottle=(0.40, 0.34, 0)),
        P("basket far-left, tape near-left with the pill adjacent, orange bottle near-right", basket=(0.44, -0.16, 0), tape_roll=(0.20, -0.20, 0), pill_bottle=(0.20, -0.08, 0), roar_bottle=(0.22, 0.26, 0)),
        P("basket near-right, tape far-left between the two bottles", basket=(0.26, 0.28, 0), tape_roll=(0.46, -0.12, 0), pill_bottle=(0.48, 0.08, 0), roar_bottle=(0.36, -0.26, 0)),
        P("basket left turned 90, tape near, orange bottle between tape and basket, pill far-right", basket=(0.40, -0.22, 90), tape_roll=(0.20, 0.08, 0), roar_bottle=(0.32, 0.04, 0), pill_bottle=(0.42, 0.26, 0)),
    ]),
    # ---- L3: full desk (basket anchored at its twin pose with its contents). Four desk arrangements (free-space scan 2026-11-08):
    # A twin desk (kit open, targets in the near-right pocket beside the cup); B kit CLOSED at its twin pose (frees the near-left strip
    # and the band right of the kit); C kit open but MIRRORED (bits flap on the other side), cup moved to the far-right pocket, targets
    # in the near-right pocket; D kit closed and moved to the right side, targets on the left and in the centre.
    "twin_l3_tape": dict(level=3, letter="a", task="tape_roll", bodies="all", targets=["tape_roll"], anchor_basket=True, placements=[
        P("A twin desk, tape in the near-right pocket", tape_roll=(0.19, 0.12, 0)),
        P("B kit closed, tape near-left, cup moved right of the kit", closed_kit=True, tape_roll=(0.22, -0.26, 0), paper_cup=(0.42, 0.14, 0)),
        P("C kit mirrored, cup to the far-right pocket, tape near-right", bits_kit=KIT_MIRROR, paper_cup=CUP_FAR_RIGHT, tape_roll=(0.20, 0.16, 0)),
        P("D kit closed and moved right, tape far-left, cup centre", closed_kit=True, bits_kit=KIT_RIGHT, tape_roll=(0.44, -0.10, 0), paper_cup=(0.30, -0.08, 0)),
    ]),
    "twin_l3_pill": dict(level=3, letter="b", task="pill_bottle", bodies="all", targets=["pill_bottle"], anchor_basket=True, placements=[
        P("A twin desk, pill in the near-right pocket", pill_bottle=(0.21, 0.15, 0)),
        P("B kit closed, pill left, cup moved right of the kit", closed_kit=True, pill_bottle=(0.30, -0.26, 0), paper_cup=(0.36, 0.16, 0)),
        P("C kit mirrored, pill in the far-right pocket, cup stays", bits_kit=KIT_MIRROR, pill_bottle=CUP_FAR_RIGHT),
        P("D kit closed and moved right, pill far-left, cup near centre", closed_kit=True, bits_kit=KIT_RIGHT, pill_bottle=(0.42, -0.12, 0), paper_cup=(0.22, -0.04, 0)),
    ]),
    "twin_l3_fold_kit": dict(level=3, letter="c", task="fold_kit", bodies="all", targets=["bits_kit"], anchor_basket=True, placements=[
        P("A twin desk, kit at its twin pose", bits_kit=TWIN_KIT),
        P("kit mirrored: bits flap on the other side", bits_kit=KIT_MIRROR),
        P("kit shifted right, cup moved to the near-left strip", bits_kit=(0.32, 0.08, 245.6), paper_cup=(0.22, -0.26, 0)),
        P("kit turned 60 deg, cup moved to the near-left strip", bits_kit=(0.30, 0.00, 305.6), paper_cup=(0.22, -0.26, 0)),
    ]),
    "twin_l3_close_lid": dict(level=3, letter="d", task="close_lid", bodies="all", targets=["headphone_case"], anchor_basket=True, placements=[
        P("v1 base: kit closed at its twin pose, case far-left, lid toward the robot", closed_kit=True, headphone_case=(0.46, -0.40, 210)),
        P("kit closed and moved right, case near the centre-left", closed_kit=True, bits_kit=KIT_RIGHT, headphone_case=(0.34, -0.20, 210)),
        P("kit closed and moved right, case mid-left turned so the lid opens to the robot's left", closed_kit=True, bits_kit=KIT_RIGHT, headphone_case=(0.42, -0.16, 255)),
        P("kit closed at its twin pose, case at the far end of the feasible band", closed_kit=True, headphone_case=(0.50, -0.34, 225)),
    ]),
    # ---- L4: full desk, sequences; the same four desk arrangements.
    "twin_l4_tape_cup": dict(level=4, letter="a", task=["tape_roll", "paper_cup"], bodies="all", targets=["tape_roll", "paper_cup"], anchor_basket=True, placements=[
        P("A twin desk, tape beside the cup in the near-right pocket", tape_roll=(0.19, 0.12, 0), paper_cup=CUP_TWIN),
        P("B kit closed, tape near-left, cup right of the kit", closed_kit=True, tape_roll=(0.22, -0.26, 0), paper_cup=(0.42, 0.14, 0)),
        P("C kit mirrored, cup in the far-right pocket, tape near-right", bits_kit=KIT_MIRROR, paper_cup=CUP_FAR_RIGHT, tape_roll=(0.20, 0.14, 0)),
        P("D kit closed and moved right, tape far-left, cup near centre", closed_kit=True, bits_kit=KIT_RIGHT, tape_roll=(0.44, -0.14, 0), paper_cup=(0.22, -0.02, 0)),
    ]),
    # the twin's loose bit pile (bit_1..bit_6 next to the robot base) is cut down to the two targets, so "the two bits" is unambiguous
    "twin_l4_two_bits": dict(level=4, letter="b", task=["bit_1", "bit_2"], text="Put the two bits in the basket", bodies="all", exclude=["bit_3", "bit_4", "bit_5", "bit_6"],
                             targets=["bit_1", "bit_2"], anchor_basket=True, placements=[
        P("A twin desk, one bit between kit and basket, one at the far end among the clippers", bit_1=(0.50, 0.24, 30), bit_2=(0.64, 0.02, -45)),
        P("B kit closed, bits far-left and far-right", closed_kit=True, bit_1=(0.44, -0.24, 60), bit_2=(0.50, 0.26, 0)),
        P("C kit mirrored, cup far-right pocket, one bit at the far end, one wide right beside the robot", bits_kit=KIT_MIRROR, paper_cup=CUP_FAR_RIGHT, bit_1=(0.64, 0.02, 0), bit_2=(0.30, 0.42, 90)),
        P("D kit closed and moved right, bits far-left and wide left beside the robot, cup centre-left", closed_kit=True, bits_kit=KIT_RIGHT, bit_1=(0.44, -0.22, 45),
          bit_2=(0.24, -0.44, -30), paper_cup=(0.30, -0.16, 0)),
    ]),
    "twin_l4_pill_roar": dict(level=4, letter="c", task=["pill_bottle", "roar_bottle"], bodies="all", targets=["pill_bottle", "roar_bottle"], anchor_basket=True, placements=[
        P("A twin desk, pill near-right, orange bottle far-right pocket", pill_bottle=(0.20, 0.12, 0), roar_bottle=(0.37, 0.28, 0)),
        P("B kit closed, orange bottle near-left, pill right of the kit", closed_kit=True, roar_bottle=(0.22, -0.26, 0), pill_bottle=(0.42, 0.12, 0)),
        P("C kit mirrored, both bottles in the near-right pocket, cup far-right", bits_kit=KIT_MIRROR, paper_cup=CUP_FAR_RIGHT, roar_bottle=(0.20, 0.14, 0), pill_bottle=(0.22, 0.26, 0)),
        P("D kit closed and moved right, pill far-left, orange bottle near centre, cup left", closed_kit=True, bits_kit=KIT_RIGHT, pill_bottle=(0.42, -0.14, 0), roar_bottle=(0.24, -0.06, 0), paper_cup=(0.30, -0.24, 0)),
    ]),
    "twin_l4_three": dict(level=4, letter="d", task=["pill_bottle", "paper_cup", "tape_roll"], text="Put the pill bottle, the cup and the tape roll in the basket", bodies="all", targets=["pill_bottle", "paper_cup", "tape_roll"], anchor_basket=True, placements=[
        P("A twin desk, tape and cup near-right, pill far-right pocket", tape_roll=(0.19, 0.13, 0), pill_bottle=(0.33, 0.28, 0), paper_cup=(0.21, 0.26, 0)),
        P("B kit closed, tape and pill on the near-left strip, cup right of the kit", closed_kit=True, tape_roll=(0.20, -0.26, 0), pill_bottle=(0.30, -0.26, 0), paper_cup=(0.42, 0.14, 0)),
        P("C kit mirrored, cup far-right pocket, tape and pill near-right", bits_kit=KIT_MIRROR, paper_cup=CUP_FAR_RIGHT, tape_roll=(0.20, 0.14, 0), pill_bottle=(0.22, 0.26, 0)),
        P("D kit closed and moved right, tape far-left, pill near centre, cup left", closed_kit=True, bits_kit=KIT_RIGHT, tape_roll=(0.44, -0.16, 0), pill_bottle=(0.22, -0.02, 0), paper_cup=(0.32, -0.24, 0)),
    ]),
}
assert all(len(c["placements"]) == 4 for c in CONFIGS.values())


def check_model_dir(check):
    # check models must sit directly under the package scenes/ folder (assetdir .. and the robot include are relative); removed at the end
    return SCENES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--check", action="store_true", help="validate every placement, write nothing into the package")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--preview", action="store_true", help="--check plus ext/wrist renders and sheets (task string on top) in renders/preview/")
    args = ap.parse_args()
    args.check = args.check or args.preview
    out_renders = RENDERS / "preview" if args.preview else RENDERS
    draw = (not args.check or args.preview) and not args.no_render
    if not args.check:
        copied = V1.copy_assets(); print(f"assets: {len(copied)} files under {V1.ASSETS.relative_to(WS)}")
    twin = V1.TwinText()
    # bits (2026-11-08): the hex mesh stays the visual, collision = a box of its bounds at the compiled mesh frame (quat 0.5 x4: long axis
    # along body x), stiff contact. The mesh hull under a pinch let the bit squeeze out or sink into the desk; the box with the finger pads lifts.
    for b in [b for b in twin.bodies if re.fullmatch(r"bit_\d", b)]:
        twin.bodies[b], n = re.subn(
            r'<geom type="mesh" mesh="bit_hex" mass="0.006" (rgba="[^"]*") condim="4" friction="0.6 0.005 0.0001" />',
            r'<geom type="mesh" mesh="bit_hex" mass="0" \1 contype="0" conaffinity="0" />' + "\n      "
            r'<geom type="box" size="0.00346 0.003 0.0125" quat="0.5 0.5 0.5 0.5" mass="0.006" group="3" condim="4" '
            r'friction="0.6 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001 0.5 2" />', twin.bodies[b])
        assert n == 1, b
    print(f"twin scene snapshot {twin.snapshot}")
    out_root = check_model_dir(args.check)
    checks_path = HERE / "checks.json"
    checks = {} if args.configs == list(CONFIGS) or not checks_path.exists() else json.loads(checks_path.read_text())
    checks["_twin_snapshot"] = twin.snapshot
    checks["_rules"] = dict(placements="hand-designed (v2)", margin_m=V1.MARGIN_M, reach_m=V1.REACH_M, basket_reach_m=V1.BASKET_REACH_M, settle_s=V1.SETTLE_S,
                            settle_tol_m=V1.SETTLE_TOL_M, start_contact_tol_m=V1.PEN_TOL_M, region_xy=V1.REGION_XY.tolist())
    out_renders.mkdir(exist_ok=True, parents=True)
    failures, all_tiles = [], []
    rng = np.random.default_rng(0)  # unused: jitter is off
    for name in args.configs:
        spec = CONFIGS[name]
        bodies = [b for b in twin.all_bodies if b not in spec.get("exclude", [])] if spec["bodies"] == "all" else list(spec["bodies"])
        entries, tiles = {}, []
        for k, pl in enumerate(spec["placements"], start=1):
            folder_name = f"{name}_p{k}"
            folder = out_root / (("_check_" if args.check else "") + folder_name); folder.mkdir(exist_ok=True, parents=True)
            closed = pl["closed_kit"]
            # the model every pose is checked in: this placement's bodies at the twin poses (kit open or closed)
            (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{folder_name}", bodies, {}, closed_kit=closed))
            scene = V1.Scene(folder / "scene.xml", bodies)
            spec_k = dict(spec, base={b: tuple(float(v) for v in p) for b, p in pl["poses"].items()}, jitter=[])
            if spec.get("anchor_basket"):
                spec_k["base"].pop("basket", None)
            poses, info = V1.place(twin, scene, spec_k, bodies, rng, jitter=False)
            if poses is None:
                failures.append((folder_name, info))
                print(f"  {folder_name}: FAIL {json.dumps(info, default=str)[:600]}")
                continue
            (folder / "scene.xml").write_text(twin.scene_xml(f"scene_{folder_name}", bodies, poses, closed_kit=closed))
            chk = V1.settle(folder, bodies)
            watched = list(poses) + (["bits_kit"] if closed and "bits_kit" not in poses else [])
            moved = {b: chk["moved_mm"][b] for b in watched}
            ok_settle = max(moved.values()) <= V1.SETTLE_TOL_M * 1e3
            if not ok_settle:
                failures.append((folder_name, dict(failed="settle", moved_mm=moved)))
                print(f"  {folder_name}: FAIL settle {moved}")
                continue
            info["targets"] = spec["targets"]
            meta = dict(config=name, level=spec["level"], letter=spec["letter"], placement=k, tier=f"twin_l{spec['level']}", note=pl["note"],
                        generator="lerobot/scripts/sim_builders/twin_eval_v2/build_twin_envs_v2.py", twin_snapshot=twin.snapshot, source=str(twin.src.relative_to(WS)),
                        rules=checks["_rules"], task_objects=spec["targets"], basket_anchored=bool(spec.get("anchor_basket", False)),
                        kit_closed=closed, kit_closed_angles_deg=V1.CLOSED_KIT_DEG if closed else None,
                        objects=V1.pose_meta(poses, twin, bodies, info), unchanged=[b for b in bodies if b not in poses],
                        ik=info["ik"], start_contact_mm=info["pen_mm"], settle=chk,
                        assets="meshes and textures under scenes/assets/twin/ and the package photoreal folders (assetdir ..); room = assets/twin/room/room.xml; no background composite")
            (folder / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
            entries[folder_name] = dict(note=pl["note"], kit_closed=closed,
                                        poses={b: dict(xy=[round(float(v), 4) for v in p[0][:2]], yaw_deg=V1.yaw_deg(p[1])) for b, p in poses.items() if b in info["movable"]},
                                        ik=info["ik"], start_contact_mm=info["pen_mm"], settle_watched_moved_mm=moved, settle_all_max_mm=chk["max_moved_mm"],
                                        worst_contact_dist_mm=chk["worst_contact_dist_mm"])
            placed = ", ".join(f"{b} ({p[0][0]:.2f}, {p[0][1]:.2f}) {np.degrees(V1.yaw_of(p[1])):.0f}" for b, p in poses.items() if b in info["movable"])
            print(f"  {folder_name}: ok, settle max {max(moved.values()):.1f} mm, start contact {info['pen_mm']:.1f} mm; {placed}")
            if draw:
                imgs = V1.render(folder, cams=("ext", "wrist"))
                Image.fromarray(imgs["ext"]).save(out_renders / f"{folder_name}_ext.png"); Image.fromarray(imgs["wrist"]).save(out_renders / f"{folder_name}_wrist.png")
                tiles += [(f"p{k}: {pl['note']}", imgs["ext"]), (f"p{k} wrist", imgs["wrist"])]
                all_tiles.append((f"L{spec['level']}{spec['letter']} {folder_name}", imgs["ext"]))
        if tiles:  # ext + wrist per placement, the task string in a band on top
            V1.sheet(tiles, out_renders / f"sheet_{name}.png", cols=4, size=(640, 480))
            im = Image.open(out_renders / f"sheet_{name}.png"); top = Image.new("RGB", (im.width, 34), (45, 45, 45))
            ImageDraw.Draw(top).text((8, 6), f"L{spec['level']}{spec['letter']} {name}   task: \"{spec.get('text', spec['task'])}\"", fill=(255, 220, 120), font=ImageFont.load_default(size=20))
            full = Image.new("RGB", (im.width, im.height + 34)); full.paste(top, (0, 0)); full.paste(im, (0, 34)); full.save(out_renders / f"sheet_{name}.png")
        checks[name] = dict(level=spec["level"], letter=spec["letter"], tier=f"twin_l{spec['level']}", bodies=bodies, targets=spec["targets"],
                            anchor_basket=bool(spec.get("anchor_basket", False)), placements=entries)
    if draw and not args.preview:
        tiles = [(f"L{c['level']}{c['letter']} {n}_p{k}", np.asarray(Image.open(RENDERS / f"{n}_p{k}_ext.png")))
                 for n, c in CONFIGS.items() for k in range(1, 5) if (RENDERS / f"{n}_p{k}_ext.png").exists()]
        V1.sheet(tiles, RENDERS / "contact_sheet_all.png", cols=4, size=(480, 360))
    if not args.check:
        checks_path.write_text(json.dumps(checks, indent=1) + "\n")
    if args.check:
        for p in SCENES.glob("_check_twin_*"):
            shutil.rmtree(p)
    print(f"\n{len(failures)} failing placement(s)" + (": " + ", ".join(f for f, _ in failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
