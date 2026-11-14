"""Unit check of the twin scorers without a policy (ENVS_REPORT.md, verification).

    MUJOCO_GL=egl PYTHONSAFEPATH=1 PYTHONPATH=lerobot/src uv run python lerobot/scripts/sim_builders/digital_twin/envs/scorer_check.py \
        [--scene twin_l1_tape_basket_right_p2] [--fold twin_l3_fold_kit_p2] [--sequence twin_l4_three_p2] [--lid twin_l3_tape_p1]

Pick-and-place: the arm is driven kinematically (IK of build_twin_envs.Scene, mj_forward only, the clock advanced 1/30 s per
tick) from the park pose over the object, down to a pinch, the gripper closed until both fingers touch, then the object rides
with the gripper (fixed offset in the gripper_end frame) up and toward the basket. Run A releases it 10 cm outside the basket
footprint (drop record); run B lowers it into the basket and releases it (inside). The stage timeline and the dense curve of
every tick are printed, monotonicity is asserted. Fold: the kit's hinges are driven directly (inner flap, then outer flap)
after a pinch at the inner flap's edge; the ladder must reach 1.0 with the arm back at park. Sequence: every object of the
layout's task placed in order with the same driver. Lid: the registered twin_l3_close_lid scorer (CloseLidScorer) on a full-desk
scene's headphone case, the lid hinge driven 0 -> closed after a pinch at a lid wall (wiring check; no twin_l3_close_lid scene yet).
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_twin_envs import SCENES, Scene  # noqa: E402

from lerobot.robots.rebot_b601_sim.tasks import task_for  # noqa: E402

DT = 1.0 / 30


class Driver:
    def __init__(self, scene_name):
        self.name = scene_name
        folder = SCENES / scene_name
        import json
        bodies = list(json.loads((folder / "objects.json").read_text())["objects"])
        self.s = Scene(folder / "scene.xml", bodies)
        self.m, self.d = self.s.m, self.s.d
        self.q = slice(self.s.q0, self.s.q0 + 6)
        self.g = slice(self.s.q0 + 6, self.s.q0 + 8)
        self.ge = self.m.body("gripper_end").id
        # the robot's own 0.5 s settle, then the scorer
        for _ in range(int(0.5 / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)
        self.scorer = task_for(scene_name).scorer(self.m, self.d, lambda: 0.0)
        self.log = []
        self.last_stage = None

    def tick(self, why=""):
        mujoco.mj_forward(self.m, self.d)
        self.d.time += DT
        self.scorer.poll()
        sc = self.scorer
        dense = getattr(sc, "dense", sc.reward)  # CloseLidScorer (lid run) has no dense value
        self.log.append((round(sc.t, 2), sc.stage_name, sc.reward, round(dense, 4), why))
        if sc.stage_name != self.last_stage:
            print(f"    t={sc.t:6.2f}  stage -> {sc.stage_name:<22} reward {sc.reward:.3f}  dense {dense:.3f}   ({why})")
            self.last_stage = sc.stage_name

    def arm_to(self, target, orient=True):
        res = self.s.ik_top_pinch(np.asarray(target, float), orient=orient)
        self.d.qpos[self.q] = np.deg2rad(res["q_deg"])
        mujoco.mj_forward(self.m, self.d)
        return res

    def tip(self):
        R = self.d.xmat[self.ge].reshape(3, 3)
        return self.d.xpos[self.ge] + self.s.tip_len * (R @ self.s.approach)

    def free_adr(self, body):
        bid = self.m.body(body).id
        return int(self.m.jnt_qposadr[self.m.body_jntadr[bid]])

    def set_body(self, body, pos, quat):
        a = self.free_adr(body)
        self.d.qpos[a:a + 3] = pos; self.d.qpos[a + 3:a + 7] = quat
        mujoco.mj_forward(self.m, self.d)

    def move_tip(self, target, n, orient=True, glued=None, why=""):
        """n ticks interpolating the tip from where it is to `target`; `glued` = (body, rel_pos, rel_quat) in the gripper_end frame."""
        start = self.tip().copy()
        for i in range(1, n + 1):
            self.arm_to(start + (np.asarray(target) - start) * i / n, orient=orient)
            if glued is not None:
                self.place_glued(*glued)
            self.tick(why)

    def place_glued(self, body, rel_pos, rel_quat):
        R = self.d.xmat[self.ge].reshape(3, 3); p = self.d.xpos[self.ge] + R @ rel_pos
        q = np.zeros(4); mujoco.mju_mulQuat(q, self.d.xquat[self.ge], rel_quat)
        self.set_body(body, p, q)

    def grasp(self, body):
        """Close the gripper until both finger bodies touch `body`; returns the body pose in the gripper_end frame."""
        bid = self.m.body(body).id
        for slide in np.linspace(0.05, 0.0, 51):
            self.d.qpos[self.g] = slide
            mujoco.mj_forward(self.m, self.d)
            if self.scorer.pinched(bid):
                break
        R = self.d.xmat[self.ge].reshape(3, 3)
        rel_pos = R.T @ (self.d.xpos[bid] - self.d.xpos[self.ge])
        inv = np.zeros(4); mujoco.mju_negQuat(inv, self.d.xquat[self.ge]); rel_quat = np.zeros(4); mujoco.mju_mulQuat(rel_quat, inv, self.d.xquat[bid])
        return (body, rel_pos, rel_quat), float(self.d.qpos[self.g][0]), self.scorer.pinched(bid)


def pick_place_run(scene_name, body, drop_outside):
    print(f"\n== {scene_name}: {'run A, drop 10 cm outside the basket' if drop_outside else 'run B, place inside'}")
    dr = Driver(scene_name)
    m, d, s = dr.m, dr.d, dr.s
    bid = m.body(body).id
    basket = m.body("basket").id
    p_obj = d.xpos[bid].copy(); top = s.top_z(body)
    z_pinch = p_obj[2] if body in ("tape_roll",) else max(p_obj[2], top - 0.03)
    for _ in range(5):
        dr.tick("start")
    dr.move_tip([p_obj[0], p_obj[1], top + 0.10], 15, why="approach from above")
    dr.move_tip([p_obj[0], p_obj[1], z_pinch], 10, why="descend to the pinch height")
    glued, slide, pinched = dr.grasp(body)
    print(f"    gripper slide {slide:.3f} m, both fingers touching: {pinched}")
    for _ in range(3):
        dr.tick("closed on the object")
    # lift straight up, then carry
    tip0 = dr.tip().copy()
    dr.move_tip(tip0 + [0, 0, 0.12], 10, glued=glued, why="lift")
    pb = d.xpos[basket].copy(); rim = pb[2] + dr.scorer.ladder.foot_hi[2] if hasattr(dr.scorer, "ladder") else pb[2] + 0.212
    if drop_outside:
        # 10 cm outside the footprint on the side facing the object
        Rb = d.xmat[basket].reshape(3, 3); lo, hi = dr.scorer.ladder.foot_lo, dr.scorer.ladder.foot_hi
        q = Rb.T @ (d.xpos[bid] - pb); edge = np.clip(q[:2], lo[:2], hi[:2]); out = q[:2] - edge
        out = out / (np.linalg.norm(out) + 1e-9) * (np.linalg.norm(out) and 0.10)
        target_xy = (pb + Rb @ np.array([edge[0] + out[0], edge[1] + out[1], 0.0]))[:2]
        carry_z = tip0[2] + 0.12
        dr.move_tip([target_xy[0], target_xy[1], carry_z], 20, glued=glued, why="carry toward the basket edge")
        # release: open the gripper, the object falls to the table at that xy
        d.qpos[dr.g] = 0.05
        rest_quat = np.array([1.0, 0, 0, 0])
        dr.set_body(body, [d.xpos[bid][0], d.xpos[bid][1], p_obj[2]], rest_quat)
        for _ in range(10):
            dr.tick("released outside")
        print(f"    drop record: {dr.scorer.ladder.drop}")
    else:
        dr.move_tip([pb[0], pb[1], rim + 0.12 + (tip0[2] - p_obj[2])], 20, glued=glued, why="carry over the basket")
        for _ in range(3):
            dr.tick("over the basket")
        floor_z = pb[2] + 0.012
        dr.move_tip([pb[0], pb[1], floor_z + (p_obj[2] - 0.0) + (tip0[2] - p_obj[2]) + 0.005], 15, glued=glued, why="lower into the basket")
        d.qpos[dr.g] = 0.05
        dr.set_body(body, [pb[0], pb[1], floor_z + p_obj[2]], np.array([1.0, 0, 0, 0]))
        dr.move_tip(dr.tip() + [0, 0, 0.15], 10, why="retreat")
        for _ in range(20):
            dr.tick("hold")
    dense = [row[3] for row in dr.log]
    mono = all(b >= a - 1e-9 for a, b in zip(dense, dense[1:]))
    print(f"    ticks {len(dr.log)}, final stage {dr.scorer.stage_name}, reward {dr.scorer.reward}, dense {dr.scorer.dense:.4f}, monotone {mono}, "
          f"carried {dr.scorer.ladder.carried_s:.2f} s")
    print("    dense curve (every 5th tick): " + " ".join(f"{v:.2f}" for v in dense[::5]))
    assert mono
    return dr


def fold_run(scene_name):
    print(f"\n== {scene_name}: drive the hinges")
    dr = Driver(scene_name)
    m, d, s = dr.m, dr.d, dr.s
    for _ in range(5):
        dr.tick("start")
    targets = s.flap_edge_targets()
    inner = targets["bits_kit_left"]
    dr.move_tip(inner + [0, 0, 0.08], 10, why="approach the bits flap edge")
    dr.move_tip(inner - [0, 0, 0.004], 8, why="pinch into the flap edge")
    d.qpos[dr.g] = 0.0
    for _ in range(3):
        dr.tick("pinch")
    print(f"    raw: {dr.scorer.raw}")
    # park the arm again, then drive the hinges kinematically
    d.qpos[dr.q] = m.qpos0[dr.q]; d.qpos[dr.g] = m.qpos0[dr.g]
    adr = dr.scorer.qadr
    for side, label in (("inner", "bits flap"), ("outer", "card flap")):
        for frac in np.linspace(0, 1, 25):
            for a in adr[side]:
                d.qpos[a] = frac * np.pi / 2  # strip hinge + flap hinge each to 90 deg = 180 deg total
            dr.tick(f"fold the {label}")
    print(f"    angles {dr.scorer.angles_deg()}, final stage {dr.scorer.stage_name}, reward {dr.scorer.reward}, dense {dr.scorer.dense:.4f}")
    dense = [row[3] for row in dr.log]
    mono = all(b >= a - 1e-9 for a, b in zip(dense, dense[1:]))
    print("    dense curve (every 4th tick): " + " ".join(f"{v:.2f}" for v in dense[::4]) + f"   monotone {mono}")
    assert mono and dr.scorer.dense >= 1.0 - 1e-9
    return dr


def sequence_run(scene_name):
    bodies = task_for(scene_name).scorer.BODIES
    print(f"\n== {scene_name}: {' then '.join(bodies)}")
    dr = Driver(scene_name)

    def place(body, slot):
        m, d, s = dr.m, dr.d, dr.s
        bid = m.body(body).id; basket = m.body("basket").id
        p_obj = d.xpos[bid].copy(); top = s.top_z(body)
        z_pinch = p_obj[2] if body.startswith("tape_roll") else max(p_obj[2], top - 0.03)
        dr.move_tip([p_obj[0], p_obj[1], top + 0.10], 12, why=f"approach {body}")
        dr.move_tip([p_obj[0], p_obj[1], z_pinch], 8, why=f"descend {body}")
        glued, slide, pinched = dr.grasp(body)
        for _ in range(3):
            dr.tick(f"closed on {body}")
        tip0 = dr.tip().copy()
        dr.move_tip(tip0 + [0, 0, 0.15], 8, glued=glued, why=f"lift {body}")
        pb = d.xpos[basket].copy()
        dr.move_tip([pb[0], pb[1], pb[2] + 0.212 + 0.12 + (tip0[2] - p_obj[2])], 15, glued=glued, why=f"carry {body}")
        dr.move_tip([pb[0], pb[1], pb[2] + 0.012 + p_obj[2] + (tip0[2] - p_obj[2]) + 0.06], 10, glued=glued, why=f"lower {body}")
        d.qpos[dr.g] = 0.05
        # the objects already inside take the far corners, the new one a free spot of the floor
        off = [(0.0, 0.0), (0.06, -0.05), (-0.06, 0.05), (0.06, 0.05)][slot]
        dr.set_body(body, [pb[0] + off[0], pb[1] + off[1], pb[2] + 0.012 + p_obj[2] + 0.03], np.array([1.0, 0, 0, 0]))
        dr.move_tip(dr.tip() + [0, 0, 0.15], 8, why=f"retreat from {body}")
        for _ in range(20):
            dr.tick("hold")

    for _ in range(5):
        dr.tick("start")
    for i, body in enumerate(bodies):
        place(body, i)
    dense = [row[3] for row in dr.log]
    mono = all(b >= a - 1e-9 for a, b in zip(dense, dense[1:]))
    print(f"    final stage {dr.scorer.stage_name}, reward {dr.scorer.reward}, dense {dr.scorer.dense:.4f}, monotone {mono}")
    print(f"    objects: { {b: dict(placed=o['placed'], carried=o['carried_time_s'], drops=o['drops']) for b, o in dr.scorer.extra()['objects'].items()} }")
    print("    dense curve (every 6th tick): " + " ".join(f"{v:.2f}" for v in dense[::6]))
    assert mono and dr.scorer.dense >= 1.0 - 1e-9
    return dr


def lid_run(scene_name):
    from lerobot.robots.rebot_b601_sim.tasks import TASKS
    print(f"\n== {scene_name}: twin_l3_close_lid scorer on this scene's headphone case, lid hinge driven")
    dr = Driver(scene_name)
    m, d, s = dr.m, dr.d, dr.s
    dr.scorer = TASKS["twin_l3_close_lid"].scorer(m, d, lambda: 0.0)
    for _ in range(5):
        dr.tick("start")
    wall = s.lid_wall_targets()["headphone_lid_wall_xp"]
    res = dr.arm_to(wall + [0, 0, 0.08]); dr.tick("over the near lid wall")
    res = dr.arm_to(wall - [0, 0, 0.005]); dr.tick("pinch into the wall (IK, may be out of reach here)")
    print(f"    IK at the wall: ok {res['ok']}, pos err {res['pos_err_mm']} mm; reach stage {dr.scorer.stage_name} (midpoint-to-lid {dr.scorer._predicates()['reach']})")
    d.qpos[dr.q] = m.qpos0[dr.q]; d.qpos[dr.g] = m.qpos0[dr.g]
    adr = dr.scorer.qadr
    for frac in np.linspace(0, 1, 30):
        d.qpos[adr] = frac * dr.scorer.closed
        dr.tick("close the lid")
    print(f"    hinge {np.degrees(d.qpos[adr]):.1f} deg of {np.degrees(dr.scorer.closed):.1f}, final stage {dr.scorer.stage_name}, reward {dr.scorer.reward}")
    assert dr.scorer.reward >= 1.0 - 1e-9
    return dr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="twin_l1_tape_basket_right_p2")
    ap.add_argument("--body", default=None)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--sequence", default=None, help="a twin_l4 scene: every object of its task placed in order")
    ap.add_argument("--lid", default=None, help="a full-desk twin scene: drive the headphone lid hinge under the twin_l3_close_lid scorer")
    args = ap.parse_args()
    if args.fold:
        fold_run(args.fold)
    elif args.lid:
        lid_run(args.lid)
    elif args.sequence:
        sequence_run(args.sequence)
    else:
        body = args.body or task_for(args.scene).scorer.BODY
        pick_place_run(args.scene, body, drop_outside=True)
        pick_place_run(args.scene, body, drop_outside=False)
