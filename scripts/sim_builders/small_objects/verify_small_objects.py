"""Scripted pick and place of the small objects on the twin desk: can the sim gripper take a die, a plug, a screwdriver and a
nut off the table and put each where it belongs (cup, its compartment, the case, the pin)?

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/small_objects/verify_small_objects.py

Top pinch with the fingers closing across the object, close, lift, carry over the place, correct the tip by the object's
measured offset once, lower, open, retreat. The position-controlled arm hangs below its command (6 mm low and 4 mm short at
0.4 m), which is more than a nut is wide: over the object and over the place the script measures where the tip is and shifts
the commands by the difference (`fix`). One row per move. `main` runs one scene with the four pairs (model in
scenes/_check_small_objects/, removed at the end), then the nut alone with the tip off at the pin, off at the grasp, and at other finger force
caps; frames of the first run go to renders/.
"""
import shutil
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tube_peg"))
sys.path.insert(0, str(HERE))
import small_objects as SO  # noqa: E402
import verify_tube_peg as VT  # noqa: E402

V1 = VT.V1
CAP_N = 10.0
CLOSE_DEG = 0.0
CARRY_Z = 0.060
VIA_M = 0.07  # a place higher than CARRY_Z (the cup) is reached through a point this far before it, already at its height
# per object: tip height at the grasp (the pads' lower edge is 0.4 mm above the tip point, so the thin objects are taken with the
# pads 1.4 mm off the table), fingers close along the body's y (None = any way), tip height at the release and over the place
# (above the place's origin)
KINDS = dict(
    die=dict(grasp_z=0.004, across=True, release_z=SO.CUP_H + 0.012, carry_z=SO.CUP_H + 0.025),
    plug=dict(grasp_z=0.001, across=True, release_z=SO.BOX[2] + 0.006, carry_z=CARRY_Z),
    screwdriver=dict(grasp_z=0.001, across=True, release_z=SO.CASE[2] + 0.004, carry_z=CARRY_Z),
    nut=dict(grasp_z=0.001, across=None, release_z=2 * SO.HOLDER_PLATE[2] + 0.005, carry_z=CARRY_Z),
)


def yaw_of(d, body):
    return float(np.arctan2(d.xmat[body][3], d.xmat[body][0]))


class Pinch:
    """IK of the top pinch on a copy of the scene (never the stepped model): tips at a point, approach down, and (optionally)
    the fingers' closing direction along a given horizontal direction."""

    def __init__(self, folder, bodies):
        self.s = V1.Scene(folder / "scene.xml", bodies)
        m, d = self.s.m, self.s.d
        ge = m.body("gripper_end").id
        slide = (d.xpos[m.body("gripper_right").id] - d.xpos[m.body("gripper_left").id]) @ d.xmat[ge].reshape(3, 3)
        self.slide = slide / np.linalg.norm(slide)
        self.last = None

    def __call__(self, xy, z, closing=None):
        s = self.s
        m, d = s.m, s.d
        ge = m.body("gripper_end").id
        target = np.array([xy[0], xy[1], z]); down = np.array([0.0, 0.0, -1.0])
        lo, hi = s.limits; q = slice(s.q0, s.q0 + 6)
        yaw = np.arctan2(xy[1], xy[0])
        starts = ([self.last] if self.last is not None else []) + [np.array(v) for v in ((0, -1.0, -1.0, 0.5, 0, 0), (yaw, -0.8, -1.5, 0.8, 0, 0), (yaw, -1.5, -0.5, 0.0, 0, 0), (yaw, -0.3, -2.0, 0.5, 0, 0))]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for start in starts:
            d.qpos[q] = np.clip(start, lo, hi); d.qpos[s.q0 + 6:s.q0 + 8] = 0.05
            for _ in range(400):
                mujoco.mj_forward(m, d)
                R = d.xmat[ge].reshape(3, 3); a = R @ s.approach; tip = d.xpos[ge] + s.tip_len * a
                mujoco.mj_jac(m, d, jp, jr, tip, ge)
                turn = np.cross(a, down) + (np.cross(R @ self.slide, closing) if closing is not None else 0.0)
                e = np.concatenate([target - tip, 0.05 * turn])
                J = np.vstack([jp[:, s.dofs], 0.05 * jr[:, s.dofs]])
                d.qpos[q] = np.clip(d.qpos[q] + 0.5 * np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ e), lo, hi)
            mujoco.mj_forward(m, d)
            R = d.xmat[ge].reshape(3, 3); a = R @ s.approach
            ok = np.linalg.norm(target - d.xpos[ge] - s.tip_len * a) < 0.002 and a @ down > np.cos(np.radians(5)) and (closing is None or (R @ self.slide) @ closing > np.cos(np.radians(5)))
            if ok:
                self.last = d.qpos[q].copy()
                return self.last
        return None


def build(pairs):
    """Check scene: `pairs` = [(object body, generator, (x, y, yaw deg)), (place body, generator, (x, y, yaw deg))] flattened."""
    twin = V1.TwinText()
    poses = {}
    for name, block, (x, y, yaw) in pairs:
        twin.bodies[name] = block
        poses[name] = (np.array([x, y, 0.0005 + SO.REST_Z.get(SO.kind_of(name), 0.0)]), V1.quat_z(np.radians(yaw)))
    twin.pose.update(poses)
    folder = V1.SCENES / "_check_small_objects"; folder.mkdir(exist_ok=True)
    (folder / "scene.xml").write_text(twin.scene_xml("scene_check_small_objects", list(poses), poses))
    return folder


def run(folder, moves, cap=CAP_N, miss_mm=0.0, grasp_miss_mm=0.0, frames=None, poll=None, video=None):
    """`moves` = [dict(body, place, at=(x, y) of the object's origin in the place body's frame, yaw=the object's x axis in that
    frame (deg; None = as it comes), up=extra release height (a nut onto the nut already on the pin))], done in order in one simulation. `miss_mm` / `grasp_miss_mm` = the tip is this far off along x at the place / at the grasp. `poll(model, data)` returns a callable run once per control tick. `video` = mp4 path: the whole run, top camera | wrist camera."""
    bodies = [b for mv in moves for b in (mv["body"], mv["place"])]
    scene = V1.Scene(folder / "scene.xml", bodies)
    m, d = scene.m, scene.d
    pinch = Pinch(folder, bodies)
    fingers = [m.actuator("joint_left").id, m.actuator("joint_right").id]
    m.actuator_forcerange[fingers] = (-cap, cap)
    renderer = mujoco.Renderer(m, 480, 640) if frames is not None or video is not None else None
    writer = imageio.get_writer(video, fps=V1.FPS) if video is not None else None
    tick = poll(m, d) if poll is not None else None
    ge = m.body("gripper_end").id
    force = []

    def go(q, grip_deg, seconds, tag=None):
        start = d.ctrl.copy(); goal = np.concatenate([q, [grip_deg / VT.GRIPPER_DEG_OPEN * VT.SLIDE_OPEN] * 2])
        n = int(seconds / VT.DT)
        for i in range(n):
            d.ctrl[:] = start + (goal - start) * min(1.0, (i + 1) / (0.8 * n))
            mujoco.mj_step(m, d)
            if tick is not None and i % V1.SUBSTEPS == 0:
                tick()
            force.append(float(np.abs(d.actuator_force[fingers]).max()))
            if writer is not None and i % V1.SUBSTEPS == 0:
                shots = []
                for cam in ("ext", "wrist"):
                    renderer.update_scene(d, cam); shots.append(renderer.render().copy())
                writer.append_data(np.hstack(shots))
        if frames is not None and tag is not None:
            for cam in ("ext", "wrist"):
                renderer.update_scene(d, cam); frames.append((f"{tag} ({cam})", renderer.render().copy()))

    bias = np.zeros(3)

    def reach(xy, z, closing, grip_deg, seconds, tag=None):
        go(pinch(np.asarray(xy) + bias[:2], z + bias[2], closing), grip_deg, seconds, tag)

    def fix(xy, z, closing, grip_deg):
        """Shift the commands by what the tip is off its target now, and go there."""
        bias[:] += np.array([*xy, z]) - d.xpos[ge] - scene.tip_len * (d.xmat[ge].reshape(3, 3) @ scene.approach)
        reach(xy, z, closing, grip_deg, 0.6)

    for _ in range(int(0.5 / VT.DT)):
        mujoco.mj_step(m, d)
    rows = []
    for mv in moves:
        body, place = m.body(mv["body"]).id, m.body(mv["place"]).id
        kind = KINDS[SO.kind_of(mv["body"])]
        xy0, z0 = d.xpos[body][:2] + np.array([grasp_miss_mm * 1e-3, 0.0]), float(d.xpos[body][2])
        place_xy0 = d.xpos[place][:2].copy()
        Rp = d.xmat[place].reshape(3, 3)
        goal_xy = (d.xpos[place] + Rp @ np.array([*mv["at"], 0.0]))[:2] + np.array([miss_mm * 1e-3, 0.0])
        release_z = float(d.xpos[place][2]) + kind["release_z"] + mv.get("up", 0.0)
        back = xy0 - goal_xy
        via_xy = goal_xy + back * min(1.0, VIA_M / np.linalg.norm(back)) if kind["carry_z"] > CARRY_Z else goal_xy
        plan = None
        for sign in (1.0, -1.0):  # the fingers may close either way round; take the one the wrist can also place with
            a0 = yaw_of(d, body) + np.pi / 2
            close0 = sign * np.array([np.cos(a0), np.sin(a0), 0.0]) if kind["across"] else None
            a1 = yaw_of(d, place) + np.radians(mv["yaw"]) + np.pi / 2 if mv.get("yaw") is not None else None
            close1 = sign * np.array([np.cos(a1), np.sin(a1), 0.0]) if a1 is not None else None
            pinch.last = None
            qs = [pinch(xy0, CARRY_Z, close0), pinch(xy0, kind["grasp_z"], close0), pinch(via_xy, kind["carry_z"], close1), pinch(goal_xy, kind["carry_z"], close1), pinch(goal_xy, release_z, close1)]
            if all(q is not None for q in qs):
                plan = (close0, close1)
                break
        assert plan is not None, f"no top pinch for {mv}"
        close0, close1 = plan
        pinch.last = None
        bias[:] = 0.0
        reach(xy0, CARRY_Z, close0, VT.OPEN_DEG, 2.0, f"over {mv['body']}")
        fix(xy0, CARRY_Z, close0, VT.OPEN_DEG)
        reach(xy0, kind["grasp_z"], close0, VT.OPEN_DEG, 1.0)
        reach(xy0, kind["grasp_z"], close0, CLOSE_DEG, 0.7)
        k0 = len(force)
        reach(xy0, kind["grasp_z"], close0, CLOSE_DEG, 0.5, f"closed on {mv['body']}")
        reach(xy0, CARRY_Z, close0, CLOSE_DEG, 1.0)
        lift = float(d.xpos[body][2]) - z0
        hold_force = float(np.median(force[k0:]))
        hold_deg = float(d.qpos[scene.q0 + 6] / VT.SLIDE_OPEN * VT.GRIPPER_DEG_OPEN)
        reach(via_xy, kind["carry_z"], close1, CLOSE_DEG, 2.0)
        reach(goal_xy, kind["carry_z"], close1, CLOSE_DEG, 1.5)
        fix(goal_xy, kind["carry_z"], close1, CLOSE_DEG)
        off = d.xpos[body][:2] - goal_xy  # where the object sits in the grasp: correct the tip once
        tip_xy = goal_xy - off
        reach(tip_xy, kind["carry_z"], close1, CLOSE_DEG, 0.7, f"{mv['body']} over {mv['place']}")
        for z in np.linspace(kind["carry_z"], release_z, 5)[1:]:
            reach(tip_xy, z, close1, CLOSE_DEG, 0.4)
        reach(tip_xy, release_z, close1, CLOSE_DEG, 0.3, f"{mv['body']} lowered")
        reach(tip_xy, release_z, close1, VT.OPEN_DEG, 0.7)
        reach(tip_xy, kind["carry_z"], close1, VT.OPEN_DEG, 1.0)
        reach(tip_xy, kind["carry_z"], close1, VT.OPEN_DEG, 1.0, f"{mv['body']} released")
        rel = d.xmat[place].reshape(3, 3).T @ (d.xpos[body] - d.xpos[place])
        rows.append(dict(body=mv["body"], place=mv["place"], cap=cap, miss_mm=miss_mm, grasp_miss_mm=grasp_miss_mm, lift_mm=round(lift * 1e3, 1), hold_deg=round(hold_deg, 1), hold_force_N=round(hold_force, 1),
                         in_grasp_mm=(off * 1e3).round(1).tolist(), rel_mm=(rel * 1e3).round(1).tolist(), goal_mm=[round(v * 1e3, 1) for v in mv["at"]],
                         yaw_in_place_deg=round(float(np.degrees(yaw_of(d, body) - yaw_of(d, place))), 1),
                         place_moved_mm=round(float(np.linalg.norm(d.xpos[place][:2] - place_xy0)) * 1e3, 1)))
    if renderer is not None:
        renderer.close()
    if writer is not None:
        writer.close()
    return rows


CHECK = [("die", SO.die_body(), (0.40, 0.20, 20)), ("dice_cup", SO.cup_body(), (0.30, 0.28, 0)),
         ("plug_yellow", SO.plug_body("plug_yellow", "yellow"), (0.42, 0.06, 40)), ("plug_box", SO.plug_box_body(), (0.30, 0.10, 0)),
         ("screwdriver", SO.screwdriver_body(), (0.40, -0.08, 70)), ("screwdriver_case", SO.case_body(), (0.36, -0.20, -90)),
         ("nut", SO.nut_body(), (0.30, -0.06, 0)), ("nut_holder", SO.nut_holder_body(), (0.24, -0.14, 0))]
PLUG_AT = {colour: ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 - 0.006) for colour, (lo, hi) in SO._box_compartments().items()}  # in front of the plug already there
DRIVER_AT = (0.010, -0.022)
MOVES = [dict(body="die", place="dice_cup", at=(0.0, 0.0)), dict(body="plug_yellow", place="plug_box", at=PLUG_AT["yellow"], yaw=0.0),
         dict(body="screwdriver", place="screwdriver_case", at=DRIVER_AT, yaw=90.0), dict(body="nut", place="nut_holder", at=(0.0, 0.0))]


def main():
    folder = build(CHECK)
    print("settle", V1.settle(folder, [name for name, _, _ in CHECK]), flush=True)
    frames = []
    out = HERE / "renders"; out.mkdir(exist_ok=True)
    for row in run(folder, MOVES, frames=frames, video=out / "scripted_moves.mp4"):
        print(row, flush=True)
    nut = [MOVES[3]]
    for kw in [dict(miss_mm=v) for v in (0.5, 1.0, 1.5, 2.0, 3.0)] + [dict(grasp_miss_mm=v) for v in (2.0, 4.0, 6.0, -2.0, -4.0, -6.0)] + [dict(cap=v) for v in (5.0, 20.0, 36.0)]:
        print(run(folder, nut, **kw)[0], flush=True)
    V1.sheet(frames, out / "scripted_moves.png", cols=4, size=(640, 480))
    shutil.rmtree(folder)


if __name__ == "__main__":
    main()
