"""Scripted grasp, lift and insertion of the tube on the twin desk: can the sim hold the tube and put it on the peg?

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/tube_peg/verify_tube_peg.py

Open loop: top pinch on the standing tube, close, lift, carry over the peg (with an xy miss), lower, open, retreat. One row per
case: gripper close command (deg), finger force cap (N per finger; None = the MJCF's), xy miss at the peg (mm). The model sits
in scenes/_check_tube_peg/ (removed at the end); frames of the first case go to renders/.
"""
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "digital_twin/envs"))
sys.path.insert(0, str(HERE))
import build_twin_envs as V1  # noqa: E402
import tube_peg as TP  # noqa: E402

TUBE_XY, HOLDER_XY = (0.30, 0.10), (0.34, -0.08)
GRIPPER_DEG_OPEN, SLIDE_OPEN = -238.8, 0.05
OPEN_DEG = -60.0  # the recordings open to -50..-65 deg around the tube
M_PER_RAD = SLIDE_OPEN / np.deg2rad(-GRIPPER_DEG_OPEN)  # finger travel per motor radian
TIP_Z = 0.005  # finger hull tips above the surface the tube stands on (pad face spans ~2..15 mm)
HIGH_Z = 0.060
DT = 1.0 / (V1.FPS * V1.SUBSTEPS)


def build():
    twin = V1.TwinText()
    twin.bodies.update(tube=TP.tube_body(), tube_holder=TP.holder_body())
    poses = {"tube": (np.array([*TUBE_XY, 0.0005]), V1.quat_z(0.0)), "tube_holder": (np.array([*HOLDER_XY, 0.0005]), V1.quat_z(0.0))}
    twin.pose.update(poses)
    folder = V1.SCENES / "_check_tube_peg"; folder.mkdir(exist_ok=True)
    (folder / "scene.xml").write_text(twin.scene_xml("scene_check_tube_peg", ["tube_holder", "tube"], poses))
    return folder


def run(folder, close_deg, cap, miss_mm, frames=None):
    scene = V1.Scene(folder / "scene.xml", ["tube_holder", "tube"])
    m, d = scene.m, scene.d
    solver = V1.Scene(folder / "scene.xml", ["tube_holder", "tube"])  # IK writes qpos: never on the stepped model
    fingers = [m.actuator("joint_left").id, m.actuator("joint_right").id]
    if cap is not None:
        m.actuator_forcerange[fingers] = (-cap, cap)
    tube, holder = m.body("tube").id, m.body("tube_holder").id
    renderer = mujoco.Renderer(m, 480, 640) if frames is not None else None

    def ik(xy, z):
        r = solver.ik_top_pinch(np.array([xy[0], xy[1], z]), max_it=400)
        assert r["ok"], r
        return np.deg2rad(r["q_deg"])

    def slide(deg):
        return deg / GRIPPER_DEG_OPEN * SLIDE_OPEN

    log = dict(force=[], torque=[], trace=[])
    ge = m.body("gripper_end").id

    def go(q, grip_deg, seconds, tag=None):
        start = d.ctrl.copy(); goal = np.concatenate([q, [slide(grip_deg)] * 2])
        n = int(seconds / DT)
        for i in range(n):
            d.ctrl[:] = start + (goal - start) * min(1.0, (i + 1) / (0.8 * n))
            mujoco.mj_step(m, d)
            f = d.actuator_force[fingers]
            log["force"].append(float(np.abs(f).max())); log["torque"].append(float(-f.sum() * M_PER_RAD / 2))
        if tag is not None:  # tube in the gripper frame (mm) and its tilt from upright (deg)
            in_hand = (d.xpos[tube] - d.xpos[ge]) @ d.xmat[ge].reshape(3, 3)
            log["trace"].append((tag, (in_hand * 1e3).round(1).tolist(), round(float(np.degrees(np.arccos(np.clip(d.xmat[tube].reshape(3, 3)[2, 2], -1, 1)))), 1)))
        if renderer is not None and tag is not None:
            for cam in ("ext", "wrist"):
                renderer.update_scene(d, cam); frames.append((f"{tag} ({cam})", renderer.render().copy()))

    for _ in range(int(0.5 / DT)):
        mujoco.mj_step(m, d)
    z_rest = float(d.xpos[tube][2])
    txy = d.xpos[tube][:2].copy()
    go(ik(txy, HIGH_Z), OPEN_DEG, 2.0, "over the tube")
    go(ik(txy, TIP_Z), OPEN_DEG, 1.0, "at the tube")
    go(ik(txy, TIP_Z), close_deg, 0.7)
    k0 = len(log["torque"])
    go(ik(txy, TIP_Z), close_deg, 0.5, "closed")
    go(ik(txy, HIGH_Z), close_deg, 1.0, "lifted")
    lift = float(d.xpos[tube][2]) - z_rest
    hold_torque = float(np.median(log["torque"][k0:])); hold_force = float(np.median(log["force"][k0:]))
    hold_deg = float(d.qpos[scene.q0 + 6] / SLIDE_OPEN * GRIPPER_DEG_OPEN)
    grip_off = d.xpos[tube][:2] - txy  # tube shift inside the grasp
    pxy = d.xpos[holder][:2] + np.array([miss_mm * 1e-3, 0.0])
    plate_top = 2 * TP.PLATE[2]
    go(ik(pxy, HIGH_Z), close_deg, 2.0)
    go(ik(pxy, HIGH_Z), close_deg, 0.5, "over the peg")
    off_before = float(np.linalg.norm(d.xpos[tube][:2] - d.xpos[holder][:2]))
    for z in np.linspace(HIGH_Z, plate_top + TIP_Z + 0.002, 6)[1:]:
        go(ik(pxy, z), close_deg, 0.4)
    go(ik(pxy, plate_top + TIP_Z + 0.002), close_deg, 0.3, "lowered")
    go(ik(pxy, plate_top + TIP_Z + 0.002), OPEN_DEG, 0.7)
    go(ik(pxy, HIGH_Z), OPEN_DEG, 1.0)
    go(ik(pxy, HIGH_Z), OPEN_DEG, 1.0, "released")
    rel = d.xpos[tube] - d.xpos[holder]
    up = float(d.xmat[tube].reshape(3, 3)[2, 2])
    on_peg = bool(np.linalg.norm(rel[:2]) < TP.TUBE_R_IN and rel[2] < TP.PEG_TOP - 0.005 and up > 0.9)
    if renderer is not None:
        renderer.close()
    return dict(close_deg=close_deg, cap=cap, miss_mm=miss_mm, lift_mm=round(lift * 1e3, 1), hold_deg=round(hold_deg, 1), hold_force_N=round(hold_force, 1),
                hold_torque_Nm=round(hold_torque, 3), grip_shift_mm=round(float(np.linalg.norm(grip_off)) * 1e3, 2), off_before_mm=round(off_before * 1e3, 2),
                tube_rel_mm=(rel * 1e3).round(1).tolist(), upright=round(up, 3), holder_moved_mm=round(float(np.linalg.norm(d.xpos[holder][:2] - np.array(HOLDER_XY))) * 1e3, 1),
                on_peg=on_peg, trace=log["trace"])


def main():
    folder = build()
    frames = []
    cases = [(0.0, None, 0.0), (0.0, 26.0, 0.0), (-14.0, None, 0.0), (-14.0, 26.0, 0.0), (-20.0, 26.0, 0.0)]
    cases += [(-14.0, 26.0, miss) for miss in (1.0, 2.0, 3.0, 4.0)]
    for i, (close_deg, cap, miss) in enumerate(cases):
        print(run(folder, close_deg, cap, miss, frames if i == 3 else None), flush=True)
    out = HERE / "renders"; out.mkdir(exist_ok=True)
    V1.sheet(frames, out / "scripted_insert.png", cols=4, size=(640, 480))
    for tag, img in frames:
        if "wrist" in tag and ("closed" in tag or "lowered" in tag or "released" in tag):
            Image.fromarray(img).save(out / f"wrist_{tag.split(' (')[0].replace(' ', '_')}.png")
    shutil.rmtree(folder)


if __name__ == "__main__":
    main()
