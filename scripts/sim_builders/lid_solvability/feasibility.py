"""Kinematic feasibility of holding a rim pinch through the closing sweep, within the follower's soft joint limits.
For each pinch family and hinge angle theta: multi-start IK for a gripper pose with the tips on the pinch point (riding with the lid),
the slide axis across the wall (hard) and the tool axis inside the band that keeps the 2 cm pads on the 2.4 cm wall. Reports the
largest theta with a solution (tipping point 1.83 rad) and the joint values there. No physics. Writes feasibility.json."""
import json
from pathlib import Path
import mujoco, numpy as np

WS = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parent
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml"
LIM_DEG = np.array([(-145, 145), (-170, 1), (-200, 1), (-80, 90), (-90, 90), (-90, 90)], float)
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
m = mujoco.MjModel.from_xml_path(str(SCENE)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
q0 = int(m.jnt_qposadr[m.joint("joint1").id]); jids = [m.joint(f"joint{i}").id for i in range(1, 7)]; dof = [int(m.jnt_dofadr[j]) for j in jids]
LO = np.maximum(np.deg2rad(LIM_DEG[:, 0]), m.jnt_range[jids, 0]); HI = np.minimum(np.deg2rad(LIM_DEG[:, 1]), m.jnt_range[jids, 1])
ge = m.body("gripper_end").id; lid = m.body("headphone_lid").id; hj = m.joint("lid_hinge").id
H = d.xpos[lid].copy(); AX = d.xmat[lid].reshape(3, 3) @ m.jnt_axis[hj]
rng = np.random.default_rng(0)


def rot(axis, a):
    axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def pose(q):
    d.qpos[q0:q0 + 6] = q; mujoco.mj_kinematics(m, d); return d.xpos[ge].copy(), d.xmat[ge].reshape(3, 3).copy()


def ik(target_p, slide_dir, tool_dir, q_init, w_tool=1.0, iters=120):
    q = q_init.copy(); jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[q0:q0 + 6] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p, R = d.xpos[ge], d.xmat[ge].reshape(3, 3); mujoco.mj_jacBody(m, d, jp, jr, ge); Jp, Jr = jp[:, dof], jr[:, dof]
        s, t = R[:, 1], R[:, 0]; sd = slide_dir if np.dot(s, slide_dir) >= 0 else -slide_dir
        e = np.concatenate([target_p - p, np.cross(s, sd), w_tool * np.cross(t, tool_dir)])
        J = np.vstack([Jp, (np.eye(3) - np.outer(s, s)) @ Jr, w_tool * (np.eye(3) - np.outer(t, t)) @ Jr])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ e); q = np.clip(q + 0.5 * dq, LO, HI)
    p, R = pose(q)
    return q, float(np.linalg.norm(target_p - p)), float(np.degrees(np.arccos(min(1, abs(np.dot(R[:, 1], slide_dir)))))), float(np.degrees(np.arccos(np.clip(np.dot(R[:, 0], tool_dir), -1, 1))))


def solve(target_p, slide_dir, tool_dirs, q_warm, band_deg, starts=10):
    """Any tool direction of tool_dirs (band centre candidates); accept pos err < 3 mm, slide err < 3 deg, tool within band_deg of the candidate."""
    best = None
    for tool_dir in tool_dirs:
        inits = [q_warm] + [LO + rng.random(6) * (HI - LO) for _ in range(starts)]
        for qi in inits:
            q, ep, es, et = ik(target_p, slide_dir, tool_dir, qi)
            ok = ep < 0.003 and es < 3 and et <= band_deg
            score = (0 if ok else 1, ep + 0.01 * es / 3 + 0.01 * max(0, et - band_deg) / 10)
            if best is None or score < best[0]: best = (score, q, ep, es, et, ok)
            if ok and qi is q_warm: break  # the warm start suffices
        if best[5]: break
    return best


Y = np.array([0, 1.0, 0]); X = np.array([1.0, 0, 0]); DOWN = np.array([0, 0, -1.0])
lid_down = lambda th: rot(AX, th) @ DOWN
lid_x = lambda th: rot(AX, th) @ X
INS = 0.018; BAND = 60.0  # 18 mm insertion of the 2 cm pads: tool within ~60 deg of the lid's down keeps the pads on the wall
rim = 0.0316
xp_y = d.geom_xpos[m.geom("headphone_lid_wall_xp").id][1]; xn_y = d.geom_xpos[m.geom("headphone_lid_wall_xn").id][1]; far_x = d.geom_xpos[m.geom("headphone_lid_wall_far").id][0]


def fam_side(y, x):
    """Side wall: slide = y; tool in the xz plane within BAND of lid_down(th): candidates = lid_down tilted by -60..60 about the hinge axis."""
    return dict(P0=np.array([x, y, rim - INS]), slide=lambda th: Y, tools=lambda th: [rot(AX, th + np.deg2rad(a)) @ DOWN for a in (0, 30, -30, 55, -55)], band=8.0)


def fam_far(y):
    """Far wall: slide = lid_x(th); tool in span(y, lid_down): candidates tilted sideways by psi."""
    return dict(P0=np.array([far_x, y, rim - INS]), slide=lambda th: lid_x(th), tools=lambda th: [np.cos(np.deg2rad(a)) * lid_down(th) + np.sin(np.deg2rad(a)) * Y for a in (0, 30, -30, 55, -55, 75, -75)], band=8.0)


FAMILIES = {
    "side_xp_far_end": fam_side(xp_y, 0.235), "side_xn_far_end": fam_side(xn_y, 0.235), "side_xp_mid": fam_side(xp_y, 0.265), "side_xp_near_hinge": fam_side(xp_y, 0.30),
    "far_rim_centre": fam_far(0.0174), "far_rim_left": fam_far(0.07), "far_rim_right": fam_far(-0.035),
}
thetas = np.round(np.arange(0, 2.101, 0.1), 2)
report = {}
for name, F in FAMILIES.items():
    q_warm = d.qpos[q0:q0 + 6].copy(); rows = []; max_ok = None; at_max = None
    for th in thetas:
        P = H + rot(AX, th) @ (F["P0"] - H)
        score, q, ep, es, et, ok = solve(P, F["slide"](th), F["tools"](th), q_warm, F["band"])
        rows.append((float(th), bool(ok), round(ep, 4), round(es, 1), round(et, 1), np.degrees(q).round(0).tolist()))
        if ok: q_warm = q; max_ok = float(th); at_max = dict(zip(NAMES, np.degrees(q).round(1).tolist()))
        else: break
    report[name] = dict(max_theta_ok=max_ok, past_tipping=(max_ok or 0) >= 1.85, joints_deg_at_max=at_max, rows=rows)
    print(f"{name:20s} max theta with a pinch pose inside the limits: {max_ok}  (tipping 1.83)  joints at max: {at_max}")
    print("   fail row:", rows[-1] if not rows[-1][1] else "none")
json.dump(report, open(OUT / "feasibility.json", "w"), indent=1)
