"""Scripted solvability test of scenes/close_lid: IK-driven rim pinch, lift + sweep about the hinge, release past the tipping
point; also a pinch on the loop handle (fixed base). Headless: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/lid_solvability/solve_lid.py [grasp ...]
Writes results.json, <grasp>.mp4 (top camera) and <grasp>_sheet.jpg per grasp into this folder.

Gripper frame (gripper_end body, measured in gripper_geometry.py): +x = tool axis, the finger tips at x = 0 (the body origin), the
pads are the last 2 cm; y = slide axis (gap = 2*slide - 1.6 mm); z = pad width (1 cm at the tip, 4 cm at the heel). Palm 7-16 cm
behind the tips, 18.7 cm wide along y.
"""
import json, sys
from pathlib import Path
import cv2, mujoco, numpy as np

WS = Path(__file__).resolve().parents[4]; OUT = Path(__file__).resolve().parent
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/close_lid/scene.xml"
FPS, SUB = 30, 17
ARM_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_end", "gripper_left", "gripper_right")
# follower soft limits (config_rebot_b601_follower.py), degrees, motor order = joint1..6
LIM_DEG = np.array([(-145, 145), (-170, 1), (-200, 1), (-80, 90), (-90, 90), (-90, 90)], float)
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
OPEN, CLOSED = 0.05, 0.0
RELEASE_RAD = 2.05  # past the ~1.83 rad tipping point
INSERT = 0.015  # tips below the rim top

spec = mujoco.MjSpec.from_file(str(SCENE))
for n in ARM_BODIES: spec.body(n).gravcomp = 1.0
m = spec.compile(); m.opt.timestep = 1.0 / (FPS * SUB); d = mujoco.MjData(m)
q0 = int(m.jnt_qposadr[m.joint("joint1").id]); jids = [m.joint(f"joint{i}").id for i in range(1, 7)]; dof = [int(m.jnt_dofadr[j]) for j in jids]
LO, HI = np.deg2rad(LIM_DEG[:, 0]), np.deg2rad(LIM_DEG[:, 1])
LO = np.maximum(LO, m.jnt_range[jids, 0]); HI = np.minimum(HI, m.jnt_range[jids, 1])
ge = m.body("gripper_end").id; lid = m.body("headphone_lid").id; case = m.body("headphone_case").id
tips = [m.body("gripper_left").id, m.body("gripper_right").id]; grip_bodies = {ge, *tips}
hj = m.joint("lid_hinge").id; hq = int(m.jnt_qposadr[hj])
geom_name = lambda g: m.geom(g).name


def reset():
    mujoco.mj_resetData(m, d); d.ctrl[:] = d.qpos[q0:q0 + 8]; mujoco.mj_forward(m, d)
    for _ in range(int(0.5 / m.opt.timestep)): mujoco.mj_step(m, d)


def rot(axis, a):
    axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def fk(q):
    """Pose of gripper_end for arm joints q (does not touch the physics state permanently)."""
    save = d.qpos.copy(); d.qpos[q0:q0 + 6] = q; mujoco.mj_kinematics(m, d)
    p, R = d.xpos[ge].copy(), d.xmat[ge].reshape(3, 3).copy(); d.qpos[:] = save; mujoco.mj_kinematics(m, d); return p, R


def ik(target_p, slide_dir, tool_pref, q_init, w_tool=0.08, iters=150):
    """Tips at target_p, gripper y axis (slide) along +-slide_dir (hard), gripper x axis (tool) toward tool_pref (soft). DLS with
    the soft joint limits clipped each step. Returns q, position error, slide-axis angle error (deg), tool angle (deg)."""
    q = q_init.copy(); jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); save = d.qpos.copy()
    for _ in range(iters):
        d.qpos[q0:q0 + 6] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p, R = d.xpos[ge], d.xmat[ge].reshape(3, 3); mujoco.mj_jacBody(m, d, jp, jr, ge)
        Jp, Jr = jp[:, dof], jr[:, dof]
        s, t = R[:, 1], R[:, 0]
        sd = slide_dir if np.dot(s, slide_dir) >= 0 else -slide_dir  # the fingers are symmetric: either sign
        # rotation-vector errors (s x sd takes s onto sd); the rotation about the axis itself is free, so project it out
        e_s = np.cross(s, sd); Js = (np.eye(3) - np.outer(s, s)) @ Jr
        e_t = np.cross(t, tool_pref); Jt = (np.eye(3) - np.outer(t, t)) @ Jr
        e = np.concatenate([target_p - p, e_s, w_tool * e_t]); J = np.vstack([Jp, Js, w_tool * Jt])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ e)
        q = np.clip(q + 0.6 * dq, LO, HI)
    d.qpos[q0:q0 + 6] = q; mujoco.mj_kinematics(m, d); p, R = d.xpos[ge].copy(), d.xmat[ge].reshape(3, 3).copy(); d.qpos[:] = save; mujoco.mj_kinematics(m, d)
    s = R[:, 1]; ang_s = np.degrees(np.arccos(min(1, abs(np.dot(s, slide_dir))))); ang_t = np.degrees(np.arccos(np.clip(np.dot(R[:, 0], tool_pref), -1, 1)))
    return q, float(np.linalg.norm(target_p - p)), float(ang_s), float(ang_t)


class Run:
    def __init__(self, name):
        self.name = name; self.r = mujoco.Renderer(m, 480, 640); self.vw = cv2.VideoWriter(str(OUT / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE; self.cam.lookat[:] = [0.33, 0.017, 0.05]; self.cam.distance = 0.55; self.cam.azimuth = 90; self.cam.elevation = -25
        self.keyframes = []; self.log = []; self.t = 0.0; self.grasp_lid = None; self.ctrl_peak = np.full(6, -np.inf); self.ctrl_min = np.full(6, np.inf)
        self.contact_friction = set(); self.ncon_fingers_lid = []

    def frame(self, label):
        self.r.update_scene(d, "ext"); top = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(top, f"{self.name} {label} t={self.t:.1f}s hinge={np.degrees(d.qpos[hq]):.0f}deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return top

    def side(self):
        self.r.update_scene(d, self.cam); return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)

    def tick(self, label):
        self.t += 1.0 / FPS
        top = self.frame(label); self.vw.write(top)
        touching = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in grip_bodies and b2 in (lid, case)) or (b2 in grip_bodies and b1 in (lid, case)):
                touching.add((m.body(b1).name, m.body(b2).name, geom_name(c.geom1), geom_name(c.geom2)))
                if (b1 in tips or b2 in tips): self.contact_friction.add(round(float(c.friction[0]), 3))
        tipmid = (d.xpos[tips[0]] + d.xpos[tips[1]]) / 2
        slip = None
        if self.grasp_lid is not None:
            gw = d.xpos[lid] + d.xmat[lid].reshape(3, 3) @ self.grasp_lid; slip = float(np.linalg.norm(gw - d.xpos[ge]))
        self.log.append(dict(t=round(self.t, 3), phase=label, hinge=float(d.qpos[hq]), tips=d.xpos[ge].round(4).tolist(), slip=slip, slide=float(d.qpos[q0 + 6]),
                             touching=sorted({(a, b) for a, b, _, _ in touching}), n_finger_lid_contacts=sum(1 for a, b, _, _ in touching if "gripper_" in a + b and "lid" in a + b)))
        return top

    def key(self, label):
        self.keyframes.append(np.hstack([self.frame(label), self.side()]))

    def drive(self, q_target, slide, seconds, label):
        """Linear ctrl ramp from the present ctrl to (q_target, slide) over `seconds`, physics at FPS*SUB."""
        c0 = d.ctrl.copy(); c1 = np.concatenate([q_target, [slide, slide]]); n = max(1, int(round(seconds * FPS)))
        for k in range(1, n + 1):
            d.ctrl[:] = c0 + (c1 - c0) * k / n
            self.ctrl_peak = np.maximum(self.ctrl_peak, d.ctrl[:6]); self.ctrl_min = np.minimum(self.ctrl_min, d.ctrl[:6])
            for _ in range(SUB): mujoco.mj_step(m, d)
            self.tick(label)

    def close(self):
        self.vw.release(); self.r.close()
        rows = [np.hstack(self.keyframes[i:i + 2]) if i + 1 < len(self.keyframes) else np.hstack([self.keyframes[i], np.zeros_like(self.keyframes[i])]) for i in range(0, len(self.keyframes), 2)]
        cv2.imwrite(str(OUT / f"{self.name}_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def run_grasp(name, pinch_world, slide_fn, tool_pref_fn, sweep=True, lift_only=False, note=""):
    """pinch_world: tip target at hinge angle 0. slide_fn(theta) / tool_pref_fn(theta): directions in the world at lid angle theta.
    The pinch point rides with the lid: P(theta) = H + R_axis(theta) (P0 - H)."""
    reset(); run = Run(name); res = dict(name=name, note=note, pinch_world=np.round(pinch_world, 4).tolist())
    H = d.xpos[lid].copy(); axis = d.xmat[lid].reshape(3, 3) @ m.jnt_axis[hj]
    P = lambda th: H + rot(axis, th) @ (np.asarray(pinch_world) - H)
    q = d.qpos[q0:q0 + 6].copy(); ikerr = []
    # 1. pre-grasp 6 cm back along the preferred tool axis, fingers open
    tool0 = tool_pref_fn(0.0); pre = P(0) - 0.06 * tool0
    q, e, es, et = ik(pre, slide_fn(0.0), tool0, q); ikerr.append(("pre", e, es, et)); run.drive(q, OPEN, 2.0, "approach"); run.key("approach")
    # 2. descend
    q, e, es, et = ik(P(0), slide_fn(0.0), tool0, q); ikerr.append(("pinch", e, es, et)); run.drive(q, OPEN, 1.5, "descend"); run.key("descend")
    # 3. close
    run.drive(q, CLOSED, 1.0, "close")
    run.grasp_lid = d.xmat[lid].reshape(3, 3).T @ (d.xpos[ge] - d.xpos[lid]); run.key("pinch")
    pinched = set(tips) <= {int(m.geom_bodyid[c.geom2]) if int(m.geom_bodyid[c.geom1]) == lid else int(m.geom_bodyid[c.geom1]) for c in d.contact[:d.ncon] if lid in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))}
    loop_touch = {geom_name(c.geom1) if "loop" in geom_name(c.geom1) else geom_name(c.geom2) for c in d.contact[:d.ncon] if "loop" in geom_name(c.geom1) + geom_name(c.geom2) and (int(m.geom_bodyid[c.geom1]) in tips or int(m.geom_bodyid[c.geom2]) in tips)}
    pinch_geoms = sorted({(geom_name(c.geom1), geom_name(c.geom2)) for c in d.contact[:d.ncon] if {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} & set(tips) and {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} & {lid, case}})
    res.update(pinched_lid=bool(pinched), finger_loop_contacts=sorted(loop_touch), slide_at_pinch=float(d.qpos[q0 + 6]), gap_mm=float(2 * d.qpos[q0 + 6] * 1000 - 1.6), pinch_contacts=pinch_geoms)
    if lift_only:
        q, e, es, et = ik(P(0) + [0, 0, 0.05], slide_fn(0.0), tool0, q); ikerr.append(("lift", e, es, et)); run.drive(q, CLOSED, 2.0, "lift 5 cm"); run.key("lift")
        res.update(hinge_after_lift=float(d.qpos[hq]), tips_after_lift=d.xpos[ge].round(4).tolist(), still_touching_loop=bool({1 for c in d.contact[:d.ncon] if "loop" in geom_name(c.geom1) + geom_name(c.geom2) and (int(m.geom_bodyid[c.geom1]) in tips or int(m.geom_bodyid[c.geom2]) in tips)}))
        run.drive(q, OPEN, 0.5, "release"); run.key("release")
    elif sweep:
        # 4. sweep: the tip point rides the arc, the slide axis follows slide_fn, the tool preference tool_pref_fn
        thetas = np.linspace(0, RELEASE_RAD, 60); worst = 0.0; qs = []
        for th in thetas[1:]:
            q, e, es, et = ik(P(th), slide_fn(th), tool_pref_fn(th), q); worst = max(worst, e); qs.append((th, q.copy(), e, es, et))
        res["sweep_ik_worst_pos_err_m"] = worst; res["sweep_ik"] = [(round(th, 3), round(e, 4), round(es, 1), round(et, 1)) for th, _, e, es, et in qs[::10]]
        run.drive(qs[0][1], CLOSED, 0.3, "sweep")
        for i, (th, qq, *_r) in enumerate(qs):
            run.drive(qq, CLOSED, 6.0 / len(qs), "sweep")
            if i in (len(qs) // 3, 2 * len(qs) // 3): run.key("sweep")
        res.update(hinge_at_release_cmd=float(RELEASE_RAD), hinge_at_release=float(d.qpos[hq]), slip_at_release_m=run.log[-1]["slip"])
        sweep = [x for x in run.log if x["phase"] == "sweep"]; lost = None; streak = 0
        for x in sweep:
            streak = streak + 1 if x["n_finger_lid_contacts"] < 2 else 0
            if streak == 6 and lost is None: lost = x
        res["pinch_lost_at_hinge_rad"] = None if lost is None else round(lost["hinge"], 3)
        res["pinch_held_to_release"] = lost is None
        run.key("before release")
        # 5. release and retreat along -tool, then let it fall
        run.drive(q, OPEN, 0.5, "release")
        _, R = fk(q); back = d.xpos[ge] - 0.10 * R[:, 0]
        q2, e, es, et = ik(back, slide_fn(RELEASE_RAD), tool_pref_fn(RELEASE_RAD), q); ikerr.append(("retreat", e, es, et)); run.drive(q2, OPEN, 1.5, "retreat")
        run.drive(q2, OPEN, 2.0, "settle"); run.key("settled")
        clear = not any(len(x["touching"]) for x in run.log[-5:])
        res.update(hinge_final=float(d.qpos[hq]), hinge_final_deg=float(np.degrees(d.qpos[hq])), closed=bool(d.qpos[hq] > 2.9), gripper_clear=clear)
    # slip trace: max drift of the tip point relative to the lid-fixed grasp point while pinched
    slips = [x["slip"] for x in run.log if x["slip"] is not None and x["phase"] in ("sweep", "lift 5 cm")]
    res.update(slip_max_m=max(slips) if slips else None, ik=[(n, round(e, 4), round(es, 1), round(et, 1)) for n, e, es, et in ikerr],
               ctrl_peak_deg={n: [round(float(np.degrees(a)), 1), round(float(np.degrees(b)), 1)] for n, a, b in zip(NAMES, run.ctrl_min, run.ctrl_peak)},
               limits_deg={n: LIM_DEG[i].tolist() for i, n in enumerate(NAMES)},
               at_limit=[n for i, n in enumerate(NAMES) if run.ctrl_min[i] <= LO[i] + 1e-3 or run.ctrl_peak[i] >= HI[i] - 1e-3],
               finger_contact_friction=sorted(run.contact_friction))
    hinge_trace = [(x["t"], round(x["hinge"], 3), x["slip"] if x["slip"] is None else round(x["slip"], 4), x["n_finger_lid_contacts"]) for x in run.log[::15]]
    res["trace_t_hinge_slip_ncontacts"] = hinge_trace
    run.close(); json.dump(run.log, open(OUT / f"{name}_log.json", "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "trace_t_hinge_slip_ncontacts"}, indent=None, default=str)); return res


reset(); H = d.xpos[lid].copy(); AX = d.xmat[lid].reshape(3, 3) @ m.jnt_axis[hj]; Rl0 = d.xmat[lid].reshape(3, 3)
wall = lambda n: (d.geom_xpos[m.geom(n).id].copy(), d.geom_xmat[m.geom(n).id].reshape(3, 3).copy())
xp_c, _ = wall("headphone_lid_wall_xp"); xn_c, _ = wall("headphone_lid_wall_xn"); far_c, _ = wall("headphone_lid_wall_far")
rim_top = 0.0316
Y = np.array([0, 1.0, 0]); X = np.array([1.0, 0, 0]); DOWN = np.array([0, 0, -1.0])
lid_down = lambda th: rot(AX, th) @ DOWN  # the lid's "into the tray" direction
def fwd_down(deg):  # tool preference: down, tilted toward +x by deg (reaching forward)
    a = np.deg2rad(deg); return np.array([np.sin(a), 0, -np.cos(a)])
def band(th, start_deg, end_deg):
    """Tool preference riding with the lid: lid_down(th) tilted (about the hinge axis) by an angle ramping start -> end over the sweep."""
    phi = np.deg2rad(start_deg + (end_deg - start_deg) * th / RELEASE_RAD)
    return rot(AX, th) @ fwd_down(np.degrees(phi))
def side_tilt(th, deg):  # lid_down(th) tilted sideways toward +y (along the far wall)
    return np.cos(np.deg2rad(deg)) * lid_down(th) + np.sin(np.deg2rad(deg)) * Y
GRASPS = {
    # side rim near the far end (as the T071 demos), slide along the hinge axis (world y), pads flat on the wall.
    # tool fixed in the world (pitch 35 deg forward): the pads leave the wall once the lid has turned ~45 deg past them
    "side_xp": dict(pinch_world=[0.215, xp_c[1], rim_top - INSERT], slide_fn=lambda th: Y, tool_pref_fn=lambda th: fwd_down(35), note="tool pitch fixed 35 deg fwd"),
    "side_xn": dict(pinch_world=[0.215, xn_c[1], rim_top - INSERT], slide_fn=lambda th: Y, tool_pref_fn=lambda th: fwd_down(35), note="tool pitch fixed 35 deg fwd"),
    # tool pitch rides with the lid inside the band that keeps the 2 cm pads on the 2.4 cm wall: +40 deg (fwd) at open -> -40 deg at release
    "side_xp_track": dict(pinch_world=[0.235, xp_c[1], rim_top - 0.018], slide_fn=lambda th: Y, tool_pref_fn=lambda th: band(th, 40, -40), note="tool pitch tracks the lid, +40 -> -40 deg, 18 mm insertion, 2.3 cm from the far corner"),
    "side_xn_track": dict(pinch_world=[0.235, xn_c[1], rim_top - 0.018], slide_fn=lambda th: Y, tool_pref_fn=lambda th: band(th, 40, -40), note="mirror of side_xp_track"),
    "side_xp_mid": dict(pinch_world=[0.265, xp_c[1], rim_top - 0.018], slide_fn=lambda th: Y, tool_pref_fn=lambda th: band(th, 40, -40), note="pinch at the middle of the side wall (7.5 cm lever)"),
    # tool fixed relative to the lid (full 117 deg pitch change)
    "side_xp_lidfixed": dict(pinch_world=[0.215, xp_c[1], rim_top - INSERT], slide_fn=lambda th: Y, tool_pref_fn=lambda th: rot(AX, th) @ fwd_down(35), note="tool fixed to the lid"),
    # far rim (nearest the robot): slide across the wall = lid x, which rotates with the lid; the tool must stay normal to lid x
    "far_rim": dict(pinch_world=[far_c[0], 0.0174, rim_top - INSERT], slide_fn=lambda th: rot(AX, th) @ X, tool_pref_fn=lambda th: rot(AX, th) @ fwd_down(35), note="tool fixed to the lid"),
    "far_rim_tilt": dict(pinch_world=[far_c[0], 0.0174, rim_top - 0.018], slide_fn=lambda th: rot(AX, th) @ X, tool_pref_fn=lambda th: side_tilt(th, 60 * th / RELEASE_RAD), note="tool tilts sideways along the wall 0 -> 60 deg"),
    "far_rim_sideways": dict(pinch_world=[far_c[0], 0.0174, rim_top - INSERT], slide_fn=lambda th: rot(AX, th) @ X, tool_pref_fn=lambda th: -Y, note="tool along the hinge axis"),
    # the loop handle on the fixed base (far end): pinch, lift 5 cm
    "loop_apex": dict(pinch_world=[0.5285, 0.0174, 0.0309], slide_fn=lambda th: X, tool_pref_fn=lambda th: DOWN, lift_only=True, note="apex, slide across x, tool down"),
    "loop_leg": dict(pinch_world=[0.5136, -0.0056, 0.0265], slide_fn=lambda th: Y, tool_pref_fn=lambda th: fwd_down(30), lift_only=True, note="leg loop_0 midpoint, slide y, tool 30 deg fwd"),
    "loop_apex_fwd": dict(pinch_world=[0.5285, 0.0174, 0.0309], slide_fn=lambda th: np.array([np.cos(np.deg2rad(30)), 0, np.sin(np.deg2rad(30))]), tool_pref_fn=lambda th: fwd_down(30), lift_only=True, note="apex, tool 30 deg fwd, slide tilted with it"),
}
names = sys.argv[1:] or list(GRASPS)
results = {n: run_grasp(n, **GRASPS[n]) for n in names}
prev = json.load(open(OUT / "results.json")) if (OUT / "results.json").exists() else {}
prev.update(results); json.dump(prev, open(OUT / "results.json", "w"), indent=1, default=str)
