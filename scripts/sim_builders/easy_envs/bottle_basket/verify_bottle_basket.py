"""Compile + render + physics checks for scenes/bottle_basket.
MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/easy_envs/bottle_basket/verify_bottle_basket.py
"""
import json
from pathlib import Path
import cv2, mujoco, numpy as np

WS = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1]
SCENE = WS / "lerobot/src/lerobot/robots/rebot_b601_sim/scenes/bottle_basket/scene.xml"
FPS, SUBSTEPS = 30, 17
m = mujoco.MjModel.from_xml_path(str(SCENE)); m.opt.timestep = 1.0 / (FPS * SUBSTEPS); d = mujoco.MjData(m)
q0 = int(m.jnt_qposadr[m.joint("joint1").id])
def reset():
    mujoco.mj_resetData(m, d); d.ctrl[:] = d.qpos[q0:q0 + 8]; mujoco.mj_forward(m, d)
reset()
report = {}
bottle, basket = m.body("bottle").id, m.body("basket").id
print("compiled: nbody", m.nbody, "ngeom", m.ngeom, "bottle", d.xpos[bottle].round(4), "basket", d.xpos[basket].round(4))

# ---- renders at the park pose ----
r = mujoco.Renderer(m, 480, 640)
def render(cam, seg=False):
    if seg: r.enable_segmentation_rendering()
    r.update_scene(d, cam); im = r.render()
    if seg: r.disable_segmentation_rendering()
    return im
bg = cv2.cvtColor(cv2.imread(str(SCENE.parent / "background_top.png")), cv2.COLOR_BGR2RGB)
# the deployed composite (config_rl_sim.yaml top camera): colour gain/offset on the render, the sim table rendered with its
# own gain, every other world geom replaced by the real background
GAIN, OFF, STATIC, STATIC_GAIN = np.array([1.222, 1.278, 1.165]), np.array([8.0, 8.5, 28.1]), ["table", "table_top"], np.array([1.054, 1.231, 1.403])
rgb = render("ext"); geom = render("ext", seg=True)[:, :, 0]
rgb = np.clip(rgb * GAIN + OFF, 0, 255).astype(np.uint8)
static = (geom >= 0) & np.isin(np.clip(geom, 0, None), [m.geom(n).id for n in STATIC])
rgb[static] = np.clip(rgb[static] * STATIC_GAIN, 0, 255).astype(np.uint8)
mask = ((geom >= 0) & ((m.geom_bodyid[np.clip(geom, 0, None)] != 0) | static)).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32); mask = cv2.GaussianBlur(mask, (0, 0), 2)
comp = (mask[:, :, None] * rgb + (1 - mask[:, :, None]) * bg).astype(np.uint8)
# basket colour on the deployed path vs the real basket wall in the 10-21 top frame (ep0_top_t0.png, wall region x 545-600, y 40-110)
real = cv2.cvtColor(cv2.imread(str(OUT / "basket_reference/ep0_top_t0.png")), cv2.COLOR_BGR2RGB)[40:110, 545:600].reshape(-1, 3).mean(0)
wall = (geom >= 0) & (m.geom_bodyid[np.clip(geom, 0, None)] == basket) & np.isin(np.clip(geom, 0, None), [m.geom("basket_wall_outer_visual").id, m.geom("basket_rim_visual").id])
sim = comp[wall].reshape(-1, 3).mean(0)
report["basket_colour_top_cam"] = dict(real_rgb=real.round(1).tolist(), sim_rgb=sim.round(1).tolist(), suggested_extra_gain=(real / sim).round(3).tolist())
print("basket colour:", json.dumps(report["basket_colour_top_cam"]))
cv2.imwrite(str(OUT / "bottle_basket_top.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT / "bottle_basket_top_raw.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT / "bottle_basket_wrist.png"), cv2.cvtColor(render("wrist"), cv2.COLOR_RGB2BGR))
# close-up of the basket alone (free camera), two angles
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = d.xpos[basket] + [0, 0, 0.10]
views = []
for az, el, dist in [(-150, -35, 0.62), (-60, -60, 0.55)]:
    cam.azimuth, cam.elevation, cam.distance = az, el, dist; r.update_scene(d, cam); views.append(r.render())
# wrist colour space check: outer-wall pixels of the first close-up vs the real wrist patch the texture came from
r.enable_segmentation_rendering(); cam.azimuth, cam.elevation, cam.distance = -150, -35, 0.62; r.update_scene(d, cam); seg = r.render()[:, :, 0]; r.disable_segmentation_rendering()
wall_px = views[0][(seg >= 0) & np.isin(np.clip(seg, 0, None), [m.geom("basket_wall_outer_visual").id])].reshape(-1, 3).mean(0)
patch_rgb = np.array(json.load(open(SCENE.parent.parent / "assets/photoreal/basket/texture_sources.json"))["weave"]["mean_bgr"])[::-1]
report["basket_colour_wrist_space"] = dict(real_wrist_patch_rgb=patch_rgb.round(1).tolist(), sim_wall_rgb=wall_px.round(1).tolist(), suggested_extra_gain=(patch_rgb / wall_px).round(3).tolist())
print("wrist-space colour:", json.dumps(report["basket_colour_wrist_space"]))
cv2.imwrite(str(OUT / "bottle_basket_closeup.png"), cv2.cvtColor(np.hstack(views), cv2.COLOR_RGB2BGR))

# ---- 2 s of physics at rest ----
x0 = d.xpos[bottle].copy(); qb0 = d.xquat[bottle].copy(); xb0 = d.xpos[basket].copy()
worst = 0.0
for _ in range(2 * FPS * SUBSTEPS):
    mujoco.mj_step(m, d)
    for c in d.contact[:d.ncon]:
        if bottle in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) or basket in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            worst = min(worst, c.dist)
moved = float(np.linalg.norm(d.xpos[bottle] - x0)); tilt = float(np.degrees(2 * np.arccos(min(1, abs(np.dot(d.xquat[bottle], qb0))))))
report["rest_2s"] = dict(bottle_moved_m=moved, bottle_tilt_deg=tilt, basket_moved_m=float(np.linalg.norm(d.xpos[basket] - xb0)), worst_penetration_m=float(worst),
                         contacts=sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]}))
print("rest 2 s:", json.dumps(report["rest_2s"]))

# ---- drop test: bottle released 0.26 m up above the basket centre ----
reset(); adr = m.jnt_qposadr[m.body_jntadr[bottle]]
d.qpos[adr:adr + 3] = d.xpos[basket] + [0.03, -0.04, 0.26]; mujoco.mj_forward(m, d)
for _ in range(3 * FPS * SUBSTEPS): mujoco.mj_step(m, d)
rel = d.xpos[bottle] - d.xpos[basket]
inside = abs(rel[0]) < 0.133 and abs(rel[1]) < 0.108 and 0.05 < rel[2] < 0.07
report["drop_test"] = dict(bottle_rel_basket=rel.round(4).tolist(), inside_on_floor=bool(inside), basket_moved_m=float(np.linalg.norm(d.xpos[basket] - xb0)),
                           contacts=sorted({(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]}))
print("drop:", json.dumps(report["drop_test"]))
cam.azimuth, cam.elevation, cam.distance = -60, -60, 0.55; r.update_scene(d, cam)
cv2.imwrite(str(OUT / "bottle_basket_drop_test.png"), cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))

# ---- gripper opening vs bottle diameter (finger pad gap at the tips) ----
reset()
def finger_pts(body):
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == m.body(body).id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] | m.geom_conaffinity[g]):
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(pts)
ge = m.body("gripper_end").id
def frame_pts(s):
    d.qpos[q0 + 6:q0 + 8] = s; mujoco.mj_forward(m, d); Rg = d.xmat[ge].reshape(3, 3); og = d.xpos[ge]
    return (finger_pts("gripper_left") - og) @ Rg, (finger_pts("gripper_right") - og) @ Rg, (d.xpos[m.body("gripper_right").id] - d.xpos[m.body("gripper_left").id]) @ Rg
L, Rr, slide = frame_pts(0.05); ax = int(np.argmax(np.abs(slide))); sgn = np.sign(slide[ax])
ext = np.ptp(np.vstack([L, Rr]), axis=0); approach = int(np.argmax(np.where(np.arange(3) == ax, -1, ext)))  # the long finger axis
gaps = {}
for s in [0.0, 0.025, 0.05]:
    L, Rr, _ = frame_pts(s)
    tipL = L[L[:, approach] > np.percentile(L[:, approach], 85)]; tipR = Rr[Rr[:, approach] > np.percentile(Rr[:, approach], 85)]
    gaps[f"slide_{s}"] = float((tipR[:, ax].min() - tipL[:, ax].max()) if sgn > 0 else (tipL[:, ax].min() - tipR[:, ax].max()))
report["gripper"] = dict(tip_gap_m=gaps, bottle_diameter_m=0.055, graspable=bool(gaps["slide_0.05"] > 0.055 + 0.005))
print("gripper:", json.dumps(report["gripper"]))

# ---- IK reach (position only): can gripper_end sit over the bottle and over the basket interior ----
def ik(target, max_it=300):
    d.qpos[q0:q0 + 6] = [0, -1.0, -1.0, 0.5, 0, 0]; d.qpos[q0 + 6:q0 + 8] = 0.05
    ge = m.body("gripper_end").id; jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    jids = [m.joint(f"joint{i}").id for i in range(1, 7)]; dof = [m.jnt_dofadr[j] for j in jids]
    for _ in range(max_it):
        mujoco.mj_forward(m, d); mujoco.mj_jacBody(m, d, jp, jr, ge)
        e = target - d.xpos[ge]; J = jp[:, dof]
        dq = np.linalg.solve(J.T @ J + 1e-5 * np.eye(6), J.T @ e)
        d.qpos[q0:q0 + 6] = np.clip(d.qpos[q0:q0 + 6] + 0.5 * dq, m.jnt_range[jids, 0], m.jnt_range[jids, 1])
    mujoco.mj_forward(m, d)
    return dict(pos_err_m=float(np.linalg.norm(target - d.xpos[ge])), q_deg=np.degrees(d.qpos[q0:q0 + 6]).round(1).tolist())
reset(); bp = d.xpos[bottle].copy(); kp = d.xpos[basket].copy()
report["ik"] = {"gripper_end_over_bottle_z0.15": ik(bp + [0, 0, 0.105]), "gripper_end_over_basket_centre_z0.35": ik(kp + [0, 0, 0.35])}
print("ik:", json.dumps(report["ik"]))
# render the drop pose for the eye
cv2.imwrite(str(OUT / "bottle_basket_ik_drop_pose.png"), cv2.cvtColor(render("ext"), cv2.COLOR_RGB2BGR))
(OUT / "bottle_basket" / "verification.json").write_text(json.dumps(report, indent=1))
