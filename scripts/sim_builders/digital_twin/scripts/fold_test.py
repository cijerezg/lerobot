"""Tests + renders for assets/bits_kit_fold/bits_kit_fold.xml on a table plane. Each fabric hinge is a strip body with hinge A (centre-to-strip,
bits_kit_hinge_<side>) and hinge B (strip-to-flap, bits_kit_hinge_<side>_flap); flap tilt = A + B. Tests: settle open 1 s (drift, rest angles),
press each flap with 1 N (flap mid and outer edge) and release, close inner (right) then outer (left) flap with a pure torque on the flap body
toward flat (A + B = pi) and release (rest angles, stack height, penetration, closed block size), drop the closed kit from 2 cm, reopen.
Renders open / half / closed, oblique + top, 480x640, contact sheet. usage: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/digital_twin/scripts/fold_test.py"""
import json, pathlib, re, numpy as np, cv2, mujoco
A = pathlib.Path("assets/sim_sources/digital_twin/assets/bits_kit_fold").resolve(); OUT = A / "renders"; OUT.mkdir(exist_ok=True)
src = re.sub(r'file="([^"/]+)"', lambda m: f'file="{A / m.group(1)}"', (A / "bits_kit_fold.xml").read_text()); inc = A / "_inc_tmp.xml"; inc.write_text(src)
scene = f"""<mujoco><option gravity="0 0 -9.81" timestep="0.001"/><visual><headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/><global offwidth="640" offheight="480"/></visual>
<asset><texture name="wood" type="2d" builtin="flat" rgb1="0.55 0.36 0.2" width="64" height="64"/><material name="wood" texture="wood" texrepeat="4 4" reflectance="0.05"/></asset>
<worldbody><light pos="0.3 -0.3 1.0" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7"/><light pos="-0.4 0.4 0.8" dir="0.4 -0.4 -0.8" diffuse="0.4 0.4 0.4"/><geom name="floor" type="plane" size="1 1 0.01" material="wood" friction="0.8 0.005 0.0001"/></worldbody>
<include file="{inc.name}"/></mujoco>"""
tmp = A / "_scene_tmp.xml"; tmp.write_text(scene); model = mujoco.MjModel.from_xml_path(str(tmp)); tmp.unlink(); inc.unlink(); data = mujoco.MjData(model); dt = model.opt.timestep
J = {s: dict(A=model.joint(f"bits_kit_hinge_{s}").id, B=model.joint(f"bits_kit_hinge_{s}_flap").id) for s in ("left", "right")}
qa = {s: {k: model.jnt_qposadr[j] for k, j in J[s].items()} for s in J}; da = {s: {k: model.jnt_dofadr[j] for k, j in J[s].items()} for s in J}
B = {s: model.body(f"bits_kit_{s}").id for s in ("left", "right")}; B["centre"] = model.body("bits_kit_centre").id; B["left_strip"] = model.body("bits_kit_left_strip").id; B["right_strip"] = model.body("bits_kit_right_strip").id
kit_bodies = set(B.values()); SIGN = {"left": -1, "right": 1}; AXIS = {"left": np.array([0, 1.0, 0]), "right": np.array([0, -1.0, 0])}; FLAP_W = {"left": 0.113, "right": 0.114}  # flap slab width beyond hinge B
def run(T, ctrl=None):
    for _ in range(int(T / dt)):
        data.qfrc_applied[:] = 0; data.xfrc_applied[:] = 0
        if ctrl: ctrl()
        mujoco.mj_step(model, data)
def ang(s): a, b = float(data.qpos[qa[s]["A"]]), float(data.qpos[qa[s]["B"]]); return dict(A_deg=np.degrees(a), B_deg=np.degrees(b), tilt_deg=np.degrees(a + b))
def kit_geoms(): return [g for g in range(model.ngeom) if model.geom_group[g] == 3 and model.geom_bodyid[g] in kit_bodies]
def kit_aabb():
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in kit_geoms():
        Rg = data.geom_xmat[g].reshape(3, 3); ext = np.abs(Rg) @ model.geom_size[g]; lo = np.minimum(lo, data.geom_xpos[g] - ext); hi = np.maximum(hi, data.geom_xpos[g] + ext)
    return lo, hi
def contacts():
    pen = 0.0; pairs = {}
    for i in range(data.ncon):
        c = data.contact[i]; b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if b1 in kit_bodies and b2 in kit_bodies: pen = min(pen, c.dist); k = f"{model.body(b1).name}-{model.body(b2).name}"; pairs[k] = pairs.get(k, 0) + 1
    return -pen, pairs
def edge_height(s):  # world z of the flap's outer slab edge (slab top face centre line)
    b = B[s]; return float((data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array([SIGN[s] * FLAP_W[s], 0, 0]))[2])
def flap_face_z(s):  # world z of the flap slab's outer face (the face that is up when folded)
    g = model.geom(f"bits_kit_{s}_slab").id; Rg = data.geom_xmat[g].reshape(3, 3); n = Rg[:, 2]; return float(data.geom_xpos[g][2] + (-n[2] if n[2] < 0 else n[2]) * model.geom_size[g][2] * (1 if n[2] < 0 else 1))
R = {}; rend = mujoco.Renderer(model, 480, 640); opt = mujoco.MjvOption(); sheet = []
def shot(name):
    pos = data.xpos[B["centre"]]
    for k, cam_kw in (("oblique", dict(azimuth=135, elevation=-35, distance=0.6, z=0.03)), ("top", dict(azimuth=90, elevation=-89, distance=0.6, z=0.0))):
        cam = mujoco.MjvCamera(); cam.lookat[:] = [pos[0], pos[1], cam_kw["z"]]; cam.distance = cam_kw["distance"]; cam.azimuth = cam_kw["azimuth"]; cam.elevation = cam_kw["elevation"]
        rend.update_scene(data, cam, opt); im = np.ascontiguousarray(rend.render()[:, :, ::-1]); t = f"{name} {k}"
        cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3); cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1); cv2.imwrite(str(OUT / f"{name}_{k}.jpg"), im); sheet.append(im)
# 1) settle open 1 s
mujoco.mj_resetData(model, data); p0 = data.qpos[:3].copy(); run(1.0); p1 = data.qpos[:3].copy(); pen, pairs = contacts()
R["settle_open"] = dict(drift_mm=float(1e3 * np.linalg.norm(p1[:2] - p0[:2])), centre_z_mm=float(1e3 * p1[2]), left=ang("left"), right=ang("right"), left_edge_height_mm=1e3 * edge_height("left"), right_edge_height_mm=1e3 * edge_height("right"), kit_kit_penetration_mm=1e3 * pen, kit_kit_contacts=pairs)
print("settle open:", json.dumps(R["settle_open"], default=float)); shot("open_rest"); q_open = data.qpos.copy()
# 2) press each flap with 1 N at its mid (half flap width from hinge B) and at its outer edge, hold 1 s, release 1.5 s
def press(s, d, F=1.0):
    def f():
        b = B[s]; pw = data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array([SIGN[s] * d, 0, 0]); mujoco.mj_applyFT(model, data, np.array([0, 0, -F]), np.zeros(3), pw, b, data.qfrc_applied)
    return f
R["press"] = {}
for s in ("left", "right"):
    for where, d in (("mid", FLAP_W[s] / 2), ("edge", FLAP_W[s])):
        data.qpos[:] = q_open; data.qvel[:] = 0; run(0.3); t0 = ang(s)["tilt_deg"]; h0 = edge_height(s); run(1.0, press(s, d)); t1 = ang(s); h1 = edge_height(s); run(1.5); t2 = ang(s)["tilt_deg"]
        R["press"][f"{s}_{where}"] = dict(before_tilt_deg=t0, pressed=t1, pressed_edge_height_mm=1e3 * h1, after_release_tilt_deg=t2, sag_deg=t0 - t1["tilt_deg"]); print("press 1 N", s, where, json.dumps(R["press"][f"{s}_{where}"], default=float))
# 3) close: pure torque on the flap body about the hinge axis, PD on the flap tilt toward pi, 2 s; then 1.5 s more with a PD torque on the strip
# body toward 150 deg (a hand laying the hinge strip over; without it the strip stands past vertical with the flap tilted down onto the far
# contents and locked there by friction, and a press on the flap only locks it harder); release 1.5 s
def close(s, target=np.pi, kp=0.6, kd=0.05, strip_target=None):
    def f():
        tilt = data.qpos[qa[s]["A"]] + data.qpos[qa[s]["B"]]; vel = data.qvel[da[s]["A"]] + data.qvel[da[s]["B"]]
        data.xfrc_applied[B[s], 3:] = AXIS[s] * np.clip(kp * (target - tilt) - kd * vel, -1.0, 1.0)
        if strip_target is not None: data.xfrc_applied[model.body(f"bits_kit_{s}_strip").id, 3:] = AXIS[s] * np.clip(0.3 * (strip_target - data.qpos[qa[s]["A"]]) - 0.02 * data.qvel[da[s]["A"]], -0.5, 0.5)
    return f
def state(label):
    lo, hi = kit_aabb(); pen, pairs = contacts()
    d = dict(left=ang("left"), right=ang("right"), stack_height_mm=1e3 * hi[2], block_size_mm=(1e3 * (hi - lo)).round(1).tolist(), right_face_z_mm=1e3 * flap_face_z("right"), left_face_z_mm=1e3 * flap_face_z("left"), kit_kit_penetration_mm=1e3 * pen, kit_kit_contacts=pairs,
             centre_drift_mm=1e3 * float(np.linalg.norm(data.qpos[:2] - q_open[:2]))); print(label, json.dumps(d, default=float)); return d
data.qpos[:] = q_open; data.qvel[:] = 0; run(0.3)
run(2.0, close("right")); run(1.5, close("right", strip_target=np.radians(150))); R["inner_closed_driven"] = state("inner (right) driven:"); run(1.5); R["inner_closed"] = state("inner (right) released:"); shot("half")
run(2.0, close("left")); run(1.5, close("left", strip_target=np.radians(150))); R["closed_driven"] = state("outer (left) driven:"); run(1.5); R["closed"] = state("both released (closed):"); shot("closed"); q_closed = data.qpos.copy()
# 4) hold check 3 s more (slides open?), then drop closed from 2 cm
run(5.0); R["closed_after_5s"] = state("closed after 5 s more:"); run(5.0); R["closed_after_10s"] = state("closed after 10 s more:")
data.qpos[:] = q_closed; data.qvel[:] = 0; data.qpos[2] += 0.02; run(1.5); R["drop_closed_2cm"] = state("drop closed 2 cm:"); R["drop_closed_2cm"]["stays_closed"] = bool(R["drop_closed_2cm"]["left"]["tilt_deg"] > 150 and R["drop_closed_2cm"]["right"]["tilt_deg"] > 150)
# 5) reopen the outer flap (torque toward tilt 0.3 rad), release
run(2.0, close("left", target=0.3)); run(1.5); R["reopen_left"] = ang("left"); print("reopen left:", json.dumps(R["reopen_left"], default=float))
R["model"] = dict(total_mass_kg=float(model.body_subtreemass[B["centre"]]), ngeom_collision=len(kit_geoms()), timestep=dt)
json.dump(R, open(A / "test_results.json", "w"), indent=1, default=float)
rows = [np.hstack(sheet[i:i + 2]) for i in range(0, len(sheet), 2)]; cv2.imwrite(str(OUT / "contact_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", OUT / "contact_sheet.jpg")
