"""Render a clamshell asset alone (plain floor, default back-face culling) in the open, half-open and closed states from six
angles, and print the closed-state hinge numbers: lid-base contacts, min hull distance, rim gap and footprint overhang per angle.
usage: hp_render_states.py asset.xml work_dir out_dir [--hinge J]"""
import argparse, json, pathlib, re, numpy as np, cv2, mujoco, open3d as o3d
p = argparse.ArgumentParser(); p.add_argument("asset"); p.add_argument("work"); p.add_argument("out"); p.add_argument("--hinge", default="headphone_case_hinge"); a = p.parse_args()
asset = pathlib.Path(a.asset).resolve(); OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True); W = pathlib.Path(a.work)
src = re.sub(r'file="([^"/]+)"', lambda m: f'file="{asset.parent / m.group(1)}"', asset.read_text()); body = re.sub(r"</?mujoco>", "", src)
scene = ('<mujoco><visual><headlight ambient=".35 .35 .35" diffuse=".5 .5 .5" specular=".1 .1 .1"/><global offwidth="1280" offheight="960"/></visual>'
         '<asset><texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".65 .65 .65" width="512" height="512"/><material name="grid" texture="grid" texrepeat="12 12" reflectance="0"/></asset>'
         '<worldbody><light pos="0.6 -0.6 1.2" dir="-0.4 0.4 -1" diffuse=".5 .5 .5"/><light pos="-0.6 0.6 1.2" dir="0.4 -0.4 -1" diffuse=".4 .4 .4"/><geom name="floor" type="plane" size="2 2 0.01" material="grid"/></worldbody>' + body + "</mujoco>")
model = mujoco.MjModel.from_xml_string(scene); data = mujoco.MjData(model); hj = model.joint(a.hinge); hq = hj.qposadr[0]
lid_body = hj.bodyid[0] if hasattr(hj, "bodyid") else model.jnt_bodyid[hj.id]; base_body = model.body_parentid[lid_body]
lid_col = [g for g in range(model.ngeom) if model.geom_bodyid[g] == lid_body and model.geom_group[g] == 3]; base_col = [g for g in range(model.ngeom) if model.geom_bodyid[g] == base_body and model.geom_group[g] == 3]
meta = json.load(open(asset.parent / f"{asset.stem}_meta.json")); hp = np.array(meta["hinge_point_m"]); tab = meta.get("lid_tab")
R = mujoco.Renderer(model, 360, 480); opt = mujoco.MjvOption(); opt_col = mujoco.MjvOption(); opt_col.geomgroup[:] = 0; opt_col.geomgroup[3] = 1; opt_col.geomgroup[0] = 1
views = {"from_lid_side": (0, -30, 0.6), "from_front": (90, -30, 0.6), "from_base_side": (180, -30, 0.6), "from_back": (270, -30, 0.6), "top": (90, -89, 0.6), "low_lid_side": (20, -8, 0.5)}
def shot(label, option=None, lookat=(0, 0, 0.03)):
    ims = []
    for k, (az, el, dist) in views.items():
        cam = mujoco.MjvCamera(); cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el; R.update_scene(data, cam, option or opt); im = R.render().copy()
        cv2.putText(im, f"{label} {k}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); ims.append(im)
    return np.hstack(ims)
rows = []; rows_col = []
for label, q in (("open", 0.0), ("half", np.pi / 2), ("closed", np.pi)):
    data.qpos[:] = 0; data.qpos[3] = 1; data.qpos[hq] = q; mujoco.mj_forward(model, data)
    rows.append(shot(label)); rows_col.append(shot(label, opt_col))
    pairs = [(c.geom1, c.geom2, c.dist) for c in data.contact[:data.ncon] if {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} == {lid_body, base_body}]
    ft = np.zeros(6); dmin = min(mujoco.mj_geomDistance(model, data, g1, g2, 0.1, ft) for g1 in lid_col for g2 in base_col)
    print("%6s: hinge %.2f rad, lid-base contacts %d (deepest %.1f mm), min lid-base hull distance %.1f mm" % (label, q, len(pairs), -min([d for *_, d in pairs], default=0) * 1000, dmin * 1000))
cv2.imwrite(str(OUT / "hp_states.jpg"), np.vstack(rows)[:, :, ::-1]); cv2.imwrite(str(OUT / "hp_states_hulls.jpg"), np.vstack(rows_col)[:, :, ::-1])
# tab close-up, open state
data.qpos[:] = 0; data.qpos[3] = 1; mujoco.mj_forward(model, data)
if tab:
    u = np.array(json.load(open(W / "hinge.json"))["lid_side_dir"]); perp = np.cross([0, 0, 1], u); c = hp + u * tab["centre_ts_m"][0] + perp * tab["centre_ts_m"][1]; c[2] = (tab["top_z_m"] + tab["rim_z_m"]) / 2
    views = {"tab_outside": (0, -20, 0.22), "tab_oblique": (40, -35, 0.25), "tab_inside": (180, -25, 0.22), "tab_top": (0, -89, 0.25)}; R2 = mujoco.Renderer(model, 480, 640); R = R2
    cv2.imwrite(str(OUT / "hp_tab_closeup.jpg"), shot("open", lookat=c + np.array([0, 0, -0.03]))[:, :, ::-1])
# closed close-up: the seam between the halves from four sides
data.qpos[:] = 0; data.qpos[3] = 1; data.qpos[hq] = np.pi; mujoco.mj_forward(model, data); R = mujoco.Renderer(model, 480, 640)
views = {"closed_lid_end": (0, -15, 0.32), "closed_side": (90, -15, 0.32), "closed_base_end": (180, -15, 0.32), "closed_high": (45, -45, 0.35)}
cv2.imwrite(str(OUT / "hp_closed_closeup.jpg"), shot("closed", lookat=(-0.08, 0, 0.035))[:, :, ::-1])
# geometric rim check from the meshes: flip the lid by pi about the hinge, compare rims per angle around the base ring centre
h = json.load(open(W / "hinge.json")); u = np.array(h["lid_side_dir"][:2]); perp = np.array([-u[1], u[0]]); piv = np.array(h["point_m"][:2]); hz = hp[2]
def ts(V): return np.c_[(V[:, :2] - piv) @ u, (V[:, :2] - piv) @ perp, V[:, 2]]
B = ts(np.asarray(o3d.io.read_triangle_mesh(str(W / "base.ply")).vertices)); L = ts(np.asarray(o3d.io.read_triangle_mesh(str(W / "lid.ply")).vertices)); L[:, 0] *= -1; L[:, 2] = 2 * hz - L[:, 2]
ring = B[(B[:, 2] > 0.015) & (B[:, 2] < 0.035)][:, :2]; c = ring.mean(0)
def polar(X): rel = X[:, :2] - c; return np.arctan2(rel[:, 1], rel[:, 0]), np.linalg.norm(rel, axis=1)
ab, rb = polar(B); al, rl = polar(L); rows = []
for a0 in np.arange(-np.pi, np.pi, np.pi / 12):
    mb = (ab >= a0) & (ab < a0 + np.pi / 12) & (B[:, 2] > 0.012); ml = (al >= a0) & (al < a0 + np.pi / 12) & (L[:, 2] < 2 * hz - 0.012)
    if mb.sum() < 30 or ml.sum() < 30: continue
    rbm = np.percentile(rb[mb], 99); rlm = np.percentile(rl[ml], 99); ob = mb & (rb > rbm - 0.008); ol = ml & (rl > rlm - 0.008)
    rows.append((np.degrees(a0), (np.percentile(L[ol, 2], 5) - np.percentile(B[ob, 2], 95)) * 1000, (rlm - rbm) * 1000))
rows = np.array(rows); hinge_side = np.abs(np.abs(rows[:, 0] + 7.5) - 0) < 40  # bins facing the hinge (+t from the base centre)
print("rim gap (flipped lid rim - base rim) per 15 deg bin, mm (0 deg = towards the hinge):"); print(" ".join("%4.0f:%+5.1f" % (d, g) for d, g, _ in rows))
print("lid footprint overhang beyond the base outline per bin, mm (+ = lid covers past the base edge):"); print(" ".join("%4.0f:%+5.1f" % (d, o) for d, _, o in rows))
print("rim gap away from the hinge side: median %+.1f mm, min %+.1f, max %+.1f; overhang median %+.1f mm, min %+.1f, max %+.1f" % (np.median(rows[~hinge_side, 1]), rows[~hinge_side, 1].min(), rows[~hinge_side, 1].max(), np.median(rows[~hinge_side, 2]), rows[~hinge_side, 2].min(), rows[~hinge_side, 2].max()))
print("wrote", OUT / "hp_states.jpg", OUT / "hp_states_hulls.jpg", OUT / "hp_tab_closeup.jpg", OUT / "hp_closed_closeup.jpg")
