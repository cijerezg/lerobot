"""Asset XML on a wooden table plane: settle 1 s, stills oblique | top | close, hulls view, side-by-side with a real frame.
usage: render_asset.py asset.xml out_dir --real frame.jpg [--dist 0.5]  (MUJOCO_GL=egl)"""
import argparse, pathlib, re, numpy as np, cv2, mujoco
p = argparse.ArgumentParser(); p.add_argument("asset"); p.add_argument("out"); p.add_argument("--real", required=True); p.add_argument("--dist", type=float, default=0.5); a = p.parse_args()
asset = pathlib.Path(a.asset).resolve(); OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True); src = asset.read_text()
src = re.sub(r'file="([^"/]+)"', lambda m: f'file="{asset.parent / m.group(1)}"', src); body = re.search(r'<body name="([^"]+)"', src).group(1)
inc = asset.parent / "_asset_tmp.xml"; inc.write_text(src)
scene = f"""<mujoco><option gravity="0 0 -9.81"/><visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><global offwidth="640" offheight="480"/></visual>
<asset><texture name="wood" type="2d" builtin="flat" rgb1="0.55 0.36 0.2" width="64" height="64"/><material name="wood" texture="wood" texrepeat="4 4" reflectance="0.05"/></asset>
<worldbody><light pos="0.3 -0.3 1.0" dir="-0.3 0.3 -1" diffuse="0.7 0.7 0.7"/><light pos="-0.4 0.4 0.8" dir="0.4 -0.4 -0.8" diffuse="0.4 0.4 0.4"/><geom name="floor" type="plane" size="1 1 0.01" material="wood"/></worldbody>
<include file="{inc.name}"/></mujoco>"""
tmp = asset.parent / "_render_tmp.xml"; tmp.write_text(scene); model = mujoco.MjModel.from_xml_path(str(tmp)); tmp.unlink(); inc.unlink(); data = mujoco.MjData(model)
for _ in range(int(1.0 / model.opt.timestep)): mujoco.mj_step(model, data)
b = model.body(body); pos = data.xpos[b.id]; print("body", body, "mass %.3f kg" % model.body_subtreemass[b.id], "settled z %.4f m" % pos[2], "ngeom", model.ngeom)
d = a.dist; cams = {"oblique": dict(lookat=[pos[0], pos[1], 0.02], distance=d, azimuth=135, elevation=-35), "top": dict(lookat=[pos[0], pos[1], 0.0], distance=d, azimuth=90, elevation=-89),
                    "close": dict(lookat=[pos[0], pos[1], 0.02], distance=d * 0.55, azimuth=200, elevation=-20)}
R = mujoco.Renderer(model, 480, 640); opt = mujoco.MjvOption(); opt_h = mujoco.MjvOption(); opt_h.geomgroup[:] = 0; opt_h.geomgroup[3] = 1; opt_h.geomgroup[0] = 1
def shot(c, option):
    cam = mujoco.MjvCamera(); cam.lookat[:] = c["lookat"]; cam.distance = c["distance"]; cam.azimuth = c["azimuth"]; cam.elevation = c["elevation"]; R.update_scene(data, cam, option); return R.render().copy()[:, :, ::-1]
def label(im, t): cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3); cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1); return im
ims = [label(np.ascontiguousarray(shot(c, opt)), k) for k, c in cams.items()]; hull = label(np.ascontiguousarray(shot(cams["oblique"], opt_h)), "hulls")
real = cv2.imread(a.real); real = label(cv2.resize(real, (int(real.shape[1] * 480 / real.shape[0]), 480)), "real frame")
for k, im in zip(cams, ims): cv2.imwrite(str(OUT / f"{k}.jpg"), im)
cv2.imwrite(str(OUT / "hulls.jpg"), hull); cv2.imwrite(str(OUT / "side_by_side.jpg"), np.hstack([real] + ims + [hull]), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", OUT / "side_by_side.jpg")
