"""Drop an asset into the ReBot DM scene, settle, optionally drive a hinge closed, record video + stills.
usage: test_scene.py asset.xml out_dir --pos X Y Z [--yaw deg] [--hinge JOINT] [--table_z Z] [--mass_check]"""
import argparse, os, pathlib, re, subprocess, numpy as np, cv2, mujoco
p = argparse.ArgumentParser(); p.add_argument("asset"); p.add_argument("out"); p.add_argument("--pos", nargs=3, type=float, default=[0.35, 0.0, 0.0])
p.add_argument("--yaw", type=float, default=0.0); p.add_argument("--hinge", default=None); p.add_argument("--table_z", type=float, default=0.0)
p.add_argument("--settle", type=float, default=1.0); p.add_argument("--drive", type=float, default=3.0); p.add_argument("--hold", type=float, default=1.0); a = p.parse_args()
ROOT = pathlib.Path(__file__).resolve().parents[5]; MJCF = ROOT / "lerobot/src/lerobot/robots/rebot_b601_follower/mjcf"; OUT = pathlib.Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
asset = pathlib.Path(a.asset).resolve(); src = asset.read_text()
src = re.sub(r'file="([^"/]+)"', lambda m: f'file="{asset.parent / m.group(1)}"', src)  # absolute mesh paths: meshdir of the robot model then does not apply
q = np.array([np.cos(np.deg2rad(a.yaw) / 2), 0, 0, np.sin(np.deg2rad(a.yaw) / 2)])
src = re.sub(r'(<body name="[^"]+" )pos="0 0 0"', lambda m: m.group(1) + f'pos="{a.pos[0]} {a.pos[1]} {a.pos[2]}" quat="{q[0]:.5f} 0 0 {q[3]:.5f}"', src, count=1)
if a.hinge:
    src = src.replace("</mujoco>", f'  <actuator><position name="hinge_drive" joint="{a.hinge}" kp="0.6" kv="0.05" ctrlrange="0 3.1416"/></actuator>\n</mujoco>')
obj_tmp = MJCF / f"_r2s_obj_{os.getpid()}.xml"; scene_tmp = MJCF / f"_r2s_scene_{os.getpid()}.xml"
scene_src = (MJCF / "scene_dm.xml").read_text().replace('name="floor"', f'name="floor" pos="0 0 {a.table_z:.4f}"')
scene_src = scene_src.replace("</mujoco>", f'  <include file="{obj_tmp.name}"/>\n</mujoco>')
obj_tmp.write_text(src); scene_tmp.write_text(scene_src)
try:
    model = mujoco.MjModel.from_xml_path(str(scene_tmp))
finally:
    obj_tmp.unlink(); scene_tmp.unlink()
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
body = model.body(re.search(r'<body name="([^"]+)"', src).group(1)); print("asset body", body.name, "mass %.3f kg (subtree)" % model.body_subtreemass[body.id], "ngeom", model.ngeom)
cams = {"oblique": dict(lookat=[a.pos[0], a.pos[1], 0.05], distance=0.9, azimuth=135, elevation=-35), "top": dict(lookat=[a.pos[0], a.pos[1], 0.0], distance=0.8, azimuth=90, elevation=-89),
        "close": dict(lookat=[a.pos[0], a.pos[1], 0.03], distance=0.45, azimuth=160, elevation=-25)}
R = mujoco.Renderer(model, 480, 640); opt = mujoco.MjvOption(); opt_col = mujoco.MjvOption(); opt_col.geomgroup[:] = 0; opt_col.geomgroup[3] = 1; opt_col.geomgroup[0] = 1
def shot(name, option=None):
    ims = []
    for k, c in cams.items():
        cam = mujoco.MjvCamera(); cam.lookat[:] = c["lookat"]; cam.distance = c["distance"]; cam.azimuth = c["azimuth"]; cam.elevation = c["elevation"]
        R.update_scene(data, cam, option or opt); ims.append(R.render().copy())
    im = np.hstack(ims); cv2.imwrite(str(OUT / f"{name}.jpg"), im[:, :, ::-1]); return im
fps = 30; dt = model.opt.timestep; steps_per_frame = int(round(1 / fps / dt)); frames = []; log = []
T = a.settle + (a.drive + a.hold if a.hinge else 0); n_frames = int(T * fps); q0 = data.qpos.copy()
hinge_id = model.joint(a.hinge).qposadr[0] if a.hinge else None
for f in range(n_frames):
    t = f / fps
    if a.hinge: data.ctrl[model.actuator("hinge_drive").id] = np.pi * np.clip((t - a.settle) / a.drive, 0, 1)
    for _ in range(steps_per_frame): mujoco.mj_step(model, data)
    frames.append(shot("_frame")[:, :1280]); pos = data.xpos[body.id]
    log.append((t, float(data.qpos[hinge_id]) if a.hinge else 0.0, *pos))
    if f == 0: shot("still_start"); shot("still_start_hulls", opt_col)
    if a.hinge and abs(t - (a.settle + a.drive / 2)) < 0.5 / fps: shot("still_mid")
shot("still_end"); shot("still_end_hulls", opt_col)
vid = OUT / "frames"; vid.mkdir(exist_ok=True)
for i, fr in enumerate(frames): cv2.imwrite(str(vid / f"{i:04d}.jpg"), fr[:, :, ::-1])
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(vid / "%04d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(OUT / "test.mp4")], check=True)
log = np.array(log); np.savetxt(OUT / "log.csv", log, delimiter=",", header="t,hinge_rad,x,y,z", comments="")
print("settled body pos", log[int(a.settle * fps) - 1, 2:].round(4), "start", log[0, 2:].round(4)); 
if a.hinge: print("hinge final %.2f rad (target 3.14); body drift during close %.4f m" % (log[-1, 1], np.linalg.norm(log[-1, 2:4] - log[int(a.settle * fps) - 1, 2:4])))
print("wrote", OUT / "test.mp4")
