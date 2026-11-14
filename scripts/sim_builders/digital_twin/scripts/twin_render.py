"""Render the twin: ext + wrist at recorded robot states (frames of rebot_teleop_2026-10-29-cut-v1 ep 0) next to the real frames,
plus a 1280x720 oblique beauty view. Real top frames are decoded from the dataset video with pyav; wrist frames come from room/renders.
Run from the LeRobot root: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/digital_twin/scripts/twin_render.py [FRAMES...]"""
import sys, pathlib, numpy as np, cv2, mujoco, pandas as pd, av
sys.path.insert(0, "lerobot/src"); from lerobot.probes.action_trace_sim import to_qpos
ROOT = pathlib.Path("lerobot/scripts/sim_builders/digital_twin"); OUT = ROOT / "twin" / "renders"; OUT.mkdir(parents=True, exist_ok=True)
DS = pathlib.Path("outputs/rebot_teleop_2026-10-29-cut-v1"); frames = [int(f) for f in sys.argv[1:]] or [0, 300, 2400]
states = np.stack(pd.read_parquet(DS / "data/chunk-000/file-000.parquet", columns=["observation.state"])["observation.state"].to_numpy())
def label(im, t):
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im
def decode(path, idx):
    want = set(idx); got = {}
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want: got[i] = fr.to_ndarray(format="bgr24")
            if i >= max(want): break
    return got
real_top = decode(DS / "videos/observation.images.top/chunk-000/file-000.mp4", frames)
real_wr = {f: cv2.imread(str(ROOT / "room/renders" / f"real_wrist_f{f}.png")) for f in frames}
missing = [f for f in frames if real_wr[f] is None]
if missing: real_wr.update(decode(DS / "videos/observation.images.wrist/chunk-000/file-000.mp4", missing))
m = mujoco.MjModel.from_xml_path(str(ROOT / "twin/scene_twin.xml")); d = mujoco.MjData(m); r = mujoco.Renderer(m, 480, 640)
def render(fr, cam, ren=r):
    d.qpos[:8] = to_qpos(states[fr]); mujoco.mj_forward(m, d); ren.update_scene(d, cam); return np.ascontiguousarray(ren.render()[:, :, ::-1])
rows = []
for fr in frames:
    top, wr = render(fr, "ext"), render(fr, "wrist")
    cv2.imwrite(str(OUT / f"sim_top_f{fr}.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 92]); cv2.imwrite(str(OUT / f"sim_wrist_f{fr}.jpg"), wr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUT / f"real_top_f{fr}.jpg"), real_top[fr], [cv2.IMWRITE_JPEG_QUALITY, 92])
    row = np.hstack([label(real_top[fr].copy(), f"real top f{fr}"), label(top.copy(), "twin top"), label(real_wr[fr].copy(), f"real wrist f{fr}"), label(wr.copy(), "twin wrist")])
    cv2.imwrite(str(OUT / f"side_by_side_f{fr}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 90]); rows.append(row)
cv2.imwrite(str(OUT / "side_by_side_all.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
# beauty view: oblique from the front-left, above the desk, looking at the desk centre
rb = mujoco.Renderer(m, 720, 1280); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.33, 0.0, 0.0]; cam.distance = 2.1; cam.azimuth = 300; cam.elevation = -32   # v3: azimuth 215 put the camera at x = 1.79, outside the (now opaque) front wall; 300 = from the right-back, inside the room
d.qpos[:8] = to_qpos(states[0]); mujoco.mj_forward(m, d); rb.update_scene(d, cam); beauty = np.ascontiguousarray(rb.render()[:, :, ::-1])
cv2.imwrite(str(OUT / "beauty_oblique.jpg"), beauty, [cv2.IMWRITE_JPEG_QUALITY, 92]); print("wrote", OUT)
