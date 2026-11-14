"""Object appearance in the wrist view, real vs sim, independent of the scene misalignment: colour statistics of the pixels
inside the real object mask (depth sidecar: 4-40 mm above the table; clippers also metal-coloured) and inside the sim object
mask (segmentation), pooled over the watched frames. usage: MUJOCO_GL=egl uv run python scripts/object_color_check.py SCENE_XML FRAMES_DIR ROOT EP [--gain r g b]"""
import argparse, json, pathlib, sys, cv2, mujoco, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import scene_wrist as Wr
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("frames"); ap.add_argument("root"); ap.add_argument("ep", type=int); ap.add_argument("--gain", type=float, nargs=3); a = ap.parse_args()
model = mujoco.MjModel.from_xml_path(a.scene); data = mujoco.MjData(model); r = mujoco.Renderer(model, 480, 640); frames = pathlib.Path(a.frames)
pool = {n: dict(real=[], sim=[]) for n in ("clipper_case", "clipper_large", "clipper_small")}
for f, st in json.load(open(frames / "states.json")).items():
    f = int(f); data.qpos[:8] = to_qpos(np.asarray(st, float)); mujoco.mj_forward(model, data)
    r.update_scene(data, "wrist"); sim = r.render()[:, :, ::-1].astype(float); sim = np.clip(sim * np.array(a.gain)[::-1], 0, 255) if a.gain else sim
    r.enable_segmentation_rendering(); r.update_scene(data, "wrist"); seg = r.render()[:, :, 0]; r.disable_segmentation_rendering()
    real = cv2.imread(str(frames / f"wrist_{f:05d}.jpg")).astype(float); hsv = cv2.cvtColor(real.astype(np.uint8), cv2.COLOR_BGR2HSV)
    d16, _ = Wr.load_depth(a.root, a.ep, f); dm = d16 * Wr.UNITS_M; valid = (d16 > 0) & (dm < 0.8); pts = Wr.backproject(np.where(valid, dm, 0))
    for name in pool:
        body = model.body(name).id; sm = (seg >= 0) & (model.geom_bodyid[np.clip(seg, 0, None)] == body)
        if sm.sum() < 1500: continue
        region = cv2.dilate(sm.astype(np.uint8), np.ones((201, 201), np.uint8)) > 0; tab = valid & ~region if (valid & ~region).sum() > 3000 else valid
        P = pts[tab]; n, d, _ = Wr.ransac_plane(P[np.random.default_rng(0).choice(len(P), min(6000, len(P)), replace=False)])
        h = np.full(dm.shape, np.nan); h[valid] = pts[valid] @ n + d; above = (h > 0.004) & (h < 0.04) & region
        if name != "clipper_case": above &= (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 70)
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(above.astype(np.uint8));
        if nlab < 2: continue
        rm = lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if rm.sum() < 300: continue
        pool[name]["real"].append(real[rm]); pool[name]["sim"].append(sim[sm])
for name, p in pool.items():
    if not p["real"]: continue
    R = np.concatenate(p["real"]); S = np.concatenate(p["sim"])
    hr = [np.histogram(R[:, c], 32, (0, 256))[0] / len(R) for c in range(3)]; hs = [np.histogram(S[:, c], 32, (0, 256))[0] / len(S) for c in range(3)]
    inter = np.mean([np.minimum(hr[c], hs[c]).sum() for c in range(3)])
    print(f"{name}: {len(p['real'])} frames | mean BGR real {R.mean(0).round(0)} sim {S.mean(0).round(0)} | std real {R.std(0).round(0)} sim {S.std(0).round(0)} | histogram overlap {inter:.2f}")
