"""Per-channel gain for the sim wrist camera: real = gain * sim, fitted on table pixels (sim
segmentation = the textured plane) of the watched frames, using only pixels brighter than their frame's median in both images
so the arm's shadow (most of the real table) does not pull the fit dark. Prints the config_rl_sim.yaml lines.
usage: MUJOCO_GL=egl uv run python scripts/wrist_color_fit.py SCENE_XML FRAMES_DIR"""
import json, pathlib, sys, cv2, mujoco, numpy as np
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos
scene, frames = sys.argv[1], pathlib.Path(sys.argv[2]); model = mujoco.MjModel.from_xml_path(scene); data = mujoco.MjData(model); r = mujoco.Renderer(model, 480, 640)
plane = model.geom("table_top").id; R, S = [], []
for f, st in json.load(open(frames / "states.json")).items():
    data.qpos[:8] = to_qpos(np.asarray(st, float)); mujoco.mj_forward(model, data)
    r.disable_segmentation_rendering(); r.update_scene(data, "wrist"); sim = r.render(); r.enable_segmentation_rendering(); r.update_scene(data, "wrist"); seg = r.render()[:, :, 0]
    real = cv2.imread(str(frames / f"wrist_{int(f):05d}.jpg"))[:, :, ::-1]; m = cv2.erode((seg == plane).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if m.sum() < 2000: continue
    lit = m & (real.mean(2) > np.median(real.mean(2)[m])) & (sim.mean(2) > np.median(sim.mean(2)[m])); R.append(real[lit]); S.append(sim[lit])
# gain only: the lit sim table is nearly uniform, so an affine fit has no slope information.
R = np.concatenate(R).astype(float); S = np.concatenate(S).astype(float); gain = R.mean(0) / S.mean(0)
print(f"lit table px {len(R)}: real mean {R.mean(0).round(1)}, sim mean {S.mean(0).round(1)}")
print("        color_gain: [" + ", ".join(f"{g:.3f}" for g in gain) + "]\n        color_offset: [0.0, 0.0, 0.0]")
