"""Real frames vs the sim robot's observation at the watched frames of an episode, with the robot built from the sim yaml's
env.robot block (scene, cameras, background, colour correction: exactly what the policy gets). Robot posed from the recorded
states, objects at the scene's start pose, through RebotB601Sim.get_observation().
usage: MUJOCO_GL=egl uv run python scripts/photoreal_check.py CONFIG_YAML FRAMES_DIR OUT_DIR [--tag t] [--no-background]"""
import argparse, json, pathlib, cv2, draccus, mujoco, numpy as np, yaml
from lerobot.envs.configs import EnvConfig
from lerobot.robots.rebot_b601_sim import RebotB601Sim
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import to_qpos
import lerobot.rl.gym_manipulator  # noqa: F401  registers the robot/camera config choices
ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("frames"); ap.add_argument("out"); ap.add_argument("--tag", default=""); ap.add_argument("--no-background", action="store_true"); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); frames = pathlib.Path(a.frames); states = json.load(open(frames / "states.json"))
cfg = draccus.decode(EnvConfig, yaml.safe_load(open(a.config))["env"]).robot
if a.no_background:
    for cam in cfg.cameras.values(): cam.background = None
robot = RebotB601Sim(cfg); robot.connect()
def label(im, t):
    cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3); cv2.putText(im, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1); return im
rows_top, rows_wrist = [], []
for f, state in sorted(states.items(), key=lambda kv: int(kv[0])):
    f = int(f); robot._qpos_robot()[:] = to_qpos(np.asarray(state, float)); mujoco.mj_forward(robot.model, robot.data)
    obs = robot.get_observation(); top = obs["top"][:, :, ::-1].copy(); wrist = obs["wrist"][:, :, ::-1].copy()
    real_top = cv2.imread(str(frames / f"top_{f:05d}.jpg")); real_wrist = cv2.imread(str(frames / f"wrist_{f:05d}.jpg"))
    rows_top.append(np.hstack([label(real_top.copy(), f"real top f{f}"), label(top.copy(), "sim obs"), label(cv2.addWeighted(real_top, 0.5, top, 0.5, 0), "blend")]))
    rows_wrist.append(np.hstack([label(real_wrist.copy(), f"real wrist f{f}"), label(wrist.copy(), "sim obs"), label(cv2.addWeighted(real_wrist, 0.5, wrist, 0.5, 0), "blend")]))
    print(f"f{f}: mean |real - sim| top {np.abs(real_top.astype(float) - top).mean():.1f}, wrist {np.abs(real_wrist.astype(float) - wrist).mean():.1f} (0-255)")
cv2.imwrite(str(out / f"top{a.tag}.jpg"), np.vstack(rows_top), [cv2.IMWRITE_JPEG_QUALITY, 85]); cv2.imwrite(str(out / f"wrist{a.tag}.jpg"), np.vstack(rows_wrist), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("wrote", out)
