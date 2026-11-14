"""Drive the sim robot through the RTC runtime's env seam without a policy: env + processors from the yaml's
env block (gym_manipulator.make_robot_env / make_processors, the calls rtc_actor_runtime makes), scripted targets
through step_env_and_process_transition, keys/dtypes against the real follower's, step time, renders, park() reset.
usage: MUJOCO_GL=egl uv run python lerobot/scripts/sim_builders/real2sim/scripts/sim_robot_check.py config_rl_sim.yaml OUT_DIR"""
import pathlib, sys, time, cv2, draccus, numpy as np, torch, yaml
import lerobot.rl.gym_manipulator as gm
from lerobot.envs.configs import EnvConfig
from lerobot.processor import TransitionKey, create_transition
from lerobot.robots.rebot_b601_sim import STOP_EPISODES

cfg_path, out = sys.argv[1], pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
env_cfg = draccus.decode(EnvConfig, yaml.safe_load(open(cfg_path))["env"])
env_cfg.robot.episodes = 1
env, teleop = gm.make_robot_env(env_cfg)
env_proc, act_proc = gm.make_processors(env, teleop, env_cfg, "cpu")
robot = env.robot
print("observation_space:", {k: (v.shape, v.dtype.name) for k, v in env.observation_space.spaces.items()})
obs, info = env.reset()
raw = {k: (tuple(v.shape), str(v.dtype)) for k, v in obs.items() if hasattr(v, "shape")}
raw.update({f"pixels/{k}": (v.shape, v.dtype.name) for k, v in obs["pixels"].items()})
print("raw obs:", raw)
transition = env_proc(create_transition(observation=obs, info=info))
print("processed obs:", {k: (tuple(v.shape), str(v.dtype)) for k, v in transition[TransitionKey.OBSERVATION].items() if hasattr(v, "shape")})

names = [k.removesuffix(".pos") for k in robot.action_features]
home = np.zeros(7); raised = np.array([20.0, -60.0, -90.0, 0.0, -30.0, 0.0, -200.0])  # open gripper, arm over the table
script = np.concatenate([np.tile(home, (15, 1)), np.linspace(home, raised, 60), np.tile(raised, (30, 1)), np.linspace(raised, home, 60)])
bodies = ["clipper_case", "clipper_large", "clipper_small"]
start_pos = {b: robot.data.body(b).xpos.copy() for b in bodies}
times, cmd, meas = [], [], []
for i, target in enumerate(script):
    t0 = time.perf_counter()
    transition = gm.step_env_and_process_transition(env, transition, torch.as_tensor(target, dtype=torch.float32), env_proc, act_proc)
    times.append(time.perf_counter() - t0)
    cmd.append(target); meas.append([float(transition[TransitionKey.OBSERVATION][f"{n}.pos"]) for n in names])
    if i in (0, 74, 100, 164):
        o = env.get_raw_observation(); top = o["pixels"]["top"][:, :, ::-1]; wrist = o["pixels"]["wrist"][:, :, ::-1]
        depth = o["wrist.depth"].numpy() if isinstance(o["wrist.depth"], torch.Tensor) else o["wrist.depth"]
        dm = depth * 1e-4; dcol = cv2.applyColorMap((255 * np.clip((dm - 0.05) / 0.45, 0, 1)).astype(np.uint8), cv2.COLORMAP_VIRIDIS); dcol[depth == 0] = 0
        cv2.imwrite(str(out / f"step_{i:03d}.jpg"), np.hstack([top, wrist, dcol]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"step {i}: depth valid {100 * (depth > 0).mean():.0f}%, median {np.median(dm[depth > 0]):.3f} m, state {np.round(meas[-1], 1)}")
cmd, meas = np.array(cmd), np.array(meas)
lag = np.abs(cmd - meas).max(axis=0)
print(f"step time: mean {1e3 * np.mean(times):.1f} ms, max {1e3 * np.max(times):.1f} ms (budget 33.3)")
print("max |command - state| per joint (deg):", np.round(lag, 1))
print("object drift during the script (mm):", {b: round(1e3 * float(np.linalg.norm(robot.data.body(b).xpos - start_pos[b])), 1) for b in bodies})
# displace an object, then park: the scene must come back to its start pose
qadr = robot.model.jnt_qposadr[robot.model.body("clipper_large").jntadr[0]]; robot.data.qpos[qadr:qadr + 3] += [0.05, 0.05, 0.02]
try:
    robot.park(on_step=lambda sent: None); sigint = False
except KeyboardInterrupt:
    sigint = True
print("park: arm at", np.round([float(v) for k, v in robot.get_observation().items() if k.endswith(".pos")], 2),
      "| objects back:", {b: round(1e3 * float(np.linalg.norm(robot.data.body(b).xpos - start_pos[b])), 2) for b in bodies},
      "| episodes_done", robot.episodes_done, "| STOP_EPISODES", STOP_EPISODES.is_set(), "| SIGINT delivered", sigint)
print("teleop events after stop:", teleop.get_teleop_events())
env.close(); teleop.disconnect(); print("wrote", out)
