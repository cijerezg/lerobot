"""Replay the actions of one real tube episode in the sim and compare the velocity and gripper torque it reports with the recording.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/tube_peg/replay_real_episode.py

Episode 1 of the val root (1,878 frames, three grasps). The sim tube stands where the third grasp happens and the holder where
that tube is released (fingertip pad centre from the recorded state, with the joint zero correction of the tube layouts). Open loop, so the tube is only caught if sim and real
kinematics agree to a few mm. Velocity: q01 / q99 per joint and the correlation with the recording. Gripper torque: share of
frames per prompt level (8 equal levels over policy.gripper_torque low / high of config_rl.yaml, 4-frame mean).
"""
import glob
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_tube_peg as V  # noqa: E402

from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import RebotB601SimRobotConfig  # noqa: E402
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import RebotB601Sim, to_qpos  # noqa: E402

ROOT = "outputs/data/rebot_val_2026-11-15-tube-merged-v1"
EPISODE = 1
LOW, HIGH, LEVELS = -0.215, 0.632, 8
GRASP = (1545, 1650)  # frames of the third grasp: closing starts, tube lifted


def levels(torque):
    smooth = np.convolve(torque, np.ones(4) / 4, mode="same")
    return np.clip(np.floor((smooth - LOW) / (HIGH - LOW) * LEVELS), 0, LEVELS - 1).astype(int) + 1


ZERO = [0.0, -3.43, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint_zero_offset_deg of the tube layouts
GRASP_AT, RELEASE_AT = 1600, 1825  # frames whose recorded state places the tube and the holder


def pad_xy(folder, state):
    """Fingertip pad centre of a recorded state in the corrected sim."""
    m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml")); d = mujoco.MjData(m)
    q0 = int(m.jnt_qposadr[m.joint("joint1").id])
    d.qpos[q0 : q0 + 8] = to_qpos(state + np.array(ZERO)); mujoco.mj_forward(m, d)
    p = d.geom_xpos[[m.geom("gripper_left_pad").id, m.geom("gripper_right_pad").id]].mean(0)
    print(f"  pad centre mm {(p * 1e3).round(1).tolist()}")
    return float(p[0]), float(p[1])


def replay(folder, act, slew):
    robot = RebotB601Sim(RebotB601SimRobotConfig(scene=str(folder / "scene.xml"), pos_vel_velocity=slew, gripper_force_cap=10.0, joint_zero_offset_deg=ZERO))
    robot.connect()
    names = robot.motor_names
    tube = robot.model.body("tube").id
    z0 = float(robot.data.xpos[tube][2]); holder = robot.model.body("tube_holder").id
    rows = []
    for a in act:
        robot.send_action({f"{n}.pos": float(v) for n, v in zip(names, a)})
        robot.skip_render = True
        obs = robot.get_observation()
        rows.append([obs[f"{n}.{k}"] for k in ("pos", "vel", "torque") for n in names] + [float(robot.data.xpos[tube][2]) - z0])
    rel = (robot.data.xpos[tube] - robot.data.xpos[holder]) * 1e3
    print(f"  tube relative to the holder at the end (mm): {rel.round(1).tolist()}")
    robot.disconnect()
    rows = np.array(rows)
    return rows[:, :7], rows[:, 7:14], rows[:, 14:21], rows[:, 21]


def main():
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(ROOT + "/data/*/*.parquet")))
    ep = df[df.episode_index == EPISODE].sort_values("frame_index")
    act, st, vel, eff = (np.stack(ep[k]) for k in ("action", "observation.state", "observation.velocity", "observation.effort"))
    folder = V.build()
    V.TUBE_XY, V.HOLDER_XY = pad_xy(folder, st[GRASP_AT]), pad_xy(folder, st[RELEASE_AT])
    folder = V.build()
    for slew in (None, [50.0] * 6 + [180.0]):
        pos, svel, storque, lift = replay(folder, act, slew)
        print(f"\npos_vel_velocity = {slew}")
        print("joint: real velocity q01 / q99 | sim q01 / q99 | corr | state error (deg, median abs)")
        for j in range(7):
            print(f"  {j}: {np.quantile(vel[:, j], 0.01):7.1f} {np.quantile(vel[:, j], 0.99):7.1f} | {np.quantile(svel[:, j], 0.01):7.1f} {np.quantile(svel[:, j], 0.99):7.1f}"
                  f" | {np.corrcoef(vel[:, j], svel[:, j])[0, 1]:.2f} | {np.median(np.abs(pos[:, j] - st[:, j])):.2f}")
        print("gripper torque level share (%), levels 1..8")
        print("  real:", (np.bincount(levels(eff[:, 6]), minlength=9)[1:] / len(eff) * 100).round(1).tolist())
        print("  sim: ", (np.bincount(levels(storque[:, 6]), minlength=9)[1:] / len(eff) * 100).round(1).tolist())
        a, b = GRASP
        print(f"third grasp, frames {a}-{b}: real torque median {np.median(eff[a:b, 6]):.3f}, sim {np.median(storque[a:b, 6]):.3f}; sim tube lift at {b + 30}: {lift[b + 30] * 1e3:.1f} mm")
        moving = np.abs(vel[:, 6]) > 20
        print(f"gripper moving (real |vel| > 20 deg/s, {moving.sum()} frames): real torque q05 / q95 {np.quantile(eff[moving, 6], 0.05):.2f} {np.quantile(eff[moving, 6], 0.95):.2f}, sim {np.quantile(storque[moving, 6], 0.05):.2f} {np.quantile(storque[moving, 6], 0.95):.2f}")
        still = ~moving
        print(f"gripper still: real torque q05 / q95 {np.quantile(eff[still, 6], 0.05):.2f} {np.quantile(eff[still, 6], 0.95):.2f}, sim {np.quantile(storque[still, 6], 0.05):.2f} {np.quantile(storque[still, 6], 0.95):.2f}")
    shutil.rmtree(folder)


if __name__ == "__main__":
    main()
