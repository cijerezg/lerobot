"""Branch a recorded sim rollout and execute hand-written corrective actions from one of its frames.

A rollout recorded with ``--record_inference_dataset=true`` holds every executed action, and the sim is deterministic,
so frame t is restored by replaying actions[:t] from the scene reset. From there:

  look   dump what a labeller gets on the real robot: top / wrist RGB (plain and with a pixel grid), wrist depth,
         joint state, camera calibration. No object poses.
  run    execute a plan file's actions (N x 7 joint targets, degrees) from frame t, and the policy's own next N actions
         from the same frame; write a side-by-side video (policy | plan) per camera and the ground-truth metrics.

Run from the workspace root:
  MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/sim_branch/simlab.py look twin_l3_close_lid_p2 600
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

from lerobot.rl.inference_video import VideoWriter
from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import MujocoCameraConfig, RebotB601SimRobotConfig
from lerobot.robots.rebot_b601_sim.easy_tasks import LID, LID_HINGE, body_surface_distance
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import DEPTH_UNITS_M, GRIPPER_DEG_OPEN, RebotB601Sim

HERE = Path(__file__).resolve().parent
OUT = Path("outputs/llm_corrections_sim_2026-11-09")
ROLLOUTS = OUT / "rollouts/validation/step_00000000/sim"
H, W = 480, 640


def make_robot(scene: str) -> RebotB601Sim:
    # the robot block of config_rl_sim.yaml
    cameras = {
        "wrist": MujocoCameraConfig(camera="wrist", use_depth=True, height=H, width=W, fps=30),
        "top": MujocoCameraConfig(camera="ext", height=H, width=W, fps=30),
    }
    robot = RebotB601Sim(RebotB601SimRobotConfig(scene=scene, cameras=cameras, fps=30, substeps=17, gravcomp=True, pos_vel_velocity=None))
    robot.connect()
    return robot


def load_rollout(layout: str) -> tuple[np.ndarray, np.ndarray]:
    """(actions, states) of episode 0, degrees, frames in order."""
    files = sorted((ROLLOUTS / layout / "inference_dataset" / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[df["episode_index"] == 0].sort_values("frame_index")
    return np.stack(df["action"].to_numpy()), np.stack(df["observation.state"].to_numpy())


def send(robot: RebotB601Sim, action_deg: np.ndarray) -> None:
    robot.send_action({f"{m}.pos": float(v) for m, v in zip(robot.motor_names, action_deg, strict=True)})


def restore(layout: str, t: int) -> tuple[RebotB601Sim, np.ndarray, np.ndarray]:
    actions, states = load_rollout(layout)
    robot = make_robot(layout)
    for a in actions[:t]:
        send(robot, a)
    gap = float(np.abs(robot._present_deg() - states[t]).max())
    print(f"restored {layout} frame {t}: max |replayed - recorded| joint = {gap:.3f} deg")
    return robot, actions, states


# ---- gripper frame -------------------------------------------------------------------------------------------------
class Gripper:
    """Tool frame from the model: approach axis (gripper_end toward the finger tips), jaw axis (left finger to right
    finger), tip = the point between the finger tips."""

    def __init__(self, model: mujoco.MjModel, limits_deg: list[tuple[float, float]]):
        self.m = model
        self.soft_lo, self.soft_hi = np.deg2rad(np.asarray(limits_deg[:6])).T  # the follower's soft limits
        self.d = mujoco.MjData(model)  # scratch data: IK never touches the running sim
        self.ge = model.body("gripper_end").id
        self.left, self.right = model.body("gripper_left").id, model.body("gripper_right").id
        self.q0 = int(model.jnt_qposadr[model.joint("joint1").id])
        self.dofs = [int(model.jnt_dofadr[model.joint(f"joint{i}").id]) for i in range(1, 7)]
        d = self.d
        mujoco.mj_resetData(model, d)
        d.qpos[self.q0 + 6 : self.q0 + 8] = 0.05
        mujoco.mj_forward(model, d)
        rg, og = d.xmat[self.ge].reshape(3, 3), d.xpos[self.ge]
        mid = (d.xpos[self.left] + d.xpos[self.right]) / 2  # the finger bodies sit behind the tips; gripper_end is at the tip end
        self.approach_local = rg.T @ (og - mid) / np.linalg.norm(og - mid)
        jaw = d.xpos[self.right] - d.xpos[self.left]
        self.jaw_local = rg.T @ jaw / np.linalg.norm(jaw)
        reach = 0.0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in (self.left, self.right) and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[k] : model.mesh_vertadr[k] + model.mesh_vertnum[k]]
                world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                reach = max(reach, float(((world - og) @ (rg @ self.approach_local)).max()))
        self.tip_len = reach

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tip, approach, jaw) in the world frame."""
        rg = data.xmat[self.ge].reshape(3, 3)
        approach = rg @ self.approach_local
        return data.xpos[self.ge] + self.tip_len * approach, approach, rg @ self.jaw_local

    def fk(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.d.qpos[self.q0 : self.q0 + 6] = np.deg2rad(q_deg[:6])
        mujoco.mj_forward(self.m, self.d)
        tip, approach, jaw = self.pose(self.d)
        return tip.copy(), approach.copy(), jaw.copy()

    def ik(self, q_deg: np.ndarray, tip: np.ndarray, approach: np.ndarray, jaw: np.ndarray, iters: int = 100) -> np.ndarray:
        """Arm joints (deg) putting the tool frame at (tip, approach, jaw), seeded from q_deg: damped least squares,
        at most 0.05 rad per joint per iteration. The jaw is symmetric, so the sign nearer the seed is taken."""
        m, d = self.m, self.d
        q = np.deg2rad(np.asarray(q_deg[:6], dtype=float))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        approach = approach / np.linalg.norm(approach)
        jaw = jaw - (jaw @ approach) * approach
        jaw = jaw / np.linalg.norm(jaw)
        d.qpos[self.q0 : self.q0 + 6] = q
        mujoco.mj_forward(m, d)
        if self.pose(d)[2] @ jaw < 0:
            jaw = -jaw
        for _ in range(iters):
            d.qpos[self.q0 : self.q0 + 6] = q
            mujoco.mj_forward(m, d)
            p, a, j = self.pose(d)
            mujoco.mj_jac(m, d, jp, jr, p, self.ge)
            e_rot = 0.5 * (np.cross(a, approach) + np.cross(j, jaw))
            e = np.concatenate([tip - p, 0.2 * e_rot])
            jac = np.vstack([jp[:, self.dofs], 0.2 * jr[:, self.dofs]])
            dq = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(6), jac.T @ e)
            dq *= min(1.0, 0.05 / max(float(np.abs(dq).max()), 1e-9))
            q = np.clip(q + dq, self.soft_lo, self.soft_hi)
        return np.rad2deg(q)


# ---- cameras -------------------------------------------------------------------------------------------------------
def camera_calibration(robot: RebotB601Sim, cam: str) -> dict:
    """Pinhole intrinsics and the camera pose in the world (robot base) frame: on the real robot the top camera's pose
    is the fixed calibration and the wrist camera's is FK times the hand-eye calibration."""
    cid = robot.model.camera(robot.config.cameras[cam].camera).id
    f = 0.5 * H / np.tan(np.deg2rad(robot.model.cam_fovy[cid]) / 2)
    return dict(f=float(f), cx=W / 2, cy=H / 2, pos=robot.data.cam_xpos[cid].tolist(), mat=robot.data.cam_xmat[cid].reshape(3, 3).tolist())


def deproject(calib: dict, u: float, v: float, depth_m: float) -> np.ndarray:
    """Pixel + depth along the optical axis -> world point (MuJoCo camera: x right, y up, looking along -z)."""
    x = (u - calib["cx"]) / calib["f"] * depth_m
    y = -(v - calib["cy"]) / calib["f"] * depth_m
    return np.asarray(calib["pos"]) + np.asarray(calib["mat"]) @ np.array([x, y, -depth_m])


def project(calib: dict, point: np.ndarray) -> tuple[float, float]:
    local = np.asarray(calib["mat"]).T @ (np.asarray(point) - np.asarray(calib["pos"]))
    return calib["cx"] + calib["f"] * local[0] / -local[2], calib["cy"] - calib["f"] * local[1] / -local[2]


def ray_to_height(calib: dict, u: float, v: float, z: float) -> np.ndarray:
    """World point where the pixel's ray meets the horizontal plane at height z (top camera: no depth)."""
    origin = np.asarray(calib["pos"])
    direction = np.asarray(calib["mat"]) @ np.array([(u - calib["cx"]) / calib["f"], -(v - calib["cy"]) / calib["f"], -1.0])
    return origin + direction * (z - origin[2]) / direction[2]


def grid(rgb: np.ndarray, step: int = 40) -> np.ndarray:
    out = cv2.resize(rgb, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
    for x in range(0, W, step):
        cv2.line(out, (2 * x, 0), (2 * x, 2 * H), (255, 255, 0), 1)
        cv2.putText(out, str(x), (2 * x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for y in range(0, H, step):
        cv2.line(out, (0, 2 * y), (2 * W, 2 * y), (255, 255, 0), 1)
        cv2.putText(out, str(y), (2, 2 * y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return out


def observe(robot: RebotB601Sim) -> dict:
    obs = robot.get_observation()
    return dict(
        top=obs["top"].copy(), wrist=obs["wrist"].copy(), wrist_depth_m=obs["wrist.depth"].astype(np.float32) * DEPTH_UNITS_M,
        state_deg=robot._present_deg(), calib={cam: camera_calibration(robot, cam) for cam in ("top", "wrist")},
    )


# ---- ground truth (evaluation only, never given to the plan) --------------------------------------------------------
def truth(robot: RebotB601Sim, gripper: Gripper) -> dict:
    m, d = robot.model, robot.data
    j = m.joint(LID_HINGE).id
    lo, hi = m.jnt_range[j]
    tip, approach, jaw = gripper.pose(d)
    return dict(
        lid_closed_frac=float(d.qpos[m.jnt_qposadr[j]] / (hi if lo == 0 else lo)),
        tip_to_lid_mm=1e3 * body_surface_distance(m, d, m.body(LID).id, tip),
        tip=tip.round(4).tolist(), approach=approach.round(3).tolist(), jaw=jaw.round(3).tolist(),
        gripper_deg=float(robot._present_deg()[6]),
    )


def rollout(robot: RebotB601Sim, gripper: Gripper, actions: np.ndarray) -> tuple[list[dict], list[dict]]:
    frames, metrics = [], []
    for a in actions:
        send(robot, a)
        obs = robot.get_observation()
        frames.append(dict(top=obs["top"].copy(), wrist=obs["wrist"].copy()))
        metrics.append(truth(robot, gripper))
    return frames, metrics


def label(rgb: np.ndarray, text: str) -> np.ndarray:
    out = rgb.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def dump(obs: dict, gripper: Gripper, out: Path, **extra) -> None:
    """What the labeller sees of one state."""
    out.mkdir(parents=True, exist_ok=True)
    for cam in ("top", "wrist"):
        cv2.imwrite(str(out / f"{cam}.png"), obs[cam][:, :, ::-1])
        cv2.imwrite(str(out / f"{cam}_grid.png"), grid(obs[cam])[:, :, ::-1])
    np.save(out / "wrist_depth_m.npy", obs["wrist_depth_m"])
    tip, approach, jaw = gripper.fk(obs["state_deg"])
    (out / "state.json").write_text(json.dumps(dict(
        **extra, state_deg=obs["state_deg"].round(3).tolist(), calib=obs["calib"],
        tool=dict(tip=tip.round(4).tolist(), approach=approach.round(3).tolist(), jaw=jaw.round(3).tolist(), tip_len=round(gripper.tip_len, 4)),
    ), indent=1))


def main() -> None:
    cmd, layout, t = sys.argv[1], sys.argv[2], int(sys.argv[3])
    robot, actions, states = restore(layout, t)
    gripper = Gripper(robot.model, list(robot.config.joint_limits.values()))
    if cmd == "look":
        out = OUT / "states" / f"{layout}_t{t:04d}"
        dump(observe(robot), gripper, out, layout=layout, frame=t)
        print(f"wrote {out}")
        return
    # run: each plan file exposes plan(obs, gripper) -> (N, 7) joint targets in degrees; the files are stages, each
    # planned from the observation the previous one ended in
    sys.path.insert(0, str(HERE))
    plan_paths = [Path(p) for p in sys.argv[4:]]
    out = OUT / "runs" / f"{layout}_t{t:04d}_{plan_paths[-1].stem}"
    start = truth(robot, gripper)
    stages, plan_frames, plan_metrics = [], [], []
    for plan_path in plan_paths:
        spec = importlib.util.spec_from_file_location(plan_path.stem, plan_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stages.append(np.asarray(module.plan(observe(robot), gripper), dtype=float))
        frames, metrics = rollout(robot, gripper, stages[-1])
        plan_frames += frames
        plan_metrics += metrics
    planned = np.vstack(stages)
    dump(observe(robot), gripper, out / "end_state", layout=layout, frame=t, steps=len(planned))
    np.save(out / "actions_plan.npy", planned)
    robot.disconnect()
    robot, _, _ = restore(layout, t)
    policy_frames, policy_metrics = rollout(robot, gripper, actions[t : t + len(planned)])
    for cam in ("top", "wrist"):
        writer = VideoWriter(out / f"{cam}.mp4", 30, 2 * W, H)
        for i in range(len(planned)):
            left = label(policy_frames[min(i, len(policy_frames) - 1)][cam], f"policy ckpt 800  +{i + 1}")
            writer.write(np.hstack([left, label(plan_frames[i][cam], f"plan  +{i + 1}")]))
        writer.close()
        for name, i in (("90", min(89, len(planned) - 1)), ("end", len(planned) - 1)):
            cv2.imwrite(str(out / f"{cam}_{name}.png"), np.hstack([policy_frames[min(i, len(policy_frames) - 1)][cam], plan_frames[i][cam]])[:, :, ::-1])
    step = np.abs(np.diff(np.vstack([states[t][None], planned]), axis=0)).max(axis=0) * 30
    keys = ("lid_closed_frac", "tip_to_lid_mm", "gripper_deg")
    summary = dict(
        layout=layout, frame=t, stage_steps=[len(s) for s in stages], start=start,
        policy_90=policy_metrics[min(89, len(policy_metrics) - 1)], plan_90=plan_metrics[min(89, len(planned) - 1)],
        policy_end=policy_metrics[-1], plan_end=plan_metrics[-1], plan_max_joint_speed_deg_s=step.round(1).tolist(),
    )
    (out / "metrics.json").write_text(json.dumps(dict(
        **summary, policy=[{k: m[k] for k in keys} for m in policy_metrics[::10]], plan=[{k: m[k] for k in keys} for m in plan_metrics[::10]],
    ), indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
