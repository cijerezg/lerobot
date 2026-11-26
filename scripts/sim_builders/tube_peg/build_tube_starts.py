"""Start levels of the single-tube layouts (tasks.py tiers ``tube_half`` / ``tube_near``; user, 2026-11-25): the same scene with the
arm starting halfway to the tube, or at the end of the approach just above it, instead of at the park pose.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/tube_peg/build_tube_starts.py

Writes scenes/tube_peg_{left,right}_{half,near}_p1/{scene.xml,objects.json} in the robot package and renders/starts.png here. The
scene.xml is the base placement's with ``<custom><numeric name="start_deg">`` added: the arm's start pose in follower degrees,
which the sim robot puts the arm at on every scene reset (rebot_b601_sim.py).

near: the recordings' pose at the last frame of an "approach tube ..." window. Of the approach ends of the three tube training
      roots, the one whose fingertips are nearest to the layout's tube in xy is moved onto the tube by position-only IK (the wrist
      keeps the recorded tilt), fingertips at the recordings' median height at that frame, gripper at their median opening.
half: the point of the joint-space line from the park pose to `near` (gripper included) where the fingertips have covered half
      of the distance to the tube.

Per start: no arm contact, tube not moved by the 0.5 s settle, and the sim robot's first observation is the start pose.
"""
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_tube_peg as V  # noqa: E402

from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import MujocoCameraConfig, RebotB601SimRobotConfig  # noqa: E402
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import GRIPPER_IDX, RebotB601Sim, to_qpos  # noqa: E402
from lerobot.robots.rebot_b601_sim.tasks import task_for  # noqa: E402

V1 = V.V1
ROOTS = ("outputs/data/rebot_teleop_2026-11-14-tube-annotated-v2", "outputs/data/rebot_teleop_2026-11-14-tube2-annotated-v1",
         "outputs/data/rebot_teleop_2026-11-15-tube3-annotated-v1")
BASES = ("tube_peg_left_p1", "tube_peg_right_p1")
LEVELS = ("half", "near")


def approach_ends():
    """Follower state (7,) at the last frame of every "approach tube ..." window that a grasp window follows."""
    out = []
    for root in ROOTS:
        df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(root + "/data/*/*.parquet")))
        for ep, windows in json.loads(Path(root, "meta/subtask_windows.json").read_text())["episodes"].items():
            e = df[df.episode_index == int(ep)].sort_values("frame_index")
            states = np.stack(e["observation.state"].to_numpy())
            first = int(e["index"].iloc[0])  # the windows count frames from the root's first frame
            for w, nxt in zip(windows[:-1], windows[1:], strict=True):
                if w["subtask"].startswith("approach") and nxt["subtask"].startswith("grasp"):
                    out.append(states[w["to_index"] - 1 - first])
    return np.stack(out)


class Arm:
    """Kinematics of the corrected sim arm in follower degrees (the layouts' joint zero offset applied)."""

    def __init__(self, scene_xml, zero_deg):
        self.s = V1.Scene(scene_xml, ["tube", "tube_holder"])
        self.zero = np.asarray(zero_deg)
        self.ge = self.s.m.body("gripper_end").id

    def put(self, deg):
        s = self.s
        s.d.qpos[s.q0 : s.q0 + 8] = to_qpos(np.asarray(deg) + self.zero)
        mujoco.mj_forward(s.m, s.d)

    def tip(self, deg):
        """Fingertip point and the finger axis."""
        self.put(deg)
        axis = self.s.d.xmat[self.ge].reshape(3, 3) @ self.s.approach
        return self.s.d.xpos[self.ge] + self.s.tip_len * axis, axis

    def arm_contact_mm(self, deg):
        """Deepest contact of an arm body with anything else at this pose (mm, negative = penetration); None = no contact."""
        self.put(deg)
        s = self.s
        dists = [c.dist for c in s.d.contact[: s.d.ncon] if (s.m.geom_bodyid[c.geom1] in s.robot_bodies) != (s.m.geom_bodyid[c.geom2] in s.robot_bodies)]
        return round(min(dists) * 1e3, 2) if dists else None

    def move_tip(self, deg, target):
        """Position-only IK from `deg`: the smallest joint change that puts the fingertip point at `target`."""
        s = self.s
        deg = np.array(deg, dtype=float)
        lo, hi = np.degrees(s.limits[0]) - self.zero[:6], np.degrees(s.limits[1]) - self.zero[:6]
        jp = np.zeros((3, s.m.nv))
        for _ in range(200):
            tip, _ = self.tip(deg)
            mujoco.mj_jac(s.m, s.d, jp, None, tip, self.ge)
            J = jp[:, s.dofs]
            deg[:6] = np.clip(deg[:6] + 0.5 * np.degrees(J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(3), target - tip)), lo, hi)
        return deg


def main():
    ends = approach_ends()
    task = task_for(BASES[0])
    robot_fields = dict(task.robot)
    zero = robot_fields["joint_zero_offset_deg"]
    arm = Arm(V1.SCENES / BASES[0] / "scene.xml", zero)
    end_tips = np.stack([arm.tip(e)[0] for e in ends])
    near_z = float(np.median(end_tips[:, 2]))
    near_grip = float(np.median(ends[:, GRIPPER_IDX]))
    tilts = [float(np.degrees(np.arccos(-arm.tip(e)[1][2]))) for e in ends]
    print(f"{len(ends)} recorded approach ends: fingertip height median {near_z * 1e3:.0f} mm (q10 {np.quantile(end_tips[:, 2], 0.1) * 1e3:.0f}, "
          f"q90 {np.quantile(end_tips[:, 2], 0.9) * 1e3:.0f}), gripper median {near_grip:.0f} deg, finger tilt from vertical median {np.median(tilts):.0f} deg")
    cameras = {"wrist": MujocoCameraConfig(camera="wrist", height=480, width=640, fps=30), "top": MujocoCameraConfig(camera="ext", height=480, width=640, fps=30)}
    tiles, ok_all = [], True
    for base in BASES:
        arm = Arm(V1.SCENES / base / "scene.xml", zero)
        tube_xyz = arm.s.d.xpos[arm.s.m.body("tube").id].copy()
        target = np.array([tube_xyz[0], tube_xyz[1], near_z])
        park = np.zeros(7)
        near = None
        for k in np.argsort(np.linalg.norm(end_tips[:, :2] - tube_xyz[:2], axis=1)):  # nearest recorded approach end first
            cand = arm.move_tip(ends[k], target)
            cand[GRIPPER_IDX] = near_grip
            if np.linalg.norm(arm.tip(cand)[0] - target) < 1e-3 and arm.arm_contact_mm(cand) is None:
                near = cand.round(2)
                seed = dict(recorded_state=ends[k].round(1).tolist(), recorded_tip_to_tube_xy_mm=round(float(np.linalg.norm(end_tips[k, :2] - tube_xyz[:2])) * 1e3, 1),
                            joint_change_deg=np.abs(cand - ends[k])[:6].round(1).tolist())
                break
        assert near is not None, f"{base}: no recorded approach end reaches the tube"
        home_dist = np.linalg.norm(arm.tip(park)[0] - tube_xyz)
        fraction = next(f for f in np.linspace(0, 1, 201) if np.linalg.norm(arm.tip(park + f * (near - park))[0] - tube_xyz) <= home_dist / 2)
        starts = {"half": (park + fraction * (near - park)).round(2), "near": near}
        base_xml = (V1.SCENES / base / "scene.xml").read_text()
        base_meta = json.loads((V1.SCENES / base / "objects.json").read_text())
        for level, deg in {"home": park, **starts}.items():
            name = base if level == "home" else base.replace("_p1", f"_{level}_p1")
            tip, axis = arm.tip(deg)
            info = dict(start_deg=deg.tolist(), tip_to_tube_mm=round(float(np.linalg.norm(tip - tube_xyz)) * 1e3), tip_height_mm=round(float(tip[2]) * 1e3),
                        finger_tilt_deg=round(float(np.degrees(np.arccos(-axis[2])))), arm_contact_mm=arm.arm_contact_mm(deg))
            if level != "home":
                folder = V1.SCENES / name; folder.mkdir(exist_ok=True)
                numeric = f'  <custom>\n    <numeric name="start_deg" data="{" ".join(str(v) for v in deg.tolist())}" />\n  </custom>\n</mujoco>'
                (folder / "scene.xml").write_text(base_xml.replace(f'model="scene_{base}"', f'model="scene_{name}"').replace("</mujoco>", numeric))
            # the sim robot on the scene: first observation, tube after the reset's settle, scorer state
            robot = RebotB601Sim(RebotB601SimRobotConfig(scene=name, cameras=cameras, auto_subtasks=True, **{k: list(v) if isinstance(v, tuple) else v for k, v in robot_fields.items()}))
            robot.connect()
            obs = robot.get_observation()
            state = np.array([obs[f"{n}.pos"] for n in robot.motor_names])
            tube_moved = float(np.linalg.norm(robot.data.xpos[robot.model.body("tube").id] - tube_xyz))
            cursor = robot.subtask_scorer.poll()
            info.update(observed_state_error_deg=round(float(np.abs(state - deg).max()), 2), tube_moved_mm=round(tube_moved * 1e3, 2),
                        first_subtask=task_for(name).eval_subtasks[cursor], reward_at_start=robot.subtask_scorer.reward)
            ok = info["arm_contact_mm"] is None and info["observed_state_error_deg"] < 0.5 and tube_moved < 1e-3
            ok_all &= ok or level == "home"  # home is the base layout, listed for reference
            tiles += [(f"{name} ({cam})", obs[cam]) for cam in ("top", "wrist")]
            robot.disconnect()
            print(f"{name}: {'reference' if level == 'home' else 'ok' if ok else 'FAILED'}  {info}", flush=True)
            if level != "home":
                meta = dict(layout=name, tier=task_for(name).tier, note=f"{base_meta['note']}; arm starts {level} ({base})", generator="scripts/sim_builders/tube_peg/build_tube_starts.py",
                            objects=base_meta["objects"], start=dict(level=level, base=base, **info, **(dict(near_from=seed) if level == "near" else dict(fraction_of_joint_line=round(float(fraction), 3)))))
                (folder / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
    out = HERE / "renders"; out.mkdir(exist_ok=True)
    V1.sheet(tiles, out / "starts.png", cols=6, size=(480, 360))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
