"""Start levels of the small objects layouts (tasks.py tiers ``small_near`` / ``small_turned``; user, 2026-11-26): the same scene with
the arm starting near the layout's first object, and near it with the wrist turned, instead of at the park pose. The near starts
differ on purpose (user: probe behaviour, not one fixed pose): on top of the object, close or midway, left or right of it
(PLAN), and the arm is moving toward the object when the episode starts.

    MUJOCO_GL=egl PYTHONSAFEPATH=1 uv run python lerobot/scripts/sim_builders/small_objects/build_small_starts.py

Run after build_small_layouts.py. Writes scenes/<layout>_{near,turned}_p1/{scene.xml,objects.json} in the robot package and
renders/starts.png here. The scene.xml is the base layout's with ``<custom><numeric name="start_deg">`` added (the arm's start
pose in follower degrees, rebot_b601_sim.py), as the tube start levels (tube_peg/build_tube_starts.py).

ontop:  the recordings' pose at the last frame of an "approach <object>" segment that a grasp follows (annotation pass
        small_objects_2026-11-26). Of the approach ends of the layout's object family, the one whose fingertips are nearest to
        the object in xy is moved onto the object by position-only IK (the wrist keeps the recorded tilt), fingertips at the
        family's median height at that frame, gripper at its median opening: only the grasp is left to do.
close:  the same, SIDE_M["close"] left or right of the object (y, the robot's left / right).
midway: the same, SIDE_M["midway"] to the side and MIDWAY_Z_M above the table: fingertips about 13 cm from the object, half
        of the distance from the park pose (25 to 31 cm).
near / turned of a layout: PLAN[level][layout's number in its family] = (ontop | close | midway, left | right); a layout's two
        starts differ in distance. turned = wrist_roll TURN_DEG further (the other way where the joint limit is in the way),
        fingertips put back on the same point.
speed:  every start has the fingertips moving straight at the object (``start_vel``, joint deg/s, gripper 0) at the recordings'
        median fingertip speed at that distance from the object (approach segments of the family, frames within 20 % of the
        distance). The sim robot runs the arm in at that velocity (rebot_b601_sim.py START_RUN_IN_S).

Per start: no arm contact (at the start pose and where the run-in begins), object not moved by the reset, and the sim robot's
first observation is the start pose (within 1 deg: the arm trails its moving target) with the start velocity.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tube_peg"))
import build_small_layouts as B  # noqa: E402
import small_objects as SO  # noqa: E402
from build_tube_starts import Arm  # noqa: E402

from lerobot.robots.rebot_b601_sim.config_rebot_b601_sim import MujocoCameraConfig, RebotB601SimRobotConfig  # noqa: E402
from lerobot.robots.rebot_b601_sim.rebot_b601_sim import GRIPPER_IDX, START_RUN_IN_S, RebotB601Sim  # noqa: E402
from lerobot.robots.rebot_b601_sim.tasks import task_for  # noqa: E402

V1 = B.V1
PASS = "outputs/annotation/passes/small_objects_2026-11-26"
WRIST_ROLL = 5
TURN_DEG = 45.0
SIDE_M = dict(ontop=0.0, close=0.04, midway=0.08)
MIDWAY_Z_M = 0.10
LEFT, RIGHT = 1.0, -1.0
PLAN = dict(near=(("ontop", LEFT), ("close", LEFT), ("midway", RIGHT), ("ontop", LEFT)),
            turned=(("midway", LEFT), ("ontop", LEFT), ("close", RIGHT), ("close", LEFT)))


def approaches():
    """Family -> the follower states (frames, 7) of every "approach <object>" segment that a grasp segment follows."""
    frames, out = {}, {kind: [] for kind in SO.PLACE}
    for labels in sorted(glob.glob(PASS + "/*/labels.json")):
        window = json.loads(Path(labels).with_name("window.json").read_text())
        root = window["root"]
        if root not in frames:
            df = pd.concat(pd.read_parquet(f, columns=["observation.state", "episode_index", "frame_index"]) for f in sorted(glob.glob(root + "/data/*/*.parquet")))
            frames[root] = {ep: np.stack(e.sort_values("frame_index")["observation.state"].to_numpy()) for ep, e in df.groupby("episode_index")}
        segments = json.loads(Path(labels).read_text())["segments"]
        for seg, nxt in zip(segments[:-1], segments[1:], strict=True):
            kind = seg["subtask"].split()[-1]  # "approach yellow plug" -> plug
            if seg["class"] == "approach" and nxt["class"] == "grasp" and kind in out:
                out[kind].append(frames[root][window["episode"]][seg["start"]:seg["end"]])  # segments count the episode's frames
    return out


def main():
    segments = approaches()
    ends = {kind: np.stack([seg[-1] for seg in segs]) for kind, segs in segments.items()}
    robot_fields = dict(task_for("die_cup_left_p1").robot)
    zero = robot_fields["joint_zero_offset_deg"]
    cameras = {"wrist": MujocoCameraConfig(camera="wrist", height=480, width=640, fps=30), "top": MujocoCameraConfig(camera="ext", height=480, width=640, fps=30)}
    arm = Arm(V1.SCENES / "die_cup_left_p1" / "scene.xml", zero, ["die", "dice_cup"])
    end_tips = {kind: np.stack([arm.tip(e)[0] for e in states]) for kind, states in ends.items()}
    for kind, tips in end_tips.items():
        tilts = [float(np.degrees(np.arccos(-arm.tip(e)[1][2]))) for e in ends[kind]]
        print(f"{kind}: {len(tips)} recorded approach ends, fingertip height median {np.median(tips[:, 2]) * 1e3:.0f} mm (q10 {np.quantile(tips[:, 2], 0.1) * 1e3:.0f}, q90 {np.quantile(tips[:, 2], 0.9) * 1e3:.0f}), "
              f"gripper median {np.median(ends[kind][:, GRIPPER_IDX]):.0f} deg, finger tilt from vertical median {np.median(tilts):.0f} deg, wrist_roll q10 {np.quantile(ends[kind][:, WRIST_ROLL], 0.1):.0f} q90 {np.quantile(ends[kind][:, WRIST_ROLL], 0.9):.0f} deg")
    # recorded fingertip speed against the distance to the object (taken as the table point under the approach end)
    speed_at = {}
    for kind, segs in segments.items():
        rows = []
        for seg in segs:
            tips = np.stack([arm.tip(state)[0] for state in seg])
            rows.append(np.stack([np.linalg.norm(tips[:-1] - [tips[-1, 0], tips[-1, 1], 0.0], axis=1), np.linalg.norm(np.diff(tips, axis=0), axis=1) * V1.FPS], axis=1))
        speed_at[kind] = np.concatenate(rows)
    tiles, ok_all, seen = [], True, []
    for base, spec in B.LAYOUTS.items():
        bodies = [b for b in spec if b != "note"]
        body = next(b for b in bodies if SO.kind_of(b) is not None)  # the first object of the script
        kind = SO.kind_of(body)
        arm = Arm(V1.SCENES / base / "scene.xml", zero, bodies)
        xyz = arm.s.d.xpos[arm.s.m.body(body).id].copy()
        number = seen.count(kind); seen.append(kind)  # the layout's number in its family
        near_z = float(np.median(end_tips[kind][:, 2]))
        grip = float(np.median(ends[kind][:, GRIPPER_IDX]))
        lo, hi = np.degrees(arm.s.limits[0][WRIST_ROLL]) - zero[WRIST_ROLL], np.degrees(arm.s.limits[1][WRIST_ROLL]) - zero[WRIST_ROLL]
        starts = None
        for k in np.argsort(np.linalg.norm(end_tips[kind][:, :2] - xyz[:2], axis=1)):  # nearest recorded approach end first
            starts, vels = {}, {}
            for level in ("near", "turned"):
                dist, side = PLAN[level][number]
                deg = arm.move_tip(ends[kind][k], np.array([xyz[0], xyz[1] + side * SIDE_M[dist], MIDWAY_Z_M if dist == "midway" else near_z]))
                deg[GRIPPER_IDX] = grip
                tip = arm.tip(deg)[0]
                if level == "turned":
                    deg[WRIST_ROLL] += TURN_DEG if deg[WRIST_ROLL] + TURN_DEG < hi else -TURN_DEG
                    deg = arm.move_tip(deg, tip)
                away = xyz - arm.tip(deg)[0]
                rows = speed_at[kind][np.abs(speed_at[kind][:, 0] / np.linalg.norm(away) - 1.0) < 0.2]
                vel = np.append(arm.joint_vel(deg, float(np.median(rows[:, 1])) * away / np.linalg.norm(away)), 0.0)
                if lo < deg[WRIST_ROLL] < hi and np.linalg.norm(arm.tip(deg)[0] - tip) < 1e-3 and arm.arm_contact_mm(deg) is None and arm.arm_contact_mm(deg - vel * START_RUN_IN_S) is None:
                    starts[level] = deg.round(2)
                    vels[level] = vel.round(2)
            if len(starts) == 2:
                seed = dict(recorded_state=ends[kind][k].round(1).tolist(), recorded_tip_to_object_xy_mm=round(float(np.linalg.norm(end_tips[kind][k, :2] - xyz[:2])) * 1e3, 1))
                break
            starts = None
        assert starts is not None, f"{base}: no recorded approach end reaches {body}"
        base_xml = (V1.SCENES / base / "scene.xml").read_text()
        base_meta = json.loads((V1.SCENES / base / "objects.json").read_text())
        for level, deg in {"home": np.zeros(7), **starts}.items():
            name = base if level == "home" else base.replace("_p1", f"_{level}_p1")
            tip, axis = arm.tip(deg)
            info = dict(start_deg=deg.tolist(), object=body, **({} if level == "home" else dict(zip(("distance", "side"), (PLAN[level][number][0], "none" if PLAN[level][number][0] == "ontop" else "left" if PLAN[level][number][1] > 0 else "right"), strict=True),
                                                                                            start_vel=vels[level].tolist(), tip_speed_mm_s=round(float(np.linalg.norm(arm.tip_vel(deg, vels[level]))) * 1e3))),
                        tip_to_object_mm=round(float(np.linalg.norm(tip - xyz)) * 1e3), tip_left_of_object_mm=round(float(tip[1] - xyz[1]) * 1e3), tip_height_mm=round(float(tip[2]) * 1e3),
                        finger_tilt_deg=round(float(np.degrees(np.arccos(-axis[2])))), arm_contact_mm=arm.arm_contact_mm(deg))
            if level != "home":
                folder = V1.SCENES / name; folder.mkdir(exist_ok=True)
                numeric = f'  <custom>\n    <numeric name="start_deg" data="{" ".join(str(v) for v in deg.tolist())}" />\n    <numeric name="start_vel" data="{" ".join(str(v) for v in vels[level].tolist())}" />\n  </custom>\n</mujoco>'
                (folder / "scene.xml").write_text(base_xml.replace(f'model="scene_{base}"', f'model="scene_{name}"').replace("</mujoco>", numeric))
            # the sim robot on the scene: first observation, object after the reset's settle, scorer state
            robot = RebotB601Sim(RebotB601SimRobotConfig(scene=name, cameras=cameras, auto_subtasks=True, **{k: list(v) if isinstance(v, tuple) else v for k, v in robot_fields.items()}))
            robot.connect()
            obs = robot.get_observation()
            state = np.array([obs[f"{n}.pos"] for n in robot.motor_names])
            moved = float(np.linalg.norm(robot.data.xpos[robot.model.body(body).id] - xyz))
            cursor = robot.subtask_scorer.poll()
            info.update(observed_state_error_deg=round(float(np.abs(state - deg).max()), 2), observed_vel=[round(obs[f"{n}.vel"], 1) for n in robot.motor_names], object_moved_mm=round(moved * 1e3, 2),
                        first_subtask=task_for(name).eval_subtasks[cursor], reward_at_start=robot.subtask_scorer.reward)
            ok = info["arm_contact_mm"] is None and info["observed_state_error_deg"] < 1.0 and moved < 1e-3
            ok_all &= ok or level == "home"  # home is the base layout, listed for reference
            tiles += [(f"{name} ({cam})", obs[cam]) for cam in ("top", "wrist")]
            robot.disconnect()
            print(f"{name}: {'reference' if level == 'home' else 'ok' if ok else 'FAILED'}  {info}", flush=True)
            if level != "home":
                meta = dict(layout=name, tier=task_for(name).tier, note=f"{base_meta['note']}; arm starts {level} ({base})", generator="scripts/sim_builders/small_objects/build_small_starts.py",
                            objects=base_meta["objects"], start=dict(level=level, base=base, **info, near_from=seed))
                (folder / "objects.json").write_text(json.dumps(meta, indent=1) + "\n")
    out = HERE / "renders"; out.mkdir(exist_ok=True)
    V1.sheet(tiles, out / "starts.png", cols=6, size=(480, 360))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
