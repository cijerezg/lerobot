"""Expand every ``waypoints.json`` of a correction pass into its corrected chunk.

  uv run python -m lerobot.annotation.corrections.build <work> [--source NAME ...]

``--source`` expands only those sources and keeps the other sources' rows of ``corrections.parquet`` as they are.
For each frame of ``<work>/frames.parquet`` whose folder holds a ``waypoints.json`` writes, in that folder,
  correction.npy   30 x 7 joint targets (deg): the first 1 s of the manoeuvre, held at its end when it is shorter
  manoeuvre.npy    N x 7: the whole manoeuvre
  check.json       the chunk check (``arm.check``), the travel numbers, the IK miss, or the error that stopped it
and for the pass
  <work>/corrections.parquet                  one row per labelled frame: source, root, episode, frame, kind, grade,
                                              subtask, contact (the labeller's), ok, recorded and corrected travel,
                                              cruise, contact-pose tip (the first waypoint), tip_spread_m,
                                              inconsistent, error, anchor_frame (the grasp frame of the frame's
                                              anchor, ``anchor.anchor``; -1 = none), anchor_cm (contact-pose tip
                                              to the anchor's measured tip), anchor_tried, anchor_grade, tier
  <work>/<source>/action_trace_records.npz    the records ``lerobot.probes.action_trace_sim`` and ``inspector`` read
                                              (demonstrated chunk = the recorded one, sample = the correction)

Pace: every manoeuvre cruises at the arm's speed limit (``MAX_SPEED_DEG_S``), so a correction covers more ground
than the recorded chunk whenever its goal is that far; ``--cruise D`` builds the pass at D deg/s instead (the
demonstrations move at 7-15). ``inconsistent`` is a review flag: within one episode and
subtask some other correction with the same contact has its contact-pose tip more than 3 cm away. A frame that fails
the check or cannot be expanded stays in the table with ok = False; so does a frame the labeller skipped
(``{"skip": reason}``, cut footage only), with error = "skip: <reason>".

``tier`` says where the contact pose of a ``grasp ...`` frame comes from, most trusted first:
  anchor         within 1 cm of the anchor's measured tip, no closing attempt between the frame and that grasp
  anchor-tried   the same, but an attempt failed in between: the object may have moved; the labeller kept the anchor
  shifted        the anchor's pose moved level onto the object as the frame shows it (``"xy"`` on the first waypoint)
  free           placed by the labeller from the images (about 2.5 cm from the real grasp, ``grasp_test``)
  other          not a ``grasp ...`` subtask
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from lerobot.annotation.corrections.anchor import ANCHOR_M, anchor, closings
from lerobot.annotation.corrections.arm import FPS, MAX_SPEED_DEG_S, Arm, check, frame
from lerobot.annotation.corrections.source import Episode, sources
from lerobot.annotation.vocab import code_for
from lerobot.probes.action_trace_probe import _write_records_npz

CHUNK = 30
RAMP = 12  # frames of each sine ramp
SPREAD_M = 0.03


def travel(state: np.ndarray, actions: np.ndarray) -> float:
    """Fastest-joint travel of a chunk commanded from ``state``, degrees: per step the largest arm-joint move, or a
    fifth of the gripper's (the gripper may turn 5x faster than an arm joint)."""
    step = np.abs(np.diff(np.vstack([state[None], actions]), axis=0))
    return float(np.maximum(step[:, :6].max(axis=1), step[:, 6] / 5).sum())


def smooth(arm: Arm, episode: Episode, t: int, spec: dict, cruise_deg_s: float) -> tuple[np.ndarray, list]:
    """Waypoints (correction_rubric.md) -> (N, 7) joint targets for the whole manoeuvre, and the waypoint tool poses.
    A move is a straight tip line, or with "arrive" a quadratic Bezier that ends travelling along that direction;
    orientation and gripper finish at 70 % of each move. The path is timed as one move that starts at the arm's
    current speed, cruises at ``cruise_deg_s`` and has sine ramps."""
    g = arm.gripper
    state = episode.states[t]
    _, _, jaw = g.fk(state)
    grip = state[6]
    poses = []
    for w in spec["waypoints"]:
        if "like_frame" in w:
            tip, approach, jaw = g.fk(episode.states[w["like_frame"]])
            if "xy" in w:  # the object is elsewhere on the table than at that frame: the same pose, moved level
                tip = np.array([*w["xy"], tip[2]], dtype=float)
            tip = tip + np.asarray(w.get("shift", [0.0, 0.0, 0.0]), dtype=float)  # where the tool got to, put right by the offset seen
            grip = float(w.get("gripper", episode.states[w["like_frame"]][6]))
        else:
            tip = np.asarray(w["tip"], dtype=float)
            approach = np.asarray(w.get("approach", [0.0, 0.0, -1.0]), dtype=float)
            jaw = np.asarray(w["jaw"], dtype=float) if "jaw" in w else jaw
            grip = float(w.get("gripper", grip))
        arrive = w.get("arrive")
        arrive = approach if arrive == "approach" else arrive
        poses.append((tip, approach, jaw, grip, None if arrive is None else np.asarray(arrive, dtype=float)))

    fine = [state]
    for tip1, approach1, jaw1, grip1, arrive in poses:
        tip0, approach0, jaw0 = g.fk(fine[-1])
        r0, r1 = frame(approach0, jaw0), frame(approach1, jaw1)
        flipped = r1 * np.array([-1.0, -1.0, 1.0])  # the jaw is symmetric: take the nearer sign
        if Rotation.from_matrix(r0.T @ flipped).magnitude() < Rotation.from_matrix(r0.T @ r1).magnitude():
            r1 = flipped
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0, r1])))
        control = (tip0 + tip1) / 2
        if arrive is not None:
            control = tip1 - min(0.10, 0.5 * np.linalg.norm(tip1 - tip0)) * arrive / np.linalg.norm(arrive)
        grip0 = fine[-1][6]
        for s in np.linspace(0.0, 1.0, 41)[1:]:
            early = min(1.0, s / 0.7)
            r = slerp(early).as_matrix()
            tip = (1 - s) ** 2 * tip0 + 2 * s * (1 - s) * control + s**2 * tip1
            fine.append(np.append(g.ik(fine[-1], tip, r[:, 2], r[:, 0], iters=30), grip0 + early * (grip1 - grip0)))
    fine = np.asarray(fine)
    step = np.abs(np.diff(fine[:, :6], axis=0)).max(axis=1)
    step = np.maximum(step, np.abs(np.diff(fine[:, 6])) / 5)  # the same metric as travel()
    length = np.concatenate([[0.0], np.cumsum(step)])
    cruise = cruise_deg_s / FPS
    v0 = min(1.0, float(np.abs(state - episode.states[max(t - 1, 0)])[:6].max()) / cruise)
    n = max(2 * RAMP, int(np.ceil(length[-1] / cruise)) + RAMP)
    k = np.arange(n) + 0.5
    up = v0 + (1 - v0) * np.sin(np.pi / 2 * np.minimum(1.0, k / RAMP)) ** 2
    speed = np.minimum(up, np.sin(np.pi / 2 * np.minimum(1.0, (n - k) / RAMP)) ** 2)
    at = np.cumsum(speed) / speed.sum() * length[-1]
    actions = np.column_stack([np.interp(at, length, fine[:, j]) for j in range(7)])
    return np.clip(actions, arm.limits[:, 0], arm.limits[:, 1]), poses


def tier(row: dict, first: dict, a: dict | None) -> str:
    """How far the contact pose of a correction can be trusted (the docstring's ``tier``)."""
    if not row["subtask"].startswith("grasp"):
        return "other"
    if a is None:
        return "free"
    if row["anchor_cm"] < 100 * ANCHOR_M:
        return "anchor-tried" if a["tried"] else "anchor"
    return "shifted" if first.get("like_frame") == a["frame"] and "xy" in first else "free"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--cruise", type=float, default=MAX_SPEED_DEG_S)
    args = parser.parse_args()
    work = args.work
    frames = pd.read_parquet(work / "frames.parquet")
    if args.source is not None:
        frames = frames[frames.source.isin(args.source)]
    folders = [work / r.source / f"ep{r.episode:03d}" / f"f{r.frame:06d}" for r in frames.itertuples()]
    frames = frames.assign(folder=folders)[[(f / "waypoints.json").exists() for f in folders]]
    arm = Arm()
    rows = []
    for name, source in sources(json.loads((work / "pass.json").read_text()), list(frames.source.unique())).items():
        records, worked = [], {}
        for r in frames[frames.source == name].itertuples():
            e, t = Episode(source, r.episode), r.frame
            if r.episode not in worked:
                worked[r.episode] = closings(arm, e)
            state, recorded = e.states[t], e.actions[t : t + CHUNK]
            row = dict(
                source=name, root=str(source.root), episode=r.episode, frame=t, kind=r.kind, grade=r.grade, subtask=r.subtask,
                contact="", verdict="", flag="", ok=False, recorded_travel_deg=travel(state, recorded), cruise_deg_s=args.cruise, error="",
            )
            rows.append(row)
            try:  # the labeller's file is the one input nobody checked
                spec = json.loads((r.folder / "waypoints.json").read_text())
                row["flag"] = str(spec.get("flag", ""))
                if "skip" in spec:
                    row["error"] = f"skip: {spec['skip']}"
                    continue
                row["contact"], row["verdict"] = str(spec["contact"]), str(spec.get("verdict", ""))
                code_for(row["contact"])  # a name outside the vocabulary is an error
                full, poses = smooth(arm, e, t, spec, row["cruise_deg_s"])
            except Exception as error:  # noqa: BLE001
                row["error"] = f"{type(error).__name__}: {error}"
                (r.folder / "check.json").write_text(json.dumps(dict(ok=False, error=row["error"]), indent=1))
                print("FAILED", r.folder, row["error"])
                continue
            correction = np.vstack([full[:CHUNK], np.repeat(full[-1:], max(0, CHUNK - len(full)), axis=0)])
            result = check(arm, state, correction)
            row.update(
                ok=result["ok"], corrected_travel_deg=travel(state, correction), manoeuvre_actions=len(full),
                missed_mm=1e3 * float(np.linalg.norm(arm.gripper.fk(full[-1])[0] - poses[-1][0])),  # last waypoint, IK reach
                tip_x=poses[0][0][0], tip_y=poses[0][0][1], tip_z=poses[0][0][2], anchor_frame=-1,
            )
            a = anchor(e, worked[r.episode], t)
            if a is not None:
                away = float(np.linalg.norm(poses[0][0] - arm.gripper.fk(e.states[a["frame"]])[0]))
                row.update(anchor_frame=a["frame"], anchor_cm=100 * away, anchor_tried=a["tried"], anchor_grade=a["grade"])
            row["tier"] = tier(row, spec["waypoints"][0], a)
            np.save(r.folder / "correction.npy", correction)
            np.save(r.folder / "manoeuvre.npy", full)
            keys = ("recorded_travel_deg", "corrected_travel_deg", "cruise_deg_s", "manoeuvre_actions", "missed_mm")
            (r.folder / "check.json").write_text(json.dumps(dict(result, **{k: row[k] for k in keys}), indent=1))
            if not result["ok"]:
                print("NOT OK", r.folder, result)
            records.append(dict(
                episode=r.episode, frame=t, global_idx=e.start + t, subtask=r.subtask, state=state, gt_chunk=recorded,
                sample_chunks=[correction], measured_states=e.states[t : t + CHUNK],
                sample_clearance=np.array([result["lowest_tip_z_m"]]),
                camera_frames={cam: e.images(cam, [t])[0][::2, ::2] for cam in ("top", "wrist")},
            ))
        if records:
            _write_records_npz(records, str(work / name), fps=float(FPS), table_z=0.0, dataset_repo_id=name, dataset_root=str(source.root))

    table = pd.DataFrame(rows)
    if args.source is not None:
        kept = pd.read_parquet(work / "corrections.parquet")
        table = pd.concat([kept[~kept.source.isin(args.source)], table], ignore_index=True)
    table["tip_spread_m"] = 0.0
    built = table[table.error == ""]
    for index in built.groupby(["source", "episode", "subtask", "contact"]).groups.values():
        tips = built.loc[index, ["tip_x", "tip_y", "tip_z"]].to_numpy()
        table.loc[index, "tip_spread_m"] = np.linalg.norm(tips[:, None] - tips[None], axis=2).max(axis=1)
    table["inconsistent"] = table.tip_spread_m > SPREAD_M
    table.to_parquet(work / "corrections.parquet")
    print(
        f"wrote {work / 'corrections.parquet'}: {len(table)} labelled frames, {int(table.ok.sum())} pass the check, "
        f"{int((table.error != '').sum())} could not be expanded, {int(table.inconsistent.sum())} flagged inconsistent"
    )


if __name__ == "__main__":
    main()
