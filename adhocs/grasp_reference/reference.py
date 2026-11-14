"""How the good human demonstrations grasp each object, measured from the annotated rebot roots of config_rl.yaml
(the object table of annotation/rubrics/correction_rubric.md).

  table    every grasp segment whose close is graded 4 or 5 -> grasps.csv + the per-object tables
  frames   wrist | top at the close of 3 lid and 3 kit-flap grasps
`frames` reads grasps.csv, so run `table` first.

A grasp = a "grasp ..." segment with a close event inside (the last one). Three moments, dataset-global indices:
  close  the close event: the gripper command has risen 25 deg above its most-open value
  stall  the first frame from there where the measured gripper moves < 2 deg in the next 10 frames (fingers on the object)
  before 30 frames (1 s) before the close event
Good = every frame grade from `before` to `stall` is 4 or 5 (split_segments_by_quality_spans).

  MUJOCO_GL=egl PYTHONPATH=lerobot/src .venv/bin/python lerobot/adhocs/grasp_reference/reference.py table
"""

import glob
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.corrections.arm import Arm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.rl.offline_dataset_utils import split_segments_by_quality_spans

CONFIG = Path("config_rl.yaml")
OUT = Path("outputs/grasp_reference")
SECOND = 30


def roots() -> list[Path]:
    """The annotated internal roots of the live config: no external pool, no correction roots."""
    names = sorted(set(re.findall(r"^\s*root:\s*(outputs/\S+)", CONFIG.read_text(), flags=re.M)))
    return [Path(n) for n in names if "rebot" in n and "external_rebot" not in n and "rebot_corrections" not in n]


def group(subtask: str, following: str) -> str:
    name = subtask.removeprefix("grasp the ")
    head = re.split(r" (?:of|on|in|under|next|near\w*|far\w*|lying|with|between|left|right|hanging)\b", name)[0]  # drop "next to the ..."
    if "flap" in name:
        return "kit flap, " + ("bits side" if "bits" in name else "white-card side")
    if head in ("headphone case lid", "lid"):
        return "shaker lid" if "shaker" in following else "headphone case lid"
    if "bottle" in head:
        kind = "pill bottle" if "pill" in head else head
        return kind + (", then stood upright" if "upright" in following else ", then carried")
    if "clipper" in head:
        return "large nail clipper" if "large" in head else "nail clipper (small / unnamed)"
    for text in (head, name):  # "the lowest of the three bits" has its noun after the cut
        for word in ("shirt", "sock", "bit", "cup", "tape roll"):
            if re.search(rf"\b{word}s?\b", text):
                return word
    return head


def tilt(v: np.ndarray) -> float:
    """Angle between v and straight down, degrees: 0 = straight down, 90 = horizontal, above 90 = rising."""
    return float(np.degrees(np.arccos(np.clip(-v[2] / np.linalg.norm(v), -1, 1))))


def grasps(root: Path, arm: Arm) -> tuple[list[dict], dict[str, list[float]]]:
    frames = pd.concat([pd.read_parquet(f) for f in glob.glob(str(root / "data/**/*.parquet"), recursive=True)]).sort_values("index")
    assert (frames["index"].to_numpy() == np.arange(len(frames))).all()  # from_index / to_index are dataset-global rows
    states, actions = np.stack(frames["observation.state"].to_numpy()), np.stack(frames["action"].to_numpy())
    episode, frame_index = frames.episode_index.to_numpy(), frames.frame_index.to_numpy()
    meta = root / "meta"
    segments = pd.read_parquet(meta / "episode_metadata.parquet")
    grade = np.zeros(len(frames), dtype=int)
    for piece in split_segments_by_quality_spans(segments.to_dict("records"), pd.read_parquet(meta / "quality_spans.parquet")):
        grade[piece["from_index"] : piece["to_index"]] = piece["quality"]
    contact = pd.read_parquet(meta / "contact.parquet").set_index(["episode_index", "segment_index"]).contact_slug
    events = pd.read_parquet(meta / "depth_gripper_events.parquet")
    closes = events[events.event_type == "close"]
    following = segments.groupby("episode_index").subtask.shift(-1).fillna("")
    rows, speeds = [], {}
    for seg, then in zip(segments.itertuples(), following, strict=True):
        if not seg.subtask.startswith("grasp"):
            continue
        inside = closes[(closes["index"] >= seg.from_index) & (closes["index"] < seg.to_index)]
        if inside.empty:
            rows.append(dict(root=root.name, group=group(seg.subtask, then), good=False, no_close_event=True))
            continue
        close, interval_end = int(inside["index"].iloc[-1]), int(inside["index"].iloc[-1] - inside.frame_index.iloc[-1] + inside.closed_interval_stop.iloc[-1])
        before = max(close - SECOND, seg.from_index)
        grip = states[:, 6]
        stall = next((t for t in range(close, min(close + 3 * SECOND, interval_end - 10)) if abs(grip[t + 10] - grip[t]) < 2), close)
        hold = slice(stall + 15, max(stall + 16, min(stall + 45, interval_end)))
        tip, approach, jaw = arm.gripper.fk(states[stall])
        tip0, approach0, _ = arm.gripper.fk(states[before])
        tip1 = arm.gripper.fk(states[close])[0]
        travel = tip1 - tip0
        heading = np.degrees(np.arctan2(approach[1], approach[0]))
        start = close  # the approach: the run of good frames leading to the close
        while start > seg.from_index and grade[start - 1] >= 4:
            start -= 1
        windows = [float(np.abs(states[t + SECOND, :6] - states[t, :6]).max()) for t in range(start, close - SECOND + 1, 10)]
        name = group(seg.subtask, then)
        good = bool(grade[before : stall + 1].min() >= 4)
        if good:
            speeds.setdefault(name, []).extend(windows)
        rows.append(dict(
            root=root.name, episode=int(episode[close]), close_frame=int(frame_index[close]), stall_frame=int(frame_index[stall]),
            subtask=seg.subtask, then=then, group=name, contact=contact[(seg.episode_index, seg.segment_index)],
            good=good, whole_segment_good=bool(grade[seg.from_index : seg.to_index].min() >= 4), no_close_event=False,
            tilt=tilt(approach), tilt_before=tilt(approach0),
            heading=heading, heading_vs_base=(heading - np.degrees(np.arctan2(tip[1], tip[0])) + 180) % 360 - 180,
            jaw_elevation=float(np.degrees(np.arcsin(abs(jaw[2])))),
            x_cm=100 * tip[0], y_cm=100 * tip[1], z_cm=100 * tip[2], z_before_cm=100 * tip0[2],
            travel_cm=100 * np.linalg.norm(travel), travel_flat_cm=100 * np.linalg.norm(travel[:2]), travel_down_cm=-100 * travel[2],
            travel_tilt=tilt(travel), travel_vs_fingers=float(np.degrees(np.arccos(np.clip(travel @ approach0 / np.linalg.norm(travel), -1, 1)))),
            close_to_stall_cm=100 * np.linalg.norm(tip - tip1), close_to_stall_s=(stall - close) / SECOND,
            opening=float(np.median(grip[before : close - 10])) if close - 10 > before else float(grip[before]),
            stall_measured=float(grip[stall]), hold_measured=float(np.median(grip[hold])), hold_command=float(np.median(actions[hold, 6])),
            last_second_deg=float(np.abs(states[close, :6] - states[before, :6]).max()),
        ))
    return rows, speeds


def quartiles(values: pd.Series, digits: int = 0) -> str:
    if values.empty:
        return "-"
    q = np.percentile(values, [25, 50, 75])
    return f"{q[1]:.{digits}f} ({q[0]:.{digits}f}..{q[2]:.{digits}f})"


def table(arm: Arm) -> None:
    rows, speeds = [], {}
    for root in roots():
        new_rows, new_speeds = grasps(root, arm)
        rows += new_rows
        for name, values in new_speeds.items():
            speeds.setdefault(name, []).extend(values)
    everything = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    everything.to_csv(OUT / "grasps.csv", index=False)
    print(f"{len(everything)} grasp segments; {int(everything.no_close_event.sum())} without a close event inside (dropped); "
          f"{int(everything.good.sum())} good; wrote {OUT / 'grasps.csv'}")
    good = everything[everything.good]
    pose, arrival, gripper = [], [], []
    for name, g in sorted(good.groupby("group"), key=lambda item: -len(item[1])):
        if len(g) < 10 and not re.search("flap|lid|bottle|clipper", name):
            continue
        tilted, moved = g[g.tilt >= 20], g[g.travel_cm >= 1]
        pose.append(dict(
            object=name, n=len(g), n_whole_segment_good=int(g.whole_segment_good.sum()),
            contact=", ".join(f"{k} {v}" for k, v in g.contact.value_counts().items()),
            tilt_at_close=quartiles(g.tilt), tilt_under_45=f"{(g.tilt < 45).mean():.0%}", tilt_max=round(g.tilt.max()),
            heading_vs_base=f"{quartiles(tilted.heading_vs_base)} n={len(tilted)}",
            across_base_line=f"{tilted.heading_vs_base.abs().between(45, 135).mean():.0%}", jaw_elevation=quartiles(g.jaw_elevation),
            tip_z_cm=quartiles(g.z_cm, 1),
        ))
        arrival.append(dict(
            object=name, tilt_1s_before=quartiles(g.tilt_before), tip_z_1s_before_cm=quartiles(g.z_before_cm, 1),
            travel_cm=quartiles(g.travel_cm, 1), flat_cm=quartiles(g.travel_flat_cm, 1), down_cm=quartiles(g.travel_down_cm, 1),
            travel_tilt=f"{quartiles(moved.travel_tilt)} n={len(moved)}", travel_vs_fingers=quartiles(moved.travel_vs_fingers),
            close_to_stall=f"{g.close_to_stall_s.median():.2f} s, {g.close_to_stall_cm.median():.1f} cm",
        ))
        approach = np.asarray(speeds.get(name, [np.nan]))
        gripper.append(dict(
            object=name, opening_in_last_second=quartiles(g.opening), stall_measured=quartiles(g.stall_measured),
            hold_measured=quartiles(g.hold_measured), hold_command=quartiles(g.hold_command),
            command_minus_measured=quartiles(g.hold_command - g.hold_measured),
            approach_deg_per_s=f"{np.nanmedian(approach):.0f} / {np.nanpercentile(approach, 90):.0f} (n={len(approach)})",
            last_second_deg_per_s=f"{g.last_second_deg.median():.0f} / {g.last_second_deg.quantile(0.9):.0f}",
        ))
    pooled = np.concatenate([np.asarray(v) for v in speeds.values()])
    for title, block in (("POSE AT THE STALL", pose), ("ARRIVAL: the last 1 s before the close event", arrival), ("GRIPPER (deg) AND SPEED (fastest arm joint, deg per 1 s: median / p90)", gripper)):
        print(f"\n{title}  -  median (q25..q75)\n{pd.DataFrame(block).to_string(index=False)}")
    print(f"\nall good grasps pooled: approach {np.median(pooled):.0f} / {np.percentile(pooled, 90):.0f} deg per s over {len(pooled)} windows; "
          f"last second {good.last_second_deg.median():.0f} / {good.last_second_deg.quantile(0.9):.0f}")


def images(root: Path, episode: int, frame: int) -> dict[str, np.ndarray]:
    dataset = LeRobotDataset(repo_id=root.name, root=root)
    out = {}
    for key in dataset.meta.camera_keys:  # top / wrist; external_0 / wrist_0 in the two oldest roots
        start = float(dataset.meta.episodes[episode][f"videos/{key}/from_timestamp"])
        decoded = decode_video_frames(dataset.root / dataset.meta.get_video_file_path(episode, key), [start + frame / dataset.fps],
                                      dataset.tolerance_s, backend=dataset._video_backend, return_uint8=True)
        out["wrist" if "wrist" in key else "top"] = decoded[0].permute(1, 2, 0).numpy().copy()
    return out


def save_pair(name: str, pair: dict[str, np.ndarray]) -> None:
    cv2.imwrite(str(OUT / name), np.hstack([pair["wrist"], pair["top"]])[:, :, ::-1])
    print("wrote", OUT / name)


def frames() -> None:
    good = pd.read_csv(OUT / "grasps.csv").query("good")
    for label, pattern in (("lid", "headphone case lid"), ("flap", "kit flap")):
        picks = good[good.group.str.contains(pattern)].sample(3, random_state=0)
        for i, g in enumerate(picks.itertuples()):
            save_pair(f"{label}_{i}.png", images(Path("outputs") / g.root, int(g.episode), int(g.stall_frame)))
            print(f"  {g.root} ep {g.episode:.0f} frame {g.stall_frame:.0f}: {g.subtask} | {g.contact} | tilt {g.tilt:.0f} z {g.z_cm:.1f} cm "
                  f"heading vs base {g.heading_vs_base:.0f} jaw elevation {g.jaw_elevation:.0f} gripper {g.stall_measured:.0f}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "table":
        table(Arm())
    elif command == "frames":
        frames()
