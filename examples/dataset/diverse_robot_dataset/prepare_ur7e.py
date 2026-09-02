#!/usr/bin/env python

"""Nominate and prepare review material for the production UR7e component.

UR7e is the smallest and the most unusual of the four sources. The whole
100-episode payload is 0.86 GB across six files, every episode runs about 70 s at
a native 30 Hz, and all 100 clear the boundary policy, so acquisition is trivial
and selection is purely about execution diversity inside one repetitive scripted
stacking task.

Two things make it different from DROID and they drive this script:

* **The action is a scripted waypoint goal, not a dense per-step command.** It
  equals `skill.goal_position.joint` on 83% of rows, changes on 18.6% of frames,
  and steps as much as 1.39 rad in one frame while the state does not move. So
  activity screening reads `observation.state`, which is where the arm's motion
  actually is; screening the action would report the arm as stationary for
  seconds at a time and then fire on a teleport. Section 7.5 permits exactly this
  per-source choice, and the values themselves are preserved untouched.
* **There are no publisher keep intervals.** The field does not exist, which is
  not the same as a DROID episode whose `keep_ranges` is empty, so
  `build_annotations` is called with `enforce_keep_intervals=False`.

At 30 Hz the extractor takes the native-rate branch, so this is the one component
where an all-zero `source.action_interpolated_mask` is correct rather than a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from prepare_droid import (  # noqa: E402
    ACTION_SECONDS,
    ANCHOR_STRIDE_S,
    contact_sheet,
    overview_offsets,
    read_json,
    write_json,
)

FPS = 30.0
HISTORY_SECONDS = 6.0
# UR7e joint positions are radians and the gripper is a [0,1] ratio. Both sit above
# this source's own per-frame noise floor, measured on the staged shard.
JOINT_ACTIVITY_RAD = 0.02
GRIPPER_ACTIVITY = 0.02
CAMERAS = ("observation.images.realsense_topview", "observation.images.realsense_wrist")


@dataclass
class EpisodeSummary:
    episode_index: int
    frames: int
    duration_s: float
    source_task: str
    candidate_anchors: int
    active_anchors: int
    active_anchor_fraction: float
    state_path_length: float
    gripper_path_length: float
    max_state_step: float
    max_action_step: float
    action_change_fraction: float
    max_tracking_error: float
    max_clock_step_s: float
    descriptor: list[float]


def anchor_grid(relative: np.ndarray) -> np.ndarray:
    last = relative[-1] - ACTION_SECONDS
    if last < HISTORY_SECONDS:
        return np.empty(0, dtype=np.float64)
    count = int(np.floor((last - HISTORY_SECONDS) / ANCHOR_STRIDE_S)) + 1
    return HISTORY_SECONDS + np.arange(count, dtype=np.float64) * ANCHOR_STRIDE_S


def active_mask(relative: np.ndarray, states: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Score anchors by motion of the measured state inside the future window."""
    active = []
    for anchor in anchors:
        window = (relative >= anchor - 1e-9) & (relative <= anchor + ACTION_SECONDS + 1e-9)
        future = states[window]
        if len(future) < 2:
            active.append(False)
            continue
        joints = float(np.ptp(future[:, :6], axis=0).max())
        gripper = float(np.ptp(future[:, 6]))
        active.append(joints >= JOINT_ACTIVITY_RAD or gripper >= GRIPPER_ACTIVITY)
    return np.asarray(active, dtype=bool)


def scan(metadata_root: Path, staging_root: Path, repo_id: str, output: Path) -> dict:
    source = (staging_root / repo_id.replace("/", "__")).resolve()
    info = read_json(metadata_root / "meta/info.json")
    rows = pq.read_table(
        sorted((metadata_root / "meta/episodes").glob("**/*.parquet")),
        columns=["episode_index", "length", "tasks", "dataset_from_index",
                 "data/chunk_index", "data/file_index"],
    ).to_pylist()
    summaries: list[EpisodeSummary] = []
    by_file: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        by_file.setdefault((int(row["data/chunk_index"]), int(row["data/file_index"])), []).append(row)
    for (chunk, index), group in sorted(by_file.items()):
        path = source / info["data_path"].format(chunk_index=chunk, file_index=index)
        shard = pq.read_table(
            path, columns=["episode_index", "timestamp", "observation.state", "action"]
        )
        base = min(int(item["dataset_from_index"]) for item in group)
        for row in group:
            offset = int(row["dataset_from_index"]) - base
            length = int(row["length"])
            block = shard.slice(offset, length)
            episode = np.asarray(block["episode_index"].to_pylist())
            if not np.all(episode == int(row["episode_index"])):
                raise ValueError(f"Shard slice does not map to episode {row['episode_index']}")
            timestamps = np.asarray(block["timestamp"].to_pylist(), dtype=np.float64)
            states = np.asarray(block["observation.state"].to_pylist(), dtype=np.float64)
            actions = np.asarray(block["action"].to_pylist(), dtype=np.float64)
            relative = timestamps - timestamps[0]
            anchors = anchor_grid(relative)
            active = active_mask(relative, states, anchors) if len(anchors) else np.empty(0, bool)
            state_steps = np.linalg.norm(np.diff(states[:, :6], axis=0), axis=1)
            action_steps = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)
            tasks = [str(item) for item in (row.get("tasks") or []) if str(item).strip()]
            descriptor = np.concatenate([
                states[0, :6], states[-1, :6],
                np.asarray([
                    float(relative[-1]),
                    float(state_steps.sum()),
                    float(np.abs(np.diff(states[:, 6])).sum()),
                    float(active.mean()) if len(active) else 0.0,
                ]),
            ])
            summaries.append(EpisodeSummary(
                episode_index=int(row["episode_index"]),
                frames=length,
                duration_s=float(relative[-1]),
                source_task=tasks[0] if tasks else "",
                candidate_anchors=int(len(anchors)),
                active_anchors=int(active.sum()),
                active_anchor_fraction=float(active.mean()) if len(active) else 0.0,
                state_path_length=float(state_steps.sum()),
                gripper_path_length=float(np.abs(np.diff(states[:, 6])).sum()),
                max_state_step=float(state_steps.max(initial=0.0)),
                max_action_step=float(action_steps.max(initial=0.0)),
                action_change_fraction=float(np.mean(action_steps > 1e-9)),
                max_tracking_error=float(np.linalg.norm(actions[:, :6] - states[:, :6], axis=1).max()),
                max_clock_step_s=float(np.diff(timestamps).max()),
                descriptor=descriptor.tolist(),
            ))
    value = {
        "source": repo_id,
        "episodes_scanned": len(rows),
        "valid_episodes": len(summaries),
        "motion_thresholds": {
            "joint_activity_radians_ptp": JOINT_ACTIVITY_RAD,
            "gripper_activity_ptp": GRIPPER_ACTIVITY,
            "measured_from": "observation.state, because the action is a held waypoint goal",
        },
        "summaries": [asdict(item) for item in summaries],
    }
    write_json(output, value)
    return value


def nominate(scan_path: Path, count: int, output: Path, *, min_active: int) -> dict:
    scanned = read_json(scan_path)
    pool = [item for item in scanned["summaries"] if item["active_anchors"] >= min_active]
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} episodes clear the screen for {count} candidates")
    descriptors = np.asarray([item["descriptor"] for item in pool], dtype=np.float64)
    median = np.median(descriptors, axis=0)
    scale = np.quantile(descriptors, 0.75, axis=0) - np.quantile(descriptors, 0.25, axis=0)
    scale[scale < 1e-9] = 1.0
    descriptors = (descriptors - median) / scale
    selected: list[int] = []
    minimum = np.full(len(pool), np.inf)
    while len(selected) < count:
        if not selected:
            score = np.asarray([item["active_anchors"] for item in pool], dtype=np.float64)
        else:
            minimum = np.minimum(minimum, np.linalg.norm(descriptors - descriptors[selected[-1]], axis=1))
            score = minimum.copy()
        for index in selected:
            score[index] = -np.inf
        selected.append(int(np.argmax(score)))
    chosen = [pool[index] for index in selected]
    value = {
        "source": scanned["source"],
        "component": "stack_block",
        "selection_status": "nominated_pending_visual_review",
        "candidate_count": count,
        "pool_size": len(pool),
        "rejection_screen": {
            "minimum_active_anchors": min_active,
            "activity_measured_from": "observation.state",
            "discontinuity_screen": (
                "not applied: the waypoint action steps by design, so a per-frame action "
                "threshold cannot separate a teleport from normal operation"
            ),
        },
        "grouping": (
            "single scripted task on one physical setup, so there is no robot, scene or operator "
            "group to balance; diversity is execution variation only"
        ),
        "candidates": chosen,
    }
    write_json(output, value)
    return value


def proxies(candidates_path: Path, metadata_root: Path, staging_root: Path, repo_id: str,
            output_root: Path) -> list[Path]:
    manifest = read_json(candidates_path)
    info = read_json(metadata_root / "meta/info.json")
    rows = {int(row["episode_index"]): row for row in pq.read_table(
        sorted((metadata_root / "meta/episodes").glob("**/*.parquet"))).to_pylist()}
    source = (staging_root / repo_id.replace("/", "__")).resolve()
    written: list[Path] = []
    for item in manifest["candidates"]:
        index = int(item["episode_index"])
        row = rows[index]
        stem = output_root / f"episode_{index:06d}"
        for camera in CAMERAS:
            label = camera.rsplit(".", 1)[-1].replace("realsense_", "")
            video = source / info["video_path"].format(
                video_key=camera,
                chunk_index=int(row[f"videos/{camera}/chunk_index"]),
                file_index=int(row[f"videos/{camera}/file_index"]),
            )
            start = float(row[f"videos/{camera}/from_timestamp"])
            duration = float(row[f"videos/{camera}/to_timestamp"]) - start
            header = (f"ep {index}  {duration:.1f}s  {label}  "
                      f"active={item['active_anchors']}/{item['candidate_anchors']}")
            written.append(contact_sheet(video, start, overview_offsets(duration),
                                         stem.with_suffix(f".{label}.jpg"), header=header))
        template = stem.with_suffix(".annotations.json")
        if not template.exists():
            write_json(template, {
                "source_episode_index": index,
                "episode_duration_s": float(item["duration_s"]),
                "review_status": "pending",
                "task": "",
                "source_task": item["source_task"],
                "outcome": "",
                "reviewer_notes": "",
                "segments": [],
                "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
                "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
            })
            written.append(template)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path,
                        default=Path("outputs/diverse_robot_dataset_build/ur7e/metadata/ur7e_stack"))
    parser.add_argument("--staging-root", type=Path,
                        default=Path("outputs/diverse_robot_dataset_build/ur7e/staging"))
    parser.add_argument("--repo-id", default="Cache-SCA/UR7e-CaP-Stack_Block-100epi")
    sub = parser.add_subparsers(dest="command", required=True)
    scanner = sub.add_parser("scan")
    scanner.add_argument("--output", type=Path, required=True)
    nomination = sub.add_parser("nominate")
    nomination.add_argument("--scan", type=Path, required=True)
    nomination.add_argument("--count", type=int, default=8)
    nomination.add_argument("--min-active", type=int, default=3)
    nomination.add_argument("--output", type=Path, required=True)
    proxy = sub.add_parser("proxies")
    proxy.add_argument("--candidates", type=Path, required=True)
    proxy.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scan":
        value = scan(args.metadata_root, args.staging_root, args.repo_id, args.output)
        print(json.dumps({k: v for k, v in value.items() if k != "summaries"}, indent=2))
    elif args.command == "nominate":
        value = nominate(args.scan, args.count, args.output, min_active=args.min_active)
        print("pool %d; candidates:" % value["pool_size"])
        for item in value["candidates"]:
            print("  ep %-4d %5.1fs active %2d/%2d  path %.2f" % (
                item["episode_index"], item["duration_s"], item["active_anchors"],
                item["candidate_anchors"], item["state_path_length"]))
    else:
        written = proxies(args.candidates, args.metadata_root, args.staging_root,
                          args.repo_id, args.output_root)
        print("\n".join(str(path) for path in written))


if __name__ == "__main__":
    main()
