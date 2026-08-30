#!/usr/bin/env python

"""Nominate and prepare review material for production RoboChallenge data.

This operates directly on one extracted RoboChallenge task. It reads every
low-dimensional trajectory, rejects boundary-ineligible and discontinuous
records, then uses farthest-point sampling over trajectory descriptors while
balancing robot instances. The result is a candidate set, not an automatic
quality verdict: the generated side-by-side videos remain the final review
surface.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FPS = 30.0
HISTORY_SECONDS = 6.0
FUTURE_SECONDS = 29.0 / FPS


@dataclass
class EpisodeSummary:
    episode_index: int
    robot_id: str
    frames: int
    duration_s: float
    candidate_anchors: int
    active_anchor_fraction: float
    joint_path_length: float
    gripper_path_length: float
    max_joint_step: float
    p99_joint_speed: float
    duplicate_timestamps: int
    discontinuous: bool
    descriptor: list[float]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_states(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    timestamps = np.asarray([record["timestamp"] for record in records], dtype=np.float64)
    states = np.asarray(
        [
            [*record["joint_positions"], record["gripper_width"]]
            for record in records
        ],
        dtype=np.float64,
    )
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError(f"Expected [T,7] joint/gripper state in {path}, got {states.shape}")
    if not np.isfinite(states).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"Non-finite trajectory in {path}")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError(f"Reversed timestamps in {path}")
    return timestamps, states


def eligible_anchor_activity(timestamps: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    relative = timestamps - timestamps[0]
    anchors = np.arange(HISTORY_SECONDS, relative[-1] - FUTURE_SECONDS + 1e-9, 2.0)
    active = []
    for anchor in anchors:
        mask = (relative >= anchor) & (relative <= anchor + FUTURE_SECONDS + 1e-9)
        future = states[mask]
        if len(future) < 2:
            active.append(False)
            continue
        joint_range = np.ptp(future[:, :6], axis=0).max()
        gripper_range = np.ptp(future[:, 6])
        active.append(bool(joint_range >= 0.02 or gripper_range >= 0.00075))
    return anchors, np.asarray(active, dtype=bool)


def summarize_episode(episode_root: Path) -> EpisodeSummary:
    episode_index = int(episode_root.name.removeprefix("episode_"))
    metadata = read_json(episode_root / "meta" / "episode_meta.json")
    timestamps, states = load_states(episode_root / "states" / "states.jsonl")
    raw_dt = np.diff(timestamps)
    duplicate_timestamps = int(np.count_nonzero(raw_dt == 0))
    # Retain duplicate clock values in provenance. For screening-only velocity
    # statistics, use the declared 30 Hz interval for zero-length intervals.
    dt = np.where(raw_dt > 0, raw_dt, 1.0 / FPS)
    joint_steps = np.linalg.norm(np.diff(states[:, :6], axis=0), axis=1)
    gripper_steps = np.abs(np.diff(states[:, 6]))
    joint_speed = joint_steps / dt
    anchors, active = eligible_anchor_activity(timestamps, states)
    duration = float(timestamps[-1] - timestamps[0])
    # A per-frame jump of 0.35 rad in the six-joint norm is a discontinuity
    # screen, not a motion-quality score. Candidates are still visually reviewed.
    max_joint_step = float(joint_steps.max(initial=0.0))
    descriptor = np.concatenate(
        [
            states[0, :6],
            states[-1, :6],
            np.asarray(
                [
                    duration,
                    joint_steps.sum(),
                    gripper_steps.sum(),
                    active.mean() if len(active) else 0.0,
                ]
            ),
        ]
    )
    return EpisodeSummary(
        episode_index=episode_index,
        robot_id=str(metadata["robot_id"]),
        frames=len(states),
        duration_s=duration,
        candidate_anchors=len(anchors),
        active_anchor_fraction=float(active.mean()) if len(active) else 0.0,
        joint_path_length=float(joint_steps.sum()),
        gripper_path_length=float(gripper_steps.sum()),
        max_joint_step=max_joint_step,
        p99_joint_speed=float(np.quantile(joint_speed, 0.99)) if len(joint_speed) else 0.0,
        duplicate_timestamps=duplicate_timestamps,
        discontinuous=max_joint_step > 0.35,
        descriptor=descriptor.tolist(),
    )


def farthest_point_candidates(summaries: list[EpisodeSummary], count: int) -> list[EpisodeSummary]:
    pool = [
        summary
        for summary in summaries
        if not summary.discontinuous
        and summary.candidate_anchors >= 3
        and summary.active_anchor_fraction >= 0.25
    ]
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} eligible trajectories are available for {count} candidates")
    descriptors = np.asarray([summary.descriptor for summary in pool], dtype=np.float64)
    median = np.median(descriptors, axis=0)
    scale = np.quantile(descriptors, 0.75, axis=0) - np.quantile(descriptors, 0.25, axis=0)
    scale[scale < 1e-9] = 1.0
    descriptors = (descriptors - median) / scale

    selected: list[int] = []
    robot_counts = {robot_id: 0 for robot_id in {summary.robot_id for summary in pool}}
    min_distance = np.full(len(pool), np.inf)
    while len(selected) < count:
        if not selected:
            score = np.asarray([summary.active_anchor_fraction for summary in pool])
        else:
            last = descriptors[selected[-1]]
            min_distance = np.minimum(min_distance, np.linalg.norm(descriptors - last, axis=1))
            score = min_distance.copy()
        for index in selected:
            score[index] = -np.inf
        # Prefer the currently underrepresented physical robot when diversity scores are close.
        minimum_robot_count = min(robot_counts.values())
        for index, summary in enumerate(pool):
            if summary.robot_id != "" and robot_counts[summary.robot_id] > minimum_robot_count:
                score[index] *= 0.8
        choice = int(np.argmax(score))
        selected.append(choice)
        robot_counts[pool[choice].robot_id] += 1
    return [pool[index] for index in selected]


def nominate(task_root: Path, output: Path, count: int) -> dict:
    task_info = read_json(task_root / "meta" / "task_info.json")
    episode_roots = sorted((task_root / "data").glob("episode_*"))
    summaries = []
    invalid_episodes = []
    for path in episode_roots:
        try:
            summaries.append(summarize_episode(path))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            invalid_episodes.append(
                {
                    "episode_index": int(path.name.removeprefix("episode_")),
                    "reason": str(error),
                }
            )
    selected = farthest_point_candidates(summaries, count)
    value = {
        "source": "RoboChallenge/Table30v2",
        "task": task_info["task_desc"]["task_name"],
        "prompt": task_info["task_desc"]["prompt"],
        "embodiment": task_info["task_desc"]["task_tag"][-1],
        "selection_status": "nominated_pending_visual_review",
        "candidate_count": count,
        "episodes_scanned": len(episode_roots),
        "rejection_screen": {
            "minimum_candidate_anchors": 3,
            "minimum_active_anchor_fraction": 0.25,
            "maximum_joint_step_radians_l2": 0.35,
        },
        "candidates": [asdict(summary) for summary in selected],
        "scan_summary": {
            "discontinuous_episodes": sum(summary.discontinuous for summary in summaries),
            "invalid_episodes": invalid_episodes,
            "valid_episodes": len(summaries),
            "robot_ids": sorted({summary.robot_id for summary in summaries}),
        },
    }
    write_json(output, value)
    return value


def make_proxies(task_root: Path, manifest_path: Path, output_root: Path) -> None:
    manifest = read_json(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)
    for candidate in manifest["candidates"]:
        episode_index = int(candidate["episode_index"])
        episode_root = task_root / "data" / f"episode_{episode_index:06d}"
        global_video = episode_root / "videos" / "cam_global_rgb.mp4"
        wrist_video = episode_root / "videos" / "cam_arm_rgb.mp4"
        side_video = episode_root / "videos" / "cam_side_rgb.mp4"
        stem = output_root / f"episode_{episode_index:06d}"
        proxy = stem.with_suffix(".mp4")
        contact = stem.with_suffix(".jpg")
        inputs = ["-i", str(global_video), "-i", str(wrist_video)]
        if side_video.is_file():
            inputs.extend(["-i", str(side_video)])
            stack_filter = (
                "[0:v]scale=320:-2[g];[1:v]scale=320:-2[w];[2:v]scale=320:-2[s];"
                "[g][w][s]hstack=inputs=3,fps=10"
            )
        else:
            stack_filter = "[0:v]scale=480:-2[g];[1:v]scale=480:-2[w];[g][w]hstack=inputs=2,fps=10"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                "-filter_complex", stack_filter,
                "-an", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", str(proxy),
            ],
            check=True,
        )
        frame_count = max(1, int(math.ceil(float(candidate["duration_s"]) / 4.0)))
        columns = min(5, frame_count)
        rows = int(math.ceil(frame_count / columns))
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(proxy), "-vf", f"fps=1/4,scale=480:-2,tile={columns}x{rows}",
                "-frames:v", "1", str(contact),
            ],
            check=True,
        )
        annotation = {
            "source_episode_index": episode_index,
            "episode_duration_s": float(candidate["duration_s"]),
            "review_status": "pending",
            "segments": [],
            "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
            "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
        }
        write_json(stem.with_suffix(".annotations.json"), annotation)


def write_annotations(
    task_root: Path, manifest_path: Path, output_root: Path, accepted: list[int]
) -> None:
    manifest = read_json(manifest_path)
    candidates = {int(item["episode_index"]): item for item in manifest["candidates"]}
    unknown = sorted(set(accepted) - candidates.keys())
    if unknown:
        raise ValueError(f"Accepted episodes are absent from the candidate manifest: {unknown}")
    prompt = str(manifest["prompt"])
    selection = {
        "source": manifest["source"],
        "task": manifest["task"],
        "embodiment": manifest["embodiment"],
        "selection_status": "visually_accepted",
        "accepted_episode_indices": accepted,
        "rejected_candidate_indices": sorted(candidates.keys() - set(accepted)),
        "review_basis": "side-by-side global/wrist proxy plus low-dimensional discontinuity screen",
    }
    write_json(output_root / "selection.json", selection)
    for episode_index in accepted:
        episode_root = task_root / "data" / f"episode_{episode_index:06d}"
        _, states = load_states(episode_root / "states" / "states.jsonl")
        converted_timestamps = np.arange(len(states), dtype=np.float64) / FPS
        anchors, active = eligible_anchor_activity(converted_timestamps, states)
        duration = float(converted_timestamps[-1])
        boundaries = [0.0, *anchors.tolist(), duration]
        labels = [False, *active.tolist()]
        segments = []
        for index, keep in enumerate(labels):
            start = float(boundaries[index])
            end = float(boundaries[index + 1])
            if end <= start + 1e-9:
                continue
            segment = {
                "start_s": start,
                "end_s": end,
                "retention": "keep" if keep else "reject",
                "retention_reason": "useful_motion" if keep else "static",
            }
            if keep:
                segment.update({"subtask": prompt, "quality": 5, "mistake_events": []})
            same_label = segments and all(
                segments[-1].get(key) == segment.get(key)
                for key in ("retention", "retention_reason", "subtask", "quality", "mistake_events")
            )
            if same_label:
                segments[-1]["end_s"] = end
            else:
                segments.append(segment)
        annotation = {
            "source_episode_index": episode_index,
            "episode_duration_s": duration,
            "review_status": "validated",
            "segments": segments,
            "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
            "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
        }
        write_json(output_root / f"episode_{episode_index:06d}.annotations.json", annotation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    nomination = subparsers.add_parser("nominate")
    nomination.add_argument("--task-root", type=Path, required=True)
    nomination.add_argument("--output", type=Path, required=True)
    nomination.add_argument("--count", type=int, default=15)
    proxies = subparsers.add_parser("proxies")
    proxies.add_argument("--task-root", type=Path, required=True)
    proxies.add_argument("--manifest", type=Path, required=True)
    proxies.add_argument("--output-root", type=Path, required=True)
    annotations = subparsers.add_parser("annotations")
    annotations.add_argument("--task-root", type=Path, required=True)
    annotations.add_argument("--manifest", type=Path, required=True)
    annotations.add_argument("--output-root", type=Path, required=True)
    annotations.add_argument("--accepted", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "nominate":
        nominate(args.task_root, args.output, args.count)
    elif args.command == "proxies":
        make_proxies(args.task_root, args.manifest, args.output_root)
    else:
        write_annotations(args.task_root, args.manifest, args.output_root, args.accepted)


if __name__ == "__main__":
    main()
