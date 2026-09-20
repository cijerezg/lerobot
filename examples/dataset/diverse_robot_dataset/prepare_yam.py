#!/usr/bin/env python

"""Audit, scan, nominate, render, and finalize the three single-arm YAM sets for v3.

Three LeRobot v3 repos on the Hub (yam_sources.json), one task each: six joint radians plus a
gripper ratio in observation.state and a real commanded action that leads the measurement by
a few ticks. Episodes are short (7-40 s) and each covers the task once, so the review unit is
the whole episode, as for MolmoAct: one keep span carrying the task as its subtask, a quality,
and mistake events. Selection is four episodes per set spread over the recording order.

The three repos do not share one gripper convention (yam-pick-place even stores its measured
gripper as 1 - command), so `polarity` renders dense wrist sheets around the first gripper
transition with the values printed on every tile; the coordinator reads them and sets
gripper_transform per spec in yam_sources.json, which build_corpus.py applies at ingest.

    prepare_yam.py --spec yam_duster audit
    prepare_yam.py --spec yam_duster scan
    prepare_yam.py --spec yam_duster nominate --per-set 4
    prepare_yam.py --spec yam_duster proxies
    prepare_yam.py --spec yam_duster polarity
    # review the sheets, write verdicts.json, then
    prepare_yam.py --spec yam_duster finalize --verdicts .../verdicts.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "lerobot/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "lerobot/src"))
from prepare_droid import _grab_frame, contact_sheet, overview_offsets, read_json, write_json  # noqa: E402

SOURCE_CONFIG = Path(__file__).with_name("yam_sources.json")
SHEETS_ROOT = REPO_ROOT / "outputs/_annotation/diverse_v3_review/yam"
COMPONENTS = {"yam_duster": "duster", "yam_espresso": "espresso", "yam_pick_place": "pick_place"}
# What the frames show, which the source task strings do not all say: the duster is placed in
# the box, the espresso set's "fold napkins" is simply wrong, the pencil string is kept.
TASKS = {
    "yam_duster": "pick up the duster and place it in the red box",
    "yam_espresso": "insert the portafilter into the espresso machine",
    "yam_pick_place": "pick up the pencil from the left sharpener and put it into the right sharpener",
}
GRIPPER = 6
ACTION_SECONDS = 29.0 / 30.0
SCREEN_STRIDE_S = 1.0
# Per one-second cell, the joint peak-to-peak (measured state) is bimodal in all three sets: a
# resting floor at ~4e-4 rad (encoder noise while the arm holds still) and a moving mode from
# ~0.03 rad up, with nothing between 1e-3 and 3e-2. 0.02 rad sits in that gap. The gripper
# rests at ~1e-4 and a deliberate open/close moves >= 0.3, so 0.05 is far from both.
JOINT_ACTIVITY_RAD = 0.02
GRIPPER_ACTIVITY = 0.05
# The largest measured joint step between consecutive frames is 0.19 rad at 25-30 Hz; a
# step above this is a dropped span or a re-indexed take, not motion.
JOINT_STEP_LIMIT_RAD = 0.35
# An episode whose measured gripper never moves this much never grasped (the duster card's
# "aborted without closing the gripper"); every real grasp in the three sets moves >= 0.4.
NO_GRASP_RANGE = 0.2
LAG_MAX_TICKS = 15
# Polarity sheets: dense wrist tiles around the first gripper transition of the measured value.
POLARITY_MOVE = 0.15
POLARITY_STRIDE_S = 0.2
POLARITY_BEFORE_S = 0.8
POLARITY_TILES = 16
POLARITY_COLUMNS = 4
POLARITY_TILE_WIDTH = 400
REVIEWER_MODEL = "claude-fable-5-1"
KEEP_REASONS = {"useful_motion", "informative_mistake", "recovery", "task_required_hold"}
MISTAKE_TYPES = {"failed_close", "slip", "drop", "knock", "wrong_target"}


class Paths:
    def __init__(self, spec_name: str, build_root: Path, sheets_root: Path) -> None:
        from lerobot.datasets.diverse_pilot import load_source_specs

        self.spec = next(item for item in load_source_specs(SOURCE_CONFIG) if item.name == spec_name)
        self.component = COMPONENTS[spec_name]
        self.root = build_root / "yam"
        self.metadata_root = self.root / "metadata" / spec_name
        self.staging_root = self.root / "staging"
        self.source_root = self.staging_root / self.spec.repo_id.replace("/", "__")
        self.review_root = self.root / "review" / self.component
        self.sheets_root = sheets_root / self.component
        self.scan = self.review_root / "episode_scan.json"
        self.candidates = self.review_root / "candidates.json"
        self.audit = self.root / "audits" / f"{spec_name}.json"
        self.index = self.root / "index.json"


def video_keys(info: dict) -> list[str]:
    return [name for name, feature in info["features"].items() if feature.get("dtype") == "video"]


def camera_by_role(info: dict) -> dict[str, str]:
    """{"external": camera key, "wrist": camera key} from the shared yam role map."""
    from lerobot.datasets.diverse_actor_selection import CAMERA_ROLE_MAP

    roles = {}
    for camera in video_keys(info):
        role = CAMERA_ROLE_MAP["yam"][camera.rsplit(".", 1)[-1]]
        roles[role.rsplit("_", 1)[0]] = camera
    return roles


def episode_rows(metadata_root: Path) -> list[dict]:
    files = sorted((metadata_root / "meta/episodes").glob("**/*.parquet"))
    table = pq.read_table(files)
    columns = [name for name in table.column_names if not name.startswith("stats/")]
    return table.select(columns).to_pylist()


def episode_arrays(paths: Paths, info: dict, rows: list[dict], wanted: set[int] | None = None):
    """Yield (row, timestamps, states, actions) per wanted episode, reading each data shard once.

    ``rows`` is the whole episode table: a shard's row offset is relative to the first episode
    stored in that file, wanted or not.
    """
    by_file: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_file[(int(row["data/chunk_index"]), int(row["data/file_index"]))].append(row)
    for (chunk, index), group in sorted(by_file.items()):
        base = min(int(item["dataset_from_index"]) for item in group)
        group = [row for row in group if wanted is None or int(row["episode_index"]) in wanted]
        if not group:
            continue
        path = paths.source_root / info["data_path"].format(chunk_index=chunk, file_index=index)
        shard = pq.read_table(path, columns=["episode_index", "timestamp", "observation.state", "action"])
        for row in group:
            block = shard.slice(int(row["dataset_from_index"]) - base, int(row["length"]))
            episode = np.asarray(block["episode_index"].to_pylist())
            if not np.all(episode == int(row["episode_index"])):
                raise ValueError(f"Shard slice does not map to episode {row['episode_index']}")
            timestamps = np.asarray(block["timestamp"].to_pylist(), dtype=np.float64)
            states = np.asarray(block["observation.state"].to_pylist(), dtype=np.float64)
            actions = np.asarray(block["action"].to_pylist(), dtype=np.float64)
            yield row, timestamps, states, actions


def screen(relative: np.ndarray, states: np.ndarray) -> tuple[list[float], list[bool]]:
    """One-second cells from t=0; a cell is active when its future window moves a joint or the gripper."""
    last = float(relative[-1]) - ACTION_SECONDS
    if last < 0.0:
        return [0.0], [True]
    count = int(math.floor(last / SCREEN_STRIDE_S)) + 1
    anchors = np.arange(count, dtype=np.float64) * SCREEN_STRIDE_S
    active = []
    for anchor in anchors:
        mask = (relative >= anchor - 1e-9) & (relative <= anchor + ACTION_SECONDS + 1e-9)
        future = states[mask]
        if len(future) < 2:
            active.append(False)
            continue
        joints = float(np.ptp(future[:, :GRIPPER], axis=0).max())
        grip = float(np.ptp(future[:, GRIPPER]))
        active.append(joints >= JOINT_ACTIVITY_RAD or grip >= GRIPPER_ACTIVITY)
    return [float(value) for value in anchors], active


def lead_ticks(command: np.ndarray, measured: np.ndarray, max_lag: int = LAG_MAX_TICKS) -> int | None:
    """Ticks by which the command leads the measurement (argmax of the cross-correlation)."""
    command = command - command.mean()
    measured = measured - measured.mean()
    if command.std() < 1e-3 or measured.std() < 1e-3:
        return None
    best = None
    for lag in range(-max_lag, max_lag + 1):
        head, tail = (command[: len(command) - lag], measured[lag:]) if lag >= 0 else (command[-lag:], measured[: len(measured) + lag])
        score = float(np.corrcoef(head, tail)[0, 1])
        if best is None or score > best[1]:
            best = (lag, score)
    return best[0]


def gripper_transitions(gripper: np.ndarray) -> int:
    """Crossings of the episode's own mid-range with hysteresis, whichever way is 'closed'."""
    low, high = float(gripper.min()), float(gripper.max())
    if high - low < NO_GRASP_RANGE:
        return 0
    normalized = (gripper - low) / (high - low)
    side = normalized[0] >= 0.5
    count = 0
    for value in normalized:
        if side and value <= 0.35 or not side and value >= 0.65:
            side = not side
            count += 1
    return count


def video_mismatch(row: dict, info: dict) -> list[str]:
    """Cameras whose episode span in the shard does not hold the episode's frame count."""
    fps = float(info["fps"])
    return [
        camera
        for camera in video_keys(info)
        if round((float(row[f"videos/{camera}/to_timestamp"]) - float(row[f"videos/{camera}/from_timestamp"])) * fps) != int(row["length"])
    ]


def exclusions(row: dict, info: dict, states: np.ndarray, joint_steps: np.ndarray) -> list[str]:
    reasons = []
    if video_mismatch(row, info):
        reasons.append("video_data_mismatch")
    if row.get("episode_success") is False:
        reasons.append("source_flagged_failure")
    if float(np.ptp(states[:, GRIPPER])) < NO_GRASP_RANGE:
        reasons.append("no_grasp")
    if float(joint_steps.max(initial=0.0)) > JOINT_STEP_LIMIT_RAD:
        reasons.append("discontinuous")
    return reasons


def fetch_local_metadata(paths: Paths) -> Path:
    """The staging mirror is a complete Hub snapshot; copy its metadata instead of re-downloading."""
    for pattern in paths.spec.metadata_patterns:
        for path in paths.source_root.glob(pattern):
            target = paths.metadata_root / path.relative_to(paths.source_root)
            if path.is_dir():
                shutil.copytree(path, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return paths.metadata_root


def audit(paths: Paths) -> dict:
    from huggingface_hub import HfApi

    from lerobot.datasets.diverse_pilot import audit_source

    fetch_local_metadata(paths)
    repo_info = HfApi().dataset_info(paths.spec.repo_id, revision=paths.spec.revision)
    value = audit_source(
        paths.spec,
        paths.metadata_root,
        resolved_revision=repo_info.sha,
        repo_tags=list(repo_info.tags or []),
        gated=repo_info.gated,
    )
    # snapshot_download stamps the commit it mirrored into every file's .metadata sidecar.
    sidecar = paths.source_root / ".cache/huggingface/download/meta/info.json.metadata"
    value["repository"]["staged_commit"] = sidecar.read_text().splitlines()[0]
    write_json(paths.audit, value)
    return value


def scan(paths: Paths) -> dict:
    info = read_json(paths.metadata_root / "meta/info.json")
    fps = float(info["fps"])
    rows = episode_rows(paths.metadata_root)
    summaries = []
    for row, timestamps, states, actions in episode_arrays(paths, info, rows):
        relative = timestamps - timestamps[0]
        anchors, active = screen(relative, states)
        joint_steps = np.abs(np.diff(states[:, :GRIPPER], axis=0)).max(axis=1)
        tasks = [str(item) for item in (row.get("tasks") or []) if str(item).strip()]
        summaries.append(
            {
                "episode_index": int(row["episode_index"]),
                "frames": int(row["length"]),
                "duration_s": float(relative[-1]),
                "fps": fps,
                "measured_fps": float(1.0 / np.median(np.diff(timestamps))),
                "task": tasks[0] if tasks else "",
                "annotated_task": TASKS[paths.spec.name],
                "source_success": row.get("episode_success"),
                "screen_cells": len(anchors),
                "active_cells": int(sum(active)),
                "active_fraction": float(np.mean(active)) if active else 0.0,
                "joint_path_length_rad": float(np.abs(np.diff(states[:, :GRIPPER], axis=0)).sum()),
                "max_joint_step_rad": float(joint_steps.max(initial=0.0)),
                "max_action_step_rad": float(np.abs(np.diff(actions[:, :GRIPPER], axis=0)).max(initial=0.0)),
                "max_clock_step_s": float(np.diff(timestamps).max()),
                "state_gripper_range": [float(states[:, GRIPPER].min()), float(states[:, GRIPPER].max())],
                "action_gripper_range": [float(actions[:, GRIPPER].min()), float(actions[:, GRIPPER].max())],
                "state_gripper_start": float(states[0, GRIPPER]),
                "action_gripper_start": float(actions[0, GRIPPER]),
                "gripper_transitions": gripper_transitions(states[:, GRIPPER]),
                "action_lead_ticks": [lead_ticks(actions[:, j], states[:, j]) for j in range(GRIPPER + 1)],
                "video_mismatch": video_mismatch(row, info),
                "excluded": exclusions(row, info, states, joint_steps),
            }
        )
    excluded: dict[str, list[int]] = defaultdict(list)
    for item in summaries:
        for reason in item["excluded"]:
            excluded[reason].append(item["episode_index"])
    leads = np.asarray([item["action_lead_ticks"] for item in summaries], dtype=np.float64)
    value = {
        "source": paths.spec.repo_id,
        "component": paths.component,
        "episodes_scanned": len(rows),
        "excluded": dict(sorted(excluded.items())),
        "pool_size": sum(1 for item in summaries if not item["excluded"] and item["active_cells"] >= 2),
        "measured": {
            "fps_median": float(np.median([item["measured_fps"] for item in summaries])),
            "action_lead_ticks_median": [None if np.isnan(v) else float(v) for v in np.nanmedian(leads, axis=0)],
            "state_gripper_min": float(min(item["state_gripper_range"][0] for item in summaries)),
            "state_gripper_max": float(max(item["state_gripper_range"][1] for item in summaries)),
            "action_gripper_min": float(min(item["action_gripper_range"][0] for item in summaries)),
            "action_gripper_max": float(max(item["action_gripper_range"][1] for item in summaries)),
        },
        "screen": {
            "stride_s": SCREEN_STRIDE_S,
            "joint_activity_rad": JOINT_ACTIVITY_RAD,
            "gripper_activity": GRIPPER_ACTIVITY,
            "joint_step_limit_rad": JOINT_STEP_LIMIT_RAD,
            "no_grasp_range": NO_GRASP_RANGE,
            "measured_from": "observation.state (the commanded action leads it by a few ticks)",
        },
        "summaries": summaries,
    }
    write_json(paths.scan, value)
    return value


def nominate(paths: Paths, per_set: int) -> dict:
    scanned = read_json(paths.scan)
    pool = sorted(
        (item for item in scanned["summaries"] if not item["excluded"] and item["active_cells"] >= 2),
        key=lambda item: item["episode_index"],
    )
    # Spread the picks over the set's recording order so two episodes are unlikely to come
    # from the same stretch of the session, then take the most active one inside each slice.
    chosen = [
        max((pool[int(i)] for i in chunk), key=lambda item: (item["active_fraction"], item["joint_path_length_rad"]))
        for chunk in np.array_split(np.arange(len(pool)), min(per_set, len(pool)))
        if len(chunk)
    ]
    value = {
        "source": scanned["source"],
        "component": paths.component,
        "selection_status": "nominated_pending_visual_review",
        "per_set": per_set,
        "pool_size": len(pool),
        "candidate_count": len(chosen),
        "rejection_screen": {
            "excluded": scanned["excluded"],
            "joint_step_limit_rad": JOINT_STEP_LIMIT_RAD,
            "minimum_active_cells": 2,
        },
        "grouping": "one physical YAM per set and one task per set, so the quota is per set spread over recording order",
        "candidates": chosen,
    }
    write_json(paths.candidates, value)
    return value


def episode_video(row: dict, info: dict, camera: str, paths: Paths) -> tuple[Path, float, float]:
    video = paths.source_root / info["video_path"].format(
        video_key=camera,
        chunk_index=int(row[f"videos/{camera}/chunk_index"]),
        file_index=int(row[f"videos/{camera}/file_index"]),
    )
    start = float(row[f"videos/{camera}/from_timestamp"])
    return video, start, float(row[f"videos/{camera}/to_timestamp"]) - start


def trace_sheet(
    relative: np.ndarray, states: np.ndarray, actions: np.ndarray, tile_times: list[float], destination: Path, *, header: str, joint_names: list[str]
) -> Path:
    """Gripper and six joint traces (measured and commanded) with the sheet's tile times marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(GRIPPER + 1, 1, figsize=(12, 15), sharex=True)
    order = [GRIPPER, *range(GRIPPER)]
    for axis, dim in zip(axes, order, strict=True):
        axis.plot(relative, actions[:, dim], color="tab:orange", linewidth=1.0, label="commanded")
        axis.plot(relative, states[:, dim], color="tab:blue", linewidth=1.2, label="measured")
        for number, tile_time in enumerate(tile_times):
            axis.axvline(tile_time, color="0.6", linestyle="--", linewidth=0.8)
            if dim == GRIPPER:
                axis.text(tile_time, axis.get_ylim()[1], str(number), fontsize=8, ha="center", va="bottom", color="0.3")
        axis.set_ylabel(joint_names[dim] if dim != GRIPPER else f"{joint_names[dim]} (gripper)")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("episode time (s)")
    figure.suptitle(header, fontsize=11)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination


def proxies(paths: Paths) -> list[Path]:
    info = read_json(paths.metadata_root / "meta/info.json")
    roles = camera_by_role(info)
    candidates = {int(item["episode_index"]): item for item in read_json(paths.candidates)["candidates"]}
    rows = episode_rows(paths.metadata_root)
    written = []
    for row, timestamps, states, actions in episode_arrays(paths, info, rows, set(candidates)):
        index = int(row["episode_index"])
        item = candidates[index]
        stem = paths.sheets_root / f"episode_{index:06d}"
        tile_times = None
        for role, camera in sorted(roles.items()):
            video, start, duration = episode_video(row, info, camera, paths)
            tile_times = overview_offsets(duration)
            label = camera.rsplit(".", 1)[-1]
            header = f"ep {index}  {duration:.1f}s  {role} ({label})  {item['annotated_task'][:70]}"
            destination = stem.with_suffix(f".{role}.jpg")
            if not destination.exists():
                contact_sheet(video, start, tile_times, destination, header=header)
            written.append(destination)
        destination = stem.with_suffix(".traces.png")
        if not destination.exists():
            trace_sheet(
                timestamps - timestamps[0], states, actions, tile_times, destination,
                header=f"ep {index}  {paths.component}  measured vs commanded; dashed = tile times of the camera sheets",
                joint_names=list(paths.spec.joint_names),
            )
        written.append(destination)
    return written


def labelled_sheet(video: Path, shard_start_s: float, offsets_s: list[float], labels: list[str], destination: Path, *, header: str) -> Path:
    """A contact sheet whose tile captions carry the gripper values, not only the time."""
    from PIL import Image, ImageDraw, ImageFont

    from prepare_droid import FONT_PATH, LABEL_HEIGHT

    font = ImageFont.truetype(FONT_PATH, 15)
    header_font = ImageFont.truetype(FONT_PATH, 18)
    tiles = [_grab_frame(video, shard_start_s + offset, POLARITY_TILE_WIDTH) for offset in offsets_s]
    tile_height = max(tile.height for tile in tiles)
    rows = int(math.ceil(len(tiles) / POLARITY_COLUMNS))
    sheet = Image.new("RGB", (POLARITY_COLUMNS * POLARITY_TILE_WIDTH, LABEL_HEIGHT + rows * (tile_height + LABEL_HEIGHT)), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 3), header, fill=(255, 255, 255), font=header_font)
    for index, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        x = (index % POLARITY_COLUMNS) * POLARITY_TILE_WIDTH
        y = LABEL_HEIGHT + (index // POLARITY_COLUMNS) * (tile_height + LABEL_HEIGHT)
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + tile_height + 2), label, fill=(255, 220, 120), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)
    return destination


def first_transition_s(relative: np.ndarray, gripper: np.ndarray) -> float:
    moved = np.flatnonzero(np.abs(gripper - gripper[0]) >= POLARITY_MOVE)
    return float(relative[moved[0]]) if len(moved) else 0.0


def polarity(paths: Paths, count: int, role: str) -> list[Path]:
    info = read_json(paths.metadata_root / "meta/info.json")
    camera = camera_by_role(info)[role]
    candidates = [int(item["episode_index"]) for item in read_json(paths.candidates)["candidates"]][:count]
    rows = episode_rows(paths.metadata_root)
    written = []
    for row, timestamps, states, actions in episode_arrays(paths, info, rows, set(candidates)):
        index = int(row["episode_index"])
        relative = timestamps - timestamps[0]
        video, start, duration = episode_video(row, info, camera, paths)
        first = first_transition_s(relative, states[:, GRIPPER])
        offsets = [round(first - POLARITY_BEFORE_S + i * POLARITY_STRIDE_S, 3) for i in range(POLARITY_TILES)]
        offsets = [min(max(offset, 0.0), duration - 0.05) for offset in offsets]
        frames = np.searchsorted(relative, offsets, side="left").clip(0, len(relative) - 1)
        labels = [
            f"t={offset:6.2f}s  measured g={states[frame, GRIPPER]:.2f}  cmd g={actions[frame, GRIPPER]:.2f}"
            for offset, frame in zip(offsets, frames, strict=True)
        ]
        header = (
            f"ep {index}  {paths.component}  {camera.rsplit('.', 1)[-1]}  first gripper move at {first:.2f}s  "
            f"start g={states[0, GRIPPER]:.2f}  range [{states[:, GRIPPER].min():.2f}, {states[:, GRIPPER].max():.2f}]"
        )
        destination = paths.sheets_root / "polarity" / f"episode_{index:06d}.{role}.jpg"
        if not destination.exists():
            labelled_sheet(video, start, offsets, labels, destination, header=header)
        written.append(destination)
    return written


def segments_for(review: dict, relative: np.ndarray, states: np.ndarray) -> list[dict]:
    """Reviewer spans subdivided by the activity screen; static cells become rejects."""
    duration = float(relative[-1])
    anchors, active = screen(relative, states)
    boundaries = [*anchors, duration]
    spans = sorted(review["spans"], key=lambda span: float(span["start_s"]))
    cursor = 0.0
    for span in spans:
        if abs(float(span["start_s"]) - cursor) > 1e-6 or float(span["end_s"]) <= float(span["start_s"]):
            raise ValueError("Reviewer spans must be contiguous and ordered")
        cursor = float(span["end_s"])
    if abs(cursor - duration) > 0.05:
        raise ValueError(f"Reviewer spans end at {cursor:.3f}s, episode lasts {duration:.3f}s")
    spans[-1]["end_s"] = duration
    cuts = sorted({0.0, duration, *boundaries, *[float(span["start_s"]) for span in spans]})
    segments = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end - start < 1e-6:
            continue
        span = next(item for item in spans if float(item["start_s"]) <= start < float(item["end_s"]))
        cell = max(i for i, anchor in enumerate(anchors) if anchor <= start + 1e-9)
        segment = {"start_s": round(start, 6), "end_s": round(end, 6), "retention": span["retention"], "retention_reason": span["retention_reason"]}
        if span["retention"] == "keep":
            if not active[cell] and span["retention_reason"] != "task_required_hold":
                segment.update(retention="reject", retention_reason="static")
            else:
                segment.update(subtask=span["subtask"], quality=int(span["quality"]),
                               mistake_events=[event for event in span.get("mistake_events", []) if float(event["start_s"]) < end and float(event["end_s"]) > start])
        segments.append(segment)
    merged: list[dict] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        same = previous is not None and all(previous.get(k) == segment.get(k) for k in ("retention", "retention_reason", "subtask", "quality")) and previous.get("mistake_events", []) == segment.get("mistake_events", [])
        if same:
            previous["end_s"] = segment["end_s"]
        else:
            merged.append(segment)
    return merged


def validate_verdicts(verdicts: dict, candidates: dict[int, dict]) -> list[int]:
    """Every nominee needs an explicit verdict; accepted ones need a valid quality and events."""
    expected = {str(index) for index in candidates}
    supplied = set(verdicts["episodes"])
    if supplied != expected:
        raise ValueError(
            "Finalization requires a verdict for every nominated candidate; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}. "
            "Omitted candidates are unreviewed, not rejected. No output was written."
        )
    for index, verdict in verdicts["episodes"].items():
        if not isinstance(verdict.get("accept"), bool):
            raise ValueError(f"episode {index}: accept must be an explicit boolean")
        if not verdict["accept"]:
            continue
        quality = int(verdict["quality"])
        if not 1 <= quality <= 5:
            raise ValueError(f"episode {index}: quality {quality} outside 1..5")
        for event in verdict.get("mistake_events", []):
            if event["kind"] not in MISTAKE_TYPES or float(event["end_s"]) <= float(event["start_s"]):
                raise ValueError(f"episode {index}: bad mistake event {event}")
        keep_reason = verdict.get("retention_reason", "informative_mistake" if verdict.get("mistake_events") else "useful_motion")
        if keep_reason not in KEEP_REASONS:
            raise ValueError(f"episode {index}: bad keep reason {keep_reason}")
    return sorted(int(k) for k, v in verdicts["episodes"].items() if v["accept"])


def finalize(paths: Paths, verdicts_path: Path, review_prompt: str) -> dict:
    """Write reviews, annotations, selection, and the acquisition manifest for accepted episodes."""
    from lerobot.datasets.diverse_pilot import resolve_lerobot_payload

    verdicts = read_json(verdicts_path)
    candidates = {int(item["episode_index"]): item for item in read_json(paths.candidates)["candidates"]}
    accepted = validate_verdicts(verdicts, candidates)
    info = read_json(paths.metadata_root / "meta/info.json")
    rows = episode_rows(paths.metadata_root)
    report = {}
    for row, timestamps, states, actions in episode_arrays(paths, info, rows, set(accepted)):
        index = int(row["episode_index"])
        verdict = verdicts["episodes"][str(index)]
        item = candidates[index]
        quality = int(verdict["quality"])
        events = verdict.get("mistake_events", [])
        relative = timestamps - timestamps[0]
        duration = float(relative[-1])
        keep_reason = verdict.get("retention_reason", "informative_mistake" if events else "useful_motion")
        task_text = verdict.get("task") or item["annotated_task"]
        review = {
            "source_episode_index": index,
            "task": task_text,
            "source_task": item["task"],
            "outcome": verdict.get("outcome", "success"),
            "reviewer_notes": verdict.get("notes", ""),
            "spans": [
                {
                    "start_s": 0.0, "end_s": duration, "retention": "keep", "retention_reason": keep_reason,
                    "subtask": task_text, "quality": quality,
                    "mistake_events": events,
                }
            ],
        }
        write_json(paths.review_root / f"episode_{index:06d}.review.json", review)
        segments = segments_for(review, relative, states)
        annotations = {
            "source_episode_index": index,
            "task": review["task"],
            "source_task": item["task"],
            "outcome": review["outcome"],
            "reviewer_notes": review["reviewer_notes"],
            "episode_duration_s": duration,
            "review_status": "validated",
            "review_basis": "twelve-tile contact sheets of the external and wrist cameras plus the measured/commanded trace sheet over the whole episode; model review under the ReBot rubric",
            "review_provenance": "model_reviewed",
            "quality_provenance": "model_reviewed_rebot_rubric",
            "reviewer_model": verdict.get("reviewer") or verdicts.get("reviewer") or REVIEWER_MODEL,
            "activity_screen_subdivisions": {"stride_s": SCREEN_STRIDE_S, "joint_activity_rad": JOINT_ACTIVITY_RAD, "gripper_activity": GRIPPER_ACTIVITY},
            "source_keep_intervals_s": [],
            "segments": segments,
            "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
            "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
        }
        write_json(paths.review_root / f"episode_{index:06d}.annotations.json", annotations)
        kept = [s for s in segments if s["retention"] == "keep"]
        report[index] = {"quality": quality, "mistake_events": len(events), "kept_s": round(sum(s["end_s"] - s["start_s"] for s in kept), 2), "segments": len(segments)}
    selection = {
        "source": paths.spec.repo_id,
        "component": paths.component,
        "selection_status": "visually_accepted",
        "accepted_episode_indices": accepted,
        "rejected_candidate_indices": sorted(set(candidates) - set(accepted)),
        "review_basis": "twelve-tile contact sheets of the external and wrist cameras plus the trace sheet over the whole episode",
        "review_provenance": "model_reviewed",
        "reviewer_model": verdicts.get("reviewer") or REVIEWER_MODEL,
        "review_date": verdicts.get("review_date"),
        "review_prompt": review_prompt,
        "selection_rule": f"up to {read_json(paths.candidates)['per_set']} episodes per set spread over recording order; a nominee is rejected only on a visible quality or scope failure",
        "accepted": {str(index): {"task": verdicts["episodes"][str(index)].get("task") or candidates[index]["annotated_task"], "source_task": candidates[index]["task"], "outcome": verdicts["episodes"][str(index)].get("outcome", "success"), "notes": verdicts["episodes"][str(index)].get("notes", "")} for index in accepted},
        "rejections": {str(k): v.get("notes", "") for k, v in verdicts["episodes"].items() if not v.get("accept")},
    }
    write_json(paths.review_root / "selection.json", selection)
    manifest = resolve_lerobot_payload(paths.spec, read_json(paths.audit), paths.metadata_root, accepted)
    manifest["source_format"] = "lerobot_v3"
    write_json(paths.review_root / "acquisition_manifest.json", manifest)
    index = read_json(paths.index) if paths.index.is_file() else {"dataset": "diverse_robot_dataset_v3", "source": "yam", "components": []}
    entry = {
        "component": paths.component,
        "spec_name": paths.spec.name,
        "config_name": SOURCE_CONFIG.name,
        "embodiment": "YAM",
        "status": "validated",
        "source_episodes": len(accepted),
        "dataset_root": f"review/{paths.component}",
        "manifest_path": f"review/{paths.component}/acquisition_manifest.json",
        "selection_path": f"review/{paths.component}/selection.json",
        "annotations_root": f"review/{paths.component}",
        "review_round": "v3",
        "quality_values": sorted({int(v["quality"]) for v in report.values()}),
        "mistake_events": sum(v["mistake_events"] for v in report.values()),
    }
    index["components"] = [item for item in index["components"] if item["component"] != paths.component] + [entry]
    write_json(paths.index, index)
    return {"accepted": len(accepted), "rejected": len(selection["rejected_candidate_indices"]), "episodes": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, choices=sorted(COMPONENTS))
    parser.add_argument("--build-root", type=Path, default=Path("outputs/diverse_robot_dataset_v3_build"))
    parser.add_argument("--sheets-dir", type=Path, default=SHEETS_ROOT, help="review sheets root; the component name is appended")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("scan")
    nomination = sub.add_parser("nominate")
    nomination.add_argument("--per-set", type=int, default=4)
    sub.add_parser("proxies")
    polarity_parser = sub.add_parser("polarity")
    polarity_parser.add_argument("--count", type=int, default=3)
    polarity_parser.add_argument("--role", default="wrist", choices=["wrist", "external"])
    final = sub.add_parser("finalize")
    final.add_argument("--verdicts", type=Path, required=True)
    final.add_argument("--review-prompt", default="ReBot rubric: quality 1-5 over the whole episode; mistakes only as bounded failed_close/slip/drop/knock/wrong_target events; reject on uncontrolled motion, human intervention, severe occlusion, or an unverifiable outcome; unclear = reject.")
    args = parser.parse_args()
    paths = Paths(args.spec, args.build_root, args.sheets_dir)
    if args.command == "audit":
        value = audit(paths)
        print(json.dumps({"repository": value["repository"], "admission": value["admission"]}, indent=2))
    elif args.command == "scan":
        value = scan(paths)
        print(json.dumps({k: v for k, v in value.items() if k != "summaries"}, indent=2))
    elif args.command == "nominate":
        value = nominate(paths, args.per_set)
        print(json.dumps({k: v for k, v in value.items() if k != "candidates"}, indent=2))
        print("nominees:", [item["episode_index"] for item in value["candidates"]])
    elif args.command == "proxies":
        written = proxies(paths)
        print(f"{len(written)} sheets under {paths.sheets_root}")
    elif args.command == "polarity":
        written = polarity(paths, args.count, args.role)
        print(f"{len(written)} polarity sheets under {paths.sheets_root / 'polarity'}")
    else:
        print(json.dumps(finalize(paths, args.verdicts, args.review_prompt), indent=2))


if __name__ == "__main__":
    main()
