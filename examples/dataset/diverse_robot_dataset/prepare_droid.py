#!/usr/bin/env python

"""Nominate and prepare review material for production DROID components.

DROID is already LeRobot v3 on the Hub, so acquisition is a two-stage payload
problem rather than a tar extraction: every episode's low-dimensional record
lives in one of 18 shared data shards (77 MB each, ~766 episodes per shard),
while its three 720x1280 h264 cameras live in three separate ~203 MB video
shards. This script therefore downloads the complete low-dimensional collection
once, screens and nominates against real commanded-action trajectories, and only
then resolves the much larger video payload for the nominated candidates.

The four selection lessons from the RoboChallenge build carry over:

* balance is a hard quota, not a score penalty. DROID has no `robot_id`, so the
  grouping key is the scene family: lab plus external-camera serial pair;
* the primary still-review surface is a single external camera at a legible tile
  size, never a three-camera side-by-side sheet;
* the motion screen cannot see a human in the workspace or a wrong agent moving,
  so nomination is never an acceptance;
* episode-length redundancy, not anchor quality, controls component size, so the
  candidate report states candidate anchor counts per episode up front.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

FPS = 15.0
HISTORY_SECONDS = 6.0
ACTION_SECONDS = 29.0 / 30.0
ANCHOR_STRIDE_S = 2.0
STATE_FIELDS = ("observation.joint_position", "observation.gripper_position")
ACTION_FIELDS = ("action.joint_position", "action.gripper_position")
CAMERAS = ("observation.images.left_external", "observation.images.wrist", "observation.images.right_external")
# Franka commanded joints are radians and the Robotiq command is a [0,1] absolute
# width. Both thresholds sit above the per-frame quantization floor measured on
# the staged low-dimensional collection; see meta/motion_thresholds.json.
JOINT_ACTIVITY_RAD = 0.02
GRIPPER_ACTIVITY = 0.02
DISCONTINUITY_RAD = 0.35
# Hysteresis band on the DROID absolute gripper command (0 open, 1 closed). A close event
# is one open-to-closed transition, so repeated closes inside one episode mark repeated
# grasp attempts, which is the cheapest low-dimensional signal for recovery behaviour.
GRIPPER_CLOSED = 0.6
GRIPPER_OPEN = 0.3


@dataclass
class EpisodeSummary:
    episode_index: int
    uuid: str
    lab: str
    user: str
    day: str
    scene_family: str
    left_serial: str
    right_serial: str
    frames: int
    duration_s: float
    candidate_anchors: int
    active_anchor_fraction: float
    keep_eligible_anchors: int
    keep_covered_fraction: float
    keep_interval_count: int
    source_task: str
    gripper_close_events: int
    joint_path_length: float
    gripper_path_length: float
    max_joint_step: float
    p99_joint_speed: float
    max_clock_step_s: float
    min_clock_step_s: float
    duplicate_timestamps: int
    discontinuous: bool
    descriptor: list[float]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def episode_table(metadata_root: Path):
    files = sorted((metadata_root / "meta/episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata beneath {metadata_root}")
    return pq.read_table(files)


def source_root(staging_root: Path, repo_id: str) -> Path:
    return (staging_root / repo_id.replace("/", "__")).resolve()


def anchor_grid(timestamps: np.ndarray) -> np.ndarray:
    """Boundary-eligible anchors: six seconds of history and a complete future chunk."""
    relative = timestamps - timestamps[0]
    last = relative[-1] - ACTION_SECONDS
    if last < HISTORY_SECONDS:
        return np.empty(0, dtype=np.float64)
    count = int(math.floor((last - HISTORY_SECONDS) / ANCHOR_STRIDE_S)) + 1
    return HISTORY_SECONDS + np.arange(count, dtype=np.float64) * ANCHOR_STRIDE_S


def eligible_anchor_activity(
    timestamps: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Score every stride-2 s anchor by motion inside its future action target."""
    relative = timestamps - timestamps[0]
    anchors = anchor_grid(timestamps)
    active = []
    for anchor in anchors:
        mask = (relative >= anchor - 1e-9) & (relative <= anchor + ACTION_SECONDS + 1e-9)
        future = actions[mask]
        if len(future) < 2:
            active.append(False)
            continue
        # The gripper is the last column and the joints are everything before it, so this
        # works for DROID's 7+1 Franka vector and UR7e's 6+1 vector without a per-source case.
        joint_range = float(np.ptp(future[:, :-1], axis=0).max())
        gripper_range = float(np.ptp(future[:, -1]))
        active.append(joint_range >= JOINT_ACTIVITY_RAD or gripper_range >= GRIPPER_ACTIVITY)
    return anchors, np.asarray(active, dtype=bool)


def gripper_close_events(gripper: np.ndarray) -> int:
    """Count open-to-closed transitions in the commanded gripper channel."""
    events = 0
    closed = bool(gripper[0] >= GRIPPER_CLOSED)
    for value in gripper[1:]:
        if not closed and value >= GRIPPER_CLOSED:
            events += 1
            closed = True
        elif closed and value <= GRIPPER_OPEN:
            closed = False
    return events


def keep_interval_windows(keep_ranges: list[list[int]]) -> list[tuple[float, float]]:
    """Source keep intervals as episode-relative seconds."""
    return [(float(start) / FPS, float(end - 1) / FPS) for start, end in keep_ranges if end - start >= 2]


def anchors_inside_keep(anchors: np.ndarray, keep_ranges: list[list[int]]) -> np.ndarray:
    """Anchors whose complete [-6 s, +0.9667 s] window lies inside one keep interval.

    `keep_interval_index` is -1 outside these intervals. The source excludes those
    frames as inconsistent, so a retained window must not straddle or enter one.
    """
    windows = keep_interval_windows(keep_ranges)
    inside = []
    for anchor in anchors:
        start, end = anchor - HISTORY_SECONDS, anchor + ACTION_SECONDS
        inside.append(any(low <= start + 1e-9 and end <= high + 1e-9 for low, high in windows))
    return np.asarray(inside, dtype=bool)


def _column(shard, field: str) -> np.ndarray:
    values = np.asarray(shard[field].to_pylist())
    return values[:, None] if values.ndim == 1 else values


def load_episode_arrays(shard, offset: int, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = shard.slice(offset, length)
    timestamps = np.asarray(rows["timestamp"].to_pylist(), dtype=np.float64)
    states = np.concatenate([_column(rows, field) for field in STATE_FIELDS], axis=1)
    actions = np.concatenate([_column(rows, field) for field in ACTION_FIELDS], axis=1)
    keep = np.asarray(rows["keep_interval_index"].to_pylist(), dtype=np.int64).reshape(-1)
    return timestamps, states.astype(np.float64), actions.astype(np.float64), keep


def scan(metadata_root: Path, staging_root: Path, repo_id: str, output: Path) -> dict:
    table = episode_table(metadata_root)
    rows = table.to_pylist()
    root = source_root(staging_root, repo_id)
    by_file: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_file[(int(row["data/chunk_index"]), int(row["data/file_index"]))].append(row)

    summaries: list[EpisodeSummary] = []
    invalid: list[dict] = []
    columns = ["timestamp", "episode_index", "keep_interval_index", *STATE_FIELDS, *ACTION_FIELDS]
    for (chunk_index, file_index), file_rows in sorted(by_file.items()):
        path = root / f"data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing staged low-dimensional shard: {path}")
        shard = pq.read_table(path, columns=columns)
        base = min(int(item["dataset_from_index"]) for item in file_rows)
        for row in file_rows:
            index = int(row["episode_index"])
            try:
                summaries.append(
                    summarize_episode(row, shard, int(row["dataset_from_index"]) - base)
                )
            except (KeyError, ValueError) as error:
                invalid.append({"episode_index": index, "reason": str(error)})
    value = {
        "source": repo_id,
        "episodes_scanned": len(rows),
        "valid_episodes": len(summaries),
        "invalid_episodes": invalid,
        "motion_thresholds": {
            "joint_activity_radians_ptp": JOINT_ACTIVITY_RAD,
            "gripper_activity_ptp": GRIPPER_ACTIVITY,
            "discontinuity_radians_l2_step": DISCONTINUITY_RAD,
            "measured_from": "native commanded action vector",
        },
        "summaries": [asdict(item) for item in summaries],
    }
    write_json(output, value)
    return value


def summarize_episode(row: dict, shard, offset: int) -> EpisodeSummary:
    length = int(row["length"])
    timestamps, states, actions, keep = load_episode_arrays(shard, offset, length)
    if len(timestamps) != length:
        raise ValueError(f"Shard slice length {len(timestamps)} does not match metadata {length}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state or action")
    if length < 2:
        raise ValueError("Episode is shorter than two frames")
    steps = np.diff(timestamps)
    if np.any(steps <= 0):
        raise ValueError("Timestamps are not strictly increasing")
    joint_steps = np.linalg.norm(np.diff(actions[:, :7], axis=0), axis=1)
    gripper_steps = np.abs(np.diff(actions[:, 7]))
    joint_speed = joint_steps / steps
    anchors, active = eligible_anchor_activity(timestamps, actions)
    keep_ranges = [list(item) for item in (row.get("keep_ranges") or [])]
    inside = anchors_inside_keep(anchors, keep_ranges) if len(anchors) else np.empty(0, dtype=bool)
    duration = float(timestamps[-1] - timestamps[0])
    uuid = str(row["uuid"])
    lab, user, stamp = (uuid.split("+") + ["", "", ""])[:3]
    tasks = [str(item) for item in (row.get("tasks") or []) if str(item).strip()]
    covered = sum(int(end) - int(start) for start, end in keep_ranges)
    descriptor = np.concatenate(
        [
            actions[0, :7],
            actions[-1, :7],
            np.asarray(
                [
                    duration,
                    float(joint_steps.sum()),
                    float(gripper_steps.sum()),
                    float(active.mean()) if len(active) else 0.0,
                ]
            ),
        ]
    )
    return EpisodeSummary(
        episode_index=int(row["episode_index"]),
        uuid=uuid,
        lab=lab,
        user=user,
        day=stamp[:10],
        scene_family=f"{lab}+{row['left_serial']}+{row['right_serial']}",
        left_serial=str(row["left_serial"]),
        right_serial=str(row["right_serial"]),
        frames=length,
        duration_s=duration,
        candidate_anchors=int(len(anchors)),
        active_anchor_fraction=float(active.mean()) if len(active) else 0.0,
        keep_eligible_anchors=int(np.count_nonzero(active & inside)),
        keep_covered_fraction=float(covered / length),
        keep_interval_count=len(keep_ranges),
        source_task=tasks[0] if tasks else "",
        gripper_close_events=gripper_close_events(actions[:, -1]),
        joint_path_length=float(joint_steps.sum()),
        gripper_path_length=float(gripper_steps.sum()),
        max_joint_step=float(joint_steps.max(initial=0.0)),
        p99_joint_speed=float(np.quantile(joint_speed, 0.99)),
        max_clock_step_s=float(steps.max()),
        min_clock_step_s=float(steps.min()),
        duplicate_timestamps=0,
        discontinuous=bool(joint_steps.max(initial=0.0) > DISCONTINUITY_RAD),
        descriptor=descriptor.tolist(),
    )


TILE_WIDTH = 400
TILE_COLUMNS = 4
TILE_ROWS = 3
LABEL_HEIGHT = 22
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _grab_frame(video: Path, timestamp_s: float, width: int) -> "Image.Image":
    """Decode one frame at an absolute shard timestamp."""
    from PIL import Image

    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{timestamp_s:.6f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={width}:-2",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def contact_sheet(
    video: Path,
    shard_start_s: float,
    offsets_s: list[float],
    destination: Path,
    *,
    header: str,
    tile_width: int = TILE_WIDTH,
    columns: int = TILE_COLUMNS,
) -> Path:
    """Write one labelled still sheet.

    Every tile carries its episode-relative source time so a visual event maps
    straight onto an annotation span. Sheets stay under a dozen tiles because a
    taller sheet is downscaled to illegibility by the time a reviewer sees it,
    which is the RoboChallenge three-camera lesson applied to still review.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(FONT_PATH, 16)
    header_font = ImageFont.truetype(FONT_PATH, 18)
    tiles = [_grab_frame(video, shard_start_s + offset, tile_width) for offset in offsets_s]
    tile_height = max(tile.height for tile in tiles)
    rows = int(math.ceil(len(tiles) / columns))
    sheet = Image.new(
        "RGB",
        (columns * tile_width, LABEL_HEIGHT + rows * (tile_height + LABEL_HEIGHT)),
        (16, 16, 16),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 3), header, fill=(255, 255, 255), font=header_font)
    for index, (tile, offset) in enumerate(zip(tiles, offsets_s, strict=True)):
        column, row = index % columns, index // columns
        x = column * tile_width
        y = LABEL_HEIGHT + row * (tile_height + LABEL_HEIGHT)
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 6, y + tile_height + 2),
            f"t={offset:6.2f}s",
            fill=(255, 220, 120),
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)
    return destination


def _episode_video(row: dict, info: dict, camera: str, root: Path) -> tuple[Path, float, float]:
    path = root / info["video_path"].format(
        video_key=camera,
        chunk_index=int(row[f"videos/{camera}/chunk_index"]),
        file_index=int(row[f"videos/{camera}/file_index"]),
    )
    start = float(row[f"videos/{camera}/from_timestamp"])
    end = float(row[f"videos/{camera}/to_timestamp"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing staged camera shard: {path}")
    return path, start, end - start


def overview_offsets(duration_s: float, tiles: int = TILE_COLUMNS * TILE_ROWS) -> list[float]:
    return [round(duration_s * (index + 0.5) / tiles, 3) for index in range(tiles)]


def detail_pages(duration_s: float, stride_s: float, per_page: int) -> list[list[float]]:
    count = max(1, int(math.floor(duration_s / stride_s)) + 1)
    offsets = [round(min(index * stride_s, duration_s - 0.05), 3) for index in range(count)]
    return [offsets[index : index + per_page] for index in range(0, len(offsets), per_page)]


def annotation_template(summary: dict, keep_ranges: list[list[int]]) -> dict:
    """Seed the reviewer file with the motion screen and source keep intervals."""
    return {
        "source_episode_index": int(summary["episode_index"]),
        "uuid": summary["uuid"],
        "episode_duration_s": float(summary["duration_s"]),
        "review_status": "pending",
        "task": "",
        "source_task": str(summary.get("source_task", "")),
        "gripper_close_events": int(summary.get("gripper_close_events", 0)),
        "outcome": "",
        "reviewer_notes": "",
        "source_keep_intervals_s": [
            [round(start / FPS, 3), round((end - 1) / FPS, 3)] for start, end in keep_ranges
        ],
        "candidate_anchors_s": [],
        "segments": [],
        "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
        "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
    }


def proxies(
    candidates_path: Path,
    metadata_root: Path,
    staging_root: Path,
    repo_id: str,
    output_root: Path,
    *,
    mode: str,
    episodes: list[int] | None,
    stride_s: float,
    cameras: list[str] | None = None,
) -> list[Path]:
    manifest = read_json(candidates_path)
    summaries = {int(item["episode_index"]): item for item in manifest["candidates"]}
    wanted = episodes or sorted(summaries)
    info = read_json(metadata_root / "meta/info.json")
    rows = {row["episode_index"]: row for row in episode_table(metadata_root).to_pylist()}
    root = source_root(staging_root, repo_id)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for episode_index in wanted:
        summary = summaries[episode_index]
        row = rows[episode_index]
        stem = output_root / f"episode_{episode_index:06d}"
        if mode == "overview":
            for camera, label in (
                ("observation.images.left_external", "left"),
                ("observation.images.right_external", "right"),
            ):
                video, start, duration = _episode_video(row, info, camera, root)
                task = str(summary.get("source_task", ""))
                header = (
                    f"ep {episode_index}  {summary['uuid']}  {duration:.1f}s  "
                    f"{label}_external  anchors={summary['keep_eligible_anchors']}  "
                    f"closes={summary.get('gripper_close_events', 0)}"
                    + (f"  | {task[:70]}" if task else "")
                )
                written.append(
                    contact_sheet(
                        video,
                        start,
                        overview_offsets(duration),
                        stem.with_suffix(f".{label}.jpg"),
                        header=header,
                    )
                )
        else:
            detail = [
                (f"observation.images.{name}", "left" if name == "left_external" else name.split("_")[0])
                for name in (cameras or ("left_external", "wrist"))
            ]
            for camera, label in detail:
                video, start, duration = _episode_video(row, info, camera, root)
                for page, offsets in enumerate(detail_pages(duration, stride_s, TILE_COLUMNS * TILE_ROWS)):
                    header = (
                        f"ep {episode_index}  {label}  page {page + 1}  "
                        f"{offsets[0]:.1f}-{offsets[-1]:.1f}s of {duration:.1f}s"
                    )
                    written.append(
                        contact_sheet(
                            video,
                            start,
                            offsets,
                            stem.with_suffix(f".detail.{label}.p{page + 1}.jpg"),
                            header=header,
                        )
                    )
        template_path = stem.with_suffix(".annotations.json")
        if not template_path.exists():
            template = annotation_template(summary, [list(item) for item in (row["keep_ranges"] or [])])
            template["candidate_anchors_s"] = [
                round(float(value), 3)
                for value in anchor_grid(np.arange(summary["frames"], dtype=np.float64) / FPS)
            ]
            write_json(template_path, template)
            written.append(template_path)
    return written


def group_quotas(pool: list[dict], count: int, key: str) -> dict[str, int]:
    """Split candidate slots as evenly as the pool allows across grouping keys.

    RoboChallenge showed that expressing balance as a score penalty collapses as
    soon as one group sits far from the pool median in descriptor space. DROID has
    no `robot_id`, so the scene family (lab plus external-camera serial pair) and
    the operator take its place, and the split is a hard quota.
    """
    available: dict[str, int] = collections.Counter(str(item[key]) for item in pool)
    quotas = dict.fromkeys(available, 0)
    for _ in range(count):
        eligible = [name for name in sorted(available) if quotas[name] < available[name]]
        if not eligible:
            raise ValueError(f"Only {sum(available.values())} episodes are available for {count}")
        quotas[min(eligible, key=lambda name: (quotas[name], -available[name], name))] += 1
    return quotas


def farthest_point_candidates(pool: list[dict], count: int, key: str) -> list[dict]:
    quotas = group_quotas(pool, count, key)
    descriptors = np.asarray([item["descriptor"] for item in pool], dtype=np.float64)
    median = np.median(descriptors, axis=0)
    scale = np.quantile(descriptors, 0.75, axis=0) - np.quantile(descriptors, 0.25, axis=0)
    scale[scale < 1e-9] = 1.0
    descriptors = (descriptors - median) / scale
    selected: list[int] = []
    counts = dict.fromkeys(quotas, 0)
    min_distance = np.full(len(pool), np.inf)
    while len(selected) < count:
        if not selected:
            score = np.asarray([item["keep_eligible_anchors"] for item in pool], dtype=np.float64)
        else:
            min_distance = np.minimum(
                min_distance, np.linalg.norm(descriptors - descriptors[selected[-1]], axis=1)
            )
            score = min_distance.copy()
        for index, item in enumerate(pool):
            if counts[str(item[key])] >= quotas[str(item[key])]:
                score[index] = -np.inf
        for index in selected:
            score[index] = -np.inf
        choice = int(np.argmax(score))
        selected.append(choice)
        counts[str(pool[choice][key])] += 1
    return [pool[index] for index in selected]


def screen_pool(
    summaries: list[dict],
    lab: str,
    min_anchors: int,
    min_active: float,
    min_close_events: int = 0,
) -> list[dict]:
    """Screen one lab's episodes.

    `min_close_events` is the recovery filter. Plan Section 1.6 found recovery to be
    the one Section 1.3 gap the DROID Failure contribution left mostly open, and an
    episode whose gripper closes twice or more has attempted a grasp twice or more,
    which is where a retry after a slip or a miss lives. It is a nomination signal
    only; whether a retry is a genuine recovery is decided by proxy review.
    """
    return [
        item
        for item in summaries
        if item["lab"] == lab
        and not item["discontinuous"]
        and item["keep_eligible_anchors"] >= min_anchors
        and item["active_anchor_fraction"] >= min_active
        and item.get("gripper_close_events", 0) >= min_close_events
    ]


def video_files(row: dict, info: dict) -> list[str]:
    return [
        info["video_path"].format(
            video_key=camera,
            chunk_index=int(row[f"videos/{camera}/chunk_index"]),
            file_index=int(row[f"videos/{camera}/file_index"]),
        )
        for camera in CAMERAS
    ]


def nominate(
    scan_path: Path,
    metadata_root: Path,
    lab: str,
    count: int,
    output: Path,
    *,
    min_anchors: int,
    min_active: float,
    group_key: str,
    min_close_events: int = 0,
) -> dict:
    scanned = read_json(scan_path)
    pool = screen_pool(scanned["summaries"], lab, min_anchors, min_active, min_close_events)
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} eligible {lab} episodes are available for {count} candidates")
    for item in pool:
        item["group_key"] = f"{item['scene_family']}|{item['user']}" if group_key == "scene_user" else item[group_key]
    selected = farthest_point_candidates(pool, count, "group_key")
    info = read_json(metadata_root / "meta/info.json")
    rows = {row["episode_index"]: row for row in episode_table(metadata_root).to_pylist()}
    payload: set[str] = set()
    for item in selected:
        row = rows[item["episode_index"]]
        payload.update(video_files(row, info))
        payload.add(
            info["data_path"].format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            )
        )
    value = {
        "source": scanned["source"],
        "component": lab,
        "grouping_key": group_key,
        "selection_status": "nominated_pending_visual_review",
        "candidate_count": count,
        "pool_size": len(pool),
        "lab_episodes_scanned": sum(1 for item in scanned["summaries"] if item["lab"] == lab),
        "rejection_screen": {
            "minimum_keep_eligible_active_anchors": min_anchors,
            "minimum_active_anchor_fraction": min_active,
            "maximum_joint_step_radians_l2": DISCONTINUITY_RAD,
            "minimum_gripper_close_events": min_close_events,
            "anchor_window_must_lie_inside_one_source_keep_interval": True,
        },
        "group_balance": {
            "policy": "even per-scene-family/operator quota enforced before descriptor diversity",
            "pool": dict(collections.Counter(item["group_key"] for item in pool)),
            "selected": dict(collections.Counter(item["group_key"] for item in selected)),
        },
        "payload_files": sorted(payload),
        "candidates": selected,
    }
    write_json(output, value)
    return value


REVIEWER_REJECT_REASONS = {"erratic", "low_quality", "human_intervention", "out_of_scope", "static"}
REVIEWER_KEEP_REASONS = {
    "useful_motion",
    "informative_mistake",
    "recovery",
    "task_required_hold",
}


def _cell_labels(timestamps: np.ndarray, actions: np.ndarray) -> tuple[list[float], list[bool]]:
    """Anchor-grid activity cells, matching the extractor's anchor set exactly."""
    anchors, active = eligible_anchor_activity(timestamps, actions)
    duration = float(timestamps[-1] - timestamps[0])
    boundaries = [0.0, *[float(value) for value in anchors], duration]
    labels = [False, *[bool(value) for value in active]]
    return boundaries, labels


def _label_at(boundaries: list[float], labels: list[bool], timestamp: float) -> bool:
    for index, label in enumerate(labels):
        if boundaries[index] <= timestamp < boundaries[index + 1]:
            return label
    return labels[-1]


def _span_at(spans: list[dict], timestamp: float) -> dict:
    for span in spans:
        if float(span["start_s"]) <= timestamp < float(span["end_s"]):
            return span
    return spans[-1]


def _inside_keep(windows: list[tuple[float, float]], start: float, end: float) -> bool:
    return any(low <= start + 1e-6 and end <= high + 1e-6 for low, high in windows)


def build_annotations(
    review: dict,
    timestamps: np.ndarray,
    actions: np.ndarray,
    keep_ranges: list[list[int]],
    *,
    enforce_keep_intervals: bool = True,
) -> dict:
    """Intersect reviewer spans, the activity screen, and the source keep intervals.

    The reviewer decides intent, outcome, mistake spans, and deliberate holds. The
    activity screen only subdivides an approved span, and the source keep intervals
    only remove windows the publisher already flagged as inconsistent. Automatic
    statistics never promote a span the reviewer rejected.

    `enforce_keep_intervals` distinguishes two different things that both look like an
    empty `keep_ranges`. On DROID the field exists and an empty value means the
    publisher found no consistent interval, which is evidence against the episode, so
    the default excludes it. A source that has no such field at all -- UR7e, for
    instance -- declares no exclusions, and passing False skips the rule rather than
    rejecting every window.
    """
    duration = float(timestamps[-1] - timestamps[0])
    spans = sorted(review["spans"], key=lambda item: float(item["start_s"]))
    cursor = 0.0
    for span in spans:
        start, end = float(span["start_s"]), float(span["end_s"])
        if abs(start - cursor) > 1e-6 or end <= start:
            raise ValueError(f"Reviewer spans must be contiguous and ordered near {start:.3f}s")
        if span["retention"] == "keep" and span["retention_reason"] not in REVIEWER_KEEP_REASONS:
            raise ValueError(f"Unknown keep reason {span['retention_reason']!r}")
        if span["retention"] == "reject" and span["retention_reason"] not in REVIEWER_REJECT_REASONS:
            raise ValueError(f"Unknown reject reason {span['retention_reason']!r}")
        cursor = end
    if abs(cursor - duration) > 0.05:
        raise ValueError(f"Reviewer spans end at {cursor:.3f}s but the episode ends at {duration:.3f}s")
    spans[-1]["end_s"] = duration

    boundaries, labels = _cell_labels(timestamps, actions)
    windows = keep_interval_windows(keep_ranges)
    edges = {0.0, duration}
    edges.update(boundaries)
    edges.update(float(span["start_s"]) for span in spans)
    edges.update(float(span["end_s"]) for span in spans)
    for low, high in windows:
        edges.update((low, high))
    ordered = sorted(value for value in edges if -1e-9 <= value <= duration + 1e-9)

    segments: list[dict] = []
    screen_disagreements = 0
    for start, end in zip(ordered, ordered[1:], strict=False):
        if end <= start + 1e-6:
            continue
        middle = 0.5 * (start + end)
        span = _span_at(spans, middle)
        reason = str(span["retention_reason"])
        active = _label_at(boundaries, labels, middle)
        if span["retention"] == "reject":
            segment = {"retention": "reject", "retention_reason": reason}
        elif (
            enforce_keep_intervals
            and not _inside_keep(windows, start, end)
            and not span.get("override_source_keep", False)
        ):
            segment = {"retention": "reject", "retention_reason": "source_excluded"}
        elif not active and reason != "task_required_hold":
            screen_disagreements += 1
            segment = {"retention": "reject", "retention_reason": "static"}
        else:
            events = [
                dict(event)
                for event in span.get("mistake_events", [])
                if float(event["start_s"]) < end and float(event["end_s"]) > start
            ]
            segment = {
                "retention": "keep",
                "retention_reason": reason,
                "subtask": str(span["subtask"]),
                "quality": int(span["quality"]),
                "mistake_events": events,
            }
        segment["start_s"] = round(float(start), 6)
        segment["end_s"] = round(float(end), 6)
        if segments and all(
            segments[-1].get(key) == segment.get(key)
            for key in ("retention", "retention_reason", "subtask", "quality", "mistake_events")
        ):
            segments[-1]["end_s"] = segment["end_s"]
        else:
            segments.append(segment)
    return {
        "source_episode_index": int(review["source_episode_index"]),
        "uuid": review.get("uuid", ""),
        "task": str(review["task"]),
        "outcome": str(review.get("outcome", "")),
        "reviewer_notes": str(review.get("reviewer_notes", "")),
        "episode_duration_s": duration,
        "review_status": "validated",
        "review_basis": (
            "labelled global-camera contact sheets at a two-second stride over the full episode, "
            "intersected with the commanded-action activity screen and the source keep intervals"
        ),
        "activity_screen_subdivisions": screen_disagreements,
        "source_keep_intervals_s": [[round(low, 3), round(high, 3)] for low, high in windows],
        "segments": segments,
        "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
        "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
    }


def write_annotations(
    review_root: Path,
    metadata_root: Path,
    staging_root: Path,
    repo_id: str,
    accepted: list[int],
) -> dict:
    rows = {row["episode_index"]: row for row in episode_table(metadata_root).to_pylist()}
    root = source_root(staging_root, repo_id)
    columns = ["timestamp", "episode_index", "keep_interval_index", *STATE_FIELDS, *ACTION_FIELDS]
    report = {}
    for episode_index in accepted:
        row = rows[episode_index]
        review = read_json(review_root / f"episode_{episode_index:06d}.review.json")
        shard = pq.read_table(
            root / f"data/chunk-{int(row['data/chunk_index']):03d}/file_{int(row['data/file_index']):03d}.parquet",
            columns=columns,
        )
        table = episode_table(metadata_root).to_pylist()
        base = min(
            int(item["dataset_from_index"])
            for item in table
            if item["data/chunk_index"] == row["data/chunk_index"]
            and item["data/file_index"] == row["data/file_index"]
        )
        timestamps, _, actions, _ = load_episode_arrays(
            shard, int(row["dataset_from_index"]) - base, int(row["length"])
        )
        annotations = build_annotations(
            review, timestamps, actions, [list(item) for item in (row["keep_ranges"] or [])]
        )
        write_json(review_root / f"episode_{episode_index:06d}.annotations.json", annotations)
        keep = [item for item in annotations["segments"] if item["retention"] == "keep"]
        report[episode_index] = {
            "task": annotations["task"],
            "outcome": annotations["outcome"],
            "segments": len(annotations["segments"]),
            "kept_seconds": round(sum(item["end_s"] - item["start_s"] for item in keep), 2),
            "quality_values": sorted({item["quality"] for item in keep}),
            "mistake_events": sum(len(item["mistake_events"]) for item in keep),
            "keep_reasons": sorted({item["retention_reason"] for item in keep}),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scanner = subparsers.add_parser("scan", help="summarize every source episode from staged data shards")
    scanner.add_argument("--metadata-root", type=Path, required=True)
    scanner.add_argument("--staging-root", type=Path, required=True)
    scanner.add_argument("--repo-id", default="jnogga/droid_failure")
    scanner.add_argument("--output", type=Path, required=True)
    nomination = subparsers.add_parser("nominate", help="nominate one component's review candidates")
    nomination.add_argument("--scan", type=Path, required=True)
    nomination.add_argument("--metadata-root", type=Path, required=True)
    nomination.add_argument("--lab", required=True)
    nomination.add_argument("--count", type=int, default=24)
    nomination.add_argument("--min-anchors", type=int, default=3)
    nomination.add_argument("--min-active", type=float, default=0.25)
    nomination.add_argument("--group-key", default="scene_user")
    nomination.add_argument(
        "--min-close-events",
        type=int,
        default=0,
        help="require at least this many gripper close events; 2 targets repeated grasp attempts",
    )
    nomination.add_argument("--output", type=Path, required=True)
    proxy = subparsers.add_parser("proxies", help="render labelled review sheets and annotation templates")
    proxy.add_argument("--candidates", type=Path, required=True)
    proxy.add_argument("--metadata-root", type=Path, required=True)
    proxy.add_argument("--staging-root", type=Path, required=True)
    proxy.add_argument("--repo-id", default="jnogga/droid_failure")
    proxy.add_argument("--output-root", type=Path, required=True)
    proxy.add_argument("--mode", choices=("overview", "detail"), default="overview")
    proxy.add_argument("--episodes", type=int, nargs="*")
    proxy.add_argument("--stride", type=float, default=ANCHOR_STRIDE_S)
    proxy.add_argument(
        "--cameras",
        nargs="+",
        choices=("left_external", "wrist", "right_external"),
        help="detail-mode cameras; defaults to the left external view plus the wrist",
    )
    annotate = subparsers.add_parser("annotations", help="compile reviewer spans into validated annotations")
    annotate.add_argument("--review-root", type=Path, required=True)
    annotate.add_argument("--metadata-root", type=Path, required=True)
    annotate.add_argument("--staging-root", type=Path, required=True)
    annotate.add_argument("--repo-id", default="jnogga/droid_failure")
    annotate.add_argument("--accepted", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "scan":
        value = scan(args.metadata_root, args.staging_root, args.repo_id, args.output)
        print(json.dumps({k: v for k, v in value.items() if k != "summaries"}, indent=2))
    elif args.command == "nominate":
        value = nominate(
            args.scan,
            args.metadata_root,
            args.lab,
            args.count,
            args.output,
            min_anchors=args.min_anchors,
            min_active=args.min_active,
            group_key=args.group_key,
            min_close_events=args.min_close_events,
        )
        print(
            json.dumps(
                {
                    "component": value["component"],
                    "pool_size": value["pool_size"],
                    "group_balance": value["group_balance"],
                    "payload_file_count": len(value["payload_files"]),
                    "candidates": [
                        {
                            "episode_index": item["episode_index"],
                            "uuid": item["uuid"],
                            "duration_s": round(item["duration_s"], 1),
                            "keep_eligible_anchors": item["keep_eligible_anchors"],
                            "active_anchor_fraction": round(item["active_anchor_fraction"], 3),
                            "gripper_close_events": item.get("gripper_close_events", 0),
                            "source_task": item.get("source_task", ""),
                        }
                        for item in value["candidates"]
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "proxies":
        written = proxies(
            args.candidates,
            args.metadata_root,
            args.staging_root,
            args.repo_id,
            args.output_root,
            mode=args.mode,
            episodes=args.episodes,
            stride_s=args.stride,
            cameras=args.cameras,
        )
        print("\n".join(str(path) for path in written))
    elif args.command == "annotations":
        report = write_annotations(
            args.review_root, args.metadata_root, args.staging_root, args.repo_id, args.accepted
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
