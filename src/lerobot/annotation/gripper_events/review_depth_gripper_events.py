#!/usr/bin/env python

"""Render manual-review videos for depth gripper-event labels.

The renderer reads the materialized sidecars that training will consume and burns
their values over the dataset's synchronized top and wrist videos. It does not
recompute events or targets.

By default, two event-rich episodes are selected from each annotated dataset:

    python review_depth_gripper_events.py

Specific episodes can also be rendered:

    python review_depth_gripper_events.py \
        --data-dir outputs/rebot_val-annotated-v2 --episodes 0 2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TOP = "observation.images.top"
WRIST = "observation.images.wrist"
GRIPPER_FEATURE = "gripper.pos"
CLOSE_THRESHOLD = -60.0
OPEN_THRESHOLD = -90.0
CUTOFF_FRAMES = 150
DEFAULT_DATASETS = [
    Path("outputs/rebot_socks_basket-annotated-v2"),
    Path("outputs/rebot_shirts_bin-annotated-v2"),
    Path("outputs/rebot_two_container-annotated-v2"),
    Path("outputs/rebot_val-annotated-v2"),
]

VIDEO_W = 1280
CAMERA_H = 480
TIMELINE_H = 92
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class ReviewEpisode:
    root: Path
    episode_index: int
    start: int
    stop: int
    fps: float
    n_close: int
    n_open: int

    @property
    def frames(self) -> int:
        return self.stop - self.start

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    @property
    def n_events(self) -> int:
        return self.n_close + self.n_open


def _read_info(root: Path) -> tuple[dict, float, int]:
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)
    fps = float(info["fps"])
    names = info["features"]["action"]["names"]
    matches = [i for i, name in enumerate(names) if name == GRIPPER_FEATURE]
    if len(matches) != 1:
        raise ValueError(f"{root}: expected exactly one {GRIPPER_FEATURE!r} action feature")
    return info, fps, matches[0]


def _episode_metadata(root: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"{root}: no episode metadata parquet")
    return pd.concat([pd.read_parquet(path) for path in paths]).set_index("episode_index")


def episode_table(roots: list[Path]) -> list[ReviewEpisode]:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        _info, fps, _gripper_dim = _read_info(root)
        episodes = _episode_metadata(root)
        events = pd.read_parquet(root / "meta" / "depth_gripper_events.parquet")
        for episode_index, episode in episodes.iterrows():
            episode_events = events[events["episode_index"] == episode_index]
            rows.append(
                ReviewEpisode(
                    root=root,
                    episode_index=int(episode_index),
                    start=int(episode["dataset_from_index"]),
                    stop=int(episode["dataset_to_index"]),
                    fps=fps,
                    n_close=int((episode_events["event_type"] == "close").sum()),
                    n_open=int((episode_events["event_type"] == "open").sum()),
                )
            )
    return rows


def pick_event_rich(rows: list[ReviewEpisode], per_dataset: int) -> list[ReviewEpisode]:
    by_root: dict[Path, list[ReviewEpisode]] = {}
    for row in rows:
        by_root.setdefault(row.root, []).append(row)
    picked = []
    for root in DEFAULT_DATASETS:
        candidates = sorted(
            by_root.get(root, []),
            key=lambda row: (-row.n_events, -min(row.n_close, row.n_open), row.episode_index),
        )
        picked.extend(candidates[:per_dataset])
    return picked


def _load_episode_values(review: ReviewEpisode, gripper_dim: int) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    root = review.root
    actions = pd.concat(
        [
            pd.read_parquet(path, columns=["action"])
            for path in sorted((root / "data").glob("chunk-*/file-*.parquet"))
        ],
        ignore_index=True,
    )
    gripper = np.array(
        [np.asarray(action, dtype=np.float32)[gripper_dim] for action in actions["action"]],
        dtype=np.float32,
    )[review.start : review.stop]
    labels = pd.read_parquet(root / "meta" / "depth_gripper_event_labels.parquet").iloc[
        review.start : review.stop
    ]
    events = pd.read_parquet(root / "meta" / "depth_gripper_events.parquet")
    events = events[events["episode_index"] == review.episode_index].copy()

    expected_frame = np.arange(review.frames, dtype=np.int64)
    expected_index = np.arange(review.start, review.stop, dtype=np.int64)
    if len(labels) != review.frames:
        raise ValueError(f"{root} ep{review.episode_index}: label length disagrees with episode bounds")
    if not np.array_equal(labels["frame_index"].to_numpy(), expected_frame):
        raise ValueError(f"{root} ep{review.episode_index}: label frame identities are not local and contiguous")
    if not np.array_equal(labels["index"].to_numpy(), expected_index):
        raise ValueError(f"{root} ep{review.episode_index}: label global identities disagree with bounds")
    return gripper, labels, events


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}".replace(".", ",")


def format_countdown(delta: int, target: float, fps: float) -> str:
    if delta < 0:
        return ">5s / none   target 0.000"
    return f"{delta / fps:4.1f}s ({delta:3d}f)   target {target:.3f}"


def build_srt(
    review: ReviewEpisode,
    gripper: np.ndarray,
    labels: pd.DataFrame,
    events: pd.DataFrame,
    step_frames: int = 3,
) -> str:
    """Update numerical labels at 10 Hz and flash each event for half a second."""
    close_delta = labels["depth_gripper_close_delta"].to_numpy(dtype=np.int16)
    open_delta = labels["depth_gripper_open_delta"].to_numpy(dtype=np.int16)
    close_target = labels["depth_gripper_close_target"].to_numpy(dtype=np.float32)
    open_target = labels["depth_gripper_open_target"].to_numpy(dtype=np.float32)
    close_events = set(events.loc[events["event_type"] == "close", "frame_index"].astype(int))
    open_events = set(events.loc[events["event_type"] == "open", "frame_index"].astype(int))
    flash_frames = max(1, int(round(0.5 * review.fps)))

    cues = []
    for cue_index, start in enumerate(range(0, review.frames, step_frames), start=1):
        stop = min(start + step_frames, review.frames)
        recent_start = max(0, start - flash_frames + 1)
        recent_close = any(frame in close_events for frame in range(recent_start, stop))
        recent_open = any(frame in open_events for frame in range(recent_start, stop))
        banner = []
        if recent_close:
            banner.append("*** CLOSE EVENT ***")
        if recent_open:
            banner.append("*** OPEN EVENT ***")
        lines = [
            (
                f"ep{review.episode_index}   frame {start}/{review.frames - 1}   "
                f"global {review.start + start}   command {gripper[start]:.1f} deg"
            ),
            f"next CLOSE  {format_countdown(int(close_delta[start]), float(close_target[start]), review.fps)}",
            f"next OPEN   {format_countdown(int(open_delta[start]), float(open_target[start]), review.fps)}",
        ]
        if banner:
            lines.append("    ".join(banner))
        cues.append(
            f"{cue_index}\n{_timestamp(start / review.fps)} --> {_timestamp(stop / review.fps)}\n"
            + "\n".join(lines)
            + "\n"
        )
    return "\n".join(cues)


def _x(frame: int, frames: int) -> int:
    return int(frame / max(frames - 1, 1) * (VIDEO_W - 1))


def build_timeline(
    review: ReviewEpisode,
    gripper: np.ndarray,
    labels: pd.DataFrame,
    events: pd.DataFrame,
    path: Path,
) -> None:
    """Command trace, thresholds, event ticks, and both future-target intensity strips."""
    image = Image.new("RGB", (VIDEO_W, TIMELINE_H), (16, 16, 18))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 11)
    small = ImageFont.truetype(FONT_PATH, 9)

    command_top, command_bottom = 3, 49

    def command_y(value: float) -> int:
        clipped = min(20.0, max(-300.0, value))
        return int(command_bottom - (clipped + 300.0) / 320.0 * (command_bottom - command_top))

    for threshold, color in ((CLOSE_THRESHOLD, (235, 80, 80)), (OPEN_THRESHOLD, (80, 150, 245))):
        y = command_y(threshold)
        for x0 in range(0, VIDEO_W, 10):
            draw.line((x0, y, min(x0 + 5, VIDEO_W - 1), y), fill=color, width=1)

    # Downsample only for drawing; annotations still come from every stored label row.
    points = [(_x(frame, review.frames), command_y(float(value))) for frame, value in enumerate(gripper)]
    if len(points) > 1:
        draw.line(points, fill=(225, 225, 225), width=1)

    close_target = labels["depth_gripper_close_target"].to_numpy(dtype=np.float32)
    open_target = labels["depth_gripper_open_target"].to_numpy(dtype=np.float32)
    strip_specs = (
        (close_target, 54, 69, (240, 68, 68), "CLOSE"),
        (open_target, 73, 88, (70, 140, 245), "OPEN"),
    )
    for target, y0, y1, color, name in strip_specs:
        for x0 in range(VIDEO_W):
            lo = x0 * review.frames // VIDEO_W
            hi = max(lo + 1, (x0 + 1) * review.frames // VIDEO_W)
            value = float(target[lo:min(hi, review.frames)].max(initial=0.0))
            fill = tuple(int(channel * value) for channel in color)
            draw.line((x0, y0, x0, y1), fill=fill)
        draw.text((4, y0 + 1), name, fill=(255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0))

    for event in events.itertuples(index=False):
        x0 = _x(int(event.frame_index), review.frames)
        color = (255, 70, 70) if event.event_type == "close" else (70, 150, 255)
        draw.line((x0, 0, x0, 8), fill=color, width=2)
    draw.text((4, 2), "command", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    image.save(path)


def render(review: ReviewEpisode, out_dir: Path, crf: int, preset: str) -> dict:
    root = review.root.resolve()
    out_dir = out_dir.resolve()
    info, fps, gripper_dim = _read_info(root)
    episodes = _episode_metadata(root)
    episode = episodes.loc[review.episode_index]
    gripper, labels, events = _load_episode_values(review, gripper_dim)
    if fps != review.fps:
        raise ValueError("FPS changed between review selection and rendering")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{root.name}__ep{review.episode_index:02d}__depth_gripper_events"
    with tempfile.TemporaryDirectory(prefix=f".{stem}.", dir=out_dir) as temporary_dir:
        temporary = Path(temporary_dir)
        subtitle = temporary / "labels.srt"
        subtitle.write_text(build_srt(review, gripper, labels, events))
        timeline = temporary / "timeline.png"
        build_timeline(review, gripper, labels, events, timeline)
        playhead = temporary / "playhead.png"
        Image.new("RGB", (3, TIMELINE_H), (255, 255, 255)).save(playhead)

        inputs = []
        for key in (TOP, WRIST):
            video_path = root / info["video_path"].format(
                video_key=key,
                chunk_index=int(episode[f"videos/{key}/chunk_index"]),
                file_index=int(episode[f"videos/{key}/file_index"]),
            )
            inputs += [
                "-ss",
                f"{float(episode[f'videos/{key}/from_timestamp']):.3f}",
                "-t",
                f"{review.duration:.3f}",
                "-i",
                str(video_path),
            ]
        # A single-frame overlay is repeated by the overlay filter. Using image2's
        # loop option here can multiply the effective output frame rate.
        inputs += ["-i", str(timeline), "-i", str(playhead)]

        style = (
            "FontName=DejaVu Sans,FontSize=14,Alignment=7,MarginV=7,MarginL=10,"
            "BorderStyle=3,Outline=1,Shadow=0,PrimaryColour=&H00FFFFFF,BackColour=&H90000000"
        )
        output_height = CAMERA_H + TIMELINE_H
        filter_graph = (
            f"[0:v]scale=640:{CAMERA_H}[top];[1:v]scale=640:{CAMERA_H}[wrist];"
            f"[top][wrist]hstack=inputs=2[cameras];"
            f"[cameras]pad={VIDEO_W}:{output_height}:0:0:color=0x101012[canvas];"
            f"[canvas][2:v]overlay=0:{CAMERA_H}[with_timeline];"
            f"[with_timeline][3:v]overlay=x='{VIDEO_W - 3}*t/{review.duration:.6f}':y={CAMERA_H}[marked];"
            f"[marked]fps={fps:g},subtitles={subtitle.name}:force_style='{style}'[out]"
        )
        destination = out_dir / f"{stem}.mp4"
        temporary_output = temporary / destination.name
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
        result = subprocess.run(command, cwd=temporary, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {stem}:\n{result.stderr.strip()[:2000]}")
        os.replace(temporary_output, destination)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt:format=duration",
            "-of",
            "json",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    details = json.loads(probe.stdout)
    stream = details["streams"][0]
    actual_duration = float(details["format"]["duration"])
    if stream["codec_name"] != "h264" or stream["width"] != VIDEO_W or stream["height"] != CAMERA_H + TIMELINE_H:
        raise ValueError(f"Unexpected rendered video format: {details}")
    if abs(actual_duration - review.duration) > 0.25:
        raise ValueError(
            f"Rendered duration {actual_duration:.3f}s disagrees with episode duration {review.duration:.3f}s"
        )
    return {
        "dataset": review.root.name,
        "episode_index": review.episode_index,
        "frames": review.frames,
        "duration_seconds": review.duration,
        "close_events": review.n_close,
        "open_events": review.n_open,
        "path": str(destination),
        "size_mb": destination.stat().st_size / 1e6,
        "video": {**stream, "duration": actual_duration},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--per-dataset", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/_annotation/depth_gripper_event_review"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [args.data_dir] if args.data_dir else DEFAULT_DATASETS
    rows = episode_table(roots)
    if args.data_dir and args.episodes is not None:
        selected = [row for row in rows if row.episode_index in args.episodes]
        missing = sorted(set(args.episodes) - {row.episode_index for row in selected})
        if missing:
            raise ValueError(f"Unknown episode indices for {args.data_dir}: {missing}")
    elif args.data_dir:
        selected = sorted(rows, key=lambda row: row.episode_index)
    else:
        selected = pick_event_rich(rows, args.per_dataset)

    print(f"{'dataset':34s} {'ep':>3s} {'min':>6s} {'close':>6s} {'open':>5s}")
    for review in selected:
        print(
            f"{review.root.name:34s} {review.episode_index:3d} {review.duration / 60:6.2f} "
            f"{review.n_close:6d} {review.n_open:5d}"
        )
    if args.list:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(render, review, args.out, args.crf, args.preset): review for review in selected
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"  {result['path']}  ({result['duration_seconds']:.0f}s, "
                f"{result['size_mb']:.0f} MB)",
                flush=True,
            )

    results.sort(key=lambda result: (result["dataset"], result["episode_index"]))
    manifest = args.out / "manifest.json"
    temporary = args.out / ".manifest.json.tmp"
    temporary.write_text(json.dumps({"videos": results}, indent=2) + "\n")
    os.replace(temporary, manifest)
    print(f"done -> {args.out} ({len(results)} videos; manifest: {manifest})")


if __name__ == "__main__":
    main()
