#!/usr/bin/env python

"""Materialize action-derived future gripper-event targets for the depth loss.

Only ``meta/info.json``, episode bounds, and the source parquet columns ``action``,
``episode_index``, ``frame_index``, and ``index`` are read. In particular, this
script never reads robot state or any semantic annotation.

Example:

    python depth_gripper_event_annotate.py \
        --data-dir outputs/rebot_socks_basket-annotated-v2
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUBRIC_VERSION = "depth-gripper-event-labels-v1"
EXPECTED_FPS = 30.0
GRIPPER_FEATURE = "gripper.pos"
EXPECTED_GRIPPER_DIM = 6
CLOSE_THRESHOLD = -60.0
OPEN_THRESHOLD = -90.0
MIN_CLOSED_FRAMES = 15
HALF_LIFE_FRAMES = 30
CUTOFF_FRAMES = 150

EVENT_FILENAME = "depth_gripper_events.parquet"
LABEL_FILENAME = "depth_gripper_event_labels.parquet"
INFO_FILENAME = "depth_gripper_event_labels_info.json"
PLOT_DIRNAME = "depth_gripper_event_qa"

EVENT_COLUMNS = [
    "episode_index",
    "event_type",
    "frame_index",
    "index",
    "gripper_command",
    "closed_interval_start",
    "closed_interval_stop",
    "closed_interval_length",
]
LABEL_COLUMNS = [
    "episode_index",
    "frame_index",
    "index",
    "depth_gripper_close_target",
    "depth_gripper_open_target",
    "depth_gripper_close_delta",
    "depth_gripper_open_delta",
]

EVENT_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.int64()),
        pa.field("event_type", pa.string()),
        pa.field("frame_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("gripper_command", pa.float32()),
        pa.field("closed_interval_start", pa.int64()),
        pa.field("closed_interval_stop", pa.int64()),
        pa.field("closed_interval_length", pa.int64()),
    ]
)
LABEL_SCHEMA = pa.schema(
    [
        pa.field("episode_index", pa.int64()),
        pa.field("frame_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("depth_gripper_close_target", pa.float32()),
        pa.field("depth_gripper_open_target", pa.float32()),
        pa.field("depth_gripper_close_delta", pa.int16()),
        pa.field("depth_gripper_open_delta", pa.int16()),
    ]
)


@dataclass(frozen=True)
class EpisodeBounds:
    episode_index: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def _parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")
    return files


def load_info(root: Path) -> tuple[dict, int]:
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)

    fps = float(info.get("fps", -1))
    if fps != EXPECTED_FPS:
        raise ValueError(f"Expected {EXPECTED_FPS:g} FPS, found {fps:g}")

    try:
        action_feature = info["features"]["action"]
        names = action_feature["names"]
        shape = action_feature["shape"]
    except (KeyError, TypeError) as exc:
        raise ValueError("meta/info.json has no valid action feature metadata") from exc
    if not isinstance(names, list):
        raise ValueError("features.action.names must be a list")
    matches = [i for i, name in enumerate(names) if name == GRIPPER_FEATURE]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {GRIPPER_FEATURE!r} action feature, found {len(matches)}")
    gripper_dim = matches[0]
    if gripper_dim != EXPECTED_GRIPPER_DIM:
        raise ValueError(
            f"Expected {GRIPPER_FEATURE!r} at action dimension {EXPECTED_GRIPPER_DIM}, found {gripper_dim}"
        )
    if shape != [len(names)]:
        raise ValueError(f"Action shape {shape!r} does not agree with {len(names)} feature names")
    return info, gripper_dim


def load_episode_bounds(root: Path) -> list[EpisodeBounds]:
    columns = ["episode_index", "dataset_from_index", "dataset_to_index"]
    frames = [pd.read_parquet(path, columns=columns) for path in _parquet_files(root / "meta" / "episodes")]
    episodes = pd.concat(frames, ignore_index=True).sort_values("episode_index", kind="stable")
    if episodes["episode_index"].duplicated().any():
        raise ValueError("Episode metadata contains duplicate episode indices")

    bounds = [
        EpisodeBounds(int(row.episode_index), int(row.dataset_from_index), int(row.dataset_to_index))
        for row in episodes.itertuples(index=False)
    ]
    if not bounds:
        raise ValueError("Episode metadata is empty")
    expected_start = 0
    for bound in bounds:
        if bound.start != expected_start or bound.stop <= bound.start:
            raise ValueError(
                f"Episode {bound.episode_index} has invalid/noncontiguous bounds "
                f"[{bound.start}, {bound.stop}); expected start {expected_start}"
            )
        expected_start = bound.stop
    return bounds


def load_source_frames(root: Path, action_width: int) -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["action", "episode_index", "frame_index", "index"]
    frames = [pd.read_parquet(path, columns=columns) for path in _parquet_files(root / "data")]
    source = pd.concat(frames, ignore_index=True)
    try:
        action = np.stack(source["action"].to_numpy()).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError("Action rows do not have a uniform vector shape") from exc
    if action.ndim != 2 or action.shape[1] != action_width:
        raise ValueError(f"Expected action shape [N, {action_width}], found {action.shape}")
    if not np.isfinite(action[:, EXPECTED_GRIPPER_DIM]).all():
        raise ValueError("Raw gripper action contains non-finite values")
    return source, action


def validate_source_identities(source: pd.DataFrame, bounds: Sequence[EpisodeBounds], total_frames: int) -> None:
    if len(source) != total_frames:
        raise ValueError(f"Source has {len(source)} rows; metadata records {total_frames}")
    expected_global = np.arange(total_frames, dtype=np.int64)
    if not np.array_equal(source["index"].to_numpy(dtype=np.int64), expected_global):
        raise ValueError("Source global indices are not contiguous and in global order")

    for bound in bounds:
        rows = source.iloc[bound.start : bound.stop]
        if len(rows) != bound.length:
            raise ValueError(f"Episode {bound.episode_index} bounds exceed source data")
        if not np.all(rows["episode_index"].to_numpy() == bound.episode_index):
            raise ValueError(f"Episode identity disagrees with bounds for episode {bound.episode_index}")
        if not np.array_equal(rows["frame_index"].to_numpy(dtype=np.int64), np.arange(bound.length)):
            raise ValueError(f"Local frame indices are not contiguous in episode {bound.episode_index}")
        if not np.array_equal(rows["index"].to_numpy(dtype=np.int64), expected_global[bound.start : bound.stop]):
            raise ValueError(f"Global frame indices disagree with bounds for episode {bound.episode_index}")


def closed_intervals(gripper: np.ndarray, min_frames: int = MIN_CLOSED_FRAMES) -> list[tuple[int, int]]:
    """Return retained half-open closed intervals from the locked hysteresis rule."""
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    closed = False
    for frame, command in enumerate(gripper):
        closed = bool(command > CLOSE_THRESHOLD) if not closed else bool(command >= OPEN_THRESHOLD)
        if closed and start is None:
            start = frame
        elif not closed and start is not None:
            if frame - start >= min_frames:
                intervals.append((start, frame))
            start = None
    if start is not None and len(gripper) - start >= min_frames:
        intervals.append((start, len(gripper)))
    return intervals


def build_events(
    gripper: np.ndarray, bounds: Sequence[EpisodeBounds]
) -> tuple[pd.DataFrame, dict[int, list[tuple[int, int]]]]:
    rows: list[dict] = []
    intervals_by_episode: dict[int, list[tuple[int, int]]] = {}
    for bound in bounds:
        episode_gripper = gripper[bound.start : bound.stop]
        intervals = closed_intervals(episode_gripper)
        intervals_by_episode[bound.episode_index] = intervals
        for start, stop in intervals:
            common = {
                "episode_index": bound.episode_index,
                "closed_interval_start": start,
                "closed_interval_stop": stop,
                "closed_interval_length": stop - start,
            }
            if start > 0:
                rows.append(
                    {
                        **common,
                        "event_type": "close",
                        "frame_index": start,
                        "index": bound.start + start,
                        "gripper_command": episode_gripper[start],
                    }
                )
            if stop < bound.length:
                rows.append(
                    {
                        **common,
                        "event_type": "open",
                        "frame_index": stop,
                        "index": bound.start + stop,
                        "gripper_command": episode_gripper[stop],
                    }
                )

    rows.sort(key=lambda row: (row["index"], row["event_type"]))
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for column in EVENT_COLUMNS:
        if column == "event_type":
            events[column] = events[column].astype("string")
        elif column == "gripper_command":
            events[column] = events[column].astype(np.float32)
        else:
            events[column] = events[column].astype(np.int64)
    return events, intervals_by_episode


def future_event_targets(length: int, event_frames: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute cutoff next-event deltas and exponential targets in one backward pass."""
    event_mask = np.zeros(length, dtype=bool)
    event_array = np.asarray(event_frames, dtype=np.int64)
    if event_array.size:
        if event_array.min() < 0 or event_array.max() >= length or len(np.unique(event_array)) != len(event_array):
            raise ValueError("Event frames must be unique and inside the episode")
        event_mask[event_array] = True

    deltas = np.full(length, -1, dtype=np.int16)
    next_event = -1
    for frame in range(length - 1, -1, -1):
        if event_mask[frame]:
            next_event = frame
        if next_event >= 0 and next_event - frame <= CUTOFF_FRAMES:
            deltas[frame] = next_event - frame

    targets = np.zeros(length, dtype=np.float32)
    valid = deltas >= 0
    targets[valid] = np.exp2(-deltas[valid].astype(np.float32) / HALF_LIFE_FRAMES).astype(np.float32)
    return deltas, targets


def build_labels(
    source: pd.DataFrame, events: pd.DataFrame, bounds: Sequence[EpisodeBounds]
) -> pd.DataFrame:
    total_frames = len(source)
    close_delta = np.full(total_frames, -1, dtype=np.int16)
    open_delta = np.full(total_frames, -1, dtype=np.int16)
    close_target = np.zeros(total_frames, dtype=np.float32)
    open_target = np.zeros(total_frames, dtype=np.float32)

    for bound in bounds:
        episode_events = events[events["episode_index"] == bound.episode_index]
        for event_type, delta_out, target_out in (
            ("close", close_delta, close_target),
            ("open", open_delta, open_target),
        ):
            event_frames = episode_events.loc[episode_events["event_type"] == event_type, "frame_index"]
            delta, target = future_event_targets(bound.length, event_frames.to_numpy(dtype=np.int64))
            delta_out[bound.start : bound.stop] = delta
            target_out[bound.start : bound.stop] = target

    return pd.DataFrame(
        {
            "episode_index": source["episode_index"].to_numpy(dtype=np.int64),
            "frame_index": source["frame_index"].to_numpy(dtype=np.int64),
            "index": source["index"].to_numpy(dtype=np.int64),
            "depth_gripper_close_target": close_target,
            "depth_gripper_open_target": open_target,
            "depth_gripper_close_delta": close_delta,
            "depth_gripper_open_delta": open_delta,
        },
        columns=LABEL_COLUMNS,
    )


def _validate_target_head(
    labels: pd.DataFrame, events: pd.DataFrame, bounds: Sequence[EpisodeBounds], event_type: str
) -> None:
    target = labels[f"depth_gripper_{event_type}_target"].to_numpy(dtype=np.float32)
    delta = labels[f"depth_gripper_{event_type}_delta"].to_numpy(dtype=np.int16)
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError(f"{event_type} targets are non-finite or outside [0, 1]")
    if np.any((delta < -1) | (delta > CUTOFF_FRAMES)):
        raise ValueError(f"{event_type} deltas are outside the allowed range")
    if not np.array_equal(delta == -1, target == 0):
        raise ValueError(f"{event_type} delta=-1 and target=0 do not agree")

    event_indices = events.loc[events["event_type"] == event_type, "index"].to_numpy(dtype=np.int64)
    expected_event_mask = np.zeros(len(labels), dtype=bool)
    expected_event_mask[event_indices] = True
    if not np.array_equal(delta == 0, expected_event_mask):
        raise ValueError(f"{event_type} zero deltas do not exactly identify retained events")
    if not np.all(target[expected_event_mask] == np.float32(1.0)):
        raise ValueError(f"{event_type} event targets are not exactly one")

    ratio = np.float32(2.0 ** (1.0 / HALF_LIFE_FRAMES))
    for bound in bounds:
        d = delta[bound.start : bound.stop]
        y = target[bound.start : bound.stop]
        consecutive = (d[:-1] > 0) & (d[1:] >= 0) & (d[:-1] == d[1:] + 1)
        if not np.allclose(y[1:][consecutive], y[:-1][consecutive] * ratio, rtol=2e-6, atol=1e-7):
            raise ValueError(f"{event_type} target ramp failed in episode {bound.episode_index}")

        episode_event_frames = events.loc[
            (events["episode_index"] == bound.episode_index) & (events["event_type"] == event_type),
            "frame_index",
        ].to_numpy(dtype=np.int64)
        expected_delta, expected_target = future_event_targets(bound.length, episode_event_frames)
        if not np.array_equal(d, expected_delta) or not np.array_equal(y, expected_target):
            raise ValueError(f"{event_type} labels cross a boundary or use the wrong event")


def validate_materialization(
    source: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    bounds: Sequence[EpisodeBounds],
    intervals_by_episode: dict[int, list[tuple[int, int]]],
) -> None:
    if list(events.columns) != EVENT_COLUMNS or list(labels.columns) != LABEL_COLUMNS:
        raise ValueError("Output columns do not match the locked schema")
    if len(labels) != len(source):
        raise ValueError("Frame-label row count does not match the source")
    identity = ["episode_index", "frame_index", "index"]
    if not np.array_equal(labels[identity].to_numpy(), source[identity].to_numpy()):
        raise ValueError("Frame-label identities do not exactly match the source")
    if not events["event_type"].isin(["open", "close"]).all():
        raise ValueError("Event table contains an invalid event type")

    expected_event_rows = []
    for bound in bounds:
        for start, stop in intervals_by_episode[bound.episode_index]:
            if stop - start < MIN_CLOSED_FRAMES:
                raise ValueError("A retained closed interval is too short")
            expected_event_rows.extend(
                (bound.episode_index, event_type, frame)
                for event_type, frame, observable in (
                    ("close", start, start > 0),
                    ("open", stop, stop < bound.length),
                )
                if observable
            )
    actual_event_rows = list(
        events[["episode_index", "event_type", "frame_index"]].itertuples(index=False, name=None)
    )
    if sorted(expected_event_rows) != sorted(actual_event_rows):
        raise ValueError("Retained intervals do not map one-to-one to observable events")
    if np.any(events["closed_interval_length"].to_numpy() < MIN_CLOSED_FRAMES):
        raise ValueError("Event table contains a short closed interval")

    _validate_target_head(labels, events, bounds, "close")
    _validate_target_head(labels, events, bounds, "open")


def _atomic_write_parquet(frame: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
        pq.write_table(table, temporary)
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict, path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _target_statistics(values: np.ndarray) -> dict:
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        "nonzero_fraction": float(np.count_nonzero(values) / len(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "quantiles": {
            f"q{int(q * 100):02d}": float(value)
            for q, value in zip(quantiles, np.quantile(values, quantiles), strict=True)
        },
    }


def build_info(
    source: pd.DataFrame,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    intervals_by_episode: dict[int, list[tuple[int, int]]],
    command: str,
    gripper_dim: int,
    plot_paths: Sequence[Path],
) -> dict:
    return {
        "rubric_version": RUBRIC_VERSION,
        "creation_date": date.today().isoformat(),
        "source_signal": "raw, unnormalized action",
        "resolved_gripper_feature": GRIPPER_FEATURE,
        "resolved_gripper_dimension": gripper_dim,
        "fps": int(EXPECTED_FPS),
        "thresholds_degrees": {"close": CLOSE_THRESHOLD, "open": OPEN_THRESHOLD},
        "persistence": {"frames": MIN_CLOSED_FRAMES, "seconds": MIN_CLOSED_FRAMES / EXPECTED_FPS},
        "target_half_life": {"frames": HALF_LIFE_FRAMES, "seconds": HALF_LIFE_FRAMES / EXPECTED_FPS},
        "target_cutoff": {"frames": CUTOFF_FRAMES, "seconds": CUTOFF_FRAMES / EXPECTED_FPS},
        "row_counts": {
            "source_frames": len(source),
            "event_rows": len(events),
            "frame_label_rows": len(labels),
        },
        "event_counts": {
            "retained_closed_intervals": sum(map(len, intervals_by_episode.values())),
            "close": int((events["event_type"] == "close").sum()),
            "open": int((events["event_type"] == "open").sum()),
        },
        "target_statistics": {
            "close": _target_statistics(labels["depth_gripper_close_target"].to_numpy()),
            "open": _target_statistics(labels["depth_gripper_open_target"].to_numpy()),
        },
        "inputs": {
            "data_columns_read": ["action", "episode_index", "frame_index", "index"],
            "episode_columns_read": ["episode_index", "dataset_from_index", "dataset_to_index"],
            "excluded_inputs": (
                "observation.state, RGB, language, subtasks, quality labels, mistake labels, and all "
                "semantic annotations were not read"
            ),
        },
        "qa_plots": [str(path) for path in plot_paths],
        "command": command,
    }


def _atomic_save_figure(figure, path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=140, bbox_inches="tight")
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_qa_plots(
    root: Path,
    gripper: np.ndarray,
    labels: pd.DataFrame,
    events: pd.DataFrame,
    bounds: Sequence[EpisodeBounds],
    plot_all: bool,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = list(bounds) if plot_all else list(bounds[:2])
    plot_dir = root / "meta" / PLOT_DIRNAME
    plot_dir.mkdir(exist_ok=True)
    paths = []
    for bound in selected:
        path = plot_dir / f"episode_{bound.episode_index:04d}.png"
        local_gripper = gripper[bound.start : bound.stop]
        local_labels = labels.iloc[bound.start : bound.stop]
        local_events = events[events["episode_index"] == bound.episode_index]
        seconds = np.arange(bound.length) / EXPECTED_FPS

        figure, (axis_command, axis_target) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axis_command.plot(seconds, local_gripper, color="black", linewidth=0.8, label="gripper command")
        axis_command.axhline(CLOSE_THRESHOLD, color="tab:red", linestyle="--", label="close threshold")
        axis_command.axhline(OPEN_THRESHOLD, color="tab:blue", linestyle="--", label="open threshold")
        for event_type, color, marker in (("close", "tab:red", "v"), ("open", "tab:blue", "^")):
            event_frames = local_events.loc[local_events["event_type"] == event_type, "frame_index"].to_numpy()
            if len(event_frames):
                axis_command.scatter(
                    event_frames / EXPECTED_FPS,
                    local_gripper[event_frames],
                    color=color,
                    marker=marker,
                    s=24,
                    zorder=3,
                    label=f"{event_type} event",
                )
        axis_command.set_ylabel("command (deg)")
        axis_command.legend(loc="best", ncols=3, fontsize=8)
        axis_command.grid(alpha=0.2)

        axis_target.plot(
            seconds,
            local_labels["depth_gripper_close_target"],
            color="tab:red",
            linewidth=0.9,
            label="future close",
        )
        axis_target.plot(
            seconds,
            local_labels["depth_gripper_open_target"],
            color="tab:blue",
            linewidth=0.9,
            label="future open",
        )
        axis_target.set(xlabel="episode time (s)", ylabel="soft target", ylim=(-0.02, 1.02))
        axis_target.legend(loc="best", fontsize=8)
        axis_target.grid(alpha=0.2)
        figure.suptitle(f"{root.name}: episode {bound.episode_index}")
        _atomic_save_figure(figure, path)
        plt.close(figure)
        paths.append(path.relative_to(root))
    return paths


def _validate_arrow_schema(path: Path, expected: pa.Schema) -> None:
    actual = pq.read_schema(path)
    if actual != expected:
        raise ValueError(f"Unexpected parquet schema in {path}:\nexpected {expected}\nactual {actual}")


def materialize(root: Path, overwrite: bool, plot_all: bool, command: str) -> dict:
    root = root.resolve()
    meta_dir = root / "meta"
    event_path = meta_dir / EVENT_FILENAME
    label_path = meta_dir / LABEL_FILENAME
    info_path = meta_dir / INFO_FILENAME
    destinations = [event_path, label_path, info_path]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace existing sidecars without --overwrite: {names}")

    info, gripper_dim = load_info(root)
    bounds = load_episode_bounds(root)
    total_frames = int(info.get("total_frames", -1))
    if int(info.get("total_episodes", -1)) != len(bounds):
        raise ValueError("Episode count in info.json does not match episode metadata")
    if bounds[-1].stop != total_frames:
        raise ValueError("Final episode bound does not match total_frames in info.json")

    source, action = load_source_frames(root, len(info["features"]["action"]["names"]))
    validate_source_identities(source, bounds, total_frames)
    gripper = action[:, gripper_dim]
    events, intervals_by_episode = build_events(gripper, bounds)
    labels = build_labels(source, events, bounds)
    validate_materialization(source, events, labels, bounds, intervals_by_episode)

    _atomic_write_parquet(events, EVENT_SCHEMA, event_path)
    _atomic_write_parquet(labels, LABEL_SCHEMA, label_path)

    reopened_events = pd.read_parquet(event_path)
    reopened_labels = pd.read_parquet(label_path)
    _validate_arrow_schema(event_path, EVENT_SCHEMA)
    _validate_arrow_schema(label_path, LABEL_SCHEMA)
    validate_materialization(source, reopened_events, reopened_labels, bounds, intervals_by_episode)

    plot_paths = write_qa_plots(root, gripper, reopened_labels, reopened_events, bounds, plot_all)
    provenance = build_info(
        source,
        reopened_events,
        reopened_labels,
        intervals_by_episode,
        command,
        gripper_dim,
        plot_paths,
    )
    _atomic_write_json(provenance, info_path)
    with info_path.open() as f:
        if json.load(f) != provenance:
            raise ValueError("Reopened provenance JSON does not match the generated metadata")
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace existing sidecars and QA plots")
    parser.add_argument(
        "--plot-all-episodes",
        action="store_true",
        help="plot every episode instead of the first two (required for the validation root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = shlex.join([str(Path(sys.executable).resolve()), *sys.argv])
    result = materialize(args.data_dir, args.overwrite, args.plot_all_episodes, command)
    counts = result["event_counts"]
    stats = result["target_statistics"]
    print(
        f"{args.data_dir}: {result['row_counts']['frame_label_rows']} labels, "
        f"{counts['retained_closed_intervals']} intervals, {counts['close']} close events, "
        f"{counts['open']} open events"
    )
    print(
        f"targets: close nonzero={stats['close']['nonzero_fraction']:.1%} "
        f"mean={stats['close']['mean']:.4f}; open nonzero={stats['open']['nonzero_fraction']:.1%} "
        f"mean={stats['open']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
