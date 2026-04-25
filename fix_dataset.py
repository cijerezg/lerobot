#!/usr/bin/env python
"""Repair a LeRobot dataset whose table has trailing rows missing from video.

By default this fixes the known local dataset by writing a sibling copy with
``-fixed`` appended. The original dataset is left untouched unless ``--in-place``
is provided.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("outputs/cube-subtasks-e30-base120trim-0-9-101-end")
DEFAULT_TOLERANCE_S = 1e-4


@dataclass(frozen=True)
class VideoBounds:
    path: Path
    frame_count: int
    last_timestamp: float


def _count_video_frames(video_path: Path, fps: float) -> VideoBounds:
    frame_count = 0
    last_timestamp: float | None = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame_count += 1
            if frame.pts is not None:
                last_timestamp = float(frame.pts * frame.time_base)

    if frame_count == 0:
        raise ValueError(f"No decoded frames found in {video_path}")

    if last_timestamp is None:
        last_timestamp = (frame_count - 1) / fps

    return VideoBounds(
        path=video_path,
        frame_count=frame_count,
        last_timestamp=last_timestamp,
    )


def _load_table_parts(root: Path) -> list[tuple[Path, pd.DataFrame]]:
    parts: list[tuple[Path, pd.DataFrame]] = []
    for path in sorted((root / "data").glob("*/*.parquet")):
        parts.append((path, pd.read_parquet(path)))
    if not parts:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return parts


def _write_table_parts(parts: list[tuple[Path, pd.DataFrame]], data: pd.DataFrame) -> None:
    """Rewrite data files preserving the existing single-file layout when present."""
    if len(parts) != 1:
        raise NotImplementedError(
            "This repair script only rewrites single-file datasets. "
            f"Found {len(parts)} data parquet files."
        )

    data.to_parquet(parts[0][0], index=False)


def _video_key_path(root: Path, info: dict, video_key: str, chunk_index: int, file_index: int) -> Path:
    video_path_template = info["video_path"]
    return root / video_path_template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )


def _prepare_target(source: Path, output: Path | None, in_place: bool, overwrite: bool) -> Path:
    if in_place:
        return source

    target = output or source.with_name(f"{source.name}-fixed")
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target} already exists. Remove it, pass --overwrite, or choose --output."
            )
        shutil.rmtree(target)

    shutil.copytree(source, target)
    return target


def _repair_dataset(root: Path, tolerance_s: float) -> int:
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    info = json.loads(info_path.read_text())
    fps = float(info["fps"])
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    if not video_keys:
        raise ValueError("Dataset has no video features to validate")

    table_parts = _load_table_parts(root)
    data = pd.concat([part for _, part in table_parts], ignore_index=True)
    episodes = pd.read_parquet(episodes_path).copy()

    bounds_cache: dict[tuple[str, int, int], VideoBounds] = {}
    for _, episode in episodes.iterrows():
        for video_key in video_keys:
            chunk_index = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            cache_key = (video_key, chunk_index, file_index)
            if cache_key in bounds_cache:
                continue
            video_path = _video_key_path(root, info, video_key, chunk_index, file_index)
            bounds_cache[cache_key] = _count_video_frames(video_path, fps)

    keep_masks: list[pd.Series] = []
    dropped_rows = 0

    for _, episode in episodes.sort_values("episode_index").iterrows():
        episode_index = int(episode["episode_index"])
        episode_df = data[data["episode_index"] == episode_index].copy()
        if episode_df.empty:
            raise ValueError(f"Episode {episode_index} has no frame rows")

        max_relative_timestamp = float("inf")
        for video_key in video_keys:
            chunk_index = int(episode[f"videos/{video_key}/chunk_index"])
            file_index = int(episode[f"videos/{video_key}/file_index"])
            from_timestamp = float(episode[f"videos/{video_key}/from_timestamp"])
            video_bounds = bounds_cache[(video_key, chunk_index, file_index)]
            max_relative_timestamp = min(
                max_relative_timestamp,
                video_bounds.last_timestamp - from_timestamp,
            )

        episode_keep = episode_df["timestamp"].astype(float) <= max_relative_timestamp + tolerance_s
        dropped_rows += int((~episode_keep).sum())
        keep_masks.append(pd.Series(episode_keep.to_numpy(), index=episode_df.index))

    if dropped_rows == 0:
        return 0

    keep_mask = pd.concat(keep_masks).sort_index()
    repaired = data.loc[keep_mask].copy().reset_index(drop=True)
    repaired["index"] = np.arange(len(repaired), dtype=np.int64)

    new_from_index = 0
    for row_idx, episode in episodes.sort_values("episode_index").iterrows():
        episode_index = int(episode["episode_index"])
        episode_mask = repaired["episode_index"] == episode_index
        episode_length = int(episode_mask.sum())
        if episode_length == 0:
            raise ValueError(f"Repair would remove all rows from episode {episode_index}")

        episode_row_indices = repaired.index[episode_mask]
        repaired.loc[episode_row_indices, "frame_index"] = np.arange(episode_length, dtype=np.int64)

        episodes.loc[row_idx, "length"] = episode_length
        episodes.loc[row_idx, "dataset_from_index"] = new_from_index
        episodes.loc[row_idx, "dataset_to_index"] = new_from_index + episode_length
        new_from_index += episode_length

        for video_key in video_keys:
            from_timestamp = float(episodes.loc[row_idx, f"videos/{video_key}/from_timestamp"])
            episodes.loc[row_idx, f"videos/{video_key}/to_timestamp"] = (
                from_timestamp + episode_length / fps
            )

    info["total_frames"] = int(len(repaired))

    _write_table_parts(table_parts, repaired)
    episodes.to_parquet(episodes_path, index=False)
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    return dropped_rows


def _validate_last_items(root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/fixed-dataset", root=root, video_backend="pyav")
    for episode in dataset.meta.episodes:
        last_index = int(episode["dataset_to_index"]) - 1
        _ = dataset[last_index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix trailing frame rows that point past the end of LeRobot videos."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root to repair. Defaults to {DEFAULT_DATASET}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset path. Defaults to '<dataset>-fixed'. Ignored with --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Repair the dataset in place instead of writing a fixed copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help="Timestamp tolerance used when comparing rows to video frames.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the final per-episode last-frame decode validation.",
    )
    args = parser.parse_args()

    source = args.dataset
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")

    target = _prepare_target(source, args.output, args.in_place, args.overwrite)
    dropped_rows = _repair_dataset(target, args.tolerance_s)

    if not args.skip_validation:
        _validate_last_items(target)

    print(f"dataset: {target}")
    print(f"dropped_rows: {dropped_rows}")
    if dropped_rows:
        print("status: repaired")
        print("note: meta/stats.json was not recomputed; counts may be stale by the dropped rows.")
    else:
        print("status: no repair needed")


if __name__ == "__main__":
    main()
