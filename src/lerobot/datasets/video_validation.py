#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.video_utils import get_video_duration_in_s


def _episodes_to_dataframe(episodes) -> pd.DataFrame:
    if isinstance(episodes, pd.DataFrame):
        return episodes.copy()
    if hasattr(episodes, "to_pandas"):
        return episodes.to_pandas()
    return pd.DataFrame(list(episodes))


def _is_missing(value) -> bool:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        if len(value) == 0:
            return True
        value = value[0]
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def validate_video_metadata(
    root: Path | str,
    episodes,
    video_keys: Iterable[str],
    video_path: str | None,
    fps: int,
    tolerance_s: float | None = None,
    episode_indices: Iterable[int] | None = None,
) -> None:
    """Validate LeRobot per-episode video metadata against referenced MP4 files.

    The checks are intentionally focused on metadata corruption that can make a
    structurally valid dataset play the wrong frames, especially repeated
    ``from_timestamp = 0`` rows produced by broken batched encoding.
    """
    video_keys = list(video_keys)
    if len(video_keys) == 0:
        return
    if video_path is None:
        raise ValueError("Invalid video metadata: dataset has video features but no video_path is set.")

    root = Path(root)
    episodes_df = _episodes_to_dataframe(episodes)
    if len(episodes_df) == 0:
        return

    if episode_indices is not None:
        episode_indices = set(episode_indices)
        episodes_df = episodes_df[episodes_df["episode_index"].isin(episode_indices)].copy()
        if len(episodes_df) == 0:
            return

    tolerance_s = tolerance_s if tolerance_s is not None else max(0.05, 2.0 / fps)
    overlap_tolerance_s = max(1e-4, 0.25 / fps)
    errors: list[str] = []
    duration_cache: dict[tuple[str, int, int], tuple[Path, float | None]] = {}

    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"
        required_cols = [chunk_col, file_col, from_col, to_col]

        missing_cols = [col for col in required_cols if col not in episodes_df.columns]
        if missing_cols:
            errors.append(f"{video_key}: missing required episode metadata columns: {missing_cols}")
            continue

        key_df = episodes_df[
            ["episode_index", "length", chunk_col, file_col, from_col, to_col]
        ].copy()

        for _, row in key_df.iterrows():
            ep_idx = int(row["episode_index"])
            missing_values = [col for col in required_cols if _is_missing(row[col])]
            if missing_values:
                errors.append(
                    f"{video_key} episode {ep_idx}: missing video metadata values: {missing_values}"
                )
                continue

            from_ts = float(row[from_col])
            to_ts = float(row[to_col])
            if to_ts <= from_ts:
                errors.append(
                    f"{video_key} episode {ep_idx}: to_timestamp ({to_ts}) must be greater than "
                    f"from_timestamp ({from_ts})."
                )
                continue

            expected_duration = float(row["length"]) / fps
            actual_range_duration = to_ts - from_ts
            if abs(actual_range_duration - expected_duration) > tolerance_s:
                errors.append(
                    f"{video_key} episode {ep_idx}: metadata duration "
                    f"{_format_seconds(actual_range_duration)}s does not match episode length/fps "
                    f"{_format_seconds(expected_duration)}s within tolerance "
                    f"{_format_seconds(tolerance_s)}s."
                )

            chunk_idx = int(row[chunk_col])
            file_idx = int(row[file_col])
            cache_key = (video_key, chunk_idx, file_idx)
            if cache_key not in duration_cache:
                path = root / video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                if not path.exists():
                    errors.append(
                        f"{video_key} episode {ep_idx}: referenced video file does not exist: "
                        f"{path.relative_to(root) if path.is_relative_to(root) else path}"
                    )
                    duration_cache[cache_key] = (path, None)
                else:
                    duration_cache[cache_key] = (path, get_video_duration_in_s(path))

            _, mp4_duration = duration_cache[cache_key]
            if mp4_duration is not None and to_ts > mp4_duration + tolerance_s:
                errors.append(
                    f"{video_key} episode {ep_idx}: to_timestamp {_format_seconds(to_ts)}s exceeds "
                    f"MP4 duration {_format_seconds(mp4_duration)}s for chunk={chunk_idx} file={file_idx}."
                )

        complete_rows = key_df.dropna(subset=required_cols)
        for (chunk_idx, file_idx), group in complete_rows.groupby([chunk_col, file_col], sort=True):
            chunk_idx = int(chunk_idx)
            file_idx = int(file_idx)
            group = group.sort_values("episode_index").copy()
            if len(group) == 0:
                continue

            from_values = group[from_col].astype(float).to_numpy()
            to_values = group[to_col].astype(float).to_numpy()
            lengths = group["length"].astype(float).to_numpy()
            episode_ids = group["episode_index"].astype(int).tolist()
            expected_group_duration = float(lengths.sum() / fps)
            _, mp4_duration = duration_cache.get((video_key, chunk_idx, file_idx), (Path(), None))

            if len(group) > 1 and np.all(np.isclose(from_values, 0.0, atol=overlap_tolerance_s)):
                errors.append(
                    f"Invalid video metadata for {video_key} chunk={chunk_idx} file={file_idx}: "
                    f"{len(group)} episodes reference the same file, but all from_timestamp values are 0 "
                    f"and the MP4 duration is {_format_seconds(mp4_duration)}s while expected grouped "
                    f"duration is {_format_seconds(expected_group_duration)}s. "
                    "This likely indicates failed batched video encoding."
                )

            if mp4_duration is not None and expected_group_duration > mp4_duration + tolerance_s:
                errors.append(
                    f"Invalid video metadata for {video_key} chunk={chunk_idx} file={file_idx}: "
                    f"{len(group)} episodes reference this file with expected grouped duration "
                    f"{_format_seconds(expected_group_duration)}s, but the MP4 duration is only "
                    f"{_format_seconds(mp4_duration)}s. This likely indicates failed batched video encoding."
                )

            for pos in range(1, len(group)):
                prev_ep = episode_ids[pos - 1]
                ep_idx = episode_ids[pos]
                if from_values[pos] < from_values[pos - 1] - overlap_tolerance_s:
                    errors.append(
                        f"{video_key} chunk={chunk_idx} file={file_idx}: timestamps are not monotonic by "
                        f"episode order between episodes {prev_ep} and {ep_idx}."
                    )
                if from_values[pos] < to_values[pos - 1] - overlap_tolerance_s:
                    errors.append(
                        f"{video_key} chunk={chunk_idx} file={file_idx}: timestamp ranges overlap between "
                        f"episodes {prev_ep} and {ep_idx}."
                    )
                if abs(from_values[pos] - from_values[pos - 1]) <= overlap_tolerance_s:
                    errors.append(
                        f"{video_key} chunk={chunk_idx} file={file_idx}: duplicate from_timestamp "
                        f"{_format_seconds(from_values[pos])}s for distinct episodes {prev_ep} and "
                        f"{ep_idx}. This likely indicates failed batched video encoding."
                    )

    if errors:
        raise ValueError("Invalid LeRobotDataset video metadata:\n- " + "\n- ".join(errors))
