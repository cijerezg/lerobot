#!/usr/bin/env python3
"""Recover a LeRobotDataset whose recording finished but whose video encoding failed.

Background
----------
``LeRobotDataset.save_episode`` writes each episode's frames + metadata to disk
immediately, but when ``batch_encoding_size > 1`` it defers MP4 encoding to a
batch step at the end. There is a bug in
``LeRobotDataset._batch_save_episode_video`` where the FIRST batch crashes with
``TypeError: 'NoneType' object is not subscriptable`` because ``meta.episodes``
hasn't been reloaded from the just-flushed parquet yet. The good news: by the
time the crash propagates through the ``finally`` block in ``lerobot_record``,
``dataset.finalize()`` has already been called and every episode's frame
parquet, image directory, and metadata parquet are valid on disk — only the
videos are missing.

This script picks up from that state:

  * loads the dataset from its local root,
  * for every episode that lacks ``videos/<key>/chunk_index`` metadata, encodes
    a video per camera from its ``images/<key>/episode-NNNNNN/`` directory,
  * places each encoded video under ``videos/<key>/chunk-XXX/file-XXX.mp4``
    using the same chunk/file rotation logic that recording would have used,
  * merges the resulting metadata back into ``meta/episodes/.../*.parquet``,
  * refreshes ``meta/info.json`` with the encoded video info,
  * (optionally) pushes the recovered dataset to the Hugging Face Hub.

Usage
-----
    uv run scripts/recover_video_encoding.py \
        --repo-id jackvial/so101_pickplace_recap_pickplace_20260429_e20

    # Re-encode using a different codec
    uv run scripts/recover_video_encoding.py \
        --repo-id <repo> --vcodec libsvtav1

    # Skip pushing to the Hub
    uv run scripts/recover_video_encoding.py --repo-id <repo> --no-push
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes, write_info
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_VCODEC = "libsvtav1"


def _episode_already_encoded(ep_row: dict, video_keys: list[str]) -> bool:
    for key in video_keys:
        col = f"videos/{key}/chunk_index"
        if col not in ep_row or ep_row[col] is None or pd.isna(ep_row[col]):
            return False
    return True


def recover(
    repo_id: str,
    root: Path | None = None,
    vcodec: str | None = None,
    push_to_hub: bool = True,
    private: bool = False,
    tags: list[str] | None = None,
) -> None:
    if root is None:
        root = HF_LEROBOT_HOME / repo_id
    root = Path(root)

    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"No dataset found at {root}")

    logging.info("Loading dataset metadata from %s", root)
    # Load metadata directly. ``LeRobotDataset.__init__`` would try to validate
    # that every episode's video file exists on disk (or download from the Hub),
    # which is exactly what we are about to fix — so build a minimal stand-in
    # via ``__new__`` that has just what ``_save_episode_video`` and
    # ``_encode_temporary_episode_video`` need.
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    ds = LeRobotDataset.__new__(LeRobotDataset)
    ds.repo_id = repo_id
    ds.root = root
    ds.meta = meta
    ds.vcodec = vcodec or DEFAULT_VCODEC
    ds.image_writer = None
    ds.writer = None
    ds.episode_buffer = None
    ds.latest_episode = None
    ds.batch_encoding_size = 1
    ds.episodes_since_last_encoding = 0
    ds.episodes = None

    video_keys = list(ds.meta.video_keys)
    if not video_keys:
        logging.info("Dataset has no video keys; nothing to encode.")
        return

    total_eps = ds.meta.total_episodes
    logging.info(
        "Dataset has %d episodes, %d video keys (%s).",
        total_eps,
        len(video_keys),
        ", ".join(video_keys),
    )

    # Group episodes by their (data/chunk_index, data/file_index) so we can edit
    # the matching meta/episodes/.../*.parquet file in place once per group.
    episode_rows: list[dict] = [dict(ds.meta.episodes[i]) for i in range(total_eps)]

    # Plan: which episodes still need encoding?
    todo: list[int] = [
        i for i, row in enumerate(episode_rows) if not _episode_already_encoded(row, video_keys)
    ]
    if not todo:
        logging.info("All episodes already have encoded videos. Nothing to do.")
    else:
        logging.info("Episodes needing encoding: %s", todo)

    # ``_save_episode_video`` uses ``meta.latest_episode`` to decide whether to
    # start a new video file or concatenate to the existing one. We need to
    # build that state up incrementally as we encode in episode order.
    # ``meta.episodes`` is also consulted, but only for the "we are resuming
    # recording" branch — that branch is the one that breaks recovery. Force the
    # function down the "first episode" path by clearing ``meta.episodes`` for
    # the duration of the encoding loop.
    saved_meta_episodes = ds.meta.episodes
    ds.meta.episodes = None
    ds.meta.latest_episode = None

    new_video_metadata: dict[int, dict] = {}
    try:
        for ep_idx in todo:
            logging.info("Encoding episode %d / %d ...", ep_idx, total_eps - 1)
            ep_meta: dict = {}
            for video_key in video_keys:
                ep_meta.update(ds._save_episode_video(video_key, ep_idx))
            ep_meta.pop("episode_index", None)
            new_video_metadata[ep_idx] = ep_meta

            # Track latest_episode in the dict-of-lists shape that
            # ``_save_episode_video`` expects (mirrors ``_flush_metadata_buffer``).
            ds.meta.latest_episode = {k: [v] for k, v in ep_meta.items()}
    finally:
        ds.meta.episodes = saved_meta_episodes

    if not new_video_metadata:
        if push_to_hub:
            _push(repo_id, root, tags=tags, private=private)
        return

    # Group new video metadata by the underlying meta/episodes parquet file
    # (one file per (meta/episodes/chunk_index, meta/episodes/file_index) pair).
    by_meta_file: dict[tuple[int, int], list[int]] = {}
    for ep_idx in new_video_metadata:
        row = episode_rows[ep_idx]
        key = (
            int(row["meta/episodes/chunk_index"]),
            int(row["meta/episodes/file_index"]),
        )
        by_meta_file.setdefault(key, []).append(ep_idx)

    for (meta_chunk, meta_file), ep_indices in by_meta_file.items():
        ep_path = root / DEFAULT_EPISODES_PATH.format(
            chunk_index=meta_chunk, file_index=meta_file
        )
        logging.info("Updating %s with video metadata for %d episodes", ep_path, len(ep_indices))

        ep_df = pd.read_parquet(ep_path)

        # Build a DataFrame indexed by the parquet row positions corresponding
        # to each episode_index.  Episodes are written sequentially, so the row
        # number relative to the start of this parquet file is
        # ``ep_idx - first_ep_idx_in_file``.
        first_ep_in_file = min(
            i
            for i, r in enumerate(episode_rows)
            if int(r["meta/episodes/chunk_index"]) == meta_chunk
            and int(r["meta/episodes/file_index"]) == meta_file
        )
        rows_for_df = []
        index_for_df = []
        for ep_idx in ep_indices:
            rows_for_df.append(new_video_metadata[ep_idx])
            index_for_df.append(ep_idx - first_ep_in_file)
        new_cols_df = pd.DataFrame(rows_for_df, index=index_for_df).convert_dtypes(
            dtype_backend="pyarrow"
        )

        merged = ep_df.combine_first(new_cols_df)
        # ``combine_first`` may reorder columns; preserve the original order
        # plus the new video columns at the end.
        col_order = list(ep_df.columns) + [c for c in new_cols_df.columns if c not in ep_df.columns]
        merged = merged[col_order]
        merged.to_parquet(ep_path)

    # Refresh in-memory metadata and persist info (which now knows about
    # encoded video properties for episode 0 — already updated by
    # ``_save_episode_video`` itself).
    ds.meta.episodes = load_episodes(root)
    write_info(ds.meta.info, root)
    logging.info("Recovered %d episodes worth of video encoding.", len(new_video_metadata))

    if push_to_hub:
        _push(repo_id, root, tags=tags, private=private)


def _push(repo_id: str, root: Path, *, tags: list[str] | None, private: bool) -> None:
    """Push the now-complete dataset to the Hub via a fully-constructed LeRobotDataset."""
    logging.info("Loading recovered dataset for push (this validates video files exist) ...")
    ds = LeRobotDataset(repo_id, root=root)
    logging.info("Pushing recovered dataset to %s", repo_id)
    ds.push_to_hub(tags=tags, private=private)
    logging.info("Push complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="HF Hub repo id, e.g. user/dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root (defaults to HF_LEROBOT_HOME/repo_id).",
    )
    parser.add_argument(
        "--vcodec",
        default=None,
        help="Override video codec (e.g. libsvtav1, libx264). Defaults to dataset's recorded codec.",
    )
    parser.add_argument("--no-push", dest="push_to_hub", action="store_false")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument("--tag", action="append", dest="tags", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s:%(lineno)d %(message)s",
    )

    recover(
        repo_id=args.repo_id,
        root=args.root,
        vcodec=args.vcodec,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
