#!/usr/bin/env python
"""Upload the repaired cube-subtasks dataset to the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ROOT = Path("outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed")
DEFAULT_REPO_ID = "jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed"


def _validate_decode(dataset: LeRobotDataset) -> None:
    """Decode the final frame of each episode before starting an upload."""
    for episode in dataset.meta.episodes:
        last_index = int(episode["dataset_to_index"]) - 1
        _ = dataset[last_index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a fixed LeRobot dataset to Hugging Face Hub.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local dataset root. Defaults to {DEFAULT_ROOT}.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Destination Hub dataset repo id. Defaults to {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload the Hub dataset as private.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional Hub branch to upload to.",
    )
    parser.add_argument(
        "--no-large-folder",
        action="store_true",
        help="Use regular upload_folder instead of upload_large_folder.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip local last-frame decode validation before upload.",
    )
    parser.add_argument(
        "--push-without-videos",
        action="store_true",
        help="Upload metadata/data only and skip videos.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {args.root}. Run fix_dataset.py first or pass --root."
        )

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        video_backend="pyav",
    )

    if not args.skip_validation:
        print("Validating final frame decode for each episode...")
        _validate_decode(dataset)

    print(f"Uploading {args.root} to https://huggingface.co/datasets/{args.repo_id}")
    dataset.push_to_hub(
        branch=args.branch,
        tags=["LeRobot", "robotics"],
        private=args.private,
        push_videos=not args.push_without_videos,
        upload_large_folder=not args.no_large_folder,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
