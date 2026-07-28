#!/usr/bin/env python

"""Build the independent replay caches declared by an offline RL config."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def _configured_sources(config: dict) -> list[dict]:
    dataset = config.get("dataset") or {}
    sources = list(dataset.get("sources") or [])
    legacy = list(dataset.get("additional_offline_dataset_paths") or [])
    if sources and legacy:
        raise ValueError("Use dataset.sources or additional_offline_dataset_paths, not both.")
    if sources:
        return sources
    primary_root = dataset.get("root")
    if primary_root is None:
        raise ValueError("The cache builder requires local dataset roots.")
    return [
        {"name": Path(primary_root).name, "root": primary_root},
        *({"name": Path(path).name, "root": path} for path in legacy),
    ]


def _storage_size_args(policy: dict) -> list[str]:
    size = policy.get("image_storage_size")
    if size is None:
        return ["raw"]
    if len(size) != 2:
        raise ValueError(f"policy.image_storage_size must have two values or be null, got {size!r}.")
    return [str(size[0]), str(size[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Offline RL YAML containing dataset.sources.")
    parser.add_argument("--cache-dir", default=None, help="Override config buffer_cache_dir.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=512)
    parser.add_argument("--drop-cache", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild caches that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print builder commands without running them.")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    dataset = config.get("dataset") or {}
    policy = config.get("policy") or {}
    cache_dir = args.cache_dir or config.get("buffer_cache_dir")
    if not cache_dir:
        raise ValueError("Set buffer_cache_dir in the config or pass --cache-dir.")

    builder = Path(__file__).with_name("lerobot_memmap_buffer_cache.py")
    fallback_repo_id = dataset.get("repo_id") or "local/dataset"
    video_backend = dataset.get("video_backend", "pyav")
    image_dtype = policy.get("image_storage_dtype", "uint8")
    image_stride = int(policy.get("image_stride", 1))

    sources = _configured_sources(config)
    print(f"Preparing {len(sources)} independent replay caches under {cache_dir}")
    for source in sources:
        root = source.get("root")
        if not root:
            raise ValueError(f"Dataset source is missing root: {source!r}")
        name = source.get("name") or Path(root).name
        command = [
            sys.executable,
            str(builder),
            "--data-dir",
            str(root),
            "--repo-id",
            str(source.get("repo_id") or fallback_repo_id),
            "--cache-dir",
            str(cache_dir),
            "--video-backend",
            str(video_backend),
            "--image-storage-dtype",
            str(image_dtype),
            "--image-storage-size",
            *_storage_size_args(policy),
            "--image-stride",
            str(image_stride),
            "--flush-every",
            str(args.flush_every),
        ]
        if args.num_workers is not None:
            command.extend(["--num-workers", str(args.num_workers)])
        if args.drop_cache:
            command.append("--drop-cache")
        if not args.rebuild:
            command.append("--skip-existing")
        print(f"[{name}] {root}")
        if args.dry_run:
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
