#!/usr/bin/env python

"""Safely acquire and audit the pinned AgiBot RGB-D pilot archive.

The pilot intentionally starts with the smallest archive published by
``agibot-world/AgiBotWorld2026``. It verifies the immutable Hub metadata and
local SHA-256, scans every tar member before extraction, then audits the
LeRobot parquet, videos, and AgiBot-specific annotation metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import av
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

READ_CHUNK = 8 * 1024 * 1024
REJECTED_TYPES = {
    tarfile.LNKTYPE: "hard_link",
    tarfile.SYMTYPE: "symlink",
    tarfile.CHRTYPE: "character_device",
    tarfile.BLKTYPE: "block_device",
    tarfile.FIFOTYPE: "fifo",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def hub_entry(repo_id: str, revision: str, archive_path: str) -> dict[str, Any]:
    info = HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    matches = [entry for entry in info.siblings if entry.rfilename == archive_path]
    if len(matches) != 1:
        raise ValueError(f"Expected one Hub entry for {archive_path!r}; found {len(matches)}")
    entry = matches[0]
    lfs = entry.lfs
    if entry.size is None or lfs is None or not lfs.sha256:
        raise ValueError(f"Hub entry {archive_path!r} lacks size or LFS SHA-256 metadata")
    return {
        "repo_sha": info.sha,
        "path": entry.rfilename,
        "bytes": int(entry.size),
        "sha256": str(lfs.sha256),
    }


def acquire(manifest: dict[str, Any], archive_spec: dict[str, Any], download_root: Path) -> Path:
    published = hub_entry(
        manifest["source_repo_id"], manifest["source_revision"], archive_spec["path"]
    )
    expected = {
        "repo_sha": manifest["source_revision"],
        "path": archive_spec["path"],
        "bytes": int(archive_spec["bytes"]),
        "sha256": archive_spec["sha256"],
    }
    if published != expected:
        raise ValueError(f"Pinned archive metadata changed: expected {expected}, got {published}")
    local = Path(
        hf_hub_download(
            repo_id=manifest["source_repo_id"],
            repo_type="dataset",
            filename=archive_spec["path"],
            revision=manifest["source_revision"],
            local_dir=download_root,
        )
    )
    actual = {"bytes": local.stat().st_size, "sha256": sha256(local)}
    for key in ("bytes", "sha256"):
        if actual[key] != expected[key]:
            raise ValueError(f"Local archive {key} mismatch: expected {expected[key]}, got {actual[key]}")
    return local


def member_rejection(member: tarfile.TarInfo) -> str | None:
    if member.type in REJECTED_TYPES:
        return REJECTED_TYPES[member.type]
    name = member.name
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return "absolute_path"
    parts = Path(name).parts
    if ".." in parts:
        return "parent_traversal"
    if not parts or parts[0] != "data":
        return "outside_data_root"
    if not member.isfile() and not member.isdir():
        return "unsupported_entry_type"
    return None


def scan_archive(archive_path: Path) -> dict[str, Any]:
    rejected = []
    members = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            reason = member_rejection(member)
            if reason:
                rejected.append({"name": member.name, "reason": reason})
            members.append(
                {"name": member.name, "bytes": member.size, "kind": "file" if member.isfile() else "dir"}
            )
    return {
        "member_count": len(members),
        "file_count": sum(item["kind"] == "file" for item in members),
        "uncompressed_file_bytes": sum(item["bytes"] for item in members if item["kind"] == "file"),
        "members": members,
        "rejected_members": rejected,
    }


def extract_archive(archive_path: Path, extract_root: Path, scan: dict[str, Any]) -> Path:
    if scan["rejected_members"]:
        raise ValueError(f"Refusing extraction: {scan['rejected_members']}")
    dataset_root = extract_root / "data"
    if dataset_root.exists():
        return dataset_root
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(path=extract_root, filter="data")
    return dataset_root


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def arrow_type_shape(field_type: Any) -> list[int] | None:
    shape = []
    while hasattr(field_type, "list_size"):
        shape.append(int(field_type.list_size))
        field_type = field_type.value_type
    return shape or None


def numeric_summary(values: list[Any]) -> dict[str, Any]:
    flat = []
    for value in values:
        if isinstance(value, list):
            flat.extend(value)
        elif isinstance(value, (int, float)):
            flat.append(value)
    numeric = [float(value) for value in flat if isinstance(value, (int, float))]
    finite = [value for value in numeric if math.isfinite(value)]
    result: dict[str, Any] = {"values": len(numeric), "finite_values": len(finite)}
    if finite:
        result.update({"min": min(finite), "max": max(finite), "mean": sum(finite) / len(finite)})
    return result


def audit_parquet(dataset_root: Path) -> dict[str, Any]:
    paths = sorted((dataset_root / "data").rglob("*.parquet"))
    files = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        columns = {}
        table = parquet.read()
        for field in schema:
            column = table[field.name]
            entry: dict[str, Any] = {
                "arrow_type": str(field.type),
                "null_count": column.null_count,
            }
            shape = arrow_type_shape(field.type)
            if shape:
                entry["fixed_shape"] = shape
            if field.name in {"timestamp", "frame_index", "episode_index", "task_index", "action", "observation.state"}:
                entry["summary"] = numeric_summary(column.to_pylist())
            columns[field.name] = entry
        timestamps = table["timestamp"].to_pylist() if "timestamp" in table.column_names else []
        deltas = [float(b) - float(a) for a, b in zip(timestamps, timestamps[1:], strict=False)]
        files.append(
            {
                "path": str(path.relative_to(dataset_root)),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "columns": columns,
                "timestamp": {
                    "start": timestamps[0] if timestamps else None,
                    "end": timestamps[-1] if timestamps else None,
                    "strictly_increasing": all(delta > 0 for delta in deltas),
                    "median_delta": sorted(deltas)[len(deltas) // 2] if deltas else None,
                },
            }
        )
    return {"file_count": len(files), "total_rows": sum(item["rows"] for item in files), "files": files}


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_name,codec_long_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,duration,nb_frames,nb_read_frames:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def decoded_samples(path: Path, is_depth: bool) -> dict[str, Any]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    total = int(stream.frames or 0)
    wanted = {0}
    if total:
        wanted.update({total // 2, total - 1})
    samples = []
    decoded = 0
    for index, frame in enumerate(container.decode(stream)):
        if index in wanted:
            array = frame.to_ndarray()
            item: dict[str, Any] = {
                "frame_index": index,
                "pts": frame.pts,
                "time": float(frame.time) if frame.time is not None else None,
                "format": frame.format.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "min": int(array.min()),
                "max": int(array.max()),
            }
            if is_depth:
                item["zero_fraction"] = float((array == 0).sum() / array.size)
                item["unique_values"] = int(len(set(array.reshape(-1).tolist())))
            samples.append(item)
        decoded = index + 1
    container.close()
    return {"decoded_frames": decoded, "samples": samples}


def audit_videos(dataset_root: Path) -> dict[str, Any]:
    paths = sorted((dataset_root / "videos").rglob("*.mp4"))
    videos = []
    for path in paths:
        key = path.parent.name
        probe = ffprobe(path)
        videos.append(
            {
                "path": str(path.relative_to(dataset_root)),
                "key": key,
                "is_depth": key.endswith("depth"),
                "bytes": path.stat().st_size,
                "probe": probe,
                "decode": decoded_samples(path, key.endswith("depth")),
            }
        )
    counts = Counter(item["decode"]["decoded_frames"] for item in videos)
    return {
        "file_count": len(videos),
        "decoded_frame_count_distribution": dict(sorted(counts.items())),
        "videos": videos,
    }


def find_annotation_fields(value: Any, prefix: str = "") -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    keywords = ("instruction", "segment", "frame", "success", "error", "intervention", "task")
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if any(word in key.lower() for word in keywords):
                found.setdefault(path, []).append(child)
            nested = find_annotation_fields(child, path)
            for nested_path, items in nested.items():
                found.setdefault(nested_path, []).extend(items)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nested = find_annotation_fields(child, f"{prefix}[{index}]")
            for nested_path, items in nested.items():
                found.setdefault(nested_path, []).extend(items)
    return found


def audit_metadata(dataset_root: Path) -> dict[str, Any]:
    info = read_json(dataset_root / "meta" / "info.json")
    episodes = jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = jsonl(dataset_root / "meta" / "tasks.jsonl")
    annotation_fields = find_annotation_fields(info)
    for index, episode in enumerate(episodes):
        nested = find_annotation_fields(episode, f"episodes[{index}]")
        for path, values in nested.items():
            annotation_fields.setdefault(path, []).extend(values)
    return {
        "info": info,
        "episodes": episodes,
        "tasks": tasks,
        "annotation_fields": annotation_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=here / "agibot_pilot.json")
    parser.add_argument(
        "--work-root", type=Path, default=Path("outputs/diverse_robot_dataset/agibot/pilot")
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    if len(manifest["archives"]) != 1:
        raise ValueError("The pilot auditor currently expects exactly one archive")
    archive_spec = manifest["archives"][0]
    if args.skip_download:
        archive_path = args.work_root / archive_spec["path"]
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        actual = {"bytes": archive_path.stat().st_size, "sha256": sha256(archive_path)}
        expected = {key: archive_spec[key] for key in ("bytes", "sha256")}
        if actual != expected:
            raise ValueError(f"Local archive does not match manifest: {actual} != {expected}")
    else:
        archive_path = acquire(manifest, archive_spec, args.work_root)

    scan = scan_archive(archive_path)
    dataset_root = extract_archive(archive_path, args.work_root / "extracted", scan)
    report = {
        "source": manifest,
        "local_archive": str(archive_path),
        "archive_scan": scan,
        "dataset_root": str(dataset_root),
        "metadata": audit_metadata(dataset_root),
        "parquet": audit_parquet(dataset_root),
        "videos": audit_videos(dataset_root),
        "tools": {
            "ffprobe": shutil.which("ffprobe"),
            "pyav": av.__version__,
        },
    }
    report_path = args.work_root / "agibot_pilot_audit.json"
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "rows": report["parquet"]["total_rows"], "videos": report["videos"]["file_count"]}, indent=2))


if __name__ == "__main__":
    main()
