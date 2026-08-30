#!/usr/bin/env python

"""Acquire one RoboChallenge task archive into bounded staging.

This covers the first two steps of the production recipe: download only the
requested task at the pinned revision and verify published bytes and SHA-256
values, then scan the concatenated tar before extracting it. The scan is a
separate streaming pass so traversal paths, links, and device entries are
rejected before anything is written to disk.

The split `.tar.part-NNNN` files are a plain byte split of one tar archive, so
they are read as a single concatenated stream rather than extracted one by one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SOURCE_REPO_ID = "RoboChallenge/Table30v2"
SOURCE_REVISION = "c58ad2cc76ce722ea54de51f9fae03012a698f47"
REJECTED_TYPES = {
    tarfile.LNKTYPE: "hard_link",
    tarfile.SYMTYPE: "symlink",
    tarfile.CHRTYPE: "character_device",
    tarfile.BLKTYPE: "block_device",
    tarfile.FIFOTYPE: "fifo",
}
READ_CHUNK = 8 * 1024 * 1024


@dataclass
class ArchivePart:
    path: str
    bytes: int
    sha256: str


class ConcatenatedParts:
    """Read several archive parts as the single tar stream they were split from."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._index = 0
        self._stream = paths[0].open("rb")

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            buffer = b""
            while True:
                block = self.read(READ_CHUNK)
                if not block:
                    return buffer
                buffer += block
        buffer = b""
        while len(buffer) < size:
            block = self._stream.read(size - len(buffer))
            if block:
                buffer += block
                continue
            if self._index + 1 >= len(self._paths):
                break
            self._stream.close()
            self._index += 1
            self._stream = self._paths[self._index].open("rb")
        return buffer

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> ConcatenatedParts:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def published_parts(task: str, revision: str) -> list[ArchivePart]:
    entries = HfApi().list_repo_tree(
        SOURCE_REPO_ID, repo_type="dataset", revision=revision, recursive=True, expand=True
    )
    parts = []
    for entry in entries:
        path = getattr(entry, "path", "")
        if not path.startswith(f"{task}.tar.part-"):
            continue
        lfs = getattr(entry, "lfs", None)
        if lfs is None or not getattr(lfs, "sha256", None):
            raise ValueError(f"{path} does not publish an LFS SHA-256")
        parts.append(ArchivePart(path=path, bytes=int(entry.size), sha256=str(lfs.sha256)))
    if not parts:
        raise ValueError(f"No archive parts found for task {task!r} at revision {revision}")
    return sorted(parts, key=lambda part: part.path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download_task(task: str, downloads_root: Path, revision: str, expected_bytes: int | None) -> dict:
    parts = published_parts(task, revision)
    total = sum(part.bytes for part in parts)
    if expected_bytes is not None and total != expected_bytes:
        raise ValueError(
            f"{task} publishes {total} archive bytes; the production manifest expects {expected_bytes}"
        )
    destination = downloads_root / task
    destination.mkdir(parents=True, exist_ok=True)
    verified = []
    for part in parts:
        local = Path(
            hf_hub_download(
                SOURCE_REPO_ID,
                part.path,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
            )
        )
        actual_bytes = local.stat().st_size
        if actual_bytes != part.bytes:
            raise ValueError(f"{part.path}: downloaded {actual_bytes} bytes, published {part.bytes}")
        actual_sha256 = file_sha256(local)
        if actual_sha256 != part.sha256:
            raise ValueError(
                f"{part.path}: SHA-256 {actual_sha256} does not match published {part.sha256}"
            )
        verified.append({"path": part.path, "bytes": part.bytes, "sha256": part.sha256})
        print(f"verified {part.path} ({part.bytes} bytes)", flush=True)
    return {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": revision,
        "task": task,
        "archive_files": verified,
        "archive_bytes": total,
        "download_root": str(destination),
    }


def _member_rejection(member: tarfile.TarInfo, task: str) -> str | None:
    if member.type in REJECTED_TYPES:
        return REJECTED_TYPES[member.type]
    name = member.name
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return "absolute_path"
    parts = Path(name).parts
    if ".." in parts:
        return "parent_traversal"
    if not parts or parts[0] != task:
        return "outside_task_root"
    if not member.isfile() and not member.isdir():
        return "unsupported_entry_type"
    return None


def scan_archive(task: str, part_paths: list[Path]) -> dict:
    rejected: list[dict] = []
    files = 0
    directories = 0
    member_bytes = 0
    with ConcatenatedParts(part_paths) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            reason = _member_rejection(member, task)
            if reason is not None:
                rejected.append({"name": member.name, "reason": reason})
                if len(rejected) >= 50:
                    break
                continue
            if member.isdir():
                directories += 1
            else:
                files += 1
                member_bytes += member.size
    return {
        "members_files": files,
        "members_directories": directories,
        "member_bytes": member_bytes,
        "rejected_members": rejected,
        "checks": [
            "absolute_path",
            "parent_traversal",
            "outside_task_root",
            "hard_link",
            "symlink",
            "character_device",
            "block_device",
            "fifo",
        ],
    }


def extract_archive(task: str, part_paths: list[Path], raw_root: Path) -> dict:
    destination = raw_root / task
    if (destination / task).exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction: {destination / task}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with ConcatenatedParts(part_paths) as stream, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            reason = _member_rejection(member, task)
            if reason is not None:
                raise ValueError(f"Rejected tar member {member.name!r} during extraction: {reason}")
            archive.extract(member, path=destination, filter="data")
            extracted += 1
    episodes = sorted((destination / task / "data").glob("episode_*"))
    return {
        "raw_root": str(destination / task),
        "extracted_members": extracted,
        "episode_directories": len(episodes),
    }


def local_part_paths(task: str, downloads_root: Path, manifest: dict) -> list[Path]:
    paths = [downloads_root / task / entry["path"] for entry in manifest["archive_files"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing downloaded archive parts: {missing}")
    return paths


def acquire(
    task: str,
    downloads_root: Path,
    raw_root: Path,
    revision: str,
    production_manifest: Path | None,
    skip_download: bool,
) -> dict:
    expected_bytes = None
    embodiment = None
    if production_manifest is not None:
        entries = {item["name"]: item for item in read_json(production_manifest)["tasks"]}
        if task not in entries:
            raise ValueError(f"{task!r} is not listed in {production_manifest}")
        expected_bytes = int(entries[task]["archive_bytes"])
        embodiment = entries[task]["embodiment"]

    record_path = raw_root / task / "raw_acquisition.json"
    if skip_download and record_path.is_file():
        acquisition = read_json(record_path)
    else:
        acquisition = download_task(task, downloads_root, revision, expected_bytes)
        write_json(record_path, acquisition)
    if embodiment is not None:
        acquisition["embodiment"] = embodiment

    part_paths = local_part_paths(task, downloads_root, acquisition)
    print(f"scanning {len(part_paths)} archive parts before extraction", flush=True)
    scan = scan_archive(task, part_paths)
    acquisition["archive_scan"] = scan
    write_json(record_path, acquisition)
    if scan["rejected_members"]:
        raise ValueError(
            f"Archive scan rejected {len(scan['rejected_members'])} members; extraction refused"
        )
    print(f"scan passed: {scan['members_files']} files, {scan['member_bytes']} bytes", flush=True)

    acquisition["extraction"] = extract_archive(task, part_paths, raw_root)
    acquisition["cleanup_policy"] = (
        "raw archive, extracted task, and temporary full-rate conversion are removed only after"
        " packed validation passes"
    )
    write_json(record_path, acquisition)
    return acquisition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--downloads-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=Path(__file__).with_name("robochallenge_production.json"),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse an existing verified raw_acquisition.json and only scan and extract",
    )
    args = parser.parse_args()
    result = acquire(
        args.task,
        args.downloads_root,
        args.raw_root,
        args.revision,
        args.production_manifest,
        args.skip_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
