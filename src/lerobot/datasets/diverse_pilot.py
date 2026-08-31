#!/usr/bin/env python

"""Admission-gated acquisition helpers for the diverse real-robot pilot.

Metadata inspection is deliberately separate from payload acquisition. A
source cannot produce a download plan until its audit proves that a usable
joint-action field exists. Native commanded actions are preserved when
available. A source may explicitly opt into copying its measured joint state
into the action field; this exception is recorded in provenance. Values are
only selected or linearly interpolated in time: no normalization, delta
conversion, joint remapping, clipping, or gripper transformation is
implemented here.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

HISTORY_OFFSETS_SECONDS = np.asarray([-6, -5, -4, -3, -2, -1, 0], dtype=np.float64)
ACTION_OFFSETS_SECONDS = np.arange(30, dtype=np.float64) / 30.0
PACKED_EXTENSION = "lerobot-v3-packed-sample"
PACKED_EXTENSION_VERSION = 1
PAYLOAD_PREFIXES = ("data/", "videos/", "images/")


class AdmissionError(RuntimeError):
    """Payload acquisition was attempted for an unaudited source."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    revision: str
    source_format: str
    pilot_episodes: int
    robot_type: str
    license: str | None
    real_robot_evidence: str
    state_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    gripper_field: str | None
    state_semantics: str
    action_semantics: str
    gripper_semantics: str
    joint_names: tuple[str, ...]
    metadata_patterns: tuple[str, ...]
    action_source: str = "native"
    usage_basis: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceSpec:
        value = dict(value)
        for key in ("state_fields", "action_fields", "joint_names", "metadata_patterns"):
            value[key] = tuple(value.get(key, ()))
        return cls(**value)

    def __post_init__(self) -> None:
        if self.action_source not in {"native", "copy_state"}:
            raise ValueError(f"Unsupported action_source: {self.action_source}")
        if self.action_source == "copy_state" and not self.state_fields:
            raise ValueError("copy_state action sources require configured state_fields")


@dataclass(frozen=True)
class ActionChunk:
    values: np.ndarray
    target_timestamps: np.ndarray
    source_timestamps: np.ndarray
    source_indices: np.ndarray
    source_values: np.ndarray
    interpolated_mask: np.ndarray


@dataclass(frozen=True)
class NumericalPackedSample:
    anchor_timestamp: float
    observation_values: np.ndarray
    observation_timestamps: np.ndarray
    observation_frame_indices: np.ndarray
    observation_timing_error: np.ndarray
    action: ActionChunk


@dataclass(frozen=True)
class SourceEpisodeArrays:
    episode_index: int
    tasks: tuple[str, ...]
    timestamps: np.ndarray
    states: np.ndarray
    actions: np.ndarray


def load_source_specs(config_path: Path) -> list[SourceSpec]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return [SourceSpec.from_dict(item) for item in config["sources"]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _feature_summary(features: dict[str, dict], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for name in fields:
        feature = features.get(name)
        result.append(
            {
                "field": name,
                "present": feature is not None,
                "dtype": None if feature is None else feature.get("dtype"),
                "shape": None if feature is None else feature.get("shape"),
                "names": None if feature is None else feature.get("names"),
                "fps": None if feature is None else feature.get("fps"),
            }
        )
    return result


def _camera_summary(features: dict[str, dict]) -> list[dict[str, Any]]:
    cameras = []
    for name, feature in features.items():
        if feature.get("dtype") not in {"image", "video"}:
            continue
        info = feature.get("info", {})
        cameras.append(
            {
                "field": name,
                "dtype": feature.get("dtype"),
                "shape": feature.get("shape"),
                "fps": info.get("video.fps"),
                "codec": info.get("video.codec"),
                "pixel_format": info.get("video.pix_fmt"),
                "is_depth": bool(info.get("video.is_depth_map", False)),
            }
        )
    return cameras


def _task_vocabulary(metadata_root: Path, limit: int = 50) -> tuple[int | None, list[str]]:
    path = metadata_root / "meta/tasks.parquet"
    if not path.exists():
        return None, []
    table = pq.read_table(path)
    values: list[str] = []
    for name in (name for name in table.column_names if name != "task_index"):
        values.extend(str(value) for value in table[name].to_pylist() if value is not None)
    return table.num_rows, values[:limit]


def fetch_metadata(spec: SourceSpec, metadata_root: Path, *, token: str | None = None) -> Path:
    """Fetch only allowlisted metadata; payload prefixes are forbidden."""
    for pattern in spec.metadata_patterns:
        normalized = pattern.lstrip("/")
        if normalized.startswith(PAYLOAD_PREFIXES):
            raise ValueError(f"Metadata pattern must not include a payload prefix: {pattern}")
    local_dir = metadata_root / spec.name
    snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=local_dir,
        allow_patterns=list(spec.metadata_patterns),
        token=token,
    )
    return local_dir


def audit_source(
    spec: SourceSpec,
    metadata_root: Path,
    *,
    resolved_revision: str | None = None,
    repo_tags: list[str] | None = None,
    gated: bool | str | None = None,
) -> dict[str, Any]:
    """Build an admission audit from previously fetched metadata."""
    info_path = metadata_root / "meta/info.json"
    info = _read_json(info_path) if info_path.exists() else {}
    features = info.get("features", {})
    task_count, vocabulary = _task_vocabulary(metadata_root)
    state_fields = _feature_summary(features, spec.state_fields)
    action_fields = _feature_summary(features, spec.action_fields)
    cameras = _camera_summary(features)
    native_joint_action = (
        spec.action_source == "native"
        and bool(spec.action_fields)
        and all(item["present"] for item in action_fields)
    )
    copied_state_action = spec.action_source == "copy_state" and bool(spec.state_fields)
    training_action_available = native_joint_action or copied_state_action
    state_present = bool(spec.state_fields) and all(item["present"] for item in state_fields)
    declared_license = spec.license or _license_from_tags(repo_tags or [])

    failures = []
    if not declared_license and not spec.usage_basis:
        failures.append("No source license or usage grant is declared.")
    if not spec.real_robot_evidence:
        failures.append("Real-robot provenance is not documented.")
    if not training_action_available:
        failures.append("No usable joint-action field or approved state-copy exception is available.")
    if spec.source_format == "lerobot_v3" and not state_present:
        failures.append("Configured native joint-state field is missing from meta/info.json.")
    if spec.source_format == "lerobot_v3" and info.get("codebase_version") != "v3.0":
        failures.append("Source is not LeRobot v3 as configured.")

    return {
        "audit_version": 1,
        "repository": {
            "repo_id": spec.repo_id,
            "requested_revision": spec.revision,
            "resolved_revision": resolved_revision or spec.revision,
            "gated": gated,
            "license": declared_license,
            "usage_basis": spec.usage_basis,
            "usage_restrictions": "See source license/card; no additional grant inferred.",
        },
        "real_robot_evidence": spec.real_robot_evidence,
        "source_format": spec.source_format,
        "lerobot_version": info.get("codebase_version"),
        "robot": {"model": spec.robot_type, "embodiment": "single-arm"},
        "episodes": {
            "count": info.get("total_episodes"),
            "task_count": info.get("total_tasks", task_count),
            "task_vocabulary_sample": vocabulary,
        },
        "state": {
            "fields": state_fields,
            "semantics": spec.state_semantics,
            "joint_names": list(spec.joint_names) or ["unknown"],
        },
        "action": {
            "fields": state_fields if copied_state_action else action_fields,
            "semantics": spec.action_semantics,
            "source": spec.action_source,
            "native_commanded_joint_action": native_joint_action,
            "copied_from_state": copied_state_action,
            "training_action_available": training_action_available,
            "policy": (
                "Copy the configured measured joint-state vector exactly into action."
                if copied_state_action
                else "Preserve values; concatenate configured fields in listed order only."
            ),
        },
        "gripper": {
            "field": spec.gripper_field,
            "semantics": spec.gripper_semantics,
            "policy": "No inversion, thresholding, clipping, scaling, or canonicalization.",
        },
        "timing": {
            "nominal_state_action_fps": info.get("fps"),
            "measured_rate": "pending low-dimensional shard inspection",
            "timestamp_field": "timestamp" if "timestamp" in features else "documented source timestamp",
            "clock_behavior": "pending low-dimensional shard inspection",
        },
        "cameras": cameras,
        "depth": [camera for camera in cameras if camera["is_depth"]],
        "annotations": {
            "source_tasks": bool(vocabulary),
            "pilot_subtask_quality_mistake": "pending full-episode review",
        },
        "layout": {
            "data_path": info.get("data_path"),
            "video_path": info.get("video_path"),
            "minimum_payload": (
                "one data shard and each camera shard referenced by episode metadata"
                if info
                else "task archive parts; episode minimum unresolved without archive index"
            ),
        },
        "notes": spec.notes,
        "admission": {"passed": not failures, "failures": failures},
    }


def _license_from_tags(tags: list[str]) -> str | None:
    return next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), None)


def run_source_audits(
    specs: list[SourceSpec],
    metadata_root: Path,
    output_root: Path,
    *,
    fetch: bool = True,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve revisions, fetch metadata only, and persist one audit per source."""
    api = HfApi(token=token)
    output_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for spec in specs:
        repo_info = api.dataset_info(spec.repo_id, revision=spec.revision)
        root = fetch_metadata(spec, metadata_root, token=token) if fetch else metadata_root / spec.name
        audit = audit_source(
            spec,
            root,
            resolved_revision=repo_info.sha,
            repo_tags=list(repo_info.tags or []),
            gated=repo_info.gated,
        )
        write_json(output_root / f"{spec.name}.json", audit)
        audits.append(audit)
    return audits


def read_episode_table(metadata_root: Path):
    candidates = sorted((metadata_root / "meta/episodes").glob("**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No episode metadata Parquet beneath {metadata_root}")
    return pq.read_table(candidates)


def _table_column_array(table, field: str) -> np.ndarray:
    values = table[field].to_pylist()
    leaf_type = table.schema.field(field).type
    while isinstance(leaf_type, pa.ExtensionType):
        leaf_type = leaf_type.storage_type
    while (
        pa.types.is_list(leaf_type)
        or pa.types.is_large_list(leaf_type)
        or pa.types.is_fixed_size_list(leaf_type)
    ):
        leaf_type = leaf_type.value_type
    array = np.asarray(values, dtype=leaf_type.to_pandas_dtype())
    if array.ndim == 1:
        array = array[:, None]
    return array


def load_staged_lerobot_episode(
    spec: SourceSpec,
    manifest: dict[str, Any],
    metadata_root: Path,
    staging_root: Path,
    episode_index: int,
) -> SourceEpisodeArrays:
    """Read one episode's native low-dimensional arrays from a staged shard."""
    staged_source_format = manifest.get("source_format", spec.source_format)
    if staged_source_format != "lerobot_v3":
        raise ValueError(f"Unsupported staged source format: {staged_source_format}")
    source_root = (staging_root / manifest["repo_id"].replace("/", "__")).resolve()
    episode_table = read_episode_table(metadata_root)
    all_rows = episode_table.to_pylist()
    row = next((item for item in all_rows if item["episode_index"] == episode_index), None)
    if row is None:
        raise KeyError(f"Episode {episode_index} is absent from source metadata")
    episode_manifest = next(
        (item for item in manifest["episodes"] if item["episode_index"] == episode_index), None
    )
    if episode_manifest is None:
        raise KeyError(f"Episode {episode_index} is absent from acquisition manifest")
    data_path = (source_root / episode_manifest["files"]["data"]).resolve()
    if source_root not in data_path.parents or not data_path.is_file():
        raise FileNotFoundError(f"Missing staged data shard: {data_path}")

    same_file_rows = [
        item
        for item in all_rows
        if item["data/chunk_index"] == row["data/chunk_index"]
        and item["data/file_index"] == row["data/file_index"]
    ]
    file_global_start = min(item["dataset_from_index"] for item in same_file_rows)
    offset = int(row["dataset_from_index"] - file_global_start)
    length = int(row["length"])
    fields = ["timestamp", "episode_index", *spec.state_fields]
    if spec.action_source == "native":
        fields.extend(spec.action_fields)
    shard = pq.read_table(data_path, columns=fields).slice(offset, length)
    episode_ids = np.asarray(shard["episode_index"].to_pylist(), dtype=np.int64)
    if len(episode_ids) != length or not np.all(episode_ids == episode_index):
        raise ValueError(f"Shard slice does not map exactly to source episode {episode_index}")
    timestamps = np.asarray(shard["timestamp"].to_pylist(), dtype=np.float64).reshape(-1)
    states = np.concatenate([_table_column_array(shard, field) for field in spec.state_fields], axis=1)
    actions = (
        states.copy()
        if spec.action_source == "copy_state"
        else np.concatenate([_table_column_array(shard, field) for field in spec.action_fields], axis=1)
    )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Episode {episode_index} contains NaN or infinity")
    _validated_timestamps(timestamps)
    return SourceEpisodeArrays(
        episode_index=episode_index,
        tasks=tuple(row.get("tasks") or ()),
        timestamps=timestamps,
        states=states,
        actions=actions,
    )


def nominate_lerobot_candidates(
    metadata_root: Path,
    *,
    count: int,
    min_duration_s: float = 8.0,
    prefer_shared_payload_files: bool = True,
) -> list[dict[str, Any]]:
    """Nominate metadata candidates; final selection requires visual review."""
    info = _read_json(metadata_root / "meta/info.json")
    table = read_episode_table(metadata_root)
    identity_columns = {"episode_index", "length", "tasks"}
    shard_columns = {
        name for name in table.column_names if name.endswith("/chunk_index") or name.endswith("/file_index")
    }
    columns = [name for name in table.column_names if name in identity_columns | shard_columns]
    rows = table.select(columns).to_pylist()
    fps = float(info["fps"])
    rows = [row for row in rows if float(row["length"]) / fps >= min_duration_s]
    if not rows:
        return []

    def semantic_key(row: dict[str, Any]) -> str:
        tasks = row.get("tasks") or []
        words = [word.strip(".,").lower() for word in (tasks[0].split() if tasks else [])]
        return " ".join(words[:3]) or "unlabeled"

    group_columns = sorted(shard_columns)
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in rows:
        # Prefer two episodes sharing every data/video shard, not merely the
        # low-dimensional Parquet file. This minimizes temporary media.
        key = tuple(int(row.get(name, 0)) for name in group_columns)
        groups.setdefault(key, []).append(row)
    ordered_groups = sorted(groups.values(), key=lambda group: (-len(group), group[0]["episode_index"]))
    pool = ordered_groups[0] if prefer_shared_payload_files else rows
    selected = []
    seen_semantics: set[str] = set()
    has_tasks = any(candidate.get("tasks") for candidate in pool)
    for row in sorted(pool, key=lambda value: (-value["length"], value["episode_index"])):
        key = semantic_key(row)
        if has_tasks and key in seen_semantics:
            continue
        selected.append(_candidate_record(row, fps, has_tasks))
        seen_semantics.add(key)
        if len(selected) == count:
            break
    if len(selected) < count:
        used = {item["episode_index"] for item in selected}
        for row in sorted(pool, key=lambda value: (-value["length"], value["episode_index"])):
            if row["episode_index"] not in used:
                selected.append(_candidate_record(row, fps, False))
            if len(selected) == count:
                break
    return selected


def _candidate_record(row: dict[str, Any], fps: float, semantic: bool) -> dict[str, Any]:
    return {
        **row,
        "duration_s": row["length"] / fps,
        "selection_status": "nominated_pending_proxy_review",
        "selection_rationale": (
            "duration boundary pass plus metadata semantic diversity"
            if semantic
            else "duration boundary pass; visual diversity requires proxy review"
        ),
    }


def _render_path(template: str, chunk: int, file_index: int, *, video_key: str | None = None) -> str:
    return template.format(chunk_index=chunk, file_index=file_index, video_key=video_key)


def resolve_lerobot_payload(
    spec: SourceSpec,
    audit: dict[str, Any],
    metadata_root: Path,
    episode_indices: list[int],
) -> dict[str, Any]:
    """Resolve exact payload paths; refuses unless source admission passed."""
    if not audit.get("admission", {}).get("passed", False):
        reasons = "; ".join(audit.get("admission", {}).get("failures", []))
        raise AdmissionError(f"{spec.name} is not admitted: {reasons}")
    if spec.source_format != "lerobot_v3":
        raise AdmissionError(f"No exact resolver for source format {spec.source_format!r}")
    info = _read_json(metadata_root / "meta/info.json")
    rows = {row["episode_index"]: row for row in read_episode_table(metadata_root).to_pylist()}
    cameras = [
        name for name, feature in info["features"].items() if feature.get("dtype") in {"image", "video"}
    ]
    files: set[str] = set()
    episodes = []
    for episode_index in episode_indices:
        row = rows[episode_index]
        data_path = _render_path(info["data_path"], int(row["data/chunk_index"]), int(row["data/file_index"]))
        files.add(data_path)
        episode_files: dict[str, Any] = {"data": data_path, "videos": {}}
        for camera in cameras:
            prefix = f"videos/{camera}"
            path = _render_path(
                info["video_path"],
                int(row[f"{prefix}/chunk_index"]),
                int(row[f"{prefix}/file_index"]),
                video_key=camera,
            )
            files.add(path)
            episode_files["videos"][camera] = {
                "path": path,
                "from_timestamp": float(row[f"{prefix}/from_timestamp"]),
                "to_timestamp": float(row[f"{prefix}/to_timestamp"]),
            }
        episodes.append({"episode_index": episode_index, "files": episode_files})
    return {
        "manifest_version": 1,
        "repo_id": spec.repo_id,
        "revision": audit["repository"]["resolved_revision"],
        "episodes": episodes,
        "files": sorted(files),
    }


def attach_payload_sizes(manifest: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    """Attach Hub-reported file sizes without fetching payload content."""
    info = HfApi(token=token).dataset_info(
        manifest["repo_id"], revision=manifest["revision"], files_metadata=True
    )
    sizes = {sibling.rfilename: sibling.size for sibling in info.siblings}
    manifest = dict(manifest)
    manifest["file_sizes_bytes"] = {path: sizes.get(path) for path in manifest["files"]}
    known = [size for size in manifest["file_sizes_bytes"].values() if size is not None]
    manifest["total_payload_bytes"] = sum(known) if len(known) == len(manifest["files"]) else None
    return manifest


def download_manifest_payload(
    manifest: dict[str, Any], staging_root: Path, *, token: str | None = None
) -> list[Path]:
    """Download exactly manifest-listed paths beneath controlled staging."""
    source_root = staging_root / manifest["repo_id"].replace("/", "__")
    source_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for relative in manifest["files"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.startswith(PAYLOAD_PREFIXES):
            raise ValueError(f"Unsafe or non-payload manifest path: {relative}")
        snapshot_download(
            repo_id=manifest["repo_id"],
            repo_type="dataset",
            revision=manifest["revision"],
            local_dir=source_root,
            allow_patterns=[relative],
            token=token,
        )
        downloaded.append(source_root / relative)
    return downloaded


def validate_staging_capacity(
    manifest: dict[str, Any], staging_root: Path, *, reserve_ratio: float = 1.2
) -> None:
    """Fail before acquisition when known payload bytes exceed free space."""
    required = manifest.get("total_payload_bytes")
    if required is None:
        raise ValueError("Manifest lacks complete payload size metadata")
    staging_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging_root).free
    if free < required * reserve_ratio:
        raise OSError(
            f"Insufficient staging space: need {required * reserve_ratio:.0f} bytes "
            f"including reserve, found {free}"
        )


def generate_review_proxies(manifest: dict[str, Any], staging_root: Path, output_root: Path) -> list[Path]:
    """Create low-resolution full-episode MP4s and contact sheets per source camera."""
    source_root = (staging_root / manifest["repo_id"].replace("/", "__")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for episode in manifest["episodes"]:
        episode_index = int(episode["episode_index"])
        durations = [
            float(video["to_timestamp"]) - float(video["from_timestamp"])
            for video in episode["files"]["videos"].values()
        ]
        annotation = {
            "source_episode_index": episode_index,
            "episode_duration_s": min(durations),
            "review_status": "pending",
            "segments": [],
            "required_segment_fields": [
                "start_s",
                "end_s",
                "retention",
                "retention_reason",
            ],
            "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
        }
        write_json(output_root / f"episode_{episode_index:06d}.annotations.json", annotation)
        for camera, video in episode["files"]["videos"].items():
            source = (source_root / video["path"]).resolve()
            if source_root not in source.parents or not source.is_file():
                raise FileNotFoundError(f"Missing staged source video: {source}")
            start = float(video["from_timestamp"])
            duration = float(video["to_timestamp"]) - start
            camera_slug = camera.replace("/", "_").replace(".", "_")
            stem = output_root / f"episode_{episode_index:06d}_{camera_slug}"
            proxy = stem.with_suffix(".mp4")
            contact = stem.with_suffix(".jpg")
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.9f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{duration:.9f}",
                    "-vf",
                    "scale=-2:360",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "28",
                    str(proxy),
                ],
                check=True,
            )
            frame_count = max(1, int(math.ceil(duration / 5.0)))
            columns = min(5, frame_count)
            rows = int(math.ceil(frame_count / columns))
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(proxy),
                    "-vf",
                    f"fps=1/5,scale=320:-1,tile={columns}x{rows}",
                    "-frames:v",
                    "1",
                    str(contact),
                ],
                check=True,
            )
            outputs.extend([proxy, contact])
    return outputs


def _validated_annotations(path: Path, episode_end_s: float) -> dict[str, Any]:
    annotations = _read_json(path)
    if annotations.get("review_status") != "validated":
        raise ValueError(f"Annotations must have review_status='validated': {path}")
    segments = sorted(annotations.get("segments", []), key=lambda item: float(item["start_s"]))
    if not segments:
        raise ValueError(f"No annotation segments in {path}")
    cursor = 0.0
    for segment in segments:
        start, end = float(segment["start_s"]), float(segment["end_s"])
        if start > cursor + 1e-6 or start < cursor - 1e-6 or end <= start:
            raise ValueError(f"Annotation segments must be contiguous and ordered in {path}")
        if segment.get("retention") not in {"keep", "reject"}:
            raise ValueError(f"Annotation retention must be 'keep' or 'reject' in {path}")
        if not str(segment.get("retention_reason", "")).strip():
            raise ValueError(f"Annotation retention_reason must be non-empty in {path}")
        if segment["retention"] == "keep":
            if not 1 <= int(segment["quality"]) <= 5:
                raise ValueError(f"Kept annotation quality must be in [1,5] in {path}")
            if not str(segment["subtask"]).strip():
                raise ValueError(f"Kept annotation subtask must be non-empty in {path}")
        for event in segment.get("mistake_events", []):
            if float(event["end_s"]) <= float(event["start_s"]):
                raise ValueError(f"Mistake event has non-positive duration in {path}")
        cursor = end
    if cursor < episode_end_s - 1e-3:
        raise ValueError(f"Annotation segments do not cover the episode through {episode_end_s:.6f}s")
    return annotations


def annotation_window_eligibility(
    annotations: dict[str, Any],
    window_start_s: float,
    window_end_s: float,
    *,
    action_start_s: float | None = None,
) -> dict[str, Any]:
    """Apply full-window quality rejection and future-only static rejection."""
    if window_end_s <= window_start_s:
        raise ValueError("Annotation window must have positive duration")
    overlapping = [
        segment
        for segment in annotations["segments"]
        if float(segment["start_s"]) <= window_end_s and float(segment["end_s"]) > window_start_s
    ]
    if not overlapping:
        raise ValueError(
            f"No annotation segment overlaps source window [{window_start_s:.6f}, {window_end_s:.6f}]"
        )
    rejected = []
    for segment in overlapping:
        if segment["retention"] == "keep":
            continue
        reason = str(segment["retention_reason"])
        if reason in {"static", "no_op", "stationary"} and action_start_s is not None:
            overlaps_future = (
                float(segment["start_s"]) <= window_end_s and float(segment["end_s"]) > action_start_s
            )
            if not overlaps_future:
                continue
        rejected.append(segment)
    return {
        "eligible": not rejected,
        "rejection_reasons": sorted({str(segment["retention_reason"]) for segment in rejected}),
    }


def annotation_at(annotations: dict[str, Any], timestamp_s: float) -> dict[str, Any]:
    for segment_index, segment in enumerate(annotations["segments"]):
        start, end = float(segment["start_s"]), float(segment["end_s"])
        if start <= timestamp_s < end or (
            segment_index == len(annotations["segments"]) - 1 and math.isclose(timestamp_s, end)
        ):
            events = [
                event
                for event in segment.get("mistake_events", [])
                if float(event["start_s"]) <= timestamp_s < float(event["end_s"])
            ]
            return {
                "subtask": str(segment["subtask"]),
                "quality": int(segment["quality"]),
                "mistake": bool(events),
                "mistake_events": events,
                "retention_reason": str(segment["retention_reason"]),
            }
    raise ValueError(f"No annotation segment covers source timestamp {timestamp_s:.6f}s")


def _camera_feature_schema(source_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema = {}
    labels = ("tm6", "tm5", "tm4", "tm3", "tm2", "tm1", "t")
    for camera, feature in source_info["features"].items():
        if feature.get("dtype") not in {"image", "video"}:
            continue
        if feature.get("info", {}).get("video.is_depth_map", False):
            raise NotImplementedError(
                f"Depth feature {camera} requires a lossless source-specific decoder; refusing RGB conversion"
            )
        for label in labels:
            schema[f"{camera}.{label}"] = {
                "dtype": "video",
                "shape": tuple(feature["shape"]),
                "names": feature.get("names"),
            }
    return schema


def _decode_sparse_camera_frames(
    source_info: dict[str, Any],
    episode: SourceEpisodeArrays,
    episode_manifest: dict[str, Any],
    samples: list[NumericalPackedSample],
    source_root: Path,
) -> dict[str, np.ndarray]:
    from lerobot.datasets.video_utils import decode_video_frames

    labels = ("tm6", "tm5", "tm4", "tm3", "tm2", "tm1", "t")
    decoded: dict[str, np.ndarray] = {}
    for camera, video in episode_manifest["files"]["videos"].items():
        path = (source_root / video["path"]).resolve()
        if source_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Missing staged camera shard: {path}")
        feature = source_info["features"][camera]
        fps = float(feature.get("info", {}).get("video.fps") or source_info["fps"])
        episode_video_start = float(video["from_timestamp"])
        query = [
            episode_video_start + float(timestamp - episode.timestamps[0])
            for sample in samples
            for timestamp in sample.observation_timestamps
        ]
        frames = decode_video_frames(
            path,
            query,
            tolerance_s=0.51 / fps,
            backend="pyav",
            return_uint8=True,
        )
        height, width, channels = feature["shape"]
        frames = frames.permute(0, 2, 3, 1).numpy().reshape(len(samples), 7, height, width, channels)
        for offset, label in enumerate(labels):
            decoded[f"{camera}.{label}"] = frames[:, offset]
    return decoded


def _packed_record(
    spec: SourceSpec,
    revision: str,
    episode_index: int,
    sample: NumericalPackedSample,
    annotation: dict[str, Any],
    robot_type: str | None = None,
) -> dict[str, Any]:
    return {
        "observation.state": sample.observation_values[-1],
        "observation.state_history": sample.observation_values[:-1],
        "action": sample.action.values,
        "source.episode_index": np.asarray([episode_index], dtype=np.int64),
        "source.anchor_timestamp": np.asarray([sample.anchor_timestamp], dtype=np.float64),
        "source.observation_timestamps": sample.observation_timestamps,
        "source.observation_frame_indices": sample.observation_frame_indices,
        "source.observation_timing_error": sample.observation_timing_error,
        "source.action_timestamps": sample.action.target_timestamps,
        "source.action_source_timestamps": sample.action.source_timestamps,
        "source.action_source_indices": sample.action.source_indices,
        "source.action_source_values": sample.action.source_values,
        "source.action_interpolated_mask": sample.action.interpolated_mask,
        "source.repo_id": spec.repo_id,
        "source.revision": revision,
        "source.robot_type": robot_type or spec.robot_type,
        "source.joint_names": json.dumps(spec.joint_names),
        "source.state_semantics": spec.state_semantics,
        "source.action_semantics": spec.action_semantics,
        "source.gripper_semantics": spec.gripper_semantics,
        "annotation.subtask": annotation["subtask"],
        "annotation.retention_reason": annotation["retention_reason"],
        "annotation.quality": np.asarray([annotation["quality"]], dtype=np.int64),
        "annotation.mistake": np.asarray([annotation["mistake"]], dtype=bool),
    }


def write_packed_v3_dataset(
    spec: SourceSpec,
    audit: dict[str, Any],
    manifest: dict[str, Any],
    metadata_root: Path,
    staging_root: Path,
    annotations_root: Path,
    dataset_root: Path,
    *,
    output_repo_id: str,
    stride_s: float,
    min_chunks_per_episode: int = 3,
) -> dict[str, Any]:
    """Write reviewed staged episodes as a packed LeRobot v3 dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not audit.get("admission", {}).get("passed", False):
        raise AdmissionError(f"Cannot extract unadmitted source {spec.name}")
    if min_chunks_per_episode < 1:
        raise ValueError("min_chunks_per_episode must be positive")
    if dataset_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset root: {dataset_root}")
    source_info = _read_json(metadata_root / "meta/info.json")
    physical_robot_type = str(source_info.get("robot_type") or spec.robot_type)
    episode_indices = [int(item["episode_index"]) for item in manifest["episodes"]]
    episodes = [
        load_staged_lerobot_episode(spec, manifest, metadata_root, staging_root, episode_index)
        for episode_index in episode_indices
    ]
    state_dimension = episodes[0].states.shape[1]
    action_dimension = episodes[0].actions.shape[1]
    if any(episode.states.shape[1] != state_dimension for episode in episodes):
        raise ValueError("State dimension changes across selected source episodes")
    if any(episode.actions.shape[1] != action_dimension for episode in episodes):
        raise ValueError("Action dimension changes across selected source episodes")

    state_dtype = np.dtype(episodes[0].states.dtype).name
    action_dtype = np.dtype(episodes[0].actions.dtype).name
    if state_dtype not in {"float32", "float64"} or action_dtype not in {"float32", "float64"}:
        raise TypeError(
            f"Packed output supports native float32/float64 arrays, got "
            f"state={state_dtype}, action={action_dtype}"
        )
    if any(np.dtype(episode.states.dtype).name != state_dtype for episode in episodes):
        raise ValueError("Physical state dtype changes across selected source episodes")
    if any(np.dtype(episode.actions.dtype).name != action_dtype for episode in episodes):
        raise ValueError("Physical action dtype changes across selected source episodes")

    features = packed_feature_schema(
        state_dimension, action_dimension, state_dtype=state_dtype, action_dtype=action_dtype
    )
    features.update(_camera_feature_schema(source_info))
    features.update(
        {
            key: {"dtype": "string", "shape": (1,), "names": None}
            for key in (
                "source.repo_id",
                "source.revision",
                "source.robot_type",
                "source.joint_names",
                "source.state_semantics",
                "source.action_semantics",
                "source.gripper_semantics",
                "annotation.subtask",
                "annotation.retention_reason",
            )
        }
    )
    features["annotation.quality"] = {"dtype": "int64", "shape": (1,), "names": None}
    features["annotation.mistake"] = {"dtype": "bool", "shape": (1,), "names": None}
    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=dataset_root,
        robot_type=physical_robot_type,
        fps=1,
        features=features,
        use_videos=True,
        image_writer_threads=4,
    )

    source_root = (staging_root / manifest["repo_id"].replace("/", "__")).resolve()
    extraction_episodes = []
    try:
        for episode, episode_manifest in zip(episodes, manifest["episodes"], strict=True):
            provenance_episode_index = int(
                episode_manifest.get("source_episode_index", episode.episode_index)
            )
            candidate_samples = pack_numerical_episode(
                episode.timestamps,
                episode.states,
                episode.timestamps,
                episode.actions,
                native_action_rate_hz=float(source_info["fps"]),
                stride_s=stride_s,
            )
            annotations_path = annotations_root / f"episode_{provenance_episode_index:06d}.annotations.json"
            annotations = _validated_annotations(
                annotations_path, float(episode.timestamps[-1] - episode.timestamps[0])
            )
            samples = []
            rejection_reasons: Counter[str] = Counter()
            for sample in candidate_samples:
                relative_anchor = sample.anchor_timestamp - episode.timestamps[0]
                decision = annotation_window_eligibility(
                    annotations,
                    relative_anchor + float(HISTORY_OFFSETS_SECONDS[0]),
                    relative_anchor + float(ACTION_OFFSETS_SECONDS[-1]),
                    action_start_s=relative_anchor,
                )
                if decision["eligible"]:
                    samples.append(sample)
                else:
                    rejection_reasons.update(decision["rejection_reasons"])
            if len(samples) < min_chunks_per_episode:
                raise ValueError(
                    f"Episode {episode.episode_index} has only {len(samples)} eligible chunks; "
                    f"requires at least {min_chunks_per_episode}. Review another episode or "
                    "explicitly lower --min-chunks-per-episode for a rare event."
                )
            camera_frames = _decode_sparse_camera_frames(
                source_info, episode, episode_manifest, samples, source_root
            )
            for sample_index, sample in enumerate(samples):
                relative_anchor = sample.anchor_timestamp - episode.timestamps[0]
                labels = annotation_at(annotations, relative_anchor)
                record = _packed_record(
                    spec,
                    manifest["revision"],
                    provenance_episode_index,
                    sample,
                    labels,
                    physical_robot_type,
                )
                for camera_key, frames in camera_frames.items():
                    record[camera_key] = frames[sample_index]
                # DROID Failure ships empty task strings, so the reviewed task
                # identity from proxy review is the only real label available.
                reviewed_task = str(annotations.get("task", "")).strip()
                record["task"] = (
                    episode.tasks[0]
                    if episode.tasks
                    else reviewed_task or f"source episode {provenance_episode_index}"
                )
                dataset.add_frame(record)
            dataset.save_episode()
            extraction_episodes.append(
                {
                    "source_episode_index": provenance_episode_index,
                    "converted_episode_index": episode.episode_index,
                    "source_frames": len(episode.timestamps),
                    "candidate_anchors": len(candidate_samples),
                    "packed_samples": len(samples),
                    "rejected_anchors": len(candidate_samples) - len(samples),
                    "rejection_reasons": dict(sorted(rejection_reasons.items())),
                    "first_anchor_s": None if not samples else samples[0].anchor_timestamp,
                    "last_anchor_s": None if not samples else samples[-1].anchor_timestamp,
                    "max_observation_timing_error_s": (
                        None
                        if not samples
                        else max(float(np.abs(sample.observation_timing_error).max()) for sample in samples)
                    ),
                }
            )
        dataset.finalize()
    except Exception:
        dataset.finalize()
        raise

    extension = {
        "name": PACKED_EXTENSION,
        "version": PACKED_EXTENSION_VERSION,
        "outer_fps_is_synthetic": True,
        "history_offsets_seconds": HISTORY_OFFSETS_SECONDS.tolist(),
        "action_offsets_seconds": ACTION_OFFSETS_SECONDS.tolist(),
        "anchor_stride_seconds": stride_s,
        "minimum_chunks_per_episode": min_chunks_per_episode,
        "source_spec": {
            "repo_id": spec.repo_id,
            "revision": manifest["revision"],
            "state_fields": list(spec.state_fields),
            "action_fields": list(spec.action_fields),
            "action_source": spec.action_source,
            "robot_type": physical_robot_type,
            "action_copied_from_state_fields": (
                list(spec.state_fields) if spec.action_source == "copy_state" else []
            ),
            "gripper_field": spec.gripper_field,
            "physical_state_dtype": state_dtype,
            "physical_action_dtype": action_dtype,
            "declared_state_dtypes": {
                field: source_info["features"][field].get("dtype") for field in spec.state_fields
            },
            "declared_action_dtypes": {
                field: source_info["features"][field].get("dtype")
                for field in (
                    spec.state_fields if spec.action_source == "copy_state" else spec.action_fields
                )
            },
        },
    }
    write_json(dataset_root / "meta/packed_extension.json", extension)
    write_json(dataset_root / "meta/source_audit.json", audit)
    write_json(dataset_root / "meta/acquisition_manifest.json", manifest)
    report = {
        "packed_extension": extension,
        "episodes": extraction_episodes,
        "total_packed_samples": sum(item["packed_samples"] for item in extraction_episodes),
    }
    write_json(dataset_root / "meta/extraction_report.json", report)
    annotations_output = dataset_root / "meta/annotations"
    annotations_output.mkdir(parents=True, exist_ok=True)
    for episode_manifest in manifest["episodes"]:
        episode_index = int(
            episode_manifest.get("source_episode_index", episode_manifest["episode_index"])
        )
        source = annotations_root / f"episode_{episode_index:06d}.annotations.json"
        shutil.copy2(source, annotations_output / source.name)
    return report


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_physical_action_provenance(dataset_root: Path, physical_action_dtype: str) -> float:
    columns = [
        "action",
        "source.action_timestamps",
        "source.action_source_timestamps",
        "source.action_source_values",
    ]
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No packed Parquet files beneath {dataset_root / 'data'}")
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in data_files])
    action = _table_column_array(table, "action")
    target_timestamps = _table_column_array(table, "source.action_timestamps")
    source_timestamps = _table_column_array(table, "source.action_source_timestamps")
    source_values = _table_column_array(table, "source.action_source_values")
    if np.dtype(action.dtype).name != physical_action_dtype:
        raise ValueError(
            f"Physical action Parquet dtype mismatch: expected {physical_action_dtype}, "
            f"got {np.dtype(action.dtype).name}"
        )
    if np.dtype(source_values.dtype).name != physical_action_dtype:
        raise ValueError(
            f"Physical action provenance dtype mismatch: expected {physical_action_dtype}, "
            f"got {np.dtype(source_values.dtype).name}"
        )
    denominator = source_timestamps[:, :, 1] - source_timestamps[:, :, 0]
    weight = np.divide(
        target_timestamps - source_timestamps[:, :, 0],
        denominator,
        out=np.zeros_like(target_timestamps, dtype=np.float64),
        where=denominator != 0,
    )
    expected = source_values[:, :, 0] + weight[:, :, None] * (source_values[:, :, 1] - source_values[:, :, 0])
    tolerance = 16 * np.finfo(np.dtype(physical_action_dtype)).eps
    np.testing.assert_allclose(action, expected, rtol=tolerance, atol=tolerance)
    return float(np.abs(action - expected).max(initial=0.0))


def _loader_interpolation_atol(
    action_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Bound action error induced by float32 timestamps in the standard loader.

    Physical Parquet provenance is validated separately at its stored dtype.
    Reconstructing from the loader view can shift an interpolation weight by
    one timestamp ULP, so scale that quantization by the interval action slope.
    """
    timestamp_spacing = abs(float(np.spacing(np.asarray(action_timestamps).max())))
    denominator = source_timestamps[:, 1] - source_timestamps[:, 0]
    slope = np.divide(
        np.abs(source_values[:, 1] - source_values[:, 0]),
        np.abs(denominator)[:, None],
        out=np.zeros_like(source_values[:, 0], dtype=np.float64),
        where=denominator[:, None] != 0,
    )
    return 1e-5 + timestamp_spacing * slope


def validate_packed_v3_dataset(dataset_root: Path, repo_id: str) -> dict[str, Any]:
    """Decode every pilot row and verify packed temporal/interpolation integrity."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    extension = _read_json(dataset_root / "meta/packed_extension.json")
    if extension.get("name") != PACKED_EXTENSION or extension.get("version") != PACKED_EXTENSION_VERSION:
        raise ValueError("Packed extension identity/version mismatch")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, video_backend="pyav")
    source_spec = extension["source_spec"]
    physical_state_dtype = source_spec["physical_state_dtype"]
    physical_action_dtype = source_spec["physical_action_dtype"]
    expected_physical_dtypes = {
        "observation.state": physical_state_dtype,
        "observation.state_history": physical_state_dtype,
        "action": physical_action_dtype,
        "source.action_source_values": physical_action_dtype,
    }
    for key, expected_dtype in expected_physical_dtypes.items():
        actual_dtype = dataset.meta.features[key]["dtype"]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"Physical dtype mismatch for {key}: expected {expected_dtype}, got {actual_dtype}"
            )
    physical_action_reconstruction_error = _validate_physical_action_provenance(
        dataset_root, physical_action_dtype
    )
    image_keys = [
        key for key, feature in dataset.meta.features.items() if feature.get("dtype") in {"image", "video"}
    ]
    state_dimension = action_dimension = None
    max_observation_error = 0.0
    interpolation_points = 0
    for index in range(len(dataset)):
        item = dataset[index]
        state = _as_numpy(item["observation.state"])
        history = _as_numpy(item["observation.state_history"])
        action = _as_numpy(item["action"])
        action_timestamps = _as_numpy(item["source.action_timestamps"])
        source_timestamps = _as_numpy(item["source.action_source_timestamps"])
        source_values = _as_numpy(item["source.action_source_values"])
        interpolated = _as_numpy(item["source.action_interpolated_mask"]).astype(bool)
        observation_timestamps = _as_numpy(item["source.observation_timestamps"])
        observation_error = _as_numpy(item["source.observation_timing_error"])
        if history.shape != (6, state.shape[-1]) or action.shape[0] != 30:
            raise ValueError(f"Packed matrix shape mismatch at row {index}")
        if source_timestamps.shape != (30, 2) or source_values.shape != (30, 2, action.shape[1]):
            raise ValueError(f"Action provenance shape mismatch at row {index}")
        # Parquet stores these as float64. The standard LeRobot loader returns
        # floating features as float32 tensors, so validate at loader
        # serialization tolerance (the physical Parquet values remain exact).
        timestamp_spacing = abs(float(np.spacing(np.asarray(action_timestamps).max())))
        cadence_atol = max(1e-9, 2 * timestamp_spacing)
        np.testing.assert_allclose(np.diff(action_timestamps), 1 / 30, rtol=0, atol=cadence_atol)
        if not np.all(np.diff(observation_timestamps) > 0):
            raise ValueError(f"Observation history is not chronological at row {index}")
        denominator = source_timestamps[:, 1] - source_timestamps[:, 0]
        weight = np.divide(
            action_timestamps - source_timestamps[:, 0],
            denominator,
            out=np.zeros(30, dtype=np.float64),
            where=denominator != 0,
        )
        expected = source_values[:, 0] + weight[:, None] * (source_values[:, 1] - source_values[:, 0])
        loader_action_atol = _loader_interpolation_atol(
            action_timestamps, source_timestamps, source_values
        )
        loader_action_error = np.abs(action - expected)
        loader_action_tolerance = loader_action_atol + 1e-5 * np.abs(expected)
        if np.any(loader_action_error > loader_action_tolerance):
            violation = loader_action_error - loader_action_tolerance
            point, dimension = np.unravel_index(np.argmax(violation), violation.shape)
            raise AssertionError(
                "Loader action reconstruction mismatch at "
                f"row {index}, point {point}, dimension {dimension}: "
                f"error={loader_action_error[point, dimension]}, "
                f"tolerance={loader_action_tolerance[point, dimension]}"
            )
        if not np.array_equal(interpolated, denominator != 0):
            raise ValueError(f"Interpolation mask/provenance mismatch at row {index}")
        if not np.isfinite(state).all() or not np.isfinite(history).all() or not np.isfinite(action).all():
            raise ValueError(f"Non-finite state/action at row {index}")
        for key in image_keys:
            image = _as_numpy(item[key])
            if image.ndim != 3 or not np.isfinite(image).all():
                raise ValueError(f"Camera feature failed decode at row {index}: {key}")
        state_dimension = state.shape[-1]
        action_dimension = action.shape[-1]
        max_observation_error = max(max_observation_error, float(np.abs(observation_error).max()))
        interpolation_points += int(interpolated.sum())
    report = {
        "passed": True,
        "rows": len(dataset),
        "state_dimension": state_dimension,
        "action_dimension": action_dimension,
        "physical_state_dtype": physical_state_dtype,
        "physical_action_dtype": physical_action_dtype,
        "camera_streams": image_keys,
        "interpolated_action_points": interpolation_points,
        "max_physical_action_reconstruction_error": physical_action_reconstruction_error,
        "max_observation_timing_error_s": max_observation_error,
    }
    write_json(dataset_root / "meta/validation_report.json", report)
    return report


def safe_cleanup_payload(manifest: dict[str, Any], staging_root: Path) -> list[Path]:
    """Delete only resolved manifest files within controlled staging."""
    source_root = (staging_root / manifest["repo_id"].replace("/", "__")).resolve()
    removed = []
    for relative in manifest["files"]:
        target = (source_root / relative).resolve()
        if source_root not in target.parents:
            raise ValueError(f"Refusing cleanup outside controlled staging: {target}")
        if target.is_file():
            target.unlink()
            removed.append(target)
    return removed


def nearest_observations(
    timestamps: np.ndarray, values: np.ndarray, anchor_timestamp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = _validated_timestamps(timestamps)
    values = np.asarray(values)
    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("Observation timestamps and values have different lengths")
    targets = anchor_timestamp + HISTORY_OFFSETS_SECONDS
    right = np.clip(np.searchsorted(timestamps, targets, side="left"), 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets)
    indices = np.where(choose_left, left, right)
    actual = timestamps[indices]
    return values[indices], actual, indices.astype(np.int64), actual - targets


def sample_action_chunk(
    timestamps: np.ndarray,
    values: np.ndarray,
    anchor_timestamp: float,
    *,
    native_rate_hz: float,
    coincidence_tolerance_s: float = 1e-6,
) -> ActionChunk:
    """Create a 30 Hz chunk while preserving every action component."""
    timestamps = _validated_timestamps(timestamps)
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != timestamps.shape[0]:
        raise ValueError("Actions must have shape [time, action_dimension]")
    targets = anchor_timestamp + ACTION_OFFSETS_SECONDS
    if targets[0] < timestamps[0] or targets[-1] > timestamps[-1]:
        raise ValueError("Action chunk crosses an episode boundary")
    right = np.clip(np.searchsorted(timestamps, targets, side="left"), 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    exact = np.isclose(timestamps[right], targets, atol=coincidence_tolerance_s, rtol=0)
    left = np.where(exact, right, left)

    if math.isclose(native_rate_hz, 30.0, rel_tol=0, abs_tol=1e-6):
        choose_right = np.abs(timestamps[right] - targets) < np.abs(timestamps[left] - targets)
        selected = np.where(choose_right, right, left)
        output = values[selected].copy()
        source_indices = np.stack([selected, selected], axis=1).astype(np.int64)
        source_times = np.stack([timestamps[selected], timestamps[selected]], axis=1)
        source_values = np.stack([values[selected], values[selected]], axis=1)
        interpolated = np.zeros(30, dtype=bool)
    else:
        t0, t1 = timestamps[left], timestamps[right]
        denominator = t1 - t0
        weight = np.divide(targets - t0, denominator, out=np.zeros_like(targets), where=denominator != 0)
        output = values[left] + weight[:, None] * (values[right] - values[left])
        source_indices = np.stack([left, right], axis=1).astype(np.int64)
        source_times = np.stack([t0, t1], axis=1)
        source_values = np.stack([values[left], values[right]], axis=1)
        interpolated = left != right
    return ActionChunk(
        values=output.astype(values.dtype, copy=False),
        target_timestamps=targets,
        source_timestamps=source_times,
        source_indices=source_indices,
        source_values=source_values,
        interpolated_mask=interpolated,
    )


def _validated_timestamps(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("Timestamps must be one-dimensional with at least two samples")
    if not np.isfinite(timestamps).all():
        raise ValueError("Timestamps contain NaN or infinity")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must increase; duplicates and clock reversals are rejected")
    return timestamps


def anchor_timestamps(timestamps: np.ndarray, stride_s: float) -> np.ndarray:
    timestamps = _validated_timestamps(timestamps)
    if stride_s <= 0:
        raise ValueError("Anchor stride must be positive")
    first = timestamps[0] + 6.0
    last = timestamps[-1] - ACTION_OFFSETS_SECONDS[-1]
    if first > last:
        return np.empty(0, dtype=np.float64)
    count = int(math.floor((last - first) / stride_s)) + 1
    return first + np.arange(count, dtype=np.float64) * stride_s


def pack_numerical_episode(
    observation_timestamps: np.ndarray,
    observation_values: np.ndarray,
    action_timestamps: np.ndarray,
    action_values: np.ndarray,
    *,
    native_action_rate_hz: float,
    stride_s: float,
) -> list[NumericalPackedSample]:
    samples = []
    for anchor in anchor_timestamps(action_timestamps, stride_s):
        observations, actual, indices, error = nearest_observations(
            observation_timestamps, observation_values, float(anchor)
        )
        action = sample_action_chunk(
            action_timestamps, action_values, float(anchor), native_rate_hz=native_action_rate_hz
        )
        samples.append(
            NumericalPackedSample(
                anchor_timestamp=float(anchor),
                observation_values=observations,
                observation_timestamps=actual,
                observation_frame_indices=indices,
                observation_timing_error=error,
                action=action,
            )
        )
    return samples


def packed_feature_schema(
    state_dimension: int,
    action_dimension: int,
    *,
    state_dtype: str = "float32",
    action_dtype: str = "float32",
) -> dict[str, dict[str, Any]]:
    """Matrix-valued feature schema accepted by LeRobot v3."""
    return {
        "observation.state": {"dtype": state_dtype, "shape": (state_dimension,), "names": None},
        "observation.state_history": {"dtype": state_dtype, "shape": (6, state_dimension), "names": None},
        "action": {"dtype": action_dtype, "shape": (30, action_dimension), "names": None},
        "source.episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "source.anchor_timestamp": {"dtype": "float64", "shape": (1,), "names": None},
        "source.observation_timestamps": {"dtype": "float64", "shape": (7,), "names": None},
        "source.observation_frame_indices": {"dtype": "int64", "shape": (7,), "names": None},
        "source.observation_timing_error": {"dtype": "float64", "shape": (7,), "names": None},
        "source.action_timestamps": {"dtype": "float64", "shape": (30,), "names": None},
        "source.action_source_timestamps": {"dtype": "float64", "shape": (30, 2), "names": None},
        "source.action_source_indices": {"dtype": "int64", "shape": (30, 2), "names": None},
        "source.action_source_values": {
            "dtype": action_dtype,
            "shape": (30, 2, action_dimension),
            "names": None,
        },
        "source.action_interpolated_mask": {"dtype": "bool", "shape": (30,), "names": None},
    }


def loader_view(item: dict[str, Any]) -> dict[str, Any]:
    """Return seven state observations and one [30,D] native action chunk."""
    history = np.asarray(item["observation.state_history"])
    current = np.asarray(item["observation.state"])
    action = np.asarray(item["action"])
    states = np.concatenate([history, current[None]], axis=0)
    if states.shape[0] != 7 or action.shape[0] != 30:
        raise ValueError(f"Invalid packed shapes: states={states.shape}, action={action.shape}")
    return {"observation.state": states, "action": action, "raw": item}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
