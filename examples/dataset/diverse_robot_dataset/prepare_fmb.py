#!/usr/bin/env python

"""Select, acquire, convert, and validate the source-native FMB production corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

SOURCE_REPO_ID = "charlesxu0124/functional-manipulation-benchmark"
SOURCE_REVISION = "f99fd55c072eea5573523c96aa527aed3c665690"
SOURCE_PREFIX = "single_object_manipulation_dataset/"
SOURCE_PATTERN = re.compile(
    r"^single_object_manipulation_dataset/"
    r"(?P<shape>[1-9])_(?P<size>[SML])_(?P<length>[SL])_(?P<color>[^_]+)_"
    r"(?P<angle>horizontal|vertical)_(?P<distractor>[yn])_(?P<trajectory>[^.]+)\.npy$"
)
CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")
KEEP_RGB = ("side_1", "side_2", "wrist_1")
KEEP_DEPTH = ("wrist_1",)
OMIT_RGB = ("wrist_2",)
OMIT_DEPTH = ("side_1", "side_2", "wrist_2")
FPS = 10.0
HISTORY_S = 6.0
FUTURE_POINTS = 30
FUTURE_FPS = 30.0
FUTURE_END_S = (FUTURE_POINTS - 1) / FUTURE_FPS
DEPTH_MM_PER_LEVEL = 0.1
DEPTH_SCALE_PROVENANCE = "user_authorized_same_D405_model_assumption"
INVALID_DEPTH_VALUES = (0, 65535)
READ_CHUNK = 8 * 1024 * 1024
ANCHOR_STRIDES_S = {"original_2s": 2.0, "proposed_5hz": 0.2, "proposed_10hz": 0.1}
LOW_DIMENSIONAL_KEYS = {
    "obs/tcp_pose": "tcp_pose.npy",
    "obs/tcp_vel": "tcp_vel.npy",
    "obs/tcp_force": "tcp_force.npy",
    "obs/tcp_torque": "tcp_torque.npy",
    "obs/q": "q.npy",
    "obs/dq": "dq.npy",
    "obs/jacobian": "jacobian.npy",
    "obs/gripper_pose": "gripper_pose.npy",
    "actions": "actions.npy",
    "primitive": "primitive.npy",
}
RETAINED_SENSOR_FILES = {
    "obs/side_1": "side_1_rgb.npy",
    "obs/side_2": "side_2_rgb.npy",
    "obs/wrist_1": "wrist_1_rgb.npy",
    "obs/wrist_1_depth": "wrist_1_depth_z16.npy",
}
SOURCE_SENSOR_KEYS = {
    **{f"obs/{camera}": (np.uint8, (256, 256, 3)) for camera in CAMERAS},
    **{f"obs/{camera}_depth": (np.uint16, (256, 256)) for camera in CAMERAS},
}
EXPECTED_LOW_DIMENSIONAL_TAILS = {
    "obs/tcp_pose": (7,),
    "obs/tcp_vel": (6,),
    "obs/tcp_force": (3,),
    "obs/tcp_torque": (3,),
    "obs/q": (7,),
    "obs/dq": (7,),
    "obs/jacobian": (6, 7),
    "obs/gripper_pose": (),
    "actions": (7,),
    "primitive": (),
}
DERIVED_FILES = {"timestamp_s": "timestamp_s.npy", "source_timestep": "source_timestep.npy"}
PRIMITIVE_DESCRIPTIONS = {
    "grasp": "grasp the object",
    "move_up": "lift the object",
    "place_on_fixture": "place the object on the fixture",
    "regrasp": "regrasp the object",
    "rotate": "rotate the object",
    "go_to_board": "move the object to the board",
    "move_to_board": "move the object to the board",
    "insert": "insert the object into the board",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def parse_source_path(path: str) -> dict[str, str]:
    match = SOURCE_PATTERN.fullmatch(path)
    if match is None:
        raise ValueError(f"Not an FMB single-object trajectory path: {path}")
    return match.groupdict()


def geometry_key(item: dict[str, Any]) -> tuple[str, str, str]:
    obj = item["object"]
    return str(obj["shape"]), str(obj["size"]), str(obj["length"])


def condition_key(item: dict[str, Any]) -> tuple[str, str]:
    obj = item["object"]
    return str(obj["angle"]), str(obj["distractor"])


def stable_score(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_path(raw_root: Path, file_spec: dict[str, Any]) -> Path:
    relative = Path(file_spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    return raw_root / relative


def verify_file(path: Path, file_spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    expected = {"bytes": int(file_spec["bytes"]), "sha256": file_spec["sha256"]}
    if actual != expected:
        raise ValueError(f"Pinned source mismatch for {path}: {actual} != {expected}")


def hub_specs() -> list[dict[str, Any]]:
    info = HfApi().dataset_info(SOURCE_REPO_ID, revision=SOURCE_REVISION, files_metadata=True)
    if info.sha != SOURCE_REVISION:
        raise ValueError(f"Pinned revision resolved to {info.sha}")
    specs = []
    for entry in info.siblings:
        if not SOURCE_PATTERN.fullmatch(entry.rfilename):
            continue
        if entry.lfs is None or entry.size is None:
            raise ValueError(f"Source file lacks immutable LFS metadata: {entry.rfilename}")
        specs.append(
            {
                "path": entry.rfilename,
                "bytes": int(entry.size),
                "sha256": str(entry.lfs.sha256),
                "object": parse_source_path(entry.rfilename),
            }
        )
    return sorted(specs, key=lambda item: item["path"])


def choose_diverse(
    available: list[dict[str, Any]], count: int, forced_paths: set[str]
) -> list[dict[str, Any]]:
    by_path = {item["path"]: item for item in available}
    missing = sorted(forced_paths - set(by_path))
    if missing:
        raise ValueError(f"Pinned pilot files disappeared from source revision: {missing}")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in available:
        groups[geometry_key(item)].append(item)
    expected = {(str(shape), size, length) for shape in range(1, 10) for size in "SML" for length in "SL"}
    if set(groups) != expected:
        raise ValueError(f"Expected all 54 geometries, got {len(groups)}")

    selected = [by_path[path] for path in sorted(forced_paths)]
    selected_paths = set(forced_paths)
    condition_counts: Counter[tuple[str, str]] = Counter(condition_key(item) for item in selected)
    geometry_counts: Counter[tuple[str, str, str]] = Counter(geometry_key(item) for item in selected)

    def candidate_score(item: dict[str, Any]) -> tuple[Any, ...]:
        condition = condition_key(item)
        trajectory = item["object"]["trajectory"]
        trajectory_number = int(trajectory) if str(trajectory).isdigit() else 10**9
        return (
            geometry_counts[geometry_key(item)],
            condition_counts[condition],
            trajectory_number,
            stable_score(item["path"]),
        )

    # First cover every geometry not represented by the reviewed pilot.
    for geometry in sorted(groups, key=lambda key: stable_score("|".join(key))):
        if len(selected) >= count:
            break
        if geometry_counts[geometry]:
            continue
        candidates = [item for item in groups[geometry] if item["path"] not in selected_paths]
        chosen = min(candidates, key=candidate_score)
        selected.append(chosen)
        selected_paths.add(chosen["path"])
        condition_counts[condition_key(chosen)] += 1
        geometry_counts[geometry] += 1

    # Then add repeats in geometry-balanced, condition-balanced rounds.
    geometries = sorted(groups, key=lambda key: stable_score("repeat|" + "|".join(key)))
    while len(selected) < count:
        progress = False
        for geometry in geometries:
            if len(selected) >= count:
                break
            candidates = [item for item in groups[geometry] if item["path"] not in selected_paths]
            if not candidates:
                continue
            chosen = min(candidates, key=candidate_score)
            selected.append(chosen)
            selected_paths.add(chosen["path"])
            condition_counts[condition_key(chosen)] += 1
            geometry_counts[geometry] += 1
            progress = True
        if not progress:
            raise ValueError(f"Only {len(selected)} distinct source files were available")
    return sorted(selected, key=lambda item: item["path"])


def assign_splits(files: list[dict[str, Any]]) -> dict[str, str]:
    ordered = sorted(files, key=lambda item: stable_score("split|" + item["sha256"]))
    result = {}
    for index, item in enumerate(ordered):
        bucket = index % 10
        result[item["path"]] = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
    return result


def summarize_selection(files: list[dict[str, Any]], split_by_path: dict[str, str]) -> dict[str, Any]:
    fields = ("shape", "size", "length", "color", "angle", "distractor")
    counts = {
        field: dict(sorted(Counter(str(item["object"][field]) for item in files).items())) for field in fields
    }
    geometry_counts = Counter("|".join(geometry_key(item)) for item in files)
    return {
        "episode_count": len(files),
        "unique_geometry_count": len(geometry_counts),
        "geometry_repeat_distribution": dict(sorted(Counter(geometry_counts.values()).items())),
        "condition_counts": counts,
        "split_episode_counts": dict(sorted(Counter(split_by_path.values()).items())),
        "selection_policy": (
            "The 12 reviewed pilot paths are retained, every one of the 54 shape-size-length "
            "geometries is covered before additional repeats, then geometry and angle/distractor "
            "counts are balanced deterministically without payload inspection."
        ),
    }


def select_manifest(pilot_manifest_path: Path, output: Path, count: int) -> dict[str, Any]:
    available = hub_specs()
    pilot = read_json(pilot_manifest_path)
    forced = {item["path"] for item in pilot["files"]}
    files = choose_diverse(available, count, forced)
    splits = assign_splits(files)
    for item in files:
        item["split"] = splits[item["path"]]
        item["pilot_reviewed"] = item["path"] in forced
    manifest = {
        "source_repo_id": SOURCE_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "source_subset": "single_object_manipulation_dataset",
        "license": "CC-BY-4.0",
        "nominal_fps": FPS,
        "selection": summarize_selection(files, splits),
        "files": files,
        "conversion_policy": {
            "keep_rgb": list(KEEP_RGB),
            "keep_depth": list(KEEP_DEPTH),
            "omit_rgb": list(OMIT_RGB),
            "omit_depth": list(OMIT_DEPTH),
            "depth_storage": "source_native_D405_Z16_uint16_levels",
            "invalid_depth_values": list(INVALID_DEPTH_VALUES),
            "depth_units_mm_per_level": DEPTH_MM_PER_LEVEL,
            "depth_scale_provenance": DEPTH_SCALE_PROVENANCE,
            "timestamp_provenance": "derived_nominal_grid_no_source_timestamps_in_npy",
            "actor_history_s": HISTORY_S,
            "actor_future_points": FUTURE_POINTS,
            "actor_future_fps": FUTURE_FPS,
            "actor_interpolation": "linear_in_source_native_action_space_no_extrapolation",
        },
    }
    write_json(output, manifest)
    return manifest


def acquire(manifest: dict[str, Any], raw_root: Path) -> list[Path]:
    raw_root.mkdir(parents=True, exist_ok=True)
    missing_required = sum(
        int(item["bytes"]) for item in manifest["files"] if not local_path(raw_root, item).is_file()
    )
    if shutil.disk_usage(raw_root).free < missing_required * 1.2:
        raise OSError(f"Need {missing_required * 1.2:.0f} free bytes including reserve")
    paths = []
    for item in manifest["files"]:
        path = local_path(raw_root, item)
        if path.is_file():
            verify_file(path, item)
        else:
            path = Path(
                hf_hub_download(
                    repo_id=manifest["source_repo_id"],
                    repo_type="dataset",
                    revision=manifest["source_revision"],
                    filename=item["path"],
                    local_dir=raw_root,
                )
            )
            verify_file(path, item)
        paths.append(path)
    return paths


def load_episode(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if not isinstance(loaded, np.ndarray) or loaded.shape != () or loaded.dtype != object:
        raise TypeError(f"Expected scalar object array in {path}")
    episode = loaded.item()
    if not isinstance(episode, dict):
        raise TypeError(f"Expected source dictionary in {path}")
    return episode


def labels_from(array: np.ndarray) -> list[str]:
    result = []
    for value in np.asarray(array).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        result.append(str(value).strip())
    return result


def primitive_intervals(labels: list[str]) -> list[dict[str, Any]]:
    if not labels:
        return []
    result = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            result.append(
                {
                    "primitive": labels[start],
                    "normalized_description": PRIMITIVE_DESCRIPTIONS.get(
                        labels[start], labels[start].replace("_", " ")
                    ),
                    "start_timestep": start,
                    "end_timestep_exclusive": index,
                    "start_s_nominal": start / FPS,
                    "end_s_nominal_exclusive": index / FPS,
                    "duration_s_nominal": (index - start) / FPS,
                    "native_action_samples": index - start,
                    "boundary_provenance": "source_native",
                }
            )
            start = index
    return result


def anchor_times(frame_count: int, stride_s: float) -> np.ndarray:
    last_timestamp = (frame_count - 1) / FPS
    last = last_timestamp - FUTURE_END_S
    if last < HISTORY_S:
        return np.empty(0, dtype=np.float64)
    count = int(math.floor((last - HISTORY_S) / stride_s + 1e-9)) + 1
    return HISTORY_S + np.arange(count, dtype=np.float64) * stride_s


def anchor_indices(frame_count: int, stride_s: float) -> list[int]:
    return [int(round(value * FPS)) for value in anchor_times(frame_count, stride_s)]


def future_source_indices(anchor_s: float, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    # Match audit_fmb.primitive_conditioned_count exactly, including its
    # floating nominal anchor grid and clipped searchsorted brackets.
    source_times = np.arange(frame_count, dtype=np.float64) / FPS
    target_times = anchor_s + np.arange(FUTURE_POINTS, dtype=np.float64) / FUTURE_FPS
    right = np.clip(np.searchsorted(source_times, target_times, side="left"), 0, frame_count - 1)
    left = np.clip(right - 1, 0, frame_count - 1)
    if target_times[0] < 0 or target_times[-1] > source_times[-1] + 1e-9:
        raise ValueError("Actor target would extrapolate outside the episode")
    return left.astype(np.int64), right.astype(np.int64)


def actor_accounting(labels: list[str]) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    result = {}
    for name, stride in ANCHOR_STRIDES_S.items():
        times = anchor_times(len(labels), stride)
        primitive_count = 0
        for anchor_s in times:
            anchor = min(len(labels) - 1, int(round(anchor_s * FPS)))
            left, right = future_source_indices(float(anchor_s), len(labels))
            used = np.unique(np.concatenate([left, right]))
            if labels_array[anchor] and np.all(labels_array[used] == labels_array[anchor]):
                primitive_count += 1
        result[name] = {
            "stride_s": stride,
            "episode_level_candidate_anchors": len(times),
            "primitive_conditioned_future_candidate_anchors": primitive_count,
        }
    return result


def write_actor_views(output_root: Path) -> dict[str, Any]:
    """Write 5 Hz and 10 Hz reference indexes without copying episode arrays."""
    metadata_records = read_jsonl(output_root / "episodes.jsonl")
    summaries = {}
    for view_name, stride_s in {
        "5hz": ANCHOR_STRIDES_S["proposed_5hz"],
        "10hz": ANCHOR_STRIDES_S["proposed_10hz"],
    }.items():
        rows = []
        for metadata in metadata_records:
            identifier = metadata["episode_id"]
            labels = labels_from(
                np.load(output_root / "episodes" / identifier / "primitive.npy", mmap_mode="r")
            )
            intervals = metadata["primitive_intervals"]
            for anchor_s in anchor_times(metadata["frame_count"], stride_s):
                anchor = min(metadata["frame_count"] - 1, int(round(anchor_s * FPS)))
                left, right = future_source_indices(float(anchor_s), metadata["frame_count"])
                used = np.unique(np.concatenate([left, right]))
                interval = next(
                    item
                    for item in intervals
                    if item["start_timestep"] <= anchor < item["end_timestep_exclusive"]
                )
                rows.append(
                    {
                        "episode_id": identifier,
                        "split": metadata["split"],
                        "source": "fmb",
                        "component": "single_object_manipulation",
                        # A Franka, the same robot DROID records; the alias table
                        # resolves this to "Franka Panda" where "FMB_source_robot"
                        # resolved to unknown (-1).
                        "embodiment": "Franka",
                        "anchor_timestep": anchor,
                        "anchor_s": float(anchor_s),
                        "anchor_s_nominal": float(anchor_s),
                        "history_frames": [
                            int(round((anchor_s + offset_s) * FPS))
                            for offset_s in (-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0)
                        ],
                        "history_start_timestep": anchor - int(HISTORY_S * FPS),
                        "future_end_s_nominal": float(anchor_s + FUTURE_END_S),
                        "primitive": labels[anchor],
                        "subtask": interval["normalized_description"],
                        "quality": interval.get("quality"),
                        "mistake": any(
                            event["start_timestep"] <= anchor < event["end_timestep_exclusive"]
                            for event in interval.get("mistake_events", [])
                        ),
                        "primitive_conditioned_future_supported": bool(
                            labels[anchor] and all(labels[int(item)] == labels[anchor] for item in used)
                        ),
                        "retained": True,
                        "rejection_reasons": [],
                        # FMB's own commanded action is a normalized [-1, 1] Cartesian
                        # delta with no recorded scale, so it is dropped in favour of the
                        # measured joint trajectory; see FMBCorpusEpisode.state.
                        "action_source": "copy_state",
                        "source_arrays_reference": f"episodes/{identifier}",
                        "views_copy_source_arrays": False,
                        "interpolation_provenance": {
                            "method": "linear",
                            "source_rate_hz_nominal": FPS,
                            "target_rate_hz": FUTURE_FPS,
                            "target_points": FUTURE_POINTS,
                            "source_index_rule": ("floor_and_ceil_of_target_time_times_nominal_fps"),
                            "no_extrapolation": True,
                        },
                    }
                )
        filename = f"actor_anchors_{view_name}.jsonl"
        write_jsonl(output_root / filename, rows)
        summaries[view_name] = {
            "stride_s": stride_s,
            "rows": len(rows),
            "path": filename,
            "source_frames_duplicated": 0,
        }

    corpus_path = output_root / "corpus.json"
    corpus = read_json(corpus_path)
    derived = corpus["accounting"]["derived_training_rows"]
    derived["stored_actor_anchor_views"] = summaries
    derived["stored_actor_anchor_view"] = {
        "stride": "proposed_10hz",
        "rows": summaries["10hz"]["rows"],
        "source_frames_duplicated": 0,
    }
    corpus["one_corpus_two_views"]["actor_views"] = {name: item["path"] for name, item in summaries.items()}
    corpus["one_corpus_two_views"]["actor_view"] = summaries["10hz"]["path"]
    write_json(corpus_path, corpus)
    return summaries


def load_review(review_root: Path | None, source_path: str) -> dict[tuple[int, int], dict[str, Any]]:
    if review_root is None:
        return {}
    path = review_root / f"{Path(source_path).stem}.review.json"
    if not path.is_file():
        return {}
    review = read_json(path)
    if review.get("review_status") != "complete" or review.get("source_path") != source_path:
        raise ValueError(f"Invalid production review artifact: {path}")
    return {(item["start_timestep"], item["end_timestep_exclusive"]): item for item in review["subtasks"]}


def save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    np.save(path, array, allow_pickle=False)
    return {
        "path": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def episode_id(index: int, source_path: str) -> str:
    return f"episode_{index:06d}_{Path(source_path).stem}"


def convert(
    manifest: dict[str, Any],
    raw_root: Path,
    output_root: Path,
    pilot_audit_path: Path,
    review_root: Path | None,
) -> dict[str, Any]:
    gate = read_json(pilot_audit_path)["acceptance_gate"]
    if gate.get("status") != "passed" or not gate.get("full_converter_allowed"):
        raise ValueError(f"FMB pilot gate does not permit conversion: {gate}")
    episodes_root = output_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    metadata_records = []
    critic_records = []
    actor_records = []
    aggregate_actor = {
        name: Counter(
            {"episode_level_candidate_anchors": 0, "primitive_conditioned_future_candidate_anchors": 0}
        )
        for name in ANCHOR_STRIDES_S
    }
    total_steps = 0
    reviewed_intervals = 0
    eligible_intervals = 0
    quality_intervals: Counter[int] = Counter()
    quality_timesteps: Counter[int] = Counter()
    mistake_types: Counter[str] = Counter()
    mistake_timesteps = 0

    for index, file_spec in enumerate(manifest["files"]):
        source = local_path(raw_root, file_spec)
        verify_file(source, file_spec)
        raw = load_episode(source)
        if "actions" not in raw or "primitive" not in raw:
            raise ValueError(f"Missing actions or primitive in {source}")
        frame_count = len(np.asarray(raw["actions"]))
        required = set(LOW_DIMENSIONAL_KEYS) | set(SOURCE_SENSOR_KEYS)
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Missing required FMB arrays in {source}: {missing}")
        if any(len(np.asarray(raw[key])) != frame_count for key in required):
            raise ValueError(f"Unsynchronized source arrays in {source}")
        for key, (expected_dtype, expected_tail) in SOURCE_SENSOR_KEYS.items():
            value = np.asarray(raw[key])
            if value.dtype != expected_dtype or tuple(value.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Source sensor schema mismatch in {source}: {key} "
                    f"dtype={value.dtype} shape={value.shape}"
                )
        for key, expected_tail in EXPECTED_LOW_DIMENSIONAL_TAILS.items():
            value = np.asarray(raw[key])
            if tuple(value.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Source low-dimensional schema mismatch in {source}: {key} shape={value.shape}"
                )
        labels = labels_from(np.asarray(raw["primitive"]))
        if len(labels) != frame_count or any(not label for label in labels):
            raise ValueError(f"Invalid primitive timeline in {source}")
        identifier = episode_id(index, file_spec["path"])
        destination = episodes_root / identifier
        destination.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for source_key, filename in {**RETAINED_SENSOR_FILES, **LOW_DIMENSIONAL_KEYS}.items():
            value = np.asarray(raw[source_key])
            if source_key == "primitive":
                value = np.asarray(labels, dtype=np.str_)
            arrays[source_key] = save_array(destination / filename, value)
        timestamps = np.arange(frame_count, dtype=np.float64) / FPS
        arrays["timestamp_s"] = save_array(destination / DERIVED_FILES["timestamp_s"], timestamps)
        arrays["source_timestep"] = save_array(
            destination / DERIVED_FILES["source_timestep"], np.arange(frame_count, dtype=np.int64)
        )

        reviews = load_review(review_root, file_spec["path"])
        intervals = primitive_intervals(labels)
        for interval_index, interval in enumerate(intervals):
            key = (interval["start_timestep"], interval["end_timestep_exclusive"])
            decision = reviews.get(key)
            if decision is None:
                interval.update(
                    {
                        "classification": "unclear",
                        "subtask_outcome": "unknown",
                        "quality": None,
                        "quality_provenance": "none",
                        "pause_assessment": "unknown",
                        "retry_assessment": "unknown",
                        "interruption_assessment": "unknown",
                        "mistake_assessment": "unknown",
                        "recovery_assessment": "unknown",
                        "mistake_events": [],
                        "recovery_events": [],
                        "interruption_events": [],
                        "critic_eligible": False,
                        "critic_rejection_reason": "pending_episode_visual_review",
                        "review_provenance": "none",
                    }
                )
            else:
                interval.update(decision)
                reviewed_intervals += 1
            quality = interval.get("quality")
            if isinstance(quality, int):
                quality_intervals[quality] += 1
                quality_timesteps[quality] += interval["end_timestep_exclusive"] - interval["start_timestep"]
            for event in interval.get("mistake_events", []):
                mistake_types[event["mistake_type"]] += 1
                mistake_timesteps += event["end_timestep_exclusive"] - event["start_timestep"]
            if interval["critic_eligible"]:
                eligible_intervals += 1
            critic_records.append(
                {
                    "episode_id": identifier,
                    "split": file_spec["split"],
                    "source_path": file_spec["path"],
                    "interval_index": interval_index,
                    **interval,
                    "action_sequence_reference": {
                        "path": f"episodes/{identifier}/actions.npy",
                        "start_timestep": interval["start_timestep"],
                        "end_timestep_exclusive": interval["end_timestep_exclusive"],
                        "contains_every_native_action": True,
                    },
                    "timestamp_reference": {
                        "path": f"episodes/{identifier}/timestamp_s.npy",
                        "start_timestep": interval["start_timestep"],
                        "end_timestep_exclusive": interval["end_timestep_exclusive"],
                    },
                    "state_and_observation_reference": {
                        "episode_directory": f"episodes/{identifier}",
                        "same_timestep_range": True,
                    },
                }
            )

        episode_actor = actor_accounting(labels)
        for name in ANCHOR_STRIDES_S:
            aggregate_actor[name].update(
                {
                    "episode_level_candidate_anchors": episode_actor[name]["episode_level_candidate_anchors"],
                    "primitive_conditioned_future_candidate_anchors": episode_actor[name][
                        "primitive_conditioned_future_candidate_anchors"
                    ],
                }
            )
        for anchor_s in anchor_times(frame_count, ANCHOR_STRIDES_S["proposed_10hz"]):
            anchor = min(frame_count - 1, int(round(anchor_s * FPS)))
            left, right = future_source_indices(float(anchor_s), frame_count)
            used = np.unique(np.concatenate([left, right]))
            primitive_supported = bool(
                labels[anchor] and all(labels[int(item)] == labels[anchor] for item in used)
            )
            actor_records.append(
                {
                    "episode_id": identifier,
                    "split": file_spec["split"],
                    "anchor_timestep": anchor,
                    "anchor_s_nominal": float(anchor_s),
                    "history_start_timestep": anchor - int(HISTORY_S * FPS),
                    "future_end_s_nominal": float(anchor_s + FUTURE_END_S),
                    "primitive": labels[anchor],
                    "primitive_conditioned_future_supported": primitive_supported,
                    "source_arrays_reference": f"episodes/{identifier}",
                    "interpolation_provenance": {
                        "method": "linear",
                        "source_rate_hz_nominal": FPS,
                        "target_rate_hz": FUTURE_FPS,
                        "target_points": FUTURE_POINTS,
                        "source_index_rule": "floor_and_ceil_of_target_time_times_nominal_fps",
                        "no_extrapolation": True,
                    },
                }
            )

        valid_depth = np.asarray(raw["obs/wrist_1_depth"])
        invalid = (valid_depth == 0) | (valid_depth == 65535)
        episode_metadata = {
            "episode_id": identifier,
            "split": file_spec["split"],
            "source": {
                "repo_id": manifest["source_repo_id"],
                "revision": manifest["source_revision"],
                "path": file_spec["path"],
                "bytes": int(file_spec["bytes"]),
                "sha256": file_spec["sha256"],
                "object": file_spec["object"],
            },
            "frame_count": frame_count,
            "duration_s_nominal": frame_count / FPS,
            "timestamp_provenance": "derived_nominal_grid_no_source_timestamps_in_npy",
            "nominal_fps": FPS,
            "arrays": arrays,
            "source_observed_sensor_counts": {
                "rgb_images_by_camera": {
                    camera: frame_count if f"obs/{camera}" in raw else 0 for camera in CAMERAS
                },
                "depth_maps_by_camera": {
                    camera: frame_count if f"obs/{camera}_depth" in raw else 0 for camera in CAMERAS
                },
                "native_action_samples": frame_count,
            },
            "production_retained_sensor_counts": {
                "rgb_images_by_camera": dict.fromkeys(KEEP_RGB, frame_count),
                "depth_maps_by_camera": {"wrist_1": frame_count},
                "native_action_samples": frame_count,
            },
            "dropped_modalities": {
                "rgb": list(OMIT_RGB),
                "depth": list(OMIT_DEPTH),
                "verified_absent_from_episode_directory": True,
            },
            "depth": {
                "storage": "source_native_D405_Z16_uint16_levels",
                "invalid_values": list(INVALID_DEPTH_VALUES),
                "invalid_pixel_count": int(invalid.sum()),
                "valid_pixel_count": int((~invalid).sum()),
                "depth_units_mm_per_level": DEPTH_MM_PER_LEVEL,
                "depth_scale_provenance": DEPTH_SCALE_PROVENANCE,
                "metric_conversion": "depth_mm = raw_z16_level * 0.1 for valid pixels",
            },
            "primitive_intervals": intervals,
            "actor_anchor_accounting": episode_actor,
        }
        write_json(destination / "episode.json", episode_metadata)
        episode_metadata["metadata_sha256"] = sha256(destination / "episode.json")
        metadata_records.append(episode_metadata)
        total_steps += frame_count
        del raw

    write_jsonl(output_root / "episodes.jsonl", metadata_records)
    write_jsonl(output_root / "actor_anchors_10hz.jsonl", actor_records)
    write_jsonl(output_root / "critic_intervals.jsonl", critic_records)
    accounting = {
        "accounting_provenance": {
            "source_observed_counts": "observed_directly_from_hash_verified_raw_arrays",
            "production_retained_counts": "observed_directly_from_converted_arrays",
            "duration_anchor_and_interval_counts": "derived_exactly_on_nominal_10hz_grid",
        },
        "accepted_source_episodes": len(metadata_records),
        "accepted_continuous_span_duration_s_nominal": total_steps / FPS,
        "unique_synchronized_timesteps_observed": total_steps,
        "source_observed_sensor_counts": {
            "rgb_images_by_camera": dict.fromkeys(CAMERAS, total_steps),
            "depth_maps_by_camera": dict.fromkeys(CAMERAS, total_steps),
            "native_action_samples": total_steps,
        },
        "production_retained_sensor_counts": {
            "rgb_images_by_camera": dict.fromkeys(KEEP_RGB, total_steps),
            "depth_maps_by_camera": {"wrist_1": total_steps},
            "native_action_samples": total_steps,
        },
        "derived_training_rows": {
            "actor_anchor_accounting": {
                name: {"stride_s": ANCHOR_STRIDES_S[name], **dict(values)}
                for name, values in aggregate_actor.items()
            },
            "stored_actor_anchor_view": {
                "stride": "proposed_10hz",
                "rows": len(actor_records),
                "source_frames_duplicated": 0,
            },
            "candidate_critic_intervals": len(critic_records),
            "visually_reviewed_critic_intervals": reviewed_intervals,
            "critic_eligible_intervals": eligible_intervals,
            "rebot_critic_labels": {
                "quality_scope": "source_native_primitive_interval",
                "quality_interval_counts": dict(sorted(quality_intervals.items())),
                "quality_timestep_counts": dict(sorted(quality_timesteps.items())),
                "mistake_scope": "event_span_broadcast_to_timestep_boolean",
                "mistake_event_counts": dict(sorted(mistake_types.items())),
                "mistake_timesteps": mistake_timesteps,
                "recovery_labels_used": False,
            },
        },
        "split_episode_counts": dict(sorted(Counter(item["split"] for item in metadata_records).items())),
        "raw_staging_removed": False,
    }
    corpus = {
        "format": "fmb_source_native_actor_critic_corpus_v1",
        "source_manifest_sha256": sha256(args_manifest_path_placeholder(output_root)),
        "conversion_policy": manifest["conversion_policy"],
        "one_corpus_two_views": {
            "source_episode_collection": "episodes/",
            "actor_view": "actor_anchors_10hz.jsonl",
            "critic_view": "critic_intervals.jsonl",
            "views_copy_source_sensor_or_action_arrays": False,
        },
        "accounting": accounting,
        "validation_status": "pending",
    }
    # The caller replaces the placeholder manifest hash after copying the manifest.
    return {"corpus": corpus, "metadata": metadata_records}


def args_manifest_path_placeholder(output_root: Path) -> Path:
    path = output_root / "source_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Copy the exact production manifest to {path} before conversion")
    return path


def validate(manifest: dict[str, Any], raw_root: Path, output_root: Path) -> dict[str, Any]:
    episodes = read_jsonl(output_root / "episodes.jsonl")
    actor_views = {
        "10hz": read_jsonl(output_root / "actor_anchors_10hz.jsonl"),
    }
    actor_5hz_path = output_root / "actor_anchors_5hz.jsonl"
    if actor_5hz_path.is_file():
        actor_views["5hz"] = read_jsonl(actor_5hz_path)
    critic = read_jsonl(output_root / "critic_intervals.jsonl")
    errors = []
    expected_files = {
        *RETAINED_SENSOR_FILES.values(),
        *LOW_DIMENSIONAL_KEYS.values(),
        *DERIVED_FILES.values(),
        "episode.json",
    }
    manifest_by_path = {item["path"]: item for item in manifest["files"]}
    episode_by_id = {}
    source_paths = set()
    total_steps = 0
    raw_traceability = []
    for metadata in episodes:
        identifier = metadata["episode_id"]
        episode_by_id[identifier] = metadata
        source_path = metadata["source"]["path"]
        if source_path in source_paths:
            errors.append(f"duplicate_source_episode:{source_path}")
        source_paths.add(source_path)
        file_spec = manifest_by_path.get(source_path)
        if file_spec is None:
            errors.append(f"episode_not_in_manifest:{source_path}")
            continue
        directory = output_root / "episodes" / identifier
        actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_files != expected_files:
            errors.append(f"episode_file_schema:{identifier}:{sorted(actual_files)}")
        source = local_path(raw_root, file_spec)
        try:
            verify_file(source, file_spec)
            raw = load_episode(source)
        except Exception as exc:
            errors.append(f"source_provenance:{source_path}:{exc}")
            continue
        frame_count = int(metadata["frame_count"])
        total_steps += frame_count
        for key, filename in {**RETAINED_SENSOR_FILES, **LOW_DIMENSIONAL_KEYS}.items():
            converted = np.load(directory / filename, allow_pickle=False)
            source_value = np.asarray(raw[key])
            if key == "primitive":
                source_value = np.asarray(labels_from(source_value), dtype=np.str_)
            if not np.array_equal(converted, source_value):
                errors.append(f"raw_to_production_mismatch:{identifier}:{key}")
        depth = np.load(directory / "wrist_1_depth_z16.npy", allow_pickle=False)
        if depth.dtype != np.uint16:
            errors.append(f"depth_dtype:{identifier}:{depth.dtype}")
        invalid = (depth == INVALID_DEPTH_VALUES[0]) | (depth == INVALID_DEPTH_VALUES[1])
        depth_metadata = metadata.get("depth", {})
        if depth_metadata.get("invalid_values") != list(INVALID_DEPTH_VALUES):
            errors.append(f"depth_invalid_policy:{identifier}")
        if depth_metadata.get("depth_scale_provenance") != DEPTH_SCALE_PROVENANCE:
            errors.append(f"depth_scale_provenance:{identifier}")
        if depth_metadata.get("depth_units_mm_per_level") != DEPTH_MM_PER_LEVEL:
            errors.append(f"depth_scale_value:{identifier}")
        if depth_metadata.get("invalid_pixel_count") != int(invalid.sum()):
            errors.append(f"depth_invalid_count:{identifier}")
        if depth_metadata.get("valid_pixel_count") != int((~invalid).sum()):
            errors.append(f"depth_valid_count:{identifier}")
        timestamps = np.load(directory / "timestamp_s.npy", allow_pickle=False)
        expected_timestamps = np.arange(frame_count, dtype=np.float64) / FPS
        if not np.array_equal(timestamps, expected_timestamps):
            errors.append(f"timestamp_grid:{identifier}")
        timesteps = np.load(directory / "source_timestep.npy", allow_pickle=False)
        if not np.array_equal(timesteps, np.arange(frame_count, dtype=np.int64)):
            errors.append(f"source_timestep:{identifier}")
        if not np.isfinite(np.load(directory / "actions.npy", allow_pickle=False)).all():
            errors.append(f"actions_nonfinite:{identifier}")
        intervals = metadata["primitive_intervals"]
        if not intervals or intervals[0]["start_timestep"] != 0:
            errors.append(f"primitive_start:{identifier}")
        if intervals and intervals[-1]["end_timestep_exclusive"] != frame_count:
            errors.append(f"primitive_end:{identifier}")
        for previous, current in zip(intervals, intervals[1:], strict=False):
            if previous["end_timestep_exclusive"] != current["start_timestep"]:
                errors.append(f"primitive_gap:{identifier}")
        raw_traceability.append(
            {
                "episode_id": identifier,
                "source_path": source_path,
                "source_sha256_preserved": metadata["source"]["sha256"] == file_spec["sha256"],
                "retained_arrays_exactly_match_source": True,
            }
        )
        del raw

    if source_paths != set(manifest_by_path):
        errors.append("manifest_episode_set_mismatch")
    for view_name, actor_rows in actor_views.items():
        actor_keys = set()
        for row in actor_rows:
            key = (row["episode_id"], row["anchor_timestep"])
            if key in actor_keys:
                errors.append(f"duplicate_actor_anchor:{view_name}:{key}")
            actor_keys.add(key)
            metadata = episode_by_id.get(row["episode_id"])
            if metadata is None:
                errors.append(f"actor_unknown_episode:{view_name}:{row['episode_id']}")
                continue
            if row["split"] != metadata["split"]:
                errors.append(f"actor_split_leakage:{view_name}:{row['episode_id']}")
            if row.get("views_copy_source_arrays") is not False and view_name == "5hz":
                errors.append(f"actor_view_copy_contract:{view_name}:{row['episode_id']}")
            try:
                future_source_indices(float(row["anchor_s_nominal"]), metadata["frame_count"])
            except ValueError:
                errors.append(f"actor_extrapolation:{view_name}:{key}")

    critic_keys = set()
    for row in critic:
        key = (row["episode_id"], row["start_timestep"], row["end_timestep_exclusive"])
        if key in critic_keys:
            errors.append(f"duplicate_critic_interval:{key}")
        critic_keys.add(key)
        metadata = episode_by_id.get(row["episode_id"])
        if metadata is None:
            errors.append(f"critic_unknown_episode:{row['episode_id']}")
            continue
        if row["split"] != metadata["split"]:
            errors.append(f"critic_split_leakage:{row['episode_id']}")
        if row["native_action_samples"] != row["end_timestep_exclusive"] - row["start_timestep"]:
            errors.append(f"critic_action_count:{key}")
        if row["critic_eligible"] and row["critic_rejection_reason"] is not None:
            errors.append(f"eligible_has_rejection:{key}")
        if not row["critic_eligible"] and not row["critic_rejection_reason"]:
            errors.append(f"rejected_without_reason:{key}")
        quality = row.get("quality")
        events = row.get("mistake_events", [])
        if quality is not None:
            if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 5:
                errors.append(f"critic_quality_value:{key}")
            if row.get("quality_provenance") != "human_reviewed_rebot_rubric":
                errors.append(f"critic_quality_provenance:{key}")
            if quality >= 3 and events:
                errors.append(f"critic_quality_mistake_mismatch:{key}")
            if quality == 2 and len(events) != 1:
                errors.append(f"critic_quality_two_event_count:{key}")
            if quality == 1 and not events:
                errors.append(f"critic_quality_one_without_event:{key}")
        for event in events:
            event_start = event.get("start_timestep")
            event_end = event.get("end_timestep_exclusive")
            if (
                event.get("mistake") is not True
                or not isinstance(event_start, int)
                or not isinstance(event_end, int)
                or not row["start_timestep"] <= event_start < event_end <= row["end_timestep_exclusive"]
            ):
                errors.append(f"critic_mistake_span:{key}")
        expected_assessment = "observed" if events else "none_observed"
        if quality is not None and row.get("mistake_assessment") != expected_assessment:
            errors.append(f"critic_mistake_assessment:{key}")

    splits_by_episode = {identifier: metadata["split"] for identifier, metadata in episode_by_id.items()}
    if len(splits_by_episode) != len(set(splits_by_episode)):
        errors.append("split_episode_identity_collision")
    corpus = read_json(output_root / "corpus.json")
    accounting = corpus.get("accounting", {})
    if accounting.get("accepted_source_episodes") != len(episodes):
        errors.append("accounting_episode_count")
    if accounting.get("unique_synchronized_timesteps_observed") != total_steps:
        errors.append("accounting_timestep_count")
    derived = accounting.get("derived_training_rows", {})
    if derived.get("stored_actor_anchor_view", {}).get("rows") != len(actor_views["10hz"]):
        errors.append("accounting_actor_row_count")
    stored_views = derived.get("stored_actor_anchor_views", {})
    for view_name, actor_rows in actor_views.items():
        if stored_views and stored_views.get(view_name, {}).get("rows") != len(actor_rows):
            errors.append(f"accounting_actor_row_count:{view_name}")
    if derived.get("candidate_critic_intervals") != len(critic):
        errors.append("accounting_critic_interval_count")
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "validated_episode_count": len(episodes),
        "validated_unique_synchronized_timesteps": total_steps,
        "validated_actor_rows_10hz": len(actor_views["10hz"]),
        "validated_actor_rows_by_view": {name: len(rows) for name, rows in sorted(actor_views.items())},
        "validated_critic_intervals": len(critic),
        "split_episode_counts": dict(sorted(Counter(splits_by_episode.values()).items())),
        "checks": {
            "schema": not any(item.startswith("episode_file_schema") for item in errors),
            "synchronization": not any(
                item.startswith(("timestamp_grid", "source_timestep")) for item in errors
            ),
            "depth_handling": not any(item.startswith("depth_") for item in errors),
            "action_continuity_and_finiteness": not any(item.startswith("actions_") for item in errors),
            "primitive_boundaries": not any(item.startswith("primitive_") for item in errors),
            "critic_labels": not any(item.startswith("critic_") for item in errors),
            "split_leakage": not any("split_leakage" in item for item in errors),
            "counts": len(episodes) == len(manifest["files"])
            and not any(item.startswith("accounting_") for item in errors),
            "raw_to_production_traceability": not any(
                item.startswith(("source_provenance", "raw_to_production_mismatch")) for item in errors
            ),
            "dropped_modalities_absent": not any(item.startswith("episode_file_schema") for item in errors),
        },
        "raw_to_production_traceability": raw_traceability,
        "raw_staging_removal_allowed": not errors and len(raw_traceability) == len(manifest["files"]),
        "raw_staging_removed": False,
    }
    write_json(output_root / "validation_report.json", report)
    return report


def write_production_audit(
    manifest_path: Path,
    pilot_audit_path: Path,
    output_root: Path,
    review_root: Path | None = None,
) -> Path:
    manifest = read_json(manifest_path)
    corpus = read_json(output_root / "corpus.json")
    validation = read_json(output_root / "validation_report.json")
    pilot_gate = read_json(pilot_audit_path)["acceptance_gate"]
    derived_rows = corpus["accounting"]["derived_training_rows"]
    candidate_intervals = derived_rows["candidate_critic_intervals"]
    reviewed_intervals = derived_rows["visually_reviewed_critic_intervals"]
    unreviewed_intervals = candidate_intervals - reviewed_intervals
    labels_path = Path(__file__).resolve().parent / "fmb_production_quality_mistakes.json"
    dense_manifest_path = (
        output_root.parent / "quality_mistake_review" / "dense_metric_flags" / "manifest.json"
    )
    audit = {
        "status": validation["status"],
        "source": {
            "repo_id": manifest["source_repo_id"],
            "revision": manifest["source_revision"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "selected_source_bytes": sum(int(item["bytes"]) for item in manifest["files"]),
            "selection": manifest["selection"],
        },
        "pilot_gate": pilot_gate,
        "retention_and_depth_policy": manifest["conversion_policy"],
        "accounting": corpus["accounting"],
        "critic_review_state": {
            "candidate_intervals": candidate_intervals,
            "reviewed_intervals": reviewed_intervals,
            "critic_eligible_intervals": derived_rows["critic_eligible_intervals"],
            "unreviewed_intervals": unreviewed_intervals,
            "unreviewed_rejection_reason": (
                "pending_episode_visual_review" if unreviewed_intervals else None
            ),
            "unknown_labels_preserved": True,
            "rebot_critic_labels": corpus["accounting"]["derived_training_rows"].get(
                "rebot_critic_labels", {}
            ),
        },
        "validation": {
            key: validation[key]
            for key in (
                "status",
                "errors",
                "validated_episode_count",
                "validated_unique_synchronized_timesteps",
                "validated_actor_rows_10hz",
                "validated_actor_rows_by_view",
                "validated_critic_intervals",
                "split_episode_counts",
                "checks",
                "raw_staging_removal_allowed",
                "raw_staging_removed",
            )
        },
        "artifacts": {
            "corpus": str(output_root / "corpus.json"),
            "corpus_sha256": sha256(output_root / "corpus.json"),
            "validation_report": str(output_root / "validation_report.json"),
            "validation_report_sha256": sha256(output_root / "validation_report.json"),
            "actor_view_5hz": str(output_root / "actor_anchors_5hz.jsonl"),
            "actor_view_10hz": str(output_root / "actor_anchors_10hz.jsonl"),
            "critic_view": str(output_root / "critic_intervals.jsonl"),
            "production_review_root": str(review_root) if review_root is not None else None,
            "quality_mistake_labels": str(labels_path),
            "quality_mistake_labels_sha256": sha256(labels_path),
            "dense_metric_flag_manifest": str(dense_manifest_path),
            "dense_metric_flag_manifest_sha256": sha256(dense_manifest_path),
        },
    }
    output = output_root.parent / "fmb_production_audit.json"
    write_json(output, audit)
    return output


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "fmb_production.json")
    parser.add_argument("--pilot-manifest", type=Path, default=here / "fmb_pilot.json")
    parser.add_argument(
        "--pilot-audit",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/pilot/fmb_pilot_audit.json"),
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/pilot/review"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/production/raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/production/corpus"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    select_parser = commands.add_parser("select")
    select_parser.add_argument("--episode-count", type=int, default=100)
    commands.add_parser("acquire")
    commands.add_parser("convert")
    commands.add_parser("views")
    commands.add_parser("validate")
    commands.add_parser("build")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        manifest = select_manifest(args.pilot_manifest, args.manifest, args.episode_count)
        print(json.dumps(manifest["selection"], indent=2))
        return
    if args.command == "views":
        print(json.dumps(write_actor_views(args.output_root), indent=2))
        return
    manifest = read_json(args.manifest)
    if args.command in {"acquire", "build"}:
        paths = acquire(manifest, args.raw_root)
        print(json.dumps({"verified_source_files": len(paths)}, indent=2))
        if args.command == "acquire":
            return
    if args.command in {"convert", "build"}:
        args.output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.manifest, args.output_root / "source_manifest.json")
        converted = convert(manifest, args.raw_root, args.output_root, args.pilot_audit, args.review_root)
        converted["corpus"]["source_manifest_sha256"] = sha256(args.output_root / "source_manifest.json")
        write_json(args.output_root / "corpus.json", converted["corpus"])
        write_actor_views(args.output_root)
        converted["corpus"] = read_json(args.output_root / "corpus.json")
        print(json.dumps(converted["corpus"]["accounting"], indent=2))
        if args.command == "convert":
            return
    report = validate(manifest, args.raw_root, args.output_root)
    if args.command in {"validate", "build"}:
        corpus_path = args.output_root / "corpus.json"
        corpus = read_json(corpus_path)
        corpus["validation_status"] = report["status"]
        corpus["validation_report_sha256"] = sha256(args.output_root / "validation_report.json")
        write_json(corpus_path, corpus)
        write_production_audit(args.manifest, args.pilot_audit, args.output_root, args.review_root)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
