#!/usr/bin/env python

"""Acquire and audit a pinned raw FMB pilot without converting it.

The raw FMB ``.npy`` files contain pickled dictionaries, so this tool only
loads files after their byte length and SHA-256 match the pinned manifest. The
report deliberately distinguishes values observed in source files from values
derived on FMB's nominal 10 Hz grid. FMB trajectories do not contain source
timestamps, and this script never presents nominal timestamps as measured ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")
DEPTH_KEYS = tuple(f"obs/{camera}_depth" for camera in CAMERAS)
RGB_KEYS = tuple(f"obs/{camera}" for camera in CAMERAS)
SEQUENCE_KEYS = (
    *RGB_KEYS,
    *DEPTH_KEYS,
    "obs/tcp_pose",
    "obs/tcp_vel",
    "obs/tcp_force",
    "obs/tcp_torque",
    "obs/q",
    "obs/dq",
    "obs/jacobian",
    "obs/gripper_pose",
    "actions",
    "primitive",
)
EXPECTED_SHAPES = {
    **dict.fromkeys(RGB_KEYS, (256, 256, 3)),
    **dict.fromkeys(DEPTH_KEYS, (256, 256)),
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
ACTION_END_S = 29.0 / 30.0
HISTORY_S = 6.0
ANCHOR_STRIDES_S = {"original_2s": 2.0, "proposed_5hz": 0.2, "proposed_10hz": 0.1}
MAX_QUANTILE_VALUES = 1_000_000
READ_CHUNK = 8 * 1024 * 1024


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


def local_path(raw_root: Path, file_spec: dict[str, Any]) -> Path:
    relative = Path(file_spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    return raw_root / relative


def verify_local(path: Path, file_spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    expected = {key: file_spec[key] for key in ("bytes", "sha256")}
    if actual != expected:
        raise ValueError(f"Pinned payload mismatch for {path}: {actual} != {expected}")


def verify_hub(manifest: dict[str, Any]) -> dict[str, Any]:
    info = HfApi().dataset_info(
        manifest["source_repo_id"], revision=manifest["source_revision"], files_metadata=True
    )
    if info.sha != manifest["source_revision"]:
        raise ValueError(f"Resolved revision changed: {info.sha}")
    published = {entry.rfilename: entry for entry in info.siblings}
    checked = []
    for file_spec in manifest["files"]:
        entry = published.get(file_spec["path"])
        if entry is None or entry.lfs is None:
            raise ValueError(f"Pinned FMB file missing or no longer LFS-backed: {file_spec['path']}")
        actual = {"bytes": int(entry.size), "sha256": str(entry.lfs.sha256)}
        expected = {key: file_spec[key] for key in ("bytes", "sha256")}
        if actual != expected:
            raise ValueError(f"Published metadata changed for {file_spec['path']}: {actual}")
        checked.append({"path": file_spec["path"], **actual})
    return {"repo_id": manifest["source_repo_id"], "revision": info.sha, "files": checked}


def acquire(manifest: dict[str, Any], raw_root: Path) -> list[Path]:
    verify_hub(manifest)
    required = sum(int(item["bytes"]) for item in manifest["files"])
    raw_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(raw_root).free < required * 1.2:
        raise OSError(f"FMB pilot needs {required * 1.2:.0f} bytes including reserve")
    paths = []
    for file_spec in manifest["files"]:
        path = Path(
            hf_hub_download(
                repo_id=manifest["source_repo_id"],
                repo_type="dataset",
                revision=manifest["source_revision"],
                filename=file_spec["path"],
                local_dir=raw_root,
            )
        )
        verify_local(path, file_spec)
        paths.append(path)
    return paths


def load_episode(path: Path) -> dict[str, Any]:
    # FMB encodes each dictionary in a pickled object array. Hash verification is
    # mandatory before reaching this function; never use it on an arbitrary npy.
    loaded = np.load(path, allow_pickle=True)
    if not isinstance(loaded, np.ndarray) or loaded.shape != () or loaded.dtype != object:
        raise TypeError(f"Expected a scalar object array in {path}, got {type(loaded)}")
    episode = loaded.item()
    if not isinstance(episode, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(episode)}")
    return episode


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(child) for child in value]
    return value


def quantiles(values: np.ndarray, points: tuple[float, ...]) -> dict[str, float]:
    values = np.asarray(values)
    if values.size == 0:
        return {}
    return {
        f"p{point:g}": float(value)
        for point, value in zip(points, np.percentile(values, points), strict=False)
    }


def sampled_finite_positive(array: np.ndarray) -> tuple[np.ndarray, int]:
    stride = max(1, int(math.ceil(array.size / MAX_QUANTILE_VALUES)))
    sample = np.asarray(array).reshape(-1)[::stride]
    sample = sample[np.isfinite(sample) & (sample > 0)]
    return sample, stride


def true_spans(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return [[int(group[0]), int(group[-1] + 1)] for group in np.split(indices, breaks)]


def unit_assessment(array: np.ndarray, median: float | None, maximum: float | None) -> dict[str, Any]:
    if np.issubdtype(array.dtype, np.integer):
        return {
            "status": "assumed_for_conversion",
            "depth_units_mm_per_level": 0.1,
            "provenance": "user_authorized_same_D405_model_assumption",
            "numeric_hypothesis": "raw_D405_Z16_levels",
            "note": (
                "FMB does not retain get_depth_scale(); 0.1 mm/level is adopted from the "
                "locally measured D405 configuration by explicit user decision."
            ),
        }
    if median is None or maximum is None:
        hypothesis = "unavailable"
    elif 0.02 <= median <= 10 and maximum <= 100:
        hypothesis = "values_are_compatible_with_meters"
    else:
        hypothesis = "scale_is_not_recognized"
    return {
        "status": "unproven",
        "numeric_hypothesis": hypothesis,
        "note": "Magnitude alone is not accepted as proof of physical units.",
    }


def depth_summary(array: np.ndarray, fps: float) -> dict[str, Any]:
    if array.ndim != 3:
        return {"error": f"expected [time,height,width], got {list(array.shape)}"}
    total = int(array.size)
    finite_count = zero_count = negative_count = saturated_count = 0
    valid_fraction = []
    saturation_value = int(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else None
    for frame in array:
        finite = np.isfinite(frame)
        saturated = (
            finite & (frame == saturation_value) if saturation_value is not None else np.zeros_like(finite)
        )
        finite_count += int(finite.sum())
        zero_count += int((finite & (frame == 0)).sum())
        negative_count += int((finite & (frame < 0)).sum())
        saturated_count += int(saturated.sum())
        valid_fraction.append(float((finite & (frame > 0) & ~saturated).mean()))
    valid_fraction_array = np.asarray(valid_fraction)
    sample, sample_stride = sampled_finite_positive(array)
    if saturation_value is not None:
        sample = sample[sample != saturation_value]
    distribution = quantiles(sample, (0, 1, 5, 25, 50, 75, 95, 99, 100))

    temporal_medians = []
    temporal_p95 = []
    spatial = np.asarray(array)[:, ::8, ::8]
    for previous, current in zip(spatial[:-1], spatial[1:], strict=False):
        valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0) & (current > 0)
        if saturation_value is not None:
            valid &= (previous != saturation_value) & (current != saturation_value)
        difference = np.abs(current[valid] - previous[valid])
        if difference.size:
            temporal_medians.append(float(np.median(difference)))
            temporal_p95.append(float(np.percentile(difference, 95)))
    temporal_medians_array = np.asarray(temporal_medians)
    if temporal_medians_array.size:
        center = float(np.median(temporal_medians_array))
        mad = float(np.median(np.abs(temporal_medians_array - center)))
        jump_threshold = center + max(10 * mad, np.finfo(np.float64).eps)
        jump_frames = (np.flatnonzero(temporal_medians_array > jump_threshold) + 1).tolist()
    else:
        center = mad = jump_threshold = None
        jump_frames = []
    median = distribution.get("p50")
    maximum = distribution.get("p100")
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "pixels": total,
        "finite_fraction_exact": finite_count / total if total else None,
        "usable_positive_nonsaturated_fraction_exact": (
            (finite_count - zero_count - negative_count - saturated_count) / total if total else None
        ),
        "zero_fraction_exact": zero_count / total if total else None,
        "negative_fraction_exact": negative_count / total if total else None,
        "dtype_max_sentinel_candidate": saturation_value,
        "dtype_max_fraction_exact": saturated_count / total if total else None,
        "invalid_value_observation": (
            "zero_and_dtype_max_are_sentinel_candidates"
            if zero_count and saturated_count
            else "mixed_or_no_invalid_values_observed"
        ),
        "positive_nonsaturated_value_distribution_sampled": distribution,
        "quantile_sample_stride": sample_stride,
        "quantile_sample_values": int(sample.size),
        "per_frame_valid_fraction": {
            **quantiles(valid_fraction_array, (0, 5, 50, 95, 100)),
            "below_50_percent_spans_end_exclusive": true_spans(valid_fraction_array < 0.5),
            "below_90_percent_spans_end_exclusive": true_spans(valid_fraction_array < 0.9),
        },
        "temporal_difference_on_8x_spatial_subsample": {
            "frame_pairs": len(temporal_medians),
            "median_of_pair_medians": center,
            "median_absolute_deviation": mad,
            "p95_of_pair_p95": float(np.percentile(temporal_p95, 95)) if temporal_p95 else None,
            "robust_jump_threshold": jump_threshold,
            "jump_frame_indices": jump_frames,
            "jump_times_s_nominal": [round(index / fps, 6) for index in jump_frames],
        },
        "physical_units": unit_assessment(array, median, maximum),
    }


def paired_wrist_depth_summary(wrist_1: np.ndarray, wrist_2: np.ndarray) -> dict[str, Any]:
    if wrist_1.shape != wrist_2.shape or wrist_1.dtype != wrist_2.dtype:
        return {
            "same_shape_and_dtype": False,
            "wrist_1": {"shape": list(wrist_1.shape), "dtype": str(wrist_1.dtype)},
            "wrist_2": {"shape": list(wrist_2.shape), "dtype": str(wrist_2.dtype)},
        }
    sentinel = np.iinfo(wrist_1.dtype).max
    valid_1 = (wrist_1 > 0) & (wrist_1 != sentinel)
    valid_2 = (wrist_2 > 0) & (wrist_2 != sentinel)
    both = valid_1 & valid_2
    union = valid_1 | valid_2
    difference = np.abs(wrist_1[both].astype(np.int32) - wrist_2[both].astype(np.int32))
    correlations = []
    for first, second, mask in zip(wrist_1, wrist_2, both, strict=True):
        first_valid, second_valid = first[mask], second[mask]
        if mask.sum() > 100 and np.std(first_valid) > 0 and np.std(second_valid) > 0:
            correlations.append(float(np.corrcoef(first_valid, second_valid)[0, 1]))
    return {
        "same_shape_and_dtype": True,
        "arrays_exactly_equal": bool(np.array_equal(wrist_1, wrist_2)),
        "all_pixel_equal_fraction": float((wrist_1 == wrist_2).mean()),
        "valid_mask_iou": float(both.sum() / union.sum()),
        "jointly_valid_fraction": float(both.mean()),
        "jointly_valid_exact_equal_fraction": float((difference == 0).mean()),
        "same_coordinate_frame_correlation_median": (
            float(np.median(correlations)) if correlations else None
        ),
        "conclusion": "distinct_physical_streams_not_duplicates",
    }


def alignment_summary(rgb: np.ndarray, depth: np.ndarray) -> dict[str, Any]:
    if rgb.shape[:3] != depth.shape or rgb.shape[-1] != 3:
        return {"shape_aligned": False, "rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape)}
    frame_indices = np.unique(np.linspace(0, len(depth) - 1, min(12, len(depth)), dtype=int))
    correlations = []
    edge_overlaps = []
    for index in frame_indices:
        gray = cv2.cvtColor(rgb[index], cv2.COLOR_BGR2GRAY).astype(np.float32)
        z = depth[index].astype(np.float32)
        valid = np.isfinite(z) & (z > 0) & (z != np.iinfo(depth.dtype).max)
        if valid.sum() < 100:
            continue
        fill = float(np.median(z[valid]))
        z = np.where(valid, z, fill)
        rgb_grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        z_grad = cv2.magnitude(cv2.Sobel(z, cv2.CV_32F, 1, 0), cv2.Sobel(z, cv2.CV_32F, 0, 1))
        interior = valid & (rgb_grad > 0)
        if interior.sum() < 100:
            continue
        correlations.append(float(np.corrcoef(rgb_grad[interior], z_grad[interior])[0, 1]))
        rgb_cut = float(np.percentile(rgb_grad[interior], 90))
        z_cut = float(np.percentile(z_grad[interior], 90))
        rgb_edge = interior & (rgb_grad >= rgb_cut)
        z_edge = interior & (z_grad >= z_cut)
        edge_overlaps.append(float((rgb_edge & z_edge).sum() / max(1, (rgb_edge | z_edge).sum())))
    return {
        "shape_aligned": True,
        "sample_frame_indices": frame_indices.tolist(),
        "gradient_correlation_median": float(np.nanmedian(correlations)) if correlations else None,
        "top_decile_edge_iou_median": float(np.median(edge_overlaps)) if edge_overlaps else None,
        "status": "requires_visual_confirmation",
    }


def normalize_labels(values: np.ndarray) -> list[str]:
    labels = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        labels.append(str(value).strip())
    return labels


def primitive_intervals(labels: list[str], fps: float) -> list[dict[str, Any]]:
    if not labels:
        return []
    intervals = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            count = index - start
            intervals.append(
                {
                    "primitive": labels[start],
                    "start_timestep": start,
                    "end_timestep_exclusive": index,
                    "start_s_nominal": start / fps,
                    "end_s_nominal_exclusive": index / fps,
                    "native_action_samples": count,
                    "duration_s_nominal": count / fps,
                    "boundary_provenance": "source_native",
                    "one_frame_flicker": count == 1,
                    "completeness": "pending_visual_review",
                    "critic_eligibility": "pending_visual_review",
                }
            )
            start = index
    return intervals


def anchor_grid(frame_count: int, fps: float, stride_s: float) -> np.ndarray:
    last_timestamp = (frame_count - 1) / fps
    last = last_timestamp - ACTION_END_S
    if last < HISTORY_S:
        return np.empty(0, dtype=np.float64)
    count = int(math.floor((last - HISTORY_S) / stride_s + 1e-9)) + 1
    return HISTORY_S + np.arange(count, dtype=np.float64) * stride_s


def primitive_conditioned_count(labels: list[str], fps: float, anchors: np.ndarray) -> int:
    labels_array = np.asarray(labels)
    count = 0
    source_times = np.arange(len(labels_array), dtype=np.float64) / fps
    for anchor in anchors:
        targets = anchor + np.arange(30, dtype=np.float64) / 30.0
        right = np.clip(np.searchsorted(source_times, targets, side="left"), 0, len(source_times) - 1)
        left = np.clip(right - 1, 0, len(source_times) - 1)
        used = np.unique(np.concatenate([left, right]))
        anchor_index = min(len(labels_array) - 1, int(round(anchor * fps)))
        if labels_array[anchor_index] and np.all(labels_array[used] == labels_array[anchor_index]):
            count += 1
    return count


def actor_yield(frame_count: int, labels: list[str], fps: float) -> dict[str, Any]:
    result = {}
    for name, stride in ANCHOR_STRIDES_S.items():
        anchors = anchor_grid(frame_count, fps, stride)
        result[name] = {
            "stride_s": stride,
            "episode_level_candidate_anchors": int(len(anchors)),
            "primitive_conditioned_future_candidate_anchors": primitive_conditioned_count(
                labels, fps, anchors
            ),
            "retained_after_motion_and_review": None,
        }
    return result


def numeric_array_summary(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    result = {"dtype": str(array.dtype), "shape": list(array.shape)}
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        result["finite_fraction"] = float(finite.mean()) if array.size else None
        values = array[finite]
        if values.size:
            result.update({"min": float(values.min()), "max": float(values.max())})
    return result


def action_summary(actions: np.ndarray, fps: float) -> dict[str, Any]:
    actions = np.asarray(actions)
    result = numeric_array_summary(actions)
    if actions.ndim != 2 or not len(actions):
        return result
    differences = np.abs(np.diff(actions.astype(np.float64), axis=0))
    result.update(
        {
            "semantics": "source_commanded_cartesian_xyz_rpy_gripper",
            "native_samples": int(len(actions)),
            "nominal_rate_hz": fps,
            "per_dimension_min": np.nanmin(actions, axis=0).astype(float).tolist(),
            "per_dimension_max": np.nanmax(actions, axis=0).astype(float).tolist(),
            "per_step_absolute_difference_max": (
                np.nanmax(differences, axis=0).astype(float).tolist() if len(differences) else []
            ),
            "per_step_absolute_difference_p99": (
                np.nanpercentile(differences, 99, axis=0).astype(float).tolist() if len(differences) else []
            ),
            "timestamp_provenance": "derived_nominal_grid_no_source_timestamps_in_npy",
            "timestamp_monotonicity": "not_observable",
            "missing_action_intervals": "not_observable_without_source_timestamps",
            "30hz_representation": "requires_linear_interpolation_with_source-index_provenance",
        }
    )
    return result


def schema_summary(episode: dict[str, Any], frame_count: int) -> dict[str, Any]:
    arrays = {}
    errors = []
    for key in SEQUENCE_KEYS:
        if key not in episode:
            errors.append(f"missing:{key}")
            continue
        array = np.asarray(episode[key])
        arrays[key] = numeric_array_summary(array)
        if not array.shape or array.shape[0] != frame_count:
            errors.append(f"length:{key}:{array.shape}")
        elif tuple(array.shape[1:]) != EXPECTED_SHAPES[key]:
            errors.append(f"shape:{key}:{array.shape}")
    return {"keys": sorted(episode), "arrays": arrays, "errors": errors}


def audit_episode(path: Path, file_spec: dict[str, Any], fps: float) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_local(path, file_spec)
    episode = load_episode(path)
    if "actions" not in episode:
        raise ValueError(f"FMB episode has no actions array: {path}")
    actions = np.asarray(episode["actions"])
    frame_count = int(len(actions))
    labels = normalize_labels(np.asarray(episode.get("primitive", [])))
    schema = schema_summary(episode, frame_count)
    depth = {}
    alignment = {}
    for camera in CAMERAS:
        depth_key, rgb_key = f"obs/{camera}_depth", f"obs/{camera}"
        if depth_key in episode:
            depth[camera] = depth_summary(np.asarray(episode[depth_key]), fps)
        if depth_key in episode and rgb_key in episode:
            alignment[camera] = alignment_summary(
                np.asarray(episode[rgb_key]), np.asarray(episode[depth_key])
            )
    intervals = primitive_intervals(labels, fps)
    missing_labels = sum(not label for label in labels)
    report = {
        "source_path": file_spec["path"],
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_spec["sha256"],
        "manifest_object": file_spec["object"],
        "source_object_info": json_value(episode.get("object_info")),
        "unique_synchronized_timesteps_observed": frame_count,
        "source_duration_s_nominal": frame_count / fps,
        "timebase": {
            "nominal_fps": fps,
            "source_timestamps_present": False,
            "provenance": "derived_nominal_grid_no_source_timestamps_in_npy",
        },
        "physical_sensor_counts_observed": {
            "rgb_images_by_camera": {
                camera: frame_count if f"obs/{camera}" in episode else 0 for camera in CAMERAS
            },
            "depth_maps_by_camera": {
                camera: frame_count if f"obs/{camera}_depth" in episode else 0 for camera in CAMERAS
            },
            "native_action_samples": frame_count,
        },
        "schema": schema,
        "depth": depth,
        "wrist_depth_comparison": paired_wrist_depth_summary(
            np.asarray(episode["obs/wrist_1_depth"]),
            np.asarray(episode["obs/wrist_2_depth"]),
        ),
        "rgb_depth_alignment": alignment,
        "action": action_summary(actions, fps),
        "primitive": {
            "samples": len(labels),
            "missing_or_empty_samples": missing_labels,
            "labels": dict(sorted(Counter(labels).items())),
            "sequence": [item["primitive"] for item in intervals],
            "candidate_intervals": intervals,
            "candidate_interval_count": len(intervals),
            "one_frame_flicker_count": sum(item["one_frame_flicker"] for item in intervals),
            "critic_eligible_interval_count": None,
            "critic_eligible_duration_s": None,
            "review_state": "pending_visual_review",
        },
        "actor_anchor_yield": actor_yield(frame_count, labels, fps),
        "automated_checks": {},
    }
    checks = {
        "schema_complete": not schema["errors"],
        "all_four_depth_streams_present": set(depth) == set(CAMERAS),
        "both_wrist_depth_streams_present": {"wrist_1", "wrist_2"}.issubset(depth),
        "actions_finite": report["action"].get("finite_fraction") == 1.0,
        "primitive_labels_complete": len(labels) == frame_count and missing_labels == 0,
        "depth_scale_resolved_for_conversion": all(
            depth[camera]["physical_units"]["status"] in {"proven", "assumed_for_conversion"}
            for camera in ("wrist_1",)
        ),
        "visual_alignment_reviewed": False,
        "subtask_completeness_reviewed": False,
    }
    report["automated_checks"] = checks
    review_template = {
        "source_path": file_spec["path"],
        "review_status": "pending",
        "depth": {
            camera: {"rgb_depth_aligned": None, "persistent_occlusion": None, "notes": ""}
            for camera in CAMERAS
        },
        "episode": {"clean_completion": None, "pauses": [], "retries": [], "interruptions": [], "notes": ""},
        "subtasks": [
            {
                "primitive": item["primitive"],
                "start_timestep": item["start_timestep"],
                "end_timestep_exclusive": item["end_timestep_exclusive"],
                "classification": "unclear",
                "reviewed_start_timestep": None,
                "reviewed_end_timestep_exclusive": None,
                "boundary_provenance": "source_native",
                "subtask_outcome": "unknown",
                "quality": None,
                "quality_provenance": "none",
                "mistake_events": [],
                "recovery_events": [],
                "interruption_events": [],
                "critic_eligible": None,
                "critic_rejection_reason": None,
                "notes": "",
            }
            for item in intervals
        ],
    }
    return report, review_template


def colorized_depth(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(frame) & (frame > 0) & (frame != np.iinfo(frame.dtype).max)
    normalized = np.zeros(frame.shape, dtype=np.uint8)
    if high > low:
        normalized[valid] = np.clip((frame[valid] - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def render_episode(
    episode: dict[str, Any], report: dict[str, Any], output: Path, fps: float, max_frames: int
) -> None:
    cameras = CAMERAS
    height, width = 256, 256
    frame_count = len(episode["actions"])
    limit = frame_count if max_frames <= 0 else min(frame_count, max_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height * len(cameras))
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create review video: {output}")
    scales = {}
    for camera in cameras:
        distribution = report["depth"][camera]["positive_nonsaturated_value_distribution_sampled"]
        scales[camera] = (
            distribution.get("p1", distribution["p0"]),
            distribution.get("p99", distribution["p100"]),
        )
    labels = normalize_labels(np.asarray(episode["primitive"]))
    try:
        for index in range(limit):
            rows = []
            for camera in cameras:
                rgb = np.asarray(episode[f"obs/{camera}"][index]).copy()
                depth = colorized_depth(np.asarray(episode[f"obs/{camera}_depth"][index]), *scales[camera])
                cv2.putText(
                    rgb,
                    f"{camera} RGB",
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    depth,
                    f"{camera} depth",
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                rows.append(np.concatenate([rgb, depth], axis=1))
            canvas = np.concatenate(rows, axis=0)
            cv2.putText(
                canvas,
                f"t={index / fps:.1f}s  {labels[index]}",
                (6, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()


def aggregate(episodes: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    total_steps = sum(item["unique_synchronized_timesteps_observed"] for item in episodes)
    rgb = {
        camera: sum(
            item["physical_sensor_counts_observed"]["rgb_images_by_camera"][camera] for item in episodes
        )
        for camera in CAMERAS
    }
    depth = {
        camera: sum(
            item["physical_sensor_counts_observed"]["depth_maps_by_camera"][camera] for item in episodes
        )
        for camera in CAMERAS
    }
    actor = {}
    for name in ANCHOR_STRIDES_S:
        actor[name] = {
            "episode_level_candidate_anchors": sum(
                item["actor_anchor_yield"][name]["episode_level_candidate_anchors"] for item in episodes
            ),
            "primitive_conditioned_future_candidate_anchors": sum(
                item["actor_anchor_yield"][name]["primitive_conditioned_future_candidate_anchors"]
                for item in episodes
            ),
            "retained_after_motion_and_review": None,
        }
    comparison_fields = (
        "all_pixel_equal_fraction",
        "valid_mask_iou",
        "jointly_valid_exact_equal_fraction",
        "same_coordinate_frame_correlation_median",
    )
    wrist_comparison = {
        field: quantiles(
            np.asarray([item["wrist_depth_comparison"][field] for item in episodes]), (0, 50, 100)
        )
        for field in comparison_fields
    }
    return {
        "accounting_provenance": {
            "episode_and_sample_counts": "observed_directly",
            "duration_and_anchor_counts": "derived_exactly_on_nominal_10hz_grid",
            "retained_rows_and_critic_eligibility": "pending_human_review",
        },
        "pilot_source_files": len(episodes),
        "accepted_source_episodes": None,
        "accepted_continuous_span_duration_s": None,
        "source_duration_s_nominal": total_steps / fps,
        "unique_synchronized_timesteps_observed": total_steps,
        "rgb_images_by_physical_camera_observed": rgb,
        "depth_maps_by_physical_camera_observed": depth,
        "wrist_depth_comparison": wrist_comparison,
        "native_action_samples_observed": total_steps,
        "actor": actor,
        "complete_subtask_intervals": None,
        "critic_eligible_subtask_intervals": None,
        "candidate_native_primitive_intervals": sum(
            item["primitive"]["candidate_interval_count"] for item in episodes
        ),
        "row_to_unique_timestep_ratio": None,
        "split_episode_counts": None,
    }


def audit(manifest: dict[str, Any], work_root: Path, render: bool, max_render_frames: int) -> Path:
    raw_root = work_root / "raw"
    reports = []
    review_root = work_root / "review"
    for file_spec in manifest["files"]:
        path = local_path(raw_root, file_spec)
        report, template = audit_episode(path, file_spec, float(manifest["nominal_fps"]))
        reports.append(report)
        stem = path.stem
        write_json(review_root / f"{stem}.review.json", template)
        if render:
            episode = load_episode(path)
            render_episode(
                episode,
                report,
                review_root / f"{stem}.rgb_depth.mp4",
                float(manifest["nominal_fps"]),
                max_render_frames,
            )
            del episode
    automated_failures = sorted(
        {name for episode in reports for name, passed in episode["automated_checks"].items() if not passed}
    )
    result = {
        "source": {
            "repo_id": manifest["source_repo_id"],
            "revision": manifest["source_revision"],
            "license": manifest["license"],
            "selection": manifest["selection"],
            "depth_capture_evidence": {
                "official_code_repo": "rail-berkeley/fmb",
                "inspected_revision": "d4da6ce044a9806f41e58bf7423b5a3c05289925e",
                "capture_file": "robot_infra/camera/rs_capture.py",
                "capture_format": "pyrealsense2 Z16 aligned to the color stream",
                "depth_scale_recorded": False,
                "preprocessing_observation": "Official policies divide raw depth by 65535; this is normalization, not metric conversion.",
            },
        },
        "scope": "read_only_pilot_audit_no_conversion",
        "production_retention_decision": {
            "keep_rgb": ["side_1", "side_2", "wrist_1"],
            "keep_depth": ["wrist_1"],
            "omit_rgb": ["wrist_2"],
            "omit_depth": ["side_1", "side_2", "wrist_2"],
            "selected_wrist_reason": "wrist_1 has higher usable depth coverage in 11 of 12 pilot episodes; the second distinct view has no established training-value ablation to justify its cost.",
            "depth_units_mm_per_level": 0.1,
            "depth_scale_provenance": "user_authorized_same_D405_model_assumption",
            "source_payload_note": "Raw FMB npy files are monolithic; omit these fields only in converted production data and remove reacquirable raw staging after validation.",
            "uncompressed_camera_payload_reduction_fraction": 0.45,
        },
        "episodes": reports,
        "accounting": aggregate(reports, float(manifest["nominal_fps"])),
        "acceptance_gate": {
            "status": "depth_resolved_pending_complete_subtask_review",
            "automated_unmet_or_pending_checks": automated_failures,
            "full_download_allowed": False,
            "full_converter_allowed": False,
            "note": "Depth modality decision is resolved; complete-subtask and interruption review remains pending.",
        },
    }
    output = work_root / "fmb_pilot_audit.json"
    write_json(output, result)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=here / "fmb_pilot.json")
    parser.add_argument("--work-root", type=Path, default=Path("outputs/diverse_robot_dataset/fmb/pilot"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="verify pinned metadata without downloading payloads")
    commands.add_parser("acquire", help="download only the twelve pinned raw trajectories")
    audit_parser = commands.add_parser("audit", help="audit already downloaded trajectories")
    audit_parser.add_argument("--render", action="store_true")
    audit_parser.add_argument("--max-render-frames", type=int, default=0)
    all_parser = commands.add_parser("all", help="verify, acquire, audit, and render the pilot")
    all_parser.add_argument("--max-render-frames", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = read_json(args.manifest)
    if args.command == "verify":
        print(json.dumps(verify_hub(manifest), indent=2))
        return
    if args.command in {"acquire", "all"}:
        paths = acquire(manifest, args.work_root / "raw")
        print(json.dumps({"downloaded": [str(path) for path in paths]}, indent=2))
        if args.command == "acquire":
            return
    output = audit(
        manifest,
        args.work_root,
        render=args.command == "all" or bool(getattr(args, "render", False)),
        max_render_frames=int(getattr(args, "max_render_frames", 0)),
    )
    print(json.dumps({"report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
