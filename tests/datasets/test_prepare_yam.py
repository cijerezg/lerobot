"""Tests for the YAM source preparation script and the corpus ingest's gripper transform."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/dataset/diverse_robot_dataset"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_yam = _load("prepare_yam")
build_corpus = _load("build_corpus")

FPS = 25.0


def _episode(seconds: float, *, still_until_s: float = 0.0, gripper_only_from_s: float | None = None):
    """Seven-wide state at 25 Hz: still, then joints moving, then optionally only the gripper."""
    frames = int(seconds * FPS) + 1
    relative = np.arange(frames) / FPS
    rng = np.random.default_rng(0)
    states = rng.normal(0.0, 1e-4, size=(frames, 7))
    moving = relative >= still_until_s
    states[moving, 0] += 0.5 * np.sin(relative[moving])
    states[moving, 3] += 0.2 * relative[moving]
    if gripper_only_from_s is not None:
        tail = relative >= gripper_only_from_s
        states[tail, :6] = states[tail][:, :6].mean(axis=0)
        states[tail, 6] = np.linspace(0.0, 0.8, int(tail.sum()))
    return relative, states


def test_screen_marks_still_cells_inactive_and_moving_cells_active() -> None:
    relative, states = _episode(8.0, still_until_s=3.0, gripper_only_from_s=6.0)
    anchors, active = prepare_yam.screen(relative, states)
    assert anchors == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # Cells 0-2 sit entirely inside the still span (cell 2 ends at 2.97 s); the move starts at 3 s.
    assert active[:3] == [False, False, False]
    assert active[3:6] == [True, True, True]
    # From 6 s only the gripper moves, and a gripper move alone keeps the cell active.
    assert active[6:] == [True, True]


def _summary(index: int, active_fraction: float, excluded: list[str] | None = None) -> dict:
    return {
        "episode_index": index,
        "active_cells": 5,
        "active_fraction": active_fraction,
        "joint_path_length_rad": 1.0,
        "excluded": excluded or [],
    }


def _scan_paths(tmp_path: Path, summaries: list[dict]):
    class Paths:
        component = "unit"
        scan = tmp_path / "episode_scan.json"
        candidates = tmp_path / "candidates.json"

    prepare_yam.write_json(Paths.scan, {"source": "unit/repo", "excluded": {}, "summaries": summaries})
    return Paths()


def test_nominate_spreads_the_quota_over_recording_order_and_takes_the_most_active_per_slice(tmp_path: Path) -> None:
    # Twenty episodes; the most active one of each quarter is the pick, so no two picks share a slice.
    activity = {3: 0.9, 8: 0.95, 13: 0.85, 17: 0.99}
    summaries = [_summary(i, activity.get(i, 0.5)) for i in range(20)]
    value = prepare_yam.nominate(_scan_paths(tmp_path, summaries), per_set=4)
    assert [item["episode_index"] for item in value["candidates"]] == [3, 8, 13, 17]
    assert value["pool_size"] == 20 and value["candidate_count"] == 4


def test_nominate_never_picks_an_excluded_take(tmp_path: Path) -> None:
    summaries = [_summary(i, 0.5) for i in range(8)]
    summaries[1] = _summary(1, 1.0, ["source_flagged_failure"])
    summaries[5] = _summary(5, 1.0, ["video_data_mismatch"])
    summaries[6] = _summary(6, 1.0, ["no_grasp"])
    value = prepare_yam.nominate(_scan_paths(tmp_path, summaries), per_set=4)
    chosen = {item["episode_index"] for item in value["candidates"]}
    assert chosen.isdisjoint({1, 5, 6})
    assert value["pool_size"] == 5


def _info(fps: float = FPS, cameras: tuple[str, ...] = ("observation.images.top",)) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": [7]},
        "action": {"dtype": "float32", "shape": [7]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }
    for camera in cameras:
        features[camera] = {"dtype": "video", "shape": [48, 64, 3], "info": {"video.fps": fps}}
    return {
        "codebase_version": "v3.0",
        "fps": fps,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }


def _row(length: int, video_frames: int, **extra) -> dict:
    return {
        "episode_index": 0,
        "length": length,
        "videos/observation.images.top/from_timestamp": 0.0,
        "videos/observation.images.top/to_timestamp": video_frames / FPS,
        **extra,
    }


def test_exclusions_name_the_source_flags_and_array_faults() -> None:
    _, states = _episode(4.0)
    steps = np.abs(np.diff(states[:, :6], axis=0)).max(axis=1)
    info = _info()
    assert prepare_yam.exclusions(_row(101, 101), info, states, steps) == ["no_grasp"]
    grasped = states.copy()
    grasped[50:, 6] = 0.9
    assert prepare_yam.exclusions(_row(101, 101), info, grasped, steps) == []
    # The published video holds a different take than the parquet rows (yam-pick-place 15/24/37/39).
    assert prepare_yam.exclusions(_row(101, 321), info, grasped, steps) == ["video_data_mismatch"]
    assert prepare_yam.exclusions(_row(101, 101, episode_success=False), info, grasped, steps) == ["source_flagged_failure"]
    assert prepare_yam.exclusions(_row(101, 101, episode_success=True), info, grasped, steps) == []
    jump = steps.copy()
    jump[10] = 0.5
    assert prepare_yam.exclusions(_row(101, 101), info, grasped, jump) == ["discontinuous"]


def test_validate_verdicts_requires_every_nominee_and_explicit_booleans() -> None:
    candidates = {3: {}, 90: {}}
    with pytest.raises(ValueError, match="missing=\\[.90.\\]"):
        prepare_yam.validate_verdicts({"episodes": {"3": {"accept": True, "quality": 4}}}, candidates)
    with pytest.raises(ValueError, match="explicit boolean"):
        prepare_yam.validate_verdicts({"episodes": {"3": {"accept": "yes"}, "90": {"accept": False}}}, candidates)
    with pytest.raises(ValueError, match="outside 1..5"):
        prepare_yam.validate_verdicts({"episodes": {"3": {"accept": True, "quality": 7}, "90": {"accept": False}}}, candidates)
    with pytest.raises(ValueError, match="bad mistake event"):
        prepare_yam.validate_verdicts(
            {"episodes": {"3": {"accept": True, "quality": 3, "mistake_events": [{"kind": "wobble", "start_s": 1.0, "end_s": 2.0}]}, "90": {"accept": False}}},
            candidates,
        )
    accepted = prepare_yam.validate_verdicts(
        {"episodes": {"90": {"accept": True, "quality": 5}, "3": {"accept": False, "notes": "human hand in frame"}}}, candidates
    )
    assert accepted == [90]


def test_lead_ticks_recovers_a_known_command_lead() -> None:
    relative = np.arange(300) / 30.0
    command = np.sin(relative)
    measured = np.roll(command, 3)
    assert prepare_yam.lead_ticks(command, measured) == 3
    assert prepare_yam.lead_ticks(np.zeros(300), measured) is None


def test_gripper_transitions_count_crossings_whichever_way_is_closed() -> None:
    gripper = np.concatenate([np.zeros(20), np.full(20, 0.6), np.zeros(20), np.full(20, 0.6), np.zeros(20)])
    assert prepare_yam.gripper_transitions(gripper) == 4
    assert prepare_yam.gripper_transitions(1.0 - gripper) == 4
    assert prepare_yam.gripper_transitions(np.full(60, 0.3)) == 0


# ── gripper_transform at corpus ingest on a tiny synthetic v3 shard ────────────────────


def _synthetic_video(path: Path, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={int(FPS)}",
            "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _write_v3_source(root: Path, repo_id: str, states: np.ndarray, actions: np.ndarray) -> tuple[Path, Path]:
    """One-episode LeRobot v3 mirror under root/staging plus its metadata copy under root/metadata."""
    frames = len(states)
    staged = root / "staging" / repo_id.replace("/", "__")
    metadata = root / "metadata" / "yam_unit"
    info = _info()
    for base in (staged, metadata):
        (base / "meta").mkdir(parents=True, exist_ok=True)
        (base / "meta/info.json").write_text(json.dumps(info))
    timestamps = (np.arange(frames, dtype=np.float32) / np.float32(FPS)).astype(np.float32)
    data = pa.table(
        {
            "episode_index": pa.array(np.zeros(frames, dtype=np.int64)),
            "timestamp": pa.array(timestamps),
            "observation.state": pa.array(states.astype(np.float32).tolist(), type=pa.list_(pa.float32(), 7)),
            "action": pa.array(actions.astype(np.float32).tolist(), type=pa.list_(pa.float32(), 7)),
        }
    )
    (staged / "data/chunk-000").mkdir(parents=True)
    pq.write_table(data, staged / "data/chunk-000/file-000.parquet")
    episodes = pa.table(
        {
            "episode_index": [0],
            "tasks": [["unit task"]],
            "length": [frames],
            "dataset_from_index": [0],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "videos/observation.images.top/chunk_index": [0],
            "videos/observation.images.top/file_index": [0],
            "videos/observation.images.top/from_timestamp": [0.0],
            "videos/observation.images.top/to_timestamp": [frames / FPS],
        }
    )
    (metadata / "meta/episodes/chunk-000").mkdir(parents=True)
    pq.write_table(episodes, metadata / "meta/episodes/chunk-000/file-000.parquet")
    _synthetic_video(staged / "videos/observation.images.top/chunk-000/file-000.mp4", frames)
    return staged, metadata


def _yam_component(root: Path, transform: dict[str, str], states: np.ndarray, actions: np.ndarray):
    from lerobot.datasets.diverse_pilot import SourceSpec

    repo_id = "unit/yam-synthetic"
    staged, metadata = _write_v3_source(root, repo_id, states, actions)
    review = root / "review/unit"
    review.mkdir(parents=True)
    manifest = {
        "manifest_version": 1,
        "repo_id": repo_id,
        "revision": "deadbeef",
        "source_format": "lerobot_v3",
        "episodes": [
            {
                "episode_index": 0,
                "files": {
                    "data": "data/chunk-000/file-000.parquet",
                    "videos": {
                        "observation.images.top": {
                            "path": "videos/observation.images.top/chunk-000/file-000.mp4",
                            "from_timestamp": 0.0,
                            "to_timestamp": len(states) / FPS,
                        }
                    },
                },
            }
        ],
    }
    (review / "acquisition_manifest.json").write_text(json.dumps(manifest))
    (review / "selection.json").write_text(json.dumps({"accepted_episode_indices": [0]}))
    annotations = {
        "review_status": "validated",
        "task": "unit task",
        "episode_duration_s": (len(states) - 1) / FPS,
        "segments": [
            {"start_s": 0.0, "end_s": (len(states) - 1) / FPS, "retention": "keep", "retention_reason": "useful_motion", "subtask": "unit task", "quality": 4, "mistake_events": []}
        ],
    }
    (review / "episode_000000.annotations.json").write_text(json.dumps(annotations))
    spec = SourceSpec(
        name="yam_unit", repo_id=repo_id, revision="deadbeef", source_format="lerobot_v3", pilot_episodes=1,
        robot_type="yam", license="apache-2.0", real_robot_evidence="unit", state_fields=("observation.state",),
        action_fields=("action",), gripper_field="action[6]", state_semantics="unit", action_semantics="unit",
        gripper_semantics="unit", joint_names=(), metadata_patterns=("meta/**",), gripper_transform=transform,
    )
    return build_corpus.Component(
        source="yam", component="unit", embodiment="YAM", dataset_root=review, staging_root=root / "staging",
        metadata_root=metadata, spec=spec, video_origin="staged_v3", manifest_path=review / "acquisition_manifest.json",
        selection_path=review / "selection.json", annotations_root=review, review_round="v3",
    )


@pytest.mark.parametrize(
    "transform",
    [{"state": "identity", "action": "identity"}, {"state": "flip_0_1", "action": "identity"}, {"state": "flip_0_1", "action": "flip_0_1"}],
)
def test_ingest_applies_the_configured_gripper_transform_per_slot(tmp_path: Path, monkeypatch, transform: dict[str, str]) -> None:
    frames = 50
    relative = np.arange(frames) / FPS
    states = np.stack([*(0.1 * i + np.sin(relative) for i in range(6)), np.linspace(0.0, 0.7, frames)], axis=1)
    actions = np.stack([*(0.1 * i + np.sin(relative + 0.1) for i in range(6)), np.linspace(1.0, 0.3, frames)], axis=1)
    component = _yam_component(tmp_path, transform, states, actions)
    monkeypatch.setattr(build_corpus, "BUILD_ROOT", tmp_path)
    corpus = tmp_path / "corpus"
    record = build_corpus.ingest_episode(component, 0, "train", corpus, hash_cache={}, overwrite=False)
    assert record["gripper_transform"] == transform
    assert record["native_rate_hz"] == FPS and record["rate_provenance"] == "declared_by_the_source_dataset_info"
    stored_state = np.load(corpus / "episodes" / record["episode_id"] / "state.npy")
    stored_action = np.load(corpus / "episodes" / record["episode_id"] / "action.npy")
    expected_state = states.astype(np.float32).astype(np.float64)
    expected_action = actions.astype(np.float32).astype(np.float64)
    if transform["state"] == "flip_0_1":
        expected_state[:, 6] = 1.0 - expected_state[:, 6]
    if transform["action"] == "flip_0_1":
        expected_action[:, 6] = 1.0 - expected_action[:, 6]
    np.testing.assert_allclose(stored_state, expected_state, atol=1e-6)
    np.testing.assert_allclose(stored_action, expected_action, atol=1e-6)
    # Joints are never touched by the gripper transform.
    np.testing.assert_allclose(stored_state[:, :6], states[:, :6].astype(np.float32), atol=1e-6)


def test_twenty_five_hertz_float32_timestamps_pass_the_declared_rate_check_and_interpolate() -> None:
    from lerobot.datasets.diverse_pilot import sample_action_chunk

    frames = 251
    timestamps = (np.arange(frames, dtype=np.float32) / np.float32(FPS)).astype(np.float64)
    measured = 1.0 / float(np.median(np.diff(timestamps)))
    assert abs(FPS - measured) / FPS < 1e-3
    actions = np.stack([timestamps, np.cos(timestamps)], axis=1)
    chunk = sample_action_chunk(timestamps, actions, 6.0, native_rate_hz=FPS)
    # 25 Hz samples coincide with the 30 Hz grid only every 1/5 s, so at least 24 of 30 points
    # interpolate (the packer flags a coincidence only when float32 rounds onto or above the
    # grid point, so some of the six coincident points carry a ~1e-6 interpolation weight).
    assert not chunk.interpolated_mask[0]
    assert chunk.interpolated_mask.sum() >= 24
