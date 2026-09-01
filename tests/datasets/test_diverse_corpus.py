"""Tests for the source-native corpus and its actor and critic views."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.diverse_corpus import DiverseCorpus, time_stratified_indices

BUILDER_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/dataset/diverse_robot_dataset/build_corpus.py"
)
_spec = importlib.util.spec_from_file_location("build_corpus", BUILDER_PATH)
build_corpus = importlib.util.module_from_spec(_spec)
sys.modules["build_corpus"] = build_corpus  # dataclasses resolve types through sys.modules
_spec.loader.exec_module(build_corpus)


def _annotations(segments: list[dict], duration_s: float, **extra) -> dict:
    return {
        "review_status": "validated",
        "episode_duration_s": duration_s,
        "segments": segments,
        **extra,
    }


def _keep(start: float, end: float, subtask: str, quality: int = 4, **extra) -> dict:
    return {
        "start_s": start,
        "end_s": end,
        "retention": "keep",
        "retention_reason": extra.pop("retention_reason", "useful_motion"),
        "subtask": subtask,
        "quality": quality,
        "mistake_events": extra.pop("mistake_events", []),
        **extra,
    }


def _reject(start: float, end: float, reason: str) -> dict:
    return {"start_s": start, "end_s": end, "retention": "reject", "retention_reason": reason}


def _write_episode(
    corpus_root: Path,
    episode_id: str,
    *,
    frames: int,
    fps: float,
    annotations: dict,
    split: str = "train",
    outcome: str = "success",
) -> dict:
    directory = corpus_root / "episodes" / episode_id
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = np.arange(frames, dtype=np.float64) / fps
    state = np.stack([timestamps, np.sin(timestamps)], axis=1)
    action = state.copy()
    np.save(directory / "timestamp_s.npy", timestamps)
    np.save(directory / "state.npy", state)
    np.save(directory / "action.npy", action)
    record = {
        "episode_id": episode_id,
        "source": "unit",
        "component": "test",
        "embodiment": "TestArm",
        "split": split,
        "task": "unit task",
        "frames": frames,
        "duration_s": float(timestamps[-1]),
        "native_rate_hz": fps,
        "state_dimension": 2,
        "action_dimension": 2,
        "action_source": "native",
        "source_episode_index": 0,
        "source_repo_id": "unit/test",
        "source_revision": "0" * 40,
        "robot_type": "TestArm",
        "state_semantics": "unit",
        "action_semantics": "unit",
        "gripper_semantics": "unit",
        "cameras": [],
        "annotations": annotations,
        "review": {
            "outcome": outcome,
            "quality_provenance": "human_reviewed",
            "review_provenance": "unit",
        },
    }
    (directory / "episode.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def test_static_gap_inside_one_subtask_becomes_a_pause_not_a_new_interval() -> None:
    annotations = _annotations(
        [
            _reject(0.0, 6.0, "static"),
            _keep(6.0, 10.0, "reach for the block"),
            _reject(10.0, 12.0, "static"),
            _keep(12.0, 20.0, "reach for the block"),
        ],
        20.0,
    )
    intervals = build_corpus._merge_subtask_intervals(annotations)
    assert len(intervals) == 1
    assert intervals[0]["start_s"] == 6.0
    assert intervals[0]["end_s"] == 20.0
    assert len(intervals[0]["pause_events"]) == 1
    assert intervals[0]["pause_events"][0]["reason"] == "static"


def test_static_gap_between_different_subtasks_does_not_merge_them() -> None:
    annotations = _annotations(
        [
            _keep(0.0, 8.0, "grasp the cup"),
            _reject(8.0, 10.0, "static"),
            _keep(10.0, 18.0, "place the cup"),
        ],
        18.0,
    )
    intervals = build_corpus._merge_subtask_intervals(annotations)
    assert [item["subtask"] for item in intervals] == ["grasp the cup", "place the cup"]
    assert all(not item["pause_events"] for item in intervals)


def test_non_pause_rejection_closes_the_interval_as_interrupted() -> None:
    annotations = _annotations(
        [
            _keep(0.0, 8.0, "open the drawer"),
            _reject(8.0, 9.0, "human_intervention"),
            _keep(9.0, 16.0, "open the drawer"),
        ],
        16.0,
    )
    intervals = build_corpus._merge_subtask_intervals(annotations)
    assert len(intervals) == 2
    assert intervals[0]["interruption_events"][0]["reason"] == "human_intervention"
    assert intervals[0]["end_boundary"] == "interruption:human_intervention"


def test_interrupted_intervals_are_not_critic_eligible(tmp_path: Path) -> None:
    annotations = _annotations(
        [
            _keep(0.0, 8.0, "open the drawer"),
            _reject(8.0, 9.0, "human_intervention"),
            _keep(9.0, 16.0, "open the drawer"),
        ],
        16.0,
    )
    record = _write_episode(tmp_path, "unit__test__ep000000", frames=481, fps=30.0, annotations=annotations)
    rows = build_corpus.critic_intervals(record, tmp_path)
    assert rows[0]["critic_eligible"] is False
    assert rows[0]["critic_rejection_reason"] == "truncated_by_reviewed_interruption"
    assert rows[1]["critic_eligible"] is True


def test_critic_interval_references_every_native_action_in_its_span(tmp_path: Path) -> None:
    annotations = _annotations([_keep(0.0, 16.0, "stack the blocks")], 16.0)
    record = _write_episode(tmp_path, "unit__test__ep000001", frames=481, fps=30.0, annotations=annotations)
    row = build_corpus.critic_intervals(record, tmp_path)[0]
    assert row["action_sequence_reference"]["contains_every_native_action"] is True
    assert row["native_action_samples"] == row["end_timestep_exclusive"] - row["start_timestep"]
    assert row["end_timestep_exclusive"] - row["start_timestep"] == 480


def test_single_subtask_episode_carries_the_reviewed_outcome_others_stay_unknown(
    tmp_path: Path,
) -> None:
    single = _annotations([_keep(0.0, 16.0, "stack the blocks")], 16.0)
    record = _write_episode(tmp_path, "unit__test__ep000002", frames=481, fps=30.0, annotations=single)
    assert build_corpus.critic_intervals(record, tmp_path)[0]["subtask_outcome"] == "success"

    multi = _annotations(
        [_keep(0.0, 8.0, "grasp the cup"), _keep(8.0, 16.0, "place the cup")], 16.0
    )
    record = _write_episode(tmp_path, "unit__test__ep000003", frames=481, fps=30.0, annotations=multi)
    outcomes = {row["subtask_outcome"] for row in build_corpus.critic_intervals(record, tmp_path)}
    assert outcomes == {"unknown"}


def test_actor_anchors_use_the_requested_stride_and_reviewed_eligibility(tmp_path: Path) -> None:
    annotations = _annotations(
        [
            _reject(0.0, 6.0, "static"),
            _keep(6.0, 14.0, "reach"),
            _reject(14.0, 16.0, "static"),
            _keep(16.0, 20.0, "place"),
        ],
        20.0,
    )
    record = _write_episode(tmp_path, "unit__test__ep000004", frames=601, fps=30.0, annotations=annotations)
    rows = build_corpus.actor_anchors(record, tmp_path, 0.2)
    anchors = [row["anchor_s"] for row in rows]
    assert anchors[0] == pytest.approx(6.0)
    assert np.allclose(np.diff(anchors), 0.2)
    assert anchors[-1] <= 20.0 - 29 / 30.0 + 1e-9
    retained = [row for row in rows if row["retained"]]
    assert all(len(row["history_frames"]) == 7 for row in rows)
    # An anchor whose one-second future lands in the static span is rejected, and the
    # six-second history reaching back into a static span is not a rejection by itself.
    assert any(row["anchor_s"] > 13.5 and not row["retained"] for row in rows)
    assert any(row["anchor_s"] > 16.0 and row["retained"] for row in rows)
    assert len(retained) < len(rows)


def test_denser_stride_multiplies_rows_over_the_same_timeline(tmp_path: Path) -> None:
    annotations = _annotations([_keep(0.0, 30.0, "wipe the table")], 30.0)
    record = _write_episode(tmp_path, "unit__test__ep000005", frames=901, fps=30.0, annotations=annotations)
    sparse = build_corpus.actor_anchors(record, tmp_path, 2.0)
    dense = build_corpus.actor_anchors(record, tmp_path, 0.2)
    assert len(dense) == pytest.approx(10 * len(sparse), rel=0.05)
    assert {row["episode_id"] for row in dense} == {row["episode_id"] for row in sparse}
    assert all(row["views_copy_source_arrays"] is False for row in dense)


def test_split_assignment_reserves_validation_and_test_episodes() -> None:
    ten = build_corpus.assign_splits(list(range(10)))
    assert sorted(ten.values()).count("train") == 8
    assert sorted(ten.values()).count("validation") == 1
    assert sorted(ten.values()).count("test") == 1
    four = build_corpus.assign_splits([3, 7, 11, 15])
    assert set(four.values()) == {"train", "validation", "test"}


def test_critic_subsampling_keeps_the_endpoints_and_reports_padding(tmp_path: Path) -> None:
    annotations = _annotations([_keep(0.0, 16.0, "stack the blocks")], 16.0)
    record = _write_episode(tmp_path, "unit__test__ep000006", frames=481, fps=30.0, annotations=annotations)
    build_corpus.refresh_episode_index(tmp_path)
    build_corpus.write_jsonl(
        tmp_path / "critic_intervals.jsonl", build_corpus.critic_intervals(record, tmp_path)
    )
    build_corpus.write_jsonl(tmp_path / "actor_anchors_5hz.jsonl", [])
    corpus = DiverseCorpus(tmp_path)
    row = corpus.critic_intervals()[0]
    sample = corpus.critic_sample(row, action_points=16)
    assert sample["native_action_samples"] == 480
    assert sample["action"].shape[0] == 480
    assert sample["action.subsampled"].shape == (16, 2)
    assert sample["action.subsampled_mask"].all()
    assert sample["action.subsampled_indices"][0] == 0
    assert sample["action.subsampled_indices"][-1] == 479


def test_time_stratified_indices_pad_short_sequences() -> None:
    timestamps = np.arange(5) / 30.0
    actions = np.zeros((5, 2))
    indices, mask = time_stratified_indices(timestamps, actions, 8)
    assert list(indices) == [0, 1, 2, 3, 4]
    assert mask.tolist() == [True] * 5 + [False] * 3


def test_trailing_source_excluded_span_is_a_clip_boundary_not_an_interruption(
    tmp_path: Path,
) -> None:
    annotations = _annotations(
        [
            _keep(0.0, 15.0, "pack the box"),
            _reject(15.0, 16.0, "source_excluded"),
        ],
        16.0,
    )
    record = _write_episode(
        tmp_path, "unit__test__ep000007", frames=481, fps=30.0, annotations=annotations
    )
    row = build_corpus.critic_intervals(record, tmp_path)[0]
    assert row["classification"] == "complete_to_clip_boundary"
    assert row["critic_eligible"] is True
    assert row["end_boundary"] == "source_keep_range_end"
    assert row["end_boundary_provenance"] == "source_native_keep_range"


def test_interior_source_excluded_span_still_truncates_the_interval(tmp_path: Path) -> None:
    annotations = _annotations(
        [
            _keep(0.0, 8.0, "pack the box"),
            _reject(8.0, 9.0, "source_excluded"),
            _keep(9.0, 16.0, "pack the box"),
        ],
        16.0,
    )
    record = _write_episode(
        tmp_path, "unit__test__ep000008", frames=481, fps=30.0, annotations=annotations
    )
    rows = build_corpus.critic_intervals(record, tmp_path)
    assert rows[0]["critic_eligible"] is False
    assert rows[0]["critic_rejection_reason"] == "truncated_by_source_range"
