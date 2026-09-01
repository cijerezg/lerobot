from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "examples/dataset/diverse_robot_dataset/audit_fmb.py"
SPEC = importlib.util.spec_from_file_location("audit_fmb", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_fmb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_fmb)


def test_primitive_intervals_preserve_source_runs() -> None:
    intervals = audit_fmb.primitive_intervals(["grasp", "grasp", "rotate", "grasp", "grasp"], fps=10.0)

    assert [
        (item["primitive"], item["start_timestep"], item["end_timestep_exclusive"]) for item in intervals
    ] == [
        ("grasp", 0, 2),
        ("rotate", 2, 3),
        ("grasp", 3, 5),
    ]
    assert intervals[1]["one_frame_flicker"] is True
    assert all(item["critic_eligibility"] == "pending_visual_review" for item in intervals)


def test_actor_yield_requires_history_and_future_support() -> None:
    labels = ["grasp"] * 80
    result = audit_fmb.actor_yield(80, labels, fps=10.0)

    assert result["original_2s"]["episode_level_candidate_anchors"] == 1
    assert result["proposed_5hz"]["episode_level_candidate_anchors"] == 5
    assert result["proposed_10hz"]["episode_level_candidate_anchors"] == 10
    assert result["proposed_10hz"]["primitive_conditioned_future_candidate_anchors"] == 10


def test_depth_summary_keeps_units_unproven() -> None:
    depth = np.full((3, 16, 16), 0.4, dtype=np.float32)
    depth[1, :2, :2] = 0

    result = audit_fmb.depth_summary(depth, fps=10.0)

    assert result["finite_fraction_exact"] == 1.0
    assert result["zero_fraction_exact"] > 0
    assert result["physical_units"]["status"] == "unproven"
    assert result["physical_units"]["numeric_hypothesis"] == "values_are_compatible_with_meters"


def test_alignment_summary_rejects_shape_mismatch() -> None:
    rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    depth = np.zeros((2, 7, 8), dtype=np.float32)

    assert audit_fmb.alignment_summary(rgb, depth)["shape_aligned"] is False


def test_integer_depth_uses_explicit_d405_scale_assumption() -> None:
    depth = np.full((2, 16, 16), 2500, dtype=np.uint16)

    result = audit_fmb.depth_summary(depth, fps=10.0)

    assert result["physical_units"]["status"] == "assumed_for_conversion"
    assert result["physical_units"]["depth_units_mm_per_level"] == 0.1


def test_paired_wrist_depth_detects_distinct_streams() -> None:
    wrist_1 = np.full((2, 16, 16), 2500, dtype=np.uint16)
    wrist_2 = wrist_1.copy()
    wrist_2[:, 4:12, 4:12] += 100

    result = audit_fmb.paired_wrist_depth_summary(wrist_1, wrist_2)

    assert result["arrays_exactly_equal"] is False
    assert result["conclusion"] == "distinct_physical_streams_not_duplicates"
    assert result["jointly_valid_exact_equal_fraction"] == 0.75
