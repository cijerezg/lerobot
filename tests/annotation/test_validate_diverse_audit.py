from __future__ import annotations

import numpy as np

from lerobot.annotation.validate_diverse_audit import (
    classify_containment,
    episode_authority_intervals,
    effective_frame_labels,
    expected_precision_bounds,
    expected_quality_bounds,
    half_second_frames,
    merge_intervals,
)


def test_merge_intervals_preserves_excluded_gaps() -> None:
    assert merge_intervals([(20, 30), (0, 10), (10, 15), (40, 50), (25, 35)]) == [
        (0, 15),
        (20, 35),
        (40, 50),
    ]


def test_episode_authority_uses_common_keep_segments_and_fmb_reviewed_primitives() -> None:
    timestamps = np.arange(10, dtype=np.float64) / 10
    common = {
        "annotations": {
            "segments": [
                {"start_s": 0.0, "end_s": 0.3, "retention": "keep"},
                {"start_s": 0.3, "end_s": 0.6, "retention": "reject"},
                {"start_s": 0.6, "end_s": 1.0, "retention": "keep"},
            ]
        }
    }
    assert episode_authority_intervals("corpus", common, timestamps) == [(0, 3), (6, 10)]

    fmb = {
        "primitive_intervals": [
            {
                "start_timestep": 0,
                "end_timestep_exclusive": 4,
                "reviewed_start_timestep": 1,
                "reviewed_end_timestep_exclusive": 3,
            },
            {"start_timestep": 4, "end_timestep_exclusive": 8},
        ]
    }
    assert episode_authority_intervals("fmb", fmb, timestamps) == [(1, 3), (4, 8)]


def test_containment_candidates_separate_padding_semantics_and_gaps() -> None:
    retained = [(10, 20), (30, 40)]
    assert classify_containment(retained, (8, 15), (10, 15)) == {
        "containment_class": "padding_only",
        "boundary_class": "outer_edge",
    }
    assert classify_containment(retained, (15, 35), (15, 35)) == {
        "containment_class": "semantic_gap",
        "boundary_class": "internal_gap",
    }
    assert classify_containment(retained, (3, 8), (3, 8)) == {
        "containment_class": "semantic_outside_retention",
        "boundary_class": "outer_edge",
    }
    assert classify_containment(retained, (11, 18), (11, 18)) is None


def test_quality_headroom_is_applied_once_and_clipped_to_one_interval() -> None:
    row = {
        "raw_from_index": 12,
        "raw_to_index": 18,
        "quality": 3,
        "native_rate_hz": 15.0,
    }
    assert half_second_frames(15.0) == 8
    assert half_second_frames(25.0) == 13
    assert expected_quality_bounds(row, [(10, 20), (30, 40)]) == (10, 20)
    row["quality"] = 5
    assert expected_quality_bounds(row, [(10, 20), (30, 40)]) == (12, 18)


def test_precision_headroom_is_clipped_without_moving_the_annotated_end() -> None:
    row = {
        "raw_from_index": 33,
        "to_index": 39,
        "native_rate_hz": 10.0,
    }
    assert expected_precision_bounds(row, [(10, 20), (30, 40)]) == (30, 39)
    row["to_index"] = 41
    assert expected_precision_bounds(row, [(10, 20), (30, 40)]) is None


def test_effective_labels_match_the_training_precedence_rules() -> None:
    quality = [
        {"from_index": 10, "to_index": 20, "quality": 5},
        {"from_index": 12, "to_index": 18, "quality": 3},
        {"from_index": 14, "to_index": 16, "quality": 2},
    ]
    mistakes = [{"from_index": 15, "to_index": 17}]
    precision = [
        {"from_index": 10, "to_index": 20, "precision": 3},
        {"from_index": 14, "to_index": 16, "precision": 5},
    ]
    assert effective_frame_labels(9, quality, mistakes, precision) == (4, False, 1)
    assert effective_frame_labels(13, quality, mistakes, precision) == (3, False, 3)
    assert effective_frame_labels(15, quality, mistakes, precision) == (2, True, 5)
