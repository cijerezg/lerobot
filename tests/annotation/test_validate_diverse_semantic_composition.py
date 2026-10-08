from __future__ import annotations

import json
from pathlib import Path

from lerobot.annotation import diverse_semantic_sweep_compiler as compiler
from lerobot.annotation import validate_diverse_semantic_composition as review


def test_current_reports_have_exact_once_decision_accounting() -> None:
    workspace = Path(__file__).resolve().parents[3]
    sweep = workspace / "migration" / "diverse_annotation_audit_2026-10-07" / "sweeps"
    reports = {family: json.loads((sweep / name).read_text()) for family, name in compiler.REPORTS.items()}
    edits = compiler.collect_edits(reports)
    accounting, errors = review._decision_accounting(reports, edits)
    assert errors == []
    assert sum(accounting["compiled_edits"].values()) == 3400
    assert accounting["confirmed_edit_records"] == 3194
    assert accounting["rejected_boundary_assertion_records"] == 206
    assert accounting["exact_duplicate_groups"] == []


def test_normalized_overlap_is_fail_closed_for_boundary_and_quality_conflicts() -> None:
    episode_id = "sample"
    specs = {
        episode_id: {
            "replacements": {
                "subtask_atoms.jsonl": [
                    {"parent_interval_index": 0, "atom_index": 0, "start_timestep": 0,
                     "end_timestep_exclusive": 12, "subtask": "grasp object"},
                    {"parent_interval_index": 0, "atom_index": 1, "start_timestep": 12,
                     "end_timestep_exclusive": 20, "subtask": "move object"},
                ],
                "quality_spans.jsonl": [
                    {"raw_from_index": 5, "raw_to_index": 10, "from_index": 0, "to_index": 12}
                ],
            }
        }
    }
    edits = [
        {"kind": "boundary", "family": "x", "episode_id": episode_id,
         "old_boundary_frame": 9, "new_boundary_frame": 11},
        {"kind": "quality_delete_selector", "family": "x", "episode_id": episode_id,
         "raw_from_index": 5, "raw_to_index": 10},
    ]
    rows, conflicts, unresolved = review._normalized_overlap(specs, edits)
    assert [row["status"] for row in rows] == ["conflict", "conflict"]
    assert len(conflicts) == 2
    assert unresolved == conflicts


def test_normalized_spec_validation_preserves_gap_authority(tmp_path: Path) -> None:
    spec = {
        "store": "corpus", "episode_id": "sample",
        "actor_anchors": {"mode": "remap", "retained_intervals": [[0, 5], [8, 12]]},
        "replacements": {
            "subtask_atoms.jsonl": [
                {"episode_id": "sample", "parent_interval_index": 0, "atom_index": 0,
                 "start_timestep": 0, "end_timestep_exclusive": 5, "native_rate_hz": 10.0},
                {"episode_id": "sample", "parent_interval_index": 1, "atom_index": 0,
                 "start_timestep": 8, "end_timestep_exclusive": 12, "native_rate_hz": 10.0},
            ]
        },
    }
    (tmp_path / "sample.json").write_text(json.dumps(spec))
    containment = {
        "episodes": [{"episode_id": "sample", "store": "corpus", "native_rate_hz": 10.0,
                      "retained_intervals": [[0, 5], [8, 12]]}]
    }
    specs, summary, errors = review._validate_normalized_specs(tmp_path, containment)
    assert set(specs) == {"sample"}
    assert summary["spec_count"] == 1
    assert errors == []


def test_rejected_boundary_conflict_requires_source_reversion() -> None:
    episode_id = "robochallenge__pick_out_the_green_blocks__ep000285"
    reports = {
        "droid": {"reviews": {"gripper_boundaries_before_settle": []}},
        "robochallenge": {"categories": {"gripper_boundaries_before_settle": {"decisions": [
            {"status": "rejected", "episode_id": episode_id, "parent_interval_index": 0,
             "left_atom_index": 2, "right_atom_index": 3, "current_boundary_frame": 484},
        ]}}},
        "yam": {"decisions": []},
    }
    specs = {episode_id: {"replacements": {"subtask_atoms.jsonl": [
        {"parent_interval_index": 0, "atom_index": 2, "start_timestep": 384,
         "end_timestep_exclusive": 490},
        {"parent_interval_index": 0, "atom_index": 3, "start_timestep": 490,
         "end_timestep_exclusive": 800},
    ]}}}
    rows, conflicts, unresolved = review._rejected_boundary_overlap(reports, specs)
    assert len(rows) == len(conflicts) == 1
    assert conflicts[0]["selector"]["expected_boundary"] == 484
    assert conflicts[0]["selector"]["normalized_boundary"] == 490
    assert conflicts[0]["resolution"]["precedence"] == "systematic_rejection_keeps_source"
    assert unresolved == []
