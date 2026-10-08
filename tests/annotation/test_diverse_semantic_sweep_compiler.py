import json
from pathlib import Path

import pytest

from lerobot.annotation import diverse_semantic_sweep_compiler as compiler


def test_open_count_understands_each_family_schema() -> None:
    assert compiler._open_count("droid", {"counts": {"open": 8}}) == 8
    assert compiler._open_count("droid_success", {"counts": {"open": 0}}) == 0
    assert compiler._open_count("fmb", {"counts": {"open": 2}}) == 2
    assert compiler._open_count("molmoact", {"summary": {"open_items": 3}}) == 3
    assert compiler._open_count(
        "robochallenge",
        {"status": "complete", "second_reader_resolution": {"status": "complete"},
         "categories": {"x": {"decisions": [{"status": "confirmed", "open_detail": None}]}}},
    ) == 0
    assert compiler._open_count("ur7e", {"opens": [], "summary_counts": {"open_items": 0}}) == 0
    assert compiler._open_count("yam", {"decisions": [{"open": False}, {"open": True}]}) == 1


def test_load_gate_refuses_any_open_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sweep = tmp_path / "sweeps"
    sweep.mkdir()
    (sweep / "droid.json").write_text(json.dumps({"counts": {"open": 1}}))
    containment = tmp_path / "containment.json"
    containment.write_text(json.dumps({"open_candidates": [], "validation": {"open_candidates": 0}}))
    monkeypatch.setattr(compiler, "REPORTS", {"droid": "droid.json"})
    with pytest.raises(compiler.SweepCompileError, match="open audit decisions remain"):
        compiler.load_and_validate(tmp_path / "source", sweep, containment)


def test_all_current_family_adapters_emit_only_typed_edits() -> None:
    workspace = Path(__file__).resolve().parents[3]
    sweep = workspace / "migration" / "diverse_annotation_audit_2026-10-07" / "sweeps"
    reports = {family: json.loads((sweep / filename).read_text()) for family, filename in compiler.REPORTS.items()}
    edits = compiler.collect_edits(reports)
    assert len(edits) == 3_400
    assert sum(row["kind"] == "quality_restore_full_pause" for row in edits) == 141
    assert sum(bool(row.get("rejected_candidate_reversion") or row.get("rejected_candidate_assertion"))
               for row in edits) == 206
    assert len({row["episode_id"] for row in edits}) > 700
    assert all(isinstance(row.get("kind"), str) and isinstance(row.get("episode_id"), str) for row in edits)
    assert {"boundary", "text", "quality_delete_exact", "assert_structural_gap"}.issubset(
        {row["kind"] for row in edits}
    )


def test_full_pause_restoration_coalesces_duplicate_semantic_identity() -> None:
    episode = "sample"
    source_rows = [
        {"episode_id": episode, "uid": "sample_p0a0", "parent_interval_index": 0,
         "atom_index": 0, "quality": 3, "cause": "hover", "native_rate_hz": 2.0,
         "from_index": 0, "raw_from_index": 0, "raw_to_index": 2, "to_index": 3,
         "note": "first"},
        {"episode_id": episode, "uid": "sample_p0a0", "parent_interval_index": 0,
         "atom_index": 0, "quality": 3, "cause": "hover", "native_rate_hz": 2.0,
         "from_index": 3, "raw_from_index": 4, "raw_to_index": 6, "to_index": 7,
         "note": "second"},
    ]
    rows = [dict(row, _source_line=index) for index, row in enumerate(source_rows, 1)]
    exact = {"from_index": 0, "raw_from_index": 2, "raw_to_index": 4, "to_index": 5}
    edits = [
        {"kind": "quality_restore_full_pause", "uid": "sample_p0a0",
         "parent_interval_index": 0, "atom_index": 0, "quality": 3,
         "current": {key: row[key] for key in
                     ("from_index", "raw_from_index", "raw_to_index", "to_index")},
         "replacement_rows": [exact]}
        for row in source_rows
    ]
    atoms = [{"episode_id": episode, "parent_interval_index": 0, "atom_index": 0,
              "start_timestep": 0, "end_timestep_exclusive": 10, "native_rate_hz": 2.0,
              "_source_atom_key": [episode, 0, 0]}]
    ledger: list[dict] = []
    result = compiler._apply_quality_edits(
        rows, edits, episode, ledger, source_rows=source_rows,
        retained_intervals=[[0, 10]], atoms=atoms,
    )
    restored = [row for row in result if row["raw_from_index"] == 2 and row["raw_to_index"] == 4]
    assert len(restored) == 1
    assert restored[0]["note"] == "first"
    assert sum(row.get("resolution", "").startswith("coalesced_duplicate") for row in ledger) == 1


def test_fmb_retention_trim_updates_reviewed_primitive_authority(tmp_path: Path) -> None:
    episode_dir = tmp_path / "fmb" / "episodes" / "ep"
    episode_dir.mkdir(parents=True)
    source = {
        "primitive_intervals": [
            {"start_timestep": 0, "end_timestep_exclusive": 4,
             "reviewed_start_timestep": 0, "reviewed_end_timestep_exclusive": 4},
            {"start_timestep": 4, "end_timestep_exclusive": 10,
             "reviewed_start_timestep": 4, "reviewed_end_timestep_exclusive": 10},
        ]
    }
    (episode_dir / "episode.json").write_text(json.dumps(source))

    record = compiler._fmb_episode_record_for_retention(tmp_path, "ep", [[0, 8]])

    assert record is not None
    assert record["primitive_intervals"][1]["end_timestep_exclusive"] == 10
    assert record["primitive_intervals"][1]["reviewed_end_timestep_exclusive"] == 8
