from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.annotation.diverse_fix_normalizer import NormalizationError, normalize_all, normalize_one


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _base(episode_id: str = "ep") -> dict:
    return {
        "schema_version": "diverse_annotation_audit_fix_v1",
        "episode_id": episode_id,
        "store": "corpus",
        "retained_intervals": [[0, 30]],
        "operations": {},
    }


def test_normalizes_explicit_complete_episode_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    episode = source / "corpus" / "episodes" / "ep"
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text("{}", encoding="utf-8")
    spec = _base()
    for name in (
        "subtask_atoms.jsonl",
        "contact_atoms.jsonl",
        "precision_atoms.jsonl",
        "speed_atoms.jsonl",
        "speed_atoms_hybrid_v1.jsonl",
        "quality_spans.jsonl",
        "mistakes_v2.jsonl",
        "precision_windows.jsonl",
    ):
        spec["operations"][name] = {"replace_complete_episode_rows": []}
    spec["operations"]["critic_intervals.jsonl"] = {"replace_complete_episode_rows": []}
    spec["operations"]["actor_anchors_5hz.jsonl"] = {"rebuild_complete_episode_rows": True}
    path = tmp_path / "spec.json"
    _write(path, spec)

    normalized = normalize_one(path, source)
    assert normalized["schema_version"] == 1
    assert normalized["status"] == "resolved"
    assert normalized["actor_anchors"] == {"mode": "remap", "retained_intervals": [[0, 30]]}
    assert normalized["critic_intervals"] == {"mode": "replace", "rows": []}
    assert set(normalized["replacements"]) == {
        "subtask_atoms.jsonl",
        "contact_atoms.jsonl",
        "precision_atoms.jsonl",
        "speed_atoms.jsonl",
        "speed_atoms_hybrid_v1.jsonl",
        "quality_spans.jsonl",
        "mistakes_v2.jsonl",
        "precision_windows.jsonl",
    }


def test_refuses_surgical_and_shorthand_variants(tmp_path: Path) -> None:
    surgical = _base("surgical")
    surgical["operations"] = {"quality_spans.jsonl": {"add": [{"episode_id": "surgical"}]}}
    path = tmp_path / "surgical.json"
    _write(path, surgical)
    with pytest.raises(NormalizationError, match="surgical operation"):
        normalize_one(path, tmp_path)

    shorthand = _base("shorthand")
    shorthand["operations"] = {"contact_atoms.jsonl": {"replace_episode_rows": [[0, 0, 0, 30, 1]]}}
    path = tmp_path / "shorthand.json"
    _write(path, shorthand)
    with pytest.raises(NormalizationError, match="must be JSON objects"):
        normalize_one(path, tmp_path)


def test_batch_dry_run_reports_refused_specs_without_writing(tmp_path: Path) -> None:
    fixes = tmp_path / "fixes"
    fixes.mkdir()
    valid = _base("valid")
    valid["operations"] = {"quality_spans.jsonl": {"replace_complete_episode_rows": []}}
    _write(fixes / "valid.json", valid)
    invalid = _base("invalid")
    invalid["operations"] = {"quality_spans.jsonl": {"remove": []}}
    _write(fixes / "invalid.json", invalid)

    normalized, refused = normalize_all(tmp_path, fixes)
    assert [path.name for path, _ in normalized] == ["valid.json"]
    assert len(refused) == 1
    assert refused[0]["source_spec"].endswith("invalid.json")
