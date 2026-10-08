import json
from pathlib import Path

import pytest

from lerobot.annotation import diverse_fix_normalizer as v1
from lerobot.annotation import diverse_fix_normalizer_v2 as normalizer


EPISODE = "episode"


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_surgical_operations_expand_to_full_episode_rows(tmp_path: Path) -> None:
    source = tmp_path / "root"
    original = {"episode_id": EPISODE, "from_index": 1, "to_index": 3, "quality": 3}
    replacement = {**original, "quality": 5}
    _write_rows(source / "fmb" / "quality_spans.jsonl", [original, {"episode_id": "other"}])
    spec = {"operations": {"quality_spans.jsonl": {"replace": [
        {"match": original, "replacement": replacement, "reason": "reviewed"}
    ]}}}

    rows = normalizer._expand_surgical(spec, source, "fmb", EPISODE, tmp_path / "fix.json")

    assert rows == {"quality_spans.jsonl": [replacement]}


def test_surgical_operations_fail_closed_on_unknown_instruction(tmp_path: Path) -> None:
    spec = {"operations": {"quality_spans.jsonl": {"splice": []}}}
    with pytest.raises(v1.NormalizationError, match="unsupported surgical keys"):
        normalizer._expand_surgical(spec, tmp_path, "fmb", EPISODE, tmp_path / "fix.json")


def test_legacy_speed_is_rekeyed_to_corrected_atom(tmp_path: Path) -> None:
    source = tmp_path / "root"
    old = {
        "episode_id": EPISODE, "parent_interval_index": 4, "atom_index": 0,
        "start_timestep": 10, "end_timestep_exclusive": 20, "subtask": "move block",
        "speed": 4, "speed_source": "duration_v4",
    }
    _write_rows(source / "corpus" / "speed_atoms.jsonl", [old])
    atom = {
        "episode_id": EPISODE, "parent_interval_index": 7, "atom_index": 2,
        "start_timestep": 12, "end_timestep_exclusive": 24, "subtask": "move block",
        "verb": "move", "native_rate_hz": 10.0,
    }

    rows = normalizer._derive_legacy_speed(source, "corpus", EPISODE, [atom], {(7, 2): 3})

    assert rows[0]["parent_interval_index"] == 7
    assert rows[0]["atom_index"] == 2
    assert rows[0]["end_timestep_exclusive"] == 24
    assert rows[0]["speed"] == 3
    assert rows[0]["speed_source"] == "round1_duration_prior"
