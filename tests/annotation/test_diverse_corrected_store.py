from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.annotation.diverse_corrected_store import StoreError, load_corrections, remap_actor_rows


def _write_spec(path: Path, **updates) -> None:
    value = {
        "schema_version": 1,
        "status": "resolved",
        "store": "corpus",
        "episode_id": "ep",
        "replacements": {"quality_spans.jsonl": []},
        "unresolved": [],
        "second_read_required": False,
    }
    value.update(updates)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loader_refuses_unresolved_and_duplicate_specs(tmp_path: Path) -> None:
    _write_spec(tmp_path / "a.json", status="open")
    with pytest.raises(StoreError, match="status must be 'resolved'"):
        load_corrections(tmp_path)
    (tmp_path / "a.json").unlink()

    _write_spec(tmp_path / "a.json")
    _write_spec(tmp_path / "b.json")
    with pytest.raises(StoreError, match="duplicate correction spec"):
        load_corrections(tmp_path)


def test_atom_change_requires_aligned_channels_and_index_rebuild(tmp_path: Path) -> None:
    atom = {
        "episode_id": "ep",
        "parent_interval_index": 0,
        "atom_index": 0,
        "start_timestep": 0,
        "end_timestep_exclusive": 30,
        "subtask": "grasp the object",
    }
    _write_spec(tmp_path / "fix.json", replacements={"subtask_atoms.jsonl": [atom]})
    with pytest.raises(StoreError, match="also requires full replacements"):
        load_corrections(tmp_path)


def test_actor_remap_excludes_history_or_future_crossing_gap() -> None:
    atom = {
        "episode_id": "ep",
        "parent_interval_index": 0,
        "atom_index": 0,
        "start_timestep": 5,
        "end_timestep_exclusive": 30,
        "subtask": "grasp the object",
        "quality": 4,
        "quality_provenance": "audit",
        "mistake_events": [],
    }
    base = {
        "episode_id": "ep",
        "native_rate_hz": 10.0,
        "anchor_s": 1.0,
        "anchor_frame": 10,
        "future_end_s": 1.2,
        "history_frames": [4, 5, 6, 7, 8, 9, 10],
        "retained": True,
        "subtask": "old",
        "quality": 3,
        "mistake": False,
    }
    inside = {
        **base,
        "anchor_s": 2.0,
        "anchor_frame": 20,
        "future_end_s": 2.2,
        "history_frames": [14, 15, 16, 17, 18, 19, 20],
    }
    rows = remap_actor_rows([base, inside], "ep", [atom], [(5, 30)], "corpus", np.arange(31) / 10.0)
    assert rows[0]["retained"] is False
    assert rows[0]["retention_reason"] == "audit_excluded"
    assert rows[1]["retained"] is True
    assert rows[1]["subtask"] == "grasp the object"
