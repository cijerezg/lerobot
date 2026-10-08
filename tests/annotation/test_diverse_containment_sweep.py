from __future__ import annotations

import json

import numpy as np

from lerobot.annotation import diverse_containment_sweep
from lerobot.annotation.diverse_containment_sweep import (
    actor_window_decision,
    operation_name,
    remap_v2_row,
    source_retained_intervals,
    subtract_intervals,
)


def _atom(source: tuple[str, int, int], target: tuple[str, int, int], start: int, stop: int) -> dict:
    return {
        "source_atom_key": list(source),
        "target_atom_key": list(target),
        "start_timestep": start,
        "end_timestep_exclusive": stop,
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_source_authority_preserves_common_gaps_and_all_fmb_primitives() -> None:
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
    assert source_retained_intervals("corpus", common, timestamps) == [(0, 3), (6, 10)]

    fmb = {
        "primitive_intervals": [
            {"start_timestep": 0, "end_timestep_exclusive": 3, "critic_eligible": True},
            {"start_timestep": 3, "end_timestep_exclusive": 7, "critic_eligible": False},
        ]
    }
    assert source_retained_intervals("fmb", fmb, timestamps) == [(0, 7)]


def test_staged_atoms_cannot_replace_episode_retention_authority(tmp_path, monkeypatch) -> None:
    """A staged atom spanning a source gap must be split, not treated as authority."""
    monkeypatch.setattr(diverse_containment_sweep, "STORES", ("corpus",))
    episode_id = "droid__staged"
    store = tmp_path / "corpus"
    episode_dir = store / "episodes" / episode_id
    episode_dir.mkdir(parents=True)

    _write_jsonl(
        store / "episodes.jsonl",
        [
            {
                "episode_id": episode_id,
                "source": "droid",
                "split": "train",
                "frames": 10,
                "native_rate_hz": 10.0,
                "directory": f"episodes/{episode_id}",
            }
        ],
    )
    _write_jsonl(
        store / "critic_intervals.jsonl",
        [
            {
                "episode_id": episode_id,
                "interval_index": 0,
                "start_timestep": 0,
                "end_timestep_exclusive": 10,
            }
        ],
    )
    atom = {
        "episode_id": episode_id,
        "parent_interval_index": 0,
        "atom_index": 0,
        "start_timestep": 0,
        "end_timestep_exclusive": 10,
        "subtask": "move",
    }
    _write_jsonl(store / "subtask_atoms.jsonl", [atom])
    for name in ("contact_atoms.jsonl", "speed_atoms_hybrid_v1.jsonl", "precision_atoms.jsonl"):
        atom_key = {key: atom[key] for key in ("episode_id", "parent_interval_index", "atom_index")}
        _write_jsonl(store / name, [atom_key])
    empty_sidecars = (
        "quality_spans.jsonl",
        "mistakes_v2.jsonl",
        "precision_windows.jsonl",
        "actor_anchors_5hz.jsonl",
    )
    for name in empty_sidecars:
        _write_jsonl(store / name, [])

    (episode_dir / "episode.json").write_text(
        json.dumps(
            {
                "annotations": {
                    "segments": [
                        {"start_s": 0.0, "end_s": 0.3, "retention": "keep"},
                        {"start_s": 0.3, "end_s": 0.6, "retention": "reject"},
                        {"start_s": 0.6, "end_s": 1.0, "retention": "keep"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    np.save(episode_dir / "timestamp_s.npy", np.arange(10, dtype=np.float64) / 10)

    manifest = diverse_containment_sweep.build_manifest(tmp_path)

    assert manifest["episodes"][0]["retained_intervals"] == [[0, 3], [6, 10]]
    assert manifest["episodes"][0]["excluded_internal_gaps"] == [[3, 6]]
    assert [(row["start_timestep"], row["end_timestep_exclusive"]) for row in manifest["target_parents"]] == [
        (0, 3),
        (6, 10),
    ]
    assert manifest["atom_operations"] == [
        {
            "store": "corpus",
            "source_line": 1,
            "source_atom_key": [episode_id, 0, 0],
            "source_range": [0, 10],
            "action": "split",
            "targets": [
                {
                    "target_atom_key": [episode_id, 0, 0],
                    "start_timestep": 0,
                    "end_timestep_exclusive": 3,
                    "subtask": "move",
                    "inherit_per_atom_sidecars_from": [episode_id, 0, 0],
                },
                {
                    "target_atom_key": [episode_id, 1, 0],
                    "start_timestep": 6,
                    "end_timestep_exclusive": 10,
                    "subtask": "move",
                    "inherit_per_atom_sidecars_from": [episode_id, 0, 0],
                },
            ],
        }
    ]


def test_interval_actions_and_subtraction() -> None:
    assert subtract_intervals([(0, 10)], [(0, 3), (6, 10)]) == [(3, 6)]
    assert operation_name((0, 10), [(0, 3), (6, 10)]) == "split"
    assert operation_name((0, 10), [(0, 3)]) == "clip"
    assert operation_name((0, 10), []) == "drop"
    assert operation_name((0, 10), [(0, 10)], rekeyed=True) == "rekey"


def test_v2_rows_split_on_gap_and_rematerialize_fields() -> None:
    episode_id = "yam__espresso86"
    source_key = (episode_id, 0, 0)
    atoms = [
        _atom(source_key, (episode_id, 0, 0), 0, 10),
        _atom(source_key, (episode_id, 1, 0), 20, 30),
    ]
    retained = [(0, 10), (20, 30)]
    base = {
        "episode_id": episode_id,
        "parent_interval_index": 0,
        "atom_index": 0,
        "uid": "espresso86_p0a0",
        "native_rate_hz": 10.0,
    }

    quality = remap_v2_row(
        "quality_spans.jsonl",
        {**base, "raw_from_index": 8, "raw_to_index": 22, "from_index": 0, "to_index": 27, "quality": 2},
        retained,
        atoms,
    )
    assert [(x["fields"]["raw_from_index"], x["fields"]["raw_to_index"]) for x in quality] == [
        (8, 10),
        (20, 22),
    ]
    assert [(x["fields"]["from_index"], x["fields"]["to_index"]) for x in quality] == [
        (0, 10),
        (20, 27),
    ]
    assert [x["fields"]["quality"] for x in quality] == [2, 2]
    assert [x["target_atom_key"] for x in quality] == [
        [episode_id, 0, 0],
        [episode_id, 1, 0],
    ]

    precision = remap_v2_row(
        "precision_windows.jsonl",
        {**base, "raw_from_index": 8, "from_index": 0, "to_index": 22, "commit_index": 15, "precision": 3},
        retained,
        atoms,
    )
    assert [x["fields"]["commit_index"] for x in precision] == [9, 20]
    assert [x["fields"]["precision"] for x in precision] == [3, 3]

    mistake = remap_v2_row(
        "mistakes_v2.jsonl",
        {**base, "from_index": 8, "to_index": 22, "mistake": True, "mistake_type": "slip"},
        retained,
        atoms,
    )
    assert [(x["fields"]["from_index"], x["fields"]["to_index"]) for x in mistake] == [(8, 10), (20, 22)]
    assert all(x["fields"]["mistake"] for x in mistake)


def test_actor_windows_must_stay_inside_anchor_interval() -> None:
    timestamps = np.arange(40, dtype=np.float64) / 10
    retained = [(0, 10), (20, 40)]
    base = {"anchor_frame": 22, "history_frames": [20, 21, 22], "future_end_s": 2.8}
    assert actor_window_decision("corpus", base, retained, timestamps) == (True, [], 28)

    keep, reasons, endpoint = actor_window_decision(
        "corpus", {**base, "history_frames": [9, 20, 22]}, retained, timestamps
    )
    assert not keep and reasons == ["history_crosses_source_retention"] and endpoint == 28

    keep, reasons, endpoint = actor_window_decision(
        "corpus", {**base, "future_end_s": 4.0}, retained, timestamps
    )
    assert not keep and reasons == ["future_crosses_source_retention"] and endpoint == 40

    assert actor_window_decision(
        "corpus", {**base, "anchor_frame": 35, "history_frames": [33, 34, 35], "future_end_s": 3.9000005}, retained, timestamps
    ) == (False, ["future_crosses_source_retention"], 40)
