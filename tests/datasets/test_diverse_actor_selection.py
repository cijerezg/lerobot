"""Tests for the diverse actor training selection (integration plan phase A).

The pure-function tests run everywhere. The selection tests read the real federated
corpus and skip when it is not present, matching test_diverse_corpus.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.datasets.diverse_corpus import CONTACT_ATOMS_VIEW, PRECISION_ATOMS_VIEW
from lerobot.datasets.diverse_actor_selection import (
    ACTION_LAYOUTS,
    CANONICAL_CAMERA_ROLES,
    EXPECTED_ANCHORS,
    EXPECTED_ANCHORS_BY_SOURCE,
    EXPECTED_EPISODES,
    holdout_episode_ids,
    PACKED_CURRENT_SLOT,
    PACKED_OBSERVATIONS,
    action_layout_for,
    audit_selection,
    camera_roles_for,
    check_sample_contract,
    open_federated_corpus,
    packed_history_slots,
    packed_slot_for_age,
    select_actor_anchors,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "outputs/diverse_robot_dataset_v3"


def _corpus():
    if not (DATA_ROOT / "corpus" / "episodes.jsonl").is_file():
        pytest.skip("federated corpora not present")
    return open_federated_corpus(DATA_ROOT)


def _selection():
    return select_actor_anchors(_corpus())


# ── Packed observation window ────────────────────────────────────────────────


def test_history_ages_select_the_packed_slots_they_name() -> None:
    assert packed_history_slots([6.0, 4.0, 2.0]) == [0, 2, 4]
    # memory.history_offsets_seconds is written negative; it means the same instants.
    assert packed_history_slots([-6.0, -4.0, -2.0]) == [0, 2, 4]
    assert packed_slot_for_age(0.0) == PACKED_CURRENT_SLOT
    assert PACKED_OBSERVATIONS == 7


def test_an_age_the_corpus_never_packed_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of the packed observations"):
        packed_slot_for_age(1.5)


def test_history_may_not_claim_the_current_observation() -> None:
    with pytest.raises(ValueError, match="current observation"):
        packed_history_slots([4.0, 2.0, 0.0])


# ── Camera roles ─────────────────────────────────────────────────────────────


def test_two_and_three_camera_sources_map_onto_canonical_roles() -> None:
    assert camera_roles_for("droid", ["left_external", "right_external", "wrist"]) == {
        "external_0": "left_external",
        "external_1": "right_external",
        "wrist_0": "wrist",
    }
    # UR5 records no second external view: the role is absent, never filled in.
    assert camera_roles_for("robochallenge", ["global", "wrist"]) == {
        "external_0": "global",
        "wrist_0": "wrist",
    }
    assert set(camera_roles_for("ur7e", ["realsense_topview", "realsense_wrist"])) <= set(
        CANONICAL_CAMERA_ROLES
    )


def test_an_unmapped_camera_is_a_hard_error() -> None:
    with pytest.raises(KeyError, match="unmapped"):
        camera_roles_for("droid", ["left_external", "overhead_2"])


# ── Action layouts ───────────────────────────────────────────────────────────


def test_layout_indices_are_unique_and_dense() -> None:
    indices = [layout.index for layout in ACTION_LAYOUTS]
    assert indices == list(range(len(ACTION_LAYOUTS)))
    assert len({layout.name for layout in ACTION_LAYOUTS}) == len(ACTION_LAYOUTS)


def test_two_franka_sources_do_not_share_one_layout() -> None:
    droid = action_layout_for("droid", "Franka")
    fmb = action_layout_for("fmb", "Franka")
    assert droid.index != fmb.index
    assert droid.dim == fmb.dim == 8
    assert (droid.action_source, fmb.action_source) == ("native", "copy_state")


def test_an_unregistered_convention_refuses_to_guess() -> None:
    with pytest.raises(KeyError, match="no action layout registered"):
        action_layout_for("droid", "UR5")


# ── Selection over the real corpus ───────────────────────────────────────────


def test_selection_is_the_whole_accepted_corpus() -> None:
    selection = _selection()
    assert len(selection.episode_ids) == EXPECTED_EPISODES
    assert len(selection.rows) == EXPECTED_ANCHORS
    per_source = audit_selection(selection)
    assert {name: audit.anchors for name, audit in per_source.items()} == EXPECTED_ANCHORS_BY_SOURCE


def test_holdout_episodes_are_absent_and_accounted_for() -> None:
    selection = _selection()
    holdout = holdout_episode_ids(DATA_ROOT)
    assert holdout and set(selection.held_out) == holdout
    assert not holdout & set(selection.episode_ids)
    assert sum(selection.held_out.values()) == 1460


def test_every_anchor_carries_the_speed_of_its_reviewed_atom() -> None:
    """Subtask text and speed are read off the reviewed atom by native timestep: the
    anchor's own frame (``anchor_frame`` common, ``anchor_timestep`` FMB) must land inside
    exactly one atom, and the row keeps that atom's text and 1-5 bucket (the speed
    sidecar is joined to the atom by (episode, parent, atom) index here, independently
    of the selection's span lookup)."""
    selection = _selection()
    atoms: dict[tuple[str, str], list[dict]] = {}
    for key, store in (("common", selection.corpus.common), ("fmb", selection.corpus.fmb)):
        speeds = {
            (str(a["episode_id"]), int(a["parent_interval_index"]), int(a["atom_index"])): int(a["speed"])
            for a in store.speed_atoms()
        }
        for atom in store.subtask_atoms():
            key_ = (str(atom["episode_id"]), int(atom["parent_interval_index"]), int(atom["atom_index"]))
            atoms.setdefault((key, str(atom["episode_id"])), []).append({**atom, "speed": speeds[key_]})
    for row in selection.rows:
        frame = int(row["anchor_timestep"] if row["corpus_key"] == "fmb" else row["anchor_frame"])
        hits = [
            atom
            for atom in atoms[(row["corpus_key"], str(row["episode_id"]))]
            if int(atom["start_timestep"]) <= frame < int(atom["end_timestep_exclusive"])
        ]
        assert len(hits) == 1, (row["episode_id"], frame)
        assert row["speed"] == int(hits[0]["speed"])
        assert 1 <= row["speed"] <= 5
        # The step text is the atom's, not the parent interval's (kept as parent_subtask).
        assert row["subtask"] == hits[0]["subtask"]
        assert row["parent_subtask"] != "" and "subtask" in row
    per_source = audit_selection(selection)
    assert all(sum(audit.speed_values.values()) == audit.anchors for audit in per_source.values())


def test_without_sidecars_precision_and_contact_are_minus_one() -> None:
    """The optional channels: a store with no sidecar selects exactly as before, and
    every one of its rows carries the -1 that omits the clause."""
    selection = _selection()
    for key, sub in (("common", "corpus"), ("fmb", "fmb")):
        rows = [row for row in selection.rows if row["corpus_key"] == key]
        assert rows
        if not (DATA_ROOT / sub / PRECISION_ATOMS_VIEW).is_file():
            assert {row["precision"] for row in rows} == {-1}
        if not (DATA_ROOT / sub / CONTACT_ATOMS_VIEW).is_file():
            assert {row["contact"] for row in rows} == {-1}


def _label_sidecar(store, drop: str) -> tuple[list[dict], list[dict]]:
    """Synthetic precision / contact rows over the store's own atoms, in memory (never
    written into the corpus root), with every atom of episode ``drop`` left out."""
    precision, contact = [], []
    for atom in store.subtask_atoms():
        if str(atom["episode_id"]) == drop:
            continue
        span = {name: atom[name] for name in (
            "episode_id", "parent_interval_index", "atom_index", "start_timestep", "end_timestep_exclusive"
        )}
        precision.append({**span, "precision": 1 + int(atom["atom_index"]) % 5})
        contact.append({**span, "contact": (3 * int(atom["parent_interval_index"]) + int(atom["atom_index"])) % 15})
    return precision, contact


def test_precision_and_contact_follow_their_atom_and_a_missing_atom_is_minus_one(monkeypatch) -> None:
    corpus = _corpus()
    dropped = {}
    for key, store in (("common", corpus.common), ("fmb", corpus.fmb)):
        dropped[key] = str(store.subtask_atoms()[0]["episode_id"])
        precision, contact = _label_sidecar(store, dropped[key])
        monkeypatch.setattr(store, "precision_atoms", lambda rows=precision: rows)
        monkeypatch.setattr(store, "contact_atoms", lambda rows=contact: rows)
    selection = select_actor_anchors(corpus)

    atoms: dict[tuple[str, str], list[dict]] = {}
    for key, store in (("common", corpus.common), ("fmb", corpus.fmb)):
        for atom in store.subtask_atoms():
            atoms.setdefault((key, str(atom["episode_id"])), []).append(atom)
    missing = 0
    for row in selection.rows:
        if str(row["episode_id"]) == dropped[row["corpus_key"]]:
            assert (row["precision"], row["contact"]) == (-1, -1)
            missing += 1
            continue
        frame = int(row["anchor_timestep"] if row["corpus_key"] == "fmb" else row["anchor_frame"])
        (atom,) = [
            atom
            for atom in atoms[(row["corpus_key"], str(row["episode_id"]))]
            if int(atom["start_timestep"]) <= frame < int(atom["end_timestep_exclusive"])
        ]
        assert row["precision"] == 1 + int(atom["atom_index"]) % 5
        assert row["contact"] == (3 * int(atom["parent_interval_index"]) + int(atom["atom_index"])) % 15
    # The dropped episodes are not held out, so their rows are selected and carry -1.
    assert missing > 0
    per_source = audit_selection(selection)
    assert sum(audit.precision_values[-1] for audit in per_source.values()) == missing
    assert all(sum(audit.contact_values.values()) == audit.anchors for audit in per_source.values())


def test_split_stays_on_every_row_as_provenance() -> None:
    selection = _selection()
    assert all(row["split"] in {"train", "validation", "test"} for row in selection.rows)
    # ... and all three appear, which is what makes filtering on it a live hazard.
    assert {row["split"] for row in selection.rows} == {"train", "validation", "test"}


def test_mistake_flags_are_anchor_level_not_segment_level() -> None:
    """The stored flag over-claims on 219 common anchors in v3; ReBot's column is per-frame."""
    selection = _selection()
    assert selection.mistake_flags_corrected == 219
    corrected = [row for row in selection.rows if row["mistake"] != row["mistake_flag_as_stored"]]
    # Every correction removes a claim, never adds one.
    assert all(row["mistake_flag_as_stored"] and not row["mistake"] for row in corrected)
    for row in corrected:
        record = selection.episode_records[row["episode_id"]]
        segment = next(
            item
            for item in record["annotations"]["segments"]
            if item["start_s"] <= row["anchor_s"] < item["end_s"]
        )
        events = segment.get("mistake_events") or []
        assert events, "a correction only makes sense inside a mistake-bearing segment"
        assert not any(e["start_s"] <= row["anchor_s"] < e["end_s"] for e in events)


def test_every_row_carries_a_layout_and_a_camera_role_set() -> None:
    selection = _selection()
    for row in selection.rows:
        assert row["native_action_dim"] in (7, 8)
        assert row["action_layout_id"] == action_layout_for(row["source"], row["embodiment"]).index
        assert set(row["camera_roles"]) <= set(CANONICAL_CAMERA_ROLES)
        assert "wrist_0" in row["camera_roles"]
        assert row["has_depth"] == (row["source"] == "fmb")


def test_sample_contract_holds_on_real_samples() -> None:
    check_sample_contract(_selection(), per_source=3)
