import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.annotation import diverse_audit
from lerobot.annotation.diverse_audit import (
    FAMILY_ORDER,
    allocate_families,
    array_schema,
    camera_modalities,
    choose_family_rows,
    handoff,
    refresh,
    require_prior_rounds_complete,
    resolve_active_root,
    sample,
    source_retention,
)


def test_source_retention_uses_source_segments_and_preserves_gaps() -> None:
    record = {
        "episode_id": "episode",
        "annotations": {
            "segments": [
                {"start_s": 0.0, "end_s": 0.3, "retention": "keep", "retention_reason": "useful_motion"},
                {"start_s": 0.3, "end_s": 0.5, "retention": "reject", "retention_reason": "source_excluded"},
                {"start_s": 0.5, "end_s": 1.0, "retention": "keep", "retention_reason": "recovery"},
            ]
        },
    }

    timestamps = np.arange(10, dtype=np.float64) / 10.0
    source, retained, excluded, provenance = source_retention(record, "corpus", 10.0, 10, timestamps)

    assert retained == [[0, 3], [5, 10]]
    assert excluded == [[3, 5]]
    assert [row["retention"] for row in source] == ["keep", "reject", "keep"]
    assert "annotations.segments" in provenance


def test_source_retention_uses_stored_timestamps_not_nominal_rate() -> None:
    record = {
        "episode_id": "episode",
        "annotations": {
            "segments": [
                {"start_s": 0.0, "end_s": 0.3, "retention": "keep", "retention_reason": "useful_motion"},
            ]
        },
    }
    timestamps = np.array([0.0, 0.1, 0.2, 0.2999999], dtype=np.float64)

    source, retained, excluded, provenance = source_retention(record, "corpus", 10.0, 4, timestamps)

    assert source[0]["end_timestep_exclusive"] == 4
    assert retained == [[0, 4]]
    assert excluded == []
    assert "stored native timestamps" in provenance


def test_fmb_source_retention_uses_reviewed_primitive_bounds() -> None:
    record = {
        "episode_id": "fmb-episode",
        "primitive_intervals": [
            {
                "start_timestep": 0,
                "end_timestep_exclusive": 10,
                "reviewed_start_timestep": 2,
                "reviewed_end_timestep_exclusive": 8,
                "primitive": "grasp",
            },
            {
                "start_timestep": 10,
                "end_timestep_exclusive": 20,
                "primitive": "insert",
            },
        ],
    }

    source, retained, excluded, provenance = source_retention(record, "fmb", 10.0, 20)

    assert retained == [[2, 8], [10, 20]]
    assert excluded == [[0, 2], [8, 10]]
    assert source[0]["source_start_timestep"] == 0
    assert source[0]["source_end_timestep_exclusive"] == 10
    assert source[0]["retention_reason"] == "production_reviewed_primitive"
    assert "reviewed bounds" in provenance


def _pools(*, empty: tuple[str, ...] = ()) -> dict[str, list[dict]]:
    return {family: ([] if family in empty else [{"family": family}] * 20) for family in FAMILY_ORDER}


def test_family_allocation_rotates_past_exhausted_family_and_wraps() -> None:
    prior = [{"family": family} for family in FAMILY_ORDER for _ in range(4)]
    allocation = allocate_families(_pools(empty=("ur7e",)), prior, round_number=5)
    assert allocation["yam"] == 2
    assert allocation["fmb"] == 2
    assert sum(allocation.values()) == 8

    wrapped = allocate_families(_pools(), prior, round_number=9)
    assert wrapped["droid_success"] == 2
    assert sum(wrapped.values()) == 8


def test_within_round_choice_updates_all_coverage_dimensions() -> None:
    candidates = [
        {"episode_id": "old", "family": "droid", "split": "train", "component": "A", "task": "one", "embodiment": "R1"},
        {"episode_id": "new-a", "family": "droid", "split": "validation", "component": "B", "task": "two", "embodiment": "R2"},
        {"episode_id": "new-b", "family": "droid", "split": "test", "component": "C", "task": "three", "embodiment": "R3"},
    ]
    seen = {
        "split": {("droid", "train")},
        "component": {("droid", "A")},
        "task": {("droid", "one")},
        "embodiment": {("droid", "R1")},
    }

    chosen = choose_family_rows(candidates, 2, random.Random(4), seen)

    assert {row["episode_id"] for row in chosen} == {"new-a", "new-b"}
    assert ("droid", "validation") in seen["split"]
    assert ("droid", "test") in seen["split"]


def _write_completed_round(
    work: Path,
    number: int,
    *,
    wrong: int = 0,
    minor: int = 0,
    systematic: bool = False,
    families: tuple[str, ...] = FAMILY_ORDER,
    sweeps_open: int = 0,
) -> list[str]:
    (work / "rounds").mkdir(exist_ok=True)
    (work / "checks").mkdir(exist_ok=True)
    episode_ids = [f"episode-{number}-{index}" for index in range(8)]
    picks = [
        {"episode_id": episode_id, "family": families[index % len(families)]}
        for index, episode_id in enumerate(episode_ids)
    ]
    round_row = {
        "round": number,
        "status": "complete",
        "completion": {"fixes_open": 0, "second_reads_open": 0, "sweeps_open": sweeps_open},
        "pre_fix_counts": {"wrong": wrong, "minor": minor, "open": 0},
        "systematic_issue_detected": systematic,
        "picks": picks,
    }
    (work / "rounds" / f"round_{number:02d}.json").write_text(json.dumps(round_row))
    severities = ["wrong"] * wrong + ["minor"] * minor
    for index, episode_id in enumerate(episode_ids):
        issues = [{"severity": severity} for severity in severities] if index == 0 else []
        (work / "checks" / f"round_{number:02d}__{episode_id}.json").write_text(json.dumps({
            "episode_id": episode_id,
            "status": "complete",
            "second_read_required": False,
            "issues": issues,
        }))
    return episode_ids


def test_prior_round_gate_requires_explicit_closed_work(tmp_path: Path) -> None:
    episode_ids = _write_completed_round(tmp_path, 1)

    require_prior_rounds_complete(tmp_path, 2)

    check_path = tmp_path / "checks" / f"round_01__{episode_ids[0]}.json"
    check_path.write_text(json.dumps({
        "episode_id": episode_ids[0], "status": "complete", "second_read_required": True, "issues": []
    }))
    with pytest.raises(ValueError, match="check remains open"):
        require_prior_rounds_complete(tmp_path, 2)


def test_prior_round_gate_checks_counts_and_full_round_size(tmp_path: Path) -> None:
    _write_completed_round(tmp_path, 1)
    round_path = tmp_path / "rounds" / "round_01.json"
    row = json.loads(round_path.read_text())
    row["pre_fix_counts"]["wrong"] = 1
    round_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="counts do not match"):
        require_prior_rounds_complete(tmp_path, 2)

    row["pre_fix_counts"]["wrong"] = 0
    row["picks"].pop()
    round_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="exactly 8 picks"):
        require_prior_rounds_complete(tmp_path, 2)


def test_prior_round_gate_requires_integer_zero_completion_counts(tmp_path: Path) -> None:
    _write_completed_round(tmp_path, 1)
    round_path = tmp_path / "rounds" / "round_01.json"
    row = json.loads(round_path.read_text())
    row["completion"]["fixes_open"] = False
    round_path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="must record completion as zero"):
        require_prior_rounds_complete(tmp_path, 2)


def test_existing_round_rerun_still_checks_prior_completion(tmp_path: Path) -> None:
    _write_completed_round(tmp_path, 1)
    round_one = tmp_path / "rounds" / "round_01.json"
    row = json.loads(round_one.read_text())
    row["status"] = "reviewed_needs_repair"
    round_one.write_text(json.dumps(row))
    (tmp_path / "rounds" / "round_02.json").write_text("{}")

    with pytest.raises(ValueError, match="round 1 status"):
        sample(SimpleNamespace(work=str(tmp_path), round=2, seed=7))


def test_sampler_refuses_short_exhausted_round(tmp_path: Path) -> None:
    (tmp_path / "rounds").mkdir()
    (tmp_path / "checks").mkdir()
    (tmp_path / "inventory.jsonl").write_text("".join(
        json.dumps({
            "episode_id": f"episode-{index}",
            "family": "droid",
            "split": "train",
            "component": "component",
            "task": "task",
            "embodiment": "robot",
        }) + "\n"
        for index in range(7)
    ))

    with pytest.raises(ValueError, match="short final batch"):
        sample(SimpleNamespace(work=str(tmp_path), round=1, seed=7))
    assert not (tmp_path / "rounds" / "round_01.json").exists()


def test_gate_stops_after_two_clean_rounds_with_family_coverage(tmp_path: Path) -> None:
    (tmp_path / "inventory.jsonl").write_text("".join(
        json.dumps({"episode_id": f"inventory-{family}", "family": family}) + "\n"
        for family in FAMILY_ORDER
    ))
    _write_completed_round(tmp_path, 1)
    _write_completed_round(tmp_path, 2)

    with pytest.raises(ValueError, match="stopping rule already met"):
        require_prior_rounds_complete(tmp_path, 3)


def test_systematic_issue_resets_clean_streak_and_open_sweep_blocks(tmp_path: Path) -> None:
    _write_completed_round(tmp_path, 1)
    _write_completed_round(tmp_path, 2, systematic=True)
    require_prior_rounds_complete(tmp_path, 3)

    row_path = tmp_path / "rounds" / "round_02.json"
    row = json.loads(row_path.read_text())
    row["completion"]["sweeps_open"] = 1
    row_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="must record completion"):
        require_prior_rounds_complete(tmp_path, 3)


def test_clean_round_requires_explicit_systematic_issue_record(tmp_path: Path) -> None:
    _write_completed_round(tmp_path, 1)
    row_path = tmp_path / "rounds" / "round_01.json"
    row = json.loads(row_path.read_text())
    del row["systematic_issue_detected"]
    row_path.write_text(json.dumps(row))

    with pytest.raises(ValueError, match="must explicitly record systematic_issue_detected"):
        require_prior_rounds_complete(tmp_path, 2)


def _write_handoff_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    work, active_root = tmp_path / "work", tmp_path / "active"
    work.mkdir()
    active_root.mkdir()
    inventory = [
        {"episode_id": "one", "family": "droid", "split": "train", "native_rate_hz": 15.0,
         "label_store": str(active_root / "corpus")},
        {"episode_id": "two", "family": "fmb", "split": "test", "native_rate_hz": 10.0,
         "label_store": str(active_root / "fmb")},
    ]
    (work / "inventory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inventory))
    (work / "population.json").write_text(json.dumps({
        "root": str(active_root),
        "episodes": 2,
        "families": {"droid": 1, "fmb": 1},
        "splits": {"train": 1, "test": 1},
        "provenance": {},
    }))
    return work, active_root, inventory


def test_handoff_updates_only_audit_review_pointers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, active_root, inventory = _write_handoff_fixture(tmp_path)
    corrected_root = tmp_path / "corrected"
    corrected_root.mkdir()
    (corrected_root / "corrected_store_manifest.json").write_text(json.dumps({
        "source_root": str(active_root), "post_write_validation": {"stores": {}}
    }))
    corrected_inventory = [
        {**row, "label_store": str(corrected_root / ("fmb" if row["family"] == "fmb" else "corpus"))}
        for row in inventory
    ]
    monkeypatch.setattr(diverse_audit, "build_inventory", lambda root: (corrected_inventory, {"root": str(root)}))

    handoff(SimpleNamespace(work=str(work), corrected_root=str(corrected_root)))

    population = json.loads((work / "population.json").read_text())
    assert population["root"] == str(active_root)
    assert population["active_root"] == str(active_root)
    assert population["review_root"] == str(corrected_root.resolve())
    assert all(str(corrected_root) in row["label_store"] for row in diverse_audit.read_jsonl(work / "inventory.jsonl"))


def test_handoff_rejects_wrong_lineage_or_changed_sampling_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, active_root, inventory = _write_handoff_fixture(tmp_path)
    corrected_root = tmp_path / "corrected"
    corrected_root.mkdir()
    manifest_path = corrected_root / "corrected_store_manifest.json"
    manifest_path.write_text(json.dumps({
        "source_root": str(tmp_path / "other"), "post_write_validation": {"stores": {}}
    }))
    with pytest.raises(ValueError, match="does not equal prior audit review root"):
        handoff(SimpleNamespace(work=str(work), corrected_root=str(corrected_root)))

    manifest_path.write_text(json.dumps({
        "source_root": str(active_root), "post_write_validation": {"stores": {}}
    }))
    changed = [{**row, "split": "validation" if row["episode_id"] == "one" else row["split"]} for row in inventory]
    monkeypatch.setattr(diverse_audit, "build_inventory", lambda root: (changed, {"root": str(root)}))
    with pytest.raises(ValueError, match="changed sampling identity"):
        handoff(SimpleNamespace(work=str(work), corrected_root=str(corrected_root)))


def test_refresh_refuses_to_mix_active_and_handed_off_review_roots(tmp_path: Path) -> None:
    work, active_root, _ = _write_handoff_fixture(tmp_path)
    population_path = work / "population.json"
    population = json.loads(population_path.read_text())
    population["review_root"] = str(tmp_path / "corrected")
    population_path.write_text(json.dumps(population))
    config = tmp_path / "config.yaml"
    config.write_text(f"diverse:\n  enabled: true\n  root: {active_root}\n")

    with pytest.raises(ValueError, match="would mix review pointers"):
        refresh(SimpleNamespace(config=str(config), work=str(work), root=None))


def test_fmb_schema_and_camera_modalities_are_explicit() -> None:
    record = {
        "arrays": {
            "obs/q": {"path": "q.npy", "shape": [12, 7], "dtype": "float64"},
            "actions": {"path": "actions.npy", "shape": [12, 7], "dtype": "float64"},
        },
        "production_retained_sensor_counts": {
            "rgb_images_by_camera": {"side_1": 12, "wrist_1": 12},
            "depth_maps_by_camera": {"wrist_1": 12},
        },
        "dropped_modalities": {"rgb": ["wrist_2"], "depth": ["side_1"]},
    }

    assert array_schema(record, ["obs/q"])["obs/q"]["shape"] == [12, 7]
    assert camera_modalities(record, "fmb") == {
        "rgb": ["side_1", "wrist_1"],
        "depth": ["wrist_1"],
        "dropped_rgb": ["wrist_2"],
        "dropped_depth": ["side_1"],
    }


def test_active_root_must_match_supplied_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("diverse:\n  enabled: true\n  root: outputs/current\n")

    root, metadata = resolve_active_root(config, "outputs/current")
    assert root == Path("outputs/current")
    assert metadata["sha256"]
    with pytest.raises(ValueError, match="does not match"):
        resolve_active_root(config, "outputs/stale")
