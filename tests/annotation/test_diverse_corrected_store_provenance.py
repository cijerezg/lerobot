from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.annotation import assemble_diverse_corrected_store as assembler
from lerobot.annotation import diverse_corrected_store as store


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    for store_name, episode_id in (("corpus", "episode"), ("fmb", "fmb_episode")):
        root = source / store_name
        episode = root / "episodes" / episode_id
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text(
            json.dumps({"episode_id": episode_id, "frame_count": 2}), encoding="utf-8"
        )
        (episode / "payload.bin").write_bytes(f"payload-{episode_id}".encode())
        _write_jsonl(root / "episodes.jsonl", [{"episode_id": episode_id, "frames": 2}])
        for name in sorted(set(store.REPLACEABLE) | {store.ACTOR_VIEW, store.CRITIC_VIEW}):
            _write_jsonl(root / name, [])

    fix_dir = tmp_path / "fixes"
    fix_dir.mkdir()
    replacement_record = {"episode_id": "episode", "frame_count": 2, "annotations": {"reviewed": True}}
    spec = {
        "schema_version": 1,
        "status": "resolved",
        "store": "corpus",
        "episode_id": "episode",
        "replacements": {"quality_spans.jsonl": []},
        "episode_record": replacement_record,
        "unresolved": [],
        "second_read_required": False,
    }
    (fix_dir / "episode.json").write_text(json.dumps(spec), encoding="utf-8")
    return source, fix_dir


def test_pre_publish_failure_never_exposes_output(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    output = tmp_path / "corrected"

    def fail(_temporary: Path, _manifest: dict) -> None:
        raise store.StoreError("injected provenance failure")

    with pytest.raises(store.StoreError, match="injected provenance failure"):
        store.assemble(source, output, fix_dir, False, pre_publish=fail)
    assert not output.exists()
    assert not list(tmp_path.glob(".corrected.assembling-*"))


def test_manifest_hashes_tables_records_and_symlink_targets(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    output = tmp_path / "corrected"
    store.assemble(source, output, fix_dir, False)

    manifest = json.loads((output / "corrected_store_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["source_semantic_tables"]
    assert manifest["output_semantic_tables"]
    assert manifest["changed_episode_records"][0]["path"] == "corpus/episodes/episode/episode.json"
    assert {row["path"] for row in manifest["symlink_targets"]} == {
        "corpus/episodes/episode/payload.bin",
        "fmb/episodes/fmb_episode/payload.bin",
    }
    assert store.validate_assembled_store(source, output, fix_dir)["provenance_status"] == "verified"

    with (output / "corpus" / "quality_spans.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(store.StoreError, match="output_semantic_tables"):
        store.validate_assembled_store(source, output, fix_dir)


def test_validate_detects_link_target_content_drift(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    output = tmp_path / "corrected"
    store.assemble(source, output, fix_dir, False)
    (source / "corpus" / "episodes" / "episode" / "payload.bin").write_bytes(b"changed")

    with pytest.raises(store.StoreError, match="symlink_targets"):
        store.validate_assembled_store(source, output, fix_dir)


def test_wrapper_validate_only_gate_checks_completed_manifest(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    output = tmp_path / "corrected"
    assembler.assemble(source, output, fix_dir, False)

    report = assembler.validate_assembled_store(source, output, fix_dir)
    assert report["provenance_status"] == "verified"
    assert report["speed_trace_artifacts"] == []

    manifest_path = output / "corrected_store_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["speed_trace_artifacts"] = [{"unexpected": True}]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(store.StoreError, match="speed_trace_artifacts"):
        assembler.validate_assembled_store(source, output, fix_dir)
