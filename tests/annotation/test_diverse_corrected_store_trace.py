import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.annotation import assemble_diverse_corrected_store as assembler

EPISODE = "episode"


def _fixture(tmp_path: Path, *, gap: bool = False) -> tuple[Path, Path]:
    source = tmp_path / "source"
    fix_dir = tmp_path / "fixes"
    episode = EPISODE
    episode_dir = source / "corpus" / "episodes" / episode
    episode_dir.mkdir(parents=True)
    np.save(episode_dir / "state.npy", np.zeros((4, 3), dtype=np.float64))
    artifact = fix_dir / "artifacts" / "corpus" / "speed_hybrid_v1" / f"{episode}.npz"
    artifact.parent.mkdir(parents=True)
    mask = np.array([True, False, not gap])
    valid = np.array([True, not gap, not gap])
    score = np.array([2.0, 2.5 if not gap else np.nan, 3.0 if not gap else np.nan])
    np.savez_compressed(
        artifact,
        joint_speed_rad_s=np.array([0.1, 0.2, 0.3]),
        motion_score=np.array([2.0, 2.5, 3.0]),
        duration_score=np.array([3.0, 3.0, 3.0]),
        speed_score=score,
        valid=valid,
        supervision_mask=mask,
        speed=np.array([3, 0, 3 if not gap else 0], dtype=np.uint8),
        native_rate_hz=np.asarray(10.0, dtype=np.float64),
    )
    second_start = 3 if gap else 2
    spec = {
        "schema_version": 1, "status": "resolved", "store": "corpus", "episode_id": episode,
        "replacements": {
            "subtask_atoms.jsonl": [
                {"episode_id": episode, "parent_interval_index": 0, "atom_index": 0,
                 "start_timestep": 0, "end_timestep_exclusive": 2},
                {"episode_id": episode, "parent_interval_index": 1, "atom_index": 0,
                 "start_timestep": second_start, "end_timestep_exclusive": 4},
            ],
            "speed_atoms_hybrid_v1.jsonl": [
                {"parent_interval_index": 0, "atom_index": 0, "speed": 3, "speed_score": 3.0},
                {"parent_interval_index": 1, "atom_index": 0, "speed": 3, "speed_score": 3.0},
            ],
            "contact_atoms.jsonl": [],
            "precision_atoms.jsonl": [],
            "speed_atoms.jsonl": [],
        },
        "actor_anchors": {"mode": "remap", "retained_intervals": [[0, 4]]},
        "critic_intervals": {"mode": "from_atoms"},
        "artifacts": {"speed_hybrid_v1": {
            "path": artifact.relative_to(fix_dir).as_posix(),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "keys": sorted(assembler.TRACE_KEYS),
        }},
    }
    (fix_dir / "episode.json").write_text(json.dumps(spec))
    return source, fix_dir


def test_trace_gate_accepts_unsupervised_transition_between_adjacent_atoms(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    result = assembler.validate_trace_artifacts(source, fix_dir)
    assert result[0]["transitions"] == 3


def test_trace_gate_rejects_scored_excluded_gap(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path, gap=True)
    artifact = next((fix_dir / "artifacts").rglob("*.npz"))
    with np.load(artifact) as z:
        payload = {key: z[key] for key in z.files}
    payload["valid"][1] = True
    payload["speed_score"][1] = 2.5
    np.savez_compressed(artifact, **payload)
    spec_path = fix_dir / "episode.json"
    spec = json.loads(spec_path.read_text())
    spec["artifacts"]["speed_hybrid_v1"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(assembler.StoreError, match="excluded transitions"):
        assembler.validate_trace_artifacts(source, fix_dir)


def test_trace_gate_uses_recursive_correction_inventory_and_rejects_orphans(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    nested = fix_dir / "nested"
    nested.mkdir()
    (fix_dir / "episode.json").rename(nested / "episode.json")

    assert len(assembler.validate_trace_artifacts(source, fix_dir)) == 1

    artifact = next((fix_dir / "artifacts").rglob("*.npz"))
    orphan = artifact.with_name("orphan.npz")
    orphan.write_bytes(artifact.read_bytes())
    with pytest.raises(assembler.StoreError, match="orphan"):
        assembler.validate_trace_artifacts(source, fix_dir)


def test_validate_copied_trace_reopens_output_npz(tmp_path: Path) -> None:
    source, fix_dir = _fixture(tmp_path)
    out = tmp_path / "out"
    target = out / "corpus" / "speed_hybrid_v1" / f"{EPISODE}.npz"
    target.parent.mkdir(parents=True)
    artifact = next((fix_dir / "artifacts").rglob("*.npz"))
    target.write_bytes(artifact.read_bytes())

    result = assembler.validate_copied_trace_artifacts(source, out, fix_dir)
    assert result[0]["output_path"] == f"corpus/speed_hybrid_v1/{EPISODE}.npz"

    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(assembler.StoreError, match="hash mismatch"):
        assembler.validate_copied_trace_artifacts(source, out, fix_dir)
