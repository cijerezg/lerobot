"""Assemble normalized diverse corrections, including canonical speed traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.annotation import diverse_corrected_store as _base


_base.ATOM_ALIGNED = (
    "contact_atoms.jsonl",
    "precision_atoms.jsonl",
    "speed_atoms.jsonl",
    "speed_atoms_hybrid_v1.jsonl",
)
_base.REPLACEABLE = frozenset((_base.SUBTASK_VIEW, *_base.ATOM_ALIGNED, *_base.V2_VIEWS))

StoreError = _base.StoreError
load_corrections = _base.load_corrections
validate_tables = _base.validate_tables
TRACE_KEYS = {
    "joint_speed_rad_s", "motion_score", "duration_score", "speed_score",
    "valid", "supervision_mask", "speed", "native_rate_hz",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_artifact_path(fix_dir: Path, value: Any, spec: Path) -> Path:
    if not isinstance(value, str):
        raise StoreError(f"{spec}: trace artifact path must be a relative string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise StoreError(f"{spec}: trace artifact path escapes --fix-dir")
    path = (fix_dir / relative).resolve()
    if fix_dir.resolve() not in path.parents:
        raise StoreError(f"{spec}: trace artifact path escapes --fix-dir")
    return path


def _atom_trace_specs(fix_dir: Path) -> list[tuple[Path, dict[str, Any], Path]]:
    """Return the exact recursive set of atom-changing specs and their unique NPZs."""
    corrections = _base.load_corrections(fix_dir)
    output, referenced_paths, referenced_keys = [], set(), set()
    for correction in corrections:
        if _base.SUBTASK_VIEW not in correction.replacements:
            continue
        spec_path = correction.path
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        artifact = spec.get("artifacts", {}).get("speed_hybrid_v1")
        if not isinstance(artifact, dict):
            raise StoreError(f"{spec_path}: atom replacement requires a speed_hybrid_v1 trace artifact")
        path = _safe_artifact_path(fix_dir, artifact.get("path"), spec_path)
        key = (correction.store, correction.episode_id)
        if key in referenced_keys or path in referenced_paths:
            raise StoreError(f"{spec_path}: trace artifact is not one-to-one with an atom-changing correction")
        referenced_keys.add(key)
        referenced_paths.add(path)
        output.append((spec_path, spec, path))
    discovered = {path.resolve() for path in fix_dir.rglob("*.npz") if path.is_file()}
    if discovered != referenced_paths:
        missing = sorted(str(path) for path in referenced_paths - discovered)
        orphan = sorted(str(path) for path in discovered - referenced_paths)
        raise StoreError(f"trace artifact inventory mismatch; missing={missing}, orphan={orphan}")
    return output


def _validate_trace_file(
    path: Path, spec_path: Path, spec: dict[str, Any], source_root: Path, *, require_regular: bool = False
) -> dict[str, Any]:
    replacements = spec["replacements"]
    atoms = replacements["subtask_atoms.jsonl"]
    artifact = spec["artifacts"]["speed_hybrid_v1"]
    if require_regular and path.is_symlink():
        raise StoreError(f"{spec_path}: corrected output trace must be a copied regular file: {path}")
    if not path.is_file():
        raise StoreError(f"{spec_path}: trace artifact is missing: {path}")
    digest = _sha256(path)
    if digest != artifact.get("sha256"):
        raise StoreError(f"{spec_path}: trace artifact hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != TRACE_KEYS or set(artifact.get("keys", [])) != TRACE_KEYS:
            raise StoreError(f"{spec_path}: trace artifact has noncanonical keys")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    episode = str(spec["episode_id"])
    store = str(spec["store"])
    episode_dir = source_root / store / "episodes" / episode
    state = episode_dir / ("q.npy" if store == "fmb" else "state.npy")
    transition_count = int(np.load(state, mmap_mode="r").shape[0]) - 1
    vector_keys = TRACE_KEYS - {"native_rate_hz"}
    if any(arrays[key].shape != (transition_count,) for key in vector_keys):
        raise StoreError(f"{spec_path}: trace arrays must all have shape ({transition_count},)")
    if arrays["native_rate_hz"].shape != () or arrays["native_rate_hz"].dtype != np.float64:
        raise StoreError(f"{spec_path}: native_rate_hz must be scalar float64")
    if arrays["valid"].dtype != np.bool_ or arrays["supervision_mask"].dtype != np.bool_:
        raise StoreError(f"{spec_path}: valid and supervision_mask must be bool")
    if arrays["speed"].dtype != np.uint8:
        raise StoreError(f"{spec_path}: speed must be uint8")
    for key in ("joint_speed_rad_s", "motion_score", "duration_score", "speed_score"):
        if arrays[key].dtype != np.float64:
            raise StoreError(f"{spec_path}: {key} must be float64")
    expected_mask = np.zeros(transition_count, dtype=bool)
    ordered_atoms = sorted(atoms, key=lambda row: int(row["start_timestep"]))
    for atom in ordered_atoms:
        start, stop = int(atom["start_timestep"]), int(atom["end_timestep_exclusive"])
        expected_mask[start : stop - 1] = True
    if not np.array_equal(arrays["supervision_mask"], expected_mask):
        raise StoreError(f"{spec_path}: supervision mask does not exactly match corrected atoms")
    retained_transitions = expected_mask.copy()
    for left, right in zip(ordered_atoms, ordered_atoms[1:], strict=False):
        stop = int(left["end_timestep_exclusive"])
        if stop == int(right["start_timestep"]):
            retained_transitions[stop - 1] = True
    excluded = ~retained_transitions
    if arrays["valid"][excluded].any() or np.isfinite(arrays["speed_score"][excluded]).any():
        raise StoreError(f"{spec_path}: excluded transitions remain valid/scored")
    if arrays["speed"][~expected_mask].any():
        raise StoreError(f"{spec_path}: excluded transitions have nonzero labels")
    by_key = {
        (int(row["parent_interval_index"]), int(row["atom_index"])): row
        for row in replacements["speed_atoms_hybrid_v1.jsonl"]
    }
    for atom in atoms:
        key = (int(atom["parent_interval_index"]), int(atom["atom_index"]))
        row = by_key.get(key)
        if row is None:
            raise StoreError(f"{spec_path}: trace has no aggregate row for atom {key}")
        start, stop = int(atom["start_timestep"]), int(atom["end_timestep_exclusive"])
        scores = arrays["speed_score"][start : stop - 1]
        reason = len(scores) < 2 or not np.isfinite(scores).all()
        score = 3.0 if reason else float(np.median(scores))
        expected_speed = 3 if reason else int(np.clip(np.floor(score + 0.5), 1, 5))
        if int(row["speed"]) != expected_speed or not np.isclose(float(row["speed_score"]), score):
            raise StoreError(f"{spec_path}: aggregate speed disagrees with trace at atom {key}")
    return {"store": store, "episode_id": episode, "sha256": digest, "transitions": transition_count}


def validate_trace_artifacts(source_root: Path, fix_dir: Path) -> list[dict[str, Any]]:
    """Validate the exact 1:1 set of declared NPZ artifacts recursively."""
    results = []
    for spec_path, spec, path in _atom_trace_specs(fix_dir):
        result = _validate_trace_file(path, spec_path, spec, source_root)
        results.append({**result, "path": str(path)})
    return results


def validate_copied_trace_artifacts(source_root: Path, out_root: Path, fix_dir: Path) -> list[dict[str, Any]]:
    """Revalidate the copied output NPZ bytes and schema, not only fix-dir inputs."""
    results = []
    for spec_path, spec, source_path in _atom_trace_specs(fix_dir):
        store, episode = str(spec["store"]), str(spec["episode_id"])
        output_relative = Path(store) / "speed_hybrid_v1" / f"{episode}.npz"
        result = _validate_trace_file(out_root / output_relative, spec_path, spec, source_root, require_regular=True)
        results.append({**result, "path": str(source_path), "output_path": output_relative.as_posix()})
    return results


def assemble(source_root: Path, out_root: Path, fix_dir: Path, dry_run: bool) -> dict[str, Any]:
    source_root, out_root, fix_dir = source_root.resolve(), out_root.resolve(), fix_dir.resolve()
    artifacts = validate_trace_artifacts(source_root, fix_dir)
    copied_artifacts: list[dict[str, Any]] = []
    original_copy = _base._copy_scaffold

    def copy_with_traces(source: Path, temporary: Path, corrections: list[_base.Correction]) -> None:
        original_copy(source, temporary, corrections)
        for store in _base.STORES:
            source_dir = source / store / "speed_hybrid_v1"
            destination = temporary / store / "speed_hybrid_v1"
            destination.mkdir()
            if source_dir.is_dir():
                for path in source_dir.glob("*.npz"):
                    (destination / path.name).symlink_to(path.resolve())
        for item in artifacts:
            target = temporary / item["store"] / "speed_hybrid_v1" / f"{item['episode_id']}.npz"
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copy2(item["path"], target)

    _base._copy_scaffold = copy_with_traces
    try:
        def complete_manifest(temporary: Path, _manifest: dict[str, Any]) -> dict[str, Any]:
            copied = validate_copied_trace_artifacts(source_root, temporary, fix_dir)
            copied_artifacts.extend(copied)
            return {"speed_trace_artifacts": copied}

        report = _base.assemble(
            source_root, out_root, fix_dir, dry_run,
            pre_publish=None if dry_run else complete_manifest,
        )
    finally:
        _base._copy_scaffold = original_copy
    report["speed_trace_artifacts"] = artifacts
    if not dry_run:
        report["speed_trace_artifacts"] = copied_artifacts
    return report


def validate_assembled_store(source_root: Path, out_root: Path, fix_dir: Path) -> dict[str, Any]:
    source_root, out_root, fix_dir = source_root.resolve(), out_root.resolve(), fix_dir.resolve()
    report = _base.validate_assembled_store(source_root, out_root, fix_dir)
    declared = validate_trace_artifacts(source_root, fix_dir)
    traces = validate_copied_trace_artifacts(source_root, out_root, fix_dir)
    if [(row["store"], row["episode_id"], row["sha256"]) for row in declared] != [
        (row["store"], row["episode_id"], row["sha256"]) for row in traces
    ]:
        raise StoreError("declared and copied speed trace inventories differ")
    manifest = json.loads((out_root / "corrected_store_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("speed_trace_artifacts") != traces:
        raise StoreError("corrected-store manifest mismatch for speed_trace_artifacts")
    report["speed_trace_artifacts"] = traces
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--fix-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.validate_only:
        parser.error("--dry-run and --validate-only are mutually exclusive")
    if args.validate_only:
        report = validate_assembled_store(args.source_root, args.out_root, args.fix_dir)
    else:
        report = assemble(args.source_root, args.out_root, args.fix_dir, args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
