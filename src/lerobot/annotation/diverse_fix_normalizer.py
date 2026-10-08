"""Normalize diverse round repair specs into ``diverse_corrected_store`` schema.

The round-one agents wrote several presentation schemas.  This adapter accepts only
semantically equivalent *full episode replacement* forms.  Surgical edits, shorthand
rows requiring derivation, and pending recomputation gates are refused.  Originals
are read-only; normalized files are written to a fresh directory.

Usage::

    uv run python -m lerobot.annotation.diverse_fix_normalizer \
      --source-root outputs/diverse_robot_dataset_v3 \
      --fix-dir migration/diverse_annotation_audit_2026-10-07/fixes/round1 \
      --out-dir migration/diverse_annotation_audit_2026-10-07/fixes/round1_normalized \
      --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CHANNELS = {
    "subtask_atoms.jsonl",
    "contact_atoms.jsonl",
    "precision_atoms.jsonl",
    "speed_atoms.jsonl",
    "speed_atoms_hybrid_v1.jsonl",
    "quality_spans.jsonl",
    "mistakes_v2.jsonl",
    "precision_windows.jsonl",
}
READY_STATUSES = {None, "resolved", "ready_for_versioned_build"}


class NormalizationError(ValueError):
    """A source spec cannot be converted without inventing data."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _objects(value: Any, where: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise NormalizationError(f"{where}: full replacement rows must be JSON objects")
    return value


def _infer_store(spec: dict[str, Any], path: Path) -> str:
    store = spec.get("store")
    if store in {"corpus", "fmb"}:
        return store
    pointer = str(spec.get("source_store", spec.get("active_root", ""))).rstrip("/")
    if pointer.endswith("/corpus"):
        return "corpus"
    if pointer.endswith("/fmb"):
        return "fmb"
    raise NormalizationError(f"{path}: cannot infer corpus/fmb store")


def _retained_intervals(spec: dict[str, Any], path: Path) -> list[list[int]]:
    value = spec.get("retained_intervals")
    retention = spec.get("retention")
    if isinstance(retention, dict) and retention.get("replacement_retained_intervals") is not None:
        value = retention["replacement_retained_intervals"]
    if not isinstance(value, list) or not value:
        raise NormalizationError(f"{path}: actor rebuild has no explicit retained intervals")
    result, previous = [], -1
    for interval in value:
        if not isinstance(interval, list) or len(interval) != 2:
            raise NormalizationError(f"{path}: invalid retained interval {interval!r}")
        start, stop = map(int, interval)
        if start < previous or stop <= start:
            raise NormalizationError(f"{path}: retained intervals are not sorted disjoint half-open ranges")
        result.append([start, stop])
        previous = stop
    return result


def _extract_mapping(spec: dict[str, Any], path: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract all recognized full-row representations; reject partial operations."""
    output: dict[str, list[dict[str, Any]]] = {}

    direct = spec.get("replacement_episode_rows")
    if isinstance(direct, dict):
        for name, rows in direct.items():
            if name in CHANNELS:
                output[name] = _objects(rows, f"{path}:replacement_episode_rows.{name}")

    for section_name in ("replacements", "operations"):
        section = spec.get(section_name)
        if not isinstance(section, dict):
            continue
        for name, operation in section.items():
            if name not in CHANNELS and name not in {"critic_intervals.jsonl", "actor_anchors_5hz.jsonl"}:
                continue
            if not isinstance(operation, dict):
                raise NormalizationError(f"{path}:{section_name}.{name}: operation must be an object")
            partial = set(operation) & {"add", "remove", "replace"}
            if partial:
                raise NormalizationError(
                    f"{path}:{name}: surgical operation {sorted(partial)} must first be resolved to full episode rows"
                )
            candidates = [
                operation.get("rows"),
                operation.get("replace_complete_episode_rows"),
                operation.get("replace_episode_rows"),
            ]
            rows = next((candidate for candidate in candidates if candidate is not None), None)
            if rows is None:
                continue
            parsed = _objects(rows, f"{path}:{section_name}.{name}")
            if name in output and output[name] != parsed:
                raise NormalizationError(f"{path}:{name}: conflicting full replacements")
            output[name] = parsed
    return output


def _episode_record(source_root: Path, store: str, episode_id: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    replacement = spec.get("episode_record")
    if replacement is not None:
        if not isinstance(replacement, dict):
            raise NormalizationError("episode_record must be a JSON object")
        return replacement
    annotations = spec.get("episode_json_annotations_replacement")
    if annotations is None:
        return None
    if not isinstance(annotations, dict):
        raise NormalizationError("episode_json_annotations_replacement must be an object")
    path = source_root / store / "episodes" / episode_id / "episode.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["annotations"] = annotations
    return record


def normalize_one(path: Path, source_root: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise NormalizationError(f"{path}: top level must be an object")
    if spec.get("status") not in READY_STATUSES:
        raise NormalizationError(f"{path}: unresolved status {spec.get('status')!r}")
    if spec.get("second_read_required"):
        raise NormalizationError(f"{path}: second read remains required")
    if spec.get("unresolved"):
        raise NormalizationError(f"{path}: unresolved items remain")
    episode_id = str(spec.get("episode_id", ""))
    if not episode_id:
        raise NormalizationError(f"{path}: episode_id is missing")
    store = _infer_store(spec, path)
    rows = _extract_mapping(spec, path)

    # A full atom replacement is useful only when every joined channel has its own
    # complete replacement. speed_atoms is included even though the active trainer
    # reads hybrid speed; it is the duration input and must not retain stale bounds.
    if "subtask_atoms.jsonl" in rows:
        required = {
            "contact_atoms.jsonl",
            "precision_atoms.jsonl",
            "speed_atoms.jsonl",
            "speed_atoms_hybrid_v1.jsonl",
            "quality_spans.jsonl",
            "mistakes_v2.jsonl",
            "precision_windows.jsonl",
        }
        missing = sorted(required - set(rows))
        if missing:
            raise NormalizationError(f"{path}: atom repair lacks full replacements for {missing}")

    critic_rows = rows.pop("critic_intervals.jsonl", None)
    actor_rows = rows.pop("actor_anchors_5hz.jsonl", None)
    actor_rebuild_claimed = any(
        isinstance(container, dict)
        and isinstance(container.get("actor_anchors_5hz.jsonl"), dict)
        and any("rebuild" in key for key in container["actor_anchors_5hz.jsonl"])
        for container in (spec.get("operations"), spec.get("replacements"))
    ) or "actor_anchor_history_filter" in spec or "index_and_anchor_impacts" in spec
    actor = (
        {"mode": "replace", "rows": actor_rows}
        if actor_rows is not None
        else {"mode": "remap", "retained_intervals": _retained_intervals(spec, path)}
        if actor_rebuild_claimed or "subtask_atoms.jsonl" in rows
        else {"mode": "preserve"}
    )
    critic = (
        {"mode": "replace", "rows": critic_rows}
        if critic_rows is not None
        else {"mode": "from_atoms"}
        if "subtask_atoms.jsonl" in rows
        else {"mode": "preserve"}
    )

    # The original UR7e shorthand is deliberately not expanded here: it explicitly
    # says canonical speed recomputation remains a completion gate. Correct the two
    # known metadata defects in-memory if a later resolved full-row revision is used.
    if episode_id == "ur7e__stack_block__ep000041":
        validation = spec.get("validation_invariants")
        if isinstance(validation, dict) and validation.get("retained_frames") not in (None, 1555):
            validation = {**validation, "retained_frames": 1555}
        for row in rows.get("precision_windows.jsonl", []):
            uid = str(row.get("uid", ""))
            row["uid"] = uid.replace("ur7e__stack_block__ep000041_", "stack_block__ep000041_")

    return {
        "schema_version": 1,
        "status": "resolved",
        "store": store,
        "episode_id": episode_id,
        "replacements": rows,
        "actor_anchors": actor,
        "critic_intervals": critic,
        "episode_record": _episode_record(source_root, store, episode_id, spec),
        "unresolved": [],
        "second_read_required": False,
        "normalization_provenance": {
            "source_spec": str(path),
            "source_spec_sha256": _sha256(path),
            "source_schema_version": spec.get("schema_version"),
            "full_episode_replacements_only": True,
        },
    }


def normalize_all(source_root: Path, fix_dir: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[dict[str, str]]]:
    normalized, refused = [], []
    for path in sorted(fix_dir.glob("*.json")):
        try:
            normalized.append((path, normalize_one(path, source_root)))
        except (NormalizationError, KeyError, TypeError, ValueError) as exc:
            refused.append({"source_spec": str(path), "error": str(exc)})
    if not normalized and not refused:
        raise NormalizationError(f"no JSON repair specs in {fix_dir}")
    return normalized, refused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--fix-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_root, fix_dir, out_dir = args.source_root.resolve(), args.fix_dir.resolve(), args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite normalized output directory: {out_dir}")
    normalized, refused = normalize_all(source_root, fix_dir)
    report = {
        "source_root": str(source_root),
        "fix_dir": str(fix_dir),
        "out_dir": str(out_dir),
        "normalized": [str(path) for path, _ in normalized],
        "refused": refused,
        "dry_run": args.dry_run,
    }
    if refused:
        report["status"] = "REFUSED_UNRESOLVED_SPECS"
    else:
        report["status"] = "PASS"
    if not args.dry_run:
        if refused:
            raise NormalizationError(json.dumps(report, indent=2))
        out_dir.mkdir(parents=True)
        for source_path, spec in normalized:
            (out_dir / source_path.name).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out_dir / "normalization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
