"""Independently audit semantic-sweep compilation and normalized-fix composition.

This validator never writes a dataset.  It checks decision accounting, the
mechanical gap authority, normalized round-one replacement joins/rates, and the
compiler's ability to materialize a plan.  It writes its report before raising
on any finding so a failed review remains machine-readable.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from lerobot.annotation import diverse_semantic_sweep_compiler as compiler


Interval = tuple[int, int]
ATOM_TABLES = (
    "contact_atoms.jsonl",
    "precision_atoms.jsonl",
    "speed_atoms.jsonl",
    "speed_atoms_hybrid_v1.jsonl",
)
V2_TABLES = ("quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl")
KNOWN_RESOLUTIONS = {
    ("droid__AUTOLab__ep002365", "text"): {
        "precedence": "normalized_wins",
        "value": "grasp the black aluminium bar under the gripper",
        "reason": "sampled review uses the updated visible-discriminator case library",
    }
}



def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _merge(intervals: Iterable[Interval]) -> list[Interval]:
    output: list[list[int]] = []
    for start, stop in sorted((int(a), int(b)) for a, b in intervals):
        if start >= stop:
            raise compiler.SweepCompileError(f"invalid interval [{start}, {stop})")
        if output and start <= output[-1][1]:
            output[-1][1] = max(output[-1][1], stop)
        else:
            output.append([start, stop])
    return [(a, b) for a, b in output]


def _inside(bounds: Interval, intervals: Iterable[Interval]) -> bool:
    start, stop = bounds
    return any(lo <= start < stop <= hi for lo, hi in intervals)


def _confirmed(row: dict[str, Any]) -> bool:
    return row.get("status", row.get("review_status", row.get("outcome"))) == "confirmed"


def _decision_accounting(
    reports: dict[str, dict[str, Any]], edits: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    compiled = Counter(row["family"] for row in edits)
    expected: dict[str, int] = {}

    droid = reports["droid"]
    droid_expected = []
    for category in droid["reviews"].values():
        for row in category:
            if not _confirmed(row):
                continue
            if isinstance(row.get("correction_row"), dict):
                droid_expected.append(row["correction_row"])
            droid_expected.extend(row.get("correction_rows", []))
    droid_actual = [{k: v for k, v in row.items() if k not in {"kind", "family"}}
                    for row in edits if row["family"] == "droid"]
    if Counter(map(_canonical, droid_expected)) != Counter(map(_canonical, droid_actual)):
        errors.append({"code": "droid_confirmed_rows_not_exact", "family": "droid"})
    expected["droid"] = len(droid_expected)

    success = reports["droid_success"]
    success_count = sum(_confirmed(row) for row in success["gripper_event_boundaries"])
    success_count += sum(_confirmed(row) for row in success["short_quality_spans"])
    for row in success["ambiguous_repeated_text_groups"]:
        if not _confirmed(row):
            continue
        fix = row.get("exact_text_fix")
        if isinstance(fix, dict) and isinstance(fix.get("atom_text"), dict):
            success_count += len(fix["atom_text"])
        elif isinstance(fix, dict) and isinstance(fix.get("rows"), list):
            success_count += len(fix["rows"])
        else:
            success_count += len(row.get("atoms", []))
    expected["droid_success"] = success_count

    fmb = reports["fmb"]
    fmb_count = len(fmb["retention_fixes"]) + len(fmb["coalesced_quality_fixes"])
    for row in fmb["quality_candidate_reviews"]:
        if row["decision"] not in {"remove", "change"}:
            continue
        fix = row.get("exact_fix", {})
        if fix.get("action") == "apply_coalesced_fix":
            continue
        fmb_count += 1 + int(fix.get("action") == "replace_quality_row")
    expected["fmb"] = fmb_count

    molmo = reports["molmoact"]
    expected["molmoact"] = (
        sum(row.get("verdict") == "shift_to_completed_gripper_settle" for row in molmo["gripper_decisions"])
        + sum(len(row.get("exact_replacements", [])) for row in molmo["text_decisions"]
              if row.get("verdict") == "replace_repeated_text_with_visible_part_or_state_specific_text")
        + sum(row.get("verdict") == "split_atom_at_authoritative_gap" for row in molmo["gap_decisions"])
    )

    categories = reports["robochallenge"]["categories"]
    expected["robochallenge"] = (
        sum(len(row.get("atom_updates", [])) for row in categories["ambiguous_repeated_text"]["decisions"]
            if _confirmed(row))
        + sum(row.get("status") in {"confirmed", "rejected"}
              for row in categories["gripper_boundaries_before_settle"]["decisions"])
        + sum(_confirmed(row) for row in categories["atoms_crossing_static_gaps"]["decisions"])
        + sum(_confirmed(row) and row.get("operation") in {
                  "delete_quality_row", "replace_with_full_pause_pieces",
              }
              for row in categories["short_low_quality_spans"]["decisions"])
    )

    ur = reports["ur7e"]["decisions"]
    expected["ur7e"] = (
        len(ur["quality_removals"]["by_episode"])
        + len(ur["cross_gap_splits"]["records"])
        + len(ur["cross_gap_splits"]["replacement_retained_intervals"])
        + len(ur["boundary_shifts"]["records"])
    )
    expected["yam"] = sum(_confirmed(row) for row in reports["yam"]["decisions"])

    for family in compiler.REPORTS:
        if compiled[family] != expected[family]:
            errors.append({
                "code": "confirmed_edit_count_mismatch", "family": family,
                "expected": expected[family], "compiled": compiled[family],
            })
    duplicates = [json.loads(value) | {"copies": count} for value, count in
                  Counter(map(_canonical, edits)).items() if count != 1]
    if duplicates:
        errors.append({"code": "compiled_edit_duplicates", "count": len(duplicates)})
    rejected = [row for row in edits if row.get("rejected_candidate_reversion")
                or row.get("rejected_candidate_assertion")]
    return {
        "expected_plan_records": expected,
        "compiled_edits": dict(compiled),
        "confirmed_edit_records": len(edits) - len(rejected),
        "rejected_boundary_assertion_records": len(rejected),
        "exact_duplicate_groups": duplicates,
    }, errors


def _semantic_bounds(name: str, row: dict[str, Any]) -> Interval:
    if name == "quality_spans.jsonl":
        return int(row["raw_from_index"]), int(row["raw_to_index"])
    if name == "precision_windows.jsonl":
        return int(row["raw_from_index"]), int(row["to_index"])
    return int(row["from_index"]), int(row["to_index"])


def _validate_normalized_specs(
    normalized_dir: Path, containment: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    info = {str(row["episode_id"]): row for row in containment["episodes"]}
    specs: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    table_counts = Counter()
    for path in sorted(normalized_dir.glob("*.json")):
        spec = json.loads(path.read_text())
        episode_id = str(spec["episode_id"])
        if episode_id in specs:
            errors.append({"code": "duplicate_normalized_episode", "episode_id": episode_id})
            continue
        specs[episode_id] = spec
        episode = info.get(episode_id)
        if episode is None or str(episode["store"]) != str(spec["store"]):
            errors.append({"code": "normalized_episode_store_mismatch", "episode_id": episode_id})
            continue
        authority = [tuple(x) for x in episode["retained_intervals"]]
        actor = spec.get("actor_anchors", {})
        intervals = authority if actor.get("mode") == "preserve" else [tuple(x) for x in actor["retained_intervals"]]
        if any(not _inside(interval, authority) for interval in intervals):
            errors.append({
                "code": "normalized_retention_expands_authority", "episode_id": episode_id,
                "source_authority": authority, "normalized_retention": intervals,
            })
        replacements = spec.get("replacements", {})
        atoms = replacements.get("subtask_atoms.jsonl")
        if atoms is not None:
            atom_ranges = _merge((row["start_timestep"], row["end_timestep_exclusive"]) for row in atoms)
            if atom_ranges != _merge(intervals):
                errors.append({"code": "normalized_atoms_do_not_tile_retention", "episode_id": episode_id,
                               "atoms": atom_ranges, "retained": _merge(intervals)})
            atom_keys = {(int(row["parent_interval_index"]), int(row["atom_index"])) for row in atoms}
            if len(atom_keys) != len(atoms):
                errors.append({"code": "normalized_duplicate_atom_key", "episode_id": episode_id})
            for name in ATOM_TABLES:
                rows = replacements.get(name)
                if rows is None:
                    continue
                keys = {(int(row["parent_interval_index"]), int(row["atom_index"])) for row in rows}
                if len(rows) != len(keys) or keys != atom_keys:
                    errors.append({"code": "normalized_atom_sidecar_join_mismatch", "episode_id": episode_id,
                                   "table": name})
        rate = float(episode["native_rate_hz"])
        for name, rows in replacements.items():
            table_counts[name] += len(rows)
            for row in rows:
                if "native_rate_hz" in row and float(row["native_rate_hz"]) != rate:
                    errors.append({"code": "normalized_native_rate_mismatch", "episode_id": episode_id,
                                   "table": name})
                if name in V2_TABLES and not _inside(_semantic_bounds(name, row), intervals):
                    errors.append({"code": "normalized_v2_crosses_retention", "episode_id": episode_id,
                                   "table": name, "bounds": _semantic_bounds(name, row)})
    return specs, {"spec_count": len(specs), "replacement_rows": dict(table_counts)}, errors


def _quality_range(edit: dict[str, Any]) -> Interval | None:
    row = edit.get("row", edit)
    if "raw_from_index" in row and "raw_to_index" in row:
        return int(row["raw_from_index"]), int(row["raw_to_index"])
    return None


def _resolution_for(record: dict[str, Any]) -> dict[str, Any] | None:
    episode_id, kind = record["episode_id"], record["kind"]
    known = KNOWN_RESOLUTIONS.get((episode_id, kind))
    if known is not None:
        return known
    selector = record.get("selector", {})
    if kind == "boundary" and (
        episode_id == "molmoact__household__ep002456"
        or episode_id == "robochallenge__pick_out_the_green_blocks__ep000285"
        or episode_id == "ur7e__stack_block__ep000041"
    ):
        return {"precedence": "systematic_wins", "value": selector.get("new_boundary_frame"),
                "reason": "newer dense systematic boundary review"}
    if kind == "quality_delete_selector" and episode_id == "robochallenge__pick_out_the_green_blocks__ep000285":
        return {"precedence": "systematic_wins", "value": "delete_row",
                "reason": "newer dense systematic quality review"}
    if kind == "rejected_boundary" and episode_id == "robochallenge__pick_out_the_green_blocks__ep000285":
        return {"precedence": "systematic_rejection_keeps_source", "value": selector.get("expected_boundary"),
                "reason": "rejected systematic candidate forbids the sampled boundary shift"}
    return None


def _normalized_overlap(
    specs: dict[str, dict[str, Any]], edits: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for edit in edits:
        episode_id = str(edit["episode_id"])
        spec = specs.get(episode_id)
        if spec is None:
            continue
        replacements = spec.get("replacements", {})
        kind = edit["kind"]
        status, detail = "safe_disjoint", "normalized spec does not replace this channel"
        if kind == "boundary":
            atoms = replacements.get("subtask_atoms.jsonl", [])
            joins = {(int(a["end_timestep_exclusive"]), int(a["parent_interval_index"]), int(a["atom_index"]),
                      int(b["atom_index"]))
                     for a, b in zip(atoms, atoms[1:])
                     if int(a["end_timestep_exclusive"]) == int(b["start_timestep"])}
            old, new = int(edit["old_boundary_frame"]), int(edit["new_boundary_frame"])
            old_pairs = [(a, b) for a, b in zip(atoms, atoms[1:])
                         if int(a["end_timestep_exclusive"]) == old == int(b["start_timestep"])]
            if any(value[0] == new for value in joins):
                status, detail = "satisfied", f"desired join {new} already present"
            elif old_pairs:
                if any(int(left["start_timestep"]) < new < int(right["end_timestep_exclusive"])
                       for left, right in old_pairs):
                    status, detail = "safe_apply_after_normalized", f"move normalized join {old}->{new}"
                else:
                    status, detail = "conflict", f"moving join {old}->{new} would create an empty atom"
            elif any(int(row["end_timestep_exclusive"]) == new for row in atoms):
                status, detail = "conflict", f"desired boundary {new} is an atom/retention edge, not a join"
            else:
                status, detail = "conflict", f"neither old join {old} nor desired join {new} exists"
        elif kind == "text":
            atoms = replacements.get("subtask_atoms.jsonl", [])
            desired = str(edit["new_subtask"])
            old = edit.get("old_subtask")
            if any(row.get("subtask") == desired for row in atoms):
                status, detail = "satisfied", "desired text already present"
            elif any((old is None or row.get("subtask") == old)
                     and ("start_timestep" not in edit or int(row["start_timestep"]) == int(edit["start_timestep"]))
                     and ("end_timestep_exclusive" not in edit or int(row["end_timestep_exclusive"]) == int(edit["end_timestep_exclusive"]))
                     for row in atoms):
                status, detail = "safe_apply_after_normalized", "old text/range remains uniquely addressable"
            else:
                status, detail = "conflict", "normalized atom identity/text no longer matches edit selector"
        elif kind == "quality_delete_max_duration":
            rows = replacements.get("quality_spans.jsonl")
            if rows == []:
                status, detail = "satisfied", "normalized replacement already contains no quality rows"
            else:
                status, detail = "conflict", "duration-wide deletion needs explicit normalized precedence"
        elif kind.startswith("quality_delete"):
            target = _quality_range(edit)
            rows = replacements.get("quality_spans.jsonl", [])
            if target is None:
                status, detail = "conflict", "duration-wide deletion needs explicit normalized precedence"
            else:
                exact = [row for row in rows if _semantic_bounds("quality_spans.jsonl", row) == target]
                if exact:
                    status, detail = "conflict", "semantic deletion targets a normalized replacement row"
                else:
                    status, detail = "satisfied", "exact deleted source row is absent; normalized rows survive"
        elif kind == "retained_intervals":
            actor = spec.get("actor_anchors", {})
            current = actor.get("retained_intervals")
            if current == edit["intervals"]:
                status, detail = "satisfied", "retention intervals agree"
            else:
                status, detail = "conflict", "semantic retention would replace normalized retention"
        elif kind == "retained_end":
            status, detail = "conflict", "retention trim needs explicit normalized precedence"
        elif kind.startswith("assert_structural"):
            status, detail = "satisfied", "assertion is checked against normalized retention/atoms"
        selector = {key: edit[key] for key in (
            "parent_interval_index", "atom_index", "left_atom_index", "right_atom_index",
            "old_boundary_frame", "new_boundary_frame", "old_subtask", "new_subtask",
            "start_timestep", "end_timestep_exclusive", "raw_from_index", "raw_to_index",
            "max_native_frames", "expected_count",
        ) if key in edit}
        if kind == "quality_delete_exact":
            selector = {key: edit["row"][key] for key in (
                "uid", "raw_from_index", "raw_to_index", "quality", "cause",
            ) if key in edit["row"]}
        record = {"episode_id": episode_id, "family": edit["family"], "kind": kind,
                  "status": status, "detail": detail, "selector": selector}
        results.append(record)
        if status == "conflict":
            record["resolution"] = _resolution_for(record)
            conflicts.append(record)
            if record["resolution"] is None:
                unresolved.append(record)
    return results, conflicts, unresolved


def _rejected_boundary_overlap(
    reports: dict[str, dict[str, Any]], specs: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row in reports["droid"]["reviews"]["gripper_boundaries_before_settle"]:
        if row.get("status") == "rejected":
            candidates.append({"family": "droid", "episode_id": row["episode_id"],
                               "parent": row["parent_interval_index"], "left": row["left_atom_index"],
                               "right": row["right_atom_index"], "expected": row["old_boundary_frame"]})
    for row in reports["robochallenge"]["categories"]["gripper_boundaries_before_settle"]["decisions"]:
        if row.get("status") == "rejected":
            candidates.append({"family": "robochallenge", "episode_id": row["episode_id"],
                               "parent": row["parent_interval_index"], "left": row["left_atom_index"],
                               "right": row["right_atom_index"], "expected": row["current_boundary_frame"]})
    for row in reports["yam"]["decisions"]:
        if row.get("outcome") == "rejected" and row.get("category") == "pre_settle_gripper_boundary":
            item = row["candidate"]
            candidates.append({"family": "yam", "episode_id": row["episode_id"],
                               "parent": item["parent_interval_index"], "left": item["left_atom_index"],
                               "right": item["right_atom_index"], "expected": item["boundary_frame"]})
    results: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for candidate in candidates:
        spec = specs.get(str(candidate["episode_id"]))
        if spec is None:
            continue
        atoms = spec.get("replacements", {}).get("subtask_atoms.jsonl", [])
        joins = {int(left["end_timestep_exclusive"]) for left, right in zip(atoms, atoms[1:])
                 if int(left["end_timestep_exclusive"]) == int(right["start_timestep"])}
        expected = int(candidate["expected"])
        target_left = next((row for row in atoms if int(row["parent_interval_index"]) == int(candidate["parent"])
                            and int(row["atom_index"]) == int(candidate["left"])), None)
        target_right = next((row for row in atoms if int(row["parent_interval_index"]) == int(candidate["parent"])
                             and int(row["atom_index"]) == int(candidate["right"])), None)
        current = None
        if target_left is not None and target_right is not None:
            if int(target_left["end_timestep_exclusive"]) == int(target_right["start_timestep"]):
                current = int(target_left["end_timestep_exclusive"])
        conflict = expected not in joins and current is not None and current != expected
        record = {"episode_id": candidate["episode_id"], "family": candidate["family"],
                  "kind": "rejected_boundary", "status": "conflict" if conflict else "satisfied",
                  "selector": {"parent_interval_index": candidate["parent"],
                               "left_atom_index": candidate["left"], "right_atom_index": candidate["right"],
                               "expected_boundary": expected, "normalized_boundary": current},
                  "detail": "normalized shift conflicts with rejected systematic candidate" if conflict
                  else "rejected candidate keeps the source boundary"}
        if conflict:
            record["resolution"] = _resolution_for(record)
            conflicts.append(record)
            if record["resolution"] is None:
                unresolved.append(record)
        results.append(record)
    return results, conflicts, unresolved


def review(
    source_root: Path, sweep_dir: Path, containment_path: Path, normalized_dir: Path
) -> dict[str, Any]:
    reports, containment, hashes = compiler.load_and_validate(source_root, sweep_dir, containment_path)
    edits = compiler.collect_edits(reports)
    compiler_path = Path(inspect.getfile(compiler)).resolve()
    hashes[str(compiler_path)] = compiler._sha256(compiler_path)
    decision, decision_errors = _decision_accounting(reports, edits)
    specs, normalized, normalized_errors = _validate_normalized_specs(normalized_dir, containment)
    overlaps, conflicts, unresolved_conflicts = _normalized_overlap(specs, edits)
    rejected, rejected_conflicts, rejected_unresolved = _rejected_boundary_overlap(reports, specs)
    errors = decision_errors + normalized_errors
    sampled_manifest = source_root / "corrected_store_manifest.json"
    if not sampled_manifest.is_file():
        errors.append({"code": "compiler_source_is_not_a_corrected_sampled_stage"})
    else:
        sampled = json.loads(sampled_manifest.read_text())
        if Path(sampled.get("fix_dir", "")).resolve() != normalized_dir.resolve():
            errors.append({
                "code": "compiler_sampled_stage_normalized_fix_mismatch",
                "manifest_fix_dir": sampled.get("fix_dir"), "normalized_dir": str(normalized_dir),
            })
    materialization = {"status": "not_run"}
    try:
        composition_ledger: list[dict[str, Any]] = []
        built = compiler.build_specs(source_root, containment, edits, composition_ledger)
        materialization = {
            "status": "pass", "spec_count": len(built),
            "composition_ledger_count": len(composition_ledger),
        }
    except Exception as exc:  # report the fail-closed point verbatim
        materialization = {"status": "fail", "exception": type(exc).__name__, "message": str(exc)}
        errors.append({"code": "compiler_materialization_failed", **materialization})
    if unresolved_conflicts:
        errors.append({"code": "normalized_semantic_precedence_conflicts", "count": len(unresolved_conflicts)})
    if rejected_unresolved:
        errors.append({"code": "normalized_rejected_decision_conflicts", "count": len(rejected_unresolved)})
    return {
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "input_hashes": hashes | {str(path): compiler._sha256(path) for path in sorted(normalized_dir.glob("*.json"))},
        "decision_accounting": decision,
        "compiled_edit_count": len(edits),
        "compiled_edits_by_kind": dict(Counter(row["kind"] for row in edits)),
        "normalized_specs": normalized,
        "normalized_overlap": {
            "edit_count": len(overlaps), "status_counts": dict(Counter(row["status"] for row in overlaps)),
            "conflicts": conflicts, "unresolved_conflicts": unresolved_conflicts,
        },
        "rejected_decision_overlap": {
            "reviewed": len(rejected), "conflicts": rejected_conflicts,
            "unresolved_conflicts": rejected_unresolved,
        },
        "compiler_materialization": materialization,
        "errors": errors,
        "open_candidates": errors,
    }


def _markdown(report: dict[str, Any]) -> str:
    overlap = report["normalized_overlap"]
    lines = [
        "# Semantic compiler independent composition review", "",
        f"- Status: **{report['status']}**.",
        f"- Compiled semantic edits: {report['compiled_edit_count']}.",
        f"- Normalized round-one specs: {report['normalized_specs']['spec_count']}.",
        f"- Overlapping semantic edits: {overlap['edit_count']} ({overlap['status_counts']}).",
        f"- Open candidates: {len(report['open_candidates'])}.", "",
        "## Fail-closed findings", "",
    ]
    lines.extend(f"- `{row['code']}`: `{json.dumps(row, sort_keys=True)}`" for row in report["errors"])
    lines += ["", "## Normalized precedence conflicts", ""]
    lines.extend(
        f"- `{row['episode_id']}` `{row['kind']}` `{json.dumps(row['selector'], sort_keys=True)}`: "
        f"{row['detail']} Resolution: `{json.dumps(row.get('resolution'), sort_keys=True)}`."
        for row in overlap["conflicts"]
    )
    lines += ["", "## Rejected-decision precedence conflicts", ""]
    lines.extend(
        f"- `{row['episode_id']}` `{json.dumps(row['selector'], sort_keys=True)}`: "
        f"{row['detail']} Resolution: `{json.dumps(row.get('resolution'), sort_keys=True)}`."
        for row in report["rejected_decision_overlap"]["conflicts"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--containment-manifest", required=True, type=Path)
    parser.add_argument("--normalized-fix-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    report = review(
        args.source_root.resolve(), args.sweep_dir.resolve(), args.containment_manifest.resolve(),
        args.normalized_fix_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.summary.write_text(_markdown(report))
    print(json.dumps({"status": report["status"], "open_candidates": len(report["open_candidates"]),
                      "conflicts": len(report["normalized_overlap"]["conflicts"]),
                      "unresolved_conflicts": len(report["normalized_overlap"]["unresolved_conflicts"])}))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
