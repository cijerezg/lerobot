"""Compile heterogeneous semantic-sweep decisions into exact edit operations.

The compiler is intentionally fail-closed: every required family report and the
mechanical containment manifest must be present, source hashes must still match,
and no report may contain an open decision.  Family adapters emit only confirmed,
machine-resolvable edits; free-form prose is never interpreted as an operation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.annotation import diverse_fix_normalizer_v2 as fix_normalizer
from lerobot.annotation.diverse_containment_sweep import remap_v2_row


REPORTS = {
    "droid": "droid_semantic_sweep.json",
    "droid_success": "droid_success_semantic_sweep.json",
    "fmb": "fmb_semantic_sweep.json",
    "molmoact": "molmoact_semantic_sweep.json",
    "robochallenge": "robochallenge_semantic_sweep.json",
    "ur7e": "ur7e_semantic_sweep.json",
    "yam": "yam_semantic_sweep.json",
}


class SweepCompileError(ValueError):
    """The semantic correction set is incomplete or not mechanically exact."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SweepCompileError(f"required audit input is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SweepCompileError(f"{path}: top level must be an object")
    return value


def _open_count(family: str, report: dict[str, Any]) -> int:
    if family == "droid":
        return int(report.get("counts", {}).get("open", -1))
    if family == "droid_success":
        return int(report.get("counts", {}).get("open", -1))
    if family == "fmb":
        return int(report.get("counts", {}).get("open", -1))
    if family == "molmoact":
        return int(report.get("summary", {}).get("open_items", -1))
    if family == "robochallenge":
        decisions = [
            row for category in report.get("categories", {}).values()
            for row in category.get("decisions", [])
        ]
        unresolved = sum(row.get("status") == "open" or bool(row.get("open_detail")) for row in decisions)
        if report.get("status") != "complete" or report.get("second_reader_resolution", {}).get("status") != "complete":
            unresolved += 1
        return unresolved
    if family == "ur7e":
        return max(len(report.get("opens", [])), int(report.get("summary_counts", {}).get("open_items", -1)))
    if family == "yam":
        return sum(bool(row.get("open")) for row in report.get("decisions", []))
    raise SweepCompileError(f"unknown family {family}")


def load_and_validate(
    source_root: Path, sweep_dir: Path, containment_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str]]:
    reports, hashes, opens = {}, {}, {}
    for family, filename in REPORTS.items():
        path = sweep_dir / filename
        report = _load(path)
        reports[family] = report
        hashes[str(path)] = _sha256(path)
        count = _open_count(family, report)
        if count:
            opens[family] = count
    containment = _load(containment_path)
    hashes[str(containment_path)] = _sha256(containment_path)
    structural_open = len(containment.get("open_candidates", []))
    if structural_open or int(containment.get("validation", {}).get("open_candidates", -1)) != 0:
        opens["mechanical_containment"] = max(structural_open, 1)
    if opens:
        raise SweepCompileError(f"open audit decisions remain: {opens}")
    for relative, expected in containment.get("source_hashes", {}).items():
        path = source_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise SweepCompileError(f"source hash mismatch: {relative}")
    return reports, containment, hashes


def _key(value: str) -> tuple[int, int]:
    if not value.startswith("p") or "a" not in value:
        raise SweepCompileError(f"invalid atom key {value!r}")
    parent, atom = value[1:].split("a", 1)
    return int(parent), int(atom)


def _confirmed(row: dict[str, Any]) -> bool:
    return row.get("status", row.get("review_status", row.get("outcome"))) == "confirmed"


def collect_edits(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []

    droid = reports["droid"]["correction_rows"]
    for row in droid["gripper_boundaries"]:
        edits.append({"kind": "boundary", "family": "droid", **row})
    for row in droid["text"]:
        edits.append({"kind": "text", "family": "droid", **row})
    for row in droid["quality_span_deletions"]:
        edits.append({"kind": "quality_delete_selector", "family": "droid", **row})

    success = reports["droid_success"]
    for row in success["gripper_event_boundaries"]:
        if _confirmed(row):
            fix = row.get("exact_frame_fix")
            if not isinstance(fix, dict) or "proposed_shared_boundary_frame" not in fix:
                raise SweepCompileError("confirmed DROID-success boundary lacks exact_frame_fix")
            edits.append({
                "kind": "boundary", "family": "droid_success", "episode_id": row["episode_id"],
                "parent_interval_index": row["parent_interval_index"],
                "old_boundary_frame": fix["current_shared_boundary_frame"],
                "new_boundary_frame": fix["proposed_shared_boundary_frame"],
            })
    for row in success["ambiguous_repeated_text_groups"]:
        if _confirmed(row):
            old = row.get("current_text", row.get("current_subtask", row.get("candidate_subtask", row.get("repeated_text"))))
            fix = row.get("exact_text_fix")
            replacements = []
            if isinstance(fix, dict) and isinstance(fix.get("atom_text"), dict):
                replacements = [{"atom_key": key, "proposed_subtask": value, "action": "rename"}
                                for key, value in fix["atom_text"].items()]
            elif isinstance(fix, dict) and isinstance(fix.get("rows"), list):
                replacements = fix["rows"]
            else:
                replacements = []
                for atom in row.get("atoms", []):
                    atom_key = atom.get("atom_key")
                    if atom_key is None:
                        atom_key = f"p{atom['parent_interval_index']}a{atom['atom_index']}"
                    replacements.append({"atom_key": atom_key,
                                         "proposed_subtask": atom.get("proposed_text", atom.get("proposed_subtask")),
                                         "action": "rename"})
            if not replacements:
                raise SweepCompileError("confirmed DROID-success text edit has no exact replacements")
            for replacement in replacements:
                parent, atom_index = _key(replacement["atom_key"])
                if replacement.get("action") == "remove_as_fake_regrasp_and_merge_adjacent_unfold_atoms":
                    edits.append({"kind": "merge_fake_atom", "family": "droid_success",
                                  "episode_id": row["episode_id"], "parent_interval_index": parent,
                                  "atom_index": atom_index})
                    continue
                proposed = replacement.get("proposed_subtask")
                if not isinstance(proposed, str) or not proposed:
                    raise SweepCompileError("confirmed DROID-success rename lacks proposed_subtask")
                edits.append({
                    "kind": "text", "family": "droid_success", "episode_id": row["episode_id"],
                    "parent_interval_index": parent, "atom_index": atom_index,
                    "old_subtask": old, "new_subtask": proposed,
                })
    for row in success["short_quality_spans"]:
        if _confirmed(row):
            edits.append({
                "kind": "quality_delete_selector", "family": "droid_success", "episode_id": row["episode_id"],
                "raw_from_index": row["native_half_open_frames"][0],
                "raw_to_index": row["native_half_open_frames"][1],
            })

    fmb = reports["fmb"]
    for row in fmb["quality_candidate_reviews"]:
        if row["decision"] in {"remove", "change"}:
            fix = row.get("exact_fix", {})
            if fix.get("action") == "apply_coalesced_fix":
                continue
            exact = fix.get("remove_exact")
            if not isinstance(exact, dict):
                raise SweepCompileError("confirmed FMB quality edit lacks remove_exact")
            edits.append({"kind": "quality_delete_exact", "family": "fmb", "episode_id": row["episode_id"], "row": exact})
            if fix.get("action") == "replace_quality_row":
                addition = fix.get("add_exact")
                if not isinstance(addition, dict):
                    raise SweepCompileError("FMB replace_quality_row lacks add_exact")
                edits.append({"kind": "quality_add_exact", "family": "fmb",
                              "episode_id": row["episode_id"], "row": addition})
    for row in fmb["retention_fixes"]:
        edits.append({"kind": "retained_end", "family": "fmb", **row})
    for row in fmb["coalesced_quality_fixes"]:
        edits.append({"kind": "quality_coalesce", "family": "fmb", **row})

    molmo = reports["molmoact"]
    for row in molmo["gripper_decisions"]:
        if row.get("verdict") == "shift_to_completed_gripper_settle":
            edits.append({
                "kind": "boundary", "family": "molmoact", "episode_id": row["episode_id"],
                "parent_interval_index": row["parent_interval_index"],
                "left_atom_index": row["left_atom_index"], "right_atom_index": row["right_atom_index"],
                "old_boundary_frame": row["old_boundary_frame"], "new_boundary_frame": row["confirmed_boundary_frame"],
            })
    for row in molmo["text_decisions"]:
        if row.get("verdict") == "replace_repeated_text_with_visible_part_or_state_specific_text":
            for replacement in row.get("exact_replacements", []):
                edits.append({"kind": "text", "family": "molmoact", "episode_id": row["episode_id"], **replacement})
    for row in molmo["gap_decisions"]:
        if row.get("verdict") == "split_atom_at_authoritative_gap":
            edits.append({"kind": "assert_structural_gap", "family": "molmoact", "episode_id": row["episode_id"], **row["exact_fix"]})

    rc = reports["robochallenge"]["categories"]
    for row in rc["ambiguous_repeated_text"]["decisions"]:
        if _confirmed(row):
            for update in row.get("atom_updates", []):
                parent, atom = _key(update["atom_key"])
                edits.append({"kind": "text", "family": "robochallenge", "episode_id": row["episode_id"],
                              "parent_interval_index": parent, "atom_index": atom,
                              "old_subtask": update["from"], "new_subtask": update["to"]})
    for row in rc["gripper_boundaries_before_settle"]["decisions"]:
        if _confirmed(row):
            edits.append({"kind": "boundary", "family": "robochallenge", **row,
                          "old_boundary_frame": row["current_boundary_frame"],
                          "new_boundary_frame": row["proposed_boundary_frame"]})
        elif row.get("status") == "rejected":
            # A rejected shift is also authoritative: the source boundary is
            # correct. This assertion reverses any older sampled shift on the
            # same keyed join and otherwise remains idempotent.
            if row.get("rejected_intervals_hit"):
                edits.append({
                    "kind": "assert_boundary_superseded_by_gap", "family": "robochallenge", **row,
                    "rejected_candidate_assertion": True,
                })
            else:
                edits.append({"kind": "boundary", "family": "robochallenge", **row,
                              "old_boundary_frame": row["current_boundary_frame"],
                              "new_boundary_frame": row["current_boundary_frame"],
                              "rejected_candidate_reversion": True})
    for row in rc["atoms_crossing_static_gaps"]["decisions"]:
        if _confirmed(row):
            edits.append({"kind": "assert_structural_fragments", "family": "robochallenge", **row})
    for row in rc["short_low_quality_spans"]["decisions"]:
        if not _confirmed(row):
            continue
        if row.get("operation") == "delete_quality_row":
            edits.append({"kind": "quality_delete_selector", "family": "robochallenge", **row["current"],
                          "episode_id": row["episode_id"], "uid": row["uid"],
                          "quality": row["quality"], "cause": row["cause"]})
        elif row.get("operation") == "replace_with_full_pause_pieces":
            replacements = row.get("replacement_rows")
            if not isinstance(replacements, list):
                raise SweepCompileError("confirmed RoboChallenge pause restoration lacks exact replacement_rows")
            edits.append({
                "kind": "quality_restore_full_pause", "family": "robochallenge",
                "episode_id": row["episode_id"], "uid": row["uid"],
                "parent_interval_index": row["parent_interval_index"],
                "atom_index": row["atom_index"], "quality": row["quality"],
                "current": row["current"], "replacement_rows": replacements,
            })

    ur = reports["ur7e"]["decisions"]
    for episode, count in ur["quality_removals"]["by_episode"].items():
        edits.append({"kind": "quality_delete_max_duration", "family": "ur7e", "episode_id": episode,
                      "max_native_frames": 30, "expected_count": count})
    for row in ur["cross_gap_splits"]["records"]:
        edits.append({"kind": "assert_structural_gap", "family": "ur7e", **row})
    for episode, intervals in ur["cross_gap_splits"]["replacement_retained_intervals"].items():
        edits.append({"kind": "retained_intervals", "family": "ur7e", "episode_id": episode, "intervals": intervals})
    for row in ur["boundary_shifts"]["records"]:
        left_parent, left_atom = _key(row["left_atom"])
        right_parent, right_atom = _key(row["right_atom"])
        edits.append({"kind": "boundary", "family": "ur7e", **row,
                      "parent_interval_index": left_parent, "left_atom_index": left_atom,
                      "right_parent_interval_index": right_parent, "right_atom_index": right_atom,
                      "old_boundary_frame": row["old_boundary"], "new_boundary_frame": row["new_boundary"]})

    for row in reports["yam"]["decisions"]:
        if not _confirmed(row):
            continue
        if row["category"] == "pre_settle_gripper_boundary":
            candidate = row["candidate"]
            edits.append({"kind": "boundary", "family": "yam", "episode_id": row["episode_id"],
                          "parent_interval_index": candidate["parent_interval_index"],
                          "left_atom_index": candidate["left_atom_index"], "right_atom_index": candidate["right_atom_index"],
                          "old_boundary_frame": candidate["boundary_frame"], "new_boundary_frame": row["new_boundary_frame"]})
        elif row["category"] == "short_quality_span":
            edits.append({"kind": "quality_delete_exact", "family": "yam", "episode_id": row["episode_id"], "row": row["candidate"]})
        elif row["category"] == "declared_gap":
            edits.append({"kind": "assert_structural_gap", "family": "yam", "episode_id": row["episode_id"],
                          "gap": row["candidate"]["gap_frame_range"]})
    return edits


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _atom_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["episode_id"]), int(row["parent_interval_index"]), int(row["atom_index"])


def _update_bounds(row: dict[str, Any], start: int, stop: int, rate: float) -> None:
    row["start_timestep"], row["end_timestep_exclusive"] = start, stop
    if "start_s" in row:
        row["start_s"] = start / rate
    if "end_s_exclusive" in row:
        row["end_s_exclusive"] = stop / rate
    if "duration_s" in row:
        row["duration_s"] = (stop - start) / rate


def materialize_atoms(
    source_rows: list[dict[str, Any]], operations: list[dict[str, Any]], episode_id: str
) -> list[dict[str, Any]]:
    """Apply the containment manifest's complete source-to-target atom ledger."""
    output = []
    for operation in operations:
        if operation["source_atom_key"][0] != episode_id:
            continue
        source = dict(source_rows[int(operation["source_line"]) - 1])
        if _atom_key(source) != tuple(operation["source_atom_key"]):
            raise SweepCompileError(f"{episode_id}: atom source_line/key mismatch")
        for target in operation["targets"]:
            row = dict(source)
            key = target["target_atom_key"]
            row.update(episode_id=key[0], parent_interval_index=int(key[1]), atom_index=int(key[2]),
                       subtask=target["subtask"])
            _update_bounds(row, int(target["start_timestep"]), int(target["end_timestep_exclusive"]),
                           float(row["native_rate_hz"]))
            row["_source_atom_key"] = list(operation["source_atom_key"])
            output.append(row)
    return sorted(output, key=lambda row: (row["start_timestep"], row["parent_interval_index"], row["atom_index"]))


def materialize_v2(
    source_rows: list[dict[str, Any]], operations: list[dict[str, Any]], episode_id: str
) -> list[dict[str, Any]]:
    """Apply explicit target field overrides while preserving unmentioned source fields."""
    output = []
    for operation in operations:
        if operation["episode_id"] != episode_id:
            continue
        source = dict(source_rows[int(operation["source_line"]) - 1])
        if str(source.get("uid")) != str(operation["source_uid"]):
            raise SweepCompileError(f"{episode_id}: v2 source_line/uid mismatch")
        for target in operation["targets"]:
            row = dict(source)
            row.update(target["fields"])
            row["_source_line"] = int(operation["source_line"])
            output.append(row)
    return output


def _edit_atom_matches(atom: dict[str, Any], edit: dict[str, Any], side: str | None = None) -> bool:
    source = atom.get("_source_atom_key", _atom_key(atom))
    parent_field = "right_parent_interval_index" if side == "right" else "parent_interval_index"
    atom_field = f"{side}_atom_index" if side else "atom_index"
    parent = edit.get(parent_field, edit.get("parent_interval_index"))
    index = edit.get(atom_field)
    return (parent is None or int(source[1]) == int(parent)) and (index is None or int(source[2]) == int(index))


def apply_atom_edits(
    atoms: list[dict[str, Any]], edits: list[dict[str, Any]], episode_id: str,
    composition_conflicts: list[dict[str, Any]] | None = None,
    sampled_episode: bool = False,
    retained_intervals: list[list[int]] | None = None,
) -> None:
    for edit in edits:
        kind = edit["kind"]
        if kind == "boundary":
            old, new = int(edit["old_boundary_frame"]), int(edit["new_boundary_frame"])
            left = [row for row in atoms if int(row["end_timestep_exclusive"]) == old and _edit_atom_matches(row, edit, "left")]
            right = [row for row in atoms if int(row["start_timestep"]) == old and _edit_atom_matches(row, edit, "right")]
            if edit.get("rejected_candidate_reversion") and old == new and len(left) == len(right) == 1:
                continue
            if edit.get("rejected_candidate_reversion") and not sampled_episode:
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id, "field": "shared_atom_boundary",
                        "rejected_source_boundary": old,
                        "materialized_left_matches": len(left),
                        "materialized_right_matches": len(right),
                        "resolution": "containment_proven_original_join_not_materializable_assertion_noop",
                    })
                continue
            if not left and not right:
                # Rekeying in the sampled stage can move the audit's source
                # atom indices while preserving the exact native boundary.
                left = [row for row in atoms if int(row["end_timestep_exclusive"]) == old]
                right = [row for row in atoms if int(row["start_timestep"]) == old]
            if not left and not right:
                already_left = [row for row in atoms if int(row["end_timestep_exclusive"]) == new]
                already_right = [row for row in atoms if int(row["start_timestep"]) == new]
                if len(already_left) == len(already_right) == 1:
                    continue
                keyed_left = [row for row in atoms if _edit_atom_matches(row, edit, "left")]
                keyed_right = [row for row in atoms if _edit_atom_matches(row, edit, "right")]
                if keyed_left and keyed_right and retained_intervals:
                    if edit.get("rejected_candidate_reversion"):
                        last_left = max(keyed_left, key=lambda row: int(row["end_timestep_exclusive"]))
                        first_right = min(keyed_right, key=lambda row: int(row["start_timestep"]))
                        structural_override = any(
                            candidate["kind"] == "assert_structural_fragments"
                            and (
                                int(candidate.get("atom_index", -1)) == int(edit.get("right_atom_index", -2))
                                or any(
                                    int(update.get("atom_index", -1)) == int(edit.get("left_atom_index", -2))
                                    for update in candidate.get("second_reader_resolution", {}).get("neighbor_updates", [])
                                )
                            )
                            for candidate in edits
                        )
                        if structural_override and int(last_left["end_timestep_exclusive"]) < int(first_right["start_timestep"]):
                            if composition_conflicts is not None:
                                composition_conflicts.append({
                                    "episode_id": episode_id, "field": "shared_atom_boundary",
                                    "rejected_source_boundary": old,
                                    "retained_pre_gap_edge": int(last_left["end_timestep_exclusive"]),
                                    "post_gap_start": int(first_right["start_timestep"]),
                                    "resolution": "confirmed_structural_second_reader_supersedes_continuous_boundary_assertion",
                                })
                            continue
                    gap_left = max(
                        (row for row in keyed_left if int(row["end_timestep_exclusive"]) <= new),
                        key=lambda row: int(row["end_timestep_exclusive"]), default=None,
                    )
                    gap_right = min(
                        (row for row in keyed_right if int(row["start_timestep"]) >= new),
                        key=lambda row: int(row["start_timestep"]), default=None,
                    )
                    run = next(
                        (interval for interval in retained_intervals
                         if int(interval[0]) <= old < int(interval[1])), None,
                    )
                    if gap_left is not None and gap_right is not None and run is not None:
                        retained_edge = int(run[1])
                        if int(gap_left["end_timestep_exclusive"]) == new and retained_edge == new:
                            continue
                        if int(gap_left["end_timestep_exclusive"]) == old and retained_edge <= int(gap_right["start_timestep"]):
                            clamped = min(new, retained_edge)
                            if not int(gap_left["start_timestep"]) < clamped:
                                raise SweepCompileError(f"{episode_id}: hard-gap clamp empties the left atom")
                            _update_bounds(
                                gap_left, int(gap_left["start_timestep"]), clamped,
                                float(gap_left["native_rate_hz"]),
                            )
                            if composition_conflicts is not None:
                                composition_conflicts.append({
                                    "episode_id": episode_id, "field": "shared_atom_boundary",
                                    "systematic_proposed_value": new, "retained_edge": retained_edge,
                                    "post_gap_atom": [
                                        int(gap_right["parent_interval_index"]), int(gap_right["atom_index"])
                                    ],
                                    "resolution": "sweep_fix_clamped_by_hard_gap_preserving_post_gap_subtask",
                                })
                            continue
                if len(keyed_left) == len(keyed_right) == 1 and (
                    int(keyed_left[0]["end_timestep_exclusive"])
                    == int(keyed_right[0]["start_timestep"])
                ):
                    sampled_boundary = int(keyed_left[0]["end_timestep_exclusive"])
                    if sampled_boundary == new:
                        continue
                    if sampled_episode and composition_conflicts is not None:
                        composition_conflicts.append({
                            "episode_id": episode_id,
                            "field": "shared_atom_boundary",
                            "atom_keys": [
                                [int(keyed_left[0]["parent_interval_index"]), int(keyed_left[0]["atom_index"])],
                                [int(keyed_right[0]["parent_interval_index"]), int(keyed_right[0]["atom_index"])],
                            ],
                            "sampled_value": sampled_boundary,
                            "systematic_expected_old": old,
                            "systematic_proposed_value": new,
                            "resolution": (
                                "systematic_rejection_restores_source_boundary_over_sampled_fix"
                                if edit.get("rejected_candidate_reversion")
                                else "newer_dense_systematic_boundary_supersedes_sampled_fix"
                            ),
                        })
                        left, right = keyed_left, keyed_right
            if not left and len(right) == 1 and retained_intervals:
                keyed_left = [row for row in atoms if _edit_atom_matches(row, edit, "left")]
                prior = max(keyed_left, key=lambda row: int(row["end_timestep_exclusive"]), default=None)
                post_gap_run = next(
                    (interval for interval in retained_intervals if int(interval[0]) == old), None
                )
                if (
                    prior is not None and post_gap_run is not None
                    and int(prior["end_timestep_exclusive"]) < old
                    and old < new < int(right[0]["end_timestep_exclusive"])
                ):
                    # The shared source join falls exactly after a hard gap.
                    # Preserve the gap, create a short post-gap fragment for
                    # the settling left subtask, and move only the right start.
                    fragment = copy.deepcopy(prior)
                    fragment["parent_interval_index"] = int(right[0]["parent_interval_index"])
                    for parent_field in ("parent_critic_eligible", "parent_note"):
                        if parent_field in right[0]:
                            fragment[parent_field] = right[0][parent_field]
                    _update_bounds(fragment, old, new, float(fragment["native_rate_hz"]))
                    _update_bounds(
                        right[0], new, int(right[0]["end_timestep_exclusive"]),
                        float(right[0]["native_rate_hz"]),
                    )
                    atoms.append(fragment)
                    atoms.sort(key=lambda row: int(row["start_timestep"]))
                    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for row in atoms:
                        by_parent[int(row["parent_interval_index"])].append(row)
                    for rows in by_parent.values():
                        for index, row in enumerate(rows):
                            row["atom_index"] = index
                    if composition_conflicts is not None:
                        composition_conflicts.append({
                            "episode_id": episode_id, "field": "shared_atom_boundary",
                            "hard_gap": [int(prior["end_timestep_exclusive"]), old],
                            "post_gap_settle_fragment": [old, new],
                            "resolution": "split_post_gap_settle_without_bridging_hard_gap",
                        })
                    continue
            if not left and not right:
                already_left = [row for row in atoms if int(row["end_timestep_exclusive"]) == new]
                already_right = [row for row in atoms if int(row["start_timestep"]) == new]
                if len(already_left) == len(already_right) == 1:
                    continue
            if len(left) != 1 or len(right) != 1:
                raise SweepCompileError(f"{episode_id}: boundary {old} matched left={len(left)} right={len(right)}")
            if sampled_episode and composition_conflicts is not None and not any(
                row.get("episode_id") == episode_id
                and row.get("field") == "shared_atom_boundary"
                and row.get("systematic_proposed_value") == new
                for row in composition_conflicts
            ):
                composition_conflicts.append({
                    "episode_id": episode_id,
                    "field": "shared_atom_boundary",
                    "sampled_value": old,
                    "systematic_proposed_value": new,
                    "resolution": (
                        "systematic_rejection_restores_source_boundary_over_sampled_fix"
                        if edit.get("rejected_candidate_reversion")
                        else "newer_dense_systematic_boundary_applied_to_sampled_episode"
                    ),
                })
            if new >= int(right[0]["end_timestep_exclusive"]):
                # A confirmed native-frame settle can fall just inside a
                # source-excluded gap.  In the retained dataset that consumes
                # the entire short right atom: extend the left atom only to the
                # authoritative retained edge and remove the empty right atom.
                parent = int(right[0]["parent_interval_index"])
                same_parent_later = [row for row in atoms if int(row["parent_interval_index"]) == parent
                                     and int(row["start_timestep"]) >= int(right[0]["end_timestep_exclusive"])]
                if same_parent_later:
                    raise SweepCompileError(f"{episode_id}: boundary {new} crosses a nonterminal atom")
                post_gap = sorted(
                    (row for row in atoms if int(row["start_timestep"]) >= new),
                    key=lambda row: int(row["start_timestep"]),
                )
                if not post_gap or post_gap[0]["subtask"] != right[0]["subtask"]:
                    raise SweepCompileError(
                        f"{episode_id}: gap-clamped boundary does not preserve the intended next subtask"
                    )
                retained_edge = int(right[0]["end_timestep_exclusive"])
                _update_bounds(left[0], int(left[0]["start_timestep"]), retained_edge,
                               float(left[0]["native_rate_hz"]))
                atoms.remove(right[0])
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id,
                        "field": "shared_atom_boundary",
                        "systematic_proposed_value": new,
                        "retained_edge": retained_edge,
                        "resolution": "sweep_fix_clamped_by_hard_gap_retention_authority",
                        "post_gap_atom": [
                            int(post_gap[0]["parent_interval_index"]), int(post_gap[0]["atom_index"])
                        ],
                    })
                continue
            if not int(left[0]["start_timestep"]) < new < int(right[0]["end_timestep_exclusive"]):
                raise SweepCompileError(f"{episode_id}: boundary {new} creates an empty atom")
            _update_bounds(left[0], int(left[0]["start_timestep"]), new, float(left[0]["native_rate_hz"]))
            _update_bounds(right[0], new, int(right[0]["end_timestep_exclusive"]), float(right[0]["native_rate_hz"]))
            left[0]["end_boundary_provenance"] = "semantic_sweep_visual_settle"
            right[0]["boundary_provenance"] = "semantic_sweep_visual_settle"
        elif kind == "text":
            matches = [row for row in atoms if _edit_atom_matches(row, edit)]
            if "start_timestep" in edit:
                ranged = [row for row in atoms if int(row["start_timestep"]) == int(edit["start_timestep"])]
                if len(ranged) == 1:
                    matches = ranged
            if not matches:
                raise SweepCompileError(f"{episode_id}: text edit matched no atom")
            old = edit.get("old_subtask")
            mismatches = [row for row in matches if old and row["subtask"] not in {old, edit["new_subtask"]}]
            if mismatches:
                if composition_conflicts is None:
                    raise SweepCompileError(f"{episode_id}: text edit old_subtask mismatch")
                for row in mismatches:
                    composition_conflicts.append({
                        "episode_id": episode_id,
                        "field": "subtask",
                        "atom_key": [int(row["parent_interval_index"]), int(row["atom_index"])],
                        "sampled_value": row["subtask"],
                        "systematic_expected_old": old,
                        "systematic_proposed_value": edit["new_subtask"],
                        "resolution": "preserve_compliant_episode_specific_sampled_fix",
                    })
                matches = [row for row in matches if row not in mismatches]
            for row in matches:
                if row["subtask"] == edit["new_subtask"]:
                    continue
                row["subtask"] = edit["new_subtask"]
                words = str(edit["new_subtask"]).split(maxsplit=1)
                if words:
                    row["verb"] = words[0]
                if len(words) > 1 and "object" in row:
                    row["object"] = words[1]
        elif kind == "merge_fake_atom":
            matches = [index for index, row in enumerate(atoms) if _edit_atom_matches(row, edit)]
            if len(matches) != 1 or matches[0] == 0 or matches[0] == len(atoms) - 1:
                raise SweepCompileError(f"{episode_id}: fake atom merge is not uniquely interior")
            index = matches[0]
            left, fake, right = atoms[index - 1 : index + 2]
            if not (left["end_timestep_exclusive"] == fake["start_timestep"] and
                    fake["end_timestep_exclusive"] == right["start_timestep"] and
                    left["parent_interval_index"] == fake["parent_interval_index"] == right["parent_interval_index"]):
                raise SweepCompileError(f"{episode_id}: fake atom neighbors are not contiguous")
            _update_bounds(left, int(left["start_timestep"]), int(right["end_timestep_exclusive"]),
                           float(left["native_rate_hz"]))
            atoms[index - 1 : index + 2] = [left]
            parent = int(left["parent_interval_index"])
            for atom_index, row in enumerate(row for row in atoms if int(row["parent_interval_index"]) == parent):
                row["atom_index"] = atom_index


def assert_structural_edits(atoms: list[dict[str, Any]], edits: list[dict[str, Any]], episode_id: str) -> None:
    for edit in edits:
        if edit["kind"] == "assert_structural_gap":
            gap = edit.get("gap", edit.get("excluded_frame_range", edit.get("declared_gap_frame_range")))
            if gap and any(int(row["start_timestep"]) < int(gap[1]) and int(gap[0]) < int(row["end_timestep_exclusive"])
                           for row in atoms):
                raise SweepCompileError(f"{episode_id}: atom still crosses confirmed gap {gap}")
        elif edit["kind"] == "assert_structural_fragments":
            expected = [tuple(item) for item in edit["replacement_fragments"]]
            for boundary in edits:
                if (
                    boundary["kind"] == "boundary"
                    and int(boundary.get("parent_interval_index", -1)) == int(edit["parent_interval_index"])
                    and int(boundary.get("left_atom_index", -1)) == int(edit["atom_index"])
                    and int(boundary["new_boundary_frame"]) > int(boundary["old_boundary_frame"])
                    and int(boundary["old_boundary_frame"]) >= expected[-1][1]
                ):
                    old = int(boundary["old_boundary_frame"])
                    new = int(boundary["new_boundary_frame"])
                    if old == expected[-1][1]:
                        expected[-1] = (expected[-1][0], new)
                    else:
                        expected.append((old, new))
            expected.sort()
            source_key = [episode_id, int(edit["parent_interval_index"]), int(edit["atom_index"])]
            actual = [(int(row["start_timestep"]), int(row["end_timestep_exclusive"])) for row in atoms
                      if row.get("_source_atom_key") == source_key]
            expected_gaps = [(left[1], right[0]) for left, right in zip(expected, expected[1:])]
            actual_gaps = [(left[1], right[0]) for left, right in zip(actual, actual[1:])]
            if len(actual) != len(expected) or actual_gaps != expected_gaps:
                raise SweepCompileError(
                    f"{episode_id}: structural fragment gaps {actual_gaps} != {expected_gaps}"
                )
        elif edit["kind"] == "assert_boundary_superseded_by_gap":
            for item in edit.get("rejected_intervals_hit", []):
                gap = item[:2]
                if any(int(row["start_timestep"]) < int(gap[1]) and int(gap[0]) < int(row["end_timestep_exclusive"])
                       for row in atoms):
                    raise SweepCompileError(f"{episode_id}: rejected boundary still bridges hard gap {gap}")


def _intersections(start: int, stop: int, intervals: list[list[int]]) -> list[tuple[int, int]]:
    return [(max(start, lo), min(stop, hi)) for lo, hi in intervals if max(start, lo) < min(stop, hi)]


def _subtract_gap(intervals: list[list[int]], gap: list[int]) -> list[list[int]]:
    start, stop = map(int, gap)
    if stop <= start:
        raise SweepCompileError(f"invalid confirmed gap {gap}")
    output = []
    for lo, hi in intervals:
        if hi <= start or stop <= lo:
            output.append([lo, hi])
            continue
        if lo < start:
            output.append([lo, start])
        if stop < hi:
            output.append([stop, hi])
    return output


def _clip_atoms(atoms: list[dict[str, Any]], intervals: list[list[int]]) -> list[dict[str, Any]]:
    output = []
    for atom in atoms:
        for start, stop in _intersections(int(atom["start_timestep"]), int(atom["end_timestep_exclusive"]), intervals):
            row = copy.deepcopy(atom)
            _update_bounds(row, start, stop, float(row["native_rate_hz"]))
            output.append(row)
    # Existing structural parents remain ordered; re-index only within each surviving parent.
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_parent[int(row["parent_interval_index"])].append(row)
    for rows in by_parent.values():
        rows.sort(key=lambda row: row["start_timestep"])
        for index, row in enumerate(rows):
            row["atom_index"] = index
    return sorted(output, key=lambda row: row["start_timestep"])


def apply_structural_fragment_edits(
    atoms: list[dict[str, Any]], edits: list[dict[str, Any]], episode_id: str,
    composition_conflicts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output = list(atoms)
    for edit in edits:
        if edit["kind"] != "assert_structural_fragments":
            continue
        source_key = [episode_id, int(edit["parent_interval_index"]), int(edit["atom_index"])]
        matches = [row for row in output if row.get("_source_atom_key") == source_key]
        if not matches:
            raise SweepCompileError(f"{episode_id}: exact structural replacement matched no source atom")
        actual_fragments = sorted(
            (int(row["start_timestep"]), int(row["end_timestep_exclusive"])) for row in matches
        )
        expected_fragments = sorted(tuple(map(int, item)) for item in edit["replacement_fragments"])
        terminal_end = max(int(row["end_timestep_exclusive"]) for row in output)
        selected: set[int] = set()
        applied_fragments = []
        for start, stop in expected_fragments:
            candidates = [
                row for row in matches
                if max(start, int(row["start_timestep"])) < min(stop, int(row["end_timestep_exclusive"]))
            ]
            if len(candidates) != 1 or id(candidates[0]) in selected:
                raise SweepCompileError(
                    f"{episode_id}: reviewed fragment [{start},{stop}) has {len(candidates)} unique containment owners; "
                    f"available={actual_fragments}"
                )
            owner = candidates[0]
            selected.add(id(owner))
            applied_stop = stop
            if int(owner["end_timestep_exclusive"]) == stop + 1 == terminal_end:
                applied_stop = terminal_end
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id, "field": "terminal_structural_fragment",
                        "source_atom_key": source_key, "reviewed_end_exclusive": stop,
                        "retention_authority_end_exclusive": terminal_end,
                        "resolution": "mechanical_terminal_authority_supersedes_off_by_one_review_fragment",
                    })
            applied_fragments.append([start, applied_stop])
            _update_bounds(owner, start, applied_stop, float(owner["native_rate_hz"]))
        edit["replacement_fragments"] = applied_fragments
        output = [
            row for row in output
            if row.get("_source_atom_key") != source_key or id(row) in selected
        ]
        # The fresh containment manifest has already split and reparented each
        # fragment. Re-cloning from matches[0] would collapse post-gap pieces
        # into the first parent and corrupt all joined sidecars.
        resolution = edit.get("second_reader_resolution", {})
        for update in resolution.get("neighbor_updates", []):
            neighbor_key = [episode_id, int(edit["parent_interval_index"]), int(update["atom_index"])]
            neighbors = [row for row in output if row.get("_source_atom_key") == neighbor_key]
            if not neighbors:
                raise SweepCompileError(
                    f"{episode_id}: structural neighbor update matched no atoms"
                )
            if "end_timestep_exclusive" in update and "start_timestep" not in update:
                neighbor = max(neighbors, key=lambda row: int(row["end_timestep_exclusive"]))
            elif "start_timestep" in update and "end_timestep_exclusive" not in update:
                neighbor = min(neighbors, key=lambda row: int(row["start_timestep"]))
            elif len(neighbors) == 1:
                neighbor = neighbors[0]
            else:
                raise SweepCompileError(f"{episode_id}: ambiguous two-sided structural neighbor update")
            start = int(update.get("start_timestep", neighbor["start_timestep"]))
            stop = int(update.get("end_timestep_exclusive", neighbor["end_timestep_exclusive"]))
            _update_bounds(neighbor, start, stop, float(neighbor["native_rate_hz"]))
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_parent[int(row["parent_interval_index"])].append(row)
    for rows in by_parent.values():
        rows.sort(key=lambda row: int(row["start_timestep"]))
        for index, row in enumerate(rows):
            row["atom_index"] = index
    return sorted(output, key=lambda row: int(row["start_timestep"]))


def _aligned_rows(
    source_rows: list[dict[str, Any]], atoms: list[dict[str, Any]], episode_id: str
) -> list[dict[str, Any]]:
    source = {_atom_key(row): row for row in source_rows if str(row["episode_id"]) == episode_id}
    output = []
    for atom in atoms:
        inherited = tuple(atom.get("_source_atom_key", _atom_key(atom)))
        if inherited not in source:
            raise SweepCompileError(f"{episode_id}: no aligned source row for {inherited}")
        row = copy.deepcopy(source[inherited])
        row.update(
            episode_id=episode_id,
            parent_interval_index=int(atom["parent_interval_index"]),
            atom_index=int(atom["atom_index"]),
            start_timestep=int(atom["start_timestep"]),
            end_timestep_exclusive=int(atom["end_timestep_exclusive"]),
            subtask=atom["subtask"],
        )
        if "verb" in row:
            row["verb"] = atom.get("verb", row["verb"])
        if "duration_s" in row:
            row["duration_s"] = (row["end_timestep_exclusive"] - row["start_timestep"]) / float(atom["native_rate_hz"])
        output.append(row)
    return output


def _remap_v2_to_final_authority(
    name: str, rows: list[dict[str, Any]], intervals: list[list[int]], atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_atoms = [
        {
            "source_atom_key": list(row.get("_source_atom_key", _atom_key(row))),
            "target_atom_key": list(_atom_key(row)),
            "start_timestep": int(row["start_timestep"]),
            "end_timestep_exclusive": int(row["end_timestep_exclusive"]),
        }
        for row in atoms
    ]
    output = []
    for row in rows:
        for target in remap_v2_row(name, row, [tuple(item) for item in intervals], target_atoms):
            materialized = copy.deepcopy(row)
            materialized.update(target["fields"])
            output.append(materialized)
    return output


def _matches(row: dict[str, Any], selector: dict[str, Any]) -> bool:
    ignored = {"kind", "family", "sidecar", "operation", "effective_after_delete", "row", "expected_count",
               "max_native_frames", "reason", "remap_all_dependent_sidecars"}
    return all(key in ignored or key not in selector or row.get(key) == value for key, value in selector.items())


def _sampled_quality_removals(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load hash-verified exact removals from the immutable sampled stage."""
    manifest_path = source_root / "corrected_store_manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = _load(manifest_path)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_root = Path(manifest.get("source_root", ""))
    fix_dir = Path(manifest.get("fix_dir", ""))
    baseline_path = fix_dir.parents[1] / "sweeps" / "round1_mechanical_containment.json"
    baseline = _load(baseline_path)
    if Path(baseline.get("source_root", "")).resolve() != original_root.resolve():
        raise SweepCompileError("original containment source root does not match sampled-stage provenance")
    source_tables: dict[str, list[dict[str, Any]]] = {}
    for correction in manifest.get("corrections", []):
        spec_path = Path(correction["spec"])
        if not spec_path.is_file() or _sha256(spec_path) != correction["sha256"]:
            raise SweepCompileError(f"sampled correction hash mismatch: {spec_path}")
        spec = _load(spec_path)
        provenance = spec.get("normalization_provenance", {})
        original_path = Path(provenance.get("source_spec", ""))
        expected = provenance.get("source_spec_sha256")
        if not original_path.is_file() or not expected or _sha256(original_path) != expected:
            raise SweepCompileError(f"sampled source-spec provenance mismatch: {original_path}")
        original = _load(original_path)
        rows = original.get("operations", {}).get("quality_spans.jsonl", {}).get("remove", [])
        if not isinstance(rows, list):
            raise SweepCompileError(f"sampled quality removal provenance is malformed: {original_path}")
        for row in rows:
            if not isinstance(row, dict):
                raise SweepCompileError(f"sampled quality removal is not an object: {original_path}")
            output[str(correction["episode_id"])].append({"row": row, "authority": "sampled_fix"})
        quality_operation = original.get("replacement_tables", {}).get("quality_spans.jsonl")
        if quality_operation is None:
            quality_operation = original.get("tables", {}).get("quality_spans.jsonl")
        if quality_operation is None:
            quality_operation = original.get("replacements", {}).get("quality_spans.jsonl")
        if quality_operation is None:
            operation = original.get("operations", {}).get("quality_spans.jsonl")
            if isinstance(operation, dict):
                replacement = next(
                    (operation[key] for key in ("replace_episode_rows", "replace_complete_episode_rows")
                     if key in operation), None,
                )
                if replacement is not None:
                    quality_operation = {"operation": "replace_all_episode_rows", "rows": replacement}
        if not isinstance(quality_operation, dict) or quality_operation.get("operation") != "replace_all_episode_rows":
            continue
        store = str(correction["store"])
        table_path = original_root / store / "quality_spans.jsonl"
        expected_table_hash = baseline.get("source_hashes", {}).get(f"{store}/quality_spans.jsonl")
        spec_table_hash = original.get("provenance", {}).get("source_file_sha256", {}).get("quality_spans.jsonl")
        if not expected_table_hash or (spec_table_hash and spec_table_hash != expected_table_hash):
            raise SweepCompileError(f"missing or conflicting trusted quality-table hash for {store}")
        if not table_path.is_file() or _sha256(table_path) != expected_table_hash:
            raise SweepCompileError(f"sampled source quality-table provenance mismatch: {table_path}")
        source_table_hash = _sha256(table_path)
        if store not in source_tables:
            source_tables[store] = _read_jsonl(table_path)
        episode_id = str(correction["episode_id"])
        replacements = spec.get("replacements", {}).get("quality_spans.jsonl", [])
        replacement_ids = {_stable_quality_identity(row) for row in replacements}
        for row in source_tables[store]:
            if str(row.get("episode_id")) == episode_id and _stable_quality_identity(row) not in replacement_ids:
                output[episode_id].append({
                    "row": row, "authority": "sampled_fix", "source_table_sha256": source_table_hash,
                })
    return output


def _stable_quality_identity(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("uid"), row.get("raw_from_index"), row.get("raw_to_index")


def _proven_prior_quality_removal(
    selector: dict[str, Any], prior_removals: list[dict[str, Any]], current_rows: list[dict[str, Any]]
) -> str | None:
    matches = [entry for entry in prior_removals if _matches(entry["row"], selector)]
    if len(matches) != 1:
        return None
    identity = _stable_quality_identity(matches[0]["row"])
    if any(_stable_quality_identity(row) == identity for row in current_rows):
        return None
    return str(matches[0]["authority"])


def _transformed_quality_hits(
    selector: dict[str, Any], prior_removals: list[dict[str, Any]], current_rows: list[dict[str, Any]]
) -> list[int]:
    matches = [entry for entry in prior_removals
               if entry.get("authority") == "structural_containment_transform"
               and _matches(entry["row"], selector)]
    if len(matches) != 1:
        return []
    source_line = int(matches[0]["source_line"])
    return [index for index, row in enumerate(current_rows) if int(row.get("_source_line", -1)) == source_line]


def _apply_quality_edits(
    rows: list[dict[str, Any]], edits: list[dict[str, Any]], episode_id: str,
    composition_conflicts: list[dict[str, Any]] | None = None,
    prior_removals: list[dict[str, Any]] | None = None,
    sampled_episode: bool = False,
    source_rows: list[dict[str, Any]] | None = None,
    retained_intervals: list[list[int]] | None = None,
    atoms: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]

    def remap_addition(row: dict[str, Any]) -> list[dict[str, Any]]:
        if retained_intervals is None or atoms is None:
            raise SweepCompileError(f"{episode_id}: quality addition lacks atom/retention authority")
        target_atoms = [
            {
                "source_atom_key": list(atom.get("_source_atom_key", _atom_key(atom))),
                "target_atom_key": list(_atom_key(atom)),
                "start_timestep": int(atom["start_timestep"]),
                "end_timestep_exclusive": int(atom["end_timestep_exclusive"]),
            }
            for atom in atoms
        ]
        candidate = copy.deepcopy(row)
        candidate.setdefault("episode_id", episode_id)
        targets = remap_v2_row(
            "quality_spans.jsonl", candidate,
            [tuple(item) for item in retained_intervals], target_atoms,
        )
        if not targets:
            raise SweepCompileError(f"{episode_id}: confirmed quality addition has no retained authority")
        added = []
        for target in targets:
            materialized = copy.deepcopy(candidate)
            materialized.update(target["fields"])
            added.append(materialized)
        return added

    for edit in edits:
        kind = edit["kind"]
        if kind == "quality_delete_exact":
            selector = edit["row"]
            hits = [index for index, row in enumerate(output) if _matches(row, selector)]
            transformed = _transformed_quality_hits(selector, prior_removals or [], output) if not hits else []
            if transformed:
                output = [row for index, row in enumerate(output) if index not in set(transformed)]
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id, "field": "quality_spans.jsonl", "selector": selector,
                        "structural_descendants_deleted": len(transformed),
                        "resolution": "systematic_delete_applied_to_all_structural_descendants",
                    })
                continue
            if not hits and composition_conflicts is not None:
                authority = _proven_prior_quality_removal(selector, prior_removals or [], output)
                if authority is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id,
                        "field": "quality_spans.jsonl",
                        "selector": selector,
                        "sampled_value": "row_absent",
                        "systematic_proposed_value": "delete_exact_row",
                        "resolution": f"already_applied_by_{authority}",
                    })
                    continue
            if len(hits) != 1:
                raise SweepCompileError(f"{episode_id}: exact quality deletion matched {len(hits)} rows")
            if sampled_episode and composition_conflicts is not None:
                composition_conflicts.append({
                    "episode_id": episode_id, "field": "quality_spans.jsonl", "selector": selector,
                    "sampled_value": output[hits[0]],
                    "systematic_proposed_value": "delete_exact_row",
                    "resolution": "newer_dense_systematic_quality_supersedes_sampled_fix",
                })
            output.pop(hits[0])
        elif kind == "quality_delete_selector":
            selector = {key: edit[key] for key in
                        ("uid", "raw_from_index", "raw_to_index", "quality", "cause")
                        if key in edit}
            if not selector:
                selector = {key: edit[key] for key in ("from_index", "to_index") if key in edit}
            hits = [index for index, row in enumerate(output) if _matches(row, selector)]
            if not hits and sampled_episode and {"raw_from_index", "raw_to_index"} <= edit.keys():
                # A sampled full-row replacement can legitimately rekey the
                # source row while retaining the same reviewed semantic span.
                # Dense review is authoritative for that span, but the match
                # must remain unique before it may delete anything.
                semantic_selector = {
                    **({"uid": edit["uid"]} if "uid" in edit else {}),
                    "raw_from_index": edit["raw_from_index"],
                    "raw_to_index": edit["raw_to_index"],
                    **({"quality": edit["quality"]} if "quality" in edit else {}),
                    **({"cause": edit["cause"]} if "cause" in edit else {}),
                }
                hits = [index for index, row in enumerate(output) if _matches(row, semantic_selector)]
            transformed = _transformed_quality_hits(selector, prior_removals or [], output) if not hits else []
            if transformed:
                output = [row for index, row in enumerate(output) if index not in set(transformed)]
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id, "field": "quality_spans.jsonl", "selector": selector,
                        "structural_descendants_deleted": len(transformed),
                        "resolution": "systematic_delete_applied_to_all_structural_descendants",
                    })
                continue
            if not hits and composition_conflicts is not None:
                authority = _proven_prior_quality_removal(selector, prior_removals or [], output)
                if authority is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id,
                        "field": "quality_spans.jsonl",
                        "selector": selector,
                        "sampled_value": "row_absent",
                        "systematic_proposed_value": "delete_exact_selected_row",
                        "resolution": f"already_applied_by_{authority}",
                    })
                    continue
            if len(hits) != 1:
                raise SweepCompileError(f"{episode_id}: quality selector matched {len(hits)} rows: {selector}")
            if sampled_episode and composition_conflicts is not None:
                composition_conflicts.append({
                    "episode_id": episode_id, "field": "quality_spans.jsonl", "selector": selector,
                    "sampled_value": output[hits[0]],
                    "systematic_proposed_value": "delete_exact_selected_row",
                    "resolution": "newer_dense_systematic_quality_supersedes_sampled_fix",
                })
            output.pop(hits[0])
        elif kind == "quality_delete_max_duration":
            maximum = int(edit["max_native_frames"])
            expected = int(edit["expected_count"])
            source_candidates = [
                (line, row) for line, row in enumerate(source_rows or [], 1)
                if str(row.get("episode_id")) == episode_id
                and int(row["raw_to_index"]) - int(row["raw_from_index"]) <= maximum
            ]
            if len(source_candidates) == expected:
                selected_lines = {line for line, _ in source_candidates}
                hits = [index for index, row in enumerate(output)
                        if int(row.get("_source_line", -1)) in selected_lines]
                surviving_lines = {int(output[index]["_source_line"]) for index in hits}
                dropped_lines = {
                    int(entry["source_line"]) for entry in (prior_removals or [])
                    if entry.get("authority") == "structural_containment" and "source_line" in entry
                }
                if surviving_lines | (selected_lines & dropped_lines) != selected_lines:
                    missing = sorted(selected_lines - surviving_lines - dropped_lines)
                    raise SweepCompileError(
                        f"{episode_id}: duration deletion has unproven missing source lines {missing}"
                    )
                output = [row for index, row in enumerate(output) if index not in set(hits)]
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id, "field": "quality_spans.jsonl",
                        "max_native_frames": maximum, "confirmed_source_rows": expected,
                        "surviving_descendants_deleted": len(hits),
                        "containment_dropped_source_rows": len(selected_lines & dropped_lines),
                        "resolution": "duration_delete_selected_by_original_source_provenance",
                    })
                continue
            sampled_prior = [
                entry for entry in (prior_removals or [])
                if entry.get("authority") == "sampled_fix"
                and int(entry["row"]["raw_to_index"]) - int(entry["row"]["raw_from_index"]) <= maximum
            ]
            identities = {_stable_quality_identity(entry["row"]) for entry in sampled_prior}
            if len(source_candidates) == 0 and len(identities) == expected and not any(
                _stable_quality_identity(row) in identities for row in output
            ):
                if composition_conflicts is not None:
                    composition_conflicts.append({
                        "episode_id": episode_id, "field": "quality_spans.jsonl",
                        "max_native_frames": maximum, "confirmed_source_rows": expected,
                        "source_table_sha256": sorted({entry["source_table_sha256"] for entry in sampled_prior}),
                        "resolution": "already_applied_by_sampled_fix",
                    })
                continue
            raise SweepCompileError(
                f"{episode_id}: original duration selector matched {len(source_candidates)}, expected {expected}"
            )
        elif kind == "quality_restore_full_pause":
            if source_rows is None or retained_intervals is None or atoms is None:
                raise SweepCompileError(f"{episode_id}: pause restoration lacks source/authority context")
            selector = {
                "episode_id": episode_id,
                "uid": edit["uid"],
                "parent_interval_index": edit["parent_interval_index"],
                "atom_index": edit["atom_index"],
                "quality": edit["quality"],
                **edit["current"],
            }
            source_hits = [
                (line, row) for line, row in enumerate(source_rows, 1)
                if _matches(row, selector)
            ]
            if len(source_hits) != 1:
                raise SweepCompileError(
                    f"{episode_id}: full-pause source selector matched {len(source_hits)} rows"
                )
            source_line, source = source_hits[0]
            descendant_indices = {
                index for index, row in enumerate(output)
                if int(row.get("_source_line", -1)) == source_line
            }
            desired = [dict(row) for row in edit["replacement_rows"]]
            desired_raw = sorted(
                (int(row["raw_from_index"]), int(row["raw_to_index"])) for row in desired
            )
            if any(lo >= hi for lo, hi in desired_raw) or any(
                left[1] > right[0] for left, right in zip(desired_raw, desired_raw[1:])
            ):
                raise SweepCompileError(f"{episode_id}: invalid/overlapping full-pause pieces")
            target_atoms = [
                {
                    "source_atom_key": list(row.get("_source_atom_key", _atom_key(row))),
                    "target_atom_key": list(_atom_key(row)),
                    "start_timestep": int(row["start_timestep"]),
                    "end_timestep_exclusive": int(row["end_timestep_exclusive"]),
                }
                for row in atoms
            ]
            if not desired:
                current_targets = remap_v2_row(
                    "quality_spans.jsonl", dict(source),
                    [tuple(item) for item in retained_intervals], target_atoms,
                )
                if current_targets or descendant_indices:
                    raise SweepCompileError(
                        f"{episode_id}: empty full-pause replacement is not a proven containment drop"
                    )
            replacements = []
            for exact in desired:
                bounds = [int(exact[key]) for key in
                          ("from_index", "raw_from_index", "raw_to_index", "to_index")]
                if not (bounds[0] <= bounds[1] < bounds[2] <= bounds[3]):
                    raise SweepCompileError(f"{episode_id}: malformed full-pause replacement bounds {bounds}")
                candidate = copy.deepcopy(source)
                candidate.update(exact)
                targets = remap_v2_row(
                    "quality_spans.jsonl", candidate,
                    [tuple(item) for item in retained_intervals], target_atoms,
                )
                if len(targets) != 1:
                    raise SweepCompileError(
                        f"{episode_id}: full-pause piece has {len(targets)} authority intersections"
                    )
                fields = targets[0]["fields"]
                for key in ("from_index", "raw_from_index", "raw_to_index", "to_index"):
                    if int(fields[key]) != int(exact[key]):
                        raise SweepCompileError(
                            f"{episode_id}: full-pause exact {key}={exact[key]} conflicts with authority {fields[key]}"
                        )
                candidate.update(fields)
                candidate["_source_line"] = source_line
                candidate["_restored_full_pause"] = True
                replacements.append(candidate)
            produced_raw = sorted(
                (int(row["raw_from_index"]), int(row["raw_to_index"])) for row in replacements
            )
            if produced_raw != desired_raw:
                raise SweepCompileError(f"{episode_id}: full-pause raw-union mismatch")
            before = [output[index] for index in sorted(descendant_indices)]
            output = [row for index, row in enumerate(output) if index not in descendant_indices]
            output.extend(replacements)
            if composition_conflicts is not None:
                composition_conflicts.append({
                    "episode_id": episode_id, "field": "quality_spans.jsonl",
                    "source_uid": edit["uid"], "source_line": source_line,
                    "removed_containment_descendants": len(before),
                    "replacement_piece_count": len(replacements),
                    "resolution": (
                        "confirmed_full_pause_wholly_outside_retention_drop"
                        if not replacements else "confirmed_full_pause_restoration_materialized"
                    ),
                })
        elif kind == "quality_add_exact":
            output.extend(remap_addition(edit["row"]))
        elif kind == "quality_coalesce":
            ranges = {tuple(item) for item in edit["remove_raw_ranges"]}
            hits = [row for row in output if (int(row["raw_from_index"]), int(row["raw_to_index"])) in ranges]
            if len(hits) != len(ranges):
                raise SweepCompileError(f"{episode_id}: coalesced deletion matched {len(hits)}/{len(ranges)} rows")
            output = [row for row in output if (int(row["raw_from_index"]), int(row["raw_to_index"])) not in ranges]
            template = copy.deepcopy(hits[0])
            template.update(edit["add"])
            output.extend(remap_addition(template))
    identity_fields = (
        "episode_id", "uid", "parent_interval_index", "atom_index", "quality", "cause",
        "from_index", "raw_from_index", "raw_to_index", "to_index",
    )
    restored_identities = {
        tuple(row.get(key) for key in identity_fields)
        for row in output if row.get("_restored_full_pause")
    }
    restored_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        identity = tuple(row.get(key) for key in identity_fields)
        if identity in restored_identities:
            restored_groups[identity].append(row)
    dropped_ids: set[int] = set()
    for identity, group in restored_groups.items():
        if len(group) == 1:
            continue
        ordered = sorted(
            group,
            key=lambda row: (int(row["_source_line"]), json.dumps(row, sort_keys=True)),
        )
        winner = ordered[0]
        dropped_ids.update(id(row) for row in ordered[1:])
        all_fields = set().union(*(row.keys() for row in ordered)) - set(identity_fields) - {
            "_source_line", "_restored_full_pause",
        }
        divergent = sorted(
            key for key in all_fields
            if len({json.dumps(row.get(key), sort_keys=True, default=str) for row in ordered}) > 1
        )
        if composition_conflicts is not None:
            composition_conflicts.append({
                "episode_id": episode_id, "field": "quality_spans.jsonl",
                "semantic_identity": dict(zip(identity_fields, identity, strict=True)),
                "candidate_source_lines": [int(row["_source_line"]) for row in ordered],
                "selected_source_line": int(winner["_source_line"]),
                "divergent_nonsemantic_fields": divergent,
                "resolution": "coalesced_duplicate_full_pause_semantic_identity_by_lowest_source_line",
            })
    output = [row for row in output if id(row) not in dropped_ids]
    for row in output:
        row.pop("_restored_full_pause", None)
    return output


def _merge_frame_intervals(intervals: list[list[int]] | list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, stop in sorted((int(start), int(stop)) for start, stop in intervals):
        if start >= stop:
            raise SweepCompileError(f"invalid retention interval [{start}, {stop})")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def _fmb_episode_record_for_retention(
    source_root: Path, episode_id: str, intervals: list[list[int]],
) -> dict[str, Any] | None:
    """Materialize FMB reviewed primitive authority when semantic retention changes."""
    path = source_root / "fmb" / "episodes" / episode_id / "episode.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    primitives = record.get("primitive_intervals")
    if not isinstance(primitives, list) or not primitives:
        raise SweepCompileError(f"{episode_id}: FMB episode lacks primitive_intervals")

    def bounds(row: dict[str, Any]) -> tuple[int, int]:
        return (
            int(row.get("reviewed_start_timestep", row["start_timestep"])),
            int(row.get("reviewed_end_timestep_exclusive", row["end_timestep_exclusive"])),
        )

    target = _merge_frame_intervals(intervals)
    current = _merge_frame_intervals([bounds(row) for row in primitives])
    if current == target:
        return None

    updated = []
    for source_row in primitives:
        start, stop = bounds(source_row)
        pieces = _intersections(start, stop, target)
        if len(pieces) > 1:
            raise SweepCompileError(
                f"{episode_id}: FMB primitive [{start}, {stop}) would split across retention"
            )
        if not pieces:
            continue
        row = copy.deepcopy(source_row)
        row["reviewed_start_timestep"], row["reviewed_end_timestep_exclusive"] = pieces[0]
        updated.append(row)
    materialized = _merge_frame_intervals([bounds(row) for row in updated])
    if materialized != target:
        raise SweepCompileError(
            f"{episode_id}: FMB metadata cannot materialize retention {target}; got {materialized}"
        )
    record["primitive_intervals"] = updated
    return record


def _corpus_episode_record_for_retention(
    source_root: Path, episode_id: str, intervals: list[list[int]]
) -> dict[str, Any]:
    """Materialize corpus ``annotations.segments`` for exact frame retention."""
    record_path = source_root / "corpus" / "episodes" / episode_id / "episode.json"
    record = _load(record_path)
    timestamps = np.load(record_path.parent / "timestamp_s.npy", mmap_mode="r")
    frames = len(timestamps)
    rate = float(record["native_rate_hz"])
    target = _merge_frame_intervals([tuple(map(int, item)) for item in intervals])
    if not target or any(start < 0 or stop > frames or start >= stop for start, stop in target):
        raise SweepCompileError(f"{episode_id}: invalid corpus retention {target}")

    def frame_time(frame: int) -> float:
        if frame < frames:
            return float(timestamps[frame])
        if frame == frames:
            return float(timestamps[-1]) + 1.0 / rate
        raise SweepCompileError(f"{episode_id}: metadata boundary {frame} exceeds {frames}")

    def frame_bounds(row: dict[str, Any]) -> tuple[int, int]:
        return (
            int(np.searchsorted(timestamps, float(row["start_s"]), side="left")),
            int(np.searchsorted(timestamps, float(row["end_s"]), side="left")),
        )

    annotations = copy.deepcopy(record.get("annotations", {}))
    source_segments = annotations.get("segments")
    if not isinstance(source_segments, list) or not source_segments:
        raise SweepCompileError(f"{episode_id}: corpus metadata lacks annotation segments")
    updated: list[dict[str, Any]] = []
    for source_row in source_segments:
        start, stop = frame_bounds(source_row)
        if stop <= start:
            continue
        if source_row.get("retention") != "keep":
            updated.append(copy.deepcopy(source_row))
            continue
        pieces = _intersections(start, stop, [list(item) for item in target])
        cursor = start
        for keep_start, keep_stop in pieces:
            if cursor < keep_start:
                updated.append({
                    "start_s": frame_time(cursor), "end_s": frame_time(keep_start),
                    "retention": "reject", "retention_reason": "terminal_idle",
                })
            row = copy.deepcopy(source_row)
            row["start_s"], row["end_s"] = frame_time(keep_start), frame_time(keep_stop)
            updated.append(row)
            cursor = keep_stop
        if cursor < stop:
            updated.append({
                "start_s": frame_time(cursor), "end_s": frame_time(stop),
                "retention": "reject", "retention_reason": "terminal_idle",
            })
    updated.sort(key=lambda row: (float(row["start_s"]), float(row["end_s"])))
    materialized = _merge_frame_intervals([
        frame_bounds(row) for row in updated if row.get("retention") == "keep"
    ])
    if materialized != target:
        raise SweepCompileError(
            f"{episode_id}: corpus metadata cannot materialize retention {target}; got {materialized}"
        )
    annotations["segments"] = updated
    annotations["source_keep_intervals_s"] = [
        [frame_time(start), frame_time(stop)] for start, stop in target
    ]
    annotations["reviewer_notes"] = (
        str(annotations.get("reviewer_notes", ""))
        + " Targeted diverse audit removed visually confirmed terminal idle tails."
    ).strip()
    record["annotations"] = annotations
    return record


def build_specs(
    source_root: Path, containment: dict[str, Any], edits: list[dict[str, Any]],
    composition_conflicts: list[dict[str, Any]] | None = None,
    *, sampled_quality_provenance: bool = True,
) -> list[dict[str, Any]]:
    """Create canonical full-row specs from structural ledgers plus semantic edits."""
    edits_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edit in edits:
        edits_by_episode[edit["episode_id"]].append(edit)
    episode_info = {str(row["episode_id"]): row for row in containment["episodes"]}
    atom_ops_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for operation in containment["atom_operations"]:
        atom_ops_by_episode[str(operation["source_atom_key"][0])].append(operation)
    v2_ops: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: defaultdict(list) for name in containment["v2_operations"]
    }
    for name, operations in containment["v2_operations"].items():
        for operation in operations:
            v2_ops[name][str(operation["episode_id"])].append(operation)
    structural_atom_changed = {
        str(operation["source_atom_key"][0]) for operation in containment["atom_operations"]
        if operation["action"] != "keep"
    }
    structural_v2_changed = {
        str(operation["episode_id"]) for operations in containment["v2_operations"].values()
        for operation in operations if operation["action"] != "keep"
    }
    actor_changed = {str(row["episode_id"]) for row in containment["actor_anchor_drops"]}
    affected = set(edits_by_episode) | structural_atom_changed | structural_v2_changed | actor_changed
    table_names = (
        "subtask_atoms.jsonl", "contact_atoms.jsonl", "precision_atoms.jsonl", "speed_atoms.jsonl",
        "speed_atoms_hybrid_v1.jsonl", "quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl",
    )
    tables = {store: {name: _read_jsonl(source_root / store / name) for name in table_names}
              for store in ("corpus", "fmb")}
    sampled_manifest_path = source_root / "corrected_store_manifest.json"
    sampled_episodes = set()
    if sampled_manifest_path.is_file():
        sampled_episodes = {
            str(row["episode_id"]) for row in _load(sampled_manifest_path).get("corrections", [])
        }
    prior_quality_removals = (
        _sampled_quality_removals(source_root)
        if sampled_quality_provenance
        else defaultdict(list)
    )
    for operation in containment["v2_operations"]["quality_spans.jsonl"]:
        episode_id = str(operation["episode_id"])
        store = str(episode_info[episode_id]["store"])
        source_row = tables[store]["quality_spans.jsonl"][int(operation["source_line"]) - 1]
        if str(source_row.get("uid")) != str(operation["source_uid"]):
            raise SweepCompileError(f"{episode_id}: dropped quality source_line/uid mismatch")
        if not operation.get("targets"):
            prior_quality_removals[episode_id].append({
                "row": source_row, "authority": "structural_containment",
                "source_line": int(operation["source_line"]),
            })
        elif operation.get("action") != "keep":
            prior_quality_removals[episode_id].append({
                "row": source_row, "authority": "structural_containment_transform",
                "source_line": int(operation["source_line"]),
            })
    specs = []
    atom_edit_kinds = {
        "boundary", "text", "merge_fake_atom", "retained_end", "retained_intervals",
        "assert_structural_gap", "assert_structural_fragments",
    }
    quality_kinds = {"quality_delete_exact", "quality_delete_selector", "quality_delete_max_duration",
                     "quality_add_exact", "quality_coalesce", "quality_restore_full_pause"}
    for episode_id in sorted(affected):
        info = episode_info.get(episode_id)
        if info is None:
            raise SweepCompileError(f"semantic edit references episode absent from containment: {episode_id}")
        store = str(info["store"])
        ep_edits = edits_by_episode.get(episode_id, [])
        intervals = copy.deepcopy(info["retained_intervals"])
        for edit in ep_edits:
            if edit["kind"] == "retained_intervals":
                intervals = copy.deepcopy(edit["intervals"])
            elif edit["kind"] == "retained_end":
                stop = int(edit["set_retained_end_exclusive"])
                intervals = [[a, min(b, stop)] for a, b in intervals if a < stop]
                intervals = [[a, b] for a, b in intervals if a < b]
        retention_changed = any(edit["kind"] in {"retained_intervals", "retained_end"} for edit in ep_edits)
        for edit in ep_edits:
            if edit["kind"] == "assert_structural_gap":
                gap = edit.get("gap", edit.get("excluded_frame_range", edit.get("declared_gap_frame_range")))
                if not isinstance(gap, list) or len(gap) != 2:
                    raise SweepCompileError(f"{episode_id}: confirmed gap lacks exact frame range")
                intervals = _subtract_gap(intervals, gap)
            elif edit["kind"] == "assert_structural_fragments":
                gaps = edit.get("source_declared_static_gaps")
                if not isinstance(gaps, list) or not gaps:
                    raise SweepCompileError(f"{episode_id}: structural split lacks declared static gaps")
                for gap in gaps:
                    if not isinstance(gap, list) or len(gap) != 2:
                        raise SweepCompileError(f"{episode_id}: malformed declared static gap")
                    intervals = _subtract_gap(intervals, gap)
        atom_changed = episode_id in structural_atom_changed or any(edit["kind"] in atom_edit_kinds for edit in ep_edits)
        needs_atom_context = atom_changed or any(
            edit["kind"] in {"quality_restore_full_pause", "quality_add_exact", "quality_coalesce"}
            for edit in ep_edits
        )
        replacements: dict[str, list[dict[str, Any]]] = {}
        atoms = None
        if needs_atom_context:
            atoms = materialize_atoms(tables[store]["subtask_atoms.jsonl"], atom_ops_by_episode[episode_id], episode_id)
            atoms = _clip_atoms(atoms, intervals)
            atoms = apply_structural_fragment_edits(atoms, ep_edits, episode_id, composition_conflicts)
            apply_atom_edits(
                atoms, ep_edits, episode_id, composition_conflicts,
                sampled_episode=episode_id in sampled_episodes,
                retained_intervals=intervals,
            )
            assert_structural_edits(atoms, ep_edits, episode_id)
        if atom_changed:
            assert atoms is not None
            replacements["subtask_atoms.jsonl"] = atoms
            for name in ("contact_atoms.jsonl", "precision_atoms.jsonl", "speed_atoms.jsonl"):
                replacements[name] = _aligned_rows(tables[store][name], atoms, episode_id)
            old_hybrid = {_atom_key(row): row for row in tables[store]["speed_atoms_hybrid_v1.jsonl"]
                          if str(row["episode_id"]) == episode_id}
            priors = {}
            duration_used = {}
            for atom in atoms:
                source_key = tuple(atom.get("_source_atom_key", _atom_key(atom)))
                if source_key not in old_hybrid:
                    raise SweepCompileError(f"{episode_id}: missing hybrid duration prior for {source_key}")
                target_key = (int(atom["parent_interval_index"]), int(atom["atom_index"]))
                priors[target_key] = float(old_hybrid[source_key]["duration_speed"])
                duration_used[target_key] = bool(old_hybrid[source_key]["duration_used"])
            replacements["speed_atoms_hybrid_v1.jsonl"] = fix_normalizer._hybrid_rows(
                source_root, store, episode_id, atoms, priors, duration_used
            )
        for name in ("quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl"):
            needs = atom_changed or episode_id in structural_v2_changed or (
                name == "quality_spans.jsonl" and any(edit["kind"] in quality_kinds for edit in ep_edits)
            )
            if not needs:
                continue
            rows = materialize_v2(tables[store][name], v2_ops[name][episode_id], episode_id)
            if name == "quality_spans.jsonl":
                rows = _apply_quality_edits(
                    rows, ep_edits, episode_id, composition_conflicts,
                    prior_quality_removals.get(episode_id, []),
                    sampled_episode=episode_id in sampled_episodes,
                    source_rows=tables[store]["quality_spans.jsonl"],
                    retained_intervals=intervals,
                    atoms=atoms,
                )
            if atom_changed or retention_changed:
                if atoms is None:
                    raise SweepCompileError(f"{episode_id}: retention change lacks final atom authority")
                rows = _remap_v2_to_final_authority(name, rows, intervals, atoms)
            replacements[name] = rows
        for rows in replacements.values():
            for row in rows:
                row.pop("_source_atom_key", None)
                row.pop("_source_line", None)
        if not replacements and episode_id not in actor_changed:
            continue
        episode_record = None
        if retention_changed:
            episode_record = (
                _fmb_episode_record_for_retention(source_root, episode_id, intervals)
                if store == "fmb"
                else _corpus_episode_record_for_retention(source_root, episode_id, intervals)
            )
        specs.append({
            "schema_version": 1, "status": "resolved", "store": store, "episode_id": episode_id,
            "replacements": replacements,
            "actor_anchors": {"mode": "remap", "retained_intervals": intervals}
            if atom_changed or episode_id in actor_changed else {"mode": "preserve"},
            "critic_intervals": {"mode": "from_atoms"} if atom_changed else {"mode": "preserve"},
            "episode_record": episode_record, "unresolved": [], "second_read_required": False,
            "semantic_compiler": {"edits": ep_edits},
        })
    return specs


def validate_plan(source_root: Path, sweep_dir: Path, containment_path: Path) -> dict[str, Any]:
    reports, containment, hashes = load_and_validate(source_root, sweep_dir, containment_path)
    edits = collect_edits(reports)
    rejected = [row for row in edits if row.get("rejected_candidate_reversion")
                or row.get("rejected_candidate_assertion")]
    return {
        "status": "VALIDATED_SEMANTIC_EDIT_PLAN",
        "source_root": str(source_root),
        "input_hashes": hashes,
        "structural_open_items": len(containment.get("open_candidates", [])),
        "plan_record_count": len(edits),
        "confirmed_edit_records": len(edits) - len(rejected),
        "rejected_boundary_assertion_records": len(rejected),
        "full_pause_restoration_records": sum(row["kind"] == "quality_restore_full_pause" for row in edits),
        "edits_by_kind": dict(sorted(Counter(row["kind"] for row in edits).items())),
        "affected_episodes": len({row["episode_id"] for row in edits}),
    }


def compile_directory(
    source_root: Path, sweep_dir: Path, containment_path: Path, out_dir: Path, *, dry_run: bool
) -> dict[str, Any]:
    """Materialize full-row semantic corrections and canonical trace artifacts."""
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    reports, containment, hashes = load_and_validate(source_root, sweep_dir, containment_path)
    edits = collect_edits(reports)
    conflicts: list[dict[str, Any]] = []
    specs = build_specs(source_root, containment, edits, conflicts)
    rejected = [row for row in edits if row.get("rejected_candidate_reversion")
                or row.get("rejected_candidate_assertion")]
    if len(edits) != 3400 or len(rejected) != 206 or len(edits) - len(rejected) != 3194:
        raise SweepCompileError(
            f"semantic decision accounting drift: total={len(edits)} confirmed={len(edits)-len(rejected)} "
            f"rejected_assertions={len(rejected)}"
        )
    if sum(row["kind"] == "quality_restore_full_pause" for row in edits) != 141:
        raise SweepCompileError("full-pause restoration decision coverage is not exactly 141")
    restore_resolutions = Counter(
        row.get("resolution") for row in conflicts
        if "full_pause" in str(row.get("resolution", ""))
    )
    if restore_resolutions["confirmed_full_pause_restoration_materialized"] != 131:
        raise SweepCompileError(f"nonempty full-pause coverage drift: {restore_resolutions}")
    if restore_resolutions["confirmed_full_pause_wholly_outside_retention_drop"] != 10:
        raise SweepCompileError(f"zero-authority full-pause coverage drift: {restore_resolutions}")

    artifacts: list[tuple[Path, bytes]] = []
    encoded_specs: list[tuple[Path, bytes, dict[str, Any]]] = []
    seen_names: set[str] = set()
    for spec in specs:
        episode_id = str(spec["episode_id"])
        name = f"{episode_id}.json"
        if name in seen_names:
            raise SweepCompileError(f"duplicate emitted spec filename: {name}")
        seen_names.add(name)
        if "subtask_atoms.jsonl" in spec["replacements"]:
            payload = fix_normalizer._trace_artifact(source_root, spec)
            relative = Path("artifacts") / spec["store"] / "speed_hybrid_v1" / f"{episode_id}.npz"
            spec["artifacts"] = {
                "speed_hybrid_v1": {
                    "path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(),
                    "keys": sorted(fix_normalizer.TRACE_KEYS),
                }
            }
            for row in spec["replacements"]["speed_atoms_hybrid_v1.jsonl"]:
                row["state_speed_file"] = f"{spec['store']}/speed_hybrid_v1/{episode_id}.npz"
            artifacts.append((relative, payload))
        payload = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode()
        encoded_specs.append((Path(name), payload, spec))
    inventory = [
        {"episode_id": spec["episode_id"], "store": spec["store"], "path": path.as_posix(),
         "sha256": hashlib.sha256(payload).hexdigest()}
        for path, payload, spec in encoded_specs
    ]
    report = {
        "schema_version": 1,
        "status": "VALID_DRY_RUN" if dry_run else "COMPILED",
        "source_root": str(source_root), "sweep_dir": str(sweep_dir),
        "containment_manifest": str(containment_path), "out_dir": str(out_dir),
        "input_hashes": hashes,
        "plan_record_count": len(edits), "confirmed_edit_records": len(edits) - len(rejected),
        "rejected_boundary_assertion_records": len(rejected),
        "full_pause_restoration_records": 141,
        "spec_count": len(specs), "trace_artifact_count": len(artifacts),
        "conflict_ledger_count": len(conflicts),
        "resolution_counts": dict(sorted(Counter(str(row.get("resolution")) for row in conflicts).items())),
        "edits_by_kind": dict(sorted(Counter(row["kind"] for row in edits).items())),
        "spec_inventory": inventory,
    }
    if dry_run:
        return report
    temporary = out_dir.with_name(f".{out_dir.name}.compiling-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary compiler directory: {temporary}")
    try:
        temporary.mkdir(parents=True)
        for relative, payload, _ in encoded_specs:
            (temporary / relative).write_bytes(payload)
        for relative, payload in artifacts:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (temporary / "semantic_conflict_ledger.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in conflicts), encoding="utf-8"
        )
        (temporary / "semantic_compiler_manifest.txt").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from lerobot.annotation import assemble_diverse_corrected_store as store_assembler
        store_assembler.load_corrections(temporary)
        store_assembler.validate_trace_artifacts(source_root, temporary)
        temporary.rename(out_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--containment-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    report = compile_directory(
        args.source_root.resolve(), args.sweep_dir.resolve(), args.containment_manifest.resolve(),
        args.out_dir.resolve(), dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
