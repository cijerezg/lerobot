"""Normalize round-1 diverse audit fixes, including documented compact forms.

This module is deliberately separate from the audit renderer.  It expands the
three compact correction formats used in round 1 into the same full-row,
full-episode replacement contract accepted by ``diverse_corrected_store``.
Unknown compact or surgical operations remain fatal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.annotation import diverse_fix_normalizer as v1
from lerobot.annotation.speed import hybrid_motion_speed, joint_motion_speed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _source_rows(root: Path, store: str, name: str, episode: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(root / store / name)
    return [copy.deepcopy(row) for row in rows if str(row.get("episode_id")) == episode]


def _replace_exact(rows: list[dict[str, Any]], old: dict[str, Any], new: dict[str, Any], where: str) -> None:
    hits = [i for i, row in enumerate(rows) if row == old]
    if len(hits) != 1:
        raise v1.NormalizationError(f"{where}: exact replacement matched {len(hits)} rows, expected one")
    rows[hits[0]] = copy.deepcopy(new)


def _remove_exact(rows: list[dict[str, Any]], item: dict[str, Any], where: str) -> None:
    hits = [i for i, row in enumerate(rows) if row == item]
    if len(hits) != 1:
        raise v1.NormalizationError(f"{where}: exact removal matched {len(hits)} rows, expected one")
    rows.pop(hits[0])


def _expand_surgical(
    doc: dict[str, Any], root: Path, store: str, episode: str, origin: Path
) -> dict[str, list[dict[str, Any]]]:
    """Expand the narrowly defined FMB add/remove/replace operations."""
    out: dict[str, list[dict[str, Any]]] = {}
    operations = doc.get("operations", {})
    if not isinstance(operations, dict):
        raise v1.NormalizationError(f"{origin}: operations must be an object")
    for name, operation in operations.items():
        if name not in v1.CHANNELS or not isinstance(operation, dict):
            continue
        if "replace_complete_episode_rows" in operation:
            rows = operation["replace_complete_episode_rows"]
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise v1.NormalizationError(f"{origin}: {name} replacement rows must be objects")
            out[name] = copy.deepcopy(rows)
            continue
        allowed = {"add", "remove", "replace"}
        if not set(operation).issubset(allowed):
            raise v1.NormalizationError(f"{origin}: unsupported surgical keys for {name}: {sorted(set(operation) - allowed)}")
        rows = _source_rows(root, store, name, episode)
        for item in operation.get("remove", []):
            if not isinstance(item, dict):
                raise v1.NormalizationError(f"{origin}: {name}.remove entries must be objects")
            _remove_exact(rows, item, f"{origin}:{name}.remove")
        for item in operation.get("replace", []):
            if not isinstance(item, dict) or not {"match", "replacement"}.issubset(item) or set(item) - {"match", "replacement", "reason", "note"}:
                raise v1.NormalizationError(f"{origin}: {name}.replace needs exact match and replacement")
            if not isinstance(item["match"], dict) or not isinstance(item["replacement"], dict):
                raise v1.NormalizationError(f"{origin}: {name}.replace values must be objects")
            _replace_exact(rows, item["match"], item["replacement"], f"{origin}:{name}.replace")
        additions = operation.get("add", [])
        if not isinstance(additions, list) or any(not isinstance(row, dict) for row in additions):
            raise v1.NormalizationError(f"{origin}: {name}.add entries must be objects")
        rows.extend(copy.deepcopy(additions))
        out[name] = rows
    return out


def _nearest_template(rows: list[dict[str, Any]], start: int, end: int, subtask: str) -> dict[str, Any]:
    if not rows:
        return {}
    def score(row: dict[str, Any]) -> tuple[int, int, int]:
        a = int(row.get("start_timestep", 0))
        b = int(row.get("end_timestep_exclusive", a))
        overlap = max(0, min(end, b) - max(start, a))
        same = int(row.get("subtask") == subtask or row.get("verb") == subtask)
        distance = -abs(a - start)
        return same, overlap, distance
    return copy.deepcopy(max(rows, key=score))


def _derive_legacy_speed(
    root: Path, store: str, episode: str, atoms: list[dict[str, Any]], duration_priors: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    source = _source_rows(root, store, "speed_atoms.jsonl", episode)
    out = []
    for atom in atoms:
        start, end = int(atom["start_timestep"]), int(atom["end_timestep_exclusive"])
        key = (int(atom["parent_interval_index"]), int(atom["atom_index"]))
        row = _nearest_template(source, start, end, str(atom["subtask"]))
        row.update({
            "episode_id": episode,
            "parent_interval_index": key[0],
            "atom_index": key[1],
            "start_timestep": start,
            "end_timestep_exclusive": end,
            "duration_s": (end - start) / float(atom["native_rate_hz"]),
            "subtask": atom["subtask"],
            "verb": atom.get("verb", atom["subtask"]),
        })
        if duration_priors and key in duration_priors:
            prior = float(duration_priors[key])
            row.update({"speed": prior, "raw_speed": prior, "speed_source": "round1_duration_prior"})
        out.append(row)
    return out


def _merge_common(common: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(common)
    merged.update(copy.deepcopy(row))
    return merged


def _expand_layout(operation: dict[str, Any], where: str) -> list[dict[str, Any]]:
    rows = operation.get("replace_episode_rows")
    layout = operation.get("row_layout")
    common = operation.get("common_row_requirements", {})
    if not isinstance(rows, list) or not isinstance(layout, list) or not isinstance(common, dict):
        raise v1.NormalizationError(f"{where}: compact rows require row_layout and common_row_requirements")
    expanded = []
    for values in rows:
        if not isinstance(values, list) or len(values) != len(layout):
            raise v1.NormalizationError(f"{where}: row does not match row_layout")
        expanded.append(_merge_common(common, dict(zip(layout, values, strict=True))))
    return expanded


def _expand_object_rows(operation: dict[str, Any], where: str) -> list[dict[str, Any]]:
    rows = operation.get("replace_episode_rows")
    common = operation.get("common_row_requirements", {})
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) or not isinstance(common, dict):
        raise v1.NormalizationError(f"{where}: compact replacement must contain object rows")
    return [_merge_common(common, row) for row in rows]


def _complete_atoms(rows: list[dict[str, Any]], fps: float, episode: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        start, end = int(row["start_timestep"]), int(row["end_timestep_exclusive"])
        row.update({
            "episode_id": episode,
            "native_rate_hz": fps,
            "start_s": start / fps,
            "end_s_exclusive": end / fps,
            "duration_s": (end - start) / fps,
        })
        row.setdefault("boundary_provenance", "diverse_sampled_audit_round1")
        out.append(row)
    return out


def _join_atom_fields(rows: list[dict[str, Any]], atoms: list[dict[str, Any]], episode: str) -> list[dict[str, Any]]:
    by_key = {(int(row["parent_interval_index"]), int(row["atom_index"])): row for row in atoms}
    out = []
    for row in rows:
        key = (int(row["parent_interval_index"]), int(row["atom_index"]))
        atom = by_key.get(key)
        if atom is None:
            raise v1.NormalizationError(f"{episode}: row references unknown atom {key!r}")
        row.update({
            "episode_id": episode,
            "source": atom.get("source"),
            "embodiment": atom.get("embodiment"),
            "subtask": atom["subtask"],
            "verb": atom.get("verb", atom["subtask"]),
            "start_timestep": atom["start_timestep"],
            "end_timestep_exclusive": atom["end_timestep_exclusive"],
        })
        out.append(row)
    return out


def _group(atom: dict[str, Any]) -> str:
    source = str(atom["source"])
    if source in {"droid", "droid_success"}:
        return "droid"
    if source == "robochallenge":
        return f"robochallenge/{atom['embodiment']}"
    return source


@lru_cache(maxsize=2)
def _motion_pools(root: Path) -> dict[str, np.ndarray]:
    groups: dict[str, list[np.ndarray]] = {}
    for store in ("corpus", "fmb"):
        store_root = root / store
        episodes: dict[str, list[dict[str, Any]]] = {}
        for atom in _read_jsonl(store_root / "subtask_atoms.jsonl"):
            episodes.setdefault(str(atom["episode_id"]), []).append(atom)
        for episode, ep_atoms in episodes.items():
            if not any(atom.get("split") == "train" for atom in ep_atoms):
                continue
            episode_dir = store_root / "episodes" / episode
            q = np.load(episode_dir / ("q.npy" if store == "fmb" else "state.npy"), mmap_mode="r")
            if store != "fmb":
                q = q[:, :-1]
            values, valid = joint_motion_speed.motion_trace(q, float(ep_atoms[0]["native_rate_hz"]))
            bucket = groups.setdefault(_group(ep_atoms[0]), [])
            for atom in ep_atoms:
                if atom.get("split") != "train":
                    continue
                a, b = int(atom["start_timestep"]), int(atom["end_timestep_exclusive"])
                bucket.append(np.asarray(values[a : b - 1][valid[a : b - 1]], dtype=np.float64))
    return {group: np.sort(np.concatenate(values)) for group, values in groups.items() if values}


def _hybrid_rows(
    root: Path,
    store: str,
    episode: str,
    atoms: list[dict[str, Any]],
    priors: dict[str, float],
    duration_used: dict[tuple[int, int], bool] | None = None,
) -> list[dict[str, Any]]:
    episode_dir = root / store / "episodes" / episode
    state_path = episode_dir / "state.npy"
    q_path = episode_dir / "q.npy"
    if state_path.exists():
        q = np.load(state_path, mmap_mode="r")[:, :-1]
        state_file = str(state_path)
    elif q_path.exists():
        q = np.load(q_path, mmap_mode="r")
        state_file = str(q_path)
    else:
        raise v1.SpecError(f"{episode}: cannot recompute hybrid speed without state.npy or q.npy")
    fps = float(atoms[0]["native_rate_hz"])
    pool_map = _motion_pools(root)
    group = _group(atoms[0])
    if group not in pool_map:
        raise v1.NormalizationError(f"{episode}: no motion calibration pool for {group}")
    inputs = []
    for atom in atoms:
        key = (int(atom["parent_interval_index"]), int(atom["atom_index"]))
        prior = priors.get(key)
        if prior is None or not math.isfinite(prior):
            raise v1.NormalizationError(f"{episode}: missing duration prior for {key}")
        inputs.append({
            "start_timestep": int(atom["start_timestep"]),
            "end_timestep_exclusive": int(atom["end_timestep_exclusive"]),
            "duration_speed": float(prior),
            "use_duration": (
                bool(duration_used[key]) if duration_used is not None and key in duration_used
                else str(atom.get("verb")) != "return"
            ),
        })
    trace = hybrid_motion_speed.hybrid_trace(np.asarray(q), fps, inputs, pool_map[group])
    source_templates = _source_rows(root, store, "speed_atoms_hybrid_v1.jsonl", episode)
    out = []
    for atom_input, atom in zip(inputs, atoms, strict=True):
        a, b = atom_input["start_timestep"], atom_input["end_timestep_exclusive"]
        span = slice(a, max(a, b - 1))
        values = trace["speed_score"][span]
        reason = "fewer than two state transitions in the atom" if len(values) < 2 else (
            "nonfinite state data" if not np.isfinite(values).all() else ""
        )
        score = float(np.median(values)) if not reason else 3.0
        motion_score = float(np.median(trace["motion_score"][span])) if not reason else 3.0
        measured = float(np.median(trace["joint_speed_rad_s"][span])) if not reason else None
        flags = [reason] if reason else []
        if abs(int(hybrid_motion_speed.quantize(motion_score)) - atom_input["duration_speed"]) >= 2:
            flags.append("motion and duration differ by at least two buckets; adjustment capped")
        row = _nearest_template(source_templates, a, b, str(atom["subtask"]))
        row.update({
            "episode_id": episode,
            "parent_interval_index": int(atom["parent_interval_index"]),
            "atom_index": int(atom["atom_index"]),
            "start_timestep": a,
            "end_timestep_exclusive": b,
            "duration_s": (b - a) / fps,
            "subtask": atom["subtask"],
            "class": atom.get("verb"),
            "group": group,
            "annotation_layer": "atoms",
            "confidence": atom.get("confidence", "confident"),
            "method": hybrid_motion_speed.METHOD,
            "duration_speed": atom_input["duration_speed"],
            "duration_used": atom_input["use_duration"],
            "speed": 3 if reason else int(hybrid_motion_speed.quantize(score)),
            "speed_score": score,
            "speed_source": "default_unclear" if reason else hybrid_motion_speed.METHOD,
            "speed_default_reason": reason,
            "speed_flags": flags,
            "motion_speed": int(hybrid_motion_speed.quantize(motion_score)),
            "motion_score": motion_score,
            "motion_median_rad_s": measured,
            "state_speed_file": state_file,
        })
        out.append(row)
    return out


TRACE_KEYS = {
    "joint_speed_rad_s", "motion_score", "duration_score", "speed_score",
    "valid", "supervision_mask", "speed", "native_rate_hz",
}


def _trace_artifact(root: Path, spec: dict[str, Any]) -> bytes:
    """Build the canonical per-transition hybrid trace for an atom replacement."""
    store, episode = str(spec["store"]), str(spec["episode_id"])
    atoms = spec["replacements"]["subtask_atoms.jsonl"]
    speed_rows = spec["replacements"]["speed_atoms_hybrid_v1.jsonl"]
    by_key = {
        (int(row["parent_interval_index"]), int(row["atom_index"])): row for row in speed_rows
    }
    if len(by_key) != len(atoms):
        raise v1.NormalizationError(f"{episode}: hybrid speed rows do not cover corrected atoms")
    episode_dir = root / store / "episodes" / episode
    q = np.load(episode_dir / ("q.npy" if store == "fmb" else "state.npy"), mmap_mode="r")
    if store != "fmb":
        q = q[:, :-1]
    inputs = []
    for atom in atoms:
        key = (int(atom["parent_interval_index"]), int(atom["atom_index"]))
        row = by_key.get(key)
        if row is None:
            raise v1.NormalizationError(f"{episode}: no hybrid row for atom {key}")
        inputs.append({
            "start_timestep": int(atom["start_timestep"]),
            "end_timestep_exclusive": int(atom["end_timestep_exclusive"]),
            "duration_speed": float(row["duration_speed"]),
            "use_duration": bool(row["duration_used"]),
        })
    fps = float(atoms[0]["native_rate_hz"])
    pools = _motion_pools(root)
    group = _group(atoms[0])
    if group not in pools:
        raise v1.NormalizationError(f"{episode}: no motion calibration pool for {group}")
    trace = hybrid_motion_speed.hybrid_trace(np.asarray(q), fps, inputs, pools[group])
    labels = np.zeros(len(trace["supervision_mask"]), dtype=np.uint8)
    labels[trace["supervision_mask"]] = 3
    usable = trace["supervision_mask"] & trace["valid"]
    labels[usable] = hybrid_motion_speed.quantize(trace["speed_score"][usable])
    payload = {**trace, "speed": labels, "native_rate_hz": np.asarray(fps, dtype=np.float64)}
    if set(payload) != TRACE_KEYS:
        raise v1.NormalizationError(f"{episode}: internal trace schema mismatch")
    output = io.BytesIO()
    np.savez_compressed(output, **payload)
    return output.getvalue()


def _normalize_doc(doc: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    store = v1._infer_store(doc, path)
    episode = str(doc.get("episode_id", ""))
    if not episode:
        raise v1.NormalizationError(f"{path}: episode_id is missing")
    retained = v1._retained_intervals(doc, path)
    operations = doc.get("operations", {})
    if episode == "ur7e__stack_block__ep000041":
        mapping: dict[str, list[dict[str, Any]]] = {}
        atom_op = operations.get("subtask_atoms.jsonl", {})
        atoms = _complete_atoms(_expand_object_rows(atom_op, f"{path}:subtask_atoms"), 30.0, episode)
        eligible = {
            int(row["interval_index"]): bool(row["critic_eligible"])
            for row in operations["critic_intervals.jsonl"]["replace_episode_rows"]
        }
        for atom in atoms:
            atom["parent_critic_eligible"] = eligible[int(atom["parent_interval_index"])]
            atom.setdefault("boundary_provenance", "diverse_sampled_audit_round1")
            atom.setdefault("end_boundary_provenance", "diverse_sampled_audit_round1")
            atom.setdefault("note", "")
            atom.setdefault("parent_note", "")
        mapping["subtask_atoms.jsonl"] = atoms
        for name in ("contact_atoms.jsonl", "precision_atoms.jsonl"):
            mapping[name] = _join_atom_fields(_expand_layout(operations[name], f"{path}:{name}"), atoms, episode)
        hybrid_compact = _expand_layout(operations["speed_atoms_hybrid_v1.jsonl"], f"{path}:hybrid_speed")
        priors = {
            (int(row["parent_interval_index"]), int(row["atom_index"])): float(row["duration_speed_prior"])
            for row in hybrid_compact
        }
        mapping["speed_atoms.jsonl"] = _derive_legacy_speed(root, store, episode, atoms, priors)
        mapping["speed_atoms_hybrid_v1.jsonl"] = _hybrid_rows(root, store, episode, atoms, priors)
        for name in ("quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl"):
            rows = operations[name].get("replace_episode_rows")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise v1.NormalizationError(f"{path}:{name}: replacement rows must be objects")
            mapping[name] = copy.deepcopy(rows)
        critic_rows = _expand_object_rows(operations["critic_intervals.jsonl"], f"{path}:critic")
        for row in critic_rows:
            start, end = int(row["start_timestep"]), int(row["end_timestep_exclusive"])
            row.update({
                "episode_id": episode,
                "start_s": start / 30.0,
                "end_s_exclusive": end / 30.0,
                "duration_s": (end - start) / 30.0,
                "native_action_samples": end - start,
            })
        critic = {"mode": "replace", "rows": critic_rows}
        prefix = "stack_block__ep000041_"
        for row in mapping.get("precision_windows.jsonl", []):
            uid = str(row.get("uid", ""))
            if uid:
                row["uid"] = prefix + uid.split("__ep000041_", 1)[-1]
        record = json.loads((root / store / "episodes" / episode / "episode.json").read_text())
        episode_op = operations.get("episode.json", {})
        annotations = copy.deepcopy(record.get("annotations", {}))
        segments = copy.deepcopy(episode_op.get("replace_annotations_segments", []))
        timestamps = np.load(root / store / "episodes" / episode / "timestamp_s.npy", mmap_mode="r")
        terminal_end_s = float(timestamps[-1]) + 1.0 / 30.0
        for segment in segments:
            for field in ("start_s", "end_s"):
                source_time = float(segment[field])
                frame = round(source_time * 30.0)
                if field == "end_s" and source_time >= float(timestamps[-1]):
                    segment[field] = terminal_end_s
                elif 0 <= frame < len(timestamps):
                    segment[field] = float(timestamps[frame])
                elif frame == len(timestamps):
                    segment[field] = terminal_end_s
                else:
                    raise v1.NormalizationError(
                        f"{path}: UR7e episode segment boundary frame {frame} is outside timestamp authority"
                    )
        keep_intervals = [[float(timestamps[round(start * 30.0)]),
                           terminal_end_s if float(stop) >= float(timestamps[-1]) else
                           float(timestamps[round(stop * 30.0)])]
                          for start, stop in episode_op.get("replace_annotations_source_keep_intervals_s", [])]
        annotations.update({
            "segments": segments,
            "source_keep_intervals_s": keep_intervals,
            "reviewer_notes": episode_op.get("replace_annotations_reviewer_notes", ""),
        })
        record["annotations"] = annotations
        retained_frames = 1555
    else:
        mapping = _expand_surgical(doc, root, store, episode, path)
        direct = doc.get("replacement_episode_rows", {})
        if isinstance(direct, dict):
            for name, rows in direct.items():
                if name in v1.CHANNELS:
                    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                        raise v1.NormalizationError(f"{path}: {name} replacement rows must be objects")
                    mapping[name] = copy.deepcopy(rows)
        atoms = mapping.get("subtask_atoms.jsonl")
        if atoms is not None and "speed_atoms.jsonl" not in mapping:
            mapping["speed_atoms.jsonl"] = _derive_legacy_speed(root, store, episode, atoms)
        critic = {"mode": "from_atoms"} if atoms is not None else {"mode": "preserve"}
        record = v1._episode_record(root, store, episode, doc)
        if episode == "droid__WEIRD__ep013616" and record is None:
            # This sampled correction extends the visually confirmed release
            # through f350. Carry the same authority into episode.json so the
            # new store never pairs corrected atoms with source-era retention.
            record_path = root / store / "episodes" / episode / "episode.json"
            record = json.loads(record_path.read_text())
            timestamps = np.load(record_path.parent / "timestamp_s.npy", mmap_mode="r")
            if retained != [[90, 350]]:
                raise v1.NormalizationError(f"{path}: unexpected WEIRD retention {retained}")
            end_s = float(timestamps[350])
            annotations = copy.deepcopy(record.get("annotations", {}))
            segments = annotations.get("segments")
            if not isinstance(segments, list):
                raise v1.NormalizationError(f"{path}: WEIRD episode annotations lack segments")
            keep_rows = [row for row in segments if row.get("retention") == "keep"]
            if not keep_rows:
                raise v1.NormalizationError(f"{path}: WEIRD episode annotations lack keep segments")
            final_keep = max(keep_rows, key=lambda row: float(row["end_s"]))
            old_end = float(final_keep["end_s"])
            following = [row for row in segments if row.get("retention") != "keep"
                         and math.isclose(float(row["start_s"]), old_end, abs_tol=1e-5)]
            if len(following) != 1:
                raise v1.NormalizationError(f"{path}: WEIRD final keep/reject boundary is not unique")
            final_keep["end_s"] = end_s
            following[0]["start_s"] = end_s
            annotations["segments"] = segments
            annotations["source_keep_intervals_s"] = [[float(timestamps[90]), end_s]]
            annotations["reviewer_notes"] = (
                str(annotations.get("reviewer_notes", ""))
                + " Sampled audit extends the confirmed release through native frame 350."
            ).strip()
            record["annotations"] = annotations
        if episode == "yam__espresso__ep000086" and record is None:
            record_path = root / store / "episodes" / episode / "episode.json"
            record = json.loads(record_path.read_text())
            timestamps = np.load(record_path.parent / "timestamp_s.npy", mmap_mode="r")
            if retained != [[0, 20], [82, 843]]:
                raise v1.NormalizationError(f"{path}: unexpected YAM retention {retained}")
            episode_op = operations.get("episode.json", {})
            fields = copy.deepcopy(episode_op.get("replace_annotation_fields"))
            if not isinstance(fields, dict):
                raise v1.NormalizationError(f"{path}: YAM episode.json replacement fields are missing")
            segments = fields.get("segments")
            if not isinstance(segments, list) or len(segments) != 4:
                raise v1.NormalizationError(f"{path}: YAM episode segments are not the reviewed four-way split")
            boundaries = [0, 20, 82, 843, 924]
            for segment, start, stop in zip(segments, boundaries[:-1], boundaries[1:], strict=True):
                segment["start_s"] = float(timestamps[start])
                segment["end_s"] = float(timestamps[stop])
            fields["segments"] = segments
            fields["source_keep_intervals_s"] = [
                [float(timestamps[0]), float(timestamps[20])],
                [float(timestamps[82]), float(timestamps[843])],
            ]
            annotations = copy.deepcopy(record.get("annotations", {}))
            annotations.update(fields)
            record["annotations"] = annotations
            for dotted, value in episode_op.get("review_fields_to_update_consistently", {}).items():
                if not dotted.startswith("review."):
                    raise v1.NormalizationError(f"{path}: unsupported YAM episode field {dotted}")
                record.setdefault("review", {})[dotted.split(".", 1)[1]] = value
        retained_frames = sum(stop - start for start, stop in retained)

    if "subtask_atoms.jsonl" in mapping:
        required = v1.CHANNELS - {"subtask_atoms.jsonl"}
        missing = sorted(required - set(mapping))
        if missing:
            raise v1.NormalizationError(f"{path}: atom repair lacks full replacements for {missing}")
        actor = {"mode": "remap", "retained_intervals": retained}
    else:
        actor = {"mode": "preserve"}
    return {
        "schema_version": 1,
        "status": "resolved",
        "store": store,
        "episode_id": episode,
        "episode_record": record,
        "replacements": dict(sorted(mapping.items())),
        "actor_anchors": actor,
        "critic_intervals": critic,
        "unresolved": [],
        "second_read_required": False,
        "normalization_provenance": {
            "source_spec": str(path),
            "source_spec_sha256": v1._sha256(path),
            "source_schema_version": doc.get("schema_version"),
            "full_episode_replacements_only": True,
            "corrected_retained_frames": retained_frames,
        },
    }


def normalize_one(path: Path, root: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    episode = str(doc.get("episode_id", ""))
    if episode in {
        "droid__WEIRD__ep013616", "episode_000061_6_S_L_4_vertical_n_5",
        "robochallenge__pick_out_the_green_blocks__ep000285", "ur7e__stack_block__ep000041",
        "yam__espresso__ep000086",
    }:
        result = _normalize_doc(doc, path, root)
    else:
        result = v1.normalize_one(path, root)
    if episode == "droid__AUTOLab__ep002365":
        record = result.get("episode_record")
        if not isinstance(record, dict):
            raise v1.NormalizationError(f"{path}: AUTOLab episode_record is required")
        timestamps = np.load(root / result["store"] / "episodes" / episode / "timestamp_s.npy", mmap_mode="r")
        annotations = record.get("annotations", {})
        expected = [[float(timestamps[0]), float(timestamps[564])]]
        if annotations.get("source_keep_intervals_s") != expected:
            raise v1.NormalizationError(
                f"{path}: AUTOLab source_keep_intervals_s must be exact [0,564) timestamps"
            )
        segments = annotations.get("segments", [])
        kept = [(float(row["start_s"]), float(row["end_s"])) for row in segments
                if row.get("retention") == "keep"]
        if not kept or min(row[0] for row in kept) != expected[0][0] or max(row[1] for row in kept) != expected[0][1]:
            raise v1.NormalizationError(f"{path}: AUTOLab keep segments do not cover exact [0,564)")
    return result


def normalize_directory(source_root: Path, fix_dir: Path, out_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    if not source_root.is_dir() or not fix_dir.is_dir():
        raise ValueError("source-root and fix-dir must exist")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    normalized: list[tuple[Path, dict[str, Any]]] = []
    refused = []
    for path in sorted(fix_dir.glob("*.json")):
        try:
            normalized.append((path, normalize_one(path, source_root)))
        except Exception as exc:
            refused.append({"source": path.name, "error": str(exc)})
    artifacts: list[tuple[Path, bytes]] = []
    if not refused:
        for _, result in normalized:
            if "subtask_atoms.jsonl" not in result.get("replacements", {}):
                continue
            payload = _trace_artifact(source_root, result)
            relative = Path("artifacts") / result["store"] / "speed_hybrid_v1" / f"{result['episode_id']}.npz"
            digest = hashlib.sha256(payload).hexdigest()
            result["artifacts"] = {
                "speed_hybrid_v1": {
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "keys": sorted(TRACE_KEYS),
                }
            }
            for row in result["replacements"]["speed_atoms_hybrid_v1.jsonl"]:
                row["state_speed_file"] = f"{result['store']}/speed_hybrid_v1/{result['episode_id']}.npz"
            artifacts.append((relative, payload))
    report = {"source_root": str(source_root), "fix_dir": str(fix_dir), "out_dir": str(out_dir),
              "dry_run": dry_run, "normalized": len(normalized), "trace_artifacts": len(artifacts), "refused": refused}
    if refused:
        report["status"] = "REFUSED_UNRESOLVED_SPECS"
        return report
    report["status"] = "VALID_DRY_RUN" if dry_run else "NORMALIZED"
    if not dry_run:
        out_dir.mkdir(parents=True)
        for source, result in normalized:
            (out_dir / source.name).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        for relative, payload in artifacts:
            target = out_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (out_dir / "normalization_manifest.txt").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fix-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = normalize_directory(args.source_root.resolve(), args.fix_dir.resolve(), args.out_dir.resolve(), dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["refused"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
