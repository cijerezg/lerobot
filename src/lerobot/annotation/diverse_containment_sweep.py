"""Build a mechanical correction manifest for diverse retention containment.

The active source root is read-only.  This module writes only a manifest and a short
Markdown summary; a separate corrected-store builder consumes the manifest later.

Authority is source-declared retained footage:

* common corpus: ``episode.json`` annotation segments with ``retention == "keep"``,
  converted to native half-open frames with the stored timestamp array;
* FMB: every source-native primitive interval (``critic_eligible`` is a critic-view
  decision, not a retention decision).

The manifest contains full target parent/atom mappings, v2 sidecar row operations and
actor-anchor drops.  Per-atom contact/speed/legacy-precision rows inherit through the
atom mapping.  Every source row receives a deterministic keep/rekey/clip/split/drop
decision; ``open_candidates`` is empty or generation fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from lerobot.annotation.validate_diverse_audit import (
    half_second_frames,
    interval_at,
    merge_intervals,
    read_jsonl,
)
from lerobot.annotation.vocab import MISTAKE_TYPES
from lerobot.datasets.diverse_pilot import COPY_STATE_LEAD_S

STORES = ("corpus", "fmb")
PER_ATOM_SIDECARS = (
    "contact_atoms.jsonl",
    "speed_atoms_hybrid_v1.jsonl",
    "precision_atoms.jsonl",
)
V2_SIDECARS = ("quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl")
Interval = tuple[int, int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intersections(bounds: Interval, retained: Sequence[Interval]) -> list[Interval]:
    start, stop = bounds
    return [(max(start, lo), min(stop, hi)) for lo, hi in retained if max(start, lo) < min(stop, hi)]


def subtract_intervals(whole: Sequence[Interval], covered: Sequence[Interval]) -> list[Interval]:
    """Parts of ``whole`` not covered by ``covered`` (all half-open)."""
    out: list[Interval] = []
    covered = merge_intervals(covered)
    for start, stop in whole:
        cursor = start
        for lo, hi in covered:
            if hi <= cursor or stop <= lo:
                continue
            if cursor < lo:
                out.append((cursor, min(lo, stop)))
            cursor = max(cursor, hi)
            if cursor >= stop:
                break
        if cursor < stop:
            out.append((cursor, stop))
    return out


def source_retained_intervals(
    store_name: str,
    record: dict[str, Any],
    timestamps: np.ndarray,
) -> list[Interval]:
    if store_name == "fmb":
        spans = [
            (
                int(row["reviewed_start_timestep"] if "reviewed_start_timestep" in row else row["start_timestep"]),
                int(
                    row["reviewed_end_timestep_exclusive"]
                    if "reviewed_end_timestep_exclusive" in row
                    else row["end_timestep_exclusive"]
                ),
            )
            for row in record["primitive_intervals"]
        ]
    else:
        spans = [
            (
                int(np.searchsorted(timestamps, float(row["start_s"]), side="left")),
                int(np.searchsorted(timestamps, float(row["end_s"]), side="left")),
            )
            for row in record["annotations"]["segments"]
            if row["retention"] == "keep"
        ]
    return merge_intervals((start, stop) for start, stop in spans if start < stop)


def operation_name(source: Interval, targets: Sequence[Interval], *, rekeyed: bool = False) -> str:
    if not targets:
        return "drop"
    if len(targets) > 1:
        return "split"
    if targets[0] != source:
        return "clip"
    return "rekey" if rekeyed else "keep"


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["episode_id"]), int(row["parent_interval_index"]), int(row["atom_index"])


def _key_json(key: tuple[str, int, int]) -> dict[str, Any]:
    return {"episode_id": key[0], "parent_interval_index": key[1], "atom_index": key[2]}


def _uid(key: tuple[str, int, int]) -> str:
    return f"{key[0].split('__', 1)[-1]}_p{key[1]}a{key[2]}"


def _owner_for_piece(
    piece: Interval,
    source_key: tuple[str, int, int],
    target_atoms: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    same_source = [row for row in target_atoms if tuple(row["source_atom_key"]) == source_key]
    candidates = same_source or list(target_atoms)
    if not candidates:
        raise ValueError(f"No target atom can own {source_key} range {piece}")

    def score(row: dict[str, Any]) -> tuple[int, int, int]:
        overlap = max(0, min(piece[1], row["end_timestep_exclusive"]) - max(piece[0], row["start_timestep"]))
        distance = min(abs(piece[0] - row["end_timestep_exclusive"]), abs(piece[1] - row["start_timestep"]))
        return overlap, -distance, -row["start_timestep"]

    return max(candidates, key=score)


def remap_v2_row(
    channel: str,
    row: dict[str, Any],
    retained: Sequence[Interval],
    episode_atoms: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Target fragments for one v2 row, with effective bounds rematerialized once."""
    if channel == "quality_spans.jsonl":
        semantic = (int(row["raw_from_index"]), int(row["raw_to_index"]))
    elif channel == "precision_windows.jsonl":
        semantic = (int(row["raw_from_index"]), int(row["to_index"]))
    else:
        semantic = (int(row["from_index"]), int(row["to_index"]))
    targets = []
    source_key = _key(row)
    fps = float(row["native_rate_hz"])
    for fragment_index, piece in enumerate(intersections(semantic, retained)):
        owner = _owner_for_piece(piece, source_key, episode_atoms)
        target_key = tuple(owner["target_atom_key"])
        fields: dict[str, Any] = {
            "parent_interval_index": target_key[1],
            "atom_index": target_key[2],
            "uid": _uid(target_key),
        }
        containing = interval_at(retained, piece[0])
        if containing is None or piece[1] > containing[1]:
            raise AssertionError((piece, retained))
        if channel == "quality_spans.jsonl":
            fields.update(raw_from_index=piece[0], raw_to_index=piece[1], quality=int(row["quality"]))
            if int(row["quality"]) == 5:
                effective = piece
            else:
                effective = (
                    max(containing[0], piece[0] - round(fps)),
                    min(containing[1], piece[1] + half_second_frames(fps)),
                )
            fields.update(from_index=effective[0], to_index=effective[1])
        elif channel == "precision_windows.jsonl":
            fields.update(
                raw_from_index=piece[0],
                from_index=max(containing[0], piece[0] - round(fps)),
                to_index=piece[1],
                commit_index=min(max(int(row["commit_index"]), containing[0]), containing[1] - 1),
                precision=int(row["precision"]),
            )
        else:
            mistake_type = str(row["mistake_type"])
            if mistake_type not in MISTAKE_TYPES:
                raise ValueError(f"unknown mistake type: {mistake_type}")
            fields.update(
                from_index=piece[0],
                to_index=piece[1],
                mistake=bool(row["mistake"]),
                mistake_type=mistake_type,
            )
        targets.append(
            {
                "fragment_index": fragment_index,
                "target_atom_key": list(target_key),
                "fields": fields,
            }
        )
    return targets


def actor_window_decision(
    store_name: str,
    row: dict[str, Any],
    retained: Sequence[Interval],
    timestamps: np.ndarray,
) -> tuple[bool, list[str], int]:
    anchor = int(row.get("anchor_frame", row.get("anchor_timestep")))
    owner = interval_at(retained, anchor)
    reasons: list[str] = []
    if owner is None:
        return False, ["anchor_outside_source_retention"], -1
    history = [int(frame) for frame in row.get("history_frames", [])]
    if not history or any(not owner[0] <= frame < owner[1] for frame in history):
        reasons.append("history_crosses_source_retention")
    future_end_s = (
        float(row["future_end_s_nominal"]) + float(COPY_STATE_LEAD_S)
        if store_name == "fmb"
        else float(row["future_end_s"])
    )
    future_frame = int(np.searchsorted(timestamps, future_end_s, side="left"))
    if future_end_s > float(timestamps[-1]) + 1e-6 or not owner[0] <= future_frame < owner[1]:
        reasons.append("future_crosses_source_retention")
    return not reasons, reasons, future_frame


def _episode_source(store_name: str, summary: dict[str, Any]) -> str:
    return "fmb" if store_name == "fmb" else str(summary["source"])


def _episode_frames(summary: dict[str, Any]) -> int:
    return int(summary.get("frames", summary.get("frame_count")))


def _episode_rate(summary: dict[str, Any]) -> float:
    return float(summary.get("native_rate_hz", summary.get("nominal_fps")))


def build_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest: dict[str, Any] = {
        "format": "diverse_mechanical_containment_v1",
        "source_root": str(root),
        "authority": {
            "corpus": "episode.json annotations.segments where retention=keep, mapped through timestamp_s.npy",
            "fmb": "episode.json primitive_intervals; critic_eligible does not alter actor retention",
        },
        "per_atom_sidecars_inherited": list(PER_ATOM_SIDECARS),
        "source_hashes": {},
        "episodes": [],
        "target_parents": [],
        "atom_operations": [],
        "v2_operations": {name: [] for name in V2_SIDECARS},
        "actor_anchor_drops": [],
        "open_candidates": [],
    }
    counts: dict[str, Any] = {
        "episodes": Counter(),
        "parents": Counter(),
        "atoms": Counter(),
        "actor_anchors": Counter(),
        "v2": {name: Counter() for name in V2_SIDECARS},
    }
    all_target_atoms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_episode_ids: set[str] = set()

    for store_name in STORES:
        store = root / store_name
        required = ["episodes.jsonl", "critic_intervals.jsonl", "subtask_atoms.jsonl", "actor_anchors_5hz.jsonl"]
        required += list(PER_ATOM_SIDECARS) + list(V2_SIDECARS)
        for name in required:
            path = store / name
            if not path.is_file():
                raise FileNotFoundError(path)
            manifest["source_hashes"][f"{store_name}/{name}"] = sha256(path)

        summaries = {str(row["episode_id"]): row for row in read_jsonl(store / "episodes.jsonl")}
        if len(summaries) != len(read_jsonl(store / "episodes.jsonl")):
            raise ValueError(f"{store}: duplicate episode ids")
        collision = source_episode_ids & set(summaries)
        if collision:
            raise ValueError(f"cross-store episode ids: {sorted(collision)[:3]}")
        source_episode_ids |= set(summaries)

        parents_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(store / "critic_intervals.jsonl"):
            parents_by_episode[str(row["episode_id"])].append(row)
        atoms_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        source_atoms: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in read_jsonl(store / "subtask_atoms.jsonl"):
            atoms_by_episode[str(row["episode_id"])].append(row)
            source_atoms[_key(row)] = row

        for episode_id, summary in summaries.items():
            episode_dir = store / str(summary.get("directory", f"episodes/{episode_id}"))
            record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
            timestamps = np.load(episode_dir / "timestamp_s.npy", mmap_mode="r")
            frames, rate = _episode_frames(summary), _episode_rate(summary)
            if len(timestamps) != frames:
                raise ValueError(f"{episode_id}: timestamp rows {len(timestamps)} != {frames}")
            # Retention authority always comes from the episode record, including
            # for staged corrected stores. Treating the current atom table as
            # authority here would make a re-scan circular: malformed atoms would
            # define their own allowed footage and unresolved source gaps would be
            # hidden. Audited retention revisions must therefore be materialized
            # explicitly in the staged episode.json.
            retained = source_retained_intervals(store_name, record, timestamps)
            if not retained:
                raise ValueError(f"{episode_id}: source declares no retained footage")
            gaps = [(left[1], right[0]) for left, right in zip(retained, retained[1:]) if left[1] < right[0]]
            manifest["episodes"].append(
                {
                    "store": store_name,
                    "episode_id": episode_id,
                    "source": _episode_source(store_name, summary),
                    "split": str(summary["split"]),
                    "native_rate_hz": rate,
                    "frames": frames,
                    "retained_intervals": [list(x) for x in retained],
                    "excluded_internal_gaps": [list(x) for x in gaps],
                }
            )
            counts["episodes"][store_name] += 1

            # Intersect every current parent with source authority, then assign target
            # parent indices in native temporal order.
            parent_pieces: list[dict[str, Any]] = []
            for parent in parents_by_episode[episode_id]:
                source_parent = int(parent["interval_index"])
                source_range = (int(parent["start_timestep"]), int(parent["end_timestep_exclusive"]))
                for fragment_index, piece in enumerate(intersections(source_range, retained)):
                    parent_pieces.append(
                        {
                            "source_parent_interval_index": source_parent,
                            "source_line": int(parent["_line"]),
                            "fragment_index": fragment_index,
                            "start_timestep": piece[0],
                            "end_timestep_exclusive": piece[1],
                        }
                    )
            parent_pieces.sort(key=lambda row: (row["start_timestep"], row["source_parent_interval_index"]))
            for target_parent, piece in enumerate(parent_pieces):
                piece["store"] = store_name
                piece["episode_id"] = episode_id
                piece["target_parent_interval_index"] = target_parent
                piece["split"] = str(summary["split"])
                piece["native_rate_hz"] = rate
                manifest["target_parents"].append(piece)
            target_parent_union = merge_intervals(
                (row["start_timestep"], row["end_timestep_exclusive"]) for row in parent_pieces
            )
            missing = subtract_intervals(retained, target_parent_union)
            if missing:
                raise ValueError(f"{episode_id}: parents do not cover authority {missing}")
            counts["parents"]["source"] += len(parents_by_episode[episode_id])
            counts["parents"]["target"] += len(parent_pieces)

            target_parents_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in parent_pieces:
                target_parents_by_source[int(row["source_parent_interval_index"])].append(row)
            atom_pieces: list[dict[str, Any]] = []
            source_to_pieces: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
            for atom in atoms_by_episode[episode_id]:
                source_key = _key(atom)
                source_range = (int(atom["start_timestep"]), int(atom["end_timestep_exclusive"]))
                for parent_piece in target_parents_by_source[source_key[1]]:
                    target_parent_range = (
                        int(parent_piece["start_timestep"]),
                        int(parent_piece["end_timestep_exclusive"]),
                    )
                    for piece in intersections(source_range, [target_parent_range]):
                        target = {
                            "store": store_name,
                            "episode_id": episode_id,
                            "source_atom_key": list(source_key),
                            "source_line": int(atom["_line"]),
                            "target_parent_interval_index": int(parent_piece["target_parent_interval_index"]),
                            "start_timestep": piece[0],
                            "end_timestep_exclusive": piece[1],
                            "subtask": str(atom["subtask"]),
                            "split": str(summary["split"]),
                            "native_rate_hz": rate,
                        }
                        atom_pieces.append(target)
                        source_to_pieces[source_key].append(target)
            by_target_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in atom_pieces:
                by_target_parent[int(row["target_parent_interval_index"])].append(row)
            for target_parent, rows in by_target_parent.items():
                rows.sort(key=lambda row: (row["start_timestep"], row["source_atom_key"]))
                parent = next(row for row in parent_pieces if row["target_parent_interval_index"] == target_parent)
                cursor = int(parent["start_timestep"])
                for target_atom, row in enumerate(rows):
                    if row["start_timestep"] != cursor:
                        raise ValueError(f"{episode_id} P{target_parent}: atom gap at {cursor}->{row['start_timestep']}")
                    row["target_atom_index"] = target_atom
                    row["target_atom_key"] = [episode_id, target_parent, target_atom]
                    cursor = int(row["end_timestep_exclusive"])
                if cursor != int(parent["end_timestep_exclusive"]):
                    raise ValueError(f"{episode_id} P{target_parent}: atoms end {cursor}, parent ends {parent['end_timestep_exclusive']}")
                all_target_atoms[episode_id].extend(rows)

            for atom in atoms_by_episode[episode_id]:
                source_key = _key(atom)
                source_range = (int(atom["start_timestep"]), int(atom["end_timestep_exclusive"]))
                targets = sorted(source_to_pieces[source_key], key=lambda row: row["start_timestep"])
                target_ranges = [(row["start_timestep"], row["end_timestep_exclusive"]) for row in targets]
                rekeyed = bool(targets) and tuple(targets[0]["target_atom_key"]) != source_key
                manifest["atom_operations"].append(
                    {
                        "store": store_name,
                        "source_line": int(atom["_line"]),
                        "source_atom_key": list(source_key),
                        "source_range": list(source_range),
                        "action": operation_name(source_range, target_ranges, rekeyed=rekeyed),
                        "targets": [
                            {
                                "target_atom_key": row["target_atom_key"],
                                "start_timestep": row["start_timestep"],
                                "end_timestep_exclusive": row["end_timestep_exclusive"],
                                "subtask": row["subtask"],
                                "inherit_per_atom_sidecars_from": list(source_key),
                            }
                            for row in targets
                        ],
                    }
                )
                counts["atoms"][operation_name(source_range, target_ranges, rekeyed=rekeyed)] += 1
            counts["atoms"]["source"] += len(atoms_by_episode[episode_id])
            counts["atoms"]["target"] += len(atom_pieces)

        # Per-atom join sources must be complete before a builder inherits them.
        for name in PER_ATOM_SIDECARS:
            rows = read_jsonl(store / name)
            keys = [_key(row) for row in rows]
            if len(keys) != len(set(keys)) or set(keys) != set(source_atoms):
                raise ValueError(f"{store_name}/{name}: per-atom join is not one-to-one")

        # V2 operations are full-source ledgers.  The builder copies unchanged fields
        # and applies each target's explicit ``fields`` override.
        retained_by_episode = {
            row["episode_id"]: [tuple(x) for x in row["retained_intervals"]]
            for row in manifest["episodes"]
            if row["store"] == store_name
        }
        for channel in V2_SIDECARS:
            for row in read_jsonl(store / channel):
                episode_id = str(row["episode_id"])
                targets = remap_v2_row(channel, row, retained_by_episode[episode_id], all_target_atoms[episode_id])
                if channel == "quality_spans.jsonl":
                    source_semantic = (int(row["raw_from_index"]), int(row["raw_to_index"]))
                elif channel == "precision_windows.jsonl":
                    source_semantic = (int(row["raw_from_index"]), int(row["to_index"]))
                else:
                    source_semantic = (int(row["from_index"]), int(row["to_index"]))
                target_ranges = []
                for target in targets:
                    fields = target["fields"]
                    if channel == "quality_spans.jsonl":
                        target_ranges.append((fields["raw_from_index"], fields["raw_to_index"]))
                    elif channel == "precision_windows.jsonl":
                        target_ranges.append((fields["raw_from_index"], fields["to_index"]))
                    else:
                        target_ranges.append((fields["from_index"], fields["to_index"]))
                unchanged = len(target_ranges) == 1 and target_ranges[0] == source_semantic
                source_owner = _key(row)
                rekeyed = bool(targets) and tuple(targets[0]["target_atom_key"]) != source_owner
                action = operation_name(source_semantic, target_ranges, rekeyed=rekeyed)
                if unchanged and not rekeyed:
                    # Even a semantically unchanged row may need headroom clipping.
                    explicit = targets[0]["fields"]
                    if int(row["from_index"]) != int(explicit["from_index"]) or int(row["to_index"]) != int(explicit["to_index"]):
                        action = "clip"
                operation = {
                    "store": store_name,
                    "source_line": int(row["_line"]),
                    "episode_id": episode_id,
                    "source_atom_key": list(source_owner),
                    "source_uid": str(row["uid"]),
                    "source_semantic_range": list(source_semantic),
                    "action": action,
                    "targets": targets,
                }
                manifest["v2_operations"][channel].append(operation)
                counts["v2"][channel][action] += 1
                counts["v2"][channel]["source"] += 1
                counts["v2"][channel]["target"] += len(targets)

        # Drop only currently retained rows whose loaded observation or future action
        # crosses source authority. Existing non-retained rows stay non-retained.
        timestamp_cache: dict[str, np.ndarray] = {}
        retained_for_store = {
            row["episode_id"]: [tuple(x) for x in row["retained_intervals"]]
            for row in manifest["episodes"]
            if row["store"] == store_name
        }
        for row in read_jsonl(store / "actor_anchors_5hz.jsonl"):
            counts["actor_anchors"]["source"] += 1
            if not bool(row.get("retained", True)):
                counts["actor_anchors"]["already_not_retained"] += 1
                continue
            counts["actor_anchors"]["source_retained"] += 1
            episode_id = str(row["episode_id"])
            if episode_id not in timestamp_cache:
                summary = summaries[episode_id]
                episode_dir = store / str(summary.get("directory", f"episodes/{episode_id}"))
                timestamp_cache[episode_id] = np.load(episode_dir / "timestamp_s.npy", mmap_mode="r")
            keep, reasons, future_frame = actor_window_decision(
                store_name, row, retained_for_store[episode_id], timestamp_cache[episode_id]
            )
            if keep:
                counts["actor_anchors"]["target_retained"] += 1
                continue
            anchor = int(row.get("anchor_frame", row.get("anchor_timestep")))
            manifest["actor_anchor_drops"].append(
                {
                    "store": store_name,
                    "source_line": int(row["_line"]),
                    "episode_id": episode_id,
                    "anchor_frame": anchor,
                    "anchor_index": int(row.get("anchor_index", -1)),
                    "split": str(summaries[episode_id]["split"]),
                    "native_rate_hz": _episode_rate(summaries[episode_id]),
                    "history_frames": [int(x) for x in row.get("history_frames", [])],
                    "future_endpoint_frame": future_frame,
                    "reasons": reasons,
                }
            )
            counts["actor_anchors"]["drop"] += 1
            for reason in reasons:
                counts["actor_anchors"][reason] += 1

    # Generated-target validation: containment, unique target keys and q1/q2 ownership.
    target_keys = [tuple(row["target_atom_key"]) for rows in all_target_atoms.values() for row in rows]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("duplicate target atom key")
    target_v2: dict[str, dict[str, list[dict[str, Any]]]] = {
        channel: defaultdict(list) for channel in V2_SIDECARS
    }
    for channel, operations in manifest["v2_operations"].items():
        for operation in operations:
            for target in operation["targets"]:
                target_v2[channel][operation["episode_id"]].append(target["fields"])
    retained_all = {row["episode_id"]: [tuple(x) for x in row["retained_intervals"]] for row in manifest["episodes"]}
    for channel, episodes in target_v2.items():
        for episode_id, rows in episodes.items():
            for row in rows:
                bounds = (int(row["from_index"]), int(row["to_index"]))
                owner = interval_at(retained_all[episode_id], bounds[0])
                if owner is None or bounds[1] > owner[1]:
                    raise ValueError(f"generated {channel} row crosses retention: {episode_id} {bounds}")
    for episode_id, rows in target_v2["mistakes_v2.jsonl"].items():
        for mistake in rows:
            if not any(
                int(span["quality"]) <= 2
                and int(span["raw_from_index"]) <= int(mistake["from_index"])
                and int(mistake["to_index"]) <= int(span["raw_to_index"])
                for span in target_v2["quality_spans.jsonl"][episode_id]
            ):
                raise ValueError(f"generated mistake lacks raw q1/q2 containment: {episode_id} {mistake}")

    manifest["counts"] = {
        "episodes": dict(counts["episodes"]),
        "parents": dict(counts["parents"]),
        "atoms": dict(counts["atoms"]),
        "actor_anchors": dict(counts["actor_anchors"]),
        "v2": {name: dict(counter) for name, counter in counts["v2"].items()},
    }
    manifest["validation"] = {
        "open_candidates": 0,
        "target_atom_keys_unique": True,
        "target_parents_cover_source_authority": True,
        "target_atoms_tile_target_parents": True,
        "target_v2_inside_one_retained_interval": True,
        "target_mistakes_inside_raw_q1_q2": True,
        "actor_keep_windows_inside_one_retained_interval": True,
        "split_and_native_rate_preserved": True,
    }
    return manifest


def summary_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Round 1 mechanical containment sweep",
        "",
        f"- Source root: `{manifest['source_root']}` (read-only).",
        "- Authority: source-declared retained intervals in each episode record.",
        "- Output: correction manifest only; no store was built or activated.",
        f"- Open candidates: **{manifest['validation']['open_candidates']}**.",
        "",
        "## Counts",
        "",
        f"- Episodes: {sum(counts['episodes'].values())} ({counts['episodes']}).",
        f"- Parents: {counts['parents'].get('source', 0)} source -> {counts['parents'].get('target', 0)} target.",
        f"- Atoms: {counts['atoms'].get('source', 0)} source -> {counts['atoms'].get('target', 0)} target; "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts["atoms"].items()) if key not in {"source", "target"})
        + ".",
        f"- Actor anchors dropped: {counts['actor_anchors'].get('drop', 0)} of "
        f"{counts['actor_anchors'].get('source_retained', 0)} currently retained "
        f"(history={counts['actor_anchors'].get('history_crosses_source_retention', 0)}, "
        f"future={counts['actor_anchors'].get('future_crosses_source_retention', 0)}).",
        "",
        "| v2 channel | source | target | keep | rekey | clip | split | drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in V2_SIDECARS:
        row = counts["v2"][channel]
        lines.append(
            f"| {channel} | {row.get('source', 0)} | {row.get('target', 0)} | {row.get('keep', 0)} | "
            f"{row.get('rekey', 0)} | {row.get('clip', 0)} | {row.get('split', 0)} | {row.get('drop', 0)} |"
        )
    lines += [
        "",
        "## Builder contract",
        "",
        "- `target_parents` and each atom operation's `targets` are the complete corrected parent/atom layout.",
        "- Contact, speed and legacy precision inherit from `inherit_per_atom_sidecars_from`.",
        "- Each v2 operation copies unchanged source fields and applies its target `fields` override; `drop` has no targets.",
        "- `actor_anchor_drops` are removed from the currently retained actor view; all other retained anchors remain.",
        "- All target ranges are native half-open frames and preserve episode split and native rate.",
        "",
        "## Validation",
        "",
    ]
    lines += [f"- {name}: `{value}`" for name, value in manifest["validation"].items()]
    return "\n".join(lines) + "\n"


def write_outputs(manifest: dict[str, Any], output: str | Path, summary: str | Path) -> None:
    output, summary = Path(output), Path(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.write_text(summary_markdown(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for target in (Path(args.output).resolve(), Path(args.summary).resolve()):
        if target == root or root in target.parents:
            raise ValueError(f"refusing to write audit output inside active root: {target}")
    manifest = build_manifest(root)
    write_outputs(manifest, args.output, args.summary)
    print(json.dumps({"status": "PASS", "open_candidates": 0, "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
