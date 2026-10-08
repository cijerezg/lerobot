"""Current-contract structural validator for the diverse sampled annotation audit.

Run from the workspace root::

    uv run python -m lerobot.annotation.validate_diverse_audit \
        --root outputs/diverse_robot_dataset_v3 \
        --output migration/diverse_annotation_audit_2026-10-07/validation/structural.json

The validator is deliberately read-only with respect to the corpus.  It validates the
two federated stores, writes one JSON report, and writes a JSONL repair-candidate file
for every v2 row that is not contained in one continuous retained interval.  It does
not apply repairs or adopt a replacement root.

Unlike the legacy atom validator, this module does not enforce parent-grade
inheritance or single-atom mistake ownership.  Mistakes and v2 windows may cross atom
boundaries, but no label, observation-history window, or future-action window may
cross an excluded gap or a retained outer edge.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from lerobot.annotation.vocab import CONTACT_SLUGS, MISTAKE_TYPES

STORES = ("corpus", "fmb")
EXPECTED_FAMILIES = {"droid", "droid_success", "molmoact", "robochallenge", "ur7e", "yam", "fmb"}
ALLOWED_SPLITS = {"train", "validation", "test"}
ATOM_KEY = ("episode_id", "parent_interval_index", "atom_index")
PER_ATOM_SIDECARS = {
    "contact_atoms.jsonl": ("contact", set(range(15))),
    "speed_atoms_hybrid_v1.jsonl": ("speed", set(range(1, 6))),
    "precision_atoms.jsonl": ("precision", set(range(1, 6))),
}
V2_SIDECARS = {
    "quality_spans.jsonl": (
        "raw_from_index", "raw_to_index", "from_index", "to_index", "quality", "cause", "confidence", "uid"
    ),
    "mistakes_v2.jsonl": (
        "from_index", "to_index", "mistake", "mistake_type", "confidence", "uid"
    ),
    "precision_windows.jsonl": (
        "commit_index", "raw_from_index", "from_index", "to_index", "precision", "confidence", "uid"
    ),
}
LOADER_ATOM_SIDECARS = ("subtask_atoms.jsonl", *PER_ATOM_SIDECARS)

Interval = tuple[int, int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects and retain one-based source lines for diagnostics."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append({**row, "_line": line_number})
    return rows


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge only overlapping or exactly adjacent half-open intervals."""
    merged: list[list[int]] = []
    for start, stop in sorted((int(a), int(b)) for a, b in intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def containing_interval(intervals: Sequence[Interval], start: int, stop: int) -> Interval | None:
    """The single continuous retained interval containing ``[start, stop)``."""
    hits = [(lo, hi) for lo, hi in intervals if lo <= start and stop <= hi]
    return hits[0] if len(hits) == 1 else None


def interval_at(intervals: Sequence[Interval], frame: int) -> Interval | None:
    return next(((lo, hi) for lo, hi in intervals if lo <= frame < hi), None)


def classify_containment(
    intervals: Sequence[Interval], effective: Interval, semantic: Interval
) -> dict[str, str] | None:
    """Classify a row outside retention for a deterministic repair queue.

    ``padding_only`` means the semantic annotation is valid and only compiler-added
    headroom must be clipped.  Otherwise the semantic annotation itself requires
    review/remapping.  ``internal_gap`` is kept separate because clipping one side is
    not enough: a semantic range crossing a gap must be split or rejected.
    """
    if containing_interval(intervals, *effective) is not None:
        return None
    semantic_inside = containing_interval(intervals, *semantic) is not None
    if intervals:
        within_outer_envelope = intervals[0][0] <= effective[0] and effective[1] <= intervals[-1][1]
    else:
        within_outer_envelope = False
    boundary = "internal_gap" if within_outer_envelope else "outer_edge"
    if semantic_inside:
        kind = "padding_only"
    elif boundary == "internal_gap":
        kind = "semantic_gap"
    else:
        kind = "semantic_outside_retention"
    return {"containment_class": kind, "boundary_class": boundary}


def half_second_frames(rate_hz: float) -> int:
    """Nearest integer frame count with .5 rounded away from zero."""
    return int(math.floor(0.5 * float(rate_hz) + 0.5))


def expected_quality_bounds(row: dict[str, Any], retained: Sequence[Interval]) -> Interval | None:
    raw = (int(row["raw_from_index"]), int(row["raw_to_index"]))
    owner = containing_interval(retained, *raw)
    if owner is None:
        return None
    if int(row["quality"]) == 5:
        return raw
    fps = float(row["native_rate_hz"])
    return max(owner[0], raw[0] - round(fps)), min(owner[1], raw[1] + half_second_frames(fps))


def expected_precision_bounds(row: dict[str, Any], retained: Sequence[Interval]) -> Interval | None:
    semantic = (int(row["raw_from_index"]), int(row["to_index"]))
    owner = containing_interval(retained, *semantic)
    if owner is None:
        return None
    fps = float(row["native_rate_hz"])
    return max(owner[0], semantic[0] - round(fps)), semantic[1]


def effective_frame_labels(
    frame: int,
    quality_rows: Sequence[dict[str, Any]],
    mistake_rows: Sequence[dict[str, Any]],
    precision_rows: Sequence[dict[str, Any]],
) -> tuple[int, bool, int]:
    """The exact v2 labels consumed by ``diverse_actor_selection`` at one frame."""
    quality = min(
        (
            int(row["quality"])
            for row in quality_rows
            if int(row["from_index"]) <= frame < int(row["to_index"])
        ),
        default=4,
    )
    mistake = any(
        int(row["from_index"]) <= frame < int(row["to_index"])
        for row in mistake_rows
    )
    precision = max(
        (
            int(row["precision"])
            for row in precision_rows
            if int(row["from_index"]) <= frame < int(row["to_index"])
        ),
        default=1,
    )
    return quality, mistake, precision


@dataclass
class Findings:
    rows: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        code: str,
        message: str,
        *,
        store: str,
        episode_id: str | None = None,
        source_file: str | None = None,
        row: dict[str, Any] | None = None,
        severity: str = "error",
        details: dict[str, Any] | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "severity": severity,
            "code": code,
            "store": store,
            "message": message,
        }
        if episode_id is not None:
            item["episode_id"] = episode_id
        if source_file is not None:
            item["source_file"] = source_file
        if row is not None and "_line" in row:
            item["line"] = int(row["_line"])
        if details:
            item["details"] = details
        self.rows.append(item)
        self.counts[code] += 1


def atom_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["episode_id"]), int(row["parent_interval_index"]), int(row["atom_index"])


def expected_uid(episode_id: str, parent: int, atom: int) -> str:
    return f"{episode_id.split('__', 1)[-1]}_p{parent}a{atom}"


def _array_frame_count(summary: dict[str, Any]) -> int:
    return int(summary.get("frames", summary.get("frame_count")))


def _episode_rate(summary: dict[str, Any]) -> float:
    return float(summary.get("native_rate_hz", summary.get("nominal_fps")))


def _source(store_name: str, summary: dict[str, Any]) -> str:
    return "fmb" if store_name == "fmb" else str(summary["source"])


def episode_authority_intervals(
    store_name: str,
    record: dict[str, Any],
    timestamps: np.ndarray,
) -> list[Interval]:
    """Retained frame intervals declared by the source episode record.

    This is intentionally independent of the atom and critic tables.  Otherwise a
    malformed corrected table could define its own allowed footage and hide a stale
    ``episode.json`` or a bridged source gap.
    """
    if store_name == "fmb":
        spans = [
            (
                int(row.get("reviewed_start_timestep", row["start_timestep"])),
                int(row.get("reviewed_end_timestep_exclusive", row["end_timestep_exclusive"])),
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


def _future_endpoint_frame(
    store_name: str,
    row: dict[str, Any],
    timestamps: np.ndarray,
) -> tuple[int, float]:
    if store_name == "fmb":
        # FMB copy_state starts one 10 Hz source tick after the anchor.
        from lerobot.datasets.diverse_pilot import COPY_STATE_LEAD_S

        end_s = float(row["future_end_s_nominal"]) + float(COPY_STATE_LEAD_S)
    else:
        end_s = float(row["future_end_s"])
    index = int(np.searchsorted(timestamps, end_s, side="left"))
    return index, end_s


def _check_metadata(
    store_name: str,
    store: Path,
    episodes: dict[str, dict[str, Any]],
    findings: Findings,
) -> dict[str, Any]:
    arrays_checked = 0
    videos_declared = 0
    for episode_id, summary in episodes.items():
        episode_dir = store / str(summary.get("directory", f"episodes/{episode_id}"))
        metadata_path = episode_dir / "episode.json"
        if not metadata_path.is_file():
            findings.add("missing_episode_metadata", str(metadata_path), store=store_name, episode_id=episode_id)
            continue
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames, fps = _array_frame_count(summary), _episode_rate(summary)
        record_frames = int(
            record.get("frame_count", record.get("arrays", {}).get("timestamp_s", {}).get("shape", [frames])[0])
        )
        record_fps = float(record.get("native_rate_hz", record.get("nominal_fps", fps)))
        if str(record.get("episode_id", episode_id)) != episode_id or record_frames != frames or record_fps != fps:
            findings.add(
                "episode_metadata_mismatch",
                "episode id/frame count/native rate disagrees with episodes.jsonl",
                store=store_name,
                episode_id=episode_id,
                details={"index_frames": frames, "metadata_frames": record_frames, "index_rate": fps, "metadata_rate": record_fps},
            )
        arrays = record.get("arrays", {})
        for name, spec in arrays.items():
            path, shape = spec.get("path"), spec.get("shape")
            if not path or not isinstance(shape, list) or not shape:
                findings.add("array_schema", f"array {name!r} lacks path/shape", store=store_name, episode_id=episode_id)
                continue
            if int(shape[0]) != frames:
                findings.add(
                    "array_declared_frames",
                    f"{path} declares {shape[0]} rows; episode has {frames}",
                    store=store_name,
                    episode_id=episode_id,
                )
            array_path = episode_dir / str(path)
            if not array_path.is_file():
                findings.add("missing_array", str(array_path), store=store_name, episode_id=episode_id)
                continue
            try:
                array = np.load(array_path, mmap_mode="r")
                arrays_checked += 1
                if tuple(array.shape) != tuple(shape):
                    findings.add(
                        "array_shape_mismatch",
                        f"{path}: actual {list(array.shape)} != declared {shape}",
                        store=store_name,
                        episode_id=episode_id,
                    )
            except Exception as exc:  # pragma: no cover - corrupt npy details vary
                findings.add("array_unreadable", f"{path}: {exc}", store=store_name, episode_id=episode_id)
        if store_name == "corpus":
            for camera in record.get("cameras", []):
                videos_declared += 1
                if int(camera.get("frames", -1)) != frames or float(camera.get("fps", -1)) != fps:
                    findings.add(
                        "camera_declared_contract",
                        f"camera {camera.get('name')!r} frame count/rate disagrees with episode",
                        store=store_name,
                        episode_id=episode_id,
                    )
                if not (episode_dir / str(camera.get("path"))).is_file():
                    findings.add(
                        "missing_camera_video",
                        f"camera {camera.get('name')!r} asset is missing",
                        store=store_name,
                        episode_id=episode_id,
                    )
    return {"array_headers_checked": arrays_checked, "videos_declared": videos_declared}


def _validate_store(
    store_name: str,
    store: Path,
    findings: Findings,
    containment_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    required = ["episodes.jsonl", "critic_intervals.jsonl", "subtask_atoms.jsonl", "actor_anchors_5hz.jsonl"]
    required += list(PER_ATOM_SIDECARS) + list(V2_SIDECARS)
    missing = [name for name in required if not (store / name).is_file()]
    for name in missing:
        findings.add("missing_store_file", name, store=store_name, source_file=name)
    if missing:
        return {"missing_files": missing}

    episode_rows = read_jsonl(store / "episodes.jsonl")
    episodes: dict[str, dict[str, Any]] = {}
    for row in episode_rows:
        episode_id = str(row.get("episode_id"))
        if episode_id in episodes:
            findings.add("duplicate_episode", episode_id, store=store_name, episode_id=episode_id, row=row)
            continue
        episodes[episode_id] = row
        split = str(row.get("split"))
        if split not in ALLOWED_SPLITS:
            findings.add("invalid_split", split, store=store_name, episode_id=episode_id, row=row)

    atoms = read_jsonl(store / "subtask_atoms.jsonl")
    critics = read_jsonl(store / "critic_intervals.jsonl")
    actor_rows = read_jsonl(store / "actor_anchors_5hz.jsonl")
    per_atom = {name: read_jsonl(store / name) for name in PER_ATOM_SIDECARS}
    v2 = {name: read_jsonl(store / name) for name in V2_SIDECARS}

    atoms_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    atoms_by_parent: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    atoms_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in atoms:
        episode_id = str(row.get("episode_id"))
        if episode_id not in episodes:
            findings.add("orphan_atom_episode", "atom references an unknown episode", store=store_name, episode_id=episode_id, row=row)
            continue
        key = atom_key(row)
        if key in atoms_by_key:
            findings.add("duplicate_atom_key", repr(key), store=store_name, episode_id=episode_id, row=row)
        atoms_by_key[key] = row
        atoms_by_parent[(episode_id, key[1])].append(row)
        atoms_by_episode[episode_id].append(row)
        frames, fps = _array_frame_count(episodes[episode_id]), _episode_rate(episodes[episode_id])
        start, stop = int(row["start_timestep"]), int(row["end_timestep_exclusive"])
        if not 0 <= start < stop <= frames:
            findings.add(
                "atom_frame_bounds", f"[{start},{stop}) outside [0,{frames})", store=store_name,
                episode_id=episode_id, row=row,
            )
        if str(row.get("source")) != _source(store_name, episodes[episode_id]) or str(row.get("split")) != str(episodes[episode_id].get("split")):
            findings.add("atom_source_split_mismatch", "atom source/split disagrees with episode", store=store_name, episode_id=episode_id, row=row)
        if float(row.get("native_rate_hz", -1)) != fps:
            findings.add("atom_rate_mismatch", "atom rate disagrees with episode", store=store_name, episode_id=episode_id, row=row)

    parents: dict[tuple[str, int], dict[str, Any]] = {}
    parents_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in critics:
        episode_id = str(row.get("episode_id"))
        key = (episode_id, int(row["interval_index"]))
        if key in parents:
            findings.add("duplicate_parent", repr(key), store=store_name, episode_id=episode_id, row=row)
        parents[key] = row
        parents_by_episode[episode_id].append(row)
        if episode_id not in episodes:
            findings.add("orphan_parent_episode", "parent references an unknown episode", store=store_name, episode_id=episode_id, row=row)
            continue
        frames = _array_frame_count(episodes[episode_id])
        start, stop = int(row["start_timestep"]), int(row["end_timestep_exclusive"])
        if not 0 <= start < stop <= frames:
            findings.add("parent_frame_bounds", f"[{start},{stop}) outside [0,{frames})", store=store_name, episode_id=episode_id, row=row)
        source, split = _source(store_name, episodes[episode_id]), str(episodes[episode_id]["split"])
        if str(row.get("source", source)) != source or str(row.get("split")) != split:
            findings.add("parent_source_split_mismatch", "parent source/split disagrees with episode", store=store_name, episode_id=episode_id, row=row)
        # Interruption seconds are source metadata.  They must start at or after the
        # retained parent's end; exact frame gaps come from adjacent parent bounds.
        parent_end_s = float(row.get("end_s_exclusive", stop / _episode_rate(episodes[episode_id])))
        tolerance = 1.01 / _episode_rate(episodes[episode_id])
        for event in row.get("interruption_events") or []:
            if float(event["start_s"]) < parent_end_s - tolerance:
                findings.add(
                    "interruption_overlaps_parent",
                    f"interruption starts at {event['start_s']} s before parent end {parent_end_s} s",
                    store=store_name, episode_id=episode_id, row=row,
                )

    retained = {
        episode_id: merge_intervals(
            (int(row["start_timestep"]), int(row["end_timestep_exclusive"])) for row in rows
        )
        for episode_id, rows in parents_by_episode.items()
    }
    gaps = {
        episode_id: [(left[1], right[0]) for left, right in zip(intervals, intervals[1:]) if left[1] < right[0]]
        for episode_id, intervals in retained.items()
    }

    # Episode metadata is the non-circular retention authority.  Require its exact
    # union to agree independently with both parents and atoms.  In particular, this
    # catches a corrected atom/critic extension whose episode.json still marks those
    # frames rejected.  FMB authority is the union of its reviewed primitive bounds;
    # critic eligibility never removes actor-retained footage.
    for episode_id, summary in episodes.items():
        episode_dir = store / str(summary.get("directory", f"episodes/{episode_id}"))
        try:
            record = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
            timestamps = np.load(episode_dir / "timestamp_s.npy", mmap_mode="r")
            authority = episode_authority_intervals(store_name, record, timestamps)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            findings.add(
                "episode_authority_schema",
                f"cannot derive retained intervals from episode metadata: {exc}",
                store=store_name,
                episode_id=episode_id,
            )
            continue
        parent_union = retained.get(episode_id, [])
        atom_union = merge_intervals(
            (int(row["start_timestep"]), int(row["end_timestep_exclusive"]))
            for row in atoms_by_episode.get(episode_id, [])
        )
        if not authority:
            findings.add(
                "episode_authority_empty",
                "episode metadata declares no retained footage",
                store=store_name,
                episode_id=episode_id,
            )
        if parent_union != authority:
            findings.add(
                "episode_authority_parent_mismatch",
                "critic-parent union disagrees with episode retention authority",
                store=store_name,
                episode_id=episode_id,
                details={"authority": [list(x) for x in authority], "parents": [list(x) for x in parent_union]},
            )
        if atom_union != authority:
            findings.add(
                "episode_authority_atom_mismatch",
                "subtask-atom union disagrees with episode retention authority",
                store=store_name,
                episode_id=episode_id,
                details={"authority": [list(x) for x in authority], "atoms": [list(x) for x in atom_union]},
            )

    for key, parent in parents.items():
        episode_id = key[0]
        children = sorted(
            atoms_by_parent.get(key, []),
            key=lambda row: (int(row["start_timestep"]), int(row["atom_index"])),
        )
        if not children:
            findings.add("parent_without_atoms", repr(key), store=store_name, episode_id=episode_id, row=parent)
            continue
        parent_range = (int(parent["start_timestep"]), int(parent["end_timestep_exclusive"]))
        if (int(children[0]["start_timestep"]), int(children[-1]["end_timestep_exclusive"])) != parent_range:
            findings.add("atom_parent_edge_mismatch", f"children do not cover parent {parent_range}", store=store_name, episode_id=episode_id, row=parent)
        for left, right in zip(children, children[1:]):
            if int(left["end_timestep_exclusive"]) != int(right["start_timestep"]):
                findings.add(
                    "atom_tiling_gap_or_overlap",
                    f"P{key[1]} boundary {left['end_timestep_exclusive']} -> {right['start_timestep']}",
                    store=store_name, episode_id=episode_id, row=right,
                )
    for key, rows in atoms_by_parent.items():
        if key not in parents:
            findings.add("atoms_without_parent", repr(key), store=store_name, episode_id=key[0], row=rows[0])
    for episode_id, rows in atoms_by_episode.items():
        for row in rows:
            bounds = (int(row["start_timestep"]), int(row["end_timestep_exclusive"]))
            if containing_interval(retained.get(episode_id, []), *bounds) is None:
                findings.add("atom_crosses_retention", str(bounds), store=store_name, episode_id=episode_id, row=row)

    for file_name, (value_field, allowed) in PER_ATOM_SIDECARS.items():
        rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in per_atom[file_name]:
            episode_id = str(row.get("episode_id"))
            try:
                key = atom_key(row)
            except (KeyError, TypeError, ValueError):
                findings.add("sidecar_schema", "missing atom key", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                continue
            if key in rows_by_key:
                findings.add("duplicate_sidecar_key", repr(key), store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            rows_by_key[key] = row
            atom = atoms_by_key.get(key)
            if atom is None:
                findings.add("orphan_sidecar_atom", repr(key), store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                continue
            for name in ("start_timestep", "end_timestep_exclusive", "subtask"):
                if row.get(name) != atom.get(name):
                    findings.add("sidecar_atom_mismatch", f"{name} disagrees with atom", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            if row.get(value_field) not in allowed:
                findings.add("sidecar_value_range", f"{value_field}={row.get(value_field)!r}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            if file_name == "contact_atoms.jsonl":
                contact = int(row["contact"])
                expected_slug = CONTACT_SLUGS[contact]
                if str(row.get("contact_slug")) != expected_slug:
                    findings.add("contact_slug_mismatch", f"expected {expected_slug!r}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
        for key in sorted(set(atoms_by_key) - set(rows_by_key)):
            findings.add("missing_sidecar_atom", repr(key), store=store_name, episode_id=key[0], source_file=file_name)
        for key in sorted(set(rows_by_key) - set(atoms_by_key)):
            findings.add("extra_sidecar_atom", repr(key), store=store_name, episode_id=key[0], source_file=file_name)

    for file_name, required_fields in V2_SIDECARS.items():
        for row in v2[file_name]:
            episode_id = str(row.get("episode_id"))
            absent = [name for name in required_fields if name not in row]
            if absent:
                findings.add("v2_schema", f"missing {absent}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                continue
            if episode_id not in episodes:
                findings.add("orphan_v2_episode", "unknown episode", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                continue
            source, fps = _source(store_name, episodes[episode_id]), _episode_rate(episodes[episode_id])
            frames = _array_frame_count(episodes[episode_id])
            if str(row.get("source")) != source or float(row.get("native_rate_hz", -1)) != fps:
                findings.add("v2_source_rate_mismatch", "source/rate disagrees with episode", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            key = atom_key(row)
            if key not in atoms_by_key:
                findings.add("v2_owner_missing", repr(key), store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            elif str(row["uid"]) != expected_uid(*key):
                findings.add("v2_uid_mismatch", f"expected {expected_uid(*key)!r}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            effective = (int(row["from_index"]), int(row["to_index"]))
            if not 0 <= effective[0] < effective[1] <= frames:
                findings.add("v2_frame_bounds", f"{effective} outside [0,{frames})", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
            if file_name == "quality_spans.jsonl":
                semantic = (int(row["raw_from_index"]), int(row["raw_to_index"]))
                if int(row["quality"]) not in {1, 2, 3, 5}:
                    findings.add("quality_value", str(row["quality"]), store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                if not effective[0] <= semantic[0] < semantic[1] <= effective[1]:
                    findings.add("quality_range_order", f"effective {effective}, raw {semantic}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                expected_bounds = expected_quality_bounds(row, retained.get(episode_id, []))
            elif file_name == "precision_windows.jsonl":
                semantic = (int(row["raw_from_index"]), int(row["to_index"]))
                if int(row["precision"]) not in range(2, 6):
                    findings.add("precision_value", str(row["precision"]), store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                if not effective[0] <= semantic[0] < semantic[1]:
                    findings.add("precision_range_order", f"effective {effective}, semantic {semantic}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)
                expected_bounds = expected_precision_bounds(row, retained.get(episode_id, []))
            else:
                semantic = effective
                expected_bounds = effective
                if row.get("mistake") is not True or str(row.get("mistake_type")) not in MISTAKE_TYPES:
                    findings.add("mistake_schema", f"mistake={row.get('mistake')!r}, type={row.get('mistake_type')!r}", store=store_name, episode_id=episode_id, source_file=file_name, row=row)

            classification = classify_containment(retained.get(episode_id, []), effective, semantic)
            if classification is not None:
                candidate = {
                    "store": store_name,
                    "channel": file_name,
                    "line": int(row["_line"]),
                    "episode_id": episode_id,
                    "source": source,
                    "split": str(episodes[episode_id]["split"]),
                    "uid": str(row.get("uid")),
                    "parent_interval_index": int(row["parent_interval_index"]),
                    "atom_index": int(row["atom_index"]),
                    "effective_range": list(effective),
                    "semantic_range": list(semantic),
                    "retained_intervals": [list(x) for x in retained.get(episode_id, [])],
                    **classification,
                    "repair": (
                        "clip_compiler_headroom_to_continuous_retained_interval"
                        if classification["containment_class"] == "padding_only"
                        else "visually_review_then_remap_split_or_remove_semantic_range"
                    ),
                }
                containment_candidates.append(candidate)
                findings.add(
                    "v2_outside_retention",
                    f"effective {effective}, semantic {semantic}: {classification['containment_class']}",
                    store=store_name, episode_id=episode_id, source_file=file_name, row=row,
                    details=classification,
                )
            if expected_bounds is not None and effective != expected_bounds:
                findings.add(
                    "headroom_not_clipped_once",
                    f"effective {effective} != expected {expected_bounds}",
                    store=store_name, episode_id=episode_id, source_file=file_name, row=row,
                )

    # Every discrete event is unique and fully contained by a raw q1/q2 attempt.
    exact_events: set[tuple[str, int, int, str]] = set()
    quality_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in v2["quality_spans.jsonl"]:
        quality_by_episode[str(row.get("episode_id"))].append(row)
    for row in v2["mistakes_v2.jsonl"]:
        episode_id = str(row["episode_id"])
        event = (episode_id, int(row["from_index"]), int(row["to_index"]), str(row["mistake_type"]))
        if event in exact_events:
            findings.add("duplicate_mistake_event", repr(event), store=store_name, episode_id=episode_id, source_file="mistakes_v2.jsonl", row=row)
        exact_events.add(event)
        start, stop = event[1], event[2]
        if not any(
            int(span["quality"]) <= 2
            and int(span["raw_from_index"]) <= start
            and stop <= int(span["raw_to_index"])
            for span in quality_by_episode[episode_id]
        ):
            findings.add(
                "mistake_not_in_raw_q1_q2",
                f"mistake [{start},{stop}) is not inside one raw q1/q2 span",
                store=store_name, episode_id=episode_id, source_file="mistakes_v2.jsonl", row=row,
            )

    # Actor observations and future action samples must remain inside the same exact
    # continuous retained interval as the anchor.  This intentionally detects both
    # internal excluded gaps and excluded prefix/suffix footage.
    timestamp_cache: dict[str, np.ndarray] = {}
    retained_actor_rows = 0
    for row in actor_rows:
        if not bool(row.get("retained", True)):
            continue
        retained_actor_rows += 1
        episode_id = str(row["episode_id"])
        if episode_id not in episodes:
            findings.add("orphan_actor_episode", "unknown episode", store=store_name, episode_id=episode_id, source_file="actor_anchors_5hz.jsonl", row=row)
            continue
        anchor = int(row.get("anchor_frame", row.get("anchor_timestep")))
        owner = interval_at(retained.get(episode_id, []), anchor)
        if owner is None:
            findings.add("retained_anchor_outside_retention", str(anchor), store=store_name, episode_id=episode_id, source_file="actor_anchors_5hz.jsonl", row=row)
            continue
        history = [int(frame) for frame in row.get("history_frames", [])]
        if not history or any(not owner[0] <= frame < owner[1] for frame in history):
            findings.add(
                "history_window_crosses_retention",
                f"anchor {anchor}, history {history}, retained {owner}",
                store=store_name, episode_id=episode_id, source_file="actor_anchors_5hz.jsonl", row=row,
            )
        episode_dir = store / str(episodes[episode_id].get("directory", f"episodes/{episode_id}"))
        if episode_id not in timestamp_cache:
            timestamp_cache[episode_id] = np.load(episode_dir / "timestamp_s.npy", mmap_mode="r")
        endpoint, endpoint_s = _future_endpoint_frame(store_name, row, timestamp_cache[episode_id])
        if endpoint >= len(timestamp_cache[episode_id]) or not owner[0] <= endpoint < owner[1]:
            findings.add(
                "action_window_crosses_retention",
                f"anchor {anchor}, future endpoint {endpoint} ({endpoint_s:.6f}s), retained {owner}",
                store=store_name, episode_id=episode_id, source_file="actor_anchors_5hz.jsonl", row=row,
            )

    metadata_metrics = _check_metadata(store_name, store, episodes, findings)
    return {
        "episodes": len(episodes),
        "parents": len(critics),
        "atoms": len(atoms),
        "retained_intervals": sum(len(rows) for rows in retained.values()),
        "internal_gaps": sum(len(rows) for rows in gaps.values()),
        "actor_rows": len(actor_rows),
        "retained_actor_rows": retained_actor_rows,
        "contact_rows": len(per_atom["contact_atoms.jsonl"]),
        "speed_rows": len(per_atom["speed_atoms_hybrid_v1.jsonl"]),
        "precision_atom_rows": len(per_atom["precision_atoms.jsonl"]),
        "quality_spans": len(v2["quality_spans.jsonl"]),
        "mistakes": len(v2["mistakes_v2.jsonl"]),
        "precision_windows": len(v2["precision_windows.jsonl"]),
        **metadata_metrics,
        "episode_ids": sorted(episodes),
        "source_splits": {
            episode_id: {"source": _source(store_name, row), "split": str(row["split"])}
            for episode_id, row in episodes.items()
        },
        "retained_by_episode": {episode_id: [list(x) for x in rows] for episode_id, rows in retained.items()},
    }


def _validate_loader(
    root: Path,
    store_rows: dict[str, dict[str, list[dict[str, Any]]]],
    findings: Findings,
) -> dict[str, Any]:
    """Open the training readers and independently verify effective anchor labels."""
    try:
        from lerobot.datasets.diverse_actor_selection import holdout_actor_selection, select_actor_anchors
        from lerobot.datasets.fmb_corpus import FederatedDiverseCorpus

        corpus = FederatedDiverseCorpus(root / "corpus", root / "fmb")
        training = select_actor_anchors(corpus, verify_counts=False)
        holdout = holdout_actor_selection(corpus)
        rows = training.rows + holdout.rows
    except Exception as exc:
        findings.add("trainer_loader_error", repr(exc), store="federated")
        return {"status": "FAIL", "error": repr(exc)}

    mismatches = 0
    for row in rows:
        store_name = "fmb" if row["corpus_key"] == "fmb" else "corpus"
        episode_id = str(row["episode_id"])
        frame = int(row.get("anchor_frame", row.get("anchor_timestep")))
        channels = store_rows[store_name]
        quality, mistake, precision = effective_frame_labels(
            frame,
            channels["quality_spans.jsonl"].get(episode_id, []),
            channels["mistakes_v2.jsonl"].get(episode_id, []),
            channels["precision_windows.jsonl"].get(episode_id, []),
        )

        def one_atom(table: str) -> dict[str, Any] | None:
            hits = [
                item for item in channels[table].get(episode_id, [])
                if int(item["start_timestep"]) <= frame < int(item["end_timestep_exclusive"])
            ]
            return hits[0] if len(hits) == 1 else None

        subtask_atom = one_atom("subtask_atoms.jsonl")
        speed_atom = one_atom("speed_atoms_hybrid_v1.jsonl")
        contact_atom = one_atom("contact_atoms.jsonl")
        actual = (
            int(row["quality"]), bool(row["mistake"]), int(row["precision"]),
            str(row["subtask"]), int(row["speed"]), int(row["contact"]),
        )
        expected = (
            quality, mistake, precision,
            None if subtask_atom is None else str(subtask_atom["subtask"]),
            None if speed_atom is None else int(speed_atom["speed"]),
            None if contact_atom is None else int(contact_atom["contact"]),
        )
        if actual != expected:
            mismatches += 1
            findings.add(
                "effective_loader_label_mismatch",
                f"frame {frame}: loader {actual} != sidecars {expected}",
                store=store_name, episode_id=episode_id,
            )
    return {
        "status": "PASS" if mismatches == 0 else "FAIL",
        "training_episodes": len(training.episode_ids),
        "training_anchors": len(training.rows),
        "holdout_episodes": len(holdout.episode_ids),
        "holdout_anchors": len(holdout.rows),
        "effective_rows_checked": len(rows),
        "effective_label_mismatches": mismatches,
    }


def validate(root: str | Path, *, run_loader: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(root)
    findings = Findings()
    candidates: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    episode_ids: set[str] = set()
    families: set[str] = set()
    loader_channels: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}

    for store_name in STORES:
        store = root / store_name
        if not store.is_dir():
            findings.add("missing_store", str(store), store=store_name)
            continue
        metrics[store_name] = _validate_store(store_name, store, findings, candidates)
        ids = set(metrics[store_name].pop("episode_ids", []))
        collisions = episode_ids & ids
        for episode_id in sorted(collisions):
            findings.add("cross_store_episode_collision", episode_id, store=store_name, episode_id=episode_id)
        episode_ids |= ids
        for value in metrics[store_name].pop("source_splits", {}).values():
            families.add(str(value["source"]))
        metrics[store_name].pop("retained_by_episode", None)
        loader_channels[store_name] = {}
        for name in (*LOADER_ATOM_SIDECARS, *V2_SIDECARS):
            by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
            path = store / name
            if path.is_file():
                for row in read_jsonl(path):
                    by_episode[str(row.get("episode_id"))].append(row)
            loader_channels[store_name][name] = dict(by_episode)

    if families != EXPECTED_FAMILIES:
        findings.add(
            "family_inventory",
            f"actual {sorted(families)} != expected {sorted(EXPECTED_FAMILIES)}",
            store="federated",
        )
    loader = _validate_loader(root, loader_channels, findings) if run_loader and all((root / s).is_dir() for s in STORES) else {"status": "SKIP"}

    candidate_counts = Counter(row["containment_class"] for row in candidates)
    channel_counts = Counter(row["channel"] for row in candidates)
    boundary_counts = Counter(row["boundary_class"] for row in candidates)
    source_counts = Counter(row["source"] for row in candidates)
    report = {
        "contract": "Diverse dataset: sampled audit (2026-10-07)",
        "root": str(root),
        "status": "PASS" if not any(row["severity"] == "error" for row in findings.rows) else "FAIL",
        "metrics": metrics,
        "loader": loader,
        "checks": {
            "finding_counts": dict(sorted(findings.counts.items())),
            "errors": sum(row["severity"] == "error" for row in findings.rows),
            "warnings": sum(row["severity"] == "warning" for row in findings.rows),
        },
        "containment_candidates": {
            "rows": len(candidates),
            "by_class": dict(sorted(candidate_counts.items())),
            "by_channel": dict(sorted(channel_counts.items())),
            "by_boundary": dict(sorted(boundary_counts.items())),
            "by_source": dict(sorted(source_counts.items())),
        },
        "findings": findings.rows,
        "notes": [
            "Retained intervals are exact merged critic-parent frame ranges; atoms must tile each parent.",
            "Semantic ranges are raw quality stretches, full precision work windows, and mistake events.",
            "Padding-only candidates can be rematerialized mechanically; semantic candidates require visual review.",
            "No parent-grade inheritance or single-atom mistake ownership is enforced.",
        ],
    }
    return report, candidates


def write_report(
    report: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    output: str | Path,
    candidates_output: str | Path | None = None,
) -> tuple[Path, Path]:
    output = Path(output)
    if candidates_output is None:
        candidates_output = output.with_name(f"{output.stem}_containment_candidates.jsonl")
    candidates_output = Path(candidates_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates_output.parent.mkdir(parents=True, exist_ok=True)
    report = {**report, "containment_candidates": {**report["containment_candidates"], "path": str(candidates_output)}}
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidates_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    return output, candidates_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="federated root containing corpus/ and fmb/")
    parser.add_argument("--output", required=True, help="JSON validation report")
    parser.add_argument("--candidates-output", help="JSONL v2 containment repair queue")
    parser.add_argument("--no-loader", action="store_true", help="skip trainer-facing loader verification")
    args = parser.parse_args()
    report, candidates = validate(args.root, run_loader=not args.no_loader)
    output, candidates_output = write_report(report, candidates, args.output, args.candidates_output)
    print(json.dumps({
        "status": report["status"],
        "errors": report["checks"]["errors"],
        "containment_candidates": len(candidates),
        "report": str(output),
        "candidates": str(candidates_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
