"""Bookkeeping for the diverse-only sampled annotation audit.

This module deliberately does not decide semantic labels. It resolves the active
federated store, inventories unique source episodes, makes reproducible family-balanced
draws without replacement, and writes the audit ledger/check skeletons outside the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FAMILY_ORDER = ("droid", "droid_success", "molmoact", "robochallenge", "ur7e", "yam", "fmb")
ROUND_SIZE = 8
COMPLETION_KEYS = ("fixes_open", "second_reads_open", "sweeps_open")
SIDECARS = (
    "subtask_atoms.jsonl", "contact_atoms.jsonl", "precision_atoms.jsonl",
    "quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl",
    "speed_atoms_hybrid_v1.jsonl",
)
ACTIVE_CONFIG = Path("lerobot/src/lerobot/rl/config_rl.yaml")
PROVENANCE_KEYS = (
    "channel", "grammar_version", "contact_vocab_version", "created", "date", "annotator",
    "rubric", "writer", "method", "work_dir", "files", "labels", "rows", "source_sha256", "problems",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "sha256": sha256(path)}
    if rows is not None:
        record["rows"] = rows
    return record


def parsed_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: payload[key] for key in PROVENANCE_KEYS if key in payload}


def camera_modalities(record: dict[str, Any], part: str) -> dict[str, list[str]]:
    if part == "fmb":
        counts = record.get("production_retained_sensor_counts", {})
        dropped = record.get("dropped_modalities", {})
        return {
            "rgb": sorted(counts.get("rgb_images_by_camera", {})),
            "depth": sorted(counts.get("depth_maps_by_camera", {})),
            "dropped_rgb": sorted(dropped.get("rgb", [])),
            "dropped_depth": sorted(dropped.get("depth", [])),
        }
    cameras = [camera if isinstance(camera, str) else camera["name"] for camera in record.get("cameras", [])]
    return {"rgb": sorted(cameras), "depth": [], "dropped_rgb": [], "dropped_depth": []}


def merged_ranges(atoms: list[dict[str, Any]]) -> list[list[int]]:
    spans = sorted((int(row["start_timestep"]), int(row["end_timestep_exclusive"])) for row in atoms)
    out: list[list[int]] = []
    for start, stop in spans:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], stop)
        else:
            out.append([start, stop])
    return out


def complement_ranges(ranges: list[list[int]], frames: int) -> list[list[int]]:
    cursor, out = 0, []
    for start, stop in ranges:
        if cursor < start:
            out.append([cursor, start])
        cursor = max(cursor, stop)
    if cursor < frames:
        out.append([cursor, frames])
    return out


def seconds_to_frame(seconds: float, rate: float, frames: int) -> int:
    """Return the first nominal native frame at or after ``seconds``."""
    return min(frames, max(0, int(math.ceil(float(seconds) * rate - 1e-9))))


def source_retention(
    record: dict[str, Any], part: str, rate: float, frames: int, timestamps: np.ndarray | None = None
) -> tuple[list[dict[str, Any]], list[list[int]], list[list[int]], str]:
    """Derive continuous retained footage from source metadata, never semantic atoms."""
    source_intervals: list[dict[str, Any]] = []
    if part == "corpus":
        if timestamps is None:
            raise ValueError(f"missing native timestamps for {record.get('episode_id')}")
        if timestamps.shape != (frames,):
            raise ValueError(
                f"timestamp shape {timestamps.shape} does not match {frames} frames for {record.get('episode_id')}"
            )
        segments = record.get("annotations", {}).get("segments", [])
        if not segments:
            raise ValueError(f"missing source retention segments for {record.get('episode_id')}")
        for segment in segments:
            start_s, end_s = float(segment["start_s"]), float(segment["end_s"])
            source_intervals.append({
                "start_timestep": int(np.searchsorted(timestamps, start_s, side="left")),
                "end_timestep_exclusive": int(np.searchsorted(timestamps, end_s, side="left")),
                "start_s": start_s,
                "end_s": end_s,
                "retention": str(segment["retention"]),
                "retention_reason": str(segment["retention_reason"]),
            })
        provenance = "episode.json annotations.segments; boundaries mapped through stored native timestamps"
    else:
        primitives = record.get("primitive_intervals", [])
        if not primitives:
            raise ValueError(f"missing FMB source primitive intervals for {record.get('episode_id')}")
        for primitive in primitives:
            source_start = int(primitive["start_timestep"])
            source_stop = int(primitive["end_timestep_exclusive"])
            start = int(primitive.get("reviewed_start_timestep", source_start))
            stop = int(primitive.get("reviewed_end_timestep_exclusive", source_stop))
            source_intervals.append({
                "start_timestep": start,
                "end_timestep_exclusive": stop,
                "start_s": start / rate,
                "end_s": stop / rate,
                "source_start_timestep": source_start,
                "source_end_timestep_exclusive": source_stop,
                "retention": "keep",
                "retention_reason": "production_reviewed_primitive",
                "primitive": primitive.get("primitive"),
            })
        provenance = "episode.json primitive_intervals reviewed bounds over the production-retained FMB arrays"
    keep = merged_ranges([
        {"start_timestep": row["start_timestep"], "end_timestep_exclusive": row["end_timestep_exclusive"]}
        for row in source_intervals if row["retention"] == "keep"
    ])
    if not keep or any(start < 0 or stop > frames or start >= stop for start, stop in keep):
        raise ValueError(f"invalid source retention for {record.get('episode_id')}: {keep}")
    return source_intervals, keep, complement_ranges(keep, frames), provenance


def array_schema(record: dict[str, Any], names: list[str]) -> dict[str, dict[str, Any]]:
    arrays = record.get("arrays", {})
    return {
        name: {key: arrays[name][key] for key in ("path", "shape", "dtype") if key in arrays[name]}
        for name in names if name in arrays
    }


def label_provenance(sidecars: dict[str, dict[str, Any]], info_cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, metadata in sidecars.items():
        channel = name.removesuffix(".jsonl")
        info_name = "quality_spans_info.json" if name in {
            "quality_spans.jsonl", "mistakes_v2.jsonl", "precision_windows.jsonl"
        } else f"{channel}_info.json"
        result[channel] = {"sidecar": metadata, "info_card": info_cards.get(info_name)}
    return result


def build_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"root": str(root), "stores": {}}
    for part in ("corpus", "fmb"):
        store = root / part
        episodes_path = store / "episodes.jsonl"
        episodes = read_jsonl(episodes_path)
        atoms: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for atom in read_jsonl(store / "subtask_atoms.jsonl"):
            atoms[str(atom["episode_id"])].append(atom)
        critics: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for interval in read_jsonl(store / "critic_intervals.jsonl"):
            critics[str(interval["episode_id"])].append(interval)
        actor_path = store / "actor_anchors_5hz.jsonl"
        actor_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for anchor in read_jsonl(actor_path):
            actor_rows[str(anchor["episode_id"])].append(anchor)
        sidecars = {
            name: file_record(store / name, rows=len(read_jsonl(store / name)))
            for name in SIDECARS
        }
        info_cards = {
            path.name: {**file_record(path), "provenance": parsed_provenance(path)}
            for path in sorted(store.glob("*_info.json"))
        }
        manifest_paths = sorted({
            *store.glob("*manifest*.json"), *store.glob("*reviewed.json"),
            *store.glob("validation_report.json"), *store.glob("views_summary.json"),
        })
        store_labels = label_provenance(sidecars, info_cards)
        provenance["stores"][part] = {
            "path": str(store),
            "episodes_index": file_record(episodes_path, rows=len(episodes)),
            "critic_intervals": file_record(store / "critic_intervals.jsonl", rows=sum(len(rows) for rows in critics.values())),
            "actor_anchor_view": file_record(actor_path, rows=sum(len(rows) for rows in actor_rows.values())),
            "sidecars": sidecars,
            "info_cards": info_cards,
            "source_manifests": {path.name: file_record(path) for path in manifest_paths},
        }
        for summary in episodes:
            eid = str(summary["episode_id"])
            episode_dir = store / summary.get("directory", f"episodes/{eid}")
            record_path = episode_dir / "episode.json"
            record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else summary
            family = "fmb" if part == "fmb" else str(summary["source"])
            rate = float(summary.get("native_rate_hz", summary.get("nominal_fps", record.get("nominal_fps"))))
            frames = int(summary.get("frames", summary.get("frame_count", record.get("frame_count"))))
            timestamps = None if part == "fmb" else np.load(episode_dir / "timestamp_s.npy", mmap_mode="r")
            source_intervals, retained, excluded, retention_source = source_retention(
                record, part, rate, frames, timestamps
            )
            anchors = actor_rows[eid]
            retained_anchors = [row for row in anchors if bool(row.get("retained"))]
            for anchor in retained_anchors:
                frame = int(anchor.get("anchor_frame", anchor.get("anchor_timestep")))
                if not any(start <= frame < stop for start, stop in retained):
                    raise ValueError(f"retained actor anchor {eid}:{frame} lies outside source retention {retained}")
            interruptions = []
            for parent in critics[eid]:
                for event in parent.get("interruption_events", []):
                    start_s, end_s = float(event["start_s"]), float(event["end_s"])
                    interruptions.append({
                        "parent_interval_index": int(parent["interval_index"]),
                        "start_timestep": seconds_to_frame(start_s, rate, frames),
                        "end_timestep_exclusive": seconds_to_frame(end_s, rate, frames),
                        "start_s": start_s,
                        "end_s": end_s,
                        "type": event.get("type", "interruption"),
                        "reason": event.get("reason"),
                    })
            modalities = camera_modalities(record, part)
            source_meta = record.get("source", {}) if part == "fmb" else {}
            if part == "fmb":
                state_names = [
                    name for name in record.get("arrays", {})
                    if name.startswith("obs/") and not any(token in name for token in ("side_", "wrist_"))
                ]
                action_names = [name for name in ("actions",) if name in record.get("arrays", {})]
                state_fields = state_names
                action_fields = action_names
                state_semantics = (
                    "Source-native FMB numeric state arrays: q/dq, TCP xyz + quaternion pose, gripper pose, "
                    "Jacobian, TCP force/torque/velocity where listed; units are not stated in episode metadata."
                )
                action_semantics = (
                    "Source-native 7-D actions array; binary commanded gripper event targets are separate arrays "
                    "and are not inferred from missing values."
                )
                gripper_semantics = "Source-native gripper pose plus commanded binary open/close event targets."
                source_metadata = {
                    key: source_meta[key] for key in ("path", "repo_id", "revision", "sha256", "bytes")
                    if key in source_meta
                }
            else:
                state_names, action_names = ["state"], ["action"]
                state_fields = record.get("state_fields", [])
                action_fields = record.get("action_fields", [])
                state_semantics = record.get("state_semantics")
                action_semantics = record.get("action_semantics")
                gripper_semantics = record.get("gripper_semantics")
                source_metadata = {
                    key: record[key] for key in (
                        "source_repo_id", "source_revision", "source_episode_index", "staged_episode_index",
                        "source", "component", "robot_type", "rate_provenance", "timestamp_provenance",
                    ) if key in record
                }
                uuid = record.get("annotations", {}).get("uuid")
                if uuid:
                    source_metadata["uuid"] = uuid
            inventory.append({
                "episode_id": eid, "family": family, "source": family, "store": part,
                "component": summary.get("component", record.get("component", "single_object_manipulation")),
                "split": str(summary.get("split", record.get("split", "train"))),
                "task": summary.get("task", record.get("task", "insert the object into the board")),
                "embodiment": summary.get("embodiment", record.get("embodiment", "Franka")),
                "native_rate_hz": rate, "frames": frames,
                "duration_s": float(summary.get("duration_s", frames / rate)),
                "retained_intervals": retained,
                "excluded_intervals": excluded,
                "source_intervals": source_intervals,
                "interruption_events": interruptions,
                "retention_provenance": {
                    "source": retention_source,
                    "atoms_used": False,
                    "actor_crosscheck": f"{len(retained_anchors)}/{len(anchors)} retained/total 5 Hz rows; all retained anchors inside source intervals",
                },
                "atoms": len(atoms[eid]),
                "cameras": modalities["rgb"],
                "camera_modalities": modalities,
                "actor_anchor_view": str(actor_path),
                "actor_anchor_count": len(retained_anchors),
                "actor_anchor_rows": len(anchors),
                "state_fields": state_fields,
                "state_semantics": state_semantics,
                "state_schema": array_schema(record, state_names),
                "action_fields": action_fields,
                "action_semantics": action_semantics,
                "action_schema": array_schema(record, action_names),
                "gripper_semantics": gripper_semantics,
                "source_identity": source_meta.get("path", summary.get("source_episode_index")),
                "source_metadata": source_metadata,
                "episode_metadata": str(record_path),
                "episode_metadata_sha256": sha256(record_path),
                "label_store": str(store),
                "label_provenance": store_labels,
            })
    inventory.sort(key=lambda row: (FAMILY_ORDER.index(row["family"]), row["episode_id"]))
    if len({row["episode_id"] for row in inventory}) != len(inventory):
        raise ValueError("episode_id is not unique across the federated stores")
    actual = tuple(family for family in FAMILY_ORDER if any(row["family"] == family for row in inventory))
    if actual != FAMILY_ORDER:
        raise ValueError(f"family inventory differs from the audit contract: {actual}")
    return inventory, provenance


def schema_table(inventory: list[dict[str, Any]]) -> list[str]:
    lines = ["| family / component / embodiment | episodes | rates (Hz) | splits | cameras | state/action schema |", "|---|---:|---|---|---|---|"]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        groups[(row["family"], str(row["component"]), str(row["embodiment"]))].append(row)
    for key, rows in sorted(groups.items()):
        rates = ", ".join(f"{x:g}" for x in sorted({row["native_rate_hz"] for row in rows}))
        splits = ", ".join(f"{k}:{v}" for k, v in sorted(Counter(row["split"] for row in rows).items()))
        cameras = ", ".join(sorted({"/".join(row["cameras"]) for row in rows}))
        schema = (str(rows[0]["state_semantics"]) + "; " + str(rows[0]["action_semantics"])).replace("|", "/")
        lines.append(f"| {' / '.join(key)} | {len(rows)} | {rates} | {splits} | {cameras} | {schema} |")
    return lines


def configured_root(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    diverse = config.get("diverse", {})
    if not diverse.get("enabled"):
        raise ValueError(f"diverse dataset is not enabled in {config_path}")
    if not diverse.get("root"):
        raise ValueError(f"diverse.root is missing in {config_path}")
    return Path(diverse["root"]), {
        **file_record(config_path), "diverse_enabled": True, "diverse_root": str(diverse["root"]),
    }


def resolve_active_root(config_path: Path, supplied_root: str | None) -> tuple[Path, dict[str, Any]]:
    root, metadata = configured_root(config_path)
    if supplied_root is not None and Path(supplied_root).resolve() != root.resolve():
        raise ValueError(f"--root {supplied_root} does not match {config_path} diverse.root {root}")
    return root, metadata


def init(args: argparse.Namespace) -> None:
    config_path, work = Path(args.config), Path(args.work)
    root, config_provenance = resolve_active_root(config_path, args.root)
    if work.exists() and (work / "population.json").is_file():
        recorded = json.loads((work / "population.json").read_text(encoding="utf-8"))
        if Path(recorded["root"]).resolve() != root.resolve():
            raise ValueError(f"recorded audit root {recorded['root']} no longer matches active root {root}")
        print(f"resume {work}")
        return
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"refusing to initialize non-empty {work}")
    work.mkdir(parents=True, exist_ok=True)
    for name in ("rounds", "checks", "evidence", "fixes", "sweeps", "validation"):
        (work / name).mkdir()
    inventory, provenance = build_inventory(root)
    write_jsonl(work / "inventory.jsonl", inventory)
    holdout_path = root / "holdout_episodes.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.is_file() else {"episode_ids": []}
    provenance.update({
        "active_config": config_provenance,
        "inventory": file_record(work / "inventory.jsonl", rows=len(inventory)),
        "holdout": {**file_record(holdout_path), "episodes": len(holdout.get("episode_ids", []))},
    })
    family_counts, split_counts = Counter(row["family"] for row in inventory), Counter(row["split"] for row in inventory)
    population = {
        "contract_date": "2026-10-07", "active_config": str(config_path),
        "root": str(root), "episodes": len(inventory), "families": dict(family_counts), "splits": dict(split_counts),
        "provenance": provenance,
        "sampling": "8 unchecked episodes per round; one per non-exhausted family, then rotating extras; without replacement",
    }
    (work / "population.json").write_text(json.dumps(population, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Diverse annotation sampled audit", "",
        "- Contract: `lerobot/src/lerobot/annotation/README.md`, “Diverse dataset: sampled audit” (2026-10-07).",
        f"- Active training pointer: `{config_path}` → `{root}`.",
        f"- Stores: `{root / 'corpus'}`, `{root / 'fmb'}`.",
        f"- Population: {len(inventory)} unique source episodes; families {dict(family_counts)}; splits {dict(split_counts)}.",
        "- Exclusions: every ReBot, external-ReBot, and rollout root. No training-config change or dataset adoption is authorized.",
        "- Sampling: without replacement; random within families; each non-exhausted family receives one slot when possible; extra slots rotate by family and prefer unseen split/component combinations.",
        "- Stopping rule: two consecutive 8-episode rounds with 0 wrong and at most 2 minor each, every represented family sampled, and no open fixes, second reads, or sweeps.",
        "- Spark status at start: unavailable (`ssh dgx`: no route to host); only bounded sampled evidence may be rendered locally. Full compilation/validation remains on Spark when needed.",
        "", "## Population schema", "", *schema_table(inventory), "", "## Rounds", "",
        "| round | seed | eligible | allocation | ordered picks | pre-fix wrong | pre-fix minor | open | clean streak |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    (work / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"work": str(work), "episodes": len(inventory), "families": family_counts, "splits": split_counts}, default=dict))


def refresh(args: argparse.Namespace) -> None:
    """Refresh the immutable population snapshot without altering the audit ledger or rounds."""
    config_path, work = Path(args.config), Path(args.work)
    if not (work / "population.json").is_file():
        raise FileNotFoundError(f"cannot refresh an uninitialized audit: {work}")
    recorded = json.loads((work / "population.json").read_text(encoding="utf-8"))
    active_root, config_provenance = configured_root(config_path)
    if Path(recorded["root"]).resolve() != active_root.resolve():
        raise ValueError(f"recorded audit root {recorded['root']} no longer matches active root {active_root}")
    recorded_review_root = Path(recorded.get("review_root", recorded["root"])).resolve()
    if args.root is None:
        root = active_root.resolve()
    else:
        root = Path(args.root).resolve()
        if root != recorded_review_root:
            raise ValueError(f"--root {root} is not the recorded audit review root {recorded_review_root}")
    if recorded_review_root != root:
        raise ValueError(
            f"audit review root is already handed off to {recorded_review_root}; "
            "refreshing from the configured active root would mix review pointers"
        )
    inventory, provenance = build_inventory(root)
    write_jsonl(work / "inventory.jsonl", inventory)
    holdout_path = root / "holdout_episodes.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.is_file() else {"episode_ids": []}
    provenance.update({
        "active_config": config_provenance,
        "inventory": file_record(work / "inventory.jsonl", rows=len(inventory)),
        "holdout": {**file_record(holdout_path), "episodes": len(holdout.get("episode_ids", []))},
    })
    family_counts = Counter(row["family"] for row in inventory)
    split_counts = Counter(row["split"] for row in inventory)
    recorded.update({
        "active_config": str(config_path), "root": str(active_root), "episodes": len(inventory),
        "families": dict(family_counts), "splits": dict(split_counts), "provenance": provenance,
    })
    (work / "population.json").write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"work": str(work), "episodes": len(inventory), "refreshed": True}))


def _sampling_identity(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str, float]]:
    identity: dict[str, tuple[str, str, float]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in identity:
            raise ValueError(f"duplicate episode_id in audit inventory: {episode_id}")
        identity[episode_id] = (str(row["family"]), str(row["split"]), float(row["native_rate_hz"]))
    return identity


def handoff(args: argparse.Namespace) -> None:
    """Point subsequent audit review at a validated corrected sibling root.

    The configured training root remains the immutable discovery root.  This command
    changes only the audit's population metadata and inventory review pointers.
    """
    work, corrected_root = Path(args.work), Path(args.corrected_root).resolve()
    population_path, inventory_path = work / "population.json", work / "inventory.jsonl"
    if not population_path.is_file() or not inventory_path.is_file():
        raise ValueError(f"audit population is not initialized under {work}")
    population = json.loads(population_path.read_text(encoding="utf-8"))
    prior_review_root = Path(population.get("review_root", population["root"])).resolve()
    if corrected_root == prior_review_root:
        raise ValueError("corrected review root must be a new sibling root")

    manifest_path = corrected_root / "corrected_store_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"corrected root is missing {manifest_path.name}: {corrected_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_source = manifest.get("source_root")
    if not isinstance(manifest_source, str) or Path(manifest_source).resolve() != prior_review_root:
        raise ValueError(
            f"corrected manifest source {manifest_source!r} does not equal prior audit review root {prior_review_root}"
        )
    if not isinstance(manifest.get("post_write_validation"), dict):
        raise ValueError("corrected manifest lacks post_write_validation")

    old_inventory = read_jsonl(inventory_path)
    new_inventory, new_provenance = build_inventory(corrected_root)
    expected_episodes = int(population["episodes"])
    if len(old_inventory) != expected_episodes or len(new_inventory) != expected_episodes:
        raise ValueError(
            f"corrected root must preserve {expected_episodes} episodes; "
            f"old/new inventory sizes are {len(old_inventory)}/{len(new_inventory)}"
        )
    old_identity, new_identity = _sampling_identity(old_inventory), _sampling_identity(new_inventory)
    if old_identity != new_identity:
        missing = sorted(old_identity.keys() - new_identity.keys())
        added = sorted(new_identity.keys() - old_identity.keys())
        changed = sorted(key for key in old_identity.keys() & new_identity.keys() if old_identity[key] != new_identity[key])
        raise ValueError(
            "corrected root changed sampling identity "
            f"(missing={missing[:5]}, added={added[:5]}, family/split/rate changed={changed[:5]})"
        )

    family_counts = Counter(row["family"] for row in new_inventory)
    split_counts = Counter(row["split"] for row in new_inventory)
    if dict(family_counts) != population["families"] or dict(split_counts) != population["splits"]:
        raise ValueError("corrected root changed recorded family or split counts")

    inventory_tmp = inventory_path.with_suffix(".jsonl.handoff.tmp")
    population_tmp = population_path.with_suffix(".json.handoff.tmp")
    write_jsonl(inventory_tmp, new_inventory)
    provenance = population.setdefault("provenance", {})
    handoffs = provenance.setdefault("review_handoffs", [])
    handoffs.append({
        "from": str(prior_review_root),
        "to": str(corrected_root),
        "manifest": file_record(manifest_path),
        "source_root": manifest_source,
    })
    provenance["review_root"] = new_provenance
    inventory_record = file_record(inventory_tmp, rows=len(new_inventory))
    inventory_record["path"] = str(inventory_path)
    provenance["inventory"] = inventory_record
    population.setdefault("active_root", population["root"])
    population["review_root"] = str(corrected_root)
    population_tmp.write_text(json.dumps(population, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inventory_tmp.replace(inventory_path)
    population_tmp.replace(population_path)
    print(json.dumps({"work": str(work), "review_root": str(corrected_root), "episodes": len(new_inventory)}))


def existing_picks(work: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((work / "rounds").glob("round_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["picks"])
    return rows


def _validate_completed_round(work: Path, number: int) -> dict[str, Any]:
    """Validate the auditable facts needed before a later round may be drawn."""
    round_path = work / "rounds" / f"round_{number:02d}.json"
    if not round_path.is_file():
        raise ValueError(f"missing {round_path}")
    row = json.loads(round_path.read_text(encoding="utf-8"))
    if row.get("round") != number:
        raise ValueError(f"round number mismatch in {round_path}: {row.get('round')!r}")
    if row.get("status") != "complete":
        raise ValueError(f"round {number} status is {row.get('status')!r}")

    completion = row.get("completion")
    if not isinstance(completion, dict) or any(
        isinstance(completion.get(key), bool)
        or not isinstance(completion.get(key), int)
        or completion.get(key) != 0
        for key in COMPLETION_KEYS
    ):
        raise ValueError(
            f"round {number} must record completion as zero {COMPLETION_KEYS}; got {completion!r}"
        )

    picks = row.get("picks")
    if not isinstance(picks, list) or len(picks) != ROUND_SIZE:
        size = len(picks) if isinstance(picks, list) else None
        raise ValueError(f"round {number} must contain exactly {ROUND_SIZE} picks; got {size}")
    episode_ids = [pick.get("episode_id") for pick in picks if isinstance(pick, dict)]
    if len(episode_ids) != ROUND_SIZE or any(not isinstance(episode_id, str) or not episode_id for episode_id in episode_ids):
        raise ValueError(f"round {number} contains a pick without an episode_id")
    if len(set(episode_ids)) != ROUND_SIZE:
        raise ValueError(f"round {number} contains duplicate episode picks")
    if any(not isinstance(pick.get("family"), str) or not pick["family"] for pick in picks):
        raise ValueError(f"round {number} contains a pick without a source family")

    issue_counts = Counter()
    for pick in picks:
        check_path = work / "checks" / f"round_{number:02d}__{pick['episode_id']}.json"
        if not check_path.is_file():
            raise ValueError(f"missing {check_path}")
        check = json.loads(check_path.read_text(encoding="utf-8"))
        if check.get("episode_id") != pick["episode_id"]:
            raise ValueError(f"episode mismatch in {check_path}")
        if check.get("status") != "complete" or check.get("second_read_required") is not False:
            raise ValueError(f"check remains open: {check_path}")
        issues = check.get("issues")
        if not isinstance(issues, list):
            raise ValueError(f"check must record an issues list: {check_path}")
        for issue in issues:
            severity = issue.get("severity") if isinstance(issue, dict) else None
            if severity not in ("wrong", "minor"):
                raise ValueError(f"invalid or unresolved issue severity {severity!r} in {check_path}")
            issue_counts[severity] += 1

    counts = row.get("pre_fix_counts")
    if not isinstance(counts, dict):
        raise ValueError(f"round {number} must record pre_fix_counts")
    for key in ("wrong", "minor", "open"):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"round {number} has invalid pre-fix {key} count: {value!r}")
    if counts["wrong"] != issue_counts["wrong"] or counts["minor"] != issue_counts["minor"]:
        raise ValueError(
            f"round {number} pre-fix counts do not match its checks: "
            f"recorded wrong/minor={counts['wrong']}/{counts['minor']}, "
            f"checks={issue_counts['wrong']}/{issue_counts['minor']}"
        )
    if counts["open"] != 0:
        raise ValueError(f"round {number} still records {counts['open']} open issues")
    return row


def _round_is_clean(row: dict[str, Any]) -> bool:
    counts = row["pre_fix_counts"]
    threshold_clean = counts["wrong"] == 0 and counts["minor"] <= 2
    if not threshold_clean:
        return False
    systematic = row.get("systematic_issue_detected")
    if not isinstance(systematic, bool):
        raise ValueError(
            f"clean-threshold round {row['round']} must explicitly record systematic_issue_detected true or false"
        )
    return not systematic


def require_prior_rounds_complete(work: Path, round_number: int) -> None:
    """Do not let a draw outrun repairs or continue after the documented stop."""
    prior_rows: list[dict[str, Any]] = []
    for number in range(1, round_number):
        try:
            prior_rows.append(_validate_completed_round(work, number))
        except ValueError as exc:
            raise ValueError(f"cannot sample round {round_number}: {exc}") from exc

    clean_streak = 0
    sampled_families: set[str] = set()
    for row in prior_rows:
        sampled_families.update(pick["family"] for pick in row["picks"])
        clean_streak = clean_streak + 1 if _round_is_clean(row) else 0

    inventory_families = {row["family"] for row in read_jsonl(work / "inventory.jsonl") if row.get("family")}
    represented_families = inventory_families or set(FAMILY_ORDER)
    if clean_streak >= 2 and represented_families <= sampled_families:
        raise ValueError(
            f"cannot sample round {round_number}: stopping rule already met "
            f"({clean_streak} consecutive clean rounds; all represented families covered)"
        )


def allocate_families(
    pools: dict[str, list[dict[str, Any]]], prior: list[dict[str, Any]], round_number: int, total: int = 8
) -> dict[str, int]:
    allocation = {family: 1 for family in FAMILY_ORDER if pools[family]}
    remaining = total - sum(allocation.values())
    shift = (round_number - 1) % len(FAMILY_ORDER)
    rotation = list(FAMILY_ORDER[shift:] + FAMILY_ORDER[:shift])
    checked = Counter(row["family"] for row in prior)
    while remaining > 0:
        available = [family for family in rotation if len(pools[family]) > allocation.get(family, 0)]
        if not available:
            break
        least_sampled = min(checked[family] + allocation.get(family, 0) for family in available)
        family = next(
            family for family in rotation
            if family in available and checked[family] + allocation.get(family, 0) == least_sampled
        )
        allocation[family] = allocation.get(family, 0) + 1
        index = rotation.index(family)
        rotation = rotation[index + 1:] + rotation[:index + 1]
        remaining -= 1
    return allocation


def choose_family_rows(
    candidates: list[dict[str, Any]], count: int, rng: random.Random,
    seen: dict[str, set[tuple[str, str]]],
) -> list[dict[str, Any]]:
    available = list(candidates)
    rng.shuffle(available)
    chosen: list[dict[str, Any]] = []
    for _ in range(min(count, len(available))):
        def score(row: dict[str, Any]) -> tuple[int, bool, bool, bool, bool]:
            family = row["family"]
            flags = tuple((family, str(row.get(field, ""))) in seen[field] for field in seen)
            return (sum(flags), *flags)
        row = min(available, key=score)
        available.remove(row)
        chosen.append(row)
        for field in seen:
            seen[field].add((row["family"], str(row.get(field, ""))))
    return chosen


def sample(args: argparse.Namespace) -> None:
    work = Path(args.work)
    path = work / "rounds" / f"round_{args.round:02d}.json"
    require_prior_rounds_complete(work, args.round)
    if path.is_file():
        print(path.read_text(encoding="utf-8"))
        return
    inventory, prior = read_jsonl(work / "inventory.jsonl"), existing_picks(work)
    checked = {row["episode_id"] for row in prior}
    seen = {
        field: {(row["family"], str(row.get(field, ""))) for row in prior}
        for field in ("split", "component", "task", "embodiment")
    }
    pools = {
        family: [row for row in inventory if row["family"] == family and row["episode_id"] not in checked]
        for family in FAMILY_ORDER
    }
    eligible = sum(len(rows) for rows in pools.values())
    allocation = allocate_families(pools, prior, args.round)
    rng = random.Random(args.seed)
    picks: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        picks.extend(choose_family_rows(pools[family], allocation.get(family, 0), rng, seen))
    if len(picks) != ROUND_SIZE:
        raise ValueError(
            f"cannot sample round {args.round}: only {len(picks)} unchecked episodes remain; "
            f"a short final batch is not a {ROUND_SIZE}-episode audit round"
        )
    rng.shuffle(picks)
    payload = {
        "round": args.round, "seed": args.seed, "eligible_pool": eligible,
        "eligible_by_family": {family: len(pools[family]) for family in FAMILY_ORDER},
        "allocation": allocation, "ordered_picks": [row["episode_id"] for row in picks], "picks": picks,
        "status": "pending_review", "pre_fix_counts": {"wrong": None, "minor": None, "open": None},
        "completion": {"fixes_open": None, "second_reads_open": None, "sweeps_open": None},
        "systematic_issue_detected": None,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in picks:
        check = {
            "round": args.round, "episode_id": row["episode_id"], "family": row["family"], "store": row["store"],
            "split": row["split"], "native_rate_hz": row["native_rate_hz"], "retained_intervals": row["retained_intervals"],
            "cameras": row["cameras"], "status": "pending", "second_read_required": False, "issues": [], "decision_note": "",
        }
        (work / "checks" / f"round_{args.round:02d}__{row['episode_id']}.json").write_text(
            json.dumps(check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    pick_text = "<br>".join(f"{row['family']}: {row['episode_id']} ({row['split']})" for row in picks)
    with (work / "audit.md").open("a", encoding="utf-8") as stream:
        stream.write(f"| {args.round} | {args.seed} | {eligible} | {allocation} | {pick_text} | pending | pending | pending | 0 |\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def labels(args: argparse.Namespace) -> None:
    work = Path(args.work)
    round_row = json.loads((work / "rounds" / f"round_{args.round:02d}.json").read_text(encoding="utf-8"))
    population = json.loads((work / "population.json").read_text(encoding="utf-8"))
    root = Path(population.get("review_root", population["root"]))
    inventory = {row["episode_id"]: row for row in read_jsonl(work / "inventory.jsonl")}
    for pick in round_row["picks"]:
        eid = pick["episode_id"]
        current = inventory.get(eid)
        if current is None:
            raise ValueError(f"sampled episode {eid} is missing from the current audit inventory")
        for key in ("family", "split", "native_rate_hz"):
            if current[key] != pick[key]:
                raise ValueError(f"sampled episode {eid} changed {key}: {pick[key]!r} -> {current[key]!r}")
        store = root / current["store"]
        out = work / "evidence" / f"round_{args.round:02d}" / eid; out.mkdir(parents=True, exist_ok=True)
        payload = {"inventory": current, "channels": {name: [row for row in read_jsonl(store / name) if str(row.get("episode_id")) == eid] for name in SIDECARS}}
        (out / "effective_labels.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote effective-label packets for {len(round_row['picks'])} episodes")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--config", default=str(ACTIVE_CONFIG))
    p.add_argument("--root")
    p.add_argument("--work", required=True)
    p.set_defaults(func=init)
    p = sub.add_parser("refresh")
    p.add_argument("--config", default=str(ACTIVE_CONFIG))
    p.add_argument("--root")
    p.add_argument("--work", required=True)
    p.set_defaults(func=refresh)
    p = sub.add_parser("handoff")
    p.add_argument("--work", required=True)
    p.add_argument("--corrected-root", required=True)
    p.set_defaults(func=handoff)
    p = sub.add_parser("sample"); p.add_argument("--work", required=True); p.add_argument("--round", type=int, required=True); p.add_argument("--seed", type=int, required=True); p.set_defaults(func=sample)
    p = sub.add_parser("labels"); p.add_argument("--work", required=True); p.add_argument("--round", type=int, required=True); p.set_defaults(func=labels)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
