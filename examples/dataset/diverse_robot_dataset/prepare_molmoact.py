#!/usr/bin/env python

"""Scan, nominate, render, and finalize the MolmoAct Dataset (Household + Tabletop) for v2.

The source is LeRobot v3 on the Hub with an end-effector state (xyz metres, three Euler
angles, gripper ratio) and no joint channel; `action.abs_ee_action` equals the state at the
same timestep, so the source is ingested under the copy_state exception. Episodes are short
(median ~8 s) and each covers one task, so the review unit is the whole episode: one keep
span carrying the annotated instruction as its subtask, a quality, and mistake events.

Selection is breadth-first: a fixed number of episodes per task (default 2), which is what
makes this source useful -- 134 tasks in home and tabletop scenes -- rather than depth.

    prepare_molmoact.py --spec molmoact_household scan
    prepare_molmoact.py --spec molmoact_household nominate --per-task 2
    prepare_molmoact.py --spec molmoact_household proxies
    # review the sheets, write verdicts.json, then
    prepare_molmoact.py --spec molmoact_household finalize --verdicts .../verdicts.json

A later round tops the quota up without re-reviewing the first round: `nominate --prior` takes
the earlier candidates.json, excludes its nominees from the pool, and writes the union of both
rounds (each record stamped `round`); `proxies --round v3 --sheets-dir` renders only the new
round into a separate review dir with a `batches.json` for the batch writer. `finalize` then
takes the union verdict file (see migration/diverse_v3_control_mode_2026-09-19/molmoact/).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "lerobot/src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "lerobot/src"))
from prepare_droid import contact_sheet, overview_offsets, read_json, write_json  # noqa: E402

SOURCE_CONFIG = Path(__file__).with_name("molmoact_sources.json")
CAMERAS = ("observation.images.primary", "observation.images.secondary", "observation.images.wrist")
ACTION_SECONDS = 29.0 / 30.0
SCREEN_STRIDE_S = 1.0
# Metres for translation, radians for the Euler triple, ratio for the gripper: the smallest
# motion that is still a deliberate move inside a one-second window.
XYZ_ACTIVITY_M = 0.01
ROT_ACTIVITY_RAD = 0.05
GRIPPER_ACTIVITY = 0.05
# A translation step above this between consecutive frames is a tracking glitch, not motion.
XYZ_STEP_LIMIT_M = 0.05
REVIEWER_MODEL = "claude-fable-5-1"
KEEP_REASONS = {"useful_motion", "informative_mistake", "recovery", "task_required_hold"}
MISTAKE_TYPES = {"failed_close", "slip", "drop", "knock", "wrong_target"}
PRIOR_ROUND = "v2"
ROUND = "v3"
# Review batches: sorted by episode index and dealt round-robin, so each batch mixes tasks.
BATCH_SIZE = 19


class Paths:
    def __init__(self, spec_name: str, build_root: Path) -> None:
        from lerobot.datasets.diverse_pilot import load_source_specs

        self.spec = next(item for item in load_source_specs(SOURCE_CONFIG) if item.name == spec_name)
        self.component = spec_name.split("_", 1)[1]  # household | tabletop
        self.root = build_root / "molmoact"
        self.metadata_root = self.root / "metadata" / spec_name
        self.staging_root = self.root / "staging"
        self.source_root = self.staging_root / self.spec.repo_id.replace("/", "__")
        self.review_root = self.root / "review" / self.component
        self.scan = self.review_root / "episode_scan.json"
        self.candidates = self.review_root / "candidates.json"
        self.audit = self.root / "audits" / f"{spec_name}.json"
        self.index = self.root / "index.json"


def episode_rows(metadata_root: Path) -> list[dict]:
    files = sorted((metadata_root / "meta/episodes").glob("**/*.parquet"))
    columns = ["episode_index", "tasks", "length", "dataset_from_index", "data/chunk_index", "data/file_index"]
    columns += [f"videos/{camera}/{field}" for camera in CAMERAS for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    return pq.read_table(files, columns=columns).to_pylist()


def annotated_tasks(metadata_root: Path) -> dict[int, str]:
    table = pq.read_table(metadata_root / "meta/tasks_annotated.parquet").to_pandas().reset_index()
    return {int(row["episode_index"]): str(row["task"]) for _, row in table.iterrows()}


def screen(relative: np.ndarray, states: np.ndarray) -> tuple[list[float], list[bool]]:
    """One-second cells from t=0; a cell is active when its future window moves."""
    last = float(relative[-1]) - ACTION_SECONDS
    if last < 0.0:
        return [0.0], [True]
    count = int(math.floor(last / SCREEN_STRIDE_S)) + 1
    anchors = np.arange(count, dtype=np.float64) * SCREEN_STRIDE_S
    active = []
    for anchor in anchors:
        mask = (relative >= anchor - 1e-9) & (relative <= anchor + ACTION_SECONDS + 1e-9)
        future = states[mask]
        if len(future) < 2:
            active.append(False)
            continue
        xyz = float(np.ptp(future[:, :3], axis=0).max())
        rot = float(np.ptp(future[:, 3:6], axis=0).max())
        grip = float(np.ptp(future[:, 6]))
        active.append(xyz >= XYZ_ACTIVITY_M or rot >= ROT_ACTIVITY_RAD or grip >= GRIPPER_ACTIVITY)
    return [float(value) for value in anchors], active


def scan(paths: Paths) -> dict:
    info = read_json(paths.metadata_root / "meta/info.json")
    fps = float(info["fps"])
    rows = episode_rows(paths.metadata_root)
    annotated = annotated_tasks(paths.metadata_root)
    by_file: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_file[(int(row["data/chunk_index"]), int(row["data/file_index"]))].append(row)
    summaries = []
    for (chunk, index), group in sorted(by_file.items()):
        path = paths.source_root / info["data_path"].format(chunk_index=chunk, file_index=index)
        shard = pq.read_table(path, columns=["episode_index", "timestamp", "observation.state"])
        base = min(int(item["dataset_from_index"]) for item in group)
        for row in group:
            offset = int(row["dataset_from_index"]) - base
            length = int(row["length"])
            block = shard.slice(offset, length)
            episode = np.asarray(block["episode_index"].to_pylist())
            if not np.all(episode == int(row["episode_index"])):
                raise ValueError(f"Shard slice does not map to episode {row['episode_index']}")
            timestamps = np.asarray(block["timestamp"].to_pylist(), dtype=np.float64)
            states = np.asarray(block["observation.state"].to_pylist(), dtype=np.float64)
            relative = timestamps - timestamps[0]
            wraps = int((np.abs(np.diff(states[:, 3:6], axis=0)) > math.pi).any(axis=1).sum())
            states[:, 3:6] = np.unwrap(states[:, 3:6], axis=0)
            anchors, active = screen(relative, states)
            xyz_steps = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)
            tasks = [str(item) for item in (row.get("tasks") or []) if str(item).strip()]
            summaries.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "frames": length,
                    "duration_s": float(relative[-1]),
                    "fps": fps,
                    "task": tasks[0] if tasks else "",
                    "annotated_task": annotated.get(int(row["episode_index"]), ""),
                    "screen_cells": len(anchors),
                    "active_cells": int(sum(active)),
                    "active_fraction": float(np.mean(active)) if active else 0.0,
                    "xyz_path_length_m": float(xyz_steps.sum()),
                    "max_xyz_step_m": float(xyz_steps.max(initial=0.0)),
                    "gripper_path_length": float(np.abs(np.diff(states[:, 6])).sum()),
                    "euler_wrap_crossings": wraps,
                    "max_clock_step_s": float(np.diff(timestamps).max()),
                    "discontinuous": bool(xyz_steps.max(initial=0.0) > XYZ_STEP_LIMIT_M),
                }
            )
    value = {
        "source": paths.spec.repo_id,
        "episodes_scanned": len(rows),
        "valid_episodes": sum(1 for item in summaries if not item["discontinuous"]),
        "screen": {
            "stride_s": SCREEN_STRIDE_S,
            "xyz_activity_m": XYZ_ACTIVITY_M,
            "rot_activity_rad": ROT_ACTIVITY_RAD,
            "gripper_activity": GRIPPER_ACTIVITY,
            "xyz_step_limit_m": XYZ_STEP_LIMIT_M,
            "measured_from": "observation.state (the source action is a copy of it)",
        },
        "summaries": summaries,
    }
    write_json(paths.scan, value)
    return value


def nominate(paths: Paths, per_task: int, prior: Path | None = None) -> dict:
    """Pick per_task episodes per task; with a prior round's candidates.json, fill only the shortfall."""
    scanned = read_json(paths.scan)
    prior_candidates = read_json(prior)["candidates"] if prior is not None else []
    prior_indices = {int(item["episode_index"]) for item in prior_candidates}
    prior_per_task = Counter(item["task"] for item in prior_candidates)
    pool = [item for item in scanned["summaries"] if not item["discontinuous"] and item["active_cells"] >= 2 and int(item["episode_index"]) not in prior_indices]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in pool:
        by_task[item["task"]].append(item)
    chosen = []
    for task, items in sorted(by_task.items()):
        quota = max(per_task - prior_per_task[task], 0)
        if quota == 0:
            continue
        items = sorted(items, key=lambda item: item["episode_index"])
        # Spread the picks over the task's recording order so two episodes are unlikely to
        # come from the same session, then take the most active one inside each slice.
        slices = np.array_split(np.arange(len(items)), min(quota, len(items)))
        for chunk in slices:
            candidates = [items[int(i)] for i in chunk]
            chosen.append(max(candidates, key=lambda item: (item["active_fraction"], item["xyz_path_length_m"])))
    value = {
        "source": scanned["source"],
        "component": paths.component,
        "selection_status": "nominated_pending_visual_review",
        "per_task": per_task,
        "tasks": len(by_task),
        "pool_size": len(pool),
        "candidate_count": len(chosen),
        "rejection_screen": {
            "discontinuous_xyz_step_m": XYZ_STEP_LIMIT_M,
            "minimum_active_cells": 2,
        },
        "grouping": "one physical Franka; diversity comes from tasks and scenes, so the quota is per task",
        "candidates": chosen,
    }
    if prior is not None:
        # The union is what finalize reviews; prior records are carried verbatim plus their round.
        value["candidates"] = [{**item, "round": PRIOR_ROUND} for item in prior_candidates] + [{**item, "round": ROUND} for item in chosen]
        value["candidate_count"] = len(value["candidates"])
        value["new_candidate_count"] = len(chosen)
        value["round_counts"] = {PRIOR_ROUND: len(prior_candidates), ROUND: len(chosen)}
        value["prior_candidates_path"] = str(prior)
        backup = paths.candidates.with_name(f"candidates_{PRIOR_ROUND}.json")
        if paths.candidates.is_file() and not backup.is_file():
            shutil.copyfile(paths.candidates, backup)
    write_json(paths.candidates, value)
    return value


def batches_for(indices: list[int]) -> list[list[int]]:
    indices = sorted(indices)
    count = math.ceil(len(indices) / BATCH_SIZE)
    return [indices[start::count] for start in range(count)]


def proxies(paths: Paths, round_name: str | None = None, sheets_dir: Path | None = None) -> list[Path]:
    """Three sheets per candidate; a round renders only its own candidates, into sheets_dir when given."""
    info = read_json(paths.metadata_root / "meta/info.json")
    rows = {int(row["episode_index"]): row for row in episode_rows(paths.metadata_root)}
    selected = [item for item in read_json(paths.candidates)["candidates"] if round_name is None or item.get("round") == round_name]
    if sheets_dir is None:
        sheets_dir = paths.review_root
    else:
        # The batch writer and the verdict merge read these next to the sheets.
        sheets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.candidates, sheets_dir / "candidates.json")
        write_json(sheets_dir / "batches.json", batches_for([int(item["episode_index"]) for item in selected]))
    written = []
    for item in selected:
        index = int(item["episode_index"])
        row = rows[index]
        stem = sheets_dir / f"episode_{index:06d}"
        for camera in CAMERAS:
            label = camera.rsplit(".", 1)[-1]
            video = paths.source_root / info["video_path"].format(
                video_key=camera,
                chunk_index=int(row[f"videos/{camera}/chunk_index"]),
                file_index=int(row[f"videos/{camera}/file_index"]),
            )
            start = float(row[f"videos/{camera}/from_timestamp"])
            duration = float(row[f"videos/{camera}/to_timestamp"]) - start
            header = f"ep {index}  {duration:.1f}s  {label}  {item['annotated_task'][:70]}"
            destination = stem.with_suffix(f".{label}.jpg")
            if not destination.exists():
                contact_sheet(video, start, overview_offsets(duration), destination, header=header)
            written.append(destination)
    return written


def segments_for(review: dict, relative: np.ndarray, states: np.ndarray) -> list[dict]:
    """Reviewer spans subdivided by the activity screen; static cells become rejects."""
    duration = float(relative[-1])
    anchors, active = screen(relative, states)
    boundaries = [*anchors, duration]
    spans = sorted(review["spans"], key=lambda span: float(span["start_s"]))
    cursor = 0.0
    for span in spans:
        if abs(float(span["start_s"]) - cursor) > 1e-6 or float(span["end_s"]) <= float(span["start_s"]):
            raise ValueError("Reviewer spans must be contiguous and ordered")
        cursor = float(span["end_s"])
    if abs(cursor - duration) > 0.05:
        raise ValueError(f"Reviewer spans end at {cursor:.3f}s, episode lasts {duration:.3f}s")
    spans[-1]["end_s"] = duration
    cuts = sorted({0.0, duration, *boundaries, *[float(span["start_s"]) for span in spans]})
    segments = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end - start < 1e-6:
            continue
        span = next(item for item in spans if float(item["start_s"]) <= start < float(item["end_s"]))
        cell = max(i for i, anchor in enumerate(anchors) if anchor <= start + 1e-9)
        segment = {"start_s": round(start, 6), "end_s": round(end, 6), "retention": span["retention"], "retention_reason": span["retention_reason"]}
        if span["retention"] == "keep":
            if not active[cell] and span["retention_reason"] != "task_required_hold":
                segment.update(retention="reject", retention_reason="static")
            else:
                segment.update(subtask=span["subtask"], quality=int(span["quality"]),
                               mistake_events=[event for event in span.get("mistake_events", []) if float(event["start_s"]) < end and float(event["end_s"]) > start])
        segments.append(segment)
    merged: list[dict] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        same = previous is not None and all(previous.get(k) == segment.get(k) for k in ("retention", "retention_reason", "subtask", "quality")) and previous.get("mistake_events", []) == segment.get("mistake_events", [])
        if same:
            previous["end_s"] = segment["end_s"]
        else:
            merged.append(segment)
    return merged


def finalize(paths: Paths, verdicts_path: Path, review_prompt: str) -> dict:
    """Write reviews, annotations, selection, and the acquisition manifest for accepted episodes."""
    from lerobot.datasets.diverse_pilot import resolve_lerobot_payload

    verdicts = read_json(verdicts_path)
    candidates = {int(item["episode_index"]): item for item in read_json(paths.candidates)["candidates"]}
    expected = {str(index) for index in candidates}
    supplied = set(verdicts["episodes"])
    if supplied != expected:
        raise ValueError(
            "Finalization requires a verdict for every nominated candidate; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}. "
            "Omitted candidates are unreviewed, not rejected. No output was written."
        )
    for index, verdict in verdicts["episodes"].items():
        if not isinstance(verdict.get("accept"), bool):
            raise ValueError(f"episode {index}: accept must be an explicit boolean")
    info = read_json(paths.metadata_root / "meta/info.json")
    rows = {int(row["episode_index"]): row for row in episode_rows(paths.metadata_root)}
    accepted = sorted(int(k) for k, v in verdicts["episodes"].items() if v["accept"])
    report = {}
    for index in accepted:
        verdict = verdicts["episodes"][str(index)]
        item = candidates[index]
        quality = int(verdict["quality"])
        if not 1 <= quality <= 5:
            raise ValueError(f"episode {index}: quality {quality} outside 1..5")
        events = verdict.get("mistake_events", [])
        for event in events:
            if event["kind"] not in MISTAKE_TYPES or float(event["end_s"]) <= float(event["start_s"]):
                raise ValueError(f"episode {index}: bad mistake event {event}")
        duration = float(item["duration_s"])
        keep_reason = verdict.get("retention_reason", "informative_mistake" if events else "useful_motion")
        if keep_reason not in KEEP_REASONS:
            raise ValueError(f"episode {index}: bad keep reason {keep_reason}")
        # The annotated sentence is occasionally truncated; a verdict may name the task instead.
        task_text = verdict.get("task") or item["annotated_task"] or item["task"]
        review = {
            "source_episode_index": index,
            "task": task_text,
            "source_task": item["task"],
            "outcome": verdict.get("outcome", "success"),
            "reviewer_notes": verdict.get("notes", ""),
            "spans": [
                {
                    "start_s": 0.0, "end_s": duration, "retention": "keep", "retention_reason": keep_reason,
                    "subtask": task_text, "quality": quality,
                    "mistake_events": events,
                }
            ],
        }
        write_json(paths.review_root / f"episode_{index:06d}.review.json", review)
        row = rows[index]
        path = paths.source_root / info["data_path"].format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        shard = pq.read_table(path, columns=["episode_index", "timestamp", "observation.state"])
        table = [r for r in rows.values() if r["data/chunk_index"] == row["data/chunk_index"] and r["data/file_index"] == row["data/file_index"]]
        base = min(int(r["dataset_from_index"]) for r in table)
        block = shard.slice(int(row["dataset_from_index"]) - base, int(row["length"]))
        timestamps = np.asarray(block["timestamp"].to_pylist(), dtype=np.float64)
        states = np.asarray(block["observation.state"].to_pylist(), dtype=np.float64)
        states[:, 3:6] = np.unwrap(states[:, 3:6], axis=0)
        relative = timestamps - timestamps[0]
        segments = segments_for(review, relative, states)
        annotations = {
            "source_episode_index": index,
            "task": review["task"],
            "source_task": item["task"],
            "outcome": review["outcome"],
            "reviewer_notes": review["reviewer_notes"],
            "episode_duration_s": float(relative[-1]),
            "review_status": "validated",
            "review_basis": "twelve-tile contact sheets of all three cameras over the whole episode; model review under the ReBot rubric",
            "review_provenance": "model_reviewed",
            "quality_provenance": "model_reviewed_rebot_rubric",
            "reviewer_model": verdict.get("reviewer") or verdicts.get("reviewer") or REVIEWER_MODEL,
            "activity_screen_subdivisions": {"stride_s": SCREEN_STRIDE_S, "xyz_activity_m": XYZ_ACTIVITY_M, "rot_activity_rad": ROT_ACTIVITY_RAD, "gripper_activity": GRIPPER_ACTIVITY},
            "source_keep_intervals_s": [],
            "segments": segments,
            "required_segment_fields": ["start_s", "end_s", "retention", "retention_reason"],
            "required_keep_segment_fields": ["subtask", "quality", "mistake_events"],
        }
        write_json(paths.review_root / f"episode_{index:06d}.annotations.json", annotations)
        kept = [s for s in segments if s["retention"] == "keep"]
        report[index] = {"quality": quality, "mistake_events": len(events), "kept_s": round(sum(s["end_s"] - s["start_s"] for s in kept), 2), "segments": len(segments)}
    selection = {
        "source": paths.spec.repo_id,
        "component": paths.component,
        "selection_status": "visually_accepted",
        "accepted_episode_indices": accepted,
        "rejected_candidate_indices": sorted(set(candidates) - set(accepted)),
        "review_basis": "twelve-tile contact sheets of all three cameras over the whole episode",
        "review_provenance": "model_reviewed",
        "reviewer_model": verdicts.get("reviewer") or REVIEWER_MODEL,
        "review_date": verdicts.get("review_date"),
        "review_prompt": review_prompt,
        "selection_rule": f"up to {read_json(paths.candidates)['per_task']} episodes per task; a nominee is rejected only on a visible quality or scope failure",
        "accepted": {str(index): {"task": verdicts["episodes"][str(index)].get("task") or candidates[index]["annotated_task"], "source_task": candidates[index]["task"], "outcome": verdicts["episodes"][str(index)].get("outcome", "success"), "notes": verdicts["episodes"][str(index)].get("notes", "")} for index in accepted},
        "rejections": {str(k): v.get("notes", "") for k, v in verdicts["episodes"].items() if not v.get("accept")},
    }
    write_json(paths.review_root / "selection.json", selection)
    manifest = resolve_lerobot_payload(paths.spec, read_json(paths.audit), paths.metadata_root, accepted)
    manifest["source_format"] = "lerobot_v3"
    write_json(paths.review_root / "acquisition_manifest.json", manifest)
    index = read_json(paths.index) if paths.index.is_file() else {"dataset": "diverse_robot_dataset_v2", "source": "molmoact", "components": []}
    entry = {
        "component": paths.component,
        "spec_name": paths.spec.name,
        "config_name": SOURCE_CONFIG.name,
        "embodiment": "Franka",
        "status": "validated",
        "source_episodes": len(accepted),
        "dataset_root": f"review/{paths.component}",
        "manifest_path": f"review/{paths.component}/acquisition_manifest.json",
        "selection_path": f"review/{paths.component}/selection.json",
        "annotations_root": f"review/{paths.component}",
        "review_round": "v2",
        "quality_values": sorted({int(v["quality"]) for v in report.values()}),
        "mistake_events": sum(v["mistake_events"] for v in report.values()),
    }
    index["components"] = [item for item in index["components"] if item["component"] != paths.component] + [entry]
    write_json(paths.index, index)
    return {"accepted": len(accepted), "rejected": len(selection["rejected_candidate_indices"]), "episodes": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, choices=["molmoact_household", "molmoact_tabletop"])
    parser.add_argument("--build-root", type=Path, default=Path("outputs/diverse_robot_dataset_v2_build"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    nomination = sub.add_parser("nominate")
    nomination.add_argument("--per-task", type=int, default=2)
    nomination.add_argument("--prior", type=Path, help="earlier round's candidates.json: its nominees are excluded and carried into the union")
    sheets = sub.add_parser("proxies")
    sheets.add_argument("--round", help="render only candidates stamped with this round (e.g. v3)")
    sheets.add_argument("--sheets-dir", type=Path, help="write sheets, batches.json and a candidates.json copy here instead of the review dir")
    final = sub.add_parser("finalize")
    final.add_argument("--verdicts", type=Path, required=True)
    final.add_argument("--review-prompt", default="ReBot rubric: quality 1-5 over the whole episode; mistakes only as bounded failed_close/slip/drop/knock/wrong_target events; reject on uncontrolled motion, human intervention, severe occlusion, or an unverifiable outcome; unclear = reject.")
    args = parser.parse_args()
    paths = Paths(args.spec, args.build_root)
    if args.command == "scan":
        value = scan(paths)
        print(json.dumps({k: v for k, v in value.items() if k != "summaries"}, indent=2))
    elif args.command == "nominate":
        value = nominate(paths, args.per_task, args.prior)
        print(json.dumps({k: v for k, v in value.items() if k != "candidates"}, indent=2))
    elif args.command == "proxies":
        written = proxies(paths, args.round, args.sheets_dir)
        print(f"{len(written)} sheets under {paths.review_root if args.sheets_dir is None else args.sheets_dir}")
    else:
        print(json.dumps(finalize(paths, args.verdicts, args.review_prompt), indent=2))


if __name__ == "__main__":
    main()
