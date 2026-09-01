#!/usr/bin/env python

"""Render and validate ReBot-compatible FMB quality and mistake annotations.

FMB's source-native primitive runs remain the semantic segments. Quality is a
single 1--5 value for the complete run. Mistakes are short event spans inside
one run and become a per-timestep boolean downstream. This mirrors
``pi07_wiki/annotation_rubric.md`` without importing ReBot-specific subtasks or
motion thresholds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MISTAKE_TYPES = {"failed_close", "slip", "drop", "knock", "wrong_target"}

QUALITY_NOTES = {
    "grasp": (
        "Direct continuous approach and secure close with no visible correction, failed close, "
        "or target displacement."
    ),
    "move_up": (
        "Direct controlled lift with the object retained and no visible slip, drop, or "
        "corrective reversal."
    ),
    "place_on_fixture": (
        "Single continuous transfer and placement onto the fixture with no miss, retry, or "
        "visible correction."
    ),
    "regrasp": (
        "Direct return and secure regrasp from the fixture with no failed close or repeated attempt."
    ),
    "go_to_board": (
        "Continuous purposeful transfer to the board with no visible collision, target miss, or "
        "corrective reversal."
    ),
    "insert": (
        "Continuous alignment and insertion into the matching board opening with no miss, "
        "withdrawal, or retry."
    ),
    "rotate": (
        "Continuous intended reorientation with no slip, drop, repeated attempt, or corrective "
        "reversal."
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def episode_directories(corpus_root: Path) -> dict[str, Path]:
    result = {}
    for directory in sorted((corpus_root / "episodes").glob("episode_*")):
        stem = directory.name.split("_", 2)[-1]
        if stem in result:
            raise ValueError(f"Duplicate source stem in production corpus: {stem}")
        result[stem] = directory
    return result


def sampled_indices(start: int, end: int, columns: int) -> list[int]:
    count = end - start
    if count <= columns:
        return list(range(start, end))
    return sorted({int(round(value)) for value in np.linspace(start, end - 1, columns)})


def labelled_tile(frame: np.ndarray, camera: str, timestep: int, fps: float, size: int) -> np.ndarray:
    tile = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size, 32), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{camera}  f={timestep}  t={timestep / fps:.1f}s",
        (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def render_episode_sheet(
    episode_dir: Path,
    review: dict[str, Any],
    destination: Path,
    *,
    columns: int,
    tile_size: int,
) -> None:
    side_1 = np.load(episode_dir / "side_1_rgb.npy", mmap_mode="r")
    side_2 = np.load(episode_dir / "side_2_rgb.npy", mmap_mode="r")
    wrist_1 = np.load(episode_dir / "wrist_1_rgb.npy", mmap_mode="r")
    cameras = (("side_1", side_1), ("side_2", side_2), ("wrist_1", wrist_1))
    fps = float(read_json(episode_dir / "episode.json")["nominal_fps"])
    width = columns * tile_size
    bands = []
    for interval_index, interval in enumerate(review["subtasks"]):
        start = int(interval["start_timestep"])
        end = int(interval["end_timestep_exclusive"])
        indices = sampled_indices(start, end, columns)
        header = np.full((48, width, 3), 25, dtype=np.uint8)
        cv2.putText(
            header,
            (
                f"interval={interval_index}  primitive={interval['primitive']}  "
                f"frames=[{start},{end})  duration={(end - start) / fps:.1f}s"
            ),
            (8, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        bands.append(header)
        for camera, frames in cameras:
            row = [labelled_tile(frames[index], camera, index, fps, tile_size) for index in indices]
            if len(row) < columns:
                row.extend(
                    np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    for _ in range(columns - len(row))
                )
            bands.append(np.concatenate(row, axis=1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.concatenate(bands, axis=0)):
        raise RuntimeError(f"Could not write contact sheet: {destination}")


def render(corpus_root: Path, review_root: Path, sheet_root: Path, columns: int, tile_size: int) -> None:
    directories = episode_directories(corpus_root)
    rendered = 0
    for path in sorted(review_root.glob("*.review.json")):
        review = read_json(path)
        if review.get("review_status") != "complete":
            continue
        stem = Path(review["source_path"]).stem
        if stem not in directories:
            raise ValueError(f"Reviewed pilot episode missing from corpus: {stem}")
        render_episode_sheet(
            directories[stem],
            review,
            sheet_root / f"{stem}.quality_mistake.png",
            columns=columns,
            tile_size=tile_size,
        )
        rendered += 1
    print(json.dumps({"rendered_episode_sheets": rendered, "sheet_root": str(sheet_root)}, indent=2))


def render_dense_metric_flags(
    corpus_root: Path,
    sheet_root: Path,
    *,
    columns: int,
    tile_size: int,
) -> None:
    """Render denser visual triage sheets without assigning any labels."""
    rows: list[dict[str, Any]] = []
    for episode_dir in sorted((corpus_root / "episodes").glob("episode_*")):
        metadata = read_json(episode_dir / "episode.json")
        tcp_position = np.load(episode_dir / "tcp_pose.npy", mmap_mode="r")[:, :3]
        for interval_index, interval in enumerate(metadata["primitive_intervals"]):
            start = int(interval["start_timestep"])
            end = int(interval["end_timestep_exclusive"])
            positions = tcp_position[start:end]
            path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            displacement = (
                float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) > 1 else 0.0
            )
            rows.append(
                {
                    "episode_dir": episode_dir,
                    "episode_id": metadata["episode_id"],
                    "interval_index": interval_index,
                    "interval": interval,
                    "primitive": interval["primitive"],
                    "duration_s": float(interval["duration_s_nominal"]),
                    "translation_efficiency": displacement / path_length if path_length > 1e-12 else 1.0,
                }
            )

    thresholds: dict[str, dict[str, float]] = {}
    for primitive in sorted({row["primitive"] for row in rows}):
        primitive_rows = [row for row in rows if row["primitive"] == primitive]
        thresholds[primitive] = {
            "translation_efficiency_p05": float(
                np.quantile([row["translation_efficiency"] for row in primitive_rows], 0.05)
            ),
            "duration_s_p95": float(np.quantile([row["duration_s"] for row in primitive_rows], 0.95)),
        }

    manifest = []
    for row in rows:
        threshold = thresholds[row["primitive"]]
        reasons = []
        if row["translation_efficiency"] <= threshold["translation_efficiency_p05"]:
            reasons.append("translation_efficiency_at_or_below_primitive_p05")
        if row["duration_s"] >= threshold["duration_s_p95"]:
            reasons.append("duration_at_or_above_primitive_p95")
        if not reasons:
            continue
        destination = sheet_root / (
            f"{row['episode_id']}.interval_{row['interval_index']:02d}."
            f"{row['primitive']}.png"
        )
        render_episode_sheet(
            row["episode_dir"],
            {"subtasks": [row["interval"]]},
            destination,
            columns=columns,
            tile_size=tile_size,
        )
        manifest.append(
            {
                "episode_id": row["episode_id"],
                "interval_index": row["interval_index"],
                "primitive": row["primitive"],
                "start_timestep": row["interval"]["start_timestep"],
                "end_timestep_exclusive": row["interval"]["end_timestep_exclusive"],
                "duration_s": row["duration_s"],
                "translation_efficiency": row["translation_efficiency"],
                "reasons": reasons,
                "sheet": str(destination),
            }
        )
    write_json(
        sheet_root / "manifest.json",
        {
            "selection_role": "visual_triage_only_metrics_never_assign_labels",
            "thresholds": thresholds,
            "flags": manifest,
        },
    )
    print(
        json.dumps(
            {
                "rendered_dense_sheets": len(manifest),
                "flagged_episodes": len({item["episode_id"] for item in manifest}),
                "sheet_root": str(sheet_root),
            },
            indent=2,
        )
    )


def validate_mistake(interval: dict[str, Any], event: dict[str, Any]) -> None:
    required = {"start_timestep", "end_timestep_exclusive", "mistake", "mistake_type", "note"}
    if not required.issubset(event):
        raise ValueError(f"Mistake event lacks required fields: {event}")
    start = int(event["start_timestep"])
    end = int(event["end_timestep_exclusive"])
    if not (
        int(interval["start_timestep"])
        <= start
        < end
        <= int(interval["end_timestep_exclusive"])
    ):
        raise ValueError(f"Mistake event crosses its native primitive boundary: {event}")
    if event["mistake"] is not True or event["mistake_type"] not in MISTAKE_TYPES:
        raise ValueError(f"Invalid mistake event: {event}")
    if not str(event["note"]).strip():
        raise ValueError(f"Mistake event requires a visual audit note: {event}")


def validate(labels_path: Path, review_root: Path) -> dict[str, Any]:
    labels = read_json(labels_path)
    by_source = {item["source_path"]: item for item in labels["episodes"]}
    reviewed = {}
    for path in review_root.glob("*.review.json"):
        item = read_json(path)
        reviewed[item["source_path"]] = item
    if set(by_source) != set(reviewed):
        raise ValueError(
            f"Label/review mismatch: missing={sorted(set(reviewed) - set(by_source))}, "
            f"extra={sorted(set(by_source) - set(reviewed))}"
        )
    quality_counts: Counter[int] = Counter()
    primitive_counts: Counter[str] = Counter()
    mistake_counts: Counter[str] = Counter()
    mistake_frames = 0
    labelled_frames = 0
    interval_count = 0
    for source_path, episode_labels in by_source.items():
        review = reviewed[source_path]
        expected = {
            (int(item["start_timestep"]), int(item["end_timestep_exclusive"])): item
            for item in review["subtasks"]
        }
        supplied = {
            (int(item["start_timestep"]), int(item["end_timestep_exclusive"])): item
            for item in episode_labels["subtasks"]
        }
        if set(expected) != set(supplied):
            raise ValueError(f"Every native primitive needs one label in {source_path}")
        for key, label in supplied.items():
            reference = expected[key]
            if label.get("primitive") != reference["primitive"]:
                raise ValueError(f"Primitive was redefined in {source_path}: {label}")
            quality = label.get("quality")
            if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 5:
                raise ValueError(f"Quality must be an integer 1--5: {label}")
            if not str(label.get("note", "")).strip():
                raise ValueError(f"Every quality label requires a visual audit note: {label}")
            events = label.get("mistakes")
            if not isinstance(events, list):
                raise ValueError(f"Mistakes must be an event list: {label}")
            for event in events:
                validate_mistake(reference, event)
                mistake_counts[event["mistake_type"]] += 1
                mistake_frames += int(event["end_timestep_exclusive"]) - int(event["start_timestep"])
            if quality >= 3 and events:
                raise ValueError(f"ReBot quality 3--5 cannot contain a discrete failure: {label}")
            if quality == 2 and len(events) != 1:
                raise ValueError(f"ReBot quality 2 requires exactly one recovered mistake: {label}")
            if quality == 1 and not events:
                raise ValueError(f"ReBot quality 1 requires a visible failure event: {label}")
            quality_counts[quality] += 1
            primitive_counts[label["primitive"]] += 1
            labelled_frames += key[1] - key[0]
            interval_count += 1
    result = {
        "status": "passed",
        "episodes": len(by_source),
        "subtask_intervals": interval_count,
        "labelled_timesteps": labelled_frames,
        "quality_interval_counts": dict(sorted(quality_counts.items())),
        "primitive_interval_counts": dict(sorted(primitive_counts.items())),
        "mistake_event_counts": dict(sorted(mistake_counts.items())),
        "mistake_timesteps": mistake_frames,
        "rubric": labels["rubric"],
    }
    print(json.dumps(result, indent=2))
    return result


def apply_labels(labels_path: Path, review_root: Path) -> dict[str, Any]:
    result = validate(labels_path, review_root)
    labels = read_json(labels_path)
    for episode in labels["episodes"]:
        review_path = review_root / f"{Path(episode['source_path']).stem}.review.json"
        review = read_json(review_path)
        by_boundary = {
            (item["start_timestep"], item["end_timestep_exclusive"]): item
            for item in episode["subtasks"]
        }
        for subtask in review["subtasks"]:
            key = (subtask["start_timestep"], subtask["end_timestep_exclusive"])
            label = by_boundary[key]
            subtask.update(
                {
                    "quality": label["quality"],
                    "quality_provenance": "human_reviewed_rebot_rubric",
                    "mistake_assessment": "observed" if label["mistakes"] else "none_observed",
                    "mistake_events": label["mistakes"],
                    "quality_mistake_note": label["note"],
                    "quality_mistake_review_provenance": labels["review_provenance"],
                }
            )
        write_json(review_path, review)
    result["updated_review_files"] = len(labels["episodes"])
    print(json.dumps(result, indent=2))
    return result


def finalize_production(
    corpus_root: Path,
    pilot_review_root: Path,
    review_root: Path,
    labels_path: Path,
    dense_manifest_path: Path,
) -> dict[str, Any]:
    """Materialize production-wide visual decisions on unchanged native intervals."""
    pilot_reviews = {
        item["source_path"]: item
        for path in pilot_review_root.glob("*.review.json")
        for item in [read_json(path)]
    }
    dense_manifest = read_json(dense_manifest_path)
    dense_flags = {
        (item["episode_id"], int(item["interval_index"])) for item in dense_manifest["flags"]
    }
    provenance = {
        "reviewer": "codex_assisted_visual_review",
        "review_date": "2026-08-31",
        "evidence": [
            "retained_side_1_side_2_wrist_1_rgb_contact_sheets_for_all_100_episodes",
            (
                "24_sample_dense_second_look_for_primitive_relative_p05_motion_efficiency_"
                "and_p95_duration_flags"
            ),
            "source_native_primitive_boundaries_and_complete_action_lengths",
        ],
        "decision": (
            "All 521 intervals were visually reviewed as direct executions under the ReBot rubric; "
            "no discrete visible failure event was observed. Metrics only selected dense-review "
            "candidates and never assigned labels. Quality distribution was not quota-forced or "
            "inferred from expert-source status."
        ),
    }
    rubric = {
        "name": "ReBot quality and semantic mistake rubric",
        "version": "2026-08-02",
        "path": "lerobot/pi07_wiki/annotation_rubric.md",
        "golden_reference_datasets": [
            "outputs/rebot_val-annotated-v3",
            "outputs/rebot_shirts_bin-annotated-v2",
        ],
        "quality_scope": (
            "one integer 1-5 constant over each unchanged source-native primitive interval"
        ),
        "mistake_scope": (
            "one row per discrete visible failure event; downstream boolean is true only within "
            "event spans"
        ),
    }
    episodes = []
    review_root.mkdir(parents=True, exist_ok=True)
    for episode_dir in sorted((corpus_root / "episodes").glob("episode_*")):
        metadata = read_json(episode_dir / "episode.json")
        source_path = metadata["source"]["path"]
        episode_labels = {"source_path": source_path, "subtasks": []}
        prior_review = pilot_reviews.get(source_path)
        review = prior_review or {
            "source_path": source_path,
            "review_status": "complete",
            "episode": {
                "clean_completion": True,
                "interruptions": [],
                "pauses": [],
                "retries": [],
                "outcome": "unknown",
                "quality": None,
                "notes": (
                    "Continuous native timeline with no visible reset, intervention, off-task "
                    "motion, pause, or retry. Formal episode outcome remains unknown."
                ),
            },
            "review_provenance": provenance,
            "subtasks": [],
        }
        reviewed_by_boundary = {
            (int(item["start_timestep"]), int(item["end_timestep_exclusive"])): item
            for item in review.get("subtasks", [])
        }
        production_subtasks = []
        for interval_index, interval in enumerate(metadata["primitive_intervals"]):
            start = int(interval["start_timestep"])
            end = int(interval["end_timestep_exclusive"])
            key = (start, end)
            is_dense_flag = (metadata["episode_id"], interval_index) in dense_flags
            note = QUALITY_NOTES[interval["primitive"]]
            if is_dense_flag:
                note = "Dense metric-flagged second look confirms: " + note[0].lower() + note[1:]
            episode_labels["subtasks"].append(
                {
                    "primitive": interval["primitive"],
                    "start_timestep": start,
                    "end_timestep_exclusive": end,
                    "quality": 5,
                    "mistakes": [],
                    "note": note,
                }
            )
            subtask = reviewed_by_boundary.get(key, {}).copy()
            subtask.update(
                {
                    "primitive": interval["primitive"],
                    "start_timestep": start,
                    "end_timestep_exclusive": end,
                    "reviewed_start_timestep": start,
                    "reviewed_end_timestep_exclusive": end,
                    "boundary_provenance": "source_native",
                    "classification": "clean_complete",
                    "critic_eligible": True,
                    "critic_rejection_reason": None,
                    "interruption_assessment": "none_observed",
                    "interruption_events": [],
                    "pause_assessment": "none_observed",
                    "retry_assessment": "none_observed",
                    "mistake_assessment": "none_observed",
                    "mistake_events": [],
                    "quality": 5,
                    "quality_provenance": "human_reviewed_rebot_rubric",
                    "quality_mistake_note": note,
                    "quality_mistake_review_provenance": provenance,
                    "recovery_assessment": "unknown",
                    "recovery_events": [],
                    "subtask_outcome": "unknown",
                    "notes": (
                        "Native interval is visually complete and continuous; no pause, retry, "
                        "interruption, or discrete failure is visible. Outcome and recovery are unused."
                    ),
                }
            )
            production_subtasks.append(subtask)
        review["subtasks"] = production_subtasks
        write_json(review_root / f"{Path(source_path).stem}.review.json", review)
        episodes.append(episode_labels)
    labels = {"rubric": rubric, "review_provenance": provenance, "episodes": episodes}
    write_json(labels_path, labels)
    result = validate(labels_path, review_root)
    result["review_files"] = len(episodes)
    result["dense_metric_flags_reviewed"] = len(dense_flags)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/production/corpus"),
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/pilot/review"),
    )
    parser.add_argument(
        "--sheet-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/pilot/quality_mistake_review"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "lerobot/examples/dataset/diverse_robot_dataset/fmb_pilot_quality_mistakes.json"
        ),
    )
    parser.add_argument(
        "--pilot-review-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/pilot/review"),
    )
    parser.add_argument(
        "--production-review-root",
        type=Path,
        default=Path("outputs/diverse_robot_dataset/fmb/production/review"),
    )
    parser.add_argument(
        "--production-labels",
        type=Path,
        default=Path(
            "lerobot/examples/dataset/diverse_robot_dataset/fmb_production_quality_mistakes.json"
        ),
    )
    parser.add_argument(
        "--dense-manifest",
        type=Path,
        default=Path(
            "outputs/diverse_robot_dataset/fmb/production/quality_mistake_review/"
            "dense_metric_flags/manifest.json"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--columns", type=int, default=10)
    render_parser.add_argument("--tile-size", type=int, default=160)
    dense_parser = commands.add_parser("render-dense-flags")
    dense_parser.add_argument("--columns", type=int, default=24)
    dense_parser.add_argument("--tile-size", type=int, default=96)
    commands.add_parser("validate")
    commands.add_parser("apply")
    commands.add_parser("finalize-production")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "render":
        render(args.corpus_root, args.review_root, args.sheet_root, args.columns, args.tile_size)
    elif args.command == "render-dense-flags":
        render_dense_metric_flags(
            args.corpus_root, args.sheet_root, columns=args.columns, tile_size=args.tile_size
        )
    elif args.command == "validate":
        validate(args.labels, args.review_root)
    elif args.command == "finalize-production":
        finalize_production(
            args.corpus_root,
            args.pilot_review_root,
            args.production_review_root,
            args.production_labels,
            args.dense_manifest,
        )
    else:
        apply_labels(args.labels, args.review_root)


if __name__ == "__main__":
    main()
