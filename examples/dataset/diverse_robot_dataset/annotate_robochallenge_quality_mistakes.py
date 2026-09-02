#!/usr/bin/env python

"""Render, validate, and apply RoboChallenge quality/mistake review labels.

RoboChallenge's original preparation path assigns quality 5 and an empty mistake
list to every motion span automatically.  This tool replaces those defaults with
segment-level visual decisions made under ``pi07_wiki/annotation_rubric.md``.

The review artifact is kept outside the training corpus.  ``apply`` first saves
the original corpus and source annotation metadata, then updates both copies.  The ordinary ``build_corpus.py views`` command must
be run afterwards to regenerate actor and critic indexes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MISTAKE_TYPES = {"failed_close", "slip", "drop", "knock", "wrong_target"}
REVIEW_PROVENANCE = "model_reviewed_rebot_rubric"
REVIEW_DATE = "2026-09-02"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def robochallenge_episodes(corpus_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    episodes = []
    for path in sorted((corpus_root / "episodes").glob("robochallenge__*/episode.json")):
        episode = read_json(path)
        if episode.get("source") == "robochallenge":
            episodes.append((path.parent, episode))
    return episodes


def sampled_frames(start_s: float, end_s: float, fps: float, columns: int) -> list[int]:
    start = max(0, int(np.floor(start_s * fps)))
    end = max(start + 1, int(np.ceil(end_s * fps)))
    if end - start <= columns:
        return list(range(start, end))
    return sorted({int(round(value)) for value in np.linspace(start, end - 1, columns)})


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open review video: {path}")
    result = {}
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index} from {path}")
        result[index] = frame
    capture.release()
    return result


def labelled_tile(
    frame: np.ndarray,
    camera: str,
    frame_index: int,
    fps: float,
    width: int,
    height: int,
) -> np.ndarray:
    tile = _letterbox(frame, width, height)
    cv2.rectangle(tile, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{camera}  f={frame_index}  t={frame_index / fps:.1f}s",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def render_episode_sheet(
    episode_dir: Path,
    episode: dict[str, Any],
    destination: Path,
    *,
    columns: int,
    tile_width: int,
    tile_height: int,
) -> None:
    fps = float(episode["native_rate_hz"])
    keep_segments = [
        (index, segment)
        for index, segment in enumerate(episode["annotations"]["segments"])
        if segment["retention"] == "keep"
    ]
    camera_paths = {
        camera["name"]: episode_dir / camera["path"] for camera in episode["cameras"]
    }
    all_indices = sorted(
        {
            frame
            for _, segment in keep_segments
            for frame in sampled_frames(segment["start_s"], segment["end_s"], fps, columns)
        }
    )
    frames = {name: video_frames(path, all_indices) for name, path in camera_paths.items()}
    width = columns * tile_width
    bands = []
    title = np.full((54, width, 3), 20, dtype=np.uint8)
    cv2.putText(
        title,
        f"{episode['episode_id']}  task={episode['task']}",
        (8, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    bands.append(title)
    for segment_index, segment in keep_segments:
        indices = sampled_frames(segment["start_s"], segment["end_s"], fps, columns)
        header = np.full((42, width, 3), 30, dtype=np.uint8)
        cv2.putText(
            header,
            (
                f"segment={segment_index}  [{segment['start_s']:.2f},{segment['end_s']:.2f})s  "
                f"duration={segment['end_s'] - segment['start_s']:.2f}s"
            ),
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        bands.append(header)
        for camera in ("global", "side", "wrist"):
            if camera not in frames:
                continue
            row = [
                labelled_tile(
                    frames[camera][index], camera, index, fps, tile_width, tile_height
                )
                for index in indices
            ]
            row.extend(
                np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                for _ in range(columns - len(row))
            )
            bands.append(np.concatenate(row, axis=1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.concatenate(bands, axis=0)):
        raise RuntimeError(f"Could not write contact sheet: {destination}")


def initialize_labels(corpus_root: Path, labels_path: Path) -> dict[str, Any]:
    episodes = []
    for _, episode in robochallenge_episodes(corpus_root):
        subtasks = []
        for segment_index, segment in enumerate(episode["annotations"]["segments"]):
            if segment["retention"] != "keep":
                continue
            subtasks.append(
                {
                    "segment_index": segment_index,
                    "start_s": segment["start_s"],
                    "end_s": segment["end_s"],
                    "quality": None,
                    "mistakes": [],
                    "note": "",
                }
            )
        episodes.append({"episode_id": episode["episode_id"], "subtasks": subtasks})
    labels = {
        "rubric": {
            "name": "ReBot quality and semantic mistake rubric",
            "version": "2026-08-02",
            "path": "lerobot/pi07_wiki/annotation_rubric.md",
            "quality_scope": "one integer 1-5 constant over each retained RoboChallenge segment",
            "mistake_scope": "one discrete visible failure event per bounded time span",
        },
        "review_provenance": {
            "reviewer": "codex_assisted_visual_review",
            "review_date": REVIEW_DATE,
            "review_basis": [
                "timestamped_global_side_wrist_contact_sheets_for_every_retained_segment",
                "dense_followup_review_for_ambiguous_corrections_and_failure_events",
                "source_native_state_and_gripper_traces_used_for_visual_triage_only",
            ],
            "automatic_labels_used_as_verdicts": False,
        },
        "episodes": episodes,
    }
    write_json(labels_path, labels)
    return {"episodes": len(episodes), "segments": sum(len(item["subtasks"]) for item in episodes)}


def render(
    corpus_root: Path,
    sheet_root: Path,
    *,
    columns: int,
    tile_width: int,
    tile_height: int,
) -> dict[str, Any]:
    episodes = robochallenge_episodes(corpus_root)
    for episode_dir, episode in episodes:
        render_episode_sheet(
            episode_dir,
            episode,
            sheet_root / episode["component"] / f"{episode['episode_id']}.png",
            columns=columns,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    result = {"rendered_episode_sheets": len(episodes), "sheet_root": str(sheet_root)}
    write_json(sheet_root / "manifest.json", result)
    return result


def validate_mistake(segment: dict[str, Any], event: dict[str, Any]) -> None:
    required = {"start_s", "end_s", "mistake_type", "note"}
    if not required.issubset(event):
        raise ValueError(f"Mistake event lacks required fields: {event}")
    start = float(event["start_s"])
    end = float(event["end_s"])
    if not float(segment["start_s"]) <= start < end <= float(segment["end_s"]):
        raise ValueError(f"Mistake event crosses its retained segment: {event}")
    if event["mistake_type"] not in MISTAKE_TYPES:
        raise ValueError(f"Unknown mistake type: {event}")
    if not str(event["note"]).strip():
        raise ValueError(f"Mistake event requires a visual audit note: {event}")


def validate(labels_path: Path, corpus_root: Path) -> dict[str, Any]:
    labels = read_json(labels_path)
    supplied = {item["episode_id"]: item for item in labels["episodes"]}
    expected = {episode["episode_id"]: episode for _, episode in robochallenge_episodes(corpus_root)}
    if set(supplied) != set(expected):
        raise ValueError(
            f"Episode mismatch: missing={sorted(set(expected) - set(supplied))}, "
            f"extra={sorted(set(supplied) - set(expected))}"
        )
    quality_counts: Counter[int] = Counter()
    mistake_counts: Counter[str] = Counter()
    labelled_duration_s = 0.0
    mistake_duration_s = 0.0
    segment_count = 0
    for episode_id, episode_labels in supplied.items():
        episode = expected[episode_id]
        keep = {
            index: segment
            for index, segment in enumerate(episode["annotations"]["segments"])
            if segment["retention"] == "keep"
        }
        provided = {int(item["segment_index"]): item for item in episode_labels["subtasks"]}
        if set(provided) != set(keep):
            raise ValueError(f"Retained segment mismatch in {episode_id}")
        for segment_index, label in provided.items():
            reference = keep[segment_index]
            if not np.isclose(float(label["start_s"]), float(reference["start_s"])) or not np.isclose(
                float(label["end_s"]), float(reference["end_s"])
            ):
                raise ValueError(f"Segment boundary changed in {episode_id}: {label}")
            quality = label.get("quality")
            if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 5:
                raise ValueError(f"Quality must be an integer 1--5: {label}")
            if not str(label.get("note", "")).strip():
                raise ValueError(f"Every segment needs a visual audit note: {label}")
            mistakes = label.get("mistakes")
            if not isinstance(mistakes, list):
                raise ValueError(f"Mistakes must be an event list: {label}")
            for event in mistakes:
                validate_mistake(reference, event)
                mistake_counts[event["mistake_type"]] += 1
                mistake_duration_s += float(event["end_s"]) - float(event["start_s"])
            if quality >= 3 and mistakes:
                raise ValueError(f"Quality 3--5 cannot contain a discrete failure: {label}")
            if quality == 2 and len(mistakes) != 1:
                raise ValueError(f"Quality 2 requires exactly one recovered failure: {label}")
            if quality == 1 and not mistakes:
                raise ValueError(f"Quality 1 requires at least one failure: {label}")
            quality_counts[quality] += 1
            labelled_duration_s += float(reference["end_s"]) - float(reference["start_s"])
            segment_count += 1
    return {
        "status": "passed",
        "episodes": len(expected),
        "segments": segment_count,
        "quality_segment_counts": dict(sorted(quality_counts.items())),
        "mistake_event_counts": dict(sorted(mistake_counts.items())),
        "labelled_duration_s": labelled_duration_s,
        "mistake_duration_s": mistake_duration_s,
        "rubric": labels["rubric"],
    }


def source_annotation_path(build_root: Path, episode: dict[str, Any]) -> Path:
    root = build_root / "robochallenge" / "review" / episode["component"]
    if episode.get("review_round") == "round2":
        root /= "round2"
    return root / f"episode_{int(episode['source_episode_index']):06d}.annotations.json"


def apply_labels(
    labels_path: Path,
    corpus_root: Path,
    build_root: Path,
    backup_root: Path,
) -> dict[str, Any]:
    result = validate(labels_path, corpus_root)
    labels = read_json(labels_path)
    by_episode = {item["episode_id"]: item for item in labels["episodes"]}
    provenance = labels["review_provenance"]
    for episode_dir, episode in robochallenge_episodes(corpus_root):
        annotation_path = source_annotation_path(build_root, episode)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing source annotation: {annotation_path}")
        annotation = read_json(annotation_path)
        annotation_keep = [
            (index, item)
            for index, item in enumerate(annotation["segments"])
            if item["retention"] == "keep"
        ]
        corpus_keep = [
            (index, item)
            for index, item in enumerate(episode["annotations"]["segments"])
            if item["retention"] == "keep"
        ]
        if [(i, s["start_s"], s["end_s"]) for i, s in annotation_keep] != [
            (i, s["start_s"], s["end_s"]) for i, s in corpus_keep
        ]:
            raise ValueError("Source/corpus segment mismatch: " + str(episode["episode_id"]))
        episode_backup = backup_root / episode["episode_id"] / "episode.json"
        if not episode_backup.exists():
            episode_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(episode_dir / "episode.json", episode_backup)
        source_backup = (
            backup_root
            / "source_annotations"
            / annotation_path.relative_to(build_root / "robochallenge" / "review")
        )
        if not source_backup.exists():
            source_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(annotation_path, source_backup)
    updated = 0
    for episode_dir, episode in robochallenge_episodes(corpus_root):
        labels_by_segment = {
            int(item["segment_index"]): item for item in by_episode[episode["episode_id"]]["subtasks"]
        }
        episode_path = episode_dir / "episode.json"
        backup_path = backup_root / episode["episode_id"] / "episode.json"
        if not backup_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(episode_path, backup_path)
        for index, segment in enumerate(episode["annotations"]["segments"]):
            if segment["retention"] != "keep":
                continue
            label = labels_by_segment[index]
            segment.update(
                {
                    "quality": label["quality"],
                    "mistake_events": label["mistakes"],
                    "quality_mistake_note": label["note"],
                    "quality_mistake_review_provenance": provenance,
                }
            )
        episode["review"].update(
            {
                "quality_provenance": REVIEW_PROVENANCE,
                "quality_mistake_review_provenance": provenance,
            }
        )
        write_json(episode_path, episode)

        annotation_path = source_annotation_path(build_root, episode)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing source annotation: {annotation_path}")
        annotation = read_json(annotation_path)
        annotation_keep = [
            (index, item)
            for index, item in enumerate(annotation["segments"])
            if item["retention"] == "keep"
        ]
        corpus_keep = [
            (index, item)
            for index, item in enumerate(episode["annotations"]["segments"])
            if item["retention"] == "keep"
        ]
        if [(i, s["start_s"], s["end_s"]) for i, s in annotation_keep] != [
            (i, s["start_s"], s["end_s"]) for i, s in corpus_keep
        ]:
            raise ValueError(f"Source/corpus segment mismatch: {episode['episode_id']}")
        for index, segment in annotation_keep:
            label = labels_by_segment[index]
            segment.update(
                {
                    "quality": label["quality"],
                    "mistake_events": label["mistakes"],
                    "quality_mistake_note": label["note"],
                    "quality_mistake_review_provenance": provenance,
                }
            )
        annotation.update(
            {
                "quality_provenance": REVIEW_PROVENANCE,
                "quality_mistake_review_provenance": provenance,
            }
        )
        write_json(annotation_path, annotation)
        updated += 1
    result.update(
        {
            "updated_episode_records": updated,
            "updated_source_annotations": updated,
            "backup_root": str(backup_root),
            "views_require_regeneration": True,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root", type=Path, default=Path("outputs/diverse_robot_dataset/corpus")
    )
    parser.add_argument(
        "--build-root", type=Path, default=Path("outputs/diverse_robot_dataset_build")
    )
    parser.add_argument(
        "--sheet-root",
        type=Path,
        default=Path(
            "outputs/diverse_robot_dataset_build/robochallenge/quality_mistake_review/sheets"
        ),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).with_name("robochallenge_quality_mistakes.json"),
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path(
            "outputs/diverse_robot_dataset_build/robochallenge/quality_mistake_review/"
            "pre_review_episode_metadata"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("initialize")
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--columns", type=int, default=24)
    render_parser.add_argument("--tile-width", type=int, default=160)
    render_parser.add_argument("--tile-height", type=int, default=100)
    commands.add_parser("validate")
    commands.add_parser("apply")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "initialize":
        result = initialize_labels(args.corpus_root, args.labels)
    elif args.command == "render":
        result = render(
            args.corpus_root,
            args.sheet_root,
            columns=args.columns,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
    elif args.command == "validate":
        result = validate(args.labels, args.corpus_root)
    else:
        result = apply_labels(
            args.labels, args.corpus_root, args.build_root, args.backup_root
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
