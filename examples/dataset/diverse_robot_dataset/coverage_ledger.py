#!/usr/bin/env python

"""Summarize diversity coverage across validated production components.

Section 13 requires a coverage review before allocating further episodes. This
reads only validated component metadata plus whatever source descriptors are
still in staging, and reports what the collection actually covers: tasks, skill
tags, embodiments, scene groups, action semantics, annotation state, chunk yield,
and storage. It makes no selection decisions.

It accepts one or more component roots, so a single ledger can span sources with
different shapes. RoboChallenge components are keyed by task and grouped by
physical `robot_id`; DROID components are keyed by lab and grouped by scene
family plus operator, and carry reviewed task strings, outcomes, and mistake
spans that RoboChallenge's auto-derived annotations do not. Everything
source-specific is read from the component metadata rather than assumed, and the
gap list at the end is computed from what the components contain, never
hardcoded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REJECTED_KEEP_REASONS = ("informative_mistake", "recovery", "task_required_hold")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def directory_bytes(root: Path) -> int:
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def component_name(component: dict) -> str:
    """RoboChallenge components are named by task, DROID components by lab."""
    return str(component.get("task") or component.get("component"))


def group_counts(nomination: dict, accepted: set[int]) -> tuple[str, dict[str, int]]:
    """Count accepted episodes per diversity group using whichever key the source has.

    RoboChallenge balances across physical robot instances. DROID has no
    `robot_id`, so `prepare_droid.py` balances across scene family plus operator
    and records the realized split as `group_key`.
    """
    candidates = {int(item["episode_index"]): item for item in nomination["candidates"]}
    for field, label in (("robot_id", "physical_robot"), ("group_key", "scene_family_operator")):
        if any(field in item for item in candidates.values()):
            counts = Counter(
                str(candidates[index][field]) for index in accepted if index in candidates
            )
            return label, dict(sorted(counts.items()))
    return "ungrouped", {}


def annotation_coverage(dataset_root: Path) -> dict:
    """Aggregate the human-facing annotation channels actually present.

    RoboChallenge annotations are auto-derived and carry neither a reviewed task
    nor an outcome, so every field here degrades to empty rather than failing.
    """
    quality: Counter = Counter()
    keep_reasons: Counter = Counter()
    outcomes: Counter = Counter()
    tasks: list[str] = []
    mistake_events = 0
    mistake_seconds = 0.0
    recovered_spans = 0
    for path in sorted((dataset_root / "meta/annotations").glob("*.json")):
        annotation = read_json(path)
        if str(annotation.get("task", "")).strip():
            tasks.append(str(annotation["task"]))
        if str(annotation.get("outcome", "")).strip():
            outcomes[str(annotation["outcome"])] += 1
        for segment in annotation["segments"]:
            if segment["retention"] != "keep":
                continue
            quality[int(segment["quality"])] += 1
            keep_reasons[str(segment["retention_reason"])] += 1
            # annotation_rubric.md defines quality 2 as exactly one failure event recovered
            # inside the segment, so a quality-2 keep span is a recovered span whichever
            # reason string it carries. DROID Failure filed these under `recovery` and DROID
            # Success under `informative_mistake`; counting both keeps the recovery coverage
            # figure comparable across components.
            if str(segment["retention_reason"]) == "recovery" or int(segment["quality"]) == 2:
                recovered_spans += 1
            for event in segment.get("mistake_events", []):
                mistake_events += 1
                mistake_seconds += float(event["end_s"]) - float(event["start_s"])
    return {
        "reviewed_tasks": tasks,
        "outcomes": dict(sorted(outcomes.items())),
        "quality_segments": {str(key): value for key, value in sorted(quality.items())},
        "keep_reasons": dict(sorted(keep_reasons.items())),
        "mistake_events": mistake_events,
        "mistake_seconds": round(mistake_seconds, 2),
        "recovered_spans": recovered_spans,
        "human_labelled": bool(tasks or outcomes or mistake_events or len(quality) > 1),
    }


def component_record(root: Path, component: dict, source: str) -> dict:
    dataset_root = root / component["dataset_root"]
    extraction = read_json(dataset_root / "meta/extraction_report.json")
    validation = read_json(dataset_root / "meta/validation_report.json")
    info = read_json(dataset_root / "meta/info.json")
    extension = read_json(dataset_root / "meta/packed_extension.json")
    nomination = read_json(dataset_root / "meta/candidate_nomination.json")
    selection = read_json(dataset_root / "meta/source_selection.json")

    name = component_name(component)
    accepted = set(selection["accepted_episode_indices"])
    group_key, groups = group_counts(nomination, accepted)
    cameras = sorted(
        {key.rsplit(".", 1)[0] for key in info["features"] if key.startswith("observation.images.")}
    )
    shape = next(
        (
            info["features"][key]["shape"]
            for key in info["features"]
            if key.startswith("observation.images.")
        ),
        None,
    )
    chunks = [episode["packed_samples"] for episode in extraction["episodes"]]
    rejections: Counter = Counter()
    candidates = 0
    for episode in extraction["episodes"]:
        candidates += episode["candidate_anchors"]
        rejections.update(episode.get("rejection_reasons", {}))

    # RoboChallenge keeps a raw task descriptor in staging; other sources do not.
    task_info_path = root / "staging/raw" / name / name / "meta/task_info.json"
    tags: list[str] = []
    prompt = nomination.get("prompt", "")
    if task_info_path.is_file():
        descriptor = read_json(task_info_path)["task_desc"]
        # task_tag mixes skill descriptors with arm count and embodiment; keep only skills.
        tags = [
            tag
            for tag in descriptor["task_tag"]
            if tag not in {"single-arm", "dual-arm"} and tag != component["embodiment"]
        ]
        prompt = descriptor["prompt"]

    specification = extension["source_spec"]
    return {
        "source": source,
        "component": name,
        "embodiment": component["embodiment"],
        "robot_type": specification.get("robot_type"),
        "prompt": prompt,
        "skill_tags": tags,
        "source_episodes": len(chunks),
        "group_key": group_key,
        "groups": groups,
        "candidate_anchors": candidates,
        "retained_chunks": int(validation["rows"]),
        "retention": round(int(validation["rows"]) / candidates, 4) if candidates else None,
        "chunks_per_episode": {
            "min": min(chunks),
            "median": sorted(chunks)[len(chunks) // 2],
            "max": max(chunks),
        },
        "rejection_reasons": dict(rejections),
        "cameras": [name.rsplit(".", 1)[-1] for name in cameras],
        "camera_streams": len(validation["camera_streams"]),
        "frame_shape": shape,
        "state_dimension": validation["state_dimension"],
        "action_dimension": validation["action_dimension"],
        "action_source": specification.get("action_source"),
        "physical_action_dtype": specification.get("physical_action_dtype"),
        "interpolated_action_points": int(validation.get("interpolated_action_points", 0)),
        "depth_streams": sum(
            1
            for key, feature in info["features"].items()
            if feature.get("info", {}).get("video.is_depth_map", False)
        ),
        "annotation": annotation_coverage(dataset_root),
        "bytes": directory_bytes(dataset_root),
    }


def collect(roots: list[Path]) -> list[dict]:
    records: list[dict] = []
    for root in roots:
        index = read_json(root / "index.json")
        for component in index["components"]:
            records.append(component_record(root, component, index["source"]))
    records.sort(key=lambda item: (item["source"], item["embodiment"], item["component"]))
    return records


def derive_gaps(records: list[dict]) -> dict:
    """State what the collection lacks, computed from the components themselves."""
    embodiments = sorted({record["embodiment"] for record in records})
    action_sources = sorted({record["action_source"] for record in records})
    dimensions = sorted({record["action_dimension"] for record in records})
    depth = sum(record["depth_streams"] for record in records)
    mistakes = sum(record["annotation"]["mistake_events"] for record in records)
    recovery = sum(record["annotation"]["recovered_spans"] for record in records)
    unlabelled = [
        record["component"] for record in records if not record["annotation"]["human_labelled"]
    ]
    absent_outcomes = []
    if not mistakes:
        absent_outcomes.append("annotated mistake events")
    if not recovery:
        absent_outcomes.append("explicit recovery spans")
    return {
        "note": (
            "Computed from validated component metadata. These are the inputs to allocating "
            "the next source episodes."
        ),
        "embodiments_present": embodiments,
        "action_dimensions_present": dimensions,
        "action_sources_present": action_sources,
        "components_without_native_actions": [
            record["component"] for record in records if record["action_source"] != "native"
        ],
        "depth_streams": depth,
        "absent_signals": ["depth"] if not depth else [],
        "absent_outcomes": absent_outcomes,
        "components_with_auto_derived_annotations": unlabelled,
        "components_without_mistake_events": [
            record["component"] for record in records if not record["annotation"]["mistake_events"]
        ],
    }


def build(roots: list[Path], output: Path) -> dict:
    records = collect(roots)
    sources: Counter = Counter()
    groups: Counter = Counter()
    tags: Counter = Counter()
    embodiments: Counter = Counter()
    rejections: Counter = Counter()
    quality: Counter = Counter()
    keep_reasons: Counter = Counter()
    outcomes: Counter = Counter()
    for record in records:
        sources[record["source"]] += record["source_episodes"]
        groups.update(record["groups"])
        tags.update(record["skill_tags"])
        embodiments[record["embodiment"]] += record["source_episodes"]
        rejections.update(record["rejection_reasons"])
        quality.update(record["annotation"]["quality_segments"])
        keep_reasons.update(record["annotation"]["keep_reasons"])
        outcomes.update(record["annotation"]["outcomes"])

    total_candidates = sum(record["candidate_anchors"] for record in records)
    total_chunks = sum(record["retained_chunks"] for record in records)
    ledger = {
        "roots": [str(root) for root in roots],
        "components": records,
        "totals": {
            "sources": len(sources),
            "components": len(records),
            "source_episodes": sum(record["source_episodes"] for record in records),
            "candidate_anchors": total_candidates,
            "retained_chunks": total_chunks,
            "retention": round(total_chunks / total_candidates, 4) if total_candidates else None,
            "interpolated_action_points": sum(
                record["interpolated_action_points"] for record in records
            ),
            "mistake_events": sum(record["annotation"]["mistake_events"] for record in records),
            "recovered_spans": sum(record["annotation"]["recovered_spans"] for record in records),
            "bytes": sum(record["bytes"] for record in records),
        },
        "coverage": {
            "source_episodes": dict(sorted(sources.items())),
            "embodiment_episodes": dict(sorted(embodiments.items())),
            "group_episodes": dict(sorted(groups.items())),
            "skill_tag_components": dict(sorted(tags.items())),
            "rejection_reasons": dict(sorted(rejections.items())),
            "quality_segments": dict(sorted(quality.items())),
            "keep_reasons": dict(sorted(keep_reasons.items())),
            "episode_outcomes": dict(sorted(outcomes.items())),
            "distinct_components": sorted(record["component"] for record in records),
            "reviewed_tasks": sorted(
                task for record in records for task in record["annotation"]["reviewed_tasks"]
            ),
        },
        "gaps": derive_gaps(records),
    }
    write_json(output, ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-root",
        type=Path,
        action="append",
        dest="component_roots",
        help="component root holding index.json; repeat for a combined ledger",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    roots = args.component_roots or [Path("outputs/diverse_robot_dataset/robochallenge")]
    output = args.output or (
        roots[0] / "coverage_ledger.json"
        if len(roots) == 1
        else roots[0].parent / "coverage_ledger.json"
    )
    ledger = build(roots, output)
    totals = ledger["totals"]
    # Lab names repeat across the two DROID splits, so the source has to be on every row.
    print(
        f"{'source':<16} {'component':<28} {'emb':<7} {'eps':>4} {'chunks':>7} "
        f"{'keep%':>6} {'mist':>5} {'rec':>4}"
    )
    for record in ledger["components"]:
        source = record["source"].split("/")[-1]
        print(
            f"{source:<16} {record['component']:<28} {record['embodiment']:<7} "
            f"{record['source_episodes']:>4} {record['retained_chunks']:>7} "
            f"{record['retention'] * 100:>5.1f}% {record['annotation']['mistake_events']:>5} "
            f"{record['annotation']['recovered_spans']:>4}"
        )
    print(
        f"\n{totals['components']} components across {totals['sources']} sources, "
        f"{totals['source_episodes']} episodes, {totals['retained_chunks']} chunks, "
        f"{totals['retention'] * 100:.1f}% retention, {totals['bytes'] / 1e6:.0f} MB"
    )
    print("embodiments:", ledger["coverage"]["embodiment_episodes"])
    print("quality segments:", ledger["coverage"]["quality_segments"])
    print("keep reasons:", ledger["coverage"]["keep_reasons"])
    print("outcomes:", ledger["coverage"]["episode_outcomes"])
    print("mistake events:", totals["mistake_events"], "| recovered spans:", totals["recovered_spans"])
    print("gaps:", json.dumps(ledger["gaps"], indent=2))
    print("written to", output)


if __name__ == "__main__":
    main()
