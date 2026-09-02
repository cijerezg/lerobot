#!/usr/bin/env python

"""Drive one DROID component through the production lifecycle.

Both `droid_failure` and `droid_success` use this driver; pick with `--source`.
They share a robot, a rate, a shard layout, and a schema, and differ only in
which split they draw from and whether the source carries language.

DROID is already LeRobot v3 on the Hub, so `acquire_robochallenge.py` and
`run_robochallenge_component.py` do not apply: there is no split tar to verify
and extract, and there is no `states.jsonl`. The stages here are

    review   nominate candidates, resolve and download only their payload,
             and render labelled review sheets;
    detail   render two-second-stride sheets for a shortlist so failure and
             recovery spans can be located in source time;
    finalize compile reviewer spans into validated annotations, extract the
             packed v3 component, validate it, attach metadata, update the index.

Visual acceptance and annotation stay human decisions between `review` and
`finalize`. Unlike RoboChallenge, these sources expose native commanded joint
actions, so the `copy_state` exception must not be used, and their 15 Hz rate
means the 30 Hz interpolation path runs and must leave a non-zero interpolation
mask.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PREPARE = Path(__file__).with_name("prepare_droid.py")
SOURCE_CONFIG = Path(__file__).with_name("droid_sources.json")
DEFAULT_SOURCE = "droid_failure"
DEFAULT_COMPONENT_ROOTS = {
    "droid_failure": "outputs/diverse_robot_dataset_build/droid",
    "droid_success": "outputs/diverse_robot_dataset_build/droid_success",
}
# Plan Section 1.6 selects droid_success for recovery, so its nomination requires at least
# two gripper close events. droid_failure was selected for failures and does not screen on it.
DEFAULT_MIN_CLOSE_EVENTS = {"droid_failure": 0, "droid_success": 2}
CANDIDATE_COUNT = 24
ACCEPTANCE_TARGET = 10
ANCHOR_STRIDE_S = 2.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


class Component:
    def __init__(self, lab: str, component_root: Path, source: str) -> None:
        from lerobot.datasets.diverse_pilot import load_source_specs

        self.source = source
        self.spec = next(item for item in load_source_specs(SOURCE_CONFIG) if item.name == source)
        self.repo_id = self.spec.repo_id
        self.source_slug = source.replace("_", "-")
        self.lab = lab
        self.slug = lab.lower()
        self.root = component_root
        self.metadata_root = component_root / "metadata" / source
        self.staging_root = component_root / "staging"
        self.scan = component_root / "review/episode_scan.json"
        self.review_root = component_root / "review" / self.slug
        self.candidates = self.review_root / "candidates.json"
        self.video_plan = component_root / "download_plans" / f"{self.slug}_videos.json"
        self.extract_plan = component_root / "download_plans" / f"{self.slug}_extract.json"
        self.dataset_root = component_root / "datasets/franka" / self.slug
        self.packed_repo_id = f"local/{self.source_slug}-franka-{self.slug}"
        self.index_path = component_root / "index.json"

    def prepare(self, *arguments: str) -> None:
        run([sys.executable, str(PREPARE), *arguments])

    def pilot(self, *arguments: str) -> None:
        run(
            [
                sys.executable, "-m", "lerobot.scripts.lerobot_diverse_pilot",
                "--config", str(SOURCE_CONFIG),
                "--metadata-root", str(self.root / "metadata"),
                "--output-root", str(self.root),
                *arguments,
            ]
        )

    def resolve_plan(self, episodes: list[int], destination: Path) -> dict:
        from lerobot.datasets.diverse_pilot import attach_payload_sizes, resolve_lerobot_payload

        audit = read_json(self.root / "audits" / f"{self.source}.json")
        manifest = resolve_lerobot_payload(self.spec, audit, self.metadata_root, episodes)
        manifest = attach_payload_sizes(manifest, token=os.environ.get("HF_TOKEN"))
        write_json(destination, manifest)
        return manifest

    def review(self, count: int, min_close_events: int) -> None:
        self.prepare(
            "nominate",
            "--scan", str(self.scan),
            "--metadata-root", str(self.metadata_root),
            "--lab", self.lab,
            "--count", str(count),
            "--min-close-events", str(min_close_events),
            "--output", str(self.candidates),
        )
        manifest = read_json(self.candidates)
        episodes = [int(item["episode_index"]) for item in manifest["candidates"]]
        plan = self.resolve_plan(episodes, self.video_plan)
        print(
            f"payload: {len(plan['files'])} files, "
            f"{plan['total_payload_bytes'] / 1e9:.1f} GB for {len(episodes)} candidates",
            flush=True,
        )
        self.pilot(
            "download",
            "--plan", str(self.video_plan),
            "--staging-root", str(self.staging_root),
        )
        self.prepare(
            "proxies",
            "--candidates", str(self.candidates),
            "--metadata-root", str(self.metadata_root),
            "--staging-root", str(self.staging_root),
            "--repo-id", self.repo_id,
            "--output-root", str(self.review_root),
            "--mode", "overview",
        )
        index = read_json(self.index_path)
        index["in_progress_component"] = {
            "component": self.lab,
            "stage": "visual_review",
            "candidates": len(episodes),
            "payload_files": len(plan["files"]),
            "review_root": f"review/{self.slug}",
            "reviewed": [],
        }
        index["next_component"] = self.lab
        index["next_milestone"] = (
            f"{self.lab} component: finish the visual pass, accept {ACCEPTANCE_TARGET}, "
            "write review records, then finalize"
        )
        index["status"] = "in_progress"
        write_json(self.index_path, index)

    def detail(self, episodes: list[int]) -> None:
        self.prepare(
            "proxies",
            "--candidates", str(self.candidates),
            "--metadata-root", str(self.metadata_root),
            "--staging-root", str(self.staging_root),
            "--repo-id", self.repo_id,
            "--output-root", str(self.review_root),
            "--mode", "detail",
            "--stride", str(ANCHOR_STRIDE_S),
            "--episodes", *[str(index) for index in episodes],
        )

    def finalize(self, accepted: list[int], minimum: int) -> None:
        if len(accepted) != ACCEPTANCE_TARGET:
            raise ValueError(
                f"The production target is {ACCEPTANCE_TARGET} accepted episodes, got {len(accepted)}"
            )
        self.prepare(
            "annotations",
            "--review-root", str(self.review_root),
            "--metadata-root", str(self.metadata_root),
            "--staging-root", str(self.staging_root),
            "--repo-id", self.repo_id,
            "--accepted", *[str(index) for index in accepted],
        )
        self.write_selection(accepted)
        self.resolve_plan(accepted, self.extract_plan)
        self.pilot(
            "extract",
            "--source", self.source,
            "--source-metadata-root", str(self.metadata_root),
            "--plan", str(self.extract_plan),
            "--staging-root", str(self.staging_root),
            "--annotations-root", str(self.review_root),
            "--dataset-root", str(self.dataset_root),
            "--repo-id", self.packed_repo_id,
            "--stride", str(ANCHOR_STRIDE_S),
            "--min-chunks-per-episode", str(minimum),
        )
        self.pilot("validate", "--dataset-root", str(self.dataset_root), "--repo-id", self.packed_repo_id)
        self.attach_metadata(accepted)
        self.update_index()

    def write_selection(self, accepted: list[int]) -> None:
        manifest = read_json(self.candidates)
        candidates = {int(item["episode_index"]): item for item in manifest["candidates"]}
        unknown = sorted(set(accepted) - candidates.keys())
        if unknown:
            raise ValueError(f"Accepted episodes are absent from the candidate manifest: {unknown}")
        reviews = {
            index: read_json(self.review_root / f"episode_{index:06d}.review.json") for index in accepted
        }
        write_json(
            self.review_root / "selection.json",
            {
                "source": self.repo_id,
                "component": self.lab,
                "selection_status": "visually_accepted",
                "accepted_episode_indices": accepted,
                "rejected_candidate_indices": sorted(candidates.keys() - set(accepted)),
                "review_basis": (
                    "labelled global-camera contact sheets over the full episode, "
                    "two-second-stride detail sheets for the shortlist"
                ),
                "accepted": {
                    str(index): {
                        "uuid": candidates[index]["uuid"],
                        "scene_family": candidates[index]["scene_family"],
                        "task": reviews[index]["task"],
                        "outcome": reviews[index].get("outcome", ""),
                        "notes": reviews[index].get("reviewer_notes", ""),
                    }
                    for index in accepted
                },
                "rejections": read_json(self.review_root / "rejections.json")
                if (self.review_root / "rejections.json").exists()
                else {},
            },
        )

    def attach_metadata(self, accepted: list[int]) -> None:
        report = read_json(self.dataset_root / "meta/validation_report.json")
        if not report.get("passed"):
            raise SystemExit(f"Validation failed for {self.lab}; component metadata not attached")
        if int(report.get("interpolated_action_points", 0)) <= 0:
            raise SystemExit(
                "DROID is a 15 Hz source, so a validated component must contain interpolated "
                "action points; got zero."
            )
        meta = self.dataset_root / "meta"
        shutil.copy2(self.candidates, meta / "candidate_nomination.json")
        shutil.copy2(self.review_root / "selection.json", meta / "source_selection.json")
        write_json(
            meta / "raw_acquisition.json",
            {
                "source": self.repo_id,
                "revision": read_json(self.extract_plan)["revision"],
                "source_format": "lerobot_v3",
                "acquisition": "Hub payload restricted to the resolved data and camera shards",
                "accepted_episode_indices": accepted,
                "action_source": "native",
                "component_grouping_key": "lab",
                "component": self.lab,
                "low_dimensional_plan": f"download_plans/{self.source}_lowdim.json",
                "video_plan": f"download_plans/{self.slug}_videos.json",
                "extract_plan": f"download_plans/{self.slug}_extract.json",
            },
        )

    def update_index(self) -> None:
        index = read_json(self.index_path)
        report = read_json(self.dataset_root / "meta/validation_report.json")
        extraction = read_json(self.dataset_root / "meta/extraction_report.json")
        annotations = [
            read_json(path) for path in sorted((self.dataset_root / "meta/annotations").glob("*.json"))
        ]
        qualities = sorted(
            {
                int(segment["quality"])
                for annotation in annotations
                for segment in annotation["segments"]
                if segment["retention"] == "keep"
            }
        )
        component = {
            "component": self.lab,
            "embodiment": "Franka",
            "status": "validated",
            "source_episodes": len(extraction["episodes"]),
            "retained_chunks": int(report["rows"]),
            "interpolated_action_points": int(report["interpolated_action_points"]),
            "quality_values": qualities,
            "mistake_events": sum(
                len(segment.get("mistake_events", []))
                for annotation in annotations
                for segment in annotation["segments"]
            ),
            "dataset_root": f"datasets/franka/{self.slug}",
            "review_root": f"review/{self.slug}",
        }
        components = [item for item in index["components"] if item["component"] != self.lab]
        components.append(component)
        index["components"] = components
        index["completed_source_episodes"] = sum(item["source_episodes"] for item in components)
        index["total_retained_chunks"] = sum(item["retained_chunks"] for item in components)
        remaining = [lab for lab in index["remaining_components"] if lab != self.lab]
        index["remaining_components"] = remaining
        index["next_component"] = remaining[0] if remaining else None
        in_progress = index.get("in_progress_component")
        if isinstance(in_progress, dict) and in_progress.get("component") == self.lab:
            index.pop("in_progress_component")
        index["next_milestone"] = (
            f"{remaining[0]} component: nominate 24, accept 10, annotate, extract, validate"
            if remaining
            else f"{self.source.replace('_', ' ').title()} complete: rebuild the combined coverage ledger"
        )
        index["status"] = "in_progress" if remaining else f"{self.source}_complete"
        write_json(self.index_path, index)
        print(json.dumps(component, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(DEFAULT_COMPONENT_ROOTS))
    parser.add_argument("--component-root", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    review = commands.add_parser("review", help="nominate, acquire, and render review sheets")
    review.add_argument("--count", type=int, default=CANDIDATE_COUNT)
    review.add_argument("--min-close-events", type=int, default=None)
    detail = commands.add_parser("detail", help="render two-second-stride sheets for a shortlist")
    detail.add_argument("--episodes", type=int, nargs="+", required=True)
    finalize = commands.add_parser("finalize", help="annotate, extract, validate, and index")
    finalize.add_argument("--accepted", type=int, nargs="+", required=True)
    finalize.add_argument("--min-chunks-per-episode", type=int, default=3)
    args = parser.parse_args()
    component_root = args.component_root or REPO_ROOT / DEFAULT_COMPONENT_ROOTS[args.source]
    component = Component(args.lab, component_root, args.source)
    if args.command == "review":
        minimum = args.min_close_events
        if minimum is None:
            minimum = DEFAULT_MIN_CLOSE_EVENTS.get(args.source, 0)
        component.review(args.count, minimum)
    elif args.command == "detail":
        component.detail(args.episodes)
    else:
        component.finalize(args.accepted, args.min_chunks_per_episode)


if __name__ == "__main__":
    main()
