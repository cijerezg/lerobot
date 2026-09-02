#!/usr/bin/env python

"""Drive one RoboChallenge task through the production component recipe.

`review` runs the nomination and proxy-rendering steps so a reviewer can accept
episodes. `finalize` runs everything after acceptance: annotations, full-rate
conversion, packed extraction, validation, component metadata, and the ledger
update. Acquisition is separate because it is the only step that touches the
network; see `acquire_robochallenge.py`.

Visual acceptance stays a human decision between the two subcommands. The
automatic motion screen only decides which anchors inside an accepted episode
survive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PREPARE = Path(__file__).with_name("prepare_robochallenge.py")
CONVERT = REPO_ROOT / "lerobot/examples/dataset/diverse_robot_pilot/convert_robochallenge.py"
PILOT_CONFIG = REPO_ROOT / "lerobot/examples/dataset/diverse_robot_pilot/config.json"
PILOT_METADATA_ROOT = REPO_ROOT / "outputs/diverse_robot_pilot/metadata"
CANDIDATE_COUNT = 15
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
    def __init__(self, task: str, component_root: Path, production_manifest: Path) -> None:
        entries = {item["name"]: item for item in read_json(production_manifest)["tasks"]}
        if task not in entries:
            raise ValueError(f"{task!r} is not listed in {production_manifest}")
        self.task = task
        self.embodiment = str(entries[task]["embodiment"])
        self.root = component_root
        self.slug = f"robochallenge-{self.embodiment.lower()}-{task.replace('_', '-')}"
        self.raw_task_root = component_root / "staging/raw" / task / task
        self.review_root = component_root / "review" / task
        self.candidates = self.review_root / "candidates.json"
        self.source_repo_id = f"local/{self.slug}-production-source"
        self.source_dataset_root = (
            component_root / "staging" / self.source_repo_id.replace("/", "__")
        )
        self.dataset_root = component_root / "datasets" / self.embodiment.lower() / task
        self.packed_repo_id = f"local/{self.slug}"
        self.index_path = component_root / "index.json"

    def review(self) -> None:
        run(
            [
                sys.executable, str(PREPARE), "nominate",
                "--task-root", str(self.raw_task_root),
                "--output", str(self.candidates),
                "--count", str(CANDIDATE_COUNT),
            ]
        )
        run(
            [
                sys.executable, str(PREPARE), "proxies",
                "--task-root", str(self.raw_task_root),
                "--manifest", str(self.candidates),
                "--output-root", str(self.review_root),
            ]
        )
        manifest = read_json(self.candidates)
        print(
            json.dumps(
                {
                    "task": manifest["task"],
                    "prompt": manifest["prompt"],
                    "embodiment": manifest["embodiment"],
                    "episodes_scanned": manifest["episodes_scanned"],
                    "candidates": [
                        {
                            "episode_index": item["episode_index"],
                            "robot_id": item["robot_id"],
                            "duration_s": round(item["duration_s"], 2),
                            "candidate_anchors": item["candidate_anchors"],
                            "active_anchor_fraction": round(item["active_anchor_fraction"], 3),
                        }
                        for item in manifest["candidates"]
                    ],
                    "review_root": str(self.review_root),
                },
                indent=2,
            )
        )

    def finalize(self, accepted: list[int]) -> None:
        if len(accepted) != 10:
            raise ValueError(f"The production target is 10 accepted episodes, got {len(accepted)}")
        run(
            [
                sys.executable, str(PREPARE), "annotations",
                "--task-root", str(self.raw_task_root),
                "--manifest", str(self.candidates),
                "--output-root", str(self.review_root),
                "--accepted", *[str(index) for index in accepted],
            ]
        )
        run(
            [
                sys.executable, str(CONVERT),
                "--raw-task-root", str(self.raw_task_root),
                "--dataset-root", str(self.source_dataset_root),
                "--repo-id", self.source_repo_id,
                "--episodes", *[str(index) for index in accepted],
            ]
        )
        pilot = [
            sys.executable, "-m", "lerobot.scripts.lerobot_diverse_pilot",
            "--config", str(PILOT_CONFIG),
            "--metadata-root", str(PILOT_METADATA_ROOT),
            "--output-root", str(self.root),
        ]
        run(
            [
                *pilot, "extract",
                "--source", "robochallenge_raw",
                "--source-metadata-root", str(self.source_dataset_root),
                "--plan", str(self.source_dataset_root / "meta/local_acquisition_manifest.json"),
                "--staging-root", str(self.root / "staging"),
                "--annotations-root", str(self.review_root),
                "--dataset-root", str(self.dataset_root),
                "--repo-id", self.packed_repo_id,
                "--stride", str(ANCHOR_STRIDE_S),
            ]
        )
        run(
            [
                *pilot, "validate",
                "--dataset-root", str(self.dataset_root),
                "--repo-id", self.packed_repo_id,
            ]
        )
        self.attach_metadata(accepted)
        self.update_index()

    def attach_metadata(self, accepted: list[int]) -> None:
        report = read_json(self.dataset_root / "meta/validation_report.json")
        if not report.get("passed"):
            raise SystemExit(f"Validation failed for {self.task}; component metadata not attached")
        meta = self.dataset_root / "meta"
        shutil.copy2(self.candidates, meta / "candidate_nomination.json")
        shutil.copy2(self.review_root / "selection.json", meta / "source_selection.json")
        acquisition = read_json(self.root / "staging/raw" / self.task / "raw_acquisition.json")
        acquisition["accepted_episode_indices"] = accepted
        acquisition["action_source"] = "copy_state"
        write_json(meta / "raw_acquisition.json", acquisition)

    def update_index(self) -> None:
        index = read_json(self.index_path)
        report = read_json(self.dataset_root / "meta/validation_report.json")
        extraction = read_json(self.dataset_root / "meta/extraction_report.json")
        component = {
            "task": self.task,
            "embodiment": self.embodiment,
            "status": "validated",
            "source_episodes": len(extraction["episodes"]),
            "retained_chunks": int(report["rows"]),
            "dataset_root": f"datasets/{self.embodiment.lower()}/{self.task}",
            "review_root": f"review/{self.task}",
        }
        components = [item for item in index["components"] if item["task"] != self.task]
        components.append(component)
        index["components"] = components
        index["completed_source_episodes"] = sum(item["source_episodes"] for item in components)
        index["total_retained_chunks"] = sum(item["retained_chunks"] for item in components)
        remaining = [task for task in index["remaining_tasks"] if task != self.task]
        index["remaining_tasks"] = remaining
        index["next_task"] = remaining[0] if remaining else None
        write_json(self.index_path, index)
        print(json.dumps(component, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--component-root",
        type=Path,
        default=REPO_ROOT / "outputs/diverse_robot_dataset_build/robochallenge",
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=Path(__file__).with_name("robochallenge_production.json"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("review", help="nominate candidates and render review proxies")
    finalize = commands.add_parser("finalize", help="convert, extract, validate, and index")
    finalize.add_argument("--accepted", type=int, nargs="+", required=True)
    args = parser.parse_args()
    component = Component(args.task, args.component_root, args.production_manifest)
    if args.command == "review":
        component.review()
    else:
        component.finalize(args.accepted)


if __name__ == "__main__":
    main()
