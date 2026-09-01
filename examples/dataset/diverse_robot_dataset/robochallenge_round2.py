#!/usr/bin/env python

"""Nominate and render a further review round for an already-built RoboChallenge task.

Round 1 accepted ten episodes per task from fifteen nominated candidates. The
extracted task staging holds roughly a thousand episodes, so further episodes
cost no download. This driver nominates a fresh candidate set from the episodes
no earlier round looked at, renders the same review surfaces, and writes the
pending annotation stubs. Visual acceptance stays a human step.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PREPARE = Path(__file__).with_name("prepare_robochallenge.py")
BUILD_CORPUS = Path(__file__).with_name("build_corpus.py")
CONVERT = REPO_ROOT / "lerobot/examples/dataset/diverse_robot_pilot/convert_robochallenge.py"
PRODUCTION_MANIFEST = Path(__file__).with_name("robochallenge_production.json")
DEFAULT_COMPONENT_ROOT = REPO_ROOT / "outputs/diverse_robot_dataset/robochallenge"
CANDIDATE_COUNT = 15


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def reviewed_episode_indices(review_root: Path) -> list[int]:
    """Every episode any earlier round nominated, accepted, or rejected."""
    seen: set[int] = set()
    for candidates in review_root.glob("**/candidates.json"):
        manifest = read_json(candidates)
        seen.update(int(item["episode_index"]) for item in manifest["candidates"])
        seen.update(int(index) for index in manifest.get("excluded_episode_indices", []))
    for annotation in review_root.glob("**/episode_*.annotations.json"):
        seen.add(int(annotation.name.removeprefix("episode_").split(".")[0]))
    return sorted(seen)


def finalize(
    task: str,
    review_round: int,
    accepted: list[int],
    task_root: Path,
    round_root: Path,
    component_root: Path,
    embodiment: str,
) -> None:
    """Write annotations for the accepted episodes, convert them, and ingest the corpus."""
    slug = f"robochallenge-{embodiment.lower()}-{task.replace('_', '-')}"
    staged_root = component_root / "staging" / f"local__{slug}-round{review_round}-source"
    run(
        [
            sys.executable, str(PREPARE), "annotations",
            "--task-root", str(task_root),
            "--manifest", str(round_root / "candidates.json"),
            "--output-root", str(round_root),
            "--accepted", *[str(index) for index in accepted],
        ]
    )
    run(
        [
            sys.executable, str(CONVERT),
            "--raw-task-root", str(task_root),
            "--dataset-root", str(staged_root),
            "--repo-id", f"local/{slug}-round{review_round}-source",
            "--episodes", *[str(index) for index in accepted],
        ]
    )
    run(
        [
            sys.executable, str(BUILD_CORPUS), "ingest-round",
            "--source", "robochallenge",
            "--task", task,
            "--round", str(review_round),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--count", type=int, default=CANDIDATE_COUNT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument(
        "--accepted",
        type=int,
        nargs="+",
        help="Visually accepted episode indices; runs annotations, conversion, and corpus ingest.",
    )
    args = parser.parse_args()

    tasks = {item["name"]: item for item in read_json(PRODUCTION_MANIFEST)["tasks"]}
    if args.task not in tasks:
        raise SystemExit(f"{args.task!r} is not listed in {PRODUCTION_MANIFEST}")
    task_root = args.component_root / "staging/raw" / args.task / args.task
    if not (task_root / "data").is_dir():
        raise SystemExit(f"Task staging is absent: {task_root}")

    review_root = args.component_root / "review" / args.task
    round_root = review_root / f"round{args.round}"
    if args.accepted:
        finalize(
            args.task,
            args.round,
            args.accepted,
            task_root,
            round_root,
            args.component_root,
            str(tasks[args.task]["embodiment"]),
        )
        return
    exclude = reviewed_episode_indices(review_root)
    candidates = round_root / "candidates.json"

    run(
        [
            sys.executable, str(PREPARE), "nominate",
            "--task-root", str(task_root),
            "--output", str(candidates),
            "--count", str(args.count),
            "--exclude", *[str(index) for index in exclude],
        ]
    )
    run(
        [
            sys.executable, str(PREPARE), "proxies",
            "--task-root", str(task_root),
            "--manifest", str(candidates),
            "--output-root", str(round_root),
        ]
    )
    manifest = read_json(candidates)
    print(
        json.dumps(
            {
                "task": manifest["task"],
                "round": args.round,
                "excluded_from_earlier_rounds": len(exclude),
                "episodes_scanned": manifest["episodes_scanned"],
                "robot_balance": manifest["robot_balance"]["selected"],
                "candidates": [
                    {
                        "episode_index": item["episode_index"],
                        "robot_id": item["robot_id"],
                        "duration_s": round(item["duration_s"], 2),
                        "active_anchor_fraction": round(item["active_anchor_fraction"], 3),
                    }
                    for item in manifest["candidates"]
                ],
                "review_root": str(round_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
