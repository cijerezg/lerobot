#!/usr/bin/env python
"""
Bulk-annotate all episodes in a LeRobot dataset with a fixed subtask template.

For datasets where every episode follows the same task structure, this avoids
manually annotating each one. You define the skills and their proportional
time boundaries, and this script applies them uniformly.

Usage:
    python scripts/bulk_annotate_subtasks.py \
    --repo-id jackvial/so101_pickplace_success_120_v2 \
    --output-dir outputs/so101_pickplace_success_120_v2_w_subtasks_v2
"""

import argparse
import json
import logging
from pathlib import Path

from rich.console import Console

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_full.annotate.subtask_annotate import (
    EpisodeSkills,
    Skill,
    create_subtask_index_array,
    create_subtasks_dataframe,
    save_subtasks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# -- Define your template here --
# Each tuple is (skill_name, start_seconds, end_seconds)
# end_seconds=None means "until end of episode"
SKILL_TEMPLATE_ABSOLUTE = [
    ("reach and grasp cube", 0,  4),
    ("move to x marker",     4,  7),
    ("return to reset",      7,  None),
]

COARSE_DESCRIPTION = "Pick up the orange cube and place it on the black X marker"


def make_skills_for_episode(episode_duration: float, template: list[tuple[str, int, int | None]]) -> list[Skill]:
    skills = []
    for name, start, end in template:
        actual_start = min(float(start), episode_duration)
        actual_end = episode_duration if end is None else min(float(end), episode_duration)
        if actual_start >= episode_duration:
            break
        skills.append(Skill(name=name, start=round(actual_start, 3), end=round(actual_end, 3)))
    return skills


def main():
    parser = argparse.ArgumentParser(description="Bulk-annotate a LeRobot dataset with a fixed subtask template")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--data-dir", type=str, default=None, help="Local dataset path (overrides repo-id for loading)")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write the annotated dataset")
    args = parser.parse_args()

    for i in range(len(SKILL_TEMPLATE_ABSOLUTE) - 1):
        _, _, end = SKILL_TEMPLATE_ABSOLUTE[i]
        _, next_start, _ = SKILL_TEMPLATE_ABSOLUTE[i + 1]
        assert end == next_start, f"Gap/overlap between skills at boundary {end} -> {next_start}"

    console.print(f"[cyan]Loading dataset: {args.repo_id}[/cyan]")
    if args.data_dir:
        dataset = LeRobotDataset(repo_id="local/dataset", root=args.data_dir)
    else:
        dataset = LeRobotDataset(repo_id=args.repo_id)

    console.print(f"[green]Loaded {dataset.meta.total_episodes} episodes, {len(dataset)} frames[/green]")

    console.print("[cyan]Applying template to all episodes...[/cyan]")
    annotations: dict[int, EpisodeSkills] = {}

    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        ep_from = ep["dataset_from_index"]
        ep_to = ep["dataset_to_index"]

        ts_first = float(dataset.hf_dataset[int(ep_from)]["timestamp"])
        ts_last = float(dataset.hf_dataset[int(ep_to - 1)]["timestamp"])
        duration = ts_last - ts_first
        if duration <= 0:
            logger.warning(f"Episode {ep_idx} has zero/negative duration ({duration:.2f}s), skipping")
            continue

        skills = make_skills_for_episode(duration, SKILL_TEMPLATE_ABSOLUTE)
        annotations[ep_idx] = EpisodeSkills(
            episode_index=ep_idx,
            description=COARSE_DESCRIPTION,
            skills=skills,
        )

    console.print(f"[green]Annotated {len(annotations)} episodes with {len(SKILL_TEMPLATE_ABSOLUTE)} skills each[/green]")

    subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe(annotations)
    subtask_indices = create_subtask_index_array(dataset, annotations, skill_to_subtask_idx)

    unannotated = int((subtask_indices == -1).sum())
    if unannotated > 0:
        console.print(f"[yellow]Warning: {unannotated} frames left unannotated (possibly timestamp edge cases)[/yellow]")

    out_path = Path(args.output_dir)
    repo_id = f"{args.repo_id}_with_subtasks"

    console.print(f"[cyan]Writing annotated dataset to {out_path}...[/cyan]")
    new_dataset = add_features(
        dataset=dataset,
        features={
            "subtask_index": (subtask_indices, {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }),
        },
        output_dir=out_path,
        repo_id=repo_id,
    )

    save_subtasks(subtasks_df, out_path, console)

    skills_data = {
        "coarse_description": COARSE_DESCRIPTION,
        "skill_to_subtask_index": skill_to_subtask_idx,
        "episodes": {str(ep_idx): ann.to_dict() for ep_idx, ann in annotations.items()},
    }
    skills_path = out_path / "meta" / "skills.json"
    with open(skills_path, "w") as f:
        json.dump(skills_data, f, indent=2)
    console.print(f"[green]Saved skills.json to {skills_path}[/green]")

    console.print("[bold green]Done! Annotated dataset ready.[/bold green]")
    console.print(f"  Path: {out_path}")
    console.print(f"  Subtasks: {list(skill_to_subtask_idx.keys())}")
    console.print(f"  Total frames: {len(subtask_indices)}")
    console.print(f"  Annotated frames: {(subtask_indices >= 0).sum()}")


if __name__ == "__main__":
    main()
