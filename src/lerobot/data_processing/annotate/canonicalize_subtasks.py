#!/usr/bin/env python

"""
One-off cleanup: merge near-duplicate subtask labels in an annotated LeRobot dataset.

The VLM annotator (subtask_annotate_gemma_4.py) produces free-form skill names, so the
same action appears under several wordings ("grasp red truck" / "grasp the red truck").
This script asks an LLM to group the labels, picks one canonical wording per group, and
rewrites the dataset in place:

  - meta/subtasks.parquet      deduped canonical set, reindexed by sorted order
  - meta/skills.json           renamed skills, rebuilt index map, adjacent duplicates merged
  - data/**/*.parquet          subtask_index column remapped (-1 stays -1)

Originals are copied to <root>/pre_canonicalization/ first. The LLM only proposes the
string grouping; it is validated to be an exact partition of the existing labels before
anything is written.

Usage:
    python canonicalize_subtasks.py --data-dir /path/to/dataset [--dry-run]

    # Validation dataset: reuse the train dataset's mapping instead of asking the
    # LLM, so both end up with one shared vocabulary (matched by name at load time):
    python canonicalize_subtasks.py --data-dir /path/to/validation --reuse-map /path/to/train
"""

import argparse
import glob
import json
import re
import shutil
import textwrap
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console

console = Console()


def create_grouping_prompt(coarse_goal: str, subtasks: list[str]) -> str:
    subtask_list = "\n".join(f"- {s}" for s in subtasks)
    return textwrap.dedent(f"""\
        You are cleaning up subtask labels for a robot manipulation dataset.
        The overall task is: "{coarse_goal}"

        The labels below were written by a video annotation model and contain
        near-duplicates: different wordings of the same physical action.

        Group the labels so that each group contains only wordings of the SAME action.
        Never merge two genuinely different actions (e.g. "lift the red truck" and
        "lower the red truck into the bowl" are different). When unsure, keep labels
        in separate groups. For each group, pick the clearest member as the canonical
        name — the canonical name MUST be copied verbatim from the group's members.

        Labels:
        {subtask_list}

        Every label must appear in exactly one group. Reply with only JSON:
        {{
          "groups": [
            {{"canonical": "<one of the members>", "members": ["<label>", ...]}}
          ]
        }}
    """)


def parse_json_response(response: str) -> dict:
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def propose_mapping(model_name: str, device: str, coarse_goal: str, subtasks: list[str]) -> dict[str, str]:
    """Ask the LLM for a grouping and return a validated old-name -> canonical-name map."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    console.print(f"[cyan]Loading {model_name}...[/cyan]")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

    prompt = create_grouping_prompt(coarse_goal, subtasks)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()

    groups = parse_json_response(response)["groups"]

    mapping = {}
    for group in groups:
        for member in group["members"]:
            if member in mapping:
                raise ValueError(f"Label appears in two groups: {member!r}")
            mapping[member] = group["canonical"]
    missing = set(subtasks) - set(mapping)
    extra = set(mapping) - set(subtasks)
    if missing or extra:
        raise ValueError(f"Grouping is not a partition of the labels. Missing: {missing}, unknown: {extra}")
    return mapping


def load_reused_mapping(source: Path, subtasks: list[str]) -> tuple[dict[str, str], Path]:
    """Build the mapping from another dataset's canonicalization_map.json.

    `source` is the canonicalized dataset root (or the json itself). Labels the
    source map doesn't know are kept unchanged — they just won't merge into the
    shared vocabulary, so they're warned about unless they already ARE one of the
    source's canonical names."""
    path = source if source.suffix == ".json" else source / "meta" / "canonicalization_map.json"
    source_map = json.loads(path.read_text())["mapping"]
    canonicals = set(source_map.values())
    mapping = {label: source_map.get(label, label) for label in subtasks}
    unknown = [label for label in subtasks if label not in source_map and label not in canonicals]
    if unknown:
        console.print(f"[yellow]{len(unknown)} labels not in the reused map (kept as-is): {unknown}[/yellow]")
    return mapping, path


def merge_adjacent_skills(skills: list[dict]) -> list[dict]:
    merged = [dict(skills[0])]
    for skill in skills[1:]:
        if skill["name"] == merged[-1]["name"]:
            merged[-1]["end"] = skill["end"]
        else:
            merged.append(dict(skill))
    return merged


def apply_mapping(root: Path, mapping: dict[str, str], model_name: str) -> None:
    old_df = pd.read_parquet(root / "meta" / "subtasks.parquet")
    skills_data = json.load(open(root / "meta" / "skills.json"))
    data_files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))

    canonical_names = sorted(set(mapping.values()))
    new_index = {name: i for i, name in enumerate(canonical_names)}
    old_to_new = {
        int(old_df.loc[name, "subtask_index"]): new_index[mapping[name]] for name in old_df.index
    }

    backup_dir = root / "pre_canonicalization"
    backup_dir.mkdir(exist_ok=True)
    for path in [root / "meta" / "subtasks.parquet", root / "meta" / "skills.json", *map(Path, data_files)]:
        dest = backup_dir / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
    console.print(f"[green]✓ Backed up originals to {backup_dir}[/green]")

    subtasks_df = pd.DataFrame(
        [{"subtask": name, "subtask_index": i} for name, i in new_index.items()]
    ).set_index("subtask")
    subtasks_df.to_parquet(root / "meta" / "subtasks.parquet", engine="pyarrow", compression="snappy")

    skills_data["skill_to_subtask_index"] = new_index
    for ep in skills_data["episodes"].values():
        for skill in ep["skills"]:
            skill["name"] = mapping[skill["name"]]
        ep["skills"] = merge_adjacent_skills(ep["skills"])
    with open(root / "meta" / "skills.json", "w") as f:
        json.dump(skills_data, f, indent=2)

    for data_file in data_files:
        df = pd.read_parquet(data_file)
        df["subtask_index"] = df["subtask_index"].map(lambda i: old_to_new.get(i, -1))
        df.to_parquet(data_file, engine="pyarrow", compression="snappy")

    audit = {"model": model_name, "mapping": mapping, "old_index_to_new": old_to_new}
    with open(root / "meta" / "canonicalization_map.json", "w") as f:
        json.dump(audit, f, indent=2)

    console.print(f"[green]✓ {len(old_df)} labels -> {len(canonical_names)} canonical subtasks[/green]")


def main():
    parser = argparse.ArgumentParser(description="Merge near-duplicate subtask labels via LLM grouping")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print the proposed mapping and exit")
    parser.add_argument(
        "--reuse-map",
        type=str,
        default=None,
        help="canonicalized dataset root (or canonicalization_map.json) whose mapping "
        "is applied instead of asking the LLM — use for validation datasets so the "
        "vocabulary matches train",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    subtasks = sorted(pd.read_parquet(root / "meta" / "subtasks.parquet").index)

    if args.reuse_map:
        mapping, map_path = load_reused_mapping(Path(args.reuse_map), subtasks)
        model_name = f"reuse-map:{map_path}"
    else:
        coarse_goal = json.load(open(root / "meta" / "skills.json"))["coarse_description"]
        mapping = propose_mapping(args.model, args.device, coarse_goal, subtasks)
        model_name = args.model

    for canonical in sorted(set(mapping.values())):
        members = [old for old, new in mapping.items() if new == canonical]
        console.print(f"[bold]{canonical}[/bold]  <-  {members}")

    if args.dry_run:
        console.print("[yellow]Dry run: nothing written.[/yellow]")
        return
    apply_mapping(root, mapping, model_name)


if __name__ == "__main__":
    main()
