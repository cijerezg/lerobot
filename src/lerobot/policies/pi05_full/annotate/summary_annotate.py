#!/usr/bin/env python

"""
One-off: generate MEM-style long-term memory summaries for an annotated LeRobot dataset.

Following the MEM paper (pi.website/download/Mem.pdf), the long-term memory m_t is a
compressed natural-language summary of what the robot has done so far, updated after each
completed subtask. This script builds the training targets for that summary chain: for
every episode it takes the ordered subtask segments (runs of the per-frame subtask_index
column) and asks an LLM for cumulative summaries, one per segment — summaries[k] is the
memory state after completing segment k. The LLM is prompted to compress and keep only
information relevant for finishing the task.

Run canonicalize_subtasks.py first so the summaries are built from clean labels.
Episodes with identical subtask sequences share one LLM call.

Output: meta/summaries.parquet with columns
    episode_index, segment_index, from_index, to_index, subtask_index, subtask, summary
where [from_index, to_index) is the segment's global dataset index range. At train time,
a frame inside segment k conditions on summaries[k-1] (empty string for the first segment).

Usage:
    python summary_annotate.py --data-dir /path/to/dataset [--dry-run]
"""

import argparse
import glob
import json
import textwrap
from pathlib import Path

import pandas as pd
import torch
from rich.console import Console

from canonicalize_subtasks import parse_json_response

console = Console()


def create_summary_prompt(coarse_goal: str, subtasks: list[str]) -> str:
    steps = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(subtasks))
    return textwrap.dedent(f"""\
        A robot is solving this task: "{coarse_goal}"

        It completed the following subtasks in order:
        {steps}

        Write the robot's running memory: after each completed subtask, a first-person
        summary of everything done so far that is still relevant for finishing the task.
        Update the previous summary at each step. Be succinct: compress aggressively,
        combine repeated or similar actions, and drop details that no longer matter for
        future steps (short transit/retract moves rarely matter). Each summary must be
        a single sentence or two, like: "I picked up the red truck and moved it to the
        bowl."

        Reply with only JSON, one summary per subtask ({len(subtasks)} total):
        {{"summaries": ["<memory after subtask 1>", "<memory after subtask 2>", ...]}}
    """)


def episode_segments(df: pd.DataFrame) -> list[dict]:
    """Ordered subtask segments of one episode: runs of subtask_index (unlabeled -1 skipped)."""
    df = df.sort_values("index")
    run_id = (df["subtask_index"] != df["subtask_index"].shift()).cumsum()
    segments = []
    for _, run in df.groupby(run_id, sort=True):
        subtask_index = int(run["subtask_index"].iloc[0])
        if subtask_index == -1:
            continue
        segments.append({
            "subtask_index": subtask_index,
            "from_index": int(run["index"].iloc[0]),
            "to_index": int(run["index"].iloc[-1]) + 1,
        })
    return segments


def generate_summaries(model, processor, device: str, coarse_goal: str, subtasks: list[str]) -> list[str]:
    prompt = create_summary_prompt(coarse_goal, subtasks)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()

    summaries = parse_json_response(response)["summaries"]
    if len(summaries) != len(subtasks) or not all(isinstance(s, str) and s.strip() for s in summaries):
        raise ValueError(f"Expected {len(subtasks)} non-empty summaries, got: {summaries}")
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Generate MEM-style summary chains for subtask segments")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print segments and unique sequences, no LLM, no writes")
    args = parser.parse_args()

    root = Path(args.data_dir)
    coarse_goal = json.load(open(root / "meta" / "skills.json"))["coarse_description"]
    subtask_names = pd.read_parquet(root / "meta" / "subtasks.parquet")["subtask_index"]
    index_to_name = {int(i): name for name, i in subtask_names.items()}

    data_files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    frames = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "index", "subtask_index"]) for f in data_files]
    )

    episodes = {
        int(ep_idx): episode_segments(ep_df) for ep_idx, ep_df in frames.groupby("episode_index")
    }
    sequences = {
        ep_idx: tuple(index_to_name[seg["subtask_index"]] for seg in segments)
        for ep_idx, segments in episodes.items()
    }
    unique_sequences = sorted(set(sequences.values()))
    console.print(
        f"{len(episodes)} episodes, {len(unique_sequences)} unique subtask sequences -> "
        f"{len(unique_sequences)} LLM calls"
    )

    if args.dry_run:
        for seq in unique_sequences:
            count = sum(1 for s in sequences.values() if s == seq)
            console.print(f"[bold]{count} episode(s):[/bold] {list(seq)}")
        return

    from transformers import AutoModelForCausalLM, AutoProcessor

    console.print(f"[cyan]Loading {args.model}...[/cyan]")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)

    summaries_by_sequence = {}
    failures = []
    for seq in unique_sequences:
        try:
            summaries_by_sequence[seq] = generate_summaries(model, processor, args.device, coarse_goal, list(seq))
        except Exception as e:
            console.print(f"[red]✗ {list(seq)}: {e}[/red]")
            failures.append(seq)
    if failures:
        raise SystemExit(f"{len(failures)} sequence(s) failed; nothing written. Rerun after inspecting.")

    rows = []
    for ep_idx in sorted(episodes):
        summaries = summaries_by_sequence[sequences[ep_idx]]
        for k, seg in enumerate(episodes[ep_idx]):
            rows.append({
                "episode_index": ep_idx,
                "segment_index": k,
                "from_index": seg["from_index"],
                "to_index": seg["to_index"],
                "subtask_index": seg["subtask_index"],
                "subtask": index_to_name[seg["subtask_index"]],
                "summary": summaries[k],
            })
    out_path = root / "meta" / "summaries.parquet"
    pd.DataFrame(rows).to_parquet(out_path, engine="pyarrow", compression="snappy")
    with open(root / "meta" / "summaries_info.json", "w") as f:
        json.dump({"model": args.model, "coarse_description": coarse_goal}, f, indent=2)
    console.print(f"[green]✓ Wrote {len(rows)} segment summaries to {out_path}[/green]")


if __name__ == "__main__":
    main()
