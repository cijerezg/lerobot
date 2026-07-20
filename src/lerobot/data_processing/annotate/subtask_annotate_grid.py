#!/usr/bin/env python

"""
Grid-based subtask annotation, memory-first: label the robot's current action every
--interval-seconds, conditioned on the long-term memory (meta/summaries.parquet, run
summary_annotate.py first) plus frames spanning the window (top view, optionally wrist).

Labels are atomic and PROGRESS-FREE ("grasp the white sock", "move the sock to the
basket") — no ordinals or counts, progress lives in the memory. Adjacent identical
labels merge into segments naturally; canonicalize_subtasks.py still applies after.

Writes a new dataset (add_features) with per-frame subtask_index, meta/subtasks.parquet,
a meta/subtask_windows.json audit, copies the summaries files, and hardlinks the depth/
sidecar.

Usage:
    python subtask_annotate_grid.py --data-dir /path/to/dataset --top-key observation.images.top \
        [--wrist-key observation.images.wrist] [--preview-episode 0] --output-dir /path/to/out
"""

import argparse
import glob
import io
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from rich.console import Console

from lerobot.data_processing.annotate.canonicalize_subtasks import parse_json_response
from lerobot.data_processing.annotate.summary_annotate import episode_windows

console = Console()

TOP_FRACS = (0.0, 0.5, 1.0)
WRIST_FRACS = (0.25, 0.75)


def create_subtask_prompt(coarse_goal: str, memory: str, n_top: int, n_wrist: int) -> str:
    wrist_clause = (
        f" The first {n_top} frames are the overhead view in time order; the last {n_wrist} are the wrist camera."
        if n_wrist else ""
    )
    return textwrap.dedent(f"""\
        A robot is working on this task: "{coarse_goal}"

        The robot's memory of what it has done so far:
        {memory or "Nothing has happened yet."}

        The frames show what the robot does during the next few seconds.{wrist_clause}
        Name the single action the robot is performing, as a short imperative like
        "grasp the white sock", "move the sock to the basket", "release the sock in
        the basket", or "return to home". Name objects by appearance only — never use
        ordinals or counts; progress is tracked in the memory, not the action name.

        Reply with only JSON: {{"subtask": "<action>"}}
    """)


def frame_at(video_path: Path, t: float) -> Image.Image | None:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    if out.returncode == 0 and out.stdout:
        return Image.open(io.BytesIO(out.stdout)).convert("RGB")
    return None


def generate_subtask(model, processor, device, frames, coarse_goal, memory, n_top, n_wrist) -> str:
    prompt = create_subtask_prompt(coarse_goal, memory, n_top, n_wrist)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, videos=frames, num_frames=len(frames), return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()
    subtask = parse_json_response(response)["subtask"]
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError(f"Empty subtask in response: {response[:200]}")
    return subtask.strip().lower().rstrip(".")


def video_path_for(root: Path, info: dict, ep, key: str) -> tuple[Path, float]:
    path = root / info["video_path"].format(
        video_key=key, chunk_index=int(ep[f"videos/{key}/chunk_index"]), file_index=int(ep[f"videos/{key}/file_index"])
    )
    return path, float(ep[f"videos/{key}/from_timestamp"])


def main():
    parser = argparse.ArgumentParser(description="Grid-based memory-conditioned subtask annotation")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--top-key", type=str, required=True)
    parser.add_argument("--wrist-key", type=str, default=None)
    parser.add_argument("--interval-seconds", type=float, default=4.0)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--preview-episode", type=int, default=None,
                        help="annotate a single episode and print labels; no writes")
    parser.add_argument("--dry-run", action="store_true", help="print windows per episode, no LLM, no writes")
    args = parser.parse_args()

    root = Path(args.data_dir)
    info = json.load(open(root / "meta" / "info.json"))
    fps = float(info["fps"])
    interval_frames = max(1, round(args.interval_seconds * fps))
    coarse_goal = str(pd.read_parquet(root / "meta" / "tasks.parquet").index[0])

    summaries = pd.read_parquet(root / "meta" / "summaries.parquet").sort_values(["episode_index", "segment_index"])
    episodes_meta = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))]
    ).set_index("episode_index")

    windows_by_episode = {
        int(ep_idx): episode_windows(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]), interval_frames)
        for ep_idx, ep in episodes_meta.iterrows()
    }
    total = sum(len(w) for w in windows_by_episode.values())
    console.print(
        f'goal: "{coarse_goal}" | {len(windows_by_episode)} episodes, {args.interval_seconds}s windows -> {total} LLM calls'
    )

    if args.dry_run:
        for ep_idx in sorted(windows_by_episode):
            console.print(f"episode {ep_idx}: {len(windows_by_episode[ep_idx])} windows")
        return
    if args.preview_episode is None and not args.output_dir:
        raise SystemExit("--output-dir is required unless --preview-episode or --dry-run is used")

    def memory_for(ep_idx: int, frame: int) -> str:
        rows = summaries[(summaries.episode_index == ep_idx) & (summaries.to_index <= frame)]
        return str(rows.iloc[-1]["summary"]) if len(rows) else ""

    from transformers import AutoModelForCausalLM, AutoProcessor

    console.print(f"[cyan]Loading {args.model}...[/cyan]")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)

    episode_indices = [args.preview_episode] if args.preview_episode is not None else sorted(windows_by_episode)
    labels_by_episode: dict[int, list[str]] = {}
    for ep_idx in episode_indices:
        ep = episodes_meta.loc[ep_idx]
        top_path, top_from = video_path_for(root, info, ep, args.top_key)
        if args.wrist_key:
            wrist_path, wrist_from = video_path_for(root, info, ep, args.wrist_key)
        ep_from_index = int(ep["dataset_from_index"])
        labels = []
        for k, win in enumerate(windows_by_episode[ep_idx]):
            t0 = (win["from_index"] - ep_from_index) / fps
            t1 = (win["to_index"] - 1 - ep_from_index) / fps
            frames = [f for fr in TOP_FRACS if (f := frame_at(top_path, top_from + t0 + fr * (t1 - t0)))]
            n_top = len(frames)
            n_wrist = 0
            if args.wrist_key:
                wrist = [f for fr in WRIST_FRACS if (f := frame_at(wrist_path, wrist_from + t0 + fr * (t1 - t0)))]
                frames += wrist
                n_wrist = len(wrist)
            memory = memory_for(ep_idx, win["from_index"])
            label = generate_subtask(model, processor, args.device, frames, coarse_goal, memory, n_top, n_wrist)
            console.print(f"[green]ep {ep_idx} {t0:5.0f}-{t1:3.0f}s:[/green] {label}")
            labels.append(label)
        labels_by_episode[ep_idx] = labels

    if args.preview_episode is not None:
        console.print("[yellow]Preview only — nothing written.[/yellow]")
        return

    # Vocabulary + per-frame index array
    names = sorted({label for labels in labels_by_episode.values() for label in labels})
    name_to_idx = {n: i for i, n in enumerate(names)}
    n_frames = int(episodes_meta["dataset_to_index"].max())
    subtask_indices = np.full(n_frames, -1, dtype=np.int64)
    for ep_idx, labels in labels_by_episode.items():
        for win, label in zip(windows_by_episode[ep_idx], labels):
            subtask_indices[win["from_index"] : win["to_index"]] = name_to_idx[label]

    from lerobot.datasets.dataset_tools import add_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(root=str(root), repo_id="local/dataset", download_videos=False)
    output_dir = Path(args.output_dir)
    new_dataset = add_features(
        dataset=dataset,
        features={"subtask_index": (subtask_indices, {"dtype": "int64", "shape": (1,), "names": None})},
        output_dir=output_dir,
        repo_id=f"{dataset.repo_id}_with_subtasks",
    )

    subtasks_df = pd.DataFrame([{"subtask": n, "subtask_index": i} for n, i in name_to_idx.items()]).set_index("subtask")
    subtasks_df.to_parquet(output_dir / "meta" / "subtasks.parquet", engine="pyarrow", compression="snappy")
    audit = {
        "model": args.model, "interval_seconds": args.interval_seconds,
        "top_key": args.top_key, "wrist_key": args.wrist_key,
        "episodes": {
            str(ep_idx): [
                {"from_index": w["from_index"], "to_index": w["to_index"], "subtask": l}
                for w, l in zip(windows_by_episode[ep_idx], labels_by_episode[ep_idx])
            ]
            for ep_idx in labels_by_episode
        },
    }
    with open(output_dir / "meta" / "subtask_windows.json", "w") as f:
        json.dump(audit, f, indent=2)
    for name in ("summaries.parquet", "summaries_info.json"):
        if (root / "meta" / name).exists():
            shutil.copy(root / "meta" / name, output_dir / "meta" / name)
    depth_dir = root / "depth"
    if depth_dir.exists():
        shutil.copytree(depth_dir, output_dir / "depth", copy_function=os.link)
    console.print(f"[bold green]✓ {len(names)} subtasks, dataset written to {new_dataset.root}[/bold green]")


if __name__ == "__main__":
    main()
