#!/usr/bin/env python

"""
Generate MEM-style long-term memory summaries from video on a fixed time grid.

The memory m_k is a compressed, strictly retrospective natural-language summary of what
the robot has done so far, updated recurrently every --window-seconds of video:
m_k = VLM(frames of window k, m_{k-1}). Summaries are independent of subtask annotation —
video only, no labels — so this can run before (or without) subtask_annotate. Content is
strictly what HAS happened plus visible scene facts; never plans or future steps.

Output: meta/summaries.parquet with columns
    episode_index, segment_index, from_index, to_index, summary
where segment_index is the window index and [from_index, to_index) its global dataset
index range. At train time a frame inside window k conditions on summaries[k-1] ("" for
the first window); the hold/update pairing (summary_label_spans) and the memory-first
decode seam consume these rows unchanged.

Usage:
    python summary_annotate.py --data-dir /path/to/dataset --video-key observation.images.top [--dry-run]
"""

import argparse
import glob
import io
import json
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console

from lerobot.data_processing.annotate.canonicalize_subtasks import parse_json_response

console = Console()

MAX_FRAMES_PER_WINDOW = 6


def create_summary_prompt(coarse_goal: str, prev_summary: str) -> str:
    prev = prev_summary or "Nothing has happened yet."
    return textwrap.dedent(f"""\
        A robot is working on this task: "{coarse_goal}"

        Its memory so far:
        {prev}

        The frames show its most recent actions. Update the memory. Aim for the
        shortest text that still tells the full story: someone reading only the
        memory should know exactly what has been done and how far along the task
        is. Ground progress in what is visible: how many socks are in the basket
        and how many are still on the table. Socks move one at a time and a pair
        is complete when both of its socks are in the basket — track partially
        moved pairs explicitly. First person, past tense. Don't invent progress
        the frames don't show.

        Reply with only JSON: {{"summary": "<updated memory>"}}
    """)


def episode_windows(from_index: int, to_index: int, window_frames: int) -> list[dict]:
    """Fixed-size windows tiling [from_index, to_index); the last window keeps the
    remainder (a trailing stub shorter than half a window merges into its predecessor)."""
    starts = list(range(from_index, to_index, window_frames))
    if len(starts) > 1 and to_index - starts[-1] < window_frames / 2:
        starts.pop()
    return [
        {"from_index": s, "to_index": starts[i + 1] if i + 1 < len(starts) else to_index}
        for i, s in enumerate(starts)
    ]


def segment_frames(video_path: Path, start_s: float, end_s: float) -> list[Image.Image]:
    """Evenly sampled RGB frames spanning [start_s, end_s], decoded via the ffmpeg CLI
    (the dataset videos are AV1; OpenCV's bundled decoder can't read them)."""
    n = int(np.clip(round(end_s - start_s), 2, MAX_FRAMES_PER_WINDOW))
    frames = []
    for t in np.linspace(start_s, end_s, n):
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        )
        if out.returncode == 0 and out.stdout:
            frames.append(Image.open(io.BytesIO(out.stdout)).convert("RGB"))
    if not frames:
        raise ValueError(f"No frames decoded from {video_path} in [{start_s:.2f}, {end_s:.2f}]s")
    return frames


def generate_summary(
    model, processor, device: str, frames: list[Image.Image], coarse_goal: str, prev_summary: str
) -> str:
    prompt = create_summary_prompt(coarse_goal, prev_summary)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, videos=frames, num_frames=len(frames), return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()

    summary = parse_json_response(response)["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"Empty summary in response: {response[:200]}")
    return summary.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate video-based MEM summary chains on a fixed time grid")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--video-key", type=str, required=True)
    parser.add_argument("--window-seconds", type=float, default=12.0, help="memory update interval")
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="print per-episode windows, no LLM, no writes")
    args = parser.parse_args()

    root = Path(args.data_dir)
    info = json.load(open(root / "meta" / "info.json"))
    fps = float(info["fps"])
    window_frames = max(1, round(args.window_seconds * fps))
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    coarse_goal = str(tasks.index[0])

    episodes_meta = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))]
    ).set_index("episode_index")

    windows_by_episode = {
        int(ep_idx): episode_windows(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]), window_frames)
        for ep_idx, ep in episodes_meta.iterrows()
    }
    total = sum(len(w) for w in windows_by_episode.values())
    console.print(
        f'goal: "{coarse_goal}" | {len(windows_by_episode)} episodes, {args.window_seconds}s windows '
        f"({window_frames} frames) -> {total} windows / LLM calls"
    )

    if args.dry_run:
        for ep_idx in sorted(windows_by_episode):
            w = windows_by_episode[ep_idx]
            console.print(f"episode {ep_idx}: {len(w)} windows, frames {w[0]['from_index']}..{w[-1]['to_index']}")
        return

    from transformers import AutoModelForCausalLM, AutoProcessor

    console.print(f"[cyan]Loading {args.model}...[/cyan]")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)

    rows = []
    for ep_idx in sorted(windows_by_episode):
        ep = episodes_meta.loc[ep_idx]
        video_path = root / info["video_path"].format(
            video_key=args.video_key,
            chunk_index=int(ep[f"videos/{args.video_key}/chunk_index"]),
            file_index=int(ep[f"videos/{args.video_key}/file_index"]),
        )
        video_from_ts = float(ep[f"videos/{args.video_key}/from_timestamp"])
        ep_from_index = int(ep["dataset_from_index"])

        summary = ""
        for k, win in enumerate(windows_by_episode[ep_idx]):
            start_s = video_from_ts + (win["from_index"] - ep_from_index) / fps
            end_s = video_from_ts + (win["to_index"] - 1 - ep_from_index) / fps
            frames = segment_frames(video_path, start_s, end_s)
            summary = generate_summary(model, processor, args.device, frames, coarse_goal, summary)
            console.print(f"[green]ep {ep_idx} win {k}:[/green] {summary}")
            rows.append({
                "episode_index": ep_idx,
                "segment_index": k,
                "from_index": win["from_index"],
                "to_index": win["to_index"],
                "summary": summary,
            })

    out_path = root / "meta" / "summaries.parquet"
    pd.DataFrame(rows).to_parquet(out_path, engine="pyarrow", compression="snappy")
    with open(root / "meta" / "summaries_info.json", "w") as f:
        json.dump(
            {"model": args.model, "coarse_description": coarse_goal, "video_key": args.video_key,
             "window_seconds": args.window_seconds},
            f, indent=2,
        )
    console.print(f"[green]✓ Wrote {len(rows)} window summaries to {out_path}[/green]")


if __name__ == "__main__":
    main()
