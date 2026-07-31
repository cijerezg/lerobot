#!/usr/bin/env python

"""
Burn summary + subtask annotations into a plain mp4 per episode -- one file you can just open.

Top and wrist are placed side by side and the annotation text is burned in as subtitles, so
there is no server, no seeking protocol and no codec negotiation involved: the output is
H.264 that any player handles. Cue boundaries are the union of the subtask (4s) and summary
(12s) grids, so both values on screen are always the ones that apply at that instant.

    uv run lerobot/src/lerobot/data_processing/annotate/bake_annotations.py \
        --data-dir outputs/rebot_sorting_clothes_v1              # all episodes
    uv run lerobot/src/lerobot/data_processing/annotate/bake_annotations.py \
        --data-dir outputs/rebot_sorting_clothes_v1 --episodes 7 # just one

Writes to <data-dir>/annotated/ep<NN>.mp4.
"""

import argparse
import glob
import json
import subprocess
import textwrap
from pathlib import Path

import pandas as pd

TOP, WRIST = "observation.images.top", "observation.images.wrist"
WRAP = 74


def ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def build_srt(subtasks: list[dict], summaries: list[dict], duration: float) -> str:
    """One cue per interval where neither the subtask nor the summary changes."""
    edges = sorted({0.0, duration} | {r[k] for r in subtasks + summaries for k in ("t0", "t1")})
    edges = [e for e in edges if 0.0 <= e <= duration]

    def at(rows, t):
        return next((r["text"] for r in rows if r["t0"] <= t < r["t1"]), None)

    cues = []
    for a, b in zip(edges, edges[1:]):
        if b - a < 0.05:
            continue
        sub, summ = at(subtasks, a), at(summaries, a)
        if cues and cues[-1][2] == sub and cues[-1][3] == summ:
            cues[-1][1] = b                      # extend, don't repeat
        else:
            cues.append([a, b, sub, summ])

    out = []
    for i, (a, b, sub, summ) in enumerate(cues, 1):
        lines = [f"[{sub}]" if sub else "[-]"]
        lines += textwrap.wrap(summ, WRAP) if summ else ["-"]
        out.append(f"{i}\n{ts(a)} --> {ts(b)}\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Burn annotations into a per-episode mp4")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="veryfast")
    args = ap.parse_args()

    root = Path(args.data_dir).resolve()
    info = json.load(open(root / "meta" / "info.json"))
    fps = float(info["fps"])
    episodes_meta = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))]
    ).set_index("episode_index")
    summaries = pd.read_parquet(root / "meta" / "summaries.parquet")
    subtasks = json.load(open(root / "meta" / "subtask_windows.json"))["episodes"]

    out_dir = root / "annotated"
    out_dir.mkdir(exist_ok=True)

    wanted = args.episodes if args.episodes else [int(i) for i in episodes_meta.index]
    for ep_idx in wanted:
        ep = episodes_meta.loc[ep_idx]
        f0, f1 = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        duration = (f1 - f0) / fps

        sm = summaries[summaries.episode_index == ep_idx].sort_values("segment_index")
        sum_rows = [{"t0": (int(r.from_index) - f0) / fps, "t1": (int(r.to_index) - f0) / fps,
                     "text": str(r.summary)} for _, r in sm.iterrows()]
        sub_rows = [{"t0": (int(w["from_index"]) - f0) / fps, "t1": (int(w["to_index"]) - f0) / fps,
                     "text": w["subtask"]} for w in subtasks[str(ep_idx)]]

        srt = out_dir / f"ep{ep_idx:02d}.srt"
        srt.write_text(build_srt(sub_rows, sum_rows, duration))

        ins = []
        for key in (TOP, WRIST):
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                file_index=int(ep[f"videos/{key}/file_index"]),
            )
            ins += ["-ss", f"{float(ep[f'videos/{key}/from_timestamp']):.3f}",
                    "-t", f"{duration:.3f}", "-i", str(path)]

        out = out_dir / f"ep{ep_idx:02d}.mp4"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *ins,
            "-filter_complex",
            f"[0:v][1:v]hstack=inputs=2[v];"
            f"[v]subtitles={srt.name}:force_style='FontName=DejaVu Sans,FontSize=15,"
            f"Alignment=1,MarginV=12,MarginL=14,BorderStyle=3,Outline=1'[o]",
            "-map", "[o]", "-an",
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
        ]
        # cwd = out_dir so the subtitles filter gets a bare filename (no path escaping)
        print(f"ep{ep_idx}: {duration:.0f}s -> {out.name} ...", flush=True)
        r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr.strip()[:1500])
            raise SystemExit(f"ffmpeg failed on ep{ep_idx}")
        srt.unlink()
        print(f"  wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
