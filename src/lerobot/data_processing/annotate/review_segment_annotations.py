#!/usr/bin/env python

"""
Watchable review videos for the per-segment quality + mistake annotations.

Reads what training actually consumes — `meta/episode_metadata.parquet`,
`meta/mistakes.parquet` and `meta/subtask_windows.json` of a relabelled dataset — so
what you watch is the artefact, not a re-derivation of it.

Each episode renders as one plain H.264 mp4: top and wrist side by side, the current
subtask + quality burned in at the top, a red MISTAKE banner while a mistake span is
live, and an always-visible quality timeline for the whole episode along the bottom
with a playhead. Sampling a few episodes is the point, so `--sample N` picks a spread
that is half worst-offenders and half clean.

    # 6 episodes spread across all four datasets — the usual entry point
    python review_segment_annotations.py --sample 6

    # everything in one dataset, or specific episodes
    python review_segment_annotations.py --data-dir outputs/rebot_val-annotated-v2
    python review_segment_annotations.py --data-dir outputs/rebot_val-annotated-v2 --episodes 0 2

    # just tell me what is in there, render nothing
    python review_segment_annotations.py --sample 6 --list
"""

import argparse
import glob
import json
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TOP, WRIST = "observation.images.top", "observation.images.wrist"
DEFAULT_DATASETS = [
    "outputs/rebot_socks_basket-annotated-v2",
    "outputs/rebot_shirts_bin-annotated-v2",
    "outputs/rebot_two_container-annotated-v2",
    "outputs/rebot_val-annotated-v2",
]
QCOLOR = {5: (46, 125, 50), 4: (124, 179, 66), 3: (249, 168, 37), 2: (239, 108, 0), 1: (198, 40, 40)}
MISTAKE_COLOR = (255, 23, 68)
BAR_H, TICK_H = 46, 14
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WRAP = 78


def load(root: Path):
    info = json.load(open(root / "meta" / "info.json"))
    segs = pd.read_parquet(root / "meta" / "episode_metadata.parquet")
    mis = pd.read_parquet(root / "meta" / "mistakes.parquet")
    eps = pd.concat([
        pd.read_parquet(f)
        for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))
    ]).set_index("episode_index")
    return info, float(info["fps"]), segs, mis, eps


def ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def build_srt(segs: pd.DataFrame, mis: pd.DataFrame, f0: int, fps: float, duration: float) -> str:
    """One cue per interval where neither the segment nor the mistake state changes."""
    edges = {0.0, duration}
    for _, r in segs.iterrows():
        edges |= {(int(r.from_index) - f0) / fps, (int(r.to_index) - f0) / fps}
    for _, r in mis.iterrows():
        edges |= {(int(r.from_index) - f0) / fps, (int(r.to_index) - f0) / fps}
    edges = sorted(e for e in edges if 0.0 <= e <= duration)

    out, n = [], 0
    for a, b in zip(edges, edges[1:]):
        if b - a < 0.05:
            continue
        f = f0 + int(a * fps) + 1
        seg = segs[(segs.from_index <= f) & (segs.to_index > f)]
        m = mis[(mis.from_index <= f) & (mis.to_index > f)]
        if seg.empty:
            continue
        s = seg.iloc[0]
        lines = [f"seg {int(s.segment_index)}  |  {s.subtask}  |  quality {int(s.quality)}/5"]
        if not m.empty:
            lines.append(f">>> MISTAKE: {m.iloc[0].mistake_type} <<<")
        lines += textwrap.wrap(str(s.note), WRAP)[:2]
        n += 1
        out.append(f"{n}\n{ts(a)} --> {ts(b)}\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def build_timeline(segs, mis, f0, f1, width, path):
    """Whole-episode quality track: one block per segment, mistake ticks above."""
    img = Image.new("RGB", (width, BAR_H), (18, 18, 18))
    d = ImageDraw.Draw(img)
    span = max(f1 - f0, 1)

    def x(f):
        return int((f - f0) / span * (width - 1))

    for _, r in segs.iterrows():
        x0, x1 = x(int(r.from_index)), x(int(r.to_index))
        d.rectangle([x0, TICK_H, max(x1 - 1, x0), BAR_H - 1], fill=QCOLOR[int(r.quality)])
        if x1 - x0 > 26:
            d.text((x0 + 3, TICK_H + 2), str(int(r.quality)), fill=(255, 255, 255),
                   font=ImageFont.truetype(FONT, 13))
    for _, r in mis.iterrows():
        x0, x1 = x(int(r.from_index)), x(int(r.to_index))
        d.rectangle([x0, 0, max(x1 - 1, x0 + 2), TICK_H - 2], fill=MISTAKE_COLOR)
    img.save(path)


def render(root: Path, ep_idx: int, out_dir: Path, crf: int, preset: str):
    # ffmpeg runs with cwd=out_dir so the subtitles filter gets a bare filename (no
    # path escaping); every dataset path therefore has to be absolute.
    root, out_dir = root.resolve(), out_dir.resolve()
    info, fps, segs, mis, eps = load(root)
    ep = eps.loc[ep_idx]
    f0, f1 = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    duration = (f1 - f0) / fps
    s = segs[segs.episode_index == ep_idx].sort_values("segment_index")
    m = mis[mis.episode_index == ep_idx] if len(mis) else mis

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{root.name}__ep{ep_idx:02d}"
    srt = out_dir / f"{stem}.srt"
    srt.write_text(build_srt(s, m, f0, fps, duration))
    bar = out_dir / f"{stem}.bar.png"
    build_timeline(s, m, f0, f1, 1280, bar)
    head = out_dir / f"{stem}.head.png"
    Image.new("RGB", (3, BAR_H), (255, 255, 255)).save(head)

    ins = []
    for key in (TOP, WRIST):
        path = root / info["video_path"].format(
            video_key=key,
            chunk_index=int(ep[f"videos/{key}/chunk_index"]),
            file_index=int(ep[f"videos/{key}/file_index"]),
        )
        ins += ["-ss", f"{float(ep[f'videos/{key}/from_timestamp']):.3f}", "-t", f"{duration:.3f}",
                "-i", str(path)]
    ins += ["-i", str(bar), "-i", str(head)]

    out = out_dir / f"{stem}.mp4"
    style = ("FontName=DejaVu Sans,FontSize=15,Alignment=7,MarginV=8,MarginL=12,"
             "BorderStyle=3,Outline=1,Shadow=0")
    filt = (
        f"[0:v]scale=640:480[a];[1:v]scale=640:480[b];[a][b]hstack=inputs=2[v];"
        f"[v]pad=1280:{480 + BAR_H}:0:0:color=0x121212[p];"
        f"[p][2:v]overlay=0:480[q];"
        # The playhead is an overlay, not a drawbox: drawbox evaluates its x once at
        # configuration and the marker never moves. overlay re-evaluates x per frame.
        # No parentheses either — filter_complex's parser eats them inside expressions,
        # and the canvas is a fixed 1280 wide so (W-3) folds to a constant.
        f"[q][3:v]overlay=x='{1280 - 3}*t/{duration:.3f}':y=480[r];"
        f"[r]subtitles={srt.name}:force_style='{style}'[o]"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *ins,
           "-filter_complex", filt, "-map", "[o]", "-an",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr.strip()[:1200])
        raise SystemExit(f"ffmpeg failed on {stem}")
    srt.unlink()
    bar.unlink()
    head.unlink()
    return out, duration, len(s), len(m)


def episode_table(datasets):
    rows = []
    for ds in datasets:
        root = Path(ds)
        if not root.exists():
            continue
        _info, fps, segs, mis, eps = load(root)
        for ep_idx in eps.index:
            s = segs[segs.episode_index == ep_idx]
            nm = len(mis[mis.episode_index == ep_idx]) if len(mis) else 0
            rows.append({
                "root": root, "ep": int(ep_idx), "segments": len(s), "mistakes": nm,
                "worst": int(s.quality.min()), "mean_q": round(float(s.quality.mean()), 2),
                "minutes": round(float((eps.loc[ep_idx, "dataset_to_index"]
                                        - eps.loc[ep_idx, "dataset_from_index"]) / fps / 60), 1),
            })
    return rows


def spread(pool, k):
    """Take k, round-robin over datasets so one dataset cannot fill the whole slate."""
    by: dict = {}
    for r in pool:
        by.setdefault(r["root"], []).append(r)
    out = []
    while len(out) < k and any(by.values()):
        for root in list(by):
            if len(out) >= k:
                break
            if by[root]:
                out.append(by[root].pop(0))
    return out


def pick(rows, n):
    """Half worst-offenders, half clean — the two things worth eyeballing."""
    dirty = sorted([r for r in rows if r["mistakes"]], key=lambda r: -r["mistakes"])
    clean = sorted([r for r in rows if not r["mistakes"]], key=lambda r: -r["mean_q"])
    want = (n + 1) // 2
    return (spread(dirty, want) + spread(clean, n - want))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/_annotation/review"))
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--preset", default="veryfast")
    args = ap.parse_args()

    datasets = [args.data_dir] if args.data_dir else DEFAULT_DATASETS
    rows = episode_table(datasets)

    if args.data_dir and args.episodes:
        todo = [r for r in rows if r["ep"] in args.episodes]
    elif args.data_dir and not args.sample:
        todo = rows
    else:
        todo = pick(rows, args.sample or 6)

    print(f"{'dataset':34s} {'ep':>3s} {'min':>5s} {'segs':>5s} {'mist':>5s} {'worst':>6s} {'mean q':>7s}")
    for r in todo:
        print(f"{r['root'].name:34s} {r['ep']:3d} {r['minutes']:5.1f} {r['segments']:5d} "
              f"{r['mistakes']:5d} {r['worst']:6d} {r['mean_q']:7.2f}")
    if args.list:
        return

    print()
    for r in todo:
        out, dur, ns, nm = render(r["root"], r["ep"], args.out, args.crf, args.preset)
        print(f"  {out}  ({dur:.0f}s, {ns} segments, {nm} mistakes, "
              f"{out.stat().st_size / 1e6:.0f} MB)")
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
