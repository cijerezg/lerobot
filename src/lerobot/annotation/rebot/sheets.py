"""Contact sheets (top over wrist) for one episode of any dataset root, 8 tiles per sheet.

    uv run python -m lerobot.annotation.rebot.sheets <root> <out dir> <episode> --step 90
    uv run python -m lerobot.annotation.rebot.sheets <root> <out dir> <episode> --start 1050 --end 1400 --step 30 --tag fine

Writes <out dir>/ep<E>_<tag>_<first>-<last>.png. Without --end the episode's last frame is added.
"""

import argparse
from pathlib import Path

from lerobot.annotation.rebot.episode import record, sheet

PER_SHEET = 8

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("episode", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--tag", default="sheet")
    args = ap.parse_args()
    rec = record(args.root, args.episode)
    frames = list(range(args.start, min(args.end or rec["frames"], rec["frames"]), args.step))
    if args.end is None:
        frames = sorted(set(frames + [rec["frames"] - 1]))
    for p in range(0, len(frames), PER_SHEET):
        part = frames[p : p + PER_SHEET]
        name = f"ep{args.episode}_{args.tag}_{part[0]:05d}-{part[-1]:05d}.png"
        print(sheet(rec, part, Path(args.out) / name, cols=4, size=(400, 300)))
