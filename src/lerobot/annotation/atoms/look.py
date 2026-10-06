"""Closeup: full-resolution frames from every camera of an episode, for one visual question.

    uv run python -m lerobot.annotation.atoms.look --episode E --frames 100 120 140
    uv run python -m lerobot.annotation.atoms.look --episode E --from-s 10 --to-s 13 --stride-s 0.5

Writes outputs/_annotation/subtask_atoms_review/sheets/closeup/<episode>_<frames>.jpg (path printed):
one column per frame, external camera on top, wrist below, any third camera under that,
each tile 520 px wide (the contact sheets use 256 px). Frames decode at 640 px width into
outputs/_annotation/subtask_atoms_review/frames_hd/ and are reused.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from lerobot.annotation.atoms.atoms_common import CORPUS, REVIEW, episode_cameras, episode_phases, load_corpus
from lerobot.annotation.atoms.sheets import font, label

FRAMES_HD = REVIEW / "frames_hd"
OUT = REVIEW / "sheets" / "closeup"
TILE_W = 520
PER_ROW = 3


def hd_path(episode_id: str, camera: str, frame: int) -> Path:
    return FRAMES_HD / episode_id / camera / f"f{frame:06d}.jpg"


def extract_hd(episode_id: str, camera: str, frames: list[int], width: int = 640) -> None:
    wanted = sorted({int(f) for f in frames if not hd_path(episode_id, camera, int(f)).exists()})
    if not wanted:
        return
    video = CORPUS / "episodes" / episode_id / "videos" / f"{camera}.mp4"
    out_dir = FRAMES_HD / episode_id / camera
    out_dir.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{f})" for f in wanted)
    tmp = out_dir / "tmp_%05d.jpg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select='{expr}',scale={width}:-2", "-vsync", "0", "-q:v", "2", str(tmp)],
        check=True,
    )
    produced = sorted(out_dir.glob("tmp_*.jpg"))
    if len(produced) != len(wanted):
        print(f"warning: {episode_id}/{camera}: asked {len(wanted)} frames, got {len(produced)}", file=sys.stderr)
    for path, f in zip(produced, wanted, strict=False):
        path.rename(hd_path(episode_id, camera, f))
    for path in out_dir.glob("tmp_*.jpg"):
        path.unlink()


def tile(episode_id: str, camera: str, frame: int) -> Image.Image:
    path = hd_path(episode_id, camera, frame)
    if not path.exists():
        return Image.new("RGB", (TILE_W, TILE_W * 9 // 16), (40, 40, 40))
    img = Image.open(path).convert("RGB")
    return img.resize((TILE_W, int(round(TILE_W * img.height / img.width))), Image.BILINEAR)


def render(episode_id: str, frames: list[int]) -> Path:
    episodes, _ = load_corpus()
    ep = episodes[episode_id]
    rate = float(ep["native_rate_hz"])
    info = episode_phases(episode_id, ep["embodiment"], rate, source=ep["source"])
    frames = [f for f in frames if 0 <= f < info["n"]]
    ext, wrist = episode_cameras(ep)
    cams = [ext, wrist] + [c for c in ep["cameras"] if c not in (ext, wrist)]
    for cam in cams:
        extract_hd(episode_id, cam, frames)
    heights = [tile(episode_id, cam, frames[0]).height for cam in cams]
    col_h = sum(heights) + 18
    rows = (len(frames) + PER_ROW - 1) // PER_ROW
    img = Image.new("RGB", (PER_ROW * (TILE_W + 8), 26 + rows * (col_h + 6)), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"{episode_id}   task: {ep['task'][:90]}   closeup rows: {' over '.join(cams)}   (f frame, seconds, g closedness 1 = shut)", font=font(13, True), fill="white")
    for i, f in enumerate(frames):
        x = (i % PER_ROW) * (TILE_W + 8)
        y = 26 + (i // PER_ROW) * (col_h + 6)
        for cam, h in zip(cams, heights, strict=True):
            img.paste(tile(episode_id, cam, f), (x, y))
            y += h
        label(d, (x + 2, y + 1), f"f{f} {f / rate:.2f}s g{info['closedness'][f]:.2f}", size=13, bold=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "_".join(f"f{f}" for f in frames) if len(frames) <= 6 else f"f{frames[0]}-f{frames[-1]}_n{len(frames)}"
    out = OUT / f"{episode_id}_{name}.jpg"
    img.save(out, quality=85)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=[])
    ap.add_argument("--from-s", type=float)
    ap.add_argument("--to-s", type=float)
    ap.add_argument("--stride-s", type=float, default=0.5)
    args = ap.parse_args()
    frames = list(args.frames)
    if args.from_s is not None and args.to_s is not None:
        episodes, _ = load_corpus()
        rate = float(episodes[args.episode]["native_rate_hz"])
        step = max(1, int(round(args.stride_s * rate)))
        frames += list(range(int(round(args.from_s * rate)), int(round(args.to_s * rate)) + 1, step))
    if not frames:
        ap.error("give --frames or --from-s/--to-s")
    if len(frames) > 12:
        ap.error(f"{len(frames)} frames; keep a closeup to 12 or fewer (use a larger stride)")
    print(render(args.episode, sorted(set(frames))))


if __name__ == "__main__":
    main()
