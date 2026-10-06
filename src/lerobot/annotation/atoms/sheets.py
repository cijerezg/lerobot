"""Contact sheets for the atom review: one overview strip + rows of candidate atoms.

    uv run python -m lerobot.annotation.atoms.sheets render [--source S] [--episode E] [--jobs 8]
    uv run python -m lerobot.annotation.atoms.sheets dense --episode E --from-s 10 --to-s 20 [--stride-s 0.5]

Frames are extracted with ffmpeg (select filter, one pass per video) into
outputs/_annotation/subtask_atoms_review/frames/<episode>/<camera>/f%06d.jpg and reused.
Sheets go to outputs/_annotation/subtask_atoms_review/sheets/<source>/<episode>_<k>.jpg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.annotation.atoms.atoms_common import (
    CORPUS,
    REVIEW,
    WORK,
    episode_cameras,
    episode_phases,
    load_corpus,
    read_jsonl,
)

FRAMES = REVIEW / "frames"
SHEETS = REVIEW / "sheets"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TILE_W = 256
OVER_W = 160
SHEET_W = 1600
MAX_ATOM_ROWS = 6


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def frame_path(episode_id: str, camera: str, frame: int) -> Path:
    return FRAMES / episode_id / camera / f"f{frame:06d}.jpg"


def extract_frames(episode_id: str, camera: str, frames: list[int], width: int = 320) -> None:
    """One ffmpeg pass per video: decode once, keep the selected frames, scale to `width`."""
    wanted = sorted({int(f) for f in frames if not frame_path(episode_id, camera, int(f)).exists()})
    if not wanted:
        return
    video = CORPUS / "episodes" / episode_id / "videos" / f"{camera}.mp4"
    out_dir = FRAMES / episode_id / camera
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start in range(0, len(wanted), 300):
        chunk = wanted[chunk_start : chunk_start + 300]
        expr = "+".join(f"eq(n\\,{f})" for f in chunk)
        tmp = out_dir / f"tmp_{chunk_start}_%05d.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video),
            "-vf", f"select='{expr}',scale={width}:-2",
            "-vsync", "0", "-q:v", "3", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        produced = sorted(out_dir.glob(f"tmp_{chunk_start}_*.jpg"))
        if len(produced) != len(chunk):
            # the last frame index can exceed the stream; keep what we got in order
            print(f"warning: {episode_id}/{camera}: asked {len(chunk)} frames, got {len(produced)}", file=sys.stderr)
        for path, f in zip(produced, chunk, strict=False):
            path.rename(frame_path(episode_id, camera, f))
        for path in out_dir.glob(f"tmp_{chunk_start}_*.jpg"):
            path.unlink()


def load_tile(episode_id: str, camera: str, frame: int, width: int) -> Image.Image:
    path = frame_path(episode_id, camera, frame)
    if not path.exists():
        img = Image.new("RGB", (width, int(width * 9 / 16)), (40, 40, 40))
        return img
    img = Image.open(path).convert("RGB")
    h = int(round(width * img.height / img.width))
    return img.resize((width, h), Image.BILINEAR)


def label(draw: ImageDraw.ImageDraw, xy, text, size=13, fill="white", bg="black", bold=False):
    f = font(size, bold)
    x, y = xy
    w = draw.textlength(text, font=f)
    draw.rectangle((x - 2, y - 1, x + w + 2, y + size + 3), fill=bg)
    draw.text((x, y), text, font=f, fill=fill)


def trace_image(info, start: int, stop: int, rate: float, atoms, width: int, height: int = 110) -> Image.Image:
    img = Image.new("RGB", (width, height), (24, 24, 24))
    d = ImageDraw.Draw(img)
    c = info["closedness"][start:stop]
    v = info["speed"][start:stop]
    vmax = max(float(v.max()), 1e-6)
    n = stop - start
    xs = np.linspace(0, width - 1, n)
    top, bottom = 8, height - 22
    for k in range(1, n):
        d.line((xs[k - 1], bottom - (bottom - top) * v[k - 1] / vmax, xs[k], bottom - (bottom - top) * v[k] / vmax), fill=(110, 110, 110))
    for k in range(1, n):
        d.line((xs[k - 1], bottom - (bottom - top) * c[k - 1], xs[k], bottom - (bottom - top) * c[k]), fill=(80, 160, 255), width=2)
    for a in atoms:
        x = xs[min(n - 1, a["start_timestep"] - start)]
        col = {"grasp": (255, 210, 80), "move": (120, 230, 120), "release": (255, 120, 120), "return": (200, 200, 200)}.get(a["verb"], (255, 255, 255))
        d.line((x, top, x, bottom), fill=col, width=2)
        d.text((x + 2, top), a["verb"][:3], font=font(11), fill=col)
    # seconds axis
    total = n / rate
    tick = 2 if total <= 40 else (5 if total <= 100 else 10)
    t = 0
    while t <= total:
        x = xs[min(n - 1, int(t * rate))]
        d.line((x, bottom, x, bottom + 4), fill=(200, 200, 200))
        d.text((x + 1, bottom + 5), f"{(start / rate) + t:.0f}s", font=font(10), fill=(200, 200, 200))
        t += tick
    d.text((4, 2), "blue: gripper closedness   grey: joint speed", font=font(10), fill=(170, 170, 170))
    return img


DENSE_TASKS = {"press_the_button", "turn_on_the_light_switch", "water_the_flowers", "wipe_the_table"}


def overview_stride_s(start: int, stop: int, rate: float, dense: bool = False) -> float:
    total = (stop - start) / rate
    if dense:
        return 1.0
    return 1.0 if total <= 45 else (2.0 if total <= 90 else 3.0)


def overview_frames(start: int, stop: int, rate: float, dense: bool = False) -> list[int]:
    stride = overview_stride_s(start, stop, rate, dense)
    frames = list(range(start, stop, int(round(stride * rate))))
    if frames[-1] != stop - 1:
        frames.append(stop - 1)
    return frames


def wants_wrist_overview(ep: dict) -> bool:
    return ep["source"] != "robochallenge" or ep["component"] in DENSE_TASKS


def needed_frames(proposals: list[dict]) -> list[int]:
    frames = set()
    for p in proposals:
        dense = p["component"] in DENSE_TASKS
        frames.update(overview_frames(p["parent_start"], p["parent_end"], p["native_rate_hz"], dense))
        for a in p["atoms"]:
            a0, a1 = a["start_timestep"], a["end_timestep_exclusive"]
            frames.update([a0, (a0 + a1) // 2, a1 - 1])
        for f in p["failed_closes"]:
            frames.update([f["close_frame"], f["open_frame"] or f["close_frame"]])
    return sorted(frames)


def render_episode(episode_id: str) -> list[Path]:
    episodes, parents = load_corpus()
    ep = episodes[episode_id]
    proposals = [p for p in read_jsonl(WORK / "proposals.jsonl") if p["episode_id"] == episode_id]
    rate = float(ep["native_rate_hz"])
    ext, wrist = episode_cameras(ep)
    frames = needed_frames(proposals)
    extract_frames(episode_id, ext, frames)
    extract_frames(episode_id, wrist, frames)
    info = episode_phases(episode_id, ep["embodiment"], rate, source=ep["source"])
    out_dir = SHEETS / ep["source"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{episode_id}_*.jpg"):
        old.unlink()
    written = []
    sheet_index = 0
    both = wants_wrist_overview(ep)
    for p in proposals:
        start, stop = p["parent_start"], p["parent_end"]
        atoms = p["atoms"]
        header = [
            f"{episode_id}   source={ep['source']}/{ep['component']}   {ep['embodiment']} @ {rate:g} Hz   task: {ep['task']}",
            f"PARENT {p['parent_interval_index']}  [{start},{stop})  {start / rate:.1f}-{stop / rate:.1f} s  quality {p['parent_quality']}  eligible={p['critic_eligible']}   text: \"{p['parent_subtask']}\"",
        ]
        ev = p["parent_events"]
        if ev["mistake_events"]:
            header.append("  reviewed mistakes: " + "; ".join(f"{e['kind']} {e['start_s']:.1f}-{e['end_s']:.1f}s ({e.get('note', '')[:70]})" for e in ev["mistake_events"]))
        if ev["recovery_events"] or ev["pause_events"] or ev["interruption_events"]:
            header.append(
                "  pauses: " + ", ".join(f"{e['start_s']:.0f}-{e['end_s']:.0f}" for e in ev["pause_events"])
                + "   interruptions: " + ", ".join(f"{e['start_s']:.0f}-{e['end_s']:.0f} {e['reason']}" for e in ev["interruption_events"])
                + "   recoveries: " + ", ".join(f"{e['start_s']:.0f}-{e['end_s']:.0f}" for e in ev["recovery_events"])
            )
        if p["failed_closes"]:
            header.append("  proprio failed closes (candidate mistakes, confirm in frames): " + ", ".join(f"close f{f['close_frame']} ({f['close_frame'] / rate:.1f}s) open f{f['open_frame']} disp {f['disp_rad']:.2f} rad" for f in p["failed_closes"]))
        if p.get("closed_on_nothing"):
            header.append("  proprio: gripper fully shut while 'carrying' (nothing between the fingers?) in cycles " + ", ".join(str(k) for k in p["closed_on_nothing"]))
        header.append("  gripper events: " + ", ".join(f"{e['kind']} f{e['frame']} ({e['frame'] / rate:.1f}s) {e['before']:.2f}->{e['after']:.2f}" + ("*" if e["secondary"] else "") for e in p["gripper_events"]))
        head_h = 16 * len(header) + 8

        # ---- page A: header + trace + overview strip
        dense = ep["component"] in DENSE_TASKS
        over = overview_frames(start, stop, rate, dense)
        per_row = SHEET_W // OVER_W
        over_h = load_tile(episode_id, ext, over[0], OVER_W).height
        rows_per_cam = (len(over) + per_row - 1) // per_row
        block_h = (2 * over_h + 16) if both else (over_h + 14)
        ov_h = rows_per_cam * block_h + 20
        H = head_h + 110 + ov_h + 8
        img = Image.new("RGB", (SHEET_W, H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        y = 4
        for i, line in enumerate(header):
            d.text((6, y), line, font=font(13, bold=(i < 2)), fill=(255, 255, 255) if i < 2 else (220, 220, 160))
            y += 16
        y += 4
        img.paste(trace_image(info, start, stop, rate, atoms, SHEET_W - 12), (6, y))
        y += 110
        stride = overview_stride_s(start, stop, rate, dense)
        d.text((6, y + 2), f"OVERVIEW every {stride:g} s: {ext}" + (f" over {wrist}" if both else "") + "   (frame index f, seconds, gripper closedness g: 1 = shut)", font=font(12, True), fill=(180, 220, 255))
        y += 18
        for i, f in enumerate(over):
            x = (i % per_row) * OVER_W
            yy = y + (i // per_row) * block_h
            img.paste(load_tile(episode_id, ext, f, OVER_W), (x, yy))
            if both:
                img.paste(load_tile(episode_id, wrist, f, OVER_W), (x, yy + over_h))
                label(d, (x + 2, yy + 2 * over_h - 2), f"f{f} {f / rate:.1f}s g{info['closedness'][f]:.2f}", size=10)
            else:
                label(d, (x + 2, yy + over_h - 2), f"f{f} {f / rate:.1f}s g{info['closedness'][f]:.2f}", size=10)
        path = out_dir / f"{episode_id}_{sheet_index:02d}.jpg"
        img.save(path, quality=82)
        written.append(path)
        sheet_index += 1

        # ---- pages B..: candidate atom rows
        tile_h = load_tile(episode_id, ext, start, TILE_W).height
        row_h = tile_h + 40
        chunks = [atoms[i : i + MAX_ATOM_ROWS] for i in range(0, len(atoms), MAX_ATOM_ROWS)]
        for ci, chunk in enumerate(chunks):
            H = head_h + 24 + len(chunk) * row_h + 10
            img = Image.new("RGB", (SHEET_W, H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            y = 4
            for i, line in enumerate(header[:2]):
                d.text((6, y), line, font=font(13, bold=True), fill=(255, 255, 255))
                y += 16
            y += 4
            d.text((6, y + 4), f"CANDIDATE ATOMS {ci + 1}/{len(chunks)}   columns: {ext} start | mid | end-1   ||   {wrist} start | mid | end-1", font=font(12, True), fill=(180, 220, 255))
            y += 24
            for a in chunk:
                a0, a1 = a["start_timestep"], a["end_timestep_exclusive"]
                mid = (a0 + a1) // 2
                idx = atoms.index(a)
                text = f"atom {idx}: [{a0},{a1}) {a0 / rate:.1f}-{a1 / rate:.1f}s ({(a1 - a0) / rate:.1f}s)  candidate verb: {a['verb'].upper()}  cycle {a['cycle']}  start from {a['start_provenance']}"
                d.text((6, y), text, font=font(13, True), fill=(255, 230, 120))
                y += 18
                for k, (cam, f) in enumerate([(ext, a0), (ext, mid), (ext, a1 - 1), (wrist, a0), (wrist, mid), (wrist, a1 - 1)]):
                    x = k * (TILE_W + 8) + (10 if k >= 3 else 0)
                    img.paste(load_tile(episode_id, cam, f, TILE_W), (x, y))
                    label(d, (x + 2, y + tile_h - 16), f"f{f} {f / rate:.1f}s g{info['closedness'][f]:.2f}", size=11)
                y += tile_h + 22
            path = out_dir / f"{episode_id}_{sheet_index:02d}.jpg"
            img.save(path, quality=82)
            written.append(path)
            sheet_index += 1
    return written


def render_dense(episode_id: str, from_s: float, to_s: float, stride_s: float, out: Path | None = None) -> Path:
    episodes, _ = load_corpus()
    ep = episodes[episode_id]
    rate = float(ep["native_rate_hz"])
    ext, wrist = episode_cameras(ep)
    info = episode_phases(episode_id, ep["embodiment"], rate, source=ep["source"])
    n = info["n"]
    f0, f1 = max(0, int(round(from_s * rate))), min(n, int(round(to_s * rate)))
    step = max(1, int(round(stride_s * rate)))
    frames = list(range(f0, f1, step))
    extract_frames(episode_id, ext, frames)
    extract_frames(episode_id, wrist, frames)
    per_row = 8
    w = SHEET_W // per_row
    th = load_tile(episode_id, ext, frames[0], w).height
    rows = (len(frames) + per_row - 1) // per_row
    img = Image.new("RGB", (SHEET_W, 24 + rows * (2 * th + 16)), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((6, 4), f"{episode_id}  dense strip {from_s:.1f}-{to_s:.1f}s every {stride_s}s   rows: {ext} over {wrist}", font=font(13, True), fill="white")
    for i, f in enumerate(frames):
        x = (i % per_row) * w
        y = 24 + (i // per_row) * (2 * th + 16)
        img.paste(load_tile(episode_id, ext, f, w), (x, y))
        img.paste(load_tile(episode_id, wrist, f, w), (x, y + th))
        label(d, (x + 2, y + 2 * th - 14), f"f{f} {f / rate:.2f}s g{info['closedness'][f]:.2f}", size=11)
    out = out or (SHEETS / "dense" / f"{episode_id}_{from_s:.1f}_{to_s:.1f}_{stride_s}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=82)
    return out


def _render(eid):
    try:
        paths = render_episode(eid)
        return eid, [str(p) for p in paths], None
    except Exception as exc:  # noqa: BLE001
        return eid, [], repr(exc)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--source", default=None)
    r.add_argument("--episode", default=None)
    r.add_argument("--jobs", type=int, default=8)
    dn = sub.add_parser("dense")
    dn.add_argument("--episode", required=True)
    dn.add_argument("--from-s", type=float, required=True)
    dn.add_argument("--to-s", type=float, required=True)
    dn.add_argument("--stride-s", type=float, default=0.5)
    args = ap.parse_args()
    if args.cmd == "dense":
        print(render_dense(args.episode, args.from_s, args.to_s, args.stride_s))
        return
    episodes, _ = load_corpus()
    ids = [e for e, ep in episodes.items() if (args.source is None or ep["source"] == args.source) and (args.episode is None or e == args.episode)]
    manifest = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for eid, paths, err in pool.map(_render, ids):
            if err:
                print("ERROR", eid, err)
            else:
                manifest[eid] = paths
                print(eid, len(paths), "sheets")
    mpath = SHEETS / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    old.update(manifest)
    mpath.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
