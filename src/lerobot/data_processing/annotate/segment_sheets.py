#!/usr/bin/env python

"""
Contact sheets for the semantic segments, one image per unit of judgement.

The rubric (pi07_wiki/annotation_rubric.md) grades whole segments from video, so the
sheet is the annotation instrument: a segment's frames sampled over time, top over
wrist, with the frame index / elapsed time / gripper value burned into every tile.

Density follows the rubric's tiers, bounded by what survives image downscaling —
sheets stay under ~2000x1200, so a tile arrives near its native 256x192:

    grasp    8 cols, 1 segment/sheet, + a zoomed strip per flagged failed close
    move     5 cols, 3 segments/sheet
    release  5 cols, 4 segments/sheet
    return   5 cols, 6 segments/sheet, top only

Frames come out of one ffmpeg pass per (episode, camera) — the videos are AV1, so
seeking per frame costs more than a single linear decode with a select filter.

    python segment_sheets.py --data-dir outputs/rebot_two_container-v1 --out SHEETS
    python segment_sheets.py --data-dir DS --out SHEETS --episodes 3 4
"""

import argparse
import glob
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TOP, WRIST = "observation.images.top", "observation.images.wrist"
GRIPPER_DIM, PAN_DIM = 6, 0
CLOSE_ON, CLOSE_OFF = -60.0, -90.0
MIN_CLOSED_FRAMES, DISP_MAX = 15, 32.0
TW, TH = 256, 192
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

COLS = {"grasp": 8, "move": 5, "release": 5, "return": 5}
PER_SHEET = {"grasp": 1, "move": 3, "release": 4, "return": 6}
CAMERAS = {"grasp": (TOP, WRIST), "move": (TOP, WRIST), "release": (TOP, WRIST), "return": (TOP,)}


def load_dataset(root: Path):
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))]
    )
    state = np.stack(data["observation.state"].values).astype(np.float32)
    eps = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))]
    ).sort_values("episode_index").set_index("episode_index")
    return state, eps


def closed_runs(grip):
    runs, start, closed = [], None, False
    for i, g in enumerate(grip):
        closed = g > CLOSE_ON if not closed else g >= CLOSE_OFF
        if closed and start is None:
            start = i
        elif not closed and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(grip)))
    return [r for r in runs if r[1] - r[0] >= MIN_CLOSED_FRAMES]


def phase_of(subtask: str) -> str:
    return "return" if subtask.startswith("return") else subtask.split()[0]


def segment_features(state, a, b):
    arm = state[a:b, :GRIPPER_DIM]
    path = float(np.linalg.norm(np.diff(arm, axis=0), axis=1).sum())
    net = float(np.linalg.norm(arm[-1] - arm[0]))
    pan = state[a:b, PAN_DIM]
    rev = 0
    if len(pan) > 30:
        sm = np.convolve(pan, np.ones(15) / 15, mode="valid")
        dv = np.sign(np.diff(sm))
        dv = dv[dv != 0]
        rev = int((np.diff(dv) != 0).sum()) if len(dv) > 1 else 0
    return {
        "dur": round((b - a) / 30.0, 1),
        "eff": round(net / path, 3) if path > 1e-6 else 0.0,
        "rev": rev,
        "pan_range": [round(float(pan.min()), 1), round(float(pan.max()), 1)],
    }


def build_plan(root: Path, episodes=None):
    """Segments + features + flagged failed closes, grouped into sheets."""
    state, eps = load_dataset(root)
    windows = json.load(open(root / "meta" / "subtask_windows.semantic.json"))["episodes"]
    ds = root.name

    segments, sheets = [], []
    for ep_idx in sorted(int(k) for k in windows):
        if episodes and ep_idx not in episodes:
            continue
        lo = int(eps.loc[ep_idx, "dataset_from_index"])
        hi = int(eps.loc[ep_idx, "dataset_to_index"])
        runs = [(lo + a, lo + b) for a, b in closed_runs(state[lo:hi, GRIPPER_DIM])]

        by_phase = {}
        for si, w in enumerate(windows[str(ep_idx)]):
            a, b = int(w["from_index"]), int(w["to_index"])
            ph = phase_of(w["subtask"])
            fails = []
            for ca, cb in runs:
                if ca >= a and cb <= b:
                    seg = state[ca:cb, :GRIPPER_DIM]
                    if float(np.linalg.norm(seg - seg[0], axis=1).max()) <= DISP_MAX:
                        fails.append([ca, cb])
            seg = {
                "ds": ds, "ep": ep_idx, "seg": si, "from": a, "to": b,
                "subtask": w["subtask"], "phase": ph, "fails": fails,
                **segment_features(state, a, b),
            }
            segments.append(seg)
            by_phase.setdefault(ph, []).append(seg)

        for ph, segs in by_phase.items():
            if ph == "grasp":
                # A grasp with a flagged close needs its own sheet: the overview strip
                # plus a zoom per close already fills the height budget. Failure-free
                # grasps are one short strip each, so they group like the other phases.
                for s in (x for x in segs if x["fails"]):
                    strips = [{"from": s["from"], "to": s["to"], "cols": COLS[ph], "seg": s["seg"], "zoom": False}]
                    for ca, cb in s["fails"][:3]:
                        strips.append({"from": max(s["from"], ca - 45), "to": min(s["to"], cb + 45),
                                       "cols": COLS[ph], "seg": s["seg"], "zoom": True})
                    sheets.append({"ds": ds, "ep": ep_idx, "phase": ph, "segs": [s["seg"]],
                                   "cameras": CAMERAS[ph], "strips": strips})
                clean = [x for x in segs if not x["fails"]]
                for i in range(0, len(clean), 3):
                    group = clean[i : i + 3]
                    sheets.append({
                        "ds": ds, "ep": ep_idx, "phase": ph, "segs": [g["seg"] for g in group],
                        "cameras": CAMERAS[ph],
                        "strips": [{"from": g["from"], "to": g["to"], "cols": COLS[ph],
                                    "seg": g["seg"], "zoom": False} for g in group],
                    })
            else:
                for i in range(0, len(segs), PER_SHEET[ph]):
                    group = segs[i : i + PER_SHEET[ph]]
                    sheets.append({
                        "ds": ds, "ep": ep_idx, "phase": ph, "segs": [g["seg"] for g in group],
                        "cameras": CAMERAS[ph],
                        "strips": [{"from": g["from"], "to": g["to"], "cols": COLS[ph],
                                    "seg": g["seg"], "zoom": False} for g in group],
                    })

    for i, sh in enumerate(sheets):
        sh["name"] = f"{sh['ds']}__ep{sh['ep']:02d}__{sh['phase']}__{'-'.join(str(s) for s in sh['segs'])}.png"
    return segments, sheets, state, eps


def sample_indices(a, b, ncols):
    """Uniform over the closed range [a, b-1] — `to_index` is exclusive, and for the
    last segment of an episode frame `b` belongs to the next episode (or past the
    end of the video file)."""
    last = b - 1
    return [a + (last - a) * i // max(ncols - 1, 1) for i in range(ncols)] if ncols > 1 else [a]


def extract_episode(args):
    """One linear ffmpeg decode per (episode, camera) -> {global_frame: png path}."""
    root, ep_idx, key, wanted, lo, video_path, from_ts, cache = args
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    offset = int(round(from_ts * 30.0))
    todo = sorted({int(g) for g in wanted})
    local = [g - lo + offset for g in todo]

    tmp = Path(tempfile.mkdtemp(dir=cache))
    expr = "+".join(f"eq(n\\,{n})" for n in local)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
         "-vf", f"select='{expr}'", "-vsync", "0", "-f", "image2", str(tmp / "%05d.png")],
        check=True,
    )
    produced = sorted(tmp.glob("*.png"))
    if len(produced) != len(todo):
        raise RuntimeError(f"ep{ep_idx} {key}: asked {len(todo)} frames, ffmpeg gave {len(produced)}")
    out = {}
    for g, p in zip(todo, produced):
        dest = cache / f"{key.split('.')[-1]}_{g}.png"
        shutil.move(str(p), dest)
        out[g] = str(dest)
    shutil.rmtree(tmp)
    return ep_idx, key, out


def compose(sheet, frames, state, fps, out_path):
    cams = sheet["cameras"]
    rows_per_strip = len(cams)
    widths = [s["cols"] * TW for s in sheet["strips"]]
    header = 18
    W = max(widths)
    H = sum(rows_per_strip * TH + header for s in sheet["strips"])
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 13)
    small = ImageFont.truetype(FONT, 11)

    y = 0
    for strip in sheet["strips"]:
        idxs = sample_indices(strip["from"], strip["to"], strip["cols"])
        tag = (f"seg{strip['seg']} [{strip['from']},{strip['to']}) "
               f"{(strip['to'] - strip['from']) / fps:.1f}s" + ("  ZOOM: failed close" if strip["zoom"] else ""))
        draw.text((4, y + 3), tag, fill=(255, 220, 90), font=font)
        y += header
        for c, g in enumerate(idxs):
            for r, key in enumerate(cams):
                img = Image.open(frames[key][g]).resize((TW, TH))
                canvas.paste(img, (c * TW, y + r * TH))
            t = (g - strip["from"]) / fps
            lab = f"{g}  t{t:.1f}  g{state[g, GRIPPER_DIM]:.0f}"
            draw.rectangle([c * TW, y, c * TW + 7 * len(lab), y + 14], fill=(0, 0, 0))
            draw.text((c * TW + 2, y + 1), lab, fill=(120, 255, 120), font=small)
        y += rows_per_strip * TH
    canvas.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    root, out = args.data_dir, args.out
    out.mkdir(parents=True, exist_ok=True)
    info = json.load(open(root / "meta" / "info.json"))
    fps = float(info["fps"])

    segments, sheets, state, eps = build_plan(root, args.episodes)
    print(f"{root.name}: {len(segments)} segments -> {len(sheets)} sheets")

    need = {}
    for sh in sheets:
        for strip in sh["strips"]:
            for g in sample_indices(strip["from"], strip["to"], strip["cols"]):
                need.setdefault((sh["ep"], tuple(sh["cameras"])), set()).add(g)

    jobs, cache = [], out / "_frames"
    per_ep_cams = {}
    for (ep_idx, cams), gs in need.items():
        for key in cams:
            per_ep_cams.setdefault((ep_idx, key), set()).update(gs)
    for (ep_idx, key), gs in sorted(per_ep_cams.items()):
        ep = eps.loc[ep_idx]
        vp = root / info["video_path"].format(
            video_key=key, chunk_index=int(ep[f"videos/{key}/chunk_index"]),
            file_index=int(ep[f"videos/{key}/file_index"]))
        jobs.append((str(root), ep_idx, key, gs, int(ep["dataset_from_index"]), str(vp),
                     float(ep[f"videos/{key}/from_timestamp"]), str(cache / f"ep{ep_idx:02d}")))

    frames = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ep_idx, key, mapping in pool.map(extract_episode, jobs):
            frames.setdefault(ep_idx, {})[key] = mapping
            print(f"  ep{ep_idx:02d} {key.split('.')[-1]}: {len(mapping)} frames", flush=True)

    for sh in sheets:
        compose(sh, frames[sh["ep"]], state, fps, out / sh["name"])
    shutil.rmtree(cache, ignore_errors=True)   # ~11 GB of 640x480 PNGs across the corpus
    print(f"composed {len(sheets)} sheets -> {out}")

    index = {"dataset": root.name, "fps": fps, "segments": segments,
             "sheets": [{k: v for k, v in sh.items() if k != "cameras"} for sh in sheets]}
    json.dump(index, open(out / f"index__{root.name}.json", "w"), indent=1)
    print(f"wrote {out / f'index__{root.name}.json'}")


if __name__ == "__main__":
    main()
