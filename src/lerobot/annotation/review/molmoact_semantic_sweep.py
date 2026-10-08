"""Render resumable MolmoAct semantic-sweep evidence on Spark.

This renderer is evidence-only. It reads the active diverse-v3 store and the audit
candidate manifest, then writes review sheets under an explicitly supplied results
directory. It never edits dataset sidecars.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


CAMERAS = ("primary", "secondary", "wrist")
TILE_W, TILE_H = 240, 180


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _frame(video: Path, index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = max(0, min(index, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {video} frame {index}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((TILE_W, TILE_H), Image.Resampling.LANCZOS)


def _sheet(episode_dir: Path, frames: list[int], title: str, out: Path) -> None:
    header = 58
    canvas = Image.new("RGB", (TILE_W * len(frames), header + TILE_H * len(CAMERAS)), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="white", font=_font(18))
    draw.text((8, 33), "frames: " + ", ".join(map(str, frames)), fill="#cccccc", font=_font(14))
    for row, camera in enumerate(CAMERAS):
        video = episode_dir / "videos" / f"{camera}.mp4"
        for col, frame_index in enumerate(frames):
            tile = _frame(video, frame_index)
            x, y = col * TILE_W, header + row * TILE_H
            canvas.paste(tile, (x, y))
            d = ImageDraw.Draw(canvas)
            d.rectangle((x, y, x + 92, y + 20), fill=(0, 0, 0))
            d.text((x + 3, y + 2), f"{camera} f{frame_index}", fill="white", font=_font(12))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    canvas.save(tmp, quality=88, optimize=True)
    os.replace(tmp, out)


def _render_one(item: dict, corpus: Path, out_dir: Path, kind: str) -> dict:
    eid = item["episode_id"]
    episode_dir = corpus / "episodes" / eid
    if kind == "gripper":
        boundary = int(item["boundary_frame"])
        settle = item.get("settle_frame_estimate")
        settle = boundary + 3 if settle is None else int(settle)
        frames = sorted(set(max(0, x) for x in (boundary - 2, boundary - 1, boundary, boundary + 1,
                                                 settle - 1, settle, settle + 1)))
        name = f"{item['_candidate_index']:04d}_{eid}_p{item['parent_interval_index']}_b{boundary}.jpg"
        title = (f"gripper candidate {item['_candidate_index']} | {eid} | p{item['parent_interval_index']} "
                 f"a{item['left_atom_index']} -> a{item['right_atom_index']} | old {boundary}, detector {settle}")
    elif kind == "text":
        frames = []
        for atom in item["atoms"]:
            a, b = int(atom["start_timestep"]), int(atom["end_timestep_exclusive"])
            frames.extend((a, (a + b - 1) // 2, b - 1))
        frames = sorted(set(frames))
        name = f"{item['_candidate_index']:03d}_{eid}_{item['subtask'].replace(' ', '_')}.jpg"
        title = f"repeated text {item['_candidate_index']} | {eid} | {item['subtask']}"
    elif kind == "gap":
        a, b = map(int, item["gap_frame_range"])
        frames = sorted(set(max(0, x) for x in (a - 15, a - 2, a - 1, a, (a + b) // 2,
                                                 b - 1, b, b + 1, b + 2, b + 15)))
        name = f"gap_{item['_candidate_index']:03d}_{eid}_f{a}-{b}.jpg"
        title = f"declared gap {item['_candidate_index']} | {eid} | f{a}-{b} {item['reason']}"
    elif kind == "quality":
        a, b = int(item["raw_from_index"]), int(item["raw_to_index"])
        frames = sorted(set(max(0, x) for x in (a - 16, a - 3, a - 1, a, b - 1, b, b + 2, b + 8)))
        name = f"quality_{item['_candidate_index']:03d}_{eid}_{item['cause']}_f{a}-{b}.jpg"
        title = f"short quality {item['_candidate_index']} | {eid} | {item['cause']} q{item['quality']} raw f{a}-{b}"
    else:
        raise ValueError(kind)
    out = out_dir / name
    if not out.exists():
        _sheet(episode_dir, frames, title, out)
    return {"candidate": item, "frames_rendered": frames, "evidence": str(out), "status": "rendered"}


def _candidates(args) -> list[dict]:
    source = json.loads(args.candidates.read_text())["categories"]
    if args.kind == "gripper":
        rows = [r for r in source["gripper_event_boundaries_before_settle"]["report"]["candidates"]
                if r["source"] == "molmoact"]
    elif args.kind == "text":
        groups = source["ambiguous_repeated_lookalike_text"]["report"]["candidates"]
        rows = [r for key in ("exact_mechanical", "visually_required") for r in groups[key]
                if r["source"] == "molmoact"]
    elif args.kind == "gap":
        rows = [r for r in source["atoms_crossing_excluded_gaps"]["report"]["candidates"]
                if r["source"] == "molmoact"]
    elif args.kind == "quality":
        rows = [r for r in _read_jsonl(args.corpus / "quality_spans.jsonl")
                if r["source"] == "molmoact" and int(r["quality"]) <= 3
                and r["cause"] in {"hover", "hold_still", "idle"}
                and int(r["raw_to_index"]) - int(r["raw_from_index"]) <= float(r["native_rate_hz"])]
    else:
        raise ValueError(args.kind)
    return [r | {"_candidate_index": i} for i, r in enumerate(rows)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kind", choices=("gripper", "text", "gap", "quality"), required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    all_rows = _candidates(args)
    end = len(all_rows) if args.end is None else min(args.end, len(all_rows))
    rows = all_rows[args.start:end]
    batch = args.out / f"{args.kind}_{args.start:04d}_{end:04d}"
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
    done, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 12))) as ex:
        futures = {ex.submit(_render_one, row, args.corpus, batch, args.kind): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # preserve exact candidate and continue the bounded batch
                errors.append({"candidate": row, "error": repr(exc)})
    done.sort(key=lambda r: r["candidate"]["_candidate_index"])
    status = {"kind": args.kind, "start": args.start, "end": end, "requested": len(rows),
              "rendered": len(done), "errors": errors, "results": done}
    (batch / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: status[k] for k in ("kind", "start", "end", "requested", "rendered")}
                     | {"errors": len(errors), "batch": str(batch)}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
