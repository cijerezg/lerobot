"""Pick empty-handed frames for the scene-aware subtask sweep and render a top-camera contact sheet.

Rule (val set, rebot_val-annotated-v3):
  * an empty-handed WINDOW is a ``grasp X`` segment from its start up to the first gripper
    close event inside it (meta/depth_gripper_events.parquet), or a ``return to home`` segment;
    subtask_windows.json is in GLOBAL indices, the events table carries both (``index`` global,
    ``frame_index`` per episode) — everything below is converted to per-episode frame indices;
  * within each window take frames at 0.5 s in and at 25 / 50 / 75 % of the window, snapped
    onto the image_stride grid (frame_index % stride == 0);
  * the empty-handed window ends 1 s BEFORE the close (the arm is already at the object there)
    and windows shorter than MIN_WINDOW_S after that trim (failed-grasp retries) are dropped;
  * scene inventory: a class is on the table from its first frame to the end of its LAST grasp
    segment, default first frame 0 — corrected by INVENTORY_OVERRIDES, read off the top camera
    (2026-09-13): ep2's white socks are added mid-episode (absent at fr 1019, present at 1200),
    ep3's burgundy shirt is a never-grasped distractor; ep3's white shirt has no vocab label, so
    "the grey shirt" is AMBIGUOUS there and excluded from the absent set. present(t) = classes
    on the table at t (GT object included); absent = grasp vocabulary minus present minus ambiguous;
  * user rules 2026-09-13: "the dark shirt" is an ambiguous label everywhere (never in a label set,
    windows whose GT is grasp the dark shirt are skipped), and a frame is kept only when every
    present class has exactly ONE instance on the table (MULTI_INSTANCE ranges, read off the top
    camera: ep0 has 3 black socks until its last reach and 2 white until fr 819; ep2 has 2 white
    socks from fr 1200 until the black-sock reach at 3571).

Usage (from the workspace root):
  uv run python lerobot/scripts/probes/subtask_scene_sweep_frames.py [root] [--stride 3] [--out outputs/probe_frames/subtask_scene_sweep_frames]

Writes frames.json and the contact sheets into --out. The reviewed frames.json is then copied to
lerobot/dataset_info/manifests/scene_sweep_frames_<date>.json, which the configs read.
"""

import argparse
import json
import os
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OFFSETS = [("0.5s", None), ("25%", 0.25), ("50%", 0.50), ("75%", 0.75)]
PRE_CLOSE_S = 1.0
MIN_WINDOW_S = 3.0
# episode -> class -> (first_frame, last_frame); None = keep the derived value (end of the last
# grasp, or the episode end for a class that is never grasped); per-episode frame indices
INVENTORY_OVERRIDES = {
    2: {"the white sock": (1200, None)},
    3: {"the red shirt": (0, None)},
}
AMBIGUOUS = {3: ["the grey shirt"]}
EXCLUDED_LABELS = ["the dark shirt"]
# episode -> class -> (first_frame, last_frame) range in which MORE than one instance is on the table
MULTI_INSTANCE = {
    0: {"the black sock": (0, 4167), "the white sock": (0, 819)},
    2: {"the white sock": (1200, 3571)},
}


def _font(size):
    import matplotlib
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


def select_frames(root: str, stride: int, fps: int) -> list[dict]:
    windows = json.load(open(f"{root}/meta/subtask_windows.json"))["episodes"]
    events = pd.read_parquet(f"{root}/meta/depth_gripper_events.parquet")
    closes = events[events.event_type == "close"]
    episodes = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
    ep_start = dict(zip(episodes.episode_index, episodes.dataset_from_index))
    vocab = pd.read_parquet(f"{root}/meta/subtasks.parquet").index
    grasp_classes = sorted(s[len("grasp "):] for s in vocab if s.startswith("grasp "))

    rows = []
    for ep_key, segs in windows.items():
        ep = int(ep_key)
        base = int(ep_start[ep])
        ep_closes = sorted(closes[closes.episode_index == ep].frame_index.tolist())
        segs = [{**s, "from_index": s["from_index"] - base, "to_index": s["to_index"] - base} for s in segs]
        grasps = [s for s in segs if s["subtask"].startswith("grasp ")]
        ep_len = segs[-1]["to_index"]
        inventory = {}
        for g in grasps:
            cls = g["subtask"][len("grasp "):]
            inventory[cls] = (0, max(g["to_index"], inventory.get(cls, (0, 0))[1]))
        for cls, (first, last) in INVENTORY_OVERRIDES.get(ep, {}).items():
            derived = inventory.get(cls, (0, ep_len))
            inventory[cls] = (derived[0] if first is None else first, derived[1] if last is None else last)
        ambiguous = EXCLUDED_LABELS + AMBIGUOUS.get(ep, [])
        multi = MULTI_INSTANCE.get(ep, {})
        for seg in segs:
            verb = seg["subtask"].split(" ")[0]
            if verb not in ("grasp", "return") or seg["subtask"][len("grasp "):] in ambiguous:
                continue
            start, stop = seg["from_index"], seg["to_index"]
            if verb == "grasp":
                inside = [c for c in ep_closes if start <= c < stop]
                stop = (inside[0] if inside else stop) - int(PRE_CLOSE_S * fps)
            if stop - start < MIN_WINDOW_S * fps:
                continue
            for name, frac in OFFSETS:
                fr = start + int(round(0.5 * fps)) if frac is None else start + int(frac * (stop - start))
                fr = int(round(fr / stride)) * stride
                if fr >= stop or fr < start:
                    continue
                present = sorted(c for c, (a, b) in inventory.items() if a <= fr < b and c not in ambiguous)
                if any(c in multi and multi[c][0] <= fr < multi[c][1] for c in present):
                    continue
                rows.append({
                    "episode_idx": ep, "frame_idx": fr, "global_idx": base + fr,
                    "window": [int(start), int(stop)], "offset": name,
                    "seconds_in": round((fr - start) / fps, 2),
                    "gt_subtask": seg["subtask"],
                    "present": present,
                    "absent": [c for c in grasp_classes if c not in present and c not in ambiguous],
                    "ambiguous": ambiguous,
                })
    return rows


def render_sheets(root: str, rows: list[dict], out_dir: str, stride: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="rebot_val", root=root)
    data = pd.read_parquet(f"{root}/data/chunk-000/file-000.parquet", columns=["observation.state"])
    grip = np.stack(data["observation.state"].values)[:, 6]

    tile_w, tile_h, cap_h, label_w = 320, 240, 58, 290
    font, small = _font(13), _font(11)
    n_cols = len(OFFSETS)
    for ep in sorted({r["episode_idx"] for r in rows}):
        ep_rows = [r for r in rows if r["episode_idx"] == ep]
        windows = []
        for r in ep_rows:
            if tuple(r["window"]) not in [tuple(w["window"]) for w in windows]:
                windows.append(r)
        W = label_w + n_cols * tile_w
        H = 30 + len(windows) * (tile_h + cap_h)
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"ep{ep} — top camera, empty-handed frames (stride {stride}); columns = offset into the window", fill="black", font=font)
        for i, w in enumerate(windows):
            y0 = 30 + i * (tile_h + cap_h)
            a, b = w["window"]
            lines = [f"row {i}: {w['gt_subtask']}", f"window fr {a}-{b} ({(b - a) / 30:.1f} s)"]
            for key in ("present", "absent", "ambiguous"):
                lines += textwrap.wrap(f"{key}: " + (", ".join(w[key]) or "-"), 38)
            for k, line in enumerate(lines):
                draw.text((8, y0 + 8 + 15 * k), line, fill="black", font=small)
            for j, (name, _) in enumerate(OFFSETS):
                r = next((r for r in ep_rows if r["window"] == w["window"] and r["offset"] == name), None)
                x0 = label_w + j * tile_w
                if r is None:
                    draw.text((x0 + 8, y0 + 8), f"{name}: (outside window)", fill="grey", font=small)
                    continue
                img = ds[r["global_idx"]]["observation.images.top"]
                img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((tile_w, tile_h))
                sheet.paste(img, (x0, y0))
                r["gripper"] = round(float(grip[r["global_idx"]]), 1)
                cap = [f"{name}: fr {r['frame_idx']} (+{r['seconds_in']} s)  global {r['global_idx']}",
                       f"gripper {r['gripper']}  ({'open' if r['gripper'] < -100 else 'closed/partial'})"]
                for k, line in enumerate(cap):
                    draw.text((x0 + 4, y0 + tile_h + 4 + 15 * k), line, fill="black", font=small)
        path = os.path.join(out_dir, f"sheet_ep{ep}.png")
        sheet.save(path)
        print(f"wrote {path}  ({len(ep_rows)} frames, {len(windows)} windows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="outputs/rebot_val-annotated-v3")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="outputs/probe_frames/subtask_scene_sweep_frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = select_frames(args.root, args.stride, args.fps)
    render_sheets(args.root, rows, args.out, args.stride)
    with open(os.path.join(args.out, "frames.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{len(rows)} frames -> {args.out}/frames.json")


if __name__ == "__main__":
    main()
