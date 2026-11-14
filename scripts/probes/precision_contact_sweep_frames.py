"""Pick pre-commit frames for the precision / contact sweep and render a contact sheet.

Why these frames: ``metadata_steering`` sweeps the precision and contact clauses on evenly
sampled frames, most of which are transport, where the true precision is 1 and the true
contact is "not applicable" and a well-trained model should move nothing. The clause can
only reach the chunk in the second before the gripper commits, so that is where this list
sits (the complementary window to ``subtask_scene_sweep_frames.py``, whose windows END 1 s
before the close).

Rule (val root, outputs/rebot_val_2026-10-14-v1-seamfix1 since 2026-11-13; the 2026-10-05 list
(precision_contact_sweep_frames_2026-10-05.json) was outputs/rebot_training_mix_2026-10-04/validation_all):
  * a WINDOW is a segment whose contact label is a real element (``meta/contact.parquet``
    code != 14); moves and return-to-home are "not applicable" and are skipped, and so is
    a pour (no gripper commit);
  * its COMMIT is the LAST gripper close inside a grasp-type segment (retries fail, the last
    close is the one that lifts) or the FIRST open inside a release-type segment
    (OPEN_COMMIT_VERBS: release, close, stand, fold, set)
    (``meta/depth_gripper_events.parquet``); a segment with no event (no depth on ep 3)
    falls back to the annotators' ``meta/precision_windows.parquet`` commit_index;
  * frames at 1.0 / 0.5 / 0.2 s before the commit, snapped onto the image_stride grid and
    kept only inside the segment; with a 1 s chunk the first frame's chunk ends at the commit;
  * the plausible contact set is hand-written per object (grasps) or target (releases) in
    GRASP_CODES / RELEASE_CODES and must contain the true code; CONTROL_CODES are two
    elements implausible everywhere (strike, pour), swept as a separate population like the
    scene sweep's absent classes;
  * the true precision comes from ``meta/precision.parquet``; the probe ramps 1..5 on every
    frame and marks the true level.

subtask_windows / episode_metadata indices are GLOBAL; the events table carries both
(``index`` global, ``frame_index`` per episode). Everything is converted to per-episode
frame indices for the rows and back to global for the probe.

Usage (from the workspace root):
  uv run python lerobot/scripts/probes/precision_contact_sweep_frames.py [root] [--stride 3] [--out outputs/probe_frames/precision_contact_sweep_frames]

Writes frames.json and the contact sheets into --out. The reviewed frames.json is then copied to
lerobot/dataset_info/manifests/precision_contact_sweep_frames_<date>.json, which the configs read.
"""

import argparse
import json
import os
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.insert(0, "lerobot/src")
from lerobot.annotation.vocab import CONTACT_VOCAB, NA_CODE, phrase_for  # noqa: E402

PRE_COMMIT_S = (1.0, 0.5, 0.2)
CAMERAS = ("observation.images.external_0", "observation.images.wrist_0")
# codes: 0 top-pinch, 1 side-pinch, 2 rim-pinch, 3 handle-grasp, 4 cloth-pinch, 9 set-down,
# 10 drop, 11 insert; see datasets/contact_vocab.py
CLOTH = [4, 0]
GRASP_CODES = {
    "the pill bottle": [0, 1],
    "the tape roll": [0, 1, 2],
    "the cup": [0, 1, 2],
    "the green eraser": [0, 1],
    "the sugar cube": [0, 1],
    "the grey shirt": CLOTH, "the navy shirt": CLOTH, "the red shirt": CLOTH,
    "the brown shirt": CLOTH, "the pink shirt": CLOTH,
    "the white sock": CLOTH, "the black sock": CLOTH,
    "the bit kit cover": CLOTH,
    # 2026-11-13: objects of outputs/rebot_val_2026-10-14-v1-seamfix1
    "the shirt": CLOTH, "the sock": CLOTH, "the white sock nearest the bin": CLOTH,
    "the outer edge of the flap with the bits": CLOTH,
    "the nail clipper": [0, 1],
    "the bit": [0, 1],
    "the yellow pill bottle": [0, 1], "the purple bottle": [0, 1], "the water bottle": [0, 1],
    "the open pill bottle": [0, 1, 2],
    "the headphone case lid": [0, 2, 3],
}
# release target (last word of the subtask) -> plausible codes
RELEASE_CODES = {"bin": [9, 10], "basket": [9, 10], "cup": [9, 10, 11], "stack": [9, 10, 11], "slot": [9, 10, 11]}
# whole-subtask rules, for wordings the two tables above cannot key
OTHER_CODES = {
    # a grasp-type commit (close) whose contact is the seating
    "stack the pieces on the white peg": [9, 10, 11],
    "release the bit in the compartment with the gray bits": [9, 10, 11],
    "close the headphone case lid": [2, 5],
    "stand the yellow pill bottle upright": [9, 10],
    "stand the purple bottle upright": [9, 10],
    "fold the flap with the bits over the middle of the kit": [9, 10],
    "set the open pill bottle down next to the box": [9, 10],
    "push the nail clipper into the small slot": [5, 6],
}
# verbs whose commit is the FIRST open inside the segment; every other verb commits on the LAST close
OPEN_COMMIT_VERBS = ("release", "close", "stand", "fold", "set")
CONTROL_CODES = [7, 12]


def _font(size):
    import matplotlib
    path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    return ImageFont.truetype(path, size)


def _plausible(subtask: str) -> list[int]:
    verb = subtask.split(" ")[0]
    if subtask in OTHER_CODES:
        return OTHER_CODES[subtask]
    if verb == "grasp":
        return GRASP_CODES[subtask[len("grasp "):]]
    if verb == "release":
        return RELEASE_CODES[subtask.split(" ")[-1]]
    raise KeyError(subtask)


def select_frames(root: str, stride: int, fps: int) -> list[dict]:
    meta = pd.read_parquet(f"{root}/meta/episode_metadata.parquet")
    contact = pd.read_parquet(f"{root}/meta/contact.parquet")
    precision = pd.read_parquet(f"{root}/meta/precision.parquet")
    segs = meta.merge(contact[["episode_index", "segment_index", "contact"]], on=["episode_index", "segment_index"])
    segs = segs.merge(precision[["episode_index", "segment_index", "precision"]], on=["episode_index", "segment_index"])
    events = pd.read_parquet(f"{root}/meta/depth_gripper_events.parquet")
    fallback = pd.read_parquet(f"{root}/meta/precision_windows.parquet")
    fallback_commit = {(int(r.episode_index), int(r.segment_index)): int(r.commit_index) for r in fallback.itertuples()}
    episodes = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
    ep_start = dict(zip(episodes.episode_index, episodes.dataset_from_index))

    rows = []
    for seg in segs.itertuples():
        if int(seg.contact) == NA_CODE:
            continue
        ep, base = int(seg.episode_index), int(ep_start[int(seg.episode_index)])
        start, stop = int(seg.from_index) - base, int(seg.to_index) - base
        verb = seg.subtask.split(" ")[0]
        if verb == "pour":
            # the pour commits on a wrist rotation, not a gripper event, and pour is a control code
            print(f"skip ep{ep} seg{seg.segment_index} {seg.subtask!r}: pour has no gripper commit")
            continue
        kind = "release" if verb in OPEN_COMMIT_VERBS else "grasp"
        ev = events[(events.episode_index == ep) & (events.event_type == ("open" if kind == "release" else "close"))]
        inside = sorted(int(f) for f in ev.frame_index if start <= f < stop)
        if inside:
            commit = inside[0] if kind == "release" else inside[-1]
        elif (ep, int(seg.segment_index)) in fallback_commit:
            commit = fallback_commit[(ep, int(seg.segment_index))] - base
        else:
            print(f"skip ep{ep} seg{seg.segment_index} {seg.subtask!r}: no commit event")
            continue
        plausible = _plausible(seg.subtask)
        if int(seg.contact) not in plausible:
            raise ValueError(f"ep{ep} seg{seg.segment_index} {seg.subtask!r}: true contact {seg.contact} not in {plausible}")
        for offset in PRE_COMMIT_S:
            fr = int(round((commit - offset * fps) / stride)) * stride
            if fr < start or fr >= commit:
                continue
            rows.append({
                "episode_idx": ep, "frame_idx": fr, "global_idx": base + fr,
                "segment_index": int(seg.segment_index), "segment": [start, stop],
                "commit_frame": int(commit), "offset_s": offset, "seconds_before_commit": round((commit - fr) / fps, 2),
                "kind": kind, "gt_subtask": seg.subtask, "quality": int(seg.quality),
                "true_contact": int(seg.contact), "true_contact_slug": CONTACT_VOCAB[int(seg.contact)].slug,
                "plausible_contacts": plausible, "control_contacts": CONTROL_CODES,
                "true_precision": int(seg.precision),
            })
    return rows


def render_sheets(root: str, rows: list[dict], out_dir: str, stride: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="rebot_val", root=root)
    data = pd.read_parquet(f"{root}/data", columns=["index", "observation.state"]).sort_values("index")
    grip = np.stack(data["observation.state"].values)[:, 6]

    tile_w, tile_h, cap_h, label_w = 256, 192, 44, 300
    font, small = _font(13), _font(11)
    n_cols = len(PRE_COMMIT_S) * len(CAMERAS)
    for ep in sorted({r["episode_idx"] for r in rows}):
        ep_rows = [r for r in rows if r["episode_idx"] == ep]
        windows = sorted({r["segment_index"] for r in ep_rows})
        W = label_w + n_cols * tile_w
        H = 30 + len(windows) * (tile_h + cap_h)
        sheet = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"ep{ep} — pre-commit frames (stride {stride}); per offset: external camera, wrist camera", fill="black", font=font)
        for i, seg_idx in enumerate(windows):
            y0 = 30 + i * (tile_h + cap_h)
            w_rows = [r for r in ep_rows if r["segment_index"] == seg_idx]
            w = w_rows[0]
            lines = [f"row {i}: {w['gt_subtask']}", f"segment fr {w['segment'][0]}-{w['segment'][1]}, commit fr {w['commit_frame']} ({w['kind']})",
                     f"true contact: {phrase_for(w['true_contact'])}   precision {w['true_precision']}   quality {w['quality']}"]
            lines += textwrap.wrap("plausible: " + ", ".join(phrase_for(c) for c in w["plausible_contacts"]), 40)
            lines += textwrap.wrap("control: " + ", ".join(phrase_for(c) for c in w["control_contacts"]), 40)
            for k, line in enumerate(lines):
                draw.text((8, y0 + 8 + 15 * k), line, fill="black", font=small)
            for j, offset in enumerate(PRE_COMMIT_S):
                r = next((r for r in w_rows if r["offset_s"] == offset), None)
                x0 = label_w + j * len(CAMERAS) * tile_w
                if r is None:
                    draw.text((x0 + 8, y0 + 8), f"-{offset} s: (outside segment)", fill="grey", font=small)
                    continue
                item = ds[r["global_idx"]]
                for c, cam in enumerate(CAMERAS):
                    img = item[cam]
                    img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((tile_w, tile_h))
                    sheet.paste(img, (x0 + c * tile_w, y0))
                r["gripper"] = round(float(grip[r["global_idx"]]), 1)
                cap = [f"-{r['seconds_before_commit']} s: fr {r['frame_idx']}  global {r['global_idx']}",
                       f"gripper {r['gripper']}"]
                for k, line in enumerate(cap):
                    draw.text((x0 + 4, y0 + tile_h + 4 + 15 * k), line, fill="black", font=small)
        path = os.path.join(out_dir, f"sheet_ep{ep}.png")
        sheet.save(path)
        print(f"wrote {path}  ({len(ep_rows)} frames, {len(windows)} windows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="outputs/rebot_val_2026-10-14-v1-seamfix1")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="outputs/probe_frames/precision_contact_sweep_frames")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = select_frames(args.root, args.stride, args.fps)
    render_sheets(args.root, rows, args.out, args.stride)
    with open(os.path.join(args.out, "frames.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{len(rows)} frames -> {args.out}/frames.json")


if __name__ == "__main__":
    main()
