#!/usr/bin/env python

"""
Emit the relabelled datasets: per-segment quality + semantic mistake spans.

Reads `outputs/_annotation/labels/*.json` and writes a NEW dataset directory per
source — never in place. data/, videos/ and depth/ are hardlinked (no copy, no risk
to the originals); meta/ is rebuilt with the new annotation files.

Written per dataset:
  meta/episode_metadata.parquet  one row per SEGMENT (not per episode): the loader
                                 broadcasts by from_index/to_index and never assumed
                                 one row per episode, so this needs no code change.
  meta/mistakes.parquet          one row per mistake EVENT, with its type.
  meta/subtask_windows.json      the semantic windows, carrying label corrections
                                 made during the vision pass.
  meta/subtasks.parquet          the dataset-local subtask vocabulary the windows
                                 resolve against (offline_dataset_utils needs it).
  meta/metadata_info.json        provenance + how much of the corpus vision reviewed.

    python segment_label_write.py --suffix annotated-v2
    python segment_label_write.py --suffix annotated-v2 --dry-run
"""

import argparse
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

LABELS = Path("outputs/_annotation/labels")
OUTPUTS = Path("outputs")
ORDER = ["rebot_socks_basket-v1", "rebot_shirts_bin-v1", "rebot_two_container-v1", "rebot_val-v1"]

DEFINITION = (
    "Quality 1-5 per semantic subtask segment, constant across the segment: 5 direct, "
    "4 one minor correction, 3 laboured but no discrete failure, 2 one failure event "
    "recovered in-segment, 1 multiple failures or an unrecovered one. Mistakes are "
    "semantic spans over discrete visible failure events (failed_close, slip, drop, "
    "knock, wrong_target), starting at the committed failing attempt and ending just "
    "after the failure is observable; recovery is excluded."
)
LIMITATION = (
    "Anchored on per-stratum proprio thresholds (pi07_wiki/annotation_rubric.md) and "
    "confirmed against per-segment contact sheets. Closes inside 'return to home' are "
    "parking behaviour and never seed a mistake. Object identity was corrected only "
    "where the video is unambiguous: the wrist camera sees the pile behind the held "
    "object, so black-vs-white sock is not always separable."
)


def hardlink_tree(src: Path, dst: Path) -> int:
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dst / rel / f
            if not target.exists():
                os.link(Path(root) / f, target)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="annotated-v2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    by_ds: dict[str, list] = {}
    for p in sorted(LABELS.glob("*.json")):
        d = json.load(open(p))
        by_ds.setdefault(d["dataset"], []).append(d)

    for ds in ORDER:
        eps = sorted(by_ds[ds], key=lambda d: d["episode"])
        src, dst = OUTPUTS / ds, OUTPUTS / f"{ds.replace('-v1', '')}-{args.suffix}"

        seg_rows, mistake_rows, windows = [], [], {}
        n_reviewed = n_frames = n_reviewed_frames = 0
        for d in eps:
            ep = d["episode"]
            windows[str(ep)] = []
            for s in d["segments"]:
                seg_rows.append({
                    "episode_index": ep, "segment_index": s["seg"],
                    "from_index": s["from"], "to_index": s["to"],
                    "subtask": s["subtask"], "quality": int(s["quality"]),
                    "vision_reviewed": bool(s.get("reviewed")),
                    "anchor_baseline": bool(s.get("anchor_baseline", True)),
                    "note": s["note"],
                })
                for m in s["mistakes"]:
                    mistake_rows.append({
                        "episode_index": ep, "from_index": m["from"], "to_index": m["to"],
                        "mistake": True, "mistake_type": m["type"], "note": m["note"],
                    })
                windows[str(ep)].append({
                    "from_index": s["from"], "to_index": s["to"], "subtask": s["subtask"],
                })
                n = s["to"] - s["from"]
                n_frames += n
                n_reviewed += bool(s.get("reviewed"))
                n_reviewed_frames += n if s.get("reviewed") else 0

        seg_df = pd.DataFrame(seg_rows)
        mis_df = pd.DataFrame(mistake_rows)
        qshare = {int(q): round(float(seg_df[seg_df.quality == q].apply(
            lambda r: r.to_index - r.from_index, axis=1).sum()) / n_frames, 4) for q in (1, 2, 3, 4, 5)}

        print(f"\n{ds} -> {dst.name}")
        print(f"  {len(seg_df)} segments, {len(mis_df)} mistakes, {n_frames} frames")
        print(f"  vision-reviewed {n_reviewed}/{len(seg_df)} segments "
              f"({n_reviewed_frames / n_frames:.0%} of frames)")
        print(f"  quality frame share {qshare}")
        if args.dry_run:
            continue

        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for sub in ("data", "videos", "depth"):
            if (src / sub).exists():
                print(f"  hardlinked {hardlink_tree(src / sub, dst / sub)} files in {sub}/")
        (dst / "meta").mkdir(exist_ok=True)
        for f in ("info.json", "stats.json", "tasks.parquet", "summaries.parquet"):
            if (src / "meta" / f).exists():
                os.link(src / "meta" / f, dst / "meta" / f)
        hardlink_tree(src / "meta" / "episodes", dst / "meta" / "episodes")

        seg_df.to_parquet(dst / "meta" / "episode_metadata.parquet", engine="pyarrow")
        mis_df.to_parquet(dst / "meta" / "mistakes.parquet", engine="pyarrow")
        labels = sorted({s["subtask"] for spans in windows.values() for s in spans})
        pd.DataFrame(
            {"subtask_index": range(len(labels))}, index=pd.Index(labels, name="subtask")
        ).to_parquet(dst / "meta" / "subtasks.parquet", engine="pyarrow", compression="snappy")
        json.dump({
            "model": "assistant-vision-pass",
            "annotator": "segment_sheets.py + segment_label_{baseline,review}.py",
            "rubric": "lerobot/pi07_wiki/annotation_rubric.md",
            "created_date": date.today().isoformat(),
            "segmentation": "semantic phases from gripper-closed intervals (semantic_segment.py)",
            "quality_scope": "per semantic segment, constant across it",
            "definition": DEFINITION,
            "limitation": LIMITATION,
            "segments": len(seg_df),
            "mistakes": len(mis_df),
            "vision_reviewed_segments": int(n_reviewed),
            "vision_reviewed_frame_fraction": round(n_reviewed_frames / n_frames, 4),
            "quality_frame_share": qshare,
        }, open(dst / "meta" / "metadata_info.json", "w"), indent=2)
        json.dump({
            "model": "assistant-vision-pass",
            "annotator": "semantic_segment.py + object-identity corrections from the vision pass",
            "created_date": date.today().isoformat(),
            "interval_seconds": None,
            "top_key": "observation.images.top",
            "wrist_key": "observation.images.wrist",
            "episodes": windows,
        }, open(dst / "meta" / "subtask_windows.json", "w"), indent=1)
        print(f"  wrote meta/ -> {dst}")


if __name__ == "__main__":
    main()
