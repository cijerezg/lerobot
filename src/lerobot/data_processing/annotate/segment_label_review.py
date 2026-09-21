#!/usr/bin/env python

"""
Apply the vision pass to one episode's label file.

The anchor baseline is already on disk; reading the contact sheets either confirms a
segment or changes it. This records that verdict compactly instead of rewriting the
whole JSON, and refuses to finish an episode with segments nobody looked at.

    --seg "SEG:Q:NOTE"      Q is a digit, or '=' to keep the baseline grade
    --mistake "SEG:FROM-TO:TYPE:NOTE"   replaces that segment's mistakes
    --drop-mistakes SEG,... clears proprio-flagged spans vision rejected
    --boundary "SEG:FRAME"  move the boundary between SEG-1 and SEG (both are rewritten,
                            so coverage stays contiguous)
    --merge SEG             fold SEG into the segment before it, keeping that one's label

Every listed segment is marked reviewed; `anchor_baseline` flips to false whenever the
grade moved off the anchor, so the two are always distinguishable afterwards.

    python segment_label_review.py rebot_val-v1 0 \
        --seg "0:=:clean approach" --seg "3:2:slip mid-carry" \
        --mistake "3:1200-1265:slip:object drops from the gripper at 1240"
"""

import argparse
import json
import os
from pathlib import Path

LABELS = Path(os.environ.get("ANNOTATE_LABELS_ROOT", "outputs/_annotation/labels"))
TYPES = {"failed_close", "slip", "drop", "knock", "wrong_target"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("episode", type=int)
    ap.add_argument("--seg", action="append", default=[])
    ap.add_argument("--mistake", action="append", default=[])
    ap.add_argument("--drop-mistakes", default="")
    ap.add_argument("--boundary", action="append", default=[],
                    help="SEG:FRAME — move the start of SEG (and the end of SEG-1) to FRAME")
    ap.add_argument("--merge", action="append", default=[],
                    help="SEG — fold this segment into the previous one (for the sub-second "
                         "stubs the proprio cut leaves behind)")
    ap.add_argument("--subtask", action="append", default=[],
                    help="SEG:TEXT — correct the label, for <object> placeholders and "
                         "majority-vote carryover errors")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    path = LABELS / f"{args.dataset}__ep{args.episode:02d}.json"
    data = json.load(open(path))
    by_seg = {s["seg"]: s for s in data["segments"]}

    for spec in args.seg:
        sid, q, note = spec.split(":", 2)
        s = by_seg[int(sid)]
        if q != "=":
            if s["quality"] != int(q):
                s["anchor_baseline"] = False
            s["quality"] = int(q)
        s["note"] = note
        s["reviewed"] = True

    for spec in args.subtask:
        sid, text = spec.split(":", 1)
        s = by_seg[int(sid)]
        if text != s["subtask"]:
            s["subtask_was"] = s["subtask"]
            s["subtask"] = text

    for sid in filter(None, args.drop_mistakes.split(",")):
        s = by_seg[int(sid)]
        s["mistakes"] = []
        s["anchor_baseline"] = False

    # Boundaries first: a later --seg/--mistake refers to the segments as renumbered here.
    for spec in args.boundary:
        sid, frame = (int(x) for x in spec.split(":"))
        segs = data["segments"]
        i = next(j for j, x in enumerate(segs) if x["seg"] == sid)
        if i == 0:
            raise SystemExit(f"seg{sid} is the first of the episode; it has no left boundary")
        prev = segs[i - 1]
        if not (prev["from"] < frame < segs[i]["to"]):
            raise SystemExit(f"frame {frame} escapes [{prev['from']},{segs[i]['to']})")
        prev["to"] = segs[i]["from"] = frame
        prev["anchor_baseline"] = segs[i]["anchor_baseline"] = False

    for spec in args.merge:
        sid = int(spec)
        segs = data["segments"]
        i = next(j for j, x in enumerate(segs) if x["seg"] == sid)
        if i == 0:
            raise SystemExit(f"seg{sid} is the first of the episode; nothing to merge it into")
        prev = segs[i - 1]
        prev["to"] = segs[i]["to"]
        prev["mistakes"] += segs[i]["mistakes"]
        prev["anchor_baseline"] = False
        segs.pop(i)
        for j, x in enumerate(segs):
            x["seg"] = j
        by_seg.clear()
        by_seg.update({x["seg"]: x for x in segs})

    grouped: dict[int, list] = {}
    for spec in args.mistake:
        sid, span, mtype, note = spec.split(":", 3)
        lo, hi = (int(x) for x in span.split("-"))
        if mtype not in TYPES:
            raise SystemExit(f"unknown mistake type {mtype!r}; expected one of {sorted(TYPES)}")
        s = by_seg[int(sid)]
        if not (s["from"] <= lo < hi <= s["to"]):
            raise SystemExit(f"seg{sid}: span [{lo},{hi}) escapes segment [{s['from']},{s['to']})")
        grouped.setdefault(int(sid), []).append(
            {"from": lo, "to": hi, "type": mtype, "note": note}
        )
    for sid, rows in grouped.items():
        by_seg[sid]["mistakes"] = rows
        by_seg[sid]["anchor_baseline"] = False

    missing = [s["seg"] for s in data["segments"] if not s.get("reviewed")]
    if missing and not args.allow_partial:
        raise SystemExit(f"{path.name}: segments not reviewed: {missing}")

    json.dump(data, open(path, "w"), indent=1)
    n = sum(1 for s in data["segments"] if s.get("reviewed"))
    changed = sum(1 for s in data["segments"] if s.get("reviewed") and not s.get("anchor_baseline"))
    nm = sum(len(s["mistakes"]) for s in data["segments"])
    print(f"{path.name}: {n}/{len(data['segments'])} reviewed, {changed} moved off anchor, {nm} mistakes")


if __name__ == "__main__":
    main()
