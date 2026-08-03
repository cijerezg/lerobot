#!/usr/bin/env python

"""
Apply the vision pass to one episode's label file.

The anchor baseline is already on disk; reading the contact sheets either confirms a
segment or changes it. This records that verdict compactly instead of rewriting the
whole JSON, and refuses to finish an episode with segments nobody looked at.

    --seg "SEG:Q:NOTE"      Q is a digit, or '=' to keep the baseline grade
    --mistake "SEG:FROM-TO:TYPE:NOTE"   replaces that segment's mistakes
    --drop-mistakes SEG,... clears proprio-flagged spans vision rejected

Every listed segment is marked reviewed; `anchor_baseline` flips to false whenever the
grade moved off the anchor, so the two are always distinguishable afterwards.

    python segment_label_review.py rebot_val-v1 0 \
        --seg "0:=:clean approach" --seg "3:2:slip mid-carry" \
        --mistake "3:1200-1265:slip:object drops from the gripper at 1240"
"""

import argparse
import json
from pathlib import Path

LABELS = Path("outputs/_annotation/labels")
TYPES = {"failed_close", "slip", "drop", "knock", "wrong_target"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("episode", type=int)
    ap.add_argument("--seg", action="append", default=[])
    ap.add_argument("--mistake", action="append", default=[])
    ap.add_argument("--drop-mistakes", default="")
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
