#!/usr/bin/env python

"""
Anchor baseline for every segment — the starting point the vision pass overrides.

The per-stratum thresholds in pi07_wiki/annotation_rubric.md are deterministic, so
applying them by hand 767 times only adds transcription errors. This writes the
baseline grade and the proprio-derived mistake spans; reading the contact sheets then
*changes* what vision disagrees with, and every such change is recorded by flipping
`anchor_baseline` to false and saying why in the note.

Never overwrites an episode that already has a label file.

    python segment_label_baseline.py            # fill in the unlabelled episodes
    python segment_label_baseline.py --dry-run
"""

import argparse
import json
from pathlib import Path

# Sheets + their feature index live in the project, not the scratchpad: the
# scratchpad is wiped between turns and took a full regeneration with it once.
SHEETS = Path("outputs/_annotation/sheets")
LABELS = Path("outputs/_annotation/labels")
ORDER = ["rebot_socks_basket-v1", "rebot_shirts_bin-v1", "rebot_two_container-v1", "rebot_val-v1"]

# mistake span around a flagged closed interval [ca, cb): 0.4 s of committed descent
# before the shut, 0.5 s after the reopen so the span covers the reveal.
LEAD, TRAIL = 12, 15


def flagged_failures(seg):
    """Closed intervals that count as failed-grasp candidates.

    A close inside `return to home` is the gripper shutting as the arm parks, not a
    grasp attempt — all 5 in the corpus were confirmed false positives on video. They
    stay in the sheet index so the zoom strip is still built and vision can catch a
    genuine event there, but they never seed a mistake or drag the grade to 1-2.
    """
    return [] if seg["phase"] == "return" else seg["fails"]


def grade(seg, is_first):
    """Returns (quality, note) from the rubric's per-stratum anchors."""
    ph, eff, rev, dur = seg["phase"], seg["eff"], seg["rev"], seg["dur"]
    nf = len(flagged_failures(seg))
    if nf:
        q = 2 if nf == 1 else 1
        return q, f"{nf} flagged failed close(s); anchor rule forces quality {q}."
    if ph == "grasp" and is_first:
        q = 5 if eff >= 0.78 else 4 if eff >= 0.72 else 3
        return q, f"grasp/first eff {eff:.2f}"
    if ph == "grasp":
        q = 5 if (eff >= 0.57 and rev <= 2) else 4 if eff >= 0.42 else 3
        return q, f"grasp/later eff {eff:.2f} rev {rev}"
    if ph == "move":
        q = 5 if eff >= 0.69 else 4 if eff >= 0.62 else 3
        return q, f"move eff {eff:.2f}"
    if ph == "release":
        basket = seg["subtask"].endswith("basket")
        hi, lo = (2.3, 3.4) if basket else (3.5, 6.5)
        q = 5 if dur <= hi else 4 if dur <= lo else 3
        return q, f"release/{'basket' if basket else 'bin'} {dur:.1f}s"
    q = 5 if eff >= 0.83 else 4 if eff >= 0.62 else 3
    return q, f"return eff {eff:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    segs = []
    for ds in ORDER:
        p = SHEETS / f"index__{ds}.json"
        if p.exists():
            segs += json.load(open(p))["segments"]

    LABELS.mkdir(parents=True, exist_ok=True)
    by_ep = {}
    for s in segs:
        by_ep.setdefault((s["ds"], s["ep"]), []).append(s)

    written = skipped = 0
    for (ds, ep), items in sorted(by_ep.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        out = LABELS / f"{ds}__ep{ep:02d}.json"
        if out.exists():
            skipped += 1
            continue
        items.sort(key=lambda s: s["seg"])
        first_grasp = min((s["seg"] for s in items if s["phase"] == "grasp"), default=None)
        rows = []
        for s in items:
            q, why = grade(s, s["seg"] == first_grasp)
            mistakes = [
                {"from": max(s["from"], ca - LEAD), "to": min(s["to"], cb + TRAIL),
                 "type": "failed_close", "note": f"proprio-flagged closed interval [{ca},{cb})"}
                for ca, cb in flagged_failures(s)
            ]
            rows.append({"seg": s["seg"], "from": s["from"], "to": s["to"], "subtask": s["subtask"],
                         "quality": q, "anchor_baseline": True, "note": why, "mistakes": mistakes})
        payload = {"dataset": ds, "episode": ep, "segments": rows}
        if not args.dry_run:
            json.dump(payload, open(out, "w"), indent=1)
        written += 1
    print(f"baseline written for {written} episodes, {skipped} already labelled")


if __name__ == "__main__":
    main()
