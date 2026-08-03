#!/usr/bin/env python

"""
Progress + resume view over the per-episode label files written during the vision pass.

Labels live in `outputs/_annotation/labels/<dataset>__ep<NN>.json`, one file per
episode, written once that episode's sheets have been read. This is the resumable
state: sheets are regenerable, judgements are not.

    python segment_label_status.py                 # what is done, what is next
    python segment_label_status.py --next          # sheets for the next unlabelled episode
    python segment_label_status.py --episode rebot_val-v1 0   # that episode's sheets + features
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# Sheets + their feature index live in the project, not the scratchpad: the
# scratchpad is wiped between turns and took a full regeneration with it once.
SHEETS = Path("outputs/_annotation/sheets")
LABELS = Path("outputs/_annotation/labels")
ORDER = ["rebot_socks_basket-v1", "rebot_shirts_bin-v1", "rebot_two_container-v1", "rebot_val-v1"]


def load_index():
    segs, sheets = [], []
    for ds in ORDER:
        p = SHEETS / f"index__{ds}.json"
        if p.exists():
            idx = json.load(open(p))
            segs += idx["segments"]
            sheets += idx["sheets"]
    return segs, sheets


def episode_key(ds, ep):
    return f"{ds}__ep{ep:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--episode", nargs=2, metavar=("DS", "EP"))
    ap.add_argument("--queue", type=int, metavar="N",
                    help="next N sheets with unreviewed segments, failure-carrying grasps first")
    args = ap.parse_args()

    if args.queue:
        segs, sheets = load_index()
        reviewed = set()
        feat = {}
        for p in LABELS.glob("*.json"):
            d = json.load(open(p))
            for s in d["segments"]:
                if s.get("reviewed"):
                    reviewed.add((d["dataset"], d["episode"], s["seg"]))
        for s in segs:
            feat[(s["ds"], s["ep"], s["seg"])] = s
        pending = [sh for sh in sheets
                   if any((sh["ds"], sh["ep"], g) not in reviewed for g in sh["segs"])]
        pending.sort(key=lambda sh: (
            0 if any(feat[(sh["ds"], sh["ep"], g)]["fails"] for g in sh["segs"]) else 1,
            ORDER.index(sh["ds"]), sh["ep"], sh["segs"][0]))
        print(f"{len(pending)} sheets pending")
        for sh in pending[: args.queue]:
            print(SHEETS / sh["name"])
            for g in sh["segs"]:
                s = feat[(sh["ds"], sh["ep"], g)]
                f = f" FAILS={s['fails']}" if s["fails"] else ""
                mark = "" if (sh["ds"], sh["ep"], g) in reviewed else "  <-"
                print(f"    seg{g:<3d} [{s['from']},{s['to']}) {s['dur']:5.1f}s eff={s['eff']:.2f} "
                      f"rev={s['rev']:<2d} {s['subtask']}{f}{mark}")
        return

    segs, sheets = load_index()
    LABELS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in LABELS.glob("*.json")}
    eps = sorted({(s["ds"], s["ep"]) for s in segs}, key=lambda x: (ORDER.index(x[0]), x[1]))

    if args.episode:
        want = [(args.episode[0], int(args.episode[1]))]
    elif args.next:
        want = [next((e for e in eps if episode_key(*e) not in done), None)]
        if want == [None]:
            print("all episodes labelled")
            return
    else:
        rows, qcount, nmist = [], Counter(), 0
        for p in LABELS.glob("*.json"):
            d = json.load(open(p))
            for s in d["segments"]:
                qcount[s["quality"]] += 1
                nmist += len(s.get("mistakes", []))
                rows.append((d["dataset"], d["episode"], s, bool(s.get("reviewed"))))
        seen = sum(1 for *_, r in rows if r)
        frames = sum(s["to"] - s["from"] for *_, s, _ in rows)
        seen_frames = sum(s["to"] - s["from"] for *_, s, r in rows if r)
        moved = sum(1 for *_, s, r in rows if r and not s.get("anchor_baseline", True))
        print(f"segments {len(rows)}   vision-reviewed {seen} ({seen/len(rows):.0%}), "
              f"{seen_frames/frames:.0%} of frames   moved off anchor {moved}   mistakes {nmist}")
        print("quality " + "  ".join(f"{q}:{qcount[q]}" for q in (5, 4, 3, 2, 1)))
        by_phase = Counter()
        tot_phase = Counter()
        for _ds, _ep, s, r in rows:
            ph = "return" if s["subtask"].startswith("return") else s["subtask"].split()[0]
            tot_phase[ph] += 1
            by_phase[ph] += r
        print("reviewed by phase  " + "  ".join(f"{p}:{by_phase[p]}/{tot_phase[p]}"
                                                for p in ("grasp", "move", "release", "return")))
        flagged = [(ds, ep, s) for ds, ep, s, _ in rows if s["mistakes"]]
        fseen = sum(1 for ds, ep, s, r in rows if s["mistakes"] and r)
        print(f"segments carrying mistakes: {len(flagged)}, of which vision-confirmed {fseen}")
        for ds in ORDER:
            de = [(d, e, s, r) for d, e, s, r in rows if d == ds]
            print(f"  {ds:26s} {sum(1 for *_, r in de if r):3d}/{len(de):3d} segments reviewed")
        nxt = next((e for e in eps if any(
            d == e[0] and ep == e[1] and not r for d, ep, _s, r in rows)), None)
        print(f"next unreviewed episode: {nxt[0]} ep{nxt[1]}" if nxt else "next: -")
        return

    for ds, ep in want:
        print(f"### {ds} ep{ep}")
        mine = [s for s in segs if s["ds"] == ds and s["ep"] == ep]
        for s in mine:
            f = f" fails={len(s['fails'])}" if s["fails"] else ""
            print(f"  seg{s['seg']:<3d} [{s['from']},{s['to']}) {s['dur']:5.1f}s "
                  f"eff={s['eff']:.2f} rev={s['rev']:<2d}{f}  {s['subtask']}")
        print("  sheets:")
        for sh in sheets:
            if sh["ds"] == ds and sh["ep"] == ep:
                print(f"    {SHEETS / sh['name']}")


if __name__ == "__main__":
    main()
