"""Audit agent reviews against the proposals: what changed, which names, which grades.

    uv run python -m lerobot.annotation.atoms.audit [--batch B] [--episode E] [--names]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from lerobot.annotation.atoms.atoms_common import REVIEW, WORK, load_corpus, read_jsonl
from lerobot.annotation.atoms.verdict import render_subtask

AGENT = REVIEW / "agent_reviews"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--episode")
    ap.add_argument("--names", action="store_true")
    args = ap.parse_args()
    episodes, parents = load_corpus()
    proposals = {(p["episode_id"], p["parent_interval_index"]): p for p in read_jsonl(WORK / "proposals.jsonl")}
    batches = json.loads((WORK / "batches.json").read_text())
    if args.episode:
        ids = [args.episode]
    elif args.batch:
        ids = batches[args.batch]["episodes"]
    else:
        ids = [p.stem for p in sorted(AGENT.glob("*.json"))]
    names = collections.defaultdict(collections.Counter)
    totals = collections.Counter()
    for eid in ids:
        path = AGENT / f"{eid}.json"
        if not path.exists():
            print(f"{eid}: no review")
            continue
        r = json.loads(path.read_text())
        ep = episodes[eid]
        rate = float(ep["native_rate_hz"])
        fam = ep["component"] if ep["source"] == "robochallenge" else ep["source"]
        for pv in r["parents"]:
            pidx = pv["parent_interval_index"]
            prop = proposals[(eid, pidx)]
            prow = next(x for x in parents[eid] if x["interval_index"] == pidx)
            pcuts = {a["start_timestep"] for a in prop["atoms"]} | {prop["parent_end"]}
            acuts = {a["start_timestep"] for a in pv["atoms"]} | {prop["parent_end"]}
            kept = len(pcuts & acuts)
            moved_or_new = len(acuts - pcuts)
            dropped = len(pcuts - acuts)
            pverbs = collections.Counter(a["verb"] for a in prop["atoms"])
            averbs = collections.Counter(a["verb"] for a in pv["atoms"])
            graded = [(a["quality"], a.get("quality_note", "")[:50]) for a in pv["atoms"] if a.get("quality") is not None]
            new_ev = [(me["kind"], me["start_s"], me["end_s"]) for a in pv["atoms"] for me in a.get("new_mistake_events", []) or []]
            for a in pv["atoms"]:
                if a["verb"] != "return":
                    names[fam][(a["verb"], a.get("object"), a.get("container"), a.get("preposition"), a.get("instrument"))] += 1
            totals["parents"] += 1
            totals["atoms"] += len(pv["atoms"])
            totals["cuts_kept"] += kept
            totals["cuts_new"] += moved_or_new
            totals["cuts_dropped"] += dropped
            totals["unsure"] += pv["confidence"] == "unsure"
            totals["graded_atoms"] += len(graded)
            totals["new_events"] += len(new_ev)
            flag = ""
            if pv["confidence"] == "unsure":
                flag += " UNSURE"
            if len(pv["atoms"]) == len(prop["atoms"]) and moved_or_new == 0 and pverbs == averbs:
                flag += " =proposal"
            print(f"{eid} P{pidx} q{prow['quality']} [{prop['parent_start']},{prop['parent_end']}) {(prop['parent_end'] - prop['parent_start']) / rate:.0f}s  proposal {len(prop['atoms'])} atoms {dict(pverbs)} -> {len(pv['atoms'])} atoms {dict(averbs)}; cuts kept {kept} new {moved_or_new} dropped {dropped}{flag}")
            if graded:
                print("    graded:", graded)
            if new_ev:
                print("    new events:", new_ev)
            if args.episode:
                print("    note:", pv["note"])
                for a in pv["atoms"]:
                    s, e = a["start_timestep"], a["end_timestep_exclusive"]
                    print(f"      [{s},{e}) {s / rate:6.1f}-{e / rate:6.1f}s  {render_subtask(a)}")
    print(dict(totals))
    if args.names:
        for fam, c in names.items():
            print("==", fam)
            for k, n in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[1])):
                print(f"  {n:4d}  {k}")


if __name__ == "__main__":
    main()
