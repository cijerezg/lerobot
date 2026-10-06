"""Run the proprio triage over every corpus episode and write proposals.jsonl.

    uv run python -m lerobot.annotation.atoms.propose [--stats]
"""

from __future__ import annotations

import argparse
import collections
import sys

import numpy as np

from lerobot.annotation.atoms.atoms_common import WORK, episode_phases, load_corpus, parent_proposal, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--episode", default=None)
    args = ap.parse_args()
    episodes, parents = load_corpus()
    proposals = []
    per_source = collections.defaultdict(lambda: collections.Counter())
    release_s = collections.defaultdict(list)
    move_s = collections.defaultdict(list)
    grasp_s = collections.defaultdict(list)
    for eid, ep in episodes.items():
        if args.episode and eid != args.episode:
            continue
        info = episode_phases(eid, ep["embodiment"], float(ep["native_rate_hz"]), source=ep["source"])
        rate = float(ep["native_rate_hz"])
        src = ep["source"]
        per_source[src]["episodes"] += 1
        per_source[src]["carries"] += len(info["carries"])
        per_source[src]["failed_closes"] += len(info["failed"])
        per_source[src]["events"] += len(info["events"])
        for r in info["carries"]:
            if r["open"] is not None:
                release_s[src].append((r["open_end"] - r["arrival"]) / rate)
                move_s[src].append((r["arrival"] - r["close"]["frame"]) / rate)
        for parent in parents[eid]:
            p = parent_proposal(ep, parent, info)
            proposals.append(p)
            per_source[src]["parents"] += 1
            per_source[src]["atoms"] += len(p["atoms"])
            for a in p["atoms"]:
                per_source[src][f"verb:{a['verb']}"] += 1
                if a["verb"] == "grasp":
                    grasp_s[src].append((a["end_timestep_exclusive"] - a["start_timestep"]) / rate)
        if args.episode:
            print(eid, ep["task"])
            print("  events:", [(e["kind"], e["frame"], round(e["frame"] / rate, 1), e["before"], e["after"]) for e in info["events"]])
            print("  carries:", [(r["close"]["frame"], r["arrival"], r["open_begin"], r["open_end"], r["disp_rad"]) for r in info["carries"]])
            print("  failed:", [(r["close"]["frame"], r["stop"], r["disp_rad"]) for r in info["failed"]])
            for p in [q for q in proposals if q["episode_id"] == eid]:
                print(f"  P{p['parent_interval_index']} [{p['parent_start']},{p['parent_end']}) q{p['parent_quality']} {p['parent_subtask']}")
                for a in p["atoms"]:
                    print(f"     [{a['start_timestep']},{a['end_timestep_exclusive']}) {(a['end_timestep_exclusive']-a['start_timestep'])/rate:5.1f}s {a['verb']} cycle={a['cycle']} {a['start_provenance']}")
    if not args.episode:
        write_jsonl(WORK / "proposals.jsonl", proposals)
        print(f"wrote {len(proposals)} proposals to {WORK / 'proposals.jsonl'}")
    if args.stats or not args.episode:
        for src, c in per_source.items():
            print(src, dict(c))
            for name, d in (("release_s", release_s), ("move_s", move_s), ("grasp_s", grasp_s)):
                if d[src]:
                    x = np.array(d[src])
                    print(f"   {name}: n={len(x)} p10/50/90 = {np.percentile(x, [10, 50, 90]).round(1).tolist()}")


if __name__ == "__main__":
    main()
