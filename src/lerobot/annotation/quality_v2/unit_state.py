"""Current labels of the units of one staging episode of a labelled ReBot pass (revision helper): for every unit
the first reader's spans / mistakes / precision with their refs (span:i, mistake:i, precision), every resolve row
on them and the added rows. Frames are staging-root frames (traces, strips). The ref numbering is the FIRST
READER's: a fix row must use it. Pool and dataset name come from <work>/pass.json.

    uv run python -m lerobot.annotation.quality_v2.unit_state <work> <staging_episode>
"""

import glob
import json
import sys
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2

KEYS = ("cause", "type", "raw_from", "raw_to", "from_index", "to_index", "grade", "confidence", "note", "what_happens")

if __name__ == "__main__":
    config = json.loads((Path(sys.argv[1]) / "pass.json").read_text())
    ep = int(sys.argv[2])
    for d in sorted((QUALITY_V2 / "classes").glob(f"{config['pool']}__*")):
        units = [json.loads(line)["uid"] for line in open(d / "units.jsonl")]
        units = [u for u in units if u.startswith(f"{config['dataset']}_ep{ep}_")]
        if not units:
            continue
        first = {}
        for f in glob.glob(str(d / "labels/*.jsonl")):
            for x in map(json.loads, filter(str.strip, open(f))):
                first[x["uid"]] = (Path(f).name, x)
        res = {}
        for f in sorted(glob.glob(str(d / "resolve/*.jsonl"))):
            for x in map(json.loads, filter(str.strip, open(f))):
                res.setdefault(x["uid"], []).append((Path(f).name, x))
        for u in sorted(units, key=lambda s: int(s.rsplit("seg", 1)[1])):
            name, r = first[u]
            print(f"\n=== {u}  class {d.name}  first reader {name}")
            print(f"  what_happens: {r.get('what_happens', '')}\n  strategy: {r.get('strategy', '')}")
            for kind, prefix in (("spans", "span"), ("mistakes", "mistake")):
                for i, row in enumerate(r.get(kind, [])):
                    print(f"  {prefix}:{i}", json.dumps({k: row.get(k) for k in KEYS if k in row}))
            if r.get("precision"):
                print("  precision", json.dumps(r["precision"]))
            for fname, x in res.get(u, []):
                for row in x.get("rows", []):
                    verdict = f"  [{fname}] {row['ref']}: {row['verdict']}"
                    print(verdict, json.dumps(row.get("row", ""))[:300], (row.get("note") or "")[:200])
                for kind, rows in x.get("added", {}).items():
                    for row in rows:
                        print(f"  [{fname}] ADDED {kind}", json.dumps(row)[:300])
