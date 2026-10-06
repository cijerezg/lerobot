"""Class references vs final labels: a reference picked as a 5 / 4 / 3 example whose final label has another kind.

uv run python -m lerobot.annotation.quality_v2.check_references <pool> [--write]
Parses the `References:` line of every <pool> class FINAL.md (`5 = uid, uid (note); 4 = ...; 3 = ...`), compares with
classes/labels_<pool>.jsonl (exemplary -> 5, strategy -> 3, else 4). --write appends one note line per class with mismatches.
"""
import json, re, sys
from datetime import date
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
pool = sys.argv[1]
L = {r["uid"]: r for r in map(json.loads, open(HERE / f"classes/labels_{pool}.jsonl"))}
kind = lambda r: 5 if any(s["cause"] == "exemplary" for s in r.get("spans", [])) else 3 if any(s["cause"] == "strategy" for s in r.get("spans", [])) else 4  # noqa: E731
for fin in sorted(list(HERE.glob(f"classes/{pool}__*/strategy/FINAL.md")) + list(HERE.glob(f"pilot/{pool}__*/strategy/FINAL.md"))):
    line = next((x for x in open(fin) if x.startswith("References:")), None)
    if not line: continue
    bad = []
    for g, part in re.findall(r"([345])\s*=\s*([^;]+)", line):
        for uid in re.findall(r"(?:rebot_all|additions|validation)_ep\d+_seg\d+", part):
            if uid in L and kind(L[uid]) != int(g): bad.append(f"{uid} (reference {g}, final {kind(L[uid])})")
    if bad:
        print(fin.parent.parent.name, "; ".join(bad))
        if "--write" in sys.argv:
            with open(fin, "a") as f:
                f.write(f"- Note (main, {date.today()}): references graded otherwise after the second read: {'; '.join(bad)}. "
                        "The reference sheets still show them; the rows and notes above are the rule.\n")
