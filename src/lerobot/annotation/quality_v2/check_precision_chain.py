"""Rubric 9 check: a carry takes the precision level of the release that follows it (rubric 6).

uv run python -m lerobot.annotation.quality_v2.check_precision_chain <pool>
Reads classes/labels_<pool>.jsonl (compile_labels.py --pool <pool>) -> prints every carry whose level differs from the next unit's
release level (None = level 1). A carry with a null window because it ends > ~10 cm from the release pose is fine when the
release has the window, so only carries with a window are compared.
"""
import json, sys
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
pool = sys.argv[1]
U = {}
for f in list(HERE.glob(f"classes/{pool}__*/units.jsonl")) + list(HERE.glob(f"pilot/{pool}__*/units.jsonl")):
    for u in map(json.loads, open(f)): U[u["uid"]] = u
L = {r["uid"]: r for r in map(json.loads, open(HERE / f"classes/labels_{pool}.jsonl"))}
lvl = lambda r: (r.get("precision") or {}).get("level", 1) if r else None  # noqa: E731
n = 0
for uid, u in sorted(U.items()):
    if u.get("action") != "carry" or not (L.get(uid) or {}).get("precision"): continue
    nxt = [v for v in U.values() if v["uid"].rsplit("_seg", 1)[0] == uid.rsplit("_seg", 1)[0] and v["from_index"] == u["to_index"]]
    if not nxt or nxt[0].get("action") != "release": continue
    a, b = lvl(L[uid]), lvl(L.get(nxt[0]["uid"]))
    if a != b:
        n += 1; print(f"{uid} carry level {a} ({L[uid]['_file']})  vs  {nxt[0]['uid']} release level {b} ({(L.get(nxt[0]['uid']) or {}).get('_file')})")
print(n, "mismatches")
