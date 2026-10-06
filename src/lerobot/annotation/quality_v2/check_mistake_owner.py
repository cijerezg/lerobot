"""Rubric 9 check: mistakes labelled outside the unit where their event starts (BRIEF_C: a mistake belongs to the unit holding m_a).

uv run python -m lerobot.annotation.quality_v2.check_mistake_owner <pool>
Reads the merged labels (classes/labels_<pool>.jsonl from compile_labels.py --pool <pool>) and every unit of the pool.
-> classes/mistake_owner_<pool>.csv: one row per mistake whose from_index is outside its unit, with the owning unit and whether
the owner already has a mistake of the same type within 1 s (duplicate) or not (to move).
"""
import csv, json, sys
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
pool = sys.argv[1]
U = {}
for f in list(HERE.glob(f"classes/{pool}__*/units.jsonl")) + list(HERE.glob(f"pilot/{pool}__*/units.jsonl")):
    for u in map(json.loads, open(f)): U[u["uid"]] = dict(u, _cls=f.parent.name)
L = {r["uid"]: r for r in map(json.loads, open(HERE / f"classes/labels_{pool}.jsonl"))}
ep = lambda uid: uid.rsplit("_seg", 1)[0]  # noqa: E731
by_ep = {}
for uid, u in U.items(): by_ep.setdefault(ep(uid), []).append(u)
rows = []
for uid, r in L.items():
    u = U[uid]
    for i, m in enumerate(r.get("mistakes", [])):
        a = int(m["from_index"])
        if u["from_index"] <= a < u["to_index"]: continue
        own = [v for v in by_ep[ep(uid)] if v["from_index"] <= a < v["to_index"]]
        o = own[0]["uid"] if own else ""
        fps = u.get("fps", 30)
        dup = any(x["type"] == m["type"] and abs(int(x["from_index"]) - a) <= fps for x in L.get(o, {}).get("mistakes", []))
        rows.append(dict(uid=uid, cls=u["_cls"], ref=f"mistake:{i}", type=m["type"], from_index=a, unit=f"{u['from_index']}-{u['to_index']}",
                         owner=o, owner_cls=own[0]["_cls"] if own else "", owner_labelled=o in L, duplicate=dup))
out = HERE / f"classes/mistake_owner_{pool}.csv"
with open(out, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["uid"]); w.writeheader(); w.writerows(rows)
print(out, len(rows), "mistakes outside their unit;", sum(r["duplicate"] for r in rows), "duplicates")
