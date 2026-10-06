"""Rubric 9: agreement of a blind second grading (5 % sample) with the final labels, default-4 scheme.

uv run python -m lerobot.annotation.quality_v2.blind_agreement <pool>
Final labels: classes/labels_<pool>.jsonl (compile_labels.py --pool <pool>). Blind labels: <class dir>/labels_blind/*.jsonl.
-> classes/blind_agreement_<pool>.json: per unit and overall
  kind      the unit's stretch kind (exemplary / strategy / none), exact match
  low       has a span of grade 1-2, match
  mistakes  same multiset of mistake types
  frames    share of the unit's frames with the same grade (spans of the unit only, headroom as in compile_labels)
"""
import glob, json, sys
from collections import Counter
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
pool = sys.argv[1]
U = {}
for f in list(HERE.glob(f"classes/{pool}__*/units.jsonl")) + list(HERE.glob(f"pilot/{pool}__*/units.jsonl")):
    for u in map(json.loads, open(f)): U[u["uid"]] = u
F = {r["uid"]: r for r in map(json.loads, open(HERE / f"classes/labels_{pool}.jsonl"))}
B = {}
for f in glob.glob(str(HERE / f"classes/{pool}__*/labels_blind/*.jsonl")) + glob.glob(str(HERE / f"pilot/{pool}__*/labels_blind/*.jsonl")):
    for r in map(json.loads, filter(str.strip, open(f))): B[r["uid"]] = r


def kind(r):
    c = {s["cause"] for s in r.get("spans", [])}
    return "exemplary" if "exemplary" in c else "strategy" if "strategy" in c else "none"


def frames(r, u):
    a0, a1, fps = u["from_index"], u["to_index"], u.get("fps", 30)
    g = [4] * (a1 - a0)
    for s in r.get("spans", []):
        gr = int(s["grade"])
        lo, hi = (s["raw_from"], s["raw_to"]) if gr == 5 else (s["raw_from"] - fps, s["raw_to"] + 0.5 * fps)
        for t in range(max(int(lo), a0), min(int(hi), a1)):
            i = t - a0
            g[i] = gr if gr == 5 and g[i] == 4 else min(g[i], gr) if gr < 5 else g[i]
    return g


rows = []
for uid, b in sorted(B.items()):
    a = F.get(uid)
    if not a: continue
    ga, gb = frames(a, U[uid]), frames(b, U[uid])
    rows.append(dict(uid=uid, cls=a["_cls"], kind_final=kind(a), kind_blind=kind(b),
                     low_final=any(int(s["grade"]) <= 2 for s in a.get("spans", [])), low_blind=any(int(s["grade"]) <= 2 for s in b.get("spans", [])),
                     mistakes_final=sorted(m["type"] for m in a.get("mistakes", [])), mistakes_blind=sorted(m["type"] for m in b.get("mistakes", [])),
                     frames_same=round(sum(x == y for x, y in zip(ga, gb)) / max(1, len(ga)), 3)))
n = len(rows)
summ = dict(units=n,
            kind_exact=sum(r["kind_final"] == r["kind_blind"] for r in rows),
            kind_confusion=Counter(f"{r['kind_final']}->{r['kind_blind']}" for r in rows if r["kind_final"] != r["kind_blind"]),
            low_agree=sum(r["low_final"] == r["low_blind"] for r in rows),
            mistakes_agree=sum(r["mistakes_final"] == r["mistakes_blind"] for r in rows),
            frames_same_mean=round(sum(r["frames_same"] for r in rows) / max(1, n), 3))
json.dump(dict(summary=summ, units=rows), open(HERE / f"classes/blind_agreement_{pool}.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
