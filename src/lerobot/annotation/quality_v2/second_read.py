"""Second-reader slices for one pool (step 3 pass D): classes/second_read_<pool>.json.

uv run python -m lerobot.annotation.quality_v2.second_read rebot_main [--work <work> ...] [--no-calibrate <class> ...]
--work (before --no-calibrate): a pass whose classes reuse other class files; each slice's `final` is then the class file
named in <work>/class_map.json (a re-annotation pool has no strategy files of its own).
Small classes (<= 16 units to read): one slice for both jobs. Otherwise unsure slices: units with open unsure refs, at most 12 per slice, one class each. Calibration slices: every unit with an
`exemplary` or `strategy` stretch, at most 24 per slice, one class each (pilot classes are not calibrated: user-reviewed).
Units already resolved (a row in <class>/resolve/*.jsonl) are skipped.
"""
import glob, json, math, sys
from pathlib import Path

from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
HERE = QUALITY_V2
pool = sys.argv[1]
skip_cal = set(sys.argv[sys.argv.index("--no-calibrate") + 1:]) if "--no-calibrate" in sys.argv else set()
works = [a for i, a in enumerate(sys.argv) if sys.argv[i - 1] == "--work"]
CMAP = {c["slug"]: str((WORKSPACE / c["final"]).resolve().relative_to(HERE)) for w in works for c in json.loads((WORKSPACE / w / "class_map.json").read_text()).values()}
EXTRA = {"rebot_main__release__deformable_to_surface": "rebot_main__release__sock_to_basket"}


def refs(r):
    out = ["unit"] if r.get("confidence") == "unsure" else []
    out += [f"span:{i}" for i, s in enumerate(r.get("spans", [])) if s.get("confidence") == "unsure"]
    out += [f"mistake:{i}" for i, m in enumerate(r.get("mistakes", [])) if m.get("confidence") == "unsure"]
    if (r.get("precision") or {}).get("confidence") == "unsure": out.append("precision")
    return out


def short(slug):
    return slug[len(pool) + 2:].replace("__", "_").replace("_to_", "_").replace("release", "rel")


slices = {}
dirs = sorted(HERE.glob(f"classes/{pool}__*")) + sorted(HERE.glob(f"pilot/{pool}__*"))
for d in [d for d in dirs if (d / "units.jsonl").exists()]:
    pilot = d.parent.name == "pilot"
    done = {json.loads(x)["uid"] for f in glob.glob(str(d / "resolve/*.jsonl")) for x in open(f) if x.strip()}
    uns, cal = [], []
    for f in sorted(glob.glob(str(d / ("labels_v3" if pilot else "labels") / "*.jsonl"))):
        for r in map(json.loads, filter(str.strip, open(f))):
            if r["uid"] in done: continue
            lf = str(Path(f).relative_to(HERE))
            if refs(r): uns.append(dict(uid=r["uid"], label_file=lf, refs=refs(r)))
            c = [f"span:{i}" for i, s in enumerate(r.get("spans", [])) if s["cause"] in ("exemplary", "strategy")]
            if c and not pilot and d.name not in skip_cal: cal.append(dict(uid=r["uid"], label_file=lf, calib_refs=c))
    final = CMAP.get(d.name) or f"{'pilot' if pilot else 'classes'}/{EXTRA.get(d.name, d.name)}/strategy/FINAL.md"
    tag = ("P_" if pilot else "") + short(d.name)
    if len({u["uid"] for u in uns + cal}) <= 16:  # small class: one slice does both jobs
        m = {}
        for u in uns + cal: m.setdefault(u["uid"], dict(uid=u["uid"], label_file=u["label_file"])).update({k: v for k, v in u.items() if k.endswith("refs")})
        if m: slices[f"D_{tag}_01"] = dict(cls=d.name, final=final, calibrate=bool(cal), units=list(m.values()))
        continue
    for kind, items, n in (("R", uns, 12), ("K", cal, 24)):
        k = max(1, math.ceil(len(items) / n)) if items else 0
        for i in range(k):
            ch = items[i * len(items) // k:(i + 1) * len(items) // k]
            slices[f"{kind}_{tag}_{i + 1:02d}"] = dict(cls=d.name, final=final, calibrate=kind == "K", units=ch)
out = HERE / "classes" / f"second_read_{pool}.json"
json.dump(slices, open(out, "w"), indent=1)
print(out, len(slices), "slices;", sum(len(s["units"]) for s in slices.values()), "unit reads")
