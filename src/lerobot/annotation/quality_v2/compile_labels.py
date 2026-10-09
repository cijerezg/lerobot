"""Pass C (default-4 scheme): compile the labels and build example sheets per frame grade.

uv run python -m lerobot.annotation.quality_v2.compile_labels [--pool <pool>] [--sheets] [--deterministic-resolvers]
Without --pool: the step 2 pilot (pilot/<class>/labels_v3) -> pilot/summary_v3.json, pilot/review_v3/.
With --pool: every classes/<pool>__*/labels and pilot/<pool>__*/labels_v3, second-reader resolutions from <class>/resolve/
-> classes/summary_<pool>.json (per class and per dataset, raw counts for progress.py), classes/labels_<pool>.jsonl
(merged final labels), and with --sheets classes/review_<pool>/img/grade<g>_<k>.jsonl (two sheets per grade).
Frame grade: lowest critique stretch (1-3) with 1 s before / 0.5 s after; else 5 inside an `exemplary` stretch (no headroom);
else 4. Spans and precision windows of every labelled unit of an episode apply to the frames of every unit of that episode.
Precision (rubric 6): p(t) = level if w_a - 1 s <= t < w_b, else 1.
Resolution rows (second reader): {"uid", "reader", "rows": [{"ref": "unit|precision|span:<i>|mistake:<i>|v1_rejected:<i>",
"verdict": "agree|change|remove", "row": {...} (change), "note"}], "added": {"spans": [...], "mistakes": [...]}}.
A `unit` row with verdict change and a full "row" replaces the whole unit (its span / mistake / precision refs then
point into the replaced row). A unit is validated when every unsure row of the first reader has a resolution row.
"""
import csv, glob, json, os, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from lerobot.annotation.paths import QUALITY_V2, WORKSPACE  # noqa: E402
REPO = WORKSPACE; os.chdir(REPO)
BASE = QUALITY_V2
_a = sys.argv; sys.argv = sys.argv[:1]
from lerobot.annotation.precision_contact import render_strips as rs  # noqa: E402
sys.argv = _a
import cv2, numpy as np  # noqa: E402

POOL = sys.argv[sys.argv.index("--pool") + 1] if "--pool" in sys.argv else None
OUTPUT_DIR = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve() if "--output-dir" in sys.argv else None
if POOL:
    DIRS = sorted(BASE.glob(f"classes/{POOL}__*")) + sorted(BASE.glob(f"pilot/{POOL}__*"))
    DIRS = [d for d in DIRS if (d / "units.jsonl").exists()]
    OUT = OUTPUT_DIR or BASE / "classes"
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY, REVIEW = OUT / f"summary_{POOL}.json", OUT / f"review_{POOL}"
else:
    if OUTPUT_DIR is not None:
        raise SystemExit("--output-dir requires --pool")
    DIRS = [BASE / "pilot" / c for c in ("rebot_main__grasp__sock", "molmoact__grasp__cup_mug")]
    OUT, SUMMARY, REVIEW = BASE / "pilot", BASE / "pilot/summary_v3.json", BASE / "pilot/review_v3"
CLASSES = [d.name for d in DIRS]
rng = np.random.default_rng(20260929)


def rows_of(r):
    """(ref, row) for every row that carries a confidence."""
    out = [("unit", r)] + [(f"span:{i}", s) for i, s in enumerate(r.get("spans", []))]
    out += [(f"mistake:{i}", m) for i, m in enumerate(r.get("mistakes", []))]
    if r.get("precision"): out.append(("precision", r["precision"]))
    return out


def merge(r, res):
    """First reader's label with the second reader's resolutions applied; sets _open (unresolved unsure refs)."""
    r = json.loads(json.dumps(r)); done = {}
    for x in (res or {}).get("rows", []): done[x["ref"]] = x
    unsure = [ref for ref, row in rows_of(r) if row.get("confidence") == "unsure"]
    r["_open"] = [ref for ref in unsure if ref not in done]; r["_resolved"] = len([ref for ref in unsure if ref in done])
    # a `unit` change may carry a row (2026-10-07: its content used to be dropped). Fields it carries overlay the unit;
    # a kind it carries (spans / mistakes / precision) replaces that kind, and the same reader's refs into it are moot
    unit = done.get("unit"); urow = (unit or {}).get("row") or {} if (unit or {}).get("verdict") == "change" else {}
    for k in ("what_happens", "strategy", "note", "v1_rejected"):
        if k in urow: r[k] = urow[k]
    for kind, prefix in (("spans", "span"), ("mistakes", "mistake")):
        if kind in urow:
            r[kind] = [dict(row, confidence="sure", second_read="change") for row in urow[kind]]
            done = {ref: x for ref, x in done.items() if not ref.startswith(prefix + ":")}
    if "precision" in urow:
        r["precision"] = dict(urow["precision"], confidence="sure", second_read="change") if urow["precision"] else None
        done.pop("precision", None)
    if any(k.startswith("precision_") for k in urow):  # a unit row carrying extra windows (precision_2, ...) replaces them all
        for k in [k for k in r if k.startswith("precision_")]: del r[k]
        r.update({k: dict(v, confidence="sure", second_read="change") for k, v in urow.items() if k.startswith("precision_")})
    # v1_rejected rows are not confidence-bearing, but a resolver may still
    # remove or replace one after an ownership/validity audit.
    for kind, prefix in (("spans", "span"), ("mistakes", "mistake"),
                         ("v1_rejected", "v1_rejected")):
        new = []
        for i, row in enumerate(r.get(kind, [])):
            x = done.get(f"{prefix}:{i}")
            if x and x["verdict"] == "remove": continue
            if x and x["verdict"] == "change": row = dict(x["row"])
            if x:
                if kind != "v1_rejected": row["confidence"] = "sure"
                row["second_read"] = x["verdict"]
            new.append(row)
        added = (res or {}).get("added", {}).get(kind, []) if kind != "v1_rejected" else []
        seen = {(x.get("cause") or x.get("type"), x.get("raw_from", x.get("from_index")), x.get("raw_to", x.get("to_index"))) for x in new}
        r[kind] = new + [dict(y, second_read="added") for y in added  # a row already in the unit row is not added twice
                         if (y.get("cause") or y.get("type"), y.get("raw_from", y.get("from_index")), y.get("raw_to", y.get("to_index"))) not in seen]
    # extra precision windows from a second reader (a unit with several commits): added.precision_* lists -> precision_<k>
    extra = [w for k, v in (res or {}).get("added", {}).items() if k.startswith("precision") for w in v]
    k0 = 1 + sum(1 for k in r if k.startswith("precision"))
    for i, w in enumerate(extra): r[f"precision_{k0 + i}"] = dict(w, second_read="added")
    for ref in ("unit", "precision"):
        x = done.get(ref)
        if not x: continue
        if ref == "precision":
            if x["verdict"] == "remove": r["precision"] = None
            elif x["verdict"] == "change":
                r["precision"] = dict(x["row"], second_read="change")
            if r["precision"]: r["precision"]["confidence"] = "sure"
        else:
            r["confidence"] = "sure"
    return r


def load():
    U, L = {}, {}
    for d in DIRS:
        for u in map(json.loads, open(d / "units.jsonl")): U[u["uid"]] = dict(u, _cls=d.name)
        sub = "labels_v3" if d.parent.name == "pilot" else "labels"
        # several readers may touch one unit (unsure slice R_/D_, calibration slice K_, main's fix pass F_): combine their rows;
        # on the same ref the later wins (order: R_/D_, then K_, then F_), `added` rows are concatenated
        res = {}
        # The historical behavior preserves glob order inside each R/D, K, F phase. The opt-in flag adds a
        # filename tie-break for reproducible post-grading compiles without changing any existing pool by default.
        def resolver_key(f):
            path = Path(f); name = path.stem.lower()
            kind = "unsure" if "unsure" in name else "precision" if "precision" in name else "other"
            phase = 3 if kind == "precision" else {"K": 1, "F": 2}.get(path.name[0], 0)
            phase = 4 if path.name.startswith("F_audit") else phase  # an audit fix is the latest reading: it wins
            return (phase, path.name) if "--deterministic-resolvers" in sys.argv else (phase,)
        for f in sorted(glob.glob(str(d / "resolve/*.jsonl")), key=resolver_key):
            for x in map(json.loads, filter(str.strip, open(f))):
                m = res.setdefault(x["uid"], {"rows": [], "added": {"spans": [], "mistakes": []}})
                # a later `unit` change carrying a full spans / mistakes list supersedes what earlier readers added
                for y in x.get("rows", []):
                    if y["ref"] == "unit" and y["verdict"] == "change":
                        for k in ("spans", "mistakes"):
                            if k in (y.get("row") or {}): m["added"][k] = []
                        if any(k.startswith("precision_") for k in (y.get("row") or {})):  # and its extra windows supersede added ones
                            for k in [k for k in m["added"] if k.startswith("precision")]: m["added"][k] = []
                m["rows"] += x.get("rows", [])
                for k, v in (x.get("added") or {}).items():
                    have = m["added"].setdefault(k, [])
                    # two readers adding the same row add it once (same values; their notes may differ)
                    same = lambda y: {f: y[f] for f in y if f not in ("note", "what_happens", "looked_at", "confidence")}  # noqa: E731
                    have.extend(y for y in v if same(y) not in map(same, have))
        for f in sorted(glob.glob(str(d / sub / "*.jsonl"))):
            for r in map(json.loads, filter(str.strip, open(f))):
                L[r["uid"]] = dict(merge(r, res.get(r["uid"])), _file=Path(f).stem, _cls=d.name)
    return U, L


def episode_grades(units, labels):
    """Frame grade and precision per frame over the frames of `units` (one episode), from all `labels` of that episode."""
    a0 = min(u["from_index"] for u in units); a1 = max(u["to_index"] for u in units); fps = units[0]["fps"]; n = a1 - a0
    crit = np.full(n, 9); five = np.zeros(n, bool); prec = np.ones(n, int)
    clip = lambda lo, hi: slice(min(max(int(lo), a0), a1) - a0, min(max(int(hi), a0), a1) - a0)  # noqa: E731
    for r in labels:
        for s in r.get("spans", []):
            g = int(s["grade"])
            if g == 5: five[clip(s["raw_from"], s["raw_to"])] = True
            else:
                sl = clip(s["raw_from"] - fps, s["raw_to"] + 0.5 * fps); crit[sl] = np.minimum(crit[sl], g)
        for k in sorted(x for x in r if x.startswith("precision")):  # precision, precision_2, ... (a unit with several commits)
            p = r[k]
            if p and p.get("level", 1) > 1 and p.get("w_a") is not None:
                sl = clip(p["w_a"] - fps, p["w_b"]); prec[sl] = np.maximum(prec[sl], int(p["level"]))
    q = np.where(crit < 9, crit, np.where(five, 5, 4))
    return {u["uid"]: (q[u["from_index"] - a0:u["to_index"] - a0], prec[u["from_index"] - a0:u["to_index"] - a0]) for u in units}


def compile_all(U, L):
    done = {uid for uid, r in L.items() if not r["_open"]}
    by_ep = defaultdict(list)
    for u in U.values(): by_ep[(u["dataset"], u["episode"])].append(u)
    FQ, FP = {}, {}
    for key, us in by_ep.items():
        us = [u for u in us if u["uid"] in done]
        if us:
            for uid, (q, p) in episode_grades(us, [L[u["uid"]] for u in us]).items(): FQ[uid], FP[uid] = q, p
    blank = lambda: dict(units_total=0, units_labelled=0, units_done=0, rows=0, unsure_rows=0, unsure_open=0, second_read=0,  # noqa: E731
                         units_exemplary=0, units_strategy=0, units_pure4=0, frames=Counter(), prec_frames=Counter(),
                         spans_by_grade=Counter(), span_frames_by_grade=Counter(), spans_by_cause=Counter(), span_frames_by_cause=Counter(),
                         low_no_mistake=0, low_no_mistake_frames=0, mistakes=Counter(), mistakes_new=0, v1_rejected=0,
                         prec_steps_2plus=0, prec_windows=0, checks=dict(mistake_outside_low_span=[], long_spans=[], idle_unlabelled=[]))
    per_cls, per_ds = defaultdict(blank), defaultdict(blank)
    seen_m = defaultdict(list)
    for uid, u in U.items():
        for R in (per_cls[u["_cls"]], per_ds[u["dataset"]]): R["units_total"] += 1
        r = L.get(uid)
        if r is None: continue
        rw = rows_of(r)
        for R in (per_cls[u["_cls"]], per_ds[u["dataset"]]):
            R["units_labelled"] += 1; R["rows"] += len(rw); R["unsure_open"] += len(r["_open"]); R["second_read"] += r["_resolved"]
            R["unsure_rows"] += len(r["_open"]) + r["_resolved"]
        if uid not in done: continue
        sp, fps = r.get("spans", []), u["fps"]
        ep, dups = seen_m[(u["dataset"], u["episode"])], []  # one event labelled by two units (crosses a boundary): count once
        for m in r.get("mistakes", []):
            dups.append(any(t == m["type"] and a < m["to_index"] and m["from_index"] < b and o != uid for o, t, a, b in ep))
            ep.append((uid, m["type"], m["from_index"], m["to_index"]))
        for R in (per_cls[u["_cls"]], per_ds[u["dataset"]]):
            R["units_done"] += 1
            R["units_exemplary"] += any(s["cause"] == "exemplary" for s in sp); R["units_strategy"] += any(s["cause"] == "strategy" for s in sp)
            R["units_pure4"] += not sp
            R["frames"].update(FQ[uid].tolist()); R["prec_frames"].update(FP[uid].tolist())
            for s in sp:
                g, nf = int(s["grade"]), int(s["raw_to"]) - int(s["raw_from"])
                R["spans_by_grade"][g] += 1; R["span_frames_by_grade"][g] += nf; R["spans_by_cause"][s["cause"]] += 1; R["span_frames_by_cause"][s["cause"]] += nf
                if g <= 2 and not any(s["raw_from"] <= m["from_index"] < s["raw_to"] for m in r.get("mistakes", [])):
                    R["low_no_mistake"] += 1; R["low_no_mistake_frames"] += nf
                if nf > 20 * fps: R["checks"]["long_spans"].append(f"{uid} {s['cause']} {s['raw_from']}-{s['raw_to']}")
            for m, dup in zip(r.get("mistakes", []), dups):
                if not dup:
                    R["mistakes"][m["type"]] += 1; R["mistakes_new"] += str(m.get("v1_row", "")).startswith("new")
                if not any(s["grade"] <= 2 and s["raw_from"] <= m["from_index"] and s["raw_to"] >= m["to_index"] for s in sp):
                    R["checks"]["mistake_outside_low_span"].append(f"{uid} {m['type']} {m['from_index']}-{m['to_index']}")
            R["v1_rejected"] += len(r.get("v1_rejected", []))
            p = r.get("precision")
            R["prec_steps_2plus"] += (p or {}).get("level", 1) > 1; R["prec_windows"] += bool(p and p.get("w_a") is not None)
            for run in u.get("still_runs", []):
                lo, hi = (run[0], run[1]) if isinstance(run, (list, tuple)) else (run.get("from"), run.get("to"))
                if hi - lo > fps and not any(s["cause"] == "idle" and s["raw_from"] < hi and s["raw_to"] > lo for s in sp):
                    R["checks"]["idle_unlabelled"].append(f"{uid} {lo}-{hi}")
    return per_cls, per_ds


def jsonable(d):
    return {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, Counter) else v) for k, v in d.items()}


U, L = load()
per_cls, per_ds = compile_all(U, L)
json.dump(dict(pool=POOL, classes={c: jsonable(v) for c, v in per_cls.items()}, datasets={c: jsonable(v) for c, v in per_ds.items()}),
          open(SUMMARY, "w"), indent=1)
if POOL:
    with open(OUT / f"labels_{POOL}.jsonl", "w") as f:
        for uid in sorted(L): f.write(json.dumps(dict({k: v for k, v in L[uid].items() if k not in ("_open", "_resolved")}, validated=not L[uid]["_open"])) + "\n")
for c, v in per_cls.items():
    tot = sum(v["frames"].values()) or 1
    print(c, f"{v['units_done']}/{v['units_labelled']}/{v['units_total']} done/labelled/total", "ex", v["units_exemplary"], "strat", v["units_strategy"],
          "pure4", v["units_pure4"], "frames%", {k: round(100 * v["frames"][k] / tot, 1) for k in (5, 4, 3, 2, 1)},
          "mistakes", dict(v["mistakes"]), "unsure open", v["unsure_open"], "checks", {k: len(x) for k, x in v["checks"].items()})


if "--sheets" in sys.argv:
    out = REVIEW / "img"; out.mkdir(parents=True, exist_ok=True)
    cand = {g: [] for g in (5, 4, 3, 2, 1)}
    for uid, r in L.items():
        u = U[uid]; fps = u["fps"]; c = u["_cls"]
        for s in r.get("spans", []):
            if s.get("confidence") == "unsure": continue
            g = int(s["grade"]); pad = int(0.5 * fps)
            cand[g].append((c, uid, s["raw_from"] - pad, s["raw_to"] + pad, f"grade {g} {s['cause']}", s["what_happens"]))
        if not r.get("spans"):  # pure 4: the commit window
            cand[4].append((c, uid, u["commit"] - int(4 * fps), u["commit"] + int(1.5 * fps), "grade 4 (default, no stretch)", r["what_happens"]))
    jobs, index = [], []
    for g, cs in cand.items():  # 8 per grade: round robin over classes in random order, a new (class, cause) first
        by_cls = defaultdict(list)
        for x in cs: by_cls[x[0]].append(x)
        order = list(by_cls); rng.shuffle(order)
        for xs in by_cls.values(): rng.shuffle(xs)
        pick, seen = [], Counter()
        for cap in (1, 2, 99):
            for c in order * 8:
                xs = [x for x in by_cls[c] if x not in pick and seen[(c, x[4])] < cap]
                if xs and len(pick) < 8: pick.append(xs[0]); seen[(c, xs[0][4])] += 1
            if len(pick) >= 8: break
        pick.sort(key=lambda x: CLASSES.index(x[0]))
        for k in range(0, len(pick), 4):
            parts = []
            for c, uid, a, b, tag, what in pick[k:k + 4]:
                u = U[uid]; e0, e1 = u["episode_range"]; a, b = max(a, e0), min(b, e1)
                gg = np.zeros(e1 + 1); gg[e0:e1] = u["gripper_episode"]
                rr = dict(u["render"], row_id=f"{uid} | {tag}", subtask=what[:70], from_index=a, to_index=b, len_s=round((b - a) / u["fps"], 1))
                rr["_video"] = {kk: tuple(v) for kk, v in rr["_video"].items()}
                parts.append((rr, gg))
            p = str(out / f"grade{g}_{k // 4 + 1}.jpg")
            jobs.append((p, [x[0] for x in parts], None, [x[1] for x in parts]))
            index.append(dict(sheet=Path(p).name, grade=g, items=[dict(cls=x[0], uid=x[1], window=[int(x[2]), int(x[3])], tag=x[4], what=x[5]) for x in pick[k:k + 4]]))

    def run(j):
        p, rows, _, gs = j
        cv2.imwrite(p, cv2.cvtColor(np.vstack([rs.strip(r, g) for r, g in zip(rows, gs)]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return p
    with Pool(12) as pool:
        for p in pool.imap_unordered(run, jobs): print(p, flush=True)
    json.dump(index, open(REVIEW / "index.json", "w"), indent=1)
