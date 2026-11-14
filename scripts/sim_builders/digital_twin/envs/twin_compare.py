"""Compare twin eval runs: per level and per configuration, success rate, mean staged reward, mean dense value, carried time, drops.
usage (LeRobot root): uv run python lerobot/scripts/sim_builders/digital_twin/envs/twin_compare.py outputs/twin_eval_res600_2026-10-30 [outputs/twin_eval_<other> ...] [--md out.md]
Pairing: the same 64 placements in every run; the last column of a two-run table is the per-level mean difference of the dense value over
the common placements with a sign test p-value (episodes paired by placement index)."""
import json, pathlib, sys, statistics as st
from math import comb
md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
runs = [a for a in sys.argv[1:] if not a.startswith("--") and a != md]
def load(run):
    D = next(pathlib.Path(run).glob("validation/step_*/sim")); out = {}
    for ev in sorted(D.glob("twin_*/subtask_events.json")):
        eps = [e for e in json.load(open(ev)) if isinstance(e, dict) and "reward" in e]
        out[ev.parent.name] = eps
    return out
data = {r: load(r) for r in runs}
def level(name): return name.split("_")[1]
def config(name): return name.rsplit("_p", 1)[0]
def agg(eps):
    r = [e["reward"] for e in eps]; dn = [e.get("reward_dense") or 0.0 for e in eps]
    succ = sum(x >= 1.0 for x in r) / len(r); car = st.mean((e.get("carried_time_s") or 0.0) for e in eps)
    drops = [e["drop"]["distance_to_basket_m"] for e in eps if e.get("drop")]
    return dict(n=len(r), success=succ, reward=st.mean(r), dense=st.mean(dn), carried=car, drops=len(drops), drop_m=st.mean(drops) if drops else float("nan"))
lines = []
def emit(s=""): lines.append(s); print(s)
for key, label in ((level, "level"), (config, "configuration")):
    emit(f"\n## by {label}\n"); emit("| run | " + label + " | eps | success | reward | dense | carried s | drops | drop dist m |"); emit("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        groups = {}
        for name, eps in data[r].items(): groups.setdefault(key(name), []).extend(eps)
        for g in sorted(groups):
            a = agg(groups[g]); emit(f"| {pathlib.Path(r).name} | {g} | {a['n']} | {a['success']:.2f} | {a['reward']:.2f} | {a['dense']:.2f} | {a['carried']:.1f} | {a['drops']} | {a['drop_m']:.2f} |")
if len(runs) == 2:
    A, B = (data[r] for r in runs); emit(f"\n## paired difference by level: {pathlib.Path(runs[1]).name} minus {pathlib.Path(runs[0]).name} (dense, per placement means)\n")
    emit("| level | placements | mean diff | wins / losses | sign-test p |"); emit("|---|---|---|---|---|")
    per = {}
    for name in sorted(set(A) & set(B)):
        da = st.mean((e.get("reward_dense") or 0.0) for e in A[name]); db = st.mean((e.get("reward_dense") or 0.0) for e in B[name]); per.setdefault(level(name), []).append(db - da)
    for L, diffs in sorted(per.items()):
        w = sum(d > 0 for d in diffs); l = sum(d < 0 for d in diffs); n = w + l
        p = sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n * 2 if n else float("nan"); p = min(p, 1.0)
        emit(f"| {L} | {len(diffs)} | {st.mean(diffs):+.3f} | {w} / {l} | {p:.3f} |")
if md: pathlib.Path(md).write_text("\n".join(lines) + "\n"); print("wrote", md)
