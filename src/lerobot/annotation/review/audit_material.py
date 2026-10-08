"""Audit material for episodes of an annotated ReBot root (README "Auditing annotated data by random sampling").

    uv run python -m lerobot.annotation.review.audit_material <root> <out dir> <episode> [...] [--onset-thr 25]

Per episode, <out dir>/ep<E>/:
- labels.txt: every label the trainer reads, frames EPISODE-LOCAL (as rebot.sheets numbers them): task, segments
  (text, contact, precision, note), quality spans, mistakes, precision windows; gripper closes and still runs.
- approach_seg<K>.png: the approach line (quality rubric 5.13) of every segment that is not a move or a return, from
  the segment start to its commit (grasp = last close onset, release = first opening, other steps = the segment end).
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.quality_v2.material import PAD0, PAD1, approach_line, commit_of, onsets, plot_approach
from lerobot.annotation.rebot.motion import closed_intervals, still_runs


def action_of(subtask):
    verb = subtask.split()[0]
    if verb == "grasp":
        return "grasp"
    if verb in ("release", "set", "put", "place", "drop"):
        return "release"
    return "end"


def table(meta, name, e):
    p = meta / f"{name}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    return d[d.episode_index == e]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--onset-thr", type=float, default=25.0)
    args = ap.parse_args()
    root = (WORKSPACE / args.root).resolve()
    meta = root / "meta"
    fps = json.loads((meta / "info.json").read_text())["fps"]
    eps = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{meta}/episodes/**/*.parquet", recursive=True)])
    eps = eps.set_index("episode_index")
    tasks = pd.read_parquet(meta / "tasks.parquet").reset_index()
    prov = {p["episode_index"]: p for p in json.loads((meta / "provenance.json").read_text())}
    files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    data = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "observation.state", "task_index"]) for f in files])
    for e in args.episodes:
        d = data[data.episode_index == e].sort_values("index")
        Q = np.stack(d["observation.state"].values).astype(float)
        n, o = len(Q), int(eps.loc[e].dataset_from_index)
        g = Q[:, 6]
        cl, op = onsets(g, 20, args.onset_thr)
        out = WORKSPACE / args.out / f"ep{e}"
        out.mkdir(parents=True, exist_ok=True)
        task = tasks.set_index("task_index").loc[int(d.task_index.iloc[0])]
        p = prov.get(e, {})
        L = [f"{root.relative_to(WORKSPACE)} episode {e}: {n} frames, fps {fps}; frames below are episode-local (global - {o})",
             f"task: {task.iloc[0] if hasattr(task, 'iloc') else task}",
             f"source: {p.get('source')} ep {p.get('source_episode')}",
             "gripper closes (relative travel 25): " + ", ".join(f"[{a},{b})" for a, b in closed_intervals(g)),
             "still runs >= 1 s: " + ", ".join(f"[{a},{b})" for a, b in still_runs(Q)), "", "SEGMENTS"]
        em = table(meta, "episode_metadata", e).sort_values("segment_index")
        co, pr = table(meta, "contact", e), table(meta, "precision", e)
        for s in em.itertuples():
            f0, f1 = int(s.from_index) - o, int(s.to_index) - o
            c = co[co.segment_index == s.segment_index].contact_slug if co is not None else []
            q = pr[pr.segment_index == s.segment_index].precision if pr is not None else []
            line = f"seg{s.segment_index} [{f0},{f1}) {(f1 - f0) / fps:.1f}s  {s.subtask!r}  contact={c.iloc[0] if len(c) else '-'}"
            line += f"  precision={q.iloc[0] if len(q) else '-'}"
            act = action_of(s.subtask)
            if not s.subtask.startswith(("move ", "return ")):
                commit, kind, found = commit_of(act, cl, op, f0, f1, fps)
                w0, w1 = max(0, f0 - int(PAD0 * fps)), min(n, f1 + int(PAD1 * fps))
                dist, turn, rises, flats = approach_line(Q, w0, w1, f0, commit, fps)
                it = dict(window=[w0, w1], from_index=f0, to_index=f1, commit=commit, uid=f"ep{e}_seg{s.segment_index}",
                          subtask=s.subtask)
                plot_approach(out / f"approach_seg{s.segment_index}.png", it, dist, turn, rises, flats)
                line += f"\n    commit f{commit} ({kind}{'' if found else ', not found'}); approach rises {rises} flat {flats}"
            L.append(line + f"\n    note: {str(s.note)[:400]}")
        for name, cols in (("quality_spans", ["raw_from_index", "raw_to_index", "quality", "cause", "confidence", "note"]),
                           ("mistakes", ["from_index", "to_index", "mistake_type", "confidence", "note"]),
                           ("precision_windows", ["segment_index", "commit_index", "from_index", "to_index", "precision"])):
            t = table(meta, name, e)
            L += ["", name.upper() + ("" if t is not None else ": table missing")]
            for r in (t.itertuples() if t is not None else []):
                v = {k: getattr(r, k) for k in cols}
                for k in v:
                    if k.endswith("index") and k != "segment_index":
                        v[k] = int(v[k]) - o
                L.append("  " + "  ".join(f"{k}={str(x)[:300]}" for k, x in v.items()))
        (out / "labels.txt").write_text("\n".join(L) + "\n")
        print(out / "labels.txt", len(em), "segments")
