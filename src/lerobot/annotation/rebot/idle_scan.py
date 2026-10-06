"""Idle cuts of a labelled pass (user 2026-10-04: the arm "really just sitting" for more than 3-4 s is cut;
mid-episode = splice, end = trim). The rule is ``motion.idle_runs``; ``motion.idle_cut`` keeps 0.5 s each side.

Segments stay labelled in source frames; build_root removes the cut frames. Runs listed in <work>/pass.json
"idle_exempt" (idx, part, run_from, why) are reported but not cut. Staging frames come from the staging root
named in pass.json.

    uv run python -m lerobot.annotation.rebot.idle_scan <work>      -> <work>/idle_runs.json, idle_cuts.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.episode import records, states
from lerobot.annotation.rebot.motion import idle_cut, idle_runs

FPS, MIN_S, THR, MARGIN = 30, 3.0, 2.0, 15

if __name__ == "__main__":
    work = Path(sys.argv[1])
    config = json.loads((work / "pass.json").read_text())
    exempt = {(e["idx"], e["part"], e["run_from"]) for e in config["idle_exempt"]}
    staging = WORKSPACE / config["staging"] / "meta"
    prov = json.loads((staging / "provenance.json").read_text())
    eps = pd.read_parquet(staging / "episodes/chunk-000/file-000.parquet").set_index("episode_index")
    found, cuts = [], []
    for r in records(work):
        d = json.loads((work / f"labels/{r['idx']:02d}.json").read_text())
        st, _ = states(r)
        for j, e in enumerate(d["episodes"]):
            if e.get("split") == "out":
                continue
            a, b = e["keep"]
            home = np.zeros(b - a, bool)
            for s in e["segments"]:
                if s["subtask"] == "return to home":
                    home[s["from_index"] - a : s["to_index"] - a] = True
            for f0, f1 in idle_runs(st, a, b, home, min_s=MIN_S, thr=THR, fps=FPS):
                where = "head" if f0 == a else "tail" if f1 == b else "mid"
                seg = [s["subtask"] for s in e["segments"] if s["from_index"] < f1 and s["to_index"] > f0]
                x = dict(idx=r["idx"], key=r["key"], part=j, keep=[a, b], run=[int(f0), int(f1)])
                x.update(seconds=round((f1 - f0) / FPS, 1), where=where, segments=seg)
                found.append(x)
                if (r["idx"], j, f0) in exempt:
                    continue
                c0, c1 = (int(c) for c in idle_cut((f0, f1), (a, b), MARGIN))
                p = next(p for p in prov if p["inventory_idx"] == r["idx"] and p["part"] == j)
                g = int(eps.loc[p["episode_index"]].dataset_from_index) - p["keep"][0]
                cuts.append(dict(x, cut=[c0, c1], seconds=round((c1 - c0) / FPS, 1)))
                cuts[-1].update(staging_episode=p["episode_index"], staging_cut=[c0 + g, c1 + g])
    for x in cuts:
        print(x["idx"], x["where"], x["cut"], x["seconds"], "s  staging ep", x["staging_episode"], x["staging_cut"], x["segments"])
    print(len(found), "runs,", len(cuts), "cuts,", round(sum(x["seconds"] for x in cuts), 1), "s cut")
    rule = (
        f'every joint and the gripper (arm joints only inside "return to home") within {THR} deg over every 1 s window for '
        f">= {MIN_S} s; cut keeps {MARGIN / FPS} s each side; tail runs trimmed, mid runs spliced (user 2026-10-04)"
    )
    (work / "idle_runs.json").write_text(json.dumps(dict(rule=rule, runs=found), indent=1))
    (work / "idle_cuts.json").write_text(json.dumps(dict(rule=rule, exempt=config["idle_exempt"], cuts=cuts), indent=1))
