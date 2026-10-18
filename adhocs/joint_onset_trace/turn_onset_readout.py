"""Read-out for the turn-onset action-trace runs: per anchor frame, how many of the K seeds turn the wrist roll
negative, keep it flat (within 5 deg) or turn it positive by the end of the 1 s chunk, next to the direction the
operator turned (turn_onset_frames.csv). Also summarizes the robot-run baseline the same way.

    uv run python lerobot/adhocs/joint_onset_trace/turn_onset_readout.py \
        outputs/probe_runs/turn_onset_2026-10-17/oct07 outputs/probe_runs/turn_onset_2026-10-17/oct14 \
        outputs/probe_runs/turn_onset_2026-10-17/oct16
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FLAT = 5.0
BASELINE = "outputs/probe_runs/clippers_trace_2026-10-17"
ROLL = 5


def load(run_dir):
    d = np.load(Path(run_dir, "action_trace/action_trace_records.npz"))
    delta = d["sample_chunks"][:, :, -1, ROLL] - d["state"][:, None, ROLL]  # (frames, K)
    gt = d["gt_chunk"][:, -1, ROLL] - d["state"][:, ROLL]
    return pd.DataFrame(dict(ep=d["episode"], frame=d["frame"], state_roll=d["state"][:, ROLL].round(1), gt_delta=gt.round(1),
                             neg=(delta < -FLAT).sum(1), flat=(np.abs(delta) <= FLAT).sum(1), pos=(delta > FLAT).sum(1),
                             median_delta=np.median(delta, 1).round(1)))


def summarize(t, label):
    k = t[["neg", "flat", "pos"]].sum()
    n = k.sum()
    print(f"\n{label}: {len(t)} frames x {n // len(t)} seeds -> neg {k.neg / n:.0%}  flat {k.flat / n:.0%}  pos {k.pos / n:.0%}")


onsets = pd.read_csv(HERE / "turn_onset_frames.csv")
runs = []
for run_dir in sys.argv[1:]:
    t = load(run_dir)
    name = Path(run_dir).name
    t = t.merge(onsets[onsets.set == name][["ep", "frame", "tag", "t_s", "op_dir", "roll_plus1s", "roll"]], on=["ep", "frame"], how="left")
    t["op_delta_1s"] = (t.roll_plus1s - t.roll).round(1)
    t.insert(0, "set", name)
    runs.append(t)
if runs:
    t = pd.concat(runs)
    cols = ["set", "ep", "tag", "t_s", "state_roll", "op_dir", "op_delta_1s", "gt_delta", "neg", "flat", "pos", "median_delta"]
    print(t[cols].to_string(index=False))
    t["with_op"] = np.where(t.op_dir == "neg", t.neg, t.pos)
    t["against_op"] = np.where(t.op_dir == "neg", t.pos, t.neg)
    for tag in ("before", "onset", "after"):
        x = t[t.tag == tag]
        n = x[["neg", "flat", "pos"]].sum().sum()
        print(f"{tag:7s}: {len(x)} frames -> with operator {x.with_op.sum() / n:.0%}  flat {x.flat.sum() / n:.0%}  against {x.against_op.sum() / n:.0%}")
    summarize(t, "training turn-onset frames, all")
if Path(BASELINE).exists():
    summarize(load(BASELINE), "robot run 10-54-00 baseline (31 frames)")
