"""Per-checkpoint read-out of the sweep in outputs/probe_runs/ckpt_sweep_2026-10-17/<step>/{reach,oct07,oct14,oct16}.

Training onset frames (24, operator direction known): seeds with / flat / against the operator's roll direction,
the model's median roll turn as a fraction of the operator's, and per joint the median |model - operator| at the
chunk end (deg) next to the same error for holding still ("hold"), which is what a nonsense checkpoint cannot beat.
Reach frames (robot run 0-8 s, 9 frames): roll neg / flat / pos, and the median commanded change of the
reach joints over the chunk, next to what the robot did.

    uv run python lerobot/adhocs/joint_onset_trace/ckpt_sweep_readout.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SWEEP = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/probe_runs/ckpt_sweep_2026-10-17")
FLAT, ROLL = 5.0, 5
JOINTS = ["pan", "lift", "elbow", "wflex", "yaw", "roll", "grip"]
onsets = pd.read_csv(HERE / "turn_onset_frames.csv")


def load(path):
    d = np.load(path)
    return d["state"], d["sample_chunks"][:, :, -1, :], d["gt_chunk"][:, -1, :], d["measured_states"][:, -1, :], d["episode"], d["frame"]


train_rows, reach_rows, err_rows = [], [], []
for step_dir in sorted(SWEEP.iterdir()):
    step = step_dir.name
    parts = []
    for name in ("oct07", "oct14", "oct16"):
        p = step_dir / name / "action_trace/action_trace_records.npz"
        if not p.exists():
            continue
        st, end, gt, _, ep, fr = load(p)
        o = onsets[onsets.set == name].set_index(["ep", "frame"]).loc[list(zip(ep, fr))]
        sign = np.where(o.op_dir.to_numpy() == "neg", -1.0, 1.0)
        parts.append((st, end, gt, sign))
    if parts:
        st, end, gt, sign = (np.concatenate(x) for x in zip(*parts))
        d = end - st[:, None, :]                      # (frames, K, 7) model change over the chunk
        g = gt - st                                   # (frames, 7) operator change
        r = d[:, :, ROLL] * sign[:, None]
        n = r.size
        ratio = np.median(d[:, :, ROLL], 1) / g[:, ROLL]
        train_rows.append(dict(step=step, frames=len(st), with_op=f"{(r > FLAT).sum() / n:.0%}", flat=f"{(abs(r) <= FLAT).sum() / n:.0%}",
                               against=f"{(r < -FLAT).sum() / n:.0%}", roll_ratio=round(float(np.median(ratio)), 2)))
        err = np.median(np.abs(np.median(d, 1) - g), 0)
        hold = np.median(np.abs(g), 0)
        err_rows.append(dict(step=step, **{j: round(float(e), 1) for j, e in zip(JOINTS, err)}))
        if len(err_rows) == 1:
            err_rows.insert(0, dict(step="hold", **{j: round(float(h), 1) for j, h in zip(JOINTS, hold)}))
    p = step_dir / "reach/action_trace/action_trace_records.npz"
    if p.exists():
        st, end, _, meas, ep, fr = load(p)
        d = end - st[:, None, :]
        m = meas - st
        sel = fr >= 120                               # 4-8 s, when the robot was reaching
        r = d[sel, :, ROLL]
        n = r.size
        med = np.median(d[sel], axis=(0, 1))
        rob = np.median(m[sel], 0)
        reach_rows.append(dict(step=step, neg=f"{(r < -FLAT).sum() / n:.0%}", flat=f"{(abs(r) <= FLAT).sum() / n:.0%}", pos=f"{(r > FLAT).sum() / n:.0%}",
                               **{f"{j}": f"{med[i]:+.0f}/{rob[i]:+.0f}" for i, j in enumerate(JOINTS)}))

pd.set_option("display.width", 200)
print("TRAINING ONSET FRAMES: seeds vs the operator's roll direction; roll_ratio = model's median turn / operator's turn")
print(pd.DataFrame(train_rows).to_string(index=False))
print("\nper joint: median |model median - operator| at the chunk end (deg); 'hold' = the same error for not moving")
print(pd.DataFrame(err_rows).to_string(index=False))
print("\nREACH FRAMES 4-8 s (robot run): roll seeds; per joint 'model/robot' = median commanded change over the chunk / what the robot did (deg)")
print(pd.DataFrame(reach_rows).to_string(index=False))
