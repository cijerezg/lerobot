"""Material for the agents of a pass under annotation_rubric.md, one folder per window (a stretch of one episode).

    uv run python -m lerobot.annotation.rebot.window_prepare <work>

Reads <work>/windows.json = {name: {"root": ..., "episode": E, "start": A, "end": B}} (end exclusive, clipped to the
episode). Writes <work>/<name>/: contact sheets (top over wrist, every 10 frames), signal plots, signals.csv and
window.json (the window + `holds`, the stretches where the gripper is gripping something).
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from lerobot.annotation.corrections.arm import Arm
from lerobot.annotation.rebot.episode import record, sheet

FPS = 30
WORK = Path(sys.argv[1])
windows = json.loads((WORK / "windows.json").read_text())
columns = ["episode_index", "frame_index", "observation.state", "action", "observation.effort"]
roots = {w["root"] for w in windows.values()}
data = {root: pd.concat(pd.read_parquet(f, columns=columns) for f in sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))) for root in roots}
arm = Arm()
for name, w in windows.items():
    root, ep, a, b = w["root"], w["episode"], w["start"], w["end"]
    out = WORK / name
    out.mkdir(exist_ok=True)
    g = data[root][data[root].episode_index == ep].sort_values("frame_index")
    q, act, tq = np.stack(g["observation.state"]), np.stack(g["action"]), np.stack(g["observation.effort"])
    b = min(b, len(q))
    tip = np.array([arm.gripper.fk(x)[0] for x in q[a:b]]) * 100  # cm
    speed = np.r_[np.linalg.norm(np.diff(tip, axis=0), axis=1) * FPS, 0]
    smooth = uniform_filter1d(speed, 9)
    wobble = np.linalg.norm(tip - uniform_filter1d(tip, 15, axis=0), axis=1) * 10  # mm off the 0.5 s mean path
    lag = np.linalg.norm(act[a:b, :6] - q[a:b, :6], axis=1)  # deg, command minus position, arm joints
    torque, hold, on = uniform_filter1d(tq[:, 6], 5), np.zeros(len(tq), bool), False
    for i, t in enumerate(torque):  # a hold starts above 0.4 N*m and lasts until the torque is back under 0.15 (it sags in long holds)
        on = t > (0.15 if on else 0.4)
        hold[i] = on
    cuts = np.flatnonzero(np.diff(np.r_[0, hold.astype(int), 0]))
    holds = [(int(s), int(e)) for s, e in zip(cuts[::2], cuts[1::2]) if e - s >= 30 and a <= s < b]
    events = sorted([s for s, _ in holds] + [min(e, b - 1) for _, e in holds])
    dist = np.full(b - a, np.nan)  # cm from the tip to where it is at the next hold start / hold end
    prev = a
    for ev in events:
        dist[prev - a : ev - a] = np.linalg.norm(tip[prev - a : ev - a] - tip[ev - a], axis=1)
        prev = ev
    x = np.arange(a, b)
    pd.DataFrame(dict(frame=x, grip_angle=q[a:b, 6], grip_command=act[a:b, 6], grip_torque=tq[a:b, 6], tip_speed_cm_s=speed, tip_speed_smooth=smooth,
                      dist_to_next_event_cm=dist, wobble_mm=wobble, command_lag_deg=lag, tip_height_cm=tip[:, 2])).round(3).to_csv(out / "signals.csv", index=False)
    json.dump(dict(root=root, episode=ep, start=a, end=b, episode_frames=len(q), fps=FPS, holds=holds), open(out / "window.json", "w"), indent=1)
    rows = [("gripper angle (black) / command (red), deg; 0 = shut", [(q[a:b, 6], "k"), (act[a:b, 6], "r")]), ("gripper torque, N*m", [(tq[a:b, 6], "k")]),
            ("tip speed cm/s (grey raw, black 0.3 s mean)", [(speed, "0.7"), (smooth, "k")]), ("distance to the tip position at the next gripper event, cm", [(dist, "k")]),
            ("wobble: tip off its 0.5 s mean path, mm", [(wobble, "k")]), ("command minus position, arm joints, deg", [(lag, "k")])]
    for lo in range(a, b, 600):
        hi = min(b, lo + 600)
        fig, axes = plt.subplots(len(rows), 1, figsize=(20, 14), sharex=True)
        for ax, (title, lines) in zip(axes, rows):
            for y, c in lines:
                ax.plot(x[lo - a : hi - a], y[lo - a : hi - a], color=c, lw=1)
            for s, e in holds:
                if s < hi and e > lo:
                    ax.axvspan(max(s, lo), min(e, hi), color="orange", alpha=0.2)
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(alpha=0.3)
        axes[-1].set_xticks(np.arange(lo, hi + 1, 30))
        axes[-1].tick_params(axis="x", labelsize=7, rotation=90)
        axes[0].set_xlim(lo, lo + 600)
        fig.suptitle(f"{name}  episode {ep}  frames {lo}-{hi}  (orange = gripper holding, torque rose above 0.4 N*m, until under 0.15)")
        fig.tight_layout()
        fig.savefig(out / f"signals_{lo:05d}-{hi:05d}.png", dpi=70)
        plt.close(fig)
    rec = record(root, ep)
    frames = list(range(a, b, 10))
    for p in range(0, len(frames), 24):
        part = frames[p : p + 24]
        sheet(rec, part, out / f"sheet_{part[0]:05d}-{part[-1]:05d}.jpg", cols=6)
    print(name, ep, a, b, "holds", holds)
