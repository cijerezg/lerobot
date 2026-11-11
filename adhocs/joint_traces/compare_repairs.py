"""Candidate repairs of the chunk-seam steps in pre-fix rollout actions, overlaid on the recorded action and state.

uv run python lerobot/adhocs/joint_traces/compare_repairs.py [--roots ...] [--n 3] [--seed 0] [--out ...]

Nothing is written to a dataset: one HTML with the overlays, and a table on stdout.
"""

import argparse
import glob
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d, median_filter

ARM = 6  # joints 0-5; the gripper is left alone

parser = argparse.ArgumentParser()
parser.add_argument("--roots", nargs="+", default=["outputs/rebot_2026-10-28-annotated-v1"])
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min_frames", type=int, default=1000)
parser.add_argument("--seam_deg", type=float, default=1.0, help="one-frame step (worst arm joint) that counts as a seam")
parser.add_argument("--max_deg", type=float, default=15.0, help="steps above this are not seams, left alone")
parser.add_argument("--bridge_half", type=int, default=6)
parser.add_argument("--blend_half", type=int, default=8)
parser.add_argument("--lowpass_sigma", type=float, default=3.0)
parser.add_argument("--lowpass_half", type=int, default=10)
parser.add_argument("--out", default="outputs/analysis/joint_traces/compare_repairs.html")
args = parser.parse_args()


def policy_runs(interv):
    edges = np.flatnonzero(np.diff(np.concatenate([[1], interv.astype(int), [1]])))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 8]


def find_seams(a):
    """a: one policy run (T, ARM). Seam s = step between frames s-1 and s. Returns seams, per-joint steps, skipped."""
    d1 = np.diff(a, axis=0)
    step = d1 - median_filter(d1, size=(5, 1), mode="nearest")  # step on top of the local velocity
    worst = np.abs(step).max(1)
    hit = np.flatnonzero(worst > args.seam_deg)
    keep = worst[hit] <= args.max_deg
    return hit[keep] + 1, step[hit[keep]], hit[~keep] + 1


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def bridge(a, seams, steps):
    """Cubic through the window around each seam, position and velocity matched to the untouched action outside."""
    out = a.copy()
    half = args.bridge_half
    groups = np.split(seams, np.flatnonzero(np.diff(seams) > 2 * half) + 1) if len(seams) else []
    for g in groups:
        lo, hi = max(g[0] - half, 3), min(g[-1] + half - 1, len(a) - 4)
        if hi - lo < 2:
            continue
        n = hi - lo
        v0 = (a[lo] - a[lo - 3]) / 3 * n
        v1 = (a[hi + 3] - a[hi]) / 3 * n
        u = (np.arange(n + 1) / n)[:, None]
        h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
        h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
        out[lo : hi + 1] = h00 * a[lo] + h10 * v0 + h01 * a[hi] + h11 * v1
    return out


def offset_blend(a, seams, steps):
    """Spread each step over the window: the old chunk fades into the new one, the detail of both is kept."""
    out = a.copy()
    t = np.arange(len(a))
    for s, d in zip(seams, steps):
        ramp = smoothstep((t - s + args.blend_half + 0.5) / (2 * args.blend_half))
        out -= ((t >= s) - ramp)[:, None] * d
    return out


def local_lowpass(a, seams, steps):
    """Gaussian low-pass, cross-faded in only around the seams."""
    w = np.zeros(len(a))
    t = np.arange(len(a))
    for s in seams:
        w = np.maximum(w, 0.5 * (1 + np.cos(np.pi * np.clip((t - s + 0.5) / args.lowpass_half, -1, 1))))
    smooth = gaussian_filter1d(a, args.lowpass_sigma, axis=0, mode="nearest")
    return a + w[:, None] * (smooth - a)


METHODS = {"bridge": (bridge, "#2ca02c"), "offset blend": (offset_blend, "#9467bd"), "local low-pass": (local_lowpass, "#ff7f0e")}


def load(root):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
    labels = pd.read_parquet(f"{root}/meta/online_labels.parquet")[["episode_index", "frame_index", "is_intervention"]]
    return df.merge(labels, how="left")


def repair_episode(e):
    action = np.stack(e.action.to_numpy()).astype(np.float64)
    interv = e.is_intervention.to_numpy().astype(bool)
    fixed = {m: action.copy() for m in METHODS}
    seam_frames, skipped_frames = [], []
    for lo, hi in policy_runs(interv):
        seams, steps, skipped = find_seams(action[lo:hi, :ARM])
        seam_frames += list(seams + lo)
        skipped_frames += list(skipped + lo)
        for m, (fn, _) in METHODS.items():
            fixed[m][lo:hi, :ARM] = fn(action[lo:hi, :ARM], seams, steps)
    return action, interv, fixed, np.array(seam_frames, dtype=int), np.array(skipped_frames, dtype=int)


def jerk(action, interv):
    """Per policy frame: change of action velocity, worst arm joint. Also the 30-frame chunk flags."""
    d2 = np.abs(np.diff(action[:, :ARM], 2, axis=0)).max(1)
    ok = ~(interv[1:-1] | interv[:-2] | interv[2:])
    c = np.concatenate([[0], np.cumsum((d2 > 3) & ok)])
    return d2[ok], (c[30:] - c[:-30]) > 0


pool = []
for root in args.roots:
    for e in json.load(open(f"{root}/meta/provenance.json")):
        if not e["spliced"] and e["keep"][1] - e["keep"][0] >= args.min_frames:
            pool.append((root, e["episode_index"]))
picks = sorted(random.Random(args.seed).sample(pool, min(args.n, len(pool))))

# table over every unspliced episode of the roots
rows = {m: dict(d2=[], chunks=[], change=[], touched=[]) for m in ["recorded", *METHODS]}
n_seams = n_skipped = n_policy = 0
data = {root: load(root) for root in args.roots}
episodes = {}
for root, ep in sorted(set(pool)):
    e = data[root][data[root].episode_index == ep].sort_values("frame_index")
    action, interv, fixed, seams, skipped = repair_episode(e)
    episodes[(root, ep)] = (e, action, interv, fixed, seams, skipped)
    n_seams += len(seams)
    n_skipped += len(skipped)
    n_policy += (~interv).sum()
    for m, x in [("recorded", action), *fixed.items()]:
        d2, chunks = jerk(x, interv)
        rows[m]["d2"].append(d2)
        rows[m]["chunks"].append(chunks)
        diff = np.abs(x[~interv, :ARM] - action[~interv, :ARM]).max(1)
        rows[m]["change"].append(diff)
print(f"{len(episodes)} episodes, {n_policy} policy frames, {n_seams} seams repaired "
      f"(one per {n_policy / max(n_seams, 1):.0f} frames), {n_skipped} steps > {args.max_deg} deg left alone")
print(f"{'':16s} {'>1 deg':>8s} {'>3 deg':>8s} {'p99':>6s} {'max':>6s} {'chunks':>8s} {'touched':>8s} {'mean chg':>9s} {'max chg':>8s}")
for m, r in rows.items():
    d2, chunks, change = np.concatenate(r["d2"]), np.concatenate(r["chunks"]), np.concatenate(r["change"])
    moved = change > 0.01
    print(f"{m:16s} {100 * (d2 > 1).mean():7.2f}% {100 * (d2 > 3).mean():7.2f}% {np.percentile(d2, 99):6.2f} {d2.max():6.1f} "
          f"{100 * chunks.mean():7.1f}% {100 * moved.mean():7.1f}% {change[moved].mean() if moved.any() else 0:9.2f} {change.max():8.2f}")

parts = []
for root, ep in picks:
    e, action, interv, fixed, seams, skipped = episodes[(root, ep)]
    info = json.load(open(f"{root}/meta/info.json"))
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]][:ARM]
    t = e.timestamp.to_numpy()
    state = np.stack(e["observation.state"].to_numpy())
    edges = np.flatnonzero(np.diff(np.concatenate([[0], interv.astype(int), [0]])))

    fig = make_subplots(rows=ARM, cols=1, shared_xaxes=True, vertical_spacing=0.012, subplot_titles=names)
    for j in range(ARM):
        def line(y, name, color, width=1.2, **kw):
            fig.add_trace(go.Scattergl(x=t, y=y, name=name, legendgroup=name, showlegend=j == 0,
                                       line=dict(color=color, width=width), **kw), row=j + 1, col=1)
        line(state[:, j], "state", "#1f77b4")
        line(action[:, j], "action (recorded)", "#d62728")
        for m, (_, color) in METHODS.items():
            line(fixed[m][:, j], m, color, width=1.6)
        for frames, name, symbol in [(seams, "seam", "line-ns-open"), (skipped, "large step, left alone", "x")]:
            fig.add_trace(go.Scattergl(x=t[frames], y=action[frames, j], mode="markers", name=name, legendgroup=name,
                                       showlegend=j == 0, marker=dict(symbol=symbol, color="black", size=9)), row=j + 1, col=1)
    for a, b in zip(edges[::2], edges[1::2]):
        fig.add_vrect(x0=t[a], x1=t[b - 1], fillcolor="gray", opacity=0.2, line_width=0, row="all", col=1)
    fig.update_xaxes(title_text="time (s)", row=ARM, col=1)
    fig.update_layout(title=f"{root} episode {ep} ({len(t)} frames, {len(seams)} seams, gray = operator intervention; "
                            "click a legend entry to hide it)",
                      height=260 * ARM, hovermode="x unified", margin=dict(l=50, r=20, t=70, b=40))
    parts.append(fig.to_html(full_html=False, include_plotlyjs=len(parts) == 0))

with open(args.out, "w") as f:
    f.write("<html><body>" + "\n".join(parts) + "</body></html>")
print(args.out, picks)
