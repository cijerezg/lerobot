"""Do the external ReBot root and the diverse corpus carry chunk-seam steps? Random episodes, same detector, one HTML.

uv run python lerobot/adhocs/joint_traces/check_other_sources.py [--n 4] [--seed 0] [--out ...]

Detector as in lerobot.annotation.rebot.seam_repair_root (one-frame step over the local velocity, 1-15 deg, worst arm
joint). Diverse joint positions are radians and are converted to degrees; the last dimension (gripper) is left out.
A pre-fix rollout episode is printed first as the reference.
"""

import argparse
import glob
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import median_filter

EXTERNAL = "outputs/external_rebot_train-annotated-v9-tasks1"
DIVERSE = "outputs/diverse_robot_dataset_v3_round18_screen125_corrected_v4_tasks2"
REFERENCE = ("outputs/rebot_2026-10-29-merged-v1", 3)  # pre-fix rollout

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="outputs/analysis/joint_traces/other_sources.html")
args = parser.parse_args()
rng = random.Random(args.seed)


def rebot_episode(root, ep):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
    df = df[df.episode_index == ep].sort_values("frame_index")
    names = [n.removesuffix(".pos") for n in json.load(open(f"{root}/meta/info.json"))["features"]["action"]["names"]]
    return np.stack(df.action.to_numpy()), np.stack(df["observation.state"].to_numpy()), df.timestamp.to_numpy(), names


def diverse_episode(store, row):
    d = f"{DIVERSE}/{store}/{row['directory']}"
    meta = json.load(open(f"{d}/episode.json"))
    if store == "fmb":  # actions are not joint positions; shown in native units
        action, state = np.load(f"{d}/actions.npy"), np.load(f"{d}/q.npy")
        names = [f"dim_{j}" for j in range(action.shape[1])]
    else:
        action, state = np.load(f"{d}/action.npy"), np.load(f"{d}/state.npy")
        action[:, :-1], state[:, :-1] = np.degrees(action[:, :-1]), np.degrees(state[:, :-1])
        names = meta.get("joint_names") or [f"dim_{j}" for j in range(action.shape[1])]
    return action, state, np.load(f"{d}/timestamp_s.npy"), names


def seams(action):
    a = action[:, :-1].astype(np.float64)
    d1 = np.diff(a, axis=0)
    worst = np.abs(d1 - median_filter(d1, size=(5, 1), mode="nearest")).max(1)
    hit = np.flatnonzero((worst > 1.0) & (worst <= 15.0)) + 1
    d2 = np.abs(np.diff(a, 2, axis=0)).max(1)
    gap = np.median(np.diff(hit)) if len(hit) > 2 else float("nan")
    return hit, 100 * len(hit) / len(a), gap, 100 * (d2 > 3).mean(), np.percentile(d2, 99)


picks = [("reference: pre-fix rollout", f"{REFERENCE[0]} ep {REFERENCE[1]}", rebot_episode(*REFERENCE))]
n_ext = json.load(open(f"{EXTERNAL}/meta/info.json"))["total_episodes"]
for ep in sorted(rng.sample(range(n_ext), args.n)):
    picks.append(("external ReBot", f"ep {ep}", rebot_episode(EXTERNAL, ep)))
rows = [(s, json.loads(line)) for s in ["corpus", "fmb"] for line in open(f"{DIVERSE}/{s}/episodes.jsonl")]
for store, row in rng.sample(rows, args.n):
    label = f"{row.get('episode_id', row['directory'])} ({row.get('native_rate_hz', '?')} Hz, action {row.get('action_source', 'native')})"
    picks.append((f"diverse {row.get('source', store)}", label, diverse_episode(store, row)))

print(f"{'source':28s} {'frames':>6s} {'seams':>6s} {'per 100 fr':>10s} {'median gap':>10s} {'d2 > 3 deg':>10s} {'d2 p99':>7s}  episode")
parts = []
for source, label, (action, state, t, names) in picks:
    hit, rate, gap, over3, p99 = seams(action)
    print(f"{source:28s} {len(action):6d} {len(hit):6d} {rate:10.2f} {gap:10.0f} {over3:9.2f}% {p99:7.2f}  {label}")
    n = len(names)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.012, subplot_titles=names)
    for j in range(n):
        for y, name, color in [(action[:, j], "action", "#d62728"), (state[:, j] if j < state.shape[1] else None, "state", "#1f77b4")]:
            if y is not None:
                fig.add_trace(go.Scattergl(x=t, y=y, name=name, legendgroup=name, showlegend=j == 0,
                                           line=dict(color=color, width=1.2)), row=j + 1, col=1)
        if j < n - 1:
            fig.add_trace(go.Scattergl(x=t[hit], y=action[hit, j], mode="markers", name="detected step", legendgroup="hit",
                                       showlegend=j == 0, marker=dict(symbol="line-ns-open", color="black", size=9)), row=j + 1, col=1)
    fig.update_xaxes(title_text="time (s)", row=n, col=1)
    fig.update_layout(title=f"{source}: {label} ({len(t)} frames, {len(hit)} detected steps)", height=230 * n,
                      hovermode="x unified", margin=dict(l=50, r=20, t=70, b=40))
    parts.append(fig.to_html(full_html=False, include_plotlyjs=len(parts) == 0))

with open(args.out, "w") as f:
    f.write("<html><body>" + "\n".join(parts) + "</body></html>")
print(args.out)
