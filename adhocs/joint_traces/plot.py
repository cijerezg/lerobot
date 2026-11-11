"""Recorded action and state per joint over whole rollout episodes, one interactive HTML.

uv run python lerobot/adhocs/joint_traces/plot.py [--n 5] [--seed 0] [--roots ...] [--out ...]
"""

import argparse
import glob
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# rollout roots recorded before the RTC window fix (2026-11-09)
ROOTS = [f"outputs/rebot_2026-{d}-annotated-v1" for d in ["10-23", "10-24", "10-26", "10-27", "10-28", "10-28b"]]

parser = argparse.ArgumentParser()
parser.add_argument("--roots", nargs="+", default=ROOTS)
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min_frames", type=int, default=1000)
parser.add_argument("--out", default="outputs/analysis/joint_traces/joint_traces.html")
args = parser.parse_args()

pool = []
for root in args.roots:
    for e in json.load(open(f"{root}/meta/provenance.json")):
        # spliced episodes have a jump at the cut seam that the robot never made
        if not e["spliced"] and e["keep"][1] - e["keep"][0] >= args.min_frames:
            pool.append((root, e["episode_index"]))
picks = sorted(random.Random(args.seed).sample(pool, min(args.n, len(pool))))

parts = []
for root, ep in picks:
    info = json.load(open(f"{root}/meta/info.json"))
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
    df = df[df.episode_index == ep].sort_values("frame_index")
    t = df.timestamp.to_numpy()
    action = np.stack(df.action.to_numpy())
    state = np.stack(df["observation.state"].to_numpy())

    labels = pd.read_parquet(f"{root}/meta/online_labels.parquet")
    labels = labels[labels.episode_index == ep].sort_values("frame_index")
    interv = labels.is_intervention.to_numpy().astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], interv, [0]])))
    spans = [(t[a], t[b - 1]) for a, b in zip(edges[::2], edges[1::2])]

    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.012, subplot_titles=names)
    for j in range(len(names)):
        fig.add_trace(go.Scattergl(x=t, y=action[:, j], name="action", legendgroup="action", showlegend=j == 0,
                                   line=dict(color="#d62728", width=1.2)), row=j + 1, col=1)
        fig.add_trace(go.Scattergl(x=t, y=state[:, j], name="state", legendgroup="state", showlegend=j == 0,
                                   line=dict(color="#1f77b4", width=1.2)), row=j + 1, col=1)
    for a, b in spans:
        fig.add_vrect(x0=a, x1=b, fillcolor="gray", opacity=0.2, line_width=0, row="all", col=1)
    fig.update_xaxes(title_text="time (s)", row=len(names), col=1)
    fig.update_layout(title=f"{root} episode {ep} ({len(t)} frames, gray = operator intervention)",
                      height=230 * len(names), hovermode="x unified", margin=dict(l=50, r=20, t=70, b=40))
    parts.append(fig.to_html(full_html=False, include_plotlyjs=len(parts) == 0))

with open(args.out, "w") as f:
    f.write("<html><body>" + "\n".join(parts) + "</body></html>")
print(args.out, picks)
