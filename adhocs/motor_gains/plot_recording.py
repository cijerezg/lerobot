"""Leader command vs follower state of a recording, per joint, as one zoomable page. Reads only.

    uv run python lerobot/adhocs/motor_gains/plot_recording.py --root=<recording> [--ref=<recording at the old gain>]

Page (``plots/<root name>.html``), one row per joint:
  left   command (``action``) and state (``observation.state``) over time, all episodes in a row; zoom is shared
  right  spectrum of the joint speed (frame difference x fps) of command and state, 0 to fps/2; with ``--ref``
         the reference recording's two spectra are drawn dashed
It also prints the rms joint speed in the tremor band (4 to 12 Hz) of command and state.
"""
import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
FPS = 30
TREMOR_HZ = (4.0, 12.0)

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--ref", default=None)
args = parser.parse_args()


def load(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = pd.read_parquet(f"{root}/data", columns=["index", "episode_index", "observation.state", "action"]).sort_values("index")
    return np.stack(d["action"].values).astype(float), np.stack(d["observation.state"].values).astype(float), d["episode_index"].values


def speed_spectrum(x: np.ndarray, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Welch spectrum of the joint speed, averaged over episodes by length. Returns (Hz, deg/s per sqrt(Hz))."""
    total, weight = 0.0, 0
    for e in np.unique(episode):
        v = np.diff(x[episode == e]) * FPS
        if len(v) < 256:
            continue
        freq, power = welch(v, fs=FPS, nperseg=256)
        total, weight = total + power * len(v), weight + len(v)
    return freq, np.sqrt(total / weight)


def band_rms(freq: np.ndarray, amplitude: np.ndarray) -> float:
    band = (freq >= TREMOR_HZ[0]) & (freq <= TREMOR_HZ[1])
    return float(np.sqrt((amplitude[band] ** 2).sum() * (freq[1] - freq[0])))


action, state, episode = load(args.root)
t = np.arange(len(action)) / FPS
ref = load(args.ref) if args.ref is not None else None

fig = make_subplots(rows=len(JOINTS), cols=2, column_widths=[0.72, 0.28], shared_xaxes="columns", vertical_spacing=0.02, horizontal_spacing=0.05, subplot_titles=[s for name in JOINTS for s in (name, f"{name} speed spectrum")])
print(f"rms joint speed in {TREMOR_HZ[0]:g} to {TREMOR_HZ[1]:g} Hz, deg/s")
print(f"{'joint':14s} {'command':>8s} {'state':>8s}" + (f" {'ref command':>12s} {'ref state':>10s}" if ref is not None else ""))
for j, name in enumerate(JOINTS):
    first = j == 0
    fig.add_trace(go.Scattergl(x=t, y=action[:, j], name="command (leader)", line={"color": "#d62728", "width": 1}, legendgroup="a", showlegend=first), row=j + 1, col=1)
    fig.add_trace(go.Scattergl(x=t, y=state[:, j], name="state (follower)", line={"color": "#1f77b4", "width": 1}, legendgroup="s", showlegend=first), row=j + 1, col=1)
    freq, spec_action = speed_spectrum(action[:, j], episode)
    _, spec_state = speed_spectrum(state[:, j], episode)
    fig.add_trace(go.Scatter(x=freq, y=spec_action, line={"color": "#d62728", "width": 1}, legendgroup="a", showlegend=False), row=j + 1, col=2)
    fig.add_trace(go.Scatter(x=freq, y=spec_state, line={"color": "#1f77b4", "width": 1}, legendgroup="s", showlegend=False), row=j + 1, col=2)
    line = f"{name:14s} {band_rms(freq, spec_action):8.2f} {band_rms(freq, spec_state):8.2f}"
    if ref is not None:
        _, ref_action = speed_spectrum(ref[0][:, j], ref[2])
        _, ref_state = speed_spectrum(ref[1][:, j], ref[2])
        fig.add_trace(go.Scatter(x=freq, y=ref_action, name="reference command", line={"color": "#d62728", "width": 1, "dash": "dot"}, legendgroup="ra", showlegend=first), row=j + 1, col=2)
        fig.add_trace(go.Scatter(x=freq, y=ref_state, name="reference state", line={"color": "#1f77b4", "width": 1, "dash": "dot"}, legendgroup="rs", showlegend=first), row=j + 1, col=2)
        line += f" {band_rms(freq, ref_action):12.2f} {band_rms(freq, ref_state):10.2f}"
    print(line)
    fig.update_yaxes(title_text="deg", row=j + 1, col=1)
    fig.update_yaxes(type="log", title_text="deg/s / sqrt(Hz)", row=j + 1, col=2)
for boundary in np.flatnonzero(np.diff(episode)) + 1:
    fig.add_vline(x=boundary / FPS, line={"color": "gray", "dash": "dash", "width": 1}, col=1)
fig.update_xaxes(title_text="s", row=len(JOINTS), col=1)
fig.update_xaxes(title_text="Hz", row=len(JOINTS), col=2)
name = os.path.basename(os.path.normpath(args.root))
fig.update_layout(height=260 * len(JOINTS), title=f"{name}: command vs state" + (f" (dashed: {os.path.basename(os.path.normpath(args.ref))})" if ref is not None else ""), hovermode="x unified", legend={"orientation": "h", "y": 1.02})

out_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{name}.html")
fig.write_html(out_path, include_plotlyjs=True)
print(f"saved {out_path}")
