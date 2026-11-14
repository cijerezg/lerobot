"""Leader command vs follower path of a recording in the Action Inspector's 3D scene. Reads only, no model.

    uv run python lerobot/adhocs/motor_gains/trace_recording.py --root=<recording> [--n=12] [--episode=0]

Per anchor frame the page (``plots/<root name>_trace.html``) draws the next second of the leader command
(``action``, black, where the policy's target comes from) and, in the slot the probe uses for policy
samples, the path the follower actually took over the same second (``observation.state``). The dotted
segment from the measured pose to the first command is the gap. The page is the action-trace probe's own
(``_analyse``, ``_figure``, ``_write_dashboard_html``); it low-passes whatever sits in the sample slot,
so the follower path is drawn after the deployed Butterworth filter.

Anchors: ``n`` frames evenly spaced over the episode's moving frames (command tip path of the next
second above 20 mm).
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.probes import action_trace_probe
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics

HORIZON = 30
MOVING_MM = 20.0

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--n", type=int, default=12)
parser.add_argument("--episode", type=int, default=0)
args = parser.parse_args()

name = os.path.basename(os.path.normpath(args.root))
dataset = LeRobotDataset(repo_id=f"cijerezg/{name}", root=args.root, episodes=[args.episode])
fps = float(dataset.fps)
d = pd.read_parquet(f"{args.root}/data", columns=["index", "episode_index", "observation.state", "action"]).sort_values("index")
d = d[d.episode_index == args.episode]
state = np.stack(d["observation.state"].values).astype(float)
action = np.stack(d["action"].values).astype(float)

kin = RebotKinematics()
starts = np.arange(0, len(state) - HORIZON, 3)
path_mm = np.array([np.linalg.norm(np.diff(kin.ee_path(action[i : i + HORIZON, :6]), axis=0), axis=1).sum() for i in starts]) * 1000
moving = starts[path_mm > MOVING_MM]
anchors = moving[np.linspace(0, len(moving) - 1, args.n).round().astype(int)]

p = SimpleNamespace(trace_table_z=0.0, trace_clearance_warn_m=0.01)
motor_speeds, joint_lower, joint_upper = action_trace_probe._resolve_actuator_config(None, action.shape[1])
motor_speeds[:6] = 40.0  # the caps the recordings were made with
records = []
for i in anchors:
    record = action_trace_probe._analyse(
        kin, state[i], action[i : i + HORIZON], [state[i : i + HORIZON]], p.trace_table_z,
        fps=fps, motor_speeds_deg_s=motor_speeds, joint_lower=joint_lower, joint_upper=joint_upper,
    )
    obs = {key: value for key, value in dataset[int(i)].items() if key.startswith("observation.images.")}
    gap_mm = 1000 * np.linalg.norm(kin.ee_path(action[i : i + 1, :6])[0] - kin.ee_path(state[i : i + 1, :6])[0])
    record.update(
        episode=args.episode, frame=int(i), global_idx=int(i), subtask="", sample_names=["follower"],
        trace_label=f"ep{args.episode} fr{i} ({i / fps:.1f} s) · gap {gap_mm:.0f} mm",
        cameras=action_trace_probe._camera_context(obs),
        camera_frames=action_trace_probe._camera_frames(obs),
        measured_states=state[i : i + HORIZON],
    )
    records.append(record)
    print(record["trace_label"])

out_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{name}_trace.html")
action_trace_probe._write_dashboard_html(
    action_trace_probe._figure(records, p, fps=fps), records, out_path,
    page_title=f"Command vs follower: {name}",
    subtitle="No model. Black = the leader command over the next second; colored = the path the follower took over the same second.",
    legend_note="Black = leader command (action). Colored = follower path (observation.state). Dotted = gap from the measured pose to the first command.",
)
print(f"saved {out_path}")
