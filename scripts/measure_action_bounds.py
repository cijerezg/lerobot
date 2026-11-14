"""Per-joint bounds for the inference action clips, measured over the training roots.

Everything is in follower degrees, rig joint order (7 wide). Prints three tables and a
YAML block for config_rl.yaml (8 wide: the padding column gets a zero entry).

  workspace  : action column q0.1 / q99.9 / min / max          -> action_clamp_limits
  step       : |a[t+1] - a[t]| within an episode, high quantiles -> action_step_limits
  excursion  : a[t0+k] - s[t0] (anchor delta), q01 / q99 at k    -> action_delta_limits
  lag        : the same envelope at k=0 (follower lag behind the leader) -> action_lag_limits
               from the embodiment stats file (deployed representation) AND recomputed
               from the raw columns as a cross-check.

Usage: uv run python lerobot/scripts/measure_action_bounds.py
"""

import glob

import numpy as np
import pandas as pd
import torch

from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerConfig

ROOTS = [
    "outputs/rebot_socks_basket-annotated-v2",
    "outputs/rebot_shirts_bin-annotated-v2",
    "outputs/rebot_two_container-annotated-v2",
    "outputs/rebot_sorting_clothes_v4-multitask-annotated-v1",
]
STATS = "outputs/stats/action_stats_anchor_diverse-rebot-v2.pt"
STATS_ROW = "rebot_b601_joint7_commanded"
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
CHUNK = 30
GRIPPER_ALIAS_BELOW = -269.0  # frames at/below this are the Damiao turn-aliasing constant

WORKSPACE_PAD_DEG = 10.0  # added outside q0.1 / q99.9, then capped by the driver's joint_limits
DELTA_MARGIN = 2.0        # times max(|q01|, |q99|) over k
LAG_MARGIN = 2.0          # times max(|q01|, |q99|) at k = 0
STEP_MARGIN = 1.5        # times the max per-step delta

frames = []
for root in ROOTS:
    files = sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet"))
    df = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "observation.state"]) for f in files
    )
    df["root"] = root
    frames.append(df)
df = pd.concat(frames, ignore_index=True).sort_values(["root", "episode_index", "frame_index"], kind="stable")
A = np.stack(df["action"].to_numpy()).astype(np.float64)[:, :7]
S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[:, :7]
ep = (df["root"] + "/" + df["episode_index"].astype(str)).to_numpy()
n_eps = len(np.unique(ep))
print(f"{len(ROOTS)} roots, {n_eps} episodes, {len(A)} frames")
alias = A[:, 6] <= GRIPPER_ALIAS_BELOW
print(f"gripper frames at/below {GRIPPER_ALIAS_BELOW}: {alias.sum()} ({100 * alias.mean():.2f}%)\n")


def table(title, rows, cols):
    print(title)
    print(f"{'joint':14s}" + "".join(f"{c:>10s}" for c in cols))
    for name, vals in zip(NAMES, rows):
        print(f"{name:14s}" + "".join(f"{v:10.2f}" for v in vals))
    print()


# ---- workspace ---------------------------------------------------------------------
qs = [0.1, 99.9]
ws = np.stack([np.percentile(A, qs[0], axis=0), np.percentile(A, qs[1], axis=0), A.min(0), A.max(0)], axis=1)
table("WORKSPACE (action column, deg)", ws, ["q0.1", "q99.9", "min", "max"])

# ---- per-step -----------------------------------------------------------------------
same = ep[1:] == ep[:-1]
consecutive = (df["frame_index"].to_numpy()[1:] - df["frame_index"].to_numpy()[:-1]) == 1
d = np.abs(A[1:] - A[:-1])[same & consecutive]
alias_pair = (alias[1:] | alias[:-1])[same & consecutive]
print(f"per-step pairs: {len(d)}, pairs touching a gripper alias frame: {alias_pair.sum()}")
pq = [99, 99.9, 99.99]
st = np.stack([*(np.percentile(d, q, axis=0) for q in pq), d.max(0)], axis=1)
table("PER-STEP |a[t+1]-a[t]| (deg/tick), all pairs", st, ["p99", "p99.9", "p99.99", "max"])
dc = d[~alias_pair]
stc = np.stack([*(np.percentile(dc, q, axis=0) for q in pq), dc.max(0)], axis=1)
table("PER-STEP |a[t+1]-a[t]| (deg/tick), alias pairs excluded", stc, ["p99", "p99.9", "p99.99", "max"])

# ---- excursion: stats file ----------------------------------------------------------
stats = torch.load(STATS, map_location="cpu", weights_only=False)
row = stats["row_names"].index(STATS_ROW)
q01 = stats["stats"]["action"]["q01"][row, :, :7].numpy()  # [CHUNK, 7]
q99 = stats["stats"]["action"]["q99"][row, :, :7].numpy()
ks = [0, 1, 5, 15, CHUNK - 1]
ex = np.stack([*(np.stack([q01[k], q99[k]], 1).reshape(7, 2) for k in ks)], 1).reshape(7, -1)
table(
    f"EXCURSION a[t0+k]-s[t0] from {STATS} row {row!r} (deg), q01/q99 at k={ks}",
    ex, [f"{s}@{k}" for k in ks for s in ("q01", "q99")],
)
env_file = np.maximum(np.abs(q01), np.abs(q99)).max(0)
lag_file = np.maximum(np.abs(q01[0]), np.abs(q99[0]))

# ---- excursion: raw cross-check at k = CHUNK-1 --------------------------------------
k = CHUNK - 1
idx = np.arange(len(A) - k)
valid = ep[idx] == ep[idx + k]
raw = A[idx + k][valid] - S[idx][valid]
rx = np.stack([np.percentile(raw, 1, axis=0), np.percentile(raw, 99, axis=0), raw.min(0), raw.max(0)], 1)
table(f"EXCURSION raw cross-check at k={k} (deg)", rx, ["q01", "q99", "min", "max"])

# ---- YAML ---------------------------------------------------------------------------
driver = RebotB601FollowerConfig(port="").joint_limits
lo = np.maximum(ws[:, 0] - WORKSPACE_PAD_DEG, [driver[n][0] for n in NAMES])
hi = np.minimum(ws[:, 1] + WORKSPACE_PAD_DEG, [driver[n][1] for n in NAMES])
D = DELTA_MARGIN * env_file
L = LAG_MARGIN * lag_file
r = STEP_MARGIN * stc[:, 3]
print(f"YAML (workspace = q0.1/q99.9 -/+ {WORKSPACE_PAD_DEG} deg capped by driver joint_limits; D = {DELTA_MARGIN} x max|q01,q99| over k;"
      f" L = {LAG_MARGIN} x max|q01,q99| at k=0; r = {STEP_MARGIN} x max step, alias pairs excluded; 8th entry = padding column)")
print(f"  action_lag_limits: [{', '.join(f'{v:.1f}' for v in L)}, 0.0]")
pairs = ", ".join(f"[{a:.1f}, {b:.1f}]" for a, b in zip(lo, hi))
print(f"  action_clamp_limits: [{pairs}, [0.0, 0.0]]")
print(f"  action_delta_limits: [{', '.join(f'{v:.1f}' for v in D)}, 0.0]")
print(f"  action_step_limits: [{', '.join(f'{v:.2f}' for v in r)}, 0.0]")
