"""Candidate anchor/donor pairs for the input_swap probe's hand-picked mode.

Usage: .venv/bin/python lerobot/scripts/probes/pick_input_swap_pairs.py [dataset_root] [--refined]
(--refined needs pinocchio: ranks by where the two demonstrations END in end-effector space).

Pairs are two frames at nearly the same arm pose whose demonstrations then diverge, so a
cell that follows the images has somewhere different to go. Windows:

* reach start: 0.3-2 s into a "grasp ..." segment (the arm leaves the container/home pose);
* approach:    0.5-3 s before a gripper close (the arm is homing on the object);
* carry:       0.5-2 s before a gripper open (holding, heading for the container).
"""
import glob
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
root = sys.argv[1] if len(sys.argv) > 1 else "outputs/rebot_val-annotated-v3"
STRIDE, H, FPS = 3, 30, 30
df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "index", "observation.state", "action"])
               for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))).sort_values("index")
S = np.stack(df["observation.state"].values).astype(np.float64); A = np.stack(df["action"].values).astype(np.float64)
ep = df.episode_index.values; fr = df.frame_index.values
em = pd.read_parquet(f"{root}/meta/episode_metadata.parquet")
ev = pd.read_parquet(f"{root}/meta/depth_gripper_events.parquet")
sub = np.array([""] * len(df), dtype=object); seg_id = np.full(len(df), -1)
for r in em.itertuples():
    sub[r.from_index:r.to_index] = r.subtask; seg_id[r.from_index:r.to_index] = r.Index
key = {(e, f): i for i, (e, f) in enumerate(zip(ep, fr))}
last = {e: int(fr[ep == e].max()) for e in np.unique(ep)}

def on_grid(i):
    return fr[i] % STRIDE == 0 and fr[i] + H <= last[ep[i]]

def event_windows(event_type, lo, hi):
    out = []
    for r in ev[ev.event_type == event_type].itertuples():
        for f in range(r.frame_index - hi, r.frame_index - lo + 1):
            if (r.episode_index, f) in key and on_grid(key[(r.episode_index, f)]):
                out.append((key[(r.episode_index, f)], f"{(r.frame_index - f) / FPS:.1f}s before {event_type}"))
    return out

def reach_starts(lo_s=0.3, hi_s=2.0):
    out = []
    for r in em[em.subtask.str.startswith("grasp")].itertuples():
        for i in range(r.from_index + int(lo_s * FPS), min(r.to_index, r.from_index + int(hi_s * FPS) + 1)):
            if on_grid(i):
                out.append((i, f"{(i - r.from_index) / FPS:.1f}s into segment"))
    return out

def delta(i):
    return A[i:i + H] - S[i]

def pairs(win, name, max_arm=6.0, cross="episode"):
    rows = []
    for a, where_a in win:
        for d, where_d in win:
            same_ep = ep[d] == ep[a]
            if cross == "episode" and same_ep: continue
            if cross == "segment" and (not same_ep or seg_id[d] == seg_id[a]): continue
            gap = np.abs(S[d, :6] - S[a, :6]).max()
            if gap > max_arm: continue
            rows.append(dict(a=a, d=d, ep_a=ep[a], fr_a=fr[a], ep_d=ep[d], fr_d=fr[d], where_a=where_a, where_d=where_d,
                             arm_gap=round(gap, 1), grip_gap=round(abs(S[d, 6] - S[a, 6])),
                             demo_gap=round(float(np.sqrt(np.mean((delta(d) - delta(a)) ** 2))), 1),
                             a_motion=round(float(np.sqrt(np.mean(delta(a) ** 2))), 1),
                             d_motion=round(float(np.sqrt(np.mean(delta(d) ** 2))), 1),
                             sub_a=sub[a], sub_d=sub[d]))
    res = pd.DataFrame(rows)
    print(f"\n=== {name} [{cross}]: {len(win)} frames, {len(res)} pairs with arm gap <= {max_arm} deg")
    if len(res):
        best = res.sort_values("demo_gap", ascending=False).drop_duplicates("a").head(25)
        print(best.to_string(index=False))
    return res

reach = reach_starts()
pairs(reach, "REACH START", cross="episode")
pairs(reach, "REACH START", cross="segment")
pairs(event_windows("close", 15, 90), "APPROACH (0.5-3 s before close)", max_arm=8.0, cross="episode")
pairs(event_windows("open", 15, 60), "CARRY (0.5-2 s before open)", cross="episode")


# ── Refined: matched grippers, both humans moving, ranked by where the demos END ─────
def ee_terminal(i):
    return KIN.ee_path(A[i:i + H, :6].astype(np.float64))[-1]

def refined(win, name, max_arm=8.0, max_grip=40.0, min_motion=5.0, cross="episode"):
    rows = []
    for a, where_a in win:
        for d, where_d in win:
            same_ep = ep[d] == ep[a]
            if cross == "episode" and same_ep: continue
            if cross == "segment" and (not same_ep or seg_id[d] == seg_id[a]): continue
            gap = np.abs(S[d, :6] - S[a, :6]).max()
            if gap > max_arm or abs(S[d, 6] - S[a, 6]) > max_grip: continue
            am, dm = np.sqrt(np.mean(delta(a) ** 2)), np.sqrt(np.mean(delta(d) ** 2))
            if min(am, dm) < min_motion: continue
            rows.append(dict(a=a, d=d, ep_a=ep[a], fr_a=fr[a], ep_d=ep[d], fr_d=fr[d], where_a=where_a, where_d=where_d,
                             arm_gap=round(gap, 1), grip_gap=round(abs(S[d, 6] - S[a, 6])),
                             start_cm=round(100 * np.linalg.norm(KIN.ee_path(S[a, :6][None])[0] - KIN.ee_path(S[d, :6][None])[0]), 1),
                             end_cm=round(100 * np.linalg.norm(ee_terminal(a) - ee_terminal(d)), 1),
                             demo_gap=round(float(np.sqrt(np.mean((delta(d) - delta(a)) ** 2))), 1),
                             a_motion=round(float(am), 1), d_motion=round(float(dm), 1), sub_a=sub[a], sub_d=sub[d]))
    res = pd.DataFrame(rows)
    print(f"\n=== REFINED {name} [{cross}]: {len(res)} pairs (arm<={max_arm}, grip<={max_grip}, both moving >= {min_motion})")
    if len(res):
        res["group"] = list(zip(res.ep_a, res.ep_d, res.sub_a, res.sub_d))
        best = res.sort_values("end_cm", ascending=False).drop_duplicates("group").head(30)
        print(best.drop(columns="group").to_string(index=False))
    return res

if "--refined" in sys.argv:
    from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
    KIN = RebotKinematics()
    refined(reach_starts(0.3, 2.5), "REACH START", cross="episode")
    refined(reach_starts(0.3, 2.5), "REACH START", cross="segment")
    refined(event_windows("close", 15, 90), "APPROACH", max_arm=10.0, cross="episode")
    refined(event_windows("close", 15, 90), "APPROACH", max_arm=10.0, cross="segment")
    refined(event_windows("open", 15, 60), "CARRY", cross="episode")
