"""Turn-onset anchors for the large-clipper grasps: the first frame of the grasp window where the wrist roll
(observation.state[5], degrees) is more than 10 deg from its value at the window start, in the direction of the
window's net change (oct14 ep 4 dips +10 before its -90 turn), plus 1 s before and after. oct16 ep 3 keeps only the
71 s grasp (its 244 s re-grasp is left out, as agreed).
Writes one trace_frames string per root to turn_onset_<set>.txt and a table to turn_onset_frames.csv
(episode-local frame, snapped down to the 3-frame image grid, which is what trace_frames expects)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOTS = {"oct07": "outputs/rebot_teleop_2026-10-07-annotated-v1-tasks1",
         "oct14": "outputs/rebot_2026-10-14-annotated-v4", "oct16": "outputs/rebot_2026-10-16-annotated-v1"}
GRASPS = {"oct07": [0], "oct14": [0, 2, 4, 10], "oct16": [3, 4, 5]}
WINDOW_T0_MAX = {("oct16", 3): 80.0}  # only the first grasp of that episode
STRIDE, FPS, THRESH = 3, 30, 10.0
rows = []
for name, root in ROOTS.items():
    df = pd.concat([pd.read_parquet(p) for p in sorted(Path(root, "data").rglob("*.parquet"))])
    wins = json.loads(Path(root, "meta/subtask_windows.json").read_text())["episodes"]
    anchors = []
    for ep in GRASPS[name]:
        e = df[df.episode_index == ep]
        st = np.stack(e["observation.state"].to_numpy())
        fr = e["frame_index"].to_numpy()
        gidx = e["index"].to_numpy()
        for w in wins[str(ep)]:
            if not (w["subtask"].startswith("grasp") and "large nail clipper" in w["subtask"]):
                continue
            a, b = int(np.searchsorted(gidx, w["from_index"])), int(np.searchsorted(gidx, w["to_index"] - 1))
            if fr[a] / FPS > WINDOW_T0_MAX.get((name, ep), np.inf):
                continue
            roll = st[a:b + 1, 5]
            sign = np.sign(roll[-1] - roll[0])
            dev = (roll - roll[0]) * sign
            hit = np.flatnonzero(dev > THRESH)
            if len(hit) == 0:
                print(f"{name} ep{ep} window {w['from_index']}: roll never leaves +-{THRESH} deg (range {roll.min():.1f}..{roll.max():.1f}), skipped")
                continue
            onset = a + int(hit[0])
            direction = "pos" if sign > 0 else "neg"
            for tag, i in (("before", onset - FPS), ("onset", onset), ("after", onset + FPS)):
                i = int(np.clip(i, 0, len(fr) - 1))
                f = int(fr[i] - fr[i] % STRIDE)
                i = int(np.searchsorted(fr, f))
                anchors.append(f"{ep}:{f}")
                rows.append(dict(set=name, root=root, ep=ep, tag=tag, frame=f, global_idx=int(gidx[i]), t_s=round(f / FPS, 2),
                                 window_t0=round(fr[a] / FPS, 1), in_window=a <= i <= b, roll=round(st[i, 5], 1),
                                 roll_w0=round(roll[0], 1), roll_wend=round(roll[-1], 1), op_dir=direction,
                                 roll_plus1s=round(st[min(i + FPS, len(st) - 1), 5], 1)))
    Path(HERE, f"turn_onset_{name}.txt").write_text(",".join(anchors))
t = pd.DataFrame(rows)
t.to_csv(HERE / "turn_onset_frames.csv", index=False)
print(t.drop(columns="root").to_string(index=False))
