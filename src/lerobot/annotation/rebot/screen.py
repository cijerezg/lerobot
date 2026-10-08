"""Numeric first look at raw recordings or rollouts, before any frame is read: per episode the length,
when the arm leaves home, idle runs (the cut rule), gripper closes, operator-teleop spans and the
subtasks recorded online.

    uv run python -m lerobot.annotation.rebot.screen <root> [<root> ...]
"""

import sys

import pandas as pd

from lerobot.annotation.rebot.episode import episodes, interventions, record, states
from lerobot.annotation.rebot.motion import closed_intervals, idle_runs, leaves_home, runs_of


def screen(root):
    print(f"\n##### {root}")
    labels = pd.read_parquet(f"{root}/meta/online_labels.parquet") if interventions(record(root, 0)) is not None else None
    for ep in episodes(root):
        rec = record(root, ep)
        s, _ = states(rec)
        n = len(s)
        idle = [(a, b, round((b - a) / 30, 1)) for a, b in idle_runs(s, 0, n)]
        line = f"  ep{ep}: {n} fr {n / 30:6.1f}s | leaves home f{leaves_home(s)} | idle >= 3 s: {idle}"
        line += f" | closes: {[(int(a), int(b)) for a, b in closed_intervals(s[:, 6])]}"
        if labels is not None:
            iv = interventions(rec)
            line += f" | teleop {iv.mean() * 100:.1f}% spans={[(int(a), int(b)) for a, b in runs_of(iv)]}"
            # a built root carries the typed subtask as recorded_subtask (build_root renames it)
            sub = labels[labels.episode_index == ep].sort_values("frame_index")
            sub = sub["subtask" if "subtask" in sub else "recorded_subtask"]
            change = sub.ne(sub.shift()).cumsum()
            line += f" | recorded subtasks: {[(v.iloc[0], len(v)) for _, v in sub.groupby(change)]}"
        print(line)


if __name__ == "__main__":
    for root in sys.argv[1:]:
        screen(root)
