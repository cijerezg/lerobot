"""Whole 2026-10-20 21-35-32 episode as a probe root: subtask windows = the subtasks the console sent online
(meta/online_labels.parquet), so probe prompts match the deployed ones. Not an annotation; quality placeholder 0.
Copied from migration/rollout_review_2026-10-17/build_probe_root.py with the paths changed.

    uv run python lerobot/adhocs/intervention_episode_reps/build_probe_root.py
"""

import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot import build_root, verify_root

SRC = "outputs/train/2026-10-20/21-35-32_rebot-mix40-ckpt1200-20261020/inference_dataset"
OUT = WORKSPACE / "outputs/rebot_rollouts_2026-10-20-probe-v1"

o = pd.read_parquet(WORKSPACE / SRC / "meta/online_labels.parquet").sort_values("frame_index")
end = len(o) - len(o) % 3
o = o[o.frame_index < end]
starts = o.index[o.subtask != o.subtask.shift()]
segments = []
for i, s in enumerate(starts):
    f0 = int(o.frame_index[s])
    f1 = int(o.frame_index[starts[i + 1]]) if i + 1 < len(starts) else end
    segments.append(dict(from_index=f0, to_index=f1, subtask=o.subtask[s], what_happens="recorded online subtask"))
row = dict(source=SRC, episode=0, keep=[0, end], cuts=[], task="Tidy up the table", key=f"{SRC} ep0",
           segments=segments, mistakes=[],
           provenance=dict(run="21-35-32", recorded_date="2026-10-20", why="probe root: whole episode, online subtasks"))
build_root.build([row], OUT, split="val")
verify_root.check(OUT)
(OUT / "README.md").write_text(
    f"# {OUT.name}\n\nWhole inference episode {SRC} ep0 (frames 0-{end}) for the representations and action-trace probes."
    " Subtask windows = the subtasks given to the policy online, NOT a reviewed annotation; episode_metadata quality is a"
    " placeholder 0. Do not train on it.\n"
)
print("DONE", OUT)
