"""Build a new root that carries the hand subtask labels of ``subtask_labeler`` (its <work>/labels/episode_<E>.json).

    uv run python -m lerobot.annotation.rebot.subtask_labels_root outputs/annotation/passes/subtasks_<root> --out outputs/data/<new root>

``--episodes 14`` / ``--skip-episodes 14`` build a root from some of the labelled episodes (a validation root and the
training root without it); ``--root`` names the labelled root when it has moved since the labels were written.

Every episode of the labelled root is taken whole (videos and depth hardlinked) and must be labelled from its first to
its last frame. Writes the subtask tables of ``build_root`` (subtasks, episode_metadata, subtask_windows, an empty
mistakes table); ``quality`` is -1 (unknown: nothing was graded). No precision, contact, speed or gripper-event tables.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.build_root import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--out", required=True)
    parser.add_argument("--root")
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--skip-episodes", type=int, nargs="+", default=[])
    args = parser.parse_args()
    work = WORKSPACE / args.work
    rows = []
    for path in sorted((work / "labels").glob("episode_*.json")):
        d = json.loads(path.read_text())
        if d["episode"] in args.skip_episodes or (args.episodes is not None and d["episode"] not in args.episodes):
            continue
        root = WORKSPACE / args.root if args.root is not None else Path(d["root"])
        episodes = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        segments = [dict(from_index=s["start"], to_index=s["end"], subtask=s["subtask"], what_happens="") for s in d["segments"]]
        assert segments[0]["from_index"] == 0 and segments[-1]["to_index"] == d["frames"], path
        assert all(a["to_index"] == b["from_index"] for a, b in zip(segments, segments[1:], strict=False)), path
        rows.append(
            dict(
                source=str(root),
                episode=d["episode"],
                key=f"{root.name}_ep{d['episode']:02d}",
                frames=d["frames"],
                keep=[0, d["frames"]],
                cuts=[],
                task=episodes.loc[d["episode"], "tasks"][0],
                segments=segments,
                mistakes=[],
                provenance=dict(kind="teleop", labels=str(path)),
            )
        )
    out = WORKSPACE / args.out
    build(rows, out, annotator=f"{work}/labels/*.json")
    table = out / "meta/episode_metadata.parquet"
    pd.read_parquet(table).assign(quality=-1).to_parquet(table, index=False)
    print("DONE", out, len(rows), "episodes")
