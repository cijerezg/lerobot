"""Build one new root per labelled root of a pass under annotation_rubric.md (table.json, frames.csv, windows.json).

    uv run python -m lerobot.annotation.rebot.window_labels_root outputs/annotation/passes/<pass> --suffix=-annotated-v1

Every labelled episode is taken whole (videos hardlinked) and must be covered by the windows from its first to its last
frame. The new root is ``<root><suffix>`` next to the labelled one. Writes the subtask tables of ``build_root``
(``quality`` -1: the pass grades segments high / low per class, not on the 0-4 scale) plus

- ``meta/mistakes.parquet``: the runs of frames whose ``mistake`` column in frames.csv is set (a mistake within the
  next second, the rule of the rubric);
- ``meta/speed_level.parquet``: episode_index, from_index, to_index (dataset frames), level (slow / normal / fast, the
  terciles of the segment's class). A segment without a level (a class too small for terciles) is ``normal``.

Two segments on either side of a window seam with the same subtask text become one subtask segment; their speed
levels stay separate.
"""

import argparse
import json

import pandas as pd

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.build_root import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work")
    parser.add_argument("--suffix", required=True)
    args = parser.parse_args()
    work = WORKSPACE / args.work
    table = pd.DataFrame(json.loads((work / "table.json").read_text()))
    frames = pd.read_csv(work / "frames.csv", keep_default_na=False)
    windows = json.loads((work / "windows.json").read_text())
    frames["root"] = frames.window.map(lambda w: windows[w]["root"])
    frames["episode"] = frames.window.map(lambda w: windows[w]["episode"])
    for root, root_table in table.groupby("root"):
        episodes = pd.read_parquet(WORKSPACE / root / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        rows, speed = [], []
        for episode, t in root_table.groupby("episode"):
            t = t.sort_values("start")
            n = int(episodes.loc[episode, "length"])
            assert t.start.iloc[0] == 0 and t.end.iloc[-1] == n and (t.start.values[1:] == t.end.values[:-1]).all(), (root, episode)
            segments = []
            for s in t.itertuples():
                if segments and s.window != window and segments[-1]["subtask"] == s.subtask:
                    segments[-1]["to_index"] = s.end
                else:
                    segments.append(dict(from_index=s.start, to_index=s.end, subtask=s.subtask, what_happens=s.evidence))
                window = s.window
                speed.append(dict(episode=episode, from_index=s.start, to_index=s.end, level=s.level or "normal"))
            f = frames[(frames.root == root) & (frames.episode == episode)].sort_values("frame")
            assert (f.frame.values == range(n)).all(), (root, episode)
            run = (f.mistake != f.mistake.shift()).cumsum()
            mistakes = [
                dict(from_index=int(g.frame.iloc[0]), to_index=int(g.frame.iloc[-1]) + 1, type=g.mistake.iloc[0], what_happens="")
                for _, g in f.groupby(run)
                if g.mistake.iloc[0]
            ]
            rows.append(
                dict(
                    source=root,
                    episode=episode,
                    key=f"{root.split('/')[-1]}_ep{episode:02d}",
                    frames=n,
                    keep=[0, n],
                    cuts=[],
                    task=episodes.loc[episode, "tasks"][0],
                    segments=segments,
                    mistakes=mistakes,
                    provenance=dict(kind="teleop", labels=str(work)),
                )
            )
        out = WORKSPACE / (root + args.suffix)
        build(rows, out, annotator=f"{work}/table.json")
        path = out / "meta/episode_metadata.parquet"
        pd.read_parquet(path).assign(quality=-1).to_parquet(path, index=False)
        # rows are in episode order, so the new episode index is the position in ``rows``
        new = pd.read_parquet(out / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        index = {r["episode"]: i for i, r in enumerate(rows)}
        speed = pd.DataFrame(speed)
        offset = speed.episode.map(lambda e: int(new.loc[index[e], "dataset_from_index"]))
        pd.DataFrame(
            dict(
                episode_index=speed.episode.map(index),
                from_index=speed.from_index + offset,
                to_index=speed.to_index + offset,
                level=speed.level,
            )
        ).to_parquet(out / "meta/speed_level.parquet", index=False)
        print("DONE", out, len(rows), "episodes")
