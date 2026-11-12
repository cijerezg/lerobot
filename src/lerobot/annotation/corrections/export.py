"""Export the corrections of a pass as a new LeRobot v3.0 root the offline trainer reads as one more source.

  uv run python -m lerobot.annotation.corrections.export <work> <out_root> [--drop-inconsistent]

Every row of ``<work>/corrections.parquet`` with ok = True and no error becomes one episode of 30 frames:
  action               the 30 corrected actions (``correction.npy``)
  observation.state    the state recorded at the corrected frame, on all 30 rows
  top / wrist video    the image recorded at the corrected frame, 30 times
  depth/wrist.depth    a copy of the source depth PNG of the corrected frame (zeros when the source has none
                       there) at frames 0, 3, .. 27
The trainer cuts a chunk as ``actions[i : i + 30]`` and refuses a start whose chunk crosses an episode end, so
frame 0 is the only start an episode has: the real observation with the corrected chunk. Frames 1 .. 29 are
never a sample.

Tables in ``meta/`` (one row per episode; what ``load_metadata_rows`` and the subtask labels need):
  episode_metadata.parquet, subtasks.parquet, subtask_windows.json   the frame's subtask, quality ``QUALITY``
  mistakes.parquet                                                   empty
  speed_hybrid_v1_runs.parquet                                       speed ``SPEED``
  contact.parquet, precision.parquet                                 the labeller's contact; the source's
                                                                     precision (no row when it is unknown)
  corrections.parquet                                                the exported rows with their episode_index
Quality and speed are the constants the deployed prompt uses (``rtc_actor_runtime``). ``--drop-inconsistent``
leaves out the rows ``build`` flagged inconsistent. Refuses an existing ``<out_root>``. Prints the config lines.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lerobot.annotation.corrections.build import CHUNK
from lerobot.annotation.corrections.source import Episode, sources
from lerobot.annotation.vocab import code_for
from lerobot.configs.video import VideoEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.offline_dataset_utils import REBOT_SPEED_TABLE

QUALITY, SPEED = 5, 5
TOP, WRIST, DEPTH = "observation.images.top", "observation.images.wrist", "wrist.depth"
DEPTH_STRIDE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("out")
    ap.add_argument("--drop-inconsistent", action="store_true")
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    rows = pd.read_parquet(work / "corrections.parquet")
    rows = rows[rows.ok & (rows.error == "")]
    if args.drop_inconsistent:
        rows = rows[~rows.inconsistent]
    rows = rows.sort_values(["source", "episode", "frame"]).reset_index(drop=True)
    found = sources(json.loads((work / "pass.json").read_text()), list(rows.source.unique()))
    info = json.loads((next(iter(found.values())).root / "meta/info.json").read_text())
    image = dict(dtype="video", shape=(480, 640, 3), names=["height", "width", "channels"])
    features = {"observation.state": info["features"]["observation.state"], WRIST: image, TOP: image, "action": info["features"]["action"]}
    dataset = LeRobotDataset.create(
        "local/corrections",
        info["fps"],
        features,
        root=out,
        robot_type=info["robot_type"],
        image_writer_threads=4,
        camera_encoder=VideoEncoderConfig(preset=12),
    )
    precisions = []
    for (name, index), group in rows.groupby(["source", "episode"], sort=False):
        ep = Episode(found[name], index)
        frames = [int(f) for f in group.frame]
        tops, wrists = ep.images("top", frames), ep.images("wrist", frames)
        for new, frame, top, wrist in zip(group.index, frames, tops, wrists):
            state = ep.states[frame].astype(np.float32)
            for action in np.load(work / name / f"ep{index:03d}" / f"f{frame:06d}" / "correction.npy"):
                dataset.add_frame({"observation.state": state, WRIST: wrist, TOP: top, "action": action.astype(np.float32), "task": ep.task})
            dataset.save_episode()
            folder = out / "depth" / DEPTH / f"episode-{new:06d}"
            folder.mkdir(parents=True)
            source = ep.source.root / "depth" / ep.source.depth / f"episode-{index:06d}"
            stored = [p for p in (source / f"frame-{f:06d}.png" for f in (frame, frame - 1, frame + 1)) if p.exists()]
            if stored:
                shutil.copy(stored[0], folder / "frame-000000.png")
            else:
                cv2.imwrite(str(folder / "frame-000000.png"), np.zeros((480, 640), dtype=np.uint16))
            for f in range(DEPTH_STRIDE, CHUNK, DEPTH_STRIDE):
                os.link(folder / "frame-000000.png", folder / f"frame-{f:06d}.png")
            precisions.append(ep.precisions[frame])
            print(new, name, index, frame, flush=True)
    dataset.finalize()
    shutil.rmtree(out / "images")  # the writer's emptied frame folders

    meta, n = out / "meta", len(rows)
    span = dict(episode_index=np.arange(n), from_index=np.arange(n) * CHUNK, to_index=np.arange(n) * CHUNK + CHUNK)
    notes = [f"correction of {r.source} ep {r.episode} f {r.frame}" for r in rows.itertuples()]
    pd.DataFrame(dict(span, segment_index=0, subtask=rows.subtask, quality=QUALITY, vision_reviewed=False, anchor_baseline=False, note=notes)).to_parquet(
        meta / "episode_metadata.parquet", index=False
    )
    mistakes = dict(episode_index="int64", from_index="int64", to_index="int64", mistake="bool", mistake_type="object", note="object")
    pd.DataFrame(columns=list(mistakes)).astype(mistakes).to_parquet(meta / "mistakes.parquet", index=False)
    pd.DataFrame(dict(span, run_index=0, speed=SPEED, speed_score=float(SPEED))).to_parquet(meta / REBOT_SPEED_TABLE, index=False)
    contact = pd.DataFrame(dict(span, segment_index=0, contact=rows.contact.map(code_for), contact_slug=rows.contact, source="correction"))
    contact.to_parquet(meta / "contact.parquet", index=False)
    precision = pd.DataFrame(dict(span, segment_index=0, precision=precisions, source="correction"))
    precision = precision[precision.precision != "unknown"].astype({"precision": "int64"})
    precision.to_parquet(meta / "precision.parquet", index=False)
    subtasks = sorted(rows.subtask.unique())
    pd.DataFrame({"subtask_index": range(len(subtasks))}, index=pd.Index(subtasks, name="subtask")).to_parquet(meta / "subtasks.parquet")
    windows = {str(i): [dict(from_index=i * CHUNK, to_index=i * CHUNK + CHUNK, subtask=s)] for i, s in enumerate(rows.subtask)}
    (meta / "subtask_windows.json").write_text(json.dumps(dict(model="corrections", annotator=str(work), top_key=TOP, wrist_key=WRIST, episodes=windows), indent=1))
    rows.assign(episode_index=np.arange(n)).to_parquet(meta / "corrections.parquet", index=False)

    print(f"{n} episodes, {n * CHUNK} frames -> {out}")
    print("config_rl.yaml, under dataset.sources (and `corrections: <share>` under diverse.rebot_group_weights):")
    print(f"  - name: corrections\n    root: {out}\n    weight: 1.0\n    sampling_group: corrections\n    normalization_source: false\n    image_stride: {CHUNK}")
    print(f"    episode_groups:\n      corrections: {list(range(n))}")


if __name__ == "__main__":
    main()
