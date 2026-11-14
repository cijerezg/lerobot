#!/usr/bin/env python
"""Finalize a recording whose save_episode was interrupted during video encoding (Ctrl-C at the SVT-AV1 stage).

The interrupted root still holds every raw frame (images/<cam>/episode-000000/frame-*.png), the depth PNGs and the
data parquet; only the videos, the episodes table, stats.json and the info.json totals are missing. This script
never touches that root: it creates <root>-recovered, hardlinks the raw frames and depth into it, rebuilds the
episode buffer from the parquet, and lets DatasetWriter.save_episode do exactly what the recorder would have done
(episode stats over sampled frames, parquet, encode, move videos, episodes table, stats.json, info.json).

Only for a root whose data parquet exists and whose FIRST episode was interrupted. A root that still has
recovery/episode-*.pkl frame logs goes through `python -m lerobot.scripts.recover_episodes` instead.

Run from the workspace root:
    ROOT=<root> uv run python lerobot/scripts/recover_interrupted_encode.py
    DRY=1 ...   builds and validates the buffer, then deletes the -recovered copy without encoding
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "lerobot/src")
from lerobot.datasets.feature_utils import validate_episode_buffer  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

SRC = Path(os.environ["ROOT"]).expanduser()
DST = SRC.with_name(SRC.name + "-recovered"); DRY = bool(os.environ.get("DRY"))
DEFAULT = {"timestamp", "frame_index", "episode_index", "index", "task_index"}  # LeRobotDatasetMetadata.create re-adds these

info = json.load(open(SRC / "meta/info.json")); fps = info["fps"]
assert info["total_episodes"] == 0, "only an interrupted FIRST episode is handled (no episodes table to append to)"
assert not DST.exists(), f"refusing to replace {DST}"
features = {k: v for k, v in info["features"].items() if k not in DEFAULT}
video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
task_of = {int(i): t for t, i in pd.read_parquet(SRC / "meta/tasks.parquet")["task_index"].items()}
df = pd.read_parquet(SRC / "data/chunk-000/file-000.parquet").sort_values("frame_index").reset_index(drop=True); n = len(df)
assert np.array_equal(df.frame_index.to_numpy(), np.arange(n)) and df.episode_index.nunique() == 1 and int(df.episode_index[0]) == 0
for k in video_keys:
    pngs = sorted((SRC / f"images/{k}/episode-000000").glob("frame-*.png"))
    assert len(pngs) == n and int(pngs[-1].stem.split("-")[1]) == n - 1, (k, len(pngs), n)
depth_pngs = len(list((SRC / "depth").rglob("*.png")))


def link_tree(src, dst):
    for p in sorted(src.rglob("*")):
        q = dst / p.relative_to(src)
        if p.is_dir(): q.mkdir(parents=True, exist_ok=True)
        else: q.parent.mkdir(parents=True, exist_ok=True); os.link(p, q)


ds = LeRobotDataset.create(repo_id=f"cijerezg/{DST.name}", fps=fps, features=features, root=DST, robot_type=info["robot_type"])
for k in video_keys: link_tree(SRC / "images" / k, DST / "images" / k)   # the encode worker rmtree's these links after encoding
link_tree(SRC / "depth", DST / "depth")

buf = ds.writer.episode_buffer  # fresh buffer, episode_index 0, same keys the recorder's add_frame fills
buf["size"] = n
buf["task"] = [task_of[int(t)] for t in df.task_index]
buf["action"] = list(df["action"]); buf["observation.state"] = list(df["observation.state"])
buf["frame_index"] = df.frame_index.tolist(); buf["timestamp"] = [i / fps for i in range(n)]  # add_frame convention
for k in video_keys: buf[k] = [str(DST / f"images/{k}/episode-000000/frame-{i:06d}.png") for i in range(n)]
validate_episode_buffer(buf, ds.meta.total_episodes, ds.meta.features)
print(f"{SRC.name}: {n} frames, {len(video_keys)} cameras, {depth_pngs} depth PNGs, task {buf['task'][0]!r}")

if DRY:
    ds.writer._finalized = True; shutil.rmtree(DST); print("DRY: buffer valid, copy removed"); sys.exit(0)
ds.save_episode()
ds.finalize()
json.dump({"source": str(SRC), "script": "lerobot/scripts/recover_interrupted_encode.py"}, open(DST / "meta/recover.json", "w"), indent=2)
out = json.load(open(DST / "meta/info.json"))
print(f"{DST.name}: {out['total_episodes']} episode(s), {out['total_frames']} frames; images dir left: {(DST / 'images').exists()}")
