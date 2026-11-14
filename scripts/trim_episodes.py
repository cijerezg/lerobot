#!/usr/bin/env python
"""Hardlinked copy of a raw ReBot root with the first HEAD and last DROP frames removed from every kept episode; EPISODES (a prefix 0..k-1)
selects which episodes survive, the rest are left out entirely.

Videos are hardlinked untouched (trailing / dropped frames are simply never addressed; each episode's to_timestamp shrinks).
Data parquet is rewritten with a fresh global `index`; episodes table gets new length / from / to / low-dim stats;
depth PNGs outside the kept range are unlinked and the rest renamed by -HEAD, dropped episodes' depth dirs are not linked.
HEAD and DROP must be multiples of 3
so the global depth phase of later episodes is preserved.

The source root is only read. Run from the workspace root:
    ROOT=<root> HEAD=300 DROP=0 EPISODES=0 uv run python lerobot/scripts/trim_episodes.py   ->  <root>-ep0-head300
    (defaults: HEAD=0, DROP=150, all episodes -> <root>-trim150)
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "lerobot/src")
from lerobot.datasets.compute_stats import compute_episode_stats  # noqa: E402

SRC = Path(os.environ["ROOT"]).expanduser(); DROP = int(os.environ.get("DROP", 150)); HEAD = int(os.environ.get("HEAD", 0))
LOWDIM = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]
info = json.load(open(SRC / "meta/info.json")); fps = float(info["fps"]); features = info["features"]
KEEP = [int(x) for x in os.environ["EPISODES"].split(",")] if os.environ.get("EPISODES") else list(range(int(info["total_episodes"])))
assert KEEP == list(range(len(KEEP))), "kept episodes must be a prefix 0..k-1 (no renumbering)"
suffix = ("" if len(KEEP) == int(info["total_episodes"]) else "-ep" + "".join(map(str, KEEP))) + (f"-head{HEAD}" if HEAD else "") + (f"-trim{DROP}" if DROP else "")
DST = SRC.with_name(SRC.name + suffix)


def link_tree(src, dst):
    for p in sorted(src.rglob("*")):
        q = dst / p.relative_to(src)
        if p.is_dir(): q.mkdir(parents=True, exist_ok=True)
        else: q.parent.mkdir(parents=True, exist_ok=True); os.link(p, q)


assert not DST.exists(), f"refusing to replace {DST}"
assert DROP % 3 == 0 and HEAD % 3 == 0 and (HEAD or DROP)
eps = pd.read_parquet(SRC / "meta/episodes/chunk-000/file-000.parquet").sort_values("episode_index").reset_index(drop=True)
df = pd.read_parquet(SRC / "data/chunk-000/file-000.parquet").sort_values("index").reset_index(drop=True)
assert len(list((SRC / "meta/episodes").rglob("*.parquet"))) == 1 and len(list((SRC / "data").rglob("*.parquet"))) == 1

DST.mkdir(); link_tree(SRC / "videos", DST / "videos"); (DST / "meta/episodes/chunk-000").mkdir(parents=True)
for ep in KEEP: link_tree(SRC / "depth/wrist.depth" / f"episode-{ep:06d}", DST / "depth/wrist.depth" / f"episode-{ep:06d}")
eps = eps[eps.episode_index.isin(KEEP)].reset_index(drop=True)
new_len = {int(r.episode_index): int(r.length) - HEAD - DROP for r in eps.itertuples()}
keep = df.episode_index.isin(KEEP) & (df.frame_index >= HEAD) & (df.frame_index < HEAD + df.episode_index.map(new_len).fillna(0))
df = df[keep].reset_index(drop=True); df["index"] = np.arange(len(df), dtype=np.int64)
df["frame_index"] = df["frame_index"] - HEAD; df["timestamp"] = (df["frame_index"] / fps).astype(np.float32)  # recorder convention: frame_index / fps
(DST / "data/chunk-000").mkdir(parents=True); df.to_parquet(DST / "data/chunk-000/file-000.parquet", index=False)

start = 0
for i, r in eps.iterrows():
    ep = int(r.episode_index); n = new_len[ep]
    eps.loc[i, "length"] = n; eps.loc[i, "dataset_from_index"] = start; eps.loc[i, "dataset_to_index"] = start + n; start += n
    for cam in [k for k in features if features[k]["dtype"] == "video"]:
        eps.loc[i, f"videos/{cam}/from_timestamp"] = float(r[f"videos/{cam}/from_timestamp"]) + HEAD / fps
        eps.loc[i, f"videos/{cam}/to_timestamp"] = float(r[f"videos/{cam}/to_timestamp"]) - DROP / fps
    ep_df = df[df.episode_index == ep]
    stats = compute_episode_stats({k: (np.stack(ep_df[k]) if k in ("action", "observation.state") else ep_df[k].to_numpy()) for k in LOWDIM}, {k: features[k] for k in LOWDIM})
    for key, values in stats.items():
        for name, value in values.items():
            col = f"stats/{key}/{name}"; old = np.asarray(eps.at[i, col])
            vals = eps[col].tolist(); vals[i] = np.asarray(value, dtype=old.dtype).reshape(old.shape)
            eps[col] = pd.Series(vals, index=eps.index, dtype=object)
    removed = 0; ep_dir = DST / "depth/wrist.depth" / f"episode-{ep:06d}"
    for p in sorted(ep_dir.glob("frame-*.png"), key=lambda p: int(p.stem.split("-")[1])):  # ascending: every rename target is already vacated
        f = int(p.stem.split("-")[1])
        if f < HEAD or f >= HEAD + n: p.unlink(); removed += 1
        elif HEAD: p.rename(ep_dir / f"frame-{f - HEAD:06d}.png")
    print(f"episode {ep}: {int(r.length)} -> {n} frames (head {HEAD}, tail {DROP}), {removed} depth png(s) unlinked")
eps.to_parquet(DST / "meta/episodes/chunk-000/file-000.parquet", index=False)

total = int(start); assert total == len(df) and np.array_equal(df["index"].to_numpy(), np.arange(total))
info["total_frames"] = total; info["total_episodes"] = len(KEEP); info["splits"] = {"train": f"0:{len(KEEP)}"}
json.dump(info, open(DST / "meta/info.json", "w"), indent=4)
os.link(SRC / "meta/tasks.parquet", DST / "meta/tasks.parquet")
root_stats = json.load(open(SRC / "meta/stats.json"))
root_stats.update({k: {n: np.asarray(v).tolist() for n, v in vals.items()} for k, vals in compute_episode_stats(
    {k: (np.stack(df[k]) if k in ("action", "observation.state") else df[k].to_numpy()) for k in LOWDIM}, {k: features[k] for k in LOWDIM}).items()})
json.dump(root_stats, open(DST / "meta/stats.json", "w"), indent=4)
json.dump({"source": str(SRC), "kept_episodes": KEEP, "dropped_head_frames_per_episode": HEAD, "dropped_tail_frames_per_episode": DROP, "script": "lerobot/scripts/trim_episodes.py"}, open(DST / "meta/trim.json", "w"), indent=2)
print(f"{DST.name}: {len(KEEP)} episode(s), {total} frames total")
