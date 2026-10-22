"""Concatenate several (already cropped) representation caches into one, so the stock page shows the episodes
side by side with cross-episode matches and the aligned grid. Row and sample indices are renumbered; episode
labels are kept, so each source stays its own episode on the page. With --concat the later caches are appended to
the first as ONE episode, simply stacked: row i is at time i x stride; each row's original clock is kept in "source", so the page's within-episode
cosine matrix shows every source and the blocks between them.

    uv run python lerobot/adhocs/intervention_episode_reps/merge_cache.py [--concat] <out_dir> <representations_dir> [<representations_dir> ...]
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

args = [a for a in sys.argv[1:] if a != "--concat"]
concat = "--concat" in sys.argv
out = Path(args[0])
srcs = [Path(p) for p in args[1:]]
dst = out / "validation" / srcs[0].parent.name / "representations" / "cache"
(dst / "thumbs").mkdir(parents=True, exist_ok=True)
metas = [json.loads((s / "cache" / "meta.json").read_text()) for s in srcs]
groups = [g for g in metas[0]["groups"] if all(g in m["groups"] for m in metas)]
rows, arrays, present = [], {g: [] for g in groups}, {g: [] for g in groups}
sample_offset = 0
first = metas[0]["rows"][0]
for src, m in zip(srcs, metas):
    for r in m["rows"]:
        new_sample = r["sample"] + sample_offset
        if concat:
            r = {**r, "episode": first["episode"], "episode_label": first["episode_label"], "task": first["task"],
                 "holdout": first["holdout"], "source": f"{r['source']} t={r['time_s']:g}",
                 "time_s": len(rows) * metas[0]["stride_s"]}  # stacked: row i sits at i x stride
        for cam in ("external_0", "wrist_0"):
            p = src / "cache" / "thumbs" / f"{r['sample']:04d}.{cam}.jpg"
            if p.exists():
                shutil.copy(p, dst / "thumbs" / f"{new_sample:04d}.{cam}.jpg")
        rows.append({**r, "row": len(rows), "sample": new_sample})
    sample_offset += max(r["sample"] for r in m["rows"]) + 1
    for g in groups:
        arrays[g].append(np.load(src / "cache" / f"{g}.npy", mmap_mode="r"))
        present[g].append(np.load(src / "cache" / f"{g}.present.npy"))
for g in groups:
    np.save(dst / f"{g}.npy", np.concatenate(arrays[g]))
    np.save(dst / f"{g}.present.npy", np.concatenate(present[g]))
meta = {**metas[0], "rows": rows, "groups": groups, "requested_groups": groups}
(dst / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"{len(rows)} rows from {len(srcs)} caches, episodes {sorted({r['episode_label'] for r in rows})} -> {dst}")
