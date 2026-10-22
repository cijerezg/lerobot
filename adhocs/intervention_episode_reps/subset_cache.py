"""Crop and thin a representations-probe cache to one time window, so the stock interactive page
(trajectories.html, plot mode) covers just that window at a readable stride. Nothing is recomputed: rows,
arrays and thumbs are copied with new row/sample indices; "episode-centered" on the page then means centered
on the window's frames.

    uv run python lerobot/adhocs/intervention_episode_reps/subset_cache.py <src_representations_dir> <out_dir> <t0> <t1> <every> [subtask substring]
    uv run python lerobot/adhocs/intervention_episode_reps/reps_episode.py --config config_rl.yaml \
        --probe_parameters.mode plot --probe_parameters.output_dir <out_dir> ...same repr_* flags as the collect
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

src, out, t0, t1, every = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
subtask_match = sys.argv[6] if len(sys.argv) > 6 else None  # optional: keep only rows whose subtask contains this
dst = out / "validation" / src.parent.name / "representations" / "cache"
(dst / "thumbs").mkdir(parents=True, exist_ok=True)
meta = json.loads((src / "cache" / "meta.json").read_text())
keep = [r for r in meta["rows"] if t0 - 1e-6 <= r["time_s"] <= t1 + 1e-6
        and (subtask_match is None or subtask_match in str(r.get("subtask", "")))]
samples = sorted({r["sample"] for r in keep})[::every]
keep = [r for r in keep if r["sample"] in samples]
old_rows = [r["row"] for r in keep]
new_sample = {s: i for i, s in enumerate(samples)}
for i, r in enumerate(keep):
    for cam in ("external_0", "wrist_0"):
        p = src / "cache" / "thumbs" / f"{r['sample']:04d}.{cam}.jpg"
        if p.exists():
            shutil.copy(p, dst / "thumbs" / f"{new_sample[r['sample']]:04d}.{cam}.jpg")
    r["row"], r["sample"] = i, new_sample[r["sample"]]
for g in meta["groups"]:
    np.save(dst / f"{g}.npy", np.load(src / "cache" / f"{g}.npy", mmap_mode="r")[old_rows])
    np.save(dst / f"{g}.present.npy", np.load(src / "cache" / f"{g}.present.npy")[old_rows])
meta["rows"] = keep
meta["stride_s"] = meta["stride_s"] * every
(dst / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"{len(keep)} frames, {keep[0]['time_s']:.1f}-{keep[-1]['time_s']:.1f} s every {meta['stride_s']} s -> {dst}")
