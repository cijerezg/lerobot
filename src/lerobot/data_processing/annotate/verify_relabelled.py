#!/usr/bin/env python

"""
Verify the relabelled datasets: the originals are untouched, and the new metadata
loads through the real training path rather than merely existing on disk.

Checks per dataset:
 1. data/ + videos/ + depth/ are HARDLINKS to the source (same inode) — nothing copied,
    nothing to diverge, originals safe.
 2. state/action are bit-identical to the source parquet.
 3. `load_metadata_rows` (the function rl_offline.py calls) parses both parquets.
 4. `ReplayBuffer.materialize_metadata` broadcasts them into per-frame columns with the
    coverage and values the label files imply.
 5. Every frame of every episode is covered by exactly one quality segment.
 6. Mistake spans lie inside their segment and inside a quality<=2 segment.
 7. subtask_windows.json tiles each episode with no gap or overlap.

    python verify_relabelled.py --suffix annotated-v2
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUTPUTS = Path("outputs")
ORDER = ["rebot_socks_basket-v1", "rebot_shirts_bin-v1", "rebot_two_container-v1", "rebot_val-v1"]


def parquet_state(root: Path):
    df = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))]
    )
    return (np.stack(df["observation.state"].values).astype(np.float32),
            np.stack(df["action"].values).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="annotated-v2")
    args = ap.parse_args()
    sys.path.insert(0, str(Path("lerobot/src").resolve()))
    from lerobot.rl.buffer import ReplayBuffer
    from lerobot.rl.offline_dataset_utils import load_metadata_rows

    failures = []

    def check(cond, msg):
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
        if not cond:
            failures.append(msg)

    for ds in ORDER:
        src = OUTPUTS / ds
        dst = OUTPUTS / f"{ds.replace('-v1', '')}-{args.suffix}"
        print(f"\n{dst.name}")

        shared = total = 0
        for sub in ("data", "videos", "depth"):
            for root, _d, files in os.walk(dst / sub):
                rel = Path(root).relative_to(dst)
                for f in files:
                    total += 1
                    a, b = Path(root) / f, src / rel / f
                    shared += b.exists() and os.stat(a).st_ino == os.stat(b).st_ino
        check(shared == total, f"{shared}/{total} data+video+depth files share the source inode")

        s_new, a_new = parquet_state(dst)
        s_old, a_old = parquet_state(src)
        check(np.array_equal(s_new, s_old) and np.array_equal(a_new, a_old),
              f"state/action bit-identical to source ({s_new.shape})")

        ep_rows, mis_rows = load_metadata_rows(dst)
        check(len(ep_rows) > 0 and len(mis_rows) >= 0,
              f"load_metadata_rows -> {len(ep_rows)} segment rows, {len(mis_rows)} mistake rows")

        n = len(s_new)
        buf = ReplayBuffer(capacity=n, device="cpu", storage_device="cpu")
        buf.size = n
        # _initialize_storage only runs on the first add(); materialize_metadata is
        # called before any transition lands, so stand the containers up the same way.
        buf.complementary_info, buf.complementary_info_keys = {}, []
        buf.materialize_metadata(ep_rows, mis_rows)
        q = buf.complementary_info["metadata_quality"].float().numpy()
        m = buf.complementary_info["metadata_mistake"].float().numpy()
        check(not (q < 0).any(), f"every frame has a quality (no -1 left); mean {q.mean():.2f}")
        check(set(np.unique(q)).issubset({1.0, 2.0, 3.0, 4.0, 5.0}), f"quality values {sorted(set(np.unique(q)))}")
        share = {int(v): round(float((q == v).mean()), 3) for v in (1, 2, 3, 4, 5)}
        print(f"       quality frame share {share}   mistake frames {m.mean():.3%}")

        cover = np.zeros(n, dtype=int)
        for r in ep_rows:
            cover[int(r["from_index"]) : int(r["to_index"])] += 1
        check((cover == 1).all(), f"every frame covered by exactly one segment (min {cover.min()}, max {cover.max()})")

        seg_by_span = {(int(r["from_index"]), int(r["to_index"])): r for r in ep_rows}
        bad = []
        for r in mis_rows:
            host = [s for (a, b), s in seg_by_span.items() if a <= r["from_index"] and r["to_index"] <= b]
            if not host or int(host[0]["quality"]) > 2:
                bad.append((r["from_index"], r["to_index"]))
        check(not bad, f"all {len(mis_rows)} mistake spans sit inside a quality<=2 segment")

        wins = json.load(open(dst / "meta" / "subtask_windows.json"))["episodes"]
        eps = pd.concat([pd.read_parquet(f) for f in sorted(
            glob.glob(str(dst / "meta/episodes/**/*.parquet"), recursive=True))]).sort_values("episode_index")
        tiled = True
        for _, e in eps.iterrows():
            w = sorted(wins[str(int(e["episode_index"]))], key=lambda x: x["from_index"])
            tiled &= (w[0]["from_index"] == int(e["dataset_from_index"])
                      and w[-1]["to_index"] == int(e["dataset_to_index"])
                      and all(a["to_index"] == b["from_index"] for a, b in zip(w, w[1:])))
        check(tiled, "subtask_windows tile every episode with no gap or overlap")

        empty = [w["subtask"] for ws in wins.values() for w in ws if not str(w["subtask"]).strip()]
        placeholders = [w["subtask"] for ws in wins.values() for w in ws if "<object>" in str(w["subtask"])]
        check(not empty, "no empty subtask labels (the 'Subtask: ;' bug class)")
        print(f"       {len(placeholders)} <object> placeholders remaining")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECKS FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
