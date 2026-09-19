#!/usr/bin/env python

"""Union two FMB stores into one v2 store without copying arrays.

The v1 single-object store and the v2 multi-object store are each written by
`prepare_fmb.py convert`, which regenerates episodes.jsonl and the views from the manifest it
was given. A union store hardlinks both episode collections, concatenates episodes.jsonl, keeps
both corpus.json / source_manifest.json under suffixed names, and rebuilds the views.

    fmb_merge_stores.py --into outputs/diverse_robot_dataset_v2/fmb \
        --store outputs/diverse_robot_dataset/fmb --store outputs/diverse_robot_dataset_v2_build/fmb_multi/store
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_fmb import read_jsonl, write_actor_views  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--into", type=Path, required=True)
    parser.add_argument("--store", type=Path, action="append", required=True)
    args = parser.parse_args()
    episodes_root = args.into / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    rows = []
    critic_rows = []
    seen = set()
    for store in args.store:
        tag = store.name if store.name != "fmb" else store.parent.name
        critic_rows.extend(read_jsonl(store / "critic_intervals.jsonl"))
        for row in read_jsonl(store / "episodes.jsonl"):
            if row["episode_id"] in seen:
                raise ValueError(f"duplicate episode id {row['episode_id']} across stores")
            seen.add(row["episode_id"])
            rows.append(row)
            target = episodes_root / row["episode_id"]
            if not target.exists():
                shutil.copytree(store / "episodes" / row["episode_id"], target, copy_function=os.link)
        for name in ("corpus.json", "source_manifest.json"):
            if (store / name).is_file():
                shutil.copy2(store / name, args.into / f"{Path(name).stem}_{tag}.json")
    import json

    with open(args.into / "episodes.jsonl", "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    with open(args.into / "critic_intervals.jsonl", "w", encoding="utf-8") as stream:
        for row in critic_rows:
            stream.write(json.dumps(row) + "\n")
    print(f"{len(rows)} episodes in {args.into}")
    print(write_actor_views(args.into))


if __name__ == "__main__":
    main()
