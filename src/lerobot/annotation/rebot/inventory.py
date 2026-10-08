"""Start or extend the inventory of a pass: one record per source episode in <work>/inventory.json.

    uv run python -m lerobot.annotation.rebot.inventory <work> <root> [--episodes E ...] [--kind rollout] [--name roll1005]

Appends to an existing inventory (idx continues). A record: idx, key (<name>_ep<E>), kind, source, episode, frames,
digest (sha256 of the episode's state and action), and ``seams``: the episode-local frames where the source root was
spliced (from its meta/provenance.json), so the video jumps there.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lerobot.annotation.paths import WORKSPACE
from lerobot.annotation.rebot.episode import episodes, kept, record, remap, states

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("root")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--kind", default="teleop")
    ap.add_argument("--name")
    args = ap.parse_args()
    work, root = WORKSPACE / args.work, (WORKSPACE / args.root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    path = work / "inventory.json"
    inventory = json.loads(path.read_text()) if path.exists() else []
    prov = root / "meta/provenance.json"
    prov = {p["episode_index"]: p for p in json.loads(prov.read_text())} if prov.exists() else {}
    for ep in args.episodes or episodes(root):
        rec = record(root, ep)
        s, a = states(rec)
        p = prov.get(int(ep), {})
        k = kept(p["keep"], p["cuts"]) if p.get("cuts") else []
        seams = [remap(k, c0) for c0, c1 in p.get("cuts", []) if p["keep"][0] < c0 and c1 < p["keep"][1]]
        inventory.append(
            dict(
                idx=len(inventory),
                key=f"{args.name or root.name}_ep{int(ep):02d}",
                kind=args.kind,
                source=str(root),
                episode=int(ep),
                frames=rec["frames"],
                digest=hashlib.sha256(np.ascontiguousarray(np.c_[s, a]).tobytes()).hexdigest(),
                seams=seams,
            )
        )
        print(inventory[-1]["idx"], inventory[-1]["key"], rec["frames"], "frames, seams", seams)
    path.write_text(json.dumps(inventory, indent=1))
