"""Embodiment stats over the SAMPLED ReBot selection (2026-10-27).

Same as lerobot.scripts.compute_diverse_stats --hand, except the ReBot row mirrors what training draws:
  * episodes in a config_rl.yaml episode group with weight 0 ("excluded") are dropped whole;
  * chunk starts whose 30-step action window overlaps a usefulness exclusion range
    (diverse.rebot_exclusions_manifest, FrameExclusions.filter_starts) are dropped;
  * states keep every frame of a sampled episode that is not inside an exclusion range.
Extra roots (unannotated, no groups, no exclusions) are passed with --extra-root.

A checkpoint's saved processors override `embodiment_stats_path` (docs/design/training.md, "Normalization
stats source"), so a new stats file only reaches a run that starts without them.

Run from the workspace root:
    uv run python lerobot/scripts/compute_filtered_stats.py [--config config_rl.yaml] \\
        [--extra-root <root> ...] --out outputs/stats/<name>.pt
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from lerobot.rl.data_sources.frame_exclusions import FrameExclusions
from lerobot.scripts.compute_diverse_stats import build_artifact, collect_diverse, collect_hand, report
from lerobot.scripts.compute_embodiment_stats import _encode, _read_columns, _read_info
from lerobot.datasets.diverse_actor_selection import action_layout_by_name

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("filtered")

p = argparse.ArgumentParser()
p.add_argument("--config", default="config_rl.yaml")
p.add_argument("--extra-root", action="append", default=[], type=Path)
p.add_argument("--chunk-size", type=int, default=30)
p.add_argument("--out", required=True)
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
group_w = cfg["diverse"]["rebot_group_weights"]
zero_groups = {g for g, w in group_w.items() if w == 0}
exclusions = FrameExclusions(cfg["diverse"]["rebot_exclusions_manifest"])
diverse_root = Path(cfg["diverse"]["root"])
stats_diverse_root = Path(str(diverse_root).removesuffix("_tasks2"))  # the store the 10-10 file used

plan = []  # (root, set of excluded episodes)
for src in cfg["dataset"]["sources"]:
    groups = src.get("episode_groups") or {}
    dropped = {ep for g, eps in groups.items() if g in zero_groups for ep in eps}
    plan.append((Path(src["root"]), dropped))
for root in args.extra_root:
    plan.append((root, set()))

T = args.chunk_size
chunk_groups, raw_groups, state_groups = [], [], []
summary = []
for root, dropped in plan:
    info = _read_info(root)
    episodes = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))], ignore_index=True)
    actions, states = _read_columns(root, info)
    n_chunks = n_states = n_frames_sampled = 0
    for _, ep in episodes.iterrows():
        idx = int(ep["episode_index"])
        if idx in dropped:
            continue
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        n_frames_sampled += hi - lo
        rows = torch.arange(lo, max(lo, hi - T + 1))
        rows = exclusions.filter_starts(rows, root=root, episode=idx, episode_start=lo, episode_end=hi, chunk_size=T)
        rows = rows.numpy()
        if len(rows):
            targets = np.stack([actions[r : r + T] for r in rows])
            anchors = states[rows]
            chunk_groups.append(_encode(targets, anchors, "anchor"))
            raw_groups.append(targets)
            n_chunks += len(rows)
        keep = np.ones(hi - lo, dtype=bool)
        for a, b in exclusions.ranges.get((str(root.resolve()), idx), []):
            keep[a:b] = False
        state_groups.append(states[lo:hi][keep])
        n_states += int(keep.sum())
    summary.append((root.name, len(episodes), len(episodes) - len(dropped), n_frames_sampled, n_chunks, n_states))
    log.info("  %-55s eps %3d sampled %3d frames %7d chunks %7d states %7d", *summary[-1])

action = np.concatenate(chunk_groups); raw = np.concatenate(raw_groups); state = np.concatenate(state_groups)
log.info("ReBot total: chunks %d states %d (sampled frames %d)", len(action), len(state), sum(s[3] for s in summary))
layout = action_layout_by_name("rebot_b601_joint7_commanded")
rebot = {layout.index: {"action": action, "state": state, "action_dim": layout.dim, "state_dim": layout.dim,
                        "sources": [str(r) for r, _ in plan], "raw_action": raw, "anchor": raw[:, 0] - action[:, 0]}}

collected = collect_diverse(stats_diverse_root, T, "anchor")
collected.update(rebot)
artifact = build_artifact(collected, T, "anchor", collect_hand(collected))
artifact["rebot_selection"] = {
    "config": args.config,
    "exclusions_manifest": cfg["diverse"]["rebot_exclusions_manifest"],
    "dropped_groups": sorted(zero_groups),
    "per_root": [dict(zip(("root", "episodes", "sampled_episodes", "sampled_frames", "chunks", "states"), s)) for s in summary],
}
report(artifact)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(artifact, args.out)
log.info("Wrote %s", args.out)
