#!/usr/bin/env python3
"""Count training episodes per exact critic text, without decoding images.

From the workspace root:
  PYTHONPATH=lerobot/src .venv/bin/python lerobot/scripts/probes/critic_text_counts.py --config config_rl.yaml

Diverse counts use the real retained-anchor selection and critic_skip rule. Release
folding changes terminals, not prompt text. The JSON is deliberately just {text: count};
a companion .meta.json records the inputs and selection totals.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml


def build_counts(config_path):
    from lerobot.datasets.diverse_actor_selection import open_federated_corpus, select_actor_anchors
    from lerobot.rl.data_sources.diverse_actor_buffer import DiverseSampleSpec, critic_view

    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if float(cfg.get("val_split", 0.0)) != 0:
        raise ValueError("Use explicit training sources; a runtime val_split cannot be counted from this YAML.")
    base = config_path.parent
    def resolve(path):
        return (base / path).resolve()
    seen = defaultdict(set)
    provenance = {"config": str(config_path), "sources": [], "rule": "unique training episodes with critic-usable exact text"}
    sources = cfg["dataset"].get("sources") or [{"root": cfg["dataset"]["root"]}]
    val_root = resolve(cfg["val_dataset_path"]) if cfg.get("val_dataset_path") else None
    for source in sources:
        root = resolve(source["root"])
        if root == val_root:
            raise ValueError(f"Validation root is also a training source: {root}")
        if float(source.get("weight", 1)) <= 0:
            continue
        vocab = pd.read_parquet(root / "meta/subtasks.parquet")
        names = (vocab["subtask"] if "subtask" in vocab.columns else vocab.index).astype(str).tolist()
        # Match the loader's normalized lookup, but count the rendered vocabulary text.
        key = lambda s: " ".join(s.strip().split()).casefold().rstrip(".")
        canonical = {key(text): text for text in names}
        if len(canonical) != len(names):
            raise ValueError(f"{root}: duplicate normalized subtask names")
        windows = json.loads((root / "meta/subtask_windows.json").read_text())["episodes"]
        selected = source.get("episodes")
        selected = None if selected is None else set(map(int, selected))
        episodes = set()
        for episode, rows in windows.items():
            if selected is not None and int(episode) not in selected:
                continue
            for row in rows:
                if int(row["to_index"]) <= int(row["from_index"]):
                    continue
                text = canonical[key(row["subtask"])]
                seen[text].add((str(root), str(episode)))
                episodes.add(str(episode))
        provenance["sources"].append({"root": str(root), "episodes": len(episodes)})
    if cfg.get("diverse", {}).get("enabled"):
        root = resolve(cfg["diverse"]["root"])
        selection = select_actor_anchors(open_federated_corpus(root))
        spec = DiverseSampleSpec(action_horizon=int(cfg["policy"]["chunk_size"]))
        chunk_seconds = spec.action_horizon / spec.action_rate_hz
        view = critic_view(selection.rows, chunk_seconds)
        for i, row in enumerate(selection.rows):
            if not view.skip[i]:
                seen[str(row["subtask"])].add((str(root), str(row["episode_id"])))
        provenance["diverse"] = {
            "root": str(root), "training_episodes": len(selection.episode_ids),
            "usable_anchors": int((~view.skip).sum()), "skipped_anchors": int(view.skip.sum()),
            "excluded_holdout_episodes": sorted(selection.held_out),
        }
    counts = {text: len(episodes) for text, episodes in sorted(seen.items())}
    provenance["n_texts"] = len(counts)
    return counts, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_rl.yaml")
    parser.add_argument("--output", default="lerobot/dataset_info/manifests/critic_text_counts.json")
    args = parser.parse_args()
    counts, provenance = build_counts(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(counts, indent=2) + "\n")
    output.with_suffix(".meta.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{len(counts)} exact texts -> {output}")


if __name__ == "__main__":
    main()
