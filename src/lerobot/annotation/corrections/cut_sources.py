"""Add the footage that was cut out of the annotated roots to a correction pass.

  uv run python -m lerobot.annotation.corrections.cut_sources <work> <annotated root> [<annotated root> ...]

Follows ``meta/provenance.json`` (source, source_episode, keep, cuts) from every given root down to the recorded
policy runs (``outputs/train/<date>/<run>/inference_dataset``). A frame of a run that none of the given
roots holds is removed footage: review cuts, head and tail trims, idle cuts, and the episodes of the run no root
took. Adds to ``<work>/pass.json``
  sources   one source of kind ``cut`` per run, named ``cut_<date>_<time>``, with ``removed``: episode -> frame
            ranges
  contact   for a subtask of the removed footage the table lacks: the commonest contact and precision that subtask
            has in the pass's annotated sources
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.annotation.corrections.source import sources
from lerobot.annotation.paths import WORKSPACE


def bounds(root: Path) -> dict:
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episodes = pd.concat([pd.read_parquet(p, columns=["episode_index", "dataset_from_index", "dataset_to_index"]) for p in files])
    return {int(e.episode_index): (int(e.dataset_from_index), int(e.dataset_to_index)) for e in episodes.itertuples()}


def entries(root: Path) -> list:
    """The provenance entries of a root that say which source frames it holds; none for a recording."""
    file = root / "meta/provenance.json"
    found = json.loads(file.read_text()) if file.exists() else []
    return [p for p in found if isinstance(p, dict) and "source" in p and "keep" in p]


def held(root: Path, known: dict) -> dict:
    """episode of ``root`` -> (recording, its episode, the recording's frame of every frame of the episode)."""
    if root not in known:
        known[root] = {}
        for p in entries(root):
            source, episode = Path(p["source"]), p["source_episode"]
            if not source.exists() or (entries(source) and episode not in held(source, known)):
                continue  # an intermediate root that was deleted: its recordings cannot be traced
            kept = np.zeros(bounds(source)[episode][1] - bounds(source)[episode][0], dtype=bool)
            kept[p["keep"][0] : p["keep"][1]] = True
            for a, b in p.get("cuts") or []:
                kept[a:b] = False
            frames = np.flatnonzero(kept)
            if entries(source):
                source, episode, below = held(source, known)[episode]
                frames = below[frames]
            known[root][p["episode_index"]] = (source, episode, frames)
    return known[root]


def main() -> None:
    work, roots = Path(sys.argv[1]), [WORKSPACE / r for r in sys.argv[2:]]
    spec = json.loads((work / "pass.json").read_text())
    known, kept = {}, {}
    for root in roots:
        lengths = bounds(root)
        for index, (source, episode, frames) in held(root, known).items():
            assert len(frames) == lengths[index][1] - lengths[index][0], (root, index)
            if source.name == "inference_dataset":
                kept.setdefault(source, {}).setdefault(episode, []).append(frames)

    missing = Counter()
    for source in sorted(kept):
        labels = pd.read_parquet(source / "meta/online_labels.parquet").sort_values("index")
        removed = {}
        for episode, (a, b) in bounds(source).items():
            gone = np.ones(b - a, dtype=bool)
            gone[np.concatenate(kept[source].get(episode, [[]])).astype(int)] = False
            edges = np.flatnonzero(np.diff(np.concatenate([[False], gone, [False]]).astype(int)))
            if len(edges):
                removed[str(episode)] = edges.reshape(-1, 2).tolist()
            missing.update(s for s in labels.subtask.to_numpy()[a:b][gone] if s not in spec["contact"])
        if removed:
            date, run = source.parts[-3], source.parts[-2]
            spec["sources"].append(dict(
                name=f"cut_{date}_{run[:8]}", root=str(source.relative_to(WORKSPACE)), kind="cut", top="rebot_2026-10", removed=removed,
            ))
            print(spec["sources"][-1]["name"], run, sum(b - a for r in removed.values() for a, b in r), "frames removed")

    seen = {}
    for source in sources(spec).values():
        if source.kind in ("rebot", "rollout"):
            for s, c, p in zip(source.subtasks, source.contacts, source.precisions, strict=True):
                if s in missing:
                    seen.setdefault(s, Counter())[(c, p)] += 1
    for subtask in missing:
        contact, precision = seen[subtask].most_common(1)[0][0] if subtask in seen else ("unknown", "unknown")
        spec["contact"][subtask] = [contact, precision]
        print(f"contact: {subtask} -> {contact}, precision {precision} ({missing[subtask]} removed frames)")
    (work / "pass.json").write_text(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
