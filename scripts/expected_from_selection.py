"""Print the EXPECTED_* constants of diverse_actor_selection.py as measured on a corpus root.

    uv run python lerobot/scripts/expected_from_selection.py --root outputs/diverse_robot_dataset_v3

The selection is built exactly like training (retained anchors of every episode minus
<root>/holdout_episodes.json) with the startup count assert skipped, so it runs on a root
whose counts the constants do not yet describe. Paste the printed block over the four
EXPECTED_* assignments in lerobot/src/lerobot/datasets/diverse_actor_selection.py after an
ingest; the held-out lines are for the comment above them.
"""
from __future__ import annotations

import argparse
from collections import Counter

from lerobot.datasets.diverse_actor_selection import (
    EXPECTED_ANCHORS,
    EXPECTED_ANCHORS_BY_SOURCE,
    EXPECTED_EPISODES,
    EXPECTED_EPISODES_BY_SOURCE,
    open_federated_corpus,
    select_actor_anchors,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/diverse_robot_dataset_v3")
    args = ap.parse_args()
    selection = select_actor_anchors(open_federated_corpus(args.root), verify_counts=False)
    episodes_by_source = Counter()
    for episode_id, rows in selection.rows_by_episode().items():
        episodes_by_source[rows[0]["source"]] += 1
    anchors_by_source = Counter(row["source"] for row in selection.rows)
    order = [*EXPECTED_ANCHORS_BY_SOURCE, *sorted(set(anchors_by_source) - set(EXPECTED_ANCHORS_BY_SOURCE))]

    print(f"EXPECTED_EPISODES = {len(selection.episode_ids)}")
    print(f"EXPECTED_ANCHORS = {len(selection.rows):_}")
    print("EXPECTED_EPISODES_BY_SOURCE = {")
    for source in order:
        print(f'    "{source}": {episodes_by_source[source]},')
    print("}")
    print("EXPECTED_ANCHORS_BY_SOURCE = {")
    for source in order:
        print(f'    "{source}": {anchors_by_source[source]:_},')
    print("}")
    print()
    print(f"# held out: {len(selection.held_out)} episodes / {sum(selection.held_out.values())} anchors")
    for episode_id, count in sorted(selection.held_out.items()):
        print(f"#   {episode_id}: {count}")
    same = (
        len(selection.episode_ids) == EXPECTED_EPISODES
        and len(selection.rows) == EXPECTED_ANCHORS
        and dict(episodes_by_source) == EXPECTED_EPISODES_BY_SOURCE
        and dict(anchors_by_source) == EXPECTED_ANCHORS_BY_SOURCE
    )
    print(f"# matches the constants currently in diverse_actor_selection.py: {same}")


if __name__ == "__main__":
    main()
