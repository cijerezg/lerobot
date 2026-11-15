# Data record

Generated 2026-11-14 22:00 by `uv run python lerobot/dataset_info/data_count.py` from `config_rl.yaml`. Do not edit by hand; rerun after every change to the data or the config.

## Training mix by sampling group

Share = fraction of all training samples. Usable = frames minus the usefulness cuts (`no manifest`; rule: `lerobot/src/lerobot/annotation/rubrics/usefulness_review_rubric.md`). ReBot hours at 30 fps.

| group | share | episodes | frames | usable frames | hours |
|---|---|---|---|---|---|
| tube | 0.7237 | 16 | 34,836 | 34,836 | 0.32 |
| tube2 | 0.2763 | 5 | 13,299 | 13,299 | 0.12 |
| **ReBot sampled** | **1** | **21** | **48,135** | **48,135** | **0.45** |

Diverse corpus: off.

## ReBot roots in the config

| source | root | episodes | frames | hours | sampled eps | groups (episodes / frames) |
|---|---|---|---|---|---|---|
| tube | `outputs/data/rebot_teleop_2026-11-14-tube-annotated-v1` | 16 | 34,836 | 0.32 | 16 | tube 16 / 34,836 |
| tube2 | `outputs/data/rebot_teleop_2026-11-14-tube2-annotated-v1` | 5 | 13,299 | 0.12 | 5 | tube2 5 / 13,299 |
| **total** | | **21** | **48,135** | **0.45** | | |


## Validation

None (`val_dataset_path: null`).

## Built, not in the config

From `lerobot/dataset_info/pending.yaml`.

None.

## Objects (sampled ReBot episodes)

From each root's `meta/episode_metadata.parquet`, groups with share > 0. Cycle = consecutive segments of one episode on the same object, split at each fresh grasp (grasp, move, release = 1; back-to-back regrasps = 1; each bit kit flap = 1; a retry after a failed close or insert = 1 more). Object = family matched in the subtask string (`FAMILIES` in data_count.py), else the noun phrase. Frames = segment frames, return to home not counted. Variants = the noun phrase of each cycle's first segment. Full subtask list: `SUBTASKS.md`.

| object | grasp cycles | internal | external | episodes | frames | hours | variants (cycles) |
|---|---|---|---|---|---|---|---|
| tube | 49 | 49 | 0 | 21 | 46,103 | 0.43 |  |

## Subtasks by verb (sampled ReBot episodes)

| verb | segments | frames | hours |
|---|---|---|---|
| grasp | 48 | 19,103 | 0.18 |
| approach | 41 | 10,792 | 0.10 |
| move | 40 | 5,684 | 0.05 |
| insert | 40 | 10,524 | 0.10 |
| return home | 10 | 2,032 | 0.02 |
