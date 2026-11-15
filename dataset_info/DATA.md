# Data record

Generated 2026-11-15 09:13 by `uv run python lerobot/dataset_info/data_count.py` from `config_rl.yaml`. Do not edit by hand; rerun after every change to the data or the config.

## Training mix by sampling group

Share = fraction of all training samples. Usable = frames minus the usefulness cuts (`no manifest`; rule: `lerobot/src/lerobot/annotation/rubrics/usefulness_review_rubric.md`). ReBot hours at 30 fps.

| group | share | episodes | frames | usable frames | hours |
|---|---|---|---|---|---|
| tube | 0.603 | 15 | 34,266 | 34,266 | 0.32 |
| tube2 | 0.234 | 5 | 13,299 | 13,299 | 0.12 |
| tube3 | 0.1629 | 4 | 9,258 | 9,258 | 0.09 |
| **ReBot sampled** | **1** | **24** | **56,823** | **56,823** | **0.53** |

Diverse corpus: off.

## ReBot roots in the config

| source | root | episodes | frames | hours | sampled eps | groups (episodes / frames) |
|---|---|---|---|---|---|---|
| tube | `outputs/data/rebot_teleop_2026-11-14-tube-annotated-v2` | 15 | 34,266 | 0.32 | 15 | tube 15 / 34,266 |
| tube2 | `outputs/data/rebot_teleop_2026-11-14-tube2-annotated-v1` | 5 | 13,299 | 0.12 | 5 | tube2 5 / 13,299 |
| tube3 | `outputs/data/rebot_teleop_2026-11-15-tube3-annotated-v1` | 4 | 9,258 | 0.09 | 4 | tube3 4 / 9,258 |
| **total** | | **24** | **56,823** | **0.53** | | |


## Validation

`outputs/data/rebot_val_2026-11-15-tube-merged-v1`: 2 episodes, 2,448 frames, 0.02 h.

## Built, not in the config

From `lerobot/dataset_info/pending.yaml`.

None.

## Objects (sampled ReBot episodes)

From each root's `meta/episode_metadata.parquet`, groups with share > 0. Cycle = consecutive segments of one episode on the same object, split at each fresh grasp (grasp, move, release = 1; back-to-back regrasps = 1; each bit kit flap = 1; a retry after a failed close or insert = 1 more). Object = family matched in the subtask string (`FAMILIES` in data_count.py), else the noun phrase. Frames = segment frames, return to home not counted. Variants = the noun phrase of each cycle's first segment. Full subtask list: `SUBTASKS.md`.

| object | grasp cycles | internal | external | episodes | frames | hours | variants (cycles) |
|---|---|---|---|---|---|---|---|
| tube | 60 | 60 | 0 | 24 | 54,634 | 0.51 |  |

## Subtasks by verb (sampled ReBot episodes)

| verb | segments | frames | hours |
|---|---|---|---|
| grasp | 56 | 20,417 | 0.19 |
| approach | 52 | 14,206 | 0.13 |
| insert | 49 | 12,953 | 0.12 |
| move | 48 | 7,058 | 0.07 |
| return home | 11 | 2,189 | 0.02 |
