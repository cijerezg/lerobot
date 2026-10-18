# Data record

Generated 2026-10-17 22:15 by `uv run python lerobot/dataset_info/data_count.py` from `config_rl.yaml`. Do not edit by hand; rerun after every change to the data or the config.

## Training mix by sampling group

Share = fraction of all training samples. Usable = frames minus the usefulness cuts (`migration/internal_usefulness_review_2026-10-16/cleanup/exclusions.json`). ReBot hours at 30 fps.

| group | share | episodes | frames | usable frames | hours |
|---|---|---|---|---|---|
| external_rebot | 0.25 | 231 | 171,876 | 171,876 | 1.59 |
| bits_clippers | 0.154 | 31 | 183,798 | 168,721 | 1.70 |
| rigid | 0.1034 | 39 | 64,824 | 64,768 | 0.60 |
| cases_kit_box_push | 0.0671 | 17 | 35,526 | 30,115 | 0.33 |
| bottles_lids | 0.065 | 17 | 59,889 | 53,109 | 0.55 |
| pills_caps | 0.0557 | 12 | 33,459 | 32,456 | 0.31 |
| clothing | 0.0548 | 79 | 280,194 | 259,848 | 2.59 |
| excluded | 0 | 13 | 20,634 | 0 | 0.19 |
| **ReBot sampled** | **0.75** | **426** | **829,566** | **780,893** | **7.68** |
| diverse corpus | 0.25 | 879 | 487,851 | | 6.39 |

## ReBot roots in the config

| source | root | episodes | frames | hours | sampled eps | groups (episodes / frames) |
|---|---|---|---|---|---|---|
| internal_main | `outputs/rebot_all-annotated-v14-tasks2` | 90 | 308,802 | 2.86 | 90 | clothing 77 / 271,947, rigid 13 / 36,855 |
| internal_additions | `outputs/rebot_main_additions-annotated-v3` | 22 | 25,416 | 0.24 | 22 | clothing 1 / 5,337, rigid 21 / 20,079 |
| internal_bits_kit | `outputs/rebot_bits_kitfold_2026-10-06-annotated-v2-tasks1` | 18 | 96,447 | 0.89 | 18 | bits_clippers 13 / 80,739, cases_kit_box_push 5 / 15,708 |
| internal_oct04 | `outputs/rebot_additions_2026-10-04-annotated-v12-tasks1` | 26 | 53,079 | 0.49 | 26 | rigid 4 / 5,046, pills_caps 12 / 33,459, cases_kit_box_push 10 / 14,574 |
| internal_oct07 | `outputs/rebot_teleop_2026-10-07-annotated-v1-tasks1` | 5 | 20,124 | 0.19 | 3 | bits_clippers 3 / 10,155, excluded 2 / 9,969 |
| internal_oct08 | `outputs/rebot_teleop_2026-10-08-annotated-v1-tasks1` | 13 | 39,009 | 0.36 | 12 | bottles_lids 12 / 33,990, excluded 1 / 5,019 |
| internal_oldval | `outputs/rebot_oldval_train-2026-10-14-v1` | 2 | 5,754 | 0.05 | 2 | clothing 1 / 2,910, rigid 1 / 2,844 |
| internal_oct14 | `outputs/rebot_2026-10-14-annotated-v4` | 13 | 54,843 | 0.51 | 13 | bits_clippers 6 / 23,700, bottles_lids 5 / 25,899, cases_kit_box_push 2 / 5,244 |
| internal_oct16 | `outputs/rebot_2026-10-16-annotated-v1` | 7 | 53,808 | 0.50 | 7 | bits_clippers 7 / 53,808 |
| internal_oct17 | `outputs/rebot_2026-10-17-merged-v1` | 2 | 15,396 | 0.14 | 2 | bits_clippers 2 / 15,396 |
| external_rebot_train | `outputs/external_rebot_train-annotated-v9-tasks1` | 241 | 177,522 | 1.64 | 231 | external_rebot 231 / 171,876, excluded 10 / 5,646 |
| **total** | | **439** | **850,200** | **7.87** | | |

## Diverse corpus

`outputs/diverse_robot_dataset_v3_round18_screen125_corrected_v4_tasks2`: corpus/ + fmb/ episodes, minus 30 holdout or excluded episodes (anchor-level retention not counted). Hours from each episode's duration.

| source | episodes | frames | hours |
|---|---|---|---|
| droid | 48 | 34,752 | 0.64 |
| droid_success | 65 | 36,203 | 0.67 |
| fmb | 160 | 33,492 | 0.93 |
| molmoact | 399 | 72,657 | 1.27 |
| robochallenge | 196 | 297,768 | 2.76 |
| ur7e | 4 | 8,382 | 0.08 |
| yam | 7 | 4,597 | 0.05 |

## Validation

`outputs/rebot_val_2026-10-14-v1`: 9 episodes, 17,022 frames, 0.16 h.

## Built, not in the config

From `lerobot/dataset_info/pending.yaml`.

None.
