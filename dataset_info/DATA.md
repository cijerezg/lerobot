# Data record

Generated 2026-11-10 13:13 by `uv run python lerobot/dataset_info/data_count.py` from `config_rl.yaml`. Do not edit by hand; rerun after every change to the data or the config.

## Training mix by sampling group

Share = fraction of all training samples. Usable = frames minus the usefulness cuts (`lerobot/dataset_info/manifests/rebot_exclusions_2026-10-18.json`; rule: `lerobot/src/lerobot/annotation/rubrics/usefulness_review_rubric.md`). ReBot hours at 30 fps.

| group | share | episodes | frames | usable frames | hours |
|---|---|---|---|---|---|
| external_rebot | 0.2 | 231 | 171,876 | 171,144 | 1.59 |
| bits_clippers | 0.155 | 55 | 280,476 | 265,399 | 2.60 |
| rigid | 0.08564 | 55 | 106,989 | 106,933 | 0.99 |
| cases_kit_box_push | 0.05119 | 24 | 44,820 | 39,409 | 0.41 |
| bottles_lids | 0.03789 | 17 | 59,889 | 53,109 | 0.55 |
| clothing | 0.03783 | 79 | 280,194 | 259,848 | 2.59 |
| pills_caps | 0.03248 | 12 | 33,459 | 32,456 | 0.31 |
| excluded | 0 | 13 | 20,634 | 0 | 0.19 |
| **ReBot sampled** | **0.6** | **473** | **977,703** | **928,298** | **9.05** |
| diverse corpus | 0.4 | 879 | 487,851 | | 6.39 (usable 6.33) |

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
| internal_oct21 | `outputs/rebot_2026-10-21-annotated-v1` | 10 | 18,285 | 0.17 | 10 | bits_clippers 10 / 18,285 |
| internal_oct26 | `outputs/rebot_2026-10-26-merged-v2` | 19 | 80,169 | 0.74 | 19 | rigid 10 / 25,620, bits_clippers 8 / 52,869, cases_kit_box_push 1 / 1,680 |
| internal_oct27 | `outputs/rebot_2026-10-27-merged-v1` | 4 | 12,171 | 0.11 | 4 | rigid 3 / 5,808, bits_clippers 1 / 6,363 |
| internal_oct29 | `outputs/rebot_2026-10-29-merged-v1` | 14 | 37,512 | 0.35 | 14 | rigid 3 / 10,737, bits_clippers 5 / 19,161, cases_kit_box_push 6 / 7,614 |
| external_rebot_train | `outputs/external_rebot_train-annotated-v9-tasks1` | 241 | 177,522 | 1.64 | 231 | external_rebot 231 / 171,876, excluded 10 / 5,646 |
| **total** | | **486** | **998,337** | **9.24** | | |

## Diverse corpus

`outputs/diverse_robot_dataset_v3_round18_screen125_corrected_v4_tasks2`: corpus/ + fmb/ episodes, minus 30 holdout or excluded episodes (anchor-level retention not counted). Hours from each episode's duration; usable = minus `lerobot/dataset_info/manifests/diverse_exclusions_2026-10-18.json`.

| source | episodes | frames | hours | usable hours |
|---|---|---|---|---|
| droid | 48 | 34,752 | 0.64 | 0.61 |
| droid_success | 65 | 36,203 | 0.67 | 0.65 |
| fmb | 160 | 33,492 | 0.93 | 0.92 |
| molmoact | 399 | 72,657 | 1.27 | 1.27 |
| robochallenge | 196 | 297,768 | 2.76 | 2.75 |
| ur7e | 4 | 8,382 | 0.08 | 0.08 |
| yam | 7 | 4,597 | 0.05 | 0.04 |

## Validation

`outputs/rebot_val_2026-10-14-v1`: 9 episodes, 17,022 frames, 0.16 h.

## Built, not in the config

From `lerobot/dataset_info/pending.yaml`.

None.

## Objects (sampled ReBot episodes)

From each root's `meta/episode_metadata.parquet`, groups with share > 0. Cycle = consecutive segments of one episode on the same object, split at each fresh grasp (grasp, move, release = 1; back-to-back regrasps = 1; each bit kit flap = 1; a retry after a failed close or insert = 1 more). Object = family matched in the subtask string (`FAMILIES` in data_count.py), else the noun phrase. Frames = segment frames, return to home not counted. Variants = the noun phrase of each cycle's first segment. Full subtask list: `SUBTASKS.md`.

| object | grasp cycles | internal | external | episodes | frames | hours | variants (cycles) |
|---|---|---|---|---|---|---|---|
| sock | 203 | 203 | 0 | 64 | 158,592 | 1.47 | sock 85, white sock 55, black sock 39, plain black sock 8, striped sock 7, plain white sock 5, plain sock 2, striped black sock 1, long black sock 1 |
| shirt | 159 | 159 | 0 | 48 | 104,315 | 0.97 | shirt 88, grey shirt 17, white shirt 15, red shirt 15, navy shirt 10, brown shirt 5, green shirt 3, black shirt 3, beige shirt 3 |
| bit | 92 | 92 | 0 | 32 | 103,825 | 0.96 | bit 79, leftmost bit 5, two bits 2, blue bit 2, two touching bits 1, slanted bit 1, three bits 1, yellow bit 1 |
| nail clipper | 77 | 77 | 0 | 35 | 76,067 | 0.70 | nail clipper 38, large nail clipper 24, small nail clipper 15 |
| pill bottle | 73 | 73 | 0 | 57 | 69,403 | 0.64 | pill bottle 52, yellow pill bottle 11, orange pill bottle 10 |
| towel | 65 | 0 | 65 | 10 | 20,058 | 0.19 | corner 34, towel 31 |
| bit kit | 64 | 64 | 0 | 34 | 47,073 | 0.44 | flap with the white card 32, flap with the bits 31, flap with the shipping label 1 |
| headphone case | 62 | 62 | 0 | 38 | 46,114 | 0.43 | headphone case 60, case with the headphones 2 |
| tape roll | 35 | 25 | 10 | 34 | 25,197 | 0.23 | tape roll 25, green tape roll 10 |
| cup | 34 | 34 | 0 | 34 | 31,317 | 0.29 |  |
| bowl | 28 | 0 | 28 | 10 | 5,814 | 0.05 |  |
| water bottle | 26 | 26 | 0 | 19 | 25,223 | 0.23 |  |
| T block | 20 | 0 | 20 | 20 | 9,699 | 0.09 |  |
| apple | 20 | 0 | 20 | 10 | 5,344 | 0.05 |  |
| charger | 20 | 0 | 20 | 20 | 7,470 | 0.07 |  |
| drawer | 20 | 0 | 20 | 10 | 5,734 | 0.05 | upper drawer 10, lower drawer 10 |
| fig bar | 20 | 0 | 20 | 10 | 6,906 | 0.06 |  |
| mallet | 20 | 0 | 20 | 10 | 2,741 | 0.03 |  |
| eraser | 19 | 9 | 10 | 19 | 9,364 | 0.09 | green eraser 15, yellow eraser 4 |
| spray bottle | 19 | 19 | 0 | 17 | 16,292 | 0.15 |  |
| disk | 18 | 0 | 18 | 18 | 9,553 | 0.09 | disk 10, black disk 4, wooden disk 4 |
| box | 12 | 1 | 11 | 11 | 4,396 | 0.04 |  |
| cap | 12 | 12 | 0 | 12 | 8,327 | 0.08 |  |
| lid | 11 | 11 | 0 | 8 | 7,162 | 0.07 |  |
| yellow piece | 11 | 0 | 11 | 9 | 3,609 | 0.03 |  |
| bolt | 10 | 0 | 10 | 10 | 2,040 | 0.02 |  |
| crayfish | 10 | 0 | 10 | 10 | 5,370 | 0.05 |  |
| microcontroller | 10 | 0 | 10 | 10 | 4,873 | 0.05 |  |
| motor | 10 | 0 | 10 | 10 | 4,555 | 0.04 |  |
| orange piece | 10 | 0 | 10 | 9 | 2,825 | 0.03 |  |
| paper ball | 10 | 0 | 10 | 10 | 2,928 | 0.03 |  |
| puck | 10 | 0 | 10 | 10 | 8,375 | 0.08 |  |
| red die | 10 | 0 | 10 | 10 | 5,327 | 0.05 |  |
| tea bag | 10 | 0 | 10 | 10 | 1,292 | 0.01 |  |
| wall plug | 10 | 10 | 0 | 6 | 8,306 | 0.08 | wall plug 5, red wall plug 3, yellow wall plug 2 |
| blue piece | 9 | 0 | 9 | 9 | 3,075 | 0.03 |  |
| packet | 8 | 0 | 8 | 7 | 4,311 | 0.04 |  |
| shaker bottle | 8 | 8 | 0 | 6 | 8,632 | 0.08 |  |
| cork | 7 | 0 | 7 | 7 | 2,019 | 0.02 |  |
| paper | 7 | 0 | 7 | 7 | 3,322 | 0.03 |  |
| sugar cube | 7 | 7 | 0 | 7 | 2,276 | 0.02 |  |
| can | 6 | 0 | 6 | 6 | 2,955 | 0.03 | can 3, red can 3 |
| noodles | 6 | 0 | 6 | 6 | 538 | 0.00 |  |
| tissue | 6 | 0 | 6 | 6 | 2,265 | 0.02 |  |
| cube | 5 | 0 | 5 | 5 | 1,635 | 0.02 |  |
| box lid | 4 | 4 | 0 | 4 | 3,039 | 0.03 |  |
| egg | 4 | 0 | 4 | 4 | 552 | 0.01 |  |
| book | 3 | 3 | 0 | 3 | 2,403 | 0.02 | top book 2, books 1 |
| floss | 3 | 3 | 0 | 3 | 2,473 | 0.02 | floss 2, dental floss 1 |
| orange bottle | 3 | 3 | 0 | 2 | 4,320 | 0.04 |  |
| block | 2 | 0 | 2 | 2 | 555 | 0.01 |  |
| blue cube | 2 | 0 | 2 | 2 | 238 | 0.00 |  |
| ham | 2 | 0 | 2 | 2 | 135 | 0.00 |  |
| oil | 2 | 0 | 2 | 2 | 240 | 0.00 |  |
| purple box | 2 | 0 | 2 | 2 | 657 | 0.01 |  |
| deck of cards | 1 | 1 | 0 | 1 | 2,058 | 0.02 |  |

## Subtasks by verb (sampled ReBot episodes)

| verb | segments | frames | hours |
|---|---|---|---|
| grasp | 1255 | 556,342 | 5.15 |
| move | 966 | 168,917 | 1.56 |
| release | 922 | 82,306 | 0.76 |
| return to home | 355 | 74,519 | 0.69 |
| fold | 94 | 15,913 | 0.15 |
| close | 73 | 18,077 | 0.17 |
| stand | 40 | 15,161 | 0.14 |
| push | 38 | 17,495 | 0.16 |
| pour | 26 | 7,495 | 0.07 |
| spread | 19 | 1,243 | 0.01 |
| take | 11 | 2,030 | 0.02 |
| open | 10 | 705 | 0.01 |
| hit | 10 | 8,375 | 0.08 |
| plug | 10 | 2,017 | 0.02 |
| set | 9 | 1,829 | 0.02 |
| press | 5 | 555 | 0.01 |
| lift | 5 | 2,091 | 0.02 |
| shake | 4 | 645 | 0.01 |
| pull | 4 | 346 | 0.00 |
| straighten | 3 | 423 | 0.00 |
| slide | 3 | 259 | 0.00 |
| turn | 3 | 558 | 0.01 |
| lay | 2 | 360 | 0.00 |
| line | 1 | 42 | 0.00 |
