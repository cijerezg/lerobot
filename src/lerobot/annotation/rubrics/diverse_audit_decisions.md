# Diverse corpus audit: decisions and known defects (2026-10-07 to 2026-10-09)

Rulings of the diverse annotation audit (909 episodes; atoms, exclusions, contact, precision, quality v2)
and the systematic defects it found but did not sweep. Rubrics of record:
[annotation_principles.md](annotation_principles.md), [subtask_atoms_rubric.md](subtask_atoms_rubric.md),
[quality_mistake_rubric_v2.md](quality_mistake_rubric_v2.md). C = audit case ruling, K = known defect.

## Status

| step | result |
|---|---|
| random rounds | 19 rounds, 17% of the 909 episodes |
| coverage screen, phase 1 | closed by the user 2026-10-09: 125 of 149 selected checks integrated, 80 episodes flagged, 181 issue rows (174 repaired, 3 rejected, 4 deliberate no-change), 91 corrected episode specs (78 corpus, 13 FMB) |
| not screened | the 24 selections without checks |
| K defects | repaired only where the predicate lands in the 125 screened episodes; the corpus-wide sweeps are not run (see [OPEN_ITEMS.md](OPEN_ITEMS.md)) |

Reference data, `lerobot/annotation_data/diverse_audit/`:

| path | holds |
|---|---|
| `screen_phase1_125_v4/` | the 91 corrected episode specs (input of `diverse_corrected_store.py`) |
| `repair_phase1.json`, `integration_phase1.json`, `selection_phase1.json` | disposition per issue, per check, and the 149 selections |
| `k4_k7_motion_predicate.json` | the K4 / K7 sweep list |
| `speed_traces/diverse_trace_map.json` + `speed_traces/<fix>/...` | speed traces of repaired episodes; `python -m lerobot.annotation.speed.hybrid_diverse runs <store> <map>` |

Per-source sweeps are in `lerobot/annotation_data/diverse_audit_sweeps/`.

## Rulings

| C | Rule |
|---|---|
| C1 | Idle under 3 s is not repaired (the idle cut starts at 3 s). |
| C4, C9 | Excluded gaps and tails with under 1 s of motion are not restored. A tail inside the final return is not a task action. |
| C5 | Contact `na` on a grasp or release atom is wrong (K9). `na` stays only where nothing is touched, e.g. a close in free air. |
| C8, C18 | Purposeful manipulation after the task is kept and relabelled for what it does. An unfinished last carry keeps a move text with no arrival. A partial close that never touches the object is not a mistake. |
| C10 | A tail cut by the length cap is restored only to the end of the interrupted action. |
| C11, C17 | Look-alike drawers, doors and shelves are named by position. Interchangeable items keep "a / the <item>". A 1-2 s pause left at 4 is not repaired. |
| C12 | A short tail that rises or returns after the last release is not flagged. |
| C13 | FMB grasp atoms ending 1.0-1.5 s after the close settles are source-native bounds and stay (case `audit_fmb_ep000061`). |
| C14 | A gripper open or close inside an excluded gap or tail is a defect at any length (K7). |
| C16 | A screener can get an empty image read. Every flag is re-checked on the frames by the repairer; a sample of clean checks gets a second read. |
| C19 | A visible miss, failed placement or redo left at effective quality 4 is a flag. |
| C20 | A flower that falls out over the rim after the release is `drop`; the attempt is 1 when never righted (case `mistake_arrange_flowers_ep000375`). |
| C21 | "push / rotate / pull the X" with no direction or outcome is a text defect (K12). A withdrawal of about 1 s folded into the last atom is atom grammar, not a defect. |

## Known defects, not swept corpus-wide

Frames decide every candidate. Screeners keep flagging the recurring ones per episode.

| K | Where | Defect | Scope | Fix |
|---|---|---|---|---|
| K1 | fmb multi_object | every atom and task says "the object" with several loose parts on the table | 60 episodes, about 400 atoms | name the grasped part (colour + shape) per episode |
| K2 | fmb single_object | "the object", one object on the table | 100 episodes | replace by a short visible noun |
| K3 | robochallenge press_the_button, turn_on_the_light_switch | press atoms end late (after lift-off); the precision window commits on the transit | 40 episodes | commit at first contact |
| K4 | droid, droid_success | excluded head under a false reason ("static", "out_of_scope") holds the approach to the first object | `k4_k7_motion_predicate.json` | restore |
| K5 | droid, droid_success | scene misread from the wrist view (rotated camera, thin handles read as empty closes): wrong action texts, grasps labelled as never happening | per episode | external views settle it |
| K6 | robochallenge shred_paper | restored head left the grasp inside "move the paper to the shredder" with contact `na` | 14 candidates of 20 | split out "grasp the <colour> paper", cloth-pinch |
| K7 | droid, droid_success, robochallenge | gripper open or close inside an excluded gap or tail | `k4_k7_motion_predicate.json` | restore the interval |
| K8 | episodes with a restored head | mistake rows, spans and windows still start at the old boundary; the restored part of a failed attempt reads as 4 | v2 spans starting exactly at a pre-restore start | regrade the restored part |
| K9 | droid, droid_success, robochallenge, ur7e | grasp atoms with contact `na` (32 atoms, 21 episodes), release atoms with `na` (48 atoms, 37 episodes) | 80 atoms | label the element |
| K10 | molmoact | "lift the X" also covers the carry, set-down and opening, no release atom | all 9 lift atoms | split |
| K11 | molmoact household | text names the wrong one of two look-alikes (task's fruit instead of the one grasped; "the bowl" with a second bowl in view) | per episode; cereal-bowl kitchen siblings ep000940, 001935, 001984, 002001, 002028 | rename from the frames |
| K12 | all | "push / rotate / pull the X" with no direction or outcome | 24 atoms in 22 episodes (molmoact 17, yam 4, droid_success 3); "pull" not yet in the predicate | add the direction or outcome seen |
| K13 | fmb | "rotate the object" also covers the set-down, opening and regrasp approach | 19 of 19 atoms | split into rotate / release / grasp, or keep as source-native (C13): not decided |
| K14 | robochallenge | excluded head ending at f180 tagged "static" holds the approach (arrange_flowers ep000985, arrange_fruits ep000305, item_classification ep000000); not caught by the K4 predicate | every head ending at 180 | restore |

K4 / K7 predicate: an excluded interval with a continuous run of more than 1 s at joint speed above 0.1 rad/s.
Hits: droid 39 of 50 episodes, droid_success 41 of 70, robochallenge 15 of 200; 111 intervals (79 heads, 28
tails, 4 gaps). Tails include real out-of-scope tasks and returns home (C12).
