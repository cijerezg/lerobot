# Annotation: open items

Questions the passes left for a decision. Remove a row when it is settled and the rule is in its rubric.

## Open class-file items

Class files are `classes/<class>/strategy/FINAL.md` under `paths.QUALITY_V2`.

| item | source | pending |
|---|---|---|
| push class has no clipper row | pass 2026-10-14 | add: 5 first press seats it, 4 after one re-press, 3 springs back or tips on the hinge |
| release__misc has no fitted-slot row | pass 2026-10-14 | add: flat approach line of more than about 1 s before the opening = `hover` 3 from where the line goes flat; no usual hover time is set for a clipper into its slot |
| rigid_misc has no clipper row | pass 2026-10-14 | write the row |
| lying_bottle has no spray-bottle row | passes 2026-10-07, later | body 5 / shoulder 4 / neck 3; a cap grip with the bottle hanging is 3 in the later passes, 4 from the side (3 only from the top) in the 10-07 ruling |
| standing bottles are graded against lying_bottle | later passes | own rows or own class |
| grasp / rigid rows are written for the tape roll | later passes | floss box and deck of cards are graded there without rows |
| kit flap grasp: label precision 3, class file 2 | later passes | one level |
| clipper slot release: label precision 5, grader 4 | later passes | one level |
| deck of cards grasp: precision 4 vs 3 between episodes | later passes | one level |

## Labels and data

| item | source | pending |
|---|---|---|
| kit-corner hover of 27 s in ep12 / ep13, text ambiguous | pass 2026-11-10b | text and grade |
| 29 lid grasps of earlier passes labelled rim-pinch, likely handle-grasp on the loop tab | teleop pass 2026-10-26 | sweep (contact rubric, rulings table) |
| pools compiled before the `compile_labels` fix dropped second-read `unit` change rows | audit 2026-10-07 | not checked which pools |
| floss teleop root: 42 idle runs (297 s, 31% of the root) are in `idle_exempt`, not cut; 26% of frames at quality 1-2; 4 of 8 episodes never seat a tube | teleop pass 2026-11-11 | cut rule for slow line-ups; training share of the root |
| FMB place on fixture = set-down, ARX5 flat flower grasp = 4, ARX5 crumb wipe = 2: applied to the labels and the rubric tables; the adjudication marked them "user decides" | diverse precision / contact pass 2026-09-25 | confirm |
| diverse known defects K1-K14: corpus-wide sweeps not run | [diverse_audit_decisions.md](diverse_audit_decisions.md) | run or accept |
| 24 screen selections never screened | diverse coverage screen 2026-10-09 | screen or accept |
| diverse audit stop rule: minor issues stayed at about 4 per round | diverse audit | what ends the audit |
| external audit stopped after 6 rounds without two clean rounds | external audit 2026-10-08 | accepted by the user for v9; rule for the next pass |

## Tools

| item | source | pending |
|---|---|---|
| `idle_scan` cannot run on episodes split into several entries | pass 2026-11-10b | support entries |
| `check_precision_chain` misses a carry before a pour | later passes | add the case |
| `render_review_videos` hardcodes `external_0` / `wrist_0` | external pass 2026-10-08 | read the camera list |
| MolmoAct reviewer brief cites `outputs/_annotation/diverse_v2_expansion_review/REVIEW_PROTOCOL.md`, not on this disk | `examples/dataset/diverse_robot_dataset/molmoact_review/` | recover the file or fold its rules into the brief |
