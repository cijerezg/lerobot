# External ReBot grading decisions (2026-10-08)

Rulings of the external re-annotation pass (241 episodes, `rebot_external__*` classes; result root
`external_rebot_train-annotated-v9`). They override the `rebot_external__*` FINAL notes where the two
conflict. General rulings are also in [quality_mistake_rubric_v2.md](quality_mistake_rubric_v2.md)
(sections 5.3, 5.7, 5.9, 5.11, 5.13, 6); this file keeps all of them with their class detail.
"Rubric" = that file.

## General

| D | Ruling |
|---|---|
| carry level | A carry's precision window takes the level of the release that follows it. The label's move-minus-one value is a legacy prior. |
| commit frames | The header commit and the measured gripper onset are both accepted when up to about 15 frames apart. On a wrong event, or before the unit start, use the measured onset. |
| D1 | Still head. A fully still head of 1 s or more that idle_scan will not cut (under 3 s) is `idle` 1 over the still run. Under 1 s gets nothing. |
| D2 | Return overshoot. Rubric 5.13 governs returns; a class row-4 "minor correction" is a correction below the 5.13 threshold. An overshoot past park that comes back, confirmed on the frames, is `search` 3, 2 only where it drags on. A wrist that rotates into the park pose in one motion while d falls gets no span. |
| D3 | Return 5s. "Stay at 4 when evidence is limited" is superseded by rubric 5.3. |
| D4 | Approach line. On push, insert, strike, fold and pour units the line points at the unit end, so it is not evidence: read the frames. The frames also win where the line misses an early detour or sits at a threshold edge. |
| D5 | Old 4 calls. Old FINAL cues ("keep 5 rare", "a small settle stays 4") and old per-unit 4 calls are superseded by rubric 5.3. A reference that is the unit being graded is not evidence. |
| D18 | Aborted approach. As after a failed close, the fresh final approach can be exemplary. |
| D23, D40 | Same-scene precision. One level per step and scene. Where sibling labels differ, settle one level on the frames, fix the labels, then the windows follow. The grader does not re-pick levels otherwise. |
| D38 | Hold at a mating face. A no-progress hold of more than 1 s at the face is `hover` 3 (user 2026-09-30), 2 past twice the external p90 (about 8 s). |
| D39 | Cause names. A back-off before a grasp is `search` 3, not `aborted_approach`, unless the gripper closed on nothing. A wrist past park and back on a return is `search`, not `stall`. |

## By class or object

| D | Class / object | Ruling |
|---|---|---|
| D6, D37 | grasp__rigid | The class keeps mixed objects. Charger: grasp__block charger rows (body centred = 5, plug end = 3); one quick close with the arm still = 5, a slow or two-stage close while the arm drifts or the wrist turns = 4. Hanoi pieces: grasp__ring_disk rows (face-on pinch across the diameter, centred = 5). Lying can or box: body grip across the middle in one aligned descent = 5. A second fixed camera that shows the grip point counts as the close view; a frozen wrist image counts as no wrist. |
| D7 | handshake | Grade per unit. Grasp unit = final reach plus close; shake unit = shake plus release. When the partner withdraws the hand, "release and clear" is not required. A firm top pinch on the back of the hand that does not slip is graded like side-on. Pre-offer waits are left to idle_scan. |
| D8 | drawers, paper ball | Drawer pulls: panel_close rows mirrored (straight along the rails in one stroke = 5). Paper-ball carry to the table: the soft_toy "box" cues do not apply; a direct low move to the table is 4 or 5. |
| D9 | peg fixtures | A flat hold of more than 1 s over the peg, piece not descending, is `hover` 3. A slow continuous descent onto the peg gets no span. Case `q3_ext_ep81` predates rubric 5.13. |
| D10, D31 | fig bar, packets | Aim for the middle; an edge grip is 3 (user 2026-09-29). A fig bar gripped at a crimped end so it hangs lengthwise is `strategy` 3; all fig-bar grasps of the breakfast repo are this grip (gripper settles at -5 to -10, apples about -160). A flat packet gripped at its only graspable edge gets no strategy critique. |
| D11, D32 | trash release (dark scene) | An opening while the end link still moves sideways at about 20 cm/s or more (FK) and the wrist view sweeps off the can is row-3 evidence without a rim view. Joint speed alone (a wrist pitching in place) does not count. Otherwise 4. |
| D12 | crayfish | A diagonal landing in a box barely larger than the plush gets no critique. |
| D13 | block_to_surface | A tall box placed on end that topples but ends centred is 4. An object centred on its outline but turned so it does not fit the outline is `poor_place` 3 (rubric 4.3), not `strategy`. |
| D14 | apples in the flat classes | The flat rows apply (centred body pinch = 5). |
| D15 | carry__ring_disk | A single arc whose approach line falls steadily gets no critique. `strategy` 3 only for a loop or reversal seen on the frames. |
| D16, D34 | strikes (air hockey) | One precision window per strike (`precision_2`, ... for the extra ones). Commit = mallet-puck contact. w_a = the puck within about 10 cm of the contact point and at least 1 s before contact; w_b = contact + 15 frames. A weak push that stops short of the centre line still counts as a strike. A missed block stays `other` with a note. |
| D17, D35 | push T | "On the target" = the T substantially overlaps the outline. Before that, one `strategy` span per drive. After it, every lift-and-re-contact is `search` 3, 2 where it repeats with no visible gain for more than about 4 s. |
| D19, D36 | paper-ball release | Split by whether the ball was at table level before the jaws parted, not by arm speed. Set down then open: 5 possible. Opened above the table on the pull-back (ball flung or clinging): `strategy` 3. |
| D20, D33 | microcontroller release | "Mid-swing" = at the measured opening onset the pan still moves at 8 deg/s or more and the horizontal end-link speed is at least the vertical speed. Lowering while opening is not mid-swing. Row 5 needs a low release; a drop from above the rim caps at 4. |
| D21 | bolt release | "Upright" = left in its set-down pose. A bolt set down lying flat that stays put fits row 5. A part that topples or pivots after the opening is row 4. |
| D22 | die bin release | Grade the release (low over the middle, then the opening). A rim release whose die bounces to the centre does not earn a 5. Where the die rests matters only for a mistake. |
| D24 | bowl grasp in a narrow gap | A rim pinch that leaves the neighbour untouched, in one clean descent, can be 5. Row 4 "reduced clearance" is a visible adjustment for the gap. |
| D25 | pour | A tip from 10-15 cm above the rim is 4 on what is visible. |
| D27 | towel | A carry where the cloth rides on open jaws is `strategy` 3. Cloth caught on the hand after the opening is `other` 2. A shake-out swing is the purpose: no `strategy` for the swing itself. |
| D28 | insert alignment | Local alignment on one socket, up to about 1 cm, is the 4 row. `strategy` 3 only when the loaded plug translates across sockets. |

Tool notes (D26): the motor repo's wrist view lags, read the top view. From idx 12 on, old
`external_ep<N>` = `ex1007_ep<N+1>`. Spans and windows past the unit end are clipped by the writer at
episode bounds only.

Demoted references (D3, D12, D13, D19, D29) are recorded in the class FINAL notes.

## Audit sweeps (6 rounds, 2026-10-08)

Patterns the random audit found every round, each closed by a mechanical sweep. Run the same sweeps on
a new external pass before the audit.

| Pattern | Sweep |
|---|---|
| Return with an overshoot past park and no critique (D2) | returns with an approach-line rise and no `search` / `stall` / `hover` span |
| Unit held at 4 by an old call or reference (D5) | units with no exemplary / strategy whose notes cite an old call, a reference or "rare 5" |
| Clean action at default 4 while the same step in a same-repo sibling is exemplary | every pure-4 unit whose repo + subtask group has exemplary siblings (at least a third of the group) |
| Near-idle stretch the idle scan does not cut (slow drift breaks its 2 deg test) and no span | `motion.idle_runs` at 6 deg / 2.5 s, minus cuts and spans graded 3 or lower, at least 2.5 s left |
| Strike unit with one window over the whole rally | D16 / D34 on every rally unit |
| Return held at 4 because the recording ends at park | read the frames; exemplary when the arm reaches park in one motion |

Checker note: the trainer's window opens 1 s before w_a (rubric 6). Compare the label's w_a with the
10 cm point, not the trainer's window start.
