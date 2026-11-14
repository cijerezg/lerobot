# Reviewer brief: precision + contact on diverse rows (addendum, 2026-09-25)

Read [AGENT_BRIEF.md](AGENT_BRIEF.md) first. Everything there holds. Only these differ on diverse rows
(`half: diverse`). Rows come from `diverse_rows.py`, sheets from `render_rows.py` / `render_strips.py`.

| item | ReBot | diverse |
|---|---|---|
| root | own roots | one per robot: droid, droid_success, molmoact (Franka), rc_arx5, rc_ur5, ur7e, yam, fmb (Franka) |
| tile offsets | 30 Hz frames | same times in native frames: to-15 (0.5 s) prints to-8 at 15 Hz |
| top tile | top camera | droid left_external, molmoact primary, robochallenge global, ur7e topview, yam outside/top, fmb side_1 |
| grip | degrees, 0 shut | percent closed over the episode's range, 100 = most closed; FMB is the binary command (0 / 100) |
| appr | FK angle | FK angle; `nan` on ARX5 and YAM (read the angle from the images); UR5 is 0 in every grasp |
| release commit | state onset | same rule in percent; none found: `steepest` = the steepest opening in the atom, else `to-0.5 s` |
| contact layout | push press strike pry pour drag fold wipe | also scrub stir spread water unfold flatten straighten tilt |
| long rows | - | contact / end-layout atoms over 4 s carry `long_segment`: the fixed tiles often show the approach, not the contact; label from the tiles you have, the row also gets an 8-frame strip |

FMB `grasp the object`: the contact comes from the FK angle (`contact_fixed` in the manifest; < 45 top-pinch,
>= 45 side-pinch). Still fill every field and label the contact you see; a disagreement is a check on the angle
rule, not an error.

Metric anchors: read slack against the object and the finger pads, not centimetres.

Derived at write, not read: move = next non-move step minus 1 (legacy prior), return 1, contact `na`. A grasp or
release atom with no close / open inside it (approach only, release tail) is labelled `na` by the reader and
inherits the nearest atom of the same subtask at write.

## Known data issues

| source | issue |
|---|---|
| MolmoAct | the wrist view never shows the pads; pad-level fields come from the fixed cameras or stay `unclear` |
| DROID, UR7e | some grasps are split into an approach atom and a close atom; the approach atom inherits |
| ARX5 | the "pink" button is red; keep the task string's name |
| MolmoAct ep7316, ep7327 | "blue mug" is white / glass |
| ep2607 | "roll of tape" is a Play-Doh tub |
| UR5 | "grapes" is one fruit |
| MolmoAct ep003862, ep003931 | identical tiles, possible duplicate episode |

Class values settled in adjudication are rows of the reference table in `../rubrics/precision_rubric.md` and the
rulings table in `../rubrics/contact_strategy_rubric.md`.
