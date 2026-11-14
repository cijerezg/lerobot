# Reviewer brief: precision + contact strategy, one batch (issued 2026-09-25; every row is one segment)

You read contact sheets for one batch of step classes and return, per row, two labels with the evidence written first.
Rubrics of record: `../rubrics/precision_rubric.md`, `../rubrics/contact_strategy_rubric.md`. The definitions
below are copied from them; if this brief and a rubric differ, the rubric wins and you say so in `flags`.

## Your inputs (read only)

- `batches/<id>/manifest.json`: one entry per row. Fields: `row_id`, `class_id`, `read_level` (`class` = one representative
  segment stands for the whole class; `verify` = a sampled segment of a class already read; `segment` = this segment only),
  `half` (rebot | diverse), `root`, `source`, `subtask`, `verb`, `n_segments`, `n_episodes` (of the class), `episode`,
  `segment_index` or `atom_index`, `from_index`, `to_index`, `sheet`, `sheet_row`, `tiles` (role + frame of each of the 4
  tiles), `wrist` (present | absent), `approach_deg` (ReBot only: gripper axis vs straight down, 0 top-down, 90 horizontal,
  at the commit tile), `gripper_state` at the two frames (degrees; 0 shut, opens toward negative), `precision_prior`,
  `precision_prior_why`, `contact_prior`, `contact_prior_rule`, `flagged_for_image`, `flags` (`no_opening_in_segment`,
  `short_segment`, `wrist absent`).
- `batches/<id>/sheet_<k>.jpg`: 6 rows x 4 tiles, 256 x 192 each, labels burned in (root, episode, subtask, tile role,
  frame, approach angle, gripper values). Tile layout by verb:

| verb | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|
| grasp | top `to-15` | wrist `to-15` (commit: pads on the object) | wrist `to-45` (object under the descending pads) | wrist `to-1` (pads closed) |
| release | top at the opening onset | wrist at the onset (commit: pads start to part) | wrist onset+8 (object gone = drop; resting = set-down; seated = insertion) | wrist at the open event (fires ~5 frames before the onset), or onset-10 where there is no event |
| push, press, strike, pry, pour, drag, fold, wipe | top `from+15` | wrist `from+15` (commit) | top `to-15` | wrist `to-15` (what the object did) |
| insert, place, lift, everything else | top `to-1` | wrist `to-1` | top `to-30` | wrist `to-15` (commit) |

- The priors are hypotheses from text rules and the rubric table. They are never the verdict: the label must rest on what
  the tiles show. Where the tiles contradict the prior, label what you see and say what you saw in the note.
- You do not open the datasets, videos or other batches. Everything you need is in your batch directory.

## Definition 1: precision (verbatim, precision_rubric.md)

precision N = how tightly the step's committing action is constrained in space. It is a property of the step and the
scene (object, target, clutter), never of the demonstration. A fumbled coarse step stays coarse; a clean fine step stays fine.
Direction: 1 = coarse, 5 = fine.

    slack = the distance the gripper (or the held object) can be off the ideal pose at the commit and the step still
            completes, taken over the tightest axis, and shortened to the nearest object that must not be disturbed if
            that is closer.

| level | slack | reads as |
|---|---|---|
| 1 | > 10 cm, or no external target | any pose in a wide region works; approach from any direction |
| 2 | 3-10 cm | a large target for the object: cup into a basket rim-to-rim, block into a bin, pillow leaned in a band |
| 3 | 1-3 cm | the gripper must straddle a rigid object about its own size (cup, bottle, apple, block, handle), or the target mouth is a few cm wider than the object (cup on a rack, button face under a fingertip) |
| 4 | 3-10 mm | the object is small against the finger pads (hex bit, screw, pen), or the target opening barely exceeds the object (bit into a compartment, paper into a shredder slot) |
| 5 | < 3 mm, or a mating fit | peg into board, plug into socket, bit into a one-bit slot; an error jams and the step cannot finish without re-aligning |

Orientation modifier, at most +1, cap 5. Add 1 when the commit also pins the gripper or object orientation to roughly
±15°: pads must land across a bit lying flat; a stem goes into a vase mouth stem-first; a sheet goes into a slot edge-on;
a peg axis must match the hole. A cylinder grasped from above (cup, bottle) has free yaw: no bump.
Metric anchors are for the ReBot scene. On other robots read the same ratios visually: slack against the object's size
and against the finger pad width.

Where the commit is, per verb: grasp = the pads on the object at closing, over the axis the object constrains; release =
the target's free opening minus the object's footprint, at the height the object is let go; insert / place on fixture /
press / turn / rotate a control = the fit or the control face against what enters it; open / close / pull / push (drawer,
door, lid) = the handle grasp is its own step, the motion is judged on where it must stop; wipe / scrub / spread / fold /
unfold / straighten / flatten / stir / hold / pour / water / dump = the area or opening the action has to stay over.
Derived, not read (you will not get rows for them): move = one level below the next step (floor 1); return to home = 1;
lift = out of a fitted fixture, the fit's level minus 1; off a free surface, 1.

Not precision: how the demonstration went (retries, search, hesitation, overshoot, slips are quality or a mistake row;
two episodes of the same step in the same scene carry the same precision whatever their quality); tempo (do not raise
precision because the operator went slowly, or lower it because they were quick); care with a fragile object ("gently" is
speed and quality); difficulty from occlusion or reach (a target hidden from the top camera, or far from the base, does not
change its slack).

## Definition 2: contact strategy (verbatim, contact_strategy_rubric.md)

contact = how the gripper, or the object it holds, engages the thing the step acts on. One value per step. `na` when the
step neither makes nor breaks a contact and does no contact work. A step that makes a contact is labelled by that contact
(grasp, push, press). A step that breaks a contact is labelled by how it lets go (set-down, drop). A step that works
through a held object is labelled by what the object does (insert, pour, tool drag). When one step makes and breaks
contact (fold the towel over), the making wins. Read from the object and the gripper at the commit. Never from how well
it went (quality), how quickly (speed) or how tight the target is (precision).

| code | element | definition | wrist read at the commit |
|---|---|---|---|
| 0 | top-pinch | pads close from above on a rigid object's sides or top, approach < 45 | object between the pads seen from above, table behind it |
| 1 | side-pinch | pads close on the body from the side, approach >= 45 (wrap on a cylinder, face pinch on a book) | the object's side wall fills the frame, table at the edge or absent |
| 2 | rim-pinch | one pad inside a mouth or ring, one outside its wall, from above | mouth or ring under the pads, interior visible |
| 3 | handle-grasp | pads close on a handle, bar, knob, tab or lever attached to a larger body | the handle between the pads, the body beyond it |
| 4 | cloth-pinch | pads close on a fold, edge, corner or bunch of a deformable or thin sheet | fabric fills the frame, a fold or edge captured |
| 5 | push | sustained lateral contact; the object slides on the surface or swings on a hinge | gripper tip (usually closed) against the object's side while it moves |
| 6 | press | contact along the normal on a control face; the object does not translate | gripper tip landing on the face, then leaving |
| 7 | strike | impulsive contact by the gripper or a held tool; the gripper keeps moving | fast pass, object gone in the next frame |
| 8 | pry | closed gripper wedged under an edge and lifted | pads sliding under a flap, the flap rising |
| 9 | set-down | held object lowered until it rests on a surface, then released | object touching the surface before the pads open |
| 10 | drop | pads open while the object is unsupported over the target | air or the container interior under the object when the pads open |
| 11 | insert | held object guided into a fitted opening or over a peg; the fit does the last centimetres | the opening around the object, the object entering it |
| 12 | tilt-pour | held container rotated so its contents leave it | container tilting, contents moving |
| 13 | tool-drag | held object pressed on a surface and moved across it | rag / brush / spoon under the pads, surface sliding past |
| 14 | na | no contact made, broken or worked | move, return to home, FMB lift and in-hand rotate, tongs |

Approach angles are the ReBot FK angle between the gripper axis and straight down (0 = top-down, 90 = horizontal); on
other robots read the same thing visually. `approach_deg` in the manifest is that number for ReBot rows.

Not contact: approach angle beyond top / side (a cloth pinch of a hanging shirt is a note, not an element); pad
orientation across / along an elongated object; sub-techniques inside an element (wad vs corner cloth pinch, body vs neck
on a bottle, fingertip vs pad pinch, open-on-arrival vs lower-then-open drops, straight vs twist insertion, tip vs body
push); grasp point, release height, carry height, effort, tempo; how the demonstration went (a failed close is a mistake
row and a quality value; its contact label is still the mode the operator attempted).

## How to read a row

1. The commit tile first (green strip, marked COMMIT), then the other wrist tiles, then the top tile for context. Fill the evidence fields from what is visible.
2. Contact: is the gripper closing on the object (grasp family), touching without closing (push / press / strike / pry),
   or letting go / working a held object (set-down / drop / insert / pour / drag)? Then which part (top / side / rim /
   handle / fold). For a release, the onset tile vs the onset+8 tile decides: object still there and resting = set-down, gone or falling =
   drop, seated in an opening = insertion.
3. Precision: object size against the pad width (commit and wrist tiles); the target's free opening against the object's footprint
   and anything already in it (release rows); orientation pin (does the pose have to match to ~15°). Pick the slack band,
   then apply the modifier once.
4. Compare with the priors. Agree: say so in one clause. Disagree: label what you see, and the note states the observation
   that overrides the prior ("cup set on the basket floor, pads open with the cup resting").
5. `wrist absent` rows: answer the pad-level fields from the top tiles where you can; where you cannot, write `unclear`,
   keep the prior for that channel, and set `image_unreadable` only if the top tiles do not settle the element either.

## Output: `labels/<id>.json`

    {"batch": "<id>", "rows": [ {
      "row_id": "...", "class_id": "...", "read_level": "class|verify|segment",
      "gripper_closes_on_object": "yes|no|unclear",
      "part_touched": "top|side|rim|handle|fold|none|unclear",
      "object_state_at_open": "resting|unsupported|fitted|n.a.|unclear",
      "held_object_action": "none|insert|pour|drag|strike|unclear",
      "contact": "<element slug or na>",
      "contact_note": "one line: what in the tiles fixes the element",
      "object_vs_pads": "deformable|larger than the pads|about pad size|small against the pads|n.a.|unclear",
      "target_opening_vs_object": "no target|> 10 cm free|3-10 cm|1-3 cm|3-10 mm|< 3 mm or mating|unclear",
      "orientation_pin": 0|1,
      "precision": 1|2|3|4|5,
      "note": "one line: what was seen that sets the level; name the departure from the prior if any",
      "flags": ["scene_sensitive"|"mixed_class"|"image_unreadable"|"rubric_conflict"] } ] }

Evidence fields come first and are filled before the labels. One object per manifest row, in manifest order. Run
the pass's label validator on `<id>` (one JSON object per manifest row, enum values only, precision 1-5; the
2026-09-25 script was not kept) and fix until it passes.

## Rules

- Label the mode the operator attempted, not the outcome. A close that misses is still that grasp mode; a drop that
  bounces out is still a drop.
- Never raise precision for a slow or careful demonstration, and never lower it for a quick one. Never move it for
  retries, hesitation or a wandering search.
- Precision is set by the object and the target, not by the name: "the object", "the block", "the cooking ingredient" get
  their level from the tiles. Two rows of the same class in the same scene carry the same values; if they cannot, flag.
- `na` is for derived steps only (move, return, FMB lift and rotate, tongs). You will not normally receive such rows; if
  a read row shows no contact being made, broken or worked, set `na` with `mixed_class` and explain.
- Release read: the tile at the onset shows the object at the height it is let go; use it for the slack, use the later
  tile for resting / unsupported / fitted.
- Do not infer from the subtask string: the element, the level, the object's size, the container's fullness, whether
  the object rests or falls. You may take the verb and the object / target names for the note.
- `verify` rows: read them cold, exactly like a class row. Do not look up the class value; you do not have it.
- Every departure from a prior has a note naming the observation. A label with no note is not reviewable.
- Unclear is not a default value. If a field is genuinely unreadable write `unclear` there, label what the remaining
  tiles support, and say in the note what you could not see.

## Escalation flags you may set

- `scene_sensitive`: the value depends on scene state that changes across segments of the class (a container filling
  up, a slot with one object's room left, objects that differ in size behind one name). The class goes per segment.
- `mixed_class`: the tiles show a different step than the string implies, or two modes within the row's own frames
  (a push that becomes a pinch, a set-down that reads as an insertion), or a read row with no contact at all.
- `image_unreadable`: the decisive tile is black, blurred or does not show the pads / object / target, and the other
  tiles do not settle the value. Keep the prior for that channel and say which tile failed.
- `rubric_conflict`: the tiles fit no element or no slack band as defined; describe what you saw.

## Forbidden

- Writing anything except `labels/<id>.json` (and, if you must render, images under `review/<id>/`). Never write
  into any dataset root, `meta/`, `rows/`, `batches/`, another batch's directory, or the rubric files.
- Reading other batches' manifests or labels, the class table, or the datasets.
- Changing the subtask string, the span, quality, mistakes or speed. Those are other channels; note, do not edit.
- Inventing an object name, a level or an element you cannot point to in a tile.

## Final reply to the lead (under 120 words)

Rows done / total; precision histogram; contact histogram; number of rows departing from each prior with the class ids;
every flag set with its row id; anything odd about the sheets (misaligned tiles, wrong frames, wrong camera roles).
