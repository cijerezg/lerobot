# Annotation rubric: precision per step (settled 2026-09-24; nothing annotated yet)

Companion to `annotation_rubric.md` (quality + mistakes) and the speed label
(`data_processing/annotate/hybrid_motion_speed.py`). Fourth metadata channel.

## What the label means

> **precision N = how tightly the step's committing action is constrained in space.**
> It is a property of the step and the scene (object, target, clutter), never of the
> demonstration. A fumbled coarse step stays coarse; a clean fine step stays fine.

Direction: **1 = coarse, 5 = fine.** 1 is "anywhere in a wide region works" (push a
door, drop a sock in a basket, return to home); 5 is a mating fit (peg into board, bit
into a one-bit slot).

Why per step and not per task: per task the label is a relabelled task id, fully
predictable from the prompt and the image, and unsteerable (no task is seen at two
levels). Per step it varies inside every episode (transport 1, grasp a bit 5), so the
model can learn *when* to be careful and the label transfers across the diverse corpus,
where hundreds of tasks share five levels.

Prompt clause (proposed, after speed): `The precision is N of 5.` Inference sets the
**true** value of the current step, not a wish; see "Inference" below.

## Units

| half | unit | storage |
|---|---|---|
| ReBot | one value per semantic segment, constant across it | `meta/precision.parquet`, rows 1:1 with `episode_metadata.parquet` (mirrors `speed.parquet`) |
| diverse | one value per reviewed atom | `precision_atoms.jsonl` keyed by `(episode_id, atom_index)` (mirrors `speed_atoms.jsonl`) |

Same broadcast machinery as quality and speed (`materialize_metadata`, atom lookup by
native timestep); no per-frame column.

## The measure: slack at the commit

Every step has one **committing action**: the gripper closing (grasp), opening (release),
the object seating (insert, place, press), or the end pose (move). Define

    slack = the distance the gripper (or the held object) can be off the ideal pose at the
            commit and the step still completes, taken over the tightest axis, and
            shortened to the nearest object that must not be disturbed if that is closer.

| level | slack | reads as |
|---|---|---|
| **1** | > 10 cm, or no external target | any pose in a wide region works; approach from any direction |
| **2** | 3-10 cm | a large target for the object: cup into a basket rim-to-rim, block into a bin, pillow leaned in a band |
| **3** | 1-3 cm | the gripper must straddle a rigid object about its own size (cup, bottle, apple, block, handle), or the target mouth is a few cm wider than the object (cup on a rack, button face under a fingertip) |
| **4** | 3-10 mm | the object is small against the finger pads (hex bit, screw, pen), or the target opening barely exceeds the object (bit into a compartment, paper into a shredder slot) |
| **5** | < 3 mm, or a mating fit | peg into board, plug into socket, bit into a one-bit slot; an error jams and the step cannot finish without re-aligning |

**Orientation modifier, at most +1, cap 5.** Add 1 when the commit also pins the
gripper or object orientation to roughly ±15°: pads must land across a bit lying flat;
a stem goes into a vase mouth stem-first; a sheet goes into a slot edge-on; a peg axis
must match the hole. A cylinder grasped from above (cup, bottle) has free yaw: no bump.

Metric anchors are for the ReBot scene. On other robots read the same ratios visually:
slack against the object's size and against the finger pad width.

### Where the commit is, per verb

The verb only says which slack to measure. The object and the target set the number;
those calls live in the table below, keyed by verb + object + target.

- **grasp**: the pads on the object at closing, over the axis the object constrains.
- **release**: the target's free opening minus the object's footprint, at the height the
  object is let go.
- **insert / place on fixture / press / turn / rotate a control**: the fit or the control
  face against what enters it.
- **open / close / pull / push** (drawer, door, lid): the handle grasp is its own step;
  the motion is judged on where it must stop.
- **wipe / scrub / spread / fold / unfold / straighten / flatten / stir / hold / pour /
  water / dump**: the area or opening the action has to stay over.

Three values are derived, not read:

- **move**: slack of its **end pose**, which is where the next step starts. One level
  below the next step (floor 1): the release or insert does the final alignment, the move
  only has to arrive within reach of it. Read `to-30` to check where it actually ended.
- **return to home**: always 1.
- **lift**: out of a fitted fixture, the fit's level minus 1; off a free surface, 1.

### Reference values (priors; the frames override, with a note)

| step class | precision | why |
|---|---|---|
| grasp the sock / shirt / cloth / towel / rag | 1 | any fold captures |
| move the sock / shirt to the basket / bin | 1 | wide hover region |
| release the sock / shirt in the basket / bin | 1 | 30 cm+ rim, deformable |
| grasp the cup / bottle / pill bottle / spray bottle / tape roll / apple / block / mug | 3 | straddle a rigid object, free yaw |
| release the cup / bottle in the basket or bin | 1-2 | rim minus object > 10 cm, 2 if the container is already crowded |
| grasp the bit (bits-box) | 5 | 6 mm wide bit lying flat: 4, +1 pads across its width |
| move the bit to the box | 4 | one below the release |
| release the bit in the slot | 5 | compartment barely wider than the bit: 4, +1 aligned with the slot |
| return to home | 1 | joint-space target |
| release the rag / watering can / any object on the table, desk or tray | 1 | open surface |
| lean the pillow against the wall (release) | 2 | 3-10 cm band |
| dump the strawberries on the cutting board | 2 | board-sized target; "gently" is speed/quality, not precision |
| press the button (corpus) | 3 | 2 cm face |
| turn on the lamp / switch | 3-4 | switch size |
| grasp the watering can; water the plant | 3; 2 | handle; spout over a 10 cm pot |
| move the rag to the desk; wipe the desk | 1; 1 | area |
| move the paper to the shredder; release in the shredder | 4; 5 | 1 cm slot, edge-on |
| grasp the flower; move to the vase; release in the vase | 3; 3; 4 | stem into a 3 cm mouth stem-first |
| move the cup to the rack; release on the rack | 2; 3 | rack slot a few cm wider |
| grasp the object; lift; move to the board; insert (FMB) | 3; 4; 4; 5 | fitted fixture; mating fit |
| place the object on the fixture (FMB) | 4 | fixture fit |
| fold / unfold the cloth | 1 | area |
| grasp the drawer handle; pull the drawer | 3; 1 | handle; free motion |

## Not precision

- **How the demonstration went.** Retries, a wandering search, hesitation, overshoot,
  slips: all of that is quality (3 for laboured, 1-2 with a failure event) or a mistake
  row. Precision is read from the object and the target, never from the arm's path.
  Two episodes of the same step in the same scene carry the same precision whatever
  their quality.
- **Tempo.** A fine step is usually slow, and speed already measures that. Do not raise
  precision because the operator went slowly, or lower it because they were quick.
- **Care with a fragile object.** "Gently" is a speed and quality property; it does not
  narrow the target.
- **Difficulty from occlusion or reach.** A target hidden from the top camera, or far
  from the base, does not change its slack.

## Procedure

Precision is a class property: the same subtask string in the same scene carries the
same value. Annotate the class, then verify, so the image budget is per class and not
per segment.

1. **Group** segments by subtask text (ReBot ~60 strings, diverse 445 atoms). The
   prior from the table is the hypothesis, never the verdict: a name does not fix a size
   (a sugar cube, a microcontroller, "the object", "the block"), so every class gets a
   vision read.
2. **One vision read per class.** One representative segment, two images at the
   **commit frame**: wrist and top. The commit frame is verb-chosen, not mid-segment:

   | verb | commit frame | why |
   |---|---|---|
   | grasp | `to-15` | pads on the object, starting to lift; the segment start shows the previous container (grasp absorbs the transit), the middle shows the approach |
   | release | the frame the gripper opens (read off the gripper channel of the state) | the object at the height it is let go; the same frame contact strategy reads, so one read serves both channels (2026-09-24, see `contact_strategy_rubric.md`) |
   | move, insert, place, press, turn | `to-1` | the end pose against the target |
   | everything else | `from+15` and `to-15` | the area or opening the action stays over |

   The annotator returns JSON in evidence-first order, so the number conditions on what
   was articulated (same device as the mistake pass): `object_vs_pads`,
   `target_opening_vs_object`, `orientation_pin`, then `precision`, `note`.
3. **Per segment** only for classes the read marks scene-sensitive: a container that
   fills, a slot with one object's room left, placeholder objects, and any class that
   fails step 4. Same two-image read on each segment.
4. **Verify.** A random sample of segments per class gets the same read. A class whose
   sample disagrees with the class value is sent whole to step 3.

| scheme | images |
|---|---|
| 3 frames x 2 cameras x every segment | ~36,000 |
| one read per class + sampled checks | ~1,500 |

What each image is read for:

| image | read |
|---|---|
| wrist at the commit | object size against the finger pads; rigid or deformable; whether the pads must land in a specific orientation; nearest neighbour within a pad width; for release, the free opening against the object's footprint, already-placed objects narrowing it |
| top at the commit | where the arm stopped relative to the target; the target's remaining room; clutter around the object |

Consistency rule: within one dataset the same (verb, object, target) triple maps to one
value unless a note explains the scene difference. Spot-check by grouping the output by
subtask text and listing the groups with more than one value.

## Coverage, not calibration

This channel is descriptive, so there is no target distribution to hit. What matters is
coverage: the model can only learn a level it has seen. Report the frame histogram per
half after the pass. Expected on today's data: ReBot 1 dominates (clothes), 3 from the
cup/bottle sessions, 4-5 only from bits-box; diverse contributes 5 through FMB inserts
and the shredder, 3-4 through flowers, buttons and racks. If level 5 is under a few
percent of frames the label for "5" rests on two sources, which is the argument for more
bits-like sessions rather than a reason to relabel.

## Output

- `meta/precision.parquet`: `episode_index, segment_index, from_index, to_index, subtask,
  precision, orientation_pin, prior, note`
- `precision_atoms.jsonl`: same fields keyed by `(episode_id, atom_index)`
- `meta/precision_info.json` / sidecar info: rubric version, annotator, the definition
  and the "not precision" list verbatim

Written to a new dataset directory, never in place.

## Inference

Quality, mistake and speed are constants at rollout (5, none, 5). Precision is not: it is
the true value of the **current step**, so the deployment path needs a per-step source.
Two options, undecided: a per-task table keyed by step class (the same priors as above,
rendered when the frozen subtask head emits the step), or a per-task constant equal to
the task's maximum step level. The probe question is whether the clause moves the chunk
at fixed speed (metadata_steering ramp, n_seeds >= 3), not whether it slows the arm.
