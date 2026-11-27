# Annotation rubric (2026-11-26)

> **Status:** in force since 2026-11-26 for every new ReBot pass, starting with
> `outputs/data/rebot_teleop_2026-11-25-cut-v2` and `outputs/data/rebot_teleop_2026-11-26-small-objects-cut-v1`.
> It replaces `archive/annotation_principles.md`, `archive/quality_mistake_rubric_v2.md`, `archive/precision_rubric.md`
> and `archive/contact_strategy_rubric.md`. Roots annotated before this date keep the labels they were printed with.

Why it changed: the 1-5 quality scale, precision and contact strategy did not steer the policy. Fewer channels, each one either computed from the signals or a yes / no
read off the frames.

## 1. What is labelled

| channel | per | values | source |
|---|---|---|---|
| subtask | segment | section 2 | gripper signal proposes the boundaries, frames decide |
| speed | segment | slow / normal / fast | computed (section 3) |
| mistake | span | type (section 4) | frames + signals |
| quality | segment | high, or nothing | frames decide, tip path as support (section 5) |
| tip target | frame | offset to the next gripper event | computed (section 6) |

Not labelled any more: precision, contact strategy, control mode, quality grades 1-4, quality spans and their causes.

## 2. Subtask

The text names the object and, for a move, where it goes. Short plain nouns, the same word for the same thing in
every episode (list below; a new object or place gets a new line there).

| class | text | from | to |
|---|---|---|---|
| approach | approach `<object>` | end of the previous segment | 2.5 s before the hold starts |
| grasp | grasp `<object>` | 2.5 s before the hold starts | the hold starts (object gripped) |
| move | move `<object>` to `<place>` | the hold starts | the object is at its place, about 1 s before the gripper opens |
| release | release `<object>` | end of move | 1 s after the gripper opens, fingers clear |
| reposition | reposition `<object>` | the fingers touch the object | they leave it |
| return home | return home | end of the last release | the arm rests |

- **Object** = what it is, with the colour only when the scene holds the same thing in several colours: `die`, `nut`,
  `ring`, `screwdriver`, `hex key`, `yellow plug`, `red plug`, `blue plug`, `bit` (screwdriver bit), `bottle` (black spray bottle), `cup` (when the cup itself is picked up).
- **Place** = where the operator puts it, as seen in the frames: `to cup`, `to case` (screwdriver case),
  `to peg` (the holder the ring / nut goes on, also when the nut ends in a hole of its base), `with the other yellow
  plugs` / `red plugs` / `blue plugs` (its compartment of the box), `to the top left drawer`, `on the other cup` (a cup stacked on a cup), `to box` (nuts dropped
  into the clear parts box), `to basket` (the wicker basket). Example texts: "approach die", "grasp die", "move die to cup", "release die",
  "move yellow plug with the other yellow plugs", "move nut to peg".
- The place is where the object ends up in that segment. Fitting or re-seating the object at its place is part of the
  move.
- A segment shorter than 1 s between a failed grasp and its retry is merged into the retry grasp.
- **Hold starts** = the first frame the gripper stops closing on the object. Read it from the gripper torque rise
  (the angle is clipped at 0 on thin objects).
- **Grasp is shorter than 2.5 s** when the previous segment ended later than that; it then starts where that one ended.
- **Reposition** = the object is pushed, nudged or dragged into a better pose without being carried. A second close
  on the same object after letting go is a new approach + grasp, not a reposition.
- **Failed grasp** (nothing held after the close): the segment stays `grasp <object>` and gets a mistake (section 4);
  it runs from 2.5 s before the fingers shut to the frame they reopen. What follows is `approach <object>`.
- **The tube episode** (episode 0 of `rebot_teleop_2026-11-25-cut-v2`, the only one) keeps the tube texts of the
  2026-11-14 pass (approach tube / grasp tube / move tube to peg / insert tube in peg, with the side naming of that
  pass).
- Segments tile the episode with no gap. Idle stretches and aimless motion are cut before labelling, as before; an
  episode ends when its last useful step is done.
- The task string stays the operator's short goal ("tidy up the table").

## 3. Speed

Computed after the subtask pass, never judged by eye.

- Measure: mean fingertip speed over the segment, cm/s, from forward kinematics of the recorded joints.
- Levels: slow / normal / fast, cut at the terciles of that measure **within the class**. "Fast grasp" means fast for a
  grasp.
- The cut points are computed once on the two roots named above and frozen in a file next to the labels; roots added
  later are labelled with the same cuts.
- A class with fewer than 30 segments has no cuts yet and gets no speed label.
- The older speed scale (`speed_hybrid_v1`, fixed to the earlier training data) is not used for these roots.

Basis (2026-11-26, phases taken from the gripper torque, 194 grasp cycles): within a class the mean speed spreads about
3x between the 10th and 90th percentile (approach 3.1-8.7 cm/s, grasp 1.2-3.7, move 2.7-10.9), 2.5-3x is left after
path length is taken out, and a fast stretch is followed by a fast one (rank correlation 0.55 approach, 0.71 move).

## 4. Mistake

| type | event |
|---|---|
| `failed_close` | the gripper closes on or at the object and the object is not held when the arm lifts or the gripper reopens |
| `ejected` | the gripper closes on the object and the object shoots out of the fingers |
| `slip` | the object leaves a closed gripper during a lift or a carry |
| `drop` | the gripper opens and the object comes to rest outside where it was being taken |
| `knock` | the arm or the held object moves another object or a container, and it stays moved |
| `other` | a failure that passes both tests below and fits no row above; say what in the note |

- **Two tests, both must hold:** you can point at the frame where it happens, and you can see the result.
- **Timing:** a mistake covers the whole mistaken motion, not the instant it shows. It starts where the motion that
  goes wrong starts (a missed grasp: the frame the gripper starts closing with the arm badly placed, staged closes
  included; a slip: the close of that grasp; a knock: the start of the sweep that hits) and ends at the event frame
  (the fingers are shut on nothing; the torque drops while the gripper is commanded shut; the object leaves the
  fingers). The recovery after the event (reopening, re-aiming) carries no mistake.
- **Look-ahead:** the policy predicts the next 1 s of actions, so a frame's label describes its next 1 s. When the
  labels are printed, the mistake starts 1 s before the annotated start (a fixed offset; the annotator marks the
  motion itself). Boundaries cannot be exact; when in doubt, start the mistake earlier, not later.
- **Not a mistake:** a half close that reopens before the fingers shut or touch the object (it stays inside the
  approach, which is then not high); a hard squeeze that holds the object; hovering, a slow or indirect approach, searching; a
  reposition; picking a different object than expected.

## 5. Quality

One label, `high`, per segment: it is judged on the whole motion and covers the segment from its first frame. A
segment is either high or carries no quality label.

**Look-ahead** (same reason as section 4): a frame is high only when its next 1 s is high. When the labels are
printed, high is dropped on the last 1 s of a high segment that is followed by a segment that is not high, and on
every frame that carries a mistake.

High = decisive and on target: the motion you would want the policy to copy. It does not have to be perfect, and it
is neither rare nor the default. The annotator judges it from the frames, wrist view first.

**Immediately high:** a move and its release that end with the object seated on a peg or in a hole (nut on the peg,
tube in the peg), whatever the path looked like, unless the object is lost on the way.

**High:**

- decisive: the arm goes to the object or place in one go and commits; it does not stop to re-aim;
- on target: in the wrist view the object (in a move: the place) sits near the centre for most of the segment, and
  the fingers arrive around it. Judged on the frames where it is in view: in an approach the object, and in a carry
  the place, comes in from the edge of the frame, and what counts is that it then comes straight to the fingers.
  Over a wide container (cup, drawer) anywhere inside the rim is on target;
- grasp: the first close holds, and the object is not lost in the carry that follows;
- release: the object lands at its place and stays.

**Not high:**

- missing a little: the fingers arrive beside the object or place and shift over, before or during the close;
- sideways motion: the arm drifts off its line and comes back, or slides sideways near the object or place;
- a search over several objects, an aborted descent, a change of slot, a stop to think;
- a mistake (section 4) overlaps the segment.

**Support, not a gate:** the fingertip path measured by `rebot/tip_motion.py` (forward kinematics of the recorded
joints). Typical values of a high segment are below; the annotator can decide against them either way and says why.

| measure | approach, move | grasp | release |
|---|---|---|---|
| directness (straight line / path length) | at least 0.6 | - | - |
| backs off (largest rise of the distance to where the tip ends) | at most 0.5 cm | at most 0.3 cm | - |
| longest pause (below 1 cm/s) | at most 0.3 s | at most 0.5 s | - |
| turns of more than 2 mm within 1 s (shaking) | at most 2 | at most 2 | at most 2 |

A release is judged with the move it ends: it is high only when that move is high.

- A retry can be high: the grasp after a failed close is judged on its own.
- How the object sits in the fingers (end grip, off centre) does not matter if it is held.
- The typical values come from the 65 segments of the 2026-11-26 pilot.
- Stored in the existing `quality` column as 5 for high and -1 otherwise.

## 6. Tip target

As built in `lerobot/adhocs/tip_offset` (2026-11-14): for every frame, the vector from the fingertip now to the
fingertip at the next gripper event of the episode, in the current tool frame.

- Events: the hold start of each grasp segment, and the frame the gripper starts opening in each release (or insert)
  segment.
- The event frame comes from the gripper torque where the angle is clipped (thin objects).
- A grasp with a `failed_close` or `ejected` mistake is not an event: the frames that would aim at it have no target.
- Frames after the last event of an episode have no target.

## 7. Order of work

`<work>` = the results folder of the pass; commands and file formats are in the README ("A pass under
annotation_rubric.md").

1. `rebot/window_prepare`: per window (a stretch of an episode) sheets, signals and the holds from the gripper torque.
2. One agent per window (`rebot/WINDOW_BRIEF.md`): segments, texts, mistakes, quality. The agents are Opus 5.5
   (`model: opus`), the same model for every window of a pass, about 8 running at a time.
3. `rebot/window_assemble`: fixed quality rules, tip path measures, speed, per-frame labels with the 1 s look-ahead.
4. `rebot/window_render`: clips with the labels burned in; the user reviews those, and the feedback goes into this
   rubric and the brief.
5. Tip target, print the root, `verify_root` (not built for this rubric yet).

## 8. Open

| item | pending |
|---|---|
| typical tip values of a high segment (section 5) | from the pilot; no shaking seen yet, so the turns value is untested |
| prompt wording of the quality, mistake and speed clauses | with the trainer change |
| gripper torque levels ("N of 8") | bounds were fit on the tube roots (max about 0.5 N*m); recompute on this data |
