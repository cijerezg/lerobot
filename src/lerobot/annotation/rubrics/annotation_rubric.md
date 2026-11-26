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
| quality | segment | high, or nothing | signals screen, frames decide (section 5) |
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
  `ring`, `screwdriver`, `hex key`, `yellow plug`, `red plug`, `blue plug`.
- **Place** = where the operator puts it, as seen in the frames: `to cup`, `to case` (screwdriver case),
  `to peg` (the holder the ring / nut goes on), `with the other yellow plugs` (its compartment of the box),
  `to the top left drawer`. Example texts: "approach die", "grasp die", "move die to cup", "release die",
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
- **Timing:** a mistake is a span of frames, not a whole segment. It is anchored on the event frame (the fingers
  finish closing on nothing; the torque drops while the gripper is commanded shut; the object leaves the fingers). It
  starts where the motion that causes it starts, at most 1 s before the event, and ends at the frame the result is
  visible (fingers reopened and the object still on the table; the object at rest). Typically 1-2 s. Half-open, in
  dataset frames. Frames outside the span carry no mistake.
- **Not a mistake:** a hard squeeze that holds the object; hovering, a slow or indirect approach, searching; a
  reposition; picking a different object than expected.

## 5. Quality

One label, `high`, per segment: it covers every frame of the segment, from its first to its last. A segment is either
high or carries no quality label.

High = the right motion, done relatively smoothly. It does not have to be perfect. Neither rare nor the default: a
clean operator gets it on roughly half of the segments.

**Immediately high:** a move and its release that end with the object seated on a peg or in a hole (nut on the peg,
tube in the peg), whatever the path looked like, unless the object is lost on the way.

**Otherwise, two parts, both must hold.**

1. Smooth, measured on the fingertip path by `rebot/tip_motion.py` (forward kinematics of the recorded joints):

| measure | approach, move | grasp | release |
|---|---|---|---|
| directness (straight line / path length) | at least 0.6 | - | - |
| backs off (largest rise of the distance to where the tip ends) | at most 0.5 cm | at most 0.3 cm | - |
| longest pause (below 1 cm/s) | at most 0.3 s | at most 0.5 s | - |
| turns of more than 2 mm within 1 s (shaking) | at most 2 | at most 2 | at most 2 |

2. The right motion, read off the frames:

- it goes to the object or place it ends at: no search over several objects, no aborted descent, no change of slot;
- grasp: the first close holds, and the object is not lost in the carry that follows;
- release: the object lands at its place and stays;
- no mistake span (section 4) overlaps the segment.

- A retry can be high: the grasp after a failed close is judged on its own.
- How the object sits in the fingers (end grip, off centre) does not matter if it is held.
- The thresholds were set on the 65 segments of the 2026-11-26 pilot and are provisional.
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

1. Boundaries proposed from the gripper signal (holds), classes filled in by rule.
2. Review on the clips (`rebot/subtask_labeler`): fix boundaries, add reposition and return home, mark failed grasps.
3. Mistakes, then quality.
4. Computed channels: speed, tip target.
5. Print the root and run `verify_root`.

## 8. Open

| item | pending |
|---|---|
| thresholds of the quality screen (section 5) | provisional, from the pilot; no shaking seen yet, so the turns limit is untested |
| prompt wording of the quality, mistake and speed clauses | with the trainer change |
| gripper torque levels ("N of 8") | bounds were fit on the tube roots (max about 0.5 N*m); recompute on this data |
