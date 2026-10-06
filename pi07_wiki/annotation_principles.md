# Annotation principles (2026-10-05)

Why this page exists: the 2026-10-05 steering probe showed the precision and contact clauses moving the action chunk by
0.04-0.05 of the flow-seed floor, the subtask labels of the bit kit and bits episodes do not tell apart targets that look
alike, and the quality grades do not survive a look at the frames. The labels were written to fill the tables, not to tell
the robot what to do. These principles apply to every annotation pass from now on, starting with
`outputs/rebot_rollouts_2026-10-05-cut-v1`. The per-channel rubrics (`quality_mistake_rubric_v2.md`, `precision_rubric.md`,
`contact_strategy_rubric.md`) stay in force where they do not contradict this page. Annotation docs live in this wiki, not in
`migration/`.

## 1. The one test

A label is written only when it removes an ambiguity the robot faces at that frame. Before writing any label ask: if the
robot had only this text and the images, would it know what to do next? If the answer does not change with the label, the
label is noise and the time goes to the ambiguity that is actually there.

## 2. Subtask text

The subtask is the instruction the robot needs at that moment. It must say, whenever the scene allows it:

- **Which object**, by something visible that separates it from its look-alikes: a feature ("the flap with the card"), a
  position relative to a landmark ("the bit nearest the basket", "the bit under the gripper"), or a part ("the top-left
  corner of the towel"). Two targets in the same scene must never share a subtask text.
- **In which order**, when order matters: "fold the card flap over the bits" and only then "fold the bit flap over the card
  flap". Identical text for the first and second step of an ordered sequence is rejected.
- **Where it goes**: the receiver or the direction, for every move, release, push, pour and fold. "Push the book off the
  box" is not enough; "push the book off the box, away from the basket" is. A move names its container; a fold names what
  the part lands on; a pour names the vessel.

Some frames cannot be resolved (two identical bits with no landmark, a push with no reference). Then write the best available
reference and flag the segment `ambiguous` in the note, so the gap is known rather than hidden behind a confident label.

| rejected | accepted |
|---|---|
| grasp the bit kit cover / grasp the bit kit cover | fold the card flap over the bits / fold the bit flap over the card flap |
| grasp the bit | grasp the bit lying beside the component box |
| grasp the towel | grasp the top-left corner of the towel |
| push the book off the box | push the book off the box toward the table edge |
| move the bit to the component box | move the bit to the component box, open compartment |

The text describes what the arm actually does in the kept frames, never the prompt (unchanged rule).

## 3. Per-step channels: precision, contact

Read from the wrist view at the commit frame, never from a text rule on the subtask noun. A clause that is a function of
the subtask text carries no information beyond the subtask, so the model cannot learn anything from it; that is what the
probe measured. Consequences:

- The contact element is what the gripper did (a flap pushed over with open fingers is a push or a pry, not a cloth pinch,
  whatever the verb in the text).
- Two steps with the same subtask text, the same contact and the same precision in different situations mean the text is
  under-specified. Fix the text, do not accept the match.
- The text rules in `contact_strategy_rubric.md` remain the prior for external data nobody will look at; on our own data they
  are a first guess that the frame read overrides.

## 4. Quality

Graded per class on strips of several episodes aligned at the commit, never one segment alone (rule already decided
2026-09-29). The grade must point at something visible in the frames; a grade without a frame to show is not written.
Idle and near-idle stretches are cut before grading, not graded.

## 5. Passes

1. Subtask text: boundaries + text under section 2; every ambiguous text is either fixed or flagged `ambiguous`.
2. Per-step channels from the frames (section 3), then quality on strips (section 4).

Pass 2 never starts on a segment whose text pass 1 rejected.
