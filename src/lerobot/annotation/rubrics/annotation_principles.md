# Annotation principles (2026-10-05)

> **Status:** in force since 2026-10-05 for every annotation pass.

Why this page exists: the 2026-10-05 steering probe showed the precision and contact clauses moving the action chunk by
0.04-0.05 of the flow-seed floor, the subtask labels of the bit kit and bits episodes do not tell apart targets that look
alike, and the quality grades do not survive a look at the frames. The labels were written to fill the tables, not to tell
the robot what to do. These principles apply to every annotation pass from now on, starting with
`outputs/rebot_rollouts_2026-10-05-cut-v1`. The per-channel rubrics (`quality_mistake_rubric_v2.md`, `precision_rubric.md`,
`contact_strategy_rubric.md`) stay in force where they do not contradict this page. Rubrics and annotation scripts live in
`src/lerobot/annotation/` (see its `README.md`), not in `migration/`.

## 1. The one test

A label is written only when it removes an ambiguity the robot faces at that frame. Before writing any label ask: if the
robot had only this text and the images, would it know what to do next? If the answer does not change with the label, the
label is noise and the time goes to the ambiguity that is actually there.

## 2. Subtask text

The subtask is the **least** text that tells the robot what to do next in this scene (user 2026-10-06: "the point of
subtask is to provide the minimal info that enables the task"). Start from the bare verb and noun, "grasp the bit", "move
the cup to the basket", and add a word only when it removes an ambiguity the robot actually has at that frame:

- **Which object**, only when the scene holds a look-alike. One bit on the table: "grasp the bit". Two bits: "grasp the bit
  next to the basket". Tell them apart by something visible: a feature ("the flap with the white card"), a position next
  to another object ("the bit nearest the basket"), or a part ("the top-left corner of the towel"). Two targets in the
  same scene must never share a text. An object already put in its container is out of play, not a look-alike: once the
  white sock is in the basket, the black sock left on the table is "grasp the sock" (audit 2026-10-07).
- **In which order**, when the same step repeats on different parts: "fold the flap with the white card over the bits"
  and only then "fold the flap with the bits over the white card".
- **Where it goes**, for every move, release, push, pour and fold: the receiver or the direction, under the same rule.
  "the basket" when there is one basket; "push the book off the box toward the table edge" because "off the box" leaves
  the direction open.

**Plain words** (user 2026-10-06). Short, simple sentences. No coined names ("card flap", "middle bit panel"): someone who
sees the image for the first time must know which thing each word means. A colour, a side or a position is a tool for
telling two things apart, not a description: "the white cup" when it is the only cup is noise. The same object grasped
again (after a drop or a slip) keeps the same text when it is still the only one of its kind.

Some frames cannot be resolved (two identical bits with no landmark, a push with no reference). Then write the best available
reference and flag the segment `ambiguous` in the note, so the gap is known rather than hidden behind a confident label.

| rejected | accepted |
|---|---|
| grasp the bit kit cover / grasp the bit kit cover | fold the flap with the white card over the bits / fold the flap with the bits over the white card |
| grasp the outer edge of the card flap of the bit kit | grasp the outer edge of the flap with the white card |
| grasp the bit (two bits on the table) | grasp the bit next to the component box |
| grasp the bit lying between the basket and the robot base (the only bit) | grasp the bit |
| move the black spray bottle to the brown basket (one bottle, one basket) | move the spray bottle to the basket |
| grasp the clear tape roll lying beside the right side of the brown basket (the roll again, after a slip) | grasp the tape roll |
| grasp the towel | grasp the top-left corner of the towel |
| push the book off the box | push the book off the box toward the table edge |
| grasp the black sock (three black socks on the table) | grasp the black sock nearest the basket |
| grasp and align the first bar / align the second bar | grasp the black aluminium bar under the gripper / release the black aluminium bar beside the parallel bars |
| twist the cap off the pill bottle (the wrist turns 3 deg, the cap is pulled off) | take the cap off the pill bottle |

The text describes what the arm actually does in the kept frames, never the prompt (unchanged rule).

## 3. Per-step channels: precision, contact

Read from the wrist view and top view at the commit frame, never from a text rule on the subtask noun. A clause that is a function of
the subtask text carries no information beyond the subtask, so the model cannot learn anything from it; that is what the
probe measured. Consequences:

- The contact element is what the gripper did (a flap pushed over with open fingers is a push or a pry, not a cloth pinch,
  whatever the verb in the text).
- Two steps with the same subtask text, the same contact and the same precision in different situations: check that the
  contact and precision were read from the frames. Change the text only when the scene has a look-alike it fails to
  separate (section 2).
- The text rules in `contact_strategy_rubric.md` remain the prior for external data nobody will look at; on our own data they
  are a first guess that the frame read overrides.

## 4. Quality

Graded per class on strips of several episodes aligned at the commit, never one segment alone (rule already decided
2026-09-29). The grade must point at something visible in the frames; a grade without a frame to show is not written.
Idle and near-idle stretches are cut before grading, not graded.

Calibration from the 2026-10-06 review (cases in `quality_mistake_rubric_v2.md` section 8.6):

- **4 is motion that goes straight to its goal.** An approach that is awkward, hesitant or indirect (it creeps with
  corrections, overshoots and comes back, hovers) is a 3, and a 2 where it drags on. A hesitation in the middle of a
  carry is a 3. Do not leave these at 4 because the arm is still making progress.
- **A clean, direct action is a 5.** Clean actions are not the norm, which is why they are marked. A short settle before
  the lift does not rule it out.
- **An episode ends when its last useful step is done.** What the arm does after that with no purpose (wandering, a
  step that was asked for and never attempted, a slow drift) is trimmed from the kept range. It is not labelled
  `return to home` and graded low.
- **Useful data, not perfect labels** (user 2026-10-06). Aimless motion (away from any object, after the last useful
  step) is cut. Hovering or searching at the object, confused but still trying (the wrist turning to find it), is kept
  and graded (3, 2 where it drags): it shows the model how to find the object. Complete, purposeful manipulation
  after the task is done (the piece picked up again and set down elsewhere) is kept and labelled for what it does
  ("grasp the black piece on the table"); only motion that touches nothing is cut (2026-10-08, diverse screen).
- **Judge the flow, not only the commit.** Hesitation and indirect approaches show on video and on dense strips over
  the whole segment; a coarse strip around the commit hides them.
- **Read directness off the joints, not the stills** (user 2026-10-06). The approach line (`quality_mistake_rubric_v2.md`
  5.13: distance of the fingertips to where they end up, from the joint angles) falls steadily on a direct approach,
  rises again on an overshoot and goes flat on a hover. A steady fall is 4 or 5, however slow and whatever the stills
  suggest.

## 5. Passes

1. Subtask text: boundaries + text under section 2; every ambiguous text is either fixed or flagged `ambiguous`.
2. Per-step channels from the frames (section 3), then quality on strips (section 4).

Pass 2 never starts on a segment whose text pass 1 rejected.
