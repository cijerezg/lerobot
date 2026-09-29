# Annotation rubric v2: quality spans, mistakes, precision windows (draft 5, 2026-09-29)

Replaces the quality and mistake sections of `annotation_rubric.md` (2026-08-02) and the
"Units" section of `precision_rubric.md`. Applies to every dataset: all ReBot roots and the
diverse corpus (DROID, MolmoAct, RoboChallenge, UR7e, YAM, FMB). The loader does not read
the new tables yet.

Draft 5 (after the step 2 pilot, 290 grasps): there is no segment-wide grade. Every frame is
4 by default, only an exceptionally clean action is 5, and critiques are stretches graded
3, 2 or 1 (user decision 2026-09-29: "everything starts at a 4, only exceptional clean
actions go to 5, critiques go down to 3, 2, 1"). Pilot files:
`migration/annotation_v2_2026-09-29/pilot/`.

Built from 70 cases read on strips and traces (20 mistakes and 50 others), two
side-by-side sheets and an idle count (`migration/quality_span_2026-09-29/`). Section 8
lists the cases.

## 1. How to use this rubric

There is no template that decides a case. The tables below give definitions, typical
values and worked cases. The annotator looks at the frames and decides.

- **Numbers start a look; they never end one.** A long slow run in the trace says "look
  here". What you see decides. In `hover_all_ep37` the trace measured a 9.2 s hover; the
  strips show a steady descent with a real hover of about 1 s.
- **Grade by comparison.** Put several instances of the same step side by side before
  grading any of them (section 5.5). What separates a 5 from a 4 is clear on the sheet
  and hard to write down.
- **Look again when unsure.** Coarse strips first, dense strips next, video last. Going
  back two or three times on one case is normal.
- **Say what you saw before you grade.** Write the description first, the grade second.
- **Say when you are not sure.** Every row carries `confidence` (sure / unsure). An
  unsure row gets a second reader. Do not guess to avoid the flag.
- **A case that fits nothing here** gets your best judgement, a note, and a line in the
  case library so the next annotator has it.

## 2. What changed from v1

| | v1 | v2 |
|---|---|---|
| quality | one grade per subtask segment | 4 by default on every frame; stretches graded 5 (exemplary) or 3, 2, 1 (critiques); nothing segment-wide |
| strategy | not judged | a worse grip point is a `strategy` stretch graded 3 over the final approach; the best one, cleanly done, is an `exemplary` stretch graded 5 |
| spread | most clean segments on one grade | 5 is rare by design; the sheets per class set what is 5 and what is 3 |
| idle | not labelled | any stretch with no motion for over 1 s is grade 1; nothing is cut |
| quality 1-2 | meant "the segment contains a mistake" | means "this stretch is poor", with or without a mistake |
| mistake | started in the committed descent | the event only |
| precision | one level per segment | the level applies near the commit; 1 elsewhere |
| wrong target | a mistake type | not a mistake (section 4.3) |

Measured on the four ReBot training roots (2,557 segments, 598,167 frames):

| fact | value |
|---|---|
| q1-q2 segments that contain a mistake | 144 of 144 |
| q1-q2 frames that lie inside a mistake span | 22.8 % |
| mistake span start, relative to the gripper starting to close | 0.8 s late (median) |

## 3. The three channels

| channel | question | values |
|---|---|---|
| mistake | Did a discrete failure event happen here? | yes / no |
| quality | Is this the motion we want the policy to produce? | 1 (worst) to 5 (best) |
| precision | How tight is the commit the arm is working on right now? | 1 (coarse) to 5 (fine) |

1. A mistake is an event with a visible outcome. Approach, hovering, searching,
   regrasping and hesitation are never mistakes. They are low quality.
2. Every mistake lies inside a quality span of grade 1 or 2. A span of grade 1 or 2 does
   not need a mistake.
3. Precision is read from the object and the target, never from how the arm moved.

## 4. Mistakes

### 4.1 Types

| type | event |
|---|---|
| `failed_close` | the gripper closes on or at the object, and the object is not held when the arm lifts or the gripper reopens |
| `slip` | the object leaves a closed gripper during a lift or a carry |
| `drop` | the gripper opens and the object comes to rest outside the target |
| `knock` | the arm or the held object moves another object or the container, and it stays moved |
| `spill` | contents leave a held container outside the target (pour, scoop) |

### 4.2 The two tests

Both must hold.

| test | question |
|---|---|
| event | Can you point at the frame where it happens? |
| outcome | Can you see the result (object still on the table, object on the floor, cup on its side)? |

If you cannot see the outcome on the coarse strip, get the dense strip. If it is still
not visible, it is not a mistake: mark a quality span and set `confidence` to unsure.

### 4.3 Not a mistake

| what | label instead |
|---|---|
| hovering, slow approach, searching | quality span |
| close, hold, let go where it lay, close again from a better angle (`lowq_droid_ep000578`) | quality span `regrasp` |
| a partial close to line up the fingers, reopened before contact | nothing, or `hover` if it is long |
| nudging or dragging the object into a better pose | quality span `reposition` |
| cloth work: grip, pull, let go, repeated (`lowq_droid_success_ep023479`) | nothing; it is the task |
| wrong object grasped, wrong container, wrong button (`mistake_robochallenge_ep000404`) | the subtask text names what was done; no mistake, no quality penalty for the choice |
| placement on the target but untidy (shirt bunched on one side of the box) | quality span `poor_place` |
| a shirt taken to the bin in several carries | nothing |

### 4.4 Span

Half-open, $[m_a, m_b)$, in frames of the dataset's own rate.

| type | start $m_a$ | end $m_b$ |
|---|---|---|
| `failed_close` | the gripper starts to close | 0.3 s after the gripper is open again, or after the arm lifts with nothing held |
| `slip` | the object first moves relative to the fingers | 0.3 s after the object is at rest |
| `drop` | the gripper starts to open | 0.3 s after the object is at rest |
| `knock` | first contact | 0.3 s after the knocked object is at rest |
| `spill` | the first contents leave the lip | 0.3 s after the last contents land |

- Take the gripper start from the trace. Take the outcome from the frames.
- One row per event. v1 rows that cover several closes must be split:
  `mistake_droid_ep013103` is one 9 s row in v1 and is two failed closes.
- A span may cross a subtask boundary.
- Typical length 1 to 4 s. Longer needs a note.

### 4.5 Hard calls

| situation | how to decide |
|---|---|
| the gripper closes fully, reopens, and the object has not moved (`hover_bits_ep7`, graded 4 in v1 with no mistake) | failed close. "Re-closing straight away" does not change it. |
| the gripper closes part way on a rigid object and reopens (`lowq_droid_ep006256`) | dense strip. Object lifted or shifted on purpose: `regrasp`. Object untouched or pushed away: `failed_close`. |
| closes on cloth with fabric between the fingers, cloth not lifted (`q3_ext_ep130`) | look at what the next motion does. Pulls and lets go: cloth work. Lifts away with nothing: `failed_close`. |
| thin object, gripper reads fully shut either way (bits) | the gripper value cannot tell. Use the wrist view after the lift. |
| object hidden by the arm in both views | not a mistake on this evidence. Quality span, unsure. |

## 5. Quality

### 5.1 What a frame's grade means

The policy predicts 1 s of actions from each frame. The grade at frame $t$ describes the
motion in the second after $t$.

$$q(t) = \begin{cases} \min_{i \in C(t)} g_i & \text{if } C(t) \ne \emptyset,\quad C(t) = \{i : g_i \le 3,\ a_i - 1\,\text{s} \le t < b_i + 0.5\,\text{s}\} \\ 5 & \text{else if some exemplary stretch has } a_j \le t < b_j \\ 4 & \text{otherwise} \end{cases}$$

The annotator marks the raw stretch $[a_i, b_i)$ and its grade $g_i$. The writer adds the
headroom to critique stretches (1 s before, 0.5 s after); exemplary stretches get none.
Critiques always win over 5. Do not pad by hand.

### 5.2 Four questions

Ask them in this order for any stretch that looks off.

| # | question | if yes | if no |
|---|---|---|---|
| 1 | **Progress.** Is the view changing toward the goal (object growing in the wrist view, arm nearer the target)? | not a hover, however slow. Slowness is the speed label. | go to 2 |
| 2 | **Purpose.** Does the motion change something useful (object pose, finger alignment, cloth laid flatter)? | `reposition` or task work; grade 3 or better | go to 3 |
| 3 | **Want it.** If the policy did exactly this when asked for quality 5, would that be fine? | no span | mark a span |
| 4 | **Compare.** How does it look next to a clean example of the same step in the same dataset? | sets the grade | |

### 5.3 Default 4, exemplary 5, critiques 3 to 1

There is no segment-wide grade. Quality is local.

| frame grade | when |
|---|---|
| 4 | the default. Ordinary, acceptable motion. A small blemish (a short settle, a slightly slow approach, an object that slides a little in the fingers) is just 4 and needs no stretch. |
| 5 | inside an `exemplary` stretch: the best strategy for the class (its 5 row, 5.4) carried out in one clean continuous motion into the close, object centred, no hesitation or correction. Rare by design. |
| 3, 2, 1 | inside a critique stretch; the cause names the issue (5.7, 5.8) |

| stretch | span |
|---|---|
| `exemplary` (5) | from the start of the final approach (arm lined up on its grip point, heading in) to the lift. Back to the segment start only when the whole motion from there is one clean move. Never over a hover, a correction or a failed close. |
| `strategy` (3) | a worse grip point for the class (its 3 row, 5.4), from the start of the final approach to the lift. The travel before it stays 4. |

There are no grade-4 stretches: anything that only deserves 4 is left unmarked.

### 5.4 Strategy

Strategy is the choice the arm makes: where on the object it grips, from which
direction it comes, where it lets go. Judge it at the commit, from the wrist view.

| question | better | worse |
|---|---|---|
| Where on the object? | the middle of the object; on cloth, a crease or a fold | right at an edge or a tip; on cloth, a flat part |
| Which part of a rigid object? | the part that gives the most stable hold (spray bottle: the body) | a part that is narrow, tapered or top-heavy in the hand (spray bottle: the top) |
| How much margin? | a small error would still catch the object | a small error would miss it or knock it |
| Approach direction? | fingers line up with the object before the descent | wrist twisted late, or the arm reaches across the object |
| Release point? | over the middle of the container, low enough to land inside | at the rim, or from a height where it can bounce out |

Two examples from the user, to show the kind of call. They are not the rule: the
general questions above are. The annotator works out the order for each object family
from its own side-by-side sheet, and the user confirms it.

| object family | 5 | 4 | 3 |
|---|---|---|---|
| sock, shirt, cloth | middle of the item, on a crease | middle, flat part | right by an edge or a tip |
| spray bottle | around the body | | from the top (4 or 3 by how secure it looks) |

Read a class row as: 5 = what qualifies for an `exemplary` stretch (with clean
execution), 4 = the default (no stretch), 3 = a `strategy` stretch.

Pilot rows (confirmed by the user 2026-09-29; full files in
`migration/annotation_v2_2026-09-29/pilot/<class>/strategy/FINAL.md`):

| class | 5 | 4 | 3 |
|---|---|---|---|
| rebot_main sock grasp | middle on a bunch, fold or heel bend, sock centred between the two fingertips, one motion | middle of a flat sock; a pile where other cloth is caught or crowds the fingers | end, tip or cuff; sock at one finger; grasped through a garment on top; reaching across from the far side |
| MolmoAct mug in the dishwasher rack | straight down onto the upturned base, body between the fingers, clear of other dishes | body grip off to one side (mug slides as the fingers close) | handle or rim edge; deep under the upper rack; wedged or barely in view |
| MolmoAct mug upright / on a peg / lying | around the body (upright), body beside the peg in one descent | rim pinch from above; peg grip where the mug rotates off | handle; rim pinch on a lying mug |
| can, jar upright | top-down across the body, one descent | jug or pitcher by the lid (open: 4 or 3) | |

Wrist-view cues from the pilot: on ReBot the fingertips are the dark triangles at the
bottom corners, and the object should sit between them; on MolmoAct the wrist camera is
off-axis, so a held mug sits low and right of centre, which is normal.

- A strategy stretch covers the final approach to the lift, not the whole segment.
- Strategy is judged even when the grasp succeeds. A grip by the edge that holds still
  gets a `strategy` 3 stretch: it is the one that is easier to miss.
- A grip by the edge that then slips gets the `strategy` stretch plus the `attempt` span
  around the slip.
- A failed close earlier in the segment does not affect the strategy call on the final
  close; it has its own `attempt` span.
- Two objects lifted together (a pair of socks, a shirt caught with the sock): other
  cloth in the fingers is the 4 row, no stretch (pilot, 3 of 3 readers).
- A person helps (a hand places the object, the operator makes the close): `other`,
  grade 1, note "human help".
- The contact label (`meta/contact.parquet`) says which strategy was used. Quality says
  whether it was a good one for that object. Use the contact label to sort a sheet.

### 5.5 Side-by-side sheets

Grade a class, never a lone segment. What separates the grades is clear on the sheet
and is left to the annotator's judgement.

| step | what |
|---|---|
| 1 | group segments by class: dataset, verb, object family (sock grasp in rebot_all; bit grasp; ring release on the stack) |
| 2 | build a sheet of 8 to 12 instances, one strip each, stacked, all covering the same time window and aligned at the commit (6 s before the close or opening, 1.5 s after) |
| 3 | rank them by eye before reading any v1 grade: strategy first, execution second |
| 4 | write down the strategies seen in the class and order them (the class's row in the table of 5.4) |
| 5 | pick two references each for 5, 4 and 3; save them in the class's entry of the case library |
| 6 | label every instance of the class against the references: which get an `exemplary` stretch, which a `strategy` stretch, which stay 4 |
| 7 | a class with under 8 instances is compared with the nearest class on the same robot |

One class as an example (rebot_all sock grasps, `side_by_side/grasp_sock_1.jpg`,
`_2.jpg`). Other classes will differ; do not carry these over:

| looks better | looks worse |
|---|---|
| grips the middle of the sock | grips right by an edge (ep4 seg3) |
| one continuous motion into the close | arrives, then waits 2 to 4 s |
| object centred between the fingers | object at the edge of the wrist view |

### 5.6 Spread

A label that is the same almost everywhere teaches nothing, but 5 must stay rare: it is
the example the policy is asked to copy at inference.

| rule | |
|---|---|
| 5 | only exemplary stretches; in the pilot 28 of 204 sock grasps and 18 of 86 mug grasps had one (6 and 10 % of frames) |
| 3 for strategy | every instance with a worse grip point, however well it went (pilot: 66 of 204, 18 of 86) |
| report | per class: grasps with an exemplary stretch, with a strategy stretch, with no stretch; frame share of each grade |
| look again | a class with no exemplary stretch at all, or with every instance on a strategy 3 |

### 5.7 Causes

| cause | what you see |
|---|---|
| `idle` | no motion at all for over 1 s, anywhere (section 5.11) |
| `hover` | the arm is at the object or the target, keeps adjusting, and does not commit |
| `search` | the arm wanders with direction reversals, or passes the object and comes back |
| `stall` | the arm drifts or lingers away from any object, or after a step is done, without stopping dead |
| `hold_still` | the arm slows to a crawl with the object held, mid-carry |
| `aborted_approach` | goes down to the object, comes back up without closing, goes down again |
| `regrasp` | closes, holds, lets go where the object lay, closes again |
| `reposition` | pushes, nudges or drags the object before grasping it |
| `stop_short` | a carry ends short of the container before the gripper opens |
| `weak_grip` | the close lands on the edge or tip of the object |
| `drag_along` | the carry pulls a second object with it |
| `poor_place` | the object ends on the target but untidy or half in |
| `oblivious` | the arm carries on after the object is gone |
| `attempt` | the stretch around a mistake (section 5.10) |
| `strategy` | a worse grip point for the class, over the final approach to the lift (5.3, 5.4); grade 3 |
| `exemplary` | an exceptionally clean action (5.3); grade 5 |
| `other` | anything else, including human help (grade 1); describe it in the note |

### 5.8 Grades

| grade | meaning | typical |
|---|---|---|
| 5 | exemplary (5.3) | best grip point, one clean continuous motion |
| 4 | the default: no stretch | a short extra wait; a small overshoot pulled back |
| 3 | laboured, but it converges and has a purpose; or a worse grip point | reposition that works; a hover past the norm (5.9); a regrasp; `strategy` |
| 2 | poor | a long hover; a search across the table; one mistake |
| 1 | never what we want | idle; very long hover or stall; a chain of mistakes; a mistake with no recovery; oblivious carry |

Things that move a grade, by judgement:

| raises the grade | lowers the grade |
|---|---|
| the step is fine (precision 4 or 5) | the step is coarse (precision 1 or 2) |
| the object is small, thin or cluttered | the arm is already lined up and still waits |
| the wait ends in a clean commit | reversals during the wait |
| the motion changes the object's pose usefully | the same motion repeats with no change |

### 5.9 Typical hover times

Time at the object before the gripper starts to close, clean closes only. Use them to
know what is normal for a dataset, not to grade.

| family | median | 90th percentile |
|---|---|---|
| ReBot clothes and rigid objects | 1.9 s | 6.0 s |
| ReBot small parts (bits) | 6.3 s | 10.3 s |
| external ReBot | 1.2 s | 4.2 s |
| diverse corpus | not measured; take it from the first annotated episodes of each source |

A wait up to the median needs no span. A wait past the 90th
percentile is almost always a span. Between the two, the four questions and the
side-by-side sheet decide. The span covers the part of the wait beyond what is normal.
A segment with no wait at all is a candidate for 5.

The trace measure (arm speed below half its peak) over-reads slow steady approaches.
Confirm every hover on the wrist view: the object should sit in about the same place in
consecutive tiles.

### 5.10 Spans around a mistake

Every mistake gets one `attempt` span that contains it.

| mistake | raw start | raw end |
|---|---|---|
| `failed_close` | start of the final descent that ends in the close; if not visible, 1 s before $m_a$ | $m_b$ |
| `slip` | start of the close that produced the weak grip | $m_b$ |
| `drop` | the frame the carry stops short | $m_b$ |
| `knock` | the frame the arm turns onto the path that hits the object; if not visible, 1 s before $m_a$ | $m_b$ |
| `spill` | the frame the container starts to tilt away from the target | $m_b$ |

- **Oblivious continuation**: extend the end to the frame the arm reacts. Grade 1.
- **Chains**: mistakes less than 10 s apart share one span. Grade 1.
- **Hover before the attempt**: its own `hover` span. Overlaps are fine; the frame takes
  the lower grade.
- **Recovery**: graded like any approach. Direct recovery has no span.
- Grade 2 for one mistake with a recovery; 1 for a chain, no recovery, or oblivious.

### 5.11 Idle

Idle stretches stay in the data (decided 2026-09-29: 34 stretches, 330 s, is too little
to justify cutting and re-caching). They are labelled so they are never trained as good
data.

| rule | |
|---|---|
| idle | in a 5 s window no arm joint moves more than 1 degree in total, and the gripper moves less than 1 % of its span |
| fully still for over 1 s, any length | a span, grade 1, cause `idle`; the span is the whole still stretch except its first 1 s |
| nothing is cut | no trimming, no splitting, no new frame indices |
| little motion (more than 1 degree in 5 s) | stays in the data; graded as `hover`, `stall` or `hold_still` by judgement |
| no allowance | precision, object size and what comes next do not excuse a fully still arm |

For scale, stretches of 5 s or more, by how much motion is allowed in the 5 s window
(`idle_explore.py`, `idle_runs.csv`). MolmoAct stores the end-effector pose, so its
tolerance is the equivalent in position (8.7 mm per degree).

| dataset | total (s) | 0.5 deg | 1 deg | 2 deg | 3 deg | 5 deg |
|---|---|---|---|---|---|---|
| rebot_all | 10,414 | 0 | 0 | 0 | 0 | 0 |
| additions | 871 | 0 | 0 | 0 | 0 | 1 |
| bits + book | 2,338 | 0 | 0 | 0 | 1 | 14 |
| external | 6,316 | 18 | 20 | 22 | 26 | 31 |
| validation | 647 | 0 | 0 | 0 | 0 | 0 |
| DROID | 2,392 | 5 | 5 | 5 | 9 | 11 |
| DROID success | 2,580 | 2 | 2 | 2 | 3 | 4 |
| MolmoAct | 5,011 | 0 | 0 | 0 | 0 | 4 |
| RoboChallenge ARX5 | 5,787 | 1 | 1 | 1 | 2 | 4 |
| RoboChallenge UR5 | 4,238 | 6 | 6 | 7 | 8 | 16 |
| UR7e, YAM | 480 | 0 | 0 | 0 | 0 | 0 |
| FMB | 3,349 | 0 | 0 | 1 | 1 | 4 |
| **all, runs** | | 32 | 34 | 38 | 50 | 89 |
| **all, seconds beyond the first** | | 311 | 330 | 358 | 421 | 643 |

At 1 degree: 5 heads, 18 tails, 11 mid-episode. The count hardly
moves between 0.5 and 2 degrees. From 3 degrees up the extra runs are fine work: bit
grasps with finger nudging, FMB inserts, a plug, a light switch. Those stay.

### 5.12 Diverse corpus

| point | rule |
|---|---|
| rates differ (10, 15, 20, 30 Hz) | think in seconds; store frames at the native rate |
| excluded gaps (`interruption_events`) | a span never crosses a gap; judge only the frames that exist |
| v1 `pause_events` (156, mostly RoboChallenge, inside atoms graded 4 or 5) | each is a candidate `idle` span; confirm on the frames |
| v1 `recovery_events` | not a label in v2; recovery is graded like any motion |
| v1 mistake kinds | `failed_grasp` = `failed_close`; `dropped_object` = `drop`; `spilled_contents` = `spill`; `misplaced_object` = `drop` if off the target, else `poor_place`; `fold_failed` = judge (cloth slipped out: `slip`); `wrong_target` = none |
| contact-rich steps (FMB insert, hang a cup on a peg) | feeling for the hole is the work as long as the part keeps moving. A part lifted clear and brought back is a span. A full stop over 1 s is `idle`. |
| cloth tasks | grip, pull, let go cycles are the work |
| gripper reading | percent closed over the episode; RoboChallenge reads width, so it runs the other way |

## 6. Precision windows

Levels and the class table in `precision_rubric.md` do not change. Only where the level
applies changes.

$$p(t) = \begin{cases} \text{level of the step} & \text{if } w_a - 1\,\text{s} \le t < w_b \\ 1 & \text{otherwise} \end{cases}$$

| bound | definition |
|---|---|
| commit | grasp: the gripper starts its final close. release: the gripper starts to open. insert, place, press: the object seats or the tip touches. |
| $w_a$ | the first frame of the final approach where the gripper is within about 10 cm of the commit pose |
| $w_b$ | 0.5 s after the gripper finishes closing or opening, or after the object is seated |

- The annotator marks $w_a$ by eye: the object (or the target mouth) fills roughly a
  third of the wrist view and stays in it. On ReBot, forward kinematics can propose
  $w_a$; the annotator confirms it.
- The window ignores subtask boundaries: the end of a carry inside the window takes the
  release's level. This replaces "move = next step minus 1".
- Failed attempts and hovers at the object stay inside the window.
- Steps with no single commit (wipe, fold, stir, pour): the level applies for the whole
  time the tool or object is over the work area.
- `return to home`: 1 throughout.

| example | level | window |
|---|---|---|
| sock grasp | 1 | none needed; 1 everywhere |
| bit grasp (`hover_bits_ep7`) | 5 | from the bit between the finger tips in the wrist view (about 12 s before the close) to the close |
| ring on the stack (`q3_ext_ep81`) | 4 | from the ring over the peg to the opening, about 6 s; the carry before it is 1 |
| cup on the rack (`pause_robochallenge_ep001017`) | 4 | from the handle at the peg to the opening |
| FMB insert | 5 | the whole insert, and the last part of the move to the board |

## 7. Procedure

### 7.1 Passes

| pass | material | use |
|---|---|---|
| 0 | side-by-side sheet of the class | learn what normal looks like before grading |
| 1 | trace + coarse strips (8 tiles per 6 s, top over wrist) | find stretches, first description |
| 2 | dense strips (8 tiles per 1 to 2 s) around the stretch | set span frames, confirm outcomes |
| 3 | video clip | anything still unclear |

A second look is required for:

| case |
|---|
| any close that reopens within 3 s |
| any span graded 1 or 2 with no mistake |
| any span longer than 10 s |
| any mistake whose outcome is not visible on the coarse strip |
| any row marked unsure |

### 7.2 Order of work

| step | what |
|---|---|
| 1 | confirm or reject each v1 mistake row; split rows that cover several events; set $m_a$, $m_b$ |
| 2 | look for mistakes v1 missed: every close that reopens, every segment over its duration flag |
| 3 | mark the `attempt` span for each mistake |
| 4 | mark every idle stretch (`idle_runs.csv` lists the long ones; the trigger finds the rest) |
| 5 | walk every segment for the other causes |
| 6 | grade each span |
| 7 | build the side-by-side sheets per class, order the strategies, pick the references, mark `exemplary` and `strategy` stretches |
| 8 | mark the precision window of every step at level 2 or higher |

### 7.3 Row fields, in this order

`what_happens`, `cause`, `raw_from`, `raw_to`, `grade`, `confidence`, `looked_at`
(coarse / dense / video), `note`.

### 7.4 Scope

Everything is annotated again: every segment and atom of every dataset in training and
validation. v1 labels are a starting point to check, not something to keep.

| dataset | root | episodes | units | hours | step classes | v1 mistakes |
|---|---|---|---|---|---|---|
| rebot_all | `outputs/rebot_all-annotated-v2` | 90 | 1,197 segments | 2.89 | 79 | 177 |
| additions | `outputs/rebot_cache_ready_2026-09-25/main_additions_train` | 24 | 114 | 0.24 | 35 | 11 |
| bits + book | `outputs/rebot_bits-book-annotated-v1` | 10 | 169 | 0.65 | 10 | 15 |
| external | `outputs/rebot_cache_ready_2026-09-25/external_rebot_train` | 240 | 1,077 | 1.75 | 135 | 20 |
| validation | `outputs/rebot_cache_ready_2026-09-25/validation_all` | 8 | 92 | 0.18 | 48 | 7 |
| **ReBot total** | | **372** | **2,649** | **5.72** | | **230** |
| DROID | `outputs/diverse_robot_dataset_v3/corpus` | 50 | 328 atoms | 0.41 | 157 | 34 |
| DROID success | same | 70 | 535 | 0.51 | 260 | 44 |
| MolmoAct | same | 417 | 1,200 | 1.31 | 392 | 11 |
| RoboChallenge | same | 200 | 1,489 | 2.42 | 118 | 5 |
| UR7e | same | 4 | 55 | 0.07 | 10 | 0 |
| YAM | same | 8 | 36 | 0.05 | 8 | 0 |
| FMB | `outputs/diverse_robot_dataset_v3/fmb` | 160 | 801 | 0.93 | 6 | 1 |
| **diverse total** | | **909** | **4,444** | **5.70** | | **95** |
| **all** | | **1,281** | **7,093** | **11.4** | | **325** |

- The rollout and inference recordings are already inside rebot_all (see its
  `meta/provenance.json`), so the older `rebot_inference_*`, `rebot_rollouts-*` and
  `rebot_val-*` roots are not annotated separately.
- Held-out splits of the diverse corpus (validation, test) are annotated like train.
- Step classes are counted by subtask text. For the side-by-side sheets they are grouped
  into verb and object family, which gives far fewer classes.
- New recordings not yet in a root are out of scope until they are built into one.

## 8. Case library

Verdicts are mine, from coarse strips and traces. "4" means no stretch (default). Strategy
stretches are set on the sheets, not here. Cases marked * need the dense strip
before the verdict is final. File names are in `migration/quality_span_2026-09-29/`
(`case_strips/`, `strips/`).

### 8.1 Waits and hovers, no mistake

| case | v1 | what happens | v2 |
|---|---|---|---|
| `hover_all_ep37` | 5 | sock; steady 9 s descent, sock grows in every tile, about 1 s wait before the close | no span (4). Slow, not hovering. |
| `hover_all_ep28` | 4 | sock; steady descent, then about 2.5 s wait at the sock | no span (4) |
| `hover_all_ep69` | 5 | sock; travel from the bin, about 2 s wait at the sock | no span (4) |
| `hover_add_ep5` | 4 | sock; arm at the sock, wrist view unchanged for 6 s with the sock at the edge of the view, then a clean close | `hover` grade 2, about 4 s |
| `q3_ext_ep81` | 3 | ring held over the peg for 5 s, then released onto the stack | no span (4; was `hover` 4 in draft 4); fine step |
| `pause_robochallenge_ep000561` | 5 | UR5 motionless over the pen for about 5 s, then a clean close | `idle` grade 1 |
| `pause_robochallenge_ep001017` | 5 | cup held at the rack, no motion for over 6 s | `idle` grade 1; the fine step does not excuse it |
| `lowq_molmoact_ep007150` | 3 | mug over the counter, 6 s slow descent, then release | `hover` grade 3; coarse target |
| `q3_all_ep57` | 3 | sock over the basket about 2.5 s before opening, lands inside | no span (4) |
| `q3_add_ep12` | 3 | eraser; direct reach, about 2 s settle, clean close | no span (4) |
| `q3_all_ep35` | 3 | sock let go over the basket with a partial opening, lands inside | no span (4) |

### 8.2 Search, stall, regrasp, reposition, no mistake

| case | v1 | what happens | v2 |
|---|---|---|---|
| `q3_all_ep30` | 3 | arm leaves the basket, swings past the table edge (wrist sees the floor), comes back, then waits over the sock about 5 s | `search` grade 2 over the swing; `hover` grade 3 |
| `q3_all_ep22` | 3 | first grasp; about 6 s near home with little motion, then a steady descent | `stall` grade 2 at the start (`idle` if the dense strip shows no motion); rest 4 |
| `q3_all_ep79` | 3 | after the bottle is in the bin the arm stays over the bin about 9 s, returns directly, then sits at home 6 s | `stall` grade 2 over the bin; at home: `stall` grade 2 (the trace shows motion, so not `idle`) |
| `hover_ext_ep63` | 4 | gripper shut, goes down onto the tray for 3 s, rises, opens, goes down again, clean close | `aborted_approach` grade 3 |
| `q3_bits_ep2` | 3 | about 40 s at a bit lying flat: nudged with the fingers until it stands, gripper re-widened twice, one clean close | `reposition` grade 3; it has a purpose and works |
| `q3_ext_ep130` * | 3 | towel; repeated pinch and tug for 19 s, fabric between the fingers each time | `search` grade 3 if the towel moves usefully; else grade 2 |
| `lowq_droid_ep000578` | 3 | tube end gripped, let go where it lay, wrist rotated, gripped again | `regrasp` grade 3 |
| `lowq_droid_ep013684` | 3 | cup held in the air, view unchanged about 4 s, then set down | `hold_still` grade 3 |
| `pause_ur7e_ep000039` | 5 | arm motionless between two blocks for about 3 s | `idle` grade 1 |
| `q3_all_ep59` * | 3 | sock carried to the basket; a second garment shifts along the path | `drag_along` grade 3 if confirmed |
| `q3_all_ep42` | 3 | sock carried to the bin on an indirect path, no stop | no span (4) |
| `q3_ext_ep179` | 3 | 2.3 s carry of a folded towel with a swing | no span (4) |
| `lowq_droid_ep013236` | 3 | 2 s fragments of an open-handed reach, cut by excluded gaps | judge each fragment alone; no span on this evidence |
| `lowq_droid_success_ep023479` | 3 | towel unfolded by three grip and pull cycles | no span; task work |
| `fmb_insert_long_ep92` | 5 | star peg at the hole for about 11 s before it seats | no span unless the peg is lifted clear; precision 5 throughout |

### 8.3 Mistakes

| case | type | what happens | v2 quality spans |
|---|---|---|---|
| `all_ep62_f227414` | failed close | sock; 3.9 s wait, empty close, good close 1 s later | `attempt` 2 (the wait: no span) |
| `all_ep1_f5608` | failed close x3 | sock; three empty closes in 10 s | `hover` 3; one `attempt` over the chain, 1 |
| `all_ep52_f180518` | drop | shirt; carry stalls short of the bin, opens on the table | `attempt` from the stall, crosses move into release, 2 |
| `all_ep40_f143930` | slip | sock held by its tip, slips, arm goes on to the basket and opens on nothing | `attempt` from the close, extended 3.2 s, 1 |
| `all_ep52_f184242` | slip | sock pinched by its end, left on the table, arm goes home shut and empty | `attempt` from the close to the episode end, 1 |
| `bits_ep0_f2330` | failed close | bit not held; arm moves off with the gripper shut for 3 s | `attempt` 1 (oblivious) |
| `hover_bits_ep7` * | failed close (missed in v1) | full close, reopens, bit still standing, closes again | `attempt` 2 |
| `mistake_droid_ep013103` | failed close x2 | glasses; 6 s moving with shut empty fingers, reopen, second close only pushes them | one `attempt`, 1 |
| `mistake_droid_ep013256` | spill | cup held tilted and motionless for 6 s, then moved to the bowl's side; pellets land on the table | `idle` 1; `attempt` 2 |
| `mistake_droid_ep006324` | knock | marker carried over a paper cup, cup knocked onto its side | `attempt` 2 |
| `add_ep3_f6320` | knock | spray bottle rolls as the tape is released | `attempt` 2 |
| `lowq_droid_ep005878` * | failed close, several | bread in a dish; repeated closes that do not lift it; v1 has one 9 s row | split into events; one `attempt`, 1 |
| `lowq_droid_ep006256` * | unclear | mug; 0.9 s close, reopens, re-approach, second close | dense strip decides `regrasp` or `failed_close` |
| `recovery_molmoact_ep000619` | slip | shorts leg slips out in the fold; re-approach and a clean second fold | `attempt` 2; recovery no span |

### 8.4 Side-by-side references (rebot_all, sock grasp)

Draft-4 wording: read "v2 base 5" as an `exemplary` stretch, "4" as no stretch, "3" as a `strategy` stretch.

| case | v1 | what happens | draft 4 base (see note above) |
|---|---|---|---|
| ep29 seg0 | 5 | one continuous descent, sock centred, single close | 5 (reference) |
| ep9 seg0 | 5 | one continuous descent, sock centred, single close | 5 (reference) |
| ep45 seg0 | 5 | at the sock after 3 s, view unchanged about 3 s, then closes | 4 (reference) |
| ep30 seg6 | 5 | at the pile after 3 s, waits about 3 s, then closes | 4 |
| ep55 seg3 | 4 | sock at the bottom edge of the view, waits about 4 s | 4, `hover` span if the dense strip confirms over 4 s |
| ep5 seg9 | 4 | sock at the bottom edge of the view, waits about 2 s | 4 (reference) |
| ep63 seg3 | 4 | waits about 2 s | 4 |
| ep4 seg3 | 4 | grips the sock right by an edge | 3: a strategy that is easier to miss |

### 8.5 Not mistakes

| case | what happens | v2 |
|---|---|---|
| `mistake_robochallenge_ep000404` | yellow button pressed before blue | subtask text "press the yellow button"; no mistake |
| `mistake_droid_success_ep003053` | shirt left bunched on one side of the box, fixed by two more passes | `poor_place` grade 3 |

## 9. Checks after a pass

| check | expected |
|---|---|
| every mistake row lies inside a span of grade 1 or 2 | all |
| frames with grade 1 or 2 and no mistake | more than zero; report the share per dataset |
| grade histogram in frames, per dataset | report |
| per class: grasps with an exemplary stretch, with a strategy stretch, with no stretch | report; no exemplary at all, or all strategy, means look again |
| idle frames left unlabelled (trigger fires, no `idle` span) | zero, or a note per run |
| share of rows marked unsure | report; a second reader does them all |
| agreement of the second reader on a 5 % sample of sure rows | report |
| spans longer than 20 s | list, each with a note |

## 10. Output

Written to new dataset directories, never in place.

| file | columns |
|---|---|
| `meta/episode_metadata.parquet` | as v1; `quality` is no longer used for training (the frame grade comes from the stretches) |
| `meta/quality_spans.parquet` | `episode_index, raw_from_index, raw_to_index, from_index, to_index, quality, cause, confidence, looked_at, note` (quality 5 for `exemplary`, 1 to 3 for critiques; from/to include the headroom for critiques only) |
| `meta/mistakes.parquet` | `episode_index, from_index, to_index, mistake, mistake_type, confidence, note` |
| `meta/precision_windows.parquet` | `episode_index, segment_index, commit_index, from_index, to_index, precision` |
| `references/<class>.json` | the strategy order and the reference segments for 5, 4 and 3 of each class, with the sheet they were picked on |
| diverse | the same tables as `.jsonl` sidecars keyed by `episode_id`, frames at the native rate |

## 11. Open

| item | state |
|---|---|
| loader: frame grade from the stretches (5.1); precision from windows | not built |
| jug or pitcher taken by the lid from the top | 4 or 3, not decided |
| clean final close right after a failed close: can it be exemplary | not decided |
| dense-strip tool | `render_strips.strip` on a 1 to 2 s window; no new renderer needed |
| side-by-side sheets | `render_strips.strip` per instance, stacked (`idle_and_sheets.py`); no new renderer needed |
| strategy order per object family | sock and spray bottle given by the user (5.4); the rest proposed by the annotator, confirmed by the user |
| idle threshold per diverse source | state units differ; set on the first episodes of each source |
| hover norms for the diverse sources | measure on the first annotated episodes |
| the 09-28 hovering recording | not found on disk; not checked against the rubric |
