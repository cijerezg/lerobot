# ReBot annotation: agent brief

Two jobs, each given to an agent with its own prompt: annotate one episode (first part), then grade quality on
the class strips ("Grading", at the end).

Agents run in parallel and share one scratchpad: keep every script you write in a folder named for your id
(`<scratchpad>/<id>/`), and re-read a label file right before you write it (audit 2026-10-07: one agent ran another's
script from the shared root).

## Annotating one episode

One agent annotates one source episode of a ReBot pass: task string, subtask segments, speed, precision, contact
and mistakes. Quality stretches and precision windows are graded afterwards, per class and side by side; do not
write grades here, but describe what you see (with frames) so the grader can use it.

Run commands from the workspace root (the folder holding `lerobot/`, `migration/`, `outputs/`) with `uv run python`.
`<work>` is the results folder of the pass, `<idx>` your episode's inventory index. Sources are read only. Write only
`<work>/labels/<idx>.json` and sheets under `<work>/review/<idx>/`. No GPU, no installs, no new scripts.

### Read first

1. `rubrics/annotation_principles.md`, in full. It overrides the other rubrics where they disagree.
2. `rubrics/quality_mistake_rubric_v2.md` sections 3, 4 (mistakes) and 5.7 (cause names, for your descriptions).
3. `rubrics/precision_rubric.md` (levels, where the commit is, reference values, "Not precision").
4. `rubrics/contact_strategy_rubric.md` (vocabulary table, rules, "What the label does not record").

### Material

| what | where |
|---|---|
| joint and gripper traces, x = episode frame | `<work>/review/trace_<idx>.png` |
| gripper closes, still runs, operator teleop runs | `<work>/review/events_<idx>.txt` |
| overview sheets, one tile per 2 s, top view over wrist view | `<work>/review/<idx>/ep<E>_overview_*.png` |
| any other window | `uv run python -m lerobot.annotation.rebot.sheets <root> <work>/review/<idx> <E> --start A --end B --step S --tag <name>` (8 tiles per sheet) |

- Gripper reading: 0 = shut, negative = open (about -270 wide open). A held rigid object stops the close short. A close
  to about 0 is often a close on nothing. The reading can also drift, so a hold is decided on the wrist view: the object
  must move with the fingers.
- On ReBot the fingertips are the dark triangles at the bottom corners of the wrist view. A held object sits between them.
- Read the overview first. Then render dense sheets (every 3 to 10 frames) around every close, opening, push, pour and
  anything unclear. Every boundary and every mistake is set on a dense sheet, not on the overview.
- Rollouts: orange TELEOP tiles are operator frames. The subtask the operator typed (`recorded_subtask`) is what was
  asked for and is often wrong; label what the arm did.

### What to decide

**Keep range.** `[a, b)` with `a % 3 == 0` and `(b - a) % 3 == 0`; the whole episode `[0, frames)` by default.
- The episode ends when its last useful step is done. If what follows has no purpose (wandering, a step asked for and
  never attempted, a slow drift), end the range about 1 s after that step settles. Do not label it `return to home`.
- A stretch of more than 3 s inside the range where the arm sits still or makes no progress is reported in `flags` with
  its frames, not cut.

**Task string.** What was actually done, in the order done, sentence case: "Put the cup in the basket and the bit in
the component box". Name a step that was attempted even if it went badly; leave out a step never attempted. Name each
object and container as the segment texts do: a colour or position word only on a kind of object some segment text
tells apart, and then on every one of that kind (audit 2026-10-07: 57 tasks kept "brown basket", "navy shirt" after
the segment texts dropped them). A plural only for several objects: one shirt grasped twice is "the shirt" (audit
round 3: template tasks such as "Put sock in basket and shirts in bin" got order and counts wrong).

**Segments.** Gapless, half-open `[from_index, to_index)`, covering the kept range exactly.

| segment | span |
|---|---|
| `grasp the X` | from the end of the previous segment (travel, searching, nudging, failed closes, regrasps) to the frame the successful close holds the object and the lift begins: the end link starts to rise after the gripper settles, not the close onset (audit 2026-10-06: 99 % of the old rebot_all grasps ended about 0.9 s early, in the close) |
| `move the X to the C` | the carry, until the arm arrives over the target; a carry under about 1 s folds into the release |
| `release the X in / on the C` | from arrival over the target until the object has left the fingers and settled; the arm moving off toward the next object belongs to the next segment, even while the operator keeps opening the gripper (audit 2026-10-07: 42 rebot_all releases ran 2-10 s into that travel) |
| contact work (push, fold, close a lid, pour, pry) | from the end of the previous segment, absorbing the approach, to the end of the motion; a `grasp the ...` segment comes first only when the gripper visibly closes on a part and holds it |
| `return to home` | only the final trip back to park |

- Move and release name the held object as its grasp did, without the location words (user 2026-10-07): after
  "grasp the white sock nearest the basket" come "move the white sock to the basket" and "release the white sock in the
  basket"; after "grasp the sock with grey stripes on top of the pile", "move the sock with grey stripes to the basket".
  Colour and feature words stay; position words (nearest, farther, on top of, on the shirt, under, next to) go.
- A retried step stays one segment. A close that holds nothing and is carried away empty stays in the grasp segment; the
  close is the mistake.
- A part released in or on its slot but not seated, then pushed in with the fingers, is its own contact-work segment
  ("push the clipper into the slot"), contact `push`: the part moves into the fit (`press` is for a face that does not
  move). User 2026-10-07: "if it didn't fit well, it makes sense to push". Pressing on a part already seated, with no
  visible change, is no progress: report it in `flags` with its frames.
- A pour where nothing comes out is still a pour ("pour the pills into the cup", `tilt-pour`): the goal is visible
  (user 2026-10-07).
- An object that is grasped, lost and grasped again gives two grasp segments. They keep the same text when it is still
  the only object of its kind (principles section 2).

**Subtask text** (principles section 2). There is no fixed list: write the least text that tells the robot what to do
next. Start from the bare verb and noun ("grasp the bit", "move the cup to the basket") and add a word only when the
scene needs it: which object when there is a look-alike, the order when a step repeats on different parts, and where it
goes for every move, release, push, pour and fold, as specific as the scene needs ("the basket" when there is one). Two
targets in the scene never share a text. Lower case, starts with the verb, describes what the arm did in these frames.
Short, simple sentences; no coined names; no colour or position on an object that has no look-alike. If
the frames cannot resolve it, write the best reference and add `ambiguous: <why>` to the segment's `note`. Add one
`NEW_SUBTASK: <string>` flag per distinct string (the validator needs it).

**Per segment.**

| field | how |
|---|---|
| `what_happens` | 1 to 3 sentences of what you saw, with frames: object, grip point, hovers, retries, pauses, TELEOP runs |
| `speed` 1 to 5 | tempo of the segment's work: 1 stalled or dithering, 2 slow and careful, 3 typical teleop pace, 4 brisk, 5 the quickest |
| `precision` 1 to 5 | slack at the commit, read on the wrist and top view at the commit frame; the reference table is a first guess. A move takes the next step's level minus 1 (floor 1); return to home 1. `orientation_pin` true only when the commit pins orientation to about 15 deg |
| `contact` | what the gripper did at the commit, read on the frames, whatever the verb in the text: one slug of `vocab.py`. Move and return are `na`. Top-pinch or side-pinch is not judged from the image: it is the FK approach angle at the commit, under 45 deg top (contact rubric); `events_<idx>.txt` prints it at every close |
| `note` | what you read at the commit frame (which frame, what the pads touch) and anything that moved a label off its prior |

Two segments with the same text, contact and precision in different situations: check that contact and precision were
read from the frames. Change the text only when the scene has a look-alike it fails to separate.

**Mistakes.** Rubric section 4: `failed_close`, `slip`, `drop`, `knock`, `spill`. Event only, one row per event, both
tests (the frame it happens, the visible outcome), span by rubric 4.4, set on a dense sheet. Every close that reopens
within about 3 s is a candidate. Not mistakes: hovering, regrasping where the object lay, nudging, a wrong object (the
text names what was done). A released cloth left draped over the container rim and carried in again is not a `drop`
(rubric 4.5): `drop` when it rests wholly outside the container, or mostly outside and is not carried in again.

### Output

```
{"idx": int, "key": str, "frames": int, "decision": "keep", "reason": "<what the episode shows>",
 "episodes": [
   {"keep": [a, b], "split": "train", "task": "...",
    "segments": [{"from_index": a, "to_index": ..., "subtask": "...", "what_happens": "...", "speed": 1-5,
                  "precision": 1-5, "orientation_pin": false, "contact": "...", "note": ""}, ...],
    "mistakes": [{"type": "...", "from_index": int, "to_index": int, "what_happens": "...", "confidence": "sure|unsure",
                  "looked_at": "coarse|dense|video", "note": ""}],
    "flags": ["NEW_SUBTASK: ..."], "cuts": []}],
 "flags": []}
```

`uv run python -m lerobot.annotation.rebot.validate_labels <work> <idx>` must print PASS.

Final message, under 200 words: the segment list (frames, text), mistakes by type, segments marked ambiguous, anything
you could not decide and why.

## Grading

The grader writes the quality stretches, confirms the mistakes and marks the precision windows of the units (segments)
it is given. `<classes>` is `classes/` in the results folder of the class pass (`paths.QUALITY_V2`). Each unit sits in a
class folder `<classes>/<slug>/` and is graded against a class file named in `<work>/class_map.json` (`final`).

### Read first

1. `rubrics/annotation_principles.md` section 4, then `rubrics/quality_mistake_rubric_v2.md` in full. It is the rule.
2. `rubrics/precision_rubric.md`: the level table and "Not precision".
3. The class file of each of your classes: the 5 / 4 / 3 rows, the reading cues, the notes. Look at two or three of its
   reference strips (`<classes>/<owner class>/aligned/<uid>.jpg` for the uids on its `References:` line) before grading.

### Material, per unit

| what | where |
|---|---|
| header (segments in the window, the annotator's description as `v1 note`, the annotator's mistake rows, still runs, commit frame) and a 4 Hz trace | `<classes>/<slug>/traces/<uid>.txt` |
| the unit plus 1 s before and 2 s after, 6 s strips, top view over wrist view | `<classes>/<slug>/coarse/<uid>.jpg` |
| 6 s before the commit to 1.5 s after, the window the references use | `<classes>/<slug>/aligned/<uid>.jpg` |
| any window, 8 tiles (use 1 to 2 s windows) | `uv run python -m lerobot.annotation.quality_v2.dense <slug> <uid> <from> <to>` |

Frames are the trace's frames (global index of the staging root). There is no earlier grade: `v1 q0` means none.

### Per unit, in this order

1. Look at the strips and say what you see. Then read the annotator's description.
2. Mistakes: confirm or reject each annotator row on a dense strip (rubric 4.2), set the span (4.4), and look for missed
   ones (every close that reopens within about 3 s, every object that ends off the target). A mistake belongs to the unit
   where its event starts.
3. One `attempt` stretch around each mistake (5.10).
4. Other critiques, with the four questions (5.2). Read the approach line first (rubric 5.13): where it falls steadily
   the approach is direct, 4 or 5, however slow, and gets no critique. Where it rises again (overshoot, back-off) or goes
   flat (hover), render dense strips (0.5 s spacing or finer) and confirm on the frames: `search` or `hover` 3, and 2 where
   it drags on, from the frame the line stops falling. A hesitation in mid-carry is `hold_still` 3.
5. Strategy: put the aligned strip next to the references and decide `exemplary`, `strategy` or nothing.
6. Precision window (rubric 6) for a level of 2 or more. A carry takes the level of the release that follows it.
   The level is the segment's `precision` in the label file (`<work>/labels/<IDX>.json`); the grader places the window,
   it does not re-pick the level. A level you think is wrong goes in the note, not in the window.

- Every grade points at frames. A stretch you cannot show on a strip is not written.
- 4 means the motion goes straight to its goal. Do not default to 4: an action that looks clean and secure is a 5 (a
  short settle before the lift does not rule it out), and one that looks awkward is a 3. The user's verdicts on the
  first three rollouts are in rubric section 8.6; read them before grading.
- Operator frames in a rollout (TELEOP) are graded like any teleop motion. `other` 1 "human help" is only for a hand in
  the scene.
- A hold or hover over the target right before the opening belongs to the release unit.
- Mark a row `unsure` rather than guess, and say what would settle it.

### Output

`<classes>/<slug>/labels/<your id>.jsonl`, one JSON object per unit:

```
{"uid": ..., "what_happens": "<2-3 sentences>", "strategy": "<grip point / path / release point>",
 "spans": [{"what_happens": ..., "cause": ..., "raw_from": int, "raw_to": int, "grade": 1|2|3|5,
            "confidence": "sure|unsure", "looked_at": "coarse|dense|video", "note": ...}],
 "mistakes": [{"what_happens": ..., "type": "failed_close|slip|drop|knock|spill", "from_index": int, "to_index": int,
               "confidence": "sure|unsure", "looked_at": "coarse|dense|video", "v1_row": "<from-to or new>", "note": ...}],
 "v1_rejected": [{"v1_row": "<from-to type>", "reason": ...}],
 "precision": null | {"level": int, "w_a": int, "commit_index": int, "w_b": int, "confidence": "sure|unsure"},
 "confidence": "sure|unsure", "looked_at": "coarse|dense|video", "note": ...}
```

Raw frames, no headroom (the writer adds it). `exemplary` is grade 5, `strategy` grade 3. Every mistake lies inside an
`attempt` stretch of grade 1 or 2 of the same unit's row.

Final message, under 200 words: per unit the stretches (cause, grade, frames), mistakes confirmed / rejected / new,
unsure rows, and any class rule that did not fit.

## Second read

A second reader settles every `unsure` row a grader left. Slices come from
`quality_v2.second_read <pool> --no-calibrate <every class of the pool>` (`classes/second_read_<pool>.json`): one class per
slice, each unit with its label file and the open refs (`unit`, `span:<i>`, `mistake:<i>`, `precision`).

- Read first as for grading (principles section 4, the quality rubric, the precision rubric, the slice's class file with its
  notes; the class notes of this pass are the latest rule).
- Per ref: read the grader's row and what it says would settle it, then look at exactly that (dense strips at 0.5 s or
  finer, `dense <slug> <uid> <from> <to>`). Decide on the frames, not on the grader's lean.
- `agree` keeps the row; `change` gives the full corrected row; `remove` drops it. A missed span or mistake goes in `added`.
- Output: `<classes>/<slug>/resolve/<reader id>.jsonl`, one object per unit:
  `{"uid", "reader", "rows": [{"ref", "verdict": "agree|change|remove", "row": {...} (change only), "note"}],
  "added": {"spans": [], "mistakes": []}, "looked_at", "note"}`. Every open ref of the unit gets a row.

## Auditing one episode

The checker of an audit round (README "Auditing annotated data by random sampling") reads every label of one episode
of an annotated root against the frames, by the same rules as the annotator and the grader. It writes no label; it
reports issues with the fix it would make.

### Read first

Everything "Annotating one episode" and "Grading" tell you to read: the principles in full, quality rubric sections 4
and 5 (5.13 approach line, 8.6 the user's verdicts), the precision rubric, the contact rubric.

### Material

| what | where |
|---|---|
| every label the trainer reads, episode-local frames; gripper closes, still runs; per step the commit and the approach line's rises and flats | `<work>/material/<root>/ep<E>/labels.txt` |
| approach line of each step (not moves and returns) | `<work>/material/<root>/ep<E>/approach_seg<K>.png` |
| sheets, overview and dense | `uv run python -m lerobot.annotation.rebot.sheets <root path> <work>/review/<root>_ep<E> <E> --step 60 --tag overview`, then `--start A --end B --step 3..10` |

`labels.txt` is written by `uv run python -m lerobot.annotation.review.audit_material <root> <work>/material/<root> <E>`.

### What to check, per segment

| field | the test |
|---|---|
| text | principles 2: the least text; tells apart every look-alike visible at that frame; names where it goes; plain words; describes what the arm did |
| boundaries, kept range | the segment table of "Annotating one episode"; the episode ends at its last useful step (aimless motion after it is cut) |
| contact, precision | read at the commit frame on the wrist and top view |
| mistakes | rubric 4: every close that reopens within about 3 s, every object that ends off its target; both tests; no invented events |
| quality | approach line first (rubric 5.13), dense sheets where it rises or goes flat; rubric 5.3 and the 8.6 verdicts: an awkward, hesitant or indirect approach is 3 (2 where it drags), a clean direct action is 5, a steady fall is 4 or 5 |
| precision window | rubric 6 |

Severity: `wrong` teaches the model something false (a text that names the wrong object or target or fails to tell a
look-alike apart; a missed or invented mistake; a grade the frames contradict, such as no critique on a visible
overshoot or hover, a critique on a direct approach, a 4 or 5 on an awkward one; a boundary more than 1 s off; a
contact of another kind; a range that keeps aimless motion). `minor` is imprecise but harmless (a boundary under 1 s
off, a one-step grade call on a borderline case, a colour or position word on an object with no look-alike).

### Output

`<work>/checks/<root>_ep<E>.json`:

```
{"root": str, "episode": int, "source": "<source root> ep <n>", "checked": "<what you looked at>",
 "issues": [{"segment": int|null, "field": "text|boundary|keep|task|contact|precision|mistake|quality|precision_window",
             "frames": [a, b], "what": "...", "fix": "<the corrected value: text, frames, cause + grade, mistake row>",
             "severity": "wrong|minor", "looked_at": "coarse|dense"}]}
```

Frames are episode-local. Every issue points at frames you looked at on a dense sheet. Final message, under 150 words:
issue counts by severity, then each `wrong` issue in one line.

## Fixing an audited source

A fixer takes some episodes of a fix pass (README "Auditing annotated data by random sampling", step 3): `<work>` is a
pass that re-annotates an annotated root (its `inventory.json` sources are that root, its `labels/<idx>.json` hold the
root's labels, its `pass.json` names the class pool, the uid prefix `dataset` and the `staging` root the pool's frames
refer to). It does two jobs, in this order, and writes only the label files and one resolve file per class folder.

### Read first

Everything "Annotating one episode" and "Grading" tell you to read, then "Auditing one episode" above.

### Job 1: the checker's issues (audited episodes only)

The check file `<audit>/checks/<root>_ep<E>.json` lists issues with a fix, in the episode-local frames of the training
root (the prompt gives the map to the label file's frames and to the staging frames). Look at each issue on a dense
sheet before applying it; apply `wrong` and `minor` alike, and say in the final message any you did not apply and why.

| field | where the fix goes |
|---|---|
| text, task, boundary, keep, contact, precision | the label file (frames in its own space); old value in the segment `note` as `audit: <old>` |
| quality, mistake, precision window | a resolve row in the class folder of the unit, `<classes>/<slug>/resolve/F_audit1006_<your id>.jsonl` |

A resolve row: `{"uid": "<dataset>_ep<staging episode>_seg<k>", "reader": "F_audit1006_<your id>", "rows": [{"ref":
"span:<i>|mistake:<i>|precision", "verdict": "change|remove", "row": {...}, "note": ...}], "added": {"spans": [...],
"mistakes": [...]}, "looked_at": "dense", "note": ...}`. `span:<i>` and `mistake:<i>` index the unit's rows in its first
reader's label file (`<classes>/<slug>/labels/*.jsonl`, the line with that uid); a span that is already gone or changed
by a later resolve file (`resolve/*.jsonl`) is checked there first. `compile_labels` applies `F_audit*` files after
every other resolve file, so an audit row wins on the same ref. Rows and frames are in the format of "Grading", in
staging frames. Find a unit's class folder with `grep -l '"uid": "<uid>"' <classes>/<pool>__*/units.jsonl`. A segment
merged, split or moved keeps its spans by frames; nothing else is needed for it.

### Job 2: text sweep (every episode given)

Every segment text against principles section 2, on the frames: an overview sheet of the episode and the frame where
each grasp starts. Tell apart every look-alike visible at that frame (same-colour socks, two shirts, several bits); drop
a colour or position word on an object with no look-alike; name the receiver of every move, release, push and fold; the
same object keeps the same text. Change the task string when it leaves out a step that was done or names one never
attempted, lists the steps out of order, or breaks the colour rule of "Task string" above. Old
text in the segment `note` as `text a: <old>`; one `NEW_SUBTASK: <string>` flag per new string.

`uv run python -m lerobot.annotation.rebot.validate_labels <work> <idx> ...` must print PASS. Final message, under 200
words: per episode the issues applied / not applied, the texts changed (count, and any you could not resolve), the
resolve files written.

### Job 3: rule sweep (spans given)

When a rule changes (README step 3), every span the old rule produced is read again under the new one. The prompt gives
a list of spans (root, episode, uid, root and staging frames, cause, grade, note). Per span: read the unit's trace
(`<classes>/<slug>/traces/<uid>.txt`, FK speed and height at 4 Hz) and a dense sheet of the span on the root, then
decide keep, change (cause or grade) or remove under the current rubric row. Change and remove go in a resolve row as in
job 1. Write `<audit>/sweeps/<rule>_<your id>.json`: one row per span `{"uid", "span", "verdict": "keep|change|remove",
"why", "frames_looked_at"}`. Final message, under 150 words: counts by verdict, resolve files written.

The same job for boundaries: the prompt gives candidate segments from a scan (for example a release that may run into
the travel to the next object, with the frame the scan thinks the arm leaves). Per candidate, read the end of the carry
before it and the end of the release on a dense sheet (top and wrist views) and the gripper and FK trace; move a
boundary that is more than about 0.5 s off in the label file (old value in the segment `note` as `audit: <old>`). Spans
stay where they are by frames. A scan marker is a hint, not the answer: cloth can stay on the fingertips after the
gripper opens.
