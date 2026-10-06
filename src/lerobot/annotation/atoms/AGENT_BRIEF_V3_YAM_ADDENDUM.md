# v3 YAM addendum to AGENT_BRIEF.md (read AGENT_BRIEF.md, REVIEW_GUIDE.md and AGENT_BRIEF_V2_ADDENDUM.md first)

This round reviews the `yam__*` episodes of the v3 corpus. EVERY command you run must be prefixed
with these three environment variables, exactly:

    DIVERSE_DATASET_ROOT=outputs/diverse_robot_dataset_v3 ATOMS_REVIEW_ROOT=outputs/_annotation/subtask_atoms_review_v3 ATOMS_WORK_ROOT=migration/subtask_atoms_v3_2026-09-19 uv run python migration/subtask_atoms_2026-09-08/<tool>.py ...

Paths differ from the brief accordingly: sheets are under
`outputs/_annotation/subtask_atoms_review_v3/sheets/yam/<episode_id>_NN.jpg` (already rendered,
2-3 pages per episode; `_00` is the overview with the trace, `_01`/`_02` the candidate-atom
pages), agent reviews go to `outputs/_annotation/subtask_atoms_review_v3/agent_reviews/<episode_id>.json`,
proposals are `migration/subtask_atoms_v3_2026-09-19/proposals.jsonl`. Dense strips and
closeups (`sheets.py dense`, `look.py`) work as in the brief and land under the v3 review root.

Source `yam` (I2RT YAM, one 6-joint arm + gripper, embodiment string `YAM`, joint layout 8).
Three sets, one task string each, each episode covers the task ONCE, so every episode has ONE
parent covering the whole episode (`[0, frames)`), quality 4 or 5, no reviewed mistake events
(v3 episode verdicts: `outputs/_annotation/diverse_v3_review/yam/<set>/verdicts.json`).

| set | corpus ids | rate | external camera | wrist camera |
|---|---|---|---|---|
| duster | `yam__duster__ep000003/090/106/174` | 25 Hz | `top` | `right_wrist` |
| espresso | `yam__espresso__ep000002/042/055/086` | 30 Hz | `outside` | `wrist` |
| pick_place (NOT in the corpus yet; license decision pending) | `yam__pick_place__ep000003/021/032/049` if admitted | 30 Hz | `cam_high` | `cam_wrist` |

The sheet columns are `<external> start | mid | end-1 || <wrist> start | mid | end-1` with the
camera names printed in the page header, as for every other source. The overview trace is the
arm-joint speed (grey) and the gripper closedness `g` (blue).

## Gripper convention (after the ingest `gripper_transform`)

Every YAM episode in the corpus stores the gripper as a 0-1 ratio with 0 = open, 1 = closed
(layout 8 `ratio_0_open_1_closed`; duster was recorded that way, espresso was flipped at
ingest, pick_place would be flipped on the state slot only). The sheets print `g` directly
from that channel (`GRIPPER["YAM"]` closed_high, scale 1.0), so `g` rises on a close and falls
on an open, the same reading as DROID and UR7e. What a close looks like in the trace:

- duster: the fingers rest fully open at `g` 0.00 for the whole approach; the close is one
  step 0.00 -> 0.53-0.57 in about 0.3 s (the fingers stall on the duster's body, so a firm
  grasp reads 0.55, never 1.0); the carry holds 0.53; the open is one step back to 0.00 within
  0.2 s. ep174's close is a gradual ramp 0.25 -> 0.57 over 9-10.5 s (verdict: slow soft close,
  then the arm holds still ~2 s before the transport).
- espresso: the fingers are NOT open at rest. `g` starts between 0.29 and 0.37 (a half-open
  hand), the close on the portafilter handle goes to 0.72-0.87, the release drops only to
  0.35-0.60, and then the fingers shut COMPLETELY (0.92-0.99, nothing between them) for the
  lock push. Read the closes from the frames on this set: a `g` of 0.35 is "open", 0.85 is
  "holding the handle", 0.95+ is "fingers shut on nothing, pushing".
- pick_place (if admitted): the fingers rest open at ~0.00 and close towards 1.0 on the
  pencil (thin object; the command overshoots slightly above 1 and the sheet clips it). Read
  the printed `g` on its sheets once rendered; nobody has looked at them yet.

The proposer's gripper events (`close`/`open` at the printed frame) came out right on all 8
ingested episodes: one close/open pair per duster episode, two closes per espresso episode
(handle grasp, then the lock push). ep086's second open is split into two small steps
(28.2 s and 29.0 s); ep042 has no open after the lock push because the fingers stay shut to
the end.

## Sets, what the frames show, and how to cut them

Cite the v3 verdict notes for what each episode does; the atoms below are what the guide's
tables give for that.

### duster (`pick up the duster and place it in the red box`, 25 Hz, 16-20 s)

`top` is a fixed overhead camera looking straight down at a white table: the red box (open,
left of centre) and the small blue duster (right of it) are both fully visible, the arm enters
from the bottom edge; the wrist camera shows the black fingers at the bottom of the frame, the
table, and people seated far behind the table (verdicts: "not in the workspace", ignore them).
Every episode is one pick-and-place cycle (verdicts ep003/090/106 q5, ep174 q4 for the ~2 s
hold after the close): the approach is long (5-9 s of hovering over the duster before the
close, the duster fills the wrist frame from ~5 s on), then a short carry to the box, a release
inside it and a 3-5 s park. Cut: `grasp the duster` (parent start to the close), `move the
duster to the red box`, `release the duster in the red box`, `return to home` (the park is
always > 3 s here, so keep it). The proposals already give exactly these four; check the
boundaries against the frames (the release settles ~1 s before the open on ep003: the open
step at 12.0 s is the atom end, the arrival at 11.0 s the start).

Nouns: `the duster` (or `the blue duster` is acceptable but keep one spelling across the four
episodes), `the red box`. Prepositions: move "to", release "in".

### espresso (`insert the portafilter into the espresso machine`, 30 Hz, 25-31 s)

`outside` is a low side view from the right of the machine (green-and-black YAM arm entering
from the top right, the silver espresso machine on the left; the arm often covers the group
head, verdicts: "partly occluded by the arm"); `wrist` looks past the green fingertips at the
portafilter basket and then at the machine's front panel. Every episode is the SAME sequence
(verdicts ep002/055 q5, ep042/086 q4): grasp the portafilter by its handle from the table,
carry it up to the group head, seat the basket (insert), release the handle, then close the
fingers fully and push the handle round from vertical to horizontal with the shut fingers
(the lock), then park. The verdicts call the closed-finger push "a deliberate contact strategy,
not a failed close": do NOT grade it as a failed_close and do not add a mistake event for it.

Cut with the guide's insert row plus a contact atom for the lock:
`grasp the portafilter` (parent start to the first close), `move the portafilter to the group
head` (or `to the espresso machine` if you prefer the task noun; keep it identical across the
four episodes), `insert the portafilter into the group head` (arrival at the head to the
opening settling; this is where the proposer's first `release` sits: rename it, the letting-go
is inside the insert), `push the portafilter handle` (from the second close through the
rotation to the final opening or the last motion; the proposer shows it as grasp + move of
cycle 1 because the fingers shut: merge those two into the one push atom), `return to home`
only when the final park is about 1 s or longer (ep002 2.5 s, ep055 2 s, ep086 2.4 s: keep;
ep042 keeps the fingers shut on the handle to the last frame, verdict "closed from 19.5 s to
the end": decide from the frames whether its last second is a park; if not, the push runs to
the parent end and there is no return). Per-episode
notes from the verdicts: ep042 releases in two steps (partial open, 1 s hold, full open) and
repositions before the push; ep086 pulls the first approach to the group head back and redoes
it (a correction inside the move: keep it in the move atom, quality 4, note it).

Nouns: `the portafilter`, `the group head` (container of the insert), `the portafilter
handle` (object of the push). Prepositions: move "to", insert "into".

### pick_place (`pick up the pencil from the left sharpener and put it into the right sharpener`, 30 Hz, 7-11 s) - only if admitted

`cam_high` is a fixed overhead camera looking down at a pale wooden table with two black
sharpeners side by side near the top of the frame (the pencil starts standing in the LEFT one),
the arm entering from the bottom; `cam_wrist` looks past the black fingers. Every episode is one
short pick-and-place (verdicts ep003/032/049 q5 "direct", ep021 q4 for a slow ~3.5 s hovering
descent before the release): the pencil is pulled up out of the left sharpener, carried
upright, lowered into the right sharpener, released, park. Cut: `grasp the pencil`, `move the
pencil to the right sharpener`, `insert the pencil into the right sharpener` (the lowering into
the hole to the opening; on these episodes the release IS the insert, so use insert, not
release), `return to home` (the verdicts put the open at 5.3 / 9.5 / 6.6 / 6.1 s in episodes of 7.4 / 10.9 / 9.0 / 8.3 s, so every park is 1.4-2.4 s: keep it).

Nouns: `the pencil`, `the right sharpener` (`the left sharpener` never appears as a container).

## Rules that bite here

- One parent per episode; the whole parent must be tiled exactly; every atom >= 0.5 s.
- Rate is 25 Hz (duster) or 30 Hz (espresso, pick_place): the sheet prints frame index AND
  seconds; write timesteps from the frame index, never from seconds x 25 or x 30 by hand.
- Do not grade the espresso lock push as a failed close; do not invent a release for it.
- Long duster approaches are still ONE `grasp` atom (grasp absorbs the approach).
- Quality: the parent quality (4 or 5) is on the header; atoms inherit it unless the frames
  show something worse inside one atom (ep174's 2 s hold after the close, ep021's slow descent,
  ep086's redone approach: put the 4 on that atom with a quality_note, siblings may stay 5).
