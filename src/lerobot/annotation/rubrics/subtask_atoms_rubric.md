# Atom review guide

> Current authority: `annotation_principles.md` for text and retention, and `quality_mistake_rubric_v2.md`
> for frame-local quality, mistakes and precision windows. The v1 parent-quality fields and review tools below
> are legacy storage/workflow details, not current grading rules. For sampled audits start at
> [the annotation README](../README.md#diverse-dataset-sampled-audit).

You are cutting the diverse corpus's reviewed subtask intervals ("parents") into atoms:
one verb plus one object per atom, at the ReBot resolution. Proprio already proposed
candidate cuts; you look at the frames and decide. The original v1 atom pass left parents unchanged. In a current
audit, retention/boundary repairs may revise parents in a new store; preserve originals and excluded gaps.

Everything is under `/home/user/Documents/Research/RL/LeRobot` (run commands from there,
with `uv run python`, never bare python; install nothing).

- sheets: `outputs/_annotation/subtask_atoms_review/sheets/<source>/<episode_id>_NN.jpg`
  - `_00` (and one `_NN` per further parent) = header, gripper/speed trace with the candidate
    cuts, and an OVERVIEW strip of frames at a fixed stride (external camera, and the wrist
    camera under it for DROID/UR7e/press/lamp/water/wipe). Labels: `f<frame> <seconds>s
    g<closedness>` where g = 1.00 means the gripper is fully shut, 0.00 fully open.
  - the following sheets = CANDIDATE ATOM rows: external start | mid | end-1, then wrist
    start | mid | end-1, with the candidate verb and the frame span.
  - a parent's sheets are consecutive: the overview page for PARENT k is followed by its
    atom pages, then the overview for PARENT k+1, and so on. Read the PARENT line.
- finer look: `uv run python -m lerobot.annotation.atoms.sheets dense --episode E --from-s 10 --to-s 20 --stride-s 0.5`
  writes `outputs/_annotation/subtask_atoms_review/sheets/dense/...jpg` (path is printed). Use it
  whenever a boundary is not obvious from the overview; the cost is small.
- proposal details (frames of every gripper event, arrival, failed closes): `migration/subtask_atoms_2026-09-08/proposals.jsonl`
  (one JSON line per parent; `grep '"episode_id": "E"'`).
- verdict skeleton: `uv run python -m lerobot.annotation.atoms.verdict skeleton --episode E`
  writes `outputs/_annotation/subtask_atoms_review/verdicts/E.json` prefilled with the
  proposal, then prints it. Edit that file (fill objects, fix verbs and cuts), then run
  `uv run python -m lerobot.annotation.atoms.verdict check --episode E` until it prints OK.
  `verdict.py show --episode E` prints the rendered subtask strings for a last read.

## The grammar

Pick-and-place, cut at physical events (the ReBot rule):

| atom | span | verb field | rendered |
|---|---|---|---|
| grasp | [end of the previous release (or parent start), gripper closes on the object) | `grasp` | `grasp the X` |
| move | [gripper closed, arrival over the target) | `move` + container | `move the X to the C` |
| lift (move with no destination) | [gripper closed, end) | `move`, container null | `lift the X` |
| release | [arrival over the target, opening settles) | `release` + container | `release the X in the C` / `on the C` |
| return | [last release, parent end): the final park only | `return` | `return to home` |

- **Completion, not onset.** Grasp ends and move begins at the first frame where the close has settled and the
  object is securely captured. Release ends only after opening has settled and the object is no longer held.
  Close/open command edges and motion onset stay inside grasp/release. A precision `commit_index` is a separate
  onset marker and must not be reused as the atom cut.
- **Hard gaps.** Every atom lies wholly inside one continuous retained interval. If an action continues across an
  excluded interval, split it at the gap and resume it as a new atom with the same text when identity and action
  are unchanged. No atom tiles excluded time.
- grasp absorbs the transit back from the previous container and any searching, nudging
  or repositioning before the close. One grasp atom per object actually lifted.
- A failed close (closes on nothing, or loses the object during the close and reopens
  without carrying it away) does NOT open a cycle: it stays inside the enclosing grasp
  atom, as a mistake event. Look at the frames: proprio flags "failed closes" in the header
  by displacement, but a deliberate close-to-push is not a failed close (it is labour,
  quality 3), and a close that lifted the object 5 cm and dropped it IS one.
- move starts when the fingers shut. release starts at arrival over the target (the arm
  settles over the container; the descent and the opening are part of release). The
  proposal's `arm_settle` cut is a proprio guess; move it if the frames show the arm
  arriving earlier or later.
- return to home is only the final park after the last release. A transit to the next
  object is the next grasp, never a return. A return shorter than about 1 s is not worth an
  atom: fold it into the last release.
- No separate move/release when the object is not transported anywhere (e.g. fold, wipe,
  stir): see contact atoms.

Contact atoms (non pick-and-place work): one atom per contact episode, the verb kept.

| work | atoms |
|---|---|
| press a button | `press the pink button` from the parent start (or the lift-off after the previous press) to the lift-off after this press; the transit to the next button belongs to the next press; the final park is `return to home` |
| turn on a lamp | `turn on the lamp` to the lift-off after the switch, then `return to home` |
| wipe / scrub | `grasp the rag` to the close; `wipe the desk with the rag` per pass, a new atom only when the arm lifts the rag off and comes back down; `release the rag on the desk`; `return to home` |
| water / pour | `grasp the watering can`; `move the watering can to the plant`; `water the plant` while tilted; `move the watering can to the plant` again if there is a second plant, `water the plant`; `move the watering can to the table`; `release the watering can on the table`; `return to home`. Pouring from a cup: `pour the cup into the red bowl`. |
| fold / unfold / spread / straighten | `grasp the towel` to the close, then `fold the towel` from the close to the opening settles (the letting-go is part of the fold); repeat per grasp |
| stir | `grasp the red spoon`; `stir the black beans with the red spoon`; `release the red spoon in the bowl` only if it is put down |
| open / close a drawer, door, lid | `grasp the drawer handle`; `open the drawer` (close to opening settles); or `close the drawer` as one push atom |
| push / pull a handle | `push the foosball handle`, `pull the foosball handle` |
| insert (toaster, channel, tube) | `move the bread slice to the toaster`; `insert the bread slice into the toaster` (arrival to opening settles) |
| hang | treat as pick-and-place: `release the red cup on the rack` |
| shredder | pick-and-place: `release the paper in the shredder` |

Allowed verbs: grasp, move, release, return, wipe, press, pour, water, turn on, turn off,
fold, unfold, stir, scrub, insert, open, close, push, pull, spread, straighten, rotate,
lift, place, hang, tilt, shake, flatten, flip, slide, drag, hold. If none fits, pick the
closest and say so in the note.

## Objects, containers, instruments

- `object` is a noun phrase starting with "the ": the thing being handled, with the colour
  or a distinguishing attribute whenever several similar things are present: `the green
  block`, `the pink flower`, `the digital scale`. Use the task string's own names where it
  has them (task says pink button: write pink even if it looks red). Otherwise a short
  common name from the frames (`the kiwi`, `the banana`, `the star fruit`).
- `container` is the destination for move/release (and the target of insert/pour): `the
  vase`, `the basket`, `the yellow box`, `the rack`, `the red block` (when stacking on it).
  `preposition` defaults to "to" for move and "in" for release; set "on" for a surface, rack,
  plate, shelf, or a block stacked on a block; "into" for insert/pour.
- `instrument` only when a tool is used on the object: `wipe the desk with the rag`,
  `stir the beans with the red spoon`, `grasp the lemon with the tongs`.
- NEVER: ordinals, counts or progress words (first, second, next, remaining, another,
  last, again, other, more, all, both, one, two ...), digits, or two objects joined by
  "and" as substitutes for visible identity. Four identical green blocks still need visible spatial references
  when more than one is in play: `grasp the green block nearest the basket`.
- Use the least text that distinguishes active look-alikes at that frame, including a visible landmark or part
  when colour is insufficient. Once only one remains in play, drop unnecessary modifiers. The same object
  regrasped in the same scene needs no invented new name. If no visible reference resolves identity, use the
  best available reference and flag `ambiguous`. Do not invent attributes you cannot see; `the flower` is better
  than a guessed colour. When you cannot tell what the object
  is, name what you see (`the small object`) and set the parent's confidence to "unsure".

## Quality and mistakes

Use `quality_mistake_rubric_v2.md`, sections 4-6. Quality is frame-local: default 4, raw exemplary stretches 5,
critiques 3/2/1. Do not inherit a parent's grade as the current verdict, cap a child at its parent's grade, or
penalize an entire atom because it contains a mistake. Legacy `quality` / `quality: null` fields are not the
v2 training verdict when frame sidecars are present.

A mistake needs a discrete visible event and outcome: `failed_close`, `slip`, `drop`, `knock`, `spill`.
Do not duplicate an existing event. Use native half-open event spans, which may cross an atom boundary but never
an excluded gap; each event is contained in a quality 1/2 attempt stretch. Mark raw stretches without adding
padding by hand. Searching, regrasping and repositioning are not automatically mistakes. `wrong_target` is
retired: name the action actually performed without a choice-of-target penalty.

Pause/recovery records are review candidates, not labels to copy automatically. Confirm idle against source-aware
traces and frames, then cut under the principles. Keep purposeful searching at the object and grade it. Record
ambiguous outcomes as unsure and obtain a second read.

## Current audit case library (2026-10-07)

| case | evidence | verdict |
|---|---|---|
| YAM retained fragments | a legacy grasp covers frames 0-200 but source frames 30-60 are excluded | split at the excluded interval; no atom, history, or future crosses it |
| DROID AUTOLab repeated bars | `first bar` / `second bar` refer to look-alikes | use visible state and position: `the black aluminium bar under the gripper`, then `beside the parallel bars` |
| UR7e stack | final close starts around f1620 and settles at f1628 | precision onset may be f1620; grasp-to-move atom cut is f1628 |
| RoboChallenge block grasp | measured-width close is still ramping at f233 and settles at f239 | grasp ends at the settled close, f239, not at the earlier mechanical candidate |

## Legacy v1 procedure per episode

For a current audit, use the README workflow and its new results folder. These commands and fixed output paths
describe the original atom pass; inspect/adapt tools before reuse, including their obsolete semantic validators.

1. Read the overview page(s): task, parent text, reviewed mistakes, gripper events. Decide
   the cycle structure (which closes lifted something, which are failed closes, where each
   release happened).
2. Read the atom pages: confirm each candidate cut and verb. Fix cuts by editing
   `start_timestep` / `end_timestep_exclusive` (integers; the atoms must tile the parent
   exactly: every atom starts where the previous ended, the first at the parent start, the
   last at the parent end). Use the dense strip for any cut you are not sure of.
3. Name objects and containers from the task string and the wrist frames.
4. Write `outputs/_annotation/subtask_atoms_review/verdicts/<episode_id>.json` (start from
   the skeleton), set `reviewer`, per-parent `confidence` ("confident" or "unsure") and a
   one-line `note` saying what you saw ("4 flowers: white, pink, yellow, white; release 3
   starts at f2130 when the arm is over the vase").
5. Run the checker until it prints OK. Do not leave a parent unreviewed.

Boundary provenance is derived automatically: a cut at a proposal gripper event keeps
`gripper_event`, at the proposal arrival keeps `arm_settle`, at the parent edge is `parent`,
anything you moved becomes `vision`. So you never write provenance.

Merge candidate atoms by deleting rows and extending the neighbour; split by inserting a
row. Every atom must be at least 0.5 s unless the parent itself is shorter.

If a parent cannot be segmented confidently (frames too dark, object unidentifiable,
gripper events contradict the frames and the dense strip does not resolve it), still write
your best atoms, set `"confidence": "unsure"` and explain in the note. Never guess silently.


## Decisions retained from the September 9 review

These conventions were consolidated from the retired root progress tracker on
2026-09-24. Episode examples and unresolved skeleton notes describe that review,
not a fresh audit of the current corpus. The inherited-grade and identical-name conventions below are historical
v1 decisions, superseded by the current sections above; do not apply them to new labels.

- **Arrivals.** The proposal's `arm_settle` cut is a proprio guess and was wrong in most
  RoboChallenge families (it fires at the end of the descent, on a mid-pass slowdown, or
  at the moment a watering can is righted). Move it to the end of the lateral travel,
  checked on a 0.2-0.5 s `sheets.py dense` strip. The two UR5 families (arrange fruits,
  item classification) are the exception: their arrivals are accurate and were kept,
  except where the proposal produced no release atom at all.
- **Fake cycles.** A close made over the destination container right after a release, or
  in mid-air during an empty transit, is not a carry: merge its move and release into the
  following grasp and add NO mistake event (matching the `000285` override).
- **Failed closes.** Only when the frames show a committed descent onto the object, the
  fingers shutting empty and reopening with the object untouched. Then the enclosing
  grasp atom gets a `failed_close` event and quality 2. Historical examples were recorded
  (green blocks `000199`, `000844`, `000859`; DROID `009488`, `011600`).
- **Regrasps are not events.** Short pinch-and-let-go closes on cloth, grip adjustments
  while stirring, and picking an object up and putting it straight back are labour, not
  discrete failures; fold them into the enclosing grasp atom and say so in the note.
- **Quality-1/2 parents with reviewed mistake events.** The checker gives the event to the
  atom with the largest overlap and then REQUIRES every sibling atom to carry its own
  grade (5 direct, 4 one minor correction, 3 laboured) with a `quality_note`. This bites
  in DROID (`006324`, `011600`, `013115`).
- **Naming.** Use the task string's names; add a colour only when two of a kind are in
  the same scene (`the black pen`/`the red pen`, `the silver phone`/`the black phone`,
  `the green mango` vs `the mango`). Identical objects keep one name and repeat verbatim.
- **Non-pick-and-place verbs** used so far: `fold`, `wipe`, `scrub`, `stir`, `water`,
  `open`, `press`, `turn on`, with the object and `instrument` fields per the guide table.
- **DROID specifics.** Episodes are multi-parent with excluded gaps; atoms tile each
  parent separately, so one action often spans a parent boundary. Several DROID episodes
  never lift the object at all (`013103`, `013116`) — those are a single grasp atom per
  parent, which is the honest reading, not a proposal-shaped grasp/move/release.
- **Never leave an unfilled skeleton** in `agent_reviews/`: a file with an empty `images`
  list fails `agent_check.py check`. `droid__IPRL__ep005570` still has one from an earlier
  session; it is harmless only because every parent of that episode has a manual override,
  so `status` never checks it.
