# Brief for a Claude visual reviewer (atom review, subtask-atoms-v1)

> Historical v1 worker brief. Its parent-grade inheritance, automatic event mapping and identical-object naming
> instructions are superseded by the current [atom rubric](../rubrics/subtask_atoms_rubric.md) and principles.
> For a new diverse sampled audit use [the central README](../README.md#diverse-dataset-sampled-audit), not this
> old pass's output paths or grading procedure.

You are one of several reviewers cutting reviewed parent intervals of the diverse corpus into
atoms (one verb, one object) by LOOKING AT THE FRAMES. Read
`lerobot/src/lerobot/annotation/rubrics/subtask_atoms_rubric.md` completely first: it defines the grammar,
the naming rules, the quality rules and the procedure. This brief adds what the guide does
not say and what earlier reviewers got wrong. Quality is the only goal: a slow, correct
review beats a fast one. Never guess silently.

Workspace: `/home/user/Documents/Research/RL/LeRobot` (run every command from there, with
`uv run python`, never bare python). No git. Install nothing. Spawn no agents.

## What you write, and only that

- `outputs/_annotation/subtask_atoms_review/agent_reviews/<episode_id>.json`, one per episode.
  Start it with `uv run python -m lerobot.annotation.atoms.agent_check skeleton --episode E`
  (prefilled from the proprio proposal; parents that already have a Codex manual override
  are left out on purpose and must stay out). Edit the file. Then
  `uv run python -m lerobot.annotation.atoms.agent_check check --episode E` until it
  prints OK, and read the rendered subtask strings it prints as a last check.
- Fill `images` with the workspace-relative paths of every sheet, dense strip and closeup you read.
- Never modify anything else: not `verdicts/`, `manual_overrides/`, `local_vlm_reviews/` (do
  not even read those: they are rejected machine drafts), `sheets/<source>/`, `proposals.jsonl`,
  or anything under `outputs/diverse_robot_dataset/`.

## Where to look

- Sheets: `outputs/_annotation/subtask_atoms_review/sheets/<source>/<episode_id>_NN.jpg`
  (Read them as images). Per parent: one overview page (header, trace, frames every 1-3 s),
  then candidate atom pages (start | mid | end-1 for the external and the wrist camera).
- Dense strip, when a boundary is not obvious (it usually is not on a 3 s overview stride):
  `uv run python -m lerobot.annotation.atoms.sheets dense --episode E --from-s 10 --to-s 14 --stride-s 0.25`
  (prints the path; 200 px tiles, external over wrist).
- Closeup at full resolution (520 px tiles, every camera), for "is something between the
  fingers", "is the object in the container", "which colour":
  `uv run python -m lerobot.annotation.atoms.look --episode E --frames 100 120 140`
  or `--from-s 10 --to-s 13 --stride-s 0.5` (12 frames max per call).
- Proposal details (gripper events, carries, failed closes, arrival frames):
  `grep '"episode_id": "E"' migration/subtask_atoms_2026-09-08/proposals.jsonl`.
- Native rate: 15 Hz for droid/droid_success, 30 Hz for robochallenge and ur7e. Cuts are frame
  indices; seconds = frame / rate.

## Calibration from direct visual checks (earlier machine reviews failed on every one of these)

1. The proposal is a hypothesis from the gripper signal. Decide the cycle structure from the
   images: a real pickup shows the object between the fingers in the wrist view during the
   carry AND gone from its origin / present in the destination afterwards in the external
   view. A close during an empty transit is NOT a cycle: fold it into the next grasp. Example:
   the green-block episode ep000285 proposed three cycles; the frames show two pickups.
2. Partially closed fingers can hold securely. A closedness of 0.30 or 0.46 on a wide or
   soft object is a grasp, not "closed on nothing". Judge by what is between the fingers.
3. An accidental drop is not a release. The object leaving the fingers before the arm is
   over the target stays inside the move (or grasp) atom; the reviewed mistake event already
   describes it. Do not create a release atom for a drop.
4. A failed close (fingers shut, nothing in them, reopen without carrying) stays inside the
   enclosing grasp atom. Confirm it in the frames before adding it as a new failed_close event;
   a deliberate closed-finger push or nudge is labour, not a failure.
5. Empty transit after a deposit belongs to the NEXT grasp. `return to home` is only the final
   park after the last release, and only when it lasts about 1 s or more; otherwise fold it
   into the last release.
6. The proposal's arrival cut (move -> release, provenance arm_settle) is a proprio guess.
   Check it with a dense strip around it. Release starts at ARRIVAL over the target: the end
   of the lateral travel, when the gripper is above the container. Any hover, wrist
   re-orientation, descent and the opening are all inside the release. The proposal often
   sits too late (at the end of a slow descent) or too early (mid-swing); move it to the frame
   where the travel ends.
7. Quality is `null` unless the rules force a value. Never raise a child above the parent.
   The only exception: a parent at quality 1-2 with reviewed mistake events; then every
   sibling atom WITHOUT an event MUST get its own grade 3/4/5 with a quality_note, and the
   atom holding the event stays null. Lower a child below its parent only with visible
   evidence, stated in quality_note. Any atom with a new_mistake_events entry gets 2 (or 1).
8. Do not re-add mistake events the parent already lists (the header shows them); they are
   mapped automatically.

## Naming

- The task string's own names win (task says "pink button": write `the pink button` even if
  it looks red). Otherwise name what you see, short and common.
- A colour or attribute only when several similar objects are present in the scene; then
  every object of that kind gets one, and identical objects keep the same name in every cycle.
- Never ordinals, counts, progress words, digits, or two objects joined by "and". Four green
  blocks give four `grasp the green block` atoms that read identically.
- Consistency across your batch matters: episodes of one robochallenge family come with two
  task strings (a sentence and a slug); use the sentence's names for both halves. Keep a list
  of the canonical names you used and report it.
- `the flower` is better than a guessed colour. If the object cannot be identified, name what
  you see and set the parent's confidence to "unsure".

## Contact tasks (no move/release inside them)

Follow the guide's table. Some proposals give ONE candidate atom for a whole parent because the
gripper never moves (button presses, lamp switch). You must split those by vision:
`press the pink button`, `press the blue button`, ... each from the lift-off after the previous
press to the lift-off after this one, using the task's colour order; final park = `return to home`.
Wipe: one atom per pass (a new atom only when the rag lifts off and comes back down). Water:
`water the plant` while tilted, carries between plants are `move the watering can to the plant`.
Fold: `grasp the towel` then `fold the towel` (close to letting go). Hang: pick-and-place with
`release the cup on the rack`. Shredder: `release the paper in the shredder`.

## Per parent, before you write

- Which closes lifted something (evidence frames), which were failed closes, where each release
  happened, how many objects, their names, the container names.
- Every cut you keep or move: state the frame and what happens there in the parent note when
  it was not obvious. A note is mandatory and must say what you saw, with frame numbers.
- Every atom at least 0.5 s (unless the parent is shorter); atoms tile the parent exactly.
- Confidence "unsure" whenever frames plus dense strips plus closeups did not resolve a cut,
  an identity, or whether an object was held. Say what is unresolved.

## When you finish the batch

Report, per episode: parents reviewed, atoms written, cuts moved from the proposal, cycles
merged or split relative to the proposal, quality values set and why, new mistake events, unsure
parents with the reason. Then the canonical names of your batch. Then anything that worried you
(a parent whose reviewed text contradicts the frames, an event that seems misplaced, a sheet
that was unreadable). Do not summarise the guide back.
