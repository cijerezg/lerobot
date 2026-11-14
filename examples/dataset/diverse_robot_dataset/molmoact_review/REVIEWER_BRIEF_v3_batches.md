# Reviewer brief: MolmoAct v3 round-2 batches (household 6-11, tabletop 0-1), 2026-09-19

You are one reviewer of one batch. Workspace `/home/user/Documents/Research/RL/LeRobot`; every
Python command is `uv run python ...` from that root; never install anything; never run git.

## Read first
1. `outputs/_annotation/diverse_v2_expansion_review/REVIEW_PROTOCOL.md` (the rubric; binding).
2. `lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/write_household_batch5.py` (the exact
   decision format you will write) and `batch_writer.py` (the assertions your file must pass).

## Your inputs
- `outputs/_annotation/diverse_v3_review/<component>/batches.json`: index `<N>` is your list of
  episode indices (18-19 episodes).
- Sheets: `outputs/_annotation/diverse_v3_review/<component>/episode_XXXXXX.{primary,secondary,wrist}.jpg`
  (1600x988, 12 tiles each with `t=` stamps, header = episode index, duration, camera, instruction).
  Read ALL THREE sheets of EVERY episode with the Read tool before deciding. Never grade a sheet
  you did not read.

## Supplemental sheets (render them yourself whenever the 12 tiles leave the outcome open)
    uv run python lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/render_terminal.py --component <component> --episodes E1 E2 [--seconds 2] [--end-s T]
        -> outputs/_annotation/diverse_v3_review/<component>/terminal_claude/episode_XXXXXX.{primary,secondary,wrist}.terminal.jpg
           (12 samples over the last 2 s of every camera, including the last source frame; decides release / placement)
    uv run python lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/render_dense.py --component <component> --episode E --camera <primary|secondary|wrist> --start A --end B --stride 0.25 [--tile-width 500]
        -> outputs/_annotation/diverse_v3_review/<component>/dense_claude/episode_XXXXXX.dense.<camera>.<A>-<B>s.jpg
           (--tile-width 500 for knobs, levers, switches, small objects; several pages if > 12 tiles)
Read every supplemental sheet you render, cite it in `evidence` ("dense secondary 6.9-7.8 s ...",
"terminal wrist 11.3-13.3 s ...") and list it in `extra_sheets` (path relative to the component
dir, e.g. `dense_claude/episode_000568.dense.secondary.06.00-09.30s.jpg`). Rendering is cheap; an
unread outcome is not. If the supplemental sheets still do not show it, withhold.

## Decision families seen in batches 0-5 (apply the same standard)
- `reviewed` (accepted): quality 5 direct / 4 one minor correction / 3 laboured with no event /
  2 one recovered event / 1 two events or unrecovered. Events require quality <= 2; quality <= 2
  requires events. Idle time is not a correction. `confidence` must be 'sure'.
- `withheld_unclear_outcome` (accept=False, quality=None, outcome='unknown', confidence='unsure'):
  the terminal/dense sheets do not show the completion. Known families: sanitizer pumps (never
  visibly pressed), toaster levers (lever off-camera), faucet/knob turns where the wrist looks away
  and the lever is unchanged, drawer/sink/table placements where the episode ends with the object
  still in the closed gripper (render the terminal sheet; if the last frame still shows the object
  fixed at the bottom edge of the wrist frame with no opening, withhold).
- `withheld_camera_integrity` (accept=False, quality=None, outcome as seen, confidence='sure'):
  a stream is frozen (the wrist shows one identical frame for the whole episode: the pillow
  session, v3 36/41/97/125/141) or otherwise unusable even when the externals show a success.
- `withheld_instruction_mismatch` (accept=False, quality=None, outcome='unknown',
  confidence='unsure'): the instruction names a different object/target than what is handled
  (v3 4991/4992 toy truck vs cat, 3353 mug vs scissors). If the instruction is merely truncated or
  understates a task that IS done, accept and give `task=` a one-sentence imperative instead.
- `rejected_person_in_workspace` (accept=False, quality=None, confidence='sure'): any hand/arm/body
  in any camera during the episode, even a clean success.
- `rejected_task_not_performed` (accept=False, quality=None, outcome='failure', confidence='sure'):
  the end state equals the start state or the wrong thing was done and not corrected.
- Also allowed when seen: `rejected_uncontrolled_motion`, `rejected_collision`.
Failures with clean behaviour that ARE visible (object dropped outside the target and left there)
are accepted with outcome='failure', quality 1 or 2 and the events listed.

## Evidence and notes
Every claim cites sheet + tiles or times: "grasp: wrist t3-t5 (1.6-2.9 s) ...; carry: secondary
t9-t11 ...; outcome: primary t12 ...; No person, no collision[, no correction]." Notes: one line
on what you saw; mandatory for anything below 5 or not accepted; say "Withheld:" / "Rejected:"
first for those. Match the density of batch 5.

## Output (the ONLY file you write besides the sheets you render)
`lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/write_<component>_batch<N>.py`, exactly
the batch-5 pattern (`D[ep] = dict(...)` per episode, then `write_batch('<component>', <N>, D)`),
docstring "<Component> v3 batch <N> (batches.json index <N>). Reviewer <your agent label>'s
explicit visual decisions, 2026-09-19." Then run it:
    uv run python lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/write_<component>_batch<N>.py
It writes `outputs/_annotation/diverse_v3_review/<component>/verdicts_batch<N>.json` and prints
the summary; it refuses to overwrite, so if it fails on an assertion fix the script and rerun
(delete only the verdicts_batch<N>.json YOUR run created, never any other file). Do not write
into `outputs/_annotation/diverse_v2_expansion_review/`, `outputs/diverse_robot_dataset*`, or any
other file.

## Report back
The summary line the writer printed, then a table: episode | accept | status | quality | events |
one-line reason, then a list of the episodes you found hardest (and why) so the coordinator can
re-read them.
