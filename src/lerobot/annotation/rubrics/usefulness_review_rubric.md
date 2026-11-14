# Usefulness review: which low-quality footage is cut from training (v2, 2026-10-18)

Decides the usefulness cuts: frames that stay in the roots but whose action targets are never drawn.
Results are the manifests in `lerobot/dataset_info/manifests/` (`rebot_exclusions_2026-10-18.json`,
`rebot_exclusions_seamfix1_2026-11-10.json`, `diverse_exclusions_2026-10-18.json`), read through
`diverse.rebot_exclusions_manifest` / `diverse.diverse_exclusions_manifest` in the run config. Idle (arm
still for 3 s or more) is cut separately by the idle scan and is not judged here.

## Unit and question

Unit = one subtask that holds at least one span graded quality 1 or 2
([quality_mistake_rubric_v2.md](quality_mistake_rubric_v2.md)). Question: should a policy be trained to
imitate the actions in this unit?

Principle: do not remove mistakes; remove behaviour that teaches nothing or that we never want copied. A
knocked bottle, a missed grasp, a drop are not low by themselves: the model should see what comes after.
When unclear, keep (user 2026-10-18: the first pass removed too much).

## Axes

| axis | values | reads as |
|---|---|---|
| predictability | 1-5 | could a policy infer the actions from the cameras plus the task text? Predictable: a reach toward the target; a second grasp after a visible miss, even from almost the same pose; a small adjustment of the jaws; a search that moves toward where the object went; a regrasp after a slip. Unpredictable: aimless pokes or sweeps; open-hand in-and-out cycles that never try to grasp; wandering away and back; motion driven by something off-camera; pauses or hovers with no visible reason; switching targets for no visible reason |
| recovery role | `recovery` / `attempt_in_recovery_chain` / `none` | `recovery`: the unit is itself a correction (re-approach, regrasp, reorienting, finding a lost object). `attempt_in_recovery_chain`: a failed try inside a sequence that ends in success or a clear correction, judged over the whole episode |

A failed attempt is predictable if a competent operator would plausibly have tried it at that point.
Eventual success does not rescue a long string of unchanged failures. A unit is neither rewarded nor
penalised for an unrelated action next to it.

## Score and verdict

| score | meaning | verdict |
|---|---|---|
| 5 | clearly useful skill or recovery (rarity or perfection not required) | keep |
| 4 | likely useful, with a caveat | keep |
| 3 | mixed: useful material plus waste, or a partial skill | keep, or trim |
| 2 | mostly unhelpful, maybe a small useful tail | remove, or trim to the tail |
| 1 | no useful skill | remove |

- **Keep**: predictable behaviour, or any recovery or attempt in a recovery chain, unless dominated by waste.
- **Trim**: a useful part plus an unpredictable or idle stretch of more than about 3 s. Give the range that
  goes, episode-local `[from, to)`.
- **Remove**: mostly unpredictable behaviour with no recovery value, or the same failed action repeated
  more than about 3 times with no change and no progress.
- `protected`: set on a unit that shows pouring or another behaviour that may be exempt from removal, with
  a note. It does not change the score.

## Calibration

| case | verdict |
|---|---|
| sock pulled out of the pile after a failed first try | 5 |
| toppled cup righted | 5 |
| cup approach corrected mid-reach | 5 |
| empty failed sock grasps, no change between tries | 1 |
| long search, then a short useful pickup | 2, trim the search |
| DROID IPRL ep5570 (unit_0099, unit_0100): about 25 s of open-hand pokes at a box in a container, fingers never close | remove (user 2026-10-18) |
| DROID IPRL ep5878 (unit_0101), bread slice: second close from a slightly adjusted pose that fails; the chain ends in a successful re-approach | keep, `attempt_in_recovery_chain` (user 2026-10-18) |

## Evidence

- Overview of the whole episode at 1 sample/s, top camera above wrist; every page is read.
- Every unit scored 1 or 2 has a dense read across its core, at least 2 samples/s; denser around a quick
  grasp, release or contact that changes the decision.
- No failure is called from gripper motion alone: the object outcome must be visible, else say uncertain.
- Old notes and grades are hints. Never score from text alone.
- Judge the whole recovery chain, including what comes after the unit.
