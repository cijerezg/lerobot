# FMB multi-object visual review (protocol of 2026-09-17)

Review protocol for the 60 trajectories of
`lerobot/examples/dataset/diverse_robot_dataset/fmb_multi_production.json`, applied with
`annotate_fmb_critic.py` in the same folder. It predates quality v2: the grades below are the per-primitive
v1 scale; span grading follows [quality_mistake_rubric_v2.md](quality_mistake_rubric_v2.md). Review origin
is `model_reviewed`, quality origin `model_reviewed_rebot_rubric`. A model decision is never recorded as a
human decision.

## Identity and evidence

- Identity is the full source path. Filenames use `board_N_trajectory_X_Y` from
  `annotate_fmb_critic.review_stem`. The board qualifier is required; never pair a sheet and a review by
  `Path(source_path).stem`.
- One sheet band per native primitive; rows side_1, side_2, wrist_1; columns carry source frame indices and
  nominal 10 Hz times. Read every camera row of every primitive, as cropped bands at full resolution.
- Black tiles after a short primitive are padding, not a failed camera. side_1 mostly shows the destination
  board, side_2 the source and fixture.
- Inspect the full trajectory before grading, so grasp, placement, fixture release, regrasp and final
  insertion are judged in hindsight. A primitive named `insert` is not assumed to have succeeded.
- A deliberate fixture release and regrasp is part of the task, not a failure. Regrasp is a planned source
  primitive.
- Every judgment cites source path, primitive bounds, camera and frame indices. Duration, path inefficiency
  or a gripper transition can ask for a dense second look; none assigns a grade. No events are invented
  between widely spaced samples.

## Decisions

- Native primitive bounds are preserved: `start_timestep`, `end_timestep_exclusive`, `primitive`,
  `boundary_provenance=source_native`.
- Completion / interruption and critic eligibility are recorded separately from quality. Pending or unclear
  primitives stay ineligible with a reason. The staging template's `clean_complete`, `critic_eligible=true`
  and `review_status=complete` are placeholders, not evidence.

| quality | per complete primitive |
|---|---|
| 5 | direct |
| 4 | one minor correction |
| 3 | repeated corrections, no discrete failure |
| 2 | one failure recovered within the primitive |
| 1 | several failures, or one not recovered within the primitive |

Recovery in a later primitive does not raise the earlier one to 2. No quotas, no all-5 default.

Events: only visible `failed_close`, `slip`, `drop`, `knock`, `wrong_target`. An event starts at the
committed attempt and ends about 0.3 s (3 native frames) after the failure is observable; the recovery is
excluded; the span is clipped to its primitive.

Label schema: per primitive `quality`, `note` and `mistakes`, each mistake with `start_timestep`,
`end_timestep_exclusive`, `mistake_type`, `note` (not the household `mistake_events` seconds format). The
validator requires matching source and primitive sets, a note, quality 1-5, no events at quality 3-5,
exactly one at 2, at least one at 1. Unknown quality stays in a separate pending artifact. A partial batch
is validated against a review directory holding exactly that batch.

## Apply

- Eligibility decisions first, then validate, then `apply`. `apply` attaches quality and mistakes only; it
  does not turn pending eligibility into reviewed.
- Do not use `finalize-production` for these trajectories: it materialises the 2026-08-31 single-object
  decisions with all-5 notes.
- The v1 / v2 stores share hardlinks: never rewrite episode arrays or metadata in place.
