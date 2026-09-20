# 07 — Data & annotation (reference)

## Datasets

- `outputs/rebot_socks_v1` — raw: 6 eps, 29,192 frames @ 30 fps, top + wrist RGB, wrist depth sidecar.
- `outputs/rebot-socks-annotated-v2` — **training-ready**: `subtask_index` per frame, 13-label canonical vocab, 81 hand-written summaries (12 s grid), quality/mistake metadata, depth hardlinked.
- Everything SO-101-era is legacy 6-dim; don't mix.
- **config_rl.yaml since 2026-09-09:** `rebot_socks_basket/shirts_bin/two_container-annotated-v3` (the 2026-09-08 audit
  revisions of the -v2 roots, frames hardlinked), `rebot_sorting_clothes_v4-multitask-annotated-v1`, and
  `outputs/rebot_rollouts-annotated-v2` (train source `rollouts`, 15 eps / 49,040 frames = the three 2026-09-08 inference
  episodes + the v1 set below); every root carries `meta/speed_hybrid_v1.parquet` (the adopted speed labels,
  `REBOT_SPEED_TABLE`), which `metadata_enabled` now requires alongside the two metadata parquets.
- `outputs/rebot_rollouts-annotated-v1` (train source `rollouts_0906`, 12 eps / 31,010 frames) + `outputs/rebot_val-annotated-v4`
  (val, 4 eps / 16,907 frames) — **the consolidated sets in config_rl.yaml since 2026-09-07**. The train set is the 11
  rollout episodes below plus one teleop demo (ep 11 = the v4-1 socks episode that was `rebot_val-annotated-v3` ep 3, flagged
  `is_intervention` True like every operator frame; 54 % of frames overall); the val set is `rebot_val-annotated-v2` (v1 ep 9,
  v2-1, v3-3 ep 0) plus the 18:55 rollout as ep 3, so it keeps four episodes and now contains mistakes. Built by
  `migration/consolidate_rollouts_2026-09-07.py` (library merge, videos stream-copied and pixel-checked against the sources,
  metadata rebased, depth hardlinked, `meta/provenance.json` per episode), sidecars regenerated, checked by
  `migration/verify_rollouts_2026-09-07.py`. The two sources below stay on disk untouched.
- `outputs/rebot_inference_2026-09-06-v6` (train) + `outputs/rebot_inference_val_2026-09-06-v2` (val) — sources of the above; **policy rollouts +
  DAgger interventions, training-ready**: 11 + 1 episodes, 29,229 + 2,886 frames from the 2026-09-06 online recorder runs
  (no-history MolmoAct2 ckpt). Train episodes 0-4 (11:28, 17:39, 18:45, 19:47, 19:54) are pure policy output; 5-10 (20:43 .. 22:06) carry teleop
  interventions (per-frame flag in `meta/online_labels.parquet`, a sidecar the loaders ignore; 52 % of train frames). Val =
  the 18:55 run (policy only, 96 s: sock and two navy-shirt carries to the bin, a beige-shirt drop, two failed closes; held
  out BECAUSE it has mistakes, for mistake-conditioned probes). Earlier versions deleted. Built by
  `migration/inference_audit_2026-09-06/build_dataset.py` (PLAN rows carry a train/val split; re-encodes the recorded AV1,
  copies low-dim verbatim, hardlinks depth; kept ranges are multiples of 3 on multiples of 3 so the depth phase survives)
  and annotated by `annotate_dataset.py <root>` (episodes matched to the plan by `meta/provenance.json`;
  `review_sheets.py` / `render_videos.py` render labelled sheets and top|wrist videos). Edits: every kept range starts 1 s (one action
  chunk) before the arm has left home by 10 deg on any arm joint (`stationary.py` prints the onsets: the 11:28 run loses its
  stationary 12.6 s prefix, the others 0-2 s); the 18:55 run cut at 96.2 s after its second failed close; 10:14, 10:24, 10:36, 10:47, 19:01,
  20:41 excluded. Task AND subtask labels describe what appears, not the prompt (typed eval-console labels discarded; the
  operator routed shirts to the basket and socks to the bin in the later runs, so e.g. "Put shirts in brown basket and
  socks in bin" is a task string and `move the red shirt to the basket` a label; a policy episode that puts the sock in
  the bin carries "Put sock and shirts in bin", not the prompt's "sock in brown basket" (user, 2026-09-07); an episode
  that moves only the sock carries "Put sock in brown basket"). Mistakes: failed closes from proprio (a close that
  reopens without carrying, operator or policy); `drop` where the gripper opens and the object stays on the table instead of
  ending up in, or hanging into, the container (release vs drop is the OUTCOME: a shirt end let go over the bin rim with most
  of the shirt on the table is a release, user decision); `slip` where it leaves a shut gripper. Operator frames are judged
  like policy frames (ep6's red shirt took six attempts: three drops). Neither a close on the wrong object (or on two
  objects) nor a carry to a container other than the task string's is a mistake (user, 2026-09-06): the grasp/move/release
  labels name what was grasped and where it went (v4's eight `wrong_target` rows were removed in v5). Vocabulary additions: `beige shirt`, `navy shirt` (= the blue shirt), and the
  shirt-to-basket / sock-to-bin labels. Train: 116 segments, 26 mistake rows (16 failed_close, 9 drop, 1 slip; 9.2 % of frames), quality frame share
  q1 15 / q2 14 / q3 19 / q4 30 / q5 22. Val: 13 segments, 3 mistakes (11 % of frames). Review material: `review/` (labelled sheets,
  per-carry strips, deposit close-ups) and `videos/` (2x, so every second frame is skipped by design). Source runs: one row
  per actor step, video frames == rows, state/action of the same step; the loop held 594-599 steps per 20 s window
  (571 in the first, warm-up), so a few 2/30 s wall-clock gaps exist that the synthetic timestamps do not show. Carry ends: the segmenter's -90 crossing when it comes within 6 s of the gripper
  opening 15 deg from its holding value, else the relative trigger (the policy's deposits open only 15-40 deg); a carry
  may fix its close-stop/release-end frames explicitly; a move under 1 s folds into the release.

Episode GT for eval sanity: ep0 white; ep1 white,black; ep2 white×2,black; ep3 +blue-heeled; ep4 +blue-striped, mistake ~162→185 s; ep5 interleaved, mistake ~176→205 s.

## Diverse corpus action layouts (`datasets/diverse_actor_selection.py`)

One row per source convention, keyed by `action_layout_id` (append-only; the per-layout
stats artifact and the buffer column index into it). `control_mode` (2026-09-19) is what
the leading channels are; it renders as the prompt clause `The control mode is joint
space.` / `... end-effector space.` right after the embodiment clause on every row
([06 §3](06_inference.md)). Anchor encoding is the same elementwise `action - state` for
every row; for layout 7 that is a pose displacement (metres, unwrapped Euler radians,
gripper ratio), well-defined because MolmoAct's action is its state (`copy_state`).

| id | name | source | embodiment | dim | action | gripper | control_mode |
|---|---|---|---|---|---|---|---|
| 0 | droid_franka_joint8_commanded | droid | Franka | 8 | native | command_0_open_1_closed | joint |
| 1 | droid_success_franka_joint8_commanded | droid_success | Franka | 8 | native | command_0_open_1_closed | joint |
| 2 | fmb_franka_joint8_measured | fmb | Franka | 8 | copy_state | source_gripper_pose_levels | joint |
| 3 | robochallenge_arx5_joint7_measured | robochallenge | ARX5 | 7 | copy_state | width_metres | joint |
| 4 | robochallenge_ur5_joint7_measured | robochallenge | UR5 | 7 | copy_state | width_metres | joint |
| 5 | ur7e_joint7_commanded | ur7e | UR7e | 7 | native | ratio_0_1 | joint |
| 6 | rebot_b601_joint7_commanded | rebot | Rebot B601 | 7 | native | ratio_0_1 | joint |
| 7 | molmoact_franka_ee7_measured | molmoact | Franka | 7 | copy_state | ratio_0_1 | end_effector |
| 8 | yam_joint7_commanded | yam | YAM | 7 | native | ratio_0_open_1_closed (see prepare_yam.py) | joint |

Layout 7 (v2): xyz metres + Euler triple (unwrapped per episode at ingest, state and
action) + gripper; no joint channel in either MolmoAct release. Layout 8 (v3): i2rt YAM
single arm, six joint radians + a real commanded gripper channel; the three Hub sets
(yam-pick-duster-200, yam-espresso, yam-pick-place) are brought to one gripper convention at
ingest. ReBot (6) is not part of the corpus; it holds an id so the mixture keys one stats
table.

## Chain (scripts in `data_processing/annotate/`)

Order matters: `summary_annotate.py` (12 s grid) → `subtask_annotate_grid.py` (4 s grid, conditions on summaries; labels atomic + progress-free) → `metadata_annotate.py annotate` + `review` (suspicion 0–10, `--threshold 4`; review UI writes the meta files). Validation sets: `--reuse-map <train-root>`. AV1 videos → ffmpeg, not cv2.

Key invariants:
- 4 s subtask grid = `subtask_regeneration_interval`; 12 s summary grid = 3:1.
- Post-annotation validation required (the empty-label → "Subtask: ;" bug class).
- v2 provenance: Gemma subtasks hand-corrected (91/242 changed, originals kept as `gemma_subtask`); summaries fully manual.

## Annotator verdicts (2026-07-18)

Gemma 4 31B: summaries unusable (counting echo loop); subtasks ~62% (misses releases, colors in clutter, fumbles→"move"). Upgrades untried: GLM-4.6V FP8 (DGX), Molmo2-8B.

## Per-new-dataset prep

annotate → norm stats + `compute_delta_stats.py --encoding anchor --chunk-size 50` → memmap cache `--image-stride 5` (must match config) → verify first steps: `loss_subtask_ce` nonzero, `subtask_index`/`summary_*_index`/`metadata_*` in batch, depth one-shot log fires.
