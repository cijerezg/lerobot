# Tip target: prompt clause + adapter (design 2026-11-14)

> **Status:** built 2026-11-14, inference path added 2026-11-15; on in the first v2 tube run (`policy.tip_target: true`), not yet tried on the robot.

Replaces the point-map depth tokens as the way depth reaches the policy. Depth is read by a
small adapter that answers one question: where, relative to the gripper tip, is the point of
the next gripper event. The answer enters the prompt as three state tokens.

## 1. Clause

Prompt order: `{depth}{subtask}{state}{velocity}{gripper_torque}{tip_target}`.

| case | text |
|---|---|
| target in the wrist view | ` The target is at <3 state tokens> from the gripper tip.` |
| not in view, or no event ahead | ` There is no target in the wrist camera view.` |
| root without targets (NaN row) | no clause |

The three tokens are the offset x, y, z (m, tool frame) binned like the state: range
`tip_target_low/high` = q01 / q99 over all `cfg.dataset.sources`, computed at launch and
pinned in the checkpoint's pack step.

## 2. Targets

`adhocs/tip_offset/build.py <root>` writes
`annotation_data/tip_offset/<root name>/targets.parquet` (+ `provenance.json`); a check sheet
goes to `outputs/analysis/tip_offset/<root name>/check.jpg`. Needs subtask labels.

Per frame, in hindsight: the next gripper event (grasp or release) of the frame's stretch,
the tip position at that event by forward kinematics, expressed in the frame's tool frame,
and its pixel in the 640 x 480 wrist image. `utils/tip_target.py` reads this into one row
per frame under `TIP_TARGET_KEY`:

`[in_view, x, y, z, u, v]`, zeros when `in_view` is 0.

A new root (train or validation) needs `build.py` run before it is used.

## 3. Adapter

`policies/molmoact2/tip_target.py`, 13M parameters (width 384, 6 layers), its own optimizer
group at lr 2e-4.

| input | source |
|---|---|
| wrist + top image tokens | the policy's vision backbone output, detached |
| wrist depth points | raw D405 depth frame, back-projected per patch (27 x 27), tool frame |
| subtask | token ids of the subtask text (16 max), from the pack step |

Output: one distribution over the 729 wrist patches plus "no target". The vector is the
probability-weighted mean of the depth points of the patches with a reading (`tip_vector`).
The label is a bump of width 0.75 patches around the target pixel, or "no target".

The image tokens are detached, so the adapter does not change the policy's gradients. The
adapter runs only when the batch carries the wrist depth frame (`TIP_DEPTH_KEY`); probe
batches render the clause from the row and skip it.

## 4. Training vs inference

| | clause is filled with |
|---|---|
| training, validation, probes | the true row from `targets.parquet` |
| actor (`build_inference_batch`) | the adapter's row `[in_view, x, y, z]`, `in_view = P(no target) <= 0.5` |

Inference is two-stage: pack once without the clause, `policy.predict_tip_target`, put the
row in the complementary data, pack again. Cost on the local GPU: about +22 ms per cycle
(13 ms vision encode + adapter, about 10 ms second pack). Both are redundant with the action
forward and can be removed if needed (reuse the image tokens, re-tokenize only the text).

The actor carries the wrist depth frame (`observation.depth.wrist_0`) whenever `tip_target`
is on, also with point-map depth off.

Not covered: `benchmark_inference.py` and subtask generation call `build_inference_batch`
without the policy, so they render no clause.

## 5. Open

- The policy trains on the true offset and runs on the adapter's. The gap is the adapter's
  error (`tip_vector_error_cm`, `tip_peak_on_patch` in the training log); the input swap
  switch `tip_target` measures how much the policy reads the clause.
- Targets exist only for the tube roots.
