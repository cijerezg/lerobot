# EE mixture loss: TODO plan

**Status:** Phase 1 done for ReBot and both Panda variants (2026-10-09, assets frozen); Phase 2 done; Phase 3 built 2026-10-09, awaiting its first end-to-end run. Decisions are final unless marked "to confirm".
Supersedes [implementation.md](implementation.md) and [loss.md](loss.md) where they differ.

Decision record (chat, 2026-10-08/09):

- Options page: https://claude.ai/artifact/SYAfEQT4Ej3NUm7PxgpVZ8
- Spec page: https://claude.ai/artifact/7Z5PpiCyymSGTFwF533ck8

## Guiding principle

Reasonably rigorous where it is easy. Things will not match perfectly and small consistent
offsets are tolerable, but every asset, constant and convention that can be checked against an
independent source or measured on the hardware with modest effort gets checked or measured.
"Consistent offset, fine" is reserved for what is genuinely hard to pin down.

## Decisions

| Item       | Decision                                                                                                                                                                                                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loss       | option 3: hand term + FK term + lambda \* joint term, weights 1, 1, 1; lambda = 0.1 ablation                                                                                                                                                                                                            |
| Hand delta | anchor minus action, same as joints: `h_t = (p0 - p_t, log(R0 R_t^T)^v, g0 - g_t)`                                                                                                                                                                                                                      |
| Units      | position m, rotation vector rad, aperture m; global per-step stats pooled over robots, one scale per block; block means so each block is 1 at chance                                                                                                                                                    |
| FK term    | FK(predicted joints) vs FK(demo joints), absolute poses, anchor cancels; rotation block = 0.5 \* Frobenius^2, no log map; on the implied clean sample `x_hat = x_tau + (1 - tau) v_hat`, natural (1 - tau)^2 weight kept                                                                                |
| Expert     | one token per step, action vector = joints (padded) + 7 hand dims, one input and one output projection                                                                                                                                                                                                  |
| FAST       | masked: hand block and joint block tokenized separately, each with its own start token, neither attends to the other, shared positions. A source without joints (MolmoAct) has no joint block at all: its suffix is the hand start token and the hand tokens, and it contributes no joint cross-entropy |
| State      | joints + hand pose from our FK: position 3, first two rotation-matrix columns 6, aperture 1; discretized with the same 201-bin scheme and the same state tokens as the joints; same dropout treatment as joints                                                                                         |
| Hand frame | fingertip midpoint, common axes (z approach, y closing), one fixed tool transform per robot                                                                                                                                                                                                             |
| Aperture   | meters, per-source polarity and stroke                                                                                                                                                                                                                                                                  |
| Init       | base MolmoAct checkpoint if its pretraining action space is EE deltas (to confirm); otherwise current                                                                                                                                                                                                   |

## Phase 1: kinematics assets, FK and IK

**Done 2026-10-09** for `rebot_b601_follower`, `panda_franka_hand` (fmb) and `panda_robotiq_2f85` (droid,
droid_success): assets in `model/kinematics_assets/`, FK/IK in `model/fk/`, roundtrip script and numbers in
each README. UR7e, ARX5, YAM still to do.

Goal: every robot's kinematic assets in one place, one FK and one IK per robot, and a roundtrip
check that says the assets are right. Nothing downstream starts before this phase closes.

**1a. Assets in one place.** `lerobot/src/lerobot/model/kinematics_assets/<robot>/` holding the
URDF (or DH table), the tool transform to the fingertip midpoint, the gripper map (raw value to
meters, polarity), and a `README` stating the joint order and sign convention as stored in the
corpus and where the asset came from.

| Robot                                       | Asset today                                                                                                                                          | To do                                                                                                                                                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ReBot (`rebot_b601_follower`)               | `robots/rebot_b601_follower/urdf/reBot-DevArm_fixend.urdf` (also RS/DM variants under `migration/rebot_public_datasets_2026-09-20/source_evidence/`) | copy the one that matches the arm; confirm joint order shoulder_pan..wrist_roll and signs against the live arm; measure the fingertip-midpoint offset; calibrate gripper stroke in meters      |
| Panda, Franka hand (FMB)                    | modified DH + flange + hand TCP in `annotation/precision_contact/panda_fk.py`                                                                        | copy the table in; hand TCP already at 0.1034 m, yaw -45 deg; stroke 0.08 m                                                                                                                    |
| Panda, Robotiq 2F-85 (DROID, droid_success) | same DH                                                                                                                                              | own tool transform for the Robotiq fingertip midpoint; stroke 0.085 m; command 0 open 1 closed                                                                                                 |
| UR7e (RoboChallenge `ur7e`)                 | none                                                                                                                                                 | UR DH parameters for the UR7e from Universal Robots; gripper model and stroke to confirm from the source; polarity from wrist frames                                                           |
| ARX5 (RoboChallenge)                        | none                                                                                                                                                 | URDF from the ARX5 SDK repo; gripper already in meters                                                                                                                                         |
| YAM                                         | none                                                                                                                                                 | URDF from the i2rt YAM repo; stroke from spec or measured; 0 open 1 closed                                                                                                                     |
| MolmoAct (pose only)                        | none, no joints                                                                                                                                      | not a kinematics asset: an Euler-to-matrix convention (axis order, intrinsic or extrinsic) plus a tool transform from MolmoAct's EE frame to the fingertip midpoint; Franka hand stroke 0.08 m |

**1b. FK.** `lerobot/src/lerobot/model/fk/`: one pure-torch chain per robot built from the asset,
`fk_r(q) -> (p, R, g)` at the fingertip midpoint, batched and differentiable. Not placo
(`model/kinematics.py` is placo-based, not differentiable through torch). Registry keyed by the
corpus `robot_type`.

**1c. IK.** Damped least squares on the same chain, seeded at a given joint vector, iterating to a
target pose with the FK Jacobian (autograd). For 6-DOF arms seeded at the current joints it returns
the branch it started on. For the Panda it returns one point on the null-space line nearest the seed.
Not used in the loss; used for the check below and later as a tool.

**1d. Roundtrip check, the acceptance test.** Script `scripts/kinematics_roundtrip.py`: for each
robot, take 200 recorded frames spread over its episodes, run `q -> fk -> ik(seed = q + noise) -> q'`
and report per robot: pose error of `fk(q')` vs `fk(q)` (position mm, rotation deg), and joint error
`q' - q` where the arm is non-redundant. Pass: pose error at numerical precision, joint error at
numerical precision on the 6-DOF arms, and a table of frames where IK did not converge (near
singularities, expected rare). Also run `q -> fk` on the first frame of 20 episodes per source and
render the fingertip as a point over the global camera frame, reusing the trace renderer in
`annotation/rebot/traces.py` or `probes/action_trace_probe.py`. A wrong sign or offset shows there
at once. For DROID, additionally compare `fk(q)` with the source `cartesian_position` shard up to
the tool transform.

**1e. Convention record.** Each robot's README gets the roundtrip numbers and the rendered frame
check. Then the asset is frozen; the same files serve training and inference.

## Phase 2: targets, stats, state

Decisions taken while building (2026-10-09):

- **No offline hand targets.** The pose is a pure function of the joints and the row's
  `action_layout_id`, so it is computed in the processor (`policies/molmoact2/hand_block.py`,
  `MolmoAct2HandBlockProcessorStep` after `AnchorEncodeStep`) from the raw joints every batch already
  carries. No new per-episode files, no cache rebuild, one code path for training and inference.
  The stats script runs the same functions over every chunk.
- **Delta sign follows the code, not the spec text.** The joint encoding is `action - anchor`
  (`anchor_encoding._encode`), so the hand delta is `(p_t - p_0, log(R_t R_0^T), g_t - g_0)`.
- **Stats live in the per-layout artifact.** `compute_diverse_stats.py --hand` appends the hand
  columns (identical in every row: global per-step stats, one scale per block) and widens the
  artifact to 15/18. The artifact carrying `hand_block` is what turns the step on;
  `MolmoAct2Config.hand_block` must agree.
- **Band, not unit variance.** The QUANTILES normalizer maps [q01, q99] to [-1, 1] and the clamp
  step cuts there, so the hand columns use q01/q99 = mean -/+ 2.326 block scales (a Gaussian's
  1st/99th percentiles). That gives the hand block the joints' clip rate (about 2 %) and the
  joints' chance-MSE (about 0.18 per coordinate) instead of the spec's "1 at chance": every block,
  joints included, sits at the same chance level, which is what the equal weights need. Measured
  clip rates are logged by the script. Pooling is by chunk, so ReBot (97 % of the chunks with a
  chain) sets the scales; revisit if the Franka sources need their own weight.
- **Masks with a hole.** Joints keep prefix-valid padding inside slots 0..7; the hand block is
  valid or padding as a whole, so a 7-DoF row has a hole at slot 7. Width consumers that must see
  joints only (FAST tokenization, the deployed slice) use `action_layout.native_joint_widths`.
- **MolmoAct has no hand block yet** (its Euler convention is Phase 1 work still open), nor do
  ARX5, UR5, UR7e, YAM; those rows carry joints only, as today.

- [x] **Hand targets.** Computed on the fly, see above (was: stored offline).
- [x] **Hand deltas** in `hand_block.py` (after the anchor step): anchor pose from the anchor state, delta per step `(p_t - p_0, log(R_t R_0^T), g_t - g_0)`.
- [x] **Global per-step stats**: `compute_diverse_stats.py --hand`, pooled over the layouts with a chain, one scale per block, means per coordinate, written into every row of the per-layout artifact.
- [x] **Hand state**: slots 8..17 of the state (and of every history state), filled by the hand step; normalized by the artifact's hand state columns and discretized by the existing state-token path, so it gets the same tokens as the joints.
- [x] **Action layout**: `JOINT_SLOTS = 8`, hand block in 8..14, `native_joint_widths` for joint-only consumers; per-row hand validity from the layout's chain. MolmoAct's joint block is still its pose vector (unchanged); it has no hand block until its convention is set.
- [x] **Unnormalize-then-FK path**: `hand_block.hand_pose_from_normalized_joints` (`a = s0 + d`), tested to reproduce the hand targets on demo joints with finite gradients (`tests/policies/test_molmoact2_hand_block.py`).

## Phase 3: model and loss

**Built 2026-10-09, unit-tested, not yet run end-to-end** (no checkpoint or GPU in the build
session). Code: `policies/molmoact2/hand_loss.py` (the terms), `hand_block.py` (FK keys, monitor),
`modeling_molmoact2.py` (wiring, span isolation, vocabulary rows, hand span decode),
`processor_molmoact2.py` (hand FAST span, labels, positions), `configuration_molmoact2.py`
(`HandBlockConfig`). Tests: `tests/policies/test_molmoact2_hand_loss.py`.

Decisions taken while building:

- **Terms and weights.** `loss = hand + fk_weight * FK + joint_weight * joints`, every term a
  block mean, so each sits at the joints' chance level (the band normalization of Phase 2). The
  hand term is the mean of its three block means (position, rotation, aperture), not their sum.
  Rows without a chain (MolmoAct, ARX5, UR5, UR7e, YAM) have no hand block and no FK term, and
  their joint block keeps weight 1: it is their only action signal, and lambda is about making
  joints follow a hand they have.
- **FK term target.** FK of the demo joints (this page), not the stop-gradient predicted hand of
  `loss.md`. Residual in the hand columns' normalized band (per-step half-bands of the position,
  rotation and aperture columns, shipped by the hand step as `hand_fk_block_scale`): position
  `|dp|^2 / (3 s_p^2)`, rotation `0.5 |R_hat - R|_F^2 / (3 s_r^2)` (the squared angle for small
  rotations, matching the flow block's `|dr|^2 / (3 s_r^2)`), aperture `dg^2 / s_g^2`, mean of the
  three. Computed on `x_hat = x_tau + (1 - tau) v_hat`, so the `(1 - tau)^2` weight is natural.
  The FK runs on the model's own joint sample through `hand_pose_from_normalized_joints`; its
  inputs (anchor, layout, the row's joint q01/q99, FK targets) ride the batch as `hand_fk_*` keys
  written by the hand step from the stats artifact (`hand_loss.HAND_FK_KEYS`).
- **Feature widths.** `hand_block: true` widens the action feature to 15 and the state feature to
  18 in `MolmoAct2Config.__post_init__` and remembers the joint widths
  (`hand_joint_action_dim`, `hand_joint_state_dim`) for the deployed command and the per-joint
  limits. The trainer's width check and `val_loss` read the widened features; the actor passes
  the 15-wide chunk to the postprocessor whole.
- **Inference.** `MolmoAct2HandMonitorProcessorStep` sits after the unnormalizer: it compares the
  hand block with FK of the decoded joints (mm, deg, mm of aperture; `complementary["hand_monitor"]`,
  `step.last`, a log line every 20 chunks) and then trims the action and the anchor to the joint
  block, so the anchor decode and the width restore (now to the joint width) see what they see
  with the flag off. Tested: the executed joints equal the joint-only pipeline's.
- **FAST.** `hand.fast_layout`: `joints` (as before), `masked` (default), `hand_first` (ablation
  B). Masked: the answer is `<action_start> joints <action_end><hand_start> hand <hand_end>`, each
  chunk FAST-tokenized at its own width; the pack step ships `fast_spans` and `position_ids` with
  the hand span renumbered to the joint span's positions; the joint flow pass adds the cross-span
  block to the attention bias (`isolate_fast_spans`), so neither span attends the other; labels
  cover both spans; the action expert's encoder mask excludes both. A row without a hand block
  writes the joint span alone, a row without joints (none today) the hand span alone. Masked
  needs `action_mode: both` (the isolation lives in the joint pass); the FAST ordinal auxiliary is
  refused with a hand span.
- **Framing tokens.** `<hand_start>` / `<hand_end>` (configurable) are added to the tokenizer when
  absent, and the policy grows `lm_head` and the embedding's `new_embedding` rows to cover them,
  seeded from the `<action_start>` / `<action_end>` rows. **Not verified against the live
  checkpoint**: the first forward on the real model is the test (the grow path assumes
  `MolmoAct2Embedding.embedding` + `new_embedding`). The zero-risk alternative is to name two
  framing tokens the checkpoint already predicts in `hand.hand_start_token` / `hand_end_token`.
  A checkpoint trained without the hand span cannot be resumed into a run with it (the head
  changed shape); start from the base checkpoint.
- **Discrete generation.** The joint span generates as before. The hand span is decoded into
  slots 8..14 when the generated answer carries one (`_decode_hand_span`), but under the masked
  layout generation runs without the isolation mask, so that span is only meaningful under
  `hand_first`; it is never executed either way.
- **Ablation C** (`hand.hand_state: false`): the state hand slots stay zero and padding, the
  prompt renders the joint state tokens only (`prompt_state_width`), history states keep their
  hand slots zero.

- [x] **Expert input/output**: block-mean flow loss (`hand_loss.block_mean_flow_loss`), masked per row.
- [x] **FK term** (`hand_loss.hand_fk_term`), weight `hand.fk_weight`, skipped for rows without a chain.
- [x] **FAST, masked**: two spans, own framing tokens, isolation mask, shared positions, CE over both.
- [x] **Config**: `hand_block`, `hand.joint_weight` (lambda), `hand.fk_weight`, `hand.fast_layout`,
      `hand.hand_state`, `hand.hand_start_token` / `hand_end_token`. Knowledge insulation untouched.
- [x] **Inference path**: hand state from FK at the anchor (Phase 2 step); the output hand block is
      a logged monitor; the executed joints equal the joint-only path (tested).
- [ ] **Base checkpoint**: confirm MolmoAct's pretraining action space (EE deltas or not) from the repo or model card; pick the init accordingly. Not reachable from the build session.
- [ ] **First end-to-end run**: the vocabulary growth, the 18-wide state projector (fresh weights,
      as before) and the sequence budget with two spans are only unit-tested. Flip `hand_block`
      in `config_rl.yaml`, point `embodiment_stats_path` at the `-hand-` artifact, and watch the
      `hand_flow_*` / `hand_fk_*` metrics on the first steps.

## Phase 4: telemetry and probes

Probe design is revisited later. What runs from the first checkpoint:

- [ ] **Joint-vs-hand agreement, the basic telemetry.** At every validation: FK of the predicted
      joint block vs the predicted hand block, in the global hand units, per block (position mm,
      rotation deg, aperture mm). Needs no target, so it runs on every sample including rollouts.
      Logged as scalars per robot.
- [ ] **Same, against the target.** FK of the predicted joints vs the demonstrated hand, and
      predicted hand vs demonstrated hand, per robot. The two together say which output is wrong
      when they disagree.
- [ ] **Probe: agreement by robot and by frame.** Extend the telemetry into a probe in the suite
      (`probes/base.py`, `manifest.py`, viewer): distribution of the disagreement per robot, and the
      worst frames rendered with the global camera view and the two FK fingertips drawn, to see
      whether hard frames cluster (near singularities, large within-chunk rotations, gripper
      transitions, specific sources).
- [ ] **True eval, later.** Tasks demonstrated only on the Franka, run on ReBot, old vs new
      checkpoint. Not designed yet.

## Phase 5: runs

1. **Control**: today's loss, current data mix (already have checkpoints).
2. **Main**: option 3, masked FAST, hand state on, weights 1/1/1.
3. **Ablation A**: lambda = 0.1.
4. **Ablation B**: FAST hand-first instead of masked.
5. **Ablation C**: hand state off.

Each run: probe every checkpoint overnight (memory: chase_validate.sh), transfer eval on the best two.

## Order of work

Phase 1 first and fully, starting with 1a for ReBot and the two Panda variants, then 1b to 1d on those, then the remaining robots as their URDFs are found. Nothing downstream is touched until the roundtrip check passes per robot. ARX5, YAM and UR7e can land later; until then those sources contribute joints only, as today.
