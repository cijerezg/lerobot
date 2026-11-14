# FAST roundtrip pre-flight: MolmoAct end-effector row vs the joint-space layouts

2026-09-19. Numbers in `fast_roundtrip.json` (same folder).

> Saved 2026-11-13 from the scratch folder `diverse_v3_control_mode_2026-09-19/fast_roundtrip`. The
> script (`fast_roundtrip.py`, ~10 s, no model) was not copied: it reads the v2 corpus and the -v3 stats
> file of that date.

## What was measured

200 anchors per layout (seed 20260919) from the v2 training selection
(`select_actor_anchors` over `outputs/diverse_robot_dataset_v2`, holdouts excluded, the same rows
the actor cache stores) plus 200 ReBot chunks from `outputs/rebot_all-annotated-v1` (image-stride-
aligned 30-frame windows inside one episode, as `ReplayBuffer.sample` draws them). Each chunk goes
through the trainer's own preprocessor step objects: `MolmoAct2UnifiedLayoutProcessorStep` ->
`AnchorEncodeStep` (a - s0) -> `MolmoAct2MaskedNormalizerProcessorStep` (per-layout, per-timestep
q01/q99 row from `action_stats_anchor_diverse-rebot-v3.pt`, row = `action_layout_id`) ->
`MolmoAct2ClampNormalizedProcessorStep` -> `_tokenize_discrete_action` at native width (DCT, x10,
round, snap, BPE; `outputs/MolmoAct-FAST-tokenizer`). Decode is the tokenizer's own (BPE ->
bins -> coefficients -> inverse DCT). Error is against the tokenizer's input (the clamped
normalized chunk); physical units come back through the same stats row (the inverse quantile map
is affine per timestep and dimension). -v4 stats do not exist yet; -v3 is the current corpus'
artifact and the MolmoAct row will not change in -v4 (same rows, same selection).

## Per-layout roundtrip

| id | layout | mode | rel L2 (all chunks) | per-chunk p50 / p90 | energy kept | tokens p10/p50/p90/max | nonzero DCT coeffs p50 (of 210/240) | chunks with a snapped bin |
|---|---|---|---|---|---|---|---|---|
| 0 | droid_franka_joint8_commanded | joint | 0.049 | 0.064 / 0.116 | 99.76% | 21/28/42/92 | 25 | 9.5% |
| 1 | droid_success_franka_joint8_commanded | joint | 0.046 | 0.054 / 0.114 | 99.79% | 20/28/47/87 | 25 | 10.5% |
| 2 | fmb_franka_joint8_measured | joint | 0.070 | 0.078 / 0.106 | 99.51% | 118/124/132/144 | 215 | 12.5% |
| 3 | robochallenge_arx5_joint7_measured | joint | 0.076 | 0.089 / 0.139 | 99.42% | 99/105/122/136 | 186 | 9.5% |
| 4 | robochallenge_ur5_joint7_measured | joint | 0.076 | 0.081 / 0.129 | 99.42% | 101/110/120/134 | 178 | 18.0% |
| 5 | ur7e_joint7_commanded | joint | 0.045 | 0.053 / 0.119 | 99.80% | 37/38/89/114 | 22 | 10.0% |
| 6 | rebot_b601_joint7_commanded | joint | 0.044 | 0.047 / 0.083 | 99.81% | 15/22/33/73 | 20 | 16.5% |
| 7 | molmoact_franka_ee7_measured | end_effector | 0.081 | 0.100 / 0.157 | 99.34% | 64/78/104/132 | 72 | 11.5% |

rel L2 = ||decoded - input||_F / ||input||_F over the whole normalized chunk; "energy kept" =
1 - rel L2^2. Rounding alone (no snap, no floor) gives the same rel L2 to the third decimal on
every layout (JSON `rel_l2_quantization_only`), so the alphabet fix costs nothing measurable.

Dropped coefficients: 0 on the deployed path (the snap asserts the count). 9.5-18% of chunks
contain at least one bin the shipped tokenizer cannot encode; the stock
`UniversalActionProcessor` deletes a coefficient from exactly those chunks (measured directly:
identical fraction per layout). No chunk reaches the bin floor (coefficient < min_token).

## Per-dimension error, layout 7 vs the joint layouts

Physical units: layouts 0-5 joints in radians, 6 in degrees (gripper raw `gripper.pos`), 7
translation metres / rotation radians / gripper 0-1 ratio; grippers 0/1 command ratio, 2 source
levels, 3/4 width metres, 5 ratio.

| id | group | dims | rel L2 | RMSE (normalized) | RMSE (physical) | signal RMS (physical) | max abs err (physical) | share of chunk energy |
|---|---|---|---|---|---|---|---|---|
| 7 | translation | 0-2 | 0.069 | 0.024 | 2.3 mm | 48 mm | 17.7 mm | 0.50 |
| 7 | rotation | 3-5 | 0.094 | 0.027 | 5.8 mrad (0.33 deg) | 85 mrad | 51.8 mrad (3.0 deg) | 0.34 |
| 7 | gripper | 6 | 0.087 | 0.030 | 0.013 | 0.125 | 0.066 | 0.16 |
| 6 | joints | 0-5 | 0.044 | 0.015 | 0.29 deg | 7.2 deg | 2.9 deg | 0.88 |
| 6 | gripper | 6 | 0.045 | 0.014 | 1.5 | 36.9 | 12.8 | 0.12 |
| 0 | joints | 0-6 | 0.050 | 0.015 | 6.7 mrad | 139 mrad | 57 mrad | 0.83 |
| 0 | gripper | 7 | 0.044 | 0.016 | 0.011 | 0.26 | 0.085 | 0.17 |
| 3 | joints | 0-5 | 0.073 | 0.029 | 9.2 mrad | 126 mrad | 70 mrad | 0.92 |
| 3 | gripper | 6 | 0.104 | 0.030 | 0.8 mm | 9.0 mm | 3.2 mm | 0.08 |
| 4 | joints | 0-5 | 0.074 | 0.030 | 4.0 mrad | 56 mrad | 28 mrad | 0.92 |
| 4 | gripper | 6 | 0.100 | 0.030 | 1.1 mm | 8.5 mm | 5.2 mm | 0.08 |

Layout 7 per dimension (x, y, z, roll, pitch, yaw, grip): rel L2 0.069 / 0.068 / 0.069 / 0.108 /
0.085 / 0.090 / 0.087; RMSE normalized 0.024 / 0.023 / 0.024 / 0.029 / 0.025 / 0.027 / 0.030.
Rotation is the worst group but only by 1.4x over translation; roll (0.108) and the gripper
carry the largest errors. No dimension dominates the way "Euler blows up" would look: rotation
holds 34% of the chunk energy and 45% of the squared error.

## Answer

An EE chunk keeps 99.3% of its energy through FAST (rel L2 0.081) against 99.8% for ReBot
(0.044) and DROID (0.049) and 99.4-99.5% for the measured-joint layouts FMB / ARX5 / UR5
(0.070-0.076). In physical terms the roundtrip moves a MolmoAct target by 2.3 mm RMS (max
18 mm), 0.33 deg RMS (max 3.0 deg) and 0.013 gripper ratio RMS (max 0.066) over a 1 s
horizon whose signal is 48 mm / 4.9 deg / 0.125 RMS. Layout 7 is the worst row, but it sits
with the other copy_state rows, not apart from them: the driver is measured-vs-commanded, not
end-effector-vs-joint.

## Surprises

1. **The error floor is the quantizer, and the copy_state rows sit on it.** RMSE in normalized
   units is 0.029-0.030 on every copy_state group (FMB, ARX5, UR5 joints and grippers, MolmoAct
   gripper) and 0.024-0.027 on MolmoAct translation / rotation, against 0.012-0.017 for the
   commanded rows (ReBot, DROID, UR7e joints). 0.1 / sqrt(12) = 0.0289 is the RMS error of
   rounding a coefficient to the 0.1 grid, i.e. the copy_state chunks carry broadband content
   above the rounding step in nearly every coefficient. Measured: DCT coefficient RMS in the
   upper half of the spectrum (k >= 15, t=0 artifact removed) is 0.036-0.041 on MolmoAct
   translation / rotation and 0.102 on its gripper, vs 0.005-0.013 on ReBot and DROID
   (one-off on the same rows; not stored by the script). That is measurement noise plus the
   15 -> 30 Hz linear interpolation kinks, and a step-like gripper. Consequence: 72 nonzero
   coefficients / 78 tokens per MolmoAct chunk vs 20 / 22 for ReBot, and the roundtrip error
   concentrates in t = 0..5 (per-timestep rel L2 0.12-0.15 vs 0.06 later; ReBot is flat at
   0.04-0.05).

2. **The anchor delta at t=0 is a broken target on every copy_state layout.** copy_state means
   a[0] == s0 exactly, so the -v3 stats row has q01 == q99 == 0 at t=0 and the normalizer's
   eps denominator maps the exact zero to -1 in all dims (layouts 2, 3, 4: 100% of values).
   MolmoAct (15 Hz) is worse in a different way: its timestamps drift off the 0.2 s anchor
   grid by more than `sample_action_chunk`'s 1e-6 s coincidence tolerance later in an
   episode, so a[0] is interpolated and the t=0 delta is a sub-micrometre residual; the stats
   row's t=0 quantiles are +-3e-8..1e-6, and the normalized t=0 target is noise anywhere in
   [-1, 1] on y, z, roll, yaw (98-100% of chunks outside +-0.1), -1 on the gripper (99%), ~0
   on x and pitch. Cost, measured by zeroing t=0 before FAST (JSON `layouts_t0_zeroed`):

   | id | tokens p50 deployed -> t0=0 | rel L2 deployed -> t0=0 | nonzero coeffs p50 |
   |---|---|---|---|
   | 2 FMB | 124 -> 76 | 0.070 -> 0.064 | 215 -> 73 |
   | 3 ARX5 | 105 -> 60 | 0.076 -> 0.066 | 186 -> 54 |
   | 4 UR5 | 110 -> 88 | 0.076 -> 0.071 | 178 -> 78 |
   | 7 MolmoAct | 78 -> 73 | 0.081 -> 0.079 | 72 -> 68 |

   The step at t=0 costs the 30 Hz copy_state rows 30-45% of their action tokens and about
   two thirds of their nonzero coefficients; for MolmoAct it is a minor token cost but the
   model is still trained to emit a meaningless t=0 value on five of seven dims. It is a
   stats / encoding question (t=0 span in `compute_diverse_stats.py`, or the coincidence
   tolerance in `sample_action_chunk`), outside this agent's files; flagged for the -v4
   rebuild. The ReBot row is untouched by any fix there (commanded != measured at t=0).

3. Clamp before FAST: 1.4-2.6% of normalized values and 15-27% of chunks hit the [-1, 1] clamp
   on every layout, ReBot included, which is the q01/q99 definition (2% tails) and not layout
   specific; it happens before the tokenizer and is not in the roundtrip numbers above.

## Files

- `fast_roundtrip.py` (probe)
- `fast_roundtrip.json` (all numbers, per layout: aggregate / per-chunk / per-timestep rel L2,
  per-dim and per-group errors in normalized and physical units, token and nonzero-coefficient
  distributions, snap / floor / stock-drop / clamp fractions, t=0 means, and the t0-zeroed
  counterfactual for 2/3/4/7)
