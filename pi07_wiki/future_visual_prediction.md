# Future visual prediction for RGB memory

This optional auxiliary trains the existing temporal RGB ViT to retain information
useful for predicting a frame four seconds ahead. It adds a training head; the
policy's temporal layers, language tokens, action head, and action-loss weights
keep their existing definitions.

## Enable or disable

The workspace-root `config_rl.yaml` contains the defaults under
`policy.future_visual_loss`. It is **disabled**. Set `enabled: true` to allocate
the auxiliary and sample future frames. Set it back to `false` to remove both the
auxiliary modules and future-frame reads, including when loading a checkpoint
that was trained with the auxiliary.

To test visual memory, also restore RGB history. This is a partial config edit:

```yaml
policy:
  future_visual_loss:
    enabled: true
    weight: 0.1
    horizon_seconds: 4.0
    target_update_steps: 100
    latent_dim: 128
    mistake_weight: 1.5
  memory:
    history_keys:
      - observation.images.external_0
      - observation.images.external_1
      - observation.images.wrist_0
```

The existing offsets `[-6, -4, -2]` seconds apply. State and depth history can stay
off; current state remains available to the policy. Leaving `history_keys: []`
is supported as the corresponding prediction-without-memory experiment.
Choose the starting checkpoint through `policy.pretrained_path` as usual.

This implementation is wired into `rl_offline.py` and the MolmoAct2 trainer for
LeRobot replay datasets, including multiple role-aligned ReBot sources. It uses
existing RGB caches. The source-native diverse actor cache does not expose future
RGB targets; enabling this auxiliary with `diverse.enabled: true` raises an early
error. Generic SFT and online replay setup are not wired for this objective.
Mistake metadata and trainable RGB ViT layers are required.

## Features and loss

The local MolmoAct2 checkpoint declares 27 ViT layers and executes layers 0–24.
Its configured output concatenates layers 24 and 18: two 1152-channel taps, or
2304 channels at each of 729 spatial positions. The predictor reads this same
output **after the existing temporal attention**, before image pooling and the
VLM projector. It receives neither state nor future images.

A small head maps 2304 → 256, applies two spatial transformer blocks shared across
cameras, and predicts 128 values at each spatial position. Spatial attention lets
a prediction use information from other locations in the current visual grid.

The target is a separate frozen copy of the RGB ViT, running each future frame
alone, with the same image preprocessing and feature taps. There is no temporal
attention in the target encoder. Both source and target features are normalized
per 1152-channel tap before prediction or projection.

For calibration pairs, form `delta = z_future - z_current` using this frozen
single-frame encoder. Compute the top 128 directions of the **uncentered** moment
`E[delta deltaᵀ]`. Not subtracting the mean change preserves consistent motion.
A static background cancels from these differences; camera motion can still
contribute substantial change. There is no whitening.

The prediction target is the **absolute future feature** projected into that
change subspace, `z_future @ basis`. It is not a prediction of the difference.
The loss is L1 after separate layer normalization of the predicted and target
128-vectors, averaged over patches and present cameras. Recorded mistake samples
get relative weight 1.5, with normalization by the valid sample weights. The
result is multiplied by `weight` and added to the existing losses. Mistake labels
do not modify the action loss through this objective.

## Target and PCA refresh

Every 100 actual actor optimizer updates, copy the RGB ViT weights to the target.
Gradient-accumulation microbatches do not advance this counter. Re-encode the
calibration image pairs with the refreshed target, recompute PCA, then align its
basis to the preceding basis with orthogonal Procrustes. This removes arbitrary
sign changes and rotations; it cannot eliminate a real change in the subspace.

Calibration uses a bounded reservoir of 16 current/future image pairs, drawn
across training batches on rank zero. Keeping images permits recalibration under
the new encoder without mixing old and new feature spaces. Up to 64 spatial
positions per present camera contribute to the moment. Target encoding processes
at most two images at once. These sizes are configurable in `future_visual_loss`.
The refreshed basis is broadcast to all DDP ranks.

The target, basis, reservoir, and optimizer-step counter are checkpointed. A
checkpoint without an auxiliary starts it fresh. If calibration has insufficient
samples or zero change energy, the auxiliary contributes zero and retries as
calibration data arrives. It also pauses after a degenerate refresh instead of
using a stale basis with a new target encoder.

Future targets cannot cross an episode boundary, truncation, or the replay write
head. Samples lacking a full four-second future are excluded. The horizon must
land on the cache's image-stride grid. Default DrQ uses the same crop shift for
current, historical, and future images within each camera; custom augmentation
requires disabling DrQ for this objective.

## Read the experiment

Aim records `future_visual_loss`, `future_visual_ready`, valid fraction, target
feature standard deviation, temporal change energy, retained PCA energy, and the
last target-update step. A low auxiliary loss alone is not evidence that memory
helps, and delayed targets do not guarantee against collapse.

With `val_loss_frames > 0`, the fixed held-out sample also records:

- `val_loss_future_visual`: prediction loss.
- `val_loss_future_visual_persistence`: copy the current single-frame target
  features into the future, with the same projection, loss, masks, and weights.
- `val_loss_future_visual_zero`: predict zero, with that same loss and weighting.

Validation does not update the target or PCA. Compare the history-enabled and
history-disabled experiments on held-out action behavior as well as these
baselines. Improved prediction still does not establish recovery from mistakes.
Inference does not run the predictor or target; loading with the switch off also
avoids their memory cost. Training pays for an extra frozen ViT and future-frame
encoding, plus periodic calibration.

`tests/policies/test_molmoact2_future_visual*.py` cover gradient routing, target
refresh/resume, PCA alignment, mixed precision, degenerate calibration, processor
isolation, and two-process DDP synchronization. Replay tests cover boundaries,
missing cameras, and shared augmentation. `scripts/smoke_future_visual.py` runs a
bounded synthetic forward/backward using local trained ViT weights without
writing a checkpoint; its module docstring includes the command.

## Implementation verification (2026-09-07)

The focused suite passed **52 tests**, including the two-process CPU/Gloo check:

```bash
# From lerobot/ inside the workspace:
../.venv/bin/python -m pytest \
  tests/policies/test_molmoact2_future_visual.py \
  tests/policies/test_molmoact2_future_visual_distributed.py \
  tests/rl/test_future_visual_replay.py \
  tests/rl/test_molmoact2_loss_logging.py \
  tests/policies/test_molmoact2_mem_encoder.py \
  tests/policies/test_molmoact2_depth_gripper_event_loss.py -q
```

The synthetic GPU smoke passed with the saved `molmoact2_rebot_nohistory_v1`
step-1200 ViT, the actual 2304-channel taps, bf16 autocast, gradient checkpointing,
128-dimensional PCA, backward into image history, and target refresh. This is a
component integration check, not a full-policy training run or evidence of
improved mistake recovery. The root policy YAML parsed with the switch both on
and off.

A broader replay run yielded 74 passed and 8 failed (3 dataset export/import tests
excluded). The failures were in `tests/utils/test_replay_buffer.py`: two exact
float equality assertions against bf16 storage, single-row sampling, chunk
sampling/alignment, and two augmentation tests requesting a 50-frame chunk from
a one-row buffer. These failures occur in existing storage/sampling logic before
the new future-frame branches. An in-memory reconstruction with those branches
removed reproduced seven; the eighth passed on that repeat. The existing chunk
check reads uninitialized capacity rows in that one-row fixture, making that case
unstable. Those broader buffer issues were not changed by this work.

## Fixed mistake / clean case browser

`future_visual_cases` adds a small, annotation-selected case study on the held-out
set. For `outputs/rebot_val-annotated-v4`, it selects 15 fixed anchors: episode 3's
drop and two failed closes, a failed-close sequence in episode 2 with recorded
after-span frames, clean same-subtask grasp counterparts, a clean release, and
clean transport. The release counterpart has the same verb but a different
object; the browser labels that weaker match. Clean comparisons have no mistake
labels anywhere from the earliest lookback through the future target.

Each recorded event gets anchors two seconds before, near its midpoint, and two
seconds after its end, snapped to the episode-local image grid. Overlapping
mistakes stay visible: **after a span does not imply recovery**. Two selected late
episode-3 anchors lack a complete +4-second target; their prediction error stays
unavailable. The after-frame for the last mistake is outside the recording and
is listed as omitted. No tail clamping invents a future or recovery.

The generated preview is at
`outputs/probe/rebot_val_v4_cases/future_visual_cases/cases.html`. It contains real
lookback/current/future RGB, but no model measurements. Recreate it from the
workspace root without loading a model:

```bash
.venv/bin/python -m lerobot.probes.future_visual_cases \
  --config config_rl.yaml --preview \
  --probe_parameters.output_dir=outputs/probe/rebot_val_v4_cases
```

To inspect a trained checkpoint, omit `--preview` and supply
`--policy.pretrained_path=outputs/<run>/checkpoints/<step>/pretrained_model`.
The standalone probe loads the checkpoint's saved architecture, memory, and
auxiliary settings. A missing or untrained future head is reported as unavailable;
an absent history input is reported separately. It does not initialize PCA from
validation examples. Attention remains inspectable on a memory checkpoint that
has no future head.

For the existing periodic validation loop, set
`probe_parameters.enable_future_visual_cases: true` and enable its usual validation
cadence. `future_visual_case_episode: 3` is the recorded, zero-based episode ID.
The switch defaults off. Cases depend only on annotations and the horizon/history
settings, never on the checkpoint's scores. `cases.json` records all anchors,
subtask/event labels, unavailable targets, and an annotation SHA-256 digest.

The browser lets you select a case, camera, and temporal layer. Spatial overlays
show current-query attention to each historical age at the same patch location;
layer bars summarize those weights. Prediction and target colors share a fixed,
seeded three-dimensional projection and scale. They are feature visualizations,
not generated images. The error map uses all latent channels and the auxiliary's
normalized L1, with **no mistake multiplier or training coefficient**, so mistake
and clean cases remain directly comparable. Hovering a patch displays its error
and attention allocation. A copy-current feature baseline is reported beside the
prediction error.

Each model case also exports an NPZ with full predicted/target vectors, per-patch
errors, per-head temporal attention, spatial age maps, and weighted value
magnitudes. The capture runs only the existing ViT path, without the VLM/action
sampler, gradients, or target/PCA updates. HTML is self-contained apart from the
optional NPZ download links. The preview was exercised in local Chromium; model
controls were exercised with explicitly synthetic readouts kept outside the
validation preview.
