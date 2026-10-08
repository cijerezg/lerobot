# Probe documentation

The exact MolmoAct2 tensor locations, equations, and dimensions used by the probes are
documented in [MODEL_TENSORS.md](MODEL_TENSORS.md). Read that note before interpreting a
plot labelled with a layer such as `L14` or `L31`.

The important distinctions are:

| Probe family | Tensor read | Shape for this checkpoint | Meaning of `Lℓ` |
|---|---|---:|---|
| `conditions_matrix`, `domain_representations`, `subspace_spans` | Token-group mean of the language block output | `[B,2560]` per group | After the complete encoder block's attention, MLP, and residual updates |
| `conditions_matrix`, `domain_representations`, `subspace_spans` (`action`) | Horizon mean of the action-expert block output | `[B,768]` | After the complete expert block's self-attention, cross-attention, MLP, and residual updates |
| `attention`, `attention_budget` | Action-expert softmax weights | cross `[B,8,30,S]`; self `[B,8,30,30]` | Inside the attention sublayer, before value mixing, output projection, and the residual update |
| `embodiment_swap` | Integrated normalized action chunk | `[B,30,8]` exposed by the mixed policy | No intermediate layer is read; the intervention is measured after the full model and flow sampler |

Layer numbering is zero-based. For example, `L31` is the 32nd of 36 layer pairs. Attention
weights and post-block representations are different measurements; neither should be
described as the other. Attention weights show routing, while Jacobian probes measure
forward sensitivity.

## Key probe: subspace spans

`subspace_spans` is part of the official full validation suite, enabled by
`probe_parameters.enable_subspace_spans: true` in `config_rl_validate.yaml`.
The training configs expose the same switch, disabled alongside the other extended
probes. `run_probes.sh` dispatches it through the standard validation registry and
requires its report in the final completeness check. The probe viewer and checkpoint
comparison discover its `index.json` automatically.

This probe measures mean-centred (per robot) representation rank, cross-robot span overlap,
principal angles, held-out energy outside each span, and the frames defining it.
Numerical rank counts singular values above a fraction of the largest singular value;
a shared mean component can change that count without removing frame differences.
These are geometry measurements, not task-performance scores.

When conditions matrix is also enabled, it runs first and subspace spans reuses its
`conditions_matrix/cache` with no additional model forwards. When subspace spans is
enabled alone, it collects that same cache without running the conditions-matrix
analysis. Sampling remains controlled by the `conditions_*` settings. `mode: collect`
only captures; `mode: plot` reads the existing cache; `mode: all` captures and analyzes.
A missing or incompatible cache fails visibly rather than substituting another step.

| Setting | Default | Meaning |
|---|---|---|
| `subspace_text` | `real` | Cached text condition (`real` or `neutral`) |
| `subspace_tolerances` | `0.3,0.1,0.05` | Relative singular-value cutoffs on the centred spectrum; middle sorted cutoff is the headline. 0.1 keeps k between 1 and the frame count at nearly every layer; 0.05 is frame-limited for state and action_output below ~100 frames |
| `subspace_layers` | `14,15,16,28,32` | Detailed spectra, principal angles, and pivots; last layer is the headline |
| `subspace_n_null` | `100` | Random subspace pairs per dimension tuple |
| `subspace_n_pivots` | `12` | Frame examples shown in the explorer |
| `subspace_pivot_group` | `img_external_0` | Headline token group |

Ranks and overlaps cover every captured layer. The additional detailed layers bracket
the observed middle-depth drop. Reports are written to `subspace_spans/explorer.html`,
with `index.json`, `summary.json`, and CSV downloads alongside it. The standalone
`python -m lerobot.probes.subspace_spans --cache_dir ...` and `--report_only` paths
remain available for saved-cache analysis.

Flow inversion remains an optional curiosity experiment in
`migration/flow_inversion_2026-09-28/flow_inversion.py`,
with its existing `flow_inversion_report` renderer. It is not registered or enabled
in the official validation suite.

### Flow inversion reconstruction experiment

The same standalone inversion runner now accepts `--inv_roundtrip`. It holds each
frame's context fixed, inverts the processed demonstrated action, and continues the
normal forward Euler sampler from selected inverse times. It compares float32
accumulation with the deployment dtype, checks every full native continuation
against `_generate_actions_from_inputs_with_rtc`, and repeats on actions generated
from known seeded noise. Padding joints are excluded from errors; episode-end
repeated chunk targets are retained, as in the original probe.

```bash
PYTHONPATH=lerobot/src .venv/bin/python migration/flow_inversion_2026-09-28/flow_inversion.py \
  --config_path=<checkpoint-compatible-config.yaml> \
  --policy.pretrained_path=<checkpoint>/pretrained_model \
  --inv_out=<new-output-directory> --inv_roundtrip \
  --inv_roundtrip_steps=20,40 --inv_roundtrip_refinements=0,4 \
  --inv_roundtrip_times=0,0.25,0.5,0.75,1 \
  --inv_val_frames_per_episode=4 --inv_val_max_episodes=2
```

This mode evaluates validation frames only. Its report leads with mean round-trip MSE,
explains flow steps and inverse corrections, and includes frame errors, joint/step
heatmaps, restart-time plots, precision controls, and interactive action overlays.
MSE averages squared coordinate errors within each frame and then across frames;
it does not square the average RMS. Existing captures can be re-rendered without
model execution. It writes `roundtrip.html`,
`roundtrip.png`, frame metrics in CSV/JSON, a summary, and complete padded inverse
and forward trajectories in one NPZ per frame. Array keys encode target kind,
step count, refinement count, and restart step. Inverse arrays have shape
`[N+1, 1, chunk_steps, padded_action_dim]`, ordered from t=0 to t=1. Forward arrays
start at the specified k/N; t=1 is a cast-only control. The generated control's
original noise is also saved. Refinement is a fixed-point iteration and can fail
to converge; it is not claimed to exactly invert the finite-precision sampler.

Rebuild the report without loading a checkpoint:

```bash
PYTHONPATH=lerobot/src .venv/bin/python migration/flow_inversion_2026-09-28/flow_inversion.py \
  --inv_out=<output-directory> --inv_roundtrip --inv_analyze_only
```

### Shared 3D PCA of recovered inputs

`--inv_noise_pca` compares recorded actions, uniform `[-1,1]`, Gaussian with
standard deviation `1/sqrt(3)`, independent random `±1`, and Student-t(3)/3 targets.
It uses exactly `--inv_pca_points` paired observation contexts (150 by default),
balanced across validation episodes and interleaved for useful partial results.
Each action family gets one target per context. This mode does not run generated
controls or a sweep of solver settings.

```bash
PYTHONPATH=lerobot/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python migration/flow_inversion_2026-09-28/flow_inversion.py \
  --config_path=config_rl_validate.yaml \
  --policy.pretrained_path=outputs/rebot-diverse-v1-2026-10-04/checkpoints/001200/pretrained_model \
  --inv_out=migration/flow_noise_pca_diverse_v1_1200/results \
  --inv_noise_pca --inv_pca_points=150 --inv_num_steps=20 --inv_refine=4
```

The five inverse trajectories share an observation context and run as independent
batch rows; every recovered input is then checked individually with the production
sampler. `noise_capture.json` and one `noise_*.npz` per context are saved as collection
progresses. `noise_pca.html` is rebuilt after 5, 15, 50, 100, and all requested contexts.
It includes interactive 3D rotation, group toggles, static projections, explained
variance, full-dimensional input magnitude, and reconstruction MSE. Plotly is bundled
locally, so the page works without network access.

PCA uses the actual sampler inputs after the expert-dtype cast, with padding excluded:
one 210-dimensional vector per 30-step, 7-joint chunk. One pooled mean and one shared
PCA basis are fitted across all groups. There is no per-group centering, scaling,
whitening, outlier removal, or filtering on reconstruction MSE. The saved
`noise_pca.npz` includes full vectors, mean, basis, scores, and explained variance;
`points.csv` identifies every point and its actual reconstruction error.

To refresh the PCA page from all completed contexts without running the model:

```bash
PYTHONPATH=lerobot/src .venv/bin/python migration/flow_inversion_2026-09-28/flow_inversion.py \
  --inv_out=migration/flow_noise_pca_diverse_v1_1200/results --inv_noise_pca --inv_analyze_only
```

### Fixed-frame action interpolation

`--inv_interpolation` holds three own-ReBot observations fixed (validation episodes
0, 1, and 6, about 40% through each episode). It blends the recorded normalized
action to one uniform, Gaussian, random ±1, and heavy-tailed endpoint, plus a
uniform-to-±1 bridge. Alpha is the action blend fraction, sampled from 0 to 1 in
0.1 increments; it is separate from the solver's flow time. Shared endpoints are
inverted once, giving 50 unique targets per frame. `--inv_interpolation_episodes`
and `--inv_interpolation_batch_size` control frame selection and inverse batching.

For model-generated endpoints instead, use `--inv_generated_actions`. This draws
three standard Gaussian inputs in the production dtype, runs the production flow
sampler, and blends each resulting action with the recorded action. It produces
31 unique targets per frame, 93 total. Outputs stay in normalized action space:
there is no second normalization or postprocessing clamp. The report marks known
sampled noises alongside recovered endpoints and separately plots action MSE and
noise recovery RMS. Generation is not a guarantee of task success.

```bash
PYTHONPATH=lerobot/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python migration/flow_inversion_2026-09-28/flow_inversion.py \
  --config_path=config_rl_validate.yaml \
  --policy.pretrained_path=outputs/rebot-diverse-v1-2026-10-04/checkpoints/001200/pretrained_model \
  --inv_out=<new-output-directory> --inv_generated_actions \
  --inv_interpolation_episodes=0,1,6 --inv_interpolation_batch_size=10 \
  --inv_num_steps=20 --inv_refine=4
```

Both modes save `interpolation.json` and `paths_*.npz` after each completed frame,
then write `interpolation.html` with camera observations, shared 3D PCA, full-space
geometry, reconstruction errors, and action overlays. To render completed frames
without loading the model, pass `--inv_interpolation --inv_analyze_only` with the
same `--inv_out`; the renderer reads the experiment type from the capture.

For the matched synthetic-target checkpoint comparison, place the two captures in
`<comparison-root>/step_000200` and `step_001200`, then run:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m lerobot.probes.flow_interpolation_compare <comparison-root>
```

The comparison verifies identical targets and uses one PCA basis and common axis
ranges across both checkpoints. Geometry and MSE always use all 210 valid action
coordinates; PCA is only a visualization. Inversion uses velocity subtraction and
fixed-point corrections, with ordinary model forwards and no backpropagation.

### Shared, distance-matched endpoints

`--inv_distance_matched` compares other recorded chunks with model-generated
chunks at similar MSE distances from the three fixed anchor actions. It uses one
immutable set of three recorded and three generated endpoint tensors across all
anchors. Recorded endpoints preserve their source-frame displacement patterns,
using saved quantile normalization and the training clamp; absolute joint
positions are not transplanted between contexts. Source images, tasks, indices,
noise seeds, and actual matching discrepancies are shown in the report.

The recorded pool contains own-ReBot validation chunks at 3-frame intervals,
excluding each anchor's 60-frame neighborhood. The generated pool reuses the nine
saved actions from `migration/flow_generated_interpolation_rebot_1200`.
Selection greedily takes three distinct pairs with the smallest worst relative
endpoint-distance mismatch over the three anchors, without using inversion
outcomes. Selected recorded windows do not overlap. The current completed run's
worst mismatch is 6.96%; endpoints are never rescaled to force a match.

Use the same checkpoint/config/solver arguments as the generated-action example,
with `--inv_distance_matched` instead of `--inv_generated_actions` and a fresh
output directory. `--inv_distance_matched --inv_analyze_only` renders saved results.
`matched.html` is refreshed after every anchor. `shared_endpoints.npz` stores the
fixed endpoints and candidate-distance audit; `matched_*.npz` stores all 61 unique
targets per anchor, their recovered inputs, and actual production reconstructions.
Matching and geometry use the same 210 valid normalized coordinates.
