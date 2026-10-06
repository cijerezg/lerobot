# Critic gradient report — v2, checkpoint 2000

[Interactive numerical report](index.html) · [Comparison with the previous critic](comparison.html).

Saved 2026-09-27. Same 8,776 frames, 136 episodes and 219 exact subtasks as the
previous checkpoint-2000 survey. All V and gradient measurements were recomputed.
The new critic uses projected continuous state and actual depth inputs.
Primary gradient: joint L2 over RGB features, projected current state and consumed
depth features **entering fusion**, after depth bounding. Text, metadata and missing
modality placeholders are excluded. Separate components remain available in the
frame explorer. These are feature derivatives, not physical state/pixel/depth derivatives.

The comparison includes matched-frame rank correlations, progress profiles at
10/20/40/80 bins, and 24 deterministic validation depth-null checks (21 real-depth frames, three missing-depth controls). The frozen
RGB/text encoder is exactly identical across the two checkpoints (416 tensors).
Combined gradient magnitudes involve different state representations; the RGB-only
comparison uses identical input coordinates. No causal attribution to architecture
alone, difficulty, failure probability or training quality is claimed.

HTML numerical data are embedded. Frame links and full CSV/JSON downloads use the
original output directory; retain it. This is a ReBot episode survey, not the
40/20/40 training sampling mixture.

```bash
xdg-open lerobot/docs/engineering_notes/reports/critic_gradients_v2_2000/index.html

PYTHONPATH=lerobot/src .venv/bin/python -m lerobot.scripts.view_probes \
  outputs/probe_runs/critic_gradients_v2_002000_train128_20260927
```

Select **Critic Input Sensitivity**. Source measurements, provenance, configuration,
logs, encoder equality check and depth intervention results are saved under that run.
No architecture or training weights were modified by this analysis.

Rebuild from saved measurements:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m lerobot.probes.critic_gradient_view \
  outputs/probe_runs/critic_gradients_v2_002000_train128_20260927/step_00002000/critic_sensitivity
.venv/bin/python migration/compare_critic_gradient_runs.py \
  outputs/probe_runs/critic_swap_002000_train128_20260926/step_00002000/critic_sensitivity \
  outputs/probe_runs/critic_gradients_v2_002000_train128_20260927/step_00002000/critic_sensitivity
node migration/summarize_critic_gradients.cjs \
  outputs/probe_runs/critic_gradients_v2_002000_train128_20260927/step_00002000/critic_sensitivity
```

These commands rebuild the live output; this docs snapshot is retained separately.

## Main observations

- Grasp gradient–progress Spearman: combined 0.239 → 0.313, but RGB-only
  0.447 → 0.358. The combined increase is not a uniformly stronger visual signal.
- Move gradient–V Spearman: 0.329 → 0.750.
- Across matched frames, old/new combined gradient ranks correlate at 0.459
  (RGB-only: 0.610); the top-10% sets overlap by 24.7%.
- Real depth is present in 6,940 of 8,776 frames. Missing-depth null features are
  excluded from the observation norm; depth-only plots omit those rows.
- Depth contributes a very small raw gradient norm, but substituting the learned
  missing-depth bank changes V in 18/21 real-depth validation checks (median absolute change
  0.0078, maximum 0.0859).
- In those same checks, 99.6–100% of depth-feature coordinates lie within 1% of
  the ±128 tanh bound. The probe measures gradients after bounding. This warrants
  a separate architecture investigation; it does not establish a disconnected
  depth path or quantify raw-depth sensitivity.

Source depth PNG grids are offset on some episodes; the existing dataset reader
used the nearest available depth frame, with 1–2 frame offsets recorded in the log.

Validation: all 8,776 component norms and frame-image paths checked; 28 Python
checks, shared numerical-statistics checks, and browser checks for six curves,
bin resolutions, component selection and links to exact frames passed.

## Policy check

[Policy checkpoint 1000: 20-frame depth saturation check](policy_depth_1000.md).
The median near-bound fraction is 0%, maximum 0.353%; this does not reproduce
the critic’s near-total saturation.

[Critic checkpoint 400: same 20-frame check](critic_depth_400.md). Median near-bound
fraction is 22.8%, rising to nearly 100% at checkpoint 2000 on the same frames.
