# Critic gradient report — checkpoint 2000

[Open the saved interactive HTML report](index.html).

This is the original architecture. See the [v2 report and comparison](../critic_gradients_v2_2000/README.md)
for the later critic with continuous state and depth. Architecture notes below describe the original implementation.

Saved on 2026-09-26. The HTML includes its numerical data and works without a
model, server, or the original summary JSON. Links to camera frames and full
CSV/JSON downloads still use the original probe output; retain that run directory.

## Open it

From the workspace root (`LeRobot/`, which contains `lerobot/` and `outputs/`):

```bash
xdg-open lerobot/docs/engineering_notes/reports/critic_gradients_2000/index.html
```

For the report and full frame explorer inside the probe viewer:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m lerobot.scripts.view_probes \
  outputs/probe_runs/critic_swap_002000_train128_20260926
```

Select **Critic Input Sensitivity**. Its first panel is the numerical report.

## What is saved

- Checkpoint: `outputs/molmoact2_critic_own_20260926/checkpoints/002000/pretrained_model`.
- 8,776 frames sampled at 1 Hz across 128 training and 8 validation episodes;
  219 exact subtask texts. These are the saved ReBot survey episodes, not the
  entire training corpus.
- Six curves: gradient against elapsed segment fraction, seconds remaining,
  V, signed ΔV, signed raw advantage, and absolute raw advantage.
- Action and transition filters; 10/20/40/80 bins; median, quartiles, mean,
  frame/episode counts, and median of within-bin episode medians.
- Expandable numerical tables, action correlation matrix, and exact bin links
  into the frame explorer. No smoothing. IQR describes spread, not uncertainty.

The current report uses the joint L2 norm over **RGB image patch features and
state-value embeddings entering critic fusion**. State values are the eight saved
`<state_N>` tokens; fixed state-clause wording, delimiters, task/subtask text,
metadata and depth/history placeholder tokens are excluded. The critic still
receives the original full prompt. These are embedding derivatives, not raw-pixel
or physical joint-coordinate derivatives. Joint norm = sqrt(image_norm² +
state_value_norm²), recovered exactly from saved disjoint token gradients.

New probe runs use this same image + state norm for primary plots, within-text
ranks and returned `grad_mags`. Saved records include `image_grad_norm`,
`state_value_grad_norm` and `image_state_grad_norm`; `grad_norm` retains its
original full-input meaning for comparison. Summaries declare
`gradient_scope: images_state`. The current restriction requires discrete
state-value tokens and rejects unsupported state formats rather than treating an
unfilled continuous-state placeholder as proprioception. A new architecture
must expose the actual consumed state features and its differentiation boundary.

**Depth:** the checkpoint-2000 critic does not consume actual depth maps. Its own
frozen encoder reads RGB image features and prompt token embeddings; the actor's
point-map injection is bypassed. Depth placeholder embeddings remain in the
prompt but are excluded from this report's gradient norm. All 8,776 saved records
mark `raw_depth_consumed: false`.

**Continuous state:** the actor's `state_projector` is not wired into the critic's
separate encoder. Checkpoint 2000 is saved with `state_format: discrete` and the
report measures derivatives of its discrete state-value embeddings. A continuous
state prompt would currently reach the critic as a placeholder without projected
state values. Using this frozen critic with a continuous-state actor requires
preserving the critic's discrete preprocessing, or implementing and training a
compatible continuous-state critic. This report change does not alter the model.

Elapsed segment fraction = `(frame - start) / (terminal - start)`.
Seconds remaining = `(1 - elapsed fraction) * segment duration`.
The segment uses critic terminals; release is folded into the preceding move.
This is recorded elapsed time, not an estimate of physical task completion.
Variable segment durations explain why the two timing axes differ when pooled.

ΔV uses matching, nonterminal, one-chunk pairs with identical conditioning.
Raw advantage is `clip(r + gamma * (1-done) * V(next), support) - V(current)`.
There are 7,499 paired ΔV values and 8,257 advantages including 758 terminals.
519 nonterminal rows lack a matching next forward under the same conditioning;
their transition measurements stay unavailable. No model rerun filled them in.

## Initial reading, not a settled interpretation

- For the image + state norm, grasp first/last 10% frame medians are
  0.0787 / 0.1744; episode-median summaries are
  0.0829 / 0.1313.
- Nonterminal signed-advantage Spearman correlation is 0.041;
  absolute-advantage correlation is 0.256.
- Earlier discussion used the all-input norm. Its numerical results should not
  be substituted for these restricted image + state measurements. The original
  all-input norms and full token breakdowns are still saved for comparison.
- Action-conditioned trends differ. Move-only late bins become sparse because
  release shares its timing segment but has a different frame label.
- These are descriptive associations in the recorded data. They do not establish
  that gradient magnitude measures difficulty or predicts execution mistakes.

## Original artifacts and reproducibility

Original run: `outputs/probe_runs/critic_swap_002000_train128_20260926`.
The gradient artifacts are under `step_00002000/critic_sensitivity/`:

- `gradient_points.json`: every plotted value, frame identity and image reference.
- `gradient_binned_summary.{html,json,csv,md}`: numerical reports and bin tables.
- `gradient_episodes.json`: episode catalog.
- `sources/<source>/epNNNN/critic_gradients.json`: full saved measurements;
  adjacent `frames/` folders contain camera images.
- `browser_check_html_report.json` and `browser_check_curves.json`: browser and
  numerical checks, including quantiles against NumPy and ranks against SciPy.

If updating older saved points, first derive the restricted component norms:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m lerobot.probes.critic_gradient_view \
  --components-only outputs/probe_runs/critic_swap_002000_train128_20260926/step_00002000/critic_sensitivity
```

Rebuild the live numerical report from saved points, without model inference:

```bash
node migration/summarize_critic_gradients.cjs \
  outputs/probe_runs/critic_swap_002000_train128_20260926/step_00002000/critic_sensitivity
```

This documentation HTML was refreshed for the image + state norm; rebuilding the live report does not
replace it. The page's frame/download links have been adjusted for its docs path.
