# Frozen-critic AWR

The prepared run is `config_rl_awr.yaml` at the workspace root. It trains the actor
with the critic from `molmoact2_critic_own_20260926_v2/checkpoints/002000`, frozen in
evaluation mode. The separate config retains the current raw-MolmoAct2 actor
initialization; set `policy.pretrained_path` to a chosen BC checkpoint to fine-tune
that actor instead. The critic checkpoint is independent of that choice.

## Weight calculation

For each critic-valid transition, using the sampler's actual reward and done flag:

```text
next_value = 0 if done else frozen_critic(next_observation, current_conditioning)
A = clamp(reward + discount * next_value, value_support_min, value_support_max)
    - frozen_critic(current_observation, current_conditioning)
group = (embodiment_index, exact_subtask_text)
z = (A - training_mean[group]) / shared_training_residual_std
log_q = clip(z, -3, 3) / 0.5 - training_log_mean_exp[group]
w = exp(log_q) / effective_batch_mean(exp(log_q))
actor_loss = mean(stop_gradient(w) * per_sample_actor_loss)
```

The final division is implemented in log space for numerical stability. It spans
all gradient accumulation microbatches and all distributed workers. There is no
second batch z-score. `advantage_lambda: 1` means no BC floor. A row whose successor
is unavailable (`critic_skip`) receives weight 1 and does not enter calibration or
weight statistics. An all-skipped batch remains ordinary BC.

Current subtask rewards are `(done - 1 - 5 * mistake_onset) / 12`, with discount
0.97 and value support [-2, 0]. The explicit mistake penalty is retained, including
on terminal chunks. No value is bootstrapped across a critic terminal. The diverse
loader supplies real successors when AWR is enabled even though critic updates
are disabled.

## Calibration and launch

Calibration draws from the actual mixed training iterator, excludes skipped rows,
and runs before any optimizer step. It estimates empirical group means, one
pooled within-group residual standard deviation, and a mean exponential per group.
It does not fit statistics to validation. Raw scalar training advantages are saved
so the probe can recompute group normalizers correctly for a temperature sweep.

From the workspace root, first run calibration alone:

```bash
PYTHONPATH=lerobot/src .venv/bin/python -m lerobot.scripts.rl_offline \
  --config_path=config_rl_awr.yaml \
  --policy.advantage_calibration_only=true
```

The default budget is 2,048 microbatches per worker, or 65,536 draws at batch size
32. Coverage is checked against every reachable, critic-valid training group.
If a rare group is missing, calibration saves a coverage report and stops before
actor updates. Increase `advantage_calibration_batches` and rerun. With no explicit
calibration path, an incomplete artifact is rebuilt; a complete matching artifact
is reused. There is no silent pooled fallback for an unseen training subtask.

Then use the same config without the calibration-only override to train. A complete
calibration is also generated automatically on first launch if none exists.

The artifact records the critic checkpoint's absolute path, size and mtime, the
reward/input settings, and the training mixture. Mismatches are rejected rather
than silently reusing stale statistics. This identity is metadata-based, not a
cryptographic hash of checkpoint bytes. Start a fresh output directory to calibrate
a changed critic or mixture. Each saved actor checkpoint includes a calibration
sidecar for its probes.

## Telemetry

At the configured `log_freq`, console and Aim show:

- Effective sample size / valid sample count and KL of normalized weights from BC.
- Minimum/maximum weight, average weight, and share carried by the top 5% of samples.
- Fraction of standardized advantages clipped at the configured threshold.
- Fraction of critic-valid rows, terminal frequency and terminal weight share.
- Mistake-onset frequency and mistake-onset weight share, derived from charged rewards.
- Number of represented groups and minimum/maximum group mean weight in the batch.

Single-worker runs also log advantage and weight histograms. Distributed scalar
metrics are computed from all workers' valid samples, including unequal valid-row
counts. Per-subtask details are written to `awr_subtask_weights.jsonl` at the same
logging interval; they are not added as thousands of Aim series.

Startup writes `awr_calibration.json`, `awr_calibration_coverage.json`, and
`awr_calibration_metrics.json`. The coverage report includes sample counts and the
number of groups with fewer than eight calibration samples. Low counts remain an
estimation limitation even when coverage is complete.

Group average weights equal one on the calibration distribution. They fluctuate
in individual training batches; final batch normalization preserves sample ratios
but does not enforce exact per-group mass in each small batch. The logged terminal
and top-5% shares expose concentration without automatically reducing aggressiveness.

Legacy runs retain `advantage_normalization: batch`. Their probe now displays that
actual raw-advantage weighting, rather than a different subtask-normalized variant.
