# draw_spread (2026-11-07)

Question: along training and val episodes, are the policy's 64 flow draws per state collapsed (memorized) or
calibrated against the cross-episode variation of the demos, does the draw mean reproduce the episode's own
style, and how do error and spread depend on motion size? Checkpoint: internal-only 1200. Anchors every 1 s,
metrics averaged over the 30 chunk steps, each split into offset (per-joint chunk mean) and shape (the rest),
in the normalized anchor-encoded action space of the flow loss. Four input-noise settings next to the clean one.

## Files

- `metrics.py`: per-anchor metrics (the docstring is the definition list).
- `report.py`: the self-contained HTML report (tables + figures) from the per-anchor table.
- `run_draw_spread.py`: the study. Prefix (ViT + LLM) once per anchor and setting, KV cache expanded to 64
  rows, 5 flow steps through the action expert at batch 64 (checked equal to the policy's own single-row path at
  the same seed). Stage 1 builds the reference pool from every training root (demo chunks at the same cadence
  with subtask string and state); each anchor's reference is its 32 nearest pool anchors by state within the same
  subtask string (verb fallback below 16 candidates), other episodes only.
- `dummy_run.py`: synthetic data through metrics + report (the alignment step; kept for the layout).

## Run

    uv run python lerobot/adhocs/draw_spread/run_draw_spread.py --config_path=config_rl_validate.yaml \
        --policy.pretrained_path=outputs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model \
        --out_dir=outputs/draw_spread/internal1200_2026-11-07

Flags: `--smoke=3` (3 anchors per episode, two training roots), `--skip_reference=true` (reuse
`out_dir/reference.npz`), `--noise_settings=state1,state3,image0.02,image0.05`, `--n_draws`, `--anchor_seconds`,
`--train_roots=root[:episode],...` (default: eight roots, each with its median-length episode).

## Outputs (under out_dir)

- `report.html`: headline table, per-episode timelines, calibration and scale scatters, style projection,
  per-subtask medians, input-noise tables, the train anchors with the smallest spread ratio.
- `anchors.csv`: one row per anchor x setting with every metric, the reference level and pool size.
- `episodes/<set>_<root>_<ep>.npz`: draws absolute (anchors x settings x 64 x 30 x 7, deg), clean draws
  normalized, demo chunks (deg and normalized), states, frames, subtasks.
- `reference.csv` + `reference.npz`: the corpus pool (chunks normalized, states, per-joint state std).
- `run.json`: checkpoint, episodes, settings, timing, the consistency check.

## Found (internal-only 1200, 2026-11-07; 1,414 anchors, 844 train / 570 val)

- Spread and error move together on both sets (log-log corr ~0.7). The lowest-spread anchors have the lowest
  error on val as well as on train; anchors with error > 3x spread are 2 % of train and 7 % of val. So low
  entropy is not a memorization signature here: where the model is confident it is usually right.
- The train-val gap is a constant factor at every spread level: mean error ~1.2x spread on train, ~1.6x on
  val (uniform overconfidence, ~2.5x in variance). It survives normalizing by motion size and holds per verb
  (grasp / move / release all 1.55-1.65 on val).
- Absolute error scales with motion size as motion^0.7; relative error is ~2x worse on the smallest motions
  than the largest. Val shifted up by ~2x in R at fixed motion.
- Object families on val: bits/clipper/kit (newest data) both most confident and best fit; clothes worst;
  differences within the scatter.
- Averaging draws: the mean of 8 draws has 21 % (train) / 15 % (val) less rms error than a single draw
  (bias-variance identity); on high-spread anchors the draws are mostly one biased blob, the gripper is the
  joint that goes bimodal, so an average would have to be mode-seeking (`trajectories/`).
- Checkpoint comparison (`compare_2026-11-07/compare.html`): mix40 10-26 at 1200 and the resume run at 1800
  (= same lineage + 10-27 data) have 2x the spread and 1.4-1.9x the error of internal-only on the same anchors,
  and a weaker spread-error correlation (0.3-0.5 vs 0.7). Calibration ratio error/spread, train / val:
  internal-only 1.20 / 1.60, mix40 1200 1.16 / 1.48, mix40 1800 1.04 / 1.28. The val-over-train gap narrows
  with the mixture and again with more training (1.33 -> 1.27 -> 1.23); it did not grow, so no aggregate
  memorization signature over 1200 -> 1800. The mixture's ratio falls steeply with spread (1.8 -> 0.8:
  overconfident when confident, underconfident when unsure), internal-only's is flat.
- Conclusion (user, 2026-11-07): the policy is better behaved than the hypothesis assumed; no low-hanging
  fruit for a reweighting or sampling change. Spread is a trustworthy confidence signal for the internal-only
  model, a weak one for the mixture.
- Follow-up scripts: `followup_spread_vs_error.py`, `followup_spread_vs_error_relative.py`,
  `followup_val_by_subtask.py`, `followup_hover_page.py` (scatter with the frame on hover),
  `followup_trajectories.py` (64 draws / mean / demo for the highest-spread anchors), `compare_runs.py` +
  `compare_page.py` (checkpoint comparison).
