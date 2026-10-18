#!/usr/bin/env bash
# Reach probe (robot run 0-8 s) + turn-onset probe (training frames) on every checkpoint 200..1400.
# 001000 is already done (symlinked from the earlier runs); pass steps as arguments to run a subset.
set -e
RUN=outputs/rebot-stratified_sampling-20261016/checkpoints
OUT=outputs/probe_runs/ckpt_sweep_2026-10-17
M=$(dirname "$0")
STEPS=${@:-000200 000400 000600 000800 001200 001400}
for step in $STEPS; do
  for s in reach:rebot_rollouts_2026-10-17-probe-v1:reach_anchors oct07:rebot_teleop_2026-10-07-annotated-v1-tasks1:turn_onset_oct07 \
           oct14:rebot_2026-10-14-annotated-v4:turn_onset_oct14 oct16:rebot_2026-10-16-annotated-v1:turn_onset_oct16; do
    IFS=: read name root anchors <<< "$s"
    [ -f $OUT/$step/$name/action_trace/action_trace_records.npz ] && continue
    uv run python $M/trace_clippers.py --config config_rl.yaml \
      --policy.pretrained_path $RUN/$step/pretrained_model \
      --dataset.root outputs/$root \
      --probe_parameters.output_dir $OUT/$step/$name \
      --probe_parameters.trace_n_samples 16 \
      --probe_parameters.trace_frames "$(cat $M/$anchors.txt)"
  done
done
