#!/usr/bin/env bash
# Optional policy evaluation of the experimental visuals. Everything written stays under this experiment.
# Usage: bash lerobot/scripts/sim_builders/photoreal_clippers/run_eval.sh CHECKPOINT [nail_ep5|nail_ep1|nail_ep2|nail_ep3]
set -euo pipefail
experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "$experiment_dir/../../../.." && pwd)"
cd "$workspace_dir"
checkpoint="${1:?Supply a pretrained_model checkpoint directory}"
layout="${2:-nail_ep5}"
case "$layout" in nail_ep5|nail_ep1|nail_ep2|nail_ep3) ;; *) echo "Unknown layout: $layout" >&2; exit 1;; esac
[[ -f "$checkpoint/config.json" ]] || { echo "Missing checkpoint config: $checkpoint" >&2; exit 1; }
result_dir="$experiment_dir/policy_runs/${layout}_$(date +%Y%m%d_%H%M%S)_$$"
[[ ! -e "$result_dir" ]] || { echo "Output already exists: $result_dir" >&2; exit 1; }
export MUJOCO_GL=egl
export PYTHONPATH="$workspace_dir/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec "$workspace_dir/.venv/bin/python" -B -m lerobot.rl.inference_async \
  --config_path="$experiment_dir/config_$layout.yaml" \
  --inference_checkpoint_path="$checkpoint" \
  --output_dir="$result_dir" \
  --job_name="photoreal-$layout" \
  --env.robot.episodes=1 \
  --aim.enable=false
