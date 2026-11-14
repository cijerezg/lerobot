#!/usr/bin/env bash
# Run the sim policy on every layout, one after the other (3 episodes each per the yamls); from the workspace root.
# usage: lerobot/scripts/sim_builders/real2sim/scripts/run_layouts.sh [config_rl_sim.yaml config_rl_sim_ep1.yaml ...]
set -uo pipefail
CONFIGS=("$@"); [ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(config_rl_sim.yaml config_rl_sim_ep1.yaml config_rl_sim_ep2.yaml config_rl_sim_ep3.yaml)
for c in "${CONFIGS[@]}"; do
  echo "=== $c"; MUJOCO_GL=egl uv run python -m lerobot.rl.inference_async --config_path="$c" || echo "run on $c exited with $?"
done
