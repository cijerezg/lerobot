#!/usr/bin/env bash
# Run the policy of one checkpoint in the MuJoCo sim on every layout of the eval set, locally or on the Spark.
# Sibling of run_probes.sh: same arguments, same output place, so chase_validate.sh / remote_validate.sh --sim
# drive it on every new checkpoint.
#
#   lerobot/scripts/run_sim_eval.sh CHECKPOINT_PRETRAINED_MODEL [FRESH_OUTPUT_DIRECTORY] [CONFIG]
#
# Output: <run>/validation/step_<step>/sim/<layout>/ per layout: subtask_events.json (stage events, reward per
# episode), logs/ and videos/ep<N>_{top,wrist,wrist_depth}.mp4 per episode (record_inference_videos; rerun.rrd only
# with rerun_save), plus sim/summary.{json,md} and index.json over all of them (reward per scene and tier), so
# view_probes / compare_probes show them as the sim probe (sim_nail = the pre-2026-10-22 nail-only eval). Each layout's task string and eval_subtasks script
# come from the registry (robots/rebot_b601_sim/tasks.py) as CLI overrides of the config's nail-task values.
# Env: LAYOUTS (default: the registry's LAYOUTS), SIM_EVAL_ARGS (extra inference_async flags, e.g.
# "--env.robot.episodes=1 --env.processor.reset.control_time_s=20").
set -uo pipefail
sim_workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$sim_workspace"
sim_checkpoint="${1:?Usage: run_sim_eval.sh CHECKPOINT_PRETRAINED_MODEL [FRESH_OUTPUT_DIRECTORY] [CONFIG]}"
sim_config="${3:-config_rl_sim.yaml}"
[[ -f "$sim_checkpoint/config.json" ]] || { echo "Missing checkpoint config: $sim_checkpoint" >&2; exit 1; }
sim_step=0
if [[ -n "${2:-}" ]]; then
  # May already hold a probe pass (remote_validate.sh --with-sim); only sim itself must be new.
  sim_run_dir="$2"
else
  # With no output override, write beside the checkpoint's training run.
  sim_step_dir="$(dirname "$sim_checkpoint")"
  sim_step_name="$(basename "$sim_step_dir")"
  [[ "$(basename "$(dirname "$sim_step_dir")")" == checkpoints && "$sim_step_name" =~ ^[0-9]+$ ]] || {
    echo "Expected RUN/checkpoints/STEP/pretrained_model" >&2; exit 1;
  }
  sim_run_dir="$(dirname "$(dirname "$sim_step_dir")")"
  sim_step="$((10#$sim_step_name))"
fi
printf -v sim_out '%s/validation/step_%08d/sim' "$sim_run_dir" "$sim_step"
[[ ! -e "$sim_out" ]] || { echo "Sim output already exists: $sim_out" >&2; exit 1; }
[[ -f "$sim_config" ]] || { echo "Missing sim config: $sim_config" >&2; exit 1; }
echo "Sim config: $sim_config"
export PYTHONPATH="lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1   # the Spark checkout has stray files under LeRobot/lerobot/ that shadow the package from the CWD
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
registry='from lerobot.robots.rebot_b601_sim.tasks import main; main()'
failed=0
for layout in ${LAYOUTS:-$(.venv/bin/python -c "$registry" layouts)}; do
  echo "=== $layout -> $sim_out/$layout"
  mapfile -t task_overrides < <(.venv/bin/python -c "$registry" overrides "$layout")
  # The sim robot raises SIGINT after its last episode; the runtime treats it as the operator's Ctrl-C and exits 0.
  # 1 h caps a stalled run (4 episodes x 90 s + resets per layout).
  # shellcheck disable=SC2086
  timeout 3600 .venv/bin/python -m lerobot.rl.inference_async \
    --config_path="$sim_config" \
    --inference_checkpoint_path="$sim_checkpoint" \
    --env.robot.scene="$layout" \
    --env.robot.id="rebot_sim_$layout" \
    --job_name="sim-$layout" \
    --output_dir="$sim_out/$layout" \
    --aim.enable=false \
    "${task_overrides[@]}" \
    ${SIM_EVAL_ARGS:-}
  rc=$?
  [[ $rc -eq 0 && -f "$sim_out/$layout/subtask_events.json" ]] || { echo "$layout failed (rc=$rc)" >&2; failed=1; }
done
.venv/bin/python lerobot/scripts/probes/sim_summary.py "$sim_out" || failed=1
exit $failed
