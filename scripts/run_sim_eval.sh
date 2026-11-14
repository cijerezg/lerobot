#!/usr/bin/env bash
# Run the policy of one checkpoint in the MuJoCo sim on every layout of the eval set, locally or on the Spark.
# Sibling of run_probes.sh: same arguments, same output place, so chase_validate.sh / remote_validate.sh --sim
# drive it on every new checkpoint.
#
#   [PARALLEL=N] lerobot/scripts/run_sim_eval.sh CHECKPOINT_PRETRAINED_MODEL [FRESH_OUTPUT_DIRECTORY] [CONFIG]
#
# Output: <run>/validation/step_<step>/sim/<layout>/ per layout: subtask_events.json (stage events, reward per
# episode), logs/ and videos/ep<N>_{top,wrist,wrist_depth}.mp4 per episode (record_inference_videos; rerun.rrd only
# with rerun_save), plus sim/summary.{json,md} and index.json over all of them (reward per scene and tier), so
# view_probes / compare_probes show them as the sim probe (sim_nail = the pre-2026-10-22 nail-only eval). Each layout's task string and eval_subtasks script
# come from the registry (robots/rebot_b601_sim/tasks.py) as CLI overrides of the config's nail-task values.
# The sim runs on a sim clock (config env.robot.inference_delay_steps, 2026-10-28): the env steps in lockstep with the
# inference worker at a fixed RTC delay, so layouts can share the GPU without changing the policy's regime.
# Env: LAYOUTS (default: the registry's eval set, `tasks.py layouts`), SIM_EVAL_ARGS (extra inference_async flags, e.g.
# "--env.robot.episodes=1 --env.processor.reset.control_time_s=20"), PARALLEL (default 1; N runs up to N layouts at
# once, each layout's console output then goes to <sim>/<layout>.console.log beside its folder), SIM_SERVER (default 1:
# one policy server process for all layouts, rl/inference_remote.py, which runs the layouts' waiting calls as one
# batch, so on a fast GPU a high PARALLEL pays: 16 on the local GPU = 55 min for the full set, 2026-11-12;
# 0 = every layout loads the checkpoint itself),
# SIM_VIDEOS (default p1: videos only for placement 1 of each configuration; all = every layout, none).
# The eval set is the twin desk (2026-11-05; the photo-composite scenes behind TWIN=1 until then): tasks.py LAYOUTS = twin_l1 ..
# twin_l4, 16 configurations x 4 placements twin_<config>_p<k> (scripts/sim_builders/digital_twin/envs/ENVS_REPORT.md), episodes per
# placement from config_rl_sim.yaml. The twin scenes render the room, so the config's top camera carries no real-background composite.
set -uo pipefail
sim_workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$sim_workspace"
sim_checkpoint="${1:?Usage: run_sim_eval.sh CHECKPOINT_PRETRAINED_MODEL [FRESH_OUTPUT_DIRECTORY] [CONFIG]}"
sim_config="${3:-config_rl_sim.yaml}"
parallel="${PARALLEL:-1}"
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
if [[ "${RESUME:-0}" == "1" ]]; then
  # RESUME=1 (2026-10-30): finish an interrupted run in place; layouts that already have subtask_events.json are skipped.
  [[ -d "$sim_out" ]] || { echo "RESUME=1 but no sim output at $sim_out" >&2; exit 1; }
else
  [[ ! -e "$sim_out" ]] || { echo "Sim output already exists: $sim_out (RESUME=1 to finish it in place)" >&2; exit 1; }
fi
[[ -f "$sim_config" ]] || { echo "Missing sim config: $sim_config" >&2; exit 1; }
echo "Sim config: $sim_config (PARALLEL=$parallel)"
export PYTHONPATH="lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1   # the Spark checkout has stray files under LeRobot/lerobot/ that shadow the package from the CWD
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
registry='from lerobot.robots.rebot_b601_sim.tasks import main; main()'

run_layout() {
  local layout="$1"
  mapfile -t task_overrides < <(.venv/bin/python -c "$registry" overrides "$layout")
  # Without videos the sim renders only the frames the policy reads (1 in ~8), which frees about a quarter of the GPU.
  [[ "$layout" == $video_layouts ]] || task_overrides+=(--record_inference_videos=false)
  # The sim robot raises SIGINT after its last episode; the runtime treats it as the operator's Ctrl-C and exits 0.
  # 1 h caps a stalled run (8 episodes x 60 s of sim time + resets per layout; the sim clock runs faster than real time).
  # shellcheck disable=SC2086
  timeout 3600 .venv/bin/python -m "$sim_module" "${server_args[@]}" \
    --config_path="$sim_config" \
    --inference_checkpoint_path="$sim_checkpoint" \
    --env.robot.scene="$layout" \
    --env.robot.id="rebot_sim_$layout" \
    --job_name="sim-$layout" \
    --output_dir="$sim_out/$layout" \
    --aim.enable=false \
    "${task_overrides[@]}" \
    ${SIM_EVAL_ARGS:-}
}

case "${SIM_VIDEOS:-p1}" in
  all) video_layouts='*' ;;
  p1) video_layouts='*_p1' ;;
  none) video_layouts='' ;;
  *) echo "SIM_VIDEOS must be all, p1 or none" >&2; exit 1 ;;
esac
sim_module=lerobot.rl.inference_async
server_args=()
if [[ "${SIM_SERVER:-1}" == "1" ]]; then
  # One process holds the checkpoint and the layouts call it (rl/inference_remote.py, 2026-11-12), instead of a model
  # load and a model copy per layout. The socket is in /tmp: a unix socket path is capped at 107 characters.
  sim_socket="/tmp/sim_policy_$$.sock"
  sim_module=lerobot.rl.inference_remote
  server_args=(--policy_socket="$sim_socket")
  mkdir -p "$sim_out"
  rm -rf "$sim_out/policy_server"  # RESUME=1: the previous attempt's server folder
  .venv/bin/python -m lerobot.rl.inference_remote --serve=true --policy_socket="$sim_socket" \
    --config_path="$sim_config" --inference_checkpoint_path="$sim_checkpoint" \
    --output_dir="$sim_out/policy_server" --aim.enable=false > "$sim_out/policy_server.console.log" 2>&1 &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null; rm -f "$sim_socket"' EXIT
  echo "Policy server pid $server_pid loading $sim_checkpoint (log: $sim_out/policy_server.console.log)"
  until [[ -S "$sim_socket" ]]; do
    kill -0 "$server_pid" 2>/dev/null || { tail -n 30 "$sim_out/policy_server.console.log" >&2; echo "Policy server died" >&2; exit 1; }
    sleep 2
  done
  # The layouts' CPU work (preprocessing, env processors) is small tensors: one thread each. With the default (a thread
  # per core in every process) 8 layouts on 32 cores slowed each other 100x from the second episode on (2026-11-12).
  export OMP_NUM_THREADS=1
fi

failed=0
finish_layout() {  # layout, rc
  # rc 139 = the EGL cleanup segfault at interpreter exit (scenes/README.md); harmless when the results file is complete
  if [[ $2 -eq 139 && -f "$sim_out/$1/subtask_events.json" ]]; then echo "$1 ok (shutdown segfault after writing results)" >&2
  else [[ $2 -eq 0 && -f "$sim_out/$1/subtask_events.json" ]] || { echo "$1 failed (rc=$2)" >&2; failed=1; }; fi
}

running_pids=()
running_layouts=()
for layout in ${LAYOUTS:-$(.venv/bin/python -c "$registry" layouts)}; do
  if [[ "${RESUME:-0}" == "1" && -f "$sim_out/$layout/subtask_events.json" ]]; then echo "skip $layout (done)"; continue; fi
  if [[ "${RESUME:-0}" == "1" && -d "$sim_out/$layout" ]]; then
    # a failed attempt left a folder without results (inference_async refuses an existing output dir): keep its console log, drop the folder
    [[ -f "$sim_out/$layout.console.log" ]] && mv "$sim_out/$layout.console.log" "$sim_out/$layout.failed.console.log"
    rm -rf "$sim_out/$layout"
  fi
  echo "=== $layout -> $sim_out/$layout"
  if [[ $parallel -le 1 ]]; then
    run_layout "$layout"
    finish_layout "$layout" $?
    continue
  fi
  if [[ ${#running_pids[@]} -ge $parallel ]]; then  # a slot frees when the oldest job ends
    wait "${running_pids[0]}"
    finish_layout "${running_layouts[0]}" $?
    running_pids=("${running_pids[@]:1}")
    running_layouts=("${running_layouts[@]:1}")
  fi
  mkdir -p "$sim_out"  # the layout folder itself must not exist yet (inference_async refuses an existing output_dir)
  run_layout "$layout" > "$sim_out/$layout.console.log" 2>&1 &
  running_pids+=("$!")
  running_layouts+=("$layout")
done
for i in "${!running_pids[@]}"; do
  wait "${running_pids[$i]}"
  finish_layout "${running_layouts[$i]}" $?
done
.venv/bin/python lerobot/scripts/probes/sim_summary.py "$sim_out" || failed=1
exit $failed
