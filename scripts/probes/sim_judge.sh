#!/usr/bin/env bash
# Judge stage of a sim eval pass (run_sim_eval.sh JUDGE=1, 2026-11-04): Gemma 4 31B-it scores every recorded episode of the
# pass after the fact, one query per 5 s window of the top + wrist videos, and writes <sim>/judge/<layout>_ep<N>.judge.json
# (timeline) + .metrics.json (judge reward = value of the furthest stage it named, agreement with the scorer) + summary.md.
# sim_summary.py folds judge/ into summary.{json,md} and index.json (reward_judge.*, judge_agreement.*).
#
#     bash lerobot/scripts/probes/sim_judge.sh <run>/validation/step_<step>/sim
#
# Spark only (the weights live there; hostname spark-5ef0). The vllm server (lerobot/scripts/spark_judge_serve.sh gemma, 0.60 of
# the unified memory) is started for the stage and stopped at its end: it cannot share the box with the sims (PARALLEL=5) or the
# probes. Env: JUDGE_EPISODES ("0 1" = those indices per layout; default all), JUDGE_LAYOUTS (glob, default *), JUDGE_WORKERS (4),
# JUDGE_MAX_QUERIES (cap per episode, smoke tests), JUDGE_START_TIMEOUT (s to wait for the server, 1200), JUDGE_KEEP_SERVER=1
# (leave vllm up afterwards), JUDGE_ENDPOINT (use a server that is already up; nothing is started or stopped).
set -uo pipefail
sim_out="${1:?Usage: sim_judge.sh <...>/validation/step_<step>/sim}"
[[ -d "$sim_out" ]] || { echo "sim_judge: no sim folder at $sim_out" >&2; exit 1; }
ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ws"
judge_dir="$sim_out/judge"
mkdir -p "$judge_dir"
export PYTHONPATH="lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export PYTHONUNBUFFERED=1
endpoint="${JUDGE_ENDPOINT:-http://localhost:8011/v1}"
server_pid=""

ready() { curl -sf --max-time 5 "$endpoint/models" > /dev/null; }
stop_server() {
  [[ -n "$server_pid" && "${JUDGE_KEEP_SERVER:-0}" != "1" ]] || return 0
  echo "sim_judge: stopping the judge server (pgid $server_pid)"
  kill -TERM -- "-$server_pid" 2>/dev/null
  for _ in $(seq 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$server_pid" 2>/dev/null; true
}
trap stop_server EXIT

if [[ -z "${JUDGE_ENDPOINT:-}" ]]; then
  if ready; then
    echo "sim_judge: a judge server already answers at $endpoint; using it (not stopped afterwards)"
  else
    [[ "$(hostname)" == spark-5ef0 ]] || { echo "sim_judge: the judge runs on the Spark only (hostname spark-5ef0)" >&2; exit 1; }
    echo "sim_judge: starting the judge server -> $judge_dir/vllm.log"
    setsid bash lerobot/scripts/spark_judge_serve.sh gemma > "$judge_dir/vllm.log" 2>&1 < /dev/null &
    server_pid=$!
    deadline=$(( $(date +%s) + ${JUDGE_START_TIMEOUT:-1200} ))
    until ready; do
      kill -0 "$server_pid" 2>/dev/null || { echo "sim_judge: the judge server exited (see $judge_dir/vllm.log)" >&2; tail -n 20 "$judge_dir/vllm.log" >&2; exit 1; }
      (( $(date +%s) < deadline )) || { echo "sim_judge: judge server not ready after ${JUDGE_START_TIMEOUT:-1200}s" >&2; exit 1; }
      sleep 10
    done
    echo "sim_judge: judge server ready"
  fi
fi

args=(--sim-run "$sim_out" --out-dir "$judge_dir" --endpoint "$endpoint" --workers "${JUDGE_WORKERS:-4}" --layouts "${JUDGE_LAYOUTS:-*}")
[[ -z "${JUDGE_EPISODES:-}" ]] || { read -r -a eps <<< "$JUDGE_EPISODES"; args+=(--episodes "${eps[@]}"); }
[[ -z "${JUDGE_MAX_QUERIES:-}" ]] || args+=(--max-queries "$JUDGE_MAX_QUERIES")
echo "sim_judge: scoring $sim_out (${args[*]})"
start=$(date +%s)
.venv/bin/python -m lerobot.rewards.vlm_judge.score_episode "${args[@]}" 2>&1 | tee "$judge_dir/score.log"
rc=${PIPESTATUS[0]}
echo "sim_judge: done rc=$rc in $(( $(date +%s) - start )) s"
exit "$rc"
