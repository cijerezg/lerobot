#!/usr/bin/env bash
# Start one service on the Spark (one shell / tmux window each; nothing is daemonised):
#     bash lerobot/scripts/spark_judge_serve.sh gemma      # google/gemma-4-31B-it, OpenAI API on :8011 (offline scorer)
#     bash lerobot/scripts/spark_judge_serve.sh assigner   # Qwen/Qwen3.5-9B on :8013 (live subtask assigner; beside gemma)
#     bash lerobot/scripts/spark_judge_serve.sh assigner_vl8b  # Qwen/Qwen3-VL-8B-Instruct on :8013 (A/B; downloaded with setup --all)
#     bash lerobot/scripts/spark_judge_serve.sh qwen       # Qwen/Qwen3-VL-32B-Instruct on :8011 (alternative judge; not together with gemma)
#     bash lerobot/scripts/spark_judge_serve.sh robometer  # lerobot/Robometer-4B FastAPI on :8012
#     bash lerobot/scripts/spark_judge_serve.sh hf gemma|qwen   # fallback judge without vllm (transformers), same API on :8011
#
# Memory (128 GB unified, ~121 GB usable): gemma 31B bf16 weights ~61 GB, qwen 32B ~66 GB, Robometer ~9 GB. One judge + Robometer fit;
# the two judges together do not. --gpu-memory-utilization is the share of the whole unified pool vllm may take for weights + KV
# cache; 0.70 = ~85 GB (61 weights + ~24 KV/activations) leaves room for Robometer and the fork's own runs.
# Context: a query is <= 16 images (2 cameras x 8 frames); Gemma 4 spends 280 tokens per image at the default budget, Qwen3-VL ~300
# for 640x480, plus ~1.2k tokens of prompt -> 16384 covers it with margin. Env knobs: PORT, JUDGE_VENV, RBM_VENV, VLLM_EXTRA
# (e.g. "--enforce-eager" if CUDA graphs fail on GB10), MAX_MODEL_LEN, GPU_UTIL.
set -euo pipefail
# This script runs ON THE SPARK only (2026-10-30: it was run on the workstation by mistake, creating ~/judge-venv there and starting a
# 60 GB download on a disk with 200 GB free). From the workstation: ssh dgx-ip 'cd ~/Documents/Research/RL/LeRobot && bash lerobot/scripts/<this script>'
[[ "$(hostname)" == spark-5ef0 ]] || { echo "refusing: run this on the Spark (hostname spark-5ef0), e.g. ssh dgx-ip 'cd ~/Documents/Research/RL/LeRobot && bash $0'" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
JUDGE_VENV="${JUDGE_VENV:-$HOME/judge-venv}"
RBM_VENV="${RBM_VENV:-$HOME/robometer-venv}"
REPO="${REPO:-$HOME/Documents/Research/RL/LeRobot}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
WHAT="${1:?gemma|assigner|assigner_vl8b|qwen|robometer|hf}"

model_id() { case "$1" in gemma) echo google/gemma-4-31B-it ;; qwen) echo Qwen/Qwen3-VL-32B-Instruct ;; assigner) echo Qwen/Qwen3.5-9B ;; assigner_vl8b) echo Qwen/Qwen3-VL-8B-Instruct ;; *) echo "unknown judge $1" >&2; exit 2 ;; esac; }

case "$WHAT" in
  gemma|qwen|assigner|assigner_vl8b)
    # gemma = offline scorer (8011, 0.60 of the 121 GB = ~73 GB for ~62 GB of weights + KV); assigner = live subtask model
    # (8013, 0.25 = ~30 GB for ~19 GB of weights + KV). Start gemma first: vllm checks free memory >= its share at startup.
    MODEL="$(model_id "$WHAT")"
    GPU_UTIL="${GPU_UTIL:-$(case $WHAT in gemma) echo 0.60 ;; qwen) echo 0.75 ;; assigner*) echo 0.25 ;; esac)}"
    [[ $WHAT == assigner* ]] && DEFAULT_PORT=8013 || DEFAULT_PORT=8011
    [[ $WHAT == assigner* ]] && MAX_MODEL_LEN="${MAX_MODEL_LEN_ASSIGNER:-8192}"
    exec "$JUDGE_VENV/bin/vllm" serve "$MODEL" \
      --host 0.0.0.0 --port "${PORT:-$DEFAULT_PORT}" \
      --dtype bfloat16 \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs 4 \
      --limit-mm-per-prompt '{"image": 16}' \
      --gpu-memory-utilization "$GPU_UTIL" \
      --served-model-name "$MODEL" \
      $VLLM_EXTRA
    ;;
  robometer)
    exec "$RBM_VENV/bin/python" "$REPO/lerobot/src/lerobot/rewards/robometer/server.py" --host 0.0.0.0 --port "${PORT:-8012}" --max-frames 8
    ;;
  hf)
    MODEL="$(model_id "${2:?gemma|qwen|assigner}")"
    [[ ${2} == assigner* ]] && DEFAULT_PORT=8013 || DEFAULT_PORT=8011
    exec "$JUDGE_VENV/bin/python" "$REPO/lerobot/scripts/spark_judge_hf_server.py" --model "$MODEL" --host 0.0.0.0 --port "${PORT:-$DEFAULT_PORT}"
    ;;
  *) echo "usage: $0 gemma|assigner|assigner_vl8b|qwen|robometer|hf gemma|qwen|assigner" >&2; exit 2 ;;
esac
