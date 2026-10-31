#!/usr/bin/env bash
# Spark (DGX Spark GB10, aarch64, CUDA 13.0 driver, sm_121) setup for the VLM judge + Robometer services. RUN ON THE SPARK:
#     bash lerobot/scripts/spark_judge_setup.sh            # venvs + the three models
#     bash lerobot/scripts/spark_judge_setup.sh --hf-only  # only the model downloads
#     bash lerobot/scripts/spark_judge_setup.sh --fallback # judge venv WITHOUT vllm (transformers + FastAPI server, spark_judge_hf_server.py)
#
# What it creates
#   ~/judge-venv      uv, python 3.12, vllm ${VLLM_VERSION} from PyPI. vllm 0.32.0 (2026-10-18) ships
#                     vllm-0.32.0-cp38-abi3-manylinux_2_28_aarch64.whl on PyPI; its release notes call the default
#                     wheels "CUDA 13.0 Python wheels for x86_64 and arm64" (the +cu129 variant is the alternate on
#                     GitHub releases). The post-install check below prints torch.version.cuda: it must start with 13.
#                     Gemma 4 (Gemma4ForConditionalGeneration) and Qwen3-VL are in vllm's supported-models table
#                     (Gemma 4 since the 0.31 line). GB10 (sm_121) is NOT a platform vllm lists; if `vllm serve`
#                     fails on kernels, try VLLM_EXTRA="--enforce-eager" in spark_judge_serve.sh, then --fallback.
#   ~/robometer-venv  upstream lerobot main with the `robometer` extra (Qwen3-VL-4B backbone, qwen-vl-utils) + fastapi.
#                     Kept apart from ~/judge-venv (vllm pins torch/transformers) and from the fork venvs.
#   ~/.cache/huggingface/hub  google/gemma-4-31B-it (bf16, ~62 GB) + Qwen/Qwen3.5-9B (bf16, ~19 GB); with --all also
#                     Qwen3-VL-8B (~17 GB), Qwen3-VL-32B (~66 GB), Robometer-4B (~9 GB). All open (no licence click).
set -euo pipefail
# This script runs ON THE SPARK only (2026-10-30: it was run on the workstation by mistake, creating ~/judge-venv there and starting a
# 60 GB download on a disk with 200 GB free). From the workstation: ssh dgx-ip 'cd ~/Documents/Research/RL/LeRobot && bash lerobot/scripts/<this script>'
[[ "$(hostname)" == spark-5ef0 ]] || { echo "refusing: run this on the Spark (hostname spark-5ef0), e.g. ssh dgx-ip 'cd ~/Documents/Research/RL/LeRobot && bash $0'" >&2; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
JUDGE_VENV="$HOME/judge-venv"
RBM_VENV="$HOME/robometer-venv"
MODE="${1:-all}"

if [[ "$MODE" != "--hf-only" ]]; then
  echo "== judge venv ($JUDGE_VENV)"
  [[ -d "$JUDGE_VENV" ]] || uv venv --python 3.12 "$JUDGE_VENV"
  if [[ "$MODE" == "--fallback" ]]; then
    uv pip install --python "$JUDGE_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match torch torchvision
    uv pip install --python "$JUDGE_VENV/bin/python" "transformers>=5.16" accelerate fastapi uvicorn pillow
  else
    uv pip install --python "$JUDGE_VENV/bin/python" "vllm==$VLLM_VERSION"
  fi
  "$JUDGE_VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
assert torch.version.cuda and torch.version.cuda.startswith("13"), "need a CUDA 13 build for the 580 driver; pass a cu130 wheel"
try:
    import vllm; print("vllm", vllm.__version__)
except ImportError:
    print("vllm not installed (fallback mode)")
PY

fi
if [[ "$MODE" != "--hf-only" && " $* " == *" --all "* ]]; then
  echo "== robometer venv ($RBM_VENV)"
  [[ -d "$RBM_VENV" ]] || uv venv --python 3.12 "$RBM_VENV"
  uv pip install --python "$RBM_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match torch torchvision
  uv pip install --python "$RBM_VENV/bin/python" "lerobot[robometer] @ git+https://github.com/huggingface/lerobot.git@main" fastapi uvicorn
  "$RBM_VENV/bin/python" -c "import torch, lerobot.rewards.robometer as r; print('robometer module', r.__file__, 'torch cuda', torch.version.cuda, torch.cuda.is_available())"
fi

echo "== models"
# 2026-10-30 (user): offline scorer = Gemma 4 31B (one query per ~5 s of video, after the fact); live subtask assigner = the
# latest small Qwen given the per-task vocabulary = Qwen3.5-9B (2026-02, natively multimodal, Qwen3_5ForConditionalGeneration;
# the newest Qwen3.8 line has no model under 27B; Qwen3-VL-8B is 2025-10). Gemma 4 is no longer gated on HF (checked 10-30).
# Robometer, Qwen3-VL-32B and Qwen3-VL-8B (A/B for the assigner) only with --all.
hf download google/gemma-4-31B-it
hf download Qwen/Qwen3.5-9B
if [[ " $* " == *" --all "* ]]; then hf download Qwen/Qwen3-VL-8B-Instruct; hf download Qwen/Qwen3-VL-32B-Instruct; hf download lerobot/Robometer-4B; fi
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null
echo "done. next: bash lerobot/scripts/spark_judge_serve.sh gemma   (offline scorer, port 8011)   |   ... assigner   (Qwen3.5-9B, port 8013)"
