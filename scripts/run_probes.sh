#!/usr/bin/env bash
# Run the configured full validation suite locally or on Spark, with no training.
set -euo pipefail
probe_workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$probe_workspace"
probe_checkpoint="${1:?Usage: run_probes.sh CHECKPOINT_PRETRAINED_MODEL FRESH_OUTPUT_DIRECTORY [CONFIG]}"
probe_run_dir="${2:?Pass a fresh probe output directory}"
probe_config="${3:-config_rl_validate.yaml}"
[[ -f "$probe_checkpoint/config.json" ]] || { echo "Missing checkpoint config: $probe_checkpoint" >&2; exit 1; }
[[ ! -e "$probe_run_dir" ]] || { echo "Use a fresh output directory: $probe_run_dir" >&2; exit 1; }
[[ -f "$probe_config" ]] || { echo "Missing probe config: $probe_config" >&2; exit 1; }
echo "Probe config: $probe_config"
export PYTHONPATH="lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python lerobot/scripts/probes/preflight.py --config "$probe_config"
# A fatal signal prints every thread's stack to the log; a pass that stalls (Spark, 2026-09-25:
# 100 % CPU, no I/O, no GPU load, no ptrace access) dumps them on `kill -USR1 <python pid>`.
# No periodic dump_traceback_later: its watchdog thread walks the other threads' frames without
# the GIL and segfaulted the 2026-09-28 pass mid-dump. 6 h caps what a stall costs (suite ~3-4 h).
timeout 21600 .venv/bin/python -c 'import faulthandler, runpy, signal, sys
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)
sys.argv = ["rl_offline"] + sys.argv[1:]
runpy.run_module("lerobot.scripts.rl_offline", run_name="__main__")' \
  --config_path="$probe_config" \
  --policy.pretrained_path="$probe_checkpoint" \
  --policy.offline_steps=0 \
  --val_on_start=true \
  --val_freq=0 \
  --save_checkpoint=false \
  --aim.enable=false \
  --offline_output_dir="$probe_run_dir"
# The shared runner catches per-probe exceptions; require every enabled report.
.venv/bin/python lerobot/scripts/probes/check_reports.py \
  "$probe_config" "$probe_run_dir/validation/step_00000000"
