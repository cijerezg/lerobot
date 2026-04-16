#!/usr/bin/env bash
# Run in background with nohup:
#   nohup bash resume_offline.sh > resume_offline.log 2>&1 &
set -euo pipefail

CHECKPOINT_DIR="outputs/recap_online_v11/checkpoints/001000/pretrained_model"

python -m lerobot.scripts.offline_learner_pi05 \
    --config_path config-recap.json \
    --policy.pi05_checkpoint "$CHECKPOINT_DIR" \
    --resume true
