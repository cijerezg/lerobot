#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="outputs/recap_online_v11/checkpoints/001000/pretrained_model"

uv run python -m lerobot.scripts.offline_learner_pi05 \
    --config_path config-recap.json \
    --policy.pi05_checkpoint "$CHECKPOINT_DIR" \
    --resume true
