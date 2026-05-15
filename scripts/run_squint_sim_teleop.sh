#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults mirror the teleop arm values from
# examples/experiments/configs/baseline_pistar06_recap.yaml. They are passed
# before "$@" so command-line flags can still override them.
SQUINT_SIM_TELEOP_MODE="${SQUINT_SIM_TELEOP_MODE:-so101_leader}"
SQUINT_SIM_TELEOP_PORT="${SQUINT_SIM_TELEOP_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A057748-if00}"
SQUINT_SIM_TELEOP_ID="${SQUINT_SIM_TELEOP_ID:-so101_leader_2026_04_16_v2}"
SQUINT_SIM_TELEOP_LEADER_USE_DEGREES="${SQUINT_SIM_TELEOP_LEADER_USE_DEGREES:-true}"
SQUINT_SIM_TELEOP_FPS="${SQUINT_SIM_TELEOP_FPS:-30}"
SQUINT_SIM_TELEOP_SENSOR_WIDTH="${SQUINT_SIM_TELEOP_SENSOR_WIDTH:-448}"
SQUINT_SIM_TELEOP_SENSOR_HEIGHT="${SQUINT_SIM_TELEOP_SENSOR_HEIGHT:-448}"
SQUINT_SIM_TELEOP_VIZ_MAX_WIDTH="${SQUINT_SIM_TELEOP_VIZ_MAX_WIDTH:-720}"
SQUINT_SIM_TELEOP_VIZ_JPEG_QUALITY="${SQUINT_SIM_TELEOP_VIZ_JPEG_QUALITY:-82}"

DEFAULT_ARGS=(
    --teleop-mode "$SQUINT_SIM_TELEOP_MODE"
    --fps "$SQUINT_SIM_TELEOP_FPS"
    --sensor-width "$SQUINT_SIM_TELEOP_SENSOR_WIDTH"
    --sensor-height "$SQUINT_SIM_TELEOP_SENSOR_HEIGHT"
    --viz-max-width "$SQUINT_SIM_TELEOP_VIZ_MAX_WIDTH"
    --viz-jpeg-quality "$SQUINT_SIM_TELEOP_VIZ_JPEG_QUALITY"
)
if [ "$SQUINT_SIM_TELEOP_MODE" != "keyboard" ]; then
    DEFAULT_ARGS+=(--teleop-port "$SQUINT_SIM_TELEOP_PORT")
    if [ -n "$SQUINT_SIM_TELEOP_ID" ]; then
        DEFAULT_ARGS+=(--teleop-id "$SQUINT_SIM_TELEOP_ID")
    fi
    case "${SQUINT_SIM_TELEOP_LEADER_USE_DEGREES,,}" in
        1|true|yes|on) DEFAULT_ARGS+=(--leader-use-degrees) ;;
    esac
fi

exec uv run --no-sync python -m lerobot.robots.squint_so101.sim.teleop_dev "${DEFAULT_ARGS[@]}" "$@"
