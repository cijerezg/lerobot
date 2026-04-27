#!/bin/bash
# =============================================================================
# DRTC Experiment Runner
# =============================================================================
#
# Starts the policy server (if not already running), then runs experiments
# defined in a YAML config. All arguments are forwarded to the Python
# experiment runner.
#
# Usage:
#   ./scripts/run_drtc_experiment.sh --config mixture_of_faults
#   ./scripts/run_drtc_experiment.sh --config spike --output_dir results/experiments
#   ./scripts/run_drtc_experiment.sh --config examples/experiments/configs/disconnect.yaml
#   ./scripts/run_drtc_experiment.sh --viz --config baseline   # enable trajectory viz
#
# Flags (consumed by this script, not forwarded to the experiment runner):
#   --viz                   - Start the trajectory visualization server (HTTP :8088)
#   --no-tui                - Disable the two-tab experiment TUI
#
# Environment variables:
#   POLICY_SERVER_DELAY_S   - Seconds to wait for policy server startup (default: 3)
#   POLICY_SERVER_HOST      - Host the client should connect to (default: localhost)
#   POLICY_SERVER_PORT      - Port to check / bind (default: 8080)
#   DRTC_TUI                - Set false/0/no/off to disable the TUI (default: true)
#
# =============================================================================

set -e

# --- Parse flags consumed by this script ----------------------------------- #
ENABLE_VIZ=false
ENABLE_TUI="${DRTC_TUI:-true}"
case "${ENABLE_TUI,,}" in
    0|false|no|off) ENABLE_TUI=false ;;
    *)              ENABLE_TUI=true ;;
esac
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --viz) ENABLE_VIZ=true ;;
        --tui) ENABLE_TUI=true ;;
        --no-tui) ENABLE_TUI=false ;;
        *)     PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done
set -- "${PASSTHROUGH_ARGS[@]}"

POLICY_SERVER_DELAY_S="${POLICY_SERVER_DELAY_S:-3}"
POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-localhost}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/policy_server_${LOG_TIMESTAMP}.log"
CLIENT_LOG_FILE="$LOG_DIR/drtc_experiment_${LOG_TIMESTAMP}.log"
STATUS_FILE="$LOG_DIR/drtc_status_${LOG_TIMESTAMP}.jsonl"
CONTROL_FILE="$LOG_DIR/drtc_controls_${LOG_TIMESTAMP}.jsonl"

# PIDs for cleanup
POLICY_SERVER_PID=""
SERVER_TIMING_TAIL_PID=""
EXPERIMENT_PID=""
STARTED_SERVER=false

cleanup() {
    EXIT_CODE=$?
    trap - SIGINT SIGTERM EXIT
    echo ""
    echo "Shutting down experiment components..."

    if [ -n "$EXPERIMENT_PID" ] && kill -0 "$EXPERIMENT_PID" 2>/dev/null; then
        echo "Stopping experiment client (PID: $EXPERIMENT_PID)..."
        kill -TERM "$EXPERIMENT_PID" 2>/dev/null || true
        wait "$EXPERIMENT_PID" 2>/dev/null || true
    fi

    if [ -n "$SERVER_TIMING_TAIL_PID" ] && kill -0 "$SERVER_TIMING_TAIL_PID" 2>/dev/null; then
        kill -TERM "$SERVER_TIMING_TAIL_PID" 2>/dev/null || true
        wait "$SERVER_TIMING_TAIL_PID" 2>/dev/null || true
    fi

    if [ "$STARTED_SERVER" = true ] && [ -n "$POLICY_SERVER_PID" ] && kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
        echo "Stopping policy server (PID: $POLICY_SERVER_PID)..."
        kill -TERM "$POLICY_SERVER_PID" 2>/dev/null || true
        wait "$POLICY_SERVER_PID" 2>/dev/null || true
    fi

    echo "Cleanup complete."
    exit "$EXIT_CODE"
}

trap cleanup SIGINT SIGTERM EXIT

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  DRTC Experiment Runner"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Arguments:    $*"
if [ "$ENABLE_TUI" = true ]; then
    echo "TUI:          enabled (use --no-tui or DRTC_TUI=false to disable)"
else
    echo "TUI:          disabled"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 1: Start Policy Server (kill existing + start fresh)
# -----------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
: >"$CLIENT_LOG_FILE"
: >"$STATUS_FILE"
: >"$CONTROL_FILE"

if ss -tlnp 2>/dev/null | grep -q ":${POLICY_SERVER_PORT} " || \
   lsof -iTCP:"${POLICY_SERVER_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[1/2] Killing existing policy server on port ${POLICY_SERVER_PORT}..."
    # Find and kill the process listening on the port
    EXISTING_PID=$(lsof -ti TCP:"${POLICY_SERVER_PORT}" -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
        kill -TERM $EXISTING_PID 2>/dev/null || true
        sleep 1
        # Force-kill if still running
        kill -0 $EXISTING_PID 2>/dev/null && kill -9 $EXISTING_PID 2>/dev/null || true
        sleep 0.5
    fi
    echo "      Old server stopped."
fi

echo "[1/2] Starting policy server..."
echo "      Policy server logs: $LOG_FILE"
POLICY_SERVER_CMD=(uv run --no-sync python -m lerobot.async_inference.policy_server_drtc
    --host="$POLICY_SERVER_HOST"
    --port="$POLICY_SERVER_PORT"
)
if [ "$ENABLE_VIZ" = true ]; then
    POLICY_SERVER_CMD+=(--trajectory_viz_enabled=true)
fi
# Verbose diagnostics surface counters like `rtc_prefix_aligned`, anchor add-back
# warnings, and per-chunk timing breakdowns -- essential for debugging anchor /
# delta encoding behaviour. Toggle with VERBOSE_DIAGNOSTICS=false to silence.
if [ "${VERBOSE_DIAGNOSTICS:-true}" = true ]; then
    POLICY_SERVER_CMD+=(--metrics_diagnostic_verbose=true)
fi
LEROBOT_DRTC_STATUS_FILE="$STATUS_FILE" "${POLICY_SERVER_CMD[@]}" >"$LOG_FILE" 2>&1 &
POLICY_SERVER_PID=$!
STARTED_SERVER=true
echo "      Policy server started (PID: $POLICY_SERVER_PID)"
if [ "$ENABLE_VIZ" = true ]; then
    echo "      Trajectory visualization: http://localhost:8088"
fi
if [ "$ENABLE_TUI" != true ] && [ "${STREAM_SERVER_TIMINGS:-true}" = true ]; then
    echo "      Streaming server timing lines to console (disable with STREAM_SERVER_TIMINGS=false)."
    tail -n 0 -F "$LOG_FILE" 2>/dev/null | grep --line-buffered -E "\\[DRTC INFER TIMING\\]|Error in inference producer loop" &
    SERVER_TIMING_TAIL_PID=$!
fi
echo "      Waiting ${POLICY_SERVER_DELAY_S}s for server to initialize..."
sleep "$POLICY_SERVER_DELAY_S"

if ! kill -0 "$POLICY_SERVER_PID" 2>/dev/null; then
    echo "ERROR: Policy server failed to start!"
    echo ""
    echo "---- policy server log (last 200 lines) ----"
    tail -n 200 "$LOG_FILE" 2>/dev/null || true
    exit 1
fi
echo "      Policy server is running."
echo ""

# -----------------------------------------------------------------------------
# Step 2: Run Experiment (foreground)
# -----------------------------------------------------------------------------
echo "[2/2] Starting experiment..."
echo "      Press Ctrl+C to stop."
if [ "$ENABLE_TUI" = true ]; then
    echo "      Client logs: $CLIENT_LOG_FILE"
    echo "      Status file: $STATUS_FILE"
    echo "      Control file: $CONTROL_FILE"
fi
echo ""
echo "----------------------------------------------"

# If the user did not pass --server_address explicitly, point the client at the
# server we just spawned. This avoids surprises when the runner default and the
# server's bind address disagree (e.g. server on localhost, runner default LAN IP).
EXTRA_ARGS=()
if ! printf '%s\n' "$@" | grep -q -E '^--server_address(=|$)'; then
    EXTRA_ARGS+=(--server_address "${POLICY_SERVER_HOST}:${POLICY_SERVER_PORT}")
fi

if [ "$ENABLE_TUI" = true ]; then
    LEROBOT_DRTC_STATUS_FILE="$STATUS_FILE" \
    LEROBOT_DRTC_CONTROL_FILE="$CONTROL_FILE" \
        uv run --no-sync python examples/experiments/run_drtc_experiment.py "${EXTRA_ARGS[@]}" "$@" \
        >"$CLIENT_LOG_FILE" 2>&1 &
    EXPERIMENT_PID=$!

    set +e
    uv run --no-sync python scripts/drtc_tui.py \
        --status-file "$STATUS_FILE" \
        --control-file "$CONTROL_FILE" \
        --client-log-file "$CLIENT_LOG_FILE" \
        --server-log-file "$LOG_FILE" \
        --watch-pid "$EXPERIMENT_PID"
    TUI_EXIT_CODE=$?
    set -e

    if [ "$TUI_EXIT_CODE" -ne 0 ] && kill -0 "$EXPERIMENT_PID" 2>/dev/null; then
        echo "TUI exited; stopping experiment client (PID: $EXPERIMENT_PID)..."
        kill -TERM "$EXPERIMENT_PID" 2>/dev/null || true
    fi

    wait "$EXPERIMENT_PID"
    EXPERIMENT_PID=""
else
    LEROBOT_DRTC_STATUS_FILE="$STATUS_FILE" \
        uv run --no-sync python examples/experiments/run_drtc_experiment.py "${EXTRA_ARGS[@]}" "$@"
fi

# Show server-side diagnostics from the log (if any DIAG_SERVER lines exist)
if [ -f "$LOG_FILE" ] && grep -q "DIAG_SERVER" "$LOG_FILE"; then
    echo ""
    echo "----------------------------------------------"
    echo "  Server diagnostics (from $LOG_FILE):"
    echo "----------------------------------------------"
    grep "DIAG_SERVER" "$LOG_FILE"
fi
echo ""
echo "Server log: $LOG_FILE"