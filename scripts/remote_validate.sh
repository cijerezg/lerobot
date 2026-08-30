#!/usr/bin/env bash
#
# remote_validate.sh — run a validation/probe pass for a local checkpoint on the DGX.
#
# Push code + checkpoint to the DGX, run the rl_offline probe suite there, stream the
# log back, pull the results into the mirrored local path, then delete the remote copy
# of the checkpoint. Both machines mirror the same absolute paths, so every path below
# is used verbatim on either side.
#
#   lerobot/scripts/remote_validate.sh outputs/molmoact2_offline_rebot_v4/checkpoints/000400
#   lerobot/scripts/remote_validate.sh lerobot-tinypi outputs/.../checkpoints/000400
#
# The remote run is launched detached (setsid + nohup), so a dropped ssh does not kill
# a two-hour probe suite — the tail reconnects, and `--attach` picks a run back up from
# a fresh shell. Note that rl_offline's per-step ticker is stdout-only and the suite
# looks frozen for long stretches by design; the log is the ground truth, not silence.
#
set -uo pipefail

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }

# ── Defaults ────────────────────────────────────────────────────────────────────
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${DGX_HOST:-dgx}"
CODE=""
CKPT=""
CONFIG=""
RUN_DIR=""
KEEP_CKPT=0
FORCE_DELETE=0
ATTACH=0
DRY=0
SSH_OPTS=(-o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ConnectTimeout=15)

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  cat <<'EOF'

Positional:
  [CODE] CKPT        CKPT is the checkpoint dir (a step dir, or a pretrained_model dir).
                     CODE is the source tree to ship, default: the workspace root.

Options:
  --host HOST        ssh target (default: $DGX_HOST, else "dgx")
  --workspace DIR    workspace root, mirrored on both boxes (default: repo parent)
  --config PATH      training config (default: <workspace>/config_rl.yaml)
  --out DIR          run/output dir (default: <workspace>/outputs/remote_val/<slug>)
  --keep-checkpoint  do not delete the remote checkpoint copy when done
  --force-delete     delete the remote checkpoint even if it pre-existed there
  --attach           skip sync+launch; re-tail and pull an already-running run
  --dry-run          print what would happen, change nothing
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)            HOST="$2"; shift 2 ;;
    --workspace)       WS="$2"; shift 2 ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --out)             RUN_DIR="$2"; shift 2 ;;
    --keep-checkpoint) KEEP_CKPT=1; shift ;;
    --force-delete)    FORCE_DELETE=1; shift ;;
    --attach)          ATTACH=1; shift ;;
    --dry-run)         DRY=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    -*)                die "unknown option: $1" ;;
    *)                 if [[ -z "$CKPT" ]]; then CKPT="$1"; else CODE="$CKPT"; CKPT="$1"; fi; shift ;;
  esac
done

[[ -n "$CKPT" ]] || { usage; exit 1; }

run() { if (( DRY )); then printf '  \033[90m$ %s\033[0m\n' "$*"; else "$@"; fi; }

# ── Resolve local paths ─────────────────────────────────────────────────────────
WS="$(cd "$WS" && pwd)" || die "workspace not found"
CODE="${CODE:-$WS}"
[[ -d "$CODE" ]] || die "code tree not found: $CODE"
CODE="$(cd "$CODE" && pwd)"
CONFIG="${CONFIG:-$WS/config_rl.yaml}"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

if   [[ -d "$CODE/src/lerobot" ]];         then SRC="$CODE/src/lerobot"
elif [[ -d "$CODE/lerobot/src/lerobot" ]]; then SRC="$CODE/lerobot/src/lerobot"
else die "no src/lerobot under $CODE"; fi

if   [[ -x "$CODE/.venv/bin/python" ]]; then VENV="$CODE/.venv"
elif [[ -x "$WS/.venv/bin/python" ]];   then VENV="$WS/.venv"
else die "no .venv found under $CODE or $WS"; fi

[[ -d "$CKPT" ]] || die "checkpoint not found: $CKPT"
CKPT="$(cd "$CKPT" && pwd)"
if [[ -d "$CKPT/pretrained_model" ]]; then MODEL="$CKPT/pretrained_model"; else MODEL="$CKPT"; fi
[[ -f "$MODEL/config.json" ]] || die "no config.json in $MODEL — is that a checkpoint?"

# Slug: <run>-<step> for .../<run>/checkpoints/<step>[/pretrained_model], else basename.
step_dir="$MODEL"; [[ "$(basename "$MODEL")" == "pretrained_model" ]] && step_dir="$(dirname "$MODEL")"
if [[ "$(basename "$(dirname "$step_dir")")" == "checkpoints" ]]; then
  SLUG="$(basename "$(dirname "$(dirname "$step_dir")")")-$(basename "$step_dir")"
else
  SLUG="$(basename "$step_dir")"
fi
RUN_DIR="${RUN_DIR:-$WS/outputs/remote_val/$SLUG}"
STATE="$RUN_DIR/.remote_val"
RLOG="$RUN_DIR/remote_val.log"          # remote log path (same string locally)
LLOG="$RUN_DIR/remote_val.log"          # local mirror we append to while tailing

info "host       $HOST"
info "workspace  $WS"
info "code       $SRC  (venv: $VENV)"
info "checkpoint $MODEL"
info "run dir    $RUN_DIR"

sshx() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

if (( ATTACH )); then
  [[ -f "$STATE/pid" ]] || die "no local state at $STATE — nothing to attach to"
  REMOTE_PID="$(cat "$STATE/pid")"
  PREEXISTED="$(cat "$STATE/preexisted" 2>/dev/null || echo 1)"
else
  # ── Preflight ─────────────────────────────────────────────────────────────────
  info "preflight"
  (( DRY )) || sshx true \
    || die "cannot ssh to '$HOST' — add a 'Host $HOST' block to ~/.ssh/config, or pass --host"

  mapfile -t ASSETS < <(
    grep -hoE '(^|[[:space:]])(root|val_dataset_path|buffer_cache_dir|base_path|discrete_action_tokenizer|action_encoding_stats_path):[[:space:]]*[^[:space:]#]+' "$CONFIG" \
      | sed -E 's/.*:[[:space:]]*//; s/^"//; s/"$//' | grep -v '^null$' | sort -u
  )
  missing=""
  (( DRY )) || missing="$(sshx "cd '$WS' 2>/dev/null || { echo '__NOWS__'; exit 0; }
                   [ -x '$VENV/bin/python' ] || echo 'venv: $VENV'
                   for p in ${ASSETS[*]@Q}; do [ -e \"\$p\" ] || echo \"asset: \$p\"; done")"
  [[ "$missing" == *__NOWS__* ]] && die "workspace $WS does not exist on $HOST"
  if [[ -n "$missing" ]]; then
    printf '%s\n' "$missing" >&2
    die "missing on $HOST (static assets are synced separately, not by this script)"
  fi
  if (( DRY )); then info "  would check ${#ASSETS[@]} static assets + venv on $HOST"
  else info "  ${#ASSETS[@]} static assets + venv present on $HOST"; fi

  PREEXISTED=0
  if (( ! DRY )) && sshx "[ -d '$MODEL' ]"; then PREEXISTED=1; fi
  (( PREEXISTED )) && warn "checkpoint already exists on $HOST — it will NOT be deleted (use --force-delete to override)"

  # ── Sync code + config ────────────────────────────────────────────────────────
  info "sync code"
  run rsync -a --delete --exclude='__pycache__/' --exclude='*.pyc' --exclude='.venv/' \
      -e "ssh ${SSH_OPTS[*]}" \
      "$SRC/" "$HOST:$SRC/" || die "code rsync failed"
  run rsync -a -e "ssh ${SSH_OPTS[*]}" "$CONFIG" "$HOST:$WS/config_rl.yaml" || die "config rsync failed"

  # ── Sync checkpoint ───────────────────────────────────────────────────────────
  info "sync checkpoint ($(du -sh "$MODEL" | cut -f1))"
  run rsync -a --partial --inplace --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
      "$MODEL/" "$HOST:$MODEL/" || die "checkpoint rsync failed"

  # ── Launch detached ───────────────────────────────────────────────────────────
  (( DRY )) || mkdir -p "$STATE"
  launcher="${TMPDIR:-/tmp}/remote_validate.run.$$.sh"
  trap 'rm -f "$launcher"' EXIT
  cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -o pipefail
export PYTHONUNBUFFERED=1
cd '$WS' || exit 97
echo \$\$ > '$STATE/pid'
uv run --no-project --python '$VENV/bin/python' python -m lerobot.scripts.rl_offline \\
    --config_path=config_rl.yaml \\
    --policy.pretrained_path='$MODEL' \\
    --policy.offline_steps=0 \\
    --val_on_start=true \\
    --save_checkpoint=false \\
    --aim.enable=false \\
    --offline_output_dir='$RUN_DIR'
echo \$? > '$STATE/exit_code'
EOF
  chmod +x "$launcher"

  if (( DRY )); then
    info "would launch on $HOST:"; sed 's/^/  /' "$launcher"; exit 0
  fi

  sshx "mkdir -p '$STATE' && rm -f '$STATE/exit_code' && : > '$RLOG'" || die "cannot prepare $RUN_DIR on $HOST"
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$launcher" "$HOST:$launcher" || die "launcher rsync failed"
  sshx "rm -f '$STATE/pid'; cd '$WS' && setsid nohup bash '$launcher' >> '$RLOG' 2>&1 < /dev/null & disown" \
    || die "failed to launch on $HOST"
  REMOTE_PID=""
  for _ in $(seq 30); do
    REMOTE_PID="$(sshx "cat '$STATE/pid' 2>/dev/null" || true)"
    [[ "$REMOTE_PID" =~ ^[0-9]+$ ]] && break
    REMOTE_PID=""; sleep 1
  done
  [[ -n "$REMOTE_PID" ]] || { sshx "tail -n 40 '$RLOG'" >&2; die "run did not start on $HOST (no pid after 30s)"; }
  echo "$REMOTE_PID"  > "$STATE/pid"
  echo "$PREEXISTED"  > "$STATE/preexisted"
  info "launched, remote pid $REMOTE_PID"
fi

# ── Stream the log, surviving dropped connections ───────────────────────────────
trap 'echo; warn "detached — the run continues on $HOST."; warn "reattach: $0 --host $HOST --out $RUN_DIR --attach $CKPT"; exit 130' INT

mkdir -p "$RUN_DIR"; : >> "$LLOG"
info "streaming $RLOG  (Ctrl-C detaches, does not kill the run)"
while :; do
  sshx "[ -f '$STATE/exit_code' ]" && break
  from=$(( $(wc -l < "$LLOG") + 1 ))
  sshx "tail -n +$from --follow=name --retry --pid=$REMOTE_PID -- '$RLOG'" 2>/dev/null | tee -a "$LLOG"
  sshx "kill -0 $REMOTE_PID 2>/dev/null" || { sleep 3; sshx "[ -f '$STATE/exit_code' ]" && break; }
  sleep 2
done
from=$(( $(wc -l < "$LLOG") + 1 ))
sshx "tail -n +$from -- '$RLOG'" 2>/dev/null | tee -a "$LLOG"
trap - INT

EXIT="$(sshx "cat '$STATE/exit_code' 2>/dev/null" || echo 1)"
[[ "$EXIT" =~ ^[0-9]+$ ]] || EXIT=1
info "remote run exited $EXIT"

# ── Pull results back ───────────────────────────────────────────────────────────
info "pull results -> $RUN_DIR"
rsync -a --info=stats1 --exclude='.remote_val/' -e "ssh ${SSH_OPTS[*]}" \
    "$HOST:$RUN_DIR/" "$RUN_DIR/" || die "results rsync failed — remote data left in place"

# ── Delete the remote checkpoint copy ───────────────────────────────────────────
if (( EXIT != 0 )); then
  warn "run failed — leaving the remote checkpoint in place for a retry"
elif (( KEEP_CKPT )); then
  info "keeping remote checkpoint (--keep-checkpoint)"
elif (( PREEXISTED )) && (( ! FORCE_DELETE )); then
  warn "remote checkpoint pre-existed this run — not deleting it"
else
  case "$MODEL" in
    "$WS"/outputs/*/*) info "delete remote checkpoint $MODEL"
        sshx "rm -rf -- '$MODEL' && rmdir -p --ignore-fail-on-non-empty \"\$(dirname '$MODEL')\" 2>/dev/null; true" \
          || warn "remote delete failed" ;;
    *) warn "refusing to delete '$MODEL' — not under $WS/outputs/" ;;
  esac
fi

echo
if (( EXIT == 0 )); then
  info "done. results: $RUN_DIR/validation/step_00000000/"
  info "view: uv run --no-project --python .venv/bin/python python -m lerobot.scripts.view_probes '$RUN_DIR'"
else
  info "log: $LLOG"
fi
exit "$EXIT"
