#!/usr/bin/env bash
#
# chase_validate.sh — keep the DGX busy for a whole training run.
#
# Watches a live run's checkpoints/ directory and hands the NEWEST unprobed checkpoint to
# remote_validate.sh, one pass at a time, until the training process exits or STOP appears.
# Checkpoints that land while a pass is in flight are marked skipped, not queued: the point
# is that the DGX is always working on the freshest weights, never on a backlog.
#
#   lerobot/scripts/chase_validate.sh outputs/molmoact2_rebot_diverse_speed_v1
#   setsid nohup lerobot/scripts/chase_validate.sh outputs/<run> > /dev/null 2>&1 &
#
# The newest checkpoint is read from the step directories themselves, not from
# checkpoints/last. A directory counts as complete only once its byte size stops changing,
# so a half-written save is never shipped.
#
# One pass at a time per box: the call takes an flock on <workspace>/outputs/remote_val/.dgx.lock.
# Re-running a checkpoint by hand while the chase is up must take the same lock:
#
#   flock outputs/remote_val/.dgx.lock lerobot/scripts/remote_validate.sh <ckpt>
#
# State is on disk under <run>/validation/.chase/, so a restart re-probes nothing:
#   <step>.ok  <step>.failed (holds the exit code)  <step>.skipped  log.<step>  chase.log
#
#   touch <run>/validation/.chase/STOP    # stop after the pass in flight
#
set -uo pipefail

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VALIDATE="$(dirname "${BASH_SOURCE[0]}")/remote_validate.sh"
POLL=120        # seconds between scans of checkpoints/
LOCKWAIT=7200   # seconds to wait for the box lock before giving up on this scan
SETTLE=20       # seconds a step dir must hold its size to count as complete
RETRIES=2       # extra attempts after a failed pass
TRAIN_PID=""
RUN=""
FWD=()          # forwarded verbatim to remote_validate.sh

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll)    POLL="$2"; shift 2 ;;
    --lockwait) LOCKWAIT="$2"; shift 2 ;;
    --settle)  SETTLE="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --pid)     TRAIN_PID="$2"; shift 2 ;;
    --)        shift; FWD=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    -*)        die "unknown option: $1 (pass remote_validate.sh flags after --)" ;;
    *)         RUN="$1"; shift ;;
  esac
done

[[ -n "$RUN" ]] || { usage; exit 1; }
[[ -d "$RUN" ]] || die "run directory not found: $RUN"
RUN="$(cd "$RUN" && pwd)"
[[ -x "$VALIDATE" ]] || die "remote_validate.sh not found next to this script"

STATE="$RUN/validation/.chase"
LOCK="$WS/outputs/remote_val/.dgx.lock"
mkdir -p "$STATE" "$(dirname "$LOCK")"
rm -f "$STATE/STOP"

log() { printf '[%s] %s\n' "$(date '+%F %H:%M:%S')" "$*" | tee -a "$STATE/chase.log"; }

# The training process to follow. Without one the chaser runs until STOP.
if [[ -z "$TRAIN_PID" ]]; then
  # -o: the OLDEST match is the launcher that lives for the whole run. The newest match
  # is often a short-lived worker forked off it, which would end the chase when it exits.
  TRAIN_PID="$(pgrep -o -f 'lerobot\.scripts\.rl_offline')"
fi
if [[ -n "$TRAIN_PID" ]]; then
  log "following training pid $TRAIN_PID"
else
  log "no rl_offline process found — running until $STATE/STOP appears"
fi

log "chasing $RUN  (poll ${POLL}s, ${RETRIES} retries)"

# Newest step directory by number, read from the directories themselves.
newest() {
  find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -xtype d -printf '%f\n' 2>/dev/null \
    | grep -E '^[0-9]+$' | sort -n | tail -1
}

# A save is complete when the model config is there and the tree stops growing.
settled() {
  local d="$1" a b
  [[ -f "$d/pretrained_model/config.json" ]] || return 1
  a="$(du -sbL "$d" 2>/dev/null | cut -f1)"
  sleep "$SETTLE"
  b="$(du -sbL "$d" 2>/dev/null | cut -f1)"
  [[ -n "$a" && "$a" == "$b" ]]
}

handled() {
  local s="$1"
  [[ -e "$STATE/$s.ok" || -e "$STATE/$s.failed" || -e "$STATE/$s.skipped" ]] && return 0
  [[ -d "$RUN/validation/$(printf 'step_%08d' "$((10#$s))")" ]]
}

while :; do
  if [[ -f "$STATE/STOP" ]]; then log "STOP — exiting"; break; fi

  alive=1
  if [[ -n "$TRAIN_PID" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    # A restarted run has not completed: adopt the new process instead of ending the chase.
    newpid="$(pgrep -o -f 'lerobot\.scripts\.rl_offline')"
    if [[ -n "$newpid" ]]; then
      log "training pid $TRAIN_PID gone, adopting $newpid (run restarted)"
      TRAIN_PID="$newpid"
    else
      alive=0
    fi
  fi

  step="$(newest)"
  if [[ -n "$step" ]] && ! handled "$step" && settled "$RUN/checkpoints/$step"; then
    # Anything older that never got a pass is skipped by design — record it, don't queue it.
    for older in $(find "$RUN/checkpoints" -mindepth 1 -maxdepth 1 -xtype d -printf '%f\n' \
                     | grep -E '^[0-9]+$' | sort -n); do
      (( 10#$older < 10#$step )) || continue
      handled "$older" || { : > "$STATE/$older.skipped"; log "skip $older (superseded by $step)"; }
    done

    rc=1; deferred=0
    for (( try=1; try<=RETRIES+1; try++ )); do
      log "probe $step  attempt $try/$((RETRIES+1))  -> $STATE/log.$step"
      exec 9>"$LOCK"
      if ! flock -w "$LOCKWAIT" 9; then
        exec 9>&-
        log "box busy: no lock in ${LOCKWAIT}s — deferring $step to the next scan"
        deferred=1; break
      fi
      "$VALIDATE" "${FWD[@]+"${FWD[@]}"}" "$RUN/checkpoints/$step" >> "$STATE/log.$step" 2>&1
      rc=$?
      exec 9>&-       # closing the fd releases the lock
      (( rc == 0 )) && break
      log "attempt $try failed rc=$rc"
      sleep 60
    done

    if (( deferred )); then
      sleep "$POLL"; continue      # nothing recorded: the step is still owed a pass
    elif (( rc == 0 )); then
      : > "$STATE/$step.ok"
      log "probe $step OK -> $RUN/validation/$(printf 'step_%08d' "$((10#$step))")"
    else
      echo "$rc" > "$STATE/$step.failed"
      log "probe $step FAILED rc=$rc after $((RETRIES+1)) attempts — see $STATE/log.$step"
    fi
    continue    # re-scan immediately: newer weights may already be on disk
  fi

  if (( alive == 0 )); then log "training pid $TRAIN_PID gone, newest checkpoint handled — exiting"; break; fi
  sleep "$POLL"
done

ok=$(ls "$STATE"/*.ok 2>/dev/null | wc -l)
bad=$(ls "$STATE"/*.failed 2>/dev/null | wc -l)
skip=$(ls "$STATE"/*.skipped 2>/dev/null | wc -l)
log "done: $ok probed, $bad failed, $skip skipped"
