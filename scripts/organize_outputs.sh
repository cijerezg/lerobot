#!/usr/bin/env bash
#
# organize_outputs.sh — sort the top level of outputs/ into data/ models/ runs/ analysis/ annotation/ media/ logs/ cache/.
#
#   lerobot/scripts/organize_outputs.sh            # print the plan, change nothing
#   lerobot/scripts/organize_outputs.sh --apply    # do it
#
# Every move is a rename inside outputs/ (same filesystem): nothing is copied, nothing is
# overwritten, an existing destination is skipped and reported. Entries no rule knows stay
# where they are and are listed. Run it on each machine; the rules read the folders, not a list.
#
#   data/                              dataset roots, dataset_repair_backups
#   models/                            MolmoAct2, MolmoAct-FAST-tokenizer (a link stays at the old path:
#                                      saved checkpoints store it)
#   cache/                             buffer caches
#   runs/<run>/                        training runs (logs, checkpoints, validation)
#   runs/<run>/probes/<step>/<name>/   stand-alone probe passes (was probe_runs/<x>-ckpt<step>-<name>)
#   analysis/                          one-off studies, probe frame sets, scratch
#   annotation/                        was _annotation
#   media/                             videos, banners
#   logs/                              logs, leader trajectory logs
#
# The ReBot buffer cache fingerprint hashes the dataset root path: a cache built before the move
# is rebuilt after it.
#
set -uo pipefail

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WS/outputs" || exit 1

moved=0; skipped=0; left=()

move() {
  local src="$1" dst="$2"
  if [[ -e "$dst" || -L "$dst" ]]; then
    printf 'SKIP   %s -> %s (destination exists)\n' "$src" "$dst"; skipped=$((skipped + 1)); return
  fi
  printf 'move   %s -> %s\n' "$src" "$dst"; moved=$((moved + 1))
  (( APPLY )) || return 0
  mkdir -p "$(dirname "$dst")" && mv -T -- "$src" "$dst"
}

drop_empty() {
  printf 'rmdir  %s (empty)\n' "$1"
  (( APPLY )) && rmdir -- "$1"
  return 0
}

is_run() { [[ -d "$1/checkpoints" || -d "$1/validation" ]] || compgen -G "$1/logs/rl_offline_*.log" > /dev/null; }

# Links that point out of their own folder with a relative target break when the folder changes depth.
echo "== relative symlinks that leave their folder (check these before --apply)"
find . -maxdepth 4 -type l -lname '../*' -printf '  %p -> %l\n'

echo "== plan"
shopt -s nullglob dotglob
for entry in *; do
  case "$entry" in
    data|models|cache|runs|stats|analysis|annotation|media|logs|remote_val|train|.cache|.gitattributes|.trackerignore|README.md)
      continue ;;
    MolmoAct2*|MolmoAct-FAST-tokenizer)
      [[ -L "$entry" ]] && continue
      move "$entry" "models/$entry"
      (( APPLY )) && ln -s "models/$entry" "$entry"
      continue ;;
    dataset_repair_backups)
      move "$entry" "data/$entry"; continue ;;
    buffer_cache*)
      move "$entry" "cache/$entry"; continue ;;
    _annotation)
      move "$entry" annotation; continue ;;
    leader_trajectory_logs)
      move "$entry" "logs/$entry"; continue ;;
    *.mp4|linkedin-banner-concepts|progress_collage_*)
      move "$entry" "media/$entry"; continue ;;
    scratch)
      for child in "$entry"/*; do move "$child" "analysis/$(basename "$child")"; done
      [[ -z "$(ls -A "$entry")" || $APPLY -eq 0 ]] && drop_empty "$entry"
      continue ;;
    probe_runs)
      continue ;;   # after the loop: the runs have to be in runs/ first
    draw_spread|grad_drift|loss_share|quality_sweep_trace|gradient_compatibility_*|rtc_seam_fix_*|llm_corrections_*|diagnostics|probe_frames|probe|twin_eval_*|kinematics_roundtrip|fast_alphabet|grasp_reference)
      move "$entry" "analysis/$entry"; continue ;;
  esac
  if [[ -d "$entry" && ! -L "$entry" && -z "$(ls -A "$entry")" ]]; then drop_empty "$entry"; continue; fi
  if [[ -f "$entry/meta/info.json" ]]; then move "$entry" "data/$entry"; continue; fi
  if [[ -d "$entry" && ! -L "$entry" ]] && is_run "$entry"; then move "$entry" "runs/$entry"; continue; fi
  left+=("$entry (no rule)")
done

if [[ -d probe_runs ]]; then
  for child in probe_runs/*/; do
    child="${child%/}"; name="$(basename "$child")"
    # The pass's own log names the checkpoint it probed.
    ckpt="$(grep -ohE 'outputs/(runs/)?[A-Za-z0-9_.-]+/checkpoints/[0-9]+' "$child"/logs/*.log 2>/dev/null | head -1)"
    if [[ -z "$ckpt" ]]; then move "$child" "analysis/$name"; continue; fi   # sweeps over several checkpoints
    run="$(basename "$(dirname "$(dirname "$ckpt")")")"
    move "$child" "runs/$run/probes/$(basename "$ckpt")/$(sed -E 's/^.*ckpt[0-9]+-//' <<< "$name")"
  done
  [[ -z "$(ls -A probe_runs)" || $APPLY -eq 0 ]] && drop_empty probe_runs
fi

echo "== left in place"
printf '  %s\n' "${left[@]+"${left[@]}"}"
echo "== $moved to move, $skipped skipped$( (( APPLY )) || echo '  (dry run; --apply to do it)')"
if (( APPLY )); then
  echo "== broken symlinks after the moves"
  find . -maxdepth 6 -xtype l -printf '  %p -> %l\n'
fi
