# State audit — 2026-07-19 (offline trainer running, pre-reconciliation)

Scope: everything the current `rl_offline.py --config config_rl.yaml` run touches (molmoact2_rl,
skip_critic=true, rebot-socks-annotated-v2 + memmap cache), plus the inference/probe seams that must
match it. Companion docs: session_status_2026-07-18.md (pipeline history), memory_build_plan.md,
depth_pointmap_design.md.

Verdict in one line: the run is a **plain BC run with subtask+metadata prompt conditioning** — the
subtask/summary *generation* training is silently off, and the advantage clause survives only on the
inference/probe side, where it no longer matches training.

---

## 1. What the current run actually does (verified in code)

Per step (`rl_offline.py` → `MolmoAct2Trainer.update_actor`):

- Samples 64 transitions from the memmap-cached buffer (image_stride 5, uint8, depth column).
- Action prompt per sample: task + discrete state + setup/control + **subtask clause** (from
  `subtask_index` via vocab, 30% dropout) + **metadata clause** (quality/mistake from
  `materialize_metadata`, 15% dropout). **No advantage clause** (skip_critic → `advantage=None` in
  `build_training_batch`).
- Loss: flow matching + discrete FAST CE (action_mode both, knowledge insulation on).
- Depth: point-map tokens injected (`_inject_depth_observations`), gate α trains in its own
  optimizer group at depth_lr=5e-4, excluded from pretrained merge. One-shot depth presence log.
- Pretrained merge fires once at step 8000 (alpha 0.2, pre/post checkpoints saved).
- No critic anywhere: `init_critic()` never called, no critic VRAM, `treat_main_dataset_as_golden`
  writes `is_golden` into the buffer but nothing reads it in this mode.

---

## 2. Out of place / functionally wrong

### 2.1 Subtask + summary generation training is OFF (highest priority)

`subtask_loss_weight` defaults to **0.0** and `subtask_max_new_tokens` defaults to **0**
([rl_molmoact2.py:59-61](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2.py#L59-L61)); **neither is set
in config_rl.yaml**. Consequences in the current run:

- `_subtask_generation_loss` is never called ([rl_molmoact2_trainer.py:915-920](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L915-L920))
  → the entire two-prompt generation seam — subtask CE **and** the MEM summary memory labels
  (hold/update-window, `summary_prev/target_index`) — trains nothing. The summary tables are loaded,
  materialized, and synced every run, then never used.
- Rollout/probe generation is disabled: `rtc_actor_runtime.py:472` gates on
  `subtask_max_new_tokens > 0`, and the probe adapter returns None
  ([adapters/molmoact2.py:154](lerobot/src/lerobot/probes/adapters/molmoact2.py#L154)), so the
  offline_inference probe's subtask/memory panel is empty.
- Mismatch it creates: the action prompt trains **with** a subtask clause 70% of the time, but at
  deployment nothing generates subtasks, so inference prompts would never carry one.

Fix: set both in config_rl.yaml (`subtask_loss_weight: 1.0` was the pi05 precedent; pick
`subtask_max_new_tokens` large enough for "Memory: … Subtask: …" — the 12-s summaries are multi-
sentence, so budget well above pi05's 24).

### 2.2 Advantage clause — training is clean, inference/probes are not (your point, confirmed)

Training no longer consumes advantage (skip_critic path passes `advantage=None`; no clause in any
training prompt). The leftovers are all on the eval side:

| Where | What happens | Status |
|---|---|---|
| [config_rl.yaml:144](config_rl.yaml#L144) `inference_advantage: 1.0` | rollout prompts get "The advantage is positive." — a clause the model never saw in training. Comment even says "matches skip_critic: false" while the config has `skip_critic: true` | set to `null`; `build_inference_batch` then drops the clause (documented behavior, [rl_molmoact2_trainer.py:1093-1095](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L1093-L1095)) |
| Probes hardcode `advantage=1.0`: [actions.py:166](lerobot/src/lerobot/probes/actions.py#L166), [offline_inference.py:369](lerobot/src/lerobot/probes/offline_inference.py#L369), [pointmap_bit_identity.py:76](lerobot/src/lerobot/probes/pointmap_bit_identity.py#L76), adapter defaults ([adapters/molmoact2.py:131,194,636](lerobot/src/lerobot/probes/adapters/molmoact2.py#L131)) | every validation probe evaluates with the OOD clause | thread `cfg.policy.inference_advantage` through instead of the literal (the critic-side probe already does the right thing — `advantage=None` at [adapters/molmoact2.py:582](lerobot/src/lerobot/probes/adapters/molmoact2.py#L582)) |
| [inference_utils.py:369,635](lerobot/src/lerobot/rl/inference_utils.py#L369) build `torch.tensor([[cfg.policy.inference_advantage]])` unconditionally | pi05-era actor/critic-logging paths; would **crash** on `inference_advantage: null` | not in the molmoact2 path — mark as pi05 legacy or guard for None when touched |
| [rl/utils.py:74-83](lerobot/src/lerobot/rl/utils.py#L74-L83) `preprocess_batch_for_pi05` injects inference_advantage into offline training prompts | pi05-only legacy in the shared utils module | leave (pi05 path) or move under rl/pi05/ |
| [probes/attention.py:881-888](lerobot/src/lerobot/probes/attention.py#L881-L888) segments the prompt by an "advantage" section | untested with the clause absent | verify once inference_advantage is null (patterns look optional, but confirm the probe doesn't misalign sections) |

### 2.3 point_gate console pollution

- The polluter: `logging.info(pointmap_gate …)` fires **every optimization step** inside
  `update_actor` ([rl_molmoact2_trainer.py:965-969](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L965-L969)).
- The values already go to wandb: `pointmap_gate`, `pointmap_gate_absmax`,
  `pointmap_gate_grad_absmax` are scalars in `accum` → `log_metrics` → `wandb_logger.log_dict` at
  log_freq. Nothing to add — just delete the logging.info block.
- Separately, `pointmap_gate` sits in `console_keys` ([line 1307](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L1307)) — that one only prints at log_freq; keep or drop by taste.
- Same category: raw `print(f"[RL_OFFLINE] step …")` every 10 steps at
  [rl_offline.py:623-624](lerobot/src/lerobot/scripts/rl_offline.py#L623-L624), duplicating the
  log_freq progress line right below it.

### 2.4 Config drift (cosmetic but misleading)

- wandb projects still SO-101-named (`so101_real-v1` / `so101_real_offline-v1`) for rebot runs.
- `policy.repo_id: cijerezg/multi-task-toys-merged-v2` — validate()-only leftover (comment admits it).
- Config header still says "Used by lerobot/src/lerobot/rl/rl_offline.py" but the entry point is
  `scripts/rl_offline.py`; also the duplicate template at `lerobot/src/lerobot/rl/config_rl.yaml`
  keeps drifting from the root one.
- `action_clamp_limits: null` and `fixed_reset_joint_positions: null` are flagged TODO in the YAML —
  needed before **online** runs only, harmless offline.

---

## 3. Clean (verified, no action)

- **Freeze/optimizer wiring**: depth modules (`pointmap_encoder`/`depth_stream` incl. gate+sink)
  explicitly whitelisted trainable, split into their own "depth" group at depth_lr, excluded from
  merge targets ([rl_molmoact2_trainer.py:304-309, 391-407](lerobot/src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L304-L309)). Matches the actor-freeze-whitelist lesson.
- **Metadata steering**: hard-error load (`load_metadata_rows`), episode-quality + window-mistake
  broadcast in `materialize_metadata`, clause renders partially without speed, 15% train dropout,
  inference prompts {quality 5, mistake false} ([rtc_actor_runtime.py:475-480](lerobot/src/lerobot/rl/rtc_actor_runtime.py#L475-L480)). Coherent end to end.
- **Summary seam plumbing**: summaries.parquet → `materialize_summaries` with update window =
  `subtask_regeneration_interval × fps` = 120 frames, text-table index offsets across additional
  datasets, `sync_summaries` into the pack step. Correct — just unconsumed until 2.1 is fixed.
- **Anchor encoding**: stats file validated for encoding/frame_conversion/chunk_size at load;
  postprocessor swaps in `policy_action_with_anchor_to_transition`; states stay absolute.
- **Pretrained merge**: missing "critic" target warns and skips (skip_critic-safe); optimizer state
  cleared post-merge; pre/post checkpoints saved.
- **skip_critic semantics**: critic never constructed (no VRAM), all advantage/critic branches
  cleanly bypassed; reward/done still stored and logged (fine).
- **Depth one-shot confirmations** on both training and inference paths guard the silent-null-bank
  failure mode.
- **Buffer hygiene**: metadata/summary fills clamp to `self.size`; additional buffers force
  `is_golden=False`; cache fingerprint includes image_stride (strided miss = hard error).
- **_subtask_generation_loss details** (once enabled): normalization identical to action path via
  prompt_mode toggle inside the pipeline; only annotated samples (`subtask_index >= 0`); summary
  indices sliced consistently. Depth keys ride along into the generation batch harmlessly (VLM
  forward ignores them).

---

## 4. Inert-but-fine (context, no action now)

- `treat_main_dataset_as_golden: true` — writes `is_golden` that only `compute_advantage` reads;
  becomes meaningful again when the critic returns.
- All critic hyperparameters in the YAML (bins, support, sigma_ratio 8.0, critic_lr, …) — parsed,
  ignored under skip_critic; values are the retuned ones, so nothing to fix.
- Short-term frame history: `memory.history_keys` defaults empty → history sampling off. Consistent
  with Phase 6 (consumption architecture) being parked; the buffer-side `_gather_history` machinery
  is ready.
- Prompt dropouts (subtask 0.3 / metadata 0.15 / summary 0.3) are pack-step defaults, not
  YAML-exposed. Fine unless you want to ablate them.
- Probes: all enabled at val_freq 400; critic probes self-skip; val set = train set (acknowledged in
  the YAML as temporary).

---

## 5. Suggested reconcile order

1. config_rl.yaml: `subtask_loss_weight` > 0, `subtask_max_new_tokens` > 0, `inference_advantage: null`
   (+ fix its comment). This makes the run match the intent in session_status "next steps" item 4.
2. Delete the per-step pointmap_gate logging.info (values already in wandb); drop the raw step print
   in rl_offline.py if the log line suffices.
3. Probes: replace hardcoded `advantage=1.0` with `cfg.policy.inference_advantage`; sanity-check the
   attention probe's prompt segmentation without the clause.
4. Verify at runtime (first 20 steps): `loss_subtask_ce` nonzero, batch carries `subtask_index` /
   `summary_*_index` / `metadata_*` from the memmap cache (the cache predates none of these for
   rebot-socks-annotated-v2, but there is no one-shot confirmation for them like depth has — worth
   one log line or a pdb check).
5. Cosmetic pass: wandb project names, repo_id, config header, template-config divergence.
