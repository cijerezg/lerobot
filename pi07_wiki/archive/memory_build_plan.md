# Memory — grand build plan (crossable checklist)

Status tracker for adding memory to the MolmoAct2 VLA. Created 2026-07-13.
Three features, one system:

1. **Short-term memory** — the last X frames spanning ~4 s (parameterizable)
   supplied to the model at every inference; not always available (episode start).
2. **Long-term memory** — the model generates subtasks (MolmoAct2 currently does
   NOT); completed subtasks accumulate as a text history in the prompt.
3. **Metadata steering** — π0.7-style episode metadata (quality / speed / mistake)
   in the prompt; train on everything, prompt for the best at inference.

**Architecture is deferred.** This plan covers plumbing + design decisions only.
How history tokens enter the trunk (MEM-style video compression, HAMLET moment
tokens, DepthStream-style gated read) is Phase 6 and intentionally empty.

Literature anchors (numbers we copy until we have reason not to):
- π0.7 (arXiv 2604.15483): up to 6 history frames per camera @ **1 s stride**;
  proprio history = one token per past state; **history dropout p=0.3**; prompt
  component dropout (subgoals 25%, subtask 30%, metadata 15%); inference-time
  metadata = quality 5, mistake false, speed at 15th percentile.
- PI MEM (pi.website/research/memory): proprio-only history ≈ 50% vs text+video
  ≈ 70% task success — short-term memory should carry visual signal eventually.
- HAMLET (arXiv 2510.00695): retrofit pattern for single-frame VLAs; causal
  confusion is real → window/stride/content/dropout must be ablatable config.

Legend: `[x]` done · `[ ]` todo · `[~]` optional/de-risking · `file.py:Lnn` touchpoint.

---

## Model-agnostic boundary (cross-cutting rule)

MolmoAct2 is the current policy, not the permanent one. Every phase splits into
a **generic layer** (survives a policy swap untouched) and a **policy seam**
(small, rewritten per policy). The discipline that makes this work:

1. **Transport structure, render late.** The done-list is a list of strings,
   metadata is a dict, history is tensors + masks — in the buffer, over gRPC,
   in complementary data, everywhere. Prompt *text* is rendered from these only
   inside the policy's processor (each policy has its own template anyway —
   molmoact2's QA format vs pi05's). Never ship a pre-rendered prompt string
   through the generic layers.
2. **One shared `MemoryConfig` dataclass** (define once, embed as a `memory`
   field on each policy RL config) instead of per-policy fields. Works with
   draccus and with `rl_offline.py`'s per-policy YAML field-stripping.
3. **Generic by construction already:** `ReplayBuffer.sample()` history
   (Phase 1), the RTC rolling deque (Phase 2), annotation synthesis, done-list
   bookkeeping, metadata labeling (Phases 4–5 logic). These live in `rl/` and
   `datasets/`-level code and never import a policy.
4. **Per-policy seam, kept minimal:** prompt template rendering, subtask
   token emission/decoding (Phase 3 model half), and how history tensors enter
   the trunk (Phase 6). Define the seam as a small interface on the
   Trainer/policy (e.g. `render_prompt(task, done_list, metadata, subtask)` +
   `consume_history(batch)`) so the next policy implements two methods, not a
   port. The pi05→molmoact2 duplication in `rl/` is the cautionary tale.

---

## Open decisions (settle each before its phase starts)

- **D1 — short-term content (v1):** image frames only, or frames + past states?
  Recommendation: both, via a generic per-key spec; states are ~free (7 floats).
- **D2 — defaults:** window 4 s, 4 samples @ ~1 s stride (+ current frame),
  parameterized in seconds (fps recorded alongside). Matches π0.7's shape.
- **D3 — episode-start padding:** use LeRobot's native delta-timestamp padding
  (repeat boundary frame + `*_is_pad` mask) vs zero-pad. Recommendation: native
  pad + mask; the mask is the "not always available" signal the model sees.
- **D4 — subtask generation cadence:** generate on fixed period vs on-demand
  (subtask-complete trigger)? v1 recommendation: fixed period + cache, mirroring
  the pi05 greedy+cached design.
- **D5/D6 — OBSOLETE 2026-07-15:** append rule and string budget applied to the
  done-list, which was built 2026-07-13 and removed 2026-07-15 in favor of the
  MEM recurrent summary m_t (see Phase 4). The summary IS the budget mechanism.
- **D7 — metadata labels for existing episodes:** speed = episode length (free);
  quality/mistake = coarse manual pass vs VLM pass vs "assume demos are quality 5".
  Recommendation for v1: demos = quality 5 / mistake false (they're curated),
  online RL rollouts = auto-labeled (length + reward/success), refine later.
- **D8 — assembly location:** history windows and prompt strings are assembled in
  the **dataset/processor layer** (model-agnostic), policy-specific code only
  formats tokens/prompt. Per the model-agnostic seam rule.

---

## Phase 0 — groundwork

Data-path ground truth (read 2026-07-13, don't re-litigate): `rl_offline.py`
explicitly disables the LeRobotDataset delta machinery
(`scripts/rl_offline.py:493` sets `delta_timestamps = None`, `delta_indices =
None`) and routes everything through `ReplayBuffer.from_lerobot_dataset` →
`ReplayBuffer.sample()` (`rl/buffer.py:353`). **History is a `sample()` feature,
not a dataset feature.** Online inference (`rl/inference_async.py`) is a thin
wrapper over `rl/rtc_actor_runtime.py`, whose shared state keeps only the
*latest* observation (`RTCSharedState.update_observation` /
`get_latest_observation`, `rtc_actor_runtime.py:221,226`) — online history
needs new code there.

- [ ] **Config surface:** shared `MemoryConfig` dataclass — history spec
      `{key: (window_s, n_samples)}`, history dropout p, prompt-component
      dropout ps, memory-list cap N, metadata enable/defaults — embedded as a
      `memory` field on each policy RL config (see model-agnostic rule 2). Must
      be real declared fields (the memmap-cache fingerprint / YAML override
      machinery strips undeclared fields — the `image_storage_*` lesson).
- [ ] **Subtask annotations for rebot data** — canonical schema DECIDED
      2026-07-13: the pi05 schema (`meta/subtasks.parquet` vocab + per-frame
      `subtask_index` feature), NOT the SARM boundary-column schema. Reasons:
      the RL path already carries `complementary_info["subtask_index"]`
      per-frame through every buffer, `offline_dataset_utils` already remaps
      vocabularies across mixed datasets, and done-list synthesis reduces to a
      scan over an existing column. Tooling (all in
      `policies/pi05_full/annotate/`): `subtask_annotate.py` (auto VLM,
      Qwen2/Qwen3-VL, any LeRobot dataset), `manual_subtask_annotate.py`
      (gradio UI + skills.yaml — usable on a few episodes before any dataset
      exists), `visualize_annotations.py`. Post-annotation validation required
      (this pipeline once emitted empty labels → "Subtask: ;").
      `sarm_annotations/` stays only if SARM is ever trained.
- [ ] **Timebase audit:** dataset fps vs buffer entry rate (per control frame —
      confirmed by `next_state = states[idx + action_chunk_size]`) vs inference
      cadence; all windows in seconds, converted once.

## Phase 1 — short-term memory in `ReplayBuffer.sample()` ✅ BUILT 2026-07-13

One implementation serves both training paths, because both sample from the
same flat circular buffer. Mirror of the existing `next_state` index math.

- [x] `MemoryConfig` in `rl/shared_config.py` (`history_keys`,
      `history_window_seconds=4.0`, `history_num_samples=4`;
      `history_offsets(fps)` → per-key step offsets, 4 s @ 30 fps → [30,60,90,120]).
      Embedded as `memory` field on `MolmoAct2RLConfig`; draccus YAML roundtrip
      verified (survives `_preprocess_config_yaml` stripping). Empty
      `history_keys` = disabled = zero behavior change.
- [x] `ReplayBuffer._gather_history()` + emission in `sample()` as
      `state["history.{key}"]` `(B, T_h, ...)` oldest→newest +
      `state["history.{key}_is_pad"]` `(B, T_h)`. Invalid slots repeat the
      earliest valid frame of the current episode (pad rule D3).
- [x] Backward validity masking (done/truncated via cummax), buffer-start
      clamp, and circular write-head clamp — all three tested.
- [x] Action history included (user call, 2026-07-13): all fill paths store one
      executed action per frame (dataset generator and cache both extract
      `action[0]`), so `history_keys: [..., action]` gathers `actions[idx − k]`
      with the same validity mask. Whether to *feed* past actions to the model
      stays a Phase 6/7 ablation (causal-confusion caution unchanged).
- [x] Pass-through wiring: `from_lerobot_dataset`, `from_cache`,
      `rl_offline.py` (computes offsets from `cfg.policy.memory` + env fps),
      `load_additional_offline_buffers` (all buffers get the same spec so the
      combined iterator emits consistent keys). No cache-format or fingerprint
      change needed (assembly is sample-time).
- [x] Decision: history dropout is NOT in `sample()` — it's a training
      regularizer, applied where history is consumed (Phase 6 trainer side).
- [x] 5 unit tests in `tests/utils/test_replay_buffer.py` (mid-episode, episode
      start, boundary crossing, circular wraparound, sample() emission) — green.
      Pre-existing fork drift: 11 upstream-style tests in that file fail
      identically with and without these changes (verified via stash baseline).
- [ ] Spot-check on a real dataset batch (rl_offline with `memory.history_keys`
      set) — do together with first training use.
- [ ] DrQ note: image augmentation applies only to current-frame image keys;
      history frames are not augmented. Revisit at Phase 6 if consistency matters.

## Phase 2 — short-term memory, actor side ✅ BUILT 2026-07-13

Goal: what the policy sees at rollout == what `sample()` assembles at training.

- [x] `assemble_history_windows()` in `rl/buffer.py` next to `_gather_history`
      (single source of truth for clamp+mask semantics): entries = completed
      (state, action) steps of the episode, distance k = `entries[-k]`, slots
      past the oldest entry repeat it + pad True; empty deque → repeat current
      state / zero action, all padded. Output `history.{key}` `(1, T_h, ...)`.
- [x] `RTCSharedState`: `configure_history()` (deque maxlen = max offset),
      `push_history` / `clear_history` / `history_snapshot`, all under the
      existing lock. Env worker pushes one entry per control step — the exact
      (state, executed_action) pair the learner buffer stores for that frame —
      right where `transition_to_send` is assembled. Cleared on episode
      restart; NOT cleared on intervention (training buffers keep contiguous
      frames across takeover, teleop action = executed action).
- [x] Inference worker merges the windows into `obs_filtered` before
      `build_inference_batch`. Disabled by default (no `memory.history_keys`
      → zero behavior change).
- [x] Both runtime entry points (`act_with_policy_rtc_inference`,
      `act_with_policy_rtc`) configure history from `cfg.policy.memory` +
      `cfg.env.fps`; the standalone online ReplayBuffer also gets
      `history_offsets` so its own sampling matches.
- [x] Transitions stay **thin** — per-step `add()` unchanged; the learner-side
      `sample()` from Phase 1 reconstructs windows. No gRPC schema change.
- [x] Parity unit test: same episode through `ReplayBuffer._gather_history` and
      `assemble_history_windows` → identical values + masks (states everywhere;
      actions on non-pad slots — pad-slot action content legitimately differs:
      zeros actor-side vs `actions[idx]` learner-side, both masked). Plus an
      empty-episode-start test. 8 history tests green.
- [ ] Known deviation to resolve at Phase 6 (consumption): actor-side windows
      hold policy-format floats at native resolution; learner-side windows hold
      buffer storage (uint8/bf16 @ image_storage_size). The Phase 6 processor
      must normalize both to the same model input. For image history, consider
      uint8 storage in the deque (float full-res frames ≈ 3.7 MB × 120 entries).
- [x] DEPTH history plumbed 2026-07-13 (user call — plumbing shouldn't wait for
      the architecture decision). Canonical key everywhere = the buffer's
      complementary name `depth.{cam}.depth`; put it in `history_keys` and it
      flows. Learner-side: `_gather_history` sources `depth.*` from
      `complementary_info` (uint16 preserved, memmap gather). Actor-side: the
      deque entry pulls depth unbatched from the unfiltered transition (same
      source as the transition's depth block); the inference worker bridges the
      current frame's `observation.depth.{cam}` to the canonical key for the
      episode-start fallback. Parity test green including uint16 dtype and the
      empty-deque case. Whether the model *reads* temporal depth stays a
      Phase 6 decision — the data is simply there now.
- [ ] Hardware smoke: run inference_async with `memory.history_keys:
      [observation.state]` and log window shapes/masks over an episode start.

## Phase 3 — subtask generation in MolmoAct2

MolmoAct2's prompt (`processor_molmoact2.py:237` `_build_robot_text`) is a fixed
QA template with no subtask concept; pi05 had the generation flow (`rl/pi05/`,
reference implementation). Long-term memory depends on this phase: without
generated subtasks there is nothing to remember.

DESIGN DECIDED 2026-07-13 (user confirmed both): **two-prompt, string-level**
integration — a dedicated generation prompt produces the subtask NAME; the
action prompt carries it as text through the Phase 4 seam. Only strings/indices
travel actor↔learner (no token passthrough — kills the pi05 BOS-mismatch bug
class). Decode = **greedy + vocab snap** (`snap_to_subtask_vocab`: normalized
exact match → fuzzy → -1); candidate-scoring is a later ablation.

- [ ] **Annotations first:** run `pi05_full/annotate/subtask_annotate.py` on
      the rebot dataset when it exists (Phase 0 item); until then the manual
      gradio tool can label a few episodes for pipeline development.
- [x] **Slice 1 — generation-text machinery ✅ BUILT 2026-07-13**
      (`processor_molmoact2.py`): `_build_subtask_generation_text` (same
      visual/state context, question form, NO action token; done-list clause
      removed 2026-07-15), `snap_to_subtask_vocab`, and
      `MolmoAct2PackInputsProcessorStep.build_subtask_generation_inputs()` —
      tokenized batch; training samples get full text = prompt+name+eos with CE
      labels on the answer span (padding-side agnostic: last answer_len non-pad
      positions), unannotated samples fully masked + `valid` mask. Verified
      against the real checkpoint tokenizer: labeled span decodes to exactly
      `grasp the cup<|im_end|>` (no BPE boundary corruption). 8 prompt tests green.
- [x] **Slice 2 — rollout generation ✅ BUILT 2026-07-13:**
      `MolmoAct2Policy.generate_subtask_tokens()` (prefill + the existing
      generic greedy loop `_continue_discrete_generation_from_output`, eos
      stop); `MolmoAct2Trainer.generate_subtask_text()` (builds the generation
      prompt from the live obs via the pack step, decodes, snaps → (raw, name,
      index)); `build_inference_batch` accepts `subtask` context → prompt
      seam (done-list context removed 2026-07-15). Runtime: generation at
      `subtask_regeneration_interval` cadence in the INFERENCE thread, gated
      on `subtask_max_new_tokens > 0` (0 = disabled = zero behavior change;
      the old hardcoded `subtask_index=-1` stamp is now the live snapshot
      value, still -1 when off); `RTCSharedState.update_subtask` keeps the
      current (name, index); cleared on episode restart, NOT on intervention;
      snap misses logged. Env worker stamps `subtask_index` per transition —
      online frames look exactly like annotated offline frames to the learner.
      12 tests green. PENDING: GPU smoke of actual generation quality (needs
      loaded checkpoint; foundation model untrained on the generation prompt —
      expect garbage until Slice 3 training).
- [x] **Slice 3 — training loss ✅ BUILT 2026-07-13** (+ fixed a slice-2 bug):
      generation packing moved INSIDE the pipeline via a `prompt_mode` toggle on
      the pack step ("action" | "subtask_generation", flipped try/finally by
      callers) — direct step calls bypassed the normalizer, so the discrete
      state string was built from RAW joints at rollout; now both generation
      paths get identical normalization to the action path.
      `MolmoAct2Trainer._subtask_generation_loss`: slices annotated samples
      (subtask_index ≥ 0), full-pipeline generation batch, LM CE via
      `policy.model(**_model_inputs(batch), labels=...)`; separate backward
      accumulating into the same grads, weighted by `subtask_loss_weight`
      (default 0 = off), logged as `loss_subtask_ce`.
      `sync_subtask_vocabulary` wires `subtasks.parquet` → pack step in
      `make_processors` AND re-syncs in `rl_offline.py` after additional
      datasets extend the vocab via the remap (base-Trainer no-op hook).
      Verified: prompt_mode branch through the real tokenizer (labeled span =
      `grasp the cup<|im_end|>`, action mode intact after toggle); 25 tests
      green; ruff at baseline.

PHASE 3 CODE COMPLETE. What remains is data-dependent: annotate → train with
`subtask_loss_weight > 0` → GPU eval of generation quality → enable at rollout
(`subtask_max_new_tokens > 0`).
- [ ] **Inference:** greedy generation + cache at cadence D4 in
      `rl/inference_async.py`. Generation MUST run in the inference thread (the
      attention-capture concurrency lesson: global flag + action-expert patches
      are not thread-safe against a concurrent worker).
- [ ] **Passthrough:** generated subtask tokens ride actor→gRPC→learner so the
      flow loss conditions on what the actor actually saw (this is the existing
      subtask-token-passthrough plan, ported from pi05 to molmoact2 — 4 known
      touchpoints there; find the molmoact2 equivalents).
- [ ] Acceptance: offline eval on held-out frames produces sensible subtask
      strings (extend the offline eval script); rollout logs show cadence +
      caching working; learner prompt == actor prompt for the same transition.

## Phase 4 — long-term memory (MEM summary in the prompt)

**Done-list REMOVED 2026-07-15** (user decision: full replacement, no ablation
toggle). The MEM paper's ablation shows naive concatenated subtask history
performs much worse than a compressed recurrent summary m_t, so the done-list
built 2026-07-13 (fill-time `materialize_done_lists`, `done_list_ids`
complementary column, `done_list_cap`, prompt clause + dropout, RTC
append-on-switch bookkeeping + transition stamping, D5/D6) was deleted from
buffer.py / shared_config.py / rl_offline.py / offline_dataset_utils.py /
processor_molmoact2.py / rl_molmoact2_trainer.py / rtc_actor_runtime.py +
tests. What survives: the current-step clause + `subtask_dropout`, the
metadata clause, per-frame `subtask_index` transport, and the two-prompt
generation machinery (Phase 3) — the summary seam builds on those.

- [x] **Prompt clause (policy seam) ✅ BUILT 2026-07-13** (done-list clause
      removed 2026-07-15): `_build_robot_text` has `The current step is
      <subtask>.` + partial metadata clauses; all-None = byte-identical legacy
      prompt (checkpoint compat, tested). `MolmoAct2PackInputsProcessorStep`
      has `subtask_names` (index→name vocab from subtasks.parquet),
      `subtask_dropout=0.3` / `metadata_dropout=0.15` (training text only),
      and extraction helpers reading complementary `subtask` (rollout strings)
      or `subtask_index` (offline batches) + `metadata` dict.
- [ ] **Summary annotation (data):** run `canonicalize_subtasks.py` then
      `summary_annotate.py` (both in `policies/pi05_full/annotate/`, gemma-4)
      on annotated-dataset-v7 → `meta/summaries.parquet` (episode_index,
      segment_index, from_index, to_index, subtask_index, subtask, summary);
      frame in segment k conditions on summaries[k-1], "" for k=0. Then the
      validation dataset with `--reuse-map <train-root>` (✅ BUILT 2026-07-17:
      applies train's canonicalization_map.json instead of the LLM; unknown
      labels kept + warned; audit stamped `reuse-map:<path>`; end-to-end
      tested on a synthetic dataset).
- [x] **Summary seam ✅ BUILT 2026-07-17** (design settled with user: summary
      conditions the GENERATION prompt only — MEM-faithful, action prompt and
      online transitions untouched; no gRPC change).
      Training side: `ReplayBuffer.materialize_summaries(segments, window,
      index_offset)` writes int64 `summary_prev_index` / `summary_target_index`
      complementary columns (fill-time, cache-compatible; buffer position ==
      dataset index — image_stride only strides image storage). Labels: target
      = summary of COMPLETED segments (row k−1 for a frame in segment k, −1 =
      empty); conditioning = same (hold pair) except the first
      `update_window_frames` of each segment condition one summary older
      (update pair — where appending is learned); window = one HL tick
      (`subtask_regeneration_interval` × fps, computed in rl_offline).
      `load_summary_segments` (offline_dataset_utils) reads
      meta/summaries.parquet; rl_offline materializes main + additional
      buffers into ONE concatenated text table via `index_offset`, synced to
      the pack step by `trainer.sync_summaries` (base no-op hook).
      Prompt seam: generation prompt gains ` Memory: <m_t>` clause
      ("none yet." for empty, absent when off); answer =
      `build_generation_answer` → `<subtask>. Memory: <summary>`;
      `parse_generation_answer` inverts it at rollout; `summary_dropout=0.3`
      drops clause AND answer span together (subtask-only sample).
      Rollout: `RTCSharedState.current_summary` (cleared on episode restart),
      generation conditions on it, decode parsed → summary replaces m_t
      (None = no memory span = keep). CE covers subtask+summary in one span
      via the existing labels machinery — `_subtask_generation_loss` only
      passes the index columns through. 26 tests green incl. parquet
      round trip; ruff at baseline.
- [ ] Follow-up (optimization, when latency measured): skip the summary span
      decode on ticks where the subtask didn't change; `subtask_max_new_tokens`
      must budget for the summary length meanwhile.
- [x] **Offline-eval tooling ✅ BUILT 2026-07-17:** `probes/offline_inference.py`
      now evaluates the HL query — `ProbablePolicy.generate_subtask` (base
      default None; molmoact2 adapter reuses `MolmoAct2Trainer().
      generate_subtask_text`, gated on `subtask_max_new_tokens > 0`),
      conditioned per frame on the GT memory via `_summary_lookup`
      (`summary_label_spans`, refactored out of `materialize_summaries` —
      single source of truth for the hold/update rule). Panel shows GT
      subtask + GT memory + per-checkpoint pred subtask + pred memory;
      renders verified with and without summaries.
- [ ] Acceptance (data-dependent): offline eval shows sensible generated
      summaries; a mock rollout carries m_t across HL steps.

## Phase 5 — metadata steering ✅ CODE COMPLETE 2026-07-13

- [x] **Labeling (v1, D7):** `ReplayBuffer.materialize_metadata(quality,
      mistake, speed_bucket_steps)` — fill-time columns `metadata_quality` /
      `metadata_mistake` / `metadata_speed` (per-episode length // bucket,
      broadcast per frame). Gated on `memory.metadata_enabled` (default off);
      defaults `metadata_default_quality=5` / `mistake=False` /
      `speed_bucket_steps=500` (π0.7). Wired in rl_offline + additional loader.
- [x] **Extraction:** `_extract_metadata` reads the columns into per-sample
      dicts (explicit `metadata` dict still wins — rollout/eval path).
- [x] **Prompt clause + 15% dropout:** built earlier with the seam.
- [x] **Inference defaults:** runtime passes `{"quality": 5, "mistake": False}`
      (speed omitted — partial rendering) through `build_inference_batch`'s
      metadata context when `metadata_enabled`.
- [ ] Online rollouts: LIVE transitions carry no metadata by design (episode
      outcome unknowable mid-episode; the 15% dropout covers the mixed regime).
      Post-hoc labeling when recorded episodes become datasets — do together
      with the first online-data transfer. Critic-derived quality labels =
      later upgrade, not built.
- [ ] Acceptance (data-dependent): toggling metadata at inference changes
      behavior measurably on the offline eval; training curves stable.

## Phase 6 — MEM video encoder + continuous state history (DECIDED 2026-07-22)

Design in [../04_memory.md](../04_memory.md) §2.4 (MEM arXiv 2603.03596).
Decisions locked: encoder path only (prompt-image history path DELETED — token
count explodes: ~1,700 LLM tokens/step for 2 cams × 4 frames); continuous
proprio-history tokens behind a text lead-in; 5 past + current @ 1 s; no gate
($e(0)=0$ is the identity story); repeat-padding v1 (no ViT attention mask).

### Config ✅ BUILT 2026-07-22
- [x] `MemoryConfig`: `history_window_seconds` 4.0→5.0, `history_num_samples`
      4→5 (offsets [30,60,90,120,150] @ 30 fps).
- [x] `DepthPointmapConfig.history_num_samples` syncs → depth slots 4→5 past
      (time-embedding table grows; from-scratch module, no checkpoint concern).
- [x] New `MolmoAct2Config` fields: `temporal_layer_stride` (4) +
      `history_stride_seconds` (1.0, a provided field — no auto-derivation, set
      both in YAML if the window changes). Enable = history tensors present.

### Processor (`processor_molmoact2.py`) ✅ BUILT 2026-07-22
- [x] DELETED `_extract_history_images`, `history_image_spans` + the
      "Images i to j are earlier frames" clause, and the per-state text budget.
- [x] `_extract_history_image_stack`: history frames through
      `image_processor.preprocess` (same resize/normalize; asserts crop_mode
      "resize" via crop count) → complementary `history_images`
      (B, cams, T_h, 729, 588) **bfloat16** + `history_image_times` (T_h,)
      seconds oldest→newest + `history_images_mask` (B,). Cameras must cover
      exactly the prompt image_keys (hard error on subset).
- [x] State-history clause: lead-in + `T_h` × `<extra_0>` (`STATE_HISTORY_TOKEN`);
      id resolved from the tokenizer and shipped per batch as
      `state_history_token_id`; `history_state_values` (B, T_h, D) float32 keeps
      the existing normalization seam.
- [x] One `history_dropout` flip per sample gates placeholders AND the sample's
      row in `history_images_mask` → exact K=1 pretrained path. Generation pack
      ships `history_images` too (no state clause there, §1.2 unchanged).

### Modeling (`modeling_molmoact2.py`) ✅ BUILT 2026-07-22
- [x] `encode_image` extension: stash-based transport (consume-once, cleared on
      history-free batches), history slices concat'd time-last, e(t) =
      PE(t) − PE(0) sinusoidal-seconds added at each temporal resblock INPUT
      (paper eq.), union-softmax at resblocks (i+1) % stride == 0 →
      {3,7,11,15,19,23}, past rows dropped after 23, current rows sliced at the
      taps (18 pre-drop). Gradient checkpointing preserved (e_t passed as arg,
      not closure — the non-leaf-capture footgun).
- [x] Attention: per-query-frame loop (peak n×(n+T) scores ≈ 550 MB fp32 at
      B=32); `/ math.sqrt` division matching the module's eager path; float32
      scores/softmax; per-sample mask −inf on the temporal block only.
- [x] `state_history_projector` (D_state→2560, the ONLY new weights) on the
      policy; projected embeds ADDED at placeholder positions in
      `build_input_embeddings` (count-checked 0-or-T_h per sample); whitelisted
      in `_apply_actor_freeze`; rides the fresh-params "depth" optimizer group —
      which also keeps it OUT of pretrained_merge_targets (the checkpoint has no
      projector weights; a merge would drag it to init).

### Verification
- [x] Unit tests (`tests/policies/test_molmoact2_mem_encoder.py`): union softmax
      ≡ naive per-query reference; causality (older frames never read newer;
      NB a constant perturbation is invisible — LayerNorm removes it);
      mask-off ≡ plain spatial block exactly; e(0) = 0; no-history
      `encode_image` bit-identical to the pre-MEM loop; token-count invariant;
      camera-count mismatch raises; stash consume-once. Prompt tests updated
      (placeholder clause, stack extractor, budget).
- [ ] Real-batch verification folded into the first `rl_offline` run (standalone
      smoke was written, passed review as a design, then deleted 2026-07-22):
      confirm on the first steps that the pack/forward doesn't crash on real
      weights, the trainable-param log lists `state_history_projector`, and loss
      is finite. The targeted invariants (image-token count, projector grads,
      mask-all-False K=1) are covered by the unit tests above; the run only has
      to prove they hold on the real checkpoint.
- [x] Buffer/actor gather parity untouched (data side unchanged beyond 4→5).
- [ ] Critic reads the same trunk → memory reaches the critic through the
      existing seam; verify when the critic comes back on (skip_critic now).

## Phase 7 — training + validation

- [ ] BC retrain on annotated data with: subtask generation + summary memory +
      metadata + (architecture permitting) short-term history. Component
      dropouts on.
- [ ] Ablation grid (cheap, offline eval): no-memory baseline vs each channel;
      the causal-confusion check = does history help or hurt action loss on
      held-out episodes.
- [ ] Hardware run on the rebot (post cable fix): rollout with live summary
      updates; verify prompt bookkeeping over a long episode.
- [ ] Reflection hook (future, out of scope here): summary memory + critic
      value trend is the substrate; nothing in this plan blocks it.

---

## Sequencing

Phase 1 is a single-function change (`sample()`) testable with a synthetic
buffer today, and it automatically covers the learner side of online training.
Phase 2 is the actor-side rolling buffer (the genuinely new online code).
Phase 3 is the critical path for everything prompt-side (4 depends on it; 5 is
independent and can go in parallel); its prerequisite is the annotation pass in
Phase 0. Phase 6 unblocks the *visual* payoff of Phases 1–2, but states-only
history can be trained the moment 1–2 land (state tokens are a trivial seam).

Suggested order of attack: **0 (config fields) → 1 → 2 → 0 (annotations) → 3 →
4+5 (together, one prompt-assembly job) → 6 → 7.**
