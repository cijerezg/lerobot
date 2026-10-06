# Depth point-map — build plan (crossable checklist)

Status tracker for building the back-projected point-map depth feature + its
co-evolving (MoT) read into MolmoAct2. Created 2026-06-13.

Design ref (read it for *why*; this doc is *what/where*):
- `depth_pointmap_design.md` — Part A the representation (LOCKED), Part B the
  co-evolving stream + additive gated SDPA read. (All the old `depth_tsdf_*` design
  docs were deleted once this path landed; this build supersedes that box/TSDF work.)

Legend: `[x]` done · `[ ]` todo · `[~]` optional/de-risking · `file.py:Lnn` touchpoint.

---

## What is true today (built 2026-06-15; validated end-to-end 2026-06-20)

The TSDF path is fully removed; the **point-map MoT** is the live and only depth path.
Built + flow-loss-only, and **validated end-to-end 2026-06-20** — training on GPU and
inference on the real robot:

- **Training (GPU, real checkpoint):** bit-identity@gate0 green; actor+critic train
  step runs; the actor gate α takes gradient and lifts off 0 (soft deadlock releasing
  as designed — gate grad ~1e-3 at step 0, `pointmap_gate_absmax` climbing thereafter).
- **Inference (on hardware, RTC):** the step-20 checkpoint loads (depth modules
  restore clean); live D405 depth (raw uint16) is ingested → back-projected → ~half of
  the 192 patches non-null and responsive to the scene; the gated read is connected
  (gate constant per-frame, nonzero). Depth's *effect* on actions is still negligible
  at this gate (≈ depth-free), as expected for a 20-step checkpoint.

Four bugs fixed to get here:
1. `fourier_position_encoding` always returned float32 → crashed the **bf16 critic**
   encoder at `pos_proj`; now cast `pe` to the proj weight dtype (`modeling_pointmap.py`).
2. the actor's `pointmap_encoder`/`depth_stream` fell through `_apply_actor_freeze`'s
   unrecognized-param **else (freeze)** — gate α never trained; added the always-trainable
   mirror branch the critic already had (`rl_molmoact2_trainer.py`). See
   `depth_pointmap_gate_gradient.md`.
3. `depth.median()` not implemented for **uint16** → live-depth trainer/inference log
   lines crashed; `.float().median()`.
4. `back_project` now accepts a bare **`(H, W)`** single live frame (inference delivers
   the depth unbatched, unlike the batched training path).

Pending = hardware/real-data only: D405 is still **top-mounted** with **placeholder K**
(382/382/320/240). For a meaningful run: mount on the wrist, calibrate K, sanity-check
`z_max_mm` (the smoke saw median ≈ 5.5 m vs `z_max=800` → far scene mostly masked), and
train long enough for the gate to grow so depth actually shapes behavior.

- **Live model read** (`policies/molmoact2/modeling_molmoact2.py`): additive gated
  SDPA + sink — `_patch_action_expert_pointmap_read`, the read added in
  `cross_attn_forward` as `SDPA(q,K_ctx,V_ctx) + tanh(α_ℓ)·gated_depth_read`.
  Encoder + `DepthStream` construction in `__init__`; training threads `depth_state`
  through `run_layer` — the per-layer gate `tanh(α_ℓ)` is read **in-region** inside
  `run_layer` (param-local, like `sink_logit`) so its gradient survives gradient
  checkpointing; inference runs the stream once in `prepare_context` (cached K/V across
  denoising). Config: `pointmap_config` on `configuration_molmoact2.py`.
- **Depth package** `policies/depth_pointmap/`: `modeling_pointmap.py` (encoder),
  `modeling_stream.py` (`DepthStream`, `gated_depth_read`, wrist-cam KV slicing),
  `configuration_pointmap.py`.
- **Data plumbing** (Track C) delivers raw depth end-to-end:
  `rl/buffer.py` (depth + `next_depth`), `rl/actor.py`, `rl/rtc_actor_runtime.py`,
  `rollout/strategies/core.py`, `scripts/lerobot_memmap_buffer_cache.py`,
  `utils/feature_utils.py` — reused unchanged (same `observation.depth.{key}` raw
  input), re-gated on `pointmap_config`.
- **Probe**: `probes/pointmap_bit_identity.py` (bit-identity@gate0 on the real ckpt).

---

## Phase 1 — point-map encoder (standalone) ✅ DONE 2026-06-13

- [x] New package `policies/depth_pointmap/` (decoupled; live model untouched).
- [x] `configuration_pointmap.py` — `DepthPointmapConfig` (intrinsics, P=40,
      image_size, z_min/z_max param, coord_scale, include_centroid_depth,
      num_wavelengths, dropout). Validates divisibility + ranges.
- [x] `modeling_pointmap.py` — `back_project` (depth→[X,Y,Z,m]), `patchify`,
      `fourier_position_encoding`, `PatchResidualBlock2d`, `PatchShapeCNN`,
      `DepthPointmapEncoder` (token = recentered-shape CNN + centroid PE; per-patch
      null bank; gate/abstain mirror DepthTsdfEncoder; `memory_from_batch`).
- [x] `tests/policies/test_depth_pointmap.py` — 10 CPU tests green; ruff clean.
- Acceptance: ✅ back-proj math, deadzone/far/hole masking, zeroed invalid coords,
  uint16 units, patch order, token count 192, empty→null, translation-invariant
  shape feature, gate shapes.

---

## Phase 2 — wire encoder into the model on the EXISTING read (de-risk) [~] 🔨 CODE DONE 2026-06-14

Goal: prove the new encoder runs in the real model (dtypes, devices, data
delivery, bit-identity@gate0) using the *already-tested* A.3 read, BEFORE the big
MoT surgery. Optional but cheap; skip only if going straight to Phase 3.

Code wiring landed: the point-map encoder is a true drop-in for the A.3 read (same
`memory_from_batch → (memory, gate)`, `.abstain_bias`, `.gate_value()`), so the
read machinery (`_project_tsdf_memory`, `_gated_tsdf_depth_read`,
`_patch_action_expert_tsdf_read`) is reused **unchanged**. `tsdf_config` and
`pointmap_config` are mutually exclusive (validated); a `_depth_encoder()` helper
returns whichever is set and the read sites call through it.

- [x] **Config field.** `pointmap_config: DepthPointmapConfig | None` in
      `configuration_molmoact2.py` (alongside `tsdf_config`). Import + discrete-mode
      guard + mutual-exclusivity guard (all three validations verified firing).
- [x] **Encoder construction.** `modeling_molmoact2.py` __init__: build
      `self.pointmap_encoder = DepthPointmapEncoder(d_mem=llm_kv_dim,
      num_read_layers=len(blocks), num_read_heads=cross_attn.num_heads)` when set;
      patch the expert once if **either** depth encoder is configured.
- [x] **Training/inference reads** routed through new `_depth_encoder()` helper
      (train read + inference handoff); no read-logic change.
- [x] **Trainable-param gating** (`_freeze_non_action_expert_parameters` +
      `get_optim_params`): match `"pointmap_encoder"` as well as `"tsdf_encoder"`.
- [x] **Intrinsics source.** Confirmed: `DepthPointmapConfig.intrinsics` is the
      single source (encoder's `memory_from_batch` reads `cfg.intrinsics` directly);
      no dataset-metadata K dependency on this path — nothing to drop.
- [x] CUDA-graph gate also disabled when `pointmap_config` set (A.3 manual read).
- [x] Probe cloned: `probes/pointmap_bit_identity.py`.
- Acceptance (GPU + real checkpoint — DONE 2026-06-20):
  - [x] A forward + train step runs with `pointmap_config` set (placeholder K, an
        existing wrist/top depth recording).
  - [x] **Bit-identity@gate0**: `uv run python -m lerobot.probes.pointmap_bit_identity
        --config config_rl.yaml` is green (chunks bitwise-equal vs depth-free).
- Risks: encoder dtype (float32 enc vs bf16 expert — `_project_tsdf_memory` already
  casts; verify on the GPU run); CUDA graphs stay OFF here (A.3 manual read
  unchanged — fixed in P3).

---

## Phase 3 — MoT co-evolving stream + joint masked-SDPA read (the real change)

Goal: replace the static A.3 read with depth tokens that **co-evolve per layer**,
read via one masked SDPA (regains CUDA graphs), with a learned sink for absolute
abstaining. This is the architectural payoff and the largest phase.

**Read mechanism DECIDED 2026-06-14 (user): additive gated SDPA + sink, NOT a single
joint softmax.** The joint softmax is *not* bit-identical at gate 0 (depth/sink keys
steal softmax mass even when depth values are zeroed). The read is two SDPA calls,
added: `out = SDPA(q,K_ctx,V_ctx) + tanh(α_ℓ)·SDPA(q,[K_d,k_⋆],[V_d,0])`. Exactly
zero at α=0 (bit-identity), compile-friendly, sink lives inside the depth softmax.
See `depth_pointmap_design.md` §B.4.

### 3a. Depth stream (co-evolution) — 🔨 standalone module DONE 2026-06-14
- [x] New `DepthStream` module (`policies/depth_pointmap/modeling_stream.py`; M
      light blocks, fresh float32, NOT in the frozen `action_expert` state_dict).
      Each block: depth self-attn + cross-attn to wrist-cam K/V + MLP. `forward`
      co-evolves N tokens → per-layer states; `read_kv` projects a state into action
      head space; owns α_ℓ + per-(layer,head) `sink_logit`. 6 CPU tests green.
- [x] Config knobs: `stream_width` (d_d=512), `stream_num_heads` (8),
      `stream_layers` (None ⇒ M=L), `stream_mlp_ratio`. Validated.
- [x] **(Stage 2)** Plumb the VLM **wrist-cam K/V per layer**: `wrist_cam_token_indices`
      / `gather_kv_at_indices` (per-row, padding-robust) slice the depth camera's run
      out of each layer's flat `(B,T,llm_kv_dim)` prefix KV. `_pointmap_wrist_meta`
      resolves (image_patch_id, num_images, cam_index) from `image_keys`.
- [x] **(Stage 2)** **Observation-only ⇒ cache once per obs**: inference runs the
      whole stream in the patched `prepare_context` and stores per-layer K/V on the
      context, reused across all denoising steps. Stream has no action dependence.

### 3b. The read (additive gated SDPA + sink) — 🔨 CODE DONE 2026-06-14 (Stage 2)
- [x] `gated_depth_read` (in `modeling_stream.py`, CPU-tested): one SDPA over
      `[K_d, k_⋆] / [V_d, 0]`, sink logit as an additive bias column. `cross_attn_forward`
      adds `tanh(α_ℓ)·gated_depth_read(...)` to the context read. New
      `_patch_action_expert_pointmap_read` (A.3 path untouched for `tsdf_config`).
- [x] **Per-layer gate** α_ℓ + per-(layer,head) `sink_logit` on `DepthStream`.
- [x] Reworked `prepare_context`/`forward_with_context`/`block_forward` (inference,
      new patch) + training `run_layer` (threads `depth_state`, evolves one block/layer,
      reads gated depth). Construction builds encoder (d_mem=stream_width) + DepthStream.
- Stream runs float32 (autocast disabled) at batch B; read K/V cast to action dtype,
  expanded to the flow-timestep batch at the training read site.
### 3c. Payoff — Stage 3 PARKED 2026-06-14 (low priority; moot under RTC)
- [~] CUDA graphs stay force-OFF when `pointmap_config` is set. **Investigated 2026-06-14:
      not worth doing now.** Reasons: (1) the action graph (`action_cuda_graph_manager`,
      in the checkpoint's `inference.py`) is only used by the NON-RTC `generate_actions_from_inputs`
      path; our config runs **RTC** (`rtc_config.enabled: true`), whose denoise loop is eager
      and never touches the graph → graphs-off costs nothing. (2) the graph only speeds the
      action denoising loop, marginal vs the un-graphed VLM prefix. (3) inference-only; early
      training gate≈0 ⇒ depth≈0 ⇒ moot anyway.
      The blocker if revisited: the checkpoint's `_clone_static_context`/`_copy_context_`
      copy a hard-coded field list (kv_contexts/masks/rope) and don't know our
      `depth_keys/depth_values`, so a replay silently drops the depth read. Fix = register
      depth K/V as static inputs via a self-managed in-place buffer (preferred — no checkpoint
      patching) or monkey-patching those helpers. REQUIRES a new gate-OPEN eager-vs-graph test
      (the gate-0 bit-identity probe can't catch depth staleness). Only pursue if switching to
      non-RTC inference AND the action loop measures as a real latency bottleneck.
- Acceptance (GPU + real checkpoint — DONE 2026-06-20):
  - [x] **Bit-identity@gate0**: `uv run python -m lerobot.probes.pointmap_bit_identity
        --config config_rl.yaml` green (probe is path-agnostic; covers the MoT read).
  - [x] A forward + train step runs with `pointmap_config` set.
  - [~] CUDA graph capture succeeds; step-time ≈ depth-free at gate0 (Stage 3 PARKED, §3c).
  - [x] A short train run shows α_ℓ moving off 0 (watch `pointmap_gate`). CONFIRMED:
        gate grad ~9.7e-4 at step 0, `pointmap_gate_absmax` 5e-5→2e-4 over the next
        steps; sink/block grads stay ~0 until the gate lifts, then co-adapt — exactly
        the soft deadlock. NOTE: the gate IS trainable from step 0 — ∂/∂α of
        `tanh(α)·read` is `read` (≠0) at α=0. Two bugs blocked this until 2026-06-20:
        the bf16-critic Fourier-PE dtype crash and the actor depth-modules being frozen
        by `_apply_actor_freeze` (see top-of-file summary + `depth_pointmap_gate_gradient.md`).
        Flow-loss-only training is viable (recon dropped, see Phase 4).
- Risks: frozen-policy integrity (gate0 probe is the guardrail); wrist-cam span /
  image_keys ordering assumption; dtype boundary (float32 stream ↔ bf16 expert).

---

## Phase 4 — masked-depth reconstruction aux loss — ❌ DROPPED 2026-06-14 (user)

Decision (user, 2026-06-14): **flow loss only for now.** No recon aux. The α=0
"deadlock" is softer than first stated (the gate is trainable from step 0; see 3c),
so flow-only training is viable. Revisit if the stream fails to learn useful geometry.
Nothing was built. (Original plan kept below for if we revisit.)

- [ ] ~~Lightweight decoder head on the depth tokens predicting masked-patch depth.~~
- [ ] ~~Mask ~50% of patches; `ℒ = ℒ_flow + λ·ℒ_recon`; wire λ + log `ℒ_recon`.~~

---

## Phase 5 — telemetry & probes — 🔨 gate uptake DONE 2026-06-14

- [x] Gate uptake: `accum["pointmap_gate"]` (mean tanh(α_ℓ)) + `pointmap_gate_absmax`
      (max |tanh α_ℓ|) in `rl_molmoact2_trainer.py`. `pointmap_gate` is on the console
      line; both ride `accum` → wandb `train/pointmap_gate{,_absmax}` automatically (base
      `log_metrics`, which skips step 0). **CONFIRMED climbing on the 2026-06-20 run**
      (absmax is the meaningful one — the mean can hide signed per-layer movement).
- [ ] (deferred) Admit-mass per layer (depth vs sink vs ctx) — needs SDPA weights,
      more invasive; add if gate telemetry isn't enough to diagnose.
- [ ] (deferred) Port `probes/tsdf_slices.py` → point-map contact sheet (back-projected
      point map / valid-mask coverage from sidecar PNGs) as a calibration/sanity view.
- [ ] Coverage/dropout-delta diagnostic (does depth change actions when present?). NOTE:
      a throwaway `[depth-check]` inference print (non_null patch count + gate) served
      this once on 2026-06-20 and was removed; a permanent version is still TODO.
- Acceptance: [~] gate telemetry readable + reviewed live; full dashboard review pending
      a real (wrist-mounted, calibrated) run.

---

## Phase 6 — cleanup / removal of the old TSDF path — ✅ DONE 2026-06-14

- [x] Removed `policies/depth_tsdf/` (encoder, config, geometry, 3D conv).
- [x] Removed `tsdf_config` from `configuration_molmoact2.py` + its validation/mutual-excl.
- [x] Removed the A.3 read (`_project_tsdf_memory`, `_gated_tsdf_depth_read`,
      `_patch_action_expert_tsdf_read`), the `tsdf_encoder` construction, training/inference
      handoffs, param-gating + cuda-graph entries, and `tsdf_gate` telemetry.
- [x] Deleted `probes/tsdf_bit_identity.py`, `probes/tsdf_slices.py`, `tests/policies/test_depth_tsdf.py`.
- [x] Re-gated the SHARED depth pipeline `tsdf_config` → `pointmap_config` (actor.py,
      rtc_actor_runtime.py, feature_utils.py, trainer depth-builder/critic-lift/logs).
- [x] Encoder cleanup: dropped the now-dead `gate`/`abstain_bias`/`num_read_layers`
      machinery (it mirrored the A.3 read); `memory_from_batch` returns tokens only.
- [x] Grep sweep clean: no `tsdf`/`DepthTsdf`/`build_tsdf` refs in source except two
      descriptive "replaces the TSDF voxel representation" comments in the pointmap pkg.
- Acceptance: [x] pointmap tests green (18); model imports clean; no dangling symbols.
- Docs (cleaned 2026-06-15): all `depth_tsdf_*.md` design docs DELETED as obsolete;
  the live read spec was merged into `depth_pointmap_design.md` Part B (code citations
  repointed there). Surviving depth docs: `depth_pointmap_design.md` + this tracker.

---

## Cross-cutting — data / hardware prerequisites (gate REAL training, not code)

The full code path is alive and **hardware-validated for inference plumbing** as of
2026-06-20 (live D405 depth → model, with D405 riding as the `top` camera and placeholder
K). The items below still gate a *meaningful* run:
- [ ] **D405 on the wrist** (currently top-mounted) + fresh recordings with depth.
- [ ] **Intrinsics K** calibrated → `DepthPointmapConfig.intrinsics` (replaces the
      placeholder 382/382/320/240). Until this lands, the point map is geometrically
      wrong even though the pipeline runs.
- [ ] **`z_max_mm` sanity** — the smoke saw median depth ≈ 5.5 m vs `z_max=800`, so most
      of the frame masks out (≈ half the patches null in a normal pose). Confirm the
      depth scale and pick a `z_max` that matches the real wrist-to-object range.
- [ ] Recordings with **hole-fill OFF** (raw depth; masking is in the encoder). Config
      currently has `depth_filters: true` on the D405 — flip for real recordings.
- [ ] Confirm `depth_units_mm=0.1` (D405 Z16) against a sidecar PNG.
- [ ] Buffer/memmap cache: point-map builds on-the-fly from raw cached depth, so the
      cache-fingerprint concern (`project_buffer_cache_fingerprint`) is moot — but
      verify image size / dtype declared so YAML overrides aren't stripped.
- [ ] `pretrained_path` / `norm_tag`: a fine-tuned checkpoint carries its own norm stats
      ⇒ `norm_tag: null` (the base `outputs/MolmoAct2-SO100_101` needs
      `so100_so101_molmoact2`). The smoke loaded fine but did not stress this.

---

## Critic depth read — ✅ DONE 2026-06-14

Decisions (user): critic OWNS its depth modules (not shared with actor); depth enters
as a **co-evolving stream** (not plain tokens). Adapted to the critic's architecture
(bidirectional, single end-of-stack read): the critic gets its own `depth_encoder` +
`depth_blocks` (DepthStreamBlock × critic_llm_depth, attending the critic's wrist-cam
tokens sliced from its obs embeds) + `depth_read_proj`; the co-evolved FINAL state is
appended to the sequence `[obs | depth | value-queries]` (fully bidirectional, no
gate/sink — critic isn't frozen). `rl_molmoact2.py`: `MolmoAct2Critic.compute_depth_tokens`
+ `forward(depth_tokens=)` + `_forward_critic_impl` wiring (V(s) live, V(s') EMA on
next_depth). Forks resolved: **TD-grad-into-encoder** → critic's own encoder, isolated;
**EMA boundary** → depth modules are part of the critic ⇒ ride `critic_target`'s generic
lerp. Trainer fix: `_apply_critic_freeze` keeps `depth_*` trainable (else-branch would
freeze them). **GPU-validated 2026-06-20**: the critic depth forward runs in the train
step (V(s)/V(s′) both go through it). The bf16-critic Fourier-PE dtype crash (bug 1 in
the status summary) surfaced here first — the critic runs bf16 while the actor stream is
float32 — and was fixed by casting `pe` to the proj dtype.

---

## Status / what's left

Phases 1–6 are DONE; the actor + critic depth paths are built, GPU-train-validated, and
hardware-inference-validated (2026-06-20). **No code work is outstanding for a first real
run.** What remains is the hardware/data track above — wrist mount, calibrated K, `z_max`
sanity, hole-fill-off recordings — plus a training run long enough for the gate to grow
so depth measurably changes behavior. Optional later: the parked CUDA-graph payoff (§3c,
moot under RTC) and the deferred telemetry probes (§Phase 5).
