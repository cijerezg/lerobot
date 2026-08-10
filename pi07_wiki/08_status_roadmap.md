# 08 — Status & roadmap

As of 2026-08-09. Values read from the code and root `config_rl.yaml`, not from
prose — several comments in those files are older than what they describe.

## Current run

Between runs. v9 was started 2026-08-08 and stopped at step 0; nothing since.

Config as it stands (`rl_offline.py`, skip_critic): the merged 3-source corpus
(38 eps / 165,740 frames) with `rebot_val-annotated-v2` held out; flow + FAST CE
with knowledge insulation, anchor encoding, chunk_size 30 / image_stride 3,
batch 32 × grad_accum 4 = 128 effective, 30000 steps, pretrained merge every 8000
at α 0.2, depth on with `depth_lr: null` (inherits joint 5e-5) and
`depth_warmup: false`. Generation CE is
**off** (`subtask_loss_weight: 0.0`). Both auxiliary losses are **off**. The full
probe suite is on, plus held-out `val_loss_*` on 128 frames.

## Feature matrix

| Feature | Code | In current config |
|---|---|---|
| Flow + FAST CE (both, knowledge insulation) | done | on |
| Anchor action encoding | done | on |
| FAST tokenizer alphabet fix (snap to nearest encodable bin) | done 2026-08-08, unit-tested | on — unconditional, in `_tokenize_discrete_action` |
| Flow trajectory auxiliary (path/shape/terminal/direction, thresholded) | done 2026-08-08 | **off** — thresholds not yet chosen (05 §2.1) |
| FAST-logit auxiliary (ordinal/path/shape) | done 2026-08-09, unit-tested, construction validated standalone | **off** — never run end-to-end (05 §2.2) |
| Held-out `val_loss_*` (flow, FAST CE, both aux) | done 2026-08-08 | on, 128 frames |
| Depth point-map into the VLM prefix | done, validated e2e | on — no gate, no DepthStream (both deleted 2026-08-03) |
| Critic depth read | still on the old stream blocks; not migrated | off (skip_critic) |
| Subtask generation (two-prompt, HL decode) | done | **training off** (`subtask_loss_weight: 0.0`); inference budget 64 tokens |
| ~~MEM summary memory (hold/update seam)~~ | **removed** — seam, `materialize_summaries` and `loss_summary_ce` all deleted | — |
| Metadata steering (quality/mistake) | done | on |
| History: buffer sample + actor deque | done, parity-tested | on |
| History consumption: MEM video encoder (images) + continuous state tokens | done | on |
| ~~History consumption: prompt path (frames as prompt images + states as text)~~ | deleted 2026-07-22 (LLM token explosion) | — |
| History consumption: depth (time-embedded pointmap slots) | done 2026-07-21 | on |
| History consumption: action | not built (causal-confusion ablation) | — |
| Distributional critic (HL-Gauss 201, σ-ratio 8.0) | done | off |
| RECAP advantage conditioning | done (pi05-proven, molmoact2 seam) | off |
| CUDA graphs with depth | parked (moot under RTC) | — |

## Before the next run

1. **Decide the flow-aux thresholds — or leave them `null`.** `action_trace` runs
   unconditionally and writes p10/p75/p90 of the four exact quantities the loss gates
   on, so the numbers arrive free at step 0 of any run; the only constraint is that
   config is read at launch. Its sample is ~17 anchors, though, so `null` thresholds
   (term applies to every valid sample) plus the in-training `action_aux/*_p75` keys
   is the better first move. Files older than 2026-08-08 have no `trajectory` block
   (05 §2.1).
2. **Smoke the discrete aux end-to-end** before trusting it in a long run — the
   real-vocabulary table build and the per-example loop have never seen a training
   batch; watch step time (05 §2.2).
3. **Decide the starting checkpoint.** `pretrained_path` currently points at
   `all-v5/checkpoints/001200`, which was trained through the FAST alphabet bug
   (4.2% of chunks scrambled under anchor encoding). Resuming from it inherits
   whatever that taught; the alternative is restarting from `outputs/MolmoAct2`.
4. **Bump `offline_output_dir`** — it still reads `-v9`, and that directory already
   exists from the aborted run.
5. The v6 memorization result (val/train flow 4.64x, FAST CE 2.97x, on an effective
   sample size of 38 episodes) is the backdrop for all of the above: prefer changes
   that are checkable on held-out frames, which is what `val_loss_*` is for.

## Then

1. Evaluate the run; action_trace `skill_vs_hold` / `skill_vs_mean` and clearance
   per checkpoint.
2. Compare against the no-history baseline (causal-confusion check). Remaining
   unbuilt: action-history consumption (still a causal-confusion ablation).
3. Hardware rollout (cable verified): live HL decode over a long episode;
   `z_max_mm` sanity on wrist-mounted depth.
4. Online data: label recorded rollouts post-hoc (metadata before mixing —
   π0.7's ablation), then critic + advantage conditioning back on.

## Parked / ideas (full text in [archive/ideas_to_revisit.md](archive/ideas_to_revisit.md))

- **Ledger memory** (candidate key innovation): m_t = (done; remaining) — a
  recurrently *generated* memory with a prospective half; supervision free from
  existing annotations ("remaining" = hindsight compression of future segments);
  crisp eval (does "remaining" shrink correctly). Risks: plan-update on
  divergence, more HL tokens (measure latency).
- **HL decode order**: memory-first (current) vs subtask-first — π0.7 paper does
  not specify; pure `build_generation_answer`/`parse_generation_answer` swap;
  legit ablation once training runs exist.
- **CFG on the metadata clause** at inference (β 1.3–2.2).
- **Subtask-level exploration** for online RL (frozen HL is greedy+cached;
  RL trains only low-level flow — a policy-gradient lever on subtask choice).
- **Speed metadata backfill** (per-segment duration, work-normalized).
- **Depth↔RGB shared 3D PE** (fallback b) if depth→wrist-cam attention doesn't
  pick up semantics; admit-mass telemetry; point-map contact-sheet probe.
- **Recon aux loss** for the depth stream — only if flow-only fails to learn
  geometry.
- Leader shadow takeover (blocked: encoder-only leader); upstream merge
  (19-conflict chunked rebase plan in memory).

## Known footguns (the ones that actually bit)

- Freeze else-branches: whitelist every new from-scratch module in
  `_apply_actor_freeze` / `_apply_critic_freeze` or its gate never trains.
- Read gate params in-region under gradient checkpointing (no closure-captured
  non-leaf tensors).
- bf16/float32 boundary: shared modules must cast PE/buffers to the consumer's
  weight dtype.
- Cache fingerprint: dtype/size mismatch silently falls back to video decode;
  declare `image_storage_*`; stride mismatch is a hard error.
- Config fields must be real declared dataclass fields or YAML overrides get
  stripped.
- `rl_learner.py` ignores `cfg.resume` — always fresh; transfer online data via
  `additional_offline_dataset_paths`.
- Depth influence: watch `depth_rgb_rms_ratio`, `depth_late_early_rms_ratio`,
  component gradient norms, and the 2×2 probe
  (`probes/depth_modality_probe.py`). The α-gate metrics are gone (2026-07-26) and
  so are `depth_attn_mass_*` / `depth_bias_mean` (2026-08-03, with the gate itself).
- **A tokenizer with no UNK deletes, it does not error.** The FAST BPE silently
  dropped any character outside its alphabet; because the coefficient layout is
  frequency-major, one dropped DC value slid every later coefficient into a
  neighbouring joint's cell and scrambled the chunk. It cost 4.2% of training chunks
  and was invisible in every loss curve. Assert the round-trip length after
  encoding, not just that encoding returned something
  ([fast_tokenizer_alphabet_bug.md](fast_tokenizer_alphabet_bug.md)).
- **Measure the deployed representation.** The first prevalence estimate for that
  bug was ~7x off because it was taken on absolute actions and whole-chunk stats
  rather than the anchor-encoded, per-timestep-normalized path the model is
  actually trained on.
- **Comments in `config_rl.yaml` and the wiki drift faster than the code.** Several
  were found stating the wrong effective batch size, a probe suite that was off when
  it was on, and telemetry keys that no longer exist. Read the value, not the note,
  and fix the note when you notice.
