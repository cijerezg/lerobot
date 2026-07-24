# 08 — Status & roadmap

As of 2026-07-21.

## Current run

Offline training on `rebot-socks-annotated-v2` (`rl_offline.py`, skip_critic):
flow + FAST CE + generation CE (weight 1.0, `subtask_max_new_tokens: 128`),
subtask/metadata prompt conditioning, depth on (gate training at `depth_lr` 5e-4),
pretrained merge at step 8000. The 2026-07-19 audit findings (generation training
silently off, `inference_advantage` OOD clause, per-step gate print) are
reconciled in config.

## Feature matrix

| Feature | Code | In current run |
|---|---|---|
| Flow + FAST CE (both, knowledge insulation) | done | on |
| Anchor action encoding | done | on |
| Depth point-map + DepthStream + gated read | done, validated e2e | on (gate growing from 0) |
| Critic depth read | done, GPU-validated | off (skip_critic) |
| Subtask generation (two-prompt, HL decode) | done | training on; rollout pending hardware |
| MEM summary memory (hold/update seam) | done, verified on real training 2026-07-19 | on |
| Metadata steering (quality/mistake) | done | on |
| History: buffer sample + actor deque | done, parity-tested | on (2026-07-22) |
| History consumption: MEM video encoder (images) + continuous state tokens | BUILT 2026-07-22, unit-tested (04_memory §2.4, build plan Phase 6); real-batch smoke pending (`smoke_mem_encoder.py`) | next run |
| ~~History consumption: prompt path (frames as prompt images + states as text)~~ | deleted 2026-07-22 (LLM token explosion) | was on in current run — those checkpoints orphaned |
| History consumption: depth (time-embedded pointmap slots) | done 2026-07-21 | on (slots 4→5 with Phase 6 config) |
| History consumption: action | not built (causal-confusion ablation) | — |
| Distributional critic (HL-Gauss 201, σ-ratio 8.0) | done | off |
| RECAP advantage conditioning | done (pi05-proven, molmoact2 seam) | off |
| CUDA graphs with depth | parked (moot under RTC) | — |

## Near-term order

1. Finish/evaluate the current offline run; offline_inference probe: subtask +
   memory quality per checkpoint.
2. Build Phase 6: MEM video encoder + continuous state history
   (04_memory §2.4; prompt-image path deleted same day its smoke passed —
   design superseded it). Then train and compare offline eval vs the
   no-history baseline (causal-confusion check). Remaining unbuilt:
   action-history consumption (still a causal-confusion ablation).
3. Hardware rollout (cable verified): live HL decode + summary bookkeeping over a
   long episode; `z_max_mm` sanity on wrist-mounted depth; gate long-run growth.
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
- Watch `pointmap_gate_absmax`, not the mean.
