# 06 — Inference runtime

Online inference is a thin wrapper (`rl/inference_async.py`) over the RTC actor
runtime ([`rl/rtc_actor_runtime.py`](../src/lerobot/rl/rtc_actor_runtime.py)).
Three cooperating threads share `RTCSharedState` under one lock: an **env worker**
(control loop at env fps, executes actions, pushes transitions), an **inference
worker** (builds batches, runs the policy), and the gRPC actor↔learner transport.

## 1. Control flow per step

1. Env worker reads the latest action chunk, executes one step, assembles the
   transition (state, action, reward, depth sidecar, `subtask_index` stamp), and
   pushes one history entry (state, executed action, depth) to the rolling deque.
2. Inference worker, at its own cadence: takes the latest observation, merges
   history windows (`assemble_history_windows`) and the current HL context
   (subtask name + metadata defaults) into the batch, runs the policy through
   **RTC** — the flow-matching denoise loop with prefix attention over the
   still-executing chunk (`execution_horizon: 5`, LINEAR schedule,
   `max_guidance_weight: 10`). The RTC denoise loop is eager (never touches CUDA
   graphs — why graphs-off for depth costs nothing).
3. Every `subtask_regeneration_interval = 4 s`, the inference worker additionally
   runs the **HL decode** ([04 §3](04_memory.md)): generation prompt conditioned
   on `current_summary`, greedy decode, parse memory-first answer, update
   `RTCSharedState` (subtask name/index + summary). Gated on
   `subtask_max_new_tokens > 0`. Must run in the inference thread — the
   action-expert patches are not thread-safe against a concurrent worker.

Episode restart clears history, subtask, and summary; **intervention clears
nothing** (training buffers keep contiguous frames across takeover; teleop action
= executed action).

## 2. Depth at inference

The D405 delivers raw uint16 z16 (0.1 mm/level, spatial + hole-fill filters OFF —
masking lives in the encoder). The policy's `prepare_context` patch back-projects
and runs the DepthStream **once per control step**, caching per-layer depth K/V
across all denoising steps ([03 §B.3](03_depth.md)). `back_project` accepts the
bare unbatched `(H, W)` live frame. A one-shot log line confirms depth presence on
the inference path (guards the silent-null-bank failure mode).

## 3. Prompt at inference

`build_inference_batch`
([rl_molmoact2_trainer.py:855](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L855))
assembles the same clauses as training, with the steering defaults:

| Clause | Inference value |
|---|---|
| subtask | latest HL decode (absent until the first decode) |
| metadata | `{quality: 5, mistake: false}` — π0.7 "prompt for the best" |
| history | live deque windows (when `memory.history_keys` set) |
| advantage | `inference_advantage` — **null under skip_critic** (clause absent; a hardcoded 1.0 would be out-of-distribution against advantage-free training) |
| summary | generation prompt only, never the action prompt |

## 4. Offline eval

`probes/offline_inference.py`: samples held-out frames, runs both the LL (action
prediction vs GT) and the HL query — per frame conditioned on the **GT** memory
via `summary_label_spans` (the same hold/update rule as training) — and renders a
panel with GT subtask + GT memory + per-checkpoint predictions. The molmoact2
adapter reuses `generate_subtask_text`; it returns None (panel empty) when
`subtask_max_new_tokens = 0`. `probes/pointmap_bit_identity.py` checks gate-0
bit-identity against the real checkpoint
(`uv run python -m lerobot.probes.pointmap_bit_identity --config config_rl.yaml`).

## 5. Latency notes

- 30 Hz control target; `torch_compile` currently off (rule out compile noise on
  the first depth run).
- HL decode budget: 128 tokens covers "Memory: … Subtask: …" (12 s summaries are
  multi-sentence). Parked optimization: skip the memory-span decode on ticks where
  the subtask didn't change — measure first (memory-first order pays full summary
  decode before the LL gets its subtask).
- Depth stream: <1% of forward, run once per control step.

## 6. Hardware context

- **rebot B601** on the daisy-chain bus; there was an intermittent power break near
  the base (cable replacement — verify with `wiggle_rebot_cable.py` before
  trusting rollouts).
- **D405** wrist-mounted on rear USB3; factory intrinsics in the config.
- **Leader arm** (planned shadow-takeover teleop): the Seeed leader is
  encoder-only — its joints cannot be driven, so the mirror-and-grab takeover
  design needs the HD variant or a clutch/delta scheme. `send_feedback` is the
  software hook (API verified).
