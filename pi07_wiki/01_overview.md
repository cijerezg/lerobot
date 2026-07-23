# 01 — Overview

## What pi07 is

"pi07" is the project name for our custom policy stack: the **π0.7 recipe**
(hierarchical subtask decoding, language memory, metadata steering, observation
history) re-implemented on the **MolmoAct2** VLA, plus one major addition the π
lineage does not have — a **metric depth stream** from the wrist-mounted RealSense
D405, injected into the frozen action expert with a zero-init gate.

Four subsystems, each with its own wiki page:

1. **Base model** ([02](02_base_model.md)) — MolmoAct2: a Qwen2.5-class VLM with a
   SigLIP vision tower and a DiT-style flow-matching action expert that
   cross-attends the VLM's per-layer K/V. Fine-tuned on rebot B601 (7-DOF) data
   with anchor-encoded actions.
2. **Depth** ([03](03_depth.md)) — the D405 depth image is back-projected to a
   metric point map, tokenized into 192 patch tokens, co-evolved through a
   DepthStream (mixture-of-transformers style), and read by the action expert via
   an additive gated SDPA that is bit-identical to the depth-free model at init.
3. **Memory & prompts** ([04](04_memory.md)) — three π0.7-style channels rendered
   into the prompt: short-term history (past states as discrete-state strings +
   past frames as extra images), a high-level subtask + recurrent MEM summary
   memory decoded every 4 s by the same model (two-prompt design), and metadata
   steering (quality / mistake clauses; train on everything, prompt for the best).
4. **RL training & critic** ([05](05_training.md)) — offline BC-style training via
   `rl_offline.py` (flow loss + discrete FAST CE + subtask/summary generation CE),
   with an optional distributional critic (HL-Gauss, 201 bins) for RECAP-style
   advantage conditioning. The current run is `skip_critic: true`.

## System diagram

```
                         ┌────────────────────────────────────────────────┐
   top RGB ──┐           │                 MolmoAct2 VLM                  │
   wrist RGB ┼──► vision │  SigLIP ViT → adapter → LLM decoder (L layers) │
   task text ┘   tower   │  per-layer K/V cached                          │
   state (discrete toks) │        │ per-layer K/V         │ lm_head       │
   prompt clauses:       └────────┼───────────────────────┼───────────────┘
    subtask, memory,              │                       │
    metadata, history             │                  HL decode (every 4 s):
                                  ▼                  "Memory: m_{t+1} Subtask: s"
   wrist depth ─► back-project ─► DepthStream        (two-prompt design)
   (D405 uint16)  → 192 tokens    (M co-evolving
                                  blocks, attends
                                  wrist-cam K/V)
                                  │ per-layer depth K/V
                                  ▼
                        ┌──────────────────────────┐
                        │      Action expert       │   out = SDPA(q,K_ctx,V_ctx)
                        │  (frozen-init DiT, flow  │       + tanh(α_ℓ)·SDPA(q,[K_d,k_⋆],[V_d,0])
                        │   matching, L blocks)    │
                        └──────────────────────────┘
                                  │
                                  ▼
                    action chunk (50 steps, anchor-encoded)
                    executed via RTC (real-time chunking)
```

## The model-agnostic rule

MolmoAct2 is the *current* policy, not the permanent one. Every feature splits into
a **generic layer** (replay buffer history gather, annotation pipelines, metadata
loading, RTC bookkeeping — lives in `rl/` and `data_processing/`, never imports a
policy) and a **policy seam** (prompt rendering, subtask token decode, how tensors
enter the trunk — small, rewritten per policy). Swapping the policy means
re-implementing the seam, not the system.

## Glossary

| Term | Meaning |
|---|---|
| **HL / LL** | High-level (subtask + memory text decode) vs low-level (action chunk) policy — same network, two prompts |
| **MoT** | Mixture-of-transformers: parallel token streams sharing the layer stack with distinct weights (DepthVLA pattern) |
| **MEM summary / m_t** | Recurrent language memory: a text summary updated each HL tick, conditioned on its previous value |
| **Gate α_ℓ** | Per-layer scalar; tanh(α_ℓ) scales the additive depth read; zero-init ⇒ bit-identity at step 0 |
| **Sink** | Extra zero-value key column in the depth softmax with a learned logit, allowing the read to abstain to exactly 0 |
| **Anchor encoding** | Actions trained as offsets from the current state: a_enc = a − s (states stay absolute) |
| **RTC** | Real-time chunking: asynchronous action-chunk execution with prefix-conditioned re-planning |
| **RECAP / advantage clause** | π0.7-style critic-conditioned prompting ("The advantage is positive.") — currently off (skip_critic) |
| **FAST** | Discrete action tokenizer; discrete action CE trains alongside the flow loss (`action_mode: both`) |
| **Knowledge insulation** | Training the discrete path without letting it disturb the continuous path (both-mode training detail) |
| **rebot B601** | Target arm; 7-dim state/action: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper |

## Timeline (major landings)

| When | What |
|---|---|
| 2026-06-15 | Depth point-map + DepthStream built (replaced TSDF path entirely) |
| 2026-06-20 | Depth validated end-to-end: GPU training + on-robot inference; gate-freeze bug found and fixed |
| 2026-07-04 | SO-101 frame conversion removed; raw arm frame everywhere (rebot migration) |
| 2026-07-13 | Memory Phases 1–3 + 5 code complete (history sampling, actor deque, subtask generation, metadata) |
| 2026-07-15 | Done-list removed, replaced by MEM recurrent summary |
| 2026-07-17 | Summary seam built; first real rebot dataset (`rebot_socks_v1`) recorded |
| 2026-07-18 | Memory-first HL decode order; metadata design revised + review tooling built; dataset annotated (`rebot-socks-annotated-v2`) |
| 2026-07-19 | State audit; generation training + inference_advantage reconciled in config |
| 2026-07-20 | Phase 6 start: proprio-history prompt clause wired; image history as extra prompt images |
