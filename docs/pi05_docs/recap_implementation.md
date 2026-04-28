# RECAP Implementation

## What is RECAP and why does it matter

[RECAP](https://arxiv.org/pdf/2511.14759) is the RL algorithm developed by [Physical Intelligence](https://www.pi.website/) that was used to train the $\pi_{0.6}$ model.

RECAP proposes an **advantage-conditioned VLA**, where the advantage comes from a critic that is trained alongside the policy (the VLA itself). The advantage is computed from the critic, binarized, and passed to the policy as an extended part of the prompt — so the same VLA decoder produces actions conditioned on whether the current state is "good" or "bad" relative to expectation.

This is useful for two reasons:

1. **The policy can learn from its own experience.** There is a grounding signal about the *goodness* of actions (the advantage from the critic) that is completely driven by reward. Without this signal, when the policy makes a mistake during online rollouts, it would just imitate that mistake. With advantage conditioning, the model is told "this is what a low-advantage state looks like," which lets it differentiate trajectories during training and lean toward the high-advantage mode at inference (when we always feed it `Advantage: positive`).
2. **Suboptimal demonstrations become useful.** The critic learns to distill which parts of a demonstration are good and which are bad — i.e., which actions have high advantage versus low. This is a big deal in the small-data regime where curated, perfectly clean demos are expensive: RECAP lets you keep the messy ones and still extract signal from them.

This document is a deep dive into how that idea is implemented in this repository, the deliberate divergences from the paper, and the code paths that wire everything together. For a higher-level usage guide see the [main RL README](../../src/lerobot/rl/README.md).

---

## How this implementation differs from the paper

RECAP was designed for a much larger model and a much larger compute budget than what is targeted here. The version in this repo keeps the core idea — advantage-conditioned VLA with a critic trained jointly — and trades a few of the paper's design choices for things that are simpler, work in the small-data regime, and fit on a single workstation.

| Aspect | Original RECAP | This implementation | Reason for the divergence |
|--------|---------------|---------------------|---------------------------|
| **Backbone** | Gemma 3 4B | PaliGemma 3B + ~300M action expert ($\pi_{0.5}$) | $\pi_{0.5}$ weights are open and run on a single GPU; PaliGemma processes 224×224 vs Gemma 3's 896×896 |
| **Return estimator** | On-policy Monte-Carlo returns over full trajectories | Off-policy 1-step TD via the Bellman equation with a target network | Off-policy TD lets us train from a fixed offline buffer and from a mixed online/offline buffer without waiting for full rollouts |
| **Critic loss** | Cross-entropy over discretized returns (HL-Gauss style) | MSE on continuous values | MSE is a smaller surface area to debug; we never observed the kind of return distribution that needs categorical heads at this scale |
| **Advantage labels** | Multiple discretized return bins | Two bins: `negative` / `positive` (split at $\tanh(A/\alpha) \geq 0.35$) | Two bins are enough to teach the policy a directional signal at the data scales we run, and the prompt stays short |
| **Advantage dropping** | Stochastic dropping of advantage labels (classifier-free guidance style) | Always pass the advantage label | We don't currently rely on guidance at inference; revisit if we want CFG-style conditional sampling |

Everything else — the value head shape, the binning thresholds, the Polyak coefficient — is documented below.

---

## Architecture overview

```
PI05RLPolicy
├── actor: PI05RLPytorch (extends PI05Pytorch)
│   └── paligemma_with_expert: PaliGemmaWithExpertModel
│       ├── vision_encoder: SigLIP-400M          [selectively frozen]
│       ├── multi_modal_projector                 [selectively frozen]
│       ├── language_model: Gemma 2B (18 layers)  [selectively frozen]
│       ├── action_expert: Gemma ~300M            [always trained]
│       │   ├── action_in_proj
│       │   └── action_out_proj
│       └── time_mlp_in / time_mlp_out            [always trained]
│
├── critic: Pi05TransformerCritic                 [trained]
│   ├── rotary_emb
│   ├── layers: 6 GemmaDecoderLayers
│   ├── norm: RMSNorm
│   ├── value_queries: [1, 32, 2048]              [learned]
│   ├── token_proj: Linear(2048, 512)
│   └── value_head: Linear(16384, 1024) → SwiGLU → Linear(512, 1)
│
└── critic_target: Pi05TransformerCritic          [no gradients, Polyak-averaged]
```

The critic is a **separate transformer**, not a head bolted onto the actor. It shares the input encoders' outputs (vision features and text embeddings) but has its own decoder layers and its own learned query tokens. We initialize its layers by deep-copying the first $N$ layers of the pretrained actor, which gives it a sensible starting point for what manipulation features look like.

---

## Critic architecture

The critic (`Pi05TransformerCritic`) is a transformer-based state value function $V(s)$ that estimates the expected discounted return from a given state.

### Components

- **Inputs**: Vision features $(B, N_{\text{patches}}, 2048)$ + text embeddings $(B, L_{\text{text}}, 2048)$.
- **Value query tokens**: 32 learned tokens $\mathbf{Q} \in \mathbb{R}^{1 \times 32 \times 2048}$, expanded to the batch dimension. These tokens are appended *after* vision and text in the sequence, and are what the value head reads out. The intuition is that the queries act as a learned bottleneck: the transformer is forced to summarize the value-relevant information about the scene into those 32 slots via self-attention.
- **Transformer stack**: 6 `GemmaDecoderLayer`s (configurable via `critic_llm_depth`) with RoPE positional embeddings. Initialized by deep-copying layers from the pretrained actor's language model.
- **Value head**: Projects the 32 query token outputs to a scalar.

$$V(s) = \text{MLP}\left(\text{flatten}\left(W_{\text{proj}} \cdot \mathbf{Q}_{\text{out}}\right)\right)$$

where:
- $\mathbf{Q}_{\text{out}} \in \mathbb{R}^{B \times 32 \times 2048}$ are the query token outputs after the transformer.
- $W_{\text{proj}} \in \mathbb{R}^{2048 \times 512}$ projects each token to 512 dims.
- `flatten` reshapes $(B, 32, 512) \to (B, 16384)$.
- MLP: `Linear(16384, 1024) → SwiGLU → Linear(512, 1)`.

The SwiGLU activation is $\text{SwiGLU}(\mathbf{x}) = \text{SiLU}(\mathbf{x}_{\text{gate}}) \odot \mathbf{x}_{\text{value}}$, where the linear layer outputs are split in half along the last dim into a gate half and a value half.

### Target network

A separate `critic_target` is maintained via Polyak averaging:

$$\boldsymbol{\theta}_{\text{target}} \leftarrow (1 - \tau) \cdot \boldsymbol{\theta}_{\text{target}} + \tau \cdot \boldsymbol{\theta}_{\text{critic}}, \quad \tau = 0.005$$

This is updated after every critic optimization step. Only parameters with `requires_grad=True` are averaged — anything frozen (e.g., embedding layers initialized from the actor and not unlocked) is skipped. This is the standard trick for stabilizing TD learning: bootstrapping off a slowly-moving target prevents the value estimate from chasing its own tail.

### Why a separate critic prompt

The critic gets a prompt that **does not contain the advantage label**. The actor prompt is:

```
Task: {task}, State: {state}, Advantage: {label};\n
```

while the critic prompt is:

```
Task: {task}, State: {state};\n
```

If the critic saw the advantage label, we would have a circular dependency: the advantage is a function of $V(s)$, and $V(s)$ would itself depend on the binarized advantage. Cutting that loop also makes the value estimate more transferable across rollouts where we feed different advantage labels to the actor.

---

## Advantage conditioning — the bridge between critic and actor

The advantage tells the actor whether the current state-action trajectory is doing better or worse than the critic expected. This is the only place the critic affects the actor: there is no policy gradient. The actor still trains with a flow-matching loss; the critic just decides which conditioning string is glued to the prompt.

### Computation

At each actor update step:

1. **Critic forward pass** (no gradients on the actor side): compute $V(s)$ with the live critic and $V_{\text{target}}(s')$ with the target.
2. **Bellman target**:

   $$Q_{\text{target}} = \frac{r}{c_r} + \gamma \cdot V_{\text{target}}(s') \cdot (1 - \text{done})$$

   where $c_r$ is `reward_normalization_constant` (1.0 in dataclass; 5.0 in the default config) and $\gamma$ is `discount` (0.97). The target is then clamped: $Q_{\text{target}} \leftarrow \min(Q_{\text{target}}, 0.05)$. The clamp reflects the bounded-from-above reward structure of these manipulation tasks (a successful episode tops out near 0 after normalization).
3. **Raw advantage**: $A = Q_{\text{target}} - V(s)$.
4. **Overrides**:
   - If the transition is from the **golden dataset** (clean demonstrations flagged with `is_golden=True`): force $A = 1.0$.
   - If the transition is a **human intervention** (flagged with `is_intervention=True` from the leader-arm teleop): force $A = 1.0$.

The overrides exist because the critic is unreliable on data it has barely seen, especially early in training, but we already know these transitions are high quality by construction. We don't want the actor learning that a clean demo is "negative" just because $V(s)$ happens to be optimistic that step.

### Tokenization

The raw advantage is squashed and discretized before being injected into the prompt:

1. **Scale**: $A_{\text{scaled}} = A / \alpha$, where $\alpha$ is `advantage_scaling` (1.0 in dataclass; 0.20 in the default config — small $\alpha$ means even modest advantages saturate $\tanh$).
2. **Squash**: $A_{\text{squashed}} = \tanh(A_{\text{scaled}})$, mapping to $[-1, 1]$ and damping outliers.
3. **Bin**: with boundaries $[-1.0, 0.35, 1.0]$:
   - $A_{\text{squashed}} < 0.35 \Rightarrow$ `"negative"`
   - $A_{\text{squashed}} \geq 0.35 \Rightarrow$ `"positive"`
4. **Inject**: the prompt becomes `"Task: {task}, State: {state}, Advantage: {label};\n"`.

The 0.35 split is asymmetric: a transition is only labeled `positive` once the squashed advantage is well above zero. In practice this means most random transitions are labeled `negative` and only the strongly-better-than-expected ones are `positive` — which matches what we want at inference, where we feed `Advantage: positive` and ask the model to imitate that mode.

### Knowledge insulation

When `knowledge_insulation` is enabled (default in the config), the advantage label affects only the action expert's path through the network, not the vision/language perception. The VLM's internal representations of the scene therefore stay clean across advantage labels, and only the action decoding is steered. This prevents the advantage signal from corrupting learned visual features and keeps inference cheap when we want to vary the advantage label without re-encoding the image.

### What the actor sees at inference

At deployment time, the critic is not consulted. Instead the actor's prompt is built with a **fixed `inference_advantage`** value (default `1.0`), so the policy is always conditioned on `Advantage: positive`. This is the whole point of training with binarized labels: at inference you can pick the mode you want and the model will produce actions consistent with that mode. The fixed inference label is set in `actor_pi05_async_utils.py`, `inference_utils.py`, and used as a placeholder in `preprocess_batch_for_pi05` — note that during training this placeholder is overwritten by the live critic-computed advantage in `_prepare_actor_batch` before the actor forward pass.

---

## Loss functions

### Actor loss

The actor loss combines three terms, all optimized jointly:

$$\mathcal{L}_{\text{actor}} = w_{\text{flow}} \cdot \mathcal{L}_{\text{flow}} + w_{\text{action\_ce}} \cdot \mathcal{L}_{\text{action\_ce}} + w_{\text{subtask\_ce}} \cdot \mathcal{L}_{\text{subtask\_ce}}$$

**Flow matching loss** (MSE on the velocity field). The action expert is a flow-matching model: given a noisy action $\mathbf{x}_t = t \cdot \boldsymbol{\epsilon} + (1-t) \cdot \mathbf{a}$ at time $t \sim U(0,1)$, it predicts the velocity that maps noise back to the clean action. We supervise that velocity directly:

$$\mathcal{L}_{\text{flow}} = \frac{1}{Td} \sum_{t,j} \left(u_{t,j} - v_{t,j}\right)^2$$

where $\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{a}$ is the ground-truth velocity (noise minus clean actions) and $\mathbf{v}$ is the model's predicted velocity via `action_out_proj`. Only the first $d$ dimensions (the actual action dims for the robot) contribute — padding dims are masked out.

**Action cross-entropy loss** (FAST token prediction). The same forward pass also predicts FAST action tokens autoregressively:

$$\mathcal{L}_{\text{action\_ce}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p(y_i \mid \mathbf{x})$$

where $\mathcal{M}$ is the set of unmasked action token positions and $y_i$ are the ground-truth FAST token indices. This gives the model a discrete-token grounding for the action chunk in addition to the continuous flow-matching prediction.

**Subtask cross-entropy loss** (subtask token prediction):

$$\mathcal{L}_{\text{subtask\_ce}} = -\frac{1}{|\mathcal{M}_s|} \sum_{i \in \mathcal{M}_s} \log p(y_i \mid \mathbf{x})$$

Same structure as the action CE but over subtask token positions. For online transitions where no subtask annotation exists yet (`subtask_index == -1`), the subtask attention mask is zeroed so those positions don't contribute to the loss while still being available as conditioning context.

All three weights default to 1.0 and are configurable.

### Critic loss

$$\mathcal{L}_{\text{critic}} = \frac{1}{B} \sum_{i=1}^{B} \left(V(s_i) - Q_{\text{target},i}\right)^2$$

with $Q_{\text{target}}$ computed exactly as in the advantage section above (normalized reward + discounted target value, clamped at 0.05). The clamp matters: without it, an over-optimistic target would push $V$ above the structural ceiling of the reward function and destabilize the advantage signal.

---

## Training loop

### Each optimization step

```
1. CRITIC UPDATE
   - Sample batch (mix online + offline if both present)
   - Preprocess WITHOUT advantage tokens (critic prompt)
   - Forward: V(s), V_target(s') → target_q = r/c_r + γ * V_target(s') * (1 - done)
   - Loss = MSE(V(s), target_q)
   - Backward, clip gradients (max norm: 2.0 in default config), step optimizer
   - Repeat for gradient_accumulation_steps (16 in default config)

2. ACTOR UPDATE (if step ≥ warmup AND step % policy_update_freq == 0)
   - Sample NEW batch
   - Run critic to compute live advantage A = target_q - V(s),
     then apply golden / intervention overrides (force A = 1.0)
   - Tokenize: inject the binarized advantage label into the actor prompt
   - Forward: predict velocity field, action tokens, subtask tokens
   - Loss = weighted sum of flow + action_ce + subtask_ce
   - Backward, clip gradients, step optimizer
   - Repeat for gradient_accumulation_steps

3. TARGET NETWORK UPDATE
   - Polyak: θ_target ← (1 - τ) * θ_target + τ * θ_critic, with τ = 0.005
```

The critic and actor each get their own batch sample so the critic's MSE doesn't see exactly the same minibatch the actor is currently fitting — this is a small but useful decoupling that keeps the TD updates from being too correlated with the policy improvement step.

### Online vs offline

**Offline training** (`offline_learner_pi05.py`, `offline_learner_val_pi05.py`):
- Trains on a fixed dataset with no actor in the loop.
- Uses `accelerate` for multi-GPU distributed training.
- All advantage values are still computed live by the critic during the actor update (the `inference_advantage` config field is *not* used as a training-time fallback — it's only for deployment).
- `offline_learner_val_pi05.py` additionally runs validation probes at `val_freq` intervals (attention maps, VLM PCA/UMAP projections, action representation analysis).
- Typical run: 8,000–10,000 steps to give the policy and critic a sensible initialization before going online.

**Online training** (`learner_pi05.py` + `actor_pi05_async.py`):
- Learner receives transitions from the actor via gRPC.
- Actor runs asynchronous inference on the robot at 30 Hz using RTC, with leader-arm interventions toggled by key `5`.
- Online and offline buffers are mixed roughly 50/50 per batch — the offline data acts as a stabilizing prior, preventing the policy from forgetting what it learned offline while it incorporates new on-robot experience.
- Critic advantages are computed live every actor update.
- Policy weights are pushed to the actor every `policy_parameters_push_frequency` steps (180 in the default config).

### Subtask token passthrough

For online transitions, the actor stores the exact subtask tokens it used during inference, and those tokens are passed through directly to the learner's actor update — bypassing the preprocessor's tokenization. This guarantees the flow loss is computed against the same conditioning context the actor actually saw, even if the subtask hadn't been annotated yet by the time the transition was sampled.

---

## Parameter freezing

The full $\pi_{0.5}$ + critic stack is too large to train end-to-end on a single workstation, so only a subset of parameters is unfrozen. The strategy is configured via `policy.trainable_params` and resolved at learner startup.

| Component | Config field | Default in `config-hiserl.json` | Total layers |
|-----------|--------------|---------------------------------|--------------|
| Vision tower (SigLIP) | `vision_tower` | `5` (train layers 5–26) | 27 |
| Multi-modal projector | `multi_modal_projector` | `true` | 1 |
| Language model (Gemma) | `language_from_layer` | `0` (train all) | 18 |
| Action expert | — | always trained | ~300M params |
| Critic decoder layers | `critic_language_from_layer` | `1` (train layers 1–5) | 6 |
| Critic value head / queries / norm | — | always trained | — |

Setting any `_from_layer` field to `null` freezes the entire component. The dataclass defaults (`None`) freeze everything by default; the practical defaults shown above come from `config-hiserl.json`.

For online training the freezing tends to be more aggressive than offline (roughly 25% of parameters trained vs. ~80% offline), purely because online imposes the additional cost of the actor process and the gRPC traffic.

---

## Reward structure

| Signal | Value | Source |
|--------|-------|--------|
| Success | `+1.0` | Reward classifier or manual key (`1`) |
| Failure / timeout | `-16.0` | `terminal_failure_reward` (config) |
| Step reward | `0.0` | Default for non-terminal steps |
| Normalization | `÷ 5.0` | `reward_normalization_constant` (config) |

Human interventions are flagged with `is_intervention=1.0` in the transition's complementary info, and their advantage is overridden to 1.0 during training, ensuring the policy treats corrective human actions as high-quality examples regardless of what the critic thinks at that step.

The asymmetry between success (+1) and failure (-16) is intentional: failure should be much more painful than success is rewarding, which biases the critic toward conservative behavior and helps prevent the policy from drifting into states it has never recovered from.

---

## Key hyperparameters

These are the values used in the default `config-hiserl.json`. The dataclass defaults in `PI05RLConfig` differ in places (e.g., `advantage_scaling=1.0`, `gradient_accumulation_steps=1`, `grad_clip_norm=40.0`, `actor_lr=critic_lr=3e-4`) — the config below is what we run.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `discount` | `0.97` | Temporal discount $\gamma$ |
| `critic_target_update_weight` | `0.005` | Polyak $\tau$ |
| `critic_lr` | `5e-5` | Critic learning rate |
| `actor_lr` | `5e-5` | Actor learning rate |
| `advantage_scaling` | `0.20` | Scales advantage before $\tanh$ |
| `gradient_accumulation_steps` | `16` | Accumulate before stepping |
| `grad_clip_norm` | `2.0` | Max gradient norm |
| `utd_ratio` | `2` | Critic updates per actor update |
| `num_inference_steps` | `5` | Flow matching denoising steps |
| `critic_llm_depth` | `6` | Critic transformer layers |
| `reward_normalization_constant` | `5.0` | Reward scaling $c_r$ |
| `terminal_failure_reward` | `-16.0` | Penalty on failure |
| `inference_advantage` | `1.0` | Fixed advantage label fed at deployment |
| `policy_update_freq` | `1` | Actor steps per learner iteration |
| `knowledge_insulation` | `true` | Advantage affects action path only |

---

## Code map

| File | Role |
|------|------|
| `rl/rl_pi05.py` | `PI05RLConfig`, `Pi05TransformerCritic`, `PI05RLPolicy` — the core RL policy and critic; also handles checkpoint detection (base $\pi_{0.5}$ vs RL checkpoint) and target-network updates. |
| `rl/pi05_train_utils.py` | `pi05_update_step`, `_update_critic`, `_update_actor`, `_compute_advantage_with_interventions` — shared training logic used by both offline and online learners. |
| `rl/utils.py` | `preprocess_batch_for_pi05` — preprocessing entry point for the offline path; also where the placeholder `inference_advantage` is filled in (later overwritten for the actor update). |
| `scripts/offline_learner_pi05.py` | Offline training without validation probes. |
| `scripts/offline_learner_val_pi05.py` | Offline training with validation probes (attention maps, PCA/UMAP). |
| `rl/learner_pi05.py` | Online learner (gRPC server, online + offline buffer mixing, weight pushes). |
| `rl/actor_pi05_async.py` | Online actor (gRPC client + RTC robot control at 30 Hz + leader-arm interventions). |
| `rl/inference_pi05_async.py` | Inference-only equivalent of the actor (no learner connection, saves an episode buffer + critic-overlay videos). |
| `policies/pi05_full/processor_pi05.py` | Advantage tokenization, prompt construction (actor prompt + critic prompt), state discretization. |
