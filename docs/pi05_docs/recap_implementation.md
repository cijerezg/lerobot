# RECAP Implementation

This document covers the architecture and implementation of our RECAP-like RL algorithm. The framework adapts [RECAP](https://arxiv.org/pdf/2511.14759) (developed by Physical Intelligence for $\pi_{0.6}$) with several deliberate differences.

## Differences from the Original RECAP

| Aspect | Original RECAP | This Implementation |
|--------|---------------|---------------------|
| **Return estimator** | On-policy Monte Carlo returns | Off-policy TD learning via Bellman equation |
| **Critic loss** | Cross-entropy over discretized returns | MSE over continuous advantage values |
| **Advantage dropping** | Stochastic dropping of advantage labels | Always pass advantage labels to the policy |
| **Backbone** | Gemma 3 4B | PaliGemma 3B + action expert (~300M params) |
| **Advantage bins** | Multiple discretized return levels | Two bins: `negative` / `positive` |

---

## Architecture Overview

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
│   ├── value_queries: [1, 32, 2048]             [learned]
│   ├── query_proj: Linear(2048, 512)
│   └── value_head: Linear(16384, 1) via SwiGLU
│
└── critic_target: Pi05TransformerCritic          [no gradients, Polyak-averaged]
```

---

## Critic Architecture

The critic (`Pi05TransformerCritic`) is a transformer-based state value function $V(s)$ that estimates the expected return from a given state.

### Components

- **Inputs**: Vision features $(B, N_{\text{patches}}, 2048)$ + text embeddings $(B, L_{\text{text}}, 2048)$
- **Value query tokens**: 32 learned tokens $\mathbf{Q} \in \mathbb{R}^{1 \times 32 \times 2048}$, expanded to batch dimension. These tokens extract value-relevant information from the input sequence via self-attention.
- **Transformer stack**: 6 `GemmaDecoderLayer`s (configurable via `critic_llm_depth`) with RoPE positional embeddings. Initialized by deep-copying layers from the pretrained actor.
- **Value head**: Projects the 32 query token outputs to a scalar:

$$V(s) = \text{MLP}\left(\text{flatten}\left(W_{\text{proj}} \cdot \mathbf{Q}_{\text{out}}\right)\right)$$

where:
- $\mathbf{Q}_{\text{out}} \in \mathbb{R}^{B \times 32 \times 2048}$ are the query token outputs after the transformer
- $W_{\text{proj}} \in \mathbb{R}^{2048 \times 512}$ projects each token to 512 dims
- `flatten` reshapes $(B, 32, 512) \to (B, 16384)$
- MLP: `Linear(16384, 1024) -> SwiGLU -> Linear(512, 1)`

The SwiGLU activation is: $\text{SwiGLU}(\mathbf{x}) = \text{SiLU}(\mathbf{x}_{\text{gate}}) \odot \mathbf{x}_{\text{value}}$ where the linear layer outputs are split into gate and value halves.

### Target Network

A separate `critic_target` is maintained via Polyak averaging:

$$\boldsymbol{\theta}_{\text{target}} \leftarrow (1 - \tau) \cdot \boldsymbol{\theta}_{\text{target}} + \tau \cdot \boldsymbol{\theta}_{\text{critic}}, \quad \tau = 0.005$$

Updated after every optimization step. Only parameters with `requires_grad=True` are averaged.

---

## Advantage Conditioning

The advantage signal bridges the critic and actor. It tells the actor whether the current state-action trajectory is better or worse than expected.

### Computation

At each actor update step:

1. **Critic forward pass** (no gradients): compute $V(s)$ and $V(s')$
2. **Bellman target**: $Q_{\text{target}} = r + \gamma \cdot V_{\text{target}}(s') \cdot (1 - \text{done})$, where $\gamma = 0.97$
3. **Raw advantage**: $A = Q_{\text{target}} - V(s)$
4. **Overrides**:
   - If the transition is from the **golden dataset** (demonstrations): $A = 1.0$
   - If the transition is a **human intervention**: $A = 1.0$

### Tokenization

The raw advantage is discretized and injected into the text prompt:

1. **Scale**: $A_{\text{scaled}} = A / \alpha$ where $\alpha$ = `advantage_scaling` (default: 0.20)
2. **Squash**: $A_{\text{squashed}} = \tanh(A_{\text{scaled}})$, mapping to $[-1, 1]$
3. **Bin**: Using boundaries $[-1.0, 0.35, 1.0]$:
   - $A_{\text{squashed}} < 0.35 \Rightarrow$ `"negative"`
   - $A_{\text{squashed}} \geq 0.35 \Rightarrow$ `"positive"`
4. **Inject**: The prompt becomes `"Task: {task}, State: {state}, Advantage: {label};\n"`

The critic receives a separate prompt **without** the advantage label, containing only `"Task: {task}, State: {state};\n"`. This prevents the critic from conditioning on its own output.

### Knowledge Insulation

When `knowledge_insulation` is enabled (default), the advantage label affects only the action decoding path, not the vision/language perception. This prevents the advantage signal from corrupting learned visual representations.

---

## Loss Functions

### Actor Loss

The actor loss combines three terms:

$$\mathcal{L}_{\text{actor}} = w_{\text{flow}} \cdot \mathcal{L}_{\text{flow}} + w_{\text{action\_ce}} \cdot \mathcal{L}_{\text{action\_ce}} + w_{\text{subtask\_ce}} \cdot \mathcal{L}_{\text{subtask\_ce}}$$

**Flow matching loss** (MSE on the velocity field):

$$\mathcal{L}_{\text{flow}} = \frac{1}{Td} \sum_{t,j} \left(u_{t,j} - v_{t,j}\right)^2$$

where $\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{a}$ is the ground-truth velocity (noise minus clean actions) and $\mathbf{v}$ is the model's predicted velocity via `action_out_proj`. Only the first $d=6$ dimensions (actual action dims) contribute.

**Action cross-entropy loss** (FAST token prediction):

$$\mathcal{L}_{\text{action\_ce}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p(y_i | \mathbf{x})$$

where $\mathcal{M}$ is the set of unmasked action token positions and $y_i$ are the ground-truth FAST token indices.

**Subtask cross-entropy loss** (subtask token prediction):

$$\mathcal{L}_{\text{subtask\_ce}} = -\frac{1}{|\mathcal{M}_s|} \sum_{i \in \mathcal{M}_s} \log p(y_i | \mathbf{x})$$

Same as action CE but over subtask token positions.

All weights default to 1.0 and are configurable.

### Critic Loss

$$\mathcal{L}_{\text{critic}} = \frac{1}{B} \sum_{i=1}^{B} \left(V(s_i) - Q_{\text{target},i}\right)^2$$

where:

$$Q_{\text{target}} = \frac{r}{c_r} + \gamma \cdot V_{\text{target}}(s') \cdot (1 - \text{done})$$

- $c_r$ = `reward_normalization_constant` (default: 5.0)
- $\gamma$ = `discount` (default: 0.97)
- $Q_{\text{target}}$ is clamped: $Q_{\text{target}} = \min(Q_{\text{target}}, 0.05)$

The clamp reflects the bounded reward structure of the manipulation task (max expected return is close to 0).

---

## Training Loop

### Each Optimization Step

```
1. CRITIC UPDATE
   - Sample batch (mix online + offline if both present)
   - Preprocess without advantage tokens (critic prompt)
   - Forward: V(s), V_target(s') -> target_q = r + gamma * V_target(s') * (1-done)
   - Loss = MSE(V(s), target_q)
   - Backward, clip gradients (max norm: 2.0), step optimizer
   - Repeat for gradient_accumulation_steps (default: 16)

2. ACTOR UPDATE (if step >= warmup AND step % policy_update_freq == 0)
   - Sample NEW batch
   - Compute advantage: A = target_q - V(s), with golden/intervention overrides
   - Tokenize: inject advantage label into prompt
   - Forward: predict velocity field, action tokens, subtask tokens
   - Loss = weighted sum of flow + action_ce + subtask_ce
   - Backward, clip gradients, step optimizer
   - Repeat for gradient_accumulation_steps

3. TARGET NETWORK UPDATE
   - Polyak average: theta_target <- 0.995 * theta_target + 0.005 * theta_critic
```

### Online vs Offline

**Offline training** (`offline_learner_val_pi05.py`):
- Trains on a fixed dataset with no actor interaction
- Uses `accelerate` for multi-GPU distributed training
- Fixed `inference_advantage` (default: 1.0) instead of critic-computed advantages during early training
- Runs validation probes at `val_freq` intervals
- Typical run: 10,000 steps

**Online training** (`learner_pi05.py` + `actor_pi05_async.py`):
- Learner receives transitions from actor via gRPC
- Actor runs asynchronous inference on the robot at 30Hz
- Online and offline buffers are mixed 50/50 per batch
- Critic computes live advantages
- Policy weights pushed to actor every `policy_parameters_push_frequency` steps (default: 180)

### Batch Mixing

When both online and offline buffers are available, each batch contains samples from both:

```
batch = concat(next(online_iterator), next(offline_iterator))
```

The offline data acts as a stabilizing prior, preventing the policy from forgetting previously learned behaviors while incorporating new online experience.

---

## Parameter Freezing

Due to VRAM constraints, only a subset of parameters are trained. The strategy is configured via `trainable_params`:

| Component | Config Field | Default | Total Layers |
|-----------|-------------|---------|-------------|
| Vision tower (SigLIP) | `vision_tower` | `5` (train layers 5-26) | 27 |
| Multi-modal projector | `multi_modal_projector` | `true` | 1 |
| Language model (Gemma) | `language_from_layer` | `0` (train all) | 18 |
| Action expert | — | always trained | ~300M params |
| Critic layers | `critic_language_from_layer` | `1` (train layers 1-5) | 6 |
| Critic value head/queries | — | always trained | — |

Setting a field to `null` freezes all layers of that component.

---

## Reward Structure

| Signal | Value | Source |
|--------|-------|--------|
| Success | `+1.0` | Reward classifier or manual key (`1`) |
| Failure / timeout | `-16.0` | `terminal_failure_reward` config |
| Step reward | `0.0` | Default for non-terminal steps |
| Normalization | $\div 5.0$ | `reward_normalization_constant` |

Human interventions are marked with `is_intervention=1.0` in the transition metadata, and their advantage is overridden to 1.0 during training, ensuring the policy learns to imitate corrective human actions.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discount` | `0.97` | Temporal discount $\gamma$ |
| `critic_target_update_weight` | `0.005` | Polyak $\tau$ |
| `critic_lr` | `5e-5` | Critic learning rate |
| `actor_lr` | `5e-5` | Actor learning rate |
| `advantage_scaling` | `0.20` | Scales advantage before tanh |
| `gradient_accumulation_steps` | `16` | Accumulate before stepping |
| `grad_clip_norm` | `2.0` | Max gradient norm |
| `utd_ratio` | `2` | Critic updates per actor update |
| `num_inference_steps` | `5` | Flow matching denoising steps |
| `critic_llm_depth` | `6` | Critic transformer layers |
| `reward_normalization_constant` | `5.0` | Reward scaling |
| `terminal_failure_reward` | `-16.0` | Penalty on failure |

---

## Code Map

| File | Role |
|------|------|
| `rl/rl_pi05.py` | `PI05RLConfig`, `Pi05TransformerCritic`, `PI05RLPolicy` — the core RL policy and critic |
| `rl/pi05_train_utils.py` | `pi05_update_step`, `_update_critic`, `_update_actor` — shared training logic |
| `scripts/offline_learner_val_pi05.py` | Offline training with validation probes |
| `scripts/offline_learner_pi05.py` | Offline training without validation |
| `rl/learner_pi05.py` | Online learner (gRPC server) |
| `rl/actor_pi05_async.py` | Online actor (gRPC client + robot control) |
| `policies/pi05_full/processor_pi05.py` | Advantage tokenization and prompt construction |
