# RECAP Implementation

## What is RECAP and why does it matter

[RECAP](https://arxiv.org/pdf/2511.14759) is the RL algorithm developed by [Physical Intelligence](https://www.pi.website/) that was used to train the $\pi_{0.6}$ model.

RECAP proposes an **advantage-conditioned VLA**, where the advantage comes from a critic that is trained alongside the policy (the VLA itself). The advantage is computed from the critic, binarized, and passed to the policy as an extended part of the prompt — so the same VLA decoder produces actions conditioned on whether the current state is "good" or "bad" relative to expectation.

This is useful for two reasons:

1. **The policy can learn from its own experience.** There is a grounding signal about the *goodness* of actions (the advantage from the critic) that is completely driven by reward. Without this signal, when the policy makes a mistake during online rollouts, it would just imitate that mistake. With advantage conditioning, the model is told "this is what a low-advantage state looks like," which lets it differentiate trajectories during training and lean toward the high-advantage mode at inference (when we always feed it `Advantage: positive`).
2. **Suboptimal demonstrations become useful.** The critic learns to distill which parts of a demonstration are good and which are bad — i.e., which actions have high advantage versus low. This is a big deal in the small-data regime where curated, perfectly clean demos are expensive: RECAP lets you keep the messy ones and still extract signal from them.

This document is a deep dive into how that idea is implemented in this repository for the **MolmoAct2** backbone (the default), the deliberate divergences from the paper, and the code paths that wire everything together. The Pi0.5 variant is documented at the bottom in [Pi0.5 variant](#pi05-variant).

---

## How this implementation differs from the paper

RECAP was designed for a much larger model and a much larger compute budget than what is targeted here. The version in this repo keeps the core idea — advantage-conditioned VLA with a critic trained jointly — and trades a few of the paper's design choices for things that are simpler, work in the small-data regime, and fit on a single workstation.

| Aspect | Original RECAP | This implementation | Reason for the divergence |
|--------|---------------|---------------------|---------------------------|
| **Backbone** | Gemma 3 4B | MolmoAct2 (Qwen2.5-7B text + SigLIP ViT + flow-matching action expert) | MolmoAct2 weights for SO100/SO101 are open and fit on a single workstation; the action expert is purpose-built for continuous control. |
| **Return estimator** | On-policy Monte-Carlo returns over full trajectories | Off-policy 1-step TD via the Bellman equation with a target network | Off-policy TD lets us train from a fixed offline buffer and from a mixed online/offline buffer without waiting for full rollouts |
| **Critic loss** | Cross-entropy over discretized returns (HL-Gauss style) | **HL-Gauss CE retained** — soft target over 201 bins for non-terminal, one-hot for exact terminal values | HL-Gauss is a better fit than MSE at this scale; we re-tuned the bin count and support after observing the target distribution. |
| **Advantage labels** | Multiple discretized return bins | Two bins: `negative` / `positive`, split at $\tanh(A/\alpha) = 0$ | Two bins are enough to teach the policy a directional signal at the data scales we run, and the prompt stays short |
| **Advantage dropping** | Stochastic dropping of advantage labels (classifier-free guidance style) | Always pass the advantage label | We don't currently rely on guidance at inference; revisit if we want CFG-style conditional sampling |

The two implementations share the loop topology and the binarized-prompt interface; everything below describes the MolmoAct2 wiring concretely.

---

## Architecture overview

```
MolmoAct2RLPolicy
├── (inherited) backbone: MolmoAct2ForConditionalGeneration
│   ├── transformer:     Qwen2.5-7B text decoder           [selectively frozen]
│   ├── vision_backbone: SigLIP ViT + MQA pool + SwiGLU    [selectively frozen]
│   └── action_expert:   flow-matching DiT                 [always trained]
│
├── critic: MolmoAct2Critic                                 [trained]
│   ├── vision_backbone:    deep-copied from actor          [selectively frozen]
│   ├── transformer_blocks: first N text blocks (deep-copy) [selectively frozen]
│   ├── rotary_emb:         deep-copied from actor
│   ├── ln_f:               deep-copied from actor          [selectively frozen]
│   ├── value_queries:      [1, num_value_bins, D]          [always trained]
│   └── bin_logit_head:     Linear(D, 1)                    [always trained]
│
└── critic_target: MolmoAct2Critic                          [no gradients, Polyak-averaged]
```

The critic is a **separate transformer**, not a head bolted onto the actor. It shares the input encoder structure (its own vision backbone consumes the same image preprocessing, and the text embeddings are read from the actor's `wte` with `detach()`) but has its own decoder layers and its own learned query tokens. We initialize it by deep-copying the actor's `vision_backbone` and the first $N$ text-transformer blocks (`critic_llm_depth`, default 12), which gives it a sensible starting point for what manipulation features look like.

---

## Critic architecture

The critic (`MolmoAct2Critic`) is a transformer-based state value function $V(s)$ that estimates the expected discounted return from a given state. It is **distributional**: it outputs a categorical distribution over a fixed grid of value bins, and the scalar value is read out as the expectation.

### Components

- **Inputs**: embedded `[text + image]` tokens of dimension $D = 2560$ (Qwen2.5-7B hidden size). Text comes from the actor's `wte` (detached); image features come from the critic's own `vision_backbone` and are spliced into the sequence at image-patch positions.
- **Value query tokens**: `num_value_bins` learned tokens (default **201**) of shape $\mathbf{Q} \in \mathbb{R}^{1 \times 201 \times D}$, expanded to batch. Appended *after* the text+image sequence; the transformer is forced to summarize all value-relevant context into those query slots via bidirectional self-attention.
- **Transformer stack**: 12 text-transformer blocks (configurable via `critic_llm_depth`), deep-copied from the actor's text model at construction, run with **bidirectional attention** (no causal mask) over the full sequence.
- **Per-query logit head**: a shared `Linear(D, 1)` projects each query token's final hidden state to a single logit. The 201 logits form a categorical distribution over the value support.

The value support is the linspace `[value_support_min, value_support_max] = [-2.0, 0.0]` divided into 201 bin centers. Given softmax probabilities $\mathbf{p}$ over these bins, the scalar value is

$$V(s) = \sum_i p_i \cdot c_i$$

where $c_i$ is the $i$-th bin center.

### Target distribution

For training, the scalar TD target $Q_{\text{target}}$ is converted into a target distribution over the same 201 bins in one of two ways:

- **Non-terminal targets** use **HL-Gauss smoothing**: each bin gets the probability mass from a Gaussian of width $\sigma = \texttt{hl\_gauss\_sigma\_ratio} \cdot \text{bin\_width}$ (default ratio 5.0), evaluated against bin boundaries. Smooth enough to give gradient signal even when the target lands between bins.
- **Terminal targets** (`done=True`) use a **one-hot** distribution at the nearest bin. Terminal values are exact (e.g. the success reward), so smearing them is just noise.

The target $Q_{\text{target}}$ itself is clamped into the support `[v_min, v_max]` before bin assignment — otherwise an out-of-range TD target would just train the head to put mass on the closest endpoint, wasting capacity.

### Target network

A separate `critic_target` is maintained via Polyak averaging:

$$\boldsymbol{\theta}_{\text{target}} \leftarrow (1 - \tau) \cdot \boldsymbol{\theta}_{\text{target}} + \tau \cdot \boldsymbol{\theta}_{\text{critic}}, \quad \tau = 0.005$$

This update fires every `critic_target_update_every` critic optimizer steps (default 4) rather than every step — a small efficiency win that doesn't change semantics. Only parameters with `requires_grad=True` are averaged; frozen parameters are *aliased* between the critic and critic_target at construction time (sharing the underlying storage) to halve their memory footprint.

### The critic prompt

The critic is run with `preprocessor({**observations, "task": ...})` — i.e. no advantage clause is built into the prompt. The actor's prompt, in contrast, gets the advantage clause appended (see next section). If the critic saw the advantage label, we would have a circular dependency: the advantage is a function of $V(s)$, and $V(s)$ would itself depend on the binarized advantage. Cutting that loop also makes the value estimate more transferable across rollouts where we feed different advantage labels to the actor.

---

## Advantage conditioning — the bridge between critic and actor

The advantage tells the actor whether the current state-action trajectory is doing better or worse than the critic expected. This is the only place the critic affects the actor: there is no policy gradient. The actor still trains with a flow-matching loss; the critic just decides which conditioning string is glued to the prompt.

### Computation

At each actor update step:

1. **Critic forward pass** (no gradients on the actor side): compute $V(s)$ with the live critic and $V_{\text{target}}(s')$ with the target.
2. **Bellman target**:

   $$Q_{\text{target}} = \frac{r}{c_r} + \gamma \cdot V_{\text{target}}(s') \cdot (1 - \text{done})$$

   where $c_r$ is `reward_normalization_constant` (default 1.0 in the live MolmoAct2 config) and $\gamma$ is `discount` (default 0.97).
3. **Raw advantage**: $A = Q_{\text{target}} - V(s)$.

The MolmoAct2 trainer overrides $A = 1.0$ for samples flagged `is_golden` (offline expert data) or `is_intervention` (online teleop corrections). The critic forward still runs on every sample so its pre-override estimate is observable via `advantage_critic_vs_golden_gap` / `advantage_critic_vs_intervention_gap` (mean of `pre_override_adv - 1.0` over the masked subset) — these shrink toward zero as the critic learns that the masked samples are indeed high-value.

### Tokenization

The raw advantage is squashed and bucketed by a per-batch dynamic threshold:

1. **Scale + squash**: $A_{\text{squashed}} = \tanh(A / \alpha)$, where $\alpha$ is `advantage_scaling` (0.2). Small $\alpha$ means even modest advantages saturate $\tanh$.
2. **Threshold (top-K)**: `threshold = quantile(squashed_adv[~override_mask], 1 - advantage_top_k_fraction)`, computed strictly over the non-override (online, critic-derived) subset of the batch. With the default `advantage_top_k_fraction = 0.3` the top 30% of non-override samples cross the threshold. If the batch is all-override (e.g., 100% golden offline), the pool is empty and `threshold = -inf` so every sample passes — equivalent to the override semantics.
3. **Bin**:
   - $A_{\text{squashed}} \geq \text{threshold} \Rightarrow$ `"positive"`
   - $A_{\text{squashed}} < \text{threshold} \Rightarrow$ `"negative"`
   Override samples have $A_{\text{squashed}} \approx 0.99991$, strictly greater than any threshold derived from the non-override pool, so they're guaranteed positive.
4. **Inject**: the prompt is built by `processor_molmoact2._build_robot_text(...)` and gets an `"The advantage is {label}."` clause inserted alongside task/setup/state/control.

The per-batch quantile replaces the prior fixed 0.35 split — it tracks RECAP's recipe (top-fraction positive) and adapts to the critic's current scale, rather than relying on a static cut. The threshold is logged as `advantage_top_k_threshold`.

### What the actor sees at inference

At deployment time the critic is not consulted: the actor's prompt is built with a fixed `inference_advantage` (default 1.0, mapped to `positive`). This is the point of training with binarized labels — at inference you pick the mode you want and the model produces actions consistent with it. `build_inference_batch` in `rl_molmoact2_trainer.py` threads the value into `complementary_data["advantage"]` along with `advantage_threshold = 0.0` (single-sample inference can't form a top-K quantile, so sign-based binning is used — positive `inference_advantage` → positive label, negative → negative).

---

## Loss functions

### Actor loss

The actor loss combines the standard MolmoAct2 training losses; advantage acts only as a prompt label and does not appear as a multiplicative weight on the loss.

$$\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{discrete\_ce}} \;(+\; \mathcal{L}_{\text{discrete\_z}})$$

- **Flow matching loss**: MSE on the velocity field predicted by the action expert. Given a noisy action $\mathbf{x}_t = t \cdot \boldsymbol{\epsilon} + (1-t) \cdot \mathbf{a}$ at $t \sim U(0,1)$, supervise $\mathbf{v}_{\theta}(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon} - \mathbf{a}$.
- **Discrete CE loss**: cross-entropy on the discrete action tokens decoded by `lm_head` between `<action_start>` and `<action_end>`.
- **Discrete-z loss** (optional, present only when the action expert exposes a separate latent CE head): an auxiliary regularizer; omitted when zero.

The advantage label affects the actor strictly through the prompt — the loss is unweighted by it.

### Critic loss

$$\mathcal{L}_{\text{critic}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{N_{\text{bins}}} t_{ij} \log \mathrm{softmax}(z_{ij})$$

where $z_i$ are the per-sample logits over the 201 bins and $t_i$ is either the HL-Gauss soft target (non-terminal) or the one-hot bin (terminal). The TD target is computed from the frozen `critic_target`, clamped into the support, then mapped to $t_i$.

Critic gradients are clipped to `optimizer_grad_clip_norm` (default 1.0). The same hyperparameter governs actor clipping; consider splitting if one head dominates the norm budget.

---

## Training loop

### Each optimization step

```
1. CRITIC UPDATE                                   [skipped if skip_critic=True]
   - Sample online batch (+ offline batch if both present); concat.
   - Preprocess WITHOUT advantage clause (critic prompt).
   - V_target(s'): forward critic_target on next_state, no grad.
   - TD target = r + γ * V_target(s') * (1 - done), then clamp to [v_min, v_max].
   - Build target distribution: HL-Gauss (non-terminal) or one-hot (terminal).
   - V(s): forward critic on current state, with grad.
   - Loss = CE(softmax(logits), soft_target).
   - Backward, clip critic grads, step "critic" optimizer.
   - Repeat for gradient_accumulation_steps.

2. ACTOR UPDATE
   - Sample fresh batch (online + offline mix).
   - If skip_critic=False: run compute_advantage to get raw A; thread into batch
     as complementary_data["advantage"] so the preprocessor binarizes it into
     a "negative"/"positive" clause in the prompt.
   - Build forward batch via preprocessor.
   - policy.forward(batch) → flow loss + discrete CE (+ discrete z).
   - Backward, clip actor grads, step "policy" optimizer.
   - Repeat for gradient_accumulation_steps.

3. TARGET NETWORK UPDATE                           [skipped if skip_critic=True]
   - Every `critic_target_update_every` calls:
       θ_target ← (1 - τ) * θ_target + τ * θ_critic,  τ = 0.005
   - Frozen-actor layers are aliased between critic and critic_target, so the
     averaging loop only touches the trainable subset.
```

The critic and actor each pull their own batch sample so the critic's CE doesn't see exactly the same minibatch the actor is fitting — this decoupling keeps the TD updates from being too correlated with the policy improvement step.

### skip_critic

`skip_critic: true` is the **default** in the shipped `config_rl.yaml` — the trainer then runs as actor-only behaviour-cloning with the flow + CE losses, no critic forward passes, no advantage clause in the prompt. Flip to `false` to enable the full RECAP loop. (Note: the flag only freezes critic *training*; pretrained critic forward passes used for logging stay unguarded, so loading a checkpoint with a trained critic still produces value overlays in episode logs.)

### Online vs offline

**Offline training** (`rl_offline.py` → `MolmoAct2Trainer`):
- Trains on a fixed dataset with no actor in the loop.
- All advantage values are still computed live by the critic during the actor update.
- Runs validation probes at `val_freq` intervals.

**Online training** (`rl_learner.py` + `rl_actor_async.py`):
- Learner receives transitions from the actor via gRPC.
- Actor runs asynchronous inference on the robot at 30 Hz using RTC, with leader-arm interventions toggled by key `5`.
- Online and offline buffers are mixed roughly 50/50 per batch — the offline data acts as a stabilizing prior, preventing the policy from forgetting what it learned offline while it incorporates new on-robot experience.
- Critic advantages are computed live every actor update (when `skip_critic=False`).
- Policy weights (trainable, non-critic params only) are pushed to the actor every `policy.actor_learner_config.policy_parameters_push_frequency` seconds (wall-clock; default 120).

---

## Parameter freezing

The full MolmoAct2 + critic stack is too large to train end-to-end on a single workstation, so only a subset of parameters is unfrozen. The strategy is configured via `policy.trainable_params` and resolved by `MolmoAct2Trainer.freeze_model`.

| Component | Config field | Default in `config_rl.yaml` | Total layers |
|-----------|--------------|------------------------------|--------------|
| Vision tower (SigLIP) | `vision_from_layer` | `16` (train layers 16–24) | 25 |
| Image pooling + projector | (linked to `vision_from_layer`) | trains when vision is on | — |
| Text model (Qwen2.5) | `language_from_layer` | `0` (train all) | 36 |
| Token embedding (`wte`) | `freeze_embedding` | `true` (frozen) | — |
| Action expert | — | always trained | — |
| Critic vision tower | `critic_vision_from_layer` | `4` (train layers 4–24) | 25 |
| Critic text blocks | `critic_language_from_layer` | `18` (train block 18 and beyond, if present) | 12 |
| Critic queries + logit head | — | always trained | — |

Setting any `_from_layer` field to `null` freezes the entire component. The shipped defaults train the upper half of the vision tower and all of the language model on the actor side, and adapt the upper critic vision layers + a slice of the critic text blocks alongside the always-trained query + logit heads. Push these layer indices lower if the critic looks under-fit; bump them higher (or set to `null`) for actor-only behaviour-cloning runs where the critic is irrelevant.

---

## Reward structure

| Signal | Value | Source |
|--------|-------|--------|
| Success | `+1.0` | Reward classifier or manual key (`1`) |
| Failure / timeout | `-10.0` | `terminal_failure_reward` (config) |
| Step reward | `0.0` | Default for non-terminal steps |
| Normalization | `÷ 1.0` (no-op by default) | `reward_normalization_constant` (config) |

The MolmoAct2 defaults are deliberately tighter than the Pi0.5 ones — value support is $[-2.0, 0.0]$, so a divisor of 1.0 keeps the success/failure pair inside the support after one discount step. Raise the normalization constant if you change the support range or the failure penalty so that $\frac{r}{c_r} + \gamma V$ stays in range.

The asymmetry between success (+1) and failure (-10) is intentional: failure should be more painful than success is rewarding, which biases the critic toward conservative behavior and helps prevent the policy from drifting into states it has never recovered from.

---

## Key hyperparameters

These are the values used in the shipped `config_rl.yaml`. The dataclass defaults in `MolmoAct2RLConfig` mostly match; differences are noted inline.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `discount` | `0.97` | Temporal discount $\gamma$ |
| `critic_target_update_weight` | `0.005` | Polyak $\tau$ |
| `critic_target_update_every` | `4` | Target update every N critic steps |
| `critic_lr` | `1e-4` | Critic learning rate |
| `advantage_scaling` | `0.2` | Scales advantage before $\tanh$ |
| `advantage_top_k_fraction` | `0.3` | Fraction of non-override samples labeled positive per batch (top-K cut) |
| `num_value_bins` | `201` | Categorical bins for V(s) |
| `value_support_min` / `max` | `-2.0` / `0.0` | Bin grid endpoints |
| `hl_gauss_sigma_ratio` | `5.0` | Soft-target width in bin-widths |
| `critic_llm_depth` | `12` | Critic transformer layers |
| `utd_ratio` | `1` | Critic updates per actor update |
| `reward_normalization_constant` | `1.0` | Reward scaling $c_r$ |
| `terminal_failure_reward` | `-10.0` | Penalty on failure |
| `inference_advantage` | `1.0` | Fixed advantage value fed at deployment; binned to `positive` by default |
| `policy_update_freq` | `1` | Actor steps per learner iteration |
| `skip_critic` | `true` | Default: actor-only run |

---

## Code map

| File | Role |
|------|------|
| `rl/molmoact2/rl_molmoact2.py` | `MolmoAct2RLConfig`, `MolmoAct2Critic`, `MolmoAct2RLPolicy` — config, distributional critic, and the policy wrapper that lazily instantiates critic + critic_target. |
| `rl/molmoact2/rl_molmoact2_trainer.py` | `MolmoAct2Trainer` — actor + critic update steps, freeze schedule, Polyak target updates, inference batch building, weight pushing. |
| `policies/molmoact2/processor_molmoact2.py` | Advantage tokenization, prompt construction (`_build_robot_text`), state discretization. |
| `policies/molmoact2/configuration_molmoact2.py`, `modeling_molmoact2.py` | The base MolmoAct2 policy (vision + text + action expert). See `policies/molmoact2/ARCHITECTURE.md` for the deep model reference. |
| `rl/rl_trainer.py` | Trainer base class — orchestration of the per-step `update_critic` / `update_actor` / `update_target_networks` calls. |
| `scripts/rl_offline.py` | Offline training entry point with validation probes. |
| `rl/rl_learner.py` | Online learner (gRPC server, online + offline buffer mixing, weight pushes). |
| `rl/rl_actor_async.py` | Online actor (gRPC client; delegates to `rtc_actor_runtime.act_with_policy_rtc`). |
| `rl/rtc_actor_runtime.py` | RTC runtime: action queue, prefix-attention schedule, 30 Hz robot control loop, leader-arm interventions. Model-agnostic; per-policy details flow through `Trainer.build_inference_batch`. |

---

## Pi0.5 variant

The Pi0.5 stack (`policies/pi05_full/`, `rl/pi05/rl_pi05.py`, `rl/pi05/rl_pi05_trainer.py`) implements the same RECAP idea on a smaller backbone. The differences worth knowing, all relative to the MolmoAct2 sections above:

| Aspect | Pi0.5 | MolmoAct2 |
|---|---|---|
| **Backbone** | PaliGemma 3B + ~300M Gemma action expert | Qwen2.5-7B + SigLIP ViT + MolmoAct2 action expert |
| **Critic queries** | 32 learned tokens with an MLP value head (`Linear(16384, 1024) → SwiGLU → Linear(512, 1)`) | 201 query tokens with a shared `Linear(D, 1)` logit head |
| **Critic loss** | MSE on continuous value targets | HL-Gauss CE over 201 bins (one-hot at terminals) |
| **Advantage bin split** | $\tanh(A/\alpha) \geq 0.35$ → positive (asymmetric fixed cut) | Per-batch top-K quantile over non-override samples (default 30%) |
| **Advantage overrides** | Force $A = 1.0$ for `is_golden` and `is_intervention` transitions | Same — force $A = 1.0$; overrides are also excluded from the top-K quantile pool |
| **Critic prompt** | `Task: {task}, State: {state};\n` — a separate, shorter prompt | Same template as the actor minus the advantage clause |
| **Knowledge insulation** | `knowledge_insulation` flag routes advantage only through the action expert | Not exposed |
| **Reward normalization** | `c_r = 5.0` | `c_r = 1.0` |
| **Terminal failure reward** | `-16.0` | `-10.0` |
| **Subtask CE loss** | Adds an explicit subtask-token CE on top of flow + action CE | Folded into the standard discrete CE |

Everything else — the loop topology (critic update → actor update → target Polyak), the online/offline mixing, the buffer mechanics, the RTC runtime, and the validation probes — is shared across both backbones.
