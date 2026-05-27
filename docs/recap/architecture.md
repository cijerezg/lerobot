# RECAP Architecture and Math Details

## Introduction to RECAP Math

RECAP introduces an advantage-conditioned policy where the return is defined as the expected sum of discounted rewards. The sequence of actions is modeled through an implicit policy $\pi_\theta(a|s)$. The value function $V^\pi(s)$ is given by:

$$ V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \middle| s_t = s \right] $$

We train the critic to minimize the Temporal Difference (TD) error using a target network:

$$ L_{critic}(\phi) = \mathbb{E} \left[ \left( r_t + \gamma V_{\phi^{-}}(s_{t+1}) - V_\phi(s_t) \right)^2 \right] $$

Where $V_{\phi^{-}}$ is the target network. The advantage $A(s, a)$ is then used to threshold the RECAP conditional objective, enabling the policy to leverage suboptimal demonstrations by conditioning on high advantage:

$$ A(s,a) \approx Q_{critic}(s,a) - V_\phi(s) $$

## Action Encodings: Absolute vs Anchor vs Delta

When using **Anchor actions**, the true continuous action $\hat{a}_t$ is offset from the start of the chunk $s_0$:

$$ \hat{a}_t = a_t - s_0 $$

For **Delta actions**, the offset is relative to the immediate previous timestep:

$$ \hat{a}_t = a_t - a_{t-1} $$

This formulation allows for smooth blending strategies when the policy executes in a receding horizon manner.

![Architecture Visualization](../../media/readme/viz_tool.png)
*Figure 1: Evolution of attention maps across training steps during the RECAP process.*

## Code Architecture Breakdown

The RL stack uses **MolmoAct2** as the default VLA backbone (Qwen2.5-7B text model + SigLIP ViT + a flow-matching ActionExpert). The companion deep reference for the network itself lives in [`policies/molmoact2/ARCHITECTURE.md`](../../src/lerobot/policies/molmoact2/ARCHITECTURE.md) — that file documents every sub-module, dtype, and shape. This document focuses on what gets *added* on top of the base policy for RL.

### Policy: `MolmoAct2RLPolicy` (`rl/molmoact2/rl_molmoact2.py`)

Extends `MolmoAct2Policy` with a lazily-instantiated **distributional value critic**:

```
MolmoAct2RLPolicy
├── (inherited) backbone: MolmoAct2ForConditionalGeneration
│   ├── transformer:     Qwen2.5-7B text decoder
│   ├── vision_backbone: SigLIP ViT + MQA pool + SwiGLU adapter
│   └── action_expert:   flow-matching DiT
├── critic:        MolmoAct2Critic              [trained]
└── critic_target: MolmoAct2Critic              [frozen, Polyak-averaged]
```

`MolmoAct2Critic` deep-copies the actor's vision backbone and the first `critic_llm_depth` (12) text transformer blocks at construction, then appends a stack of `num_value_bins` (201) learned query tokens. The transformer runs fully bidirectionally over `[text + image | value queries]`; a shared `Linear(D, 1)` head emits one logit per query, producing a categorical distribution over the value support $[-2.0, 0.0]$. The expected value is read out as $\mathbb{E}[V] = \sum_i p_i c_i$.

### Trainer: `MolmoAct2Trainer` (`rl/molmoact2/rl_molmoact2_trainer.py`)

Owns the core update loop. The critic is trained with **HL-Gauss cross-entropy** against the TD target (and against a one-hot bin for exact terminal values); the actor is trained with the standard flow-matching loss plus a CE term on the action-token decoder, conditioned on a binarized advantage label injected into the prompt.

The advantage label tokenization is built in `policies/molmoact2/processor_molmoact2.py`. The prompt template is:

```
The task is to {task}. The setup is {setup_text}. The current state of the robot
is {state}. The expected control mode is {control_text}. The advantage is
{negative|positive}. Given these, what action should the robot take to complete
the task?
```

The label is binarized per batch: `threshold = quantile(tanh(A/α)[~override_mask], 1 - top_k_fraction)`, then `positive` if $\tanh(A/\alpha) \geq \text{threshold}$ else `negative` ($\alpha$ = `advantage_scaling`, default 0.2; `advantage_top_k_fraction` default 0.3). Samples flagged `is_golden` or `is_intervention` are forced to $A = 1.0$ and excluded from the threshold pool, so they always cross.

For everything else — critic loss derivation, advantage overrides for golden/intervention transitions, Polyak update schedule, parameter freezing strategy, and hyperparameters — see [`recap_implementation.md`](recap_implementation.md).

## RTC & Asynchronous Infrastructure

Inference runs concurrently with the environment loop at 30Hz. Below is the simplified threading model:
* **Thread 1 (Env)**: Step environment $\rightarrow$ `env.step(action)`
* **Thread 2 (GPU)**: Forward pass $\rightarrow$ `model.forward(obs)`

> [!TIP]
> The buffer uses `SharedState` locks to safely pass state between the threads.

```mermaid
graph TD;
    subgraph Client [Main Process (30Hz)]
        A[Env Step] --> B{SharedQueue};
    end

    subgraph Background [Inference Thread]
        B --> C[Retrieve Observation];
        C --> D[VLA Forward Pass];
        D --> E[Produce Action Chunk];
        E --> B;
    end
```

The RTC runtime is model-agnostic: `rl/rl_actor_async.py` wires up gRPC and signal handling, and `rl/rtc_actor_runtime.py` owns the `ActionQueue`, the prefix-attention schedule, and the threading. Per-policy observation preprocessing is isolated behind `Trainer.build_inference_batch()`, so swapping VLAs does not touch the runtime.

## Replay Buffer and Memmap Database

The buffer is dynamically sized and stored entirely using `numpy.memmap` files for efficient sampling during large-scale offline runs. Image storage dtype and resolution are declared in the policy config (`image_storage_dtype`, `image_storage_size`) and participate in the cache fingerprint — mismatched configs silently fall back to video decode, so keep these fields in sync with the data on disk.

## Pi0.5 variant

The Pi0.5 stack (`policies/pi05_full/`, `rl/pi05/rl_pi05.py`, `rl/pi05/rl_pi05_trainer.py`) is still supported as an alternative backbone. The differences worth knowing:

- **Backbone**: PaliGemma 3B + ~300M Gemma action expert, instead of MolmoAct2.
- **Critic**: 32 learned query tokens with an MLP value head; MSE on continuous targets rather than HL-Gauss CE. Configured via `policy.type = "pi05_rl"` and the `PI05RLConfig` fields documented in [`recap_implementation.md`](recap_implementation.md#pi05-variant).
- **Prompt**: `Task: {task}, State: {state}, Advantage: {label};\n` — shorter, with a separate critic prompt that omits the advantage label to avoid the circular dependency.
- **Knowledge insulation**: optional flag (`knowledge_insulation`) that routes the advantage signal only through the action expert path, leaving VLM perception unconditioned. The MolmoAct2 stack does not currently expose this knob.

The training loop topology (critic update → actor update → target Polyak), the buffer, the RTC runtime, and the validation probes are the same across both backbones.
