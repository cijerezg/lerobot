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
### rl_pi05.py and pi05_train_utils.py

The core update loop computes the flow matching loss alongside the RECAP objective. The loss computation happens predominantly inside `pi05_train_utils.py`.

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

## Replay Buffer and Memmap Database

The buffer is dynamically sized and stored entirely using `numpy.memmap` files for efficient sampling during large-scale offline runs.
