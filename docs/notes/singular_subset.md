# Singular value subset training (design 2026-11-25)

> **Status:** built, on 2026-11-25; step-0 check done on the real model, no run yet. First run: `rebot-v2-tube-singular20-veltip-20261125`.

Fine-tuning where, in the selected blocks, only a random 20% of the singular values of each
weight matrix trains. The purpose is to limit capacity so that the model has to reuse the
features it already has. Baseline: `rebot-v2-tube-blocks4-6-33-35-veltip-20261125` (the same
blocks trained in full).

Code: `src/lerobot/rl/molmoact2/singular_subset.py`, switched on by
`policy.trainable_params.singular_fraction`.

## 1. Parametrisation

For one matrix $W_0 \in \mathbb{R}^{p \times q}$ with SVD $W_0 = U S V^\top$, draw a subset $T$
of the singular triplets, $|T| = k = \mathrm{round}(0.2 \min(p, q))$, uniformly at random
(fixed seed); $F$ is the rest.

$$W = U_T\, M\, V_T^\top + U_F\, D_F\, V_F^\top = W_0 + U_T (M - D_T) V_T^\top, \qquad M \in \mathbb{R}^{k \times k},\quad M_{\text{start}} = D_T .$$

Only $M$ trains. Consequences:

- $U_F, D_F, V_F$ never change, and $U_T \perp U_F$, $V_T \perp V_F$, so every triplet in $F$
  stays an exact singular triplet of $W$.
- The other singular values of $W$ are those of $M$; its singular vectors stay inside
  $\mathrm{span}(U_T)$ and $\mathrm{span}(V_T)$.
- Mass: $\sum_{i \in T} \sigma_i(W)^2 = \lVert M \rVert_F^2$, held at $m_T = \sum_{i \in T} s_i^2$.
- Clamp: no singular value of $M$ above $4 \max_{i \in T} s_i$.

## 2. How it trains

$M$ is stored as $\tilde M$ (float32) with

$$M = \sqrt{m_T}\, \frac{\tilde M}{\lVert \tilde M \rVert_F}, \qquad \lVert \tilde M \rVert_F = c = k \cdot \mathrm{rms}(W_0) .$$

The model keeps ordinary weights; nothing in the forward pass changes. Per step:

1. Backward gives $G = \partial L / \partial W$ (the weights keep `requires_grad`, but no
   optimizer owns them). The policy gradient clip (1.0) acts on these, as in the baseline.
2. $G_M = U_T^\top G V_T$, then the part along $\tilde M$ is removed (the loss does not depend
   on the length of $\tilde M$):
   $\;\partial L / \partial \tilde M = \frac{\sqrt{m_T}}{c} \big( G_M - \frac{\langle G_M, \tilde M \rangle}{c^2} \tilde M \big)$.
3. AdamW step on $\tilde M$ (the `policy` optimizer group: same lr, schedule, betas).
4. $\tilde M \leftarrow c\, \tilde M / \lVert \tilde M \rVert_F$. Weight decay only shrinks
   $\tilde M$ along itself, so this step undoes it.
5. Every 10 steps: if $\sigma_{\max}(M) > 4 \max s_T$, clip the singular values of $M$ and
   restore the mass.
6. $W \leftarrow W_0 + U_T (M - D_T) V_T^\top$, written to bfloat16 with stochastic rounding.

**Learning rate.** With $\lVert \tilde M \rVert_F = k \cdot \mathrm{rms}(W_0)$ an entry of
$\tilde M$ has the rms of an entry of $W_0$. An Adam step moves every entry by about the
learning rate, so one step is the same fraction of $\lVert M \rVert$ here as it is of
$\lVert W \rVert$ in the baseline. The baseline's `optimizer_lr` (5e-5) is used unchanged.

**Rounding.** The weights are bfloat16. After 20 steps most entries of $W$ have moved by less
than their bfloat16 spacing; rounding to nearest left 33% of the entries of one matrix
unchanged. Stochastic rounding (the same routine `StochasticRoundingAdamW` uses) is unbiased;
$\tilde M$ is the state, so rounding error does not add up over steps.

## 3. What trains

| Part | State |
| --- | --- |
| Language blocks 4-6, 33-35: q, k, v, attention output, the two halves of `ff_proj`, `ff_out` | 20% of singular values each |
| Expert blocks 4-6, 33-35: q, k, v, self-attention output, cross-attention q and output, `gate_proj`, `up_proj`, `down_proj` | 20% of singular values each |
| Tip adapter, state projector | in full (as before) |
| Norms, biases and `modulation` of those blocks; final norm, `lm_head`, expert final layer | frozen |
| Everything else | frozen |

Fused weights (`att_proj`, `ff_proj`, expert `qkv`) are split by rows into the matrices they
hold, each with its own SVD and draw. 96 matrices, 9,649,284 trained numbers (language
8,368,620, expert 1,280,664); the baseline trained about 670M in the same blocks.

Mass fraction of the draw per matrix (seed 0): language 0.189 to 0.224, median 0.202; expert
0.167 to 0.282, median 0.200.

## 4. Files and logs

- Checkpoints hold merged weights (`model.safetensors`, unchanged format). The indices, the
  chosen singular values and $M$ go to `training_state/singular_subset.pt`.
- Metrics: `singular/top_ratio` (largest $\sigma_{\max}(M) / \max s_T$ over matrices, updated
  every 10 steps), `singular/off_diagonal_lm`, `singular/off_diagonal_expert` (share of the
  mass of $M$ off its diagonal: 0 = values only re-weighted, towards 1 = directions mixed).
- Start-up computes 96 SVDs in float64 (about 2.5 min). The "Trainable" count in the log
  includes the full target weights, because they keep `requires_grad`.
- Test: `tests/rl/test_singular_subset.py`.

## 5. Step-0 check (2026-11-25, real model, no data)

- After `freeze_model` every target weight is bit-identical to the loaded one.
- 20 steps with random gradients: frozen singular values of one matrix within 2e-4 of the
  originals (values up to 6.06, bfloat16 weights); mass 6340.53 before, 6340.56 after.
