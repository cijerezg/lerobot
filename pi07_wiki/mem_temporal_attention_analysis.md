# MEM temporal attention — spec, our deviation, and what we measured

> **BUILT 2026-08-03.** `_temporal_vision_block` now runs the separable form described in
> §6: a time-only softmax over the $K$ timesteps, then the stock spatial block. The
> probe's null moved from $T/(n+T)$ to $T/(T+1)$, so **enrichment numbers in §4 are not
> comparable to anything measured after this date** — they are kept as the record of what
> the bug looked like. Unit tests:
> `tests/policies/test_molmoact2_mem_encoder.py` (11 passing, including a regression test
> that past mass is independent of the patch count).
>
> **Topology note (2026-09-01):** the deployed window later changed from five past
> frames at $-5/-4/-3/-2/-1$ s to three at $-6/-4/-2$ s. The prose, equations, and
> measurements below intentionally retain the old topology because they document the
> 2026-08-03 pre-fix experiment. Current probes derive their ages and the
> $T/(T+1)=3/4$ null from `memory.history_offsets_seconds`.

**Verdict (2026-08-03): our video encoder deviated from MEM in one place, and it was the
place that mattered.** MEM runs spatial and temporal attention as *two separate attention
operations, each with its own softmax*. We merged them into a single softmax over the
union of spatial and temporal keys. That caps the past at 0.68% of the attention budget
instead of 83%, and suppresses the gradient reaching the temporal path by the same
factor. Everything else — same-patch causal keys, no new parameters, sinusoidal $e(t)$,
every 4th layer, six frames at 1 s, history dropout 0.3 — is faithful.

Sources, read 2026-08-03:

| | where |
|---|---|
| MEM, arXiv 2603.03596 — "factorizes attention into separate spatial and temporal attention operations"; same-patch causal mask; Fig. 4 | **p. 4** |
| MEM — Appendix C: $\hat z = z + e(t)$, the $q/k/v$ reuse (Eq. 1), the $\alpha(S,T)$ operator (Eq. 2), space-only / time-only definitions, Eq. 3 | **p. 15** |
| π0.7, arXiv 2604.15483 — "follows the design of the MEM video history encoder" | **p. 4** |
| π0.7 — Sec. VI-B, history hyperparameters (6 frames, 1 s stride, dropout 0.3, state tokens) | **p. 6** |
| π0.7 — Fig. 8, memory tasks vs fine-tuned specialists | **p. 10** |

π0.7 does not redesign the encoder; it imports MEM's and changes what enters the context
(subgoal images, metadata, richer language). So MEM Appendix C is the spec.

## 1. What MEM specifies

Frames indexed by age $t \in [-K, 0]$, current frame $t = 0$; patches by $p$. The age
stamp is added at the layer input,

$$\hat z^{\,l-1}_{p,t} = z^{\,l-1}_{p,t} + e(t), \qquad e(0) = 0,$$

with $e$ sinusoidal in real seconds. Queries, keys and values reuse the ViT's own
projections (Eq. 1) — the encoder introduces **no new learnable parameters**:

$$q^{l,a}_{p,t} = W^{l,a}_Q \mathrm{LN}(\hat z^{\,l-1}_{p,t}), \qquad
k^{l,a}_{p,t} = W^{l,a}_K \mathrm{LN}(\hat z^{\,l-1}_{p,t}), \qquad
v^{l,a}_{p,t} = W^{l,a}_V \mathrm{LN}(\hat z^{\,l-1}_{p,t})$$

Eq. 2 defines an attention operator over a key set indexed by $p' \in S$, $t' \in T$, and
the paper then names two instantiations: **space-only** $\alpha(S=\{1..N\}, T=\{\})$ —
ordinary attention within one frame — and **time-only** $\alpha(S=\{\}, T=\{1..T\})$ —
the same patch $p$ across timesteps, causally masked. Every 4th layer applies both, as a
composition, each with its own softmax.

**Order: temporal first**, on the layer input, before that layer's spatial attention.
Eq. 3 nests the temporal operator inside, and MEM's cited precedent — TimeSformer
(arXiv 2102.05095), whose variant is named *T+S* for time-then-space — states it
outright: "within each block $\ell$, we first compute temporal attention by comparing
each patch $(p,t)$ with all the patches at the same spatial location in the other
frames". The same-spatial-location aperture is inherited from there too. With residuals
between the sub-steps, one temporal layer is

$$z_{\text{time}} = \mathrm{MSA}_{\text{time}}(\mathrm{LN}(\hat z)) + \hat z, \qquad
z_{\text{space}} = \mathrm{MSA}_{\text{space}}(\mathrm{LN}(z_{\text{time}})) + z_{\text{time}}, \qquad
z^{\,l} = \mathrm{MLP}(\mathrm{LN}(z_{\text{space}})) + z_{\text{space}}$$

"Before spatial attention" does not mean raw patches: at layer $l$ the input has already
passed through $l-1$ layers of spatial attention, so what is compared across time is a
contextualized representation — just not contextualized by *this* layer.

**Where MEM departs from TimeSformer: the parameters.** TimeSformer learns "distinct
query/key/value matrices for each attention step"; MEM reuses the ViT's single set to
keep its no-new-parameters property. Age addressing therefore has to ride the $e(t)$
cross-terms of §3 rather than dedicated weights — and adding a low-rank temporal-only
$\Delta W_Q, \Delta W_K$ is a partial move back toward the design MEM forked from, not a
novel intervention.

For the time-only step at patch $p$, the weights therefore normalize **over time alone**:

$$\beta_{t \to t'} = \frac{\exp\big(q_{p,t} \cdot k_{p,t'} / \sqrt d\big)}
{\sum_{s \le t} \exp\big(q_{p,t} \cdot k_{p,s} / \sqrt d\big)}, \qquad \sum_{t' \le t} \beta_{t \to t'} = 1$$

The causal mask includes the query's own timestep, so "ignore history" is reachable by
putting mass on $t' = t$. It is an available outcome, not a floor.

> Caveat on the source: Eq. 3 as printed composes $\alpha(S=\{1..N\},T=\{\})$ with an
> inner $\alpha(S=\{1..N\},T=\{1..T\})$ — the *joint* space-time attention that the body
> text on p. 4 explicitly says they avoid as "prohibitively expensive", and that
> contradicts their $O(Kn^2 + nK^2)$ complexity claim. Given the sentence immediately
> before it defines time-only as $S=\{\}$, the inner term is almost certainly a typo.
> Either reading is incompatible with a single union softmax.

## 2. What we implemented instead

[`_temporal_vision_block`](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L336)
builds both logit families and normalizes them **together**:

$$s_{i \to j} = \frac{1}{\sqrt d} \langle q^{t}_{i}, k^{t}_{j} \rangle \ \ (j = 1..n),
\qquad u_{i \to t'} = \frac{1}{\sqrt d} \langle q^{t}_{i}, k^{t'}_{i} \rangle \ \ (t' < t)$$

$$Z = \sum_{j=1}^{n} e^{s_{i \to j}} + \sum_{t' < t} e^{u_{i \to t'}}, \qquad
\alpha_{i \to j} = \frac{e^{s_{i \to j}}}{Z}, \qquad \beta_{i \to t'} = \frac{e^{u_{i \to t'}}}{Z}$$

$$o^{t}_{i} = \sum_{j=1}^{n} \alpha_{i \to j} v^{t}_{j} + \sum_{t' < t} \beta_{i \to t'} v^{t'}_{i}$$

The temporal keys are correct (same patch, causal, reused projections). The normalizer is
not: five temporal keys compete against $n = 729$ spatial ones inside a single $Z$.

**Forward consequence.** With equal logits, past mass is $T/(n+T) = 5/734 = 0.68\%$ under
ours, against $5/6 = 83\%$ under MEM's time-only softmax. Two orders of magnitude.

**Gradient consequence, which is the worse half.** The derivative of a softmax weight
w.r.t. its own logit scales like $\beta(1-\beta)$. At $\beta \approx 0.003$ the signal
reaching a temporal logit is suppressed by the same factor that suppresses its forward
contribution: the path cannot learn to be useful because it is not used, and it is not
used because it cannot learn. At $\beta \approx 1/6$ the head starts with a real stake in
the past and learns *down* if the past turns out to be useless.

## 3. How selectivity arises with no new parameters

Worth spelling out, because "reuses $W_Q, W_K, W_V$" invites the question of where any
interesting behaviour could come from, and because it is the reason recency does not
dominate by construction.

**The score is a learned bilinear form, not a similarity.** Writing $M = W_Q^\top W_K$,

$$\text{logit}(t \to t') = \frac{1}{\sqrt d}\, \mathrm{LN}(\hat z_{p,t})^\top M\, \mathrm{LN}(\hat z_{p,t'})$$

$M$ is neither symmetric nor positive definite, so a key resembling the query does not
imply a high score, and a patch attending to *itself* is not a default. A head can learn
an $M$ that fires when the past differs from the present along some direction — "was this
location occupied before?" — which is the opposite of a recency detector. A pretrained
ViT does carry a content-matching prior, so recency is favoured at initialization; it is
a prior over a $K$-way softmax, not a structural cap.

**$e(t)$ is what makes age addressable.** Since the age vector is added into both sides,
expanding the bilinear form gives four terms (LayerNorm is not linear, so this is
first-order, not exact):

$$z_t^\top M z_{t'} \;+\; z_t^\top M\, e(t') \;+\; e(t)^\top M z_{t'} \;+\; e(t)^\top M\, e(t')$$

- $z^\top M z$ — content against content: the appearance-change detector.
- $e(t)^\top M e(t')$ — a function of $(t, t')$ only: a **relative-time bias**, learnable
  through $M$ alone. This is how "prefer −3 s" is expressible with no new weights, by the
  same mechanism that gives sinusoidal position embeddings their relative-position bias.
- The two cross terms — **content-conditioned age selection**: which age to read can
  depend on what the patch looks like now. "I am occluded, go back to when I was not"
  lives here, and it is the capability the whole design is for.

The values retrieved, $v_{p,t'} = W_V \mathrm{LN}(\hat z_{p,t'})$, carry that patch's past
appearance, so the output can express "what was here 3 s ago".

Empirical support that this does develop: π0.7 p. 10 extends the same encoder to 18
frames / 54 s at inference and matches memory specialists fine-tuned per task — beyond
what a pure recency mechanism could do.

*Sections 3's expansion and the gradient argument in §2 are our derivation, not the
papers'. MEM states only the projection reuse, the same-patch causal mask, and $e(t)$.*

## 4. What we measured (pre-fix baseline)

`mem_temporal_attention`, 192 frames, 3 episodes, 16 heads, 2 cameras, on
`molmoact2_offline_rebot_all-v2` validation step 0 (policy from
`all-v1/checkpoints/000400`, 400 optimizer steps). Mean enrichment over the uniform-key
null 2.40, strongest layer 6.80, normalized age entropy 0.838.

Past-attention mass ÷ $m_{\text{uniform}}$ — `top` is fixed to the table, `wrist` travels
with the arm:

| ViT layer | top (static) | wrist (moving) |
|---|---|---|
| 3 | 9.11 | 4.48 |
| 7 | 2.91 | 1.91 |
| 11 | 2.63 | 1.50 |
| 15 | 1.59 | 0.95 |
| 19 | 0.86 | 0.61 |
| 23 | 1.31 | 0.91 |

Share of past mass by age (flat would be 0.2 each):

| camera | −5 s | −4 s | −3 s | −2 s | −1 s |
|---|---|---|---|---|---|
| top | 0.077 | 0.088 | 0.111 | 0.186 | 0.538 |
| wrist | 0.108 | 0.114 | 0.130 | 0.192 | 0.457 |

Frame-to-frame variation of past mass: CV 0.07–0.15. Downstream,
`mem_history_influence` on the same checkpoint: image history moves the chunk with RMSE
0.0021 against state history's 0.0104, GT-MSE improvement $-4.2\times10^{-5}$
($t=-1.60$), and `full` correlates with `states` at 0.991 — image history adds nothing
measurable. Context: the arm moves 3–5° over a full 1 s chunk, so −1 s is nearly the
current frame.

Three readings, and note all three were taken under the broken normalizer:

1. **The static camera reads ~2× harder than the moving one at every depth.** Both see the
   same episode at the same instant, so this is the same-patch aperture — a property of
   MEM's design, not of our bug. It is the one finding here that will survive the fix, and
   it says a moving wrist camera is the hard case for same-position matching.
2. **The read dies with depth** (chance by layer 19). Consistent with a starved channel
   that only the earliest, most low-level layer could exploit.
3. **The read barely varies with the scene** (CV ≤ 0.15) — a constant per-age bias, i.e.
   the $e(t)^\top M e(t')$ term above and nothing else. The content-conditioned terms of
   §3 never developed, which is exactly what the gradient argument predicts.

## 5. Where the 2026-08-03 setup already matched π0.7

Verified against p. 6 at the time of the recorded experiment. The history-window rows
below are historical; see the topology note at the top for the current deployment.

| detail | π0.7 | ours |
|---|---|---|
| history stride | 1 s | 1 s |
| history frames | up to 6 | 6 (5 past + current) |
| token budget | compressed to a single frame's token count | past rows dropped after the last temporal layer |
| history dropout | 0.3, whole history block | 0.3, [`shared_config.py:21`](../src/lerobot/rl/shared_config.py#L21) |
| state history | linear projection, one token per past state | `state_history_projector` |
| dropout coupling | state token masked when its history frame is dropped | one flip drops both, [`processor_molmoact2.py:1169`](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1169) |

Deliberate differences, none architectural: 378×378 crops (MolmoAct2 native) vs 448×448,
2 cameras vs up to 4, chunk 30 vs 50.

## 6. The fix, and what it forces us to re-derive

Split `_temporal_vision_block` into two attention operations: time-only over $t' \le t$
per patch, then space-only within the frame, each with its own softmax, each followed by
the standard $W_O$ + residual + MLP path.

Order is settled (§1): temporal on the layer input, then spatial, with a residual between
them — the TimeSformer T+S block with shared projections and a causal mask. The p. 4
phrase "additively adds attention over the time dimension" describes that added sub-step,
not a summation of two attention outputs.

**Decided 2026-08-03: we drop the $K=1$ identity invariant.** MEM's appendix defines
$\alpha$ as attention *weights* combined with $v$ once, which preserves exact single-image
behaviour; a TimeSformer-style pair of full sublayers does not, because a temporal softmax
over a single self-key still returns $W_O W_V \mathrm{LN}(\hat z)$ into the residual. We
are taking the sublayer form and accepting that the no-history path is no longer bit-exact
with the pretrained ViT. The property is nice-to-have, not load-bearing, and single-frame
behaviour is trackable through the existing probes.

Two consequences that follow from that choice:

1. **`history_dropout` changes meaning.** At 0.3, three samples in ten take the no-history
   path, and the comment at
   [`processor_molmoact2.py:1169`](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1169)
   claiming those "degenerate to the exact K=1 pretrained path" stops being true — they
   run the temporal sublayer against a single self-key. Correct the comment when the block
   is rebuilt; the dropout itself still does its job (no history is visible).
2. **The no-new-parameters constraint is now optional.** It only ever existed to support
   the identity claim. Dedicated temporal $W_Q, W_K$ — full or low-rank — are back on the
   table, which is what TimeSformer itself does (§1). New modules must be whitelisted in
   `_apply_actor_freeze` or they will silently not train.

Implementation note: with a separate temporal softmax, `history_on=False` must leave the
query's own timestep unmasked. Masking every temporal key to $-\infty$ gives an all-masked
row and a NaN, where the union form merely fell back to the spatial keys.

Deferred until after the fix, since they were prioritized against a broken baseline:
pose-warping the wrist history, and widening the aperture to a $k \times k$ window
(at $k=3$, 45 temporal keys; the uniform null becomes $k^2T/(n + k^2T)$). Reading 1 of §4
says the aperture is a real limitation of the design — inherited from TimeSformer — but it
should be re-measured with a working normalizer before we spend anything on it. A
low-rank temporal-only $\Delta W_Q, \Delta W_K$ (§1) belongs in the same bucket.

## 7. Scope note — this is not the mistake-memory channel

Same-patch attention over a 5 s window is the wrong instrument for "reconstruct what led
to the failure"; that needs semantic retrieval over tens of seconds and belongs to the MEM
summary $m_t$ ([04 — Memory §3.2](04_memory.md#32-the-mem-summary-m_t)). Scope the video
encoder to short-horizon dynamics — what just moved, what is occluded — and judge it on
that.

`mem_history_influence` also cannot currently see mistake-related benefit: it averages
GT-MSE improvement over evenly sampled frames, and the metadata probe finds 5 of 192
frames carrying a mistake flag. Measuring memory-for-mistakes needs a sampler that
targets mistake-adjacent frames.
