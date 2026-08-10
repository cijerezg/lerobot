# A trajectory-aware auxiliary loss for the FAST head

> # ⚠ ARCHIVED — UNCONFIRMED. DO NOT CITE ANY NUMBER IN THIS DOCUMENT.
>
> **Archived 2026-08-09.** The quantitative claims below did not survive scrutiny: the
> calculations were **imprecise and overstated the effect they were arguing for**. They
> were never independently reproduced. Treat every table, ratio, prevalence, and
> threshold in §3, §6.5 and §6.6 as **unverified and probably wrong in magnitude**, and
> treat the motivating argument — that flat BPE cross-entropy is materially hurt by its
> lack of ordinal structure — as **an open question that this document does not settle**,
> not as an established result.
>
> This note is kept only as the provenance record for the construction that shipped. It
> is **not** a specification of that code and it is **not** evidence for turning the loss
> on.
>
> **What is actually true, read from the code, not from here:**
>
> - The loss exists and is off by default:
>   `discrete_action_auxiliary_loss` in
>   [`configuration_molmoact2.py`](../../src/lerobot/policies/molmoact2/configuration_molmoact2.py),
>   implemented in
>   [`modeling_molmoact2.py`](../../src/lerobot/policies/molmoact2/modeling_molmoact2.py)
>   (`_fast_action_auxiliary_components`). See
>   [05 — Training §2.2](../05_training.md).
> - **It ships three terms, not one:** `ordinal`, `path`, `shape`. §6.7 below claims
>   Stage 1 does not compute `path_mse`/`shape_mse`. That is **no longer true** — both are
>   computed from the marginal mean $\mathbb E[c]$ without a reconstruction.
> - **The $\sigma$-smoothed soft target of §5/§6.3 was never built.** The shipped ordinal
>   term is a *cumulative-threshold* binary CE, which has no $\sigma$ at all. Every
>   sentence below about choosing $\sigma$ describes a formulation that does not exist.
> - **The slot weighting $w_k$ of §6.6 was never built.** The shipped term is a uniform
>   mean over all 210 slots. §6.6 argues at length that this is degenerate; that argument
>   rests on the same measurements this banner retracts, so it is not a reason to add
>   weighting either. If the term is ever turned on, measure the per-band contribution
>   first and decide from that.
> - The construction itself *was* validated numerically, independently of this document,
>   by [`fast_soft_decode_probe.py`](../../../fast_soft_decode_probe.py): exact one-hot
>   identity (max $|\Delta a| = 0$), exact tiling of the 210 slots on real chunks,
>   monotone degradation as the head softens. That script — not this note — is the
>   trustworthy artifact.
>
> Everything below is the original text, unedited.

---

**Status (original):** design, not built. Proposed 2026-08-08, reformulated 2026-08-09.

The FAST head is trained by flat cross-entropy over BPE tokens. That objective is
correct, but it cannot tell a coefficient off by 1 from one off by 400. This note gives
the head a **distance-aware** loss on a support where distance is trajectory-commensurate,
while staying inside the cross-entropy family — no regression term, no loss-scale
balancing.

It does **not** compute `path_mse`, `shape_mse`, or any of the flow branch's four terms.

**§6 is the section to read.** It is self-contained — every symbol defined, the two
cross-entropies compared side by side, the gradients derived, the scale and entropy floor
worked out, and the measurements that constrain $\sigma$ and $w_k$. §1–§5 are the setup;
§6 is the proposal. It also records the one measurement that rules out the frequency
weighting the geometry in §3.3 appears to recommend.

The construction adds **zero parameters**. It reads the FAST head's existing logits and
turns them into a distribution over each DCT coefficient, which is where the ordinal
structure lives.

---

## 1. Notation

Fixed once, used throughout. Nothing else is introduced later.

| symbol | meaning | value here |
|---|---|---|
| $T$ | chunk length (timesteps) | 30 |
| $D$ | action dimensions (joints) | 7 |
| $N = TD$ | coefficients per chunk | 210 |
| $\gamma$ | quantization scale | 10 |
| $m$ | bin offset (`min_token`) | −55 |
| $a$ | normalized action chunk, $a_{t,d}$ | $T \times D$ |
| $C$ | orthonormal DCT-II matrix over time | $T \times T$ |
| $c$ | quantized coefficients, $c_{k,d}$ | integers |
| $n$ | **flat slot index**, $n = Dk + d$ | $0 \dots 209$ |
| $V$ | FAST BPE vocabulary size | 1005 |

The DCT matrix is

$$C_{k,t} = \alpha_k \cos\!\Big(\frac{\pi (2t+1) k}{2T}\Big), \qquad
\alpha_0 = \sqrt{1/T}, \quad \alpha_{k \ge 1} = \sqrt{2/T}$$

and it is orthogonal: $C^\top C = I$. Encoding is

$$c_{k,d} = \operatorname{round}\Big(\gamma \sum_t C_{k,t}\, a_{t,d}\Big)$$

then the $(k,d)$ grid is flattened **frequency-major** into 210 slots, $n = Dk + d$, so
slots 0–6 hold the DC row and the span runs coarse → fine. Each slot value is shifted by
$-m$ and written as one character; byte-level BPE then merges those characters into
tokens. Decoding is exact and linear: $a = C^\top c / \gamma$.

Two per-token quantities, read off the tokenizer once:

- $\operatorname{len}(v)$ — how many coefficients token $v$ decodes to.
- $\operatorname{val}(v, r)$ — the $r$-th coefficient of token $v$, for $r < \operatorname{len}(v)$.

---

## 2. What the current loss does and does not measure

The discrete branch
([`modeling_molmoact2.py:2209`](../../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L2209))
is flat cross-entropy over BPE tokens against `lm_head`.

**That objective is correct.** It is the MLE for a discrete sequence model, a proper
scoring rule, and summed per-token CE is exactly the log-likelihood of the underlying
coefficient string — so it is tokenization-invariant and a long merged token is not
"undercharged".

The narrow claim: **CE has no ordinal structure over coefficient values**, so its
gradient away from the optimum cannot distinguish a prediction off by 1 from one off by
400. Under the v6 memorization result (val/train FAST CE 2.97x, 38 episodes as the
effective sample size) the head is far from its optimum on held-out data, which is the
regime where that matters.

In the live config the FAST head is KI-style — `action_mode: both`,
`inference_action_mode: continuous`
([`config_rl.yaml:113`](../../../config_rl.yaml#L113)). It never acts; it shapes the shared
backbone. So it is being asked to teach the backbone about trajectories through an
objective that cannot measure trajectory error.

---

## 3. Why the fix belongs at the coefficient, not the token

### 3.1 BPE token ids carry almost no ordinal information (measured)

| | |
|---|---|
| Tokens decoding to exactly 1 coefficient | 148 of 1005 |
| …of those, `id == bin value` | 122 (bins 0–121) |
| **Multi-coefficient merge tokens** | **857 of 1005** |
| **corr(token id, first coefficient)** | **+0.057** |

On real chunks: 20.6 tokens per 210 coefficients; 52% of tokens carry a single
coefficient but they cover only **5% of the slots**; the longest token in a chunk swallows
~161 coefficients. Smoothing a target distribution across neighbouring *token ids* would
spread mass over unrelated symbols.

The **coefficients** are what is ordered — consecutive integers, roughly 122 of them.

### 3.2 Coefficient error is trajectory error (Parseval)

$C$ is orthogonal, so for any coefficient error $\delta c$ with $\delta a = C^\top \delta c$:

$$\sum_{t,d} \delta a_{t,d}^2 \;=\; \sum_{k,d} \delta c_{k,d}^2$$

Verified numerically to full precision (193.440337587 on both sides). So squared distance
on coefficients and squared distance on the trajectory are the same quantity — a
*geometry* result, about which errors count how much. It does not by itself mean any
particular loss computes `path_mse`; see §6.2.

### 3.3 Smoothness is a frequency weighting

The DCT-II diagonalizes the Neumann Laplacian, so first-difference energy — the quantity
`shape_mse` penalizes — is an exact per-frequency reweighting:

$$\sum_{t,d} \big(\delta a_{t+1,d} - \delta a_{t,d}\big)^2
\;=\; \sum_{k,d} \lambda_k\, \delta c_{k,d}^2,
\qquad \lambda_k = 4\sin^2\!\Big(\frac{\pi k}{2T}\Big)$$

Verified: ratio 1.000000. So smoothness pressure costs one weight vector, no
reconstruction.

**This is a geometry result, not a recommendation.** $\lambda_0 = 0$ exactly, so using
$\lambda_k$ as the loss weight deletes the DC row — which carries almost all the
coefficient energy. §6.6 measures this and rules the pure form out.

### 3.4 Terminal and direction are the exceptions

$a_{T-1,d} = \sum_k C_{k,T-1}\, c_{k,d}$ is a **rank-1 form coupling all 30 frequencies**
(the $C_{k,T-1}$ entries range 0.0135–0.2578), and `terminal_direction_loss` is a cosine,
not a quadratic at all. Neither can be written as independent per-slot weights.

This is the dividing line, and it falls exactly on the grounding terms: **time-invariant
quadratics are free in coefficient space; time-localized or non-quadratic terms require
reconstructing $\hat a$.** Terminal and direction exist to break ties among trajectories
with equal path error — that is precisely what a diagonal metric cannot express.

---

## 4. The construction: a distribution per coefficient slot

The FAST head gives a distribution over *tokens*. We need one over the *value at each
slot*. The obstacle is that a token covers a variable-length run of slots, so under the
model's own distribution the slot boundaries would be random.

**Teacher forcing removes that.** During training the ground-truth tokens $y_1 \dots y_M$
are known, so token $i$ starts at a known slot

$$\text{start}(i) \;=\; \sum_{i' < i} \operatorname{len}(y_{i'}), \qquad
\sum_{i=1}^{M} \operatorname{len}(y_i) = N$$

Consecutive tokens therefore **tile** $\{0, \dots, N-1\}$ with no gap and no overlap.
Read that backwards: every slot $n$ is covered by exactly one token position, written
$\operatorname{tok}(n)$, at offset $\operatorname{off}(n) = n - \text{start}(\operatorname{tok}(n))$.

Let $p_i$ be the head's distribution over the $V$ action-token ids at position $i$
(gather the `lm_head` logits at those ids, softmax over them alone). The predicted
distribution over the value at slot $n$ is then

$$q_n(u) \;=\;
\frac{\sum_{v \,:\, \operatorname{val}(v,\, \operatorname{off}(n))\, =\, u} \; p_{\operatorname{tok}(n)}(v)}
     {\sum_{v \,:\, \operatorname{len}(v) \,>\, \operatorname{off}(n)} \; p_{\operatorname{tok}(n)}(v)}$$

In words: **of the probability mass sitting on tokens long enough to reach this slot, how
much of it puts value $u$ there.**

The denominator is load-bearing. Tokens shorter than $\operatorname{off}(n)$ say nothing
about slot $n$; without conditioning on "reaches this slot at all", their mass would leak
in as a phantom vote for the padding value and bias every slot toward the bin floor.

Mechanically this is a scatter-add: for each slot, distribute $p_{\operatorname{tok}(n)}$
across ~122 value bins keyed by $\operatorname{val}(\cdot, \operatorname{off}(n))$, then
divide by the covered mass. The keys are static, so they precompute to one sparse matrix
per offset. Total cost is $N \times V \approx 210\text{k}$ multiply-adds per example,
against a full VLM forward.

**Zero new parameters** — it reuses logits already computed for the CE.

### Worked example

Suppose slot $n = 8$ (so $k=1$, $d=1$) falls at offset 2 inside token position 3, the
true coefficient there is $c_8 = -12$, and the head puts its mass on four tokens:

| token $v$ | $p_3(v)$ | $\operatorname{len}(v)$ | $\operatorname{val}(v, 2)$ | contributes |
|---|---|---|---|---|
| A | 0.55 | 6 | −12 | 0.55 to $u = -12$ |
| B | 0.25 | 6 | −11 | 0.25 to $u = -11$ |
| C | 0.15 | 2 | — | nothing (too short) |
| D | 0.05 | 9 | +40 | 0.05 to $u = +40$ |

Covered mass is $0.55 + 0.25 + 0.05 = 0.85$, so
$q_8(-12) = 0.65$, $q_8(-11) = 0.29$, $q_8(+40) = 0.06$. Token C is excluded rather than
counted as a vote for zero.

---

## 5. The loss, in one line

Cross-entropy against a Gaussian-smoothed target over **coefficient values** — HL-Gauss,
the construction already used for the critic, applied here to a support that is genuinely
ordered:

$$t_n(u) \;\propto\; \exp\!\Big(-\frac{(u - c_n)^2}{2\sigma^2}\Big),
\qquad
\mathcal{L}_{\text{ord}} \;=\; \frac{1}{N}\sum_{n=0}^{N-1} w_{k(n)} \sum_u \; -\,t_n(u)\, \log q_n(u)$$

The BPE cross-entropy is untouched and remains the sampling path;
$\mathcal{L}_{\text{ord}}$ is added with its own scalar weight.

Three parameters decide whether this works, and **none of them can be set from the theory
above** — $\sigma$, the weights $w_k$, and the normalization. §6.5 and §6.6 give the
measurements that constrain them, including one result that rules out the weighting the
§3.3 geometry appears to recommend. Read §6 before implementing.

---

## 6. The loss, rigorously — this section is self-contained

Everything needed to evaluate the proposal is here. No forward or backward references
are required to read it.

### 6.1 Every symbol used below

| symbol | type | meaning |
|---|---|---|
| $T = 30,\; D = 7,\; N = TD = 210$ | ints | timesteps, joints, coefficient slots per chunk |
| $\gamma = 10$ | int | quantization scale: $c = \operatorname{round}(\gamma \cdot \text{DCT}(a))$ |
| $n$ | $0 \dots 209$ | **slot index**, $n = Dk + d$ for frequency $k$, joint $d$ |
| $k(n) = \lfloor n/D \rfloor$ | $0 \dots 29$ | the DCT frequency that slot $n$ belongs to |
| $c_n$ | integer | the **true** quantized coefficient at slot $n$. This is the label. |
| $u$ | integer | a **candidate value** for a coefficient, $u \in \{-55, \dots, 66\}$ (122 of them). The axis HL-Gauss lives on. |
| $v$ | $0 \dots 1004$ | a **FAST BPE token id**. $V = 1005$. |
| $i$ | $1 \dots M$ | a **token position** in the action span. $M \approx 20.6$ per chunk. |
| $y_i$ | token id | the **ground-truth** token at position $i$ |
| $\operatorname{len}(v)$ | int | how many coefficients token $v$ decodes to (1 … 954) |
| $\operatorname{val}(v, r)$ | integer | the $r$-th coefficient of token $v$, defined for $r < \operatorname{len}(v)$ |
| $\operatorname{tok}(n)$ | $1 \dots M$ | which token position covers slot $n$ |
| $\operatorname{off}(n)$ | int | slot $n$'s offset inside that token |
| $p_i(v)$ | distribution over $V$ | the head's prediction at position $i$ (softmax of `lm_head` logits restricted to the 1005 action ids) |
| $q_n(u)$ | distribution over 122 values | **derived**: the implied prediction for the coefficient at slot $n$ |
| $t_n(u)$ | distribution over 122 values | the HL-Gauss **target**, a Gaussian centred on $c_n$ |
| $\sigma$ | real | HL-Gauss width, **in coefficient units** |
| $w_k$ | real | per-frequency weight on the loss |

$\operatorname{tok}$ and $\operatorname{off}$ are well defined because teacher forcing
pins the boundaries: $\operatorname{start}(i) = \sum_{i'<i}\operatorname{len}(y_{i'})$,
and $\sum_i \operatorname{len}(y_i) = 210$ exactly, so the token spans tile the 210 slots
with no gap and no overlap.

### 6.2 The change of variable: tokens carry no order, values do

BPE ids are unordered (§3.1: corr(id, coefficient) $= +0.057$). Coefficient values are
consecutive integers. The construction moves the prediction from the first axis to the
second **before** any distance-based loss is applied:

$$q_n(u) \;=\;
\frac{\sum_{v \,:\, \operatorname{len}(v) > \operatorname{off}(n) \;\wedge\; \operatorname{val}(v,\operatorname{off}(n)) = u} p_{\operatorname{tok}(n)}(v)}
     {\sum_{v \,:\, \operatorname{len}(v) > \operatorname{off}(n)} p_{\operatorname{tok}(n)}(v)}$$

Numerator: mass on tokens that put value $u$ at this slot. Denominator: mass on tokens
that reach this slot at all. Tokens too short to reach slot $n$ say nothing about it and
are excluded — without that conditioning their mass would read as a vote for the padding
value and drag every slot toward the bin floor.

Concretely, at some slot with true value $c_n = -12$:

| token $v$ | $p_i(v)$ | $\operatorname{len}(v)$ | $\operatorname{val}(v, \operatorname{off}(n))$ | contributes |
|---|---|---|---|---|
| A | 0.55 | 6 | −12 | 0.55 → bin $u = -12$ |
| B | 0.25 | 6 | −11 | 0.25 → bin $u = -11$ |
| C | 0.15 | 2 | — | excluded (too short) |
| D | 0.05 | 9 | +40 | 0.05 → bin $u = +40$ |

Covered mass $= 0.85$, so $q_n(-12) = 0.65$, $q_n(-11) = 0.29$, $q_n(+40) = 0.06$.
A and B have unrelated token ids but land in adjacent value bins; that is the whole point.

### 6.3 The two cross-entropies, side by side

$$\mathcal{L}_{\text{BPE}} = -\log p_i(y_i)
\qquad\qquad
\mathcal{L}_{\text{ord},n} = -\sum_u t_n(u)\,\log q_n(u),
\quad t_n(u) \propto e^{-(u-c_n)^2/2\sigma^2}$$

| | $\mathcal{L}_{\text{BPE}}$ (exists today) | $\mathcal{L}_{\text{ord}}$ (proposed) |
|---|---|---|
| support | 1005 BPE tokens | 122 integer coefficient values |
| support ordered? | **no** | **yes** |
| target | one-hot at $y_i$ | Gaussian on $\lvert u - c_n\rvert$, width $\sigma$ |
| distribution scored | $p_i$ directly | $q_n$, a marginal of $p_{\operatorname{tok}(n)}$ |
| terms per chunk | $M \approx 20.6$ | $N = 210$ |
| minimum achievable | $0$ | $H(t_n) > 0$ — see §6.5 |
| near-miss vs far-miss | identical cost | far miss costs more |
| minimized by | the true conditional | the true conditional |

**They share a minimizer.** If $p_i$ is the true conditional then $q_n$ is the correct
induced marginal, so $\mathcal{L}_{\text{ord}}$ adds no information — it is a function of
the same $p_i$. Its entire contribution is to reshape the gradient. That is the honest
scope of the proposal: not new supervision, a different penalty surface over the same
prediction.

$\mathcal{L}_{\text{BPE}}$ is untouched and remains the sampling path.
$\mathcal{L}_{\text{ord}}$ is added with its own scalar weight.

### 6.4 Gradients

Write $i = \operatorname{tok}(n)$, $r = \operatorname{off}(n)$, and split $q_n = A_n/B_n$
with $A_n(u) = \sum_{v:\,\operatorname{val}(v,r)=u} p_i(v)$ and
$B_n = \sum_{v:\,\operatorname{len}(v)>r} p_i(v)$. Since $\sum_u t_n(u) = 1$,

$$\mathcal{L}_{\text{ord},n} = -\sum_u t_n(u)\log A_n(u) + \log B_n$$

so for a token $v$ that reaches slot $n$ (writing $u_v = \operatorname{val}(v,r)$):

$$\frac{\partial \mathcal{L}_{\text{ord},n}}{\partial p_i(v)}
= -\,\frac{t_n(u_v)}{A_n(u_v)} + \frac{1}{B_n}$$

and $0$ for tokens too short to reach it. Compare
$\partial \mathcal{L}_{\text{BPE}} / \partial p_i(v) = -\mathbb{1}[v = y_i]/p_i(v)$.

The structural difference: $\mathcal{L}_{\text{ord}}$ rewards $v$ in proportion to
$t_n(u_v)$ — how close $v$'s **value** is to the truth — pooled over every token sharing
that value, whereas $\mathcal{L}_{\text{BPE}}$ rewards only the exact id $y_i$. A token
that is one coefficient away from correct receives positive gradient under the first and
none under the second.

This is an analytic statement about the derivative. It is **not** a claim that the
resulting training dynamics are better; see §7.

### 6.5 Scale, normalization, and the entropy floor

Two traps when reading the logged number.

**Term count.** $\mathcal{L}_{\text{ord}}$ has 210 terms per chunk against
$\mathcal{L}_{\text{BPE}}$'s ~20.6. Sum both and the auxiliary is ~10x larger before any
weight is applied. Normalize $\mathcal{L}_{\text{ord}}$ by $N$ to a per-slot mean, so the
configured weight means what it looks like.

**The floor is not zero.** A perfect predictor still pays $H(t_n)$, the entropy of the
smoothed target — which is large and depends entirely on $\sigma$:

| $\sigma$ (coef units) | $H(t)$ nats | $\sigma/\gamma$ | RMS trajectory error per element |
|---|---|---|---|
| 0.5 | 0.67 | 0.05 | 0.0035 |
| 1 | 1.42 | 0.10 | 0.0069 |
| **2** | **2.11** | **0.20** | **0.0138** |
| 4 | 2.81 | 0.40 | 0.0276 |
| 8 | 3.50 | 0.80 | 0.0552 |

So at $\sigma = 8$ the loss bottoms out around 3.5 nats and never approaches zero. **Log
$\mathcal{L}_{\text{ord}} - \overline{H(t_n)}$, not the raw value**, or the curve is
mostly a constant.

**Choosing $\sigma$.** It is in coefficient units, and by Parseval an error of $\sigma$ in
one coefficient is a trajectory L2 error of $\sigma/\gamma$ over the chunk, i.e.
$\sigma/(\gamma\sqrt{N})$ RMS per element (last column). The median quantization error
already in the pipeline is **0.0133**, which lands at $\sigma \approx 2$ — smoothing
narrower than the quantization noise asks the head to resolve detail the encoding already
destroyed. **The critic's tuned $\sigma = 8$ does not transfer**: different support,
different units, and here it corresponds to a 4x-quantization-noise tolerance.

### 6.6 Slot weighting — measured, and it breaks the obvious choice

Coefficient energy is concentrated almost entirely in the lowest frequencies, and the
high-frequency slots are covered by enormous zero-run merge tokens. Measured over 97 real
chunks:

| frequency $k$ | mean $\operatorname{len}$ of the covering token | mean $\lvert c_n \rvert$ |
|---|---|---|
| 0 (DC) | 1.2 | 9.78 |
| 1 | 1.9 | 2.36 |
| 2 | 2.8 | 1.09 |
| 3 | 8.8 | 0.44 |
| 4 | 63.7 | 0.15 |
| 5 | 143.4 | 0.03 |
| 10–29 | ~161 | 0.00 |

**165 of 210 slots are covered by a token longer than 50**, and mean $\lvert c_n\rvert$
over $k \ge 3$ is **0.023** versus **4.41** for $k < 3$.

Two consequences:

1. **Unweighted, the loss is degenerate.** ~79% of slots have a deterministic zero target
   and are covered by one giant token whose probability gets replicated across all of
   them. That single prediction would dominate the loss value, and the number logged would
   mostly report "did you emit the zero run", which is nearly free.

2. **The Laplacian weighting from §3.3 is exactly backwards here.** $\lambda_k =
   4\sin^2(\pi k/2T)$ gives $\lambda_0 = \mathbf{0.0000}$ and $\lambda_{29} = 3.99$ — it
   **deletes the entire DC row**, the row carrying almost all the energy, and puts maximum
   weight on slots that are identically zero. It is principled as geometry and wrong as a
   weighting for this data.

Therefore $w_k$ must be chosen empirically, not from §3.3, and $\alpha$ in
$w_k = \alpha + \beta\lambda_k$ must be strictly positive. Two candidates worth measuring
before committing:

- **Coverage-normalized:** $w \propto 1/\operatorname{len}(y_{\operatorname{tok}(n)})$, so
  each token position contributes one unit of loss regardless of how many slots it spans.
  Restores parity with $\mathcal{L}_{\text{BPE}}$'s term count.
- **Band-limited:** apply the loss only to $k < K$ for $K$ around 4–6, where
  $\lvert c_n\rvert$ is non-negligible.

Log the per-band contribution from the first step, and pick $w_k$ from that rather than
from theory.

### 6.7 What is and is not computed

`path_mse` is $\frac{1}{TD}\sum_{t,d}(\hat a_{t,d}-a_{t,d})^2$. Computing that number
requires $\hat a$, hence a reconstruction. **Stage 1 has no $\hat a$ and does not compute
it.** What Stage 1 provides is a cross-entropy that is sensitive to distance on a
trajectory-commensurate axis — which is what §2 asked for, and is a different object.

Expected `path_mse` *is* reachable without reconstruction, and this is not obvious: by
Parseval $\|\hat a - a\|^2 = \|\hat c - c\|^2/\gamma^2$ per realization, and expectation is
linear, so only the marginals are needed — no IDCT, no independence assumption across
slots:

$$\mathbb{E}\big[\texttt{path\_mse}\big]
= \frac{1}{TD\gamma^2}\sum_n \sum_u q_n(u)\,(u-c_n)^2$$

But that is a squared-error term, not a cross-entropy, so it reopens §7. It also differs
from "reconstruct the mean, then square" by exactly the variance,
$\mathbb{E}[(u-c)^2] = (\mathbb{E}[u]-c)^2 + \operatorname{Var}(u)$, so it additionally
penalizes uncertainty and pushes $q_n$ toward a spike.

| metric | needs $\hat a$? | loss family | note |
|---|---|---|---|
| **distance-awareness** (off-by-1 ≠ off-by-400) | no | **CE** | HL-Gauss on $q_n$ — this is Stage 1 |
| expected `path_mse` | no | squared error | formula above; adds a variance penalty |
| expected `shape_mse` | no | squared error | same with $\lambda_k$ — but see §6.6 |
| `terminal_mse` | **yes** | squared error | rank-1 across all 30 frequencies (§3.4) |
| `terminal_direction_loss` | **yes** | cosine | not quadratic (§3.4) |

Row 1 is the only one that stays in the CE family and the only one that addresses §2
directly. Rows 2–3 are genuinely available without a reconstruction, but buy the literal
metric at the price of §7, so they belong with Stage 2.

## 7. What this deliberately avoids, and what it does not settle

**Avoided: mixing a regression term into a CE objective.** An earlier version of this
design collapsed $q_n$ to its mean $\hat c_n = \sum_u u\, q_n(u)$, reconstructed
$\hat a = C^\top \hat c / \gamma$, and applied squared error. That works and needs no new
parameters either, but it introduces two problems the formulation above does not have:

- **Loss-scale and gradient dynamics.** A squared-error term and a cross-entropy term
  have different magnitudes and different gradient behaviour as training proceeds, and how
  they co-evolve on a shared backbone is not something this note can predict. There is
  relevant literature — Imani & White (ICML 2018) for HL-Gauss, Farebrother et al.,
  *Stop Regressing* (ICML 2024) for classification-over-regression at scale, which is why
  the critic uses HL-Gauss — but none of it establishes the transfer to an auxiliary head
  on a VLM backbone sitting next to a token CE. **Treat that as open.**
- **Mean versus mode.** $\hat c$ under a multimodal $q_n$ can be a coefficient vector the
  model would never sample.

Comparing full distributions instead of their means removes both. That is the main reason
to prefer §4–§5.

**Not settled:** whether an ordinal auxiliary helps this backbone at all. It is a shaping
term on a head that never acts; the argument for it is inductive bias under finite data,
not correctness.

---

## 8. Invariants

1. **Slots tile exactly.** $\sum_i \operatorname{len}(y_i) = N = 210$. This was violated on
   4.2% of chunks until the alphabet fix
   ([fast_tokenizer_alphabet_bug.md](../fast_tokenizer_alphabet_bug.md) §9); it now holds
   corpus-wide and `_tokenize_discrete_action` asserts it at encode time.

2. **Per-token decode is additive.**
   $\operatorname{decode}(y_1 \dots y_M) = \operatorname{decode}(y_1) \Vert \dots \Vert
   \operatorname{decode}(y_M)$. Byte-level BPE can in principle split a multi-byte
   codepoint across a merge, which would make $\operatorname{val}$ meaningless. Verified on
   1855 real chunks with zero failures. Assert it when building the tables.

3. **Wrapper tokens carry no coefficients.** `<action_start>` / `<action_end>` sit inside
   the label span
   ([`processor_molmoact2.py:850`](../../src/lerobot/policies/molmoact2/processor_molmoact2.py#L850))
   and must take $\operatorname{len} = 0$ in the `start` cumsum. Otherwise every slot
   shifts. Most likely source of a silent off-by-one.

4. **Action-alphabet mass.** The restricted softmax discards mass placed outside the 1005
   action ids. Log $\sum_v p_i(v)$ over that set; below ~0.99 the restriction is hiding a
   head drifting off-format.

5. **Snap offset.** The alphabet fix perturbs one DC coefficient by
   $1/(\gamma\sqrt{T}) = 0.0183$ on the 4.2% of chunks that hit a hole, so $c_n$ there is
   the snapped value. That is the target, and it is self-consistent.

The $[-1,1]$ normalizer clamp is **not** a concern: on the deployed anchor path it is
0.02% of values / 0.55% of chunks. The previously recorded 31.2% came from a stats
mismatch — see [fast_tokenizer_alphabet_bug.md](../fast_tokenizer_alphabet_bug.md) §8.

---

## 9. Build order

**Stage 1 — ordinal CE on coefficient marginals (§4–§5).** Zero new parameters, stays in
the CE family, covers `path` and `shape`. Hook: inside
`_discrete_loss_from_backbone_outputs`
([`:2177`](../../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L2177)), which already
holds the logits, the per-example indices, and the within-span position ranks.

**Stage 2 — terminal and direction, only if Stage 1 pays.** These need the reconstruction
$\hat a$ (§3.4) and reintroduce everything in §6. Keep them behind a separate weight so
the two can be evaluated apart.

Config: a new `discrete_auxiliary_loss` block rather than overloading
`action_auxiliary_loss` — the flow branch scores a flow sample, this scores a
distribution, and they will not want the same weights.

---

## 10. Validation before wiring anything in

Standalone script first, in the spirit of `basis_roundtrip.py`.

1. **Tiling.** $\sum_i \operatorname{len}(y_i) = 210$ on real chunks, wrapper tokens at
   $\operatorname{len} = 0$.
2. **One-hot identity.** Force $p_i = \delta_{y_i}$. Then $q_n$ must be a point mass on the
   true $c_n$ for every slot. This single check validates $\operatorname{val}$,
   $\operatorname{len}$, the `start` cumsum, wrapper handling, and the $\gamma$/$m$
   conventions together.
3. **Denominator matters.** Compare $q_n$ with and without the covered-mass division on a
   deliberately imperfect distribution; the unnormalized version should show a measurable
   bias toward the bin floor.
4. **Real distribution.** With $p_i$ from a checkpoint, check $q_n$ concentrates near
   $c_n$ — **on val frames as well as train**, given the v6 memorization result.
