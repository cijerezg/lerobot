# Depth read redesign — decision document

Status: **DECIDED AND BUILT 2026-07-26** (§5 records the decisions, §5.2 the
final design; 03_depth.md §B.3 is the as-built reference — this file remains the
rationale/evidence record). This replaced the α-gated read of
[03_depth.md](03_depth.md) Part B. The history
mechanism is already settled and built ([depth_history_design.md](depth_history_design.md),
2026-07-25) and is untouched by everything here: whatever is chosen consumes the
encoder's 192 history-fused tokens.

Budget note (2026-07-25): depth is considered important; ~20M+ fresh parameters
are acceptable. The current stream spends 170M and trains none of them (§2).

---

## 1. Current state, precisely

Action expert: $L = 36$ layers, $H = 8$ heads, $d_h = 96$, width $d_e = 768$.
$x \in \mathbb R^{50 \times 768}$ = action-chunk hidden states, $c$ = flow-timestep
embedding.

Modulation (AdaLN), 9 vectors from $c$:

$$(\sigma^{sa}, \gamma^{sa}, g^{sa},\; \sigma^{ca}, \gamma^{ca}, g^{ca},\; \sigma^{mlp}, \gamma^{mlp}, g^{mlp}) = W_{mod}\, c$$

$$\mathrm{mod}(h, \sigma, \gamma) = h \odot (1 + \gamma) + \sigma$$

Block $\ell$ (three sublayers; depth enters only in the second):

$$x \leftarrow x + g^{sa} \odot \mathrm{SelfAttn}\big(\mathrm{mod}(\mathrm{LN}(x), \sigma^{sa}, \gamma^{sa})\big)$$
$$x \leftarrow x + g^{ca} \odot \mathrm{CrossAttn}\big(\mathrm{mod}(\mathrm{LN}(x), \sigma^{ca}, \gamma^{ca})\big)$$
$$x \leftarrow x + g^{mlp} \odot \mathrm{MLP}\big(\mathrm{mod}(\mathrm{LN}(x), \sigma^{mlp}, \gamma^{mlp})\big)$$

Inside CrossAttn, per head, per query row $i$, with $\tilde x$ the modulated
normed input:

$$q = \mathrm{qnorm}(W_Q \tilde x), \qquad
K_{ctx} = \mathrm{knorm}(W_K\, \mathrm{KV}^{LLM}_\ell), \qquad V_{ctx} = W_V\, \mathrm{KV}^{LLM}_\ell$$

$$o^{ctx}_i = \sum_{j=1}^{T_{ctx}} \frac{e^{\,q_i \cdot k_j / \sqrt{d_h}}}{\sum_{j'} e^{\,q_i \cdot k_{j'} / \sqrt{d_h}}}\; v_j$$

Depth read — separate softmax over the 192 depth tokens plus a sink column
(zero key, zero value, learned logit $\beta_{\ell h}$); $D_\ell$ = DepthStream
state after block $\ell$, $K_d = \mathrm{knorm}(W^r_K D_\ell)$, $V_d = W^r_V D_\ell$:

$$o^{d}_i = \frac{\sum_{m=1}^{192} e^{\,q_i \cdot k_{d,m} / \sqrt{d_h}}\; v_{d,m}}
{\sum_{m=1}^{192} e^{\,q_i \cdot k_{d,m} / \sqrt{d_h}} + e^{\beta_{\ell h}}}$$

The α combination and the output path:

$$o_i = o^{ctx}_i + \tanh(\alpha_\ell)\, o^{d}_i, \qquad
\mathrm{CrossAttn}(\tilde x)_i = W_O\,[\,o_i^{(1)}; \dots; o_i^{(8)}\,]$$

Net depth contribution to the hidden state at layer $\ell$:

$$\Delta x_i = g^{ca} \odot W_O\big[\tanh(\alpha_\ell)\, o^d_i\big]$$

After layer 36: $\hat v = W_{head}\,\mathrm{LN}(x)$, the flow-matching velocity.

---

## 2. Why it fails

### 2.1 Measured

| run | steps | $\tanh\alpha$ mean | $\tanh\alpha$ absmax |
|---|---|---|---|
| rebot_v1 07-19 | 222 | $-2.1\cdot10^{-4}$ | 0.0068 |
| rebot_v1 07-19 | 1025 | $-2.5\cdot10^{-4}$ | 0.0074 |
| rebot_v2 07-19 | 411 | $-4.1\cdot10^{-4}$ | 0.0088 |
| rebot_v2 07-25 | 579 | $+1.3\cdot10^{-4}$ | 0.0066 |

No growth 222 → 1025 steps. Under Adam a consistently-signed gradient moves a
parameter ≈ lr per step; at `depth_lr` $= 5\cdot10^{-4}$ that is $\alpha \approx 0.5$
by step 1000. Observed: 0.007, flat, sign-cancelling across layers ⇒ the
α-gradient is noise. Ruled out: data (88–95 % of pixels valid in
$[70, 800]$ mm, median ≈ 300 mm, 95–99 % patches non-null), plumbing (depth
present in batches), the freeze bug (whitelist + separate lr group verified).

### 2.2 Structural: the scalar bottleneck

At init ($\alpha = 0$, stream random):

$$\frac{\partial \mathcal L}{\partial \alpha_\ell}
= \Big\langle \frac{\partial \mathcal L}{\partial o},\; o^d_\ell \Big\rangle, \qquad
\frac{\partial \mathcal L}{\partial \theta_{stream}} \propto \tanh(\alpha_\ell) \approx 0.007$$

$o^d_\ell$ at init is a near-uniform softmax average of randomly-projected
tokens — approximately a query-independent constant vector. The inner product of
the output gradient with an uninformative constant is noise, so $\alpha$ random
walks; and every stream parameter's gradient carries the factor
$\tanh(\alpha) \approx 0.007$, so the 170M-parameter stream trains at <1 % rate.
A scalar can only **scale** the depth readout; it cannot **select** what is
useful in it. Contrast LoRA/ControlNet: their zero-init factor is a **matrix**
$W$, whose init gradient $\frac{\partial \mathcal L}{\partial W} =
\frac{\partial \mathcal L}{\partial x}\, a^\top$ is rank-structured and
immediately informative. Scalar zero-init gates (Flamingo, LLaMA-Adapter) are
validated only at LLM-scale data.

### 2.3 Structural: modality laziness

The multimodal literature (unimodal bias / greedy learning; FuSe observed it
directly when adding sensors to robot policies) says: if the strong modality
already explains the training data, the joint loss provides ~no gradient to the
weak one, **regardless of architecture**. Our flow loss sits at 0.02 on 6
episodes — RGB + proprio suffice, so

$$\frac{\partial \mathcal L}{\partial (\text{depth path})} \approx 0
\quad \text{is partially *correct* behavior of the loss.}$$

Consequence: an input-path fix alone may reproduce "gate at 0.007" in disguise.
Axis 3 (§4.3) is not optional garnish; it is half the fix.

### 2.4 Logging (bugs to fix in any redesign)

- `pointmap_gate` (console metric) is the signed mean over 36 layers — cancels
  to $10^{-4}$ while layers sit at $7\cdot10^{-3}$.
- `pointmap_gate_grad_absmax` is read **after** `clip_grad_norm_` (observed
  norm ≈ 4.35, clip 1.0 ⇒ reported ≈ 4× under).
- Nothing measures influence. Required regardless of choice:
  per-layer attention mass on depth keys, pre-clip depth-group grad norm, and a
  depth-ablation probe (same batch, real vs null-bank depth, action delta).

---

## 3. Literature (what is validated, and at what scale)

| mechanism | paper | validated at |
|---|---|---|
| MoT: depth tower + shared attention; tower **pretrained** for depth prediction | DepthVLA 2510.13375 | large fine-tune |
| Point tokens projected into the **pretrained 2D embedding space** | Lift3D-VLA 2607.06564 | mid |
| MLP-adapter + **addition** into least-critical blocks of a frozen expert (skip-block analysis) | PointVLA 2503.07511 | **20–80 demos** |
| Depth as **aux supervision only** (predict VQ depth tokens, CE) — no depth input | QDepth-VLA 2510.14836 | +7.7 % LIBERO, +10 % real vs π₀ |
| Scalar zero-init gated attention | Flamingo, LLaMA-Adapter 2303.16199 | LLM-scale data |
| Naive new-sensor fine-tuning → sensor ignored; fixed by cross-modal aux losses | FuSe 2501.04693 | 29k trajs |
| Corrupt the strong modality with **annealed** intensity → policy forced to wire the weak one | FACTR 2502.17432 | contact-rich real |
| Modality-wise dropout to balance RGB / point cloud | DIPOLE 2511.22445 | mid |
| 3D hygiene: LN not BN; keep per-token resolution; cross-attn consumer; **pretrained 3D encoders cut sample needs**; 3D overfits hard on small data | R3D 2604.15281 | systematic |

Lessons: (1) no small-data success uses a from-scratch tower behind a scalar
gate; (2) successful token fusion aligns new tokens to a **pretrained** space or
uses a pretrained tower; (3) the small-data winner is adapter+add into a mostly
frozen expert; (4) making the modality *necessary* (aux loss or corruption of
the strong modality) is a recurring, sometimes sufficient, ingredient.

---

## 4. The decision stack

Three orthogonal axes; every combination is coherent (one cross-constraint,
§4.4).

### 4.1 Axis 1 — input path

**(a) Joint softmax in the expert's cross-attention** (extend the context keys).
Replace the two softmaxes of §1 by one:

$$o_i = \sum_{j \in ctx}\ w_{ij}\, v_j \;+\; \sum_{m \in depth} w_{im}\, v_{d,m}, \qquad
w = \mathrm{softmax}\Big(\big[\ q_i \cdot k_j / \sqrt{d_h}\ ;\ \ q_i \cdot k_{d,m} / \sqrt{d_h} + b_\ell\ \big]\Big)$$

$b_\ell$ = optional learned per-layer bias on the depth columns (init negative,
e.g. $-2$, for a soft start: initial depth mass $\approx 192\, e^{b_\ell} / Z$).
A bias inside a softmax has healthy gradients — unlike a multiplicative scalar,
it does not zero the content gradients.

- Deletes: $\alpha$, the sink (context keys are the natural sink), the additive
  patch. Implementation ≈ 10 lines (today's `read_kv` output concatenated into
  the context K/V).
- Gradients: depth mass at init is nonzero, so gradients reach the read projections
  and everything behind them without an extra $	anh(\alpha)$ multiplier. Their
  magnitude is still proportional to the joint softmax's actual depth mass.
- Depth and context compete for one normalization mass — the mechanism every
  other modality already uses; attention sets the mix per query/head/layer.
- Free telemetry: the depth-column mass *is* the influence measure.
- Init perturbation: nonzero (mitigated by $b_\ell$); the expert is trainable
  and heals it.

**(b) Adapter + addition into selected blocks** (PointVLA pattern). Choose a
block set $\mathcal B$ (PointVLA: the *least*-critical blocks, found by a
skip-block probe — the inversion of our inject-everywhere design). Two
sub-variants, both with a **zero-init output matrix** (the ControlNet/LoRA
pattern, §2.2):

(b1) global embedding, literal PointVLA — $\bar d$ = pooled depth tokens:

$$\ell \in \mathcal B:\quad x_i \leftarrow x_i + W_2^\ell\, \mathrm{SiLU}\big(W_1^\ell\, \bar d\big), \qquad W_2^\ell = 0 \text{ at init}$$

(b2) token-wise, a dedicated cross-attention sublayer with zero-init $W_O^d$:

$$\ell \in \mathcal B:\quad x_i \leftarrow x_i + W_O^d\,\mathrm{Attn}\big(W_Q^d\,\mathrm{LN}(x_i),\ W_K^d D,\ W_V^d D\big), \qquad W_O^d = 0 \text{ at init}$$

- Bit-identical at init **and** trainable from step 0:
  $\partial \mathcal L / \partial W_O^d = (\partial \mathcal L / \partial x)\, a^\top$
  is a full-rank signal, not a scalar projection.
- Own query space for depth (the (a) path reuses queries trained for reading
  the VLM context).
- Params: (b2) ≈ $4 d_e^2 \approx 2.4$M per block; $|\mathcal B| = 9$ (every
  4th) ≈ 21M — matches the budget.
- Strongest precedent at our data scale (20–80 demos).

**(c) Depth through the pretrained vision tower as an extra camera.**
Vanishing Depth (2503.19947): a **frozen RGB encoder** consumes metric depth if
it is preprocessed into multi-channel sinusoids (per pixel, $K$ frequencies
$\lambda_k$ — the per-pixel analog of our Fourier centroid PE):

$$\Phi(u,v) = \big[\sin(2\pi z(u,v)/\lambda_k),\ \cos(2\pi z(u,v)/\lambda_k)\big]_{k=1}^{K} \longrightarrow \text{3-channel groups} \longrightarrow \text{SigLIP}$$

- Maximal reuse: pretrained encoder, pretrained fusion, and the MEM video
  encoder gives depth **history for free** (depth becomes one more camera in
  `history_keys`' image path).
- Cost: +1 camera of LLM-prefix tokens — the original latency concern; the
  prefix runs once per control step (KV cached across denoise steps), so it is
  a per-step cost, not per-denoise-step.
- Depth then also reaches the HL/subtask path (feature or bug, decide).
- Abandons the metric point-map representation (X, Y channels and mm-scale
  geometry are implicit, not explicit).

### 4.2 Axis 2 — depth encoder (what produces the tokens)

| option | prior | params | notes |
|---|---|---|---|
| current from-scratch CNN + PE | none | 9M | R3D warns from-scratch 3D + small data = overfit-prone |
| **Sonata / Utonia** (pretrained PTv3, self-supervised on 3D scans) | strong geometric | frozen + small adapter | eats our back-projected point map; R3D: pretrained 3D encoders sharply cut sample needs; caveat: room-scale pretraining vs close-range wrist view |
| sinusoidal channels into frozen SigLIP | RGB-semantic | ~0 new | only meaningful with path (c) |

Older depth-native ViTs (Omnivore, MultiMAE, ImageBind-depth) exist but are
2022-23 ViT-B class — noted, not favored.

### 4.3 Axis 3 — make depth matter (anti-laziness; §2.3)

**(i) Symmetric modality dropout** (DIPOLE). We drop depth at $p = 0.25$ but
never RGB. Add wrist-RGB dropout: with probability $p_{rgb} \approx 0.1$–$0.25$
**mask the camera's image-patch span out of the attention mask** for the sample
(as built — not a black image: masking removes the tokens from every attention
consumer, so no gradient flows through that camera's vision path and the model
cannot learn "all-black" as a cue; direct feature gathers that bypass attention
take a per-row switch derived from the same mask — 03_depth §B.3.1). Those
samples are solvable only through depth ⇒ direct, non-competing gradient to the
whole depth path:

$$\mathbb E\big[\nabla_{\theta_d} \mathcal L\big] =
p_{rgb}\, \mathbb E\big[\nabla_{\theta_d} \mathcal L \mid \text{RGB off}\big] + (1 - p_{rgb})\,\underbrace{\cdots}_{\approx 0}$$

(the second term is the lazy regime; the first is not). Five lines; permanent.

**(ii) FACTR curriculum.** Corrupt RGB with annealed intensity — e.g. Gaussian
blur with

$$\sigma(t) = \sigma_0\,\max(0,\ 1 - t/T_c)$$

so early training cannot lean on RGB and inference (σ = 0) is unaffected.
Strictly stronger schedule-version of (i); validated for exactly this failure in
contact-rich manipulation.

**(iii) Gradient modulation.** The depth optimizer group already exists;
upweight `depth_lr` (or adapt it from the pre-clip depth-grad norm). A tuning
knob, not a mechanism.

**(iv, PARKED by decision 2026-07-26)** QDepth-style aux supervision (predict
quantized wrist depth from shared features, CE). Revisit if (i)–(iii) +
input-path fix are insufficient. Note its ceiling: policy only acquires geometry
inferable from RGB — true occlusion cases still require depth input.

### 4.4 Cross-constraints

- Choosing Sonata/Utonia (axis 2) weakens (c) on axis 1: geometry-native tokens
  no longer benefit from the vision tower's RGB priors; they enter more
  naturally via (a) or (b).
- The DepthStream's fate follows axis 1: with (a) or (b) consuming encoder
  tokens directly, the 36-block / 170M co-evolving stream is not required.
  Options: delete (encoder tokens read directly at every layer — layer
  specificity then comes only from the queries), shrink to a few shared blocks
  (e.g. 4 blocks at $d_d = 512$ ≈ 19M — inside budget), or keep per-layer
  co-evolution (no evidence for it: it has never actually trained). With (c)
  the stream is deleted by construction.

---

## 5. Decisions

1. Axis 1: **(a) joint softmax — DECIDED 2026-07-26.**
2. Axis 2: **from-scratch CNN — DECIDED 2026-07-26** (close-range wrist view
   only; room-scan pretraining offers the wrong prior; the encoder is now 9M
   with history fusion built in).
3. Axis 3: **wrist-RGB dropout — DECIDED 2026-07-26** ($p_{rgb}$ value TBD at
   implementation; FACTR curriculum and lr modulation stay in reserve), paired
   with the probe suite (§5.1).
4. Stream: **KEEP — DECIDED 2026-07-26** (user). Rationale: co-evolution mirrors
   how a normal transformer's features become more semantic with depth-in-stack,
   and the same is wanted for the depth representation; the hypothesis was never
   actually tested — under the α gate the stream's gradients were ×0.007 from
   step 0, so "it didn't help" was never observed, only "it never trained".
   Under the joint softmax its gradients now flow at full rate: this is the
   fair test. Caveat carried forward (R3D): 170M from-scratch capacity on small
   data is overfit-prone — the §5.1 held-out 2×2 probe doubles as the overfit
   detector; nothing here is definitive yet.
5. Soft-start bias $b_\ell$: **include, init $-2$ — DECIDED 2026-07-26.**
   Additive inside the softmax ⇒ shrinks initial depth mass (×$e^{-2}$ ≈ 0.135)
   without zeroing content gradients (unlike the multiplicative α); doubles as
   a per-layer learned "default admission" scalar with healthy dynamics.
6. Telemetry: **build with the redesign — DECIDED 2026-07-26** (§5.1).

### 5.2 Resulting design (all decisions in — BUILT 2026-07-26)

As-built notes: `join_depth_columns` + `depth_attention_mass` in
modeling_stream.py; `depth_bias` (36,) init −2 on the DepthStream (α +
`sink_logit` + `gated_depth_read` deleted); joint mask concat in the patched
cross-attn; RGB dropout in `MolmoAct2PackInputsProcessorStep` (zeroes the depth
camera's `<im_patch>` span in `attention_mask`, after `_build_labels`, training
text only) + `cross_on` bridge kill in `DepthStreamBlock` and the critic's
`compute_depth_tokens`; telemetry in the
trainer (`depth_attn_mass_mean`/`_max` first-micro-batch capture on log steps, `depth_bias_mean`,
`depth_grad_norm_preclip`); `probes/depth_modality_probe.py` replaces the
bit-identity probe. The b_ℓ-gradient-through-SDPA-mask assumption is covered by
`test_joint_softmax_matches_eager_and_bias_gets_gradient`. Old α-era checkpoints
no longer load (missing/unexpected keys) — they were the diagnosed-broken runs.

Per expert layer $\ell$, one softmax over context and depth columns:

$$o_i = \sum_{j \in ctx} w_{ij}\, v_j + \sum_{m \in depth} w_{im}\, v_{d,m}, \qquad
w = \mathrm{softmax}\Big(\big[\ q_i \cdot k_j / \sqrt{d_h}\ ;\ \ q_i \cdot k_{d,m} / \sqrt{d_h} + b_\ell\ \big]\Big)$$

with $K_{d,\ell} = \mathrm{knorm}(W^r_K D_\ell)$, $V_{d,\ell} = W^r_V D_\ell$,
$D_\ell$ = the kept DepthStream's state after block $\ell$ (encoder → 192
history-fused tokens → stream as today), $b_\ell$ init $-2$.
**Deleted**: $\alpha$, `sink_logit`, `gated_depth_read`, the additive patch,
`pointmap_gate*` telemetry, the gate-0 bit-identity probe.
**Added**: the $b_\ell$ vector (36 scalars), wrist-RGB dropout in the
processor, §5.1 probes. Training: wrist-RGB dropout $p_{rgb}$ (TBD ~0.1–0.25)
alongside the existing depth modality dropout 0.25; the shared history-dropout
draw is unchanged.

### 5.1 Probe suite (agreed 2026-07-26)

Always-on Aim scalars (training loop):

- `depth_attn_mass_mean`/`_max` — fraction of the joint softmax's mass on depth
  columns, averaged over queries and heads, then aggregated over stream layers;
  free in path (a); the working replacement for `pointmap_gate`. The per-layer
  `{ℓ}` breakdown is probe-only (one Aim panel per layer is unreadable).
- Pre-clip depth-group gradient norm (current metric reads post-clip, ~4×
  understated at the observed grad norms).

Periodic probe script (`probes/` pattern; retires `pointmap_bit_identity.py` —
there is no gate-0 to verify anymore):

- **2×2 modality probe**: one held-out batch under
  $\{$RGB+depth, RGB-only, depth-only, neither$\}$ — null bank and RGB dropout
  already implement all four conditions. Track
  $\mathcal L(\text{RGB}) - \mathcal L(\text{RGB+depth})$ over training: the
  direct "is depth earning its keep" curve, and the direct check that the
  dropout mechanism works.
- **Jacobian sensitivity**: $\|\partial \hat a / \partial E\|$ vs
  $\|\partial \hat a / \partial (\text{RGB features})\|$, a few JVP probes.
  Attention mass measures *admission*; the Jacobian measures *causal
  sensitivity* — the pair separates "attended but ignored" from load-bearing.

## 6. Sources

DepthVLA arXiv:2510.13375 · PointVLA arXiv:2503.07511 · QDepth-VLA
arXiv:2510.14836 · Lift3D-VLA arXiv:2607.06564 · R3D arXiv:2604.15281 · FuSe
arXiv:2501.04693 · FACTR arXiv:2502.17432 · DIPOLE arXiv:2511.22445 · Vanishing
Depth arXiv:2503.19947 · LLaMA-Adapter arXiv:2303.16199 · zero-init theory
arXiv:2502.03029 · Sonata arXiv:2503.16429 / Utonia (Pointcept, ICML'26) ·
unimodal bias arXiv:2312.00935 · MEM (PI) — video encoder precedent, 04_memory.md.
