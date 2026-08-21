# Trajectory Loss Design

Reformulation of the four auxiliary terms in
[`mse_for_imitation_learning.md`](mse_for_imitation_learning.md). 2026-08-18.

§§1–4 algebra, §5 what is broken, §6 replacements, §7 what to measure first,
§8 why comparable work does not show the same failures.

---

## 1. Notation

| symbol | meaning |
|---|---|
| $T=30$, $D$ | chunk horizon, action dimensions |
| $n$ / $k$ / $d$ | **time** index / **frequency** index / dimension |
| $a \in \mathbb R^{T\times D}$ | demonstrated chunk, anchor-encoded (hold chunk $=0$) |
| $x_t=(1-t)z+ta$, $z\sim\mathcal N(0,I)$ | what the policy sees |
| $v^\ast=a-z$, $\hat v$ | target velocity, prediction |
| $r=\hat v-v^\ast$ | **velocity residual** |
| $\hat a=x_t+(1-t)\hat v$, $e=\hat a-a$ | one-step reconstruction, its error |

Tilde = frequency domain. $C\in\mathbb R^{T\times T}$ is the **orthonormal DCT-II**,
$C_{k,n}=\alpha_k\cos\big(\pi k(n+\tfrac12)/T\big)$ with $\alpha_0=\sqrt{1/T}$,
$\alpha_{k>0}=\sqrt{2/T}$, applied down the time axis: $\tilde a=Ca$, $\tilde r=Cr$.
At 30 Hz with $T=30$, frequency $k$ is roughly $k/2$ Hz; $k=29$ is Nyquist.

---

## 2. The $(1-t)$ identity

Since $a=x_t+(1-t)v^\ast$ holds identically,

$$\boxed{\;e=\hat a-a=(1-t)\,r\;}\tag{2.1}$$

exactly — no approximation, no expectation. $(1-t)$ is the **remaining step
length**: from $x_t$ the solver covers the rest of the path in one Euler step, so
velocity error is amplified by exactly that length. At $t\to1$ the step is zero
and the reconstruction is correct regardless of what was predicted.

**2.1 The reconstruction is not an independent measurement.** For any quadratic
functional $Q$, $\;Q(e)=(1-t)^2Q(r)$. Path, shape and terminal are all quadratic,
so none can see anything the flow loss cannot; each is the flow loss under a
different quadratic form. "The clean chunk is free, so score it with something
MSE cannot see" holds only for **non-quadratic** criteria.

What the reconstruction does contribute is the **denominator**: it says the
natural scale is $\|a\|^2$, not the noise-dependent $\|a-z\|^2$. That is a claim
about normalization, not about a new measurement.

**2.2 The $t$-dependence is a gate schedule.** Every quadratic term carries
$(1-t)^2$, so against a fixed threshold every term switches off at large $t$
whatever the prediction quality — the gate is a noise-level schedule. Dividing by
$(1-t)^2$ removes this **exactly**, since (2.1) is an identity, not an estimate;
only float safety argues for clamping the divisor at $10^{-4}$. Afterwards the
three quadratic terms are not auxiliary losses at all but a **per-example
reweighting of the flow loss**.

**2.3 Direction is not independent either.** With $u=\hat a_{T-1,\cdot}$,
$w=a_{T-1,\cdot}$, so $u=w+(1-t)r_{T-1,\cdot}$, splitting $r$ into parallel and
perpendicular parts gives

$$1-\cos(u,w)\;\approx\;\frac{(1-t)^2\,\|r^\perp_{T-1}\|^2}{2\,\|a_{T-1}\|^2}.\tag{2.2}$$

In the small-error regime — where a trained policy lives — the direction term is
**half the terminal term with the radial component deleted**, carrying $(1-t)^2$
like everything else. It becomes distinct only when the terminal error is
comparable to the whole demonstrated displacement.

---

## 3. The design space, and the hazard in it

After §2 everything on offer is: choose a PSD quadratic form $M$, choose a
normalizer $\nu$, score $r^\top M r/\nu(a)$.

**The two axes are orthogonal, and each answers exactly one of the original
motivations.**

| axis | answers | scope of its effect |
|---|---|---|
| $M$ — the quadratic form | *MSE is invariant to shape deformation* | what is measured **within** an example |
| $\nu$ — the normalizer | *MSE is in absolute units, biased to large motions* | the relative weight **between** examples and dimensions |

This is worth stating because it is easy to expect more of $\nu$ than it can
deliver. $\nu$ is one scalar per (example, dimension); inside an example it is a
constant and **cancels out of every within-example comparison**. It cannot change
which shapes the loss can distinguish, cannot change the loss's dynamic range in
relative terms, and cannot change where the gradient points. Those are all
properties of $M$ alone. Conversely $M$ cannot equalize across examples.

So "the value is in the denominator" is right about *motivation 2* and only that.
Motivation 1 is entirely a question about $M$.

### 3.1 The hazard: normalizing amplifies whatever is smallest

Any normalizer estimated from the data amplifies the cases where its own estimate
is small — and small energy usually means noise-dominated. The same failure
appears at every granularity one might normalize at:

| normalize per… | amplifies | which are… |
|---|---|---|
| frequency band | the lowest-energy bands | the noise bands (§6.2) |
| example | barely-moving chunks | the lowest-SNR chunks |
| dimension | barely-moving joints | the lowest-SNR joints |

§5.4 is one instance of this seen through $\lambda_k$; it is not a special case.
The scale floors are the crude defence — they cap the amplification without
diagnosing it. The principled version is to divide by the **signal** power rather
than total measured power, i.e. by $\max(P - N,\ \phi)$ with $N$ the estimated
noise floor from §7. A chunk that is entirely noise then has denominator at the
floor and bounded weight, while a chunk with real motion is scaled by its real
motion.

**Add granularity to $\nu$ only where there is enough energy to estimate it.**
That single rule decides §6.2 below.

---

## 4. The frequency basis

**4.1 Parseval — the transform alone buys nothing.** $C$ is orthonormal, so
$\sum_{n,d}x_{n,d}^2=\sum_{k,d}\tilde x_{k,d}^2$. Writing the path error in DCT
coordinates gives the *same number*. A change of orthonormal basis is not a new
loss; all content is in the weighting applied afterwards.

**4.2 Shape is a frequency weight.** Let $\Delta$ be first differencing. Then
$\sum_n(\Delta x)_n^2=x^\top(\Delta^\top\Delta)x$, and $\Delta^\top\Delta$ is the
discrete Laplacian with reflecting (Neumann) boundaries. **The DCT-II basis
vectors are exactly its eigenvectors**, with eigenvalues

$$\lambda_k=2-2\cos(\pi k/T)=4\sin^2\!\big(\tfrac{\pi k}{2T}\big).\tag{4.2}$$

Therefore

$$\underbrace{\sum_{n,d}(\Delta x)^2}_{\text{shape}}=\sum_{k,d}\lambda_k\tilde x_{k,d}^2,
\qquad
\underbrace{\sum_{n,d}x^2}_{\text{path}}=\sum_{k,d}1\cdot\tilde x_{k,d}^2.\tag{4.3}$$

**Path and shape are the same object under two frequency weights** — flat, versus
$\lambda_k$ which is $0$ at DC and $\approx4$ at Nyquist. Shape's
offset-invariance is exactly $\lambda_0=0$. The FAST/discrete branch of this
codebase already computes shape this way; the continuous branch does the
time-domain version of the same thing.

**4.3 Worked example — the counterexample from the main note.** $T=4$, one
dimension, $x_A=[1,1,1,1]$ and $x_B=[1,-1,1,-1]$:

| | spectrum | path energy | shape energy |
|---|---|---|---|
| $x_A$ | $[2,\,0,\,0,\,0]$ — pure DC | 4 | $0$ |
| $x_B$ | $[0,\,0.765,\,0,\,1.848]$ — 85% in the top mode | 4 | $12$ |

with $\lambda=[0,\,0.586,\,2,\,3.414]$. Both shape energies match direct
computation ($\sum(\Delta x_B)^2=12$). The pair is not a lucky construction: it is
DC versus the top of the band, and plain MSE is spectrally flat. Any non-constant
$w_k$ separates them.

**4.4 What the axis means.** For $T=30$ at 30 Hz:

| band | freq | content |
|---|---|---|
| $k=0$ | DC | mean offset of the chunk over the anchor |
| $k=1\text{–}2$ | $\lesssim1$ Hz | the gross sweep — where the arm goes |
| $k=3\text{–}8$ | 1–4 Hz | the maneuver: approach, decelerate, grasp |
| $k>8$ | $>4$ Hz | jitter, quantization, sensor noise |

**4.5 Not multi-spacing.** Differencing at spacing $s$ gives
$w_k\approx4\sin^2(\pi ks/2T)$ — a **comb**, exactly zero at $k=2T/s,4T/s,\dots$.
Multi-spacing adds filters that are blind by construction at specific
frequencies. Bands have no nulls and their weights are what you wrote down.

---

## 5. What is broken

**5.1 Path cannot distinguish anything.** By §2.1 it is
$(1-t)^2\|r\|^2/\|a\|^2$ — the flow loss under a per-example weight. It can
reweight examples; it cannot detect a failure mode.

**5.2 Direction overlaps terminal.** By (2.2), two of the four terms measure
overlapping quantities with an uncontrolled relative weight.

### 5.3 The joint-space cosine has no metric — the gripper

A cosine requires all coordinates to live in a common inner-product space. They
do not:

1. **Different physical quantity.** Revolute joints share units with each other.
   An aperture command does not, and in practice it is a near-binary latch.
2. **Normalization hides this.** Quantile-normalizing to $[-1,1]$ silently fixes
   the metric to $\mathrm{diag}(1/\text{range}_d^2)$ — arbitrary, not physical.
3. **The real failure is bimodality, not scale.** Within a chunk the gripper
   either does not move (contributes $0$) or traverses its whole range
   (near-maximal coordinate). The term is a *mixture*: on switch chunks it is
   almost entirely a gripper sign detector, on the rest pure arm heading, with no
   controlled weight between two disjoint sub-populations.
4. **A near-binary coordinate has no direction, only a sign.** Folding it into a
   $D$-dimensional cosine dilutes the only question worth asking about it.

The generalization: **every commensurability problem here comes from summing
squares across dimensions before normalizing.** Path, shape and terminal have it
too in milder form — their denominators sum over $d$, so the largest-excursion
dimension sets the scale for all of them.

### 5.4 The shape term measures demonstration noise, not motion

The largest crack, in six steps.

**(1) The denominator.** With $h=0$ it is the demonstration's own per-step motion
power, $\sum_{n,d}(\Delta a)^2$, which by (4.3) is $\sum_{k,d}\lambda_k\tilde
a_{k,d}^2$ — the demo's power spectrum weighted by $\lambda_k$.

**(2) Where the demo's power actually is.** A chunk is 30 samples at 30 Hz —
**one second**. An arm reaching in one second is a single smooth arc: energy at
$k=0,1,2$, maybe out to $k=4$. Energy at $k=15$ would mean the joint oscillating
fifteen times in that second. Leader-arm tremor, encoder quantization and timing
jitter are approximately white — flat across all $k$, dominant only where there
is no signal to hide behind.

**(3) What $\lambda_k$ does to each.**

| $k$ | 0 | 1 | 2 | 4 | 8 | 15 | 29 |
|---|---|---|---|---|---|---|---|
| $\lambda_k$ | 0 | 0.011 | 0.044 | 0.173 | 0.662 | 2.00 | 3.99 |

Signal sits where $\lambda_k\approx0.01\text{–}0.17$; noise sits everywhere,
including where $\lambda_k\approx2\text{–}4$. **Differencing attenuates the signal
by ~$100\times$ and amplifies the noise by ~$4\times$.**

**(4) The arithmetic.** Split $\tilde a_k^2=S_k+N$:

$$\text{denom}_\text{path}=\textstyle\sum_kS_k+TN,
\qquad
\text{denom}_\text{shape}=\textstyle\sum_k\lambda_kS_k+N\sum_k\lambda_k.$$

The noise term barely grows ($\sum_k\lambda_k=2T-2=58$ against $T=30$), but the
signal term **collapses**, since $S_k$ sits exactly where $\lambda_k\approx0.01$.
With demo power $0.5,0.3,0.12,0.05,0.03$ in bins $k=0..4$ and a flat floor:

| per-bin SNR | noise share of **path** denom | of **shape** denom |
|---|---|---|
| 40 dB | 0.3% | **24%** |
| 30 dB | 2.9% | **76%** |
| 20 dB | 23% | **97%** |

**(5) The consequence — no dynamic range.** A trained policy emits smooth
trajectories and does not reproduce tremor, so the jitter enters numerator *and*
denominator at the same magnitude and largely cancels. At 30 dB:

| low-band relative error | 0% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| $R_\text{shape}$ | 0.757 | 0.760 | 0.767 | 0.779 | 0.818 |

**A perfect prediction and a 30%-wrong one differ by 0.02.** The term is pinned
near a constant set by the noise floor.

**(6) And the gradient points the wrong way.** From $0.757$, fixing the motion
entirely buys $0.022$; *reproducing the demonstration's jitter* buys the full
$0.757$, since that is what empties the numerator. **There is ~$34\times$ more
loss available from learning to chatter than from getting the motion right.**

The scale floor does not save this — it caps near-zero denominators, and this one
is a healthy number made of the wrong thing.

**And neither does fixing the denominator.** By §3 the denominator is a constant
within an example, so it cancels from both numbers above. Replacing the
$\lambda$-weighted denominator with the demonstration's total power moves
$R_\text{shape}$ from $0.757$ to $0.056$ but leaves the relative dynamic range at
$8.0\%$ and the jitter-to-motion gradient ratio at $34.6\times$ — **identical to
three significant figures.** §5.4 is a statement about $\lambda_k$ in the
**numerator**, and only a change of numerator can fix it.

**Two checks, no new model.** *Offline:* measure $P_{k,d}=\langle\tilde
a_{k,d}^2\rangle$ per dimension, find the plateau, substitute into (4)–(5) to turn
the tables into measurements. *From existing logs:* the prediction is that
$R_\text{shape}$ sits near a constant, barely moves while the flow loss falls,
and its $0.2$ gate fires on ~100% of samples. Both the mean and the active
fraction are already logged — if that signature is there, (1)–(6) are confirmed.

**5.5 Terminal is rank-1.** It evaluates one time step, so in frequency it is
maximally spread and cannot be a band; its denominator is a single time slice,
the noisiest of the three. It also presumes the whole chunk executes — under
real-time chunking with early replanning the tail never runs.

---

## 6. Proposed reformulation

Independent changes, but 6.1 is a precondition for the rest being meaningful.

### 6.1 Normalize per dimension, always

For any quadratic form $M$,

$$R_d=\frac{r_{\cdot,d}^\top M\,r_{\cdot,d}}{a_{\cdot,d}^\top M\,a_{\cdot,d}\;\vee\;\phi_{M,d}},
\qquad R=\frac{\sum_d\omega_dR_d}{\sum_d\omega_d},\tag{6.1}$$

$\vee$ being a floor. **Scope**, since this is easy to get wrong:

| quantity | computed over | count |
|---|---|---|
| numerator, denominator | **this chunk's own time axis, dimension $d$ alone** | one per (example, dim) |
| floor $\phi_{M,d}$, activity median $m_{M,d}$ | the whole corpus, offline, once | one per dim — a dataset constant |

So yes — the denominator is per chunk, per dimension, over that chunk's own $T$
steps. Nothing is summed across dimensions until after the division. §6.2 narrows
once more, to (example, dim, band).

Each $R_d$ is invariant to dimension $d$'s units by construction. **This removes
the gripper problem structurally** — no dimension names, no per-robot constants,
no kinematic model. A ratio of two quantities in the same units is the only
dimensionless thing available.

Aggregation weight $\omega_d$: either **uniform over active dimensions**
($\omega_d=\mathbb 1[\,a_{\cdot,d}^\top Ma_{\cdot,d}>\phi_{M,d}\,]$), or
**relative activity** ($\omega_d=a_{\cdot,d}^\top Ma_{\cdot,d}/m_{M,d}$, "how much
did this dimension move relative to how much it usually moves"). Prefer relative
activity; keep uniform as the ablation.

### 6.2 Bands replace path and shape — with a *shared* denominator

Partition frequency into contiguous bands $\mathcal B_b$ and define, per dimension,

$$R_{b,d}=\frac{\sum_{k\in\mathcal B_b}\tilde r_{k,d}^2}
{\underbrace{\|a_{\cdot,d}\|^2\;\vee\;\phi_d}_{\text{one denominator, all bands}}}.\tag{6.2}$$

**The denominator is per (example, dimension) and shared across bands — it is
*not* per band.** This is the correction that §3.1 forces, and it is worth being
explicit about because per-band normalization is the tempting version.

*Why not per band.* Normalizing each band by its own power sets every band to unit
energy, so each contributes equally regardless of how much signal it carries. With
the partition below and a 30 dB floor:

| band | share of the demonstration's energy | share of signal *within* the band |
|---|---|---|
| $\{0\}$ DC | 48.6% | 99.8% |
| $\{1..2\}$ sweep | 41.0% | 99.5% |
| $\{3..8\}$ maneuver | 8.3% | 93.0% |
| $\{9..29\}$ noise | **2.0%** | **0.0%** |

Under uniform weights over four per-band ratios the noise band receives 25% of the
weight against 2.0% of the energy — a **$12.3\times$ amplification of the band that
is definitionally all noise**. That is §5.4 re-inflicted deliberately. A shared
denominator gives each band its actual energy share instead.

Per-band normalization is defensible only as a *deliberate* statement that every
timescale matters equally regardless of energy. It should then be chosen, not
inherited, and $w_b$ set knowing the effective weight is $w_b/P_b$.

*Properties of (6.2).* By Parseval $\sum_b R_{b,d}=\|r_{\cdot,d}\|^2/\|a_{\cdot,d}\|^2$
exactly — the per-dimension path NMSE. So uniform $w_b$ with no gate **recovers the
path term identically**, and the flat baseline sits inside the parameterization
rather than being a separate experiment.

*Band edges come from the measured spectrum, not taste* — put them where $P_{k,d}$
changes regime. The partition above is a starting point for $T=30$, to be
re-derived from §7.

*Drop the noise band.* This is the actual fix for §5.4, since by §3 only the
numerator can be. Selecting signal bands and discarding $\lambda_k$:

| numerator | denominator | relative dynamic range | gradient ratio |
|---|---|---|---|
| $\lambda_k$-weighted (current shape) | $\lambda_k$-weighted | 8.0% | $34.6\times$ **toward jitter** |
| $\lambda_k$-weighted | total power | 8.0% | $34.6\times$ **toward jitter** |
| bands $k\le8$, noise band excluded | total power | **2778%** | $0.1\times$ — i.e. $10\times$ **toward motion** |

Changing the denominator does nothing. Changing the numerator moves the dynamic
range by a factor of ~350 and flips the sign of what the gradient rewards.

The softer alternative to hard exclusion is the Wiener weight $w_b=P_b/(P_b+N)$,
which decays to zero where the demonstration carries no signal. Hard exclusion is
its limit; which is better is an experiment, not a derivation.

### 6.3 An exact gain/shape split replaces direction

Per dimension, over the whole chunk, project the prediction onto the
demonstration: $\gamma_d=\langle\hat a_{\cdot,d},a_{\cdot,d}\rangle/\|a_{\cdot,d}\|^2$,
$\;\hat a_{\cdot,d}=\gamma_da_{\cdot,d}+\hat a^\perp_{\cdot,d}$. Pythagoras gives an
**exact, additive** decomposition of the per-dimension NMSE:

$$\frac{\|\hat a_{\cdot,d}-a_{\cdot,d}\|^2}{\|a_{\cdot,d}\|^2}
=\underbrace{(\gamma_d-1)^2}_{G_d\ \text{gain}}
+\underbrace{\frac{\|\hat a^\perp_{\cdot,d}\|^2}{\|a_{\cdot,d}\|^2}}_{S_d\ \text{shape}}.\tag{6.3}$$

Why this beats $1-\cos$:

- **Exact and orthogonal**, not a small-angle approximation — so unlike §5.2
  there is no hidden overlap with another term.
- **Dimensionless per dimension**, so §6.1 applies and the gripper problem does
  not arise. For a latch, $\gamma_d$ is "did you open when the demo opened, and by
  how much" and $S_d$ is "did you switch at the right time" — sign and timing,
  which is all a latch has.
- **Sign errors fall out of the geometry.** $\gamma_d=0$ (did not move) gives
  $G_d=1$; $\gamma_d=2$ (doubled) gives $1$; $\gamma_d=-1$ (backwards) gives $4$.
  No $-\log((\cos+1)/2)$ construction needed.
- **Uses the whole chunk**, so it does not inherit terminal's execution problem.

**Caveat.** Substituting $\hat a=a+(1-t)r$ gives
$G_d=(1-t)^2\langle r_{\cdot,d},a_{\cdot,d}\rangle^2/\|a_{\cdot,d}\|^4$ and
$S_d=(1-t)^2\|r^\perp_{\cdot,d}\|^2/\|a_{\cdot,d}\|^2$ — **both quadratic in $r$**,
a rank-one projector and its complement. This does *not* escape §2.1; its virtues
are exactness, orthogonality and unit-freeness, not new information. It is also
the one place $1-\cos$ was doing something real: as a *ratio* of quadratics it is
genuinely non-quadratic, though (2.2) shows it linearizes in exactly the regime a
trained policy occupies. The scale-invariant form
$\sin^2\theta_d=S_d/(\gamma_d^2+S_d)$ keeps that property and costs nothing to log.

**Phase extension.** $\langle\hat a_{\cdot,d},a_{\cdot,d}\rangle$ is the zero-lag
cross-correlation; recomputing at lags $\tau\in\{-2..2\}$ splits shape error into
*shape* and *timing*. Under real-time chunking, "right motion, two frames late" is
a real failure none of the current terms detect. Diagnostic first — as a loss it
needs a penalty on $|\tau^\ast|$ or it rewards lateness.

**6.4 Divide out $(1-t)^2$.** Exact by §2.2. Clamp the divisor at $10^{-4}$.
Without it no threshold is interpretable, because the same prediction quality
crosses the gate or not depending only on which $t$ was sampled.

**6.5 Terminal — unresolved.** Rank-1, noisiest denominator, relevance depends on
the execution regime (§5.5). The principled generalization is to weight time steps
by probability of execution, making terminal the special case "the whole chunk
always runs" — but that couples the loss to the deployment config, a genuine cost.
Options in order of conservatism: keep it with a per-dimension denominator; replace
with an execution-weighted path; drop it and rely on the DC and low bands.

---

## 7. Measure before choosing

Three measurements, all cheap, none needing a model.

1. **Corpus power spectrum $P_{k,d}$ per dimension.** Sets the band edges, exposes
   the noise floor, and settles §5.4 quantitatively. Nothing else should be
   decided first.
2. **Per-dimension excursion distributions $\|a_{\cdot,d}\|^2$.** Gives the
   activity medians and floors of §6.1, and shows how far apart the gripper and
   the arm joints actually are in the normalized space now in use.
3. **Batch distributions of each proposed ratio.** Ratio families have heavy right
   tails and the aggregate is set by the smallest denominators. Log the histogram;
   aggregate with $\mathrm{mean}\log R$ or a median, not an arithmetic mean.

---

## 8. Why the literature does not show these cracks

These terms follow lines the literature already uses, so if they crack, why does
nobody report it? Not because everyone missed it. Sorting by *why* comparable work
is immune says where the frontier actually is.

*Confidence: the structural arguments (what is or is not a quadratic form) are
certain. Claims about a given paper's benchmark or action space are hypotheses
about why, not verified facts. Paper details rest on
[`..._related_work.md`](mse_for_imitation_learning_related_work.md).*

**8.1 Avoided by paying compute (§2.1).** **TCFM** reaches $\hat x_1$ by a 5-step
Euler rollout; each step evaluates the network at a point depending on the
previous output, so $\hat x_1$ is a nonlinear composition and $e=(1-t)r$ **does not
hold** — it is exactly the $S=1$ special case. So their term genuinely measures
how velocity error *compounds through the solver*. **LPL** stays at one step but
makes the criterion nonlinear ($\varphi$ = frozen VAE decoder); had $\varphi$ been
linear and orthonormal, Parseval would make the term identical to the base loss.

This inverts the framing in the related-work note. The free one-step
reconstruction was presented as an advantage over TCFM's expensive rollout. It is
better read as a **trade: the rollout is expensive precisely because the expense is
what buys the information.** Escaping §2.1 requires multi-step composition or a
nonlinear criterion. Nothing in §6 has either — so every term in §6 is a
reweighting of the flow loss. That is a legitimate contribution (Min-SNR and EDM2
are exactly that) but must be claimed as a weighting scheme, not a new measurement.

**8.2 Created by the ingredients that are not borrowed.** The $(1-t)^2$ gate
schedule needs a *hard threshold*; TCFM, OMP, REPA and LPL all use fixed weights
and no gating, and without a threshold the $t$-dependence is absorbed into the
effective weight. Likewise nothing comparable divides by a per-example,
data-derived denominator — TCFM's $\mathcal L_\text{vel}$ and DILATE's shape term
are unnormalized; Min-SNR and EDM2 normalize along the *noise* axis, never by the
example's own content. There is no denominator for a noise floor to contaminate.
They have the opposite problem: absolute units, which is §2 of the main note.
**The per-example denominator is the real idea here, and it is where both
remaining cracks live** — which is what an unexamined novel component looks like
before it is stress-tested. The borrowed parts were debugged by other people.

**8.3 Inherited by crossing domains.** Gradient-matching losses come from depth
and images, where high-frequency content is *real signal* — edges, boundaries.
Transplanted onto a smooth one-second joint trajectory at 30 Hz, the same operator
lands where the high band is all noise. The technique did not break; the domain
changed. Same for FDE, which comes from open-loop forecasting where the full
horizon is the prediction and nothing is executed. Their techniques transfer;
their unstated assumptions about **where the signal lives** do not.

**8.4 One crack is genuinely shared.** **REPA** takes cosines between
*representation* vectors, homogeneous by construction — no commensurability
problem. **OMP** takes the cosine between *velocity vectors in the action space*;
if that space mixes revolute joints with a gripper aperture, it has §5.3 in
exactly the described form. It likely goes unremarked because manipulation
benchmarks present uniformly normalized action spaces and simulated grippers are
well-scaled continuous dimensions rather than physical latches. On §5.3 the fix in
§6.1/§6.3 is ahead of the literature, for the mundane reason that the hardware
forces the issue.

**8.5 Devil's advocate: what is actually differentiated?** If the frequency
decomposition is the visible machinery, most of it is prior art. Per the
related-work note, **FocalPolicy** already pairs a time-domain chunk loss with an
orthonormal-DCT spectral loss and already argues that low coefficients carry
global motion and high ones carry execution detail. So band-splitting a
trajectory loss by frequency is not new, and should not be claimed as such.

Subtracting that, what is left:

| component | status |
|---|---|
| DCT band decomposition of the loss | **prior art** (FocalPolicy) |
| shape $=$ $\lambda_k$ weighting, so path/shape/multi-spacing are one family | analysis, not a new loss — but it is what exposes §5.4 |
| per-**example** denominator | no precedent found in training objectives |
| per-**dimension** denominator | evaluation-side precedent (Per-Group Error); none as a training denominator |
| band edges set by the measured noise floor | follows from §3.1; not seen elsewhere |

So the honest reading is the one the frequency analysis itself implies: **the
bands are not the contribution — they are the diagnostic that reveals the problem
and the coordinate system in which the fix is expressible.** The contribution, if
there is one, is the normalization structure: per example $\times$ per dimension,
against signal power rather than measured power.

That is narrow. It is also the only part that survives §2.1, since by §3 the
normalizer is the one axis that changes the relative weight of examples at all.

| crack | how comparable work avoids it |
|---|---|
| §2.1 quadratic terms collapse | pay for it: multi-step rollout (TCFM) or nonlinear criterion (LPL) |
| §2.2 $(1-t)^2$ gate schedule | no gating — fixed weights only |
| §5.4 denominator is noise | no per-example denominator at all |
| §5.4 numerator chases jitter | their domain has real high-frequency signal |
| §5.5 terminal / execution | open-loop; nothing is "executed" |
| §5.3 joint-space cosine | **OMP shares it** — probably masked by benchmark action spaces |

The cracks concentrate in the per-example denominator and the gate: the two
components that are not borrowed.

---

## 9. Summary

| current | becomes | reason |
|---|---|---|
| path | DC + low bands, per dim | §5.1: a weight on the flow loss, not a term |
| shape | high bands, per dim, edges from the measured spectrum | §4.2 it is the $\lambda_k$ weight; §5.4 ~76% jitter on both sides, <3% dynamic range, pays $34\times$ more for chatter than for correct motion |
| terminal | unresolved — §6.5 | §5.5: rank-1, noisy denominator, execution-dependent |
| direction | $G_d$ gain + $S_d$ shape, per dim | §5.2 overlaps terminal; §5.3 the cosine has no metric |
| $\times(1-t)^2$ | divided out | §2.2: it makes the gate a noise schedule |
| $\sum_d$ then divide | divide then $\sum_d$ | §5.3: the source of every unit problem |

**Priority.** §5.4 first — it is not a matter of degree but a near-inert term whose
gradient points at reproducing teleop noise. The fix is in the **numerator**
(select signal bands, drop $\lambda_k$, exclude the noise band); by §3 no
denominator change can touch it. §6.1 is the cheapest change and fixes the widest
class of problem, but it addresses motivation 2 only. Everything else waits on §7.

**Do not conflate the two axes.** $M$ answers "MSE ignores shape"; $\nu$ answers
"MSE is in absolute units." Per-band denominators try to make $\nu$ do both and
end up amplifying the noise band $12.3\times$ (§6.2).

**One line:** the four terms are four quadratic forms on the same velocity
residual, three redundant with each other, all normalized across incommensurable
dimensions. Normalize per dimension, choose the quadratic forms in the basis where
they are diagonal, and replace the cosine with the exact projection it approximates.

**The honest limit (§8.1):** with a one-step reconstruction and quadratic criteria,
none of this can measure anything the flow loss cannot. It can only weight it
better. Going further needs multi-step composition or a nonlinear criterion —
which is what the compute in TCFM and LPL buys.
