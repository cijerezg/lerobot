# Action Trajectory Losses

Single compiled note. Supersedes and absorbs `mse_for_imitation_learning.md`,
`mse_for_imitation_learning_related_work.md`, and `trajectory_loss_design.md`.
Implementation reference for what is currently wired up stays in
[`05_training.md`](05_training.md) §2.1–2.2.

- **Part I** (§1–3) the problem and what is currently implemented
- **Part II** (§4–7) analysis: what the terms actually are, and what is broken
- **Part III** (§8–11) the option catalogue and a recommended set
- **Part IV** (§12–14) relation to FAST, to the literature, and what to measure

---

# Part I — Problem and current state

## 1. Notation

| symbol | meaning |
|---|---|
| $T=30$, $D$ | chunk horizon, action dimensions |
| $n$ / $k$ / $d$ | **time** index / **frequency** index / dimension |
| $F=8$ | flow times sampled per example per step |
| $a\in\mathbb R^{T\times D}$ | demonstrated chunk, anchor-encoded (so the hold chunk is $0$) |
| $x_t=(1-t)z+ta$, $z\sim\mathcal N(0,I)$ | what the policy sees |
| $v^\ast=a-z$, $\hat v$ | target velocity, prediction |
| $r=\hat v-v^\ast$ | **velocity residual** |
| $\hat a=x_t+(1-t)\hat v$, $e=\hat a-a$ | one-step reconstruction, its error |
| $w_{k,d}$ | weight applied to frequency $k$ of dimension $d$ (§8) |
| $\bar P_{k,d}$ | **corpus-mean measured power** of the demonstration at frequency $k$, dimension $d$ |
| $N_d$ | **noise floor** of dimension $d$ — flat in $k$, hence no $k$ index; the high-$k$ plateau of $\bar P_{k,d}$ |
| $S_{k,d}=\bar P_{k,d}-N_d$ | the **signal** part of that power |

Tilde = frequency domain. $C\in\mathbb R^{T\times T}$ is the **orthonormal DCT-II**,
$C_{k,n}=\alpha_k\cos\big(\pi k(n+\tfrac12)/T\big)$, $\alpha_0=\sqrt{1/T}$,
$\alpha_{k>0}=\sqrt{2/T}$, applied down the time axis: $\tilde a=Ca$, $\tilde r=Cr$.
At 30 Hz with $T=30$, frequency $k$ is about $k/2$ Hz and $k=29$ is Nyquist.

## 2. The two complaints about MSE

**M1 — equal loss does not imply equal trajectory.** For a one-dimensional chunk
of four steps, the residuals $r_A=[1,1,1,1]$ and $r_B=[1,-1,1,-1]$ have equal mean
square. Since $\hat a-a=(1-t)(\hat v-v^\ast)$, both reach the reconstruction
unchanged in form, so the two score identically while differing in shape — one a
constant offset, the other alternating. On adjacent differences they separate: $0$
against $4$. The flow objective also gives the final step no distinct role and
does not measure direction of motion.

**M2 — the objective is in absolute units.** Error is in squared action units, so
for the same *fractional* mistake a large-displacement demonstration produces a
larger gradient than a small one. A correction of amplitude $0.01$ predicted with
the wrong sign still incurs small absolute error. The objective reports how many
normalized action units the prediction was wrong by, not how wrong it was relative
to the motion the example required.

These are the two things every proposal below is trying to fix, and §5 shows they
require **different machinery**.

## 3. What is currently implemented

The flow loss is kept and four terms are added on the reconstruction, each
evaluated at every sampled flow time and averaged per example. $h$ is the
hold-still chunk ($h=0$ in anchor space), and each term divides the prediction's
error by the error the same demonstration assigns to $h$:

$$
R_{\mathrm{path}}=\frac{\overline{(\hat a-a)^2}}{\overline{(h-a)^2}},\quad
R_{\mathrm{shape}}=\frac{\overline{(\Delta\hat a-\Delta a)^2}}{\overline{(\Delta h-\Delta a)^2}},\quad
R_{\mathrm{terminal}}=\frac{\overline{(\hat a_T-a_T)^2}}{\overline{(h_T-a_T)^2}},
$$

$$
R_{\mathrm{direction}}=1-\cos\big(\hat a_T-h_T,\;a_T-h_T\big),
$$

combined as $L=L_{\mathrm{flow}}+\sum_j w_jG_{g_j}(R_j)$ with a **hard** gate
$G_g(R)=R\cdot\mathbb 1[R>g]$, all $w_j=0.05$, all $g_j=0.2$. Denominators are
floored by corpus medians (`DEFAULT_SCALE_FLOORS`).

Reading: $0$ exact, $1$ no better than freezing the arm, $>1$ worse than freezing.

A parallel set exists on the discrete/FAST branch (`ordinal`, `path`, `shape`),
computed in DCT space directly from the teacher-forced logits. §12 covers how that
relates.

---

# Part II — Analysis

## 4. The $(1-t)$ identity

Since $a=x_t+(1-t)v^\ast$ holds identically,

$$\boxed{\;e=\hat a-a=(1-t)\,r\;}\tag{4.1}$$

exactly — no approximation, no expectation. $(1-t)$ is the **remaining step
length**: from $x_t$ the solver covers the rest of the path in one Euler step, so
velocity error is amplified by exactly that length. At $t\to1$ the step is zero and
the reconstruction is correct regardless of what was predicted.

**4.1 The reconstruction is not an independent measurement.** For any quadratic
functional $Q$, $Q(e)=(1-t)^2Q(r)$. Path, shape and terminal are all quadratic, so
none can see anything the flow loss cannot; each is the flow loss under a different
quadratic form. "The clean chunk is free, so score it with something MSE cannot
see" holds only for **non-quadratic** criteria (§10).

What the reconstruction does contribute is the **denominator**: it says the natural
scale is $\|a\|^2$, not the noise-dependent $\|a-z\|^2$. A claim about
normalization, not a new measurement.

**4.2 The $t$-dependence is a gate schedule.** Every quadratic term carries
$(1-t)^2$, so against a fixed threshold every term switches off at large $t$
whatever the prediction quality — the gate is a noise-level schedule. Dividing by
$(1-t)^2$ removes this **exactly**, since (4.1) is an identity, not an estimate;
only float safety argues for clamping the divisor at $10^{-4}$. Afterwards the
three quadratic terms are not auxiliary losses at all but a **per-example
reweighting of the flow loss**.

**4.3 Direction is not independent either.** With $u=\hat a_{T-1,\cdot}$,
$w=a_{T-1,\cdot}$, splitting $r$ parallel and perpendicular to $w$:

$$1-\cos(u,w)\;\approx\;\frac{(1-t)^2\|r^\perp_{T-1}\|^2}{2\|a_{T-1}\|^2}.\tag{4.2}$$

In the small-error regime — where a trained policy lives — the direction term is
**half the terminal term with the radial component deleted**, carrying $(1-t)^2$
like everything else.

## 5. The design space, and the hazard in it

After §4 everything on offer is: choose a PSD quadratic form $M$, choose a
normalizer $\nu$, score $r^\top Mr/\nu(a)$.

**The two axes are orthogonal, and each answers exactly one complaint.**

| axis | answers | scope of effect |
|---|---|---|
| $M$ — the quadratic form | **M1**, shape invariance | what is measured **within** an example |
| $\nu$ — the normalizer | **M2**, absolute units | relative weight **between** examples and dimensions |

$\nu$ is one scalar per (example, dimension); inside an example it is a constant
and **cancels from every within-example comparison**. It cannot change which
shapes the loss distinguishes, cannot change relative dynamic range, cannot change
where the gradient points. Those are all $M$. Conversely $M$ cannot equalize across
examples. So "the value is in the denominator" is right about **M2 and only M2**.

### 5.1 The hazard: normalizing amplifies whatever is smallest

Any normalizer estimated from data amplifies the cases where its own estimate is
small — and small energy usually means noise-dominated:

| normalize per… | amplifies | which are… |
|---|---|---|
| frequency band | lowest-energy bands | the noise bands |
| example | barely-moving chunks | the lowest-SNR chunks |
| dimension | barely-moving joints | the lowest-SNR joints |

§7.4 is one instance of this seen through $\lambda_k$, not a special case. Scale
floors are the crude defence — they cap amplification without diagnosing it. The
principled version divides by the **signal** power, $\max(P-N,\phi)$, with $N$ the
noise floor from §14. **Add granularity to $\nu$ only where there is enough energy
to estimate it.**

## 6. The frequency basis

**6.1 Parseval — the transform alone buys nothing.** $C$ is orthonormal, so
$\sum_{n,d}x^2=\sum_{k,d}\tilde x^2$. Writing the path error in DCT coordinates
gives the *same number*. A change of orthonormal basis is not a new loss; all
content is in the weighting applied afterwards.

**6.2 Shape is a frequency weight.** $\sum_n(\Delta x)_n^2=x^\top(\Delta^\top\Delta)x$,
and $\Delta^\top\Delta$ is the discrete Laplacian with reflecting (Neumann)
boundaries. **The DCT-II basis vectors are exactly its eigenvectors**, with

$$\lambda_k=2-2\cos(\pi k/T)=4\sin^2\!\big(\tfrac{\pi k}{2T}\big).\tag{6.1}$$

Therefore

$$\underbrace{\sum(\Delta x)^2}_{\text{shape}}=\sum_{k,d}\lambda_k\tilde x_{k,d}^2,
\qquad
\underbrace{\sum x^2}_{\text{path}}=\sum_{k,d}1\cdot\tilde x_{k,d}^2.\tag{6.2}$$

**Path and shape are the same object under two frequency weights** — flat, versus
$\lambda_k$ which is $0$ at DC and $\approx4$ at Nyquist. Shape's offset-invariance
is exactly $\lambda_0=0$.

**6.3 Worked example — the M1 counterexample.** $T=4$, one dimension:

| | spectrum | path energy | shape energy |
|---|---|---|---|
| $x_A=[1,1,1,1]$ | $[2,0,0,0]$ — pure DC | 4 | $0$ |
| $x_B=[1,-1,1,-1]$ | $[0,\,0.765,\,0,\,1.848]$ — 85% top mode | 4 | $12$ |

with $\lambda=[0,0.586,2,3.414]$; both shape energies match direct computation.
The pair is not a lucky construction — it is DC versus the top of the band, and
**plain MSE is spectrally flat**. Any non-constant $w_k$ separates them.

**6.4 What the axis means — and it differs by dimension.** For a *smooth* joint,
$T=30$ at 30 Hz:

| band | freq | content |
|---|---|---|
| $k=0$ | DC | mean offset of the chunk over the anchor |
| $k=1\text{–}2$ | $\lesssim1$ Hz | the gross sweep — where the arm goes |
| $k=3\text{–}8$ | 1–4 Hz | the maneuver: approach, decelerate, grasp |
| $k>8$ | $>4$ Hz | jitter, quantization, sensor noise |

**This table is wrong for the gripper.** A gripper does not trace an arc; it
switches — closed to open, open to closed. A switch is a **step edge**, and an edge
is broadband: the high band carries the *timing of the switch*, which is real signal
and is the most task-relevant thing in the chunk. Measured at a common absolute
noise floor:

| dimension | high-band ($k\ge9$) SNR | signal share of the shape denominator |
|---|---|---|
| gripper, 1–2 sample switch | **315$\times$** | 99.7% |
| gripper, 4 sample switch | **63$\times$** | 99.4% |
| smooth arm arc | 0.03$\times$ | 40% |

So the noise floor, and therefore where the bands should be cut, is a
**per-dimension** quantity. This is the same conclusion as §7.3 arriving by a
different route: it is not only the *units* that differ between a revolute joint and
a latch, it is the *shape of the spectrum*. Any global band edge is wrong for one of
them.

## 7. What is broken

**7.1 Path cannot distinguish anything.** By §4.1 it is $(1-t)^2\|r\|^2/\|a\|^2$ —
the flow loss under a per-example weight. It reweights examples; it cannot detect a
failure mode.

**7.2 Direction overlaps terminal.** By (4.2) two of four terms measure
overlapping quantities with an uncontrolled relative weight.

### 7.3 The joint-space cosine has no metric — the gripper

A cosine requires all coordinates in a common inner-product space. They are not:

1. **Different physical quantity.** Revolute joints share units with each other; an
   aperture command does not, and in practice it is a near-binary latch.
2. **Normalization hides this.** Quantile-normalizing to $[-1,1]$ silently fixes
   the metric to $\mathrm{diag}(1/\text{range}_d^2)$ — arbitrary, not physical.
3. **The real failure is bimodality, not scale.** Within a chunk the gripper either
   does not move (contributes $0$) or traverses its whole range. The term is a
   *mixture*: on switch chunks almost entirely a gripper sign detector, on the rest
   pure arm heading, with no controlled weight between two disjoint sub-populations.
4. **A near-binary coordinate has no direction, only a sign.**

Generalization: **every commensurability problem here comes from summing squares
across dimensions before normalizing.** Path, shape and terminal have it too in
milder form.

### 7.4 The shape term measures demonstration noise, not motion

**(1) The denominator.** With $h=0$ it is $\sum(\Delta a)^2$, which by (6.2) is
$\sum_{k,d}\lambda_k\tilde a_{k,d}^2$ — the demo's power spectrum weighted by
$\lambda_k$.

**(2) Where the demo's power is.** A chunk is 30 samples at 30 Hz — **one second**.
An arm reaching in one second is a single smooth arc: energy at $k=0,1,2$, maybe to
$k=4$. Energy at $k=15$ would mean the joint oscillating fifteen times in that
second. Tremor, quantization and timing jitter are approximately white — flat across
all $k$, dominant only where no signal hides them.

**(3) Where the $\lambda_k$ numbers come from.** From (6.1),
$\lambda_k=4\sin^2(\pi k/2T)$. For small $k$, $\sin x\approx x$, so

$$\lambda_k\;\approx\;4\Big(\frac{\pi k}{2T}\Big)^2=\Big(\frac{\pi k}{T}\Big)^2 .$$

That is the whole story: **differencing is differentiation, differentiating
multiplies amplitude by frequency, so it multiplies power by frequency squared.**
Low frequencies get crushed quadratically; high frequencies saturate near 4.
Evaluated at $T=30$:

| $k$ | $\pi k/2T$ | $\sin$ | $\lambda_k=4\sin^2$ | small-angle $(\pi k/T)^2$ |
|---|---|---|---|---|
| 1 | 0.05236 | 0.05234 | **0.011** | 0.011 |
| 2 | 0.10472 | 0.10453 | **0.044** | 0.044 |
| 4 | 0.20944 | 0.20791 | **0.173** | 0.175 |
| 8 | 0.41888 | 0.40674 | **0.662** | 0.702 |
| 15 | 0.78540 | 0.70711 | **2.000** | 2.467 |
| 29 | 1.51844 | 0.99863 | **3.989** | 9.223 |

Signal sits where $\lambda_k\approx0.01\text{–}0.17$; noise sits everywhere,
including where $\lambda_k\approx2\text{–}4$. **Differencing attenuates a smooth
signal by ~$100\times$ and amplifies broadband noise by ~$4\times$.**

**(4) The arithmetic.** Model the demonstration's *measured* spectrum at frequency
$k$ as

$$\tilde a_k^2 \;=\; \underbrace{S_k}_{\text{real motion}} \;+\; \underbrace{N}_{\text{noise floor}},$$

where $N$ is flat because tremor, encoder quantization and timing jitter are
broadband — approximately the same power in every bin — while $S_k$ is concentrated
at low $k$ by step (2).

Write $d_\text{path}$ and $d_\text{shape}$ for the denominators of $R_\text{path}$
and $R_\text{shape}$ from §3. Substituting:

$$d_\text{path}=\sum_k\tilde a_k^2=\underbrace{\textstyle\sum_kS_k}_{\text{signal}}+\underbrace{TN}_{\text{noise}},
\qquad
d_\text{shape}=\sum_k\lambda_k\tilde a_k^2=\underbrace{\textstyle\sum_k\lambda_kS_k}_{\text{signal}}+\underbrace{N\textstyle\sum_k\lambda_k}_{\text{noise}} .$$

Comparing the two noise terms, $\sum_k\lambda_k=2T-2=58$ against $T=30$ — barely a
factor of 2. Comparing the two signal terms, $\sum_k\lambda_kS_k$ **collapses**,
because every $S_k$ is multiplied by a $\lambda_k$ of order $0.01$. The ratio flips.

Numerically, with a stand-in smooth-joint spectrum $S=[0.5,0.3,0.12,0.05,0.03]$ in
bins $k=0..4$ (total signal power 1) and a flat floor $N$:

| per-bin SNR | noise share of $d_\text{path}$ | noise share of $d_\text{shape}$ |
|---|---|---|
| 40 dB | 0.3% | **24%** |
| 30 dB | 2.9% | **76%** |
| 20 dB | 23% | **97%** |

**(5) No dynamic range.** A trained policy emits smooth trajectories and does not
reproduce tremor, so the jitter enters numerator *and* denominator at the same
magnitude and largely cancels. At 30 dB:

| low-band relative error | 0% | 10% | 20% | 30% | 50% |
|---|---|---|---|---|---|
| $R_\text{shape}$ | 0.757 | 0.760 | 0.767 | 0.779 | 0.818 |

**A perfect prediction and a 30%-wrong one differ by 0.02.**

**(6) The gradient points the wrong way.** From $0.757$, fixing the motion buys
$0.022$; *reproducing the jitter* buys the full $0.757$. **~$34\times$ more loss is
available from learning to chatter than from getting the motion right.**

Neither the floor nor the denominator saves this. Replacing the $\lambda$-weighted
denominator with total power moves $R_\text{shape}$ from $0.757$ to $0.056$ but
leaves relative dynamic range at $8.0\%$ and the gradient ratio at $34.6\times$ —
identical to three significant figures. By §5, §7.4 is a **numerator** problem.

**Scope: this is a statement about smooth joints only.** By §6.4 the gripper's high
band is real signal at 63–315$\times$ the noise, and its shape denominator is 99%+
signal. Differencing a step gives an impulse, which is exactly the right way to ask
*when did it switch*. **So the shape term is broken for the arm joints and sound for
the gripper** — one more reason the fix has to be per dimension rather than global.

**7.5 Terminal is rank-1.** One time step, so in frequency maximally spread and not
expressible as a band; its denominator is a single time slice, the noisiest of the
three. It also presumes the whole chunk executes — under real-time chunking with
early replanning the tail never runs.

---

# Part III — Options

Recall the two complaints from §2, since everything below is sorted by which one it
answers:

> **M1 — shape-blind.** Equal MSE does not imply equal trajectory.
> **M2 — absolute units.** Error scales with how far the demonstration travels, so
> large motions dominate and small precise ones are ignored.

By §5 these need **different machinery**: $M$ (the quadratic form, §8) answers M1 and
acts *within* an example; $\nu$ (the normalizer, §9) answers M2 and acts *between*
examples and dimensions. Neither substitutes for the other.

Each entry: what it is, what it costs, what it risks.

## 8. Catalogue A — choices of $M$ (answers M1)

Every entry here is a choice of **one object**: a per-frequency weight $w_{k,d}$ in

$$\text{(the $M$ part of the score)}\;=\;\sum_k w_{k,d}\,\tilde r_{k,d}^2 .\tag{8.0}$$

One weight per *(frequency, dimension)* — not per band. A **band partition is the
special case where $w_{k,d}$ is piecewise constant in $k$.** So the whole catalogue is:

| entry | $w_{k,d}$ | effect |
|---|---|---|
| A1 flat | $1$ | the path term |
| A2 derivative order | $\lambda_k^{\,s}$ | high-pass, $s$ times |
| A3 bands | piecewise constant in $k$ | whatever you choose, per band |
| A4 whitening | $1/\max(\bar P_{k,d}-N_d,\ \phi_d)$ | **boosts** low-energy bands to parity |
| A5 Wiener | $\big(\bar P_{k,d}-N_d\big)^+/\bar P_{k,d}$ | **suppresses** low-SNR bands toward 0 |

Note A4 and A5 pull in *opposite* directions and are not interchangeable: A4
amplifies where the signal is weak, A5 attenuates where the signal-to-noise is weak.
They coincide only when weak signal and weak SNR coincide.

**A1. Flat.** $w_k=1$. The path term. *Baseline; by §7.1 it detects nothing, and it
is the constant-weight case of A3.*

**A2. Derivative order $\lambda_k^s$.** One continuous knob: $s=0$ path, $s=1$ shape,
$s=2$ acceleration, $s=3$ jerk. Attractive because jerk is *physically* meaningful —
it stresses servos and reads as bad motion. **The numbers rule it out**, normalized
weight $\lambda_k^s/\max$:

| $s$ | $k{=}1$ | $k{=}2$ | $k{=}4$ | $k{=}8$ | $k{=}15$ | $k{=}29$ |
|---|---|---|---|---|---|---|
| 1 (shape) | 0.003 | 0.011 | 0.043 | 0.166 | 0.501 | 1.0 |
| 2 (accel) | 0.000 | 0.000 | 0.002 | 0.028 | 0.251 | 1.0 |
| 3 (jerk) | 0.000 | 0.000 | 0.000 | 0.005 | 0.126 | 1.0 |

$s\ge2$ is essentially a pure Nyquist detector — **on a smooth joint a jerk penalty
is almost entirely a noise penalty.** Useful negative result: the derivative-order
family runs the wrong way, and $s$ should go *down* from 1, not up. If smooth motion
is the goal, get it by excluding the noise band (A3), not by penalizing jerk.

**A3. Frequency bands (partition).** $R_{b,d}=\sum_{k\in\mathcal B_b}\tilde r_{k,d}^2$
over a **shared** per-(example, dimension) denominator. By Parseval the bands sum to
the path term exactly, so flat weights recover the baseline inside the
parameterization. *Cost: one DCT per chunk.*

Two constraints that are not optional:

- **Band edges are per dimension.** By §6.4 a smooth joint's signal dies above
  $k\approx4$ while a gripper's switch is broadband. A single global partition is
  wrong for one of them, and excluding a global "noise band" would delete the switch.
- **Do not normalize per band.** With the §6.4 smooth-joint partition at 30 dB the
  noise band holds 2.0% of the energy, so uniform weights over four per-band ratios
  give it 25% of the weight — a $12.3\times$ amplification of a band that is
  definitionally all noise. That is §7.4 re-inflicted deliberately.

**A4. Corpus whitening.** $w_{k,d}=1/\max(\bar P_{k,d}-N_d,\ \phi_d)$ — divide by the
demonstration's *signal* power at that frequency, corpus-averaged. A **fixed** filter,
so unlike per-band normalization (§A3) it introduces no per-example amplification.
Says "every timescale matters equally on average across the dataset." The floor
$\phi_d$ is essential: without the $-N_d$ and the floor, the noise band has the
smallest $\bar P$ and therefore the largest weight — §5.1 again. *Note this is exactly
B2 applied on the frequency axis instead of the example axis.*

**A5. Wiener / SNR weight.** $w_{k,d}=\big(\bar P_{k,d}-N_d\big)^{+}\big/\bar P_{k,d}$
— the signal fraction of the measured power, i.e. the standard Wiener gain
$S/(S+N)$ with $S=\bar P-N$. It goes to $1$ where the signal dominates and to $0$
where the band is pure noise. *(The tempting $\bar P/(\bar P+N)$ is wrong: it treats
measured power as signal and bottoms out at $0.5$ in the noise band rather than
vanishing.)* This is the soft version of "drop the noise band"; hard exclusion is its
limit. Worked, with a floor at $N_d$:

| $w_{k,d}$ | $k{=}0$ | $k{=}4$ | $k{=}8$ | $k{=}29$ |
|---|---|---|---|---|
| smooth joint | 0.990 | 0.857 | **0.000** | **0.000** |
| gripper | 0.984 | 0.923 | **0.667** | **0.667** |

*One parameter per dimension, and because $N_d$ is per dimension the same formula
kills the smooth joint's noise band while preserving the gripper's switch — the §6.4
problem solved with no special case.*

**A6. Gain/shape projection.** Per dimension over the whole chunk, project the
prediction onto the demonstration,
$\gamma_d=\langle\hat a_{\cdot,d},a_{\cdot,d}\rangle/\|a_{\cdot,d}\|^2$:

$$\frac{\|\hat a_{\cdot,d}-a_{\cdot,d}\|^2}{\|a_{\cdot,d}\|^2}
=\underbrace{(\gamma_d-1)^2}_{G_d\ \text{gain}}+\underbrace{\|\hat a^\perp_{\cdot,d}\|^2/\|a_{\cdot,d}\|^2}_{S_d\ \text{shape}}.\tag{8.1}$$

Exact, orthogonal, dimensionless per dimension — so §7.3 does not arise. For a latch,
$\gamma_d$ is "did you open when the demo opened, and by how much" and $S_d$ is "did
you switch at the right time": sign and timing, all a latch has. Sign errors fall out
of the geometry — $\gamma_d=0$ gives $G_d=1$, $\gamma_d=2$ gives $1$, $\gamma_d=-1$
gives $4$ — so no $-\log((\cos+1)/2)$ construction is needed. *This is the direction
replacement.* Caveat: substituting $\hat a=a+(1-t)r$ shows both parts are quadratic
in $r$, so it does **not** escape §4.1; its virtues are exactness and unit-freeness,
not new information.

## 9. Catalogue B — choices of $\nu$ (answers M2)

**B1. Per-example baseline power.** Divide by the error a naive predictor commits on
the same chunk — the current design, with $h=0$ (freeze the arm). The baseline is a
free choice: **constant velocity** (repeat the last observed increment) is a much
stronger predictor at 30 Hz, so $R$ centres near 1 for a mediocre model and the gate
becomes discriminative rather than always-on. *Cost: needs the previous state, which
is in the batch. Trade: "1 = freezing the arm" becomes "1 = coasting."*

**B2. Signal power.** $\max(P_d-N_d,\ \phi_d)$ — subtract the estimated noise floor
before dividing. §5.1's fix, and the one that stops an all-noise chunk from
exploding: such a chunk now hits the floor and gets bounded weight instead of
maximal weight. *The highest-value change to $\nu$, and it composes with B1 (apply to
whichever baseline).*

**B3. Fixed corpus per-dimension scale.** Divide by a dataset constant per dimension,
not by anything per-example. *Fixes the dimension half of M2 with **zero**
per-example amplification risk. This is the conservative option and the right control
experiment: if B1/B2 do not beat B3, the per-example denominator is not earning its
keep.*

*Set aside:* batch rank/quantile normalization (robust to tails but loses absolute
meaning and couples examples within a batch); learned per-example uncertainty in the
EDM2/Kendall–Gal form (principled, but adds a head that can collapse to explaining
everything as noise); importance sampling instead of reweighting (same intent,
different mechanism — but plain weights are preferred here).

## 10. What is not on the table

By §4.1, nothing in §8–9 can measure anything the flow loss cannot: every entry is a
reweighting of the same residual. Escaping that class requires either **multi-step
composition** (a rollout, as TCFM does) or a **nonlinear criterion** (a learned
feature metric, as LPL does). Both are out of scope: a rollout multiplies
action-expert cost and introduces a sequential gradient, and a learned critic adds a
model and an instability.

The consequence, stated plainly rather than hidden: **this is a weighting scheme.**
It makes the flow loss count the right things — the right frequencies, on the right
per-dimension scale. It does not add information the flow loss lacks. That is the
same class as Min-SNR and EDM2, on a different axis.

## 11. A recommended minimal set

If only three things get done:

1. **Divide out $(1-t)^2$** (§4.2). Exact, one line, and without it no threshold means
   the same thing at two different flow times.
2. **Per-dimension denominators** (§7.3), against signal power (B2). Cheapest change,
   widest class of problem fixed, and what removes the gripper pathology structurally
   — no dimension names, no kinematic model.
3. **Replace shape with A3 + A5**: bands with a shared denominator, per-dimension
   edges, noise band Wiener-weighted rather than hard-excluded. This is the only fix
   for §7.4, since by §5 the denominator cannot touch it — and the per-dimension
   $N_d$ is what keeps the gripper's switch from being deleted along with the arm's
   jitter.

Then **A6** replacing direction, and **B3** as the control that says whether the
per-example denominator earns its keep at all.

Terminal stays unresolved (§7.5). Options in order of conservatism: keep it with a
per-dimension denominator; weight time steps by probability of execution; drop it and
rely on the DC and low bands.

---

# Part IV — Context

## 12. Relation to FAST

FAST already applies a DCT to the action chunk, so the obvious question is what §8
adds. **The transform is the same; the purpose is opposite.**

FAST uses the DCT to make actions **compressible** — energy concentrates in few
coefficients, high coefficients quantize to zero, BPE merges the zero runs, sequences
get short. §8 uses it to make the loss's frequency weighting **explicit**.

| | FAST | §8 |
|---|---|---|
| next step after DCT | quantize → BPE → CE over token ids | weight bands → normalize → squared error |
| optimizes | likelihood of a discrete code | a normalized metric on the trajectory |
| frequency addressable in the loss? | **no** — BPE merges slots across both $k$ and $d$ | yes, by construction |
| implicit frequency weight | ≈ bits per coefficient, so monotone in magnitude | whatever is chosen |
| normalization | **one global scale** (`action_tokenizer.scale`) | per example × per dimension |
| ordinal structure | discarded by CE (hence the `ordinal` aux term) | is the entire object |

A compressor allocates bits by energy — capacity goes where the signal is big. That is
right for compression and wrong for a manipulation metric, where the last millimetres
of an approach matter more than the transit. **FAST's implicit weighting is monotone
increasing in motion magnitude, which is M2 encoded into the representation rather
than fixed.** Token count grows with motion magnitude too, which is what the
`discrete_loss_token_weighting` options ($2/\sqrt{\text{count}}$) exist to undo.

Two consequences:

- **The base FAST CE does not have the §7.4 crack.** Quantization already discards the
  noise band — the config's own note says high-frequency slots are covered by long
  zero-run merge tokens. It gets §8's noise-band exclusion for free, for an unrelated
  reason.
- **The discrete *auxiliary* loss re-introduces it.** Its `shape` term is the
  $4\sin^2(\pi k/2T)$ reweighting divided by the hold-baseline power — §7.4 verbatim,
  in DCT space. The crack is on both branches; the continuous one got it from
  time-domain differencing, the discrete one from the aux term.

BPE is also why frequency is not addressable from the CE: one token spans several
coefficient slots of different $k$ *and* different $d$. The discrete aux loss recovers
per-slot marginals by grouping logits over the value each candidate token would place
at a slot — machinery that exists precisely because the frequency axis is otherwise
unreachable.

## 13. Relation to the literature

Compressed from the full survey. Verification status is in §13.4.

### 13.1 What each nearby work does

| work | what it does | bearing here |
|---|---|---|
| **TCFM** (arXiv:2605.08511) | $\mathcal L_\text{CFM}+1.0\mathcal L_\text{rect}+0.5\mathcal L_\text{multistep}+0.1\mathcal L_\text{vel}+0.1\mathcal L_\text{action}$, with $\mathcal L_\text{action}=\|\hat x_1-x_1\|^2$ via a **5-step Euler rollout**, $S=5$ | the direct competitor. $\mathcal L_\text{action}$ is the path term, unnormalized. $\mathcal L_\text{vel}$ differences along **flow time**, ours along **chunk index** — orthogonal |
| **LPL** (ICLR 2025, arXiv:2411.04873) | $\mathcal L_\text{diff}+\lambda\|\varphi(\hat x_0)-\varphi(x_0)\|^2$, $\varphi$ = frozen VAE decoder features | the mechanism precedent: score the free clean sample with a **nonlinear** criterion |
| **FocalPolicy** (ICML 2026, arXiv:2605.15944) | time-domain chunk loss + orthonormal-DCT spectral loss; low coefficients = global motion, high = execution detail | **prior art for A3.** Band-splitting a trajectory loss is not new |
| **OMP** (ICML 2026, arXiv:2512.19347) | $\mathcal L_\text{mse}+\lambda_\text{Disp}\mathcal L_\text{Disp}+\lambda_{\cos}\mathcal L_{\cos}$, $\mathcal L_{\cos}=-\log\frac{\cos\alpha+1}{2}$ on **velocity** vectors | shares §7.3 if its action space mixes joints with a gripper. The log form diverges at antiparallel where $1-\cos$ saturates at 2 |
| **REPA** (ICLR 2025 Oral, arXiv:2410.06940) | cosine alignment on **representations**, $\lambda\in[0.5,1]$ | coordinates are homogeneous by construction — no metric problem |
| **ACG** (ICRA 2026, arXiv:2510.22201) | training-free test-time modification for jerk/jitter | its ablation reports **intra**-chunk coherence matters more than inter-chunk |
| **DILATE** (NeurIPS 2019), **TILDE-Q** (arXiv:2210.15050) | shape + temporal-alignment decomposition; invariance to amplitude and phase | source of the M1 argument; their phase/alignment half is not pursued here |
| **MiDaS** (TPAMI 2020), **Eigen** (NeurIPS 2014) | scale-invariant loss + **multi-scale** gradient matching | not pursued; see the domain difference in §13.3 |
| **Min-SNR** (ICCV 2023), **EDM2** (CVPR 2024), **VAW** (arXiv:2506.16688) | reweight per **noise level**: $\min(\text{SNR},\gamma)/\text{SNR}$; learned $\mathcal L=\|\cdot\|^2/u(\sigma)^2+\log u(\sigma)^2$ | set aside in §9. Same motivation as our denominator, on the **orthogonal axis** |
| **Theil's $U$**, **MASE** (2006), **zero-velocity** (CVPR 2017) | divide by a naive forecast's error — as an **evaluation metric** | the precedent for $\nu$; the constant-velocity variant of B1 is the MASE convention |
| **CI-MSE** (arXiv:2606.29898) | restrict error to task-critical segments; Spearman $-0.87$ vs $-0.61$ for raw MSE | supplies the validation protocol (§14) |
| **Per-Group Error** (arXiv:2606.00253) | on an 11-DoF manipulator the lowest-aggregate-MSE checkpoint is not the best on the robot; easy joint groups mask failing ones | evaluation-side precedent for per-dimension handling (§7.3) |
| **T-MEE** (arXiv:2602.04228) | trajectory-level minimum-error-entropy *combined with* MSE | third strategy: select (CI-MSE), reshape (T-MEE), normalize (here) |

### 13.2 Why the cracks do not show up there

- **§4.1 is avoided by paying.** TCFM's rollout makes $\hat x_1$ a nonlinear
  composition, so $e=(1-t)r$ **does not hold** — it is exactly the $S=1$ special case.
  LPL keeps one step but makes $\varphi$ nonlinear. The free one-step reconstruction is
  better read as a **trade** than an advantage: the expense is what buys the
  information.
- **§4.2 and §7.4 exist only because of the un-borrowed parts.** The $(1-t)^2$ gate
  schedule needs a *hard threshold*; TCFM, OMP, REPA and LPL all use fixed weights and
  no gating. Nothing comparable divides by a per-example, data-derived denominator, so
  nothing comparable has a denominator for a noise floor to contaminate — they have the
  opposite problem, absolute units (M2). **The per-example denominator is the real idea
  here, and it is where both remaining cracks live.**
- **§7.3 is genuinely shared** with OMP, probably masked by benchmark action spaces
  being uniformly normalized and simulated grippers being well-scaled continuous
  dimensions rather than physical latches.

### 13.3 The domain-transfer pattern

Gradient matching comes from depth and images, where high-frequency content is *real
signal* — edges, boundaries. On a smooth one-second joint trajectory at 30 Hz the same
operator lands where the high band is all noise. FDE comes from open-loop forecasting,
where the full horizon is the prediction and nothing is executed. **The techniques
transfer; their unstated assumptions about where the signal lives do not.**

### 13.4 Verification status

Confirmed against primary sources: TCFM (formulas, weights, date), FocalPolicy, LPL
(ICLR 2025 poster), REPA (objective form, $\lambda$ range), OMP (cosine form), EDM2
(official NVlabs code), CI-MSE (both correlation figures), Per-Group Error, DILATE,
Martinez et al., MiDaS. **Not verified:** T-MEE's exact formulation (abstract only);
ACG's ablation details (project page only); Min-SNR and VAW formulas (secondary
sources, IDs and venues confirmed). **TILDE-Q venue is unconfirmed** — cite as
arXiv:2210.15050 with no venue. **MiDaS spacings**: the $s\in\{1,2,4,8,16\}$ figure
previously attributed to it came from a different repository; MiDaS uses downsampled
**scale levels**, which is why A4 is phrased that way.

## 14. What to measure first

Three measurements, all cheap, none needing a model. Nothing in Part III can be
parameterized without the first.

1. **Corpus power spectrum $P_{k,d}$ per dimension.** Sets band edges (A3), exposes the
   noise floor $N$ for A6 and B2, and turns the §7.4 tables from illustration into
   measurement. **Do this before anything else.**
2. **Per-dimension excursion distributions $\|a_{\cdot,d}\|^2$.** Gives the floors and
   activity medians for §7.3's fix, and shows how far apart the gripper and the arm
   joints actually are in the normalized space now in use.
3. **Batch distributions of every proposed ratio.** Ratio families have heavy right
   tails and the aggregate is set by the smallest denominators. Log the histogram;
   aggregate with $\text{mean}\log R$ or a median, not an arithmetic mean.

Two checks available **from existing logs, today**: $R_\text{shape}$ should sit near a
constant, barely move while the flow loss falls, and fire its $0.2$ gate on ~100% of
samples. Both the mean and the active fraction are already logged — if that signature
is there, §7.4 is confirmed with no new work.

**Validation protocol.** CI-MSE's standard is Spearman rank correlation between the
offline quantity and real rollout success across a spread of checkpoints. The claim
worth testing: any $R_j$ correlates with success better than raw action MSE. A cheaper
proxy first — does per-chunk hold power correlate with CI-MSE's critical-interval
labels? If it does, the per-example denominator is an unsupervised approximation of
critical-interval selection, which is the strongest available motivation for it. If it
does not, that is worth knowing before tuning any $w_j$.
