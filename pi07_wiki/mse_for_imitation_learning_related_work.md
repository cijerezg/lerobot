# Related Work for *On Mean Squared Error for Imitation Learning*

Companion note to [`mse_for_imitation_learning.md`](mse_for_imitation_learning.md).
Compiled 2026-08-17. Every claim below was checked against a primary source
(arXiv abstract page, published PDF, project page, or official code); the
verification status of each is recorded in §7 and the reference list marks what
could not be confirmed.

---

## 1. Notation

Notation follows the main note. The flow policy sees

$$
x_t = (1-t)\,z + t\,a , \qquad z \sim \mathcal N(0, I),
$$

predicts $\hat v \approx v^* = a - z$, and the clean chunk is recovered in one
step as $\hat a = x_t + (1-t)\hat v$. Write the **velocity residual**

$$
r \;=\; \hat v - v^* ,
$$

so that the **reconstruction residual** is exactly

$$
\hat a - a \;=\; (1-t)\,r .
\tag{1}
$$

$k \in \{1,\dots,T\}$ indexes chunk steps, $d$ action dimensions, $h$ is the
hold-still chunk ($h_{k,d} = 0$ in anchor-encoded space), and $R_j$, $w_j$, $g_j$
are the four ratios, their weights, and their gates.

Throughout, the four proposed terms are referred to as **path**, **shape**,
**terminal**, and **direction**.

---

## 2. Structural restatement of the proposal

Before comparing to the literature it helps to substitute (1) into each ratio,
because this determines which claims are actually contestable.

**Path.**

$$
R_{\mathrm{path}}
= (1-t)^2 \cdot
\frac{\frac{1}{N}\sum_{k,d} r_{k,d}^2}
     {\frac{1}{N}\sum_{k,d} a_{k,d}^2} \, .
\tag{2}
$$

The numerator is the per-example flow loss. So $R_{\mathrm{path}}$ is the flow
loss **reweighted** by $(1-t)^2 / \overline{a^2}$ and nothing else. It introduces
no geometric information the flow loss lacks.

**Shape.** Since $h$ is constant in $k$, $h_k - h_{k-1} = 0$, so

$$
R_{\mathrm{shape}}
= (1-t)^2 \cdot
\frac{\frac{1}{N}\sum_{k,d}\big(r_{k,d} - r_{k-1,d}\big)^2}
     {\frac{1}{N}\sum_{k,d}\big(a_{k,d} - a_{k-1,d}\big)^2} \, .
\tag{3}
$$

This *is* new information relative to (2): it is the flow residual passed through
a first-order high-pass filter, normalized by the demonstration's own per-step
motion.

**Terminal.**

$$
R_{\mathrm{terminal}}
= (1-t)^2 \cdot
\frac{\frac{1}{N}\sum_{d} r_{T,d}^2}
     {\frac{1}{N}\sum_{d} a_{T,d}^2} \, .
\tag{4}
$$

**Direction.** With $h_T = 0$, $w = a_T$ and $u = \hat a_T = a_T + (1-t)\,r_T$,

$$
R_{\mathrm{direction}}
= 1 - \frac{\langle a_T + (1-t) r_T,\; a_T\rangle}
            {\|a_T + (1-t) r_T\|\;\|a_T\|} \, .
\tag{5}
$$

**Consequence — all four terms are $t$-gated.** Terms (2)–(4) carry an explicit
$(1-t)^2$; term (5) satisfies $R_{\mathrm{direction}} \to 0$ as $t \to 1$ because
$u \to w$. Since the gate $G_{g}(R) = R\cdot\mathbb 1[R > g]$ compares $R$ against
a **fixed** $g = 0.2$, the effective gate is a *noise-level schedule*: at large
flow time every auxiliary term switches off regardless of prediction quality, and
all auxiliary supervision concentrates at small $t$. Whether this is intended
should be stated; if it is not, divide the numerators by $(1-t)^2$ to decouple
the gate from the flow-time sampling distribution.

This restatement gives the honest claim structure:

| Piece | Status |
|---|---|
| Path term | Prior art (§3.2), modulo the denominator |
| Mechanism (free one-step reconstruction, structural criterion on it) | Prior art (§3.1) |
| Shape term | Weaker version of prior art (§4.2) |
| Terminal term | Standard (final displacement error) |
| Direction term | Related prior art on velocity, not on reconstruction endpoint (§4.3) |
| **Hold-still denominator, per example** | **No precedent found** (§5) |

---

## 3. Closest work: auxiliary objectives on the reconstruction

### 3.1 Latent perceptual loss — the mechanism precedent

Berrada et al., **"Boosting Latent Diffusion with Perceptual Objectives"**
(ICLR 2025; arXiv:2411.04873) forms $\hat x_0$ by one-step prediction from the
current noisy sample, then applies a **non-MSE structural criterion** to it —
here, distances between internal decoder features $\phi(\cdot)$ — and adds this
to the base objective:

$$
\mathcal L = \mathcal L_{\mathrm{diff}} + \lambda \,\big\| \phi(\hat x_0) - \phi(x_0) \big\|^2 .
$$

The paper explicitly states compatibility with $\epsilon$-prediction,
$v$-prediction, **and flow matching**, and reports 6–20% FID improvement.

This is the same move as the present proposal, one domain over: *the clean sample
is already available for free during training, so score it with something the
denoising MSE cannot see.* It should be cited as the mechanism precedent. The
distinction to draw is that LPL's criterion is a **learned** feature metric,
whereas path/shape/terminal/direction are **analytic and interpretable**, and
none require a pretrained encoder.

### 3.2 Trajectory-Consistent Flow Matching — the direct competitor

Ahmed, Nag, Akash, Hussein & Begum, **"Trajectory-Consistent Flow Matching for
Robust Visuomotor Policy Learning"** (arXiv:2605.08511, 8 May 2026) is the
nearest neighbour in robotics. Its full objective, with the paper's own weights:

$$
\mathcal L
= \mathcal L_{\mathrm{CFM}}
+ 1.0\,\mathcal L_{\mathrm{rect}}
+ 0.5\,\mathcal L_{\mathrm{multistep}}
+ 0.1\,\mathcal L_{\mathrm{vel}}
+ 0.1\,\mathcal L_{\mathrm{action}} .
$$

Two components overlap directly:

$$
\mathcal L_{\mathrm{action}} = \mathbb E\big[\|\hat x_1 - x_1\|^2\big],
\qquad
\mathcal L_{\mathrm{vel}} = \mathbb E\Big[\tfrac{1}{S-1}\textstyle\sum_{i=1}^{S-1}
\big\|v_\theta(x_{t_{i+1}}, t_{i+1}) - v_\theta(x_{t_i}, t_i)\big\|^2\Big],
$$

with $S = 5$.

- $\mathcal L_{\mathrm{action}}$ **is the path term**, unnormalized, obtained via
  a 5-step Euler rollout rather than the free one-step reconstruction.
- $\mathcal L_{\mathrm{vel}}$ is a differencing penalty **along flow time $t$**,
  whereas the shape term (3) differences **along chunk index $k$**. These are
  orthogonal, which is a clean point of differentiation.
- $\mathcal L_{\mathrm{multistep}}$ supervises integrated displacement over
  segments, $\Delta x^* = (t_1 - t_0)(x_1 - x_0)$, with $S=4$ Euler steps and
  $K=3$ sampled segments.

TCFM's stated motivation is identical to §1 of the main note: pointwise velocity
supervision does not constrain the integrated trajectory. **It uses no baseline
normalization and no gating.** This is the paper to cite, differentiate from, and
ideally compare against.

### 3.3 Test-time and guidance-based alternatives

Lee et al., **ACG: Action Coherence Guidance for Flow-based VLA models**
(ICRA 2026; arXiv:2510.22201) attacks the same failure — jerk, pauses, jitter in
the reconstructed chunk — but as a **training-free test-time** modification of
the denoising flow. Notably, its ablation reports that *intra*-chunk coherence
matters more than *inter*-chunk coherence, which is evidence in favour of the
shape term's within-chunk formulation.

---

## 4. Term-by-term precedent

### 4.1 The Section-1 argument (equal loss $\neq$ equal trajectory)

This is the standing motivation of the shape-aware forecasting literature.
Le Guen & Thome, **DILATE** (NeurIPS 2019; arXiv:1909.09020) decomposes the
objective into a shape term (a smooth relaxation of DTW) and a temporal-alignment
term. Lee, Lee, Lim & Ko, **TILDE-Q** (arXiv:2210.15050) opens with the same
observation — that $L_p$ losses score shape-distinct predictions identically —
and constructs a loss invariant to amplitude and phase distortion.

The $r_A = [1,1,1,1]$ vs. $r_B = [1,-1,1,-1]$ construction in §1 of the main
note is the standard counterexample from this line. It is correct as written
(adjacent differences of $r_B$ are $\pm 2$, mean square $4$), but it is not novel
and should be presented as recalled motivation, with citation, not as a finding.

### 4.2 Shape

Two stronger relatives:

- **Multi-scale gradient matching.** Ranftl, Lasinger, Hafner, Schindler &
  Koltun, **MiDaS** (TPAMI 2020; arXiv:1907.01341) pairs a scale-and-shift
  invariant loss with a **multi-scale** gradient-matching loss, evaluated at
  several downsampled resolutions rather than at adjacent pixels only. The
  antecedent is Eigen, Puhrsch & Fergus (NeurIPS 2014; arXiv:1406.2283).
  The transferable point: **the shape term (3) is the single-spacing case.**
  A reconstruction that drifts away from the demonstration monotonically over
  $T=30$ steps has small adjacent differences at every step and is nearly
  invisible to (3). Adding spacings $s \in \{1,4,8\}$,

  $$
  R_{\mathrm{shape}}^{(s)} = (1-t)^2 \cdot
  \frac{\sum_{k,d}\big(r_{k,d} - r_{k-s,d}\big)^2}
       {\sum_{k,d}\big(a_{k,d} - a_{k-s,d}\big)^2},
  $$

  closes that hole at negligible cost.

- **Frequency-domain generalization.** He, Yang, Liang, Hao, Sebe & Tian,
  **FocalPolicy** (ICML 2026; arXiv:2605.15944) pairs a time-domain chunk loss
  with an orthonormal-DCT spectral loss on the macro-trajectory,
  $\mathcal L_{\mathrm{FCO}} = \mathcal L_{\mathrm{time}} + \lambda \mathcal L_{\mathrm{freq}}$,
  and argues low-frequency coefficients carry global motion while high-frequency
  coefficients carry execution detail. Multi-spacing differencing is the cheap
  approximation of this; the DCT is the principled version.

### 4.3 Direction

- Yu, Kwak, Jang, Jeong, Huang, Shin & Xie, **REPA** (ICLR 2025 **Oral**;
  arXiv:2410.06940) established the pattern of adding a cosine-alignment term to
  a flow/diffusion objective, $\mathcal L = \mathcal L_{\mathrm{velocity}} + \lambda\,\mathcal L_{\mathrm{REPA}}$,
  with $\lambda \in [0.5, 1.0]$ and $\mathcal L_{\mathrm{REPA}}$ a negative
  patchwise similarity. It aligns *representations*, not outputs — but it is the
  reason a cosine term in a flow objective will not read as surprising.
- **OMP: One-step MeanFlow Policy with Directional Alignment** (ICML 2026;
  arXiv:2512.19347) is the policy-side instance. It applies

  $$
  \mathcal L_{\cos} = -\log\!\Big(\tfrac{\cos\alpha + 1}{2}\Big),
  \qquad
  \cos\alpha = \frac{\langle v_0,\; u(z_t, r, t \mid c)\rangle}{\|v_0\|\,\|u\|},
  $$

  combined as $\mathcal L = \mathcal L_{\mathrm{mse}} + \lambda_{\mathrm{Disp}}\mathcal L_{\mathrm{Disp}} + \lambda_{\cos}\mathcal L_{\cos}$,
  with no gating.

  **The differentiator is real and worth stating explicitly:** OMP takes the
  cosine between *velocity vectors*; (5) takes the cosine between *final
  displacements of the reconstruction*. The former constrains the transport
  field, the latter constrains where the arm ends up. Note also the functional
  form: $-\log((\cos\alpha+1)/2)$ diverges as $\cos\alpha \to -1$, whereas
  $1 - \cos\alpha$ saturates at $2$. If antiparallel predictions are the failure
  mode being targeted, the log form gives a much stronger gradient there and is
  worth an ablation.

### 4.4 Terminal

Final-displacement error (FDE) is a standard trajectory-forecasting metric and
needs no defense, but it is also not a contribution. Its function in (4) is as a
reweighting that undoes the $1/T$ dilution of the last step — that is the framing
to use.

---

## 5. The denominator: where the actual novelty sits

### 5.1 Modern normalization is on the *noise* axis

Every current adaptive-weighting method in diffusion/flow training normalizes
along $t$, not along examples:

- **Min-SNR-$\gamma$** (Hang et al., ICCV 2023; arXiv:2303.09556) rescales the
  per-timestep loss by $w_t = \min(\mathrm{SNR}_t, \gamma)/\mathrm{SNR}_t$,
  treating training as multi-task learning over noise levels.
- **EDM2** (Karras, Aittala, Lehtinen, Hellsten, Aila & Laine, CVPR 2024;
  arXiv:2312.02696) learns a per-$\sigma$ uncertainty and divides by it. From the
  official NVIDIA implementation (`NVlabs/edm2`, `training/training_loop.py`):

  ```python
  denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
  loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
  ```

  i.e. $\mathcal L = \frac{1}{u(\sigma)^2}\|D(x;\sigma) - y\|^2 + \log u(\sigma)^2$,
  the Kendall–Gal homoscedastic-uncertainty form with $u(\sigma) \approx \mathrm{MLP}(c_{\mathrm{noise}}(\sigma))$.
  **The motivation is identical to the hold-still denominator — put every
  contribution on a common scale — but the scale is learned per noise level,
  not computed per example.**
- **Variational Adaptive Weighting** (Qiu & Lin, arXiv:2506.16688, June 2025)
  derives a closed-form variationally optimal weighting for diffusion *planning*
  specifically, criticizing EDM2's MLP for failing to track sharp variation early
  in training. Also noise-axis.

### 5.2 The example-axis idea has old, non-robotics precedent

The "divide by what a naive predictor would have scored" construction is
standard in forecasting, as an **evaluation metric**:

- **Theil's $U_2$**: ratio of forecast RMSE to the RMSE of the naive no-change
  forecast; $U = 1$ means no better than doing nothing, $U > 1$ worse. This is
  precisely the reading given for $R_j$.
- **MASE** (Hyndman & Koehler, *Int. J. Forecasting* 22(4), 2006): MAE divided by
  the in-sample MAE of the naive one-step forecast.
- **Zero-velocity baseline**: Martinez, Black & Romero, *On human motion
  prediction using recurrent neural networks* (CVPR 2017; arXiv:1705.02445) —
  literally $h$, "repeat the last observed pose" — shown to beat the deep models
  of its day. The abstract's phrasing is "a simple baseline that does not attempt
  to model motion at all."

**No source was found that uses such a ratio as a per-example denominator inside
a training objective**, in any of: diffusion/flow generative modelling, VLA and
visuomotor policy learning, time-series forecasting, or monocular depth.

### 5.3 The defensible claim

> Existing adaptive weighting for diffusion and flow policies asks *"how hard is
> this noise level?"*. We ask *"how much motion did this demonstration actually
> require?"*, and answer it analytically with the example's own hold-still
> baseline — the training-time counterpart of the naive-forecast denominators
> long used for evaluation (Theil's $U$, MASE, zero-velocity).

That is narrow, but it is a real gap and it is stated in a form a reviewer can
check.

---

## 6. Section 2's critique (absolute units) is an active 2026 thread

The observation that raw action MSE is measured in absolute units and therefore
misranks policies is currently being made independently by several groups. These
should be cited as converging evidence, and they also supply the evaluation
protocol the present work needs.

- **CI-MSE** — Huang, Zheng, Chen, You & Gao, *Critical Interval MSE: Toward
  Reliable Offline Validation for Robot Manipulation Policies* (arXiv:2606.29898,
  29 Jun 2026). Restricts error computation to task-critical segments. Reports
  Spearman rank correlation with rollout success of $-0.87$ for CI-MSE versus
  $-0.61$ for raw MSE across a range of checkpoints. **This is the strongest
  available quantitative statement of the problem §2 describes, and its
  correlation protocol is the natural way to validate the $R_j$ terms.**
- **Per-Group Error, Not Total MSE** — Montagut Bofi, García Blasco, Pulli &
  Vincze (arXiv:2606.00253, 29 May 2026). Shows that on an 11-DoF mobile
  manipulator the lowest-aggregate-MSE checkpoint is *not* the best on the real
  robot, because easy joint groups mask failing ones. Same disease, different
  axis (joint groups rather than motion scale).
- **T-MEE** — *Reshaping Action Error Distributions for Reliable
  Vision-Language-Action Models* (arXiv:2602.04228, 4 Feb 2026). Adds a
  trajectory-level minimum-error-entropy objective *combined with*, not replacing,
  MSE, arguing that MSE "imposes strong pointwise constraints on individual
  predictions."

Note the strategic difference: CI-MSE and Per-Group Error **select or partition**
the error; T-MEE **reshapes its distribution**; the present proposal
**normalizes** it. All three are responses to the same complaint.

---

## 7. Verification pass: corrections to earlier claims

Two claims made in the first, unverified pass do not survive checking.

1. **TILDE-Q venue.** Previously stated as ICLR. The arXiv abstract page
   (v2, 13 Mar 2024) says "submitted to ICML 2024 … under review"; the OpenReview
   forum could not be reached to confirm a final decision. **Cite as
   arXiv:2210.15050 with no venue claim** until confirmed.
2. **MiDaS gradient spacings.** Previously stated as $s \in \{1,2,4,8,16\}$
   attributed to Ranftl et al. That figure came from a GitHub issue on a
   *different* repository (Structure-Guided Ranking Loss), not from MiDaS. MiDaS
   uses multiple **downsampled scale levels**; the specific spacing set is an
   implementation variant. **The substantive recommendation in §4.2 is
   unaffected — only the numbers were misattributed.**

Confirmed against primary sources: TCFM (formulas, weights, authors, date),
FocalPolicy (title, authors, ICML 2026), LPL (ICLR 2025 poster page), REPA
(arXiv:2410.06940, ICLR 2025 Oral, objective form and $\lambda$ range), OMP
(ICML 2026, cosine loss form), EDM2 uncertainty weighting (official NVlabs code),
CI-MSE (abstract, both correlation figures), Per-Group Error (abstract), DILATE
(arXiv:1909.09020), Martinez et al. (arXiv:1705.02445, CVPR 2017), MiDaS
(arXiv:1907.01341, TPAMI 2020).

Not independently verified: T-MEE's exact formulation (abstract only); ACG's
ablation details (project page only); Min-SNR and Variational Adaptive Weighting
formulas (secondary sources, though the arXiv IDs and venues are confirmed).

---

## 8. Recommendations arising from the review

Ordered by expected impact.

1. **Fix the aggregation, not the epsilon.** The epsilon prevents division by
   zero; it does not prevent the *arithmetic mean over a batch* of $R_j$ from
   being dominated by the few examples whose denominator is near the floor. This
   is the classic pathology of ratio metrics and the forecasting literature's
   standard answer is to aggregate ratios geometrically. Concretely: average
   $\log(R_j + \varepsilon)$ across the batch, or take the median. Log the
   batch histogram of each $R_j$ before tuning any $w_j$.

2. **Decide whether the $t$-coupling is intended.** By §2, the fixed gate
   $g = 0.2$ acts as a noise-level schedule: all four terms deactivate as
   $t \to 1$. Either dividing the numerators by $(1-t)^2$ (decoupled), or keeping
   it and saying so (deliberate low-$t$ emphasis) is defensible; leaving it
   unremarked is not.

3. **Measure the distribution of $R_j$ before tuning $w_j$.** Martinez et al.
   established that hold-still is a *strong* baseline. If most chunks are slow,
   $R_j$ sits near or above $1$ routinely and the $0.2$ gate fires on
   essentially every sample, making it a no-op. If most chunks are fast, it never
   fires. Neither is visible without the histogram.

4. **Make the gate continuous, or justify the discontinuity.** $G_g(R) = R\cdot\mathbb 1[R>g]$
   jumps by $g$ at the boundary. The $\varepsilon$-insensitive form
   $\max(R - g, 0)$ is continuous, costs nothing, and is what the literature uses.
   If the hard gate is deliberate — e.g. to keep full gradient magnitude
   immediately above threshold rather than ramping from zero — say so, because
   it will be asked about.

5. **Add multi-spacing to the shape term** (§4.2). Three lines; closes the
   slow-drift blind spot of single-step differencing.

6. **Drop or justify the path term.** By (2) it is the flow loss times
   $(1-t)^2/\overline{a^2}$. The ablation must show it doing something
   $L_{\mathrm{flow}}$ does not, or it should be folded into the flow loss as an
   explicit per-example weight — which is the cleaner presentation anyway, and
   makes the connection to EDM2/Min-SNR immediate.

7. **Validate with the CI-MSE protocol.** Spearman rank correlation between the
   offline quantity and real rollout success, across a spread of checkpoints, is
   now the expected evidence standard for a claim of this shape. The
   headline result would be: $R_j$ correlates with success better than raw
   action MSE does.

8. **Consider the log-cosine form for direction** (§4.3), which unlike
   $1 - \cos$ diverges on antiparallel predictions.

---

## 9. References

**Auxiliary objectives on reconstructions / flow policies**

- T. Berrada, P. Astolfi, M. Hall, et al. *Boosting Latent Diffusion with
  Perceptual Objectives.* ICLR 2025 (poster). arXiv:2411.04873.
- R. Ahmed, S. Nag, M. Akash, M. Hussein, M. Begum. *Trajectory-Consistent Flow
  Matching for Robust Visuomotor Policy Learning.* arXiv:2605.08511, 8 May 2026.
- Q. He, Z. Yang, W. Liang, C. Hao, N. Sebe, J. Tian. *FocalPolicy:
  Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent
  Visuomotor Policy.* ICML 2026. arXiv:2605.15944.
- *ACG: Action Coherence Guidance for Flow-based Vision-Language-Action Models.*
  ICRA 2026. arXiv:2510.22201. Project page: davian-robotics.github.io/ACG/
- *OMP: One-step MeanFlow Policy with Directional Alignment.* ICML 2026 (poster).
  arXiv:2512.19347.

**Alignment / cosine objectives**

- S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, S. Xie. *Representation
  Alignment for Generation: Training Diffusion Transformers Is Easier Than You
  Think.* ICLR 2025 (Oral). arXiv:2410.06940.

**Loss weighting and normalization in diffusion / flow training**

- T. Hang, S. Gu, C. Li, et al. *Efficient Diffusion Training via Min-SNR
  Weighting Strategy.* ICCV 2023. arXiv:2303.09556.
- T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila, S. Laine. *Analyzing
  and Improving the Training Dynamics of Diffusion Models* (EDM2). CVPR 2024.
  arXiv:2312.02696. Code: github.com/NVlabs/edm2
- Z. Qiu, T. Lin. *Fast and Stable Diffusion Planning through Variational
  Adaptive Weighting.* arXiv:2506.16688, 20 Jun 2025.
- A. Kendall, Y. Gal, R. Cipolla. *Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics.* CVPR 2018. arXiv:1705.07115.
  (Origin of the EDM2 weighting form.)

**Offline metrics for manipulation policies**

- H. Huang, T. Zheng, Y. Chen, J. You, Y. Gao. *Critical Interval MSE: Toward
  Reliable Offline Validation for Robot Manipulation Policies.* arXiv:2606.29898,
  29 Jun 2026.
- P. Montagut Bofi, M. García Blasco, T. Pulli, M. Vincze. *Per-Group Error, Not
  Total MSE: Fine-Tuning Vision-Language-Action Models for 11-DoF Mobile
  Manipulation.* arXiv:2606.00253, 29 May 2026.
- *Reshaping Action Error Distributions for Reliable Vision-Language-Action
  Models* (T-MEE). arXiv:2602.04228, 4 Feb 2026.

**Shape-aware losses (pre-2022, still load-bearing)**

- V. Le Guen, N. Thome. *Shape and Time Distortion Loss for Training Deep Time
  Series Forecasting Models* (DILATE). NeurIPS 2019. arXiv:1909.09020.
- H. Lee, C. Lee, H. Lim, S. Ko. *TILDE-Q: A Transformation Invariant Loss
  Function for Time-Series Forecasting.* arXiv:2210.15050. *(Venue unconfirmed —
  see §7.)*
- R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun. *Towards Robust
  Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset
  Transfer* (MiDaS). TPAMI 2020. arXiv:1907.01341.
- D. Eigen, C. Puhrsch, R. Fergus. *Depth Map Prediction from a Single Image
  using a Multi-Scale Deep Network.* NeurIPS 2014. arXiv:1406.2283.

**Baseline-relative error measurement**

- R. J. Hyndman, A. B. Koehler. *Another look at measures of forecast accuracy*
  (MASE). *International Journal of Forecasting* 22(4):679–688, 2006.
- H. Theil. *Applied Economic Forecasting.* North-Holland, 1966. (Theil's $U$.)
- J. Martinez, M. J. Black, J. Romero. *On human motion prediction using
  recurrent neural networks.* CVPR 2017. arXiv:1705.02445.
