# Choosing a Principled Action Objective

**Status:** decision memo. This is a companion to
[action_trajectory_losses.md](action_trajectory_losses.md), not a replacement for
its band analysis. That note contains the DCT definitions, path/shape equivalence,
frequency interpretation, gripper discussion, and option catalogue. Those
derivations are not repeated here.

This memo asks a narrower question:

> Given clean, dataset-normalized action targets, which objective change is
> defensible, what assumption does it encode, and what complexity does it buy?

---

## 1. Executive conclusion

The continuous action expert and FAST/VLM branch are independent prediction
problems:

- the action expert learns a continuous conditional flow field;
- FAST learns an autoregressive distribution over BPE tokens.

They may be evaluated with the same trajectory decomposition, but they need not
use the same training loss.

The conservative recommendation is:

1. **Flow:** compare ordinary flow MSE against a **fixed equal-band quadratic
   metric used as the flow loss itself**, not added as an auxiliary. Replacement
   avoids an arbitrary auxiliary coefficient.
2. **FAST:** retain token cross-entropy. Report bandwise coefficient/ordinal
   diagnostics first. Add a trajectory-aware FAST score only if those diagnostics
   expose a failure that matters for rollout behavior.
3. **Dimension normalization:** the dataset transform already supplies a fixed
   per-dimension metric. Add another fixed dimension weight only to express an
   explicit priority not captured by that transform, and test it separately.
4. **Dataset statistics:** use fixed per-(band, dimension) power only as a separate
   ablation with the explicit interpretation “equal relative error across corpus
   bands.” Do not mix it into the default band objective.
5. **Per-example denominators:** treat them as a serious alternative when relative
   accuracy on small motions is the goal, but recognize that they perform deliberate
   importance weighting rather than neutral normalization.

The detailed motivation for the particular bands remains in
[action_trajectory_losses.md](action_trajectory_losses.md) §6–§8.

---

## 2. What existing dataset normalization already solves

Let $a\in\mathbb R^{T\times D}$ be the preprocessed action chunk. The dataset
pipeline already applies fixed action statistics per dimension.

That preprocessing chooses a coordinate metric:

- different physical units are mapped into model coordinates;
- large-range and small-range action dimensions are placed on comparable scales;
- the Gaussian base distribution used by flow matching is defined in those model
  coordinates.

The loss should therefore not introduce a second per-dimension normalization by
default. Doing so would silently replace the preprocessing metric.

Dataset normalization does **not** decide:

- how temporal scales should be weighted;
- whether every frequency coefficient or every semantic band should receive equal
  capacity;
- whether short and long motions should have equal absolute or fractional error;
- which parts of a trajectory matter for task success;
- how much a trajectory-aware FAST score should weigh against token likelihood.

Those are objective choices, not normalization bookkeeping.

---

## 3. The fixed-metric result

### 3.1 Flow notation

Use uppercase letters for random variables and lowercase letters for their
realizations. Let $A\in\mathbb R^{TD}$ be the clean action chunk (with time and
action-dimension axes flattened), $Z\in\mathbb R^{TD}$ the Gaussian base sample,
$\tau\in[0,1]$ the flow time, and $C$ the conditioning context. Define

$$
X_\tau=(1-\tau)Z+\tau A,
\qquad
U=A-Z.
\tag{3.1}
$$

The complete predictor input is $X=(X_\tau,\tau,C)$. For a realization
$x=(x_\tau,\tau,c)$ and target velocity $u$, define

$$
r_\theta(x,u)=v_\theta(x)-u.
\tag{3.2}
$$

### 3.2 Fixed quadratic metrics preserve the population target

Let $M$ be any fixed symmetric positive-definite matrix over the $TD$ action
coordinates. The population objective is

$$
\mathcal L_M(v)
=\mathbb E_{X,U}\!\left[
  (v(X)-U)^\top M(v(X)-U)
\right].
\tag{3.3}
$$

Here **risk** means expected loss in the statistical-decision-theory sense. It is
ordinary mean squared error when $M=I$; for general $M$, it is expected weighted
squared error.

The law of total expectation gives

$$
\mathcal L_M(v)
=\mathbb E_X[\mathcal R_M(v(X)\mid X)],
\tag{3.4}
$$

where

$$
\mathcal R_M(v\mid X)
=\mathbb E[(v-U)^\top M(v-U)\mid X].
\tag{3.5}
$$

This is the **pointwise conditional risk**: fix one input $X=x$, then ask which
output vector $v$ minimizes expected loss over the possible target velocities $U$
conditioned on that input. With an unrestricted function class, the population
optimum can be found independently at every $x$. A finite neural network couples
inputs through shared parameters, so changing $M$ can still change its
finite-capacity solution.

For fixed $M$, expand (3.5). Once $X$ is fixed, $v$ is a deterministic candidate
output and can be moved outside the conditional expectations:

$$
\begin{aligned}
\mathcal R_M(v\mid X)
&=\mathbb E[
  v^\top Mv-2v^\top MU+U^\top MU
  \mid X] \\
&=v^\top Mv
  -2v^\top M\mathbb E[U\mid X]
  +\mathbb E[U^\top MU\mid X].
\end{aligned}
\tag{3.6}
$$

The final term does not depend on $v$. Therefore

$$
\nabla_v\mathcal R_M
=2M\left(v-\mathbb E[U\mid X]\right),
\tag{3.7}
$$

and the unique minimizer is

$$
v_M^\star(X)=\mathbb E[U\mid X].
\tag{3.8}
$$

The fixed $M$ cancels from the first-order condition because it weights every
possible $U$ at a given $X$ in the same way.

The same conclusion holds if the metric is a positive-definite function $M(X)$ of
the predictor's observed input: after conditioning on $X$, that matrix is fixed. The
property fails when the weight uses target information not determined by $X$, as in
$M(A)$ below.

Thus a fixed positive-definite trajectory metric:

- preserves the population-optimal conditional flow field;
- changes gradient conditioning;
- changes how a finite-capacity model distributes its residual error;
- does not add information absent from the original flow target.

This is the precise sense in which a fixed band loss is principled: it is an
explicit finite-capacity preference that does not redefine the ideal flow field.

### 3.3 Target-dependent metrics do not have this property

Now let the symmetric positive-definite metric depend on the random demonstrated
chunk, $M=M(A)$. Expanding its conditional risk gives

$$
\begin{aligned}
\mathcal R_{M(A)}(v\mid X)
&=\mathbb E[(v-U)^\top M(A)(v-U)\mid X] \\
&=v^\top\mathbb E[M(A)\mid X]v
  -2v^\top\mathbb E[M(A)U\mid X] \\
&\quad+\mathbb E[U^\top M(A)U\mid X].
\end{aligned}
\tag{3.9}
$$

The cross term is the key. In general,

$$
\mathbb E[M(A)U\mid X]
\ne
\mathbb E[M(A)\mid X]\mathbb E[U\mid X],
\tag{3.10}
$$

because $M(A)$ and $U=A-Z$ both depend on the target action $A$. Differentiating
(3.9) gives

$$
\nabla_v\mathcal R_{M(A)}(v\mid X)
=2\mathbb E[M(A)\mid X]v
 -2\mathbb E[M(A)U\mid X].
$$

Setting this gradient to zero gives

$$
v_{M(A)}^\star(X)
=\left(\mathbb E[M(A)\mid X]\right)^{-1}
 \mathbb E[M(A)U\mid X],
\tag{3.11}
$$

assuming the conditional matrix expectation is invertible.

Equation (3.11) reduces to the ordinary conditional mean only in special cases,
such as when $M(A)$ is constant or when $M(A)$ and $U$ are conditionally independent
given $X$.

#### Scalar example

In one dimension, write the target-dependent weight as $m(A)>0$. At one particular
input $X=x$, suppose $U=0$ and $U=1$ are equally likely. Ordinary MSE predicts
$\mathbb E[U\mid X=x]=0.5$. If $m=1$ for the first target and $m=9$ for the second,
then (3.11) gives

$$
v_m^\star(x)
=\frac{\tfrac12(1)(0)+\tfrac12(9)(1)}
       {\tfrac12(1)+\tfrac12(9)}
=0.9.
\tag{3.12}
$$

The weight has not merely changed the numerical scale of the loss. It has moved the
desired prediction toward the more heavily weighted target.

Per-example path, band, dimension, or terminal denominators are instances of
$M(A)$. They deliberately change which demonstrated actions the model represents
most strongly.

That can be useful, but it is importance-weighted imitation—not merely a change of
units.

### 3.4 Dimension normalization and example normalization

These ideas should be separated because they answer different questions.

#### Fixed per-dimension normalization

Let $s_d>0$ be one fixed scale for action dimension $d$, estimated once from the
training corpus or chosen from physical/task semantics. A dimension-normalized
quadratic loss has metric

$$
M_{\mathrm{dim}}
=I_T\otimes
 \operatorname{diag}(s_1^{-2},\ldots,s_D^{-2}).
\tag{3.13}
$$

Because this metric is fixed across examples, (3.8) applies: it preserves the
population-optimal flow field. It changes how finite model capacity and gradients
are allocated across dimensions.

This is a good idea when raw action coordinates have different units or ranges.
However, the current data pipeline already transforms every action dimension using
fixed dataset statistics. Euclidean loss in those normalized coordinates already
corresponds to a fixed per-dimension metric in raw coordinates. Dividing again by
the same kind of scale would apply a second dimension weighting. That is not
necessarily wrong, but it must express a new preference—for example, that gripper
accuracy matters more than its normalized range implies—not be described as merely
making units comparable.

A fixed corpus scale remains a useful control if there is evidence that the current
preprocessing does not yield the intended cross-dimension metric. The scale should
then be stated explicitly: standard deviation, quantile range, typical movement
energy, actuator tolerance, or another quantity. These choices are not
interchangeable.

#### Per-example normalization

Let $q(A)>0$ be a scale computed from the current demonstrated chunk. The scalar
case is

$$
\mathcal L_{\mathrm{example}}(v)
=\mathbb E\left[
  \frac{\|v(X)-U\|_2^2}{q(A)}
\right].
\tag{3.14}
$$

This gives the whole chunk weight $1/q(A)$. If $q(A)$ is motion energy, low-motion
chunks receive more weight than high-motion chunks. That directly addresses the
concern that small, precise demonstrations are overwhelmed by large motions. It is
a coherent objective, but its optimum is the importance-weighted conditional mean
in (3.11), not the unweighted conditional mean.

A still finer version uses one scale per example and dimension:

$$
\mathcal L_{\mathrm{example,dim}}(v)
=\mathbb E\left[
  \frac1D\sum_{d=1}^D
  \frac{\|v_{\cdot,d}(X)-U_{\cdot,d}\|_2^2}{q_d(A)}
\right].
\tag{3.15}
$$

This can prevent a moving arm joint from numerically dominating a nearly static
joint or gripper within the same example. It is also more aggressive: it reweights
each example-dimension block independently, and near-zero $q_d(A)$ produces the
largest weights.

Therefore per-example normalization requires three explicit decisions:

1. **Meaning of the denominator.** Target energy, movement relative to the anchor,
   error of a hold/coast baseline, and actuator tolerance define different tasks.
2. **Behavior near zero.** A floor or shrinkage rule is unavoidable; it determines
   the maximum importance assigned to inactive chunks or dimensions.
3. **Desired estimator.** The changed conditional target must be intentional. The
   benefit is relative accuracy on small motions; the cost is biasing capacity and,
   at the population level, the represented conditional flow toward those motions.

The clean comparison is consequently:

- current dataset-normalized MSE: fixed coordinate metric, no per-example
  importance weighting;
- an additional fixed per-dimension metric: preserves the population target but
  changes cross-dimension priorities;
- a scalar per-example denominator: importance-weights whole demonstrations;
- a per-example, per-dimension denominator: importance-weights individual
  demonstration dimensions.

---

## 4. Replacement versus additive auxiliary

Suppose $M_{\mathrm{band}}$ is the chosen fixed band metric.

### Additive form

$$
L=L_{\mathrm{flow}}+\lambda L_{\mathrm{band}}.
\tag{4.1}
$$

This uses the effective metric

$$
M_{\mathrm{eff}}=I+\lambda M_{\mathrm{band}}.
\tag{4.2}
$$

The scalar $\lambda$ is unavoidable: it defines the actual geometry. Calling the
second term “auxiliary” does not remove that choice.

### Replacement form

$$
L=L_{\mathrm{band}}.
\tag{4.3}
$$

If $M_{\mathrm{band}}$ is positive definite, (3.8) still holds. Replacement has
three advantages:

1. no auxiliary coefficient;
2. the intended geometry is visible directly in the loss;
3. ordinary flow MSE can remain an unmodified diagnostic and control.

Replacement is therefore the cleaner experiment when the proposal is genuinely
“this is the metric we want.” Additive form is appropriate only when the intent is
explicitly “mostly Euclidean MSE, with a smaller secondary preference.”

---

## 5. Three defensible reductions

Let $\mathcal B_1,\ldots,\mathcal B_K$ be the fixed frequency bands justified in
the main note, and let

$$
E_{b,d}(r)
=\frac{1}{|\mathcal B_b|}
  \sum_{k\in\mathcal B_b}|\widetilde r_{k,d}|^2
\tag{5.1}
$$

be mean residual energy in band $b$, dimension $d$. The tilde denotes the
orthonormal DCT-II along the chunk-time axis, as defined in
[action_trajectory_losses.md](action_trajectory_losses.md) §1.

### 5.1 Ordinary coefficient MSE

$$
L_{\mathrm{coef}}
=\frac1D\sum_d\sum_b
  \frac{|\mathcal B_b|}{T}E_{b,d}(r).
\tag{5.2}
$$

Interpretation: every DCT coefficient receives equal weight. Wider bands receive
more total weight because they contain more coefficients. By Parseval, this is
ordinary time-domain flow MSE.

Advantages:

- canonical and parameter-free;
- no additional assumptions;
- simplest optimization target.

Drawback:

- the chosen semantic bands do not receive equal influence; band width determines
  their total contribution.

### 5.2 Equal-band loss

$$
\boxed{
L_{\mathrm{equal\ band}}
=\frac1{KD}\sum_{b,d}E_{b,d}(r).
}
\tag{5.3}
$$

Interpretation: each normalized action dimension and each declared temporal band
receives equal total weight. Coefficients inside a band share that band's weight.

Advantages:

- no dataset statistics beyond existing action preprocessing;
- no auxiliary coefficient when used as the flow loss;
- fixed positive-definite metric, so (3.8) applies;
- directly expresses the reason for introducing bands.

Drawbacks:

- equal band importance is a normative choice;
- narrow bands receive more weight per coefficient than wide bands;
- the conclusion depends on the band partition;
- task regimes within the same band remain mixed.

This is the recommended first alternative to ordinary flow MSE.

### 5.3 Corpus-relative band loss

Define fixed corpus band power

$$
P_{b,d}
=\mathbb E_{a\sim p_{\mathrm{data}}}[E_{b,d}(a)].
\tag{5.4}
$$

Then

$$
L_{\mathrm{corpus\ relative}}
=\frac1{KD}\sum_{b,d}
  \frac{E_{b,d}(r)}
       {\max(P_{b,d},\phi_{b,d})}.
\tag{5.5}
$$

Interpretation: target the same average fractional accuracy in every band and
dimension, measured relative to how much that band normally moves in the corpus.

Because $P_{b,d}$ and $\phi_{b,d}$ are fixed dataset constants, this remains a
fixed quadratic metric and preserves (3.8).

Advantages:

- addresses large corpus-scale differences that equal-band averaging leaves
  untouched;
- statistics are estimated at the interpretable band/dimension level, not at every
  coefficient or example;
- still does not reweight examples conditionally on their own targets.

Drawbacks:

- low-power or nearly inactive bands receive the largest weights;
- the floor $\phi_{b,d}$ materially defines the metric;
- a single corpus average can hide task-specific regimes;
- existing action normalization and this band normalization can partly duplicate
  each other;
- fractional accuracy is not automatically task importance.

This is a valid ablation, but not the default recommendation.

### 5.4 Per-example relative loss

Equation (5.6) is the band-and-dimension-resolved version of (3.15): its metric
depends on the current example separately for every $(b,d)$ block.

$$
L_{\mathrm{example\ relative}}
=\mathbb E\left[
  \frac1{KD}\sum_{b,d}
  \frac{E_{b,d}(r)}
       {\max(E_{b,d}(a),\phi_{b,d})}
\right].
\tag{5.6}
$$

Interpretation: every example receives comparable fractional accuracy in every
band.

This is the strongest response to “small motions are ignored,” but it is also the
most consequential:

- it changes the population flow target through (3.11);
- it emphasizes the smallest denominators;
- its floor controls a potentially large fraction of the gradient;
- it can alter the relative frequency of demonstrated action modes.

Use it only as deliberate importance weighting, not as the default flow metric.

---

## 6. Trade-off summary

| reduction | statistics | extra coefficient | population flow optimum | primary meaning |
|---|---|---:|---|---|
| ordinary MSE | none | no | preserved | equal weight per coefficient |
| fixed dimension weights | fixed $s_d$ | no, if replacement | preserved | chosen cross-dimension accuracy |
| equal band | band definitions | no, if replacement | preserved | equal weight per temporal band |
| corpus-relative band | fixed $P_{b,d}$, floors | no, if replacement | preserved | equal average fractional band accuracy |
| scalar per-example relative | current target, floor | no, if replacement | **changed** | equal fractional accuracy per chunk |
| per-example dimension/band relative | current target, floors | no, if replacement | **changed** | equal fractional accuracy per block |
| MSE + any auxiliary | depends on auxiliary | **yes** | preserved only if auxiliary metric is fixed | compromise geometry |

The first decision should be semantic, not numerical:

> Do we want equal coefficient accuracy, an explicit dimension metric, equal band
> accuracy, equal corpus-relative band accuracy, or deliberate importance weighting
> of small-motion examples?

Only after that decision should weights or floors be estimated.

---

## 7. Recommendation for the flow expert

Run the following controlled comparison:

1. ordinary flow MSE, equation (5.2);
2. equal-band flow loss, equation (5.3);
3. optionally, corpus-relative band loss, equation (5.5).

Use each as the complete flow loss rather than adding it to ordinary MSE. Keep the
other reductions as diagnostics on the same validation batches.

This experiment answers two clean questions:

1. Does equalizing semantic temporal bands improve the finite-capacity action
   expert?
2. If so, does additional corpus-relative normalization help, or merely amplify
   inactive bands?

Do not combine per-example normalization with the band change in the first
experiment. Its effect is qualitatively different and should be isolated after the
fixed-metric comparison. This is sequencing, not a rejection of the idea: if the
measured failure is poor **relative** accuracy on small motions, per-example
normalization is the objective that addresses it most directly.

For that experiment, compare the least granular variants first:

1. one scalar denominator for the whole chunk, equation (3.14);
2. one denominator per chunk and action dimension, equation (3.15);
3. only then, one denominator per chunk, dimension, and band, equation (5.6).

This ordering reveals whether the benefit comes from reweighting demonstrations,
balancing dimensions within a demonstration, or balancing temporal bands. Report
the denominator distribution, fraction of samples hitting the floor, and effective
weight quantiles; otherwise the result cannot be interpreted.

Likewise, do not add another dimension normalization automatically. First inspect
dimensionwise residuals and rollout failures under the existing dataset transform.
If a dimension is underweighted, test a fixed dimension metric as its own ablation
before making it example-dependent.

The quadratic loss should be computed directly on the velocity residual
$r_\theta$. The one-step clean-action error is

$$
\widehat a-a=(1-\tau)r_\theta,
\tag{7.1}
$$

so reconstructing and dividing by $(1-\tau)^2$ only adds numerical conditioning
problems near $\tau=1$.

---

## 8. Recommendation for FAST

FAST is not a continuous regressor. Its base objective is autoregressive token
cross-entropy:

$$
L_{\mathrm{FAST,CE}}
=-\mathbb E\sum_j
\log p_\psi(s_j\mid c,s_{<j}).
\tag{8.1}
$$

This objective handles token identity, BPE length, and autoregressive boundary
propagation. A bandwise coefficient loss cannot replace it without losing those
properties.

### 8.1 First use bands as diagnostics

Using the existing teacher-forced coefficient marginals, report by band and
dimension:

- ordinal score;
- squared error of the coefficient mean;
- expected squared coefficient error;
- correct-token mass and covered mass;
- error conditioned on gripper switching or other important events.

These diagnostics determine whether FAST CE actually has a band-specific failure.
They do not require a new training coefficient.

### 8.2 If an ordinal auxiliary is justified

A band-balanced ordinal score can mirror (5.3):

$$
L_{\mathrm{FAST,ord\ band}}
=\frac1{KD}\sum_{b,d}
  \frac1{|\mathcal B_b|}
  \sum_{k\in\mathcal B_b}
  \ell_{\mathrm{ord}}(q_{k,d},c^\star_{k,d}).
\tag{8.2}
$$

The total FAST objective would be

$$
L_{\mathrm{FAST}}
=L_{\mathrm{FAST,CE}}
+\lambda_{\mathrm{ord}}L_{\mathrm{FAST,ord\ band}}.
\tag{8.3}
$$

Unlike the flow replacement, $\lambda_{\mathrm{ord}}$ is unavoidable because CE
must remain. There is no universal theoretical value. If this experiment becomes
necessary:

1. normalize both losses by their documented reductions;
2. report the auxiliary/base gradient norm ratio on VLM parameters;
3. sweep a small fixed set of coefficients;
4. require rollout or representation-level improvement, not merely lower ordinal
   loss.

The current teacher-forced marginal is conditioned on the demonstrated BPE
boundary. It does not model the downstream shift caused by a wrong-length token.
That limitation should remain explicit.

### 8.3 Quadratic FAST caution

If a quadratic coefficient score is tested, use

$$
\mathbb E_{c\sim q}[(c-c^\star)^2],
\tag{8.4}
$$

not

$$
(\mathbb E_q[c]-c^\star)^2.
\tag{8.5}
$$

The second expression ignores predictive variance and can be zero for a bimodal
distribution whose mean equals the target.

### 8.4 Normalization choices for FAST are separate

The flow arguments above do not transfer mechanically to token cross-entropy.

A scalar per-example weight can multiply the complete FAST sequence loss:

$$
L_{\mathrm{FAST,example}}
=-\mathbb E\left[
  w(A)\sum_j\log p_\psi(s_j\mid C,s_{<j})
\right].
\tag{8.6}
$$

This is valid importance-weighted maximum likelihood. At the population level it
fits the reweighted data distribution

$$
p_w(A\mid C)
\propto w(A,C)p_{\mathrm{data}}(A\mid C),
\tag{8.7}
$$

rather than the original demonstration distribution. Thus weighting small-motion
examples in FAST is possible, but it is a separate modeling choice from applying
the same idea to the flow expert.

Per-dimension or per-band weights do not attach cleanly to BPE cross-entropy because
a BPE token need not correspond one-to-one with a single coefficient, dimension, or
band. Those weights belong naturally in a coefficient-level auxiliary such as
(8.2), where their mapping is explicit. They then require
$\lambda_{\mathrm{ord}}$ and should be justified by FAST-specific diagnostics.

---

## 9. Other defensible directions

These alternatives answer different questions and should not be presented as
variations of the same band loss.

| alternative | motivation | genuinely new information? | principal cost |
|---|---|---:|---|
| execution-probability time weighting | later chunk actions may not execute | no; fixed metric | must measure replanning/execution statistics |
| event-conditioned weighting | grasp/contact transitions matter more | no; importance weighting | labels and target-distribution shift |
| kinematic/actuator metric | Cartesian error, torque, jerk, or hardware stress matters | sometimes | robot model and cross-robot portability |
| multi-step flow rollout | one-step residual misses composition error | **yes** | multiple expert evaluations and harder gradients |
| learned trajectory features | task-relevant similarity is nonlinear | **yes** | learned model, stability, validation burden |
| robust residual penalty | outliers should have bounded influence | changes statistical estimator | additional scale/shape choice |

The fixed band metric is attractive because it is the cheapest option that changes
finite-capacity trajectory priorities while leaving the ideal flow target intact.
It should not be expected to solve failures that require task labels, dynamics, or
multi-step composition.

---

## 10. Minimal experimental plan

### Phase A: no new FAST training term

Train:

1. flow MSE;
2. equal-band flow replacement;
3. corpus-relative band replacement only if its fixed weights are numerically
   reasonable.

Keep FAST CE unchanged in all three.

### Phase B: diagnostics

For every checkpoint, report:

- flow residual under all three reductions;
- residual by band and action dimension;
- FAST ordinal and coefficient errors by the same bands;
- rollout success and failure category;
- gripper event accuracy and timing;
- executed-action chatter or jerk;
- correlation of each offline metric with rollout success.

### Phase C: only if normalization diagnostics justify it

If failures concentrate in particular action dimensions, test a fixed dimension
metric first. If failures instead concentrate in small-motion examples when measured
relatively, compare the scalar, per-dimension, and finally per-band per-example
denominators in the order given in §7. Do not change the band metric in the same
ablation.

### Phase D: only if FAST diagnostics justify it

Test the band-balanced ordinal auxiliary with an explicit coefficient sweep.

### Acceptance criterion

Prefer the simplest objective whose validation metric predicts and improves real
rollout behavior. A more elaborate loss that only improves its own value has not
earned its complexity.

---

## 11. Decisions still requiring evidence

1. Are the existing action-normalization coordinates already the intended metric
   across all dimensions and robot sources?
2. How often are all $T=30$ actions executed before replanning?
3. Are partial/padded chunks common enough to complicate the DCT reduction?
4. Do the proposed bands remain meaningful for every control rate and chunk
   horizon?
5. Does equal-band error correlate with rollout success better than ordinary MSE?
6. Does corpus-relative normalization help precise motions or mostly amplify
   inactive bands?
7. After existing dataset normalization, are rollout-relevant errors still
   concentrated in particular action dimensions?
8. Are small-motion failures better predicted by relative error than by absolute
   error, and what fraction of examples would hit a proposed denominator floor?
9. Does FAST exhibit a band-specific error after controlling for token frequency
   and BPE length?

These questions select among the objectives above. They should not be hidden inside
additional normalization constants.
