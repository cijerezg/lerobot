# On Mean Squared Error for Imitation Learning

In our imitation-learning setup, we learn a velocity field. The model does not
directly emit a finished action chunk during training. Instead, it sees a noisy
point between Gaussian noise $\varepsilon$ and the demonstrated action chunk $a$:

$$
x_t = (1-t)\varepsilon + ta,
$$

and learns the velocity that carries that point toward the demonstration:

$$
v^* = a-\varepsilon.
$$

Given a predicted velocity $v_\theta$, we can reconstruct the model's estimate of
the clean action in one step:

$$
\hat a = x_t + (1-t)v_\theta.
$$

If the velocity is exact, then $\hat a=a$. At inference time we use the same
velocity field repeatedly, starting from noise and following it with a short Euler
integration until it produces an action chunk.

The usual training objective is mean squared error on the velocity:

$$
\mathcal L_{\mathrm{flow}}
= \operatorname{mean}\left[(v_\theta-v^*)^2\right].
$$

This is a sensible objective. It is simple, stable, dense, and gives a gradient for
every action dimension at every point in the chunk. The problem is not that MSE is
wrong. The problem is that MSE has a very limited idea of what it means for an
action to be good.

## Equal MSE does not mean equal behavior

MSE treats every scalar error independently, squares it, and averages the results.
Once it has done that, the ordering and sign pattern of those errors have
disappeared.

Consider a four-step, one-dimensional example. Suppose the target is zero and two
predictions have the following errors:

$$
e_A = [1, 1, 1, 1], \qquad
e_B = [1, -1, 1, -1].
$$

Both receive exactly the same MSE:

$$
\operatorname{MSE}(e_A)=\operatorname{MSE}(e_B)=1.
$$

But they are not the same action. The first is a smooth, constant offset. The
second changes direction at every step. A robot executing the second chunk would
oscillate, even though the scalar loss says it is no worse than the first.

The same ambiguity survives action reconstruction. At a fixed flow time,

$$
\hat a-a=(1-t)(v_\theta-v^*),
$$

so both examples are merely multiplied by the same $(1-t)$ factor. Their raw
reconstructed-action MSE is still equal, while the trajectories still have very
different shapes. For the example above, the adjacent-difference error of the
constant offset is zero, whereas the oscillating error has adjacent differences
of magnitude two and a shape MSE of four.

This is the main blind spot: equal velocity MSE does not imply equal action
quality. MSE cannot tell a smooth bias from a zigzag, it does not give the last
target any special importance, and it does not directly ask whether the robot is
moving in the intended direction.

## MSE favors large motions

There is a second, less obvious problem. Absolute MSE is measured in squared
action units. For a comparable fractional mistake, a large-displacement
demonstration produces a larger error and gradient. Such examples can dominate a
batch even when the policy has
already learned their broad shape.

Subtle actions have the opposite failure mode. Imagine a delicate correction
whose amplitude is only $0.01$. A prediction that reverses the whole correction
can still have a tiny absolute MSE because every number involved is tiny. The
trajectory may be qualitatively wrong—closing instead of opening, drifting left
instead of right, or alternating around a contact—but its contribution can be
buried beneath a large arm sweep elsewhere in the batch.

In other words, absolute MSE answers, “How many normalized action units were we
wrong by?” It does not answer, “How wrong were we relative to the motion this
example required?” Those are different questions. For imitation learning, the
second one is often closer to what matters.

## The auxiliary trajectory losses

We therefore keep the flow MSE and add four losses on the reconstructed action
chunk $\hat a$. They do not require another model forward or an ODE solve: the
one-step reconstruction is already available from $x_t$ and the predicted
velocity. The metrics are computed at every sampled flow time and then averaged
per training example.

Let $h$ be the **hold-still trajectory**: the action chunk that repeats the
robot's current pose and makes no motion. In anchor-encoded action space this is
the zero trajectory. This gives each demonstration its own meaningful baseline.

### 1. Relative path loss: did we reconstruct the chunk?

The path term measures ordinary MSE over every valid time step and action
dimension, then compares it with the error made by holding still:

$$
R_{\mathrm{path}} =
\frac{\operatorname{MSE}(\hat a,a)}
{\max\left(\operatorname{MSE}(h,a),\tau_{\mathrm{path}}\right)}.
$$

This term reads naturally: zero is exact, one is no better than doing nothing,
and a value above one is worse than holding still.

Raw path MSE is not new information. At a fixed $t$ it is just the velocity MSE
scaled by $(1-t)^2$. Its value here is the **hold-relative normalization and
gating**. A small but badly reconstructed motion can now matter as much as a large
motion with the same relative error.

### 2. Relative shape loss: did we move the right way over time?

The shape term applies MSE to adjacent differences:

$$
R_{\mathrm{shape}} =
\frac{
\operatorname{MSE}\left(\Delta\hat a,\Delta a\right)
}{
\max\left(\operatorname{MSE}(\Delta h,\Delta a),\tau_{\mathrm{shape}}\right)
}.
$$

Because a constant offset disappears under $\Delta$, this loss separates a
trajectory's shape from its absolute bias. It is the term that distinguishes the
smooth `[1, 1, 1, 1]` error from `[1, -1, 1, -1]`. Reversals, jitter, and badly
timed changes become expensive even when their raw path MSE looks harmless.

### 3. Relative terminal loss: did we arrive at the right place?

An action chunk is often judged by where it ends: whether the gripper reached the
object, whether the wrist completed its rotation, or whether the arm stopped at
the intended pose. Plain path MSE dilutes a terminal miss across the whole chunk.
With a 30-step horizon, an error that appears only at the final step contributes
only one thirtieth of the corresponding per-step error.

The terminal term gives that last target its own objective:

$$
R_{\mathrm{terminal}} =
\frac{
\operatorname{MSE}(\hat a_T,a_T)
}{
\max\left(\operatorname{MSE}(h_T,a_T),\tau_{\mathrm{terminal}}\right)
}.
$$

It uses the same interpretation as the path term: zero is exact and one is no
better than never moving.

### 4. Terminal direction loss: did we at least go the right way?

Finally, we compare the predicted and demonstrated final displacements from the
hold pose:

$$
R_{\mathrm{direction}} = 1-\cos\left(
\hat a_T-h_T,\ a_T-h_T
\right).
$$

This is zero when the two displacements are aligned, one when they are
perpendicular—or when the prediction does not move—and two when they point in
opposite directions. Unlike MSE, it ignores displacement magnitude and asks only
about intent. It is especially useful when the action is small: moving a little
in the wrong direction should not look successful merely because the numbers are
small. The term is undefined and omitted when the demonstrated final
displacement is zero.

## How the losses are combined

The auxiliary terms are additions to the base objective, not replacements:

$$
\mathcal L = \mathcal L_{\mathrm{flow}} +
\sum_j \lambda_j\,\mathbf 1[R_j>\gamma_j]R_j.
$$

Each term has its own weight $\lambda_j$ and optional threshold $\gamma_j$. The
threshold is a hard gate, not a margin: below it the term contributes zero; above
it the full value contributes. The base flow MSE continues to train every sample
at every flow time, while the auxiliary losses spend extra gradient on examples
that are still behaviorally poor.

In the current configuration, all four weights are `0.05` and all four gates are
`0.2`. Thus a relative term turns on when its reconstruction error exceeds 20% of
the error made by holding still. Small denominator floors keep nearly stationary
chunks from producing unbounded ratios. Padded action dimensions and padded
timesteps are excluded from every calculation.

## What this changes

The original MSE gives the model one undifferentiated instruction: make every
velocity component numerically close to its target. The auxiliary losses turn
that into four more useful questions:

1. Is the whole reconstructed chunk better than doing nothing?
2. Does it have the right temporal shape?
3. Does it finish at the right target?
4. Does its final motion point in the right direction?

Together, these losses preserve the stability of flow-matching MSE while adding
the structure that MSE averages away. They also place large and subtle actions on
a shared, task-relative scale. A millimeter-scale correction can no longer hide
simply because a different example moves the arm across the table.

This still does not make the objective identical to task success. The losses live
in normalized joint-action space; they do not directly know about collisions,
contacts, object state, or Cartesian distance. They are better understood as
guides for MSE: inexpensive signals that tell the optimizer which equal-MSE
solutions look more like the demonstrated behavior.
