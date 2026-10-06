# On Mean Squared Error for Imitation Learning

One common objective function for robot learning is the learning of a velocity
field that transports noise to actions. The policy is not trained to emit an
action chunk directly: it sees a point between Gaussian noise $z$ and a
demonstrated chunk $a$,

$$
x_t = (1-t)\,z + t\,a ,
$$

and predicts the velocity that carries that point to the demonstration,
$v^* = a - z$. From a predicted velocity $\hat v$ the clean action is recovered
in one step,

$$
\hat a = x_t + (1-t)\,\hat v ,
$$

which equals $a$ when the velocity is exact. At inference the same field is
integrated from noise by a short Euler scheme.

The training objective is the squared error of the predicted velocity,

$$
L_{\mathrm{flow}} = \mathrm{E}\left[\ \frac{1}{N}\sum_{k,d}
\left(\hat v_{k,d} - v^*_{k,d}\right)^2\ \right] ,
$$

over demonstrations, noise, and flow time $t$, where $k$ indexes the steps of the
chunk, $d$ the action dimensions, and $N$ the number of valid pairs. Two
properties of this objective are described below, followed by four auxiliary
terms on the reconstructed chunk.

## 1. Equal loss does not imply equal trajectory

Consider a one-dimensional chunk of four steps and two predictions with velocity
residuals

$$
r_A = [\,1,\ 1,\ 1,\ 1\,], \qquad
r_B = [\,1,\ -1,\ 1,\ -1\,] ,
$$

both of mean squared value one. The residual is on the velocity, but the
reconstruction is affine in it: at fixed $t$, $\hat a - a = (1-t)(\hat v - v^*)$.
Each pattern therefore reaches the action unchanged in form and scaled by
$(1-t)$, so the two reconstructions also have equal squared error while differing
in shape — the first offset from the demonstration by a constant, the second
alternating about it.

On adjacent differences they separate: the offset gives zero, the alternating
pattern gives adjacent differences of two and an error of four, in velocity
units. Equal flow loss thus does not imply equal reconstruction. The objective
also assigns no distinct role to the final step, and does not measure the
direction of the motion.

## 2. The objective is in absolute units

The error is measured in squared action units, so for the same *fractional*
mistake a large-displacement demonstration yields a larger gradient than a small
one. Conversely, a correction of amplitude $0.01$ that is predicted with the
wrong sign still incurs a small absolute error. The objective reports how many
normalized action units the prediction was wrong by, not how wrong it was
relative to the motion the example required.

## 3. Auxiliary terms

The flow loss is kept and four terms are added on the reconstruction $\hat a$,
which is already available from $x_t$ and $\hat v$ — no extra forward pass or ODE
solve. Each is evaluated at every sampled flow time and averaged per example.

Let $h$ be the **hold-still chunk**, which repeats the current pose and never
moves (the zero trajectory in anchor-encoded action space). Each term divides the
error of the prediction by the error the same demonstration assigns to $h$. That
denominator is the demonstration's own scale, measured in the matching quantity —
total excursion for path, per-step motion for shape, final displacement for
terminal — so all terms read alike: zero is exact, one is no better than freezing
the arm, above one is worse than freezing it. Sums run over the valid entries of
the term in question, $N$ is their number, and $T$ is the last step.

**Path.** The ordinary squared error over the chunk, expressed as a fraction of
what standing still would cost. It answers whether the reconstruction as a whole
is closer to the demonstration than doing nothing.

$$
R_{\mathrm{path}} =
\frac{\frac{1}{N}\sum_{k,d}\left(\hat a_{k,d}-a_{k,d}\right)^2}
{\frac{1}{N}\sum_{k,d}\left(h_{k,d}-a_{k,d}\right)^2} .
$$

The numerator is the flow residual scaled by $(1-t)^2$; the denominator is what
the term adds.

**Shape.** Two chunks can sit equally far from the demonstration on average while
one tracks its motion and the other jitters around it. Differencing removes any
constant offset and leaves only how the motion evolves from step to step:

$$
R_{\mathrm{shape}} =
\frac{\frac{1}{N}\sum_{k,d}
\big[(\hat a_{k,d}-\hat a_{k-1,d})-(a_{k,d}-a_{k-1,d})\big]^2}
{\frac{1}{N}\sum_{k,d}
\big[(h_{k,d}-h_{k-1,d})-(a_{k,d}-a_{k-1,d})\big]^2} ,
\qquad k \ge 2 .
$$

This separates the two residuals of Section 1, which the path term scores
identically.

**Terminal.** Where the chunk ends is often what the task turns on — the gripper
reaches the object or it does not. The path term dilutes a miss at the last step
across the whole chunk, a thirtieth of its per-step size at a horizon of 30, so
the last step is also scored on its own:

$$
R_{\mathrm{terminal}} =
\frac{\frac{1}{N}\sum_{d}\left(\hat a_{T,d}-a_{T,d}\right)^2}
{\frac{1}{N}\sum_{d}\left(h_{T,d}-a_{T,d}\right)^2} .
$$

**Direction.** The three terms above still reward a small error over a correct
intent: a prediction that moves a little the wrong way scores better than one
that moves the right way and overshoots. The cosine asks only where the final
displacement points, and is unchanged if either displacement is rescaled. With
$u_d = \hat a_{T,d} - h_{T,d}$ and $w_d = a_{T,d} - h_{T,d}$,

$$
R_{\mathrm{direction}} = 1 -
\frac{\sum_{d} u_d\,w_d}
{\sqrt{\sum_{d} u_d^2}\ \sqrt{\sum_{d} w_d^2}} .
$$

This is zero for parallel displacements, one for orthogonal ones or when the
prediction does not move, and two for antiparallel ones. It is omitted when the
demonstrated final displacement is zero.

## 4. Combination

The terms are added to the base objective, not substituted for it:

$$
L = L_{\mathrm{flow}} + \sum_j w_j\, G_{g_j}(R_j),
\qquad
G_g(R) =
\begin{cases}
R, & R > g, \\
0, & \text{otherwise.}
\end{cases}
$$

The threshold $g_j$ is a hard gate, not a margin: below it the term contributes
nothing, above it its full value. All four weights are `0.05` and all four gates
are `0.2`, so a term activates once its error exceeds 20% of the hold-still
error.

## 5. Discussion

The flow loss states one requirement, that each velocity component be close to
its target. The four terms add the error over the chunk, the error in its
adjacent differences, the error at its final step, and the angle of the final
displacement, each relative to the same example's hold-still baseline. The first
three are the flow residual under different weightings; the fourth is independent
of magnitude.

They are not a measure of task success: they live in normalized joint-action
space and refer to nothing outside it — no collisions, contacts, object state, or
Cartesian distance. They are inexpensive statistics that distinguish
reconstructions the flow loss scores equally.
