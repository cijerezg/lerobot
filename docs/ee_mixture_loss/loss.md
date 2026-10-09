# The loss

**Status:** proposal, 2026-10-08. Settled: the model predicts joints and hand pose, both as
anchor deltas, through the FAST tokens (backbone) and the flow expert, knowledge insulation
stays. Open: how the two losses are coupled, and the orientation loss.

## Symbols

- $j$: the joint target (anchor delta, as today). $\hat j$: the prediction.
- $h = (t, r, g)$: the hand target. $t$ translation of the hand from the anchor, $r$ the
  rotation from the anchor orientation as a rotation vector (axis times angle), $g$ aperture.
  $\hat h$: the prediction.
- $F(\hat j)$: the hand that the predicted joints would produce. Forward kinematics of
  $s_0 + \hat j$, expressed as a delta from the anchor hand, in the same coordinates as $h$.
- $\mathrm{sg}(\cdot)$: stop gradient, the value is used, nothing flows back into it.

Both targets are normalized; $j$ per robot, $h$ with one pooled row for all robots.

## Three ways to combine the losses

**Additive.**

$$L = \|\hat j - j\|^2 + \|\hat h - h\|^2$$

The hand term aligns representations (shared last layer, same gradient direction for the same
motion on every robot). Nothing ties $\hat j$ to $\hat h$: both are scored against ground truth
and the network can satisfy both with two separate read-outs. This is the floor, and it is
already worth having.

**Product.** The form $L_j + L_h \cdot (\text{a joint term})$, or the reverse. Write it with the
joint error as the factor:

$$L = \|\hat j - j\|^2 + \|\hat h - h\|^2 \cdot \big(1 + \|\hat j - j\|^2\big)$$

The gradient on the hand is scaled by the joint error and the gradient on the joints picks up
a term proportional to the hand error. Both are still measured against ground truth, so the
product reweights samples (a sample with bad joints pushes the hand harder, and the other way
around) but never asks $\hat j$ and $\hat h$ to agree with each other. At the optimum both are
zero and the factor is 1. It does not couple; it only changes which samples count. The same is
true of $L_h + L_j \cdot (\text{hand term})$ and of putting the factor on both sides.

**Consistency.** A term in which both predictions appear:

$$L = \|\hat j - j\|^2 + \|\hat h - h\|^2 + \lambda\,\big\| F(\hat j) - \mathrm{sg}(\hat h) \big\|^2$$

The third term says: the hand that the predicted joints produce must be the hand the model
predicted. This is the only form in which the joints are asked to follow the hand, and it is
what "a term that measures the joints" has to be: the joints measured in hand space, against
the model's own hand, not against the data.

## Which way the gradient goes

The third term has two sides and the choice of where to cut the gradient decides what gets
learned.

- No cut: $\hat j$ and $\hat h$ pull on each other. The hand bends toward whatever the joints
  do, and the joints are robot-specific, so robot-specific information leaks into the hand and
  the alignment from the hand term is weakened.
- Cut on the joints: the hand becomes a read-out of the joints. Same leak, worse.
- Cut on the hand: the joints chase the hand, the hand is shaped only by its own ground-truth
  term in shared coordinates. This is the one.

So the direction is fixed by the alignment argument, not by taste.

## What the third term is, mechanically

Its gradient with respect to the joints is the Jacobian of forward kinematics, transposed, times
the hand residual. That is one Gauss-Newton step of inverse kinematics toward the predicted hand.
Training with this term is inverse kinematics done by gradient descent, one step per sample, on
each robot's own joint slots. The joint-space alternative, run a real IK solver on
$\mathrm{sg}(\hat h)$ to get a joint target and penalize $\|\hat j - \mathrm{IK}(\hat h)\|^2$,
is the same thing with the iterations done per sample, and the solver has to pick a posture.
The hand-space form picks it for free: it moves the joints from where they are.

The joint ground-truth term stays because the hand does not determine seven joints. The data
decides the posture; the third term only enforces the hand.

## Orientation

The rotation part of $h$ is the rotation from the anchor orientation $R_0$ to the step
orientation $R_k$, written as a rotation vector:

$$r = \log\!\big(R_0^{\top} R_k\big)$$

in the hand frame (use $R_k R_0^{\top}$ for the base frame; same angle, different axis
coordinates; use the same frame as $t$). Over a one-second chunk the angle is small, so $r$ is
continuous, has no wrap, and the ordinary squared error on it is the geodesic distance up to
second order. The flow loss and the FAST tokens take $r$ as three plain numbers, normalized like
the rest.

The consistency term compares two rotations and should do it as rotations, not as vectors:

$$\big\| R_F - \hat R \big\|_F^2 = 4\,(1 - \cos\theta) \approx 2\theta^2,$$

with $R_F$ the rotation from forward kinematics of the predicted joints, $\hat R$ the rotation
from the predicted $\hat r$ (no gradient), and $\theta$ the angle between them. No singularity
anywhere, and it equals the squared angle where it matters. Translation and aperture use the
squared error.

## Under flow matching

The predictions above are the terminal estimates of the flow, the one-step guess of the clean
chunk from the current noisy chunk. The first two terms are the ordinary flow losses on the two
blocks. The third term is evaluated on the terminal estimates and only for flow times at or
above one half, where the guess is a single chunk and not an average; below that the hand of an
averaged joint estimate is not a meaningful quantity. On the FAST side the backbone gets the
two token losses and no third term, since tokens carry no gradient; its alignment comes from
the hand tokens being the same tokens on every robot.

## Weights

Both targets are normalized, so the hand term takes weight 1. The consistency weight $\lambda$
starts at 0.1; the papers that tuned this kind of term found 0.05 to 0.2 useful and 0.5
harmful.

## The objective

$$L = \|\hat j - j\|^2 + \|\hat h - h\|^2 + \lambda\,\big\| F(\hat j) - \mathrm{sg}(\hat h)\big\|^2$$

Joints against the data, hand against the data in shared coordinates, and the joints' hand
against the predicted hand with the gradient flowing into the joints only.
