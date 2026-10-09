# Thoughts before the literature

**Status:** discussion, 2026-10-08. Written before the literature search in
[literature.md](literature.md); the proposal itself is [proposal.md](proposal.md).

Three principles were set:

1. The model outputs joints, always.
2. A shared quantity (the hand: pose and aperture) should bridge robots.
3. The real question is how to make the model learn useful kinematics and not a shortcut.

Principles 1 and 2 are settled. Everything below is about 3.

## Which kinematics the model has to learn

Be precise about the direction. Forward kinematics (joints to hand) we know and can compute
inside the loss; the model never has to learn it. What the model has to learn, and cannot be
given, is the inverse on each robot: hand motion plus current configuration to joint delta,
with the posture habits of that robot's demonstrations, because a 7-joint arm has a family of
joint solutions per hand pose and the demos pick one. "Learn forward kinematics" in practice
means "learn the inverse map on ReBot where ReBot has moved". The FK consistency loss is the
supervision for that inverse. It is the right loss, but it only says what the joints must
achieve, not how they are computed.

## Why predicting both is not enough on its own

Two outputs from one representation can be two independent read-outs. On the training data
both are supervised to the truth, so the relation hand = FK(joints) holds automatically and the
network is never asked to compute one from the other. The model can map pixels to joints per
robot, map pixels to hand per robot, and satisfy both losses with nothing shared in the
computation.

Generating both in one flow does couple them, but only statistically. At each flow time the
joint velocity can read the current hand guess or the observation; it uses whichever is easier.
At small flow time the hand guess is noise, so the observation wins, and small flow time is
where the mode of the chunk is chosen. So the joint-flow coupling helps late, in refinement,
and not where it matters. The consistency loss adds pressure toward agreement, but agreement
is also reachable by two correct shortcuts.

A shortcut is abandoned only when there are training samples it fails on. During ordinary
training there are none: every ReBot sample has ReBot joints, so the direct map is always
available. The hand route is only needed at transfer time, which the model never sees. That is
the whole problem in one sentence, and it says what the fix has to be: create training samples
on which the direct map fails and the hand route is the only way to the joints.

## Three ways to make the hand route necessary

In order of how hard they force it.

**Give the joint path the hand as input.** Teacher forcing: during training the joint part of
the chunk is generated with the true hand target available as conditioning; at inference it
gets the model's own hand guess. The joint path's job becomes the inverse map plus posture,
which is a much easier problem than pixels to joints, so the network takes it. Exposure bias
(true hand in training, predicted hand at test) is handled the usual way, by adding noise to
the hand input during training. This does not forbid the joint path from also reading the
image, which it needs for posture near obstacles. It makes the hand the cheap route rather
than the only route.

**Drop the image for the joint path some of the time.** With probability $p$ the joint part
sees only the hand input, the current state and the robot's name. On those samples the only way
to get the joints right is the inverse map. The rest of the time the image is there for
posture. At inference the same switch becomes a mode: run the joint path without the image when
the consistency check says the image route is misleading. This is the strong form of option 3
in the proposal, softened so that vision is not lost.

**Swap the output robot.** Take a Franka episode, keep its images and language, name ReBot in
the prompt, give ReBot's state, and ask for the ReBot joints that realize the same relative hand
motion, computed offline by inverse kinematics. No pixels-to-joints map solves this, because the
pixels show a Franka. The only route is: read the relative hand motion from the demonstration,
realize it from ReBot's own state. Relative targets make this consistent (the motion "5 cm left
and down, close" is valid for any hand), and it is the training-time version of exactly the
transfer we want. It needs a feasibility check (the motion must fit ReBot's workspace from the
chosen start state) and the IK must draw postures the way ReBot's demos do. The image and the
state disagree about which robot is present; that is the point, the joint path has to stop
trusting the image for identity.

The three are not exclusive. The first is cheap and should be on from the start. The second is a
knob. The third is data work and is the one that actually rehearses transfer.

## Where the target lives matters more than its weight

If the backbone is trained on joint losses it will store robot-specific joint knowledge, and the
shortcut lives in the representation, below any trick applied in the expert. If the backbone is
trained only on the hand (gradient assignment, option 4), the representation can only hold
hand-space task knowledge, and the shortcut can only form inside the expert, which is small and
scored in hand space. This is a bigger lever than any loss weight and costs nothing in data.

## How to tell a shortcut from a coverage failure

Two signals, both cheap.

The consistency gap at inference: the hand the executed joints produce (FK) against the hand the
model generated. Low on training tasks and high on a transfer task means the inverse map was
asked outside where it was trained. Low everywhere while transfer fails means the hand route is
not driving the joints.

The embodiment-swap probe that already exists: change the robot's name in the prompt on the same
frame. Under the hand route the hand output should stay put and the joint output should move a
lot, because the inverse map differs per robot. Under the shortcut neither moves much. That is a
direct mechanism test, not a loss curve.

## What I expect, honestly

Options 1 and 2 (shared output plus FK consistency) will lower the hand loss and probably
improve in-distribution joint accuracy on ReBot, because the Jacobian gradient is real
information the joint loss lacks. I do not expect them to produce transfer by themselves. The
teacher-forced hand input is where I expect the mechanism to appear, and the output-robot swap
is where I expect it to become reliable. The gradient assignment is the cheapest insurance
against the representation-level shortcut and I would turn it on early.

The thing I am least sure about is posture. Every forcing mechanism makes the joint path depend
on the hand and the state more and on the image less, and posture (elbow placement, approach
from the side of an obstacle) is partly visual. The image dropout rate and the inference mode
switch are the controls for that trade, and they need to be measured on ReBot, not argued.
