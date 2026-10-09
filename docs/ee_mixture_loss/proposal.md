# End-effector mixture loss

**Status:** proposal, 2026-10-08. Discussion only, nothing implemented.

## The idea

Joints are per robot. The same motion (move the hand 1 cm left, tilt it 20 degrees, close to
4 cm) is a different joint vector on every arm, and nothing in training says the two are the
same thing. Pixels and language are shared; the output is not. So the model learns each task
once per robot and has no reason to notice that a task it saw on a Franka is the same task on
ReBot.

The hand is the shared thing: where it is, how it is oriented, how open it is. Every robot's
joints map to it by forward kinematics, which we know. The proposal is to give the model the
hand as a target, in the same units on every robot, and to build the loss so that the joints
are produced *from* that hand target rather than next to it. The second part is where the
options differ, and it is the hard part.

## Notation

- $s$ is the state (joints and gripper), $s_0$ the state at the anchor frame. $a_k$ is the
  action at step $k$ of the chunk, $k = 1 \dots 30$. Today's target is $d_k = a_k - s_0$,
  normalized per robot and per step.
- The hand is a position $p$ in metres, a rotation $R$, and an aperture $g$ in cm.
  $(p, R, g) = \mathrm{FK}(s)$. FK includes the gripper calibration (command to cm) and ends at
  a tool frame defined the same way on every robot: between the fingertips, $z$ pointing out
  along the approach direction, $x$ along the closing direction.
- The hand target at step $k$ is

$$h_k = \big(\,p_k - p_0,\quad r_k,\quad g_k\,\big),$$

  where $(p_k, R_k, g_k) = \mathrm{FK}(a_k)$, $(p_0, R_0) = \mathrm{FK}(s_0)$, and $r_k$ is the
  rotation from $R_0$ to $R_k$ written as an axis-angle vector (axis times angle, three
  numbers). Seven numbers per step: translation, rotation, aperture. The same kind of delta as
  $d_k$, expressed on the hand instead of the joints.
- $h$ is normalized with a mean and std per step **pooled over all robots**, one row for
  everyone. The joint part keeps its per-robot rows. Pooling is the point: a centimetre has to
  cost the same on every robot, otherwise "same target" is not the same number.
- Flow matching gives, at flow time $\tau$, a current guess of the chunk: $\hat d_k$ for the
  joint part and $\hat h_k$ for the hand part. That guess is the only "prediction" that exists
  during training. Near $\tau = 1$ it is sharp; near $\tau = 0$ it is the average of everything
  plausible.

MolmoAct rows are already a hand pose: their $h$ comes from the recorded pose and their joint
part is padding.

## What transfer needs

A task demonstrated only on a Franka runs on ReBot if three things hold:

1. The task knowledge is stored as hand motion, shared across robots.
2. The joint output is computed from the hand motion and the current configuration, per robot.
3. ReBot's joint map is accurate on the motions the task needs. It can only be accurate where
   ReBot has actually moved.

Nothing today asks for any of this. Each option below asks for it in a different way, and each
one exists because of a way the simple version fails.

## How it fails

1. **Shortcut.** The joint output keeps coming from pixels per robot. The hand target is
   learned next to it and never used. Hand loss low, transfer zero.
2. **Frame.** The hand numbers are in a frame that is not shared, so the same motion is not the
   same numbers.
3. **Scale.** Metres, radians and normalized joints in one sum: the weights mean nothing.
4. **Averaging.** A loss on the model's guess at small $\tau$ asks the average of all plausible
   chunks to land on one hand pose. FK of an average is not the average of FKs.
5. **Coverage.** The joint map does not exist where the robot never moved. The model produces
   confident, wrong joints there.
6. **Null space.** Seven joints are not determined by the hand. Posture needs the joint loss.
7. **Calibration.** Tool frame and gripper stroke are per robot. A 2 cm error in the tool-frame
   offset, at half a radian of rotation over a chunk, is a 1 cm error in the target, which is
   the size of the signal.

Contact and force are not in the target at all. Hand motion transfers; grasping under contact
stays per robot. Not addressed here.

## The frame (failure 2) and the calibration (failure 7)

Use the robot's base frame: $p_k - p_0$ and the rotation axis in base coordinates. It is fixed
within a scene, and $z$ is gravity on every table-mounted arm, so lift and lower are shared
exactly. What is not shared is the yaw between base and camera, which differs per dataset and
per scene in DROID. The model already has to read that off the image to act at all, so the
target adds no ambiguity the pixels do not carry.

The alternative is the hand's own frame at the anchor (rotate everything by $R_0^{\top}$). That
is what the wrist camera sees, on every robot. Try it if the wrist view turns out to be what
carries the transfer.

Also append the hand at the anchor, $(p_0, R_0, g_0)$, to the state. The joint map needs the
absolute tilt and height as input; $h$ only carries the change.

| Gripper | Stroke | Source |
|---|---|---|
| Robotiq 2F-85 (DROID) | 8.5 cm | datasheet |
| Franka Hand (FMB) | 8.0 cm | datasheet |
| ReBot | to measure | calipers, open and closed |
| ARX5, UR5 tooling, YAM | to find | corpus metadata |

## The options

Every option sits on top of today's joint loss. The joint loss is never removed (failure 6).

### 1. Extend the action

The simplest thing. Each chunk step becomes $[\,h_k,\ d_k\,]$, 7 + D numbers, and the same
flow loss runs over all of it, with a weight $\lambda$ on the hand part. After pooled
normalization $\lambda = 1$ is the natural start.

What it buys: a shared output for the first time, and the joints are generated in the same
token as the hand, so the joint velocity reads the noisy hand guess while sampling.

What it does not buy: nothing forces the joints to use it (failure 1). The hand is available to
the joint part, not required by it.

### 2. Make the joints accountable in hand space

Take the model's joint guess, run it through FK, and express the result as the same seven
numbers relative to the anchor. Call that $\hat h^{\mathrm{FK}}_k$: the hand the guessed joints
would actually produce. Then penalize its distance to one of two targets.

(a) The true hand, $h_k$. This is the term in the original note.

(b) The model's own hand guess, $\hat h_k$, with no gradient flowing into $\hat h_k$:

$$L_{\mathrm{FK}} = w(\tau)\sum_k \big\|\,\hat h^{\mathrm{FK}}_k - \hat h_k\,\big\|^2
\qquad \text{(normalized units; gradient into } \hat h_k \text{ cut)}.$$

On the training data (a) and (b) agree, because $\hat h_k$ tracks $h_k$. They differ in what
the model learns. (b) makes the joint velocity a function of the hand guess sitting in the same
chunk, so at test time the joints follow whatever hand motion the shared knowledge produces.
That is the transfer path. Cutting the gradient sets the direction: joints chase the hand, the
hand never bends toward the joints.

This is not a reweighting of the joint loss. The gradient of $L_{\mathrm{FK}}$ with respect to
the joint guess is $J^{\top}$ times the hand error, where $J$ is the Jacobian of FK at the
guessed configuration: which joints move the hand, by how much, in what direction, right here.
The joint loss knows none of that; it scores each joint on its own. This term delivers the
kinematics to the weights on every sample, which is the thing the model otherwise has to
memorize pair by pair.

Two details. Compare rotations as matrices, $\|\hat R - R\|_F^2$, which is $2\theta^2$ for the
small rotations a chunk contains and has no singularity at zero. And weight by flow time,
$w(\tau) = \tau^2$, because of failure 4: at small $\tau$ the joint guess is an average, and
asking FK of the average to hit one hand pose fights the flow objective exactly where mode
selection happens.

### 3. Generate the hand first, the joints from it

Impose the factorization instead of hoping for it. Give the expert two tokens per step, one for
the hand and one for the joints, and mask the attention so the joint token reads the hand token
and not the reverse. One flow then samples "hand given observation" and "joints given hand and
observation" together, and with option 2(b) the second part is trained to be the inverse map.
Failure 1 is narrowed, not closed: the joint token still sees the image, so the shortcut is
still possible, just more expensive than reading the hand.

The strong form closes it. The joint token sees only the hand guess, the current state and the
robot's name. No image. Now the joints cannot see the task at all. The cost is failure 6:
posture is decided from state and hand alone, which is fine on a table and wrong near obstacles
the state does not describe.

What the strong form unlocks is the answer to failure 5. A joint map that sees only (hand,
state, robot) can be trained without images. Sample configurations inside ReBot's limits, sample
motions from its own demos or retarget the other robots' hand trajectories into ReBot joints by
inverse kinematics, compute the hand by FK, train on the pairs. That covers the workspace
instead of the demonstrations. The posture ambiguity of a 7-joint arm stays a distribution,
which the flow handles, but the synthetic postures have to be drawn the way the demos draw them,
or the map learns a posture nobody uses.

### 4. Who gets which gradient

The model is a backbone (VLM) and an expert. Today the backbone receives the joint loss, which
is how robot-specific joint knowledge gets into the shared representation; that is where the
shortcut lives.

Instead: the backbone is trained only by the hand loss (the hand part of the flow, or the
discrete action tokens retargeted from joints to hand), and the expert receives joint, FK and
hand losses. The knowledge-insulation switch already cuts the expert's gradient into the
backbone; the change is that the backbone's own action tokens move from joints to hand. This is
the π0.5 split with the language changed. Failure 1 is closed at the backbone. The expert could
still learn pixels-to-joints from what it reads, but it is a small learner scored in hand space
and the backbone no longer helps it.

### 5. A regression head

The first option in the original note: a small head on the representation predicts $h$. A
deterministic head predicts the average, which blurs when several motions are plausible, and
the expert never reads it, so the only link to the joints is through shared backbone weights.
Cheapest to add, weakest for the mechanism. One ablation arm, not the main line.

### 6. Swap the robot's name

"The hand motion is shared" means "the hand guess does not change when the embodiment sentence
is swapped". Penalize the difference between the hand guess under the true name and under
another robot's name. A regularizer, not a target: it doubles the backbone forward and the image
still shows the robot. Not before options 1 and 2 exist.

## What to measure

1. **Transfer.** A task class present only on the other arms, evaluated on ReBot, with and
   without the hand target. This is the leave-one-class-out ladder already planned for the
   conditions probe. In-distribution flow loss is not a readout; it will not move.
2. **Consistency gap at inference.** FK of the executed joints against the generated hand, per
   chunk. Low on training tasks and high on a transfer task means coverage (failure 5). Low
   everywhere while transfer fails means the shortcut (failure 1). Free once option 1 exists.
3. **Cross-robot decoding of $h$** from the representation with the existing conditions probe,
   as the check that the frame is not biting (failure 2).

## Recommendation

Option 1 plus option 2(b) with $w(\tau)$, base frame, pooled stats, hand appended to the state.
Option 1 is the shared output; 2(b) is what makes the joints follow it. Then read the
consistency gap:

| Observation | Meaning | Next |
|---|---|---|
| gap low in-distribution, transfer fails | shortcut | option 4 |
| gap high on transfer tasks | coverage | option 3, strong form, with kinematic pre-training |
| gap low and transfer works | the mechanism exists | collect ReBot motions, not tasks |
