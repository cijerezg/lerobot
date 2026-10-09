# Implementation: the hand block in the chunk

**Status:** proposal, 2026-10-08. Nothing implemented. Builds on [thoughts.md](thoughts.md):
the direct hand output is the alignment mechanism, forward kinematics appears only in the term
that ties the joints to it.

## 0. Why this is cheap

Four facts about the current code make this a small change.

- The action expert is 32 wide (`expected_max_action_dim`), the batch is 8 wide
  (`MAX_ACTION_DIM`), and the pack step zero-pads 8 to 32. Slots 8 to 31 are padding today.
  The hand block is 7 numbers. It takes slots 8 to 14. No new parameters anywhere.
- Knowledge insulation is on in `config_rl.yaml`. The flow loss never reaches the backbone;
  the backbone's only action signal is the FAST token loss. Tokenizing the hand block puts the
  shared motion vocabulary exactly where the task knowledge lives, with the mechanism from
  thoughts.md: the same motion is the same tokens on every robot, the language-model head is
  one matrix, so the backbone is pushed to put "move 1 cm left" in the same place for every
  robot.
- Forward kinematics runs offline once per frame to build the targets. The in-loss copy is a
  chain of at most eight 4x4 products per chunk step; for a batch of 32 with 8 flow times it is
  under ten thousand small matrix products, nothing next to the model.
- Execution is unchanged. The joint block goes through the same unnormalize, anchor decode and
  trim as today. The hand block becomes a free monitor at inference.

## 1. The hand target

Notation from the proposal: state $s$, anchor $s_0$, action $a_k$, joint target $d_k = a_k -
s_0$, hand $(p, R, g)$ with $p$ in metres, $R$ a rotation matrix, $g$ the aperture in cm, and
$\mathrm{FK}_e$ the chart map of robot $e$ including its gripper calibration.

**Offline, per frame.** Two absolute hand poses per frame, both stored as 10 numbers (position
3, first two columns of $R$ 6, aperture 1):

$$x^{s}_t = \mathrm{FK}_e(s_t), \qquad x^{a}_t = \mathrm{FK}_e(a_t).$$

The state pose is where the hand is; the action pose is where the command sends it. Both are
needed because the chunk target is "where the commands send the hand, relative to where it is".

**At sampling, per chunk step** $k = 1 \dots 30$, with $(p_0, R_0) $ from $x^s_t$ and
$(p_k, R_k, g_k)$ from $x^a_{t+k}$:

$$h_k = \Big(\, R_0^{\top}(p_k - p_0),\quad \log\!\big(R_0^{\top} R_k\big),\quad g_k \,\Big) \in \mathbb R^7 .$$

The first three numbers are the translation seen from the hand's own frame at the anchor, the
next three the rotation from the anchor orientation to the step orientation as an axis-angle
vector, the last the aperture. This is the hand-frame choice the field converged on. The
base-frame variant is one switch away: $p_k - p_0$ and $\log(R_k R_0^{\top})$. It is the first
ablation.

**Normalization.** One row for every robot:

$$\bar h_k = (h_k - \mu_k) / \sigma_k, \qquad \mu_k, \sigma_k \in \mathbb R^7
\text{ pooled over all robots and all samples at step } k .$$

The joint block keeps its per-robot rows. The pooling is what makes a centimetre cost the same
everywhere.

**Where the hand comes from, per layout.**

| Layout | Robot | Joints | Hand source | Aperture |
|---|---|---|---|---|
| 0, 1 | DROID Franka | commanded, rad | Panda DH table (exists, `panda_fk.py`) | 8.5 cm x (1 - ratio) |
| 2 | FMB Franka | measured, rad | Panda DH table | levels, calibrate once |
| 3 | RoboChallenge ARX5 | measured | ARX5 URDF, to fetch | width in metres, as recorded |
| 4 | RoboChallenge UR5 | measured | UR5 URDF (ur_description) | width in metres, as recorded |
| 6 | ReBot | commanded, deg | ReBot URDF (exists, `kinematics.py`) | stroke x (1 - ratio), measure stroke |
| 7 | MolmoAct Franka | none | recorded pose directly | 8.0 cm x (1 - ratio) |
| 8 | YAM | commanded, rad | YAM URDF (i2rt) | stroke x (1 - ratio) |

One torch module does all of it: a revolute chain read from a URDF (joint origins and axes,
composed as 4x4 products), plus the Panda DH table. The same module builds the offline targets
and runs inside the loss, so the two never disagree. The tool frame is set per robot at the
fingertip midpoint, $z$ out along the approach, $x$ along the closing direction.

## 2. The chunk

The hand block travels as its own feature key, `action.hand`, shape $[30, 7]$, with its own
pooled stats. The anchor-encode step never sees it; a small `HandDeltaStep` builds $h_k$ from
the stored poses at sampling time, in front of the normalizer, exactly where `AnchorEncodeStep`
builds $d_k$ for the joints.

Inside the policy forward the two are concatenated into the 32-wide chunk the expert already
takes:

$$A = \big[\, \bar d \;\big|\; \bar h \;\big|\; 0 \,\big], \qquad
\text{slots } 0..7 \text{ joints (robot's own, padded)},\ 8..14 \text{ hand},\ 15..31 \text{ padding}.$$

The joints keep the slots the foundation checkpoint trained them in. The pad mask is the
joint mask on slots 0 to 7, valid on 8 to 14, pad above. MolmoAct rows (layout 7) have no
joints: slots 0 to 7 pad, hand valid. They become pure hand-language samples, which is what
they always were.

After generation the chunk is split: slots 0 to 7 go down the existing joint path, slots 8 to 14
are unnormalized with the pooled stats and kept for the monitor in section 5.

## 3. Losses

Flow matching as in the design docs, with $\tau$ for flow time:

$$x_\tau = (1 - \tau)\,\varepsilon + \tau A, \qquad v^{*} = A - \varepsilon, \qquad
\hat A_\tau = x_\tau + (1 - \tau)\, v_\theta(x_\tau, \tau \mid o).$$

**Flow loss, both blocks.** Masked per slot as today, with one weight on the hand block:

$$\mathcal L_{\mathrm{flow}} = \mathbb E\Big[\, \big\| v^{d}_\theta - v^{*d} \big\|^2
+ \lambda_h \big\| v^{h}_\theta - v^{*h} \big\|^2 \Big], \qquad \lambda_h = 1 .$$

This is the alignment term. The hand slots are produced by the same final layer for every robot,
and the error on them pushes the expert's representation in the same direction for the same
motion on every robot.

**Coupling, joints to hand.** Take the terminal estimate of the joint block, put it back in the
robot's units with its own stats row, add the anchor, push it through the chart, and express the
result as the same seven numbers the hand block uses:

$$\hat d_k = \sigma^{d}_{e,k}\, \hat A^{d}_{k} + \mu^{d}_{e,k}, \qquad
\hat x_k = \mathrm{FK}_e\big(s_0 + \hat d_k\big), \qquad
\hat h^{\mathrm{FK}}_k = \mathrm{rel}\big(\hat x_k,\ x_0\big)$$

with $\mathrm{rel}$ the map of section 1 and $x_0 = \mathrm{FK}_e(s_0)$. Then

$$\mathcal L_{c} = \mathbb E\Big[\, w(\tau) \sum_{k} \rho\Big(\hat h^{\mathrm{FK}}_k,\ \hat h_k\Big) \Big],
\qquad \hat h_k = \sigma_k \,\mathrm{sg}\big(\hat A^{h}_k\big) + \mu_k, \qquad w(\tau) = \mathbb 1[\tau \ge 0.5] .$$

$\mathrm{sg}$ is stop-gradient: the hand block is the target, the joints move toward it, never
the reverse. $\rho$ is the squared error on translation and aperture, each divided by its pooled
$\sigma_k^2$, and for rotation the matrix distance

$$\rho_R = \frac{\big\| \hat R^{\mathrm{FK}}_k - \hat R_k \big\|_F^2}{2\,\sigma_{R,k}^2},
\qquad \hat R_k = \exp\big([\hat h_k^{\mathrm{rot}}]_\times\big) \text{ under no-grad},$$

which equals the normalized squared angle for small rotations and has no singularity. The gate
$w$ keeps the term off where the joint estimate is still an average of many chunks; above
$\tau = 0.5$ the chunk is pinned down and the hand of the joint estimate is a real quantity.

The gradient this term delivers to the joint estimate is

$$\frac{\partial \mathcal L_c}{\partial \hat A^{d}_k} = \sigma^{d}_{e,k}\; J_e\big(s_0 + \hat d_k\big)^{\top}\, \Sigma_k^{-1}\, r_k,$$

$J_e$ the Jacobian of the chart at the estimated configuration, $\Sigma_k$ the pooled hand
variances, $r_k$ the hand residual. That is the inverse map being taught, one sample at a time,
on each robot's own slots.

**Total.** $\mathcal L = \mathcal L_{\mathrm{flow}} + \lambda_c \mathcal L_c$ plus the existing
FAST, band and future-visual terms. Start $\lambda_c = 0.1$ (the useful range in the IK papers
is about 0.05 to 0.2 in normalized units).

**FAST tokens.** The discrete head tokenizes the 15-wide prefix $[\bar d \mid \bar h]$ instead
of the 8-wide one. The token budget already scales with the action width. Under knowledge
insulation these tokens are the backbone's whole action signal, so the hand block reaches the
representation that holds the task knowledge. A knob `discrete_action_dims: all | hand` allows
the pure form, where the backbone is trained in hand coordinates only.

## 4. Teacher forcing inside the same expert

If the follow ratio of section 6 says the joints do not yet read the hand block, make the hand
block a clean input for part of the batch. No new parameters: with probability $p_{\mathrm{tf}}$
the noisy hand slots of $x_\tau$ are replaced by the target with fixed noise, and with
probability $p_{\mathrm{null}}$ by zeros,

$$x^{h}_\tau \leftarrow \bar h + \sigma_{\mathrm{tf}}\, \varepsilon', \qquad
x^{h}_\tau \leftarrow 0 \ \text{(null)},$$

and for those samples only the joint block is scored. The joint velocity then learns to read a
nearly clean hand and realize it; the noise keeps it from integrating the hand instead of looking
at the image. Values from the one paper that measured this: $\sigma_{\mathrm{tf}} = 0.7$ in
normalized units, $p_{\mathrm{null}} = 0.15$; $p_{\mathrm{tf}} = 0.5$ to start. At inference
nothing changes, the hand block is generated as usual and the joints follow it.

## 5. Inference

Euler over the 32-wide chunk as today, 5 steps. Split. Joints: unchanged. Hand block: unnormalize
and compute the gap

$$\mathrm{gap} = \frac{1}{30} \sum_k \rho\Big(\mathrm{rel}\big(\mathrm{FK}_e(s_0 + d_k),\ x_0\big),\ \hat h_k\Big),$$

logged per chunk. Low on training tasks and high on a transfer task says coverage; low everywhere
with no transfer says the joints are not reading the hand.

## 6. Readouts

**Follow ratio.** During Euler, override the hand slots at every step with a shifted target,

$$x^{h}_i \leftarrow (1 - \tau_i)\,\varepsilon^{h} + \tau_i\,(\bar h + \bar\delta),$$

decode the joints, take the hand displacement they produce, and measure its component along the
shift:

$$\mathrm{follow} = \frac{\big\langle \Delta_{\mathrm{FK}},\ \delta \big\rangle}{\|\delta\|^2},
\qquad \Delta_{\mathrm{FK}} = \mathrm{rel}\big(\mathrm{FK}_e(s_0 + d^{\delta}),\, x_0\big) - \mathrm{rel}\big(\mathrm{FK}_e(s_0 + d),\, x_0\big).$$

Shifts of 1 cm along each axis, 10 degrees about each axis, 2 cm of aperture, over a few hundred
frames per robot. A ratio near 1 means the joints are produced from the hand. This is the
intervention measurement, not a loss curve.

**Cross-robot decoding of the hand.** The conditions probe gets $\bar h$ as a decoding target:
fit on one robot's representation, read on another's. Rising toward the within-robot ceiling is
the alignment being real in the backbone.

**Transfer.** The leave-one-class-out run on ReBot. The readout that counts.

**In-distribution.** Joint flow loss on slots 0 to 7 unchanged or better; hand-space error of
the executed joints on validation chunks, mean and tail.

## 7. Seams

Named once, so the size of the job is visible.

- Data: the FK module (URDF chain plus Panda DH), one script that writes `hand.state` and
  `hand.action` per frame for the ReBot roots and an ingest step for the diverse corpus; caches
  rebuild.
- Pipeline: `HandDeltaStep` with the frame switch; pooled stats for `action.hand` from an
  extended `compute_delta_stats`; the normalizer takes the new key.
- Policy: concatenate and split around the expert; the 32-wide mask; the joint stats rows and
  the anchor state forwarded into the batch for $\mathcal L_c$; the FK module registered per
  layout; $\mathcal L_c$ and the gate; FAST over 15 dims; the teacher-forcing switch.
- Probes: follow ratio; the decoding target; the gap in the inference log.
- Config: `hand_frame`, `lambda_h`, `lambda_c`, `tau_0`, `p_tf`, `sigma_tf`, `p_null`,
  `discrete_action_dims`.

## 8. Stages

1. Hand block in the chunk, FAST over both, pooled stats. Readouts: hand flow loss, cross-robot
   decoding, follow ratio, in-distribution joint loss. This already tests the alignment claim.
2. Add $\mathcal L_c$. Expect the follow ratio and the hand-space error to move.
3. Add teacher forcing only if the follow ratio stays low.
4. Transfer run on the best of 1 to 3, with hand frame against base frame as the first ablation.
