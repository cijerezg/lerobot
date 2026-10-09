# What the literature says

**Status:** 2026-10-08. Search done after [thoughts.md](thoughts.md); only the papers that
change a decision are listed. Citations were checked against the arXiv pages.

## Our baseline is the known failure case

**Alvarez et al. 2026**, Improving Cross-embodiment Transfer in Latent Action Models with
Action-Similarity Supervision, [arXiv 2609.19846](https://arxiv.org/abs/2609.19846). Controlled sim study, two bimanual robots
(6-DoF and 7-DoF arms), three disjoint tasks each, evaluated on the other robot's tasks. π0 with
padded joint actions: 24% transfer, 54% in-embodiment, and the embodiment is recoverable from
the action vector with 100% accuracy ("the two embodiments occupy different coordinates of the
shared action space"). Switching the output to EE deltas: 26% transfer but in-embodiment drops to
41%. Using EE motion as a similarity supervision on a shared latent while keeping native outputs:
53% transfer, 70% in-embodiment. That is our setup (padded joints, per-robot stats, embodiment
sentence) measured, and it says: keep joints as the output, use the hand as supervision.

**Feng et al. 2026**, Demystifying Action Space Design for Robotic Manipulation Policies,
[arXiv 2602.23408](https://arxiv.org/abs/2602.23408). 500+ models, 13k real rollouts. Joint deltas beat EE deltas in-embodiment
(95.9 vs 91.4 single arm), and the edge grows with data; under transfer the task space wins.
Chunk-relative deltas beat step-wise by about 10 points. Same conclusion as above from the other
direction, and it confirms the delta-from-anchor encoding we already use.

## The parallel target alone has evidence against it

**Spisak et al. 2025**, DIRIGENt, [arXiv 2501.16800](https://arxiv.org/abs/2501.16800). Diffusion over 26 joints with an FK loss
on the end effector. Dropping the FK loss leaves joint error unchanged and raises Cartesian error
(0.111 to 0.160); dropping the joint loss raises joint error 13x because the network picks other
redundant solutions. Adding the FK output as an extra diffusion target next to the joints did not
help. So: the joint loss pins the posture, and the hand only earns its place when its gradient
goes through FK into the joints. Option 1 by itself is the arrangement with no evidence.
RDT-1B (Liu et al. 2024, [arXiv 2410.07864](https://arxiv.org/abs/2410.07864)) is the large-scale version of it: a 128-slot action
vector with joint and EE slots side by side for every robot, nothing tying them together, and no
transfer attributed to the layout.

## An intermediate that the executor ignores, measured

**Xie et al. 2026**, Fast Plans, Faithful Actions, [arXiv 2609.30833](https://arxiv.org/abs/2609.30833). π0.5-style hierarchy: the
VLM emits waypoints (target configuration, gripper, duration), a flow expert executes. Erasing
the waypoint at test time changed LIBERO-Long success by 0.0 points from 91.0%: the expert had
learned the task from pixels and never read the plan. Their fix is the teacher-forcing recipe
from thoughts.md with the details filled in: the goal enters every expert layer through a
zero-initialized MLP into the conditioning, gated open only for flow time above 0.5; during
training the goal is noised (sigma 0.7 in normalized units) and replaced by a null embedding 15%
of the time. Without the noise, the expert learned to integrate the goal instead of looking at
the image (a target configuration minus the current state is roughly the integral of the action
labels), giving 15% success. With it: 96.2%, and erasing the plan now costs 7.4 points. Their
usefulness number, success with the plan minus success with it erased, is the shortcut test.
This is the closest thing to our mechanism question in print, and the label-leak warning applies
to us directly: a hand chunk is roughly the Jacobian times the joint chunk.

## Joints produced from a hand target through a loss

**Ma et al. 2024**, Hierarchical Diffusion Policy (RK-Diffuser), CVPR 2024, [arXiv 2403.03890](https://arxiv.org/abs/2403.03890).
Two diffusion branches with the same conditioning, one over EE poses, one over joints; the pose
branch's denoising objective is applied to FK(joints) through differentiable kinematics, and at
inference the joints are refined so FK(joints) tracks the pose branch. With ground-truth goals:
94.6% vs 73.6% for joint-only diffusion vs 67.2% for pose diffusion plus IK (24.6% IK failures).
Single robot, but it is option 2(b) with the stop-gradient direction we chose (joints chase the
pose branch) and it shows the gain is large when the hand branch is good.

**Jiang et al. 2026**, XL-VLA, CVPR 2026, [arXiv 2603.10158](https://arxiv.org/abs/2603.10158). Four dexterous hands, each with a
small autoencoder into one 32-d latent, trained on random joint samples with a differentiable-FK
retargeting loss that makes fingertip geometry agree across hands for the same latent. π0 then
predicts the latent; real success over 10 tasks 0.32 (raw joints) to 0.72. The most direct
precedent for "FK through the per-embodiment output forces agreement on a shared physical
quantity", and the realization map was trained from kinematics alone, no demonstrations, which
is the strong form of option 3.

## How to weight an FK loss under flow matching

**Yang et al. 2026**, MimicIK, [arXiv 2606.15148](https://arxiv.org/abs/2606.15148), and **Huang et al. 2026**, GraphDiff-IK, arXiv
2606.00086. Both put the FK loss on the clean terminal estimate (never on the velocity).
MimicIK: weight 0.1 takes mean error 6.29 to 4.65 mm and removes seed variance; 0.5 hurts again.
GraphDiff-IK gates the term off at high noise, arguing FK of a noisy joint vector is not
geometric information. So: terminal estimate, flow-time gate, small weight, and expect a
stability-sized gain in hand space rather than new behaviour.

## Where the gradient is allowed to go

**Driess et al. 2025**, Knowledge Insulating VLAs, NeurIPS 2025, [arXiv 2505.23705](https://arxiv.org/abs/2505.23705). Stop-gradient
on the expert's reads of the backbone; the backbone is trained with next-token loss on FAST
action tokens instead. Without the cut π0 needs about 7.5x the steps; with it DROID generalist
0.55 vs 0.49. The paper is explicit that the cut only works if the backbone has its own action
loss. That is option 4, and the authors note nothing in it forces the expert to use the backbone's
action tokens: it fixes what the backbone represents, not the route.

## The pixels carry the embodiment too

**Piseno et al. 2026**, Cloak, CoRL 2026, [arXiv 2606.22836](https://arxiv.org/abs/2606.22836). π0.5 fine-tuned on DROID in joint
space, no hand in the loss. At deployment the joints go through FK to two tip poses, IK on the
target robot, and the gripper is masked out of the wrist image by rendering it through FK.
Zero-shot task progress on a UMI gripper, a YAM arm and a five-finger hand: 85, 86, 82, against
88 on the source gripper; the FK/IK bridge alone gives 54 to 70 and the mask adds the rest. Two
lessons: joint outputs can be treated as an encoding of hand pose by construction, and the visual
embodiment cue is a 15 to 25 point effect on its own. Our "swap the output robot" idea has the
same target; masking the gripper in the wrist image is the cheaper version of it.

**Kareer et al. 2026**, Emergence of Human to Robot Transfer in VLAs, RSS 2026, arXiv
2512.22414. Shared chunk-relative EE representation for humans and robots; transfer from the
foreign embodiment appears only once robot pretraining is diverse enough (gains near zero at low
diversity, large at full diversity). A shared quantity is necessary, not sufficient, and with
five arms we should expect that.

## The frame

**Yuan et al. 2025**, MotionTrans, [arXiv 2509.17759](https://arxiv.org/abs/2509.17759): chunk-relative wrist pose 23.1% vs
absolute pose 10.0%. UMI, DexUMI, GEAR-VLA ([arXiv 2606.08530](https://arxiv.org/abs/2606.08530)) and GR00T N1.7 all express the
hand target as the SE(3) transform from the current hand pose to each future pose, in the
current-hand frame, with aperture in physical units. The relative part we already have. The
axis choice in the proposal (base frame) is the minority; the field uses the hand frame, which
is also base-frame-free. Cheap to switch; worth being the first ablation rather than a decision.

## What changes in the proposal

1. Option 1 alone is out as a main line (DIRIGENt, Alvarez). The hand target is supervision,
   not a sibling output.
2. The primary mechanism is the teacher-forced hand input to the joint path with the Fast Plans
   recipe: enters the expert's conditioning, gated to flow time above 0.5, noised hard, nulled
   15% of the time. The usefulness number (success with the hand chunk minus success with it
   erased) is the acceptance test.
3. The FK consistency term stays, on the terminal estimate, gated by flow time, weight around
   0.1 in normalized units (MimicIK, GraphDiff-IK, RK-Diffuser).
4. Option 4 (backbone trained in hand coordinates only) is supported and cheap, with the caveat
   from its own authors that it fixes the representation, not the route.
5. Add the wrist-image gripper mask (Cloak) as a lever beside the output-robot swap; the pixels
   identify the robot as strongly as the action vector does.
6. Hand-frame versus base-frame target becomes the first ablation.
7. Expect transfer to need the hand channel and enough embodiment diversity (Kareer); the
   held-out-task test on ReBot is still the only readout that counts.

## References

1. Alvarez, M. et al. (2026). Improving Cross-embodiment Transfer in Latent Action Models with
   Action-Similarity Supervision. arXiv:2609.19846. https://arxiv.org/abs/2609.19846
2. Feng, Y. et al. (2026). Demystifying Action Space Design for Robotic Manipulation Policies.
   arXiv:2602.23408. https://arxiv.org/abs/2602.23408
3. Spisak, J., Kerzel, M., Wermter, S. (2025). DIRIGENt: End-To-End Robotic Imitation of Human
   Demonstrations based on a Diffusion Model. arXiv:2501.16800. https://arxiv.org/abs/2501.16800
4. Liu, S. et al. (2024). RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation.
   arXiv:2410.07864. https://arxiv.org/abs/2410.07864
5. Xie, C., Ma, Li et al. (2026). Fast Plans, Faithful Actions: Closing the
   Planning-Execution Gap in Hierarchical Vision-Language-Action Models. arXiv:2609.30833.
   https://arxiv.org/abs/2609.30833
6. Ma, X. et al. (2024). Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic
   Manipulation. CVPR 2024. arXiv:2403.03890. https://arxiv.org/abs/2403.03890
7. Jiang, G. et al. (2026). Cross-Hand Latent Representation for VLA Models (XL-VLA). CVPR 2026.
   arXiv:2603.10158. https://arxiv.org/abs/2603.10158
8. Yang et al. (2026). MimicIK: Real-Time Generative Inverse Kinematics from Teleoperation
   with FK Consistency. arXiv:2606.15148. https://arxiv.org/abs/2606.15148
9. Huang, Tan, Wen, Huang, Quan (2026). Whole-Body Inverse Kinematics with Graph
   Diffusion (GraphDiff-IK). arXiv:2606.00086. https://arxiv.org/abs/2606.00086
10. Driess, D., Springenberg, J. T., Ichter, B. et al. (2025). Knowledge Insulating
    Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better. NeurIPS 2025.
    arXiv:2505.23705. https://arxiv.org/abs/2505.23705
11. Piseno, M., Tevet, G., Liu, C. K. (2026). Cloak: Zero-Shot Cross-Embodiment Manipulation by
    Masking the End-Effector from the VLA. CoRL 2026. arXiv:2606.22836.
    https://arxiv.org/abs/2606.22836 and https://tml.stanford.edu/cloak/
12. Kareer, S. et al. (2026). Emergence of Human to Robot Transfer in Vision-Language-Action
    Models. RSS 2026. arXiv:2512.22414. https://arxiv.org/abs/2512.22414
13. Yuan, C. et al. (2025). MotionTrans: Human VR Data Enable Motion-Level Learning for Robotic
    Manipulation Policies. arXiv:2509.17759. https://arxiv.org/abs/2509.17759
14. Zhang, Y. et al. (2026). GEAR-VLA: Learning Geometry-Aware Action Representations for
    Generalizable Robotic Manipulation. arXiv:2606.08530. https://arxiv.org/abs/2606.08530
15. Chi, C. et al. (2024). Universal Manipulation Interface (UMI). arXiv:2402.10329.
    https://arxiv.org/abs/2402.10329
16. Xu, M. et al. (2025). DexUMI. CoRL 2025. arXiv:2505.21864. https://arxiv.org/abs/2505.21864
17. NVIDIA GR00T team (2025, 2026). GR00T N1 (arXiv:2503.14734, https://arxiv.org/abs/2503.14734)
    and the N1.7 README at https://github.com/NVIDIA/Isaac-GR00T
