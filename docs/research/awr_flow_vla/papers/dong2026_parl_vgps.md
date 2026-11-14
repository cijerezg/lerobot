# Critic-based policy extraction without online policy gradients: Dong et al. 2026 (batch online RL), PA-RL (Sobol Mark et al. 2024), V-GPS (Nakamoto et al. 2024)

How these notes were read:
- Dong et al.: I read the full rendered PDF of both the arXiv v1 (15 pp.) and the ICLR 2026 camera-ready from proceedings.iclr.cc (15 pp.).
- PA-RL: I read the full rendered arXiv v1 PDF (30 pp.).
- V-GPS: every PDF (arXiv v1, arXiv v2, PMLR) is over the 10 MB fetch limit. I read the arXiv HTML (v2 and the unversioned page) through an extraction tool in three separate passes. The Table 1, 2 and 10 numbers came out the same in every pass, but I could not look at the V-GPS figures myself.
- Bar values quoted from Dong's Figures 5–8 and 10 are my readings of small printed bar labels or plot lines. They may be off by about 1–2 points, and the ICLR per-task labels are the least legible.

## Q0. Do the papers exist at the given IDs, and which versions were read?

### Takeaway
All three papers exist at the given arXiv IDs with matching titles.
- The Dong et al. arXiv entry is 2025 and carries no venue statement. The ICLR 2026 camera-ready is a separate PDF with slightly different figure numbers and some added text.
- PA-RL's first author's surname is "Sobol Mark", not "Mark".
- V-GPS is the CoRL 2024 paper, as its arXiv comment says.

### Cited Findings
- **2505.08078**: "What Matters for Batch Online Reinforcement Learning in Robotics?", by Perry Dong, Suvir Mirchandani, Dorsa Sadigh and Chelsea Finn (Stanford). Only v1 exists, dated 12 May 2025. The arXiv page has no Comments/venue field and no journal-ref. — [arXiv abs](https://arxiv.org/abs/2505.08078)
- **ICLR 2026 version**: the camera-ready PDF has the running header "Published as a conference paper at ICLR 2026" with the same title and authors. — [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf). ML Anthology lists it under ICLR 2026 with OpenReview forum id usw1NVkczu. — [ML Anthology](https://mlanthology.org/iclr/2026/dong2026iclr-matters/). OpenReview itself returned only a browser-verification page, so I read the proceedings PDF instead.
- **Dong, v1 vs ICLR**: the two versions have the same structure and the same claims. The ICLR version differs in these ways:
  - error bars (standard error) added to Figures 3, 5–8, 12 and 13;
  - slightly different bar values;
  - two open questions added to the Discussion;
  - architecture, data-source and evaluation-protocol details added to Appendix B.
  — [arXiv v1 PDF](https://arxiv.org/pdf/2505.08078v1); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **2412.06685**: "Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone", by Max Sobol Mark, Tian Gao, Georgia Gabriela Sampaio, Mohan Kumar Srirama, Archit Sharma, Chelsea Finn and Aviral Kumar (CMU, Stanford). Only v1 exists, dated 9 Dec 2024, with no venue on arXiv. — [arXiv abs](https://arxiv.org/abs/2412.06685); [PDF](https://arxiv.org/pdf/2412.06685)
- **2410.13816**: "Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance", by Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar and Sergey Levine. v1 is dated 17 Oct 2024 and v2 24 Feb 2025. The arXiv comment reads "Conference on Robot Learning (CoRL) 2024". — [arXiv abs](https://arxiv.org/abs/2410.13816). A search result gives the proceedings as PMLR v270, pp. 4996–5013. — [PMLR page](https://proceedings.mlr.press/v270/nakamoto25a.html)

### Inferences
- "Dong et al. 2026" is correct only for the ICLR version. The citation should give arXiv 2505.08078 (2025) plus ICLR 2026, and numbers should be quoted from the ICLR PDF.

### Gaps
- I could not open the OpenReview reviews or rebuttal for Dong because of the browser check. It is therefore unknown whether reviewers questioned the AWR tuning.

## Q1. Dong et al.: what are implicit and explicit extraction, which wins and by how much, what else matters, and does "implicit beats AWR" hold as stated?

### Takeaway
- **Result.** In simulation (6 tasks, 3 seeds × 100 evaluations), implicit extraction goes from about 38 to about 75 average normalized return after batch-online iterations. Explicit (AWR) extraction goes from about 47 to about 24 (ICLR Fig. 6).
  - AWR starts higher at iteration 0 on nearly every task.
  - AWR then gets worse over iterations on every task. It does not merely plateau.
- **Definitions.**
  - "Implicit" means the actor is trained with plain unweighted IL, and at rollout time the method draws 64 samples from it and executes argmax_a Q(s,a). This is IDQL.
  - "Explicit" means AWR, exp(β(Q−V))·log π.
- **Nuances.**
  - The AWR temperature β is never reported, and no normalization, clipping or tuning is described.
  - There is no explicit-vs-implicit comparison on a real robot.
  - The only real-robot evidence is one task with no stated trial counts.
  - Filtered-IL beats value-based RL at the smallest data scale.

### Cited Findings

**Setup (both versions, §3–4, App. B)**
- **Batch online RL (Algorithm 1)**:
  - π0 and Q0 are trained on D0.
  - At each iteration i, M rollouts are collected with Rollout(π_{i−1}, Q_{i−1}), then Q_i ← UpdateValue(∪D) and π_i ← UpdatePolicy(∪D).
  - N = 10–20 iterations with M = 200 rollouts per iteration.
  - D0 is 5–100 demonstrations, sized so that π0 reaches 30–65% success.
  — [ICLR §3–4, App. B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **Tasks**: 6 simulated tasks: Robomimic Lift, Can and Square; MimicGen Stack and Threading; Adroit Pen.
  - D0 sizes: Lift 5, Can 10, Square 100, Stack 20, Threading 50, Pen 3 (App. B Table 1).
  - Data sources (ICLR App. B): Robomimic Proficient Human, the MimicGen benchmark data, and D4RL for Adroit.
  - Evaluation rollout lengths: 400 steps for Lift, Can, Square and Threading, 200 for Stack, 100 for Pen.
  - Results are normalized return averaged over 3 seeds × 100 evaluation trials.
  — [ICLR App. B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **Critic**: IQL. L_Q = E[(r + γV(s') − Q(s,a))²] and L_V = E[L2^τ(Q_target(s,a) − V(s))], with expectile τ = 0.8, γ = 0.99, batch 256, learning rate 3e-4, Adam (§3 Eqs. 1–2; App. B Table 1). — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **Explicit extraction (the AWR baseline)**:
  - Formula: J_π(θ) = E_{(s,a)∼D}[exp(β(Q(s,a) − V(s))) log π_θ(a|s)], "where β is a hyperparameter to interpolate between behavior cloning and recovering the maximum of the Q-function" (§3).
  - §4.2 calls AWR "a canonical example of explicit policy extraction" that maximizes an RL objective plus an IL objective offline.
  - No β value, no advantage normalization, no weight clipping and no tuning sweep for AWR appear anywhere in either version, including the Appendix B hyperparameter table.
  — [ICLR §3, §4.2, App. B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [arXiv v1](https://arxiv.org/pdf/2505.08078v1)
- **Implicit extraction**:
  - "during rollouts, multiple actions {a_i} are sampled from the policy and the best action argmax_{a_i} Q(s,a_i) is selected. Since we use a diffusion policy as the base policy, this can be instantiated as … IDQL" (§4.1). The number of sampled actions is 64 (App. B Table 1).
  - Stated trade-off: implicit extraction "loses potentially useful signals from the Q-function for the policy, [but] has the advantage of disentangling the value function and policy training, which provides more stable learning" (§4.2).
  - The recipe (§5) is: "train an expressive IL policy as the actor, train a Q-function on the autonomous data, and perform implicit policy extraction using the Q-function to obtain a policy to do rollouts."
  - Both policy classes "use supervised IL objectives and the value function objectives from IQL" (§4.3).
  — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **Policy networks**:
  - Diffusion: 100 diffusion steps, variance-preserving β schedule, IDQL residual MLP with hidden size 256 and 3 residual blocks.
  - Gaussian: MLP with hidden size 256 and 3 layers.
  - These details are in ICLR App. B only.
  - Action chunking or action horizon is not mentioned in either version.
  — [ICLR App. B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

**Explicit vs implicit, Figure 6** (normalized return; Pre = π0 trained on D0, Post = after all iterations; 3 seeds × 100 trials)

Per task, arXiv v1 Fig. 6, as explicit / implicit:

| Task | Pre | Post |
|---|---|---|
| Lift | 53 / 55 | 33 / 93 |
| Can | 45 / 40 | 26 / 66 |
| Square | 62 / 47 | 11 / 85 |
| Threading | 49 / 20 | 4 / 94 |
| Stack | 43 / 39 | 24 / 60 |
| Pen | 40 / 40 | 36 / 49 |
| **Average** | **48 / 40** | **26 / 74** |

— [arXiv v1 Fig. 6](https://arxiv.org/pdf/2505.08078v1)

ICLR Fig. 6 averages: Pre 47 (explicit) / 38 (implicit); Post 24 / 75. The approximate ICLR per-task readings, as explicit / implicit:

| Task | Pre | Post |
|---|---|---|
| Lift | 49 / 55 | ~31 / 93 |
| Can | 41 / 35 | 27 / 67 |
| Square | 62 / 45 | 11 / 84 |
| Threading | 55 / 19 | ~7 / 93 |
| Stack | 42 / 33 | 30 / 63 |
| Pen | ~36 / 43 | ~35 / 50 |

— [ICLR Fig. 6](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

- **Authors' text on Figure 6** (§4.2):
  - "although explicit policy extraction achieves a stronger initial performance in nearly every benchmark task, implicit policy extraction performs significantly better after running batch online RL. In fact, explicit policy extraction does not improve the policy performance on any of the tasks."
  - Their explanation: "while explicit policy extraction approaches provide more signals from the value function for policy learning, this signal becomes detrimental when new, diverse data is added during batch online RL and causes a shift in action distribution. The policy extracted from explicit policy extraction cannot adjust to this shift as well as implicit policy extraction." No diagnostic experiment backs this explanation.
  — [ICLR §4.2](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

**Policy expressivity, Figure 7** (arXiv v1 averages, Pre → Post)
- Gaussian with explicit extraction: 19 → 18.
- Gaussian with implicit extraction: 19 → 43.
- Diffusion with implicit extraction: 40 → 74.
- ICLR averages, approximately: diffusion-implicit 38 → 75 and Gaussian-implicit ~18 → ~45.
- The paper says Gaussian-implicit is "a significant improvement over Gaussian policies with explicit policy extraction [but] considerably worse than that of an expressive policy."
- The authors' reason is that in batch online RL "the initial model needs to have captured enough of an expert action distribution to collect useful trajectories" (§4.3).
— [arXiv v1 Fig. 7](https://arxiv.org/pdf/2505.08078v1); [ICLR §4.3](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

**Algorithm class and data scale, Figures 3 and 5**
- Section headings: "Value-based RL is necessary for overcoming suboptimal convergence of Filtered-IL", and "Vanilla IL performs the worst on all tasks … will fit the failure trajectories of the autonomous rollouts. Filtered-IL, while exhibiting an initial improvement, often converges quickly to suboptimal performance" (§4.1).
- Filtered-IL definition: weight 1 on transitions from successful trajectories, 0 on failures (§3).
— [ICLR §3, §4.1](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

Data scaling, averaged over tasks (M = 50 / 100 / 200 rollouts per iteration for small / medium / large; Pen uses 100 / 200 / 300). Values are IL / filtered-IL / value-based RL:

| Scale | ICLR Fig. 5 | arXiv v1 Fig. 5 |
|---|---|---|
| Small | 43 / 60 / 54 | 43 / 59 / 54 |
| Medium | 46 / 65 / 68 | 43 / 64 / 70 |
| Large | 43 / 64 / 75 | 44 / 65 / 74 |

— [ICLR Fig. 5](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [arXiv v1 Fig. 5](https://arxiv.org/pdf/2505.08078v1)

- **Diversity**: "value-based RL methods result in much more diverse trajectories after batch online RL" (Fig. 4, end-effector state-visitation heatmaps of successful trajectories on Lift and Square). The attributed reason: "value-based RL methods can use the Q-function to determine which states and actions are desirable even in failure trajectories" (§4.1). — [ICLR §4.1, Fig. 4](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **Iterations and rollouts**:
  - In the main text, Square and Stack "were converged for IL-based methods but not for value-based RL". Longer runs are shown in App. A Fig. 12 (Square to 15 iterations, reaching about 90; Stack to 30 iterations, reaching about 58).
  - More rollouts per iteration (200 / 300 / 400) lift saturated runs (App. A Fig. 13, Can, Square and Stack).
  — [ICLR App. A](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

**Temporally correlated noise, Figure 8**
- Setup: Ornstein–Uhlenbeck noise added to executed actions during rollouts. θ = 5 and σ = 0.05 for Robomimic, θ = 5 and σ = 0.03 for MimicGen, θ = 0.1 and σ = 0.03 for Pen.
- Value-based RL without / with noise (arXiv v1): small 54 / 59, medium 70 / 74, large 74 / 80. ICLR, approximately: 54 / 57, 69 / 72, 75 / 81.
- The authors' reading: the noise "enables higher performance at each data level. However, it does not improve data scaling," and "our recipe does not hinge on it" (§5).
- It helps most on Adroit Pen (App. A).
- Limitations section: "directly adding noise may not be applicable in some deployment settings, though we find empirically that adding a small amount of noise only changes the success rate of the policy marginally."
— [ICLR §5, §7, App. A–B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

**Real robot** (§5, Fig. 10, App. C, identical in both versions)
- One task, "Tape": a Franka Research 3 (7-DoF) hangs a tape roll on a hook.
  - Inputs: RGB from a wrist and a side D435 camera plus proprioception, with a ResNet-18 encoder.
  - Control at 5 Hz, episodes of at most 200 steps.
  - Initial tape position is randomized within about 15 × 15 cm.
  - Scripted success detection (pixel colour above the hook) and automatic resets.
- Protocol: 5 demonstrations, N = 3 iterations × M = 30 rollouts.
- Baselines:
  - filtered-IL;
  - "a steering baseline adapted from (Nakamoto et al., 2024)", with Q trained on 90 rollouts plus D0 and actions sampled from the fixed π0.
- Paper's claim: "Our recipe obtains 30% improvement over the initial policy in just 3 iterations … Filtered-IL does not improve upon the initial performance … The steering baseline performs the worst, indicating that it is necessary to train the policy on the new rollouts."
- My reading of Fig. 10 (success rate, iterations 0 → 3):
  - recipe ≈45 → ≈57 → ≈70 → ≈73;
  - filtered-IL ≈67 → ≈57 → ≈60 → ≈60;
  - steering flat at ≈48.
- The number of evaluation trials is not stated anywhere.
- AWR / explicit extraction was not run on the robot.
— [ICLR §5, Fig. 10, App. C](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

**Discussion and limitations**
- The ICLR version adds two open questions:
  - "are techniques from offline RL the most optimal for learning a Q-function, or do we need something beyond pessimism specific to batch online RL?"
  - "can we improve implicit policy extraction beyond choosing the actions with the best Q-value?"
- Stated limitations:
  - continuous actions only; discrete or discretized actions may behave differently;
  - the initial policy must already have some success.
— [ICLR §6–7](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)

### Inferences
- **The "implicit" actor is never trained on the advantage.** It is plain IL on all accumulated data, and the paper describes no filtering of the actor's data for the recipe. All of the improvement comes from test-time best-of-64 under Q, plus the better data this produces in the next iteration. The claim is therefore mostly about the data flywheel, not about single-shot offline extraction.
- **AWR gets worse, it does not just plateau.** Explicit AWR falls below its own starting point on every task, for example Square 62 → 11 and Threading 49 → 4 in v1. That is worse than the unweighted-IL collapse the authors describe for IL. Two explanations fit:
  - an AWR configuration problem, such as an untuned β or unnormalized exponentiated advantages dominated by a few samples;
  - a genuine failure of AWR under a shifting data mix.
  With no β reported and no sweep, the paper cannot tell these apart.
- **The claim "implicit beats explicit" holds for iterated batch collection with a from-scratch MLP diffusion actor, an IQL Q(s,a) critic and simulation state inputs.** It is not shown for:
  - a single offline pass;
  - pretrained VLAs;
  - action chunks;
  - a real robot.
  At iteration 0, a single offline pass from demonstrations, AWR is ahead (average 47–48 vs 38–40).
- **Which policy class the explicit arm uses is unstated.** Fig. 6's explicit Pre average (47–48) is far above Fig. 7's Gaussian-explicit Pre (19), so the Fig. 6 explicit arm is probably a diffusion policy trained with an AWR-weighted diffusion loss. The paper never says so.
- **Real-robot trajectories cross, so the headline is fragile.** The filtered-IL curve starts about 20 points above the recipe at iteration 0, even though both should start from the same π0. Trial counts are also not given. The "30%" reads as roughly +28 percentage points absolute (≈45 → ≈73), and the final gap to filtered-IL is about 13 points.

### Gaps
- AWR β, advantage normalization, weight clipping, and whether the explicit arm used a diffusion or Gaussian actor: none is stated in either version.
- Evaluation trial counts on the real robot, and whether Q-selection was used during evaluation or only during data collection: not stated.
- Action chunking or temporal action representation: not studied. The only temporal element is the OU rollout noise.
- Whether the actor is retrained from scratch or fine-tuned each iteration: not stated.

## Q2. PA-RL: how it works, how it compares to AWR-style and re-ranking baselines, and real-robot results

### Takeaway
- **Method.** PA-RL samples k = 32 actions from the current policy, keeps the top m = 10 by Q, takes T = 10 gradient-ascent steps on Q with respect to the action, and distills the result into the policy with its native supervised loss. Distillation targets are either the argmax action or a draw from softmax(Q) over the candidates.
- **Simulation.** It beats IDQL, which re-ranks with no distillation: aggregate 568 → 711 vs 497 → 540 over 8 tasks.
- **AWR-style evidence is indirect.** Self-imitation / AWR on a single stale sample scores about 0 on antmaze-large-diverse, where PA-RL scores about 97.
- **Real robot.** On a WidowX, success goes from 50% (DDPM BC) to 90% and 100% with online RL. OpenVLA goes from 40% to 70% after 40 minutes online. Evaluation trial counts are not given.
- **Critic.** It needs Q(s,a) and its action gradient.

### Cited Findings

**Method (§4, Eqs. 4.1–4.5, Algorithms 1–2)**
- Global optimization: sample k actions from π_φ(·|s), rank them by Q_θ and keep the top m (Eq. 4.1). — [PA-RL §4.1](https://arxiv.org/pdf/2412.06685)
- Local optimization: T steps of a^{j+1} = a^j + α∇_a Q_θ(s, a^j) (Eq. 4.2). A step is kept only if it raises Q (Algorithm 1). The dataset action a_data(s) can also be added to the candidate set. — [PA-RL §4.1, §4.3, Alg. 1](https://arxiv.org/pdf/2412.06685)
- **Distillation target** (Eq. 4.3): π^Opt(a|s,m) = 1[a ∈ Ã^T] · exp(Q(s,a)) / Σ_{a'} exp(Q(s,a')) over the optimized candidates. There is no temperature parameter.
  - Every state in the dataset or replay buffer is annotated with a sample from π^Opt (Eq. 4.4).
  - The policy is trained with its standard loss on these annotations: the DDPM noise-prediction loss for diffusion (Eq. 4.5), cross-entropy for autoregressive policies.
  - "we avoid ever backpropagating through the denoising chain."
  — [PA-RL §4.2](https://arxiv.org/pdf/2412.06685)
- **Stated difference from AWR and filtered NLL**: those methods "largely use the dataset or replay buffer action for training via an NLL loss, PA-RL samples new actions from the policy, optimizes them against the critic, and then trains the policy via NLL on this action. This allows PA-RL to make aggressive updates, thus avoiding the 'slowness' associated with supervised regression" (§2). §4.2 adds: "we outperform methods that use advantage-weighted regression (AWR) for policy extraction since PA-RL deviates farther away from the data manifold." — [PA-RL §2, §4.2](https://arxiv.org/pdf/2412.06685)
- **Critic integration**:
  - With IQL: "simply replace the advantage-weighted regression (AWR) update with the above supervised learning update."
  - With Cal-QL: the optimized candidate set is also used for the TD-backup targets (Algorithm 2).
  — [PA-RL §4.3](https://arxiv.org/pdf/2412.06685)
- **Hyperparameters (App. B.1)**:
  - k = 32 samples, filtered to the top 10, then 10 gradient steps with step size 3e-4.
  - Critic: learning rate 3e-4, γ = 0.99, batch 256, CQL α = 0.005 (AntMaze, Kitchen) or 0.01 (CALVIN, real robot).
  - Rewards are biased so that the maximum per-step reward is 0: bias −1 for AntMaze and the real robot, −4 for Kitchen and CALVIN.
  - Diffusion policy: IDQL architecture with T = 5 diffusion steps and LN_ResNet (hidden size 256, 3 blocks).
  - Image domains use a ResNet-18 trained from scratch, and the action is concatenated at every critic MLP layer "so that the network places more importance to the actions".
  — [PA-RL App. B.1](https://arxiv.org/pdf/2412.06685)
- **Argmax vs softmax distillation** (Tables 5–6):
  - Softmax is better on kitchen-partial (95.3 vs 89.4), kitchen-complete (94.5 vs 90.3) and kitchen-mixed (75.2 vs 68.0).
  - Argmax is better on CALVIN (60.7 vs 46.6), where the standard deviation of candidate Q-values is only 0.02 (vs 1.56 / 2.66 / 11.54 on the kitchen tasks).
  - Rule given: "If the variance is too small … training only with the argmax performs better."
  — [PA-RL App. B.1](https://arxiv.org/pdf/2412.06685)

**Simulation results** (5 seeds × 32 evaluations per seed; App. A)

Table 1, offline → after 1k online episodes, Cal-QL critic:

| Task | IDQL (re-ranking, no distillation) | DQL (reparameterized gradient) | DPPO | Cal-QL (Gaussian) | PA-RL + Cal-QL |
|---|---|---|---|---|---|
| **Aggregate, 8 tasks** | **497 → 540** | **322 → 373** | **215 → 442** | **309 → 629** | **568 → 711** |
| CALVIN | 19 → 35 | 19 → 22 | 13 → 18 | 6 → 36 | 28 → 61 |
| kitchen-complete | 65 → 72 | — | — | — | 59 → 90 |
| antmaze-large-diverse | 66 → 69 | 22 → 38 | — | — | 73 → 95 |

— [PA-RL Table 1](https://arxiv.org/pdf/2412.06685)

- **Baseline tuning**: DQL was "extensively tuned", with an η sweep per environment and offline checkpoint selection. IDQL used the IDQL-Imp variant with τ = 0.9 (AntMaze) or 0.7 (other tasks). — [PA-RL App. B.2](https://arxiv.org/pdf/2412.06685)
- **AWR-type comparisons**:
  - **Self-imitation / AWR** (§5.3(4), Fig. 6). This baseline optimizes "weighted log likelihoods on one action sample from a stale policy (e.g., the behavior policy of the offline dataset or the replay buffer), turn off local optimization, and the policy distillation loss is weighed by positive action advantages (or exponentiated advantages)". It "fails to get any positive performance" on antmaze-large-diverse-v2 (≈0 over 1k episodes) while PA-RL scores ≈83 → ≈97.
  - **Standard IQL** (IQL's own extraction is AWR; Gaussian policy), Table 2, @1k episodes: kitchen-partial 40 → 60, mixed 48 → 48, complete 57 → 50. PA-RL + IQL with a diffusion policy scores 62 → 75, 69 → 73 and 63 → 88. This confounds extraction method with policy class.
  — [PA-RL §5.3, Fig. 6, Table 2](https://arxiv.org/pdf/2412.06685)
- **Ablations**:
  - Without global optimization: antmaze-large-diverse 0 → 0, vs 73 → 93 for full PA-RL. CALVIN 215 → 389 vs 234 → 455 (Table 4).
  - Number of samples k (Fig. 12): antmaze-large-diverse ≈0 at k = 1, ≈11 at k = 2, ≈61 at k = 3, ≈91 at k = 4 and ≈97 for k ≥ 5. CALVIN is "33% higher … 32 samples … compared to only 1 sample".
  - Local steps T (Fig. 11): CALVIN gains up to 20% with more steps. Antmaze is already at 96% with T = 0.
  — [PA-RL Table 4, App. D.3](https://arxiv.org/pdf/2412.06685)
- **Autoregressive transformer (128 bins)**: PA-RL + Cal-QL improves partial 33 → 95, mixed 42 → 84 and complete 8 → 90, "26% better than … tanh-Gaussian policies" and "by 224%" (Table 2). — [PA-RL Table 2](https://arxiv.org/pdf/2412.06685)

**Real robot** (§5.2, Table 3, Fig. 1, App. C)
- **Setup**: WidowX-250, 6-DoF, one third-person camera. Diffusion policies run at 5 Hz and autoregressive ones at 3 Hz. Sparse reward from detected object positions and gripper state. Manual reset after every trial. Tasks:
  - (a) cup to drying rack, with distractors, 10 demonstrations;
  - (b) pot to sink, 20 demonstrations without distractors, fine-tuned and tested with distractors;
  - (c) vegetable to sink with OpenVLA and no demonstrations.
  - 20 warm-up episodes were collected before online updates.
  — [PA-RL §5.2](https://arxiv.org/pdf/2412.06685)
- **Table 3 (diffusion)**:
  - Cup to rack: DDPM BC (offline) 50%; iterated filtered BC (online, following Zhou et al.) 50%; Cal-QL + PA-RL 55% offline → 90% online.
  - Pot to sink: DDPM 50%; filtered BC not run; PA-RL 80% → 100%.
  - "75–100% higher success rate within 40–110 minutes", i.e. 10–70 episodes of autonomous interaction.
  - There is a performance drop during the first 50 online episodes on task (a).
  — [PA-RL Table 3, §5.2.1](https://arxiv.org/pdf/2412.06685)
- **Fig. 1 summary bars**:
  - Average over 2 real diffusion tasks: DP 50%, DP + PA-RL offline only 67.5%, online 95%.
  - OpenVLA: 40%, offline only 50%, online 70%.
  — [PA-RL Fig. 1](https://arxiv.org/pdf/2412.06685)
- **OpenVLA (§5.2.2, App. C)**:
  - 50 zero-shot trials gave a 40% success rate. These trials pre-trained a Cal-QL critic using a cache of 16 OpenVLA action samples per state, and "the OpenVLA policy is not updated in this offline RL phase".
  - Online, the optimized actions were distilled via LoRA (rank 32). The action cache was recomputed after each distillation with 12 processes. The number of base-policy samples was halved for memory.
  - After about 1 hour of zero-shot trials plus 40 minutes of online fine-tuning: 70%. The main error fixed was grasping the distractor object.
  — [PA-RL §5.2.2, App. C](https://arxiv.org/pdf/2412.06685)
- **Evaluation trial counts** for Table 3 and for the 70% OpenVLA number are not reported, apart from the 50 zero-shot trials. — [PA-RL §5.2](https://arxiv.org/pdf/2412.06685)
- **Wall-clock and cost**:
  - On CALVIN, PA-RL "improves at a similar rate per unit amount of wall-clock time as IDQL" but continues to a higher level: about 60 at about 22 hours on TPU v4 (Fig. 18).
  - Policy distillation is cheaper in memory than backpropagating through the diffusion chain "by a factor equal to the number of denoising steps" (App. E).
  - Stated main limitation: "PA-RL requires sampling multiple actions from the policy, which is expensive for large foundation policies" (§6).
  — [PA-RL App. E, §6](https://arxiv.org/pdf/2412.06685)

### Inferences
- **Steering vs distillation on OpenVLA.** Because OpenVLA was frozen offline, "OpenVLA + PA-RL offline only" (50%) is effectively inference-time steering with the critic, the V-GPS style. That gives +10 points over base (40%). Distilling after 40 minutes online adds another 20 points (70%). This is the only same-robot, same-task split between inference-time and training-time implicit extraction across the three papers, with unknown trial counts.
- **The cleanest implicit-vs-weighted-imitation evidence in PA-RL is the SIL/AWR-style ablation.** It is a single task, antmaze-large-diverse, where k = 1 also fails for PA-RL itself. The failure therefore mainly reflects "one stale sample" rather than "weights" as such.
- **PA-RL's distillation is itself a weighted imitation** (softmax over exp(Q)), but over fresh on-policy candidates. The key variable the authors identify is where the imitated actions come from (sampled from the current policy and optimized), not whether weights are used.
- **Almost all gains are with online interaction** (Cal-QL online fine-tuning). The offline-only real-robot improvements are +5 points (cup) and +30 points (pot, under distractor shift).

### Gaps
- Evaluation trial counts on the real robot; seeds for the real-robot runs.
- Action chunking: PA-RL itself uses single-step actions. Chunking (6 / 8 / 4) appears only for the DPPO baseline (App. B.2).
- No direct AWR-on-diffusion (weighted DDPM loss on dataset actions) baseline at matched policy class across the benchmarks.
- Contradiction within the paper: §5.2.1 says the pre-trained diffusion policy "did not produce any success under distribution shift on task (b)", yet Table 3 lists DDPM at 50% on pot to sink. The paper does not reconcile the two.

## Q3. V-GPS: how value-guided re-ranking works, per-policy gains, when steering hurts, and comparison to fine-tuning

### Takeaway
- **Method.** V-GPS uses one language-conditioned Q(s,a,l), trained with Cal-QL (or IQL) on Bridge V2 plus Fractal. At each step it samples K actions from a frozen generalist and executes the argmax, or a softmax(Q/β) draw.
- **Real robot.** Only Octo-small-1.5 was tested, on a WidowX over 6 tasks × 20 trials: average success 0.24 → 0.44 (+20 points, +82.8% relative).
- **SIMPLER.** Average gains for 5 policies are small, +0.01 to +0.16, from mostly very low baselines. There are several per-task regressions.
- **Over-exploitation.** It shows up as K grows, especially for IQL.
- **Baselines.** There is no AWR or weighted-imitation baseline. Plain fine-tuning on Bridge hurt Octo.

### Cited Findings

**Method, critic and cost**
- **Action selection**:
  - "Sample K actions … from the generalist policy π, and query the value function Q_θ", then a_t ∼ Softmax(Q(s,a_1)/β, …, Q(s,a_K)/β) (Algorithm 1, Eq. 2).
  - Real robot: K = 50 with greedy selection (β → 0).
  - SIMPLER: "we swept over K∈{10,50} and β∈{0,0.1,1.0} and report the best result for each policy" (App. B.2).
  — [V-GPS §5.2, App. B.2](https://arxiv.org/html/2410.13816v2)
- **Critic**:
  - A single Q(s,a,l) for all policies and both embodiments, trained on Bridge V2 (45K language-annotated demonstrations) plus Fractal (130K episodes) (§6.1).
  - Architecture: ResNet-34 with FiLM language conditioning (frozen MUSE embeddings), the 7-D action concatenated after the encoder, and a 2 × 256 MLP. 25.6M parameters, vs Octo-small 27M, Octo-base 93M, RT1-X 35M and OpenVLA 7B (App. B.1, Table 13).
  - Cal-QL: α = 5.0, γ = 0.98, 1M steps, batch 512, learning rate 3e-4, clipped double-Q, crop and color-jitter augmentation.
  - IQL: τ = 0.7, 200K steps.
  - Reward: "the final 3 steps of each trajectory" get 0, the rest −1 (App. B.1).
  — [V-GPS §6.1, App. B.1](https://arxiv.org/html/2410.13816v2)
- **Inference cost** (Octo-small; Table 9, App. G): base 0.0752 s; K = 10 → 0.0963 s (1.28×); K = 30 → 1.46×; K = 50 → 0.1196 s (1.59×); K = 100 → 0.1596 s (2.12×). The authors found no significant slowdown in practice because WidowX actions are issued every 0.2 s. — [V-GPS App. G](https://arxiv.org/html/2410.13816v2)

**Real-world Table 1** (Octo-small-1.5 → V-GPS; WidowX-250; 20 trials per task with object poses randomized each trial; App. D)
- Scene A: green pepper in pot 0.15 → 0.35; sweet potato on cloth 0.30 → 0.35.
- Scene B: mushroom on cloth 0.35 → 0.70; mushroom in pot 0.30 → 0.55.
- Scene C: sushi in pot 0.10 → 0.30; spoon in pot 0.25 → 0.40.
- Average 0.24 → 0.44 (+82.8%).
- Stated failure modes fixed (§4, §6.2, Fig. 4): imprecise grasp of a slippery object, premature release, and holding the object too long.
- No other policy was evaluated on the real robot.
— [V-GPS Table 1, §6.2, App. D](https://arxiv.org/html/2410.13816v2)

**SIMPLER Table 2** (Cal-QL critic; 100 trials × 3 seeds per task for Octo and RT1-X, 50 trials for OpenVLA)

| Policy | Total average, base → V-GPS | WidowX average | Google Robot average |
|---|---|---|---|
| Octo-small | 0.27 → 0.34 | 0.30 → 0.38 | 0.22 → 0.27 |
| Octo-base | 0.17 → 0.18 | 0.17 → 0.20 | 0.17 → 0.14 |
| Octo-small-1.5 | 0.02 → 0.18 | 0.01 → 0.13 | 0.07 → 0.29 |
| RT1-X | 0.12 → 0.14 | 0.02 → 0.03 | 0.32 → 0.36 |
| OpenVLA | 0.24 → 0.27 | 0.05 → 0.07 | 0.62 → 0.69 |

Largest single gains: eggplant basket 0.49 → 0.84 (Octo-small) and 0.01 → 0.44 (Octo-small-1.5); pick can 0.05 → 0.43 (Octo-small-1.5). — [V-GPS Table 2](https://arxiv.org/html/2410.13816v2)

- **Per-task regressions** (Table 2):
  - Octo-small spoon on towel 0.52 → 0.46;
  - Octo-base spoon 0.25 → 0.21 and pick can 0.29 → 0.24;
  - RT1-X put near 0.44 → 0.42;
  - OpenVLA carrot 0.06 → 0.04.
  - The paper gives no systematic explanation for these.
  — [V-GPS Table 2](https://arxiv.org/html/2410.13816v2)
- **IQL critic** (Table 3): total averages 0.27 → 0.31, 0.17 → 0.18, 0.02 → 0.14, 0.12 → 0.15 and 0.24 → 0.31. OpenVLA eggplant 0.14 → 0.54. — [V-GPS Table 3](https://arxiv.org/html/2410.13816v2)
- **Over-exploitation vs K** (Table 10; Octo-small; baseline eggplant 0.49, pick coke 0.31):

| K | Eggplant, IQL | Eggplant, Cal-QL | Pick coke, IQL | Pick coke, Cal-QL |
|---|---|---|---|---|
| 10 | 0.59 | 0.77 | 0.30 | 0.38 |
| 30 | 0.47 | 0.81 | 0.37 | 0.38 |
| 50 | 0.42 | 0.84 | 0.31 | 0.38 |
| 100 | 0.35 | 0.63 | 0.37 | 0.36 |

  Authors: "increasing K leads to the exploitation of the value function, resulting in performance degradation" for IQL, while "the Cal-QL value function is more robust to K" (App. H). Cal-QL at K = 100 still drops relative to K = 50 (0.84 → 0.63). — [V-GPS App. H](https://arxiv.org/html/2410.13816v2)

**Controls and fine-tuning comparison**
- **Both parts are needed** (Table 12, eggplant): random selection among Octo samples 0.49 (= base); V-GPS applied to random actions 0.00; V-GPS 0.84. — [V-GPS App. J](https://arxiv.org/html/2410.13816v2)
- **Critic's own actor** (Table 11): the IQL and Cal-QL actors, which are Gaussian, score 0.00 on spoon and eggplant. The paper blames the extreme bimodal gripper open/close targets, which such an actor "is too simple to model". — [V-GPS App. I](https://arxiv.org/html/2410.13816v2)
- **Fine-tuning comparison** (Table 7, SIMPLER WidowX average):
  - Octo-small 0.30;
  - Octo fine-tuned on Bridge 0.22;
  - Octo trained from scratch on Bridge 0.01;
  - ResNet diffusion policy on Bridge 0.12;
  - V-GPS with IQL 0.34, with Cal-QL 0.38.
  - "V-GPS is the only method that achieves better performance than the generalist policy." The authors note the generalists were already pretrained on OXE, which contains Bridge and Fractal.
  - There is no AWR, advantage-weighted or filtered-BC fine-tuning baseline anywhere in the paper.
  — [V-GPS App. E](https://arxiv.org/html/2410.13816v2)
- **Data size** (Table 8, eggplant, IQL): the critic trained on 100%, 50% or 10% of Bridge scores 0.59, 0.59 and 0.55, vs base 0.49. — [V-GPS App. F](https://arxiv.org/html/2410.13816v2)
- **Limitations** (§7):
  - the policy must be stochastic;
  - extra compute "might limit its applicability in high-frequency tasks";
  - limited ability to handle "completely unseen languages and objects" because the critic sees only 2 embodiments.
  — [V-GPS §7](https://arxiv.org/html/2410.13816v2)

### Inferences
- **Steering helps most when the policy's distribution already contains good actions** that it samples with low probability, such as gripper timing and grasp point. Dong et al.'s real-robot steering baseline (flat ≈48% vs recipe ≈73%) suggests that steering alone stalls when the needed actions are rare under π0.
- **The SIMPLER gains are optimistic.** The best K and β were chosen per policy on the evaluation tasks. Many base rates are at or near 0 (for example Octo-small-1.5 at 0.01–0.02), so relative gains overstate absolute effects.
- **The critic sees only demonstration datasets.** It is trained on Bridge and Fractal "demonstrations" with a last-3-steps reward, with no explicit failure data. What the critic separates is progress within successful demonstrations, not success from failure.

### Gaps
- How chunk-predicting policies (Octo, and OpenVLA in some configs) are scored: whole chunk vs first action, and execution horizon. Three extraction passes found no statement on this.
- Whether Bridge and Fractal contain failed episodes: not stated.
- Only one policy was tested in the real world, and there is no real-robot ablation of K or β.

## Q4. What do the three papers say about critic errors, Q(s,a) vs V(s), action chunks, and side effects (slower or hesitant policies, loss of diversity)?

### Takeaway
- **Critic exploitation.** All three treat over-exploitation of the critic as the main risk of implicit extraction. They limit it by staying inside the base policy's samples, with small K or step budgets, conservative or distributional critics, and PA-RL's accept-only-if-Q-rises steps.
- **Q vs V.** All three require Q(s,a). V(s) appears only inside IQL.
- **Chunks.** None evaluates re-ranking of action chunks.
- **Side effects.** None measures hesitancy or episode length. Dong reports more state diversity with value-based RL, and PA-RL finds local optimization does not shrink action spread.

### Cited Findings
- **Critic errors, PA-RL**:
  - CEM started from random actions "finds actions where the Q-function greatly overestimates values". Predicted Q-values end up far above Monte-Carlo returns, "even much higher than the predicted Q-values further into training" (App. D.4, Fig. 14).
  - A regression (MSE) Cal-QL critic produced "large positive" Q-values although all rewards are ≤ 0. "The distributional critic bounds the predicted values by design, which limits over-estimation" (Fig. 14).
  - PA-RL switched to an HL-Gauss distributional critic whenever a seed showed "exploding Q-values", which happened on AntMaze and Kitchen. CALVIN and the real robot used MSE (App. B.1).
  - Sampling candidates from the pre-trained policy is motivated because "the critic can produce erroneous values outside the support of the dataset" (§2).
  - CEM also "can average the different modes of behavior", producing OOD actions "lying in the middle" of two modes (§5.3, App. D.5).
  — [PA-RL §2, §5.3, App. B.1, D.4–D.5](https://arxiv.org/pdf/2412.06685)
- **Critic errors, V-GPS**: IQL degrades as K grows (eggplant 0.59 at K = 10 → 0.35 at K = 100, below the 0.49 base). Cal-QL is more robust to K but still drops at K = 100 (App. H, Table 10). — [V-GPS App. H](https://arxiv.org/html/2410.13816v2)
- **Critic errors, Dong**: the stated open question is whether Q-learning needs "something beyond pessimism specific to batch online RL" (§6). Dong reports no Q over-estimation diagnostic. — [Dong ICLR §6](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf)
- **Q vs V**:
  - Dong's implicit selection uses argmax_a Q(s,a), and AWR uses Q(s,a) − V(s) from IQL (§3).
  - PA-RL needs Q for ranking and ∇_a Q for local steps (Eqs. 4.1–4.2).
  - V-GPS scores sampled actions with Q(s,a,l) (Eq. 2).
  — [Dong §3](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [PA-RL §4.1](https://arxiv.org/pdf/2412.06685); [V-GPS §5.2](https://arxiv.org/html/2410.13816v2)
- **Action chunks**:
  - Dong: none mentioned.
  - PA-RL: chunking only for the DPPO baseline (App. B.2).
  - V-GPS: no statement found.
  — [Dong App. B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [PA-RL App. B.2](https://arxiv.org/pdf/2412.06685)
- **Diversity**:
  - Dong: value-based RL with implicit extraction yields "much more diverse trajectories" than filtered-IL (Fig. 4). An expressive policy is "necessary to both generate and consume diverse data" (§1).
  - PA-RL: local optimization changes the action-sample standard deviation "by less than 0.1% on average", so it "does not lead to action sample duplication" (App. D.3, Fig. 13).
  - PA-RL: the benefit of local optimization decays during fine-tuning (L1 action change falls, Fig. 13 right).
  — [Dong §4.1](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [PA-RL App. D.3](https://arxiv.org/pdf/2412.06685)
- **Speed and latency**:
  - V-GPS: 1.28–2.12× per-step inference time for K = 10–100 with Octo-small (App. G).
  - PA-RL: sampling many actions is "expensive for large foundation policies", and they cached 16 OpenVLA samples per state (§5.2.2, §6).
  - None of the papers reports task-completion time or hesitancy.
  — [V-GPS App. G](https://arxiv.org/html/2410.13816v2); [PA-RL §5.2.2, §6](https://arxiv.org/pdf/2412.06685)
- **Weighting temperatures**:
  - Dong's AWR: β not reported.
  - PA-RL's π^Opt: exp(Q) with implicit temperature 1 over about 10 candidates, or argmax when the candidate Q standard deviation is tiny (Tables 5–6).
  - V-GPS: β ∈ {0, 0.1, 1.0}, with greedy selection on the real robot.
  - None normalizes or clips advantages.
  — [PA-RL App. B.1](https://arxiv.org/pdf/2412.06685); [V-GPS App. B.2](https://arxiv.org/html/2410.13816v2)

### Inferences
- **A V(s)-only critic cannot run any of the three implicit schemes directly.** Ranking K freshly sampled actions or chunks requires a value for untaken actions, which V(s) and a one-step TD advantage on the logged action do not provide. Options are:
  - add a Q head over the chunk, Q(s, a_{t:t+H});
  - or keep explicit weighting, the only extraction that works with dataset actions and V.
- **PA-RL's HL-Gauss finding directly supports a distributional critic.** It bounds over-estimation under action search. It says nothing about the V-only case, because no action search is done against V.
- **Per-step overhead for a large VLA is not covered by these measurements.** The 1.59× at K = 50 was for a 27M-parameter Octo. Sampling K flow chunks from a multi-billion-parameter VLA plus a separate critic encoder costs something else again. This is outside all three papers' measurements.

### Gaps
- No paper studies chunk-level Q, chunk re-ranking, or the interaction of re-ranking with open-loop chunk execution.
- No paper measures motion speed, episode length, pauses, or hesitancy under steering or weighting.

## Q5. Critique: evaluation sizes, fairness of the AWR baselines, and what transfers to a V-only critic with a single large flow VLA

### Takeaway
- **Direct AWR test.** Only Dong compares AWR to best-of-N at a matched policy class, and its AWR arm is under-specified (β, normalization and tuning are unreported) and degrades on every task.
- **Real-robot evidence.** Across the three papers it is thin:
  - Dong: 1 task, trial counts unreported;
  - PA-RL: 2 diffusion tasks plus 1 OpenVLA task, evaluation trial counts unreported;
  - V-GPS: 1 policy × 6 tasks × 20 trials.
- **What transfers.** The central lesson is to extract from fresh samples of the current expressive policy, scored by a conservative or bounded Q, and to retrain on the new data. That requires Q(s, chunk), which a V-only critic does not provide.

### Cited Findings
- **Evaluation sizes**:
  - Dong simulation: 3 seeds × 100 trials. Real robot: 5 demonstrations, 3 × 30 rollouts, evaluation trials not stated (Fig. 3, 6 captions; §5).
  - PA-RL simulation: 5 seeds × 32 evaluations (App. A). Real-robot evaluation trials not stated apart from the 50 OpenVLA zero-shot trials (§5.2, App. C).
  - V-GPS: 20 trials per real task, and SIMPLER 100 × 3 seeds (Octo, RT1-X) or 50 trials (OpenVLA) (App. D, Table 2).
  — [Dong](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [PA-RL](https://arxiv.org/pdf/2412.06685); [V-GPS](https://arxiv.org/html/2410.13816v2)
- **Baseline tuning asymmetry**:
  - Dong reports sampled-action count 64 and IQL τ = 0.8, but no AWR β (App. B Table 1).
  - PA-RL documents an extensive DQL η sweep (App. B.2), but no AWR sweep; its AWR-like SIL was tested on one task (Fig. 6).
  - V-GPS reports the best of a K × β sweep per policy in simulation (App. B.2).
  — [Dong App. B](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [PA-RL App. B.2](https://arxiv.org/pdf/2412.06685); [V-GPS App. B.2](https://arxiv.org/html/2410.13816v2)
- **Scale of the actors**:
  - Dong's actors are small from-scratch MLP diffusion or Gaussian policies (simulation) and ResNet-18 (real robot) (App. B, §5).
  - PA-RL's largest actor is OpenVLA-7B, distilled via LoRA rank 32 (§5.2.2).
  - V-GPS never updates the actor.
  — [Dong](https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf); [PA-RL](https://arxiv.org/pdf/2412.06685); [V-GPS](https://arxiv.org/html/2410.13816v2)

### Inferences
- **For a V(s) HL-Gauss critic with per-sample weighted flow-matching loss** (exp(Â/β) with Â batch-normalized and clipped to ±3), Dong's negative AWR result is weak evidence against that setup:
  - Dong's AWR used raw Q − V with an unreported β and no normalization or clipping, which is exactly the regime where exponential weights can concentrate on a few samples;
  - Dong's degradation appears only across repeated batch-collection iterations, not in a single offline pass. At iteration 0, AWR beat implicit extraction (47–48 vs 38–40).
- **What does transfer from Dong**:
  - filtered-IL (success-only BC) plateaus;
  - value-based use of failure data increases state diversity and scales with more rollouts;
  - implicit extraction plus retraining on new data keeps improving across iterations, while steering a fixed policy (V-GPS style) stalls.
- **What does not transfer**:
  - Best-of-N, PA-RL global and local optimization, and V-GPS all need Q over sampled actions. With V(s) and a one-step TD advantage, only dataset actions can be scored.
  - PA-RL's local ∇_a Q step would also need a differentiable Q over the full chunk.
  - Re-ranking K chunks from a large flow VLA multiplies the per-step sampling cost. PA-RL's workaround was an offline cache of 16 samples per state recomputed after each distillation.
- **Nearest analogues for a flow VLA with a Q-chunk critic**:
  - V-GPS-style steering at deployment, for gains without touching weights: Octo-small-1.5 real-robot 0.24 → 0.44; OpenVLA inside PA-RL 40% → 50%;
  - PA-RL-style distillation of re-ranked samples via the native flow loss, the larger gain: OpenVLA 40% → 70%, diffusion 50% → 90–100%, but with online data.
  Neither has been shown on chunked flow policies in these papers.

### Gaps
- None of the three papers tests a V-only critic, chunked actions, flow-matching actors, or a pretrained VLA with weighted-loss fine-tuning (AWR on a flow or diffusion loss) against sampling-based extraction at matched data. The key comparison for this setup is not directly measured in any of them.
- OpenReview reviews for Dong, which might show whether the AWR tuning was questioned, were inaccessible.
