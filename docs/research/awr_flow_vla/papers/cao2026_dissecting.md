# Cao et al. 2026, "Dissecting Advantage-Guided Post-Training for Vision-Language-Action Policies" (arXiv 2609.28161)

Source note: I read the full v1 PDF myself (9 pages: main text, Figs. 1-5, Tables I-II, references). There is no appendix. The project page adds no code, hyperparameters or extra numbers. Figure 4 numbers below come from the numeric labels printed in the figure; they agree with the ranges and means stated in the Sec. V-B text. Figure 5 has no numeric labels, so any value read from it is marked as read by eye.

URLs: PDF = https://arxiv.org/pdf/2609.28161 ; HTML = https://arxiv.org/html/2609.28161v1 ; abs = https://arxiv.org/abs/2609.28161 ; project page = https://dissectvla.github.io/

## Does the paper exist, with this ID, title and first author?

### Takeaway
Yes. arXiv 2609.28161v1 is exactly this paper. It was submitted 23 Sep 2026 in cs.RO, and the first author is Jiahang Cao (SJTU, working as an intern at Xiaomi Robotics). The title, ID, author and year in the description are all correct.

### Cited Findings
- Title: "Dissecting Advantage-Guided Post-Training for Vision-Language-Action Policies". Version v1 was submitted Wed 23 Sep 2026 14:10 UTC (2,652 KB), under cs.RO, with licence CC BY-NC-ND 4.0. — [arXiv abs](https://arxiv.org/abs/2609.28161)
- Authors: Jiahang Cao, Hanye Zhao, Hang Lai, Shenyu Zhang, Xiaoshen Han, Xinghang Li, Futeng Liu, Wanli Peng, Heyun Wang, Yunhong Wang, Jason Li, Yong Yu, Weinan Zhang. Cao and Zhao contributed equally. Hang Lai and Weinan Zhang are the corresponding authors. Affiliations are (1) School of Computer Science, Shanghai Jiao Tong University and (2) Xiaomi Robotics. Cao, Zhao and Han did the work as interns at Xiaomi Robotics. — [PDF p.1](https://arxiv.org/pdf/2609.28161)
- Abstract headline: across four real-world bimanual tasks, the recipe (TD advantage construction, group-wise calibration, continuous advantage weighting) improves mean task progress by 0.42 and mean success by 0.63 over the SFT initialization. — [PDF abstract](https://arxiv.org/pdf/2609.28161)
- Structure: I Intro, II Related Work, III Preliminaries (A construction, B calibration, C utilization), IV Study Design (Stage I/II/III), V Experiment Results (A setup, B RQ1, C RQ2, D RQ3), VI Conclusion. There is no appendix and no limitations section. — [PDF](https://arxiv.org/pdf/2609.28161)
- The paper builds on RECAP (ref [35], arXiv 2511.14759). The authors deliberately leave RECAP out as a condition (see the weighting section). I did not read RECAP, per instructions. — [PDF Sec. II, IV-C](https://arxiv.org/pdf/2609.28161)
- The implementation uses the LeRobot codebase (ref [43]). — [PDF Sec. V-A](https://arxiv.org/pdf/2609.28161)
- No code or dataset release is linked. The project page repeats Table II success numbers and the task descriptions, and hosts videos. — [project page](https://dissectvla.github.io/)

### Inferences
- The paper is 3 days old (as of 2026-09-26). It is v1 and has not been peer-reviewed.

### Gaps
- None on identity.

## Claim-by-claim check of the description

### Takeaway
The description is mostly accurate but compressed in ways that matter:
1. "binary weighting" is ambiguous. In the paper, the binary advantage-based scheme *is* hard filtering (Filter). The other baseline is DAgger, which gives binary weights by data source and uses no advantage at all.
2. "beats" means higher point estimates at 20 trials per task. The paper reports no confidence intervals or significance tests for Table II.
3. Several details are left out: "n-step" means n = 10 chunks (10 s), the reward is sparse, the critic is IQL, calibration is per source × value bin with dataset-level statistics, the clip is on beta·z of the calibrated advantage, and the loss is self-normalized over the batch.

### Cited Findings
| Part of the description | What the paper says | Verdict |
|---|---|---|
| "n-step TD advantages" | IQL critic, chunk-level n-step TD advantage (Eq. 3). n = 10 chunks = 300 primitive actions = 10 s, gamma = 0.96 per chunk, sparse terminal reward. It was chosen over Q−V (Eq. 2) and Monte-Carlo V using offline ordering diagnostics (Fig. 4). No other n was tried, and 1-step TD was never tested. | Correct, but underspecified |
| "value-based calibration" | "Value" grouping (Sec. IV-B, Eq. 4): V_target is clipped to [0,1] and split into B = 8 equal-width bins **per data source**. Advantages are z-scored within each bin using dataset-level statistics computed once. The floor sigma_min = 0.1 × the source's global std. Bins with < 32 samples fall back to source-level statistics. It was chosen by offline eta² (Table I, mean 0.026 vs 0.092 Raw). The real-robot ablation covers only 2 tasks (Fig. 5). The Conclusion itself uses the phrase "value-based calibration". | Correct |
| "continuous exp weighting (beta=1, clip=2)" | w_i = exp(clip(beta·z_i, −c, c)) with beta = 1 and c = 2 (Sec. IV-C). z_i is the *calibrated* (group-standardized) advantage. The clip acts on the exponent, so w lies in [e^−2, e^2] = [0.135, 7.39]. The loss is self-normalized: Σ w ℓ / Σ w over the batch (Eq. 5). | Correct |
| "beats hard filtering" | Filter: w = 1 if z_i is above the q = 0.5 quantile **of the training batch**, else 0 (Imit-RECAP rule from HABC [41]). Weight beats Filter on every task and both metrics: mean progress 0.86 vs 0.75, mean success 0.74 vs 0.50 (Table II). | Correct as point estimates |
| "beats ... binary weighting" | The only other binary scheme is **DAgger**, an "advantage-free binary rule" from HG-DAgger. Human samples (demos and takeover segments) get w = 1 and autonomous rollouts get w = 0 (Sec. IV-C). Weight beats it: 0.86 vs 0.71 progress, 0.74 vs 0.45 success (Table II). The paper calls Filter the "binary" form, as opposed to continuous (Sec. IV-C). The Conclusion says "over source-based DAgger and hard filtering". | Imprecise. The second baseline weights by data source and uses no advantage. |
| "on 4 real tasks" | Four real-world **bimanual** tasks (PhoneE, PhoneH, Box, Printer) on two 6-DoF arms with π0.5, 20 trials per condition per task (Sec. V-A, Table II). | Correct |
| (implied) "beats" | "Weight achieves the highest point estimates". No confidence intervals or significance tests for Table II. One training run per condition as far as stated; seeds are not mentioned (Sec. V-D, Table II). | Holds for point estimates, untested statistically |

Sources: [PDF Sec. IV-B, IV-C, V-A, V-B, V-D, Eqs. 3-5, Tables I-II, Figs. 4-5, Sec. VI](https://arxiv.org/pdf/2609.28161)

### Inferences
- A more accurate one-line version: "IQL critic with 10-s n-step TD advantages (sparse reward), per-source value-bin z-scoring, and w = exp(clip(z, ±2)) self-normalized per batch had the highest point estimates on 4 real bimanual tasks (20 trials each). It beat median-batch hard filtering and source-based HG-DAgger weighting, plus a demos-only SFT checkpoint that received no additional updates."

### Gaps
- No uniform-weight baseline (plain BC/SFT continued on the same mixed dataset for the same 4000 updates), so the description cannot be checked against "plain BC on the same data". See the critique section.

## Setup: model, robot, tasks, data, evaluation

### Takeaway
π0.5, a flow-matching VLA. One bimanual robot with two 6-DoF arms. Four real tasks, each with its own task-specific SFT policy. Per task, the data is about 40 h of expert demos, 100 autonomous rollouts and 100 rollouts with human takeovers. Evaluation is 20 trials per condition per task, scored as sub-task progress and success.

### Cited Findings
- Actor: "the π0.5 flow-matching actor" is the representative VLA backbone. The per-sample loss ℓ_i is π0.5's flow-matching loss. — [PDF Sec. I, III-C](https://arxiv.org/pdf/2609.28161)
- Robot: bimanual platform, two 6-DoF arms each with a gripper, and three RGB fisheye cameras (one on the base, one on each wrist). At each query the policy receives end-effector states and three undistorted images. It predicts a chunk of C = 30 end-effector actions, executed at 30 Hz (1 s per chunk) before the next query. — [PDF Sec. V-A](https://arxiv.org/pdf/2609.28161)
- Tasks (Fig. 1, Sec. V-A):
  - Pack Phone Easy (PhoneE): phone into its box, close the lid.
  - Pack Phone Hard (PhoneH): more complex box plus an accessory-insertion step, longer horizon.
  - Pack Box (Box): three objects into a carton, fold the flaps, press the side tabs.
  - Refill Printer Paper (Printer): pick up a small paper stack and load it into the tray.
  - The authors say the set spans "object insertion, long-horizon sequencing, multi-object interaction, and deformable-object handling". — [PDF Sec. I, V-A](https://arxiv.org/pdf/2609.28161)
- Data per task: a fixed offline dataset of about 40 h of expert demos (used to train the task-specific SFT initialization), 100 autonomous rollout episodes and 100 rollout episodes with human takeovers. All critic and advantage experiments use this fixed dataset. There is no online interaction. — [PDF Sec. V-A](https://arxiv.org/pdf/2609.28161)
- Sub-tasks: 4-6 per task, defined by hand with text descriptions. The detailed lists are only in the supplementary video. — [PDF Sec. IV-A](https://arxiv.org/pdf/2609.28161)
- Metrics: task progress is the fraction of sub-tasks completed at termination. Success means all sub-tasks are completed. A trial ends on success, on an unrecoverable failure (an object becomes unreachable, or the robot starts a later sub-task before finishing an earlier one), or at a task-specific timeout equal to the duration of the longest expert episode. — [PDF Sec. V-D](https://arxiv.org/pdf/2609.28161)
- Protocol: 20 real-robot trials per condition per task. All policies share the same embodiment, observation and action spaces, deployment interface, and "evaluation initial-condition schedule". — [PDF Sec. V-A, Table II](https://arxiv.org/pdf/2609.28161)

### Inferences
- Policies are per task. SFT is "task-specific", and post-training "further trains the SFT checkpoint", so the paper never deals with multiple tasks sharing one actor or one normalization.
- About 40 h of demos per task is a large demo base next to 200 rollout episodes. The rollout/takeover data is probably a minority of frames, but the paper gives no episode lengths.

### Gaps
- Rollout success rates, rollout episode lengths and hours, and the number of takeover events per episode are not stated. The only related figure is about 200 intervention pairs per task used for R_int (Fig. 4 caption).
- The batch sampling ratio between demos, rollouts and takeovers is not stated.
- SFT training steps and learning rate are not stated. Whether evaluation was blinded or conditions interleaved is not stated.

## Advantage estimator (Stage I)

### Takeaway
Seven critic × construction combinations were scored offline by two pairwise ordering diagnostics. Every n-step TD variant (n = 10 chunks) is well above chance: R_int 0.639-0.773, R_prog 0.805-0.953. Q−V and MC sit near chance (about 0.42-0.62). IQL (expectile tau = 0.8) with n-step TD was chosen: mean R_int 0.718, mean R_prog 0.897. The critic is not distributional, and neither n nor 1-step TD was ablated.

### Cited Findings
- Reward: sparse. r_t = 1 only on the terminal transition of a successful episode, 0 otherwise. d_t marks termination. Stated motive: "to avoid task-specific dense reward engineering". — [PDF Sec. III](https://arxiv.org/pdf/2609.28161)
- Time scale: t indexes action chunks, and s_{t+1} is C = 30 primitive steps (1 s) after s_t. Rewards, discounting, masks and the bootstrap state are all chunk-level. gamma = 0.96 per chunk. — [PDF Sec. III, III-A, V-A](https://arxiv.org/pdf/2609.28161)
- Eq. 1: y_t = r_t + gamma(1−d_t)V_target(s_{t+1}); L_Q = E[MSE(Q(s_t, a_t), sg[y_t])]. V_target is an EMA of V (the EMA rate is not given). — [PDF Eq. 1](https://arxiv.org/pdf/2609.28161)
- Eq. 2: A^{Q−V} = Q(s_t, a_t) − V(s_t). — [PDF Eq. 2](https://arxiv.org/pdf/2609.28161)
- Eq. 3: A^{n-step TD} = R_t^n + gamma^n m_{t,n} V_target(s_{t+n}) − V(s_t), where R_t^n = Σ_{j=0}^{n−1} gamma^j m_{t,j} r_{t+j} and m_{t,j} = Π_{k<j}(1−d_{t+k}), m_{t,0} = 1. — [PDF Eq. 3](https://arxiv.org/pdf/2609.28161)
- n = 10 chunks: nC = 300 primitive actions = 10 s of execution. This is the only n used. — [PDF Sec. V-A](https://arxiv.org/pdf/2609.28161)
- The four value-learning schemes (seven configurations):
  - IQL: Q by TD (Eq. 1), V by expectile regression on in-dataset Q with tau = 0.8.
  - SARSA: the tau = 0.5 endpoint.
  - ALOE [10]: policy-sampled TD target Q_target(s_{t+1}, â_{t+1}) with â ~ π; V derived from policy-sampled Q.
  - MC: V fit directly to Monte-Carlo returns; no Q, so it is scored only with n-step TD.
  - IQL, SARSA and ALOE are each scored with Q−V and with n-step TD, giving 3×2 + 1 = 7 configurations.
  - The paper notes MC is the target choice RECAP uses. — [PDF Sec. IV-A](https://arxiv.org/pdf/2609.28161)
- Critic architecture: each Q or V is "a separate encoder-only Transformer followed by an MLP output head, and reuses the visual embedding of the VLA policy [10]". The loss is MSE for Q and expectile/MSE for V. There is no distributional or HL-Gauss head. Whether the reused embedding is frozen is not stated. — [PDF Sec. V-A, Eq. 1](https://arxiv.org/pdf/2609.28161)
- Critic training: 10,000 updates at batch size 2048 per critic checkpoint. — [PDF Sec. V-B](https://arxiv.org/pdf/2609.28161)
- Offline diagnostics (Sec. IV-A, Fig. 3):
  - R_int: each human-takeover window is paired with the equal-length policy-controlled window just before it. Advantages are averaged within each window. R_int is the fraction of takeover events where the takeover window scores higher. About 200 pairs per task.
  - R_prog: GPT-5.6 segments trajectories into sub-tasks. All segments of successful trajectories count as "advancing". In failed trajectories, completed segments are "advancing" and the attempted-but-incomplete segment is "stalled". Advantages are averaged over six 1-s windows per segment. Each stalled segment is matched to an advancing segment by distance between SmolVLM2-500M embeddings of the three cameras, excluding same-trajectory matches and pairs beyond the task's 95th-percentile distance. R_prog is averaged within each sub-task, then across sub-tasks. About 400 pairs per task.
  - Chance is 0.5 for both. Fig. 4 shows 95% CIs. — [PDF Sec. IV-A, Fig. 3, Fig. 4 caption](https://arxiv.org/pdf/2609.28161)
- Fig. 4 values (printed labels). Means marked * are my arithmetic. The paper states the IQL means and SARSA's R_prog mean (0.898).

  | Config | R_int PhoneE | PhoneH | Box | Printer | mean | R_prog PhoneE | PhoneH | Box | Printer | mean |
  |---|---|---|---|---|---|---|---|---|---|---|
  | IQL + n-step TD (selected) | 0.700 | 0.751 | 0.684 | 0.737 | 0.718 | 0.882 | 0.910 | 0.843 | 0.953 | 0.897 |
  | SARSA + n-step TD | 0.679 | 0.655 | 0.667 | 0.773 | 0.694* | 0.908 | 0.928 | 0.806 | 0.948 | 0.898 |
  | ALOE + n-step TD | 0.691 | 0.693 | 0.680 | 0.639 | 0.676* | 0.905 | 0.930 | 0.805 | 0.906 | 0.887* |
  | IQL + Q−V | 0.594 | 0.474 | 0.553 | 0.619 | 0.560* | 0.556 | 0.482 | 0.539 | 0.576 | 0.538* |
  | SARSA + Q−V | 0.614 | 0.503 | 0.434 | 0.495 | 0.512* | 0.546 | 0.578 | 0.485 | 0.542 | 0.538* |
  | ALOE + Q−V | 0.440 | 0.465 | 0.500 | 0.423 | 0.457* | 0.490 | 0.528 | 0.496 | 0.502 | 0.504* |
  | MC (n-step TD scoring) | 0.522 | 0.444 | 0.465 | 0.505 | 0.484* | 0.568 | 0.578 | 0.537 | 0.559 | 0.561* |

  — [PDF Fig. 4, Sec. V-B](https://arxiv.org/pdf/2609.28161)
- Text: "For IQL, SARSA, and ALOE critics, n-step TD yields higher concordance than Q−V on every task and both diagnostics ... Rint ranging from 0.639 to 0.773 and Rprog from 0.805 to 0.953. In contrast, the Q−V variants and the MC baseline are generally closer to chance." — [PDF Sec. V-B](https://arxiv.org/pdf/2609.28161)
- Why Q−V fails, per the authors: real-robot datasets have limited action diversity. With narrow action coverage, V is derived from Q at the same or a similar action, so Q−V cancels the shared signal and leaves a small, noisy residual. They also note that prior work deliberately collects diverse failure actions for critic learning [42]. MC is weaker because fitting V to realized returns leaves only small TD residuals along the logged continuations. — [PDF Sec. V-B](https://arxiv.org/pdf/2609.28161)
- Why IQL: it has the highest four-task mean R_int (0.718), and its R_prog of 0.897 is close to SARSA's best 0.898. Its upper-expectile regression "can emphasize high-value in-dataset actions, a desirable property for heterogeneous datasets". — [PDF Sec. V-B](https://arxiv.org/pdf/2609.28161)

### Inferences
- IQL, SARSA and ALOE with n-step TD are within about 0.04 of each other on both means. That gap is similar to the size of the Fig. 4 95% CIs (binomial SE at about 200 pairs and p ≈ 0.7 is about ±0.06; at about 400 pairs and p ≈ 0.9 about ±0.03; my arithmetic). What separates cleanly is the construction (n-step TD vs Q−V/MC), not the critic.
- Relevance to a 1-step TD, V-only setup: the paper's evidence favours "residual along the logged continuation" (Eq. 3) over "Q minus V". The reader's A = r + gamma·V_target(s') − V(s) with s' about 1 s later is Eq. 3 with n = 1. The paper never tests n = 1 against n = 10, so it cannot tell you whether a 10-s horizon beats 1 s. With a sparse terminal reward and n = 1, the advantage is almost entirely the critic's own 1-s value change. With n = 10 it spans 10 s of logged behaviour, and gamma^10 = 0.96^10 ≈ 0.665.
- R_int compares a human window with a policy window. A critic that simply prefers human-like motion would also score well on it. R_prog (within-sub-task, embedding-matched) is the cleaner check.

### Gaps
- No ablation over n, gamma, tau or critic training steps. No comparison with GAE. No distributional critic. Whether the critic's reused visual embedding is frozen, which layer is reused, the EMA rate and the learning rate are all not stated.
- R_int and R_prog are computed on raw advantages. They are not re-checked after calibration, even though per-source z-scoring could change a takeover-vs-policy ordering if those windows fall in different sources.

## "Value-based calibration" (Stage II)

### Takeaway
Calibration is group-wise z-scoring (Eq. 4). Groups are data source × one of 8 equal-width bins of V_target clipped to [0,1]. Statistics are computed once over the whole dataset and not per batch. Its purpose is to stop sub-task and progress context from dominating advantage levels under sparse terminal reward. Offline, it cuts the sub-task-explained variance eta² from 0.092 (Raw) to 0.026 (Table I). On the robot, dropping it for Global or Temporal "degrades performance in most cases", but this was tested on only 2 tasks (Fig. 5). An un-calibrated (Raw) arm was never run on the robot.

### Cited Findings
- Eq. 4: z_i = (u_i − mu_{g_i}) / max(sigma_{g_i}, sigma_min), with mu_g and sigma_g² the within-group mean and variance. Standard normalization is the single-group special case. — [PDF Eq. 4, Sec. III-B](https://arxiv.org/pdf/2609.28161)
- On batch vs dataset normalization, the paper only says that advantages are "commonly normalized within a batch or dataset to stabilize policy optimization [37]", and that recent RL post-training normalizes within "structurally homogeneous groups" [38]. It does not compare batch-level with dataset-level statistics. Its own statistics are "computed once from the dataset, holding them constant during Stage III". — [PDF Sec. III-B, IV-B](https://arxiv.org/pdf/2609.28161)
- Motivation: "In long-horizon tasks with sparse terminal rewards, sub-tasks differ in state distribution and remaining horizon, so raw advantage levels may vary systematically with task progress." Calibration should make advantages reflect "an action's relative quality within its decision context rather than the absolute progress of that context". — [PDF Sec. IV-B](https://arxiv.org/pdf/2609.28161)
- Per-source statistics: "Since policy rollouts and expert demonstrations may come from different behavior distributions, normalization statistics are computed separately by source." Sub-task labels are used only for the eta² diagnostic and never by any calibration operator. — [PDF Sec. IV-B](https://arxiv.org/pdf/2609.28161)
- The four variants:
  - Raw: u_i unchanged.
  - Global: one z-score per source (Eq. 4).
  - Temporal: each source split into B equal-width bins of episode-relative time (after TGRPO [9]).
  - Value: V_target clipped to [0,1], each source split into B equal-width value bins.
  - Settings: B = 8 per source. Bins with < 32 samples use the source-level (Global) statistics. sigma_min = 0.1 × that source's global std. — [PDF Sec. IV-B](https://arxiv.org/pdf/2609.28161)
- Eq. 6: eta² = Σ_k n_k (z̄_k − z̄)² / Σ_i (z_i − z̄)², the fraction of advantage variance explained by sub-task identity. It is computed within each task. Lower means more comparable across sub-tasks. — [PDF Eq. 6, Sec. V-C](https://arxiv.org/pdf/2609.28161)
- Table I (eta², lower is better):

  | Calibration | PhoneE | PhoneH | Box | Printer | Mean |
  |---|---|---|---|---|---|
  | Raw | 0.072 | 0.063 | 0.106 | 0.127 | 0.092 |
  | Global | 0.057 | 0.051 | 0.090 | 0.120 | 0.080 |
  | Temporal | 0.071 | 0.023 | 0.053 | 0.028 | 0.044 |
  | Value | 0.053 | 0.027 | 0.013 | 0.010 | 0.026 |

  — [PDF Table I](https://arxiv.org/pdf/2609.28161)
- Why Value beats Temporal, per the authors: "variations in execution speed, stalls, and recoveries cause temporal bins to mix different progress levels, whereas state value more closely tracks task progress." — [PDF Sec. V-C](https://arxiv.org/pdf/2609.28161)
- Real-robot ablation (Fig. 5, Weight utilization held fixed, tasks PhoneH and Printer only): "replacing Value calibration with Global or Temporal also degrades performance in most cases". The figure prints no numeric bar values. No Raw condition was run on the robot. — [PDF Fig. 5, Sec. V-D](https://arxiv.org/pdf/2609.28161)

### Inferences
- The effect on weights: after per-source z-scoring, every source (demos, rollouts, and takeovers if they form their own source) has mean-zero calibrated advantages. So Weight does **not** down-weight rollouts as a block relative to demos. It only reweights within each source × value-bin cell.
- Value binning compares an action only with actions from states of similar estimated value. This removes the correlation between A and progress level. A pooled batch z-score, like the reader's setup, would keep any systematic A-vs-V trend.
- eta² can be driven to about 0 by any operator that destroys the signal, since pure noise gives eta² ≈ 0. So a low eta² is necessary but not sufficient. The paper has no check that calibration preserves the Stage I orderings.
- PhoneH is a small counterexample to "lower eta² is better": Temporal has lower eta² (0.023) than Value (0.027) there (Table I), yet in Fig. 5 by eye Value performs better on the robot for PhoneH.
- Cross-task value scales: the paper never faces them (per-task policies, eta² computed within task). It has no guidance on per-task vs pooled normalization for a multi-task actor. Per-source × value-bin grouping would handle scale differences within a task.

### Gaps
- There is no real-robot Raw (un-calibrated) arm, so the paper has no number for "what happens without calibration". Global is the closest proxy (Fig. 5, 2 tasks, values only in the figure).
- No batch-level vs dataset-level statistics comparison, and no per-task analysis.
- The number of sources is never listed explicitly. It is presumably demos, autonomous rollouts and takeover episodes. Whether a takeover episode's policy and human segments are separate sources is not stated.

## Weighting schemes and real-robot results (Stage III)

### Takeaway
There are three post-training arms, each run for 4000 updates at batch size 2048 on the same dataset:
- Weight (exp(clip(z, ±2)))
- Filter (0/1 at the batch median of z)
- DAgger (human = 1, autonomous = 0, no advantage)

A demos-only SFT checkpoint with no further updates serves as reference. Weight has the highest point estimate on every task and both metrics: mean progress 0.86 and success 0.74, against Filter 0.75/0.50, DAgger 0.71/0.45 and SFT 0.44/0.11. beta = 1 and c = 2 are confirmed. The clip applies to beta times the calibrated z, not to raw A.

### Cited Findings
- SFT Init is "trained on the demonstrations alone and evaluated without any further actor update". — [PDF Sec. IV-C](https://arxiv.org/pdf/2609.28161)
- DAgger is "an advantage-free binary rule following HG-DAgger [26], [27]: human samples, including both demonstrations and takeover segments, receive unit weight, while autonomous rollouts receive zero weight." — [PDF Sec. IV-C](https://arxiv.org/pdf/2609.28161)
- Weight "follows the principle of AWR [32]": w_i = exp(clip(beta·z_i, −c, c)), with z_i the calibrated advantage, beta = 1 and c = 2. — [PDF Sec. IV-C](https://arxiv.org/pdf/2609.28161)
- Filter gives "unit weight to the samples whose calibrated advantages are above the q-quantile of the training batch and zero weight to the remaining samples, with q = 0.5". This is the hard-filtering rule of the Imit-RECAP baseline in HABC [41]. — [PDF Sec. IV-C](https://arxiv.org/pdf/2609.28161)
- RECAP is left out on purpose: its advantage conditioning is introduced during pre-training, so a post-training-only re-implementation "departs from the original recipe and has been reported to yield more modest gains over supervised fine-tuning [42]". Filter is used as "a hard approximation that discards low-advantage samples rather than conditioning on them". There is no advantage-conditioning arm. — [PDF Sec. IV-C](https://arxiv.org/pdf/2609.28161)
- What each comparison isolates, per the authors: Weight vs Filter isolates the weighting form (same advantages, continuous vs binary). Weight/Filter vs DAgger compares weighting by estimated action quality with weighting by data source. All three vs SFT Init test whether post-training helps. — [PDF Sec. IV-C](https://arxiv.org/pdf/2609.28161)
- Budget: "Each post-training condition further trains the SFT checkpoint with 4000 updates and batch size 2048". All three "optimize the same π0.5 flow-matching actor through Eq. 5 using the same dataset and training budget". — [PDF Sec. IV-C, V-D](https://arxiv.org/pdf/2609.28161)
- Table II (20 trials per condition per task; progress / success):

  | Condition | PhoneE | PhoneH | Box | Printer | Mean |
  |---|---|---|---|---|---|
  | SFT Init | 0.50 / 0.30 | 0.32 / 0.00 | 0.69 / 0.15 | 0.24 / 0.00 | 0.44 / 0.11 |
  | DAgger | 0.65 / 0.35 | 0.62 / 0.30 | 0.91 / 0.70 | 0.66 / 0.45 | 0.71 / 0.45 |
  | Weight | 0.88 / 0.80 | 0.81 / 0.70 | 0.95 / 0.80 | 0.79 / 0.65 | 0.86 / 0.74 |
  | Filter | 0.78 / 0.70 | 0.69 / 0.40 | 0.91 / 0.55 | 0.60 / 0.35 | 0.75 / 0.50 |

  — [PDF Table II](https://arxiv.org/pdf/2609.28161)
- Deltas (my arithmetic from Table II; progress / success):

  | Comparison | PhoneE | PhoneH | Box | Printer | Mean |
  |---|---|---|---|---|---|
  | Weight − SFT | +0.38 / +0.50 | +0.49 / +0.70 | +0.26 / +0.65 | +0.55 / +0.65 | +0.42 / +0.63 (stated) |
  | Weight − Filter | +0.10 / +0.10 | +0.12 / +0.30 | +0.04 / +0.25 | +0.19 / +0.30 | +0.11 / +0.24 |
  | Weight − DAgger | +0.23 / +0.45 | +0.19 / +0.40 | +0.04 / +0.10 | +0.13 / +0.20 | +0.15 / +0.29 |
  | Filter − DAgger | +0.13 / +0.35 | +0.07 / +0.10 | 0.00 / −0.15 | −0.06 / −0.10 | +0.04 / +0.05 |

- Success counts out of 80 pooled trials (my arithmetic): Weight 59, Filter 40, DAgger 36, SFT Init 9. — [PDF Table II](https://arxiv.org/pdf/2609.28161)
- The authors' explanation: "Weight's finer-grained sample-level supervision may contribute to these gains: it emphasizes higher-scoring actions while retaining nonzero weights for the remaining samples, reducing sensitivity to occasional critic misranking." — [PDF Sec. V-D](https://arxiv.org/pdf/2609.28161)
- The only qualitative observation: "Weight exhibits more consistent recovery behaviors in real-robot trials, as shown in the supplementary video." — [PDF Sec. V-D](https://arxiv.org/pdf/2609.28161)
- Headline claim: "the resulting recipe produced the highest point estimates for both task progress and success on every task". — [PDF Sec. VI](https://arxiv.org/pdf/2609.28161)

### Inferences
- The scale of A before beta: A is a within-group z-score with dataset-level statistics and a floor sigma_min = 0.1 × the source's global std. It is not raw, and not batch-standardized. With beta = 1, clip(beta·z, ±2) is simply z clipped to ±2, so w ∈ [0.135, 7.39] and the max/min ratio is e^4 ≈ 55.
- The paper's beta multiplies z (an inverse temperature). The AWR-style exp(A/beta) divides by it. At beta = 1 the two are identical.
- Mapping to the reader's formula: w = exp(clip(Â, ±3)) / mean(w) with lam = 1 matches the paper's form: the paper's Eq. 5 divides by Σw over the batch, which equals dividing w by mean(w). The two differences are (a) clip 3 vs 2, which raises the weight ratio from e^4 ≈ 55 to e^6 ≈ 403, and (b) how Â is standardized (pooled per effective batch vs per source × value bin over the dataset). The paper has no lam-mixing term.
- Filter did worse than the advantage-free DAgger on success for Box (0.55 vs 0.70) and Printer (0.35 vs 0.45), and on progress for Printer (0.60 vs 0.66). Hard-thresholding at the batch median is not reliably better than weighting by data source.

### Gaps
- No sweep of beta, c or q. Baselines use single settings (q = 0.5 only), so neither the baselines nor Weight were tuned per arm. The learning rate is not stated.
- No uniform-weight (plain BC/SFT on the full mixed data) arm, no advantage-conditioning arm, and no "Weight without calibration" robot arm.

## Where the weight is applied

### Takeaway
Per training example (one state plus a 30-step action chunk). It multiplies only the π0.5 flow-matching loss, and it is self-normalized over the batch: L = Σ w_i ℓ_i / Σ w_i (Eq. 5). No other loss terms are mentioned.

### Cited Findings
- "Let ℓ_i(θ) denote the supervised policy loss for example i, which is the flow-matching loss used by the π0.5 actor ... L_π(θ) = Σ_{i∈B} w_i ℓ_i(θ) / Σ_{i∈B} w_i, Σ_{i∈B} w_i > 0." — [PDF Sec. III-C, Eq. 5](https://arxiv.org/pdf/2609.28161)
- Advantages are defined per decision step t, and t indexes action chunks. — [PDF Sec. III, III-A](https://arxiv.org/pdf/2609.28161)
- Calibration statistics are fixed dataset-level constants. Filter's threshold is recomputed per training batch (the q-quantile of the batch). — [PDF Sec. IV-B, IV-C](https://arxiv.org/pdf/2609.28161)

### Inferences
- Under Filter, Eq. 5 averages over only the retained half of each batch, which effectively halves the batch. Under DAgger, the autonomous-rollout rows are wasted. Weight uses every row. So "same budget" (updates × batch) is not the same effective sample count across arms.

### Gaps
- Not stated: whether π0.5 is trained with any auxiliary or language (FAST or subtask token) losses here, and if so whether they are weighted; whether every frame within a chunk shares one advantage; and the per-frame vs per-chunk sampling granularity of the dataloader.

## Failure modes and diagnostics

### Takeaway
The paper monitors only offline advantage-quality diagnostics: R_int and R_prog for construction, eta² for calibration. It reports no weight diagnostics (effective sample size, weight histograms, clip-hit fraction, per-source mean weight), no episode length or speed, and no failure-mode breakdown. It names ESS as something pipeline choices change but never measures it.

### Cited Findings
- The introduction names the risks: "Equal weighting may expose the actor to failures as if they were demonstrations, whereas overly aggressive filtering may discard corrective or recovery behavior." It also says changing one stage "may also alter the sample weights, sample exposure, and effective sample size produced by the complete pipeline." — [PDF Sec. I](https://arxiv.org/pdf/2609.28161)
- Diagnostics used: R_int and R_prog with 95% CIs (Fig. 4), eta² (Table I), and in Fig. 5 R_int, R_prog and 1−eta² overlaid on robot outcomes. The authors claim alignment: "higher Stage-I ordering concordance and lower Stage-II eta² generally correspond to higher real-world task progress and success." — [PDF Figs. 4-5, Table I, Sec. V-D](https://arxiv.org/pdf/2609.28161)
- The robustness argument for continuous weights is that they reduce "sensitivity to occasional critic misranking". This is stated as a possible explanation and not measured. — [PDF Sec. V-D](https://arxiv.org/pdf/2609.28161)
- Behavioural observation is limited to "more consistent recovery behaviors" (video only). — [PDF Sec. V-D](https://arxiv.org/pdf/2609.28161)

### Inferences
- With calibrated z clipped at ±2, weight concentration is bounded by construction (ratio ≤ e^4). The paper may not have needed ESS monitoring for that reason. The reader's ±3 clip on batch z allows about 7× more concentration.
- Nothing in the paper tells you whether AWR makes motion slower or more hesitant. The timeout equals the longest expert episode, so a slower policy would show up only as lower progress or success.

### Gaps
- No ESS, no weight statistics, no episode duration or time-to-completion, no sensitivity to beta or to critic errors, and no training curves.

## Data conditions: mixture, size, number of tasks, negative results

### Takeaway
The paper never varies the data mixture, dataset size or number of tasks. Every experiment uses one fixed per-task mixture (about 40 h demos, 100 autonomous rollouts, 100 takeover rollouts). Advantage weighting never hurts relative to the SFT checkpoint. But Filter falls below advantage-free DAgger on 2 of 4 tasks, and weak advantages (Q−V) visibly lower Weight's robot results on PhoneH.

### Cited Findings
- The single fixed dataset per task is described in [PDF Sec. V-A](https://arxiv.org/pdf/2609.28161).
- All three post-training conditions beat SFT Init on both four-task means. — [PDF Sec. V-D, Table II](https://arxiv.org/pdf/2609.28161)
- Filter < DAgger on success for Box (0.55 vs 0.70) and Printer (0.35 vs 0.45). — [PDF Table II](https://arxiv.org/pdf/2609.28161)
- Fig. 5 ablation (Weight fixed; PhoneH and Printer only). Six conditions: ALOE+Q−V, IQL+Q−V, SARSA+n-step and IQL+n-step (the reference), all with Value calibration, plus IQL+n-step with Temporal and IQL+n-step with Global. "Replacing the selected n-step TD construction with Q−V generally lowers both metrics." — [PDF Fig. 5 caption, Sec. V-D](https://arxiv.org/pdf/2609.28161)
- Fig. 5 by eye (no printed values; treat as ±0.1):
  - PhoneH: the Q−V conditions show roughly 0.5-0.6 progress and about 0.2-0.3 success, against 0.81/0.70 for the reference.
  - Printer: all six conditions sit within roughly 0.65-0.8 progress and 0.5-0.65 success.
  - Every ablated condition is visually far above SFT Init (PhoneH 0.32/0.00, Printer 0.24/0.00). — [PDF Fig. 5](https://arxiv.org/pdf/2609.28161)

### Inferences
- On Printer, Weight with near- or below-chance advantages (ALOE+Q−V: R_int 0.423, R_prog 0.502) still appears to lift success from 0.00 to about 0.5. Advantage-free DAgger lifts it to 0.45. So a large share of the gain over SFT Init on that task comes from 4000 more updates on rollout/takeover data, not from advantage quality. On PhoneH, advantage quality matters much more.
- Because every source is z-scored separately, the recipe keeps the between-source balance set by the sampler. How it behaves when rollouts dominate, or when demos are scarce, is untested.

### Gaps
- No experiments on dataset size, the demo:rollout ratio, a demos-only (no failures) regime, multi-task training, or more rollout rounds. No online or iterated variant.

## Critique

### Takeaway
The internal comparison of Weight vs Filter vs DAgger is fair on updates and data. Pooled over 80 trials per arm, Weight's success lead is large (59 vs 40 vs 36). Per-task differences are within the ±0.2 CI width of n = 20. Four things limit the paper:
- There is no uniform-weight BC-on-mixed-data baseline.
- The SFT reference gets neither the extra 4000 updates nor the rollout data.
- There are no seeds, CIs or tests for Table II, and no hyperparameter sweeps.
- It covers one robot, one model (π0.5) and per-task policies.

### Cited Findings
- Table II gives means over 20 trials with no variability, CIs or tests. The text uses "highest point estimates". — [PDF Table II, Sec. V-D, VI](https://arxiv.org/pdf/2609.28161)
- The SFT reference had no additional updates and saw no rollouts. The post-training arms had 4000 × 2048 more samples, including rollouts and takeovers. — [PDF Sec. IV-C, V-D](https://arxiv.org/pdf/2609.28161)
- Fixed hyperparameters: beta = 1, c = 2, q = 0.5, B = 8, n = 10, gamma = 0.96, tau = 0.8. None were swept. — [PDF Sec. IV-A, IV-B, IV-C, V-A](https://arxiv.org/pdf/2609.28161)
- The robot ablation covers 2 of 4 tasks. — [PDF Fig. 5](https://arxiv.org/pdf/2609.28161)
- The paper frames itself as a real-robot complement to simulation-based empirical studies [15]-[18]. — [PDF Sec. I](https://arxiv.org/pdf/2609.28161)

### Inferences
- Statistics (my arithmetic, not the paper's). Wilson 95% intervals at n = 20: 16/20 → about [0.58, 0.92]; 14/20 → about [0.48, 0.86]; 8/20 → about [0.22, 0.61]; 0/20 → [0, 0.16]. Per-task Weight−Filter success gaps of 0.10-0.30 are not individually clear at 0.05. Pooled and unstratified, Weight 59/80 vs Filter 40/80 gives z ≈ 3.1, and vs DAgger 36/80 gives z ≈ 3.7. So the ranking Weight > {Filter, DAgger} is likely real in aggregate. Filter vs DAgger is a wash.
- Missing baseline: uniform weights on the same mixed dataset for the same 4000 updates. The paper's intro argues that equal weighting exposes the actor to failures, but never measures it. The closest proxy is the Q−V ablation in Fig. 5, which behaves like noisy near-uniform weighting.
- The +0.42/+0.63 over SFT Init mixes three effects: extra updates, extra (rollout/takeover) data, and advantage weighting. Only the gaps between the post-training arms isolate the weighting itself.
- The Stage I and Stage II selections are within noise in places: IQL vs SARSA, and Temporal vs Value on PhoneH. The claimed alignment between diagnostics and robot outcomes rests on 6 conditions × 2 tasks with qualifiers ("generally", "in most cases").
- Generality: one lab, one bimanual 6-DoF platform, end-effector actions, π0.5 (flow-matching like MolmoAct2), a large demo base (about 40 h per task), and per-task actors. Transfer to a single 7-DoF arm, a multi-task actor, a distributional critic or 1-step TD is plausible in form but untested.

### Gaps
- Seeds, evaluation blinding and order, and per-trial failure categories are not reported. No code is released, so details the paper leaves out (learning rate, whether the critic encoder is frozen, the sampling mix) cannot be checked.
- Possible follow-ups cited by the paper (not read here): HABC, arXiv 2606.17043 (Imit-RECAP filter baseline); "Learning while deploying", arXiv 2605.00416 (post-training-only RECAP gains are modest; collecting diverse failure actions for critics); ALOE, arXiv 2602.12691; CO-RFT, arXiv 2508.02219; Stratified GRPO, arXiv 2510.06214 (group-wise normalization); Dong et al., ICLR 2026, "What matters for batch online RL in robotics?".
