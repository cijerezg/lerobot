# Fang et al. 2026 (HABC) and Zhang et al. 2026 (PTR) compared with flat AWR-style weighting

Access note: I read both papers in full: the arXiv HTML through text extraction, and every page of both PDFs rendered as images, including figures, appendices and hyperparameter tables. The only pages I skipped were Zhang's reference pages 24-25. Where a number is read off a bar chart rather than printed, it is marked "(read from bar)" with about ±2-3 pp uncertainty. Fang's "Recap" baseline is adapted from arXiv 2511.14759. Per the constraints I did not open or cite that paper, and I describe the baseline only from Fang's own text. Quick links: Fang [abs](https://arxiv.org/abs/2606.17043) / [html](https://arxiv.org/html/2606.17043) / [pdf](https://arxiv.org/pdf/2606.17043); Zhang [abs](https://arxiv.org/abs/2603.16542) / [html](https://arxiv.org/html/2603.16542) / [pdf](https://arxiv.org/pdf/2603.16542).

## Q1. Identity: do the arXiv IDs resolve to the described papers, and do the descriptions match?

### Takeaway
Both IDs resolve to the stated title, first author and 2026 date. Fang is a real-robot VLA paper (pi0.5, flow matching) whose "online RL" means rounds of collect-then-train weighted behaviour cloning with human interventions. It is not policy-gradient RL. Zhang has **no empirical AWR baseline**. It shares only AWR's exp(score/beta) weight form and compares against uniform SFT only. It also uses no reward and no value function at all.

### Cited Findings
- arXiv 2606.17043 = "Hierarchical Advantage Weighting for Online RL Fine-Tuning of VLAs from Sparse Episode Outcomes". Authors: Tongyan Fang, Siyuan Huang, Naiyu Fang, Ganlong Zhao, Zhongjin Luo, Jianbo Liu, Xiaogang Wang, Ying Dong, Hongsheng Li (ACE Robotics / Tsinghua SIGS / CUHK). v1 submitted 15 Jun 2026, cs.RO + cs.LG. — [arXiv abs](https://arxiv.org/abs/2606.17043); [PDF p.1](https://arxiv.org/pdf/2606.17043)
- The method in Fang is named "Hierarchical Advantage-Weighted Behavior Cloning (HABC)". The "hierarchy" is **two critic heads**, viability and efficiency, merged by a state-adaptive gate. It is not an episode/segment/step hierarchy. — [Fang abstract, §3.2-3.3](https://arxiv.org/html/2606.17043)
- Fang is about VLAs: "We use pi0.5 as the base VLA, initialized from its pretrained offline checkpoint". All methods share "a flow-matching VLA actor trained with a weighted flow-matching loss" (Eq. 1: L_pi = (1/B) sum_i w_i ||v_theta(s_i,a_i,sigma) - u_i||^2). — [Fang §3.1, §4.1](https://arxiv.org/pdf/2606.17043)
- Fang's "online" setup works in rounds. "Each round collects 100 autonomous rollout episodes and trains for 6k gradient steps. Among failed rollouts, approximately half receive human intervention." — [Fang App. E](https://arxiv.org/pdf/2606.17043)
- Fang's Appendix F also says: "Online fine-tuning is initialized from the pretrained pi0.5 IQL checkpoint". — [Fang App. F](https://arxiv.org/pdf/2606.17043)
- arXiv 2603.16542 = "Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting". Authors: Wanpeng Zhang, Hao Luo, Sipeng Zheng, Yicheng Feng, Haiweng Xu, Ziheng Xi, Chaoyi Xu, Haoqi Yuan, Zongqing Lu (Peking U. / Tsinghua / BeingBeyond). v1 submitted 17 Mar 2026, cs.RO. 29 pages. — [arXiv abs](https://arxiv.org/abs/2603.16542); [PDF p.1](https://arxiv.org/pdf/2603.16542)
- Zhang's baselines: "Three configurations are compared throughout: (i) SFT sets alpha=0 in Eq 21, recovering uniform supervised post-training; (ii) SFT+Belief ...; (iii) PTR enables the full pipeline with alpha=1." — [Zhang §5.1](https://arxiv.org/html/2603.16542)
- Zhang on how PTR relates to AWR: "PTR adopts the same exponential weight form as AWR but replaces reward-based advantages with a reward-free identification score derived from post-action consequences". Also: "PTR uses no reward, no value function, and no policy gradient; its connection to this literature is structural ... rather than algorithmic." — [Zhang §2](https://arxiv.org/html/2603.16542)
- Zhang's base policy is Being-H0.5, a "Mixture-of-Transformers backbone with flow matching" with a 200-dim unified action space. — [Zhang §2, §5.1](https://arxiv.org/html/2603.16542)

### Inferences
- The brief says Fang is "online RL". Read that as "iterated on-policy data collection plus offline-style weighted BC per round", not as online policy-gradient RL. Each HABC update is a weighted flow-matching step on a replay buffer, so it is structurally close to the reader's batch setting.
- The brief implies Zhang is an AWR comparison. That is incorrect. The paper is useful to the reader only for its conservative weight transform: clip [0.25, 4], mixture alpha, self-normalization, and the no-clipping ablation. Its reward-free score does not rank successes above failures.

### Gaps
- Neither paper has a code release I inspected. Fang lists github.com/ACERobotics-VLA/HABC ([PDF p.1](https://arxiv.org/pdf/2606.17043)), but I did not read it, per the "full text only" scope.

## Q2. Fang et al. 2026 (HABC): mechanism, advantages from sparse outcomes, setup, numbers, hyperparameters, and its critique of single-scalar advantage

### Takeaway
HABC is not exponential AWR:
- Transitions from failed rollouts get weight 0.
- Transitions from successful rollouts get a bounded weight g_t = 1 + tanh((1-p_v)A_v + p_v A_e), which lies in [0, 2]. It is built from two one-step TD advantages. A_v comes from a success-probability head (BCE on episode outcome). A_e comes from a steps-to-success head trained only on successes.
- SFT and intervention data get weight 1.
- Weights are unit-mean normalized. There is no temperature, no exp and no z-scoring.

Results over 50 real-robot trials on 3 bimanual tasks: success at equal budget rose from SFT 36/44/12% to HABC 60/78/22%, and the best later checkpoint reached 92/88/38%. **There is no flat exp-AWR baseline.** The closest is "Imit-Recap", which hard-thresholds a single critic's TD residual and passes only 19-27% of rollout transitions.

### Cited Findings
**Setup**
- The robot is a real ARX X5 bimanual arm. Cameras: 3 RGB streams (top RealSense D455 plus two wrist D405). Actions are end-effector frame, "chunk size of 50 during training and 25 during inference", trained on 8×A800. — [Fang §4.1](https://arxiv.org/pdf/2606.17043)
- Tasks (all contact-rich, deformable, dual-arm):
  - Pencil Pouch: insert a marker, zip the pouch.
  - Paper Bag: open a flat bag, stand it upright, insert a bottle.
  - Snack Bag: place 3 items, pull the drawstring.
  - "Each checkpoint is evaluated over 50 trials." — [Fang §4.1](https://arxiv.org/pdf/2606.17043)
- Data: 200 SFT demos per task, then rounds of 100 autonomous rollouts plus 6k gradient steps. About half of failed rollouts get a human takeover that completes the task. The Step-1 checkpoint is taken "after 3 online rounds, corresponding to 300 rollout episodes on top of 200 SFT demonstrations". Step 2 continues with "additional rounds" (number not stated). — [Fang App. E](https://arxiv.org/pdf/2606.17043)

**Critic heads (Eq. 2-4)**
- Two heads sit on a shared backbone phi(s): z_v = f_v(phi(s)), V_e = f_e(phi(s)), p_v = sigmoid(z_v). — [Fang §3.2, Eq. 2](https://arxiv.org/pdf/2606.17043)
- V_v = P(success | s) "under the current policy". It is trained with BCE against the episode outcome y on all outcome-labeled policy-execution windows, both successful and failed. — [Fang §3.2, Eq. 4](https://arxiv.org/pdf/2606.17043)
- V_e is trained only on successful windows from both policy and intervention. It uses a Huber loss on a TD target: y_e = 0 at terminal success, else -1 + sg[V_e(s_{t+1})]. Values lie in [-T_max, 0]. — [Fang §3.2, Eq. 3](https://arxiv.org/pdf/2606.17043)
- "All HABC runs use pure TD supervision for V_e". — [Fang App. F](https://arxiv.org/pdf/2606.17043)
- A separate "parent" distributional value head V_psi (logits over value bins, cross-entropy) is trained on a mixed TD/MC target, y = (1-c_t)(r_t + gamma sg[V(s_{t+1})]) + c_t R_t. The MC target is used at episode boundaries. Failed-episode transitions get r = -C with C = 100. It starts "from the pretrained IQL checkpoint". This head "is retained for compatibility with the Imit-Recap baseline, but it is not used in HABC's actor weighting." — [Fang App. A, Eq. 9; App. F](https://arxiv.org/pdf/2606.17043)

**Advantages (Eq. 5-6)**
- A_v = z_v(s_{t+1}) - z_v(s_t), with zero per-step reward. It is computed in logit space "to preserve resolution near p_v ≈ 1". — [Fang §3.3, Eq. 5](https://arxiv.org/pdf/2606.17043)
- A_e = -1 + V_e(s_{t+1}) - V_e(s_t). — [Fang §3.3, Eq. 6](https://arxiv.org/pdf/2606.17043)
- "Both are one-step TD residuals". So advantages come from learned critics via one-step TD, not from Monte-Carlo returns or group-relative baselines. — [Fang §3.3](https://arxiv.org/pdf/2606.17043)

**Gate and weights (Eq. 7-8, Alg. 1)**
- The gate is g_t = 1 + tanh((1 - p_v(s_t)) A_v + p_v(s_t) A_e). "When p_v is low, the gate emphasizes A_v ... When p_v is high, the gate emphasizes A_e". The interpolation is "per state rather than through a global training schedule". — [Fang §3.3, Eq. 7](https://arxiv.org/pdf/2606.17043)
- Pre-normalization weights:
  - g_t for successful online rollout samples.
  - **0 for failed online rollout samples.**
  - 1 for SFT and intervention samples, or g_t on intervention samples when intervention reweighting (IR) is enabled.
- These are then unit-mean normalized: c = mean over D_valid of w̃, w = w̃ / max(c, eps). The stated purpose is "decoupling the weighting from the effective learning rate". — [Fang §3.5, Eq. 8](https://arxiv.org/pdf/2606.17043)
- Algorithm 1, line 11 reads "Unit-mean-normalize non-SFT weights". — [Fang App. B.1](https://arxiv.org/pdf/2606.17043)
- Warmup: N_wu = 500 steps with g_t = 1 "until the critic heads are minimally calibrated". Even during warmup, failed-rollout samples keep w̃ = 0 (Alg. 1, lines 4-9). IR is enabled "only after an initial HABC phase". — [Fang §3.5; Alg. 1](https://arxiv.org/pdf/2606.17043)
- Fang itself contrasts AWR/AWAC, which "derive actor weights from critic advantage via exp(Â/beta)", with HABC's "bounded per-transition loss weights". — [Fang §2](https://arxiv.org/pdf/2606.17043)

**Intervention-aware credit assignment (§3.4, App. D)**
- In episodes with interventions, "only the post-intervention policy execution suffix receives the outcome label y". Earlier policy segments stay in the buffer unlabeled. Intervention windows get no outcome label but serve as imitation targets and as V_e progress targets. — [Fang §3.4](https://arxiv.org/pdf/2606.17043)
- An intervention window requires "at least 10 human-controlled steps within a 50-step window". — [Fang App. D](https://arxiv.org/pdf/2606.17043)
- The stated rationale: naive crediting "would upweight the pre-intervention policy execution segment that led to the near-failure state". — [Fang §3.4](https://arxiv.org/pdf/2606.17043)

**Baselines at equal online budget (Step 1)**
- SFT: no online data.
- Imit-DAgger: 50/50 SFT plus intervention data, no rollouts.
- Imit-Recap: single critic V_psi, Â = r + gamma(1-d)V(s') - V(s), rollout weight 1[Â ≥ eps] with eps "selected on a held-out validation set". It omits the advantage-conditioned prompt and has no intervention-aware credit assignment.
- HABC-V: V_v only.
- HABC: full method.
- Source: [Fang §4.1; App. B.2-B.3, Alg. 2-3](https://arxiv.org/pdf/2606.17043)

**Results (Figure 3, 50 trials each, Wilson 95% CI bars)**

Success rates. Only the SFT, HABC and Final values are printed in the text; the rest are read from the bars.

| Task | SFT | Imit-DAgger (read from bar) | Imit-Recap (read from bar) | HABC-V (read from bar) | HABC | HABC (cont.) (read from bar) | HABC+IR (read from bar) | Final best HABC+IR |
|---|---|---|---|---|---|---|---|---|
| Pencil Pouch | 36 | ~21 | ~38 | ~50 | **60** | ~68 | ~76 | **92** |
| Paper Bag | 44 | ~62 | ~66 | ~70 | **78** | ~80 | ~82 | **88** |
| Snack Bag | 12 | ~8 | ~14 | ~18 | **22** | ~34 | ~34 | **38** |

Sources: text numbers from [Fang §4.2 and abstract](https://arxiv.org/html/2606.17043); bar readings from the [Fang PDF Fig. 3](https://arxiv.org/pdf/2606.17043). The HTML has no numeric bar labels.

- Text claims: "HABC achieves the highest success rate on all three tasks (60%, 78%, 22%)". "HABC-V already surpasses all non-HABC baselines on every task". "Imit-DAgger underperforms SFT on Pencil Pouch and Snack Bag". — [Fang §4.2](https://arxiv.org/pdf/2606.17043)
- Trajectory length is the mean number of action frames on **successful trials only**, with std bars. "Upgrading from HABC-V to full HABC consistently reduces trajectory length on every task (−55, −162, −32 frames)". — [Fang §4.2, Fig. 3](https://arxiv.org/pdf/2606.17043)
- Trajectory lengths read from the bars:

  | Task | SFT | Imit-DAgger | Imit-Recap | HABC-V | HABC |
  |---|---|---|---|---|---|
  | Pencil Pouch | ~1500 | ~1500 | ~1970 | ~1720 | ~1665 |
  | Paper Bag | ~1720 | ~2150 | ~1900 | ~2000 | ~1840 |
  | Snack Bag | ~1240 | ~1830 | ~1680 | ~1640 | ~1610 |

  Std bars are roughly ±150-400 frames. **SFT has the shortest successful trajectories on all three tasks.** — [Fang PDF Fig. 3](https://arxiv.org/pdf/2606.17043)

**Weight diagnostics (App. G)**
- After warmup, mean pre-normalization g_t on successful autonomous transitions ≈ 0.76 / 0.68 / 0.86 (Pencil / Paper / Snack). — [Fang App. G](https://arxiv.org/pdf/2606.17043)
- Imit-Recap passes ≈ 22 / 27 / 19 of every 100 sampled autonomous transitions; "the remaining transitions are discarded". — [Fang App. G](https://arxiv.org/pdf/2606.17043)
- With IR, mean g_t on intervention windows ≈ 1.0 / 1.1 / 0.95, i.e. "near-uniform". — [Fang App. G](https://arxiv.org/pdf/2606.17043)
- No ESS and no weight histogram are reported.

**Qualitative analysis (Fig. 4-6)**
- The p_v > 0.6 region over initial pouch positions grows from step 2000 to 6000 to 10000. — [Fang §4.3, Fig. 4](https://arxiv.org/pdf/2606.17043)
- p_v drops after an OOD grasp failure and recovers during intervention. — [Fang §4.3, Fig. 4](https://arxiv.org/pdf/2606.17043)
- During an "inefficient regrasp", p_v stays high while V_e worsens, so A_e downweights the segment. — [Fang §4.3, Fig. 5](https://arxiv.org/pdf/2606.17043)
- Recovery behaviour is shown qualitatively only (Fig. 6). — [Fang §4.3, Fig. 5-6](https://arxiv.org/pdf/2606.17043)

**Hyperparameters (App. F)**
- C = 100 (failure penalty, used only for the parent head and Imit-Recap).
- N_wu = 500.
- Huber delta = 1.0.
- gamma = 0.99.
- Batch size B = 256.
- Stale-rollout cutoff = 10000 model indices.
- "keep action-expert training disabled".
- Learning rate, optimizer, eps for Imit-Recap, and per-batch source ratios are not given.
- Source: [Fang App. F](https://arxiv.org/pdf/2606.17043)

**What Fang says is wrong with single-scalar / flat advantage**
- (i) "a single scalar signal conflates the two objectives of viability and efficiency; once basic success is achieved, the binary label provides no gradient to distinguish efficient completions from slow ones". — [Fang abstract](https://arxiv.org/abs/2606.17043)
- (ii) "critic-filtered behavior cloning hard-selects transitions above a TD-advantage threshold; both approaches lose the structure that would make each signal useful at the right training stage". — [Fang §1](https://arxiv.org/pdf/2606.17043)
- (iii) Naively assigning episode outcomes across intervention boundaries leaks credit. — [Fang abstract, §3.4](https://arxiv.org/abs/2606.17043)
- Fang makes no claim about ESS collapse, exp-weight variance or temperature sensitivity of AWR. Related work cites AWM only for "a variance-reduction perspective". — [Fang §2](https://arxiv.org/pdf/2606.17043)

**Limitations stated by the authors**
- Boundary labels must be reliable.
- "V_e trains only on successful trajectories, so its signal is weakest precisely when success is rare".
- Single-task only.
- Future work: "adaptive gating, multi-step advantage estimation, and denser outcome signals". — [Fang §5](https://arxiv.org/pdf/2606.17043)

### Inferences
- **Most of HABC's work may be done by outcome filtering, not advantage weighting.** Failed-rollout samples get w = 0 at all times, including warmup, and g_t only moves successful samples within [0, 2]. HABC is therefore "success-filtered BC plus interventions plus a soft [0, 2] modulation". The paper never tests the obvious ablation: success-filtered rollouts plus interventions with g_t ≡ 1. So how much of the SFT → HABC gain comes from the critic cannot be determined.
- Imit-Recap vs HABC-V confounds three changes at once: hard vs soft weighting, single vs viability head, and with vs without intervention-aware credit. The authors themselves describe the comparison as "soft dual-head weighting with intervention-aware credit assignment".
- My Wilson 95% intervals at n = 50 overlap heavily:
  - SFT 36% → [24, 50] vs HABC 60% → [46, 72].
  - HABC-V ~50% → [37, 63] overlaps HABC.
  - Snack SFT 12% → [6, 24] vs HABC 22% → [13, 35].
- My two-proportion z-tests (unpaired):
  - Pencil 18/50 vs 30/50: z ≈ 2.4, p ≈ 0.02.
  - Paper 22/50 vs 39/50: z ≈ 3.5.
  - Snack 6/50 vs 11/50: z ≈ 1.3, p ≈ 0.18.
  - HABC-V vs HABC on Pencil (~25 vs 30 of 50): z ≈ 1.0.
  - So the efficiency head's success-rate gain is not statistically resolved. Its length gains (−32 to −162 frames) are smaller than the plotted std.
- The 92/88/38% numbers are "best checkpoint" values selected over an unstated number of Step-2 checkpoints, each evaluated on 50 trials. That selection is optimistically biased (winner's curse), so they are not equal-budget comparisons.
- On "slower policies": every online method, HABC included, has longer successful trajectories than SFT on all three tasks (e.g. Snack ~1240 vs ~1610 frames). This could be survivorship: SFT succeeds only on easy initial states. It is still the only length evidence, and it does not show HABC fixing slowness relative to BC.
- **Mapping onto the reader's setup:**
  - HABC's A_v (a logit difference of a success classifier, zero reward) and A_e (steps-to-go TD) are unscaled and summed inside tanh with no temperature or z-scoring. That contrasts with the reader's batch z-score → clip ±3 → exp(Â/1).
  - Range: HABC's weights lie in [0, 2] before normalization. The reader's lie in [e^-3, e^3] ≈ [0.05, 20], a max/min ratio of ~400 vs HABC's bounded range with outcome zeroing.
  - Normalization: HABC normalizes by the mean over valid non-SFT samples, while the reader normalizes over a 128-row batch.
  - Horizon: Fang's windows are 50 steps and its advantages are one-step. How far s_{t+1} is from s_t (one control step vs one window) is not stated, and neither is the control rate. So its TD horizon cannot be matched to the reader's ~1 s.
- **Online → offline transfer:**
  - V_v is P(success | s) "under the current policy", refreshed each round with a 10k-index stale cutoff. On a fixed offline mixture (teleop plus old rollouts), V_v would estimate success under the behaviour mixture instead. The "viability gate" logic still applies, but the round-by-round expansion of the viable region shown in Fig. 4 would not happen.
  - Intervention-aware labelling transfers directly if takeover boundaries are logged.
  - Zeroing failed-episode transitions is the reader's "success-only BC" baseline. Fang gives no evidence on whether the critic adds anything beyond it.

### Gaps
- Exact Step-1 numbers for Imit-DAgger, Imit-Recap and HABC-V are not printed; bar readings are approximate.
- The following are not reported:
  - Number of Step-2 rounds and checkpoints evaluated.
  - Learning rate and optimizer.
  - Imit-Recap's eps value and how the "held-out validation set" was scored.
  - Per-batch mix of SFT / rollout / intervention samples for HABC.
  - Whether "valid" in Eq. 8 includes zero-weighted failures.
  - Control frequency.
  - Number of interventions per round beyond "approximately half" of failures.
- Whether phi(s) is the VLA backbone or a separate critic network is not stated. Fig. 1 draws it separately from the actor.
- It is unclear what "keep action-expert training disabled" means for a flow-matching actor whose loss is Eq. 1.
- Not tested: exp-AWR with any beta; success-filtered BC without the critic; PPO/GRPO or other policy-gradient baselines; gate or normalization ablations; sensitivity to critic error; multiple seeds.

## Q3. Zhang et al. 2026 (PTR): how weights are computed, what "conservative" means, comparisons, numbers, hyperparameters, weight analysis

### Takeaway
PTR scores each training chunk by how identifiable its observed future (o_{t+Δ}) is among K mismatched futures. The score is an InfoNCE posterior-to-uniform log ratio, T_t, computed by a scorer on the policy's own hidden states. It becomes a weight w = 1 + alpha(clip_[0.25, 4](exp(T_t/beta)) - 1) inside a self-normalized flow-matching loss. "Conservative" means only the bounded density ratio from clipping: KL(q||p_D) ≤ log 16 ≈ 2.77 nats.

It is compared only to uniform SFT, on one model (Being-H0.5) with one budget:
- Clean data: small gains. LIBERO 97.8 vs 98.3 (PTR is worse); RoboCasa 55.6 vs 54.2.
- Synthetic corruptions: larger gains, up to +9.4 pp.
- Real robot, 12 tasks × 20 trials: 67.1 vs 60.8 for specialist, 63.8 vs 50.0 for generalist.

There is no AWR, filtered-BC, IQL or CQL baseline. Removing clipping costs −2.3 / −6.5 pp, with weights above 10 causing gradient spikes.

### Cited Findings
**Data tuple and loss**
- Each sample is (o_t, s_t, l, a_{t:t+L-1}, o_{t+Δ}), with L fixed and Δ that "may vary across samples". Only (o_t, s_t, l) is used at inference. — [Zhang §3](https://arxiv.org/pdf/2603.16542)
- The flow loss is ℓ_act = ||v_phi(sigma a + (1-sigma) eps, sigma, h_t, s_t) - (a - eps)||^2. — [Zhang §3, Eq. 1](https://arxiv.org/pdf/2603.16542)

**Consequence target**
- y+ = sg(g(o_{t+Δ})), where g is an EMA copy (mu = 0.999) of vision layer 12 of InternViT-300M, L2-normalized. "A frozen target space becomes misaligned ... a fully online target is unstable. EMA is a stable compromise." — [Zhang §4.2](https://arxiv.org/pdf/2603.16542)

**Candidate pool**
- Negatives are in-batch targets, cross-rank all_gather across 8 GPUs, and a FIFO queue of 1024 entries (up to 64 per sample).
- A hard-negative fraction, drawn from "same task family or similar proprioceptive state", is ramped from 0 to 0.5.
- Chunks without a valid future get w = 1.
- Source: [Zhang §4.2, §4.5](https://arxiv.org/pdf/2603.16542)

**Scorer and score (Eq. 12-14)**
- Query u_t = f(h_t, e_t): a 2-layer MLP on the pooled backbone context h_t and the pooled action-channel tokens e_t, which the action head already computes.
- Logits d_{t,i} = cos(u_t, y_i) / tau_score; posterior p̂ = softmax at index 0.
- T_t = log[p̂ / (1/|Y_t|)]. "If the posterior is uniform ... T_t = 0 and the sample falls back to uniform supervision."
- A ranking loss on permuted actions (Eq. 13) keeps the scorer action-sensitive.
- Source: [Zhang §4.2, Eq. 12-14](https://arxiv.org/pdf/2603.16542)

**Theory (Prop. 1-3, App. A)**
- Prop. 1: Bayes-optimal logits = log p(y|h,e)/p_N(y|h) + b(h,e).
- Prop. 2: As K → ∞, E[T*|h,e] → KL(p+ || p−), assuming a bounded ratio c ≤ r ≤ C.
- Prop. 3: exponential tilting reweights sources by the moment-generating function of the score.
- The exp weight form is derived as the optimizer of max_q E_q[J] − beta KL(q||p_D) (Eq. 19). "This is a classical result in the policy search literature".
- Source: [Zhang §4.3, App. A](https://arxiv.org/pdf/2603.16542)

**Weight, conservativeness and normalization (Eq. 21-30)**
- w_t = 1 + alpha(clip_[w_min, w_max](exp(T_t/beta)) − 1), with defaults w_min = 0.25, w_max = 4.0, alpha = 1. — [Zhang §4.4, Eq. 21](https://arxiv.org/pdf/2603.16542)
- The self-normalized loss is L* = Σ sg(w_t) ℓ_act / Σ sg(w_t), i.e. **per-minibatch normalization** (global batch 128). — [Zhang §4.4, Eq. 27; Table 6](https://arxiv.org/pdf/2603.16542)
- Density-ratio bound: q/p_D ∈ [w_lo/E[w], w_hi/E[w]], giving KL(q||p_D) ≤ log(w_hi/w_lo) = log 16 ≈ 2.77 nats. "This bound is the formal justification for calling PTR 'conservative'". — [Zhang §4.4, Eq. 22-26](https://arxiv.org/pdf/2603.16542)
- Source proportions are bounded: q(m)/π_m ∈ [1/16, 16]. Setting alpha → 0 recovers SFT. — [Zhang §4.4, Eq. 29-30](https://arxiv.org/pdf/2603.16542)
- On beta: "As beta → ∞ ... equivalent to standard SFT. As beta → 0+, the tilted distribution concentrates on the highest-scoring samples." — [Zhang §4.3](https://arxiv.org/pdf/2603.16542)

**Adaptive scale controller ("refiner", §4.5)**
- It tracks EMAs (decay 0.98) of identification accuracy, margin, mean T and valid-target ratio.
- If accuracy is below 0.05, tau_score and beta are multiplied by 1.01. If accuracy is above 0.35 and margin above 0.10, both are multiplied by 0.995.
- Bounds: tau ∈ [0.03, 0.20], beta ∈ [0.5, 3.0].
- Observed behaviour: tau_score goes 0.12 → ~0.03 (**the lower bound**); hard-negative ratio 0 → ~0.50 (**the upper bound**); beta 1.5 → ~1.2.
- Source: [Zhang §4.5, §5.6](https://arxiv.org/pdf/2603.16542)

**Hyperparameters (Table 6)**

| Group | Settings |
|---|---|
| Optimization | AdamW, lr 1e-4, wd 0.01, cosine schedule with 2000-step warmup, 60k steps, global batch 128 |
| Policy | Chunk L = 16; flow time ~ Beta(1.5, 1.0); 4 denoising steps |
| Scorer | lambda_id = 0.05; NCE warmup 3000 steps; tau init 0.12; beta init 1.5; logit clamp 20.0; w_min 0.25; w_max 4.0; alpha 1 |
| Other losses | lambda_rank = 0.25; belief tokens M = 4, tau_tok 1.0, lambda_ent = lambda_div = 1e-3 |

"All methods share the same backbone, action head, and training budget". — [Zhang App. B, Table 6; §5.1](https://arxiv.org/pdf/2603.16542)

**Evaluation**
- LIBERO: 4 suites × 10 tasks × 50 episodes = 2000.
- RoboCasa: 24 tasks × 50 trials across 5 layouts = 1200.
- Real: 12 tasks × 20 trials = 240 total, on 3 platforms:
  - Unitree G1 + LinkerHand O6: 26 DoF, 20 Hz.
  - PND Adam-U: 31 DoF, 20 Hz.
  - FR3 + Inspire Hand: 13 DoF, 50 Hz.
- Specialist = one checkpoint per source. Generalist = one checkpoint over 5 sources (3 real plus LIBERO plus RoboCasa).
- Source: [Zhang §5.1, §5.4, Tables 2-3](https://arxiv.org/pdf/2603.16542)

**Table 1 (clean sim, success %)**

| Method | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO Long | LIBERO avg | RoboCasa Pick&Place | RoboCasa Door/Drawer | RoboCasa Others | RoboCasa avg |
|---|---|---|---|---|---|---|---|---|---|
| SFT | 98.8 | 99.0 | 98.6 | 96.8 | **98.3** | 36.0 | 71.3 | 55.3 | 54.2 |
| SFT+Belief | 98.2 | 98.4 | 98.0 | 95.6 | 97.6 | 36.7 | 71.5 | 55.0 | 54.4 |
| PTR | 98.0 | 99.2 | 97.6 | 97.0 | 97.8 | 38.3 | 73.0 | 55.5 | **55.6** |

Source: [Zhang Table 1](https://arxiv.org/pdf/2603.16542)

**Figure 2 (corrupted training data, bar labels printed)**

Corruptions:
- ANI (action noise injection): sigma = 0.1 Gaussian noise on 30% of trajectories.
- TT (trajectory truncation): 25% of trajectories truncated to 40-70% length.
- LN (label noise): 20% of instructions reassigned.
- Combined: all three.

| Benchmark | Method | Clean | ANI | TT | LN | Combined |
|---|---|---|---|---|---|---|
| LIBERO | SFT | 98.3 | 93.2 | 91.6 | 94.8 | 85.4 |
| LIBERO | PTR | 97.8 | 95.4 | 94.0 | 96.2 | 91.0 (+5.6) |
| RoboCasa | SFT | 54.2 | 46.8 | 44.2 | 48.6 | 36.4 |
| RoboCasa | PTR | 55.6 | 52.0 | 50.4 | 52.6 | 45.8 (+9.4) |

Sources: [Zhang §5.1, §5.3, Fig. 2](https://arxiv.org/pdf/2603.16542)

**Table 3 / Table 4 (real robot, 20 trials per task)**

Specialist, SFT → PTR by suite:
- Bimanual: 55.0 → 66.7
- Long-Horizon: 63.3 → 61.7 (SFT better)
- Spatial: 75.0 → 78.3
- Robust: 50.0 → 61.7
- 12-task mean: 60.8 → 67.1

Generalist, SFT → PTR:
- Bimanual: 45.0 → 60.0
- Long-Horizon: 51.7 → 65.0
- Spatial: 63.3 → 73.3
- Robust: 40.0 → 56.7
- 12-task mean: 50.0 → 63.8

Per task, SFT-Spec / PTR-Spec / SFT-Gen / PTR-Gen:

| Task (platform) | SFT-Spec | PTR-Spec | SFT-Gen | PTR-Gen |
|---|---|---|---|---|
| Pour Water (Adam-U) | 60 | 70 | 50 | 65 |
| Handover (Adam-U) | 50 | 65 | 40 | 55 |
| Put & Close Box (G1) | 55 | 65 | 45 | 60 |
| Clear Desk (Adam-U) | 70 | 65 | 60 | 70 |
| Drawer Organization (Adam-U) | 60 | 60 | 45 | 65 |
| Scan Package (G1) | 60 | 60 | 50 | 60 |
| Arrange Flower (Adam-U) | 80 | 85 | 70 | 80 |
| Stack Bowl (FR3) | 70 | 75 | 60 | 70 |
| Water Plant (FR3) | 75 | 75 | 60 | 70 |
| Wipe Board (Adam-U) | 55 | 70 | 45 | 60 |
| Drawer Organization (FR3) | 50 | 60 | 40 | 55 |
| Wipe Board (FR3) | 45 | 55 | 35 | 55 |

- "PTR improves over SFT on 8 of 12 tasks in the specialist setting, ties on 3, and trails on 1".
- Table 3's "Overall" (71.1 vs 73.5 specialist) is the mean of LIBERO avg, RoboCasa avg and real avg.
- Sources: [Zhang Tables 3-4, §5.4](https://arxiv.org/pdf/2603.16542)

**Table 5 (ablations, LIBERO avg / RoboCasa avg)**

| Configuration | LIBERO avg | RoboCasa avg |
|---|---|---|
| Full | 97.8 | 55.6 |
| w/o belief tokens | 97.4 | 54.0 |
| w/o cross-rank gather | 97.0 | 54.3 |
| w/o EMA (frozen encoder) | 95.3 | 50.2 |
| w/o refiner | 97.1 | 51.9 |
| **w/o clipping** | **95.5** | **49.1** |
| alpha = 0 | 97.5 | 54.3 |

- "Weight clipping is the most critical component: removing it drops LIBERO by 2.3 pp and RoboCasa by 6.5 pp". — [Zhang Table 5, §5.5](https://arxiv.org/pdf/2603.16542)
- "Without clipping, the weight distribution develops heavy tails early in training: a small fraction of samples receive weights above 10, causing gradient spikes that manifest as periodic loss oscillations." — [Zhang §5.6](https://arxiv.org/pdf/2603.16542)

**Weight diagnostics (Fig. 8-11)**
- T_t stabilizes at 3.2-3.8 in the fast group (Cross-Emb, Adam-U, FR3, G1) and 2.8-3.1 in the slow group (LIBERO, RoboCasa). ℓ_act falls from 0.2-0.4 to 0.005-0.012. — [Zhang §5.6, Fig. 8](https://arxiv.org/pdf/2603.16542)
- Identification margin reaches ~1.9 (LIBERO) to ~7.5 (cross-embodiment). — [Zhang Fig. 9](https://arxiv.org/pdf/2603.16542)
- Weight histograms: at 3k steps weights are "near-uniform around w = 1". At 60k steps, LIBERO "concentrates much of its mass near the clipping bound w_max = 4.0 and retains only a thin suppressed tail near w_min". RoboCasa has "a visibly broader suppressed region". Real-robot generalist has "the widest spread". — [Zhang §5.6, Fig. 11a-c](https://arxiv.org/pdf/2603.16542)
- Relative training-loss reduction (ℓ_SFT − ℓ_PTR)/ℓ_SFT, read from the plot: ~30% LIBERO, ~45-55% RoboCasa, ~35-43% real. — [Zhang Fig. 11d](https://arxiv.org/pdf/2603.16542)
- Sensitivity sweeps on "cross-embodiment LIBERO" (normalized metrics):
  - tau_0 sweet spot ≈ 0.12.
  - beta_0 = 1.5 is best: "very small values amplify score differences into unstable weights, while very large values compress weights toward uniformity". Stability, measured as 1/Std(w), rises monotonically with beta.
  - w_max = 4 is best, and larger values "allow less controlled gradients from extreme weights".
  - Source: [Zhang §5.6, Fig. 10](https://arxiv.org/pdf/2603.16542)
- No ESS is reported. The only weight-dispersion diagnostic is 1/Std(w). — [Zhang Fig. 10](https://arxiv.org/pdf/2603.16542)

**Mixed-quality claim**
- "ambiguous samples stay near uniform weight, while clearly counter-evidential ones are down-weighted ... high-quality demonstrations produce distinctive post-action consequences". This is argued in §4.2 and tested only with the synthetic corruptions above. — [Zhang §4.2](https://arxiv.org/pdf/2603.16542)
- Limitations: PTR "does not directly optimize task success; it is a data-curation mechanism rather than a policy-optimization algorithm". With a weak backbone it "falls back to near-uniform weighting". — [Zhang §6](https://arxiv.org/pdf/2603.16542)

**Internal inconsistency**
- Table 1 lists PTR LIBERO Object/Goal as 99.2/97.6, while Table 5 "PTR (full)" lists 98.8/97.4 (both avg 97.8).
- Table 1 SFT+Belief (97.6 / 54.4) differs from Table 5 "alpha = 0 (SFT+Belief)" (97.5 / 54.3), with different per-suite cells.
- Source: [Zhang Tables 1 and 5](https://arxiv.org/pdf/2603.16542)

### Inferences
- **The score is not a quality or success signal in RL terms.** T_t measures how predictable o_{t+Δ} is given (context, action), relative to other futures. A failed grasp, a drop or a slip produces a distinctive, action-caused future, so it could easily score high. PTR's "suboptimal ⇒ less distinctive" premise was tested only on Gaussian action noise, truncation and instruction swaps, never on real failure rollouts. For the reader's data (teleop plus autonomous failures), PTR cannot replace the critic's advantage. At most it is an orthogonal "attributability" multiplier. The paper's own limitation says as much.
- **Weights largely saturate at the upper clip.** With late beta ≈ 1.2 and mean T ≈ 2.8-3.8, exp(T/beta) ≈ e^2.3-e^3.2 ≈ 10-24. That is well above w_max = 4, consistent with Fig. 11a's mass at 4. After self-normalization, most samples are therefore near-equal and a minority tail is suppressed by up to 16×. In practice PTR behaves like "soft down-weighting of a low-score tail", not fine-grained ranking. The effective lever is w_max / w_min, not beta.
- **Comparison with the reader's weights:**
  - The reader's current weights span e^-3 to e^3 (ratio ~403). The KL bound analogue is log 403 ≈ 6.0 nats, vs PTR's 2.77.
  - The unit-mean max weight is ~20 before mean-normalization. Zhang's only AWR-like failure evidence (no clipping → weights > 10 → gradient spikes, −2.3 / −6.5 pp) sits in that regime.
  - Zhang's weights are clipped **after** exp. The reader clips the z-scored Â **before** exp, which is equivalent to clipping w to [e^-3/beta, e^3/beta] before mean-normalization. The difference from PTR is the width of the range, not the mechanism.
  - Zhang also normalizes per minibatch of 128 via Σ w, the same pooling scale as the reader's 128-row effective batch.
- **Effect sizes vs noise:**
  - Clean-data gains (+1.4 pp RoboCasa, −0.5 pp LIBERO) are comparable to the ~0.4-0.8 pp per-suite discrepancies between Tables 1 and 5.
  - Real-robot per-task differences are 5-15 pp at n = 20, i.e. 1-3 trials.
  - My unpaired two-proportion tests on the 240-trial totals: specialist 146 vs 161 successes, z ≈ 1.4, p ≈ 0.15 (not significant); generalist 120 vs 153, z ≈ 3.0.
  - The strongest evidence is the generalist cross-embodiment gap and the synthetic-corruption results, not the specialist gains.
- **The ablation isolating reweighting is small.** "alpha = 0 (SFT+Belief)" vs PTR is only 0.3 / 1.3 pp. The large ablation drops (w/o clipping, w/o EMA, w/o refiner) are cases of reweighting done badly and hurting, below SFT (97.5 / 54.3), rather than evidence that reweighting helps.

### Gaps
- Numeric Δ (post-action offset) per dataset is not given. Neither is the γ^{Δ'} discount value.
- Real-robot training set size, collection method and quality mix per task are not described.
- Seeds and variance are not reported for any table.
- Not tested:
  - Any reward or value-based baseline (AWR, filtered BC, IQL, CQL).
  - Real failure data.
  - A score/critic-error sensitivity analysis beyond the tau / beta / w_max sweeps.
  - ESS.
  - Whether PTR changes policy speed or length.

## Q4. AWR failure modes reported or argued in both papers, and the diagnostics they use

### Takeaway
Neither paper runs flat exponential AWR, so neither measures its failure modes directly. The closest evidence for each failure mode:

| Failure mode | Evidence |
|---|---|
| Heavy-tailed weights / gradient spikes | Zhang's no-clipping ablation: weights > 10, periodic loss oscillation, −2.3 / −6.5 pp |
| Over-aggressive selection / data discard | Fang's hard-threshold Imit-Recap keeps only 19-27% of rollout transitions and underperforms soft weighting |
| Temperature sensitivity | Zhang's beta sweep |
| Credit leakage across human takeovers | Fang's argument |

Diagnostics are thin. Fang reports mean g_t and the filter pass rate. Zhang reports weight histograms, 1/Std(w), mean score, identification accuracy and margin. Neither reports ESS.

### Cited Findings
- Weight explosion without bounds: "a small fraction of samples receive weights above 10, causing gradient spikes that manifest as periodic loss oscillations". The ablation costs −2.3 pp LIBERO and −6.5 pp RoboCasa, ending below SFT. — [Zhang §5.5-5.6, Table 5](https://arxiv.org/pdf/2603.16542)
- Temperature: "very small [beta] values amplify score differences into unstable weights, while very large values compress weights toward uniformity and effectively disable reweighting". — [Zhang §5.6, Fig. 10b](https://arxiv.org/pdf/2603.16542)
- Clip bound: "larger values allow less controlled gradients from extreme weights". — [Zhang §5.6, Fig. 10c](https://arxiv.org/pdf/2603.16542)
- Drift toward BC is described as a feature: beta → ∞ or alpha → 0 recovers SFT, and when "demonstrations are already consistent, PTR weights stay close to uniform and the method reduces to standard post-training". — [Zhang §1, §4.3-4.4](https://arxiv.org/pdf/2603.16542)
- Stale target / moving score: a frozen target encoder costs −2.5 / −5.4 pp. Without EMA, "identification accuracy fluctuates rather than converging monotonically, and the resulting weight signal is noisy". — [Zhang §5.5-5.6](https://arxiv.org/pdf/2603.16542)
- Hard filtering throws away data: Imit-Recap "passes approximately 22, 27, and 19 out of every 100 sampled autonomous rollout transitions ... This filtering behavior is more aggressive than HABC's soft weighting". In Fig. 3, Imit-Recap < HABC-V < HABC on success on all tasks, and Imit-Recap has the longest successful trajectories on Pencil Pouch (~1970 frames, read from bar). — [Fang App. G, Fig. 3](https://arxiv.org/pdf/2606.17043)
- Single scalar has no efficiency gradient after success saturates: "once basic success is achieved, the binary label provides no gradient to distinguish efficient completions from slow ones". — [Fang abstract](https://arxiv.org/abs/2606.17043)
- Credit leakage across control authority: naive outcome crediting "would upweight the pre-intervention policy execution segment that led to the near-failure state ... if the episode fails after an intervention, it would incorrectly penalize the human's corrective actions". — [Fang §3.4](https://arxiv.org/pdf/2606.17043)
- Fang's diagnostics:
  - Mean pre-normalization g_t per task (0.76 / 0.68 / 0.86).
  - Imit-Recap pass rate.
  - Mean g_t on intervention windows (1.0 / 1.1 / 0.95).
  - Maps of p_v > 0.6 over initial states across training.
  - Per-rollout traces of p_v and V_e.
  - Trajectory length on successes.
  - Source: [Fang §4.2-4.3, App. G](https://arxiv.org/pdf/2606.17043)
- Zhang's diagnostics:
  - Weight histograms at 3k vs 60k steps.
  - 1/Std(w) "stability".
  - Mean T_t, identification accuracy and margin.
  - Belief-token entropy.
  - Relative loss reduction vs SFT.
  - Controller parameter trajectories.
  - Source: [Zhang §4.5, §5.6, Fig. 8-11](https://arxiv.org/pdf/2603.16542)

### Inferences
- Neither paper provides evidence for or against "ESS collapse" as such. Zhang's no-clip result is the closest proxy: a few samples above 10× dominating the gradient. The reader's configuration (z-clip ±3, beta = 1) allows up to e^3 ≈ 20× before mean-normalization, so it is in the regime Zhang reports as harmful.
- Both papers converge on **bounded** weights: Fang uses [0, 2] via tanh, and Zhang uses [0.25, 4] via clip. Both keep a uniform-weight anchor. Fang keeps SFT and intervention data at weight 1; Zhang uses alpha-mixing and a fallback w = 1. Neither uses unbounded or wide-range exp weights. That is design evidence, not a head-to-head AWR comparison.
- Neither measures critic-error amplification directly. Fang's warmup (g = 1 for 500 steps) and delayed IR are implicit guards against an uncalibrated critic. Zhang's controller keeps beta high while identification accuracy is below 0.05. Both are heuristics with no ablation, except Zhang's "w/o refiner" row (−0.7 / −3.7 pp).
- On "slower policies": Fang shows that all online variants produce longer successful episodes than SFT, and that its efficiency head only partially offsets this. Zhang reports no speed metric.

### Gaps
- Neither paper reports ESS, the fraction of batch weight carried by the top-k samples, or weight-vs-advantage calibration.
- Neither injects critic or score noise to measure sensitivity.
- No flat-AWR run exists in either paper to test drift toward BC or over-weighting.

## Q5. Critique: evaluation size, fairness of baselines, transfer from online to offline, what is not tested

### Takeaway
Both papers use small real-robot evaluations: Fang 50 trials per checkpoint on 3 tasks with best-checkpoint selection, and Zhang 20 trials per task. Neither includes a tuned flat-AWR baseline, so neither can settle "AWR vs X". Fang's strongest untested confound is that failed rollouts are simply dropped. Zhang's is that its score is reward-free and was never tested on real failure data. Fang's per-round update is offline weighted BC and mostly transfers to the reader's setting; Zhang's conservative weight transform transfers directly, but its score does not address success vs failure.

### Cited Findings
- Fang evaluation: 3 tasks, "Each checkpoint is evaluated over 50 trials", Wilson 95% CIs plotted, single run per method. The headline 92/88/38% come from "the best checkpoint reached by continuing HABC+IR training for more online rounds". — [Fang §4.1, Fig. 3](https://arxiv.org/pdf/2606.17043)
- Fang baseline tuning:
  - Imit-Recap's eps is "selected on a held-out validation set"; the value and selection metric are not given.
  - Imit-Recap is "not a full reproduction of Recap's advantage-conditioned actor".
  - All Step-1 methods get "an equal online fine-tuning budget" from the same SFT checkpoint.
  - Source: [Fang §4.1, App. B.2](https://arxiv.org/pdf/2606.17043)
- Fang scope: "HABC is currently evaluated on single-task fine-tuning". — [Fang §5](https://arxiv.org/pdf/2606.17043)
- Zhang evaluation: 20 trials per real task, 60 per suite, 240 total; sim 2000 LIBERO and 1200 RoboCasa episodes; no seeds reported. — [Zhang §5.1, §5.4](https://arxiv.org/pdf/2603.16542)
- Zhang baselines: only SFT and SFT+Belief, all "share the same backbone, action head, and training budget". Corruptions are synthetic. — [Zhang §5.1](https://arxiv.org/pdf/2603.16542)
- Zhang tunes its own knobs (tau, beta, w_max) by sweeps on cross-embodiment LIBERO and uses adaptive control. The SFT baseline has no weighting knobs to tune. — [Zhang §5.6, Fig. 10](https://arxiv.org/pdf/2603.16542)

### Inferences
- **Evaluation size:**
  - Fang, n = 50 per cell: 95% CI half-widths are ~11-14 pp. Only the SFT → HABC gains on Pencil Pouch and Paper Bag are clearly outside noise. HABC-V → HABC and all Snack Bag comparisons are not.
  - Zhang, n = 20 per task: half-widths ~20 pp. Only aggregates (240 trials) and the generalist gap are informative.
  - Best-checkpoint reporting in Fang inflates the 92/88/38 numbers by an unknown amount.
- **Fairness vs AWR:**
  - Neither paper tunes, or even runs, exp(A/beta) AWR. Fang's only critic baseline is a hard filter with its own eps tuning, a different critic (the parent distributional head with a C = 100 failure penalty, initialized from an IQL checkpoint), and no intervention-aware labelling. It differs from HABC in at least three ways at once.
  - Zhang's reweighting was tuned with sweeps plus an adaptive controller, while the comparison is untuned uniform SFT. The only "alpha = 0" control isolates reweighting at +0.3 / +1.3 pp.
- **Online (Fang) → offline (reader):**
  - Transfers directly: the per-round learner (flow-matching BC weighted by one-step critic advantages, bounded weights, unit-mean normalization, SFT at weight 1) and intervention-boundary labelling.
  - Does not transfer:
    - (a) V_v as "success under the current policy" refreshed every round with a stale cutoff.
    - (b) The iterative loop, where improved rollouts feed the next round. Most of the gain from 60 → 92% on Pencil Pouch came in Step 2's extra rounds (~68-76% at the Step-2 checkpoints, read from bars).
    - (c) Fresh on-policy failures that the critic can learn from.
  - Offline, one "round" corresponds roughly to Fang's Step 1 after the first 100-rollout batch. Fang does not report single-round results.
- **For the reader's specific question (AWR-style weighting on mixed teleop plus failure rollouts):**
  - Fang supports: (1) dropping or zeroing failed-episode policy segments while keeping demos and interventions at weight 1; (2) bounded weights on the rest; (3) never crediting pre-takeover segments with the episode outcome.
  - Zhang supports bounding the weight ratio (≤ 16×) and per-batch self-normalization, and shows that unclipped heavy tails hurt.
  - Neither shows that exp-advantage weighting beats success-filtered BC, and neither tests a z-scored, per-batch-pooled advantage like the reader's.
- **Not tested in either paper:**
  - Flat exp-AWR at any beta.
  - Success-only BC with no critic.
  - Critic-noise robustness.
  - ESS.
  - Multi-task training (Fang is single-task; Zhang does multi-source but has no RL signal).
  - Longer-horizon TD or multi-step advantages (Fang lists this as future work).
  - Behaviour on autonomous failure data without human takeover in an offline-only regime.
  - Effect on execution speed vs BC (Fang shows all online variants slower than SFT on successful trials).

### Gaps
- Without Fang's code or logs, I cannot check whether zero-weight failures enter the Eq. 8 normalizer, or what the actual per-batch composition was. Both determine the effective strength of the weighting.
- Neither paper reports the number of training seeds or run-to-run variance. Zhang's Table 1 vs Table 5 discrepancies suggest per-suite noise of ~0.4-0.8 pp even in simulation.
