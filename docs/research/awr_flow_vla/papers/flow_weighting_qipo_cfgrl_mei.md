# Weighting vs. guiding flow-matching policies by value/advantage: QIPO (arXiv 2503.04975), CFGRL (arXiv 2505.23458), Mei et al. / ForesightFlow (arXiv 2606.04968)

Reading basis: QIPO read from the arXiv PDF v1 (ICLR 2025 camera-ready header), pp. 3-12, 14-16, 18-28 (main text, Algs 1-5, Tables 1-5, App. A, B.1-B.3, D, E, F). CFGRL read from the arXiv PDF v1, pp. 3-12, 15-20 (Secs 3-7, Tables 1-8, Figs 1-7, Theorems 1-2); its intro/related-work pages (pp. 1-2) were only seen through a summarising fetch. Mei et al. read in full from the arXiv HTML v1 (main text, App. A-E); the PDF was over 10 MB and could not be fetched. Notation: "sim" = simulation, "real" = physical robot.

## Q1. QIPO: what energy-weighted flow matching is, whether it equals the per-sample exp-weighted CFM loss, what exactness needs, hyperparameters, results, ablations

### Takeaway
Energy-weighted flow matching (EFM) is exactly the per-sample weighted conditional flow-matching loss. Each sample gets the weight exp(-βE(x0)) (for RL, exp(βQ(s,a))), which depends only on the clean data point, is the same at every flow time, and is divided by a constant normaliser. Theorem 4.3 proves its gradient equals that of a loss whose minimiser is the exact guided flow, with no flow-time-dependent weight and no learned intermediate energy.

For conditional policies the normaliser is per state, E_{ã~μ(·|s)}[exp(βQ(s,ã))] (Eq. 5.3). QIPO estimates it with a softmax over M=16 self-generated support actions plus the dataset action, and trains on those generated actions as well.

On D4RL, QIPO's averages are roughly on par with QGPO and IDQL, not clearly better (Table 2). There is no plain BC baseline, and no robot or pixel-based RL.

### Cited Findings
- **Verification.** arXiv 2503.04975 is "Energy-Weighted Flow Matching for Offline Reinforcement Learning" by Shiyuan Zhang, Weitong Zhang and Quanquan Gu. v1 was submitted 6 Mar 2025, with the comment "28 pages, 11 figures, accepted by ICLR 2025". This matches the description. QIPO stands for "Q-weighted Iterative Policy Optimization" and is the RL instantiation of EFM. — [arXiv abs](https://arxiv.org/abs/2503.04975)
- **Target.** q(x) ∝ p(x)exp(-βE(x)), with β the "strength of the guidance" (Sec 3.2). Notation: t=0 is data and t=1 is noise (footnote 1), so x0 is the clean sample. — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Background from Lu et al. 2023 (QGPO/CEP), Theorem 3.3.**
  - The intermediate energy is E_t(x) = -log E_{p_0t(x0|x)}[exp(-βE(x0))] (Eq. 3.5).
  - It gives q_t ∝ p_t exp(-E_t) and ∇log q_t = ∇log p_t - ∇E_t (Eq. 3.6).
  - CEP must learn E_t with an auxiliary network and backpropagate through it (Sec 3.2; Remark 4.9).
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Theorem 4.1 (guided flow).** The guided velocity is û_t(x) = ∫ p_0t(x0|x) u_t0(x|x0) · exp(-βE(x0))/exp(-E_t(x)) dx0 (Eq. 4.1). It generates q_t ∝ p_t exp(-E_t), with q_0 ∝ p_0 exp(-βE). So the exact guided velocity at (x_t, t) is the posterior average of conditional velocities, reweighted by exp(-βE(x0)) and normalised per x_t by exp(-E_t(x)). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Theorem 4.3 (the key result).**
  - Energy-weighted FM loss: L_EFM = E_{t,x}[ exp(-E_t(x))/E_{p_t(x̃)}[exp(-E_t(x̃))] · ||v_θ(x) - û_t(x)||² ] (Eq. 4.2).
  - Conditional energy-weighted FM loss: L_CEFM = E_{t,x,x0}[ exp(-βE(x0))/E_{p0(x̃0)}[exp(-βE(x̃0))] · ||v_θ(x) - u_t0(x|x0)||² ] (Eq. 4.3).
  - Verbatim: "L_EFM(θ) and L_CEFM(θ) are equal up to a constant factor. Hence ∇θ L_EFM(θ) = ∇θ L_CEFM(θ)."
  - Verbatim: "the global minimum of L_CEFM is v_θ = û_t(x), given enough network capacity and infinite data."
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Why no per-x_t normaliser or time-dependent weight is needed (proof B.2, Lemma B.1).**
  - The normaliser E_{x0}[exp(-βE(x0))] "is a constant with respect to x and t" (Lemma B.1).
  - E_{p_t}[exp(-E_t)] = E_{p0}[exp(-βE)] (Eq. B.9).
  - The per-x_t factor exp(-E_t(x)) is absorbed through Bayes' rule (Eqs. B.7-B.10).
  - The only conditions are the weight's dependence on x0 alone, the global constant normaliser, and the global-minimum / infinite-capacity caveat.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Remark 4.4.** The equivalent marginal loss "places higher weight on the input x with a lower intermediate energy". Weighting therefore shifts where capacity goes across x_t; the minimiser is unchanged. — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Remark 4.6 (importance sampling).** L_CEFM equals the plain CFM loss under q_0, reweighted by q0/p0 = exp(-βE)/E[exp(-βE)]. — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Batch approximation (Algorithm 1 line 3; Algorithm 3 line 4).**
  - Unconditional case: g_i = softmax(-βE(x0^i)) over the batch.
  - The denominator "can be approximated by the empirical average Σ exp(-βE(x0^i))/N in a batch" (Sec 4.2, text after Alg. 1).
  - The loss is Σ_i λ(t_i) g_i ||·||², with one flow time per sample.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **RL form (Sec 5).**
  - Objective: argmax E_{a~π}Q(x,a) - (1/β)KL(π||μ) (Eq. 5.1), whose closed form is π ∝ μ exp(βQ). Here β is an inverse temperature: QIPO β = 1/(AWR temperature).
  - Q-weighted flow loss: L_QF = E[ exp(βQ(x,a))/E_{ã~μ(·|x)}[exp(βQ(x,ã))] · ||v_θ(a_t;x) - u_t0(a_t|a)||² ] (Eq. 5.3). The normaliser is per state x.
  - "any Q function derived from offline RL algorithms can be used" (Sec 5).
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **QIPO algorithm (Algorithms 2, 4 and 5).**
  - K1 epochs of plain BC warm-up, then K2 epochs of Q warm-up.
  - K3 policy-improvement epochs: every K_renew steps, sample M support actions a_ij from the current policy for each state and set a_i0 = the dataset action.
  - Weights: g_ij = softmax_j(βQ(x_i,a_ij)) over j=0..M, so the weights sum to 1 within each state.
  - Loss: Σ_{i,j} λ(t_ij) g_ij ||v_θ(a_ij,t_ij; x_i) - u(a_ij,t_ij|a_ij)||² (Alg. 4 lines 17-21).
  - The policy therefore also trains on its own generated actions, scored by Q(s,a).
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Iterative sharpening (Eq. 5.4).** π_{l+1} ∝ π_l exp(βQ) ∝ … ∝ μ exp((l+1)βQ), "which will implicitly increase the guidance scale β". — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Q estimator (App. D).** "We use the in-support softmax Q-learning (Lu et al., 2023)". The target is a softmax(βQ⊥)-weighted average of Q⊥ over M support actions at the next state, with Q⊥ a stop-gradient Q (Alg. 4 line 11). It is state-action Q, not advantage and not a V-only critic. — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Hyperparameters (App. E.1, E.3).**
  - Same networks as QGPO; fine-tuning LR 1e-4; DPM-Solver with 15 steps; QIPO-OT σ_min = 0.0054.
  - K3 = 100, evaluated every 5 epochs; K_renew = 10; M = 16; soft update λ = 0.005.
  - Default β = 3 (inferred from E.3: "changing the guidance scale from β=3 to β=10").
  - No weight clipping is mentioned.
  - Reported numbers are "the best performance of the reported performance over K3 = 100 episodes" (E.1), i.e. the best evaluation during training.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Table 2 (D4RL, 8 seeds, mean ± std, normalised score).** Columns: SfBC / QGPO / IDQL / SRPO / Guided Flows / QIPO-Diff / QIPO-OT.

  | Task | SfBC | QGPO | IDQL | SRPO | Guided Flows | QIPO-Diff | QIPO-OT |
  |---|---|---|---|---|---|---|---|
  | ME-HalfCheetah | 92.6 | 93.5 | 95.9 | 92.2 | 97 | 94.14±0.48 | 94.45±0.49 |
  | ME-Hopper | 108.6 | 108.0 | 108.6 | 100.1 | 105 | 112.12±0.42 | 108.02±5.19 |
  | ME-Walker2d | 109.8 | 110.7 | 112.7 | 114.0 | 94 | 110.14±0.51 | 110.87±1.04 |
  | M-HalfCheetah | 45.9 | 54.1 | 51.0 | 60.4 | 49 | 48.19±0.20 | 54.16±1.27 |
  | M-Hopper | 57.1 | 98.0 | 65.4 | 95.5 | 81 | 89.53±9.96 | 94.05±13.27 |
  | M-Walker2d | 77.9 | 86.0 | 82.5 | 84.4 | 77 | 84.99±0.46 | 87.61±1.46 |
  | MR-HalfCheetah | 37.1 | 47.6 | 45.9 | 51.4 | 42 | 45.27±0.42 | 48.04±0.79 |
  | MR-Hopper | 86.2 | 96.9 | 92.1 | 101.2 | 89 | 101.23±0.47 | 101.25±2.18 |
  | MR-Walker2d | 65.1 | 84.4 | 85.1 | 84.6 | 78 | 90.08±4.53 | 78.57±26.09 |
  | **Locomotion average** | 75.6 | 86.6 | 82.1 | 87.1 | 79.4 | 86.2 | 86.3 |
  | umaze | 92.0 | 96.4 | 94.0 | 97.1 | - | 97.5±0.53 | 93.62±7.05 |
  | umaze-diverse | 85.3 | 74.4 | 80.2 | 82.1 | - | 73.88±6.42 | 76.12±9.93 |
  | medium-play | 81.3 | 83.6 | 84.5 | 80.7 | - | 82.75±3.24 | 80.00±13.66 |
  | medium-diverse | 82.0 | 83.8 | 84.8 | 75.0 | - | 86.00±8.65 | 86.42±5.44 |
  | large-play | 59.3 | 66.6 | 63.5 | 53.6 | - | 73.25±10.90 | 55.5±29.39 |
  | large-diverse | 45.5 | 64.8 | 67.9 | 53.6 | - | 40.5±20.40 | 32.13±23.16 |
  | **AntMaze average** | 74.2 | 78.3 | 79.1 | 73.6 | - | 77.3 | 71.96 |

  The text nonetheless states that QIPO "consistently outperform[s] the baselines" (Sec 5.2). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Table 4 (App. E, extra baselines).** Columns: BEAR / TD3+BC / IQL / Diffuser / Diffusion-QL / QIPO-Diff / QIPO-Diff with β=10 (3 seeds).
  - Locomotion averages: 51.9 / 75.3 / 76.9 / 75.3 / 88.0 / 86.20 / 86.80.
  - AntMaze averages: 23.7 / 27.3 / 63.0 / - / 69.6 / 77.3 / 80.78.
  - IQL per task, locomotion (ME, M, MR for HalfCheetah/Hopper/Walker2d): 86.7, 91.5, 109.6; 47.4, 66.3, 78.3; 44.2, 94.7, 73.9.
  - IQL per task, AntMaze: 87.5, 62.2, 71.2, 70.0, 39.6, 47.5.
  - No plain BC column appears in Table 2 or Table 4.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Ablations (App. E.3).**
  - β = 3 → 10 (3 seeds, Table 4 last column). The method "demonstrates robustness to changes in the guidance scale", and the β=10 average "sometimes surpasses the original setting with β=3". Per task at β=10: AntMaze medium-play 92.67 vs 82.75, large-diverse 59.00 vs 40.5, ME-Hopper 103.56 vs 112.12.
  - M ∈ {16, 32, 64} (Fig. 5, 3 runs): a larger M "may lead to slightly improved performance", but M=16 was kept for cost.
  - K_renew ∈ {5, 10, 20, 50} (Fig. 6, 3 runs): "smaller K values generally lead to improved performance".
  - No ablation replaces the weighting scheme (for example, a batch-global normaliser versus the per-state softmax, or dataset-only versus support actions).
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Guidance versus weighting (Sec 4.3, Lemma 4.10, Table 1, Figs 1 and 3).**
  - CG and CFG compose ∇log p_t + ∇log[E_{p0t(x0|x)} p(c|x0)]^β (Eq. 4.6).
  - Energy weighting and CEP give ∇log p_t + ∇log E[p^β(c|x0)] (Eq. 4.7).
  - Eq. 4.6 "is not guaranteed to generate the desired distribution q0 when β ≠ 1". On 2-D toys, CFG at β=10 produces distorted samples while energy-weighted diffusion matches the ground truth (Figs 1, 3).
  - At β=1 the two coincide (Lemma 4.11).
  - Table 1 marks CFG as not exact but auxiliary-free, CEP as exact but auxiliary, and energy-weighted as exact and auxiliary-free.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Criticism of IQL-style weights exp((Q−V)/τ) (App. A).** It targets EDP (Kang et al. 2024): "f_IQL requires an additional value network for V_ψ(s)". It also says EDP and QVPO "are designed to maximize the Q function … [and] cannot align with the original distribution μ". — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Compute (Table 3, 1500 runs).** Per-action time: QGPO 75.05 ms, QIPO-OT 27.26 ms (−63.68%), QIPO-Diff 55.86 ms (−25.57%). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Other domains (App. F).**
  - Image synthesis on smithsonian-butterflies with β=50 is qualitative only (Fig. 9).
  - Molecule generation (Table 5, 1,000 molecules): normalised α rises −0.1977 / 0.6196 / 2.2965 / 2.2956 at β = 0 / 1 / 3 / 10.
  - There are no robot experiments and no pixel-based RL; all RL is state-based D4RL.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)

### Inferences
- Multiplying each sample's flow-matching loss by exp(A/β) is the same object QIPO calls "exact energy-weighted flow matching", not an approximation of it. The requirements are:
  - The weight is a function of the clean (state, action) sample only and is shared across all flow times t.
  - The normaliser does not depend on x_t or t.
  - A per-state constant is also harmless, since it cannot change the per-state minimiser.
- Exactness holds only at the global minimum; batch-level normalisation is a stochastic estimate of the constant (Alg. 1).
- Because exp((Q−V)/τ) = exp(Q/τ)·exp(−V/τ), and the second factor is a per-state constant, AWR's advantage weight targets the same π ∝ μ exp(Q/τ). This is inferred from Eq. 5.3 plus algebra; QIPO's App. A objection to f_IQL concerns the extra V network, not the target.
- A one-step TD advantage r + γV(s') − V(s) is a single-sample estimate of Q − V. So in expectation the reader's weight is QIPO's weight with a noisy Q; the noise itself is not analysed by QIPO (see Q3 on Mei's Remark A.6).
- What QIPO adds beyond AWR requires a state-action Q(s,a) that can score actions not in the dataset: per-state softmax over generated support actions, training on those actions, and iterative β compounding (Eq. 5.4). A V(s)-only critic cannot provide this.
- The reported averages come from the best evaluation during training (E.1) and do not beat QGPO or IDQL on average. The paper therefore supports "exact and auxiliary-free" more strongly than "better policy".

### Gaps
- QIPO reports no effective sample size, weight histogram, or weight-concentration diagnostic.
- No experiment compares QIPO against its own dataset-only per-sample weighting (AWR-like), or per-state versus batch-global normalisation.
- It is not stated whether baseline numbers were rerun or copied from prior papers.
- The per-epoch β compounding in Eq. 5.4 (l defined as (k−1) mod K_renew) is ambiguous. The effective β reached at the end of training is not stated.

## Q2. CFGRL: optimality conditioning, the guidance scale versus AWR's β, numbers versus AWR and BC, AWR's structural limitation, robot results

### Takeaway
CFGRL removes the weight entirely. It labels each dataset (s,a) with o = 1[A(s,a) ≥ 0], using an IQL advantage, and trains one flow network on o ∈ {∅, 0, 1} with 10% label dropout. At test time it samples with guidance v = (1−w)v(∅) + w·v(o=1).

- Against AWR (same IQL critic, both tuned per task): CFGRL is higher on 8 of 9 ExORL tasks. On OGBench it is clearly higher on 3 of 9, clearly lower on 1 (antmaze-large 20 vs 50), and roughly tied on the rest.
- Against BC/GCBC (goal-conditioned, no value function): w=3 is better than flow GCBC (w=1) on most of 24 tasks.
- No single table contains BC, AWR and CFGRL, and there are no robot results.

### Cited Findings
- **Verification.** arXiv 2505.23458 is "Diffusion Guidance Is a Controllable Policy Improvement Operator" by Kevin Frans, Seohong Park, Pieter Abbeel and Sergey Levine. v1 was submitted 29 May 2025; arXiv lists no venue. This matches the description. — [arXiv abs](https://arxiv.org/abs/2505.23458)
- **Product policy and improvement guarantees.**
  - π(a|s) ∝ π̂(a|s)·f(A(s,a)) (Eq. 5).
  - Remark 1 / Theorem 1: if f is non-negative and monotonically non-decreasing (and bounded), then J(π) ≥ J(π̂).
  - Remark 2 / Theorem 2: for 0 ≤ w1 < w2, π_{w2} ∝ π̂·f^{w2} improves on π_{w1} (App. A, Eqs. 36-43).
  - "Of course, there is no free lunch … the empirical performance of an over-adjusted policy may suffer due to a distribution shift" (Sec 4).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Link to AWR's β (Remark 3, Eq. 6).** KL-regularised maximisation yields π ∝ π̂·exp(A)^{1/β}.
  - Verbatim: "in their methods the β hyperparameter must be specified ahead of time".
  - Verbatim: "while the absolute scale of w and (1/β) can vary, they have a proportional relationship and share the same base case at w = (1/β) = 0" (Sec 5).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Guidance formula (Eqs. 7-12).**
  - p(o|s,a) = f(A(s,a))/Z(s), and "We will not require estimating Z(s)" (Eq. 7).
  - Bayes inversion gives ∇log π = ∇log π̂(a|s) + (∇log π̂(a|s,o) − ∇log π̂(a|s)) (Eq. 10).
  - Guided score: ∇log π̂(a|s) + w(∇log π̂(a|s,o) − ∇log π̂(a|s)) (Eq. 11), stated to correspond to π ∝ π̂(a|s)·p(o|s,a)^w (Eq. 12).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Training and sampling (Algorithms 1-2, Eq. 13).**
  - Label o ∈ {0,1}; "If rand() < 0.1, set optimality o = ∅".
  - Loss: ||v_θ(a_t, t, s, o) − (a − a0)||² with a_t = (1−t)a0 + t·a and t ~ U[0,1].
  - Sampling: v = (1−w)v_θ(a,t,s,∅) + w·v_θ(a,t,s,o=1) over N Euler steps, i.e. two network evaluations per step.
  - A single shared network with a learnable optimality embedding (App. B, Fig. 7).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Optimality label in offline RL (Eq. 15).** o = 1 if A(s,a) ≥ 0 and 0 otherwise, "Equivalently, we are assigning f = 1(A ≥ 0)". Verbatim: "there is no weighting term used in training". — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Critic.** "In all experiments, we use the same state-action value function trained via implicit Q-learning for both methods" (Sec 5), with A = Q − V. IQL expectile is 0.9 (Tables 4, 6). — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **AWR's structural limitation (Sec 5, Fig. 2).** AWR is J_AWR = E[log π_θ(a|s)·exp(A/β)] (Eq. 14).
  - Verbatim: "the weights in AWR can become peaked, such that much of the data in training is not used effectively, resulting in a weak learning signal … the magnitudes are dominated by a few outlier state-action pairs … the rest of the batch is effectively ignored".
  - Fig. 2: per-element AWR losses reach about 1000 with "Most samples are ignored"; CFGRL per-element losses are about 0-2.5 and even.
  - The second limitation is that AWR's temperature is "folded into the optimization objective", so each β needs retraining, whereas w "can be adjusted without retraining" (Sec 5).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Figure 3 (walker-run, cheetah-run).** Verbatim: "For AWR, performance saturates around a temperature of (1/β)=10. In contrast, guidance continues to improve beyond this point".
  - Values read off the plot (x = 1, 3, 5, 10, 30):
    - walker-run, AWR: about 115, 160, 230, 245, 225.
    - walker-run, CFGRL: about 140, 220, 255, 280, 280.
    - cheetah-run, AWR: about 125, 150, 150, 160, 170.
    - cheetah-run, CFGRL: about 135, 190, 205, 210, 210.
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Table 1, ExORL (RND datasets, 4 seeds), AWR vs CFGRL.**

  | Task | AWR | CFGRL |
  |---|---|---|
  | walker-stand | 603±8 | 782±8 |
  | walker-walk | 444±4 | 608±32 |
  | walker-run | 247±10 | 282±6 |
  | quadruped-walk | 776±15 | 762±25 |
  | quadruped-run | 485±7 | 571±25 |
  | cheetah-run | 168±7 | 216±15 |
  | cheetah-run-backward | 146±8 | 262±26 |
  | jaco-reach-top-right | 33±2 | 72±6 |
  | jaco-reach-top-left | 30±8 | 46±6 |

  [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Table 2, OGBench single-task (4 seeds), AWR vs CFGRL.**

  | Task | AWR | CFGRL |
  |---|---|---|
  | pointmaze-large-navigate | 70±25 | 100±0 |
  | pointmaze-teleport-navigate | 3±7 | 57±7 |
  | antmaze-large-navigate | 50±9 | 20±9 |
  | antmaze-teleport-navigate | 22±19 | 30±22 |
  | humanoidmaze-large-navigate | 3±4 | 0±0 |
  | antsoccer-arena-navigate | 7±0 | 20±5 |
  | cube-single-play | 85±8 | 82±3 |
  | scene-play | 18±3 | 17±9 |
  | puzzle-3x3-play | 3±7 | 3±4 |

  [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Tuning (Sec 5; Tables 5 and 7).**
  - Stated sweeps: 1/β ∈ {1, 3, 10, 30} for AWR and w ∈ {1, 1.25, 1.5, 2.0, 3.0} for CFGRL (Sec 5).
  - ExORL chosen values (Table 5):
    - AWR 1/β: 3, 3, 10, 3, 3, 30, 3, 3, 3.
    - CFGRL w: 30, 30, 30, 3, 10, 10, 30, 3, 3.
  - OGBench chosen values (Table 7):
    - AWR 1/β: 10, 1, 10, 10, 3, 10, 1, 3, 1.
    - CFGRL w: 1, 1, 1.25, 3, 1, 1.5, 1.5, 3, 3.
  - The ExORL w values of 10 and 30 are outside the w sweep stated in Sec 5, an internal inconsistency.
  - The largest OGBench gains (pointmaze-large 100 vs 70, pointmaze-teleport 57 vs 3) used w=1, i.e. plain optimality conditioning with no extrapolation.
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Other settings (Tables 4 and 6).**
  - Networks are MLPs: [512]×3 with Mish on ExORL, [512]×4 with GELU on OGBench.
  - Batch 1024 / 256; 1M / 500k steps; 32 / 16 flow steps; LR 3e-4.
  - Each experiment ran in at most 4 h on one A5000 (App. C).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Table 3, goal-conditioned BC (no value function; 8 seeds state-based, 4 seeds pixel-based; CFGRL w = 3, Table 8).** Order is BC / Flow BC / GCBC / Flow GCBC (= CFGRL w=1) / CFGRL.

  | Task | BC | Flow BC | GCBC | Flow GCBC | CFGRL |
  |---|---|---|---|---|---|
  | pointmaze-medium | 0±0 | 1±2 | 9±5 | 66±4 | 77±6 |
  | pointmaze-giant | 0 | 0 | 2±2 | 4±2 | 30±10 |
  | pointmaze-teleport | 0 | 0±1 | 24±7 | 46±4 | 41±8 |
  | antmaze-medium | 13±2 | 15±6 | 25±8 | 42±9 | 53±12 |
  | antmaze-large | 11±4 | 5±2 | 20±4 | 22±5 | 24±10 |
  | humanoidmaze-medium | 3±3 | 2±2 | 6±3 | 8±3 | 19±6 |
  | cube-single | 4±2 | 7±5 | 6±2 | 8±2 | 11±5 |
  | cube-double | 1±1 | 2±1 | 2±1 | 3±1 | 2±2 |
  | scene-play | 5±2 | 5±3 | 4±3 | 15±5 | 19±4 |
  | visual-cube-single | 1±1 | 1±1 | 1±2 | 13±7 | 37±9 |
  | visual-scene | 5±1 | 7±2 | 5±2 | 25±4 | 40±8 |
  | visual-cube-double | 0 | 0 | 0 | 10±4 | 7±5 |
  | puzzle-4x4 | 0 | 0 | 0 | 0 | 0 |
  | visual-puzzle-4x4 | 0 | 0 | 0 | 0 | 0 |

  In the hierarchical variant, HCFGRL vs flow HGCBC gives antmaze-medium 90±5 vs 67±8 and humanoidmaze-medium 64±10 vs 23±4. — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Varying w (Fig. 6 and App. B, 17 OGBench tasks, w ∈ {0.5, 1, 1.25, 1.5, 2, 3}).**
  - Verbatim: "performance improves as the CFG weight increases, although it sometimes declines beyond a certain point, likely because the policy deviates too far from the data distribution".
  - Read off Fig. 6: pointmaze-teleport peaks at about 52 (w=1.25) and falls to about 41 (w=3).
  - pointmaze-large falls from about 84 (w=2) to about 77 (w=3) in Fig. 4, and from about 67 to about 61 in Fig. 6. The two figures' absolute levels for this task disagree, and Fig. 4 is the one that matches Table 3.
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Shared vs separate networks (Fig. 7).** "Shared policies lead to better performance and extrapolation than separate policies". With separate networks, antmaze-medium, humanoidmaze-medium and scene-play collapse at w=3. — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Limitations (Sec 7) and robots.**
  - Verbatim: "we assume a given value function and do not make any prescriptions about how to train it".
  - Verbatim: "By itself, CFGRL does not represent a state-of-the-art RL algorithm".
  - No real-robot experiments; all tasks are simulated (ExORL, OGBench; 7 pixel-based OGBench tasks in Table 3).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)

### Inferences
- **Binary label makes w theoretically inert.** With the binary f = 1(A ≥ 0) used in the offline-RL experiments, f^w = f for every w > 0. The claimed target π̂·p(o|s,a)^w (Eq. 12) is therefore the same distribution (π̂ restricted to A ≥ 0) for all w > 0, and Theorem 2's ordering is trivial.
  - The measured gains from w > 1 (Fig. 3; Table 5, w up to 30) must come from the CFG approximation and the network's smoothing of p(o|s,a), not from the stated product distribution.
  - This matches QIPO's Lemma 4.10 that CFG at scale ≠ 1 does not sample the product.
  - The paper does not discuss this. This point is my own inference.
- **What conditioning would cost the reader.** Replacing weighting with conditioning would mean:
  - adding an optimality input to the actor with 10% dropout;
  - doubling action-expert evaluations per denoising step;
  - using only the sign of A, which discards the magnitude that the reader's standardised Â carries.
- **What the AWR comparison supports.** It shows that on MLP state-based tasks, with an IQL critic and per-task tuning, sign-conditioning plus guidance is at least as good as AWR on most tasks. Whether AWR used a flow or a Gaussian policy is not stated (Eq. 14 is written as log π_θ), so a policy-class confound cannot be ruled out.

### Gaps
- The AWR baseline's policy class (Gaussian or flow) is not specified in the pages read.
- No table reports plain BC or flow BC in the offline-RL (Tables 1-2) setting, so there is no BC/AWR/CFGRL three-way comparison.
- No effective sample size is computed for AWR beyond the single-batch illustration in Fig. 2.
- There is no study of sensitivity to critic error, or of thresholds other than A ≥ 0.

## Q3. Mei et al. 2026 (ForesightFlow): verification, what "potential-guided flow matching" is, what is weighted, setup, numbers, hyperparameters

### Takeaway
The paper exists. The actual title is slightly longer: "…for Vision-Language-Action Policy Improvement". The method is called ForesightFlow.

- **What "potential" means.** It is not a potential-based reward or a gradient guidance. The flow endpoint is augmented with a per-step success-potential vector s ∈ [0,1]^H. The advantage is A = y − V̂, where y is a manually annotated binary stage-completion label and V̂ is the model's own one-step (NFE=1) estimate of mean potential. "Guided" means best-of-K selection of sampled chunks by predicted potential.
- **What gets weighted.** The loss is advantage-weighted flow matching, with w = min(20, exp(A/0.3)) applied only to the action-velocity part of the same flow. The potential-velocity part is unweighted. No other losses (language/VLM) appear in the objective.
- **Results.** In sim (π0.5, BEHAVIOR-1K, 100 trials per task), ForesightFlow reaches 39.6% average SR vs BC 31.0% at K=5. Without best-of-K (K=1) it reaches 34.6%, below the IDQL baseline's 39.0% (IDQL uses a separate critic and IQL-style exp weighting). In real (π0, 5 bimanual tasks, trial count not stated), it reaches 35.4% vs IDQL 32.6% vs filtered BC 24.6%.

### Cited Findings
- **Verification.**
  - Title: "Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement". The description given to me said "VLA"; the paper spells it out.
  - Authors: Yunpeng Mei, Jiakai He, Hongjie Cao, Chenyu Wang, Xiaowen Zhu, Yihan Zhou, Jiamin Wang, Chenbo Xin, Peng Cheng, Yuxuan Yang, Yijie Wang, Xinhu Zheng, Gao Huang, Jie Chen, Gang Wang.
  - v1 submitted 3 Jun 2026, cs.RO. The method is named ForesightFlow.
  - [arXiv abs](https://arxiv.org/abs/2606.04968); [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Potential as a generated coordinate (Sec 3.1, Eqs. 8-9).**
  - x = [a; s] ∈ R^{H(d_a+1)} with s ∈ [0,1]^H. Verbatim: "The potential coordinate is not a separate critic network. It is an additional generated coordinate of the same flow endpoint".
  - Chunk score: Q̂_θ(c,a) = (1/H)Σ s_k (Eq. 9).
  - Architecturally, the action expert's output layer is split into an action head and a progress-potential head of dimension 1 (App. B.2), about 1K extra parameters (Table 3).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Stage labels and advantage (Sec 3.2, Eqs. 10-11).**
  - y_{i,t} = 1 "if the semantic stage assigned to chunk (i,t) is completed", else 0.
  - Potential target: s_1 = y·1_H.
  - Advantage: A_{i,t} = y_{i,t} − V̂_{i,t}.
  - "All trajectories are annotated with sparse stage-wise labels" (Sec 4, Datasets).
  - The limitations state that "the method requires stage labels" (Sec 6).
  - There is no TD bootstrapping; Theorem A.1 assumes "sparse undiscounted reward (γ = 1)".
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Baseline V̂ (Sec 3.3, Eqs. 12-14; Theorem A.1).**
  - x̂1 = x0 + sg(v_θ(x0, 0, c)), and V̂ = (1/H)Σ ŝ_k.
  - At the population optimum, with independent endpoint sampling (no minibatch OT), this equals E[Y|c] = P_D(Y=1|c).
  - It equals V^{πβ} only if the data come from a single stationary behaviour policy; in this dataset πβ "is interpreted as the dataset-mixture policy" (Remark A.4).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Weight and loss (Sec 3.4, Eqs. 15-16; Algorithm 1).**
  - w_{i,t} = min(M, exp(A_{i,t}/τ)). Verbatim: "The scalar weight w_{i,t} is broadcast across the action coordinates of the chunk; the potential velocity loss remains uniformly weighted".
  - L(θ) = E[ w_{i,t}·||v^(a)_θ(x_σ,σ,c) − u^(a)||² + ||v^(s)_θ(x_σ,σ,c) − u^(s)||² ], with the expectation over (i,t) ~ D, x0 ~ p0 and σ ~ U[0,1].
  - The weight is indexed by sample (i,t) only; it does not depend on flow time σ.
  - No normalisation (batch mean or softmax) and no standardisation of A appear in Eq. 15 or Algorithm 1 (lines 12-16).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Checking the claim "weights only the action (flow) loss, leaving other losses unweighted".**
  - Eq. 16 and Algorithm 1 line 16 contain exactly two terms: the weighted action velocity and the unweighted potential velocity.
  - No language, FAST-token, VLM or auxiliary loss is mentioned anywhere.
  - "Other losses" therefore means specifically the potential/value coordinate of the same flow.
  - Fine-tuning is "Full parameter" (Tables 7-8), so the weighted action loss also updates the VLM backbone.
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Hyperparameters.**
  - Simulation on π0.5 (Table 7):
    - action expert width 1024, depth 18, MLP 4096;
    - batch 128, chunk 30;
    - LR 1e-5, cosine schedule, AdamW (0.9, 0.95), weight decay 1e-10;
    - 30,000 steps;
    - "Number of Flow Samples 15";
    - AWR temperature τ = 0.3, clip M = 20.0;
    - full-parameter fine-tuning.
  - Real robot on π0 (Table 8): the same, except LR 2.5e-5 and 20,000 steps.
  - The BC baselines use identical settings (Tables 9-10).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Theorem A.5 (population AWR for flow matching).** The unclipped weighted action-CFM with w̄ = exp((Q(c,a) − V(c))/τ) is "equivalent, up to a positive context-dependent constant" C(c) = Z(c)exp(−V(c)/τ), to CFM under π* ∝ πβ exp(Q/τ) (Eqs. 26-34). — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Remark A.6 (bias of realised labels).** The implemented weights with realised binary labels "are not an unbiased estimator of exp((Q(c,a) − V(c))/τ)". With Y ~ Bernoulli(Q), E[exp((Y − V)/τ)|c,a] = exp(−V/τ)(1 + Q(exp(1/τ) − 1)), which is "monotone in Q(c,a) but … not the exact Boltzmann factor". Verbatim: "the practical update should be interpreted as an AWR reweighting surrogate". — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Setup (Sec 4; App. C, D).**
  - Sim: 5 BEHAVIOR-1K tasks in OmniGibson on the Galaxea R1 Pro (Radio, Trash, Spray, Hotdog, Wash). Episodes last 200-500 s (max horizons 6,000-15,000 steps). "success rate over 100 trials per task" plus a normalised stage-wise score.
  - Real: DexTeleop TeleAvatar Lite, dual 7-DoF arms, parallel grippers, a head binocular camera plus 2 wrist RGB-D cameras, 20 Hz control. Five tasks (Set-Paper-Roll, Pick-Trash, Cube-Stack, Transfer-Food, Wipe-Whiteboard) of 19-138 s, scored per stage on a 0 / 0.5 / 1 rubric (App. D.2).
  - Data per task: 200 expert demos (BEHAVIOR-1K human demos in sim, VR teleop in real) plus 100 autonomous rollouts from a pretrained π0.5 checkpoint, spanning successes, partial completions and failures (Sec 4; App. D.3).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Baselines (Sec 4.1; App. B.3-B.5).**
  - BC: full mixed data.
  - Filtered BC: "trains only on successful trajectories".
  - IDQL: a separate ~500M PaliGemma/SigLIP critic trained by TD, with the policy "fine-tuned using IQL-style advantage weighting", exponential weight β = 1, critic batch 64, LR 1e-5, 30k steps (Table 11). Failed episodes get "an additional label of −1" at the final step (App. D.3).
  - FQL: the VLM prefix and teacher frozen, a one-step student, twin-Q, α_distill = 10, α_BC = 5 (Table 12).
  - No pure AWR baseline without the potential head, and no advantage-conditioning (CFG) baseline.
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Table 1 (sim, SR %; ForesightFlow at K=5, others single-sample).**

  | Method | Radio | Trash | Spray | Hotdog | Wash | Average SR | Average score |
  |---|---|---|---|---|---|---|---|
  | BC | 42 | 35 | 0 | 65 | 13 | 31.0 | 0.35 |
  | Filtered BC | 44 | 30 | 14 | 46 | 24 | 31.6 | 0.38 |
  | FQL | 42 | 32 | 18 | 59 | 21 | 34.4 | 0.39 |
  | IDQL | 47 | 36 | 21 | 66 | 25 | 39.0 | 0.44 |
  | ForesightFlow | 51 | 38 | 20 | 59 | 30 | 39.6 | 0.46 |

  [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Table 6 (matched candidate budget, SR %).**

  | Method | Radio | Trash | Spray | Hotdog | Wash | Average SR |
  |---|---|---|---|---|---|---|
  | IDQL K=1 | 47 | 36 | 21 | 66 | 25 | 39.0 |
  | IDQL K=5 | 49 | 35 | 18 | 64 | 26 | 38.4 |
  | ForesightFlow K=1 | 45 | 28 | 21 | 53 | 26 | 34.6 |
  | ForesightFlow K=5 | 51 | 38 | 20 | 59 | 30 | 39.6 |

  Verbatim: "ForesightFlow is weaker than IDQL in average SR at K=1 (34.6% vs 39.0%), but remains ahead at K=5 (39.6% vs 38.4%)" (Sec 4.2). — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Table 2 (real, SR %; no BC or FQL rows; K and trial count not stated).**

  | Method | Paper-Roll | Pick-Trash | Cube-Stack | Transfer-Food | Whiteboard | Average SR | Average score |
  |---|---|---|---|---|---|---|---|
  | Filtered BC | 23 | 38 | 19 | 29 | 14 | 24.6 | 0.51 |
  | IDQL | 33 | 46 | 23 | 41 | 20 | 32.6 | 0.59 |
  | ForesightFlow | 36 | 51 | 26 | 45 | 19 | 35.4 | 0.62 |

  The paper states gains of "+10.8 … and IDQL by 2.8 percentage points" (Sec 4.2). — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Table 4 (decoupled vs coupled weighting; Radio only; stage completion %).**

  | Model | Stage 1 | Stage 2 | Stage 3 |
  |---|---|---|---|
  | BC | 88 | 59 | 44 |
  | Coupled (w on action and potential) | 92 | 62 | 42 |
  | Decoupled (w on action only) | 97 | 60 | 51 |

  Fig. 5 analyses 19,905 frames from five Radio rollouts: the coupled model "assigns high potentials to explicit failures such as knocking the radio over or slipping". — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Compute (Table 3; Sec 4.3).**
  - IDQL: 287 GPU-h, ~500M critic parameters, ~2.84B total, latency 119 ms (K=1) / 183 ms (K=5).
  - ForesightFlow: 178 GPU-h (−38%), ~1K extra parameters, 2.35B total, 127 / 155 ms.
  - The NFE=1 baseline takes 8.4 h per 5,000 steps vs 11 h at NFE=20; both reach comparable success by 15k steps.
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **One-step ranking fidelity (Table 5; K=8, NFE=1 vs NFE=100).** Kendall τ is 0.801-0.863 (mean 0.834); mean top-1 agreement 87.2%; MC gain 96.8%. — [HTML v1](https://arxiv.org/html/2606.04968v1)

### Inferences
- **Weighting alone versus BC (sim, K=1, vs BC).**
  - Average SR: +3.6 pp (34.6 vs 31.0).
  - Worse on Trash (28 vs 35) and Hotdog (53 vs 65); better on Spray (21 vs 0), Wash (26 vs 13) and Radio (45 vs 42).
  - IDQL at K=1 (separate critic plus exp-advantage weighting, β=1) is at or above BC on all 5 tasks, +8.0 pp on average.
  - So with this paper's own numbers, the AWR-style policy with a separate critic beat the embedded-potential weighted policy; ForesightFlow's lead depends on best-of-5 selection.
- **Weight dynamic range.**
  - Mei: A ∈ [−1, 1] and τ = 0.3 give w from exp(−3.33) ≈ 0.036 up to min(20, 28) = 20. The clip binds when A ≥ 0.3·ln 20 ≈ 0.90. The max/min ratio is about 560, with no batch normalisation.
  - Reader: Â clipped at ±3 with β = 1 gives w ∈ [0.050, 20.1] before mean normalisation, a ratio of about 403. The two dynamic ranges are of the same order.
- **Relevance of the decoupling result.** It speaks to a value head that shares parameters with the policy and is trained through the same weighted loss. The reader's critic has its own frozen encoder and its own loss, so the "self-masking" loop in App. A.4 does not arise unless a value or progress head is added to the actor.

### Gaps
- The real-world trial count per task is not stated. SR values such as 23 or 51 suggest 100 trials, but this is not confirmed.
- No seeds or error bars are reported anywhere.
- It is not stated whether Table 2 (real) and Table 4 used K=5.
- There is no τ or M ablation.
- There is no weight histogram or effective sample size.
- "Number of Flow Samples 15" is not defined.
- In Table 4, BC's Stage-3 rate on Radio (44.0) differs from its full SR in Table 1 (42.0); this is unexplained.
- FQL was trained with a frozen VLM while BC and ForesightFlow used full fine-tuning, a confound.

## Q4. Failure modes and diagnostics reported across the three papers

### Takeaway
None of the three papers reports effective sample size, weight-concentration statistics, collapse onto a few samples, drift back to BC, or slower or more hesitant motions. The failure modes they do report are:
- AWR batch gradients dominated by a few samples (CFGRL Fig. 2);
- over-guidance hurting at large w (CFGRL Fig. 6);
- CFG at scale ≠ 1 distorting the distribution (QIPO Figs 1 and 3);
- value hallucination when a shared value coordinate is trained with the same weights (Mei Table 4, Fig. 5, App. A.4);
- high seed variance on hard AntMaze tasks (QIPO Table 2).

### Cited Findings
- **Weight concentration.** CFGRL Fig. 2 shows AWR per-element losses up to about 1000 within a batch with "Most samples are ignored". Sec 5: "the rest of the batch is effectively ignored, and AWR derives the gradient only from a small subset of the available data". — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Over-extrapolation.** CFGRL Fig. 6 and App. B: performance "sometimes declines beyond a certain point, likely because the policy deviates too far from the data distribution". Fig. 7: separate (unshared) conditional networks collapse at w=3. AWR itself "saturates around (1/β)=10" (Fig. 3). — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Non-exact guidance.** QIPO Sec 4.3 and Lemma 4.10: CFG is "not guaranteed to generate the desired distribution q0 when β ≠ 1". Fig. 3(b) shows CFG samples at β=10 scattered off the modes on six 2-D toy distributions. — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Seed variance.** QIPO-OT: MR-Walker2d 78.57±26.09, AntMaze large-play 55.5±29.39, large-diverse 32.13±23.16. QIPO-Diff: large-diverse 40.5±20.40 (Table 2). These are the best evaluations over training (E.1). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Value hallucination / self-masking (Mei App. A.4, Eqs. 35-37).**
  - For failures with an overestimated potential, A ≈ −1 and w ≈ 0 "for small temperature τ".
  - Under coupled weighting "the same small weight then masks the regression signal that would reduce V̂_θ". The false-positive potential "can remain uncorrected, corrupting later advantage estimates and self-guided candidate rankings".
  - Measured in Fig. 5 and Table 4 (Radio Stage 3: 42% coupled vs 51% decoupled).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Estimator bias.** Mei Remark A.6: with realised binary labels, weights are not unbiased for exp((Q−V)/τ). Remarks A.2-A.3: there is no formal bound on ||v_θ − v*||, and the NFE=1 value is "not an exact surrogate for the integrated ODE endpoint". — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Stated limitations.**
  - Mei: "requires stage labels, sparse outcomes can still make very long-horizon credit assignment difficult, and the one-step estimator is approximate in finite networks" (Sec 6).
  - CFGRL: "we assume a given value function and do not make any prescriptions about how to train it" (Sec 7).
  - QIPO: extension to online RL is left open (Sec 6).
  - [HTML v1](https://arxiv.org/html/2606.04968v1); [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1); [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)

### Inferences
- **Sensitivity to critic error.**
  - CFGRL reduces the critic to the sign of A, so magnitude errors cannot inflate weights.
  - Mei caps weights at 20 with an unnormalised exponent.
  - QIPO normalises within each state over 17 candidates, so a single outlier state cannot dominate the batch.
  - Each design implicitly limits weight concentration in a different way; none measures the effect.

### Gaps
- No paper reports:
  - effective sample size, (Σw)²/Σw², or the fraction of the batch carrying most of the weight;
  - motion speed or duration of weighted policies;
  - behaviour under injected critic noise;
  - the performance trajectory during training showing drift toward BC.

## Q5. Critique, and where the papers agree or contradict on "weight the loss" vs "guide/condition" vs "exact energy flow"

### Takeaway
- **Agreement.** QIPO (Theorem 4.3 / Eq. 5.3) and Mei (Theorem A.5) independently establish the same fact: a per-sample, flow-time-constant exp(Q/τ) or exp(A/τ) weight on the conditional FM loss has the Boltzmann-reweighted behaviour policy as its population minimiser. "Weighting" and "exact energy flow" are therefore the same thing, not rivals.
- **Where they conflict.** On guidance, QIPO proves CFG at scale ≠ 1 is not exact, while CFGRL shows empirically that CFG with w > 1 improves return and is at least competitive with AWR.
- **Evaluation limits.**
  - QIPO and CFGRL are state-based simulation only (plus 7 pixel-based CFGRL GCBC tasks) with per-task tuning.
  - Only Mei has a VLA or robot, and its ablations are thin.
  - Where a real comparison against separate-critic weighting exists (Mei Table 6, K=1), the weighted separate-critic baseline won.

### Cited Findings
- **Agreement on exactness of per-sample weighting.** QIPO: "∇θ L_EFM = ∇θ L_CEFM" with a constant normaliser (Theorem 4.3; Lemma B.1). Mei: the weighted action CFM is equivalent "up to a positive context-dependent constant" to CFM under πβ exp(Q/τ) (Theorem A.5). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1); [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Contradiction on guidance.** QIPO: Eq. 4.6 ≠ Eq. 4.7 for β ≠ 1; toy CFG samples at β=10 are distorted (Figs 1, 3). CFGRL: Eq. 11 "corresponds to the attenuated distribution" π̂·p(o|s,a)^w (Eq. 12), and performance improves with w (Figs 3, 4, 6). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1); [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Disagreement on the value of weighting.** CFGRL says AWR "derives the gradient only from a small subset of the available data" (Sec 5, Fig. 2). QIPO and Mei adopt weighting as the core update. Mei retains weighting even with a clip of 20 and τ = 0.3 (Table 7). — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1); [HTML v1](https://arxiv.org/html/2606.04968v1)
- **QIPO critique.**
  - Averages vs baselines: locomotion 86.2 / 86.3 vs QGPO 86.6 and SRPO 87.1; AntMaze 77.3 vs IDQL 79.1 (Table 2). Yet the text says it "consistently outperform[s] the baselines".
  - Results are the best over training (E.1).
  - Ablations use 3 seeds and are shown only as bar charts (Figs 5-6).
  - No BC baseline; D4RL state-based only.
  - [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **CFGRL critique.**
  - Both methods are tuned per task (Tables 5, 7).
  - 4 seeds for offline RL.
  - OGBench AWR vs CFGRL is mixed (antmaze-large 50±9 vs 20±9).
  - The ExORL w values (10, 30) lie outside the stated w sweep (Sec 5 vs Table 5).
  - No BC in Tables 1-2 and no AWR in Table 3.
  - No robots (Sec 7; App. C).
  - [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)
- **Mei critique.**
  - Table 1 compares ForesightFlow at K=5 against single-sample baselines (caption); at matched K=1, IDQL wins 39.0 vs 34.6 (Table 6).
  - Real-world Table 2 omits BC and FQL.
  - The decoupling ablation covers one task (Table 4).
  - No τ or M sweep and no seeds.
  - The IDQL critic uses a different reward signal (an extra −1 at failure; App. D.3).
  - [HTML v1](https://arxiv.org/html/2606.04968v1)

### Inferences
- **The genuine design axis is not weight versus exact flow.** It is continuous exp(A/τ) weights versus binary conditioning plus test-time guidance, with best-of-K selection as a third, separate lever (Mei Table 6: +5.0 pp from K=5).
- **Weighting.** Keeps A's magnitude and the Boltzmann target (QIPO, Mei). It risks batch concentration (CFGRL Fig. 2) and fixes τ at training time.
- **Conditioning.** Keeps every sample at equal gradient weight and allows w to be tuned at test time (CFGRL). It uses only sign(A), the extrapolation at w > 1 has no exact target (QIPO Lemma 4.10), and inference costs double.
- **No paper runs a direct comparison of weighting versus conditioning on a VLA or real robot.**

### Gaps
- No head-to-head of exp-weighted flow matching versus CFGRL-style conditioning on the same flow VLA and data.
- Mei does not cite or compare against energy-weighted flow matching or CFGRL; its reference list was not fully checked.

## Q6. Objective (a)-(c) mapped onto the reader's setup (MolmoAct2 flow VLA; TD(1 s) advantage from a V-only HL-Gauss critic; per-sample weight exp(Â/β)/mean, constant over flow time)

### Takeaway
- **(a) Right thing to do?** Yes, in principle. Per-sample, flow-time-constant exp(A/β) weighting of the conditional FM loss is the provably exact way (at the population optimum) to fit π ∝ μ·exp(A/β) (QIPO Theorem 4.3 / Eq. 5.3; Mei Theorem A.5). "Exact energy-weighted flow" adds no flow-time or x_t-dependent term. The QIPO extras (support actions, iterative sharpening) need a Q(s,a) critic the reader does not have.
- **(b) Where the weight applies.** Weight only the action-generation loss, uniformly across flow time (Mei Eq. 16; QIPO Eq. 4.3). Any value or progress coordinate trained through the actor must stay unweighted (Mei Table 4).
- **(c) Measured gains.**
  - The only flow-VLA evidence (Mei) shows weighting alone at +3.6 pp SR over BC in sim.
  - A separate-critic exp-weighted baseline (IDQL, β=1) shows +8.0 pp over BC in sim and +8.0 pp over filtered BC on real robots.
  - CFGRL shows conditioning beating AWR on 8 of 9 ExORL tasks but mixed results on OGBench.

### Cited Findings
- **Exactness conditions.** The weight depends only on the clean sample, with a constant normaliser: "E_{x0}[exp(−βE(x0))] is a constant with respect to x and t" (Lemma B.1; Eq. 4.3). The per-state normaliser is used for policies (Eq. 5.3). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Batch normalisation is QIPO's own practical estimator.** "can be approximated by the empirical average … in a batch" (Sec 4.2; Alg. 1). — [QIPO PDF](https://arxiv.org/pdf/2503.04975v1)
- **Flow-time-constant weight in a VLA.** Mei's w_{i,t} is fixed per sample, with σ ~ U[0,1] (Eq. 16). Full-parameter fine-tuning of π0 / π0.5, batch 128, τ = 0.3, M = 20 (Tables 7-8). — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Noisy advantages bias the weights.** Mei Remark A.6: exponentiated single-sample labels are "not an unbiased estimator" of exp((Q−V)/τ). — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Mixed-quality data resembling the reader's.** Mei uses 200 teleop demos plus 100 autonomous rollouts per task, including failures (Sec 4; App. D.3). BC collapses on Spray (0%) and filtered BC underperforms BC on Hotdog, 46 vs 65 (Table 1). — [HTML v1](https://arxiv.org/html/2606.04968v1)
- **Conditioning alternative.** o = 1[A ≥ 0], 10% dropout, v = (1−w)v(∅) + w·v(o=1) (Eq. 15; Algs 1-2). Gains over AWR are largest on ExORL, e.g. walker-stand 782 vs 603 and cheetah-run-backward 262 vs 146 (Table 1). — [CFGRL PDF](https://arxiv.org/pdf/2505.23458v1)

### Inferences
- **Standardisation changes the effective temperature.** Standardising A over the effective batch makes the temperature on raw A equal to β·std_batch(A). The target temperature therefore moves as the critic's advantage spread changes during training. None of the three papers standardises A:
  - QIPO uses raw Q with a per-state softmax (β=3);
  - Mei uses raw A ∈ [−1,1] (τ=0.3, clip 20);
  - CFGRL's AWR uses raw IQL A with 1/β ∈ {1, 3, 10, 30}.
- **Mean normalisation across states.** Dividing by the batch-mean weight over 128 rows, which mix states, is a global constant per batch. Per QIPO Eq. 4.3 this leaves the minimiser unchanged in expectation. QIPO's per-state softmax (Alg. 4) instead gives every state equal total weight. Under the reader's global normalisation, states whose rows all have high Â get more total gradient. This affects capacity allocation, not the population target.
- **The clip truncates the Boltzmann target.** The ±3 clip on Â (dynamic range e^6 ≈ 403) plays the role of Mei's M = 20 (range about 560). Both cut the target's tails. Neither paper quantifies what the truncation costs.
- **Nearest published analogue to the reader's pipeline.** Mei's IDQL baseline: a separate critic plus exp-advantage-weighted flow on a π0/π0.5 VLA, β = 1 (Table 11). It gave +8.0 pp over BC in sim (Table 1) and +8.0 pp over filtered BC on real robots (Table 2), but with a Q-critic trained by TD, not a V-only TD(1 s) advantage.

### Gaps
- No paper tests:
  - a V-only one-step TD advantage;
  - batch-level standardisation of A;
  - HL-Gauss or distributional critics as the source of the weight;
  - interaction of the weight with other actor losses (e.g. discrete action tokens or language) on a VLA.
- No paper reports behaviour-level side effects (speed, hesitation) of advantage weighting on a real arm.
