# Additional 2024+ papers that measure AWR-style weighted imitation on real robots or VLAs

Scope: papers not on the exclusion list. Six papers were read in full: ARFM, SARM (RA-BC), ReTVL, PACL, ReinboT and PAC. Two more, Hi-ORS and GR-RL, were read in full as filtering comparators. SRPO was read but is thin. Numbers come only from full-text reads. Anything else is labelled "abstract only".

## Q1. Which papers qualify, what is their setting, and how were they found?

### Takeaway
Two papers are the closest match to the reader's setup: a flow-matching VLA trained with weighted imitation, tested on a real arm against plain BC and at least one other alternative. They are SARM/RA-BC (π0, bimanual YAM arms) and ReTVL (π0-style flow VLA, Piper arm). PACL has the closest data regime (demos plus mixed-quality deployment rollouts on a real 7-DoF Franka), but it uses Q-conditioning, not AWR. ARFM is the only one that studies the weight-variance tradeoff (π0; LIBERO plus UR5), but it has no critic and its offline data are successful demos only. ReinboT and PAC are the RWR/AWR-vs-BC comparisons on mixed-quality data with failures. Neither uses a flow VLA.

### Cited Findings

| Paper (venue) | Policy | Real robot | Offline data | Eval trials | How found |
|---|---|---|---|---|---|
| ARFM, Zhang et al., arXiv 2509.04063, AAAI 2026 — [arXiv](https://arxiv.org/html/2509.04063), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38944) | π0 flow VLA | UR5, 3 pick-and-place tasks (white squares, corn, cups, peppers) | UR5: about 720 **successful** trajectories (about 34,600 frames). No failures are stated. The reward is a 13-component hand-designed dense reward (Table 8). | LIBERO counts not captured. UR5 trial counts not stated. | Candidate list. Confirmed with the search "Balancing Signal and Variance Adaptive Offline RL Post-Training VLA Flow Models". |
| SARM / RA-BC, Chen et al., arXiv 2509.25358, ICLR 2026 — [arXiv](https://arxiv.org/html/2509.25358v3) | π0, LoRA, 40k steps, batch 32 | Two 6-DoF YAM arms (bimanual), 3× D405 cameras, GELLO teleop. T-shirt folding. | 200 h of human demos of mixed quality (D_all). There is also a 20 h subset of episodes shorter than 2 min (D_2min). | 12 per task (3 shirt colors × 4) | The search for SRPO returned "SARM2" (2606.10305), which led to the SARM RA-BC search. |
| ReTVL, Qin et al., arXiv 2606.24633 (v2, 2026-07-06) — [arXiv](https://arxiv.org/html/2606.24633) | Flow-matching VLA with a π0-style backbone. Action chunk 16. | AgileX Piper 6-DoF. Pick up Spoon, Stack Blocks, Fold Towel, Open Drawer. | 200 trajectories per task (80 for Spoon), with successes, failures and retries. The value model is trained on 30 annotated trajectories per task. | 20 per task | Returned by the ReinboT search ("Beyond Monotonic Progress: Retry-Supervised Value Learning"). |
| PACL, Ren et al., arXiv 2609.29000 (2026-09-24) — [arXiv](https://arxiv.org/html/2609.29000) | Diffusion Policy (not a VLA) with a small condition adapter | Franka Panda 7-DoF, 20 Hz. PickCup, StackCup, MoveSpoon. | Per task: 50 human demos plus 50 autonomous deployment rollouts (mixed success/failure). Sparse terminal reward: 1 for demos and successful rollouts, 0 for failures. | 25 per task | Search "advantage weighted diffusion policy real robot offline reinforcement learning mixed-quality demonstrations 2025". |
| ReinboT, Zhang et al., arXiv 2505.07395, ICML 2025 — [arXiv](https://arxiv.org/html/2505.07395) | GPT-style transformer (GR-1 lineage) with RTG/action/image tokens. Not a flow model. | UR5 (few-shot and OOD, Fig. 6). No numbers are given in the text. | CALVIN mixed-quality data: about 50 annotated trajectories per task, 20,000+ unlabeled teleop trajectories, and 10,000+ failure trajectories from RoboFlamingo with Gaussian noise σ ∈ {0.05, 0.1, 0.15}. Real: about 530 successful trajectories, 30 per task for few-shot. | Not captured | Candidate list. ARFM (same group) uses it as a baseline. |
| PAC (Perceiver-Actor-Critic), Springenberg et al., arXiv 2402.05546, ICML 2024 — [arXiv](https://arxiv.org/html/2402.05546) | 988M multi-task Perceiver actor-critic. Not a VLA. | Sawyer, RGB-stacking benchmark, 5 tasks | Mix of 3.64M episodes across 132 tasks. CHEF:real is RL-agent data with 12% success. | 400 per task | Prior knowledge of offline actor-critic at scale. Verified in the full text. |
| Hi-ORS, arXiv 2510.26406 (2025) — [arXiv](https://arxiv.org/html/2510.26406) | π0 (PaliGemma-3B plus a 300M action expert) | Paxini Tora One (2 tasks), Dobot X-Trainer (1 task). Raise-Hand, Pack-Detergent, Insert-Moisturizer. | Human demos, online rollouts and human corrections. About 1.5 h of real training. | 10 per checkpoint | Candidate list, plus search. |
| GR-RL (ByteDance), arXiv 2512.01801 (2025) — [arXiv](https://arxiv.org/html/2512.01801) | GR-3 VLA | Real, long-horizon dexterous task (Fig. 5) | Human demos with noisy or suboptimal parts. The critic is trained on successes and failures. | Not captured | Candidate list. |
| SRPO, arXiv 2511.15605 (2025) — [arXiv](https://arxiv.org/html/2511.15605) | π0 and π0-FAST | X-ARM 7. Apple→plate, pear→plate, fold towel, wipe whiteboard, select poker. | 30 successful plus 20 failed trajectories per task, used for reward validation. The buffer used for AWR is not specified. | Not stated | First broad search, "advantage-weighted regression vision-language-action real robot offline RL". |

### Inferences
- ReTVL and SARM share the same RA-BC weighting code path (Q2). Together they give two independent real-robot tests of one weighting rule on flow VLAs, with the value model swapped out.
- None of the included papers uses exactly the reader's recipe: a learned V(s), a 1-step TD advantage, and exp(Â/β) on dataset actions of a flow VLA. PAC comes closest on the critic (distributional, 1-step TD), but it weights actions sampled from the policy, not dataset actions.

### Gaps
- No 2024+ paper outside the exclusion list was found that is titled "advantage-weighted flow matching" or "reward-weighted flow matching" and runs on a robot. ARFM is the only flow-VLA paper whose loss is an explicit exponential reward weight on the flow loss.

## Q2. Exact weighting formulas, temperature, clipping, normalization, advantage estimator

### Takeaway
The two flow-VLA papers that report large real-robot gains (SARM, ReTVL) do not use exponential weights. They use a bounded soft filter: weight 0 for negative progress, a linear ramp over ±2σ, weight 1 above a threshold κ, and the loss normalized by Σw. ARFM uses a batch softmax exp(α·R*) with α re-solved every batch. PAC uses exp(Q/η) on policy samples plus a BC term. ReinboT's RWR multiplies log-likelihood by the raw discounted return.

### Cited Findings
- **ARFM**
  - Loss: L = Σ_i w_i(α)‖v_θ − u‖² with w_i(α) = exp(α R*_i) / Σ_j exp(α R*_j), a softmax over the batch — [ARFM](https://arxiv.org/html/2509.04063).
  - Advantage R* is a leave-one-out baseline over K samples, R*_k = K/(K−1)·(R_k − mean R). The paper never states whether K equals the batch size B — [ARFM](https://arxiv.org/html/2509.04063).
  - No learned value function is used — [ARFM](https://arxiv.org/html/2509.04063).
  - α is chosen per batch by minimizing J(α) = σ_L²[e^{2α²σ_R²} − e^{α²σ_R²}] − λασ_R². The first term is the variance of the flow-loss gradient; the second is the RL "signal". The solution comes from bisection on 4√x·e^{2x} − 2√x·e^{x} − λσ_R/σ_L² = 0, with α* = √x*/σ_R — [ARFM](https://arxiv.org/html/2509.04063).
  - σ_R² = Σ R_i²/B and σ_L² is the batch variance of the per-sample flow loss. λ = 5e-4, batch 16, 40k steps (LIBERO) and 60k steps (UR5) — [ARFM](https://arxiv.org/html/2509.04063).
  - The RWR baseline is "reproduced for flow matching". Its temperature is not stated — [ARFM](https://arxiv.org/html/2509.04063).
- **SARM RA-BC**
  - Progress delta: r̂_i = φ(o_{t+Δ}) − φ(o_t), with Δ = 25 actions (about 0.83 s at 30 fps) — [SARM](https://arxiv.org/html/2509.25358v3).
  - Soft weight: w̃_i = clip((r̂_i − (μ − 2σ))/(4σ + ε), 0, 1). μ and σ are running statistics computed with Welford's algorithm — [SARM](https://arxiv.org/html/2509.25358v3).
  - Final weight: w_i = 1 if r̂_i > κ; w̃_i if 0 ≤ r̂_i ≤ κ; 0 if r̂_i < 0 — [SARM](https://arxiv.org/html/2509.25358v3).
  - Loss: L = Σ w_i ℓ_i / (Σ w_i + ε), with κ = 0.01 ("top 5%") and ε = 1e-6 — [SARM](https://arxiv.org/html/2509.25358v3).
- **ReTVL**
  - Same form as RA-BC: r_t = V(h_{t+Δa}) − V(h_t), α_t = clip((r_t − (μ − 2σ))/(4σ + ε), 0, 1), with μ and σ from offline dataset statistics — [ReTVL](https://arxiv.org/html/2606.24633).
  - A task-specific κ sets weight 1 for chunks above it, and the loss is Σ α_t ℓ_t / (Σ α_t + ε) (Eq. 11) — [ReTVL](https://arxiv.org/html/2606.24633).
  - The value model is distributional (64 bins). Global targets are t/T for successful trajectories and 0 at failed terminal frames. A local pairwise preference loss is applied around annotated retry keypoints (λ_abs = 1, λ_pref = 3) — [ReTVL](https://arxiv.org/html/2606.24633).
- **PACL (conditioning, not weighting)**
  - Critic: chunk-level IQL with expectile τ = 0.9, chunk H_a = 8, γ = 0.99, and min over 2 Q-networks — [PACL](https://arxiv.org/html/2609.29000).
  - Chunks at or above the 60th percentile of Q get condition 1, chunks at or below the 40th percentile get 0, and the 40–60 band is an "abstain" condition ∅ trained unconditioned. Human demos are always 1 — [PACL](https://arxiv.org/html/2609.29000).
  - Inference: condition = 1, sample N = 12 chunks, and execute the chunk with the highest Q — [PACL](https://arxiv.org/html/2609.29000).
- **PAC**
  - Loss: (1−α)·KL[π_imp ‖ π_θ] + α·KL[b ‖ π_θ] + β·(distributional TD critic loss) — [PAC](https://arxiv.org/html/2402.05546).
  - The improvement term weights actions **sampled from the target policy** (N = 10) by exp(Q/η) / E[exp(Q/η)]. The BC term uses dataset actions — [PAC](https://arxiv.org/html/2402.05546).
  - Critic: categorical with 101 bins, 1-step TD, target network — [PAC](https://arxiv.org/html/2402.05546).
  - Variants: PAC uses α = 0.75. α-PAC uses a per-dataset best α. η is not reported — [PAC](https://arxiv.org/html/2402.05546).
- **ReinboT**
  - RWR baseline (Eq. 14): ∇ log π(a|·) multiplied by Σ_{i≥t} γ^{i−t} r_i with γ = 0.9. The weight is linear in return and no temperature is reported — [ReinboT](https://arxiv.org/html/2505.07395).
  - ReinboT instead predicts return-to-go with expectile regression (m = 0.9 best) and feeds the RTG features to the action head — [ReinboT](https://arxiv.org/html/2505.07395).
  - The dense reward combines sub-goal similarity (MSE/SSIM/ORB), progress, smoothness and completion, with weights 0.1 / 0.1 / 0.01 / 0.1 — [ReinboT](https://arxiv.org/html/2505.07395).
- **Hi-ORS**
  - Weight: the indicator 1[R(τ) ≥ m] on the per-trajectory return, applied to the π0 flow-matching loss. The threshold m increases across iterations. There is no temperature and no critic — [Hi-ORS](https://arxiv.org/html/2510.26406).
- **GR-RL**
  - A distributional critic gives progress ρ_t = mean Q. A sample is dropped if ρ falls by more than δ within ρ_{t:t+k}. The rest is trained with plain BC — [GR-RL](https://arxiv.org/html/2512.01801).
- **SRPO**
  - Advantage: A_{i,t} = (D_{i,t} − μ)/σ, where D is the per-step progress increment of a V-JEPA-2-latent reward, and μ and σ are taken over all trajectories. The paper states "integrates AWR" but gives no loss, β, clip or weight normalization — [SRPO](https://arxiv.org/html/2511.15605).

### Inferences
- **What the reader's standardization already covers, and what ARFM adds.** ARFM's α* = √x*/σ_R means ARFM also divides the advantage by its scale. The reader's Â/β with Â standardized is the same move. What ARFM adds is lowering the effective temperature when the per-sample flow-loss variance σ_L² is large relative to λσ_R.
- **Arithmetic from ARFM's J(α).** Write y = ασ_R for the temperature on the standardized advantage.
  - The optimum satisfies σ_L²(4y·e^{2y²} − 2y·e^{y²}) = λσ_R.
  - The reader's y = 1/β = 1 is optimal under ARFM's model only if λσ_R/σ_L² ≈ 4e² − 2e ≈ 24.1.
  - The variance factor e^{2y²} − e^{y²} is 4.67 at y = 1 and 0.37 at y = 0.5, about 13× lower.
  - This leans on ARFM's Gaussian assumption. The reader's clip at ±3 bounds the tail, so it is only a rough guide.
- **Weight ratio ceilings.** RA-BC and ReTVL weights sit in [0, 1], so the best sample weighs at most about 1/ε_min more than a down-weighted one, and negatives are zeroed. The reader's exp(Â) with Â ∈ [−3, 3] allows a ratio of up to e⁶ ≈ 403 between extreme samples in one batch. No paper here reports a successful real-robot run with ratios that large.

### Gaps
- ARFM does not say whether K in Eq. 4 is the batch size, or whether R is per-step or return-to-go.
- The temperature of ARFM's RWR baseline is not stated.
- SRPO gives no AWR hyperparameters.
- PAC's η is not reported.

## Q3. Measured results: weighted vs plain BC/SFT vs other alternatives

### Takeaway
- **Real flow VLAs.** Progress-weighted BC beat plain BC by large margins: SARM 10/12 vs 1/12 (fold from flat) and 8/12 vs 0/12 (fold from crumpled); ReTVL 80% vs 41% average over 4 tasks at n = 20 each.
- **The value model matters as much as the weighting rule.** With the formula fixed, swapping the value model moved results by 4–5 of 12 trials (SARM) and 17 points (ReTVL).
- **In simulation (π0 on LIBERO), the effects are small.** RWR gains 2.7 points on average over SFT, and ARFM gains another 1.3 over RWR.
- **Filtering and critic re-ranking baselines stay close** to the best method on real robots when trials are few (PACL).

### Cited Findings
- **SARM Table 2, real π0, 12 trials each** — [SARM](https://arxiv.org/html/2509.25358v3)

  | Task | BC-All | BC-2min (duration filter) | RA-BC, ReWiND reward | RA-BC, SARM reward |
  |---|---|---|---|---|
  | Fold from flattened | 1/12 | 7/12 | 6/12 | 10/12 |
  | Fold from crumpled | 0/12 | 0/12 | 3/12 | 8/12 |

- **ReTVL Table 2, real flow VLA, 20 trials each, success %** — [ReTVL](https://arxiv.org/html/2606.24633)

  | Task | Standard BC | Weighted BC, baseline progress-value model ("RECAP-BC" in the paper) | ReTVL-weighted BC |
  |---|---|---|---|
  | Spoon | 60 | 65 | 85 |
  | Stack | 45 | 80 | 95 |
  | Towel | 50 | 65 | 80 |
  | Drawer | 10 | 40 | 60 |
  | Average | 41 | 63 | 80 |

  The paper states +38.75 points over BC and +17.5 over the baseline weighting.

- **PACL Table I, real Franka, 25 trials each, success %.** All methods start from the same pretrained Diffusion Policy ("Baseline DP") — [PACL](https://arxiv.org/html/2609.29000)

  | Method | PickCup | StackCup | MoveSpoon |
  |---|---|---|---|
  | Baseline DP | 84 | 72 | 64 |
  | SUB (BC on demos + successes + failures) | 92 | 84 | 68 |
  | Self-Imitation (demos + successful rollouts only) | 100 | 92 | 72 |
  | SSDF (segment scoring) | 100 | 96 | 80 |
  | DQL | 88 | 76 | 72 |
  | IDQL (critic re-ranking) | 100 | 92 | 76 |
  | RISE | 100 | 96 | 80 |
  | PACL | 100 | 100 | 84 |

  The same pattern holds on Robomimic ToolHang: 46.4 / 70.0 / 78.0 / 81.2 / 56.4 / 58.8 / 64.4 / 82.8 in the same row order — [PACL](https://arxiv.org/html/2609.29000).

- **PACL conditioning ablation** (Fig. 6, Robomimic Square, single sample with no re-ranking), success % — [PACL](https://arxiv.org/html/2609.29000)

  | Condition | Success |
  |---|---|
  | Source (demo vs deployment) | 85.6 |
  | Outcome (success vs fail) | 88.0 |
  | Q − V (advantage) | 88.8 |
  | Q | 91.2 |

- **ARFM (π0)** — [ARFM](https://arxiv.org/html/2509.04063)
  - LIBERO multi-task (Table 1) average: π0 SFT 88.1, ReinboT 91.2, RWR 90.8, ARFM 92.1.
  - LIBERO-Long per method: 74.2 / 81.4 / 80.4 / 82.6.
  - Action perturbation (Table 2) average: 43.3 / 46.3 / 46.4 / 48.2.
  - Few-shot LIBERO-Long average over 30, 20 and 10 shots (Table 3): 32.5 / 33.9 / 34.6 / 36.5. At 30-shot, RWR (39.5) is below π0 (41.7).
  - Continual learning (Table 4): NBT 7.5 / 6.6 / 7.3 / 4.7 and SR 55.2 / 55.9 / 55.3 / 61.0.
  - UR5 under perturbation: Figure 5 only. The text says ARFM > ReinboT > RWR, and no numbers are given.
- **ReinboT Table 1, CALVIN ABC→D mixed-quality, average chain length** — [ReinboT](https://arxiv.org/html/2505.07395)

  | Method | Avg length |
  |---|---|
  | GR-1 on all data (plain BC) | 1.36 |
  | GR-1 on annotated only | 1.41 |
  | RWR, sparse reward | 1.38 |
  | RWR, sub-goal + sparse | 1.73 |
  | RWR, dense (scalar) | 1.82 |
  | ReinboT, sparse | 1.74 |
  | ReinboT, dense scalar | 1.90 |
  | ReinboT, dense vector | 2.26 |

  Ablation: removing RTG gives 1.36 — [ReinboT](https://arxiv.org/html/2505.07395).
  On the real UR5, "RWR performs on par with GR-1". The text gives no numbers — [ReinboT](https://arxiv.org/html/2505.07395).

- **PAC** — [PAC](https://arxiv.org/html/2402.05546)
  - Real Sawyer, CHEF:real with 12%-success data, 400 trials per task (Table 2): BC+Q (α = 1, the BC policy) 7.1% [6.1, 8.2] vs α-PAC 69.8% [67.8, 71.8]. After three self-improvement rounds (α-PAC + RLFT, with about 196k extra episodes): 93.2%.
  - Sim (Table 1), FilteredBC / BC+Q / PAC / α-PAC:

    | Domain | FilteredBC | BC+Q | PAC | α-PAC |
    |---|---|---|---|---|
    | CHEF:sim (28%-success data) | 17.0 | 11.0 | 55.0 | 52.0 |
    | Control Suite | 75.8 | 84.6 | 87.7 | 92.1 |
    | RC:Tower | 64.0 | 71.3 | 69.3 | 69.6 |
    | RC:Pyramid | 64.0 | 62.4 | 63.5 | 64.9 |

- **Hi-ORS (π0, real)** — [Hi-ORS](https://arxiv.org/html/2510.26406)
  - The text claims +23.3 points on average over BC. Per-baseline comparisons with HIL-SERL and Q-chunking appear only in the Fig. 4 curves.
  - Ablation on Insert-Moisturizer (Table I): full 80%, a learned reward classifier instead of rule rewards 60%, no human correction 0%, no no-op filter 20%, 5-step execution 10%, 25-step execution 40%.
- **GR-RL (real)** — [GR-RL](https://arxiv.org/html/2512.01801): BC on all data 45.7% → critic-filtered BC 61.6% → + symmetry augmentation 72.7% → + online RL 83.3%.
- **SRPO (real, offline AWR)** — [SRPO](https://arxiv.org/html/2511.15605): the text states average relative gains of +66.8% (π0) and +86.7% (π0-FAST) over SFT. Per-task values appear only in the Fig. 8 bar chart.

### Inferences
- **Filtering is a strong baseline.** SARM's cheap duration filter recovered most of the gap on the easier task (7/12 vs 10/12). A Fisher exact test on 10/12 vs 7/12 gives p ≈ 0.37 (my arithmetic). Only the hard task separates them (8/12 vs 0/12).
- **Where re-ranking and filtering still trail.** In PACL, filtering and IDQL re-ranking are within 0–8 points of PACL on real tasks, which is 0–2 trials out of 25. On Robomimic Square and Transport they trail by 6–10 points (filtering 86.8 and 89.2 vs PACL 93.2 and 96.0; IDQL 83.6 and 90.0).
- **BC on everything, failures included, still beat the pretrained policy** in PACL (SUB 92/84/68 vs 84/72/64). Adding more on-task data helps on its own, which confounds "weighting vs baseline" comparisons that start from a policy trained on less data.
- **Gains grow as data quality falls.** PAC gains 3× on data with 12–28% success, and nothing on near-expert RC data. ReinboT's RWR with a sparse outcome reward equals BC (1.38 vs 1.36), and only a dense progress reward helps (1.82). The reader's data (teleop plus mixed autonomous rollouts) sits in between. The value of weighting depends on how dense and correct A is.

### Gaps
- LIBERO and CALVIN evaluation episode counts were not captured for ARFM or ReinboT.
- Real-robot numbers for ARFM, ReinboT and SRPO exist only as figures.
- No paper here gives a head-to-head AWR (exponential) vs RA-BC (clipped ramp) vs filtering comparison on the same real-robot data. SARM compares against a duration filter, not a value-threshold filter.

## Q4. Failure modes and diagnostics: weight concentration, gradient variance, drift, speed, critic error

### Takeaway
Critic quality is the only failure mode with real-robot measurements. The evidence comes from swapping reward or value models under a fixed weighting rule, and from critic-quality ablations in which re-ranking with a weak critic gets worse as more candidates are sampled. Across these papers:
- Gradient variance is analyzed theoretically (ARFM) but never measured.
- No paper reports weight concentration or effective sample size.
- Only ReTVL reports how its weights distribute over labelled chunk types.

### Cited Findings
- **Critic or reward error, measured by swapping the model**
  - SARM: at a fixed RA-BC formula, ReWiND rewards give 6/12 and 3/12, against 10/12 and 8/12 with SARM — [SARM](https://arxiv.org/html/2509.25358v3).
  - SARM reward-model table: rollout Spearman ρ is −0.39 for GVL, 0.50 for ReWiND and 0.67 for SARM. The paper describes GVL as "overly pessimistic" and VLC as "excessively optimistic", and says ReWiND "fails to recognize regressions or failures" — [SARM](https://arxiv.org/html/2509.25358v3).
  - ReTVL: at a fixed formula, the baseline value model gives a 63% average and ReTVL gives 80% — [ReTVL](https://arxiv.org/html/2606.24633).
- **Weight diagnostics (ReTVL Table 7)**, reported per value model — [ReTVL](https://arxiv.org/html/2606.24633)

  | Value model | Success W. ↑ | Post-retry W. ↑ | Success Del. ↓ | Strict-bad Ret. ↓ |
  |---|---|---|---|---|
  | ReTVL | 0.769 | 0.636 | 0.146 | 0.206 |
  | Baseline progress-value model | 0.716 | 0.438 | 0.150 | 0.743 |
  | Robometer | 0.845 | 0.432 | 0.085 | 0.896 |
  | TOPReward | 0.539 | 0.498 | 0.330 | 0.695 |

  The column names are from the paper. I read them as: mean weight on success chunks, mean weight after a retry, share of success chunks zeroed, and share of strictly-bad chunks kept.
  ReTVL's value ablation: removing the preference loss leaves global VOC at 0.994 but drops Drop-AUC from 0.797 to 0.510 — [ReTVL](https://arxiv.org/html/2606.24633).
- **Critic re-ranking sensitivity** (PACL Table II, Transport; N = number of sampled candidates) — [PACL](https://arxiv.org/html/2609.29000)

  | Critic | N = 4 | N = 16 |
  |---|---|---|
  | Plain one-step critic | 79.2 | 58.8 |
  | Chunk critic | 85.2 | 63.6 |
  | PACL predictive chunk critic | 89.2 | 80.4 |

- **Critic needs failures** (PACL Table III) — [PACL](https://arxiv.org/html/2609.29000)

  | Setting | Square | Transport |
  |---|---|---|
  | PACL without failed rollouts ("PACL-success") | 77.6 | 83.2 |
  | Baseline DP | 78.8 | 84.0 |
  | + 50 failures | 88.0 | 88.8 |
  | Full data | 93.2 | 96.0 |

  The paper's wording: "failure experience provides essential supervision for critic learning".
- **Gradient variance**
  - ARFM derives the variance term σ_L²[e^{2α²σ_R²} − e^{α²σ_R²}] under its assumptions. It shows no plot of measured gradient variance, α over training, or loss curves — [ARFM](https://arxiv.org/html/2509.04063).
  - ARFM is reported insensitive to λ, and stable for bisection iterations M ≥ 10 (Fig. 4) — [ARFM](https://arxiv.org/html/2509.04063).
- **Overfitting of reweighting.** ReinboT attributes RWR's parity with BC on the real robot to "overfitting of RWR to the training data and its reliance on reweighting the data, which may lead to optimization problems when the data distribution is uneven or the amount of data is insufficient" — [ReinboT](https://arxiv.org/html/2505.07395).
- **Why some authors avoid critics.** Hi-ORS argues that chunk-level Q-values overestimate because action chunking "exponentially expands action space", and replaces the critic with outcome filtering — [Hi-ORS](https://arxiv.org/html/2510.26406). PAC relies on its BC term to prevent "overoptimism for unseen state-action pairs" — [PAC](https://arxiv.org/html/2402.05546).
- **Drift toward BC.** PAC's α ablation (App. D.1.1) reports that α can be moved from BC (α = 1) toward RL (α = 0) partway through training — [PAC](https://arxiv.org/html/2402.05546). The extraction I got quoted "~61–70% on CHEF:real" for these switch variants, but it did not make clear which variants those numbers belong to, so treat them as unverified.

### Inferences
- ReTVL's Table 7 is a cheap diagnostic the reader could copy. Log the mean weight on (a) teleop demos, (b) successful rollout chunks, and (c) chunks just before and after labelled failed grasps, drops or slips. A good critic should score high on (a) and (b) and near zero on the pre-failure chunks of (c).
- PACL's N-sweep points in the same direction as the reader's critic concern: the more a weak critic's argmax is exploited, the worse the policy. Exponential weights at β = 1 on ±3-clipped Â exploit the critic more than RA-BC's [0, 1] ramp does.

### Gaps
- No paper reports effective sample size, max-weight share, or per-batch weight entropy.
- No paper reports whether weighting slows the policy (cycle time). CO-RFT's abstract mentions a 22.3% cycle-time reduction, but that is abstract only.
- No paper reports a measured gradient-variance comparison between exponential and clipped weights.

## Q5. Critique: evaluation size, confounds, baseline fairness

### Takeaway
The real-robot evidence rests on 10–25 trials per task. Most gaps between strong alternatives (filtering, re-ranking, weighting) are within 1–3 trials. Only the large gaps against plain BC (SARM hard task, ReTVL average, PAC) are statistically solid. Several baselines are weaker than they could be: no tuned temperature, a duration filter instead of a value filter, and RWR with a linear return weight.

### Cited Findings
- SARM: 12 trials per task (3 colors × 4) — [SARM](https://arxiv.org/html/2509.25358v3).
- ReTVL: 20 per task — [ReTVL](https://arxiv.org/html/2606.24633).
- PACL: 25 per task — [PACL](https://arxiv.org/html/2609.29000).
- Hi-ORS: 10 per checkpoint — [Hi-ORS](https://arxiv.org/html/2510.26406).
- PAC: 400 per task, with Wilson intervals — [PAC](https://arxiv.org/html/2402.05546).
- ARFM: batch 16, a hand-built 13-component reward, and success-only UR5 data. Its RWR baseline temperature is not stated — [ARFM](https://arxiv.org/html/2509.04063).
- ReinboT: RWR multiplies by the raw discounted return (γ = 0.9) with no temperature. Its best number uses a vector return that RWR cannot use — [ReinboT](https://arxiv.org/html/2505.07395).
- α-PAC tunes α per dataset — [PAC](https://arxiv.org/html/2402.05546).
- PACL has no AWR baseline — [PACL](https://arxiv.org/html/2609.29000).
- ReTVL's value model needs sparse retry annotations on 30 trajectories per task. Its paper excludes wrong-object grasps, collisions and irreversible drops from retry labels — [ReTVL](https://arxiv.org/html/2606.24633).

### Inferences
- **Statistical strength, by my arithmetic.**
  - ReTVL vs standard BC (64/80 vs 33/80) is strong.
  - ReTVL vs the baseline value model (about 64/80 vs 50/80) is z ≈ 2.4.
  - SARM medium, RA-BC vs duration filter, is not significant (p ≈ 0.37).
  - PACL real-robot gaps over filtering and IDQL are 0–3 trials of 25.
- **Confounds in PAC and ARFM.** PAC's huge real-robot gap (7% → 70%) comes from data with 12% success. That regime is much worse than teleop-heavy data, so it overstates what to expect when most data are good. ARFM's LIBERO gains (+1.3 over RWR) are within the size one would expect from seed noise. I could not confirm this because no seed variance was reported.
- **The confound most relevant to the reader.** In SARM and ReTVL, the weighted and unweighted runs use the same data. The comparison against a well-tuned value-threshold filter on the same data is missing in both.

### Gaps
- None of the in-scope papers reports seed variance.
- ARFM, ReinboT and SRPO give no real-robot trial counts.

## Q6. Other papers considered and why dropped

### Takeaway
Most 2025–2026 real-robot VLA RL papers use Q-learning, PPO-style, residual or online RL, or pure filtering. They do not report an AWR-style weighted-imitation vs BC comparison.

### Cited Findings
- **CO-RFT** (2508.02219, Aug 2025): chunked TD offline RL for VLAs, using 30–60 demos. The abstract reports +57% success and −22.3% cycle time over supervised methods. There is no AWR/weighted-BC comparison in the abstract. Abstract only — [arXiv abs](https://arxiv.org/abs/2508.02219).
- **ConRFT** (2502.05450, Feb 2025): the offline stage combines BC with Q-learning on a consistency policy, and there is also online HIL fine-tuning. It reports 96.3% after 45–90 min of online training. No AWR in the abstract. Abstract only — [arXiv abs](https://arxiv.org/abs/2502.05450).
- **RLDG** (2412.09858, Dec 2024): compares RL-expert-generated data against human demos for SFT, with no weighting. Abstract only — [arXiv abs](https://arxiv.org/abs/2412.09858).
- **PLD** (2511.00091, ICLR 2026): residual RL probes, then distillation with standard SFT. Not weighted — [arXiv abs](https://arxiv.org/abs/2511.00091).
- **VLAC** (2509.15937, Sep 2025): a critic-in-the-loop online real-world RL framework. The abstract does not state weighted BC. Abstract only — [arXiv abs](https://arxiv.org/abs/2509.15937).
- **RL-100** (2510.14830): a clipped PPO surrogate inside the diffusion denoising steps, with iterative offline RL gated by off-policy evaluation. Not AWR. Search snippets only — [arXiv](https://arxiv.org/abs/2510.14830).
- **Self-Improving Embodied Foundation Models** (2509.15155, NeurIPS 2025): on-policy REINFORCE on Monte Carlo returns of a steps-to-go-difference reward, with no data reuse. This is online, not offline weighting. Numbers were seen only in third-party summaries, so none are reported — [arXiv abs](https://arxiv.org/abs/2509.15155).
- **HELP** (2607.09776, Jul 2026): rollout segments are classified by a critic (VLAC-CUT) and **filtered**, not weighted. Abstract only — [arXiv abs](https://arxiv.org/abs/2607.09776).
- **HIL-UMI** (2609.20659, 2026-09-17): advantage-**conditioned** BC with a progress-based advantage estimator, compared with SFT. Abstract only. It is relevant as a conditioning comparator but was not read — [arXiv abs](https://arxiv.org/abs/2609.20659).
- **Max-Q Selective Imitation** (2608.15088, Aug 2026): a hard winner-take-all max-Q imitation with an MC Q-chunk critic on ACT (USB insertion). It compares against HIL-SERL, EXPO and E2HiL, not AWR. Abstract only — [arXiv abs](https://arxiv.org/abs/2608.15088).
- **Intrinsic Robot Rewarding** (2609.17115): reward mechanism only; policy improvement is future work — [arXiv abs](https://arxiv.org/abs/2609.17115).

### Inferences
- If one more comparator is wanted, HIL-UMI (advantage-conditioned BC on a VLA, Sep 2026) is the most relevant unread candidate.

### Gaps
- The following surfaced in searches but were not triaged beyond their titles:
  - CounterAlign (2608.21740)
  - WCM, a world critic model (2607.29613)
  - "Failing Forward" AFIL (2605.08434), which uses failures as negative guidance for diffusion and flow VLAs
  - SARM2 (2606.10305)
  - RARM (2606.22027)
- DPPO (2409.00588) has diffusion-RWR/AWR baselines, but it is online and those baselines are sim-only. It was not read.
