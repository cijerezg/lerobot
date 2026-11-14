# AWR for flow-matching VLAs: what published results support

- No published paper runs your recipe: exponential weights on a per-batch standardized one-step TD advantage from a V(s)-only critic, on a real-robot flow VLA. Neither paper that puts exponential weights on a real flow VLA with stated details (Cao, Mei) reports plain BC on the same data.
- Where same-data comparisons exist, weighting helped a lot only when the data held many failures and the value signal had been checked: +39 pp over BC ([ReTVL T.2][retvl], bounded weights), 1/12 to 10/12 ([SARM T.2][sarm]), +0.24 success over an advantage filter with exp(clip(z, ±2)) weights ([Cao T.II][cao], 80 trials per arm). On mostly good data the gains are about −2 to +4 pp ([Zhang T.1][zhang], [ARFM T.1][arfm], [PAC T.1][pac]).
- Simulation studies rank AWR as the weakest extraction method ([Park T.1][park], [FQL Fig.5][fql], [Dong Fig.6][dong]), but they tested raw-advantage AWR against methods that need Q(s, a), which your critic does not provide.
- Swapping the value signal under a fixed weighting rule moved real-robot success by 17 pp ([ReTVL T.2][retvl]), 4-5 of 12 trials ([SARM T.2][sarm]) and roughly 0.4 on one Cao task ([Cao Fig.5][cao]), more than the 0.24 gap between Cao's weighting and filtering rules.
- Your beta = 1 on standardized advantages is Cao's setting. Your clip of ±3 (max/min weight ratio about 403) is wider than every real-robot recipe that states its range except Mei's (about 560), and no paper pools standardization statistics across tasks.

## The mechanism, without the framing

My synthesis, not a claim from any single paper. With a V(s) critic, AWR can only change how much each logged chunk counts; it cannot produce an action that is not in the data. Its best case is the data's own best behaviour at each state, so the gain is bounded by how much action quality varies across similar states in your data and by how well the critic ranks that variation. Robot states are essentially unique, so there is no per-state choice between actions: the weight works only through generalization across nearby states, and a large network can still fit every row (Park's drift-to-BC argument, [Park §5.4][park]). In practice this makes it soft data curation, which is why it behaves like filtering in most papers, why failure-rich data gives the big numbers, and why the critic's ranking quality matters more than beta or clip.

## 1. Gain over BC is large only with many failures and a checked critic

| Paper | Setting | Baseline | Weighted result | Delta | n trials | Source |
|---|---|---|---|---|---|---|
| Cao 2026 | Real. π0.5, bimanual, 4 tasks; ~40 h demos + 200 rollouts/task; exp(clip(z, ±2)) | SFT on demos, no extra updates: 0.11 success | 0.74 | +0.63 | 20/task, 80/arm | [T.II][cao] |
| Cao 2026 | Real, same | Batch-median advantage filter, same data and steps: 0.50 | 0.74 | +0.24 | 20/task | [T.II][cao] |
| Cao 2026 | Real, same | HG-DAgger weights (human 1, rollouts 0): 0.45 | 0.74 | +0.29 | 20/task | [T.II][cao] |
| ReTVL 2026 | Real. π0-style flow VLA, Piper, 4 tasks; 200 traj/task with failures and retries; ramp weights in [0, 1] | Plain BC, same data: 41% | 80% | +39 pp | 20/task | [T.2][retvl] |
| ReTVL 2026 | Real, same | Same rule, Recap-style progress-value model: 63% | 80% | +17 pp | 20/task | [T.2][retvl] |
| SARM 2025 | Real. π0 LoRA, bimanual shirt folding, 200 h mixed-quality demos; ramp weights | BC on all data: 1/12, 0/12 | 10/12, 8/12 | +9, +8 trials | 12/task | [T.2][sarm] |
| SARM 2025 | Real, same | Duration filter (episodes < 2 min): 7/12, 0/12 | 10/12, 8/12 | +3 (p ≈ 0.37), +8 | 12/task | [T.2][sarm] |
| Mei 2026 | Sim. π0.5, BEHAVIOR-1K, 200 demos + 100 rollouts/task; separate critic, exp weights, beta 1 ("IDQL", K = 1) | BC, same data: 31.0% | 39.0% | +8.0 pp | 100/task | [T.1, T.6][mei] |
| Mei 2026 | Sim, same; own weights min(20, exp(A/0.3)), K = 1 | BC: 31.0% | 34.6% | +3.6 pp | 100/task | [T.6][mei] |
| Mei 2026 | Real. π0, 5 bimanual tasks; separate-critic exp weights | Filtered BC: 24.6% (no BC row) | 32.6% | +8.0 pp | not stated | [T.2][mei] |
| Fang 2026 | Real. π0.5, 3 bimanual tasks, 200 demos + 300 rollouts; failures 0, successes in [0, 2] | SFT, no rollout data: 36/44/12% | 60/78/22% | +24/+34/+10 pp | 50/checkpoint | [Fig.3, §4.2][fang] |
| Zhang 2026 | Real. Being-H0.5 flow, 12 tasks, 3 platforms; reward-free score, clip [0.25, 4] | SFT: 60.8 (one model per source) / 50.0 (one model, 5 sources) | 67.1 / 63.8 | +6.3 / +13.8 pp | 20/task | [T.3-4][zhang] |
| Zhang 2026 | Sim. LIBERO / RoboCasa, clean data | SFT: 98.3 / 54.2 | 97.8 / 55.6 | −0.5 / +1.4 pp | 2000 / 1200 episodes | [T.1][zhang] |
| Zhang 2026 | Sim. Same, all three synthetic corruptions | SFT: 85.4 / 36.4 | 91.0 / 45.8 | +5.6 / +9.4 pp | same | [Fig.2][zhang] |
| ARFM 2025 | Sim. π0, LIBERO, success-only demos | SFT: 88.1 | RWR 90.8, ARFM 92.1 | +2.7, +4.0 | not captured | [T.1][arfm] |
| PAC 2024 | Real. Sawyer, 988M actor-critic (not a VLA), data with 12% success; weights policy samples | BC (alpha = 1): 7.1% | 69.8% | +62.7 pp | 400/task | [T.2][pac] |
| ReinboT 2025 | Sim. CALVIN mixed quality, GPT-style (not flow); RWR with raw return weight | BC: 1.36 avg length | sparse reward 1.38; dense 1.82 | +0.02; +0.46 | not captured | [T.1][reinbot] |
| Park 2024 | Sim. gc-antmaze-large, 0.1M transitions, Gaussian | AWR alpha = 0 (= BC): ≈0-1 | alpha = 3: ≈22-26 | ≈+22 | 8 seeds | [Fig.2, read from plot][park] |
| FQL 2025 | Sim. D4RL adroit, 12 tasks; flow AWR | Gaussian BC: 48 | 48 | 0 | 8 seeds | [T.2][fql] |

The weighting effect alone is isolated only where weighted and unweighted runs share data and steps: SARM and ReTVL (bounded ramp weights), Mei's separate-critic baseline, and Cao's arms against each other. Cao's +0.63 over SFT mixes 4000 extra updates at batch 2048, the rollout and takeover data, and the weighting ([Cao §IV-C][cao]). On Printer, Cao's weighted arm with near-chance Q−V advantages still reached about 0.5 success from 0.00, close to the advantage-free DAgger arm's 0.45 ([Cao Fig.5 read by eye, T.II][cao]), so extra updates and data explain most of that task's gain. Plain BC on all data, failures included, also beat the starting policy in PACL (92/84/68 vs 84/72/64, real, 25 trials; [PACL T.I][pacl]).

Data quality sets the size. PAC gains 10× on real data with 12% success and nothing on near-expert RC:Tower data (BC+Q 71.3 vs PAC 69.3, sim; [PAC T.1-2][pac]). Zhang gains −0.5 to +1.4 pp on clean data and up to +9.4 pp on corrupted data ([T.1, Fig.2][zhang]). With a sparse outcome reward, RWR equals BC; only a dense progress reward helps ([ReinboT T.1][reinbot]). Park's AWR reaches 0.9-1.0× the best extractor on noisy-expert data and 0.5× on exploratory data ([Park T.1][park]).

Dataset size and task count are barely studied. At the smallest scale filtered IL beats value-based RL (60 vs 54, sim; [Dong Fig.5][dong]), and AWR overfits at alpha = 3 on 0.1M transitions ([Park Fig.4][park]). ReinboT attributes RWR ≈ BC on its real UR5 to overfitting with too little data (no numbers; [ReinboT][reinbot]). Cao and Fang train one policy per task ([Cao §V-A][cao], [Fang §5][fang]). Zhang's one-model-for-5-sources run gained more than its per-source runs (+13.8 vs +6.3 pp; by the notes' arithmetic only the first is clearly outside 20-trial noise), but its score is reward-free ([Zhang T.3-4][zhang]). No paper weights one actor across tasks with different value scales.

## 2. Beta 1 on a z-score is standard; clip ±3 and pooled batch statistics are not

| Paper | Setting | Weight | Temperature | Max/min ratio | Normalization | Advantage | Source |
|---|---|---|---|---|---|---|---|
| Yours | Real (planned) | exp(Â/beta)/mean(w), then (1−lam) + lam·w | beta 1 on batch z | e^6 ≈ 403 | z over 128 pooled rows; unit mean | 1-step TD (1 s), HL-Gauss V(s) | - |
| Cao | Real | exp(clip(beta·z, ±2)); loss Σwℓ/Σw | beta 1 on z | e^4 ≈ 55 | z per source × 8 value bins, dataset-level, floor 0.1 × source std | 10-chunk (10 s) TD, IQL tau 0.8, sparse reward | [§III-IV, Eq.3-5][cao] |
| Mei | Sim + real | min(20, exp(A/0.3)) | tau 0.3 on A in [−1, 1] | ≈560 | none | stage label − own estimate, no TD | [Eq.15-16, T.7-8][mei] |
| Mei's IDQL baseline | Sim + real | exp weight | beta 1 | not stated | not stated | separate ~500M critic, TD | [T.11][mei] |
| Fang | Real | 1 + tanh(gated sum of two TD residuals); failures 0 | none | [0, 2] plus zeros | unit mean over non-SFT rows | two 1-step TD heads | [Eq.5-8][fang] |
| Zhang | Sim + real | 1 + alpha(clip_[0.25,4](exp(T/beta)) − 1) | beta adaptive 0.5-3 | 16 | Σw per 128-row minibatch | reward-free score | [Eq.21, 27; T.6][zhang] |
| SARM, ReTVL | Real | clip((r − (mu − 2 sigma))/(4 sigma), 0, 1); 1 above kappa; 0 if r < 0 | none | [0, 1] plus zeros | Σw; running (SARM) or dataset (ReTVL) mu, sigma | progress delta over ~0.83 s (SARM); value delta (ReTVL) | [SARM][sarm], [ReTVL Eq.11][retvl] |
| ARFM | Sim + real | batch softmax exp(alpha·R*) | alpha re-solved per batch | varies | alpha* = √x*/sigma_R per batch | leave-one-out return, no critic | [ARFM][arfm] |
| Park 2024 | Sim | exp(alpha(Q − V)) | 4-value grids from 0 to 100, tuned per cell | unbounded | none stated | IQL/SARSA Q − V | [T.3, App.C][park] |
| FQL (flow AWR) | Sim | exp(alpha(Q − V))·L_flow | alpha 0.03-10 per environment | none stated | none stated | Q − V | [Eq.10, T.6][fql] |
| CFGRL (AWR arm) | Sim | exp(A/beta) | 1/beta 1-30; saturates near 10 | - | none | IQL Q − V | [Sec.5, Fig.3][cfgrl] |
| QIPO | Sim | softmax over 16 policy samples + dataset action of beta·Q | beta 3 | per-state softmax | per state | Q(s, a) | [Alg.4, App.E][qipo] |

Beta 1 on a standardized advantage is exactly Cao's setting, and unit-mean normalization is what Cao, Zhang, SARM and ReTVL do (Σwℓ/Σw). Your lam mix is Zhang's form w = 1 + alpha(w' − 1); Zhang's alpha = 0 control puts the reweighting itself at +0.3/+1.3 pp in simulation ([Zhang Eq.21, T.5][zhang]). Clip ±3 is wider than Cao's ±2 and far wider than the bounded schemes. In simulation, the best temperature differs by environment in every paper that tuned it: FQL's flow AWR used alpha 0.03-10 ([FQL T.6][fql]), CFGRL's AWR used 1/beta 1-30 ([CFGRL T.5, T.7][cfgrl]).

No paper compares per-batch with dataset-level statistics. Cao only notes that both are common ([Cao §III-B][cao]). Cao is the only paper that groups: z = (u − mu_g)/max(sigma_g, sigma_min) with g = source × value bin, statistics computed once, bins under 32 samples falling back to source level ([Cao Eq.4, §IV-B][cao]). Its motive is that with sparse terminal reward, raw advantage levels vary with progress. Value binning cut the sub-task share of advantage variance (eta²) from 0.092 to 0.026 ([Cao T.I][cao]). On the robot, Global or Temporal grouping did worse "in most cases" on 2 tasks, figure only, and no ungrouped arm was run ([Cao Fig.5][cao]). Cao trains per-task policies, so it never meets cross-task scales. Zhang's clip bounds each source's reweighted share to [1/16, 16] ([Zhang Eq.29-30][zhang]). QIPO shows the exact target only needs a normalizer that does not depend on x_t or t: a per-state or batch-mean normalizer both qualify ([QIPO Eq.5.3, Alg.1][qipo]).

Algebra for your scheme (my inference, not from any paper):

```
Â = clip((A - mean_B(A)) / std_B(A), -3, 3)
w = exp(Â / beta) / mean_B(exp(Â / beta))
without the clip:  w = exp(A / T) / mean_B(exp(A / T)),   T = beta * std_B(A)    # mean_B(A) cancels
```

The temperature on raw A is beta·std_B(A). It moves with batch composition and, if the critic keeps training, with the critic's advantage spread. The mean matters only through where the clip bites. Standardization gives every batch the same weight spread however much of A is critic noise. Cao's sigma floor is the only guard in the papers against inflating a small spread. When tasks are pooled, the task with the widest raw A spread fills both tails, and a task with a narrow spread gets near-uniform weights. Any trend of A with progress, which Cao removes with value bins, stays in. If Â were N(0, 1) and unclipped, the effective sample fraction would be exp(−1/beta²): 0.37 at beta = 1 (≈47 of 128 rows) and 0.78 at beta = 2 (my arithmetic; no paper reports it). Under ARFM's Gaussian model, the gradient-variance factor e^(2y²) − e^(y²), with y the temperature on standardized A, is 4.67 at y = 1 and 0.37 at y = 0.5 ([ARFM][arfm]; rough because your clip bounds the tails).

On the estimator, Cao's offline tests favor a TD residual along the logged continuation over Q − V and Monte Carlo V. The means were R_int 0.72 vs ≈0.46-0.56 and R_prog 0.90 vs ≈0.50-0.56, with chance at 0.5 ([Cao Fig.4][cao]). Your A is Cao's Eq. 3 with n = 1, and n = 1 was never tested. The notes infer that with a sparse terminal reward, a 1 s advantage is mostly the critic's own 1 s value change ([Cao Eq.3][cao]). Fang uses 1-step TD residuals ([Fang §3.3][fang]). SARM's 0.83 s progress delta is the published horizon closest to yours that has a real-robot gain ([SARM][sarm]).

## 3. Failure modes are measured in simulation and argued for robots

| Failure mode | What was measured | Setting | Source |
|---|---|---|---|
| A few rows dominate the batch | per-sample AWR losses up to ~1000 in one batch; "most samples are ignored" | Sim | [CFGRL Fig.2][cfgrl] |
| Overfitting from few effective samples | alpha = 3 on 0.1M: held-out policy loss ~50+ vs training ~0-10; alpha = 0 curves overlap | Sim, 8 seeds | [Park Fig.4][park] |
| Heavy tails without a clip | weights > 10, periodic loss oscillation; −2.3/−6.5 pp, below SFT | Sim | [Zhang T.5, §5.6][zhang] |
| Drift back to BC | argued: all weights positive, so an expressive net on unique states can fit every action | Argument | [Park §5.4][park] |
| Weighting ≈ BC | RWR "on par" with BC on UR5, attributed to overfitting | Real, no numbers | [ReinboT][reinbot] |
| Decline over rounds | AWR 47 → 24 over 10-20 collect-and-retrain rounds; beta unreported | Sim, 3 seeds × 100 | [Dong Fig.6][dong] |
| Slower successful episodes | every fine-tuned method slower than SFT on successes, e.g. Snack Bag ~1240 vs ~1610 frames (std ±150-400) | Real, 50 trials | [Fang Fig.3][fang] |
| Critic error, reward model swapped | 10/12 → 6/12 and 8/12 → 3/12 at a fixed rule | Real, 12 | [SARM T.2][sarm] |
| Critic error, value model swapped | 80% → 63% at a fixed rule | Real, 20/task | [ReTVL T.2][retvl] |
| Critic error, weak advantages | Q − V advantages: PhoneH success ~0.2-0.3 vs 0.70 (by eye); Printer barely affected | Real, 2 tasks | [Cao Fig.5][cao] |
| Value head trained through the weighted loss | Radio stage 3: 42% with the weight applied to the value coordinate vs 51% without | Sim, 1 task | [Mei T.4, App.A.4][mei] |
| Credit leak across human takeovers | argued: the episode outcome credits the pre-takeover segment | Argument | [Fang §3.4][fang] |
| Biased weights from noisy labels | E[exp((Y − V)/tau)] is not exp((Q − V)/tau) | Theory | [Mei Rem.A.6][mei] |

Park's drift argument cuts both ways. Low alpha is BC by definition, and at any alpha a large network on unique states can end up fitting all actions, so weighting mostly changes which rows dominate early training ([Park §5.3-5.4][park]). No paper tested this on a pretrained VLA.

| Paper | Diagnostic actually used | What it checks |
|---|---|---|
| Cao | R_int: takeover window vs the policy window before it; R_prog: stalled vs advancing segment, embedding-matched; chance 0.5 | whether A ranks actions before any actor training ([Fig.4][cao]) |
| Cao | eta² = share of A variance explained by sub-task | whether A tracks progress level instead of action quality ([T.I, Eq.6][cao]) |
| Park | training vs held-out policy loss | overfitting from few effective samples ([Fig.4][park]) |
| CFGRL | per-sample loss spread within one batch | weight domination ([Fig.2][cfgrl]) |
| Zhang | weight histograms early vs late, mass at the clip, 1/Std(w), loss oscillation | tails and clip saturation ([Fig.10-11, §5.6][zhang]) |
| ReTVL | mean weight on success chunks and post-retry chunks; share of success chunks zeroed; share of bad chunks kept | weight by chunk category ([T.7][retvl]) |
| Fang | mean weight on successes, filter pass rate (19-27%), weight on takeover windows, success-episode length | weight level and speed ([App.G, Fig.3][fang]) |
| SARM | Spearman rho of reward vs rollout progress: −0.39 / 0.50 / 0.67 across reward models | reward-model quality ([SARM][sarm]) |

No paper reports effective sample size (Σw)²/Σw², top-k weight share or clip-hit fraction. Cao names effective sample size as something pipeline choices change but never measures it ([Cao §I][cao]).

## 4. With V(s) only, filtering and conditioning are the available alternatives

DDPG+BC, FQL, OPEX/TTT, best-of-N (SfBC, IDQL), PA-RL and V-GPS all score or differentiate actions that are not in the data, so they need Q(s, a) ([Park Eq.6-7, §6.4][park], [FQL Eq.9][fql], [Dong §4.1][dong], [PA-RL Eq.4.1-4.2][parl], [V-GPS Eq.2][vgps]). With V(s) and a TD advantage on logged chunks, you can run AWR, filtering, and conditioning on the sign of the TD advantage. The last is my inference: CFGRL itself uses IQL Q − V ([CFGRL Sec.5][cfgrl]).

| Alternative | With V(s) only? | Same-paper result vs weighting | Setting | Source |
|---|---|---|---|---|
| Advantage filter, batch median | Yes | filter 0.50 vs weight 0.74 success; filter below advantage-free DAgger on 2/4 tasks | Real, 20/task | [Cao T.II][cao] |
| Success-only filter | Yes, no critic | self-imitation 100/92/72 vs PACL 100/100/84 | Real, 25/task | [PACL T.I][pacl] |
| Success-only filter | Yes | filtered IL best at small scale (60 vs 54), plateaus at large (64 vs 75) | Sim | [Dong Fig.5][dong] |
| Success-only filter | Yes | filtered BC 31.6 vs BC 31.0 vs separate-critic exp weights 39.0 | Sim, 100/task | [Mei T.1][mei] |
| Outcome filter, rising threshold | Yes | +23.3 pp over BC claimed in text (online, with human corrections) | Real, 10/checkpoint | [Hi-ORS][hiors] |
| Critic-threshold filter | Yes | BC 45.7% → filtered 61.6% | Real, n not captured | [GR-RL][grrl] |
| Critic-threshold filter (Imit-Recap) | Yes | ~38/66/14 vs Fang's weighting 60/78/22; differs in 3 ways at once | Real, 50 | [Fang Fig.3][fang] |
| Optimality conditioning + guidance | Yes (sign of TD A) | beats AWR on 8/9 ExORL; OGBench 3 better, 1 worse (antmaze-large 20 vs 50) | Sim, 4 seeds | [CFGRL T.1-2][cfgrl] |
| Choice of conditioning label | Yes for source, outcome, advantage | source 85.6, outcome 88.0, Q − V 88.8, Q 91.2 | Sim, Square | [PACL Fig.6][pacl] |
| Best-of-64 under Q (IDQL) | No | after rounds 75 vs AWR 24; at the start 38 vs AWR 47 | Sim | [Dong Fig.6][dong] |
| Best-of-N (SfBC) | No | AWR higher in 6 of 16 settings, lower in 9 | Sim | [Park T.1][park] |
| Best-of-5, separate critic | No | K = 5 38.4 vs K = 1 39.0 | Sim | [Mei T.6][mei] |
| Inference steering (V-GPS) | No | Octo 0.24 → 0.44; no AWR baseline | Real, 6 × 20 | [V-GPS T.1][vgps] |
| DDPG+BC | No, needs grad_a Q | AWR median 0.88× DDPG+BC; DDPG+BC ≥ AWR in 15/16 | Sim, 8 seeds | [Park T.1][park] |
| FQL | No | FQL 44 vs flow AWR 16 (50 OGBench tasks); adroit 52 vs 48 | Sim | [FQL Fig.5, T.2][fql] |
| PA-RL | No | OpenVLA 40% → 50% (frozen, critic steering) → 70% (online) | Real, n not stated | [PA-RL Fig.1][parl] |

Re-ranking exploits a weak critic more as candidates grow: a one-step critic fell from 79.2 to 58.8 going from 4 to 16 candidates ([PACL T.II][pacl]), and IQL-scored V-GPS fell from 0.59 to 0.35 going from K = 10 to 100 ([V-GPS App.H][vgps]). Conditioning costs an extra actor input with 10% label dropout, two action-expert passes per denoising step, and it uses only the sign of A ([CFGRL Alg.1-2, Eq.15][cfgrl]). No paper compares weighting with conditioning on a VLA or a real robot. Cao left conditioning out on purpose ([Cao §IV-C][cao]).

## 5. Per-sample weighting of the flow loss is already the exact method

QIPO proves the per-sample weighted conditional flow loss has the same gradient as the loss whose minimizer is the exact reweighted flow. The conditions are that the weight depends only on the clean sample and the normalizer does not depend on x_t or t ([QIPO Thm 4.3, Lemma B.1][qipo]). Mei proves the same for exp((Q − V)/tau) on a flow VLA, up to a per-state constant ([Mei Thm A.5][mei]). Weighting shifts capacity across x_t without moving the minimizer ([QIPO Rem.4.4][qipo]).

```
QIPO Eq. 4.3:  L = E_{t,x0,x_t}[ w(x0)/E[w] * ||v(x_t,t) - u_t(x_t|x0)||^2 ]
Mei  Eq. 16:   L = E[ w_i * ||v_a - u_a||^2 + ||v_s - u_s||^2 ],   w_i = min(20, exp(A_i/0.3))
yours:         L = mean_B[ w_i * ||v(a_t, t | obs) - u(a_t | a)||^2 ],   w_i fixed per (s, chunk) for every t
```

The practical issues are elsewhere. Exponentiated single-sample labels are biased: E[exp((Y − V)/tau) | c, a] = exp(−V/tau)(1 + Q(exp(1/tau) − 1)), which is monotone in Q but is not the Boltzmann factor ([Mei Rem.A.6][mei]). My inference: your one-step TD advantage is also a single-sample estimate. By Jensen, rows whose next-second outcome is random (a grasp that sometimes slips) get more weight than rows with the same mean advantage and less noise.

Apply the weight to the action flow loss only. A value or progress coordinate trained through the same weighted loss hid its own errors in Mei ([T.4, App.A.4][mei]). Your critic has its own encoder and loss, so this loop does not arise. If the actor has other loss terms, no paper says whether to weight them: Cao does not say, and Mei has none ([Cao §III-C][cao], [Mei Eq.16][mei]). The weight is one scalar per chunk, broadcast over the action coordinates ([Mei Eq.15][mei]).

What QIPO adds beyond AWR needs Q(s, a) for actions outside the data: a per-state softmax over 16 generated actions, training on those actions, and sharpening across rounds as pi_(l+1) ∝ mu·exp((l+1)·beta·Q) ([QIPO Alg.4, Eq.5.4][qipo]). The compounding applies to you only if you iterate collect-and-weight rounds (my inference). On D4RL, QIPO roughly ties QGPO and IDQL (locomotion 86.2-86.3 vs 86.6 and 82.1), reporting the best evaluation over training and no BC column ([QIPO T.2, App.E.1][qipo]).

Guidance at a scale other than 1 does not sample the product distribution ([QIPO Lemma 4.10, Fig.3][qipo]), yet CFGRL gains from w > 1 ([CFGRL Fig.3, Fig.6][cfgrl]). With CFGRL's binary label 1[A ≥ 0], f^w = f, so the stated target does not change with w; the gains at w > 1 must then come from the guidance approximation itself (my inference; the paper does not discuss it). A flow policy did not rescue AWR in FQL: flow AWR scored below Gaussian AWR in 8 of 10 OGBench categories and on D4RL antmaze (44 vs 57). That comparison is confounded, because the Gaussian baseline's code clips weights at 100 and it used smaller nets ([FQL T.2][fql], [iql.py][fqlcode]).

## Where the papers agree, conflict, and why

The papers agree on bounding the weights and keeping a uniform anchor. Every recipe with a real-robot gain bounds the weight ratio: Cao e^4 ≈ 55, Mei's cap of 20 (≈560), Zhang 16, Fang [0, 2], SARM and ReTVL [0, 1]. The variants that failed in simulation are the unbounded ones ([Zhang T.5][zhang], [CFGRL Fig.2][cfgrl], [Park Fig.4][park]). Fang keeps demos and takeovers at weight 1, and Zhang mixes toward uniform with alpha, as your lam does ([Fang Eq.8][fang], [Zhang Eq.21][zhang]). They also agree that the value signal matters as much as the rule ([SARM T.2][sarm], [ReTVL T.2][retvl], [Cao Fig.5][cao]). PACL adds that the critic needs failures: without failed rollouts, PACL scored 77.6/83.2, below the base policy's 78.8/84.0, and with them 93.2/96.0 (sim; [PACL T.III][pacl]).

The main conflict is that simulation ranks AWR last ([Park T.1][park], [FQL Fig.5][fql], [Dong Fig.6][dong]) while real flow-VLA papers report large wins ([Cao T.II][cao], [ReTVL T.2][retvl], [SARM T.2][sarm], [Mei T.1-2][mei]). Four differences explain most of it:

1. Baselines. Simulation compares AWR with Q-gradient and re-ranking methods; robot papers compare it with BC, filters, and SFT that never saw the new data.
2. The AWR tested. Simulation used raw Q − V with no normalization and 3-4 temperatures including 0. Park used a fixed-std Gaussian, and Dong never reports beta ([Park App.C][park], [Dong App.B][dong]).
3. Data. Park's AWR deficit is largest on exploratory data (0.5×) and smallest on noisy-expert data (0.9-1.0×), and teleop plus rollouts is closer to the latter ([Park T.1][park]).
4. Evaluation. Simulation uses 3-8 seeds; robot papers use 12-50 trials per task, one run, no seeds.

Dong's negative result is also about repeated rounds: at the offline start, AWR led.

Weighting vs filtering also conflicts. Cao's weighting beats its filter by 0.24 success ([Cao T.II][cao]). Elsewhere filters come close: PACL's success-only filter is 0-12 pp behind the best real method, which is 0-3 trials of 25 ([PACL T.I][pacl]). SARM's duration filter was not separable from RA-BC on the easier task ([SARM T.2][sarm]). In Dong, filtered IL is best at small scale ([Dong Fig.5][dong]). Cao's filter is untuned (q = 0.5 of each batch). Because every source is z-scored to mean zero, it also drops about half the demo rows in each batch (my inference from [Cao §IV-B-C][cao]). The outcome filters in the other papers keep every demo.

| Paper | Missing control or confound | Source |
|---|---|---|
| Cao | no uniform-weight BC on the same mixed data; SFT reference got no extra data or updates; no sweeps of beta, clip, q; no CIs | [§IV-C, T.II][cao] |
| Fang | no success-filtered BC with the weight fixed at 1, so the critic's share of the gain is unknown; 92/88/38 are best-checkpoint picks | [Alg.1, Fig.3][fang] |
| Zhang | SFT is the only baseline; clean-data gains are the size of the Table 1 vs Table 5 discrepancies (0.4-0.8 pp) | [T.1, T.5][zhang] |
| Mei | main table compares K = 5 with single-sample baselines; real table omits BC; no tau or clip sweep; FQL baseline used a frozen VLM | [T.1-2, T.6][mei] |
| SARM, ReTVL | no value-threshold filter on the same data | [SARM T.2][sarm], [ReTVL T.2][retvl] |
| Dong | AWR beta, normalization and clipping never reported; no AWR on the robot; robot trial counts not stated | [App.B, Fig.10][dong] |
| Park, FQL | raw-advantage AWR; flow AWR got a smaller grid, an unspecified V, and no pixel or online runs | [Park App.C][park], [FQL App.E.2][fql] |
| PAC | 12%-success data overstates gains for teleop-heavy data; weights policy samples, not dataset actions | [T.2][pac] |
| ARFM | success-only data, hand-built 13-term reward, RWR temperature unstated | [T.8][arfm] |

## Descriptions that did not match

| Paper | What the description said | What the paper shows |
|---|---|---|
| Cao 2026 | beats hard filtering and "binary weighting" on 4 real tasks | The second baseline is source-based HG-DAgger (human data 1, autonomous rollouts 0, no advantage). The advantage-based binary scheme is the filter. There is no plain BC on the same data. "Beats" means point estimates at 20 trials per task with no CIs ([§IV-C, T.II][cao]). |
| Dong 2026 | implicit extraction beats AWR | True after 10-20 collect-and-retrain rounds in simulation (75 vs 24). At the offline start AWR was ahead (47 vs 38). AWR beta is never reported, and AWR was not run on the robot. The arXiv version is from 2025; ICLR is 2026 ([Fig.6, App.B][dong]). |
| Fang 2026 | online RL with hierarchical advantage weighting | Rounds of 100 rollouts, then 6k steps of weighted flow-matching BC, with human takeover on about half the failures; no policy gradient. The "hierarchy" is two critic heads (success probability, steps to success) mixed by a per-state gate. Failed rollouts get weight 0. There is no flat exp-AWR baseline ([§3.2-3.5, App.E][fang]). |
| Zhang 2026 | an AWR comparison | There is no AWR baseline, only uniform SFT. It shares only the exp weight form; its score is reward-free and there is no value function ([§2, §5.1][zhang]). |
| Mei 2026 | weights the action flow loss and leaves the other losses unweighted | The loss has two terms of one flow: "The scalar weight w_{i,t} is broadcast across the action coordinates of the chunk; the potential velocity loss remains uniformly weighted." There is no language or other loss. "Potential-guided" means best-of-K selection by the generated potential, not gradient guidance. The weight is not normalized, and A = stage label − own estimate, with no TD ([Eq.15-16, Alg.1, Sec.3][mei]). |

## How the additional papers were found

- ARFM: on the candidate list; confirmed by searching "Balancing Signal and Variance Adaptive Offline RL Post-Training VLA Flow Models".
- SARM (RA-BC): a search for SRPO returned SARM2 (2606.10305), which led to a search for SARM's RA-BC.
- ReTVL: returned by the ReinboT search.
- PACL: found by searching "advantage weighted diffusion policy real robot offline reinforcement learning mixed-quality demonstrations 2025".
- ReinboT: on the candidate list; ARFM uses it as a baseline.
- PAC: from prior knowledge of offline actor-critic at scale; verified in the full text.
- Hi-ORS: on the candidate list, plus search.
- GR-RL: on the candidate list.
- SRPO: from the first broad search, "advantage-weighted regression vision-language-action real robot offline RL". It gives no AWR loss, beta or clip, so it adds nothing here ([SRPO][srpo]).

## What to expect for my setup

1. If most rows are teleop demos, expect a few pp at most over lam = 0 (−2 to +4 in these papers) ([Zhang T.1][zhang], [ARFM T.1][arfm], [PAC T.1][pac]); large gains (+39 pp over BC in ReTVL, +0.24 over a filter in Cao) appeared only where failed rollouts were a large share and the value signal had been checked ([Cao T.II][cao], [ReTVL T.2][retvl], [SARM T.2][sarm]).
2. Run lam = 0 on the same data and steps as the control: no exp-weighted real flow-VLA paper has it, and on Cao's Printer task extra updates and data without any advantage gave most of the gain ([Cao Fig.5, T.II][cao]).
3. Score the critic before any actor run with Cao-style pairwise orderings and ReTVL-style mean weight on demos, successful rollouts and chunks before failed grasps, drops and slips; value-model swaps moved real success by 17 pp and 4-5 of 12 trials ([Cao Fig.4][cao], [ReTVL T.7, T.2][retvl], [SARM T.2][sarm]).
4. Your weights are wider than Cao's (ratio 403 vs 55), and per-batch z-scores pooled over tasks have no published counterpart; per-task dataset-level statistics (per value bin, as in Cao) with clip ±2, plus logged effective sample size, clip-hit fraction and mean weight per task and source, cover what the papers flag but never measured ([Cao Eq.4][cao]).
5. Two arms at 20 trials each resolve only about 30 pp and about 80 per arm resolve about 15 pp (my arithmetic); also track successful-episode length ([Fang Fig.3][fang]), and read a null result as consistent with Park's drift-back-to-BC argument ([Park §5.4][park]).

## Papers worth your reading time

- Cao et al. 2026 ([2609.28161][cao]): the closest recipe to yours (π0.5 flow VLA, real robot, TD advantage, exp(clip(z)) with Σw normalization) and the only paper that tests offline whether the advantage ranks actions before any robot time.
- Park et al. 2024 ([2406.09329][park]), §5.4 and Fig. 4: the clearest account of how exp weights fail (few effective rows, drift back to BC), and a train vs held-out loss check you can run; skip its DDPG+BC advice, which needs Q(s, a).
- ReTVL, Qin et al. 2026 ([2606.24633][retvl]): a real flow VLA on data with failures and retries like yours, a same-data BC baseline, and a weight-by-chunk-type check in Table 7 that you can copy.

[cao]: https://arxiv.org/pdf/2609.28161
[dong]: https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf
[fang]: https://arxiv.org/pdf/2606.17043
[zhang]: https://arxiv.org/pdf/2603.16542
[mei]: https://arxiv.org/html/2606.04968v1
[qipo]: https://arxiv.org/pdf/2503.04975v1
[cfgrl]: https://arxiv.org/pdf/2505.23458v1
[park]: https://arxiv.org/pdf/2406.09329v2
[fql]: https://arxiv.org/pdf/2502.02538v2
[fqlcode]: https://github.com/seohongpark/fql/blob/master/agents/iql.py
[parl]: https://arxiv.org/pdf/2412.06685
[vgps]: https://arxiv.org/html/2410.13816v2
[arfm]: https://arxiv.org/html/2509.04063
[sarm]: https://arxiv.org/html/2509.25358v3
[retvl]: https://arxiv.org/html/2606.24633
[pacl]: https://arxiv.org/html/2609.29000
[reinbot]: https://arxiv.org/html/2505.07395
[pac]: https://arxiv.org/html/2402.05546
[hiors]: https://arxiv.org/html/2510.26406
[grrl]: https://arxiv.org/html/2512.01801
[srpo]: https://arxiv.org/html/2511.15605
