# AWR for flow-matching VLAs: the picture (v2)

Companion to [`report_v1.md`](report_v1.md) (v1, full tables). This version states the mechanism, where the papers agree and conflict, and what the evidence does not cover. Numbers are repeated only where they carry the argument. Per-paper notes with page-level cites: [`papers/`](papers/).

- With a V(s) critic and logged actions, AWR is soft data curation: it decides how much each logged chunk counts and cannot produce an action that is not in the data. Everything the papers measure follows from two quantities: how much action quality varies in the data, and how well the critic ranks that variation.
- Real-robot gains are large only where failures are a big share of the data and the critic was checked before training: +39 pp over BC ([ReTVL T.2][retvl]), 1/12 to 10/12 ([SARM T.2][sarm]), 0.50 to 0.74 vs a filter ([Cao T.II][cao]). On clean demos the effect is −2 to +4 pp ([Zhang T.1][zhang], [ARFM T.1][arfm], [FQL T.2][fql]).
- Swapping the value model under a fixed rule moved real success by 17 pp ([ReTVL T.2][retvl]) and 4-5 of 12 trials ([SARM T.2][sarm]); changing the rule (weight vs filter) moved it by 0.24 in one paper ([Cao T.II][cao]) and by nothing in others ([SARM T.2][sarm], [PACL T.I][pacl]). The critic is first-order, the rule second-order.
- Every real-robot recipe that won bounds the weight ratio at 55 or less and keeps most rows near weight 1; your clip of 3 allows 403 and drops the effective sample size to 40 % of rows in every batch, signal or not.
- No paper runs your recipe (per-batch z-score of a 1 s TD advantage from V(s), exp, clip ±3, pooled across tasks). The closest pieces are Cao (exp on a z-score, clip ±2), Fang (1-step time-to-go residual), and SARM and ReTVL (chunk-length value delta, bounded ramp weights).

## 1. What advantage weighting does with a V(s) critic

The population target of every exp-weighted BC loss, including the weighted flow loss, is the tilted behaviour distribution ([QIPO Thm 4.3, Eq. 5.3][qipo], [Mei Thm A.5][mei], [Park Eq. 5][park]):

```
pi*(a | s)  ∝  mu(a | s) · exp(A(s, a) / T)          # mu = the data's behaviour policy
```

Three things follow, and they explain the whole literature.

**The ceiling is the data.** Weighting re-allocates gradient among logged chunks. The best it can do at a state is the best logged behaviour at states like it. Park shows the consequence directly: AWR actions stay inside the convex hull of dataset actions while a Q-gradient method spreads out ([Park §5.4, Fig. 3][park]). Robot states are unique, so the choice between actions happens only through generalization across nearby states, and a large network can still fit every row given enough steps ([Park §5.4][park]). In a fine-tuned VLA on a fixed offline set this is the strongest form of Park's argument, and nobody has tested it there.

**The gain is quality variance times ranking accuracy.** If all chunks are about equally good, there is nothing to re-weight. If failures are present and the critic ranks them below successes, weighting removes them, which is what a filter does. This single line orders every real result:

| Paper | Failure share in the data | Critic checked before training | Weight bound | Gain over same-data BC |
|---|---|---|---|---|
| PAC, real Sawyer | 88 % of episodes fail | 400 trials per arm; distributional critic | exp(Q/eta) on policy samples + BC term | 7.1 % → 69.8 % ([T.2][pac]) |
| ReTVL, real Piper | failures and retries in 200 traj/task | Table 7 weight-by-chunk-type; retry preference loss | [0, 1] ramp, negatives 0 | 41 % → 80 % ([T.2][retvl]) |
| SARM, real YAM | 200 h mixed-quality demos | Spearman 0.67 on rollouts vs 0.50 / −0.39 for alternatives | [0, 1] ramp, negatives 0 | 1/12 → 10/12, 0/12 → 8/12 ([T.2][sarm]) |
| Cao, real bimanual | 200 rollouts/task with takeovers on ~40 h demos | R_int 0.72, R_prog 0.90, chance 0.5 | e^4 ≈ 55 | no same-data BC; 0.50 filter → 0.74 ([T.II][cao]) |
| Fang, real ARX | 300 rollouts on 200 demos | none; 500-step warmup at weight 1 | [0, 2], failures 0 | no same-data BC; SFT 36/44/12 → 60/78/22 ([Fig. 3][fang]) |
| Mei, sim π0.5 | 100 rollouts on 200 demos | none | 560 (Mei), unstated (IDQL arm) | +3.6 pp (Mei's weights), +8.0 pp (IDQL exp, beta 1) ([T.1, T.6][mei]) |
| Zhang, sim | clean | not applicable (reward-free score) | 16 | −0.5 / +1.4 pp ([T.1][zhang]); +5.6 / +9.4 with synthetic corruption ([Fig. 2][zhang]) |
| ARFM, sim π0 | success-only | not applicable (returns) | batch softmax | RWR +2.7 pp, ARFM +4.0 pp ([T.1][arfm]) |
| ReinboT, sim | 10k failures on 20k teleop | not applicable | raw return | sparse reward +0.02, dense +0.46 chain length ([T.1][reinbot]) |
| PACL, sim | critic without failed rollouts | Table III | conditioning | 77.6 vs 78.8 base; with failures 93.2 ([T.III][pacl]) |

Two rows in that table are the cleanest statements of the mechanism. PAC gains nothing on near-expert data (BC+Q 71.3 vs PAC 69.3 on RC:Tower, [T.1][pac]) and 10× on 12 %-success data. PACL's critic trained without failed rollouts made the policy worse than its starting point ([T.III][pacl]).

**The exponential with unit-mean normalization is a fixed-shape filter.** Your weights, and Cao's, standardize A first. After that, the gradient shares depend only on beta and the clip, not on what A contains. My arithmetic for Â ~ N(0, 1) over a 128-row batch; the critic probe reports the real values (`adv_weight_ess_frac`, `adv_top5_weight_share`, `adv_clip_frac`):

| beta | clip | ESS / N | top 10 % of rows carry | bottom half carries | max / min weight | rows at the clip |
|---|---|---|---|---|---|---|
| 1 | 3 (yours) | 0.40 | 39 % | 16 % | 403 | 0.3 % |
| 1 | 2 (Cao) | 0.49 | 35 % | 17 % | 55 | 4.6 % |
| 1 | 1 | 0.69 | 21 % | 22 % | 7 | 32 % |
| 2 | 3 | 0.78 | 22 % | 31 % | 20 | 0.3 % |
| 0.5 | 3 | 0.07 | 74 % | 2.5 % | 1.6e5 | 0.3 % |
| linear ramp over ±2 sigma (SARM, ReTVL) | | 0.81 before zeroing negatives | | | | |

So at beta 1 the loss is "the top half of each batch at about five times the bottom half," whether the critic's ranking is right, wrong or noise. Per-batch standardization makes this exact per batch: a batch where A is pure critic noise gets the same 403× spread as a batch where A separates a drop from a clean grasp. Without the clip the mean cancels and the effective temperature on raw A is beta · std_B(A), which moves with batch composition and with the critic's spread as it trains (algebra in v1 §2). Cao computes statistics once over the dataset with a floor of 0.1 × the source std, which is the only published guard against inflating a small spread ([Cao Eq. 4, §IV-B][cao]). No paper standardizes per batch, and no paper pools standardization across tasks.

**The critic is first-order.** At a fixed rule, ReTVL's value model vs a progress-value baseline is 80 % vs 63 % ([ReTVL T.2][retvl]), and Table 7 shows why: the baseline kept 74 % of strictly bad chunks, ReTVL 21 % ([ReTVL T.7][retvl]). SARM's reward model vs ReWiND is 10/12 vs 6/12 and 8/12 vs 3/12, and ReWiND "fails to recognize regressions" ([SARM T.2][sarm]). Cao's Q − V advantages, near chance on the offline tests, cut PhoneH success from 0.70 to about 0.2-0.3 by eye ([Cao Fig. 4-5][cao]). Against that, the weight-vs-filter gap is 0.24 in Cao and within noise elsewhere.

## 2. Where the papers agree

- **Bound the weights and keep a uniform anchor.** Cao 55×, Zhang 16×, Fang [0, 2], SARM and ReTVL [0, 1]; Fang keeps demos at 1, Zhang mixes toward uniform with alpha as your lam does ([Fang Eq. 8][fang], [Zhang Eq. 21][zhang]). The unbounded variants are the ones that failed: weights above 10 with periodic loss oscillation, −2.3 / −6.5 pp below SFT ([Zhang T.5, §5.6][zhang]); per-sample losses near 1000 in one batch ([CFGRL Fig. 2][cfgrl]); held-out loss climbing to 50+ while training loss stays near 0 at alpha 3 on 0.1M rows ([Park Fig. 4][park]).
- **The critic needs failures.** PACL's Table III above; Cao's explanation of why Q − V is near chance is that real datasets have narrow action coverage ([Cao §V-B][cao]); Fang's efficiency head trains only on successes and the authors say its signal is "weakest precisely when success is rare" ([Fang §5][fang]).
- **Per-sample weighting of the flow loss needs no flow-specific correction.** One weight per chunk, the same at every flow time, divided by a normalizer that does not depend on x_t or t, has the tilted policy as its minimizer ([QIPO Thm 4.3, Lemma B.1][qipo], [Mei Thm A.5][mei]). "Exact energy-weighted flow" is this loss, not a rival to it. Weight only the action coordinates; a value head trained through the same weighted loss hid its own errors ([Mei T.4, App. A.4][mei]).
- **A hard filter at a critic threshold is the same operation with a sharper edge.** Every paper that compares them finds them close except Cao (below).
- **Nobody measures effective sample size.** Cao names it as something the pipeline changes and never reports it ([Cao §I][cao]). The only weight diagnostics in the set are Zhang's histograms and 1/std(w), ReTVL's weight by chunk type, and Fang's mean weight and filter pass rate. Your probe already logs more than any of them.

## 3. Where they conflict, and what resolves it

**1. Simulation ranks AWR last; real flow-VLA papers report large wins.** Park: AWR at a median 0.88× of DDPG+BC, 0.5× on exploratory data, DDPG+BC ≥ AWR in 15 of 16 settings ([Park T.1][park]). FQL: flow AWR 16 vs FQL 44 over 50 OGBench tasks ([FQL Fig. 5][fql]). Dong: AWR 47 → 24 while best-of-64 under Q goes 38 → 75 ([Dong Fig. 6][dong]). Four differences account for most of it. The sim baselines are methods that hill-climb or re-rank under Q(s, a); the robot baselines are BC, filters and SFT. The sim AWR is raw Q − V with no normalization and a 4-value temperature grid including 0; Park's actor is a fixed-std Gaussian and Dong never reports beta ([Park App. C.6][park], [Dong App. B][dong]). AWR's deficit is largest on exploratory data (0.5×) and smallest on noisy-expert data (0.9-1.0×), and teleop plus rollouts is the latter ([Park T.1][park]). Sim uses 3-8 seeds; the robot papers use 12-50 trials, one run. What does not resolve: Dong's AWR led at round 0 (47 vs 38) and then fell on every task across rounds, and the configuration that produced that is unreported. It is the one negative result in the set that could apply to iterated collect-and-retrain on your data.

**2. Weighting beats filtering (Cao) vs weighting ≈ filtering (everyone else).** Cao: 0.74 vs 0.50 success, 59 vs 40 of 80 pooled trials, z ≈ 3.1 by my arithmetic ([Cao T.II][cao]). SARM: a duration filter reached 7/12 against 10/12 on the easy task, p ≈ 0.37 ([SARM T.2][sarm]). PACL: success-only filtering within 0-3 trials of 25 of the best method ([PACL T.I][pacl]). Dong: filtered IL best at small scale, 60 vs 54 ([Dong Fig. 5][dong]). Mei: filtered BC 31.6 vs BC 31.0 vs IDQL weights 39.0 ([Mei T.1][mei]). The resolution is what each filter is. Cao's filter keeps rows above the batch median of a per-source z-score, so it drops about half the demonstrations in every batch and is untuned (q = 0.5 only); it also fell below advantage-free DAgger on 2 of 4 tasks ([Cao T.II, §IV-C][cao]). The other filters keep every demo and drop failed episodes. So "continuous beats binary" is established against a median filter, not against an outcome filter. Fang's Imit-Recap (threshold on a TD residual, passes 19-27 % of rollout rows) also trails Fang's soft weights, but it differs in the critic and the credit rule at the same time ([Fang Fig. 3, App. G][fang]).

**3. "AWR ignores most of the batch" (CFGRL) vs "weighting is the exact method" (QIPO, Mei).** Both hold. Exactness is a statement about the minimizer; concentration is a statement about the variance of the estimator at finite batch size. CFGRL's Figure 2 is raw IQL advantages at 1/beta up to 30 with no normalization ([CFGRL Sec. 5, Fig. 2][cfgrl]); the table in §1 shows what a clip does to it. The clip and beta trade bias toward BC for variance; that is the whole design space, and the winning robot recipes sit at the low-variance end (ESS ≈ 0.8).

**4. One scalar is not enough (Fang) vs one progress delta is enough (SARM, ReTVL).** Fang's argument is that a success-probability signal gives no gradient between fast and slow completions, and a steps-to-go signal trained on successes sees no failures, so they use two heads and a gate ([Fang abstract, §3.2-3.3][fang]). SARM and ReTVL use one value delta over a chunk, and it works because their value model drops at regressions: SARM's reward model has Spearman 0.67 on rollouts where ReWiND has 0.50, and ReTVL's retry preference loss lifts Drop-AUC from 0.51 to 0.80 ([SARM T.2][sarm], [ReTVL value ablation][retvl]). So the disagreement is about what the value function was trained to see, not about the number of heads. Your critic gets both: a time penalty per second, a penalty at mistake onset, and a failure terminal, so it is structurally a single head that has seen failures, closer to ReTVL than to Fang's efficiency head alone.

**5. Ten-second TD (Cao) vs one-second residuals (you, Fang, SARM, ReTVL).** Cao's offline tests favour a 10-chunk TD residual over Q − V and Monte Carlo V (R_int 0.72 vs ≈ 0.5, [Cao Fig. 4][cao]) and never try n = 1. With Cao's sparse terminal reward, a 1 s residual is the critic's own 1 s value change and nothing else. With a dense time penalty, a 1 s residual says whether this second reduced the expected remaining time by more than one second, which is Fang's A_e exactly ([Fang Eq. 6][fang]) and SARM/ReTVL's delta over one chunk ([SARM][sarm], [ReTVL Eq. 11][retvl]). Those two fit their value model to progress labels directly; yours bootstraps to a time-to-go. Whether a bootstrapped V is sharp enough at 1 s resolution is the open question; Cao's MC-vs-TD result and your own observation that TD critics stop short of the theoretical fixed point cut in opposite directions, and no paper settles it.

**6. Speed.** Fang is the only paper that measures successful-episode length, and every fine-tuned variant, including the one with an efficiency head, is slower than SFT on all three tasks (Snack Bag ≈ 1240 vs ≈ 1610 frames, std ± 150-400) ([Fang Fig. 3][fang]). Survivorship is possible (SFT succeeds only on easy starts). Nothing else in the set speaks to hesitation or cycle time.

## 4. Author claims the tables do not support

| Paper | Claim | What the numbers show |
|---|---|---|
| Cao | beats hard filtering and binary weighting | the "binary" baseline is source-based DAgger with no advantage; the filter is a batch-median z filter; no plain BC on the same data; 20 trials per task, no CIs ([§IV-C, T.II][cao]) |
| Dong | implicit extraction beats explicit | true after 10-20 rounds in sim; AWR was ahead at round 0; AWR beta unreported; AWR never run on the robot ([Fig. 6, App. B][dong]) |
| Mei | potential-guided flow beats IDQL | at matched K = 1 the separate-critic exp-weighted baseline wins, 39.0 vs 34.6; the headline needs best-of-5 ([T.6][mei]) |
| QIPO | "consistently outperforms the baselines" | locomotion 86.2-86.3 vs QGPO 86.6; AntMaze 77.3 vs IDQL 79.1; best-over-training evaluation; no BC column ([T.2, App. E.1][qipo]) |
| Fang | hierarchical advantage weighting for online RL | rounds of weighted BC with human takeovers; failures get weight 0; two heads; no flat exp-AWR arm; 92/88/38 are best-checkpoint picks ([§3.2-3.5, Fig. 3][fang]) |
| Zhang | conservative reweighting for mixed-quality data | no AWR or value baseline; reward-free score never tested on real failures; the alpha = 0 control puts the reweighting itself at +0.3 / +1.3 pp ([T.5][zhang]) |
| CFGRL | guidance scale w generalizes AWR's 1/beta | with the binary label 1[A ≥ 0], f^w = f for all w > 0, so the stated target is the same at every w; the gains at w > 1 come from the guidance approximation (my inference; [CFGRL Eq. 12, 15][cfgrl], [QIPO Lemma 4.10][qipo]) |

## 5. Alternatives measured in the same papers

With V(s) and logged chunks you can weight, filter, or condition on sign(A). Best-of-N, PA-RL, V-GPS, DDPG+BC, FQL and QIPO's support actions all score actions that are not in the data and need Q(s, a) ([Park Eq. 6-7][park], [FQL Eq. 9][fql], [PA-RL Eq. 4.1][parl], [V-GPS Eq. 2][vgps], [QIPO Alg. 4][qipo]).

| Alternative | Same-paper result vs weighting | Source |
|---|---|---|
| Outcome or success filter | within 0-3 trials of 25 on real tasks; best at small data scale in sim | [PACL T.I][pacl], [Dong Fig. 5][dong] |
| Batch-median advantage filter | 0.50 vs 0.74 success | [Cao T.II][cao] |
| Sign-of-A conditioning + guidance | beats AWR on 8/9 ExORL; OGBench mixed (antmaze-large 20 vs 50) | [CFGRL T.1-2][cfgrl] |
| Conditioning label choice | source 85.6 < outcome 88.0 < Q − V 88.8 < Q 91.2 | [PACL Fig. 6][pacl] |
| Re-ranking under Q | exploits a weak critic more as N grows: 79.2 → 58.8 (N 4 → 16); 0.59 → 0.35 (K 10 → 100) | [PACL T.II][pacl], [V-GPS App. H][vgps] |
| Inference steering, frozen VLA | Octo 0.24 → 0.44; OpenVLA 40 → 50 (frozen) → 70 (online) | [V-GPS T.1][vgps], [PA-RL Fig. 1][parl] |

No paper compares weighting with conditioning on a VLA or a real robot; Cao left conditioning out on purpose ([Cao §IV-C][cao]).

## 6. Limits of the evidence

- **Resolution.** 12-50 trials per arm, one training run, no seeds, CIs only in Fang. Twenty trials per arm resolve about 30 pp; per-task gaps below that are noise. Cao's ranking survives only after pooling 80 trials.
- **The control that matters is missing.** No exp-weighted real flow-VLA paper has uniform weights on the same data and steps. Cao's SFT reference got neither the rollout data nor the 4000 updates; on Printer, weighting with near-chance advantages still reached about 0.5 from 0.00, close to advantage-free DAgger's 0.45, so most of that task's gain is data and updates ([Cao Fig. 5, T.II][cao]). Plain BC on everything, failures included, also beat the starting policy in PACL (92/84/68 vs 84/72/64, [PACL T.I][pacl]).
- **Selection.** Fang's 92/88/38 are the best of an unstated number of checkpoints; sim papers tune per task or per cell on evaluation return; QIPO reports best-over-training.
- **Scope.** One model per paper, one lab, per-task policies (Cao, Fang). No paper weights one actor across tasks with different value scales, standardizes per batch, uses a distributional V-only critic, or tests n = 1.
- **No behaviour-level side effects** beyond Fang's episode lengths. No ESS anywhere.

## 7. Your setup against this

Your advantage is `A = −1/12 + 0.97 · V(s') − V(s)` over one second, with −5/12 added at a mistake onset and a terminal at each subtask end (0 on success, −10/12 on failure). V is a discounted time-to-go: about −0.3 for a 4 s atom, −0.6 for an 8 s ReBot segment, −2 for a 40 s bits grasp, by my arithmetic from the yaml constants. Consequences the papers speak to:

- **Nearest analogues are Fang's A_e and the SARM/ReTVL chunk delta, not Cao.** Same horizon, same "progress minus expected progress" content. What made those work was a value signal that drops at regressions (SARM, ReTVL); your mistake penalty and failure terminal are the equivalent, and `adv_weight_mistake_ratio` in the probe is the direct check.
- **Cross-task scale.** Sources whose segments are long have V several times larger in magnitude and a wider 1 s residual. A pooled per-batch z-score lets the widest-spread source fill both tails and gives a narrow-spread source near-uniform weights (v1 §2 algebra). Cao's only reason for value bins is that raw advantage level tracks progress; eta² fell from 0.092 to 0.026 ([Cao T.I][cao]). Your `adv_weight_end_ratio` is the same check in a different form. Per-task or per-source statistics computed once, as Cao does, are the published counterpart; pooled per-batch statistics have none.
- **Weight range.** Clip 3 with beta 1 gives ESS 0.40 and a 403× range; Cao's clip 2 gives 0.49 and 55×; the winning bounded recipes sit near 0.8. Beta 2 at clip 3 lands at 0.78 with a 20× range, inside the published band, at the cost of a flatter tilt. Nothing in the papers says which is right for your data; Park's Fig. 4 is the only measurement of what the sharp end costs (overfitting at alpha 3 on 0.1M rows).
- **Noisy single-sample advantages bias the weights upward** where the next second's outcome is random: E[exp((Y − V)/tau)] is monotone in Q but not the Boltzmann factor ([Mei Rem. A.6][mei]). A grasp that sometimes slips gets more weight than a grasp with the same mean advantage that never does. No paper corrects for this; a lower beta makes it worse.
- **Your diagnostics already exceed the literature's.** ESS, top-5 % share, clip fraction, weight by mistake and by time-to-end, percentile frames. The two published checks you do not have are Cao's pairwise orderings before any actor step (takeover window vs the policy window before it; stalled vs advancing segment) and ReTVL's Table 7 (mean weight on demos, on successful rollout chunks, on chunks before a failed grasp, drop or slip).

## 8. What to expect

1. If teleop demos are most rows, a few pp at most over lam = 0, and a null result is consistent with everything published ([Zhang T.1][zhang], [ARFM T.1][arfm], [FQL T.2][fql]); the large gains came only where failed rollouts were a big share and the value signal had been checked ([ReTVL T.2][retvl], [SARM T.2][sarm], [Cao T.II][cao]).
2. Run lam = 0 on the same data and steps as the control; no exp-weighted real flow-VLA paper has it, and in Cao most of one task's gain was data and updates, not the weight ([Cao Fig. 5][cao]).
3. Judge the critic before the actor run: mistake ratio well below 1, end ratio near 1, and Cao-style orderings above chance; value-model swaps moved real success more than any rule change ([ReTVL T.2, T.7][retvl], [SARM T.2][sarm], [Cao Fig. 4-5][cao]).
4. Your weights are the sharpest in the set (ESS 0.40, 403×) and the only ones standardized per batch and pooled across tasks; clip 2 or beta 2 and per-source statistics computed once put you inside the published band ([Cao Eq. 4][cao], [Zhang T.5][zhang]).
5. Twenty trials per arm resolve about 30 pp; track successful-episode length as well ([Fang Fig. 3][fang]), and read a null as Park's drift-to-BC before reading it as a critic failure ([Park §5.4][park]).

## 9. Papers worth your time

- Cao et al. 2026 ([2609.28161][cao]): the only paper that tests offline whether the advantage ranks actions before spending robot time, and the closest weighting rule to yours; read §IV-A, Fig. 4, Table I, Table II, and note what the filter and DAgger arms actually are.
- ReTVL, Qin et al. 2026 ([2606.24633][retvl]): a real flow VLA on data with failures and retries, a same-data BC baseline, a value-model swap at a fixed rule, and Table 7, the weight-by-chunk-type check you can copy.
- Park et al. 2024 ([2406.09329][park]), §5.4 and Fig. 4 only: the mechanism by which exp weights fail (few effective rows, drift back to BC) and the train-vs-held-out loss check; skip the DDPG+BC advice, which needs Q(s, a).

[cao]: https://arxiv.org/pdf/2609.28161
[dong]: https://proceedings.iclr.cc/paper_files/paper/2026/file/b650f2838de7e1e817b5408b48a19a63-Paper-Conference.pdf
[fang]: https://arxiv.org/pdf/2606.17043
[zhang]: https://arxiv.org/pdf/2603.16542
[mei]: https://arxiv.org/html/2606.04968v1
[qipo]: https://arxiv.org/pdf/2503.04975v1
[cfgrl]: https://arxiv.org/pdf/2505.23458v1
[park]: https://arxiv.org/pdf/2406.09329v2
[fql]: https://arxiv.org/pdf/2502.02538v2
[parl]: https://arxiv.org/pdf/2412.06685
[vgps]: https://arxiv.org/html/2410.13816v2
[arfm]: https://arxiv.org/html/2509.04063
[sarm]: https://arxiv.org/html/2509.25358v3
[retvl]: https://arxiv.org/html/2606.24633
[pacl]: https://arxiv.org/html/2609.29000
[reinbot]: https://arxiv.org/html/2505.07395
[pac]: https://arxiv.org/html/2402.05546
