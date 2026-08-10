# 05 — Training

Entry point: `uv run python -m lerobot.scripts.rl_offline --config config_rl.yaml`
from the repo root — and it loads the **root** `config_rl.yaml` (the copy in
`src/lerobot/rl/` is only a drifting template). Trainer:
[`MolmoAct2Trainer`](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py); config
class `MolmoAct2RLConfig` (`molmoact2_rl`,
[rl_molmoact2.py:39](../src/lerobot/rl/molmoact2/rl_molmoact2.py#L39)).

## 1. Data path

`rl_offline.py` explicitly disables the LeRobotDataset delta machinery and routes
everything through `ReplayBuffer.from_lerobot_dataset` → `ReplayBuffer.sample()`.
At fill time the buffer **materializes** the pi07 columns:

- `subtask_index` per frame (from `meta/subtasks.parquet` vocab),
- `summary_prev_index` / `summary_target_index` (`materialize_summaries`),
- `metadata_quality` / `metadata_mistake` (`materialize_metadata`),
- raw depth in `complementary_info` (uint16, memmap),
- `is_golden` when `treat_main_dataset_as_golden` (advantage = optimal; only read
  by `compute_advantage`, i.e. inert under skip_critic).

**Memmap cache**: `scripts/lerobot_memmap_buffer_cache.py` pre-decodes
image/depth rows to disk. `image_stride: 5` stores image rows every 5th frame
(30 Hz → 6 Hz) while low-dim stays dense so action chunks are exact; the stride is
part of the cache fingerprint (mismatch = hard error, not silent video-decode
fallback — but dtype/size mismatches DO silently fall back, so the policy config
must declare its `image_storage_*` fields). History windows are assembled at
sample time — no cache format change.

## 2. Actor update

`update_actor` ([rl_molmoact2_trainer.py:627](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L627)),
per optimization step: sample (online ⊕ offline iterators, concatenated), inject
recorded depth into the observation dict (`_inject_depth_observations` — no-op
without `pointmap_config`; one-shot warning if the depth column is missing so we
never silently train on the null bank), build the training batch, then:

$$\mathcal L = \underbrace{\mathcal L_{flow} + \mathcal L_{aux}}_{\text{action expert}}
 + \underbrace{\mathcal L_{FAST} + \mathcal L_{z} + \mathcal L_{aux}^{disc}}_{\text{VLM discrete head}}
 + \lambda_{gen}\,\underbrace{\mathcal L_{gen}}_{\text{subtask CE}}$$

- **Flow loss** — [02 §3.2](02_base_model.md); joint per-layer forward (VLM K/V
  collected in the same pass), depth state threaded per layer, gradient
  checkpointing on, action-dim/chunk padding masked.
- **Discrete CE** — FAST action tokens with `action_mode: both` and knowledge
  insulation; logged as `loss_discrete_ce`. The z-loss remains active as a
  logit-scale regularizer but is not a separate dashboard series.
- **Generation CE** — separate forward on the generation prompt for annotated
  samples, separate backward accumulating into the same grads, weight
  `subtask_loss_weight`. **Currently 0.0, i.e. off**; the forward still runs and
  logs `loss_subtask_ce` as a frozen-model baseline whenever the weight is > 0.
  The summary-CE half of this term was removed with the summary-memory seam —
  there is no `loss_summary_ce` any more.
- **Both auxiliary terms are off by default** (`enabled: false`, all weights 0) and
  are additive per-sample corrections, never replacements: the base flow loss and
  the base FAST CE apply to every sample regardless. §2.1 and §2.2.

Prompt conditioning per sample: subtask clause (70%), metadata clause (85%),
history clause (if `memory.history_keys` set, 70%) — `subtask_dropout` 0.3 /
`metadata_dropout` 0.15 / `history_dropout` 0.3, applied to training text only.
**No advantage clause** under `skip_critic` (`advantage=None` in
`build_training_batch`; `inference_advantage` must stay null so eval matches).

### 2.1 Flow auxiliary — `action_auxiliary_loss` (four thresholded trajectory terms)

Config block `policy.action_auxiliary_loss`; computed in
`_action_trajectory_components_with_padding` + `_thresholded_action_auxiliary_loss`
([modeling_molmoact2.py:40](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L40)).

The flow branch predicts a velocity, so the metrics are taken on the **one-step
denoised chunk** at each sampled flow time, $\hat a = x_t + (1-t)\,v_\theta$ — no
extra forward and no ODE solve. The four metrics are computed per flow time and
then averaged over flow times (`_finite_mean`, so NaN samples drop out) to one
value per example:

| Term | Quantity | Notes |
|---|---|---|
| `path` | mean squared error over all valid $(t, d)$ | the plain chunk MSE |
| `shape` | MSE of **adjacent-target differences**, $\Delta\hat a - \Delta a$ | ignores a constant offset; 0 when `chunk_size == 1` |
| `terminal` | MSE at the final target only | NaN when the last step is padding |
| `direction` | $1 - \cos$ between predicted and true **final displacement from the hold pose**, clamped to $[0,2]$ | NaN when the target does not move from hold; 0 aligned, 1 perpendicular, 2 opposite |

Path, shape, and terminal are optimized as relative errors: each raw error is
divided by the matching hold-still error for that chunk, clamped by the shared
corpus-derived scale floor. Thus 0 is exact, 1 is no better than hold, and values
above 1 are worse than hold. The raw MSEs remain available as diagnostics.
Direction is already dimensionless.

Each term is independently gated and weighted:
`total += weight × value` where `value` is finite **and** (`threshold is None` or
`value > threshold`). A threshold is a hard gate on the flow-time-averaged metric,
not a hinge — the term contributes its full value once it fires, or exactly zero.
Padding is excluded throughout via `action_horizon_is_pad` / `action_dim_is_pad`.

Requirements: `action_encoding` must not be `delta` (config rejects it — final
position and shape are not represented in delta space), and the batch must carry
`action_hold`.

**Picking thresholds.** The optimized quantities are `path_relative`,
`shape_relative`, `terminal_relative`, and `terminal_direction_loss`; these
same four quantities are what `action_trace` writes to
`validation/step_*/action_trace/action_metrics.json` under `trajectory`
(`trajectory_error_components` in
[action_metrics.py](../src/lerobot/utils/action_metrics.py) — shared by the probe
and the loss, so the units match exactly). Each carries `mean`/`median`/`p10`/`p75`/
`p90` for offline inspection.

The probe reads **nothing** from `action_auxiliary_loss` — it computes all four
components unconditionally, gated only on `enable_action_trace`. So there is no
chicken-and-egg: the numbers appear at step 0 of any run with `val_on_start: true`.
The only constraint is that config is read at launch, so a threshold meant to be
active from step 0 has to be in the file before you start. Either harvest the file
from an aborted start and relaunch, or start with `null` thresholds and set them on
a later run.

Two caveats. `action_metrics.json` files written before 2026-08-08 have the old key
set (`mse_norm`, `skill_vs_hold`) and no `trajectory` block. And the probe's sample
is small — `trace_max_anchors_per_episode` 6 × 3 val episodes ≈ 17–18 anchors — so
its tail quantiles are coarse. Live training therefore reports only the full-batch
mean, median, and gate fraction for each component.

### 2.2 Discrete auxiliary — `discrete_action_auxiliary_loss` (three FAST-logit terms)

Config block `policy.discrete_action_auxiliary_loss`; computed in
`_fast_action_auxiliary_components`
([modeling_molmoact2.py](../src/lerobot/policies/molmoact2/modeling_molmoact2.py)),
inside the discrete branch, on the teacher-forced logits it already has. **Zero new
parameters.** Requires `action_mode` `discrete` or `both`.

A FAST BPE token covers a variable run of DCT coefficient slots, but under teacher
forcing the ground-truth tokens pin every boundary, so each of the $T \times D = 210$
slots is covered by exactly one token position. At init,
`_build_fast_auxiliary_token_tables` decodes the whole 1005-token BPE vocabulary once
into `len(v)` and `val(v, r)` tables and aligns the BPE ids with LM token ids. At
train time, for slot $n$: restrict the logits to the FAST action-token columns, group
them by the coefficient value each candidate token would place at that slot, and
`logsumexp` within each group — tokens too short to reach the slot fall into a drop
bin that is excluded from the normalizer (covered-mass renormalization). That gives a
marginal $q_n$ over coefficient bins.

| Term | Quantity |
|---|---|
| `ordinal` | cumulative-threshold binary CE: for each cut $\theta$, $-\log q_n(c \le \theta)$ if $c^*_n \le \theta$ else $-\log q_n(c > \theta)$; mean over cuts, then a **uniform mean over all 210 slots** |
| `path` | $\frac{1}{N\gamma^2}\sum_n (\mathbb E_{q_n}[c] - c^*_n)^2$ — by Parseval ($a = C^\top c/\gamma$, $C$ orthonormal) this *is* the per-element trajectory MSE, computed without an IDCT |
| `shape` | $\frac{1}{(T-1)D\gamma^2}\sum_n \lambda_k (\mathbb E_{q_n}[c] - c^*_n)^2$ with $\lambda_k = 4\sin^2\!\big(\tfrac{\pi k}{2T}\big)$, $k = \lfloor n/D \rfloor$ — the exact adjacent-difference MSE as a per-frequency reweighting |

`path` and `shape` first produce the literal raw MSEs above, then divide them
by the matching hold-baseline power computed from the normalized target and
`action_hold`. The shared path/shape floors cap the weight of nearly static
chunks. Their configured weights apply to these relative values; the raw MSEs
remain diagnostics. `ordinal` stays inside the CE family and is not
power-normalized. The base FAST token CE is also unchanged.

Verified independently by
[`fast_soft_decode_probe.py`](../../fast_soft_decode_probe.py) (repo root): exact
one-hot identity (max $|\Delta a| = 0$), exact tiling of the 210 slots on real
chunks, monotone degradation as the head softens.

Telemetry is symmetric with the flow branch (§5): `loss_discrete_aux`,
`discrete_aux/{ordinal_ce,path_relative,shape_relative}_mean`, and
`val_loss_discrete_aux` on the held-out sample.
Both `loss_*_aux` keys are popped when their block is disabled, so an off term
leaves no zero-valued panel behind.

**Known gaps before this can be turned on:**

1. **`ordinal` is uniform over slots**, with no per-band breakdown to justify a
   different weighting. Coefficient energy is heavily concentrated in the low
   frequencies and the high-frequency slots are covered by long zero-run merge
   tokens, so a uniform mean is unlikely to report what it looks like it reports.
   The `discrete_aux/ordinal_ce_*` quantiles give the aggregate but not the split
   by frequency band — measure that before committing to a weight.
2. **Never run end-to-end.** Unit tests cover the pieces on synthetic tables; the
   real-vocabulary init and the per-example loop have not seen a training batch.
3. Per-example Python loop over the batch — worth timing on the first smoke run.

The design note this grew out of is
[archived and its numbers are retracted](archive/fast_soft_decode_auxiliary.md);
it is not a spec for the code above.

## 3. Freeze and optimizer rules

Two layers of gating:

1. Coarse `__init__` freeze (`_freeze_non_action_expert_parameters`) — everything
   but the action expert off.
2. Per-name authoritative freeze when `cfg.policy.trainable_params` is set:
   `_apply_actor_freeze` / `_apply_critic_freeze`
   ([rl_molmoact2_trainer.py:287,329](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L287)).
   Both end in an **else-branch that freezes unrecognized params** — so every
   fresh from-scratch module needs an explicit always-trainable branch:

```python
if "pointmap_encoder" in name or "depth_stream" in name:
    param.requires_grad = True   # independent depth CNN/ViT/pooler/projector/marker
elif ".action_expert." in name:
    ...
else:
    param.requires_grad = False  # unknown actor param — freeze
```

This else-branch silently killed the depth gate once ([03 §B.4](03_depth.md)).
The same rule will apply to any future history-consumption module.

Optimizer groups (`get_optimizer_groups` + `_split_depth_group`): a dedicated
**"depth"** group contains `pointmap_encoder`, `depth_visual` and `depth_marker` and
is excluded from pretrained merge. `depth_lr: null` makes its LR exactly
`optimizer_lr` (5e-5 in the root config); the separate optimizer is bookkeeping,
not a different update scale. Grad clip covers all actor parameters.

**Pretrained merge**: fires once at a configured step (current run: 8000,
α = 0.2) — soft-merge back toward the pretrained weights; depth group excluded;
missing "critic" target warns and skips under skip_critic; optimizer state cleared
after; pre/post checkpoints saved.

## 4. Distributional critic (built; current run has `skip_critic: true`)

`MolmoAct2Critic` ([rl_molmoact2.py:175](../src/lerobot/rl/molmoact2/rl_molmoact2.py#L175)):
a from-scratch bidirectional transformer initialized from backbone blocks, running
in bf16 over the sequence `[obs | depth | value-queries]` with **201 value-query
tokens** (one per bin), packed position ids, and a per-query scalar head.

**HL-Gauss.** The value axis is discretized into bins with centers $c_i$; a scalar
target $V^\*$ becomes a categorical target by integrating a Gaussian over each bin:

$$p_i^\* = \Phi\Big(\frac{b_{i+1} - V^\*}{\sigma}\Big) - \Phi\Big(\frac{b_i - V^\*}{\sigma}\Big), \qquad \sigma = 8.0 \times \text{bin width}$$

($\Phi$ the standard normal CDF; `sigma_ratio = 8.0` — 5.0 produced spiky
under-fit distributions). Loss is cross-entropy against the critic's softmax over
bins; the scalar value is the expectation $V = \sum_i p_i c_i$
(`value_from_probs`). A flat-looking E[V] curve is by-design smoothing, not
collapse. Known cosmetic quirk: E[V] starts near the *lower* support bound (bins
are positional + random logit head); zero-initializing the head would center it
(considered, declined).

**TD bootstrap**: target $V^\* = r + \gamma (1 - \text{done})\, V_{EMA}(s')$ with
per-step reward scaled by `reward_normalization_constant` so returns fit the
support. V(s′) runs on the EMA `critic_target` (generic lerp; depth modules
included). `skip_critic: true` skips construction entirely (no VRAM) — but note
the semantics: it only freezes critic *training*; any pretrained-critic forward
passes for logging stay unguarded.

The critic's consumer is RECAP-style **advantage conditioning**: threshold
advantage → "The advantage is positive/negative." clause in the action prompt.
Fully bypassed in the current offline run.

## 5. Telemetry

Console + wandb via the `accum` dict → `log_metrics`. The console line is a fixed
train/val-interleaved subset: `loss_flow`, `loss_action_aux`, `loss_discrete_ce`,
`loss_discrete_aux`, each next to its `val_*` twin, then `loss_subtask_ce`,
`actor_grad_norm`, `loss_critic`. W&B uses a compact allowlist rather than every
diagnostic returned by the update.

**Losses.** `loss_actor`, `loss_flow`, `loss_discrete_ce`, `loss_action_aux`,
and `loss_discrete_aux`; optional losses are popped when their term is off, so a
disabled term leaves no flat-zero panel. `loss_subtask_ce` appears only when
`subtask_loss_weight > 0`. **There is no `loss_summary_ce`** — the summary-memory
seam was removed, along with `materialize_summaries` and the hold/update split.

**Held-out losses** (`val_loss_frames: 128`, one effective batch packed once at
startup with seeded flow noise, so the comparison is apples-to-apples across
steps): `val_loss_flow`, `val_loss_discrete_ce`, plus `val_loss_action_aux` /
`val_loss_discrete_aux` whenever those terms report. This replaced the modality
ablation deltas (removed 2026-08-08: they estimated conditional MI on memorized
training frames, which is 0 by construction).

**Distributions and slices.** W&B keeps exactly three histograms:
`flow_loss_per_sample_histogram`, `discrete_ce_loss_per_sample_histogram`, and
`auxiliary_loss_per_sample_histogram` (the weighted flow + discrete auxiliary
contribution). Raw-MSE duplicates, flow-time slices, and z-loss telemetry stay out.
Detailed measurements remain in validation probes where applicable.

**Auxiliary components.** W&B keeps only
`action_aux/{path_relative,shape_relative,terminal_relative,terminal_direction_loss}_mean`
and their gate fractions. The discrete branch keeps
`discrete_aux/{ordinal_ce,path_relative,shape_relative}_mean`.
`action_trace` retains richer offline quantiles.

**Depth.** W&B keeps `depth_rgb_rms_ratio` at the text-embedding seam and the
total `depth_grad_norm_preclip`. The per-stage RMS values, ratios, and split gradient
norms remain internal debugging diagnostics and are not permanent dashboard series.

**Validation probes** at `val_freq` (action_trace carries the fit headline —
normalized MSE vs the hold-still and dataset-mean baselines; critic probes
self-skip under skip_critic). The `objective` probe's local report compares both
auxiliary losses and all components on validation versus training data. W&B receives
only `z` for the four headline objectives, plus held-out FAST top-1;
component details remain in the probe artifact. That comparison, not the live train
curve, says whether a term is buying generalization. Probes must thread
`cfg.policy.inference_advantage` — not a hardcoded advantage — so eval prompts
match training.

## 6. Config quick-reference (live values, root `config_rl.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `dataset.sources` | socks_basket / shirts_bin / two_container | merged corpus, 38 eps / 165,740 frames; `repo_id` is `cijerezg/rebot-all-v1` and sources[0] supplies norm stats |
| `val_dataset_path` | `outputs/rebot_val-annotated-v2` | 3 held-out eps, one per task family |
| `val_freq` / `val_loss_frames` | 400 / 128 | probe pass = save cadence; held-out losses every `log_freq` |
| `skip_critic` | true | plain BC + prompt conditioning |
| `batch_size` × `gradient_accumulation_steps` | 32 × 4 = **128** | effective batch (several in-file comments still say 2/64 — wrong) |
| `offline_steps` | 30000 | |
| `policy.chunk_size` / `n_action_steps` | 30 / 30 | 1.0 s at 30 Hz; divisors keep `image_stride` legal |
| `policy.image_stride` | 3 | must match the memmap cache build (fingerprinted; mismatch is a hard error) |
| `policy.action_encoding` | anchor | + `action_encoding_stats_path`, pooled over all three roots at chunk_size 30 |
| `policy.action_auxiliary_loss` | `enabled: false`, all weights 0 | §2.1 — thresholds come from `action_trace/action_metrics.json` |
| `policy.discrete_action_auxiliary_loss` | `enabled: false`, all weights 0 | §2.2 |
| `policy.subtask_regeneration_interval` | 4.0 s | HL decode cadence at rollout |
| `policy.subtask_max_new_tokens` | 64 | HL decode budget (inference only — does not train) |
| `policy.subtask_loss_weight` | **0.0** | generation CE **off** |
| `policy.memory.metadata_enabled` | true | quality/mistake clauses from dataset meta |
| `policy.memory.history_keys` | state + top/wrist images + depth | short-term history on; 5 samples over 5 s → offsets 30/60/90/120/150 |
| `policy.pointmap_config` | set | depth on; factory intrinsics; z ∈ [70, 800] mm; `rgb_dropout_prob` 0.15 on the wrist camera |
| `policy.depth_lr` | null | inherit `optimizer_lr` exactly; separate group only excludes pretrained merge |
| `policy.norm_tag` | null | stats from the dataset |
| `policy.rtc_config` | enabled, `execution_horizon: 5` | RTC inference; the horizon is the frozen-prefix length, unrelated to `image_stride` |
| `torch_compile` | false | off for the first depth run |
