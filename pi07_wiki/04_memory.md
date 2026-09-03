# 04 — Memory & prompts

The π0.7 half of pi07: everything that conditions the model beyond the current
observation. Three channels — short-term history, language conditioning (subtask
generation; the MEM summary half was **removed**, §3), metadata steering — all
rendered as **prompt clauses** in the policy's processor. Design rule: *transport structure, render late* — the
generic layers (buffer, gRPC, caches) carry tensors/strings/indices; prompt text is
rendered only inside `processor_molmoact2.py`. A clause whose data is `None` is
absent, and with everything off the prompt is byte-identical to the legacy model.

Literature anchors: π0.7 (arXiv 2604.15483 — history frames @ 1 s stride, history
dropout 0.3, component dropouts, inference metadata quality 5 / mistake false);
PI MEM (arXiv 2603.03596 — video encoder w/ temporal attention every 4th layer +
recurrent LLM-annotated summary memory; 5 past frames @ 1 s pretrain, stretches
to 18 @ inference); HAMLET (arXiv 2510.00695 —
causal-confusion caution: window/stride/dropout must be ablatable).

## 1. Prompt anatomy

### 1.1 Action prompt (`_build_robot_text`, [processor_molmoact2.py:236](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L236))

```
The task is to {task}.
[ The current step is {subtask}. ]                                   ← HL subtask
The current state of the robot is {discrete state string}.
[ The recent states of the robot, oldest to newest, are: <T_h continuous positions> ]  ← proprio history (§2.4)
[ The quality is {q} of 5. ] [ The robot made {a mistake | no mistakes}. ]     ← metadata
Given these, what action should the robot take to complete the task?
```

Image history is **not in the prompt** — it enters through the MEM video encoder
(§2.4) at zero LLM-token cost. The proprio-history clause keeps its text lead-in,
but the values are continuous embeddings (one sequence position per past state),
not digit strings.

The pretraining-era setup/control clauses ("The setup is `<setup_start>…`",
"The expected control mode is …") were **removed 2026-07-22** — they existed to
route the multi-embodiment foundation model between robots and action spaces
(pretraining vocab in the foundation `norm_stats.json`: setup = "single {arm} in
{dataset}", control = "absolute joint pose" | "delta end-effector pose"). For a
single-embodiment fine-tune they carry no information, and neither control value
is even true for anchor-encoded joint actions. They had always rendered empty in
pi07 (empty config defaults, `norm_tag: null`); the removal deleted the machinery:
config fields, `_wrap_setup_text`/`_wrap_control_text`, the `_apply_norm_tag_metadata`
auto-fill. The pack step still *accepts* the four old kwargs as ignored fields —
saved processor configs from older checkpoints load as raw kwargs.

Note: the **summary memory is NOT in the action prompt** — it conditions the
generation prompt only (MEM-faithful; the action prompt and online transitions are
untouched by the summary seam).

### 1.2 Generation prompt (`_build_subtask_generation_text`, [processor_molmoact2.py:295](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L295))

Same visual/state context, different question, **memory clause in, action tokens
out**:

```
The task is to {task}. [ Memory: {m_t | "none yet."} ]
The current state of the robot is {…}. Given these, what step should the robot perform next?
```

Answer format is **memory-first** (flipped 2026-07-18 so the subtask conditions on
the fresh summary):

```
Memory: {m_{t+1}} Subtask: {subtask}<|im_end|>
```

`build_generation_answer` / `parse_generation_answer`
([processor_molmoact2.py:332-358](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L332-L358))
are exact inverses; a decode with no `Subtask:` marker is treated as subtask-only
(memory kept). Decoded subtask text is snapped to the annotation vocabulary by
`snap_to_subtask_vocab` (normalized exact match → difflib fuzzy at cutoff 0.6 → −1).

### 1.3 Training dropouts (optional, currently off)

Applied per sample in the pack step
([processor_molmoact2.py:1118-1128](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1118-L1128)):
all three rates are explicitly `0.0` in the active config. If history deletion is
enabled for an ablation, one flip drops the **whole** state/RGB/depth history block.
The rates are explicit config fields and are saved with the checkpoint.

## 2. Short-term history

Configured by the shared, model-agnostic `MemoryConfig`
([rl/shared_config.py](../src/lerobot/rl/shared_config.py)), embedded as
`memory:` on the policy RL config:

```python
history_keys: list[str] = []                       # empty = disabled (zero behavior change)
history_offsets_seconds: list[float] | None = None # THE window shape when set: [-6, -4, -2]
history_window_seconds: float = 6.0                # uniform fallback / derived span
history_num_samples: int = 3                       # uniform fallback / derived count
history_dropout: float = 0.0
```

`history_offsets_seconds` names the lookback instants directly (sign-agnostic,
normalized to magnitudes sorted oldest → newest) and is the source of truth when
set; `history_window_seconds` / `history_num_samples` are then recomputed from it as
its span and count, so the probes' "stale" lag and the prompt token budget stay
consistent. Left `None` it falls back to the uniform ladder those two generate —
which is how every checkpoint written before 2026-09-01 loads, since none of them
carries the list. The defaults spell the same window either way (6 s / 3 → 2 s
stride → -6, -4, -2); `config_rl.yaml` states the list explicitly so the choice is
legible in the saved checkpoint config rather than implied by two numbers.

(History: 4→5 past frames 2026-07-22 to match MEM's pretraining setup — 5 past +
current, 1 s stride. Narrowed to **3 frames at -6, -4, -2 s** 2026-09-01: at 30 fps
the -1 s frame is usually a near-duplicate of the current one, so it bought little
for its ViT time slice per camera, and the -6 s reach is free now that the corpus
packs that far back. 3 slots instead of 5 is ~40% less temporal attention and 2
fewer state-history positions per sample. The sinusoidal e(t) in §2.4 is
parameterized in real seconds, so spacing/count can be changed — or stretched at
inference — without touching a parameter shape; MEM shows 6-frame training
generalizing to 18 frames / 54 s.)

Because e(t) reads real seconds, the *instants* must reach it. They do:
`MolmoAct2RLConfig.__post_init__` syncs `history_times_seconds` onto the policy
config and the point-map config. Before 2026-09-01 nothing synced them and the MEM
video encoder stamped frames off `history_stride_seconds`, whose default 1.0 was
correct only because the window happened to be 5 s / 5 samples — any other spacing
was silently mis-stamped.

`history_offsets(fps)` converts once to per-key step offsets ([180, 120, 60] at
30 fps). Every offset on an image or depth key must be a multiple of `image_stride`
(those rows are the only ones the cache stores); 180/120/60 clear the stride-3 grid.
Valid keys: any observation key, `action` (executed actions), and the canonical
depth key `depth.{cam}.depth`.

### 2.1 Learner side — `ReplayBuffer._gather_history` ([rl/buffer.py:518](../src/lerobot/rl/buffer.py#L518))

History is a **`sample()` feature, not a dataset feature** (`rl_offline.py`
disables the LeRobotDataset delta machinery). For sampled index $i$ and offset
$k$: slot value = buffer entry $i - k$, with three clamps — backward episode
validity (done/truncated via cummax), buffer start, circular write head. Invalid
slots repeat the earliest valid frame of the episode and set the pad mask
(π0.7 pad rule). Emitted as `state["history.{key}"]` of shape $(B, T_h, \dots)$
oldest→newest plus `history.{key}_is_pad` $(B, T_h)$. Depth history is gathered
from `complementary_info` with uint16 preserved.

### 2.2 Actor side — `assemble_history_windows` ([rl/buffer.py:123](../src/lerobot/rl/buffer.py#L123)) + `RTCSharedState`

A rolling deque in the RTC shared state (`configure_history` / `push_history`,
[rl/rtc_actor_runtime.py:273-299](../src/lerobot/rl/rtc_actor_runtime.py#L273)):
the env worker pushes one (state, executed action, depth) entry per control step —
exactly the pair the learner buffer stores for that frame. Cleared on episode
restart, **not** on intervention (buffers keep contiguous frames across takeover).
`assemble_history_windows` reproduces the learner's clamp+mask semantics (parity
unit-tested value-for-value); the inference worker merges the windows into the
observation before `build_inference_batch`.

### 2.3 Consumption (redesigned 2026-07-22 → MEM architecture, §2.4)

- **Proprio history** — continuous state tokens (§2.4): each past state is
  linearly projected into the backbone embedding space, one sequence position per
  timestep, behind a text lead-in clause. The previous discrete-string rendering
  (`_extract_history_states` digit strings) is replaced; normalization (same
  transform as the current state) is unchanged.
- **Image history** — MEM video encoder (§2.4): past frames enter the ViT as
  extra time slices with temporal attention every 4th layer and are dropped
  before the LLM — zero extra LLM tokens. The prompt-image path (past frames as
  extra prompt images + "Images i to j are earlier frames" clause,
  `_extract_history_images`) is **deleted 2026-07-22**: 2 cams × 4 frames cost
  ~1,700 LLM tokens per step, paid again in KV at every decode. Checkpoints
  trained with that prompt format are orphaned by design.
- **Depth history** — consumed; **rebuilt 2026-07-25** to mirror the video
  encoder ([depth_history_design.md](depth_history_design.md); the 2026-07-21
  token-concat build predated the video-encoder decision and fed `(T_h+1)·N`
  tokens to the stream). Now: the `history.depth.{cam}.depth` window rides the
  shared patch CNN as extra batch rows; a `TemporalFusion` after each CNN block
  (same-pixel attention over all frames incl. the current one, sinusoidal
  e(Δt), MLP) fuses the past into the current frame; past rows are dropped
  before pooling, so the stream always sees `N = 192` tokens. The shared
  `history_images_mask` dropout draw masks the past keys (≡ missing window ≡
  cold deque). `DepthPointmapConfig.history_num_samples` (0 = no new params),
  `history_window_seconds` and `history_times_seconds` sync from `memory.*`; the
  e(Δt) buffer is non-persistent and no parameter is sized by $T_h$ (here or in the
  ViT), so changing the window does not invalidate a checkpoint. v1 does **not** re-project
  past frames into the current camera frame (the wrist moves) — frames carry
  only the e(Δt) stamp; FK re-projection stays the fallback.
- **Action history** — plumbed through the buffers but not consumed; whether past
  *actions* should be fed at all stays a causal-confusion ablation (candidates if
  a richer channel is ever wanted: MEM-style compression, HAMLET moment tokens, a
  DepthStream-style gated stream).

### 2.4 MEM video encoder + continuous state history (design 2026-07-22)

> **Corrected 2026-08-03.** This section originally specified ONE softmax over spatial
> and temporal keys together. MEM Appendix C (p. 15) instead defines two separate
> mechanisms — space-only and time-only — each with its own softmax, and the rebuild
> follows it. The operator, the papers read line by line, and the measurements that
> caught it are in
> [mem_temporal_attention_analysis.md](mem_temporal_attention_analysis.md).

From PI's MEM paper (arXiv 2603.03596). Decisions: encoder path is THE image
history path (prompt path deleted); continuous proprio-history tokens; 3 past
frames at -6/-4/-2 s + current (5 past @ 1 s until 2026-09-01); **no gate**;
repeat-padding v1 (pad mask not threaded into ViT
attention; a repeated first frame at episode start is approximately truthful —
masking is the contained fallback if early-episode telemetry looks off). The
$e(0)=0$ identity story no longer implies a bit-exact single-frame path: the
temporal sub-step fires on the query's own timestep even with no history
(decided 2026-08-03, see the analysis doc).

**Video encoder.** $K$ frames per camera (oldest → current, $t$ = seconds before
now, current $t=0$), $n=729$ patches per frame (crop_mode "resize": one 378×378
crop per camera — "same patch across time" is the same index in one grid). All
frames share the pretrained 27-layer ViT. Most layers are spatial-only, run
per-frame exactly as today. At every 4th resblock (0-indexed 3, 7, 11, 15, 19,
23), two changes:

1. Time stamp at the layer input (paper eq.; accumulates in the residual stream):

$$z^{p,t}_{l-1} \leftarrow z^{p,t}_{l-1} + e(t), \qquad e(0)=0$$

with $e$ sinusoidal in real seconds — this is what lets stride/count stretch at
inference.

2. Space-time **separable** attention, same $W_q, W_k, W_v, W_o$, two softmaxes.
A time-only step runs first, over the query patch's own position across time
(causal, its own timestep included):

$$\beta_{t \to t'} = \frac{\exp\big(q_{t,i} \cdot k_{t',i} / \sqrt{d}\big)}
{\sum_{s \le t} \exp\big(q_{t,i} \cdot k_{s,i} / \sqrt{d}\big)}, \qquad \sum_{t' \le t} \beta_{t \to t'} = 1$$

then its output goes through $W_o$ + residual and the block's ordinary spatial
attention runs per frame, untouched. Because the temporal softmax normalizes over
the $K$ timesteps alone, an indifferent read leaves $(K-1)/K$ of that step's mass
on history — not the $T/(n+T) \approx 0.7\%$ the past got when the two key sets
shared one denominator. Complexity $\mathcal{O}(Kn^2 + nK^2)$, not
$\mathcal{O}(K^2n^2)$.

After resblock 23 (the last temporal layer) past-frame rows are discarded; only
current-frame rows reach the feature taps (`vit_layers` [−9, −3] = resblocks 18,
24 — the tap at 18 collects while past rows are still in flight, so current rows
are sliced at concat time), pooling, projector, LLM. **Invariant: the
vision-backbone output shape and the LLM sequence are byte-identical to the
single-frame model** — past frames influence the output only through attention
reads. With $K=1$ the op is *literally* the pretrained layer ($e(0)=0$, empty
temporal key set) → bit-identity test, and the history-dropout sample (whole
short-term block, images + states together) degenerates to exactly the
pretrained path.

**Continuous state history.** Each past state (normalized with the current
state's transform) is projected by one shared linear layer to one sequence
position:

$$h_{t-k} = W s_{t-k} + b, \qquad s_{t-k} \in \mathbb{R}^7,\ W \in \mathbb{R}^{2560\times 7}$$

rendered behind a text lead-in ("The recent states of the robot, oldest to
newest, are:") as $T_h$ placeholder tokens whose embeddings are overwritten by
the scatter mechanism image patches already use
([modeling_molmoact2.py:242](../src/lerobot/policies/molmoact2/modeling_molmoact2.py#L242)).
Which timestep is which = sequence order. Three positions replace the digit-string
rendering of the three historical states, and the LLM gets full float precision instead of parsing
digits. $W$ is the **only new parameter in the whole build** → freeze-whitelist
+ optimizer-group entries required (the pointmap gate lesson). The current state
stays a text clause (pretraining format); the generation prompt keeps its §1.2
shape (no proprio-history clause) but HL decodes see image history automatically
through the shared encoder.

Not adopted from our own parked list: gates, HAMLET moment tokens,
DepthStream-style streams — MEM ships none of them. Build checklist: Phase 6 of
[archive/memory_build_plan.md](archive/memory_build_plan.md).

### 2.5 What probes the memory, and what the 2026-07-26 plan left behind

The criterion the memory probes are judged against has not changed: **the robot must
not redo what it just did**, and it should continue smoothly what it was doing. Not
reconstruction fidelity, not attention mass.

Registered in the validation loop (each runs only when its enable flag is set):

| probe | question | verdict shape |
|---|---|---|
| `mem_history_influence` | *which channel* — is the model helped by the contents of image and state history, or only by history being present? | $3\times3$ real/constant/foreign factorial at fixed flow noise, paired against `real/real`. The `constant/constant` MSE penalty is what information beyond the present is worth; `foreign/foreign` asks whether it is *this* trajectory's history. RMSE/max$|\Delta|$ gate usefulness claims; dropped `none` is legacy OOD context only |
| `mem_history_regime` | *how many frames* — does history help, does it hurt, and is either real? | $z$ on $\mathrm{MSE}(\mathrm{stale})-\mathrm{MSE}(\mathrm{full})$, the paired test against the same window taken $W$ s too early: below 2 the model reacted to *a* window rather than reading *this* one. Then helped/hurt fractions against $\tau=Q_{0.9}(\|\Delta\|)$ under a reseed alone. Exists because the usefulness bar above is a mean over a population that is half positive: on v6/1600, $+0.00067$ out of 47% helped and 53% hurt. The covariate ranking that used to answer *which* frames was removed 2026-08-10 — every $\rho$ sat at its own noise band and nothing was concluded from it |
| `mem_temporal_attention` | when and where is history read? | past-attention mass against the uniform-over-time baseline $T/(T+1)$ of the temporal softmax; enrichment ~1 means no selective read survives |
| `subtask_sweep` | does the subtask clause move the actions at all? | vocabulary spread over a same-clause seed floor; this was P3 of the plan doc, and it gates every claim that memory reaches behaviour |

The rest of [archive/memory_probes_plan.md](archive/memory_probes_plan.md) (retired
2026-08-02) was written against the long-term summary memory and its HL decode.
Repetition-at-boundaries (P1), closed-loop memory drift (P2), matched-state memory
swap (P4) and memory-span decode attention (P6) all scored a decode; nothing decodes
any more (§5). The update-vs-hold CE split (P5) was built, then went the same way.
Its fix-first list is done: joint names and counts are shared
(`probes/utils.joint_names_for_dim`), pack-step dropouts are held at zero for the
duration of a capture (`probes/utils.suppress_pack_dropout`), and frame assembly is
one helper (`probes/utils.probe_frame_inputs`).

Still open from it, deliberately: **nothing trends across checkpoints.** Every probe
writes PNG/JSON under `validation/step_XXXXXXXX/` and the viewer reads them per step;
no scalar reaches Aim, so "is this getting better" is answered by eye.

## 3. Long-term memory: subtask generation

> **The MEM summary memory was removed** (commit `f5f3b327`, "remove summary memory
> seam"). There is no summary in the prompt, no `materialize_summaries` on the
> buffer, no `loss_summary_ce`, and no `current_summary` on the RTC shared state —
> `grep -rn summary src/lerobot/policies/molmoact2/` returns nothing. §3.2 below is
> kept as the design record only; **§3.1 and §3.3 describe the live path, which is
> subtask generation alone.** Long-term memory is currently an open slot — see the
> ledger-memory idea in [08 §parked](08_status_roadmap.md).

### 3.1 The two-prompt design (decided 2026-07-13)

The **same network** is both policies. Every `subtask_regeneration_interval = 4 s`
the inference thread runs an HL decode: build the generation prompt from the live
observation (through the full pipeline — a `prompt_mode` toggle on the pack step
guarantees identical state normalization to the action path), greedy-decode up to
`subtask_max_new_tokens` tokens (**now 64**, was 128 when the answer also had to
carry a summary), parse the answer, snap the subtask to the vocab. Only
**strings/indices** travel actor↔learner (no token passthrough — kills the pi05
BOS-mismatch bug class). The subtask string enters the next action prompts as
"The current step is …", and is cleared on episode restart, kept across
interventions. The env worker stamps `subtask_index` per transition, so online
frames look exactly like annotated offline frames to the learner.

Trainer entry points: `MolmoAct2Trainer.generate_subtask_text`
([rl_molmoact2_trainer.py:908](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L908)),
`MolmoAct2Policy.generate_subtask_tokens` (prefill + generic greedy loop, eos
stop). HL decode cadence and gating live in the RTC runtime
([rl/rtc_actor_runtime.py:473-549](../src/lerobot/rl/rtc_actor_runtime.py#L473)),
gated on `subtask_max_new_tokens > 0`. Generation MUST run in the inference thread
(the attention-capture concurrency lesson: the action-expert patches are not
thread-safe against a concurrent worker).

### 3.2 The MEM summary m_t — REMOVED, design record only

> Nothing in this subsection is live. It describes the summary-memory seam that was
> built 2026-07-17, verified on real training 2026-07-19, and **deleted**. Kept
> because the copy-collapse failure mode and the hold/update supervision split are
> the things to get right if long-term memory is attempted again.

Long-term memory is a **recurrent text state, rewritten wholesale each tick** —
not a growing document whose segments partition time. At HL tick $t$ the model
reads $m_t$ (one self-contained free-form text, the *entire* memory) and emits
$m_{t+1}$, a fresh rewrite conditioned on $(m_t, \text{new observations})$:

$$m_{t+1} = \pi_{HL}(m_t,\ o_t) \qquad \text{— full replacement, never } m_{t+1} = m_t \Vert \Delta_t$$

Consecutive summaries are therefore only *semantically* related — $m_{t+1}$ may
condense, restructure, or drop content of $m_t$ entirely, and can look very
different from it. Compression is the point: the MEM ablation shows naive
concatenated history (our removed "done-list") performs much worse than the
compressed recurrent state. This holds at every seam, verified in code:
annotation writes each 12 s row as the full new memory given the old one; a
frame's conditioning/target index selects **one** row of `summaries.parquet`
(`_extract_summaries` renders single indices through `summary_texts` — rows are
successive states, never joined); rollout replaces `current_summary` wholesale
(a decode without a memory span keeps the old state; `-1`/"" = empty memory).
Trained with plain CE on the annotated target text. The summary is strictly
**retrospective**: what happened, never plans. A prospective "ledger" variant
(done / remaining) is a parked candidate innovation
([08 — Status](08_status_roadmap.md)).

Training labels come from `meta/summaries.parquet` (12 s annotation grid, 3:1 with
the 4 s subtask grid). `ReplayBuffer.materialize_summaries`
([rl/buffer.py:489](../src/lerobot/rl/buffer.py#L489)) writes two int64
complementary columns per frame:

- `summary_target_index` — the CE target: summary of **completed** segments (row
  $k-1$ for a frame in segment $k$; −1 = empty memory).
- `summary_prev_index` — the conditioning $m_t$: same as the target (**hold**
  pair), except the first `update_window_frames` of each segment (= one HL tick,
  `subtask_regeneration_interval × fps` = 120 frames) condition one summary older
  (**update** pair — the window where appending is learned).

The hold/update rule lives once in `summary_label_spans`
([rl/buffer.py:93](../src/lerobot/rl/buffer.py#L93)) and is shared with the
offline-eval probe. Indices resolve into one concatenated text table across
main + additional datasets (`index_offset`), synced into the pack step via the
`sync_summaries` hook. The prompt asymmetry — conditioning clause shows $m_t$,
answer span contains $m_{t+1}$ — is what makes the decode an *update*, not a copy.

### 3.3 Generation CE loss

`_subtask_generation_loss`
([rl_molmoact2_trainer.py:1113](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L1113)):
slice the annotated samples (`subtask_index ≥ 0`), build the generation batch
through the full pipeline, and compute LM cross-entropy with labels on the answer
span only (last `answer_len` non-pad positions — padding-side agnostic;
unannotated samples fully masked):

$$\mathcal L_{gen} = -\sum_{j \in \text{answer span}} \log p_\theta\big(y_j \mid y_{<j},\ \text{generation prompt}\big)$$

backpropagated separately, weighted by `subtask_loss_weight`, logged as
`loss_subtask_ce`. With the summary gone the answer span is the subtask alone, so
there is no memory-prefix split and no `loss_summary_ce`.

**`subtask_loss_weight` is currently 0.0 — this loss is off.** The forward still
runs whenever the weight is > 0; note that under `depth_warmup` the generation
prompt carries no depth clause, so the loss touches nothing trainable and the
backward is skipped (`requires_grad` check) while the metric is still logged as a
frozen-model baseline.

The vocab is wired from `meta/subtasks.parquet` by `sync_subtask_vocabulary` and
re-synced after additional datasets extend it via the remap.

The `/hold` vs `/update` split described in §3.2 went away with the summary; if
long-term memory returns, that split is the part worth rebuilding first, because
the pooled curve hides copy-collapse.

## 4. Metadata steering

π0.7's headline result: without metadata in the prompt, adding lower-quality data
*degrades* the model; with it, more data keeps helping. Train on everything with
truthful labels, prompt for the best at inference.

Design (revised 2026-07-18):

- **Quality** — per-episode integer 1–5, scored by a human in the review UI.
- **Mistake** — boolean per 4 s subtask window. Produced by a recall-tuned LLM
  suspicion pass (evidence-first score 0–10, thresholded at review time) followed
  by human confirm/reject of flagged windows only; unflagged = clean by definition.
- **Speed** — **omitted**: single-operator data; pace variation is grasp fumbling,
  which the mistake channel already carries. The clause renders partially and the
  extractor tolerates the missing column; backfillable later.

Storage mirrors the summaries pattern (window-range parquets, no dataset rewrite):
`meta/episode_metadata.parquet`, `meta/mistakes.parquet`, `meta/metadata_info.json`.
Loading is hard-error (`load_metadata_rows`); `ReplayBuffer.materialize_metadata`
([rl/buffer.py:461](../src/lerobot/rl/buffer.py#L461)) broadcasts to per-frame
`metadata_quality` / `metadata_mistake` columns; `_extract_metadata`
([processor_molmoact2.py:1047](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1047))
turns them into per-sample dicts (an explicit `metadata` dict — the rollout/eval
path — always wins). Inference prompts `{quality: 5, mistake: false}`
([rl/rtc_actor_runtime.py:475-480](../src/lerobot/rl/rtc_actor_runtime.py#L475)).
Gated by `memory.metadata_enabled` (config: true). Live online transitions carry no
metadata by design (episode outcome is unknowable mid-episode; the 15% dropout
covers the mixed regime) — recorded rollouts get labeled post-hoc.

Cheap upgrade parked: classifier-free guidance on the metadata clause at inference
(π0.7 uses β 1.3–2.2).

## 5. What was removed

- **Done-list** (built 2026-07-13, removed 2026-07-15): fill-time
  `materialize_done_lists`, `done_list_ids` column, prompt clause, RTC bookkeeping —
  fully deleted in favor of the MEM summary. The summary IS the budget mechanism.
- **Speed metadata** — omitted (above).
- **Subtask token passthrough** (pi05 plan) — obsoleted by the string-level design.
