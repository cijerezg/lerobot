# 04 — Memory & prompts

The π0.7 half of pi07: everything that conditions the model beyond the current
observation. Three channels — short-term history, long-term language memory
(subtask + MEM summary), metadata steering — all rendered as **prompt clauses** in
the policy's processor. Design rule: *transport structure, render late* — the
generic layers (buffer, gRPC, caches) carry tensors/strings/indices; prompt text is
rendered only inside `processor_molmoact2.py`. A clause whose data is `None` is
absent, and with everything off the prompt is byte-identical to the legacy model.

Literature anchors: π0.7 (arXiv 2604.15483 — history frames @ 1 s stride, history
dropout 0.3, component dropouts, inference metadata quality 5 / mistake false);
PI MEM (recurrent LLM-annotated summary memory); HAMLET (arXiv 2510.00695 —
causal-confusion caution: window/stride/dropout must be ablatable).

## 1. Prompt anatomy

### 1.1 Action prompt (`_build_robot_text`, [processor_molmoact2.py:236](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L236))

```
The task is to {task}.
[ The current step is {subtask}. ]                                   ← HL subtask
The current state of the robot is {discrete state string}.
[ The recent states of the robot, oldest to newest, were {s_1 … s_T}. ]  ← proprio history
[ Images i to j are earlier frames from the {cam} camera, oldest to newest. ]  ← frame history
[ The quality is {q} of 5. ] [ The robot made {a mistake | no mistakes}. ]     ← metadata
Given these, what action should the robot take to complete the task?
```

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

### 1.3 Training dropouts (π0.7 recipe, training text only)

Applied per sample in the pack step
([processor_molmoact2.py:1118-1128](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1118-L1128)):
`subtask_dropout = 0.3`, `metadata_dropout = 0.15`, `history_dropout = 0.3` — one
flip drops the **whole** short-term block (states and frames describe the same
window; dropping them independently would let the model exploit whichever
survived). On the generation side, `summary_dropout = 0.3` drops the memory clause
AND the answer's memory span together (the sample degrades to subtask-only).

## 2. Short-term history

Configured by the shared, model-agnostic `MemoryConfig`
([rl/shared_config.py](../src/lerobot/rl/shared_config.py)), embedded as
`memory:` on the policy RL config:

```python
history_keys: list[str] = []        # empty = disabled (zero behavior change)
history_window_seconds: float = 4.0
history_num_samples: int = 4        # 4 s @ 30 fps → offsets [30, 60, 90, 120]
history_dropout: float = 0.3
```

`history_offsets(fps)` converts once to per-key step offsets. Valid keys: any
observation key, `action` (executed actions), and the canonical depth key
`depth.{cam}.depth`.

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

### 2.3 Consumption (Phase 6 — partially built)

What the model actually sees today:

- **Proprio history** — consumed: each past state rendered as a discrete-state
  string in the "recent states" clause, with the same normalization as the current
  state (`_extract_history_states`,
  [processor_molmoact2.py:1008](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1008)).
- **Image history** — consumed: past frames appended as extra prompt images after
  the current cameras (pretraining layout keeps cameras first) with the
  "Images i to j are earlier frames" span clause
  (`_extract_history_images`, [processor_molmoact2.py:1025](../src/lerobot/policies/molmoact2/processor_molmoact2.py#L1025)).
  The sequence-length budget accounts for them.
- **Depth history** — consumed (built 2026-07-21): the pointmap encoder takes the
  `history.depth.{cam}.depth` window through the same per-patch CNN with learned
  per-slot time embeddings (past oldest→newest + current), concatenating
  `(T_h+1)·N` tokens into the DepthStream; `DepthPointmapConfig.history_num_samples`
  (0 = no new params, old checkpoints load) syncs from `memory.history_num_samples`.
  v1 does **not** re-project past frames into the current camera frame (the wrist
  moves) — slots are only time-marked; FK re-projection is the fallback if
  α-telemetry shows history unused.
- **Action history** — plumbed through the buffers but not consumed; whether past
  *actions* should be fed at all stays a causal-confusion ablation (candidates if
  a richer channel is ever wanted: MEM-style compression, HAMLET moment tokens, a
  DepthStream-style gated stream).

## 3. Long-term memory: subtask + MEM summary

### 3.1 The two-prompt design (decided 2026-07-13)

The **same network** is both policies. Every `subtask_regeneration_interval = 4 s`
the inference thread runs an HL decode: build the generation prompt from the live
observation (through the full pipeline — a `prompt_mode` toggle on the pack step
guarantees identical state normalization to the action path), greedy-decode up to
`subtask_max_new_tokens = 128` tokens, parse `Memory: … Subtask: …`, snap the
subtask to the vocab. Only **strings/indices** travel actor↔learner (no token
passthrough — kills the pi05 BOS-mismatch bug class). The subtask string enters
the next action prompts as "The current step is …"; the summary replaces
`RTCSharedState.current_summary` and conditions the *next* HL decode. Both are
cleared on episode restart, kept across interventions. The env worker stamps
`subtask_index` per transition, so online frames look exactly like annotated
offline frames to the learner.

Trainer entry points: `MolmoAct2Trainer.generate_subtask_text`
([rl_molmoact2_trainer.py:908](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L908)),
`MolmoAct2Policy.generate_subtask_tokens` (prefill + generic greedy loop, eos
stop). HL decode cadence and gating live in the RTC runtime
([rl/rtc_actor_runtime.py:473-549](../src/lerobot/rl/rtc_actor_runtime.py#L473)),
gated on `subtask_max_new_tokens > 0`. Generation MUST run in the inference thread
(the attention-capture concurrency lesson: the action-expert patches are not
thread-safe against a concurrent worker).

### 3.2 The MEM summary m_t

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
([rl_molmoact2_trainer.py:981](../src/lerobot/rl/molmoact2/rl_molmoact2_trainer.py#L981)):
slice the annotated samples (`subtask_index ≥ 0`), build the generation batch
through the full pipeline, and compute LM cross-entropy with labels on the answer
span only (last `answer_len` non-pad positions — padding-side agnostic;
unannotated samples fully masked):

$$\mathcal L_{gen} = -\sum_{j \in \text{answer span}} \log p_\theta\big(y_j \mid y_{<j},\ \text{generation prompt}\big)$$

backpropagated separately, weighted by `subtask_loss_weight` (config: 1.0), logged
as `loss_subtask_ce` / `loss_summary_ce` (the memory-prefix split of the span).
The vocab is wired from `meta/subtasks.parquet` by `sync_subtask_vocabulary` and
re-synced after additional datasets extend it via the remap.

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
