# Memory probes — current state and proposals

**RETIRED 2026-08-02**, moved here from the repo root. Superseded by
[04 — Memory §2.5](../04_memory.md), which records what shipped and what is still
open. Read this only for the reasoning behind the proposals: P3 became
`probes/subtask_sweep.py` and P5 was built and then dropped, while P1, P2, P4 and P6
all score a language decode and died with the summary memory on 2026-08-01. The
defects listed in §3 and the fix-first list in §6 are fixed.

Status when written: v1 narrowed for the next training run, 2026-07-26.

## 0. Immediate v1 decision

Do not build a broad memory benchmark before seeing the next training run. Improve the
two probes that already run and react to their outputs:

1. `mem_history_influence.py` keeps its existing four action forwards per frame, but
   packs a real no-history condition and reports whether each history condition improves
   normalized GT action MSE. Influence says the channel is used; positive GT-MSE
   improvement says the use was helpful. Its overview adds a per-frame effect scatter,
   and `examples/` shows the actual history/current images plus GT/full/none/images/states
   action traces for the most helpful, most harmful, and strongest-effect frames.
2. `offline_inference.py` runs with configured frame history plus the GT subtask and
   rollout metadata in the action prompt. On frames with nonempty GT language memory it
   adds one cheap HL decode without memory and writes `memory_ablation.json`: GT-memory
   versus empty-memory subtask accuracy, accuracy delta, and changed-decode fraction.
   `memory_ablation.png` is the qualitative contact sheet: both current camera images,
   input memory, full raw decode, and correct/wrong marking for both conditions.
3. The offline action panel now plots and correctly labels all seven ReBot joints.
4. `mem_temporal_attention.py` retains compact per-frame/layer/camera/head/age/patch
   summaries from the same prefix forward. Its overview shows distributions across
   frames, scene dependence, history-age preference, head/camera specialization, and
   age entropy against the union-softmax uniform-key baseline. `examples/` shows
   low/median/high temporal-read frames with the most-read history image and a spatial
   overlay of which current-image patches read history. The underlying arrays are saved
   in `temporal_attention_data.npz` for follow-up analysis without another model run.

For the next run, the useful readout is deliberately small:

- influence near zero → the memory channel is not reaching the output;
- influence nonzero but GT-MSE improvement negative → it is used in the wrong direction;
- GT-memory subtask accuracy above empty-memory accuracy → language memory is helping;
- changed-decode fraction without an accuracy gain → language memory affects output but
  is not yet useful.

Sections 5–8 are a parking lot, not the implementation plan. Do not add those probes
until the basic signals from a real run justify one of them.

## 1. What memory is for

The robot must not redo what it just did. That is the criterion every probe below is
judged against — not reconstruction fidelity of the summary text, not attention mass,
not action RMSE in the abstract.
It also should help the robot smoothly continue what it was just doing. It's basically the same objective as short-term and
long-term memory is used for in humans and animals.

## 2. The chain being probed

```
obs + frame history ──► generation prompt (carries m_t) ──► decoded (m_{t+1}, subtask)
                                                                     │
                                                    subtask clause ──┘
                                                                     ▼
                                          action prompt ──► action chunk
```

Invariant that shapes everything: **m_t never enters the action prompt.**
`_build_subtask_generation_text` (`processor_molmoact2.py:267`) is a separate prompt from
`_build_robot_text`. Long-term memory reaches behavior only through the decoded subtask,
so there are two hops and either can be dead independently. A probe that measures only
the end-to-end effect cannot tell them apart.

## 3. What exists today, and why it is not enough

| probe | computes | output | limitation |
|---|---|---|---|
| `mem_history_influence.py` | 4 history conditions (full/none/images/states), seeded flow noise, chunk delta vs `none` | mean RMSE + max abs delta, one bar chart | does not touch m_t at all — only short-term frame history |
| `mem_temporal_attention.py` | temporal-read distributions by frame/layer/camera/head/history age/current-image patch | six-panel overview, low/median/high spatial examples, JSON + NPZ arrays | mechanistic rather than behavioral; now useful for locating when and where history is read, but cannot show that the read improves actions |
| `offline_inference.py` | 8 frames, action MSE + decoded text, GT memory injected | one MSE number + per-frame panels | see below |
| `loss_summary_ce` (training) | CE on the memory span | one curve | ~80% of it is copying, see §4 |

Specific defects, all confirmed in code:

- **`mem_history_influence` baseline is off-distribution.** `_variant_batch:33` drops
  `history_state_values` but leaves the placeholder tokens in the prompt carrying their
  base embedding. Training instead removes the clause entirely
  (`processor_molmoact2.py:1131`), so the `none` condition is a prompt the model has
  never seen. The docstring accepts this to keep sequence length identical; it costs the
  interpretation.
- **Scale-free outputs.** `full RMSE = 0.14` is in normalized anchor-delta space with no
  noise floor, no per-joint split, no units, averaged over frames where memory is
  irrelevant by construction.
- **`offline_inference` MSE is not comparable to anything.** Raw deg^2 over 50 steps x 7
  joints x 8 frames, ~60% of it gripper (sigma 76 deg vs 11 deg for wrist_yaw), and the
  observed 3611 sits *above* the predict-the-dataset-mean baseline (~1470). Its prompt
  also carries no subtask, metadata, or history clause — a regime that occurs in ~1.35%
  of training samples and is not the deployment regime either.
- **Live bug:** `render_sample` loops `min(n_joints, 6)` against a 6-entry
  `SO100_JOINT_NAMES` (`offline_inference.py:52`, `:264`). On the 7-dim rebot the gripper
  is never plotted and dims 4/5 are mislabeled (wrist_yaw shown as "wrist_roll",
  wrist_roll as "gripper").

## 4. Facts any new probe must respect

Dataset (`rebot-socks-annotated-v2`, 6 episodes, 29192 frames):

- 88 subtask transitions, ~15 per episode, mean run 331 frames = 11.1 s.
- 81 summary segments of 360 frames (12 s), 50 unique texts, mean 94 chars.
- Hold/update split (`summary_label_spans`, `buffer.py:93`) with a 120-frame update
  window: 33.3% update frames, but 40.7% of those have identical consecutive text, so
  only **19.7% of frames have a target that actually changes**. Hence "CE is ~80%
  copying".

Pipeline:

- Everything is teacher-forced. Training and `offline_inference` both inject GT m_t.
  Nothing measures self-generated memory, which is what rollout consumes for minutes.
- `val_dataset_path` == the training dataset (`config_rl.yaml:52`), and 6 episodes are
  memorized by step 140 (`loss_subtask_ce` 7.84 -> 0.0055). Any probe answer that could
  be recall must be treated as provisional until held-out episodes exist.
- `adapter.predict_action_chunk` (`base.py:103`) takes no generator and
  `per_episode_seed` is false, so it draws from the global RNG. Any A/B through it is
  noise-confounded. `mem_history_influence:80` sidesteps this by calling
  `adapter.policy.predict_action_chunk(..., generator=gen)` directly; new probes must do
  the same.
- `adapter._make_batch` (`adapters/molmoact2.py:80`) passes only obs + task, so a probe
  cannot currently inject a chosen subtask or metadata clause into the action prompt.
  Needed by P3. The pack step already reads `"subtask"` strings from complementary data
  (`processor_molmoact2.py:954`), so this is a small addition.
- `summary_dropout` 0.3 / `history_dropout` fire whenever a subtask name is present
  (`processor_molmoact2.py:881`). Teacher-forced probes must zero them and restore in a
  `finally`.
- `step.summary_texts` is synced only on the rl_offline path (`rl_offline.py:568`).
  Standalone probe runs silently lose the memory clause; guard or hard-error.

## 5. Proposals

### P1 — Repetition at subtask boundaries (primary)

Question: after a subtask completes, does the model move on, or re-emit the subtask it
just finished — and does memory make that difference?

Sample set: the first K frames after each of the 88 subtask transitions. This is exactly
where the image is least informative: the frame just after "grasp the white sock"
completes looks much like the frame just before, so only memory can say it is done.

Conditions per frame, identical seeded flow noise: GT memory / empty memory
("none yet.") / no memory clause.

Outputs:
1. Repetition rate per condition = fraction of boundary frames whose decoded subtask
   equals the just-completed one.
2. Per-frame panel: both camera images, the memory text in full, the decoded subtask
   under each condition marked against the GT next subtask, and the predicted chunk per
   joint in degrees for each condition with GT overlaid.

Verdict: memory does its job iff the repetition rate drops materially from empty to GT
memory. Equal rates = memory decorative. The panels show what the difference looks like
in the arm: returning to the sock versus moving to the next one.

Cost: 88 boundaries x 3 decodes + 3 chunks, subsampled to the configured episode count.

### P2 — Closed-loop memory drift

Question: does self-generated memory survive an episode?

Method: run one episode end to end, decoding (m_{t+1}, subtask) every
`subtask_regeneration_interval`, feeding the model's own memory forward. No GT injection
after the first update.

Output: a memory tape — decoded m_t against the GT summary per 12 s segment, aligned on
one time axis, marking divergence, recovery, and degeneration (repetition/looping);
below it, decoded subtask sequence vs GT subtask sequence.

Verdict: this is the deployment condition and nothing currently measures it. Divergence
within N updates predicts rollout failure regardless of how good the teacher-forced CE
looks. Works on training episodes — drift is drift even on memorized data.

Cost: ~12 decodes per episode. The cheapest proposal here.

### P3 — Subtask to action transfer (prerequisite)

Question: does the subtask clause change the actions at all?

Method: fix a state, sweep all 13 vocabulary subtasks through the action prompt, plot the
13 chunks per joint.

Verdict: if they coincide, hop two is dead, and P1/P2 are measuring a chain that cannot
reach behavior. Run this first — it is small and it gates the interpretation of
everything else.

Blocked on: complementary-data injection in `_make_batch` (§4).

### P4 — Matched-state memory swap (defer until held-out data)

Mine near-duplicate observations with different GT subtasks (nearest neighbours in
encoder representation space), cross-swap their memories, and ask whether the decode
follows the memory or the image. Sharpest test of the mechanism, but on memorized
training episodes a correct answer may be recall, so it needs held-out episodes.

### P5 — Update-vs-hold CE split (training metric, not a probe)

Split the logged memory CE three ways: hold spans, update spans where the text is
unchanged (40.7% of updates), update spans where it truly changes (19.7% of frames).
Only the third is evidence that appending is learned. Small change — the loss already
splits memory vs subtask in its `parts` dict.

### P6 — Memory-span attention during decode (mechanistic complement)

Attention mass on the memory clause tokens while decoding the subtask, against mass on
image tokens. Cheap given the existing capture machinery. Distinguishes "reads the memory
and then ignores it" from "never reads it" — a distinction P1 cannot make on its own.

## 6. Fix-first list

- `offline_inference` joint count/names on the 7-dim rebot (live bug, misreads every
  existing plot).
- Zero the pack-step dropouts inside probes; restore in `finally`.
- Guard against empty `step.summary_texts` in standalone runs.
- Shared hold/update/boundary tagging helper (extend `_summary_lookup`,
  `offline_inference.py:320`, into `probes/utils.py`) — needed by P1, P2, P5.

## 7. Proposed order

P3 (gates the rest) -> P1 (the criterion in §1) -> P2 (the deployment condition) -> P5,
with P4 and P6 after the next recording session.

## 8. Open questions for review

- Boundary window K for P1: the whole update window (120 frames) or a tighter ~1 s?
- Repetition scoring: snap the decode to the 13-entry vocabulary
  (`snap_to_subtask_vocab`) or exact string match on the raw decode?
- Should probes return scalars for wandb trend lines? Currently everything lands as
  PNG/JSON under `validation/step_XXXXXXXX/` and nothing is trended across steps
  (`rl_offline.py:208` passes no logger).
- Which of these should run at every `val_freq` versus on demand — P1 and P2 are cheap,
  P4 is not.
