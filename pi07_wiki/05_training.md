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

$$\mathcal L = \underbrace{\mathcal L_{flow}}_{\|v_\theta - (a - \varepsilon)\|^2}
 + \underbrace{\mathcal L_{FAST}}_{\text{discrete action CE (+ z-loss)}}
 + \lambda_{gen}\,\underbrace{\mathcal L_{gen}}_{\text{subtask + summary CE}}$$

- **Flow loss** — [02 §3.2](02_base_model.md); joint per-layer forward (VLM K/V
  collected in the same pass), depth state threaded per layer, gradient
  checkpointing on, action-dim/chunk padding masked.
- **Discrete CE** — FAST action tokens with `action_mode: both` and knowledge
  insulation; logged as `loss_discrete_ce` / `loss_discrete_z`.
- **Generation CE** — separate forward on the generation prompt for annotated
  samples, separate backward accumulating into the same grads, weight
  `subtask_loss_weight = 1.0` ([04 §3.3](04_memory.md)).

Prompt conditioning per sample: subtask clause (70%), metadata clause (85%),
history clause (if `memory.history_keys` set, 70%). **No advantage clause** under
`skip_critic` (`advantage=None` in `build_training_batch`; `inference_advantage`
must stay null so eval matches).

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
    param.requires_grad = True   # fresh depth modules, trained from scratch
elif ".action_expert." in name:
    ...
else:
    param.requires_grad = False  # unknown actor param — freeze
```

This else-branch silently killed the depth gate once ([03 §B.4](03_depth.md)).
The same rule will apply to any future history-consumption module.

Optimizer groups (`get_optimizer_groups` + `_split_depth_group`): per-component
learning rates (ViT / connector / LLM / action expert) plus a dedicated **"depth"
group at `depth_lr = 5e-4`** for `pointmap_encoder` + `depth_stream` (gate and sink
included), excluded from the pretrained merge. Grad clip over actor params only.

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

Console + wandb via the `accum` dict → `log_metrics`: `loss_flow`,
`loss_discrete_ce`, `loss_subtask_ce`, `loss_summary_ce`, `pointmap_gate`,
`pointmap_gate_absmax` (the meaningful one), `pointmap_gate_grad_absmax`.
Validation probes at `val_freq` (offline_inference shows GT + predicted subtask
and memory per checkpoint; critic probes self-skip under skip_critic). Probes must
thread `cfg.policy.inference_advantage` — not a hardcoded advantage — so eval
prompts match training.

## 6. Config quick-reference (live values, root `config_rl.yaml`)

| Key | Value | Meaning |
|---|---|---|
| `dataset.repo_id` | `cijerezg/rebot-socks-annotated-v2` | fully annotated rebot dataset |
| `skip_critic` | true | plain BC + prompt conditioning |
| `policy.chunk_size` / `n_action_steps` | 50 / 50 | action chunk |
| `policy.action_encoding` | anchor | + `action_encoding_stats_path` |
| `policy.subtask_regeneration_interval` | 4.0 s | HL decode cadence = summary update window |
| `policy.subtask_max_new_tokens` | 128 | HL decode budget ("Memory: … Subtask: …") |
| `policy.subtask_loss_weight` | 1.0 | generation CE on |
| `policy.memory.metadata_enabled` | true | quality/mistake clauses from dataset meta |
| `policy.memory.history_keys` | state + top/wrist images + depth | short-term history on; all three channels consumed (smoke-verified 2026-07-22) |
| `policy.pointmap_config` | set | depth on; factory intrinsics; z ∈ [70, 800] mm |
| `policy.depth_lr` | 5e-4 | depth group LR |
| `policy.image_stride` | 5 | must match the memmap cache build |
| `policy.norm_tag` | null | stats from the dataset |
| `policy.rtc_config.enabled` | true | RTC inference |
| `torch_compile` | false | off for first depth run |
