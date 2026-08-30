# RL Infrastructure Notes

Generic value-based RL training for LeRobot.  
Supports MolmoAct2 and PI05.  All model-specific logic is isolated behind the `Trainer` ABC.

---

## Architecture

```
config_rl.yaml
      │
      ▼
Trainer.for_config(cfg)          ← dispatches to MolmoAct2Trainer or PI05Trainer
      │
      ├── make_policy()          ← loads HF checkpoint + applies freeze schedule
      ├── make_processors()      ← (pre, post) processor pipeline
      ├── freeze_model()         ← actor frozen per config; critic unfrozen in RL mode
      ├── get_optimizer_groups() ← [actor group, critic group] with per-group LRs
      │
      ├── update_critic()        ← HL-Gauss distributional TD, Polyak target update
      ├── compute_advantage()    ← r + γV'(s') − V(s), tanh squash
      ├── build_training_batch() ← model-specific batch assembly (subtask inject for PI05)
      ├── actor_forward()        ← policy loss; advantage remains prompt conditioning only
      ├── update_actor()         ← full actor gradient step
      ├── update_target_networks()
      │
      ├── build_inference_batch() ← model-specific obs → tokenised batch for select_action
      ├── push_weights()          ← serialize trainable params → actor queue
      └── log_metrics()           ← Aim scalar logging
```

### Offline loop (`scripts/rl_offline.py`)
```
rl_offline.py
  └── Trainer.for_config()
        ├── make_policy / freeze_model / init_critic (if !skip_critic)
        ├── make_processors / get_optimizer_groups
        └── loop:
              update_critic (UTD−1 extra) + update_target_networks
              update_actor  (if step ≥ critic_warmup and step % policy_update_freq == 0)
              apply_pretrained_merges (if enabled)
```

### Online loop
```
rl_actor_async.py  ←gRPC→  rl_learner.py
      │                           │
  rtc_actor_runtime          Trainer.for_config()
  (ActionQueue / RTC)        update_critic / update_actor
  env_worker                 push_weights → parameters_queue
  → transitions_queue
  → interactions_queue
```

---

## Files

| File | Purpose |
|------|---------|
| `rl_trainer.py` | Abstract `Trainer` base class + `for_config()` dispatch |
| `rl_pi05_trainer.py` | `PI05Trainer` — thin wrapper over existing `pi05_train_utils.py` |
| `rl_molmoact2.py` | `MolmoAct2RLConfig`, `MolmoAct2Critic`, and `MolmoAct2RLPolicy` |
| `rl_molmoact2_trainer.py` | `MolmoAct2Trainer` — all abstract methods implemented |
| `scripts/rl_offline.py` | Generic offline training loop (actor-only and critic-trained) |
| `rl_actor_async.py` | Generic online actor entrypoint; RTC is the default runtime |
| `rtc_actor_runtime.py` | Generic RTC `ActionQueue` actor runtime ported from tested PI05 path |
| `rl_learner.py` | Generic online learner |
| `config_rl.yaml` | Unified config for both models (offline + online sections) |
| `inference_async.py` | Standalone VLA inference (no learner, no gRPC) |

**Unchanged PI05 files** (still active, not replaced):  
`actor_pi05_async.py`, `learner_pi05.py`, `pi05_train_utils.py`, `inference_pi05_async.py`

---

## MolmoAct2 Critic — Design

- Full deepcopy of actor's ViT + adapter (vision_backbone).
- First `critic_llm_depth` (default 12) text transformer blocks from the actor.
- Learnable value queries `[1, num_value_bins, 2560]` appended to token sequence.
- Bidirectional 4D attention (no causal mask).
- `bin_logit_head Linear(2560 → 1)` per query → `[B, num_bins]` logits.
- **HL-Gauss** soft target: Gaussian CDF over bin edges.
- Critic parameters inherit `requires_grad=False` from the frozen backbone deepcopy.  
  `freeze_model()` explicitly calls `requires_grad_(True)` on all critic params after `init_critic()`.

Key config fields (in `MolmoAct2RLConfig`):
```yaml
critic_llm_depth: 12
num_value_bins: 101
value_support_min: -2.0
value_support_max: 0.0
hl_gauss_sigma_ratio: 5.0
critic_lr: 1.0e-4
critic_target_update_weight: 0.005   # Polyak τ
discount: 0.97
advantage_scaling: 0.2
```

---

## Subtask critic: release folds into move

**Status:** active since 2026-08-29. Applies whenever `critic_reward_mode: subtask`.

### What the subtask critic measures

With `critic_reward_mode: subtask` the critic is a *duration* model and nothing
else: reward is `-1` per TD transition (one `chunk_size`-frame step, 1.0 s at 30 Hz),
`0` at a terminal, normalized by `reward_normalization_constant`. A transition is
terminal when a subtask boundary falls anywhere inside its chunk window, so
`V(s) ≈ -(discounted seconds until the current subtask ends)`. Terminal value is `0`
regardless of *how* the segment ended — duration is the critic's only expressive
channel. Two consequences follow, and both drove the change below:

- The last `chunk_size` frames of **every** segment are terminal with target exactly
  `0`. Short segments are mostly, or entirely, this flat region.
- Segment lengths must be roughly homogeneous, or the critic spends its capacity on
  a bimodal duration distribution rather than on within-segment progress.

### The change

A `release ...` window is **not** treated as a subtask of its own. Its boundary with
the `move ...` that precedes it is dropped, so the critic sees one
move-through-release segment whose terminal lands where the object is actually let go.

Implemented in `_subtask_terminals_from_windows` in `rl/offline_dataset_utils.py`,
gated on `CRITIC_CONTINUATION_VERBS = ("release",)`. Only an *immediately adjacent*
window folds — a frame gap means the two are not one continuous behaviour and both
boundaries are kept.

**Annotations are not modified.** `subtask_windows.json` is untouched, and
`subtask_index` — the policy's language conditioning — still labels the release
window separately, so `"release the black sock in the basket"` remains a distinct
instruction the policy is conditioned on. `_subtask_indices_from_windows` and
`_subtask_terminals_from_windows` read the same file independently; only the latter
coarsens. This is the whole point of the design: the critic's horizon and the
policy's instruction granularity do not have to agree.

### Why (measured over the five annotated roots, 2026-08-29)

Release is the outlier in a distribution that is otherwise homogeneous:

| verb | median duration |
|---|---|
| grasp | 231–425 f |
| move | 202–258 f |
| return to home | 232–335 f |
| **release** | **65–100 f** |

At 65–100 frames against a 30-frame terminal window, roughly 40% of a release sits in
the flat `V=0` region. Folding it into its move:

| root | segments | median | min | terminal frac |
|---|---|---|---|---|
| socks_basket | 261 → 179 | 247 → 348 f | 22 → 63 | 11.4% → 7.8% |
| two_container | 299 → 204 | 208 → 298 f | 9 → 26 | 13.5% → 9.2% |
| shirts_bin | 138 → 95 | 206 → 296 f | 8 → 64 | 13.4% → 9.2% |
| val | 76 → 52 | 184 → 319 f | 22 → 68 | 14.4% → 9.9% |
| sorting_clothes_v4 | 59 → 42 | 272 → 363 f | 56 → 195 | 11.3% → 8.0% |

The fold rule is unambiguous in the data: all 261 release windows across the five
roots are preceded by an immediately adjacent `move`, none opens an episode.

**Value support still fits.** Post-fold, over 572 segments: median 322 f
(`V = -0.77`), p90 522 f (`V = -1.14`), max 1061 f (`V = -1.83`). At `discount: 0.97`
and `reward_normalization_constant: 12.0`, the `[-2, 0]` floor is reached at ~1254
frames, so nothing clips. Recheck this if either constant, `chunk_size`, or the
annotation granularity changes — the asymptote is `-1/((1-γ)·N) = -2.78`, so the
headroom is not large.

### Open: `critic_mistake_penalty` likely double-counts

Recorded here because it came out of the same investigation, but **not** changed yet.

A mistake does *not* spawn a new subtask in these annotations. The vocabulary is
strictly `grasp / move / release / return to home` (19 labels, no recovery verb);
per-verb counts are balanced and there are **zero** consecutive same-label repeats in
any root. A failed grasp stays *inside* its grasp window and inflates its duration:

| root | grasp w/ flagged mistake | grasp clean |
|---|---|---|
| socks_basket | 615 f (n=20) | 371 f (n=62) |
| two_container | 616 f (n=12) | 270 f (n=83) |
| shirts_bin | 380 f (n=12) | 231 f (n=31) |

So the retry already costs 250–350 extra frames ≈ 8–11 chunk-steps ≈ `-0.7` to
`-0.95` normalized, and *proportionally to how long the recovery actually took*.
`critic_mistake_penalty: 5.0` then subtracts a further `-0.42` at one arbitrary
transition on top of that. Worth an ablation at `0.0`.

If recovery-shaped subtasks ("reset grasp" and similar) do show up, they are coming
from the rollout HL decode, not the data: `subtask_loss_weight: 0.0` leaves the
generation head untrained, so it emits free-form text off the 19-label vocabulary
every `subtask_regeneration_interval` seconds. That is a train/rollout segmentation
mismatch and a separate problem from segment granularity.

### If you revert

Set `CRITIC_CONTINUATION_VERBS = ()`. The terminal derivation returns to one marker
per annotated window with no other behavioural change. Tests:
`tests/rl/test_offline_dataset_utils.py::test_release_window_folds_into_the_preceding_move_for_the_critic`
and `::test_release_window_after_a_gap_keeps_both_boundaries`.

---

## Actor / Learner — Online RL

### `rl_actor_async.py`
- Default runtime uses `rtc_actor_runtime.py`: `ActionQueue`, latency-aware replanning, intervention resets, and smooth execution.
- `trainer.build_inference_batch()` is the model-agnostic isolation point:
  - **MolmoAct2**: calls preprocessor, returns `{input_ids, pixel_values, ...}`
  - **PI05** (future): injects subtask tokens + advantage into complementary_data
- `policy.select_action(batch)` is called `chunk_size` times per chunk:
  first call runs model + caches; subsequent calls pop from cache (MolmoAct2 behaviour).
- The old simple chunk-deque runtime remains as a debug fallback. PI05-specific `actor_pi05_async.py` remains as a reference until generic RTC is validated on robot.

### `rl_learner.py`
- Identical loop structure for any registered model.
- `trainer.push_weights()` sends only trainable params (`requires_grad=True`) to actor.
- Weight push interval: `cfg.policy.weights_push_interval` (default 180 s).
- Pretrained merges are supported via `pretrained_merge_alpha`, `pretrained_merge_every_n_steps`, and `pretrained_merge_targets`.
- Additional offline datasets are supported via `dataset.additional_offline_dataset_paths`; they are merged into the offline replay buffer with subtask-index remapping when metadata is available.
- Supports offline buffer mix (half online / half offline batches when `cfg.dataset` set).

---

## Running

### Offline Actor-Only (MolmoAct2, skip_critic: true)
```bash
cd lerobot
uv run python -m lerobot.rl.rl_offline \
    --config_path src/lerobot/rl/config_rl.yaml
```

### Offline Critic-Trained / RECAP (skip_critic: false)
Edit `config_rl.yaml`: set `skip_critic: false`, then same command.

### Online (distributed)
```bash
# Learner (runs on GPU machine with dataset)
uv run python -m lerobot.rl.rl_learner \
    --config_path src/lerobot/rl/config_rl.yaml

# Actor (runs on robot machine)
uv run python -m lerobot.rl.rl_actor_async \
    --config_path src/lerobot/rl/config_rl.yaml
```
Make sure `actor_learner_config` is uncommented in `config_rl.yaml` and `learner_host` / `learner_port` point to the learner machine.

---

## TODO

### Immediate (before first run)
- [ ] **Smoke test offline actor-only** — run `scripts/rl_offline.py` with `skip_critic: true`, verify flow loss decreases over 500 steps.
- [ ] **Smoke test RECAP** — run with `skip_critic: false`, verify critic CE loss decreases and `critic_value_mean` moves away from init.
- [ ] **PI05 regression** — run `scripts/rl_offline.py` with `policy.type: pi05_rl`, verify same loss curve as original `learner_pi05.py`.

### Short term
- [ ] **RTC for MolmoAct2** — implement `predict_action_chunk` + `ActionQueue` support in `MolmoAct2Policy`/`MolmoAct2RLPolicy`.  Once done, `rl_actor_async.py` can route through RTC for both models and `actor_pi05_async.py` can be retired.
- [ ] **`rl_pi05_trainer.py` full implementation** — currently a thin stub that delegates to `pi05_train_utils.py`.  Flesh out so PI05 can run through `scripts/rl_offline.py` and `rl_learner.py` fully.
- [ ] **`inference_async.py` cleanup** — strip gRPC transport; make it pure standalone inference only (the distributed path now lives in `rl_actor_async.py`).
- [ ] **`config_rl.yaml` — fill in `actor_learner_config`** with real IPs for the lab machines.

### Longer term
- [ ] **Unified actor** — once RTC lands for MolmoAct2, merge `actor_pi05_async.py` logic into `rl_actor_async.py` and retire the PI05-specific file.
- [ ] **`rl_pi05_trainer.py` — `build_inference_batch`** — implement subtask token injection + advantage for PI05 so the generic actor works for PI05 without RTC too.
- [ ] **Online training run** — full HILSERL loop: learner + actor on robot, verify policy improves over episodes.
