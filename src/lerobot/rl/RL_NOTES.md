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
      ├── actor_forward()        ← flow-matching loss (+ advantage weighting in RL mode)
      ├── update_actor()         ← full actor gradient step
      ├── update_target_networks()
      │
      ├── build_inference_batch() ← model-specific obs → tokenised batch for select_action
      ├── push_weights()          ← serialize trainable params → actor queue
      └── log_metrics()           ← W&B scalar logging
```

### Offline loop (`rl_offline.py`)
```
rl_offline.py
  └── Trainer.for_config()
        ├── make_policy / freeze_model / init_critic (if !skip_critic)
        ├── make_processors / get_optimizer_groups
        └── loop:
              update_critic (UTD−1 extra) + update_target_networks
              update_actor  (if step ≥ critic_warmup and step % policy_update_freq == 0)
```

### Online loop
```
rl_actor_async.py  ←gRPC→  rl_learner.py
      │                           │
  inference_worker           Trainer.for_config()
  (chunk-deque, no RTC)      update_critic / update_actor
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
| `rl_molmoact2.py` | `MolmoAct2RLConfig` + `MolmoAct2RLPolicy` (critic lifecycle) |
| `rl_molmoact2_trainer.py` | `MolmoAct2Trainer` — all abstract methods implemented |
| `rl_molmoact2_critic.py` | `MolmoAct2Critic` — distributional value critic |
| `rl_offline.py` | Generic offline training loop (BC and offline RL) |
| `rl_actor_async.py` | Generic online actor (chunk-deque, no RTC yet) |
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

## Actor / Learner — Online RL

### `rl_actor_async.py`
- Two threads: `inference_worker` (chunk-deque) + `env_worker`.
- `trainer.build_inference_batch()` is the model-agnostic isolation point:
  - **MolmoAct2**: calls preprocessor, returns `{input_ids, pixel_values, ...}`
  - **PI05** (future): injects subtask tokens + advantage into complementary_data
- `policy.select_action(batch)` is called `chunk_size` times per chunk:
  first call runs model + caches; subsequent calls pop from cache (MolmoAct2 behaviour).
- No RTC / ActionQueue for now. PI05 online RL continues to use `actor_pi05_async.py`.

### `rl_learner.py`
- Identical loop structure for any registered model.
- `trainer.push_weights()` sends only trainable params (`requires_grad=True`) to actor.
- Weight push interval: `cfg.policy.weights_push_interval` (default 180 s).
- Supports offline buffer mix (half online / half offline batches when `cfg.dataset` set).

---

## Running

### Offline BC (MolmoAct2, skip_critic: true)
```bash
cd lerobot
uv run python -m lerobot.rl.rl_offline \
    --config_path src/lerobot/rl/config_rl.yaml
```

### Offline RL / RECAP (skip_critic: false)
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
- [ ] **Smoke test offline BC** — run `rl_offline.py` with `skip_critic: true`, verify flow loss decreases over 500 steps.
- [ ] **Smoke test RECAP** — run with `skip_critic: false`, verify critic CE loss decreases and `critic_value_mean` moves away from init.
- [ ] **PI05 regression** — run `rl_offline.py` with `policy.type: pi05_rl`, verify same loss curve as original `learner_pi05.py`.

### Short term
- [ ] **RTC for MolmoAct2** — implement `predict_action_chunk` + `ActionQueue` support in `MolmoAct2Policy`/`MolmoAct2RLPolicy`.  Once done, `rl_actor_async.py` can route through RTC for both models and `actor_pi05_async.py` can be retired.
- [ ] **`rl_pi05_trainer.py` full implementation** — currently a thin stub that delegates to `pi05_train_utils.py`.  Flesh out so PI05 can run through `rl_offline.py` and `rl_learner.py` fully.
- [ ] **`inference_async.py` cleanup** — strip gRPC transport; make it pure standalone inference only (the distributed path now lives in `rl_actor_async.py`).
- [ ] **`config_rl.yaml` — fill in `actor_learner_config`** with real IPs for the lab machines.

### Longer term
- [ ] **Unified actor** — once RTC lands for MolmoAct2, merge `actor_pi05_async.py` logic into `rl_actor_async.py` and retire the PI05-specific file.
- [ ] **`rl_pi05_trainer.py` — `build_inference_batch`** — implement subtask token injection + advantage for PI05 so the generic actor works for PI05 without RTC too.
- [ ] **Online training run** — full HILSERL loop: learner + actor on robot, verify policy improves over episodes.
