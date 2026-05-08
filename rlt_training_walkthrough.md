# RLT Training Walkthrough

This checklist adapts the practical lessons from the RL Token reproduction article to this repo's RLT DRTC workflow.

Use it as a runbook: start with low-variance collection, persist the replay buffer, train cautiously, and only widen task variance after the RLT head is stable.

## 0. Pick A Policy Variant

Three RLT wrappers ship in this repo. They share the same online/offline pipeline and replay format; only the frozen VLA backbone (and its processor) differ.

| variant | `policy_type` | base policy | typical use |
| --- | --- | --- | --- |
| full PI0.5 | `pi05_rlt` | `lerobot.policies.pi05_full.PI05FullPolicy` (subtasks + FAST head + ~3B VLA) | large-scale finetunes / multi-task |
| tiny PI0.5 | `tinypi05_rlt` | `lerobot.policies.tinypi05.TinyPI05Policy` (`transformers`-based small VLA, no subtasks, no FAST) | small per-robot finetunes (e.g. SO101 pick-place) |
| tiny PI0.5 v2 | `tinypi05v2_rlt` | `lerobot.policies.tinypi05v2.TinyPI05V2Policy` (self-contained rewrite of tinypi05; no `transformers` / `pistar06` / `pi05` deps) | same as tinypi05_rlt; preferred for new tinypi05v2 finetunes |

The runbook below uses the variant-agnostic terms `<policy_type>`, `<rlt_module>`, `<embedding_trainer>`, and `<offline_trainer>`. Substitute according to the variant you picked:

| placeholder | `pi05_rlt` value | `tinypi05_rlt` value | `tinypi05v2_rlt` value |
| --- | --- | --- | --- |
| `<rlt_module>` | `lerobot.rl.rlt_pi05` | `lerobot.rl.rlt_tinypi05` | `lerobot.rl.rlt_tinypi05v2` |
| `<embedding_trainer>` | `lerobot.rl.train_pi05_rlt_embedding` | `lerobot.rl.train_tinypi05_rlt_embedding` | `lerobot.rl.train_tinypi05v2_rlt_embedding` |
| `<offline_trainer>` | `lerobot.rl.train_pi05_rlt_head_offline` | `lerobot.rl.train_tinypi05_rlt_head_offline` | `lerobot.rl.train_tinypi05v2_rlt_head_offline` |
| default `rlt_token_dim` | `2048` (= `gemma_2b` width) | `None` (defaults to `vlm_width`, e.g. `768` for `small_500m`) | `None` (same as `tinypi05_rlt` — `vlm_width` of the underlying tinypi05 architecture) |

DRTC server, replay buffer, safety tripwires, and the tuning advice in sections 6-11 apply identically to all variants. The `tinypi05_rlt` and `tinypi05v2_rlt` wrappers are interchangeable from the operator's perspective: pick `tinypi05v2_rlt` when your frozen VLA was trained with `policy_type: tinypi05v2` (e.g. the `*_so101_pickplace_160_bs64_anchor` checkpoints). The two backbones are not weight-compatible, so an embedding/head checkpoint trained against one variant cannot be reused against the other.

## 1. Define A Low-Variance Task

- [x] Pick one task where the frozen pi0.5 policy is close but unreliable, such as a fixed block placement or a repeated precision alignment task.
- [x] Fix object poses as tightly as practical for the first runs.
- [x] Fix robot start pose and camera positions.
- [x] Use one clear success criterion that can be labeled quickly at episode end.
- [x] Avoid broad object randomization until the RLT head can improve a fixed setup.

Notes:
- RLT is best treated as action refinement, not as a way to invent a completely new behavior.
- If the base policy is far from solving the task, collect more teleop or RECAP-style data before relying on RLT.

## 2. Confirm The Frozen VLA Baseline

- [ ] Run the frozen VLA pass-through with no RLT head loaded.
- [ ] Record at least 10-20 episodes.
- [ ] Estimate baseline success rate.
- [ ] Identify repeated failure modes, especially gripper hesitation, alignment error, overreach, or slow motion.
- [ ] Keep notes on whether failures are consistent enough for RLT to correct.

Config pointers:

```yaml
# pi05_rlt
policy_type: pi05_rlt
rlt_enabled: true
rlt_embedding_checkpoint: outputs/pi05_rlt_embedding_cube_subtasks_3/rlt_embedding_step_004200.pt
rlt_head_checkpoint:
```

```yaml
# tinypi05_rlt
policy_type: tinypi05_rlt
rlt_enabled: true
rlt_embedding_checkpoint: outputs/tinypi05_rlt_embedding/rlt_embedding_latest.pt
rlt_head_checkpoint:
```

```yaml
# tinypi05v2_rlt
policy_type: tinypi05v2_rlt
rlt_enabled: true
rlt_embedding_checkpoint: outputs/tinypi05v2_rlt_embedding/rlt_embedding_latest.pt
rlt_head_checkpoint:
```

Leaving `rlt_head_checkpoint` empty lets the policy collect RLT context while passing through the frozen VLA until online training reaches `rlt_execute_after_train_steps`.

### Training the RLT embedding (prerequisite for both variants)

Before any DRTC collection, train the RL-token autoencoder on top of your frozen VLA checkpoint. The autoencoder learns to compress the VLA prefix embeddings into a single token; the rest of the RLT pipeline conditions on that token.

```bash
# pi05_rlt
uv run python -m lerobot.rl.train_pi05_rlt_embedding \
  --policy_path=outputs/pi05_subtasks_good_dataset_4/checkpoints/last/pretrained_model \
  --dataset_repo_id=<your_repo_id> \
  --output_dir=outputs/pi05_rlt_embedding \
  --batch_size=8 --steps=5000
```

```bash
# tinypi05_rlt
uv run python -m lerobot.rl.train_tinypi05_rlt_embedding \
  --policy_path=outputs/train/2026-04-30/14-03-18_tinypi05_so101_pickplace_finetune/checkpoints/015000/pretrained_model \
  --dataset_repo_id=<your_repo_id> \
  --output_dir=outputs/tinypi05_rlt_embedding \
  --batch_size=8 --steps=5000
```

```bash
# tinypi05v2_rlt
uv run python -m lerobot.rl.train_tinypi05v2_rlt_embedding \
  --policy_path=outputs/train/2026-05-02/18-34-57_tinypi05_so101_pickplace_160_bs64_anchor/checkpoints/092000/pretrained_model \
  --dataset_repo_id=<your_repo_id> \
  --output_dir=outputs/tinypi05v2_rlt_embedding \
  --batch_size=8 --steps=5000
```

Both `tinypi05` variants default the RL-token width to `vlm_width` (e.g. 768 for the `small_500m` preset, 640 for `gemma3_270m_emb`). Override `--rlt_token_dim=...` only if you need a wider/narrower bottleneck; the value is persisted into the checkpoint config. `tinypi05v2_rlt` shares the autoencoder/head implementation with `tinypi05_rlt`, but you must train the embedding (and head) against the matching v2 backbone — checkpoints are not cross-compatible.

## 3. Validate The RLT Token

- [ ] Collect successful and failed episodes with similar object positions.
- [ ] Save the persisted RLT replay via `rlt_online_buffer_path`.
- [ ] Inspect whether `rl_token` features separate successful and failed behaviors.
- [ ] Prefer PCA or t-SNE over the saved token tensors for a quick sanity check.
- [ ] Continue only if tokens show task-phase structure or failure clustering.

What to look for:
- Smooth progression through normal pick/place phases.
- Distinct clusters for repeated failure states.
- Few obvious outliers caused by bad cameras, stale frames, or reset mistakes.

## 4. Collect A Persistent Buffer First

- [ ] Start with collection enabled.
- [ ] Keep training disabled for the first data pass if you want a clean seed buffer.
- [ ] Label every episode as success or failure.
- [ ] Press `9` in the DRTC TUI to discard a bad rollout before it reaches the replay buffer.
- [ ] Use teleop intervention when needed, but remember intervention transitions train toward the executed intervention action.
- [ ] Confirm the replay file is being written.

Recommended collection settings:

```yaml
rlt_online_collection_enabled: true
rlt_online_training_enabled: false
rlt_demo_buffer_path:
rlt_online_buffer_path: outputs/rlt_online/rlt_online_replay.pt
rlt_online_buffer_save_freq_transitions: 100
rlt_persist_buffer_on_shutdown: true
```

After collection, the persisted RLT buffer can seed later RLT-head training by setting either:

```yaml
rlt_demo_buffer_path: outputs/rlt_online/rlt_online_replay.pt
```

or, when resuming and appending:

```yaml
rlt_online_buffer_path: outputs/rlt_online/rlt_online_replay.pt
```

Important: this compact RLT buffer is for RLT-head training. It is not a full RECAP offline dataset because it does not store raw images, language tokens, and full transition metadata.

## 5. Train The RLT Head Offline

- [ ] Confirm `outputs/rlt_online/rlt_online_replay.pt` exists and has enough transitions for your batch size.
- [ ] Use the same frozen VLA checkpoint that produced the collected reference chunks.
- [ ] Use the same RLT embedding checkpoint that produced the collected `rl_token` tensors.
- [ ] Train only the RLT actor/critic heads from the compact replay.
- [ ] Log both console metrics and W&B metrics.

Log in to W&B first if you want online tracking:

```bash
uv run wandb login
```

For offline sync later:

```bash
export WANDB_MODE=offline
```

Run offline RLT-head training:

```bash
# pi05_rlt
uv run python -m lerobot.rl.train_pi05_rlt_head_offline \
  --policy_path=outputs/pi05_subtasks_good_dataset_4/checkpoints/last/pretrained_model \
  --replay_buffer_path=outputs/rlt_online/rlt_online_replay.pt \
  --output_dir=outputs/rlt_offline_head \
  --rlt_embedding_checkpoint=outputs/pi05_rlt_embedding_cube_subtasks_3/rlt_embedding_step_004200.pt \
  --steps=10000 \
  --batch_size=256 \
  --rlt_actor_hidden_dims='[512, 512, 512]' \
  --rlt_critic_hidden_dims='[512, 512, 512]' \
  --rlt_num_critics=4 \
  --rlt_bc_beta=0.1 \
  --rlt_jerk_beta=0.001 \
  --discount=0.985 \
  --grad_clip_norm=20.0 \
  --wandb_project=lerobot-rlt
```

```bash
# tinypi05_rlt
uv run python -m lerobot.rl.train_tinypi05_rlt_head_offline \
  --policy_path=outputs/train/2026-04-30/14-03-18_tinypi05_so101_pickplace_finetune/checkpoints/015000/pretrained_model \
  --replay_buffer_path=outputs/rlt_online/rlt_online_replay.pt \
  --output_dir=outputs/tinypi05_rlt_offline_head \
  --rlt_embedding_checkpoint=outputs/tinypi05_rlt_embedding/rlt_embedding_latest.pt \
  --steps=10000 \
  --batch_size=256 \
  --rlt_actor_hidden_dims='[512, 512, 512]' \
  --rlt_critic_hidden_dims='[512, 512, 512]' \
  --rlt_num_critics=4 \
  --rlt_bc_beta=0.1 \
  --rlt_jerk_beta=0.001 \
  --discount=0.985 \
  --grad_clip_norm=20.0 \
  --wandb_project=lerobot-rlt
```

```bash
# tinypi05v2_rlt
uv run python -m lerobot.rl.train_tinypi05v2_rlt_head_offline \
  --policy_path=outputs/train/2026-05-02/18-34-57_tinypi05_so101_pickplace_160_bs64_anchor/checkpoints/092000/pretrained_model \
  --replay_buffer_path=outputs/rlt_tinypi05v2_online/rlt_online_replay.pt \
  --output_dir=outputs/tinypi05v2_rlt_offline_head \
  --rlt_embedding_checkpoint=outputs/tinypi05v2_rlt_embedding/rlt_embedding_latest.pt \
  --steps=10000 \
  --batch_size=256 \
  --rlt_actor_hidden_dims='[512, 512, 512]' \
  --rlt_critic_hidden_dims='[512, 512, 512]' \
  --rlt_num_critics=4 \
  --rlt_bc_beta=0.1 \
  --rlt_jerk_beta=0.001 \
  --discount=0.985 \
  --grad_clip_norm=20.0 \
  --wandb_project=lerobot-rlt
```

The script instantiates the matching policy class (`PI05RLTPolicy` / `TinyPI05RLTPolicy` / `TinyPI05V2RLTPolicy`) from the frozen VLA checkpoint/config for paper-aligned head construction. The hot loop updates only `rlt_actor` and `rlt_critic`; the VLA backbone stays frozen. The replay buffer format is shared, so a buffer collected with `tinypi05_rlt` or `tinypi05v2_rlt` must be replayed against the *same* embedding checkpoint that produced its `rl_token` tensors (the script reads `rl_token.shape[-1]` to size the heads, but the encoder weights still need to match — and the v1 vs v2 backbones are not weight-compatible).

Watch these metrics:

- `train/actor_loss`
- `train/critic_loss`
- `train/kl_to_reference`
- `train/actor_q_mean`
- `train/pred_q_mean`
- `train/target_q_mean`
- `train/action_deviation_rms`
- `train/action_deviation_abs_max`
- `train/actor_grad_norm`
- `train/critic_grad_norm`

Checkpoints are written to:

```text
outputs/rlt_offline_head/rlt_head_step_<step>.pt
outputs/rlt_offline_head/rlt_head_latest.pt
```

If `kl_to_reference`, action deviation, critic loss, or Q magnitudes spike, stop and retry with higher `rlt_bc_beta`, lower learning rates, lower `rlt_num_critics`/head size, or a cleaner replay buffer.

## 6. Start Conservative Online Training

- [ ] Use a delayed actor execution gate.
- [ ] Warm up with enough replay before any online updates.
- [ ] Keep gradients clipped.
- [ ] Use multiple critics to reduce Q-value exploitation.
- [ ] Watch Q values, action deviation, actor loss, and critic loss before trusting the actor.

Recommended initial settings:

```yaml
rlt_warmup_episodes: 1
rlt_warmup_transitions: 2000
rlt_replay_capacity: 50000
rlt_batch_size: 256
rlt_utd_ratio: 3
rlt_train_freq_s: 1.0
rlt_execute_after_train_steps: 1000
rlt_grad_clip_norm: 20.0
rlt_num_critics: 4
```

Once timing and stability are confirmed, sweep `rlt_utd_ratio` toward `5-10`.

## 7. Tune Regularization

- [ ] Start with `rlt_bc_beta: 0.1` if hardware safety is uncertain.
- [ ] Try `rlt_bc_beta: 0.05` once Q/action-deviation metrics are stable.
- [ ] Avoid very low BC regularization until safety tripwires and human supervision are working.
- [ ] Use jerk regularization to reduce twitchy low-BC policies.
- [ ] If gripper exploration is too constrained, consider lower BC weight for the gripper action dimension via `rlt_bc_action_weights`.

Suggested sweep:

```yaml
rlt_bc_beta: 0.1      # cautious first run
rlt_bc_beta: 0.05     # article-like working value
rlt_bc_beta: 0.02     # only with strong supervision
rlt_jerk_beta: 0.001
rlt_discount: 0.985
```

Discount notes:
- Lower discount, such as `0.985`, encourages faster completion for short manipulation episodes.
- Higher control frequencies usually require a higher discount for the same wall-clock horizon.

## 8. Use Larger Heads Carefully

- [ ] Start with article-style `[512, 512, 512]` heads for better learning speed.
- [ ] Keep critic count above 1 when using larger heads.
- [ ] Increase UTD only after the trainer does not starve inference.
- [ ] Avoid changing `rlt_chunk_size` when resuming an existing RLT head checkpoint.

Recommended fresh-head settings:

```yaml
rlt_actor_hidden_dims: [512, 512, 512]
rlt_critic_hidden_dims: [512, 512, 512]
rlt_actor_residual_scale: 0.25
rlt_num_critics: 4
rlt_chunk_size: 10
```

Try `rlt_chunk_size: 20` only for a fresh head and after confirming action latency and RTC behavior remain acceptable.

## 9. Watch For Reward Hacking

- [ ] Stop or disable actor execution if Q values spike.
- [ ] Stop if action deviation from the VLA reference grows suddenly.
- [ ] Stop if actor loss turns sharply negative while critic loss rises.
- [ ] Stop if the robot begins pressing into the table, scraping, or making unexpectedly fast moves.
- [ ] Save the buffer and checkpoint before changing hyperparameters.

Useful tripwires:

```yaml
rlt_q_abs_max: 100.0
rlt_action_deviation_abs_max: 2.0
rlt_loss_abs_max: 1000.0
rlt_safety_patience: 3
```

If a tripwire fires:
- [ ] Disable actor execution.
- [ ] Increase `rlt_bc_beta`.
- [ ] Lower actor and critic learning rates.
- [ ] Lower `rlt_utd_ratio`.
- [ ] Check whether bad labels or unsafe intervention transitions entered replay.
- [ ] Consider collecting more fixed-task successes before resuming.

## 10. Evaluate Before Adding Variance

- [ ] Evaluate a saved RLT head on the exact fixed setup.
- [ ] Point `rlt_head_checkpoint` at `outputs/rlt_offline_head/rlt_head_latest.pt`.
- [ ] Keep `rlt_online_collection_enabled: false` and `rlt_online_training_enabled: false` for offline-head evaluation.
- [ ] Run at least 20 episodes if hardware time allows.
- [ ] Compare success rate against frozen VLA pass-through.
- [ ] Compare mean completion time and visible smoothness.
- [ ] Check whether performance regresses after restarting the client/server.

Only after fixed-task performance is reliable:
- [ ] Shift robot start pose slightly.
- [ ] Move objects within a small margin.
- [ ] Add one variance source at a time.
- [ ] Keep separate replay/checkpoint folders for each task distribution.

Eval config:

```yaml
rlt_head_checkpoint: outputs/rlt_offline_head/rlt_head_latest.pt
rlt_online_collection_enabled: false
rlt_online_training_enabled: false
```

## 11. Suggested Run Sequence

- [ ] Run frozen VLA pass-through and record baseline success rate.
- [ ] Run collect-only RLT for 20-50 episodes on a fixed task.
- [ ] Inspect token structure from the persisted replay.
- [ ] Train the RLT head offline from `outputs/rlt_online/rlt_online_replay.pt`.
- [ ] Watch console and W&B metrics, especially KL-to-reference and action deviation.
- [ ] Evaluate `outputs/rlt_offline_head/rlt_head_latest.pt` without online updates.
- [ ] Resume offline training from the latest head only after eval is stable.
- [ ] Optionally move to online fine-tuning with actor execution delayed.
- [ ] Sweep one hyperparameter at a time.
- [ ] Add variance only after fixed-task performance is repeatable.

## 12. Files To Know

- `examples/experiments/configs/baseline_pi05_rlt.yaml`: example DRTC experiment config for the full PI0.5 wrapper.
- `examples/experiments/configs/baseline_tinypi05_rlt.yaml` / `baseline_tinypi05_rlt_eval.yaml`: tinypi05 (v1) DRTC + offline-eval configs.
- `examples/experiments/configs/baseline_tinypi05v2_rlt.yaml` / `baseline_tinypi05v2_rlt_eval.yaml`: tinypi05v2 DRTC + offline-eval configs (mirror the v1 layout; only `policy_type`, the base checkpoint, and the RLT output paths change).
- `src/lerobot/async_inference/policy_server_drtc.py`: online RLT training, replay loading/saving, safety tripwires; routes `pi05_rlt`, `tinypi05_rlt`, and `tinypi05v2_rlt` through the same RLT call paths via `_is_rlt_policy()`.
- `src/lerobot/async_inference/robot_client_drtc.py`: episode collection and transition upload.
- `src/lerobot/rl/rlt_buffer.py`: compact persisted RLT replay format (variant-agnostic).
- `src/lerobot/rl/rlt_pi05.py`: full PI0.5 RLT wrapper plus the reusable `RLTokenAutoencoder`, `RLTActorHead`, `RLTCriticEnsemble`, losses, and checkpoint helpers.
- `src/lerobot/rl/rlt_tinypi05.py`: tinypi05 RLT wrapper (registered as `tinypi05_rlt`); imports the heads/losses/helpers from `rlt_pi05` and reuses them verbatim.
- `src/lerobot/rl/rlt_tinypi05v2.py`: tinypi05v2 RLT wrapper (registered as `tinypi05v2_rlt`); same head/loss/helper imports as the v1 wrapper, only the frozen VLA class changes.
- `src/lerobot/rl/train_pi05_rlt_embedding.py` / `src/lerobot/rl/train_tinypi05_rlt_embedding.py` / `src/lerobot/rl/train_tinypi05v2_rlt_embedding.py`: standalone RL-token autoencoder trainers (one per variant).
- `src/lerobot/rl/train_pi05_rlt_head_offline.py` / `src/lerobot/rl/train_tinypi05_rlt_head_offline.py` / `src/lerobot/rl/train_tinypi05v2_rlt_head_offline.py`: standalone offline RLT-head trainers with console and W&B metrics.
