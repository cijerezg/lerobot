# RLT Training Walkthrough

This checklist adapts the practical lessons from the RL Token reproduction article to this repo's `pi05_rlt` DRTC workflow.

Use it as a runbook: start with low-variance collection, persist the replay buffer, train cautiously, and only widen task variance after the RLT head is stable.

## 1. Define A Low-Variance Task

- [ ] Pick one task where the frozen pi0.5 policy is close but unreliable, such as a fixed block placement or a repeated precision alignment task.
- [ ] Fix object poses as tightly as practical for the first runs.
- [ ] Fix robot start pose and camera positions.
- [ ] Use one clear success criterion that can be labeled quickly at episode end.
- [ ] Avoid broad object randomization until the RLT head can improve a fixed setup.

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
policy_type: pi05_rlt
rlt_enabled: true
rlt_embedding_checkpoint: outputs/pi05_rlt_embedding_cube_subtasks_3/rlt_embedding_step_004200.pt
rlt_head_checkpoint:
```

Leaving `rlt_head_checkpoint` empty lets the policy collect RLT context while passing through the frozen VLA until online training reaches `rlt_execute_after_train_steps`.

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

## 5. Start Conservative Online Training

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

## 6. Tune Regularization

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

## 7. Use Larger Heads Carefully

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

## 8. Watch For Reward Hacking

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

## 9. Evaluate Before Adding Variance

- [ ] Evaluate a saved RLT head on the exact fixed setup.
- [ ] Run at least 20 episodes if hardware time allows.
- [ ] Compare success rate against frozen VLA pass-through.
- [ ] Compare mean completion time and visible smoothness.
- [ ] Check whether performance regresses after restarting the client/server.

Only after fixed-task performance is reliable:
- [ ] Shift robot start pose slightly.
- [ ] Move objects within a small margin.
- [ ] Add one variance source at a time.
- [ ] Keep separate replay/checkpoint folders for each task distribution.

## 10. Suggested Run Sequence

- [ ] Run frozen VLA pass-through and record baseline success rate.
- [ ] Run collect-only RLT for 20-50 episodes on a fixed task.
- [ ] Inspect token structure from the persisted replay.
- [ ] Train with actor execution delayed.
- [ ] Enable actor execution after warmup and watch diagnostics closely.
- [ ] Save checkpoint and replay after the first stable improvement.
- [ ] Evaluate the checkpoint without further online updates.
- [ ] Resume from checkpoint plus persisted replay only after the eval is stable.
- [ ] Sweep one hyperparameter at a time.
- [ ] Add variance only after fixed-task performance is repeatable.

## 11. Files To Know

- `examples/experiments/configs/baseline_pi05_rlt.yaml`: main DRTC experiment config.
- `src/lerobot/async_inference/policy_server_drtc.py`: online RLT training, replay loading/saving, safety tripwires.
- `src/lerobot/async_inference/robot_client_drtc.py`: episode collection and transition upload.
- `src/lerobot/rl/rlt_buffer.py`: compact persisted RLT replay format.
- `src/lerobot/rl/rlt_pi05.py`: RLT token model, actor/critic heads, losses, and checkpoint helpers.

