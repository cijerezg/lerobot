# TinyPI05v2 RLT Rollout Debug Notes

Notes from the May 2026 `tinypi05v2_rlt` SO101 pick/place rollout debugging.

## Current Working Baselines

May 9 anchor eval, after the camera was restored, behaved like the original good rollout:

```bash
./scripts/run_drtc_experiment.sh --config examples/experiments/configs/baseline_tinypi05v2_rlt_eval_20260509_anchor.yaml
```

May 10 eval currently points at the retrained anchor-recipe head:

```bash
./scripts/run_drtc_experiment.sh --config examples/experiments/configs/baseline_tinypi05v2_rlt_eval_20260510_256.yaml
```

Key paths:

- May 9 working head: `outputs/tinypi05v2_rlt_offline_head_20260509_anchor/rlt_head_step_000500.pt`
- May 9 replay: `outputs/rlt_tinypi05v2_online_20260509/rlt_online_replay_20260509.pt`
- May 10 anchor-recipe head: `outputs/tinypi05v2_rlt_offline_head_20260510_anchor_recipe/rlt_head_step_000500.pt`
- May 10 replay: `outputs/rlt_tinypi05v2_online_20260510/rlt_online_replay_20260510.pt`
- May 10 review sidecar: `outputs/rlt_tinypi05v2_online_20260510/rlt_online_replay_20260510.review.json`

## Important Debug Findings

The May 10 anchor-recipe model is better than the earlier May 10 head, but still has back-and-forth jerk. The demonstrations/interventions themselves did not look especially jerky for either dataset, so the likely issue is not raw operator motion.

The offline comparison points to actor overshoot:

```text
May 9 anchor model on May 9 replay:
  actor first-step correction RMS:   0.859
  executed first-step correction RMS: 0.944

May 10 anchor-recipe model on May 10 replay:
  actor first-step correction RMS:   0.336
  executed first-step correction RMS: 0.182
```

Interpretation:

- On May 9, the actor correction was slightly smaller than the human/executed correction.
- On May 10, the actor correction is much larger than the executed correction in the replay.
- The May 10 dataset has smoother and smaller corrections, but the learned head is amplifying them.
- This explains why the rollout can be jerky even if the recorded intervention data is not.

## Review Sidecar

The offline trainer does respect the review sidecar:

```python
RLTReplayBuffer.load(cfg.replay_buffer_path, apply_review_sidecar=True)
```

For the May 10 buffer, applying the sidecar changed the training set from:

```text
raw samples:      1606
reviewed samples: 1157
raw episodes:       76
reviewed episodes:  57
```

The sidecar deleted 19 episodes and relabelled reviewed episodes into success/failure/open.

## Current Eval Damping Knob

An eval-only blend knob was added:

```yaml
rlt_eval_actor_blend: 0.5
```

It blends only at eval time:

```text
refined = reference + blend * (actor - reference)
```

This is an engineering damping/safety knob, not part of the RLT paper objective. It is useful because the current May 10 actor appears to overshoot the smooth replay corrections.

Important plumbing bug fixed:

- `examples/experiments/run_drtc_experiment.py` had its own `ExperimentConfig` and scalar allow-list.
- The YAML field `rlt_eval_actor_blend` was initially being dropped before reaching `RobotClientDrtcConfig`.
- Logs that still show `rlt_eval_actor_blend: 1.0` mean the blend did not reach the server.
- After the plumbing fix, a correct run should report `rlt_eval_actor_blend: 0.5` in `drtc_status_*.jsonl`.

If the run is still jerky, try:

```yaml
rlt_eval_actor_blend: 0.25
```

If it becomes too close to VLA passthrough or loses the improvement, try:

```yaml
rlt_eval_actor_blend: 0.75
```

## Training Recipes

The May 9 anchor recipe that worked better:

```bash
python -m lerobot.rl.train_tinypi05v2_rlt_head_offline \
  --policy_path=outputs/train/2026-05-02/18-34-57_tinypi05_so101_pickplace_160_bs64_anchor/checkpoints/092000/pretrained_model \
  --replay_buffer_path=outputs/rlt_tinypi05v2_online_20260509/rlt_online_replay_20260509.pt \
  --output_dir=outputs/tinypi05v2_rlt_offline_head_20260509_anchor \
  --rlt_embedding_checkpoint=outputs/tinypi05v2_rlt_embedding/rlt_embedding_step_010000.pt \
  --steps=500 \
  --batch_size=128 \
  --actor_lr=0.00003 \
  --critic_lr=0.0001 \
  --rlt_actor_hidden_dims='[256, 256]' \
  --rlt_critic_hidden_dims='[256, 256]' \
  --rlt_actor_mode=gaussian \
  --rlt_num_critics=4 \
  --rlt_bc_beta=2.0 \
  --rlt_jerk_beta=0.05 \
  --rlt_reference_dropout_p=0.0 \
  --discount=0.985 \
  --grad_clip_norm=5.0 \
  --save_freq=50 \
  --wandb_project=lerobot-rlt
```

The May 10 anchor-recipe retrain used the same idea:

```bash
python -m lerobot.rl.train_tinypi05v2_rlt_head_offline \
  --policy_path=outputs/train/2026-05-02/18-34-57_tinypi05_so101_pickplace_160_bs64_anchor/checkpoints/092000/pretrained_model \
  --replay_buffer_path=outputs/rlt_tinypi05v2_online_20260510/rlt_online_replay_20260510.pt \
  --output_dir=outputs/tinypi05v2_rlt_offline_head_20260510_anchor_recipe \
  --rlt_embedding_checkpoint=outputs/tinypi05v2_rlt_embedding/rlt_embedding_step_010000.pt \
  --steps=500 \
  --batch_size=128 \
  --actor_lr=0.00003 \
  --critic_lr=0.0001 \
  --rlt_actor_hidden_dims='[256, 256]' \
  --rlt_critic_hidden_dims='[256, 256]' \
  --rlt_actor_mode=gaussian \
  --rlt_num_critics=4 \
  --rlt_bc_beta=2.0 \
  --rlt_jerk_beta=0.05 \
  --rlt_reference_dropout_p=0.0 \
  --discount=0.985 \
  --grad_clip_norm=5.0 \
  --save_freq=50 \
  --wandb_project=lerobot-rlt
```

For another May 10 retrain, make it more imitation-dominant because the actor is overshooting the executed corrections:

- Increase `rlt_bc_beta` above `2.0`.
- Increase `rlt_jerk_beta` above `0.05`.
- Lower `actor_lr` below `0.00003`, or stop earlier.
- Compare checkpoints by first-step actor correction vs executed correction, not only by within-chunk jerk.

## Paper Alignment Notes

The RLT paper uses a Gaussian actor conditioned on the RL token, proprioception, and the VLA reference action chunk. The anchoring to VLA/reference behavior is mainly from the actor loss and data, not from inference-time blending.

Runtime blending is therefore a practical safety/damping layer for robot eval, not the paper objective. It should be treated as a deployment guard while retraining or improving data.

## Blended May 10 Rollout Result

After fixing the `rlt_eval_actor_blend` config plumbing, blended May 10 rollout behavior looked much better and could succeed at the task. However, it was still not an improvement over the base VLA.

Current interpretation:

- Base VLA remains the best baseline.
- May 10 RLT unblended is too unstable.
- May 10 RLT blended is safe enough to execute, but not better than VLA.
- The current May 10 head should not be used as the behavior policy for new autonomous data collection.

Blending is acting as a stabilizer by pulling the actor back toward the VLA reference. If the stable blended behavior only matches VLA performance, the learned correction is not yet adding useful task improvement.

Best next data step:

- Collect more targeted correction data with VLA passthrough while still recording RLT context.
- Focus interventions on moments where the VLA is weak, rather than replacing whole rollouts.
- Aim for small phase-specific corrections around grasp/contact/place or post-drop stabilization.

For another May 10 retrain, make the actor even more imitation-dominant:

```bash
--rlt_bc_beta=4.0 \
--rlt_jerk_beta=0.1 \
--actor_lr=0.00001 \
--steps=300
```

Then compare early checkpoints. The key metric is whether actor first-step correction stays near the executed correction magnitude. In the May 10 anchor-recipe run, the actor was overshooting:

```text
actor correction RMS:   0.336
executed correction RMS: 0.182
```

## Phase-Focused Replay Filtering

A reasonable next experiment is to edit the existing rollout replay into a smaller grasp-focused correction dataset. The goal is not to train a whole-rollout replacement policy. The goal is to teach the RLT head when and how to nudge the VLA around grasp failures.

Keep transitions around:

- right before grasp
- during gripper closure / contact
- immediately after successful grasp
- immediately after failed grasp
- human interventions that correct grasp pose, gripper timing, or early lift
- VLA-pass-through examples where the base policy was already correct in the same phase

Remove transitions that do not teach the grasp correction:

- long approach from far away
- long stable transport after a good grasp
- post-drop oscillation
- reset/camera junk
- whole-rollout segments where the RLT head should not be learning a new behavior

Important: do not keep only failures and interventions. The filtered replay still needs examples where the right correction is near zero. Otherwise the actor may learn to always modify the VLA reference at grasp time.

Also be careful with rewards. The current review sidecar is episode-level, and the replay reward is usually attached to terminal success/failure transitions. If the filtered replay removes the terminal reward-bearing transition, critic learning may become weak or misleading. For a small grasp-phase replay, treat training as conservative behavior correction:

```bash
--rlt_bc_beta=4.0 \
--rlt_jerk_beta=0.1 \
--actor_lr=0.00001 \
--steps=300
```

If the actor still overshoots, increase BC/smoothness further or stop earlier. The expected behavior is a small phase-specific nudge, not a new autonomous policy.

Implementation preference:

- Create a new filtered `.pt` replay buffer using `episode_id` and `chunk_start_step` ranges.
- Keep the original replay untouched.
- Do not rely on the current `.review.json` for transition-level pruning, because it only supports episode-level delete/label.
- A future improvement could extend the sidecar format with transition-level include/exclude ranges, but a separate filtered replay file is simpler for now.

## Current Replay Semantics

The repo's offline RLT training currently stores:

```text
reference_chunk = VLA reference
executed_chunk = executed robot/human action
is_intervention = true/false
```

The actor loss uses `executed_chunk` as the BC target for intervention samples, but the actor is still conditioned on the original VLA reference. This partially matches the paper intervention story, but it may still be harder than conditioning on the corrected intervention reference.

## Next Steps

1. Use base VLA / VLA passthrough for the next data collection pass, not the current May 10 RLT head.
2. Collect targeted intervention corrections around the weak task phases.
3. Build a grasp-phase filtered replay from existing reviewed rollouts.
4. Retrain the May 10 head with stronger BC/smoothness and lower actor LR.
5. Evaluate early checkpoints with `rlt_eval_actor_blend` only as a safety/damping tool.
6. Consider adding an analysis script that reports:
   - actor correction first-step RMS
   - executed correction first-step RMS
   - actor correction jump between replans
   - executed correction jump between replans
   - within-chunk jerk
