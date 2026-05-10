# RLT Training And Intervention Improvements

Notes from the May 2026 `tinypi05v2_rlt` offline-head rollout, the RLT paper, and the reproduction write-up:

- RLT rollout should not linearly blend the RLT actor output with the VLA output. The paper uses a Gaussian actor conditioned on the RL token, proprioception, and VLA reference chunk. The anchor to the VLA comes from the actor loss, not inference-time blending.
- Erratic movement after offline head training is more consistent with critic exploitation / reward hacking than undertraining. The warning pattern is rising Q, negative actor loss, rising action deviation, and low critic loss on a small replay.
- With only hundreds of reviewed samples, a large `[512, 512, 512]` actor can be too expressive. Prefer a conservative seed head and continue with carefully monitored online learning.

## Safer Offline Seed Training

Recommended first-pass command for the current 2026-05-09 replay:

```bash
uv run python -m lerobot.rl.train_tinypi05v2_rlt_head_offline \
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
  --rlt_num_critics=4 \
  --rlt_bc_beta=2.0 \
  --rlt_jerk_beta=0.05 \
  --rlt_reference_dropout_p=0.0 \
  --grad_clip_norm=5.0 \
  --save_freq=50 \
  --wandb_project=lerobot-rlt
```

Rationale:

- `steps=500` avoids thousands of passes over a tiny replay.
- Smaller `[256, 256]` heads reduce the chance of fitting critic artifacts.
- Lower actor LR than critic LR makes actor exploitation slower.
- `rlt_num_critics=4` keeps the clipped-min critic more conservative.
- `rlt_reference_dropout_p=0.0` is intentional for offline seed training. Copying the VLA reference is an acceptable initial behavior; exploration can be reintroduced during online training.
- Higher `rlt_bc_beta` and `rlt_jerk_beta` keep the first deployed head close to smooth VLA-like behavior.

Watch these metrics:

- `action_deviation_rms`
- `action_deviation_abs_max`
- `actor_q_mean`
- `actor_q_abs_max`
- `critic_loss`
- `actor_loss`
- `actor_bc_loss`
- `actor_jerk_loss`

Stop or roll back if Q rises quickly while action deviation increases. For robot eval, prefer an earlier checkpoint that is still close to the VLA reference.

## Possible Per-Action BC Weights

If the last action dimension is confirmed to be gripper, consider anchoring arm joints more strongly than gripper:

```bash
--rlt_bc_action_weights='[2, 2, 2, 2, 2, 0.25]'
```

Do not use this until the action ordering is verified. The goal is to allow gripper correction while keeping high-leverage arm motion tightly anchored.

## Eval Safety

For eval after an erratic rollout, use a stricter runtime deviation guard:

```yaml
rlt_action_deviation_abs_max: 0.75
```

The server should fall back to VLA passthrough when the RLT actor exceeds this bound. This is a safety guard, not a training fix.

## Intervention Storage Improvement

The RLT paper says human intervention replaces the actor output during rollout and replaces the VLA reference in replay. The current repo behavior stores:

```text
reference_chunk = original VLA reference
executed_chunk = executed action, including intervention action
is_intervention = true
```

The actor loss then uses `executed_chunk` as the BC target when `is_intervention=true`, which partially captures the correction. However, the actor is still conditioned on the original failed VLA reference for those intervention samples.

Suggested modification:

```text
if transition.is_intervention:
    reference_chunk = executed_model
else:
    reference_chunk = source.reference_chunk
```

This makes intervention samples match the paper more closely: the correction becomes both the conditioning reference and the supervised target. It should reduce contradictory intervention training where the actor sees a bad VLA reference but is asked to output a human correction.

Implementation location:

- `src/lerobot/async_inference/policy_server_drtc.py`
- `_accept_rlt_transition`
- `RLTReplaySample(reference_chunk=...)`

Also apply the same policy to `next_reference_chunk` only if the next context corresponds to an intervention-corrected reference. Otherwise keep `next_context.reference_chunk` unchanged.

Open question:

- We may want to preserve both fields in replay:
  - `vla_reference_chunk`
  - `training_reference_chunk`

That would make offline analysis clearer and avoid losing the original VLA proposal for debugging.

## Data Collection Notes

- Keep early tasks low-variance. RLT is best used for precision refinement, not broad behavior invention.
- Review and label every rollout before offline training.
- Deleted episodes in `.review.json` must be excluded before training.
- Prefer collecting more clean, reviewed rollouts over extending offline training on a tiny replay.
- If the base VLA policy is far from competent, collect more teleop / DAgger-style data for the frozen policy before relying on RLT.

Sources:

- RLT paper: https://www.pi.website/download/rlt.pdf
- Reproduction notes: https://villekuosmanen.medium.com/research-notes-from-reproducing-rl-token-f375ecfd3c28
