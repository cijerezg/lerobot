# Diverse real-robot production dataset

Production outputs live under `outputs/diverse_robot_dataset/`. Pilot outputs
are evidence and test fixtures only; they are not part of the training set.

RoboChallenge is built first. Its production manifest selects 10 single-arm
tasks and targets 10 visually accepted source episodes per task. Every accepted
episode contributes all clean active anchors at a two-second stride. The
measured six-joint plus gripper-width state is copied unchanged into action for
this source-specific exception.

Reviewers should use `robochallenge/review/<task>/episode_XXXXXX.mp4`. Each file
shows the global and wrist cameras side by side. The `t`, `tm1`, ..., `tm6`
directories inside a finalized LeRobot dataset are model-facing temporal
features, not separate episodes and not the review interface.

Nominate and render review material for an extracted task:

```bash
uv run python lerobot/examples/dataset/diverse_robot_dataset/prepare_robochallenge.py \
  nominate \
  --task-root outputs/diverse_robot_pilot/staging/robochallenge/shred_paper \
  --output outputs/diverse_robot_dataset/robochallenge/review/shred_paper/candidates.json \
  --count 15

uv run python lerobot/examples/dataset/diverse_robot_dataset/prepare_robochallenge.py \
  proxies \
  --task-root outputs/diverse_robot_pilot/staging/robochallenge/shred_paper \
  --manifest outputs/diverse_robot_dataset/robochallenge/review/shred_paper/candidates.json \
  --output-root outputs/diverse_robot_dataset/robochallenge/review/shred_paper
```

Validated annotation JSON files feed the existing packed-v3 converter and
extractor. Final components are written under
`outputs/diverse_robot_dataset/robochallenge/datasets/<embodiment>/<task>/`.
