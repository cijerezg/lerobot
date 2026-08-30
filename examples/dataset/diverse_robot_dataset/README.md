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

Run one task component end to end. Acquisition is the only networked step:

```bash
# Steps 1-2: download at the pinned revision, verify bytes and SHA-256,
# scan the concatenated tar, then extract into bounded staging.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/acquire_robochallenge.py \
  --task item_classification \
  --downloads-root outputs/diverse_robot_dataset/robochallenge/staging/downloads \
  --raw-root outputs/diverse_robot_dataset/robochallenge/staging/raw

# Steps 3-4: nominate 15 candidates and render review proxies.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_robochallenge_component.py \
  --task item_classification review

# Review the proxies, then steps 5-8: annotations, conversion, packed
# extraction, validation, component metadata, and the index update.
uv run --no-project --python .venv/bin/python \
  lerobot/examples/dataset/diverse_robot_dataset/run_robochallenge_component.py \
  --task item_classification finalize --accepted 0 152 201 220 232 561 807 849 872 932
```

Visual acceptance is deliberately a human step between `review` and `finalize`.
Nomination balances candidates across physical robot IDs with a hard per-robot
quota; the realized balance is recorded under `robot_balance` in
`candidates.json`. `finalize` refuses to attach metadata or touch the index
unless the validation report passes.

Validated annotation JSON files feed the existing packed-v3 converter and
extractor. Final components are written under
`outputs/diverse_robot_dataset/robochallenge/datasets/<embodiment>/<task>/`.
