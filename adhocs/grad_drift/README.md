# grad_drift (2026-11-12)

Question: how fast does the gradient of a fixed set of frames change while the model trains? Take `n_probe`
frames per data group (internal / external / corrections / diverse), measure their gradient at the start
weights, train normally, and measure the same frames again (same flow times, same noise) after 1, 2, 4, ...
steps. Per group: cosine to the start gradient, gradient norm, probe loss, and how well the start gradient
predicts the loss change.

## Files

- `run_grad_drift.py`: probe gradient of the full training loss (flow + discrete CE) over every trainable
  parameter, stored as a count sketch per parameter block (98 blocks x `sketch_dim`) plus exact squared norms.
  The weight movement from the start goes through the same sketch, so dot products between gradients and
  movement can be read off later. The training steps in between are the run's own (`trainer.update_actor`,
  config batch, accumulation, clip, AdamW, learning-rate schedule from its step 0). Writes `step_<n>.npz`,
  `floor.npz` (start weights, other flow noise), `losses.csv`, `train.csv`, `frames.csv`, `run.json`.
- `plot_grad_drift.py`: `drift.png` (the four curves per group), `blocks.png` (cosine to the start per block and
  step), `pairs.png` (cosine between every two measured steps), `summary.txt`.

## Run

    uv run python lerobot/adhocs/grad_drift/run_grad_drift.py --config_path=config_rl.yaml \
        --out_dir=outputs/grad_drift/base_2026-11-12 --measure_at=0,1,2,4,8,16,32,64,128,256
    uv run python lerobot/adhocs/grad_drift/plot_grad_drift.py outputs/grad_drift/base_2026-11-12

Start weights are `--policy.pretrained_path` (null = base MolmoAct2). `--n_probe` (32), `--sketch_dim` (4096)
and `--frame_seed` are the other knobs. A measurement costs one forward and backward per group; the rest of
the time is ordinary training steps.

## Notes

- The optimizer is the run's AdamW. Plain SGD is not an option: the weights are bfloat16 and an SGD step is
  below one bf16 increment.
- With the schedule on, the first 100 steps are warmup, so the early measurements see very small movement.
  `--policy.scheduler_warmup_steps=0` gives full-size steps from the first one.
- The sketch estimates dot products; `summary.txt` ends with a check of sketched against exact squared norms.
- The floor (dotted lines in `drift.png`) is the cosine between two gradients at the same weights with
  different flow noise. A drop that stays above it is not drift.

## Found

Not run at full size yet. The smoke test (4 probe frames per group, batch 4) only shows that the pipeline works.
