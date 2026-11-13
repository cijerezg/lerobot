# loss_share (2026-11-12)

Question: does one data group (here the LLM action corrections, 10% of the mix, about 2x faster motion) take more
than its share of the flow loss and of the action expert's gradient? Same frames, same flow times and noise, every
checkpoint of a run.

## Files

- `run_loss_share.py`: draws `n_frames` chunk starts from the training mixture (config group shares), then per
  checkpoint the flow loss and the flow-loss gradient norm per frame (action expert parameters; the script checks
  that nothing else receives flow gradient), plus the summed gradient per group (internal / external / corrections
  / diverse) and the cosines between them. The docstring defines the three shares.
- `plot_loss_share.py`: `shares.png`, `distributions.png`, `summary.txt`, `groups_all.csv`.

## Run

    uv run python lerobot/adhocs/loss_share/run_loss_share.py --config_path=config_rl.yaml \
        --run_dir=outputs/rebot-corr10-nodepth-b128-20261111 --steps=400,800,1200,1600,2000 \
        --out_dir=outputs/loss_share/corr10_2026-11-12
    uv run python lerobot/adhocs/loss_share/plot_loss_share.py outputs/loss_share/corr10_2026-11-12

About 0.2 s per frame at batch 1, 40 s of setup; a finished checkpoint is skipped on rerun.

## Found (corr10 run, 1024 frames)

Corrections do not take over the flow loss: 10% of the frames, 9-11% of the loss, 11-12% of the summed per-frame
gradient norms, 4-9% of the update direction, at every checkpoint. Diverse is the over-weighted group (35% of the
frames, 43-47% of the loss, 49-87% of the update direction) and its gradient points against the three ReBot
groups (cosine -0.18 to -0.33 at step 1600). Internal is under-weighted (40% of the frames, 21% of the loss).
Flow loss only; the discrete CE that trains the backbone is not measured.
