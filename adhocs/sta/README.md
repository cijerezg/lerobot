# sta (2026-11-23)

Question: the "subspace-translated adaptation" spec edits a write layer with dW = U M V^T, V = input directions
the task uses and the earlier data barely did, U = output directions the layer already writes. Before any fitting
(its milestone M0): how much of the gradient of the tube flow loss falls inside (U, V), against random directions,
the top singular vectors of W, and plain PCA of the task inputs? Base, internal 1200 and tube 700; all 36 blocks of
the action expert, three layers each (`self_attn.out_proj`, `cross_attn.out_proj`, `mlp.down_proj`).

## Files

- `sta.py`: one policy (tube 700's config and prompt, float32), each checkpoint's weights put in place. Per layer the
  second moments of its input on D / P1 / P2, V, U, the gradient on D and the capture numbers. The docstring has
  the formulas. Writes `m0.pt`.
- `report.py`: the table of the asked blocks, `score.png`, `capture.png`, `spectrum_<model>.png`.

## Run

    uv run python lerobot/adhocs/sta/sta.py --config_path=config_rl_validate.yaml
    uv run python lerobot/adhocs/sta/report.py outputs/analysis/sta_2026-11-23/m0.pt --blocks 7,14,21,28

About 11 minutes for the three models. D = the 4 episodes of `rebot_teleop_2026-11-15-tube3-annotated-v1`, one
frame every 2 s (156 frames, 37,440 inputs per layer). P = `rebot_2026-10-14-val-v1`, 8 episodes alternating
between P1 (107 frames) and P2 (96): the internal run's training roots are no longer on disk, and nothing of the
base model's pretraining data is. P frames run with the tube prompt and the tube action statistics.

## Found (outputs/analysis/sta_2026-11-23)

- One direction passes the threshold in about 100 of 108 layers, in every model. The spectrum of the tube inputs
  against B lies on the spectrum of P2 against B except for its first value (`spectrum_*.png`). The largest value of
  P2 against P1 is 50 to 9,000, not near 1: two halves of the other data differ from each other about as much as
  the tube data differs from either.
- Capture with that one direction and 16 output directions, geometric mean over the blocks (base):

  | layer | ours | W singular | V = A_D top, U ours | random |
  |---|---|---|---|---|
  | self | 0.0017 | 0.0017 | 0.090 | 0.00003 |
  | cross | 0.017 | 0.0040 | 0.18 | 0.00002 |
  | mlp | 0.0034 | 0.0008 | 0.093 | 0.000004 |

  Ours is 100 to 1,000 times random and passes the singular vectors of W from about block 15 on (below that W wins).
  Plain PCA of the tube inputs with the same U captures 10 to 50 times more than ours in every layer kind.
- Internal gives the same picture as base. Tube 700 (flow loss on D 0.024 against 0.18 for base) has lower capture
  in the late mlp layers.

## Reuse

`--models name=path,...`, `--task_root`, `--other_root`, `--seconds`, `--k`. The file keeps the full spectra.
