# sta (2026-11-23)

Question: the "subspace-translated adaptation" spec edits a write layer with dW = U M V^T, V = input directions
the task uses and the earlier data barely did, U = output directions the layer already writes. Before any fitting
(its milestone M0): how much of the gradient of the tube flow loss falls inside (U, V), against random directions,
the top singular vectors of W, and plain PCA of the task inputs? Base, internal 1200 and tube 700; all 36 blocks of
the action expert, three layers each (`self_attn.out_proj`, `cross_attn.out_proj`, `mlp.down_proj`).

## Files

- `sta.py`: one policy (tube 700's config and prompt, float32), each checkpoint's weights put in place. Per layer the
  second moments of its input on D / P1 / P2, V, U, the gradient on D and the capture numbers. The docstring has
  the formulas. Writes `m0_joints.pt` (`m0.pt` = the first run, without the joint split).
- `report.py`: the table of the asked blocks, `score.png`, `capture.png`, `spectrum_<model>.png`.

## Run

    uv run python lerobot/adhocs/sta/sta.py --config_path=config_rl_validate.yaml
    uv run python lerobot/adhocs/sta/report.py outputs/analysis/sta_2026-11-23/m0_joints.pt --blocks 7,14,21,28

About 25 minutes for the three models (7 backward passes per frame). D = the 4 episodes of `rebot_teleop_2026-11-15-tube3-annotated-v1`, one
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

## Found, per joint and against the other data's directions (`m0_joints.pt`)

G_j = the gradient through joint j's predicted velocity (G = sum of the 7). Q = eigenvectors of the other data's
input second moment (P1 and P2 together), largest first; "left" = |G (I - P_r)|^2 / |G|^2 with P_r the first r of
them, r chosen to hold a given share of the other data's input energy.

- No joint has its own direction: capture of (U, V) is the same order for all 7 joints in every layer.
- The joints' gradients are close to orthogonal (mean cosine 0.0 to 0.4 base, under 0.25 internal and tube 700).
  Share of sum_j |G_j|^2 over all layers: base wrist yaw 0.33 + gripper 0.39; internal shoulder lift 0.27, elbow
  0.22, wrist roll 0.19; tube 700 spread over elbow, both wrist joints and gripper.
- |G|^2 left, all layers together:

  | other-data energy removed | base | internal | tube 700 |
  |---|---|---|---|
  | 0.9 | 0.33 | 0.32 | 0.57 |
  | 0.99 | 0.16 | 0.16 | 0.28 |
  | 0.999 | 0.06 | 0.06 | 0.10 |

- By depth (base, 0.99 removed): block 7 0.01 to 0.04, block 14 0.02 (self, mlp) and 0.17 (cross), block 21 0.08 /
  0.22 / 0.21, block 28 0.07 / 0.15 / 0.14 (self / cross / mlp). The same for every joint.
- What is left of the gradient follows what is left of the tube inputs' energy (a little below it): the gradient
  does not sit in the free directions more than the inputs do.

## Reuse

`--models name=path,...`, `--task_root`, `--other_root`, `--seconds`, `--k`. The file keeps the full spectra.
