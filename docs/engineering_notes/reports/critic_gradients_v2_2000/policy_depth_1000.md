# Policy checkpoint 1000: depth saturation spot check

Checked 20 deterministic real-depth frames across seven validation episodes. The eighth validation episode has no depth and was excluded. Only checkpoint 1000 was evaluated successfully.

Checkpoint: `outputs/molmoact2_rebot_mix_20260921/checkpoints/001000/pretrained_model`.

The policy depth branch does not show the near-total saturation observed in the critic. This is a small spot check, not a claim about every frame or policy action quality.

| Measurement | Minimum frame | Median frame | Maximum frame |
|---|---:|---:|---:|
| Feature RMS before bound | 18.9812 | 34.1917 | 80.8106 |
| Feature RMS after bound | 17.5747 | 30.7758 | 55.5635 |
| Coordinates within 1% of ±128 (%) | 0.0000 | 0.0000 | 0.3526 |
| Mean derivative through bound | 0.8116 | 0.9422 | 0.9811 |

Method: saved checkpoint actor weights; eval mode and bfloat16; real point-map encoder, depth visual projection, marker and production tanh bound. All depth-path tensors loaded strictly. The language/action forward was unnecessary. The unrelated legacy state-projector key warning does not affect this depth-only measurement.

Derivative is `1 - tanh(pre / 128)^2`, evaluated in float32. The reported near-bound fraction uses the actual bounded bfloat16 features.

[Per-frame measurements and identities](../../../../../outputs/probe_runs/policy_depth_saturation_20260927/policy_depth_saturation.json).

## Frames

| Episode | Frame | Subtask | Pre-bound RMS | Post-bound RMS | Near bound (%) |
|---|---:|---|---:|---:|---:|
| 0 | 300 | grasp the green shirt | 25.623 | 24.371 | 0.0000 |
| 0 | 1440 | release the white sock in the basket | 80.811 | 55.563 | 0.3526 |
| 0 | 2580 | move the black sock to the basket | 19.130 | 18.518 | 0.0000 |
| 1 | 270 | grasp the pill bottle | 28.184 | 26.635 | 0.0000 |
| 1 | 1410 | release the tape roll in the basket | 34.287 | 31.193 | 0.0012 |
| 1 | 2550 | return to home | 39.425 | 34.805 | 0.0010 |
| 2 | 360 | grasp the grey shirt | 37.880 | 34.243 | 0.0000 |
| 2 | 1800 | move the navy shirt to the bin | 21.029 | 20.305 | 0.0000 |
| 2 | 3240 | move the red shirt to the bin | 20.540 | 19.848 | 0.0000 |
| 3 | 510 | move the brown shirt to the bin | 20.003 | 19.239 | 0.0000 |
| 3 | 4530 | grasp the white sock | 44.575 | 39.626 | 0.0000 |
| 4 | 300 | grasp the white sock | 39.912 | 36.133 | 0.0000 |
| 4 | 1440 | grasp the navy shirt | 34.096 | 30.359 | 0.0000 |
| 4 | 2580 | grasp the beige shirt | 23.067 | 22.187 | 0.0000 |
| 6 | 60 | grasp the green eraser | 47.249 | 39.472 | 0.0161 |
| 6 | 360 | move the green eraser to the brown basket | 18.981 | 17.575 | 0.0000 |
| 6 | 660 | return to home | 32.405 | 29.588 | 0.0000 |
| 7 | 30 | grasp the sugar cube | 40.460 | 31.870 | 0.0232 |
| 7 | 210 | grasp the sugar cube | 38.635 | 35.376 | 0.0000 |
| 7 | 390 | return to home | 35.498 | 31.961 | 0.0000 |
