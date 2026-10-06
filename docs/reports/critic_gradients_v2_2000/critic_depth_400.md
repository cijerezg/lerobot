# Critic checkpoint 400: depth saturation spot check

Same 20 real-depth frames across seven validation episodes as the policy-1000 check.
Checkpoint: `outputs/molmoact2_critic_own_20260926_v2/checkpoints/000400/pretrained_model`.

Near bound means an actual bounded feature coordinate has `abs(z) >= 0.99 * 128`.

| Model | Minimum frame | Median frame | Maximum frame |
|---|---:|---:|---:|
| Policy 1000 | 0.000% | 0.000% | 0.353% |
| Critic 400 | 0.109% | 22.760% | 47.019% |
| Critic 2000 | 99.997% | 100.000% | 100.000% |

| Critic 400 measurement | Minimum frame | Median frame | Maximum frame |
|---|---:|---:|---:|
| Pre-bound RMS | 94.7766 | 286.8154 | 417.4408 |
| Post-bound RMS | 69.4476 | 110.2177 | 118.2706 |
| Mean derivative through bound | 0.1464 | 0.2585 | 0.7056 |

The critic already compresses depth features substantially at step 400, but the
near-total saturation at step 2000 is much more severe on these exact frames.
This bounds the comparison at two training steps; it does not locate the onset
or establish the cause. No architecture or training weights were modified.

Method: actual critic eval forward at saved bfloat16 precision; feature hooks
before the tanh bound and at fusion input; finite-feature and frame-identity checks.
The bound derivative `1-tanh(pre/128)^2` is evaluated in float32 and averaged over
coordinates per frame. Summaries then take minima/medians/maxima over the 20 frames.
Checkpoint 2000 comparison reuses its saved measurements; it was not rerun.

`../../../../../outputs/probe_runs/critic_depth_saturation_000400_20260927/critic_depth_saturation.json`.

| Episode | Frame | Subtask | Pre-bound RMS | Post-bound RMS | Near bound (%) | Mean slope |
|---|---:|---|---:|---:|---:|---:|
| 0 | 300 | grasp the green shirt | 417.441 | 118.271 | 47.019 | 0.1464 |
| 0 | 1440 | release the white sock in the basket | 202.473 | 98.705 | 7.992 | 0.4053 |
| 0 | 2580 | move the black sock to the basket | 296.719 | 111.878 | 24.639 | 0.2361 |
| 1 | 270 | grasp the pill bottle | 374.617 | 116.168 | 39.905 | 0.1764 |
| 1 | 1410 | release the tape roll in the basket | 336.575 | 113.995 | 33.096 | 0.2069 |
| 1 | 2550 | return to home | 276.912 | 108.997 | 20.881 | 0.2749 |
| 2 | 360 | grasp the grey shirt | 303.376 | 111.275 | 26.031 | 0.2443 |
| 2 | 1800 | move the navy shirt to the bin | 376.315 | 116.741 | 40.363 | 0.1683 |
| 2 | 3240 | move the red shirt to the bin | 275.378 | 109.161 | 20.396 | 0.2727 |
| 3 | 510 | move the brown shirt to the bin | 184.576 | 92.411 | 6.524 | 0.4788 |
| 3 | 4530 | grasp the white sock | 260.177 | 106.475 | 17.936 | 0.3081 |
| 4 | 300 | grasp the white sock | 357.130 | 114.706 | 36.940 | 0.1971 |
| 4 | 1440 | grasp the navy shirt | 160.679 | 87.732 | 3.788 | 0.5302 |
| 4 | 2580 | grasp the beige shirt | 412.470 | 117.952 | 46.127 | 0.1510 |
| 6 | 60 | grasp the green eraser | 205.054 | 95.597 | 9.425 | 0.4422 |
| 6 | 360 | move the green eraser to the brown basket | 94.777 | 69.448 | 0.109 | 0.7056 |
| 6 | 660 | return to home | 215.893 | 96.922 | 11.053 | 0.4266 |
| 7 | 30 | grasp the sugar cube | 117.219 | 77.309 | 0.708 | 0.6352 |
| 7 | 210 | grasp the sugar cube | 324.256 | 112.224 | 30.459 | 0.2314 |
| 7 | 390 | return to home | 369.617 | 116.352 | 39.168 | 0.1738 |
