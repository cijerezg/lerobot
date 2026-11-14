# Per-start sampling weights (ReBot)

> **Status:** abandoned, 2026-10-18. Built 2026-10-16 (commit `86c73b108`), removed two days later
> (commit `4f5d661d2`, "new experiment with uniform sampling"). Training draws starts iid inside each
> group; `tests/rl/test_episode_groups.py` asserts `rebot_start_weights` and `rebot_start_cell` are
> absent from the config. Only the outer level (`diverse.rebot_group_weights`) survives.

## Design as built (v5)

Two levels.

1. Outer: `diverse.rebot_group_weights` fixes each object group's share of all samples. A group's share
   never depends on its frame count, grasp count or the per-start parameters. (Still in force.)
2. Inside a group: each eligible start is drawn with probability proportional to its weight
   (`diverse.rebot_start_weights`, `rl/data_sources/start_weights.py`). Roots inside a group split by
   their weight mass.

Per-start rule: weight 1 = a move frame (also return and transit). Around a commit moment the weight
rises to a peak = how many times more often that start is drawn than a move start of the same group.

| event | peak | window |
|---|---|---|
| grasp (every gripper close, retries included) | `peak_ratio` x `precision_scale[precision-1]` | rise 8 s before, decay 2 s after |
| release (gripper open) | same table | 3 s before, 1 s after |
| other skills (push, pour, close, stand, plug, fold) | `skill_ratio`, flat over the segment | 1 s ramps |

Values: `peak_ratio` 5, `precision_scale` 0.3 / 0.4 / 0.6 / 0.8 / 1.0, `skill_ratio` 3, `smooth: true`
(Gaussian edges, sigma = half the window; false = rectangles).

## What it produced

| quantity | value |
|---|---|
| batch share by verb (smooth) | grasp 56 %, skill 21 %, move 8 %, release 8 %, return 6 % |
| draws per base start per 2000-step run | rigid 0.92, cases 0.86, pills 0.82, external 0.58, bottles 0.50, bits 0.31, clothing 0.15 |

The rule's ratio holds only inside a group (a precision-3 close = 3 x a move start of that group);
cross-group "per event vs move" ratios mix in the group shares.

## Versions dropped on the way

| version | rule | why dropped |
|---|---|---|
| v1 | bucket shares 70/20/6/4 + sqrt cells | batch shares barely change per-start density: grasp owns 64 % of frames |
| v2 | per-start weights, hold-to-lift-off kernel, clothing x0.4 | replaced by the simpler peak/window rule |
| v3 | the v5 rule, simulated only | became v5 once in the trainer |
| v4 | group shares derived from weight mass | clothing rose to 27 % of ReBot through the base weight (61 % of all mass is base) |

## `--density` strip of the review renderer

`annotation/review/render_review_videos.py --density <json>` still draws a "draws per start / run" strip.
The JSON for one episode has `density` (`{curve name: per-frame list}`), `ymax`, `colors`, plus `source`,
`root`, `episode`, `label`, `task`, `n`, `fps`, `closes`, `opens`, `segments`.

[`start_weight_sampling_density.py`](start_weight_sampling_density.py) is the only script that wrote
that JSON. It does NOT run today: it imports the removed `start_weights.py`, reads the removed config
block, and reads `segments.parquet` of a retired scratch folder (`subtask_frames_2026-10-15`). Kept as
the reference for the JSON layout and for the density arithmetic
(`256k samples x group share x weight / group mass`).
