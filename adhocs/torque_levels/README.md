# torque_levels

Question (2026-11-14): can the gripper torque, as a two-level value in the prompt, say whether the gripper holds
something? Approach: smooth the gripper's `observation.effort` the way the model input would be (causal weighted
mean of the last 4 frames: 0.4 on the newest, then 0.3, 0.2, 0.1), put every frame of the root in low or high, and
change the rule in the page while watching where the high stretches land. Two rules:

- latch (default): high when the torque reaches the threshold and the gripper closed by at least `closed` deg
  within the last second; stays high until the gripper has opened by `opened` deg from its tightest position since.
- threshold alone, with hysteresis: high at threshold + margin, low below threshold - margin, otherwise unchanged.

```bash
uv run python lerobot/adhocs/torque_levels/view.py <dataset root> [--threshold 0.18] [--closed 2] [--opened 2] [--margin 0.06] [--no-open]
```

Output: `outputs/analysis/torque_levels/<root>.html` and one clip per episode next to it (top and wrist views
stacked; an existing clip is reused). Needs `ffmpeg` and the column `observation.effort` (roots recorded since
2026-11-13).

## Page

- Threshold: number box, or drag the dashed line in either chart; up / down arrows move it by 0.005 N*m (the
  sensor step). The latch checkbox switches the rule; without it the gray band around the line is the margin. The
  header gives the share of frames high, the number of level changes and the number of stretches shorter than
  0.5 s, each next to the count of the threshold alone without a margin.
- "All frames": smoothed torque against gripper position (0 = closed), every frame of the root, colored by level.
  The torque axis covers the middle 99.6 % of the frames; harder squeezes are drawn on its edge. Click a point to
  see that frame.
- Episode chart: gripper position (command dashed) and torque (raw faint, smoothed solid) of the episode being
  played, high stretches shaded.
- "Every episode": one row per episode on a shared time axis, gripper position as the line, high shaded, share of
  the episode high at the right. Click to go there.

## Findings

`outputs/rebot_teleop_2026-11-14-tube-cut-v3` (16 episodes, 34,836 frames), scored against the subtask labels
(in hand = move + insert segments, 10,963 frames; free = approach + return, 9,335 frames; grasp segments not
scored). Scripts: `outputs/scratch/gripper_torque_filter_2026-11-14/{sweep,why,latch}.py`.

| rule | in-hand frames high | free frames high | carries (of 27) at least 80 % high | level changes | stretches < 0.5 s |
|---|---|---|---|---|---|
| threshold 0.10, margin 0.03 | 40 % | 16 % | 13 | 287 | 52 |
| threshold 0.15, margin 0.06 | 36 % | 3 % | 12 | 111 | 15 |
| latch 0.18, closed 2 deg, opened 2 deg | 86 % | 5 % | 27 | 116 | 4 |

- The torque level alone cannot say "in hand": the torque peaks while the gripper closes on the tube (at least
  0.19 N*m in all 27 carries) and then decays during the carry; the median torque of a carry runs from -0.15 to
  0.50 N*m, and 13 of the 27 carries sit below 0.10.
- The latch result is the same for thresholds 0.10 to 0.18 (86 % / 5 %); at 0.20 it drops to 70 %, at 0.25 to 40 %.
- Without the "closed within the last second" condition the latch also fires on the torque wobble of an open,
  still gripper (free frames high 27 %).
- Known wrong highs of the latch: a close on nothing (for example closing fully on the way home) reads high until
  the gripper opens again. The 14 % of in-hand frames it misses (1,508) are all the frames after the release at the end
  of the insert segments.
