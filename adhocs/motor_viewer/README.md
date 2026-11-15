# motor_viewer

Question (2026-11-14): what do the motor velocity and torque columns look like along a recorded episode, and does
the gripper torque show the grasp of a small object?
Approach: plot position (command and measured), velocity and torque of each motor on one time axis, synced to the
videos. No detection and no torque model; for those see `adhocs/torque_viewer` (inference roots only).

```bash
uv run python lerobot/adhocs/motor_viewer/view.py <dataset root> [--episode 0] [--joint gripper] [--no-open]
```

Builds the pages and opens the first one in the browser (links to the other episodes are in its header). An
existing clip is reused; delete it to re-encode.

Output: `outputs/analysis/motor_viewer/<root>_ep<N>.html` and the `.mp4` it plays (keep them together; top and wrist
views are stacked into one clip). Needs `ffmpeg` and the columns `observation.velocity` / `observation.effort`
(roots recorded since 2026-11-13, by `lerobot_record` or the online recorder).

## Page

Left: the clip. Right, "One joint": position (command dashed, measured solid), velocity and torque of the chosen
joint. "All joints": one measure, one row per joint, each with its own y axis. The strip at the top is the gripper
position over the whole episode (click to seek; shows the zoom window). The number at the right of each row is the
value under the playhead. Smoothing is a centered mean over 3 / 7 / 15 frames on velocity and torque, drawn over
the raw signal.

"Velocity input" (2026-11-14): one row per joint, the raw velocity and on top of it what the model would read:
causal mean of the last 4 frames (0.4 newest / 0.3 / 0.2 / 0.1, not across episodes), then the rule of the state
tokens (q01 / q99 of the root's `meta/stats.json` -> [-1, 1], clip, 256 levels). The y axis is fixed to q01..q99, so
clipped stretches run flat along the edge. "Histogram": frames per level over the whole root, one row per joint
(log counts by default); the line is the level under the playhead, hover gives level / deg/s / frames. The run also
prints a table per joint (q01, q99, deg/s per level, share clipped, levels used, share in the fullest level).

Click a chart to seek, wheel = zoom the time axis, drag = pan, double click = whole episode, space = play / pause,
arrows = one frame, shift + arrows = one second. Hover gives the values of that frame (in "One joint" also
command - position).

## Findings

Root `cijerezg/dental-clipper-bit-kit-v1_20261114_101549` (teleoperated, 3 episodes; a bit is picked and put on
the holder once in episode 0 and twice in episode 1). Gripper: 0 = closed, negative = open.

- Resolution: torque 0.014 N*m on the three big joints, 0.005 N*m on wrist and gripper. Velocity 0.28 deg/s on
  the big joints, 0.84 deg/s on wrist and gripper; at standstill it reads -0.14 / -0.42 deg/s (half a step), not 0.
  Gripper and wrist velocity jump by 10 deg/s between frames while moving; position is much cleaner.
- Gripper torque follows the direction of motion: about +0.2 N*m while closing and -0.2 N*m while opening, with or
  without an object. After a stop the torque often stays at that level for seconds (for example -0.19 N*m for
  1.5 s after an opening move in free air, episode 1, 15 s).
- The grasp of the bit is not visible. The gripper reaches its command in all three grasps (command - position
  under 0.1 deg), and nothing changes in the torque when the fingers touch the bit. Hold torque: +0.30 N*m
  (episode 1, 26.5-35 s), +0.21 N*m (episode 1, 77-85 s), +0.05 N*m (episode 0, 31.5-39 s), so the hold level
  does not separate "bit in hand" from a stop in free air.
- Not checked: in episode 1 the hold torque steps from 0.29 to 0.22 N*m within one frame at 35.3 s with no motion
  of the gripper, when the bit reaches the holder. Possibly the holder taking load off the fingers.

Velocity input, root `outputs/rebot_teleop_2026-11-14-tube-cut-v3` (16 episodes, 34836 frames):

| joint | q01 | q99 | deg/s per level | clipped | levels used | fullest level |
|---|---|---|---|---|---|---|
| shoulder_pan | -9.65 | 10.78 | 0.080 | 2.0% | 256 | 16.8% |
| shoulder_lift | -12.73 | 19.17 | 0.125 | 2.2% | 256 | 10.1% |
| elbow_flex | -9.09 | 14.69 | 0.093 | 1.9% | 256 | 13.5% |
| wrist_flex | -9.64 | 6.31 | 0.063 | 1.3% | 191 | 24.9% |
| wrist_yaw | -4.62 | 4.62 | 0.036 | 1.3% | 111 | 56.4% |
| wrist_roll | -9.65 | 10.50 | 0.079 | 1.0% | 237 | 42.3% |
| gripper | -37.36 | 34.85 | 0.283 | 1.9% | 256 | 59.6% |

The fullest level is standstill. On wrist and gripper one motor step (0.84 deg/s) is 3 to 23 levels wide, so their
levels are a comb, not a continuum.
