# torque_viewer

Question (2026-11-12): does the follower's motor torque show contact events (grasp, pushing on the table, collision)?
Approach: run a peak detector on the torque, then look at the video at each event. See
`docs/design/data_annotation.md` (motor velocity and torque columns) for how the torque is captured; roots of
2026-11-12 / 13 have it in `meta/online_labels.parquet` instead (`docs/archive/torque_capture.md`).

```bash
uv run python lerobot/adhocs/torque_viewer/view.py outputs/train/<date>/<run>/inference_dataset [--episode 0]
```

Output: `outputs/analysis/torque_viewer/<run>_ep<N>.html` and the `.mp4` it plays (keep them together; top and wrist
views are stacked into one clip). Needs `ffmpeg`. Roots recorded before 2026-11-12 have no torque columns and are
refused. Roots without `{joint}.vel` (run 14-50-14) use the derivative of `observation.state` as velocity.

## Detector

Per joint, computed in Python (`--baseline`, `--smooth`, `--floor`):

1. Clean: unwrap (the wrist motors wrap at +-10 N*m) and a 3-frame median.
2. Expected torque = slow baseline + motion model. Baseline: rolling median over 10 s (posture / gravity); the
   gripper uses the episode median, so a long hold is not absorbed. Motion model: robust (Huber) fit of the
   baseline-free torque on the joint's own velocity, `a tanh(v / v0) + b v + c dv/dt` (friction and inertia).
3. Score = |4 Hz low-pass of (torque - expected)| / sigma, with sigma = 1.4826 * median absolute deviation.
4. Candidates = `scipy.signal.find_peaks` on the score, prominence >= 2 sigma. Each carries prominence, signed
   height, extent (foot to foot, 90 % of the prominence below the top) and width at half prominence.

In the page (instant, no recomputation):

- the slider keeps candidates with prominence >= threshold (sigma; for an isolated peak, prominence = height);
- candidates of one joint with overlapping extents are one event ("N crests, one peak"); the rise and the fall of a
  bump are never separate events, since the score is a level and not a derivative;
- events of different joints whose peaks lie within the coincidence window (0.3 s) form one group: blue = several
  joints, orange = one joint alone;
- width at half prominence >= 1 s is tagged "sustained", else "bump".

"detect on: torque - baseline" skips the motion model, to compare.

## Page

Left: the clip. Top: threshold slider, detection signal, coincidence window, minimum joints per event, skip operator
spans, time window, optional position / velocity strips. Raster: the whole episode, one row per joint (tick box =
joint on / off), a bar per event with a tick at the peak, a vertical line joining coincident joints. Panels: per
enabled joint, torque with the expected torque and a dot on each peak, and the score with the threshold. Right: the
event list (peak time, kind, joints with signed N*m and sigma), sortable by time or strength.

Click the raster, a panel or a list row to seek. `n` / `p` = next / previous event, `l` = loop the selected event
(1 s before to 1 s after), space = play / pause, arrows = one frame, shift + arrows = one second, mouse wheel on the
panels = change the time window.

## Findings

- A posture (gravity) regression does not generalise: fitted on blocks of one run or across the two runs, linear,
  trigonometric and quadratic posture terms all leave a larger residual than the joint's own velocity terms alone.
  Hence the rolling baseline.
- The motion model matters for gripper, wrist_roll and shoulder_pan (residual spread 0.17 -> 0.07, 0.13 -> 0.07,
  1.1 -> 0.66 N*m on 15-09-30), little for wrist_flex and wrist_yaw. shoulder_lift and elbow_flex keep large
  unexplained swings (5-15 N*m) that come in pairs with opposite sign; probably arm motion, not contact. Not checked.
- The +-9.99 N*m frames on wrist_flex (14-50-14, 29.9-30.1 s) are a range wrap, 6 frames in a row, not single-frame
  glitches: the torque passes -10 N*m and reads +9.9. Unwrapped, it is one event of -8.3 N*m over the baseline.
- Gripper, threshold 6: 14-50-14 gives 4 events: 17.9 s (+1.9 N*m, sustained 3.1 s, extent 14.7-19.1 s, 3 crests
  of one peak), 30.8 s, 42.8 s, 76.9 s. 15-09-30 gives 6: 25.1, 83.8, 96.4 s (+1.8 to +1.9 N*m bumps), 165.3 s
  (+1.6 N*m, 3 crests, operator span), 31.8 and 33.6 s (+0.5, +0.75 N*m).
- All joints, threshold 6: 29 events on 15-09-30 (7 on several joints), 18 on 14-50-14 (5 on several joints). The
  83.8 s gripper event on 15-09-30 coincides with 5 arm joints.
- Not done: nobody has yet judged from the videos which events are contact.
