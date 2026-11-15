# tip_offset

Question (2026-11-14): can a per-frame "where is the target" vector be built from the recording alone, as the
training target of a depth adapter? Approach: the target of a frame is the gripper tip position at the next gripper
event of the same episode, minus the tip position now, in the current tool frame. Both come from forward kinematics
of `observation.state` (the twin's arm, `annotation/corrections/arm.py`); the subtask labels say which event.

```bash
uv run python lerobot/adhocs/tip_offset/build.py <subtask-labelled root>
```

Output: `lerobot/annotation_data/tip_offset/<root>/targets.parquet` (one row per frame, same order and `index` as
the root's data) with `provenance.json` (root, time, parameters, the event list), and
`outputs/analysis/tip_offset/<root>/check.jpg` (per event: the top view at the first frame that aims at it, the
wrist view 2 s before it, target circled; red = grasp, yellow = release). The root is not modified.

3D view (Rerun, after build.py; about 300 MB and 10 s per episode):

```bash
uv run python lerobot/adhocs/tip_offset/view.py <root> --episodes 0,9      # or all
uv run rerun outputs/analysis/tip_offset/<root>/episode_000.rrd
```

Scene: the arm (the twin's meshes at the recorded joints), a faded map of the table (every 3 mm cell that at least
5 depth frames of the episode saw, so an object that was moved shows at both places), the current wrist depth
frame on top in full color (every 4th pixel; holes, depth edges and everything outside 3 cm below to 30 cm above
the table dropped), the tip path of the episode (gray), the event points (purple = grasp, blue = release) and the arrow from the tip
to its target (green = the target is inside the wrist image, red = outside), the tool axes (red x, green y, blue z). Right column: subtask line, wrist image with
the target pixel, top image, offset components in cm in the axis colors. The slider is the frame index.

The cloud and the arrow do not agree exactly: the cloud is placed with the twin's wrist camera pose, the arrow
comes from the joints alone. While the arm moves, the table height read from the cloud varies by 0.6 to 0.8 cm
(sd, episodes 0 and 9); shifting the depth frames in time against the joints does not reduce it (best at 0 to -1
frames), so it is geometry (camera mount, arm model), not a delay.

## Events

- Grasp: a `grasp ...` segment followed by a `move` or `insert` segment. Event frame = start of the final hold: the
  frame after the last one where the gripper was more than 2 deg wider than the angle it leaves the segment with.
  The frames that aim at it: that segment up to the event, plus the `approach ...` segments of the same object
  right before it.
- Release: an `insert ...` segment with the `move` segments right before it. Event frame = the last frame where the
  gripper is within 2 deg of the tightest angle of that carry.
- No target (`kind` = none): `return home`, frames between an event and the end of its segment, grasp segments
  that are not followed by a carry, approaches that are not followed by their grasp.

## Columns

`kind` (grasp / release / none), `event_index` (dataset index of the event frame), `seconds_to_event`, `x` `y` `z`
(m; x = jaw axis, finger to finger; y = approach x jaw; z = approach axis, along the fingers), `distance` (m), `u`
`v` (event point in the wrist image, D405 intrinsics + the twin's wrist camera pose), `in_view` (inside the 640 x 480
image and in front of the camera), `placed` (release only: False when the next segment grasps again, so the tube
did not stay in the peg).

## Findings

`outputs/data/rebot_teleop_2026-11-14-tube-annotated-v1` (16 episodes, 34,836 frames, 123 segments): 55 events.

| kind | events | frames | event point in the wrist view | distance cm p10 / p50 / p90 |
|---|---|---|---|---|
| grasp | 28 | 17,291 | 79 % | 0.7 / 4.1 / 28.9 |
| release | 27 | 9,308 | 80 % | 0.5 / 2.8 / 13.3 |
| none | | 8,237 | | |

`outputs/data/rebot_teleop_2026-11-14-tube2-annotated-v1` (5 episodes, 13,299 frames, 56 segments): 26 events.

| kind | events | frames | event point in the wrist view | distance cm p10 / p50 / p90 |
|---|---|---|---|---|
| grasp | 13 | 7,140 | 77 % | 0.7 / 4.5 / 20.9 |
| release | 13 | 4,631 | 85 % | 0.4 / 2.3 / 12.1 |
| none | | 1,528 | | |

Both roots: 81 events, 38,370 of 48,135 frames with a target.

In the first root, 6 of the 27 releases are followed by a new grasp of the tube (`placed` False); none in the second. In the wrist views of the check
sheet the point lands on the tube between the fingers (grasp) and on the holder (release). The top-view circles
are a few cm off the objects: the top calibration is the twin's, not fitted to this camera position.

Not handled: an object that moves between the frame and the event (failed grasp attempts inside one `grasp`
segment), and the jaw axis sign (the gripper is symmetric, the model gives one sign).
