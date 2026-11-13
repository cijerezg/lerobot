# Correction rubric: corrected action chunks from hindsight clips

> **Status:** in use from 2026-11-09. Code: `annotation/corrections/`. Passes: `outputs/_annotation/corrections_v1` (grade 1-2 demos, annotated rollouts), `corrections_v2` (grade 3 demos, cut footage; skip added 2026-11-10), `corrections_v4_dense` (the 1 s grid frames between the 3 s frames of v1, v2's cut footage and v3), `corrections_v5`
(the 2026-11-11 rollouts, 1 s grid from the start). Anchors (the episode's own grasp as the target) added 2026-11-12
after the grasp test; passes v1-v5 were labelled without them.

One frame = one recorded chunk (30 actions, 1 s). The labeller reads the clip around the frame, works out what the
hand was trying to do, and writes the manoeuvre it should have done as 1-3 tool poses. `corrections.build` turns the
poses into a smooth path and keeps its first 1 s as the corrected chunk. Only corrected frames are trained on.

## Verdict first (user 2026-11-12; pilot)

Most recorded chunks are roughly right. Before writing any pose, follow what the recorded actions do: the rows and
tiles after t = 0 show where they take the tool, up to 8 s on, and `look` shows any other frame of the episode, as
often as needed. Then give the frame one verdict. The correction changes the recording as little as the verdict
allows, and a target that cannot be pinned down is not guessed.

| `verdict` | When | What to write |
|---|---|---|
| `right` | The motion goes where the subtask needs and gets there: the grasp holds, the carry arrives, the release lands. Slow is still right: a motion still on its way at +8 s is followed with `look` until it arrives | The same motion, which the builder runs at full pace: 1-3 waypoints, each a `like_frame` of a frame the tool itself passes through on the way, ending where that motion ends (for a grasp: the `ANCHOR` waypoint, close, lift; for a carry or a place: the frame where it arrives). Skip the recording's pauses and detours, change nothing else |
| `adjust` | The motion goes for the right thing and ends a little off, by an amount the tiles show: short of the object, 1-4 cm to one side, too high, closing early or late, a hesitation on the way | The pose where it ended, put right: `{"like_frame": L, "shift": [dx, dy, dz]}` with L the frame where the recorded motion ends and the shift the offset seen, in metres in the world frame, at most 4 cm. Where the episode itself gets it right later (an `ANCHOR`, a later frame at the place), that frame's `like_frame` is the adjustment and no shift is needed. A gripper that opens or closes at the wrong moment on an otherwise right motion is an `adjust` too: the same `like_frame`s with `gripper` set. `wrong` says what was off and by how much |
| `redo` | The motion is wrong (wrong object, drifting off, stalled, wandering) and the right target is known: an `ANCHOR`, a measured row of the episode at that place, or a move where 3 cm either way does not matter (a carry, a release over the basket, backing off, going to rest), or a grasp of a large object placed free (the body of a bottle standing or lying, a cup, a fold of cloth) whose `preview` on this frame or a `look` frame of the same stretch shows both pads clear of the object on either side | The manoeuvre from here, by the sections below |
| `uncorrectable` | The motion is wrong and the right target cannot be pinned down: a grasp of a small or thin object (bit, clipper, flap edge, lid tab, cap) with no `ANCHOR` and no measured row at it; a large object whose free placement cannot be shown in a preview (no frame of the stretch where the object has not moved shows it with depth), or that moves during the clip; no clear intent | `{"skip": "uncorrectable: <why>"}` as the whole file. Nothing else: no guess |

When torn between two verdicts, take the one higher in the table only if the hindsight shows it (a grasp is `right`
only when the object is seen held). `uncorrectable` is an honest answer and costs nothing: a guessed target a few
centimetres off is worse than no correction. `adjust` and `redo` can end in the same waypoints (both go to the
`ANCHOR`): the verdict then says what the recording did, `adjust` when it was going for the right thing, `redo` when
it was not. `verdict` is a field of `waypoints.json`.

The aim (user 2026-11-12, full relabel): corrections that are right in direction. A small behaviour that can be
fixed on the way (a late close, a hesitation, an early release) is fixed too; nothing has to be perfect. Be generous
with `look`: pull as many other frames of the episode as it takes to see what the hand was after and what came of
it. A frame the labeller is not sure about still gets its best verdict, plus `"flag": "<what is unsure, in a few
words>"` in `waypoints.json` (also next to `skip`). Flagged frames are reviewed apart. A flag is for doubt
about where the correction goes (which object, where it lies, a pose that may be 2 cm or more off or may push the
object); not for a doubt between `adjust` and `redo` with the same waypoints, an estimated gripper angle, or about
1 cm on a reused pose (first full pass, v5: 69% of frames flagged, mostly for those).

A release into a slot, onto a peg or into a fitted seat (a bit into its slot, a tube onto a peg) is not a release
over the basket: it needs a measured pose at that place (a `like_frame` where the episode gets there), else it is
`uncorrectable`.

A carry or a place ends where the tool arrives (over the basket, on the spot), found with `look`; the "stretch ends
at frame E" line is where the annotation changes subtask, which can be before the arrival.

## Frames of a pass

One frame per second (user 2026-11-11; `every` 30 in `pass.json`, the default of `corrections.sample`). An annotated
policy run is taken whole: policy frames, operator (teleop) frames and the footage cut out of it. Demonstrations
give their grade 1-2 frames on the same 1 s grid. The one exception, settled 2026-11-11: grade 3 demo frames stay
on the 3 s grid. Passes v1-v3 were sampled every 3 s (`every` 90 in their `pass.json`); v4_dense added their 1 s
frames.

## Unit of work

One labeller takes one job: an episode folder and a run of its frame folders (a line of `<work>/jobs.txt`; a long
episode is several jobs, done in order). It labels every frame folder of the job, in frame order.

1. Read `episode.txt` once, and `poses.txt` if the episode folder has one.
2. For each frame folder, in order: read each file once, write `waypoints.json`, move on.
3. Each time an object's contact pose or a place (drop point, set-down point) is worked out or changes, append one
   line to `poses.txt`: frame folder, object or place, `tip`, `approach`, `jaw`, gripper angles, and why it changed.
   A `poses.txt` that is already there may come from an earlier pass over the same episode on a coarser grid
   (`# from corrections_v<N>` lines): it then covers the whole episode, later frames included. For each frame use
   the line in force at that frame, and append only what a clip shows and those lines miss.
4. No code. Other frames of the episode may be looked at with `look` (Depth inputs), and the `case.txt` of another
   frame folder of this episode read, to find a row where the tool is at a place this clip does not reach.

Every frame gets a verdict (above); the table below says what the manoeuvre of a `redo` aims for.

| Frame kind (stated in `case.txt`) | What the recorded chunk is | What to write |
|---|---|---|
| policy | the policy's own actions | the manoeuvre that reaches the subtask's goal from here |
| operator | a human taking over after the policy failed | the same; the operator's later motion is a hint, not the answer |
| demo, grade 1 or 2 | a poor stretch of a human demonstration | the same |
| demo, grade 3 | a passable stretch of a demonstration: a weaker grasp choice, a search or a hover | the same; where the grasp choice is the weakness, the object table's grasp replaces it |
| policy cut, operator cut | footage of a policy run that the review removed: wandering with no clear target, resets, trims at the start and end, idle stretches | the same when the clip shows what the hand should do; otherwise skip |

**Cut footage.** The subtask on these frames is the prompt the policy had at that moment, never checked by an
annotator: a hint. Take the target from the clip (the object the hand is at or heading to, the object it holds) and
set `contact` to match. Skip the frame, with `{"skip": "<reason in a few words>"}` as the whole `waypoints.json`,
only when no correction can be written: no object the hand could be going for, a person's hand or a rearranged
scene inside the clip (objects jump between tiles), the task over with the arm parked, images too dark or blocked.
A lost or stalled policy is not a reason to skip: that is the frame the correction is for.

When the scene is no longer what the subtask assumes (object dropped, knocked over, in the wrong place), correct
toward the object where it lies now and set `contact` to what that takes: the Objects table goes by how the object
stands or lies at this frame, whatever the contact label says. The subtask label is normally right; where
it plainly disagrees with the clip (labelled a carry with an empty gripper), follow the clip and say so in `note`.
A hand heading to another object than the one the subtask names is not such a disagreement: on annotated frames
(policy, operator, demo) the subtask's object is the target and the drift is the error, so the correction goes to
the subtask's object (2026-11-11: the policy drifting to the purple bottle under "grasp the blue pill bottle").
The correction always agrees with the subtask and `contact` of its frame.
When the task is already done (object released, arm going back), go to the rest pose (`like_frame` 0) with
`contact` `na`.

**One pose per object per episode.** The first time an object's contact pose is worked out in an episode, later
frames reuse that `tip`, `approach` and `jaw` (from `poses.txt`) unless a later clip shows the object has moved by
3 cm or more, or the `preview` of the reused pose on the later frame shows a pad on the object: the preview wins.
Re-deriving it per frame makes the labels disagree by centimetres.

## Input

`<work>/<source>/ep<NNN>/episode.txt`: task, frame count, gripper stall angles for the episode's subtasks (measured
over the whole pass), then one row per frame folder: `folder | time_s | subtask | contact | frame kind`.

`<work>/<source>/ep<NNN>/f<NNNNNN>/`:

| File | Content |
|---|---|
| `case.txt` | line 1: case id, task, subtask, contact, precision, frame kind. Then the tool pose at t = 0, the stall angles, and one row per clip frame: measured tool tip, approach vector, gripper, and the recorded action's tip and gripper. A row ending in `operator` was commanded by a human. Last, when the recording has depth, the wrist depth grid |
| `wrist_a.png` | the wrist camera's clip, first half: t = -5, -2, -1, -0.5, 0, +0.2 s |
| `wrist_b.png` | the wrist camera's clip, second half: t = +0.4, +0.6, +0.8, +1.0, +1.5, +2.0 s |
| `wrist_c.png` | what came of it (passes dumped from 2026-11-12): t = +2.5, +3, +4, +5, +6.5, +8 s |
| `top.png` | top camera at t = -5, 0, +1, +2 s (and +4, +8 s from 2026-11-12). On the t = 0 tile of the 2026-11-09 rollouts: red circle = tool tip, magenta = tip path of the recorded chunk (good to 1-2 cm). Other recordings have no marks |

t = 0 is the chunk start. Frames after 0 are hindsight: what the recorded actions led to. The wrist view is the main
view. The top view is context only: what is on the table and roughly where. No position is ever read off it in
pixels. Clip times before the episode start repeat frame 0.

## Depth inputs (a pass dumped with `dump --depth`; first used 2026-11-12)

The frame folder then also holds the files below, and step 4 of the unit of work is lifted: the labeller may look
at any other frame of the episode, as often as it needs, with

`uv run python -m lerobot.annotation.corrections.look <work> <source> <episode> <frame> [<frame> ...] [--at U,V ...]`

(from the workspace root; frames are episode-local). It writes `wrist.png`, `top.png`, `wrist_depth.png` and
`depth_map.png` for each frame under `ep<NNN>/look/f<NNNNNN>/` and prints the measured tool pose, the recorded
action, the depth table, and for each `--at` pixel its reading and its point in the camera frame.

| File | Content |
|---|---|
| `wrist_depth.png` | the t = 0 wrist image at 2x; each number is the depth reading in cm at the magenta dot above it (16 x 12 grid, every 40 px of the 640 x 480 image); `u`, `v` rulers on the top and left edges |
| `depth_map.png` | the same depth image in colour: blue 7 cm to red 40 cm and beyond, black = no reading |
| `case.txt` | the 16 x 12 readings as a table, and the formula from a pixel and its reading to a point in the camera frame |

Depth is metric: with the formula, two pixels and their readings give the distance between two points in cm (finger
tip to object, object to table, object width). The camera is fixed to the hand, so a camera-frame offset between the
finger tips and a point is the move the tool has to make in the hand's own axes; the measured rows of two frames
between which the tool moved tell how the camera axes lie in the world. The sensor reads nothing nearer than about
7 cm, and readings smear by 10-20 px across an object's edge, so take them from inside a surface.

### World positions and `preview` (2026-11-12)

The wrist camera's pose on the hand is known (the twin's camera, checked on the real depth), so every depth reading
is also a point in the robot frame, the frame of `tip`. No camera fitting is needed, and positions come from the
depth rather than from rule 7's finger-width estimates.

| Where | Content |
|---|---|
| `case.txt`, `look` | under the depth table: the world position (x, y, z in cm) of the surface at each grid pixel, and the height of the largest level surface in view (normally the table) |
| `height_map.png` | the wrist image at 2x coloured by the height (world z) of the surface, with the height in cm written at the grid points |
| `look ... --at U,V` | the world position of the surface at that pixel, in metres |

A world position does not depend on which frame it was read in, as long as the object has not moved: an earlier
frame, with the camera higher up, often shows the object and the table around it better than t = 0 does. Heights
read this way carry the same offset as the measured tips of the clip (both come from the same joints), within about
1 cm; the table is not at z = 0, read it. Positions are good to about 0.5 cm away from edges; at an edge the reading
smears, and thin objects (a bit, a clipper) read as the table under them: take their x, y from the pixel and their
height as the table's plus the object's thickness.

Placing a contact pose:

1. Find the grasp point on the object in a wrist image and read its world position with `--at`. Read one or two
   more points to get the object's long axis and its top; read the table next to it.
2. `tip` x, y = the grasp point (for a side-pinch, about 1 cm past the object's axis along `approach`); `tip` z from
   the contact pose table (rule 5), counted from the heights just read. `jaw` across the narrow dimension: the
   difference of two world points along the long axis, turned 90 deg about `approach`.
3. Write `waypoints.json` and run

   `uv run python -m lerobot.annotation.corrections.preview <work> <source> <episode> <frame> [--on F]`

   It writes `preview.png` into the frame folder: the wrist image with the first waypoint drawn in (yellow circles =
   the two finger pads, magenta cross = tip, magenta line = the fingers, grey = the tool now), and prints the chunk
   check and, for the tip and each pad, the surface the camera sees at that pixel and the mark's height over it.
   `--on F` draws on another frame and writes `preview_f<F>.png`. Pixels for `--at` are those of the 640 x 480
   image (the `u`, `v` rulers), not of the 2x pictures. A pose in free space (carry, release) has nothing to check
   in a preview.
4. Look at `preview.png`. The pads must sit on either side of the part to be held, with the object between them
   and the jaw line across it; the tip must not be under the table or deep inside a rigid object. Move the pose and
   run it again until the picture is right (normally one or two rounds). A pose is not handed in before its preview
   has been looked at.

## Anchors (2026-11-12)

A grasp placed from the images lands about 2.5 cm from where the object was really taken, with or without depth
(`grasp_test`). The episode's own grasp has no such error. Where the episode goes on to grasp the subtask's object
and the grasp holds, `case.txt` ends with an `ANCHOR` line: the measured tool pose at that grasp, and the waypoint
that copies it. It outranks the contact pose table, the Objects table, `poses.txt` and anything read off the depth.

| The `ANCHOR` line says | What to write |
|---|---|
| nothing more | The anchor is the contact pose: copy the waypoint as given, then close (a waypoint with the same `like_frame` and the closing angle given; a flap, lid or cloth closes to 0 whatever the line says), then lift 5-8 cm. Run `preview` once (pilot 1, 2026-11-12: an object knocked over with no closing attempt, anchors 1.5 cm off) and go by the three rows below; expect "kept". This holds where the grasp is not the Objects table's (a bottle that was taken by the cap and held): it worked |
| `CHECK FIRST` (a closing attempt failed between this frame and the grasp) | The failed attempt may have pushed the object, so the anchor can be 1-4 cm from where the object lies at this frame. Write the anchor, run `preview` and look: do the pads sit on either side of the object as this frame shows it? If the pose is outside the image, use `--on F` with an earlier frame of the same stretch where the camera is further back. Then one of the three rows below |
| that grasp is in a stretch graded 1-3 | Use the anchor unless it breaks a rule of the Objects table (a bottle taken by the cap, the flap straight down); then the table's grasp, placed at the anchor's x, y |

After the preview of an anchor:

| Preview shows | Write |
|---|---|
| pads around the object (within about 1.5 cm; within about 0.5 cm for a bit, a clipper or another thin object, whose pads must straddle it) | the anchor as given; `note`: "anchor kept" |
| the object clearly elsewhere on the table, standing or lying the same way | the anchor moved level: `{"like_frame": G, "xy": [x, y], "gripper": O, "arrive": "approach"}` with x, y the world position of the grasp point on the object at this frame (`look --at`, or the pads' printed world positions plus the offset seen); the anchor's height, direction, jaw and opening stay. The closing waypoint carries the same `like_frame` and `xy`. Preview again until the pads are on the object. `note`: "anchor shifted" |
| the object in another posture than at the grasp (standing here, lying there; turned so the jaw no longer lies across it; the anchor on another part of it, neck against body), or not in any frame up to t = 0 | by the verdict table: a large object is placed free (Placing a contact pose), `note`: "anchor rejected: <why>"; anything else is `uncorrectable` |

When in doubt between kept and shifted, keep: the anchor is measured, the picture is a projection with up to 1 cm
of error (the grey marks show it: they should sit on the real fingers of the frame being labelled).

A frame without an `ANCHOR` line in a `grasp ...` subtask (the episode never takes the object in this stretch) is
placed free only where the verdict table allows a `redo` without a measured target; otherwise it is `uncorrectable`. The line "this stretch of the subtask ends at frame E" gives the measured pose where a non-grasp
stretch (carry, place, press) ended, a hint for its goal.

Measured grasps (867 that worked, all recordings; z = computed tip height, the table reads -2 to +3 cm):

| Object | Contact | Tip z cm (quartiles) | Tilt deg | Opening | Stall |
|---|---|---|---|---|---|
| bit | top-pinch | 1.8 - 2.3 - 2.6 | 18 | -57 | -6 |
| nail clipper | top-pinch | 1.9 - 2.6 - 3.4 | 17 | -89 | -28 |
| large nail clipper | top-pinch | 0.8 - 2.0 - 3.3 | 19 | -98 | -31 |
| sugar cube | top-pinch | 3.0 - 3.2 - 3.4 | 24 | -107 | -37 |
| eraser | top-pinch | 4.0 - 4.6 - 5.0 | 8 | -262 | -106 |
| pill bottle, standing | side-pinch | 5.0 - 6.3 - 7.1 | 63 | -198 | -128 |
| pill bottle, lying | top-pinch | 3.7 - 4.8 - 5.5 | 16 | -210 | -154 |
| spray bottle | side-pinch | 12.2 - 13.2 - 14.4 | 69 | -244 | -146 |
| water bottle, standing | side-pinch | 12.6 - 13.1 - 14.4 | 68 | -227 | -155 |
| water bottle, lying | top-pinch | 3.4 - 4.2 - 5.0 | 13 | -214 | -158 |
| tape roll | rim-pinch | 4.1 - 5.7 - 7.3 | 23 | -144 | -28 |
| cup | rim-pinch | 11.2 - 13.1 - 14.7 | 30 | -151 | -4 |
| bit-kit flap | cloth-pinch | 3.1 - 3.7 - 4.8 | 34 | -134 | -2 |
| headphone case lid | handle-grasp | 7.4 - 8.3 - 9.2 | 30 | -97 | -2 |
| sock | cloth-pinch | 3.9 - 5.0 - 6.4 | 13 | -172 | -2 |
| shirt | cloth-pinch | 6.4 - 8.3 - 10.7 | 23 | -166 | -2 |

A free placement takes its height, tilt and opening from this table, not from the depth: the depth gives x, y.

## Causal passes (the grasp test, `corrections.grasp_test`; 2026-11-12)

A pass with `"causal": true` in its `pass.json` tests the labeller: each frame is a moment of a demonstration
shortly before a grasp, and the tool pose where that grasp happened is held back as the answer. Nothing after t = 0
is shown, so there is no hindsight and no error to find: write the grasp the subtask asks for from here (contact
pose, close, lift), with `"wrong": "unknown"`. The rows before t = 0 show how the hand has been approaching.

| File | Content |
|---|---|
| `wrist_a.png` | the wrist clip, t = -8, -6, -5, -4, -3, -2.5 s |
| `wrist_b.png` | t = -2, -1.5, -1, -0.6, -0.3, 0 s |
| `top.png` | top camera at t = -5, -2, -1, 0 s |
| `case.txt` | rows for the wrist clip's frames only, then the depth grid at t = 0 |

One frame per episode, so no `poses.txt`. The only inputs are the frame folder, `episode.txt` and this rubric; the
recordings themselves are never opened. When the pass was dumped with `--depth`, `look` and `preview --on` work for
frames up to the frame being labelled and refuse later ones; without `--depth` there is no `look`.

## Geometry

| Item | Value |
|---|---|
| World frame | origin at the arm base, metres; +x away from the base (down in the top image), +y right in the top image, +z up |
| Table | z = 0 nominally. The computed tip reads -2 to +3 cm on the table depending on the recording, so take heights from this clip's rows, not from absolute numbers |
| tip | the point between the two finger tips |
| approach | unit vector along the fingers, base to tips |
| jaw | the line from one finger to the other; the fingers close along it |
| Tilt | angle of `approach` from straight down: 0 = vertical, 90 = horizontal |
| Gripper | degrees, 0 = closed, -270 = fully open, about 0.42 mm of opening per degree (-120 = 5 cm, -240 = 10 cm) |
| Wrist depth grid | distance from the wrist camera along its optical axis, cm, at the listed wrist-image pixels; `-` = no reading. Only in recordings with depth; never required |

## Output: `f<NNNNNN>/waypoints.json`

```json
{"verdict": "redo", "contact": "side-pinch", "intent": "what the hand is trying to do", "wrong": "what the recorded chunk gets wrong, or nothing",
 "note": "the manoeuvre in one line",
 "waypoints": [
  {"tip": [0.47, 0.25, 0.06], "approach": [0.66, 0.35, -0.42], "jaw": [-0.47, 0.88, 0.0], "gripper": -200, "arrive": "approach"},
  {"tip": [0.47, 0.25, 0.06], "approach": [0.66, 0.35, -0.42], "gripper": -128},
  {"tip": [0.47, 0.25, 0.13], "approach": [0.66, 0.35, -0.42]}
 ]}
```

| Field | Meaning | Default |
|---|---|---|
| `tip` | tool tip in the world frame | one of `tip`, `like_frame` is required |
| `like_frame` | copy the measured tool pose and gripper of that episode frame (any frame number of the episode that a `case.txt` row or the `ANCHOR` line lists) | |
| `xy` | with `like_frame` only: the copied pose moved level to this x, y (Anchors) | |
| `shift` | the pose moved by this offset, metres in the world frame (verdict `adjust`) | none |
| `approach` | finger direction at the pose | straight down |
| `jaw` | finger closing line at the pose | the previous pose's |
| `gripper` | commanded degrees at the pose; with `like_frame` it replaces the copied angle | the previous pose's |
| `arrive` | `"approach"` or a vector: the path curves in and ends travelling along it | straight line |

Top-level fields: `verdict`, `contact`, `intent`, `wrong`, `note`, `waypoints`, and `flag` (optional: the labeller
is not sure, and of what).

`contact` uses the names of `annotation/vocab.py` (`top-pinch`, `side-pinch`, `rim-pinch`, `handle-grasp`,
`cloth-pinch`, `pry`, `set-down`, `drop`, `insert`, `na`, ...).

## Rules

1. **Intent before geometry.** The subtask and contact label say what the hand should end up doing. Keep the
   recording's approach style unless the approach itself is the error or the object table below says otherwise.
2. **Go straight for the goal and cover ground.** The recordings move too slowly (fastest joint about 7 deg/s in the
   demonstrations, 9-13 in the rollouts). The builder runs every path at the arm's limit, 45 deg/s, so the corrected
   chunk covers as much ground as the goal allows. Your part is to give it the real destination: no hover pose, no
   intermediate small step, no pause between poses, no waiting pose, no cautious step toward the goal.
3. **The first waypoint is the contact pose.** Give the pose with the fingers in place around the object and
   `"arrive": "approach"`. The path then blends the travel and the descent into one curve. Never write a pose above
   the object followed by a vertical drop.
4. **Free-space moves** (carry, lift, back off) are one waypoint each, straight line. To settle onto a place from
   above use `"arrive": [0, 0, -1]`.
5. **Contact poses.** Tilts and heights are the medians of the grade 4-5 demonstrations.

   | Contact | Pose |
   |---|---|
   | side-pinch | tilt 60-70 deg, fingers pointing from the arm base toward the object; jaw horizontal, across the body; tip at mid-height of the body, about 1 cm past the object's axis; arrive sliding in sideways |
   | top-pinch | tilt 15-20 deg, leaning the way the recording's approach vector leans near the object (away from the arm base when it gives no lead); jaw across the narrow dimension; tip height from the measured grasps table (Anchors): about 2 cm for a bit or a clipper |
   | rim-pinch | tilt about 30 deg so one finger goes inside the mouth; jaw across the wall thickness, perpendicular to the edge line; tip 1 cm below the top of the edge |
   | cloth-pinch | tilt 15-25 deg; pads on the edge or fold to be lifted |
   | handle-grasp | fingers on either side of the tab or handle, jaw across its thin dimension |
   | drop | carry to above the target, settle to 1-3 cm above the surface, open, back off up |
   | na (carry) | straight to the next place, `gripper` left out (the builder holds the measured angle at t = 0) |

   A grasp the clip shows working (the object held, the gripper stalled) outranks this table: keep its jaw line
   and height.

6. **Gripper.**

   | Object | Approach opening | Close to |
   |---|---|---|
   | rigid (bottle, clipper, bit, tape roll) | stall angle minus 50 to 70 deg (2-3 cm wider than the object) | stall angle plus 10 deg, at most 0 |
   | thin or soft (kit flap, lid, cloth) | -100 to -140 | 0 |

   Close with a waypoint at the same pose that changes only `gripper`. The listed stall angle is for the object
   as the subtask assumes it (a bottle standing). When this clip shows the gripper stalling on the object (measured
   angle stops while the command goes on closing), use that angle; it differs when the object is held another way
   (a pill bottle lying: about -98). With no stall angle at all, estimate the object width and use the 0.42 mm per
   degree scale. To release, open to the approach opening, not to -270.
7. **Positions come from the measured rows.** The measured tips in `case.txt` are exact in the robot frame; every
   `tip` is written relative to one of them. Read the hindsight for the error and turn it round: the rows after
   t = 0 show where the recorded actions took the tool, the wrist tiles show where the object is relative to the
   fingers at those times (left or right of them, short of them, below them, by how many finger widths; a finger is
   about 2 cm wide, the open jaw is 0.42 mm per degree). If the tool drifts away from the object, the correction goes
   back toward it; if it stops short, the correction goes that much further along the same line. For a place this
   clip never reaches (the basket, a slot), take the row of a later frame folder's `case.txt` where the tool is
   there, or `like_frame`. Only when the tool never gets there in the whole episode, estimate it from the images
   relative to the nearest measured row and say so in `note`.
8. **Heights.** Take z from rows where the tool is level with or touching the object, from the depth grid when
   present, and from the object's look in the wrist view.
9. **Reach.** Keep lifts to 5-8 cm above the grasp pose.
10. **A good chunk stays.** If the recorded chunk is already right (verdict `right`), write `"wrong": "nothing"`
    and give the pose it is heading to as a `like_frame`. The builder still runs it at full pace.

## Objects

Rules set by the user, with the numbers measured on the grade 4-5 demonstrations (482 grasps). They override rule 1's
"keep the recording's style" and the generic contact poses.

| Object | Contact | Grasp |
|---|---|---|
| Bit-kit flap | cloth-pinch | Never straight down. The pads take the blue binding of the flap's free outer edge, with the tool over the table just outside the kit. Tilt 30-35 deg, fingers pointing across the line from the arm base to the kit (from the side the free edge is on). Tip 3-4 cm above the table reading. Arrive on a slant, about as much sideways as down. Open -135, close to 0 |
| Standing bottle (pill, spray, water) | side-pinch | On the middle of the body, never on the cap and never from above. Tilt 60-70 deg, fingers pointing away from the arm base. Tip z about 6 cm (pill bottle), 12 cm (spray bottle), 13 cm (water bottle). Stall about -138 (pill), -155 (spray, water) |
| Bottle lying on its side | top-pinch | Across the middle of the body from above, tilt about 15 deg, tip z about 5 cm |
| Headphone case lid | handle-grasp | By its tab: a flat dark strip about a finger wide that sticks out of the lid's free edge, opposite the hinge. Pads on the tab, toward its free end (2026-11-12: the root of the tab, placed from the depth, was 3-5 cm from the real grasps). Tilt about 30 deg, fingers pointing away from the arm base. Tip z about 8 cm with the lid open (7.4 to 9.2 measured; do not take the height from the depth). Open -95, close to 0 |
