# Correction rubric: corrected action chunks from hindsight clips

> **Status:** in use from 2026-11-09. Code: `annotation/corrections/`. Passes: `outputs/_annotation/corrections_v1` (grade 1-2 demos, annotated rollouts), `corrections_v2` (grade 3 demos, cut footage; skip added 2026-11-10).

One frame = one recorded chunk (30 actions, 1 s). The labeller reads the clip around the frame, works out what the
hand was trying to do, and writes the manoeuvre it should have done as 1-3 tool poses. `corrections.build` turns the
poses into a smooth path and keeps its first 1 s as the corrected chunk. Only corrected frames are trained on.

## Unit of work

One labeller takes one job: an episode folder and a run of its frame folders (a line of `<work>/jobs.txt`; a long
episode is several jobs, done in order). It labels every frame folder of the job, in frame order.

1. Read `episode.txt` once, and `poses.txt` if the episode folder has one.
2. For each frame folder, in order: read each file once, write `waypoints.json`, move on.
3. Each time an object's contact pose or a place (drop point, set-down point) is worked out or changes, append one
   line to `poses.txt`: frame folder, object or place, `tip`, `approach`, `jaw`, gripper angles, and why it changed.
4. No code, no second look. The only other file allowed is the `case.txt` of another frame folder of this episode,
   to find a row where the tool is at a place this clip does not reach.

Every frame gets a correction. The one exception is cut footage (below).

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
toward the object where it lies now and set `contact` to what that takes. The subtask label is normally right; where
it plainly disagrees with the clip (labelled a carry with an empty gripper), follow the clip and say so in `note`.
When the task is already done (object released, arm going back), go to the rest pose (`like_frame` 0) with
`contact` `na`.

**One pose per object per episode.** The first time an object's contact pose is worked out in an episode, later
frames reuse that `tip`, `approach` and `jaw` (from `poses.txt`) unless a later clip shows the object has moved by
3 cm or more.
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
| `top.png` | top camera at t = -5, 0, +1, +2 s. On the t = 0 tile of the 2026-11-09 rollouts: red circle = tool tip, magenta = tip path of the recorded chunk (good to 1-2 cm). Other recordings have no marks |

t = 0 is the chunk start. Frames after 0 are hindsight: what the recorded actions led to. The wrist view is the main
view. The top view is context only: what is on the table and roughly where. No position is ever read off it in
pixels. Clip times before the episode start repeat frame 0.

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
{"contact": "side-pinch", "intent": "what the hand is trying to do", "wrong": "what the recorded chunk gets wrong, or nothing",
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
| `like_frame` | copy the measured tool pose and gripper of that episode frame (any frame number of the episode that a `case.txt` row lists) | |
| `approach` | finger direction at the pose | straight down |
| `jaw` | finger closing line at the pose | the previous pose's |
| `gripper` | commanded degrees at the pose | the previous pose's |
| `arrive` | `"approach"` or a vector: the path curves in and ends travelling along it | straight line |

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
   | top-pinch | tilt 15-20 deg, leaning the way the recording's approach vector leans near the object (away from the arm base when it gives no lead); jaw across the narrow dimension; tip 2-3 cm above the table reading for flat objects |
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
   | rigid (bottle, clipper, bit, tape roll) | stall angle minus 50 to 70 deg (2-3 cm wider than the object) | stall angle plus 10 deg; never 0 |
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
10. **A good chunk stays.** If the recorded chunk is already right, write `"wrong": "nothing"` and give the pose it
    is heading to. The builder still runs it at full pace.

## Objects

Rules set by the user, with the numbers measured on the grade 4-5 demonstrations (482 grasps). They override rule 1's
"keep the recording's style" and the generic contact poses.

| Object | Contact | Grasp |
|---|---|---|
| Bit-kit flap | cloth-pinch | Never straight down. The pads take the blue binding of the flap's free outer edge, with the tool over the table just outside the kit. Tilt 30-35 deg, fingers pointing across the line from the arm base to the kit (from the side the free edge is on). Tip 3-4 cm above the table reading. Arrive on a slant, about as much sideways as down. Open -135, close to 0 |
| Standing bottle (pill, spray, water) | side-pinch | On the middle of the body, never on the cap and never from above. Tilt 60-70 deg, fingers pointing away from the arm base. Tip z about 6 cm (pill bottle), 12 cm (spray bottle), 13 cm (water bottle). Stall about -138 (pill), -155 (spray, water) |
| Bottle lying on its side | top-pinch | Across the middle of the body from above, tilt about 15 deg, tip z about 5 cm |
| Headphone case lid | handle-grasp | By its tab: a flat dark strip about a finger wide that sticks out of the lid's free edge, opposite the hinge. Pads at the root of the tab where it meets the rim, one pad inside the shell. Tilt about 30 deg, fingers pointing away from the arm base. Tip z about 8 cm with the lid open. Open -95, close to 0 |
