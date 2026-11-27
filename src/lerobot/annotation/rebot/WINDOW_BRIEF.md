# ReBot annotation under annotation_rubric.md: agent brief

One agent labels ONE window: a stretch of one teleoperated episode (ReBot arm, top camera + wrist camera, 30 fps).
The user reviews the labels on clips afterwards, so be exact about frames and say when you are unsure.

Run every command from the workspace root (the folder holding `lerobot/` and `outputs/`). Sources are read only; the
one file you write is `labels.json` in your window folder. `<work>` is the results folder of the pass, `<window>` the
name of your window.

## Read first

`lerobot/src/lerobot/annotation/rubrics/annotation_rubric.md`, in full. It is the rule. Do not label speed or the tip
target, and do not apply the 1 s look-ahead offsets (all computed later).

Two points the earlier passes got wrong:

- **Quality** (rubric section 5): be balanced and use your judgement. One pass called more than half of the segments
  high (too generous), the next one a tenth (too strict). High = decisive and on target, judged from the wrist view.
- **Mistakes** (rubric section 4) cover the whole mistaken motion: `start` is the frame the motion that goes wrong
  starts (a missed grasp or a slip: the frame the gripper command starts closing for that grasp, the first stage of a
  staged close; read it from `grip_command` in `signals.csv` and check it on the frames), `event_frame` is the event.

## Your material: `<work>/<window>/`

| file | what |
|---|---|
| `window.json` | root, episode, first and last frame (end exclusive) of the window, `episode_frames` = length of the episode (a window that ends before it stops mid-task: the episode goes on, so its last segment is not a return home), `holds` = [start, end) stretches where the gripper is gripping something (torque above 0.4 N*m, until back under 0.15) |
| `sheet_*.jpg` | every 10th frame, top view over wrist view, frame index and gripper angle burned in |
| `signals_*.png` | per frame: gripper angle / command, gripper torque, fingertip speed, distance of the tip to where it is at the next gripper event, wobble, command minus position |
| `signals.csv` | the same numbers per frame |
| `labels_prev.json` | only in a repeat pass: the labels to start from |

Look at ALL sheets. The holds are a proposal for where an object is gripped; check each against the frames (a hold
can be a failed grasp, a push with closed fingers, or two grasps merged). The list also misses holds: a nut pinched
loosely at the fingertips shows no torque (the command equals the position), so fingers that stay nearly shut (around
-9 deg) while the arm travels or waits over a place are a carry until enlarged wrist crops of the fingertips show them
empty; another nut left on the table does not show that. For every boundary, mistake and quality call
look at denser frames, wrist view first:

```
uv run python -m lerobot.annotation.rebot.sheets <root> <work>/<window>/fine <episode> --start <f> --end <f> --step 3 --tag fine
```

The fingertip path numbers of your segments (directness, backs off, longest pause, turns; rubric section 5 says what
is typical for high). Write your segments to `<work>/<window>/segments.json` (in your window folder, not a shared scratch folder: other agents run at the same time) as `{"segments": [{"start", "end", "subtask"}]}` first:

```
uv run python -m lerobot.annotation.rebot.tip_motion <root> <episode> --segments <file>
```

## Output: `<work>/<window>/labels.json`

```json
{
 "window": "<name>", "episode": 0,
 "segments": [
  {"start": 0, "end": 0, "class": "approach | grasp | move | release | reposition | return home", "subtask": "approach die",
   "quality": "high or null",
   "checks": {"centred_in_wrist_view": "mostly | partly | no", "decisive": "yes | no", "first_close_held": "yes | no | n/a"},
   "evidence": "one sentence: what makes it high, or the first thing that rules it out (frames)",
   "note": "anything unsure, or empty"}
 ],
 "mistakes": [{"start": 0, "event_frame": 0, "type": "failed_close", "note": "what happened, what the result is"}],
 "objects": "object and place names you used that are not in the rubric's list, or empty",
 "rubric_problems": "any rule in the rubric that did not fit what you saw, or empty"
}
```

Segments tile the whole window (no gap, no overlap; episode frame indices; the first starts at the window start, the
last ends at the window end). A window that starts or ends mid-episode cuts its first and last segment: label what
they show. A grasp whose result falls past the window end: say so in its note. A mistake that crosses a window edge
is recorded by the window that holds its event frame, with `start` clamped to the window start: when your window opens
on closed or reopening fingers, check whether the close just before it held anything, and when a close is still under
way at your window end, give its start frame in your summary.

Reply with a short summary: how many high per class, the mistakes with their start and event frames, and the open
doubts.
