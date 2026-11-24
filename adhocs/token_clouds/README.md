# Token clouds

**Question.** None in particular: a page for moving the tokens of one language-model layer by hand and
watching the later layers, the action expert and the tip path follow, on a few frames, over an episode, or
on one frame under the checkpoints of a run.

## Run

From the workspace root:

```bash
uv run python lerobot/adhocs/token_clouds/clouds.py --config_path=config_rl_validate.yaml
```

Open http://localhost:8773. The checkpoint (`--model`, default the tube run at step 1200) stays on the GPU,
together with those of `--others` (default `base,100,200,400,700,internal=<path>`: the base model, training
steps of the same run, and `name=path` for a checkpoint of another run, here the internal-only run at step
1200). Each of these takes 9.6 GB of GPU memory (68 GB with the defaults); `--others ""` loads none. Start
takes about a minute, a new set of frames about 3 s, an episode about 0.5 s per time step, a frame through 6
checkpoints 12 s. `--root` (default the tube validation set) is the dataset, `--frames_per_episode` (default 4)
sets how many frames are offered, `--episode_frames` (default 120) the most time steps of one episode.

## The page

- **Side by side / Episode / Training**: up to 3 frames next to each other, one episode over time, or one
  frame under every loaded checkpoint.
- **Large view**: the prompt positions at one hidden index, as points along the first three principal
  directions of that layer. Formulas are in the docstring of `clouds.py`.
- **One token**: drag a point in the large view. It is picked alone and moves in the plane through it that
  faces the screen. Right-drag orbits from anywhere.
- **Picking**: Shift-drag a lasso, click a kind, or drag a rectangle over a camera image.
- **Moving**: the handle moves, turns or scales the picked points (W / E / R); S keeps them turning. The
  change is added to the input of the chosen block, inside the three shown directions only.
- **Small views**: five later layers, two expert blocks (action positions, first flow step) and the tip path
  of the chunk. Lines run from where a point was recorded to where it is; the faint points are the recorded
  ones. The arrows on a small view step its layer.
- **Bar strip**: per hidden index, mean distance moved / mean distance from the layer's mean. Dragging across
  it (or Up / Down) puts another hidden index in the large view; the handle stays at the moved one (L goes
  back).
- **Episode**: 3 time steps per second of footage on the image grid (fewer when that is more than
  `--episode_frames`: a 39 s episode gets 98 steps, 2.5 per second). The directions are fitted once, on 6 steps
  spread over the episode. The same image patch, or the same token in order within its kind, is the same
  point at every step. Space plays, Left / Right step. A move is the same 4 x 4 on the picked points at every
  step; with a move on, a step is shown when its answer is there, and the next 8 steps are asked for ahead.
  The tip view shows the chunk on the recorded tip path (first point of every chunk).
- **Training**: the steps of the timeline are the checkpoints, base first, checkpoints of other runs last. Space
  plays through them, Left / Right step. "Checkpoints" in the side panel picks two at a time: "shown" (the
  step of the timeline) and "with". Colour = Checkpoint (on when the mode opens) draws the two as solid points,
  orange and blue, in every view; with another colour the second is faint. The directions are fitted once, on the frame under all checkpoints, so a point is in the same
  space at every checkpoint. Without a move the faint points are the frame under the checkpoint chosen at
  "with" (base at the start), the lines run from there to the checkpoint on the screen, the bar strip
  shows the distance between the two per hidden index (mean distance between the two / mean distance from the
  layer's mean, over the points far at neither) and the tip view shows both paths. With a move, the faint
  points are the checkpoint on the screen as recorded, and the same move is applied at every checkpoint. The
  prompt, the normalization and the flow noise are those of `--model` for every checkpoint; weights a
  checkpoint file does not have (base has no tip adapter) are those of `--model`. So the internal-only
  checkpoint runs with the tube run's prompt and tip adapter, not its own depth path.
- **Size = expert attention**: point size from the rank, at this step, of the action expert's cross-attention
  to the token as recorded (mean over blocks, heads and action positions, first flow step): 0.6 x for the
  least to 2 x for the most, `0.6 + 1.4 rank^3`.
- **In the cloud**: which kinds of positions are points. The directions are fitted on these. Long image
  tokens and depth are off at the start because their vectors are far longer at the input.
- Positions longer than 4 x the median length at a layer are hidden there and left out of the fit (switch
  "Long tokens" shows them). From hidden index 17 on these are about 80 of 1,018 positions, mostly image
  tokens.

## Speed

One forward per moved frame. The vision backbone is not run again. While points move, three of four answers
stop before the action expert; the answer after the last move always runs it.

| | ms |
|---|---|
| moved frame, stop before the expert | 51 |
| moved frame, with the expert (5 flow steps) | 136 |
| episode, per time step at load (attention capture 216 + forward 124 + the rest) | 470 to 500 |

The block that is moved does not change the time. An episode with a move on plays at the footage's own speed
and at 2 x (headless Chromium: 5.6 s of footage in 6 s, 11.2 s at 2 x).

## First numbers (2026-11-19)

- Frames 0 and 1164: 1,018 points (509 per frame) with the starting kinds. Three directions hold 36% of the
  spread at the input, 27 to 30% in the middle, 53% after the last block.
- First-camera tokens of frame A shifted by 0.3 along the first direction: tip moved 0.24 cm (block 1),
  0.21 cm (block 13), 0.28 cm (block 31). The other frame's tip does not move (separate forwards).
- No move gives the recorded points and the tip back exactly.
- Episode 0 (39 s, 98 steps, 516 to 521 points): three directions hold 34% at the input; first-camera tokens
  turned at block 1 move the tip 0.2 to 0.4 cm, step by step.

## Checkpoints, first numbers (2026-11-19)

Tube run `rebot-v2-tube-b128-20261114`, frame 189 of the tube validation set, 441 points. Distance from base in
units of the layer's spread, and distance of the tip path (mean over the 30 points).

| checkpoint | index 0 | 12 | 24 | 36 | tip from base, cm | tip from step 1200, cm |
|---|---|---|---|---|---|---|
| 100 | 0.86 | 0.89 | 1.11 | 1.95 | about 2 | 0.66 |
| 200 | 0.92 | 0.95 | 1.22 | 1.99 | about 2 | 0.48 |
| 400 | 1.04 | 1.06 | 1.23 | 1.81 | about 2 | 0.41 |
| 700 | 0.99 | 1.09 | 1.26 | 1.72 | about 2 | 0.61 |
| 1200 | 1.15 | 1.22 | 1.29 | 1.71 | about 2 | 0 |

- The tokens are already 0.86 of the spread away from base at the input of the language model after 100
  steps: most points are image tokens, so this is the vision path changing.
- Three directions hold 28% / 25% / 46% of the spread at hidden index 0 / 18 / 36.
- First-camera tokens shifted by 0.3 at block 13: the tip moves 0.39 cm at base, 0.01 to 0.06 cm at the
  trained checkpoints.
- No move gives the recorded points and tip back exactly at every checkpoint, in any order. A moved frame with
  the expert takes 120 ms when the checkpoint changes on every request.

One frame; nothing here is a result yet.
