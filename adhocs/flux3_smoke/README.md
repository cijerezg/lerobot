# flux3_smoke

Question: does FLUX 3 Action load and predict inside the fork's own environment, and what does one chunk cost
on the RTX PRO 6000 (latency, memory) at the rebot shape (7 wide, two cameras, chunk 42)?

Needs NATTEN in the venv and `flux-3-action-base` under `outputs/models/` (trunk, video VAE, text encoder).
No embodiment checkpoint is used: the policy is the base trunk with fresh action / state heads, which is what
a rebot finetune starts from.

```bash
uv run python lerobot/adhocs/flux3_smoke/smoke.py --base outputs/models/flux-3-action-base
# defaults: canvas 448 896 (each 480x640 view squashed to 448x448), guidance off, future video every 5th frame
# on a 256 512 canvas. Theirs: --future-stride 1 --future-canvas 448 896 --guidance 4.0. Also --steps 1, --compile
```

Inputs are noise images and a zero state, with placeholder quantiles: the numbers are load time, latency per
chunk and peak memory. The actions mean nothing (untrained heads).

`video.py` runs the same policy on frames of a rebot recording and writes the predicted future video under
the recorded one (`outputs/analysis/flux3_video_2026-11-26/`):

```bash
uv run python lerobot/adhocs/flux3_smoke/video.py --base outputs/models/flux-3-action-base \
    --data outputs/data/rebot_teleop_2026-11-15-tube3-cut-v4 --frames 300 1200   # --steps 16, --task "..."
```

Findings (2026-11-26, noise inputs, 448x896, chunk 42, 1 snapshot, 4 steps, guidance 4 / 1, not compiled):
load 25 s, 1,675 ms per chunk, 24.7 GiB peak. Guidance off 866 ms (704 compiled), + 2 steps 440 ms. The cost is the
11 future video latent frames (4,312 of ~5,100 tokens): with `--video-frames 3` 341 ms, with 1 213 ms
(176 compiled), timing only, the model was not trained with fewer. 3 snapshots not timed yet. Video on rebot frames: 11 clips written, the user has not reviewed them.

Future video at 6 Hz on 256x512 (`future_video_stride=5`, `future_canvas_hw=(256, 512)`, now the script
defaults; snapshot stays 448x896, guidance off): 3 latent frames x 8x16 = 384 video tokens instead of 4,312,
215 ms per chunk, 23.9 GiB peak. Training-side packing checked with the real VAE on a random 43-frame window
(384 tokens, time ids 3 / 70 / 136 = 0.03 / 0.70 / 1.37 s, spatial ids in snapshot-grid units). The base was
not trained with this layout; `video.py` has not been re-run with it.

Observation layout of the rebot finetune (script defaults: `--snapshots 3 --obs-stride 30`, state 21 wide =
position, velocity, torque of the 7 motors): 3 snapshots + 3 state tokens at 0 / -1 / -2 s, 348 ms per chunk,
24.2 GiB peak. One snapshot with the 7-wide state was 215 ms.

`video_checkpoint.py` (2026-11-26): the same clip for a `flux3_rl` checkpoint on frames of its training roots, with
the training inputs of the frame (3 snapshots, 3 states 21 wide, the command before now, the caption with the
annotated step and speed). Data path checked on the CPU; not run with a checkpoint yet (none exists).
