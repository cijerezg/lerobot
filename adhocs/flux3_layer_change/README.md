# flux3_layer_change (2026-11-26)

Question: after a short full-model finetune of FLUX 3 Action, in which layers do the representations differ
from the base? Input to the choice of what to freeze.

Both models get the training inputs of the same frames (same noise, same flow time). The base runs with the
checkpoint's `emb_in` / `final_layer` of the action and state streams (it has none of its own). Per layer and
stream, at the block's output: |h_ckpt - h_base| / |h_base|, cosine per token, token norm in the base. Also
|w - w0| / |w0| per block.

## Run

    uv run python lerobot/adhocs/flux3_layer_change/layer_change.py \
        --checkpoint outputs/runs/<run>/checkpoints/<step>/pretrained_model \
        --data outputs/data/rebot_teleop_2026-11-26-small-objects-cut-v2-annotated-v1 --frames 300 4000 9000

About 2 minutes, ~45 GB on the GPU (two DiTs). Prints the tables, writes
`outputs/analysis/flux3_layer_change_2026-11-26/<run>_<step>.json`.

## Found (rebot-flux3-test64-20261126, step 64, whole model trained, 6 frames)

Cosine per token, checkpoint vs base:

| layer | future video | snapshots | action | state | text |
|---|---|---|---|---|---|
| own 0 | 0.994 | 0.995 | 0.984 | 0.983 | 0.988 |
| own 3 | 0.962 | 0.936 | 0.976 | 0.976 | 0.941 |
| own 4 | 0.996 | 0.999 | 0.977 | 0.774 | 0.993 |
| joint 0 | 0.997 | 0.999 | 0.973 | 0.994 | 0.992 |
| joint 7 | 0.995 | 0.996 | 0.913 | 0.994 | 0.988 |
| joint 14 | 0.985 | 0.992 | 0.848 | 0.994 | 0.955 |
| joint 20 | 0.978 | 0.985 | 0.736 | 0.995 | 0.965 |
| joint 27 | 0.880 | 0.956 | 0.275 | 0.995 | 0.994 |

- Action tokens change steadily through the joint layers; video and snapshot tokens mostly in the last 6-8.
- State tokens have norm ~405,000 from joint 0 on and text tokens up to 1.8M, so use the cosine, not the
  relative change, for those two streams.
- Weight change is 0.006-0.008 in every block (says nothing this early); `time_in` 0.064, modulations 0.01-0.017.
- All 6 sampled flow times were between 0.90 and 1.00; low-noise steps were not measured.
