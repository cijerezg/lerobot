# FLUX 3 Action in the fork (started 2026-11-26)

> **Status:** stage 1 in progress. Policy ported and its tests pass (86 passed, 9 skipped) without weights; base trunk loads and predicts with fresh heads (866 ms per chunk at 448x896, chunk 42, 4 steps, no guidance; numbers in `adhocs/flux3_smoke/README.md`). Guidance stays off (1.0 on every stream). Future video is predicted at 6 Hz on a 256x512 canvas (`future_video_stride=5`, `future_canvas_hw=(256, 512)`; snapshots stay 448x896; future tokens keep their real times and are placed on the snapshot grid with fractional spatial ids): 384 video tokens instead of 4,312, 215 ms per chunk. The base was not trained with this layout, the finetune has to teach it. Observations: 3 snapshots and 3 state tokens at the same times, 0 / -1 / -2 s (`n_obs_steps=3`, `history_snapshots=3`, `obs_stride=30`); each state token is 21 wide (position, smoothed velocity, smoothed torque of the 7 motors, joints first), assembled by the caller; 348 ms per chunk with this layout. Caption (rendered by the trainer seam, no dropout, `caption_dropout=0`): `The task is to {task}. The current step is {subtask}. {No mistakes are made. | Mistakes are made.} The speed is {slow | normal | fast}.`; at run time the subtasks are pre-written and passed in, the mistake clause is always "No mistakes are made.". Tip offset: off for now (plan when it comes back: predicted as extra action channels, not passed in). Trainer seam built 2026-11-26 (`rl/flux3/`, policy type `flux3_rl`, runs in `rl_offline.py`): full finetune, lr 5e-5 trunk / 1e-4 heads, quantiles pooled over the training roots at start, windows that reach past the end of an episode left out of the loss. Smoke on the three annotated roots of 11-25 / 11-26 (146,739 frames): batch 8 = 75 GB and 3.1 s (batch 10 = 78 GB), a checkpoint is 13 GB, first losses video_mse 0.37, action_mse 1.7. Buffer cache: every frame at 448x448 uint8, `--no-depth` (177 GB). No rebot run yet.

Goal: replace MolmoAct2 with the world action model FLUX 3 Action (Black Forest Labs, 7B, joint video +
action flow matching), in stages. Stage 1 = the model runs natively in the fork's environment and predicts.

## What was ported

`policies/flux3/` is upstream LeRobot's policy (commit b07adcc, 2026-11-24) copied as is, plus
`policies/checkpoint.py`, `examples/flux3/` (base export) and `tests/policies/flux3/`. It runs in the fork's
venv; the old socket client and `scripts/flux3_server.py` were deleted, `rl/flux3/` now holds the trainer seam.

Differences from upstream, all because the fork's base classes are older:

| where | change |
|---|---|
| `configuration_flux3.py` | registers the draccus `torch.dtype` codec; declares `pretrained_revision`, `trust_remote_code` (upstream base-config fields) |
| `modeling_flux3.py` | `_load_as_safetensor` forwards to `_load_checkpoint` (the fork's `from_pretrained` calls the single-file hook) |
| `processor_flux3.py` | `lerobot.types`; no `trust_remote_code` on pipeline loading |
| `optim/schedulers.py` | `FrozenWarmupConstantSchedulerConfig` added (trunk frozen then warmed up, heads from step 0) |
| `processor/factory.py` | `make_default_policy_processor_steps`, `make_policy_processor_pipelines` added |
| `policies/factory.py` | `flux3` registered; pretrained processors go through the policy's own loader |
| `utils/import_utils.py` | `_natten_available` |
| tests | upstream trainer recipe, EMA and sharded-save tests skipped or dropped (`test_flux3_lora.py` not copied) |

## Stages

1. Build the base trunk with fresh rebot heads (no embodiment checkpoint; the base is what we finetune),
   time one chunk: `adhocs/flux3_smoke/`.
2. Finetune on rebot data (7 wide, so fresh heads): chunk 42 at 30 Hz, 448x448 per view (canvas 448x896, each 480x640 frame squashed), 3 history
   snapshots spaced about 1 s (needs snapshot offsets separate from the state window in
   `pack_history_video`), task string as caption. Needs a `Trainer` seam for `flux3`.
3. The fork's RTC correction inside their Euler loop (`f3/sampling.py`), then sim against the tube numbers.
4. Subtasks and the other annotations as caption text.
