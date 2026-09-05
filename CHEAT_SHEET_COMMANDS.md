# reBot Commands

> **Run everything from this directory (the workspace root) with the root `.venv`.**
> Every command below uses `uv run --no-project --python .venv/bin/python`. There is no
> `pyproject.toml` at the workspace root, and `outputs/` + `config_rl.yaml` paths in the
> config are relative to it.
>
> Do **not** use `uv run --project lerobot ...`: that resolves against
> `lerobot/pyproject.toml` and builds a *separate* environment (cu128 pins, no
> matplotlib), which fails in ways that never name the venv.

## Teleop

uv run --no-project --python .venv/bin/python lerobot/src/lerobot/scripts/lerobot_teleoperate.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true, log_path: outputs/logs/realsense_wrist.log, log_severity: info}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.variant=102HD \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_hd_v1 \
    --display_data=true

## Record

uv run --no-project --python .venv/bin/python lerobot/src/lerobot/scripts/lerobot_record.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_v1 \
    --dataset.repo_id=cijerezg/rebot_sorting_clothes_v4-2 \
    --dataset.single_task="Put shirts in bin" \
    --dataset.fps=30 \
    --dataset.depth_stride=3 \
    --dataset.num_episodes=16 \
    --dataset.episode_time_s=300 \
    --dataset.reset_time_s=120 \
    --dataset.push_to_hub=false \
    --display_data=true

`depth_stride=3` writes PNG16 wrist depth at 10Hz instead of 30Hz (actions/proprio/RGB stay 30Hz).
Depth is ~83% of on-disk size and the buffer cache only reads stride-aligned rows, so this is a
3x saving on the dominant term with nothing lost. It must equal `policy.image_stride` in
config_rl.yaml, and that must divide `policy.chunk_size`.

## Offline training prep (run once per new dataset)

Anchor action stats (chunk-size must match policy.chunk_size; writes outputs/stats/action_stats_anchor_<dataset>.pt):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.compute_delta_stats \
    --root outputs/rebot_dataset_dummy_v1 \
    --encoding anchor \
    --chunk-size 30

Memmap buffer cache (pre-decodes frames so training doesn't hold all pixels in RAM; repo-id must match dataset.repo_id in config_rl.yaml):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --repo-id cijerezg/rebot_dataset_dummy_v1 \
    --data-dir outputs/rebot_dataset_dummy_v1 \
    --cache-dir outputs/buffer_cache-rebot-dummy-v1 \
    --image-storage-dtype uint8 \
    --image-stride 3

`--image-stride` must match `policy.image_stride` in config_rl.yaml (it is part of the cache
fingerprint — a mismatch is a hard error, not a silent fallback) and must divide `chunk_size`.

## Offline training (config: config_rl.yaml at repo root)


Full run:

uv run python -m lerobot.scripts.rl_offline --config_path=config_rl.yaml

## Aim metrics UI

Aim is enabled in config_rl.yaml, so the full training command above writes metrics while it runs.
There is no account, login, or upload step. Run these commands from the workspace root.

One-time environment setup (or after dependencies change):

uv sync --project lerobot --extra training --extra molmoact2 --extra pi

Terminal 1 - start training:

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.rl_offline \
    --config_path=config_rl.yaml

Terminal 2 - start the metrics UI (during or after training):

uv run aim up --repo ./aim

Open http://127.0.0.1:43800 in a browser. Refresh the runs page after training starts if it was
initially empty. Metrics are stored locally in ./aim; the same repository lets the UI compare
runs. Stop the UI with Ctrl+C. Training does not need the UI to be running.

Useful override - disable Aim for a smoke test:

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.rl_offline \
    --config_path=config_rl.yaml \
    --aim.enable=false

To use a different metrics directory, pass the same location to both training
(--aim.repo=PATH) and the UI (aim up --repo PATH). Do not delete ./aim unless you intend to
delete all locally stored run history.

## Probe viewer

Browser UI over a run's validation probes (serves on http://127.0.0.1:7870):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.view_probes outputs/molmoact2_offline_rebot_all-v6

Takes a run directory, its `validation/`, or a single `step_*` dir. It re-scans
`<run>/validation/step_*/<probe>/` on every request, so a checkpoint that lands
mid-session shows up on refresh — no export step. `--port` to move it, `--no-open` to
skip the browser tab. Probes are written by the validation loop only when their
`probe_parameters.enable_*` flag is set in config_rl.yaml.


## Remote validation on the DGX

Run a checkpoint's probe suite on the DGX from the main PC and pull the results back into
the mirrored local path. Both boxes use the same absolute paths, so the checkpoint,
run dir and results all keep their names.

lerobot/scripts/remote_validate.sh outputs/molmoact2_offline_rebot_v4/checkpoints/000400

It syncs `lerobot/src/lerobot/` + `config_rl.yaml`, pushes the checkpoint (~11 GB), runs

    rl_offline --policy.pretrained_path=<ckpt> --policy.offline_steps=0 --val_on_start=true
               --save_checkpoint=false --aim.enable=false --offline_output_dir=<run>

streams the log, rsyncs `<run>/` back (~280 MB), then deletes the remote checkpoint copy.
Results land in `outputs/remote_val/<run>-<step>/validation/step_00000000/`, which the probe
viewer above reads directly. Which probes run is still `probe_parameters.enable_*` in
config_rl.yaml — that file is part of the sync, so edit it locally.

The remote run is detached (`setsid`), so a dropped ssh does not kill it: Ctrl-C only stops
the tail, and `--attach <ckpt>` picks it back up. The checkpoint is kept on failure, and is
never deleted if it already existed on the DGX before the run (`--force-delete` overrides).

Needs a `Host dgx` block in `~/.ssh/config` with key auth (or `--host` / `$DGX_HOST`).
Static assets — the four dataset roots, `outputs/rebot_val-annotated-v3`, `outputs/MolmoAct2`,
the FAST tokenizer, `outputs/stats/`, the buffer cache — are NOT synced by this script; it
reads their paths out of config_rl.yaml and fails fast if any is missing on the DGX.

Other flags: `--dry-run` (prints the plan, needs no DGX), `--keep-checkpoint`,
`--out DIR`, `--config PATH`, and a leading code-tree argument to ship a different
checkout, e.g. `remote_validate.sh lerobot-tinypi outputs/.../checkpoints/000400`.



# probe motors
.venv/bin/python probe_rebot_motors.py