# reBot Commands

## Teleop

uv run lerobot/src/lerobot/scripts/lerobot_teleoperate.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_v1 \
    --display_data=true

## Record

uv run lerobot/src/lerobot/scripts/lerobot_record.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_v1 \
    --dataset.repo_id=cijerezg/rebot_sorting_clothes_v3-4 \
    --dataset.single_task="Put sock in brown basket and shirts in bin" \
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

uv run python -m lerobot.scripts.compute_delta_stats \
    --root outputs/rebot_dataset_dummy_v1 \
    --encoding anchor \
    --chunk-size 30

Memmap buffer cache (pre-decodes frames so training doesn't hold all pixels in RAM; repo-id must match dataset.repo_id in config_rl.yaml):

uv run python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --repo-id cijerezg/rebot_dataset_dummy_v1 \
    --data-dir outputs/rebot_dataset_dummy_v1 \
    --cache-dir outputs/buffer_cache-rebot-dummy-v1 \
    --image-storage-dtype uint8 \
    --image-stride 3

`--image-stride` must match `policy.image_stride` in config_rl.yaml (it is part of the cache
fingerprint — a mismatch is a hard error, not a silent fallback) and must divide `chunk_size`.

## Offline training (config: config_rl.yaml at repo root)

Smoke test (few steps, no checkpoints, no wandb):

uv run python -m lerobot.scripts.rl_offline \
    --config_path=config_rl.yaml \
    --policy.offline_steps=20 \
    --val_freq=0 \
    --save_checkpoint=false \
    --wandb.enable=false

Full run:

uv run python -m lerobot.scripts.rl_offline \
    --config_path=config_rl.yaml

## Probe viewer

Browser UI over a run's validation probes (serves on http://127.0.0.1:7870):

uv run python -m lerobot.scripts.view_probes outputs/molmoact2_offline_rebot_all-v2

Takes a run directory, its `validation/`, or a single `step_*` dir. It re-scans
`<run>/validation/step_*/<probe>/` on every request, so a checkpoint that lands
mid-session shows up on refresh — no export step. `--port` to move it, `--no-open` to
skip the browser tab. Probes are written by the validation loop only when their
`probe_parameters.enable_*` flag is set in config_rl.yaml.

