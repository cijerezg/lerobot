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
    --dataset.repo_id=cijerezg/rebot_dummy-v1 \
    --dataset.single_task="Describe the task here." \
    --dataset.fps=30 \
    --dataset.num_episodes=4 \
    --dataset.episode_time_s=240 \
    --dataset.reset_time_s=60 \
    --dataset.push_to_hub=false \
    --display_data=true

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
    --image-storage-dtype uint8

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

