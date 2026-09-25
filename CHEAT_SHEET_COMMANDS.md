# reBot Commands

Run everything from the workspace root with the root `.venv`. Never `uv run --project lerobot` (builds a separate env).

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

`depth_stride` must equal `policy.image_stride` in config_rl.yaml (which must divide `policy.chunk_size`).

uv run --no-project --python .venv/bin/python lerobot/src/lerobot/scripts/lerobot_record.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_v1 \
    --dataset.repo_id=cijerezg/push-book-open-box-v1 \
    --dataset.single_task="push off the books and open the box" \
    --dataset.fps=30 \
    --dataset.depth_stride=3 \
    --dataset.num_episodes=16 \
    --dataset.episode_time_s=600 \
    --dataset.reset_time_s=120 \
    --dataset.push_to_hub=false \
    --display_data=true

## New dataset prep (once per dataset)

Anchor action stats (`--chunk-size` = `policy.chunk_size`):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.compute_delta_stats \
    --root outputs/rebot_dataset_dummy_v1 \
    --encoding anchor \
    --chunk-size 30

Memmap buffer cache (`--image-stride` = `policy.image_stride`, mismatch is a hard error):

uv run --no-project --python .venv/bin/python python -m lerobot.scripts.lerobot_memmap_buffer_cache \
    --repo-id cijerezg/rebot_dataset_dummy_v1 \
    --data-dir outputs/rebot_dataset_dummy_v1 \
    --cache-dir outputs/buffer_cache-rebot-dummy-v1 \
    --image-storage-dtype uint8 \
    --image-stride 3

## Offline training

uv run python -m lerobot.scripts.rl_offline --config_path=config_rl.yaml

Smoke test without Aim:

uv run python -m lerobot.scripts.rl_offline --config_path=config_rl.yaml --aim.enable=false

## Aim metrics UI

Env setup (once, or after dependency changes):

uv sync --project lerobot --extra training --extra molmoact2 --extra pi

Start the UI (http://127.0.0.1:43800), during or after training:

uv run aim up --repo ./aim

## Inference (standalone, RTC)

Loads `inference_checkpoint_path` from config_rl.yaml. `inference_send_actions_to_robot: false` = read-only preflight. Subtasks: `n` next, `b` back.

uv run python -m lerobot.rl.inference_async --config=config_rl.yaml

## Probe viewer

Browser UI over a run's probes (http://127.0.0.1:7870), rescans on refresh. Takes a run dir, `validation/`, or a `step_*` dir.

uv run python -m lerobot.scripts.view_probes outputs olmoact2_offline_rebot_all-v6

## Probe compare

Same probes as columns, one to four runs or checkpoints (http://127.0.0.1:7871). Same path forms as
the viewer; labels are optional and follow the paths in order. Only probes both passes ran line up,
so a pass with a different suite shows empty columns.

uv run python -m lerobot.scripts.compare_probes \
    outputs/<run>/validation/step_00000600 \
    outputs/molmoact2_rebot_diverse_stochastic_rounding_v2/validation/step_00000600 \
    --label=new-600 \
    --label=sr_v2-600

## Remote validation on the DGX

Run one checkpoint's probe suite on the DGX and pull results to `outputs/remote_val/<run>-<step>/`. Syncs code + config_rl.yaml; datasets/stats/cache must already be on the DGX.

lerobot/scripts/remote_validate.sh outputs/molmoact2_offline_rebot_v4/checkpoints/000400

Reattach to a detached run:

lerobot/scripts/remote_validate.sh --attach outputs/molmoact2_offline_rebot_v4/checkpoints/000400

Flags: `--dry-run`, `--keep-checkpoint`, `--force-delete`, `--out DIR`, `--config PATH`, `--host`.

## Chase a run's checkpoints (overnight)

Probes the newest unprobed checkpoint on the DGX until training exits or STOP. State in `<run>/validation/.chase/`.

setsid nohup lerobot/scripts/chase_validate.sh outputs/<run> > /dev/null 2>&1 &

Stop after the pass in flight:

touch outputs/<run>/validation/.chase/STOP

Probe a checkpoint by hand while the chase is up:

flock outputs/remote_val/.dgx.lock lerobot/scripts/remote_validate.sh <ckpt>

## Hardware checks

Probe follower motors:

.venv/bin/python probe_rebot_motors.py

Leader USB check after any re-plug (must be 200/200; ch341 must be under Bus 003/005 in `lsusb -t`, Bus 001 drops bytes):

.venv/bin/python migration/leader_bus_capture.py burst --trials 200 --spacing 0.033 --window 0.025

Follower chain order, base first (dead tail = cable upstream of first dead motor; one dead motor = its own board):

    #  joint          send  recv
    1  shoulder_pan      1    17
    2  shoulder_lift     2    18
    3  elbow_flex        3    19
    4  wrist_flex        4    20
    5  wrist_yaw         5    21
    6  wrist_roll        6    22
    7  gripper           7    23

Leader (102HD, ttyUSB0): ids 0-6 same order, id 7 = button board.
