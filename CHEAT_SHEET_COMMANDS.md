# reBot Commands

## Teleop

uv run lerobot/src/lerobot/scripts/lerobot_teleoperate.py \
    --robot.type=rebot_b601_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=rebot_follower_v1 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true}, top: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_leader_v1 \
    --display_data=true

