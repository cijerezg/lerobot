# camera_preview

Question: what do the wrist and top cameras see right now, without connecting the follower or the leader arm?

Run from the workspace root with the same cameras dict as `--robot.cameras` in `CHEAT_SHEET.md`:

```
uv run --no-project --python .venv/bin/python lerobot/adhocs/camera_preview/preview.py \
    --cameras="{ wrist: {type: intelrealsense, serial_number_or_name: \"427622270837\", width: 640, height: 480, fps: 30, use_depth: true, depth_filters: true, log_path: outputs/logs/realsense_wrist.log, log_severity: info}, top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._onn_USB_2.0_webcam_SN0001-video-index0, width: 640, height: 480, fps: 30}}"
```

Opens Rerun with `wrist`, `wrist.depth` and `top`. Ctrl-C to stop. Nothing is sent to any motor bus.
