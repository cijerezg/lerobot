"""Record a handheld D405 scan as Open3D reconstruction-system frames.

Writes <out>/color/NNNNNN.jpg, <out>/depth/NNNNNN.png (raw z16, no filters), <out>/intrinsic.json
(colour intrinsics after align, depth_scale). Preview window: colour | depth with the valid-depth percentage.
Recording starts immediately. Keys: space = pause/resume, q = stop. Exposure is locked after warm-up so the colours stay constant.

    uv run python lerobot/scripts/sim_builders/digital_twin/scan_record.py --out assets/sim_sources/digital_twin/scans/table_01
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--res", default="848x480")
ap.add_argument("--fps", type=int, default=15)
args = ap.parse_args()
W, H = (int(v) for v in args.res.split("x"))
os.makedirs(f"{args.out}/color", exist_ok=True)
os.makedirs(f"{args.out}/depth", exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, args.fps)
config.enable_stream(rs.stream.color, W, H, rs.format.rgb8, args.fps)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
sensor = profile.get_device().first_depth_sensor()
depth_scale = sensor.get_depth_scale()

for _ in range(45):
    pipeline.wait_for_frames()
sensor.set_option(rs.option.enable_auto_exposure, 0)
print(f"exposure locked at {sensor.get_option(rs.option.exposure):.0f} us, depth scale {depth_scale} m/unit")

intr = align.process(pipeline.wait_for_frames()).get_color_frame().profile.as_video_stream_profile().get_intrinsics()
json.dump({"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
           "depth_scale": depth_scale, "fps": args.fps}, open(f"{args.out}/intrinsic.json", "w"), indent=1)

recording = True
i = 0
t_start = None
try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue
        depth_raw = np.asanyarray(depth.get_data())
        color_bgr = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_RGB2BGR)
        if recording:
            cv2.imwrite(f"{args.out}/color/{i:06d}.jpg", color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{args.out}/depth/{i:06d}.png", depth_raw)
            i += 1

        depth_mm = depth_raw * depth_scale * 1000
        valid = depth_mm > 0
        norm = np.clip((depth_mm - 70) / (1000 - 70), 0, 1)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[~valid] = 0
        label = f"{'REC' if recording else 'paused'} {i} frames  valid {100 * valid.mean():.0f}%  median {np.median(depth_mm[valid]) if valid.any() else 0:.0f} mm"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("scan", np.hstack((color_bgr, vis)))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
            t_start = t_start or time.perf_counter()
        elif key == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
print(f"{i} frames -> {args.out}")
