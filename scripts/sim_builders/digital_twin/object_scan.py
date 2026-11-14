"""Record a D405 object scan; recording starts the moment the camera is up, q stops.

Writes scans/obj_<name>/color/NNNNNN.jpg, depth/NNNNNN.png (raw z16, no filters), intrinsic.json (colour intrinsics after
align, depth_scale) and color.mp4 (input for the real2sim object pipeline: extract_frames.py -> run_vggt.py).
Preview: colour | depth coloured over 7-50 cm with the valid-depth percentage and the median distance. Exposure locked after warm-up.

    uv run python lerobot/scripts/sim_builders/digital_twin/object_scan.py --name cup
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import pyrealsense2 as rs

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--res", default="848x480")
ap.add_argument("--fps", type=int, default=15)
args = ap.parse_args()
W, H = (int(v) for v in args.res.split("x"))
out = f"assets/sim_sources/digital_twin/scans/obj_{args.name}"
os.makedirs(f"{out}/color", exist_ok=True)
os.makedirs(f"{out}/depth", exist_ok=True)

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
print(f"exposure locked at {sensor.get_option(rs.option.exposure):.0f} us; recording to {out}, press q in the window to stop")

intr = align.process(pipeline.wait_for_frames()).get_color_frame().profile.as_video_stream_profile().get_intrinsics()
json.dump({"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
           "depth_scale": depth_scale, "fps": args.fps}, open(f"{out}/intrinsic.json", "w"), indent=1)

i = 0
try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue
        depth_raw = np.asanyarray(depth.get_data())
        color_bgr = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{out}/color/{i:06d}.jpg", color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(f"{out}/depth/{i:06d}.png", depth_raw)
        i += 1

        depth_mm = depth_raw * depth_scale * 1000
        valid = depth_mm > 0
        norm = np.clip((depth_mm - 70) / (500 - 70), 0, 1)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[~valid] = 0
        median = np.median(depth_mm[valid]) if valid.any() else 0
        cv2.putText(vis, f"REC {i}  valid {100 * valid.mean():.0f}%  median {median:.0f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(f"obj_{args.name}", np.hstack((color_bgr, vis)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", f"{out}/color/%06d.jpg",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", f"{out}/color.mp4"], check=True)
print(f"{i} frames -> {out} (color.mp4 written)")
