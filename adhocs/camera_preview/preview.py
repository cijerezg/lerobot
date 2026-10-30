"""Live preview of the cameras only: no robot, no leader arm.

Takes the same --cameras dict as --robot.cameras in the teleop command and streams colour (and depth,
where use_depth is set) into Rerun, read through the same camera classes the follower uses.
Ctrl-C to stop.
"""

import logging
import time
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun


@dataclass
class PreviewConfig:
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    fps: int = 30


@parser.wrap()
def main(cfg: PreviewConfig):
    init_logging()
    init_rerun(session_name="camera_preview")
    cameras = make_cameras_from_configs(cfg.cameras)
    for cam in cameras.values():
        cam.connect()
    logging.info(f"connected: {list(cameras)}")
    try:
        while True:
            t0 = time.perf_counter()
            obs = {}
            for key, cam in cameras.items():
                obs[key] = cam.async_read()
                if getattr(cfg.cameras[key], "use_depth", False):
                    obs[f"{key}.depth"] = cam.read_latest_depth()
            log_rerun_data(observation=obs)
            time.sleep(max(0.0, 1 / cfg.fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras.values():
            cam.disconnect()
        shutdown_rerun()


if __name__ == "__main__":
    main()
