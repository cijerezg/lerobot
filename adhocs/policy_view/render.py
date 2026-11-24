"""The camera frames as the policy gets them: each view through the model's own resize (378 x 378, one crop).

uv run python lerobot/adhocs/policy_view/render.py <inference run dir or dataset root> [--grid] [--frames=N]

--frames=N stops after the first N frames of the root's first video file. Writes outputs/analysis/policy_view/<run name>/policy_view[_grid].mp4, top | wrist. --grid draws the 14 x 14
token cells (27 patches pooled 2 x 2, the last row and column are half cells).
"""

import importlib.util
import sys
from pathlib import Path

import av
import numpy as np

from lerobot.rl.inference_video import VideoWriter

SIZE = 378
PATCH = 14
VIEWS = ("top", "wrist")

spec = importlib.util.spec_from_file_location("image_processing", "outputs/models/MolmoAct2/image_processing_molmoact2.py")
image_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_processing)

run = Path(sys.argv[1])
grid = "--grid" in sys.argv
out = Path("outputs/analysis/policy_view") / run.name / ("policy_view_grid.mp4" if grid else "policy_view.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

frames_limit = next((int(arg.split("=")[1]) for arg in sys.argv if arg.startswith("--frames=")), None)
root = run / "inference_dataset" if (run / "inference_dataset").exists() else run
containers = [av.open(str(root / f"videos/observation.images.{view}/chunk-000/file-000.mp4")) for view in VIEWS]
writer = VideoWriter(out, 30, SIZE * len(VIEWS), SIZE)
for index, frames in enumerate(zip(*(container.decode(video=0) for container in containers))):
    if index == frames_limit:
        break
    tiles = []
    for frame in frames:
        resized = image_processing.resize_image(frame.to_ndarray(format="rgb24"), [SIZE, SIZE], 2)
        tile = (resized * 255).round().astype(np.uint8)
        if grid:
            tile[:: 2 * PATCH] = 255
            tile[:, :: 2 * PATCH] = 255
        tiles.append(tile)
    writer.write(np.concatenate(tiles, axis=1))
writer.close()
print(out, writer.frames, "frames")
