"""Write per-frame uint16 depth as 16-bit PNG next to a LeRobotDataset.

Depth is deliberately kept out of LeRobot's image/video feature path (which is 8-bit and
would destroy the millimetre precision). Instead each `*.depth` observation is written as a
lossless 16-bit PNG, keyed to the dataset's episode/frame indices so the buffer-build step
can realign it into `complementary_info`.

Layout:  {dataset.root}/depth/{cam_key}.depth/episode-{ep:06d}/frame-{fr:06d}.png
"""

from pathlib import Path

import cv2


def write_depth(dataset, observation: dict) -> None:
    """Write every `*.depth` array in `observation` as a PNG16 for the frame just added.

    Must be called immediately AFTER `dataset.add_frame(...)`: the episode buffer is then
    populated (it's created lazily inside add_frame), `size` has been incremented past this
    frame, and `episode_index` is the writer's authoritative value — matching the
    (episode_index, frame_index) the dataset records, so the cache can realign by those keys.
    No-op if the observation carries no depth.
    """
    depth_items = [(k, v) for k, v in observation.items() if k.endswith(".depth")]
    if not depth_items:
        return

    ep_buffer = dataset.writer.episode_buffer
    ep_idx = ep_buffer["episode_index"]
    fr_idx = ep_buffer["size"] - 1  # add_frame already incremented size past this frame
    for key, depth in depth_items:
        ep_dir = Path(dataset.root) / "depth" / key / f"episode-{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # cv2 writes a uint16 (H, W) array as a lossless 16-bit grayscale PNG.
        cv2.imwrite(str(ep_dir / f"frame-{fr_idx:06d}.png"), depth)
