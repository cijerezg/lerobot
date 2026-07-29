"""Write per-frame uint16 depth as 16-bit PNG next to a LeRobotDataset.

Depth is deliberately kept out of LeRobot's image/video feature path (which is 8-bit and
would destroy the millimetre precision). Instead each `*.depth` observation is written as a
lossless 16-bit PNG, keyed to the dataset's episode/frame indices so the buffer-build step
can realign it into `complementary_info`.

Layout:  {dataset.root}/depth/{cam_key}.depth/episode-{ep:06d}/frame-{fr:06d}.png
"""

from pathlib import Path

import cv2


def write_depth(dataset, observation: dict, stride: int = 1) -> None:
    """Write every `*.depth` array in `observation` as a PNG16 for the frame just added.

    Must be called immediately AFTER `dataset.add_frame(...)`: the episode buffer is then
    populated (it's created lazily inside add_frame), `size` has been incremented past this
    frame, and `episode_index` is the writer's authoritative value — matching the
    (episode_index, frame_index) the dataset records, so the cache can realign by those keys.
    No-op if the observation carries no depth.

    `stride` > 1 records depth at a fraction of the dataset fps (e.g. 3 -> 10Hz on a 30Hz
    dataset), since the buffer cache only reads depth on stride-aligned rows anyway and
    PNG16 frames dominate on-disk size. The phase is taken on the GLOBAL frame index
    (`meta.total_frames + fr_idx`, the same arithmetic save_episode uses to fill the
    `index` column) rather than the per-episode one, because the cache builder gates its
    reads on its own global counter. Keying per-episode instead would drift out of phase
    whenever an episode length is not a multiple of the stride, and the cache would then
    ask for frames that were never written. `stride` must divide the consumer's
    `image_stride`, which in turn must divide `chunk_size`.
    """
    depth_items = [(k, v) for k, v in observation.items() if k.endswith(".depth")]
    if not depth_items:
        return

    ep_buffer = dataset.writer.episode_buffer
    ep_idx = ep_buffer["episode_index"]
    fr_idx = ep_buffer["size"] - 1  # add_frame already incremented size past this frame
    if (dataset.meta.total_frames + fr_idx) % stride:
        return
    for key, depth in depth_items:
        ep_dir = Path(dataset.root) / "depth" / key / f"episode-{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # cv2 writes a uint16 (H, W) array as a lossless 16-bit grayscale PNG.
        cv2.imwrite(str(ep_dir / f"frame-{fr_idx:06d}.png"), depth)
