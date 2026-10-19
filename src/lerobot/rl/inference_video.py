"""Per-episode camera videos of a standalone inference run.

``record_inference_videos: true`` writes ``{output_dir}/videos/ep<N>_<view>.mp4`` for every view in
``inference_video_views`` (``top``, ``wrist``, ``wrist.depth`` -> ``ep0_top.mp4``, ``ep0_wrist.mp4``,
``ep0_wrist_depth.mp4``), one file per episode at the env fps, from the raw robot observation the
env worker steps from. A ``<cam>.depth`` view (uint16, 0.1 mm per level, 0 = invalid) is colorized
with TURBO over the valid pixels' 2-98 percentile (render_review_videos). H.264 yuv420p through PyAV,
so the files play in a browser ``<video>``: the sim eval's ``sim_nail/index.json`` lists them and
``compare_probes`` plays them side by side. Encoding runs on a daemon thread fed by a bounded queue,
the way the rerun worker does, so the control loop never waits on libx264; a full queue drops the
frame and the count is logged at the episode end.
"""

import logging
import queue
from pathlib import Path
from threading import Thread

import av
import cv2
import numpy as np

logger = logging.getLogger(__name__)

_END_EPISODE = "end_episode"


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """uint16 depth -> RGB uint8: TURBO over the 2-98 percentile of the valid pixels, invalid (0 / 65535) black."""
    d = np.asarray(depth).astype(np.float32)
    valid = (d > 0) & (d < 65535)
    scaled = np.zeros_like(d)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        scaled[valid] = np.clip((d[valid] - lo) / max(hi - lo, 1), 0, 1)
    rgb = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
    rgb[~valid] = 0
    return rgb


class VideoWriter:
    """Streams RGB uint8 HxWx3 frames into one H.264 yuv420p mp4."""

    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        self.path = Path(path)
        self.container = av.open(str(self.path), "w")
        self.stream = self.container.add_stream("libx264", fps, options={"crf": "20", "preset": "veryfast"})
        self.stream.pix_fmt = "yuv420p"
        self.stream.width = width
        self.stream.height = height
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def view_filename(episode: int, view: str) -> str:
    return f"ep{episode}_{view.replace('.', '_')}.mp4"


class EpisodeVideos:
    """One mp4 per view per episode under ``root``; ``put`` takes the raw robot observation of a step."""

    def __init__(self, root: Path, views: list[str], fps: int) -> None:
        self.root = Path(root)
        self.views = list(views)
        self.fps = fps
        self.queue: queue.Queue = queue.Queue(maxsize=64)
        self.dropped = 0
        self.episode = 0
        self.writers: dict[str, VideoWriter] = {}
        self.thread = Thread(target=self._worker, daemon=True, name="inference_video")
        self.thread.start()

    def put(self, raw_obs: dict) -> None:
        """Queue this step's frames: ``raw_obs["pixels"][cam]`` (HxWx3 uint8) and ``raw_obs["<cam>.depth"]`` (uint16)."""
        frames = {}
        for view in self.views:
            source = raw_obs[view] if view.endswith(".depth") else raw_obs["pixels"][view]
            frames[view] = source.numpy() if hasattr(source, "numpy") else np.asarray(source)
        try:
            self.queue.put_nowait(frames)
        except queue.Full:
            self.dropped += 1

    def end_episode(self) -> None:
        self.queue.put(_END_EPISODE)
        if self.dropped:
            logger.warning("[VIDEO] episode %d: %d frames dropped (encoder behind)", self.episode, self.dropped)
        self.dropped = 0

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                self._close_writers()
                return
            if item == _END_EPISODE:
                self._close_writers()
                self.episode += 1
                continue
            for view, frame in item.items():
                rgb = colorize_depth(frame) if view.endswith(".depth") else frame
                if view not in self.writers:
                    self.root.mkdir(parents=True, exist_ok=True)
                    self.writers[view] = VideoWriter(
                        self.root / view_filename(self.episode, view), self.fps, rgb.shape[1], rgb.shape[0]
                    )
                self.writers[view].write(rgb)

    def _close_writers(self) -> None:
        for writer in self.writers.values():
            writer.close()
            logger.info("[VIDEO] wrote %s (%d frames)", writer.path, writer.frames)
        self.writers = {}

