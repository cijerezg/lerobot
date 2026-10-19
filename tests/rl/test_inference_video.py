from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
from lerobot.rl.inference_video import EpisodeVideos, VideoWriter, colorize_depth, view_filename  # noqa: E402

try:
    av.codec.Codec("libx264", "w")
except Exception:  # noqa: BLE001
    pytest.skip("libx264 encoder not available in this PyAV build", allow_module_level=True)


def _probe(path: Path) -> tuple[int, str, str, int, int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return frames, stream.codec_context.name, stream.codec_context.pix_fmt, stream.width, stream.height


def test_video_writer_streams_h264_yuv420p(tmp_path: Path) -> None:
    writer = VideoWriter(tmp_path / "clip.mp4", 30, 64, 48)
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.close()

    assert _probe(tmp_path / "clip.mp4") == (5, "h264", "yuv420p", 64, 48)


def test_colorize_depth_keeps_invalid_black() -> None:
    depth = np.zeros((8, 8), np.uint16)
    depth[2:, :] = np.linspace(1000, 5000, 48).reshape(6, 8).astype(np.uint16)

    rgb = colorize_depth(depth)

    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    assert not rgb[:2].any()
    assert rgb[2:].any()


def test_episode_videos_split_per_episode_and_view(tmp_path: Path) -> None:
    videos = EpisodeVideos(tmp_path / "videos", ["top", "wrist", "wrist.depth"], 30)
    obs = {
        "pixels": {"top": np.zeros((48, 64, 3), np.uint8), "wrist": np.zeros((32, 32, 3), np.uint8)},
        "wrist.depth": np.full((32, 32), 2000, np.uint16),
    }
    for _ in range(3):
        videos.put(obs)
    videos.end_episode()
    for _ in range(2):
        videos.put(obs)
    videos.close()

    names = sorted(p.name for p in (tmp_path / "videos").iterdir())
    assert names == [view_filename(ep, view) for ep in (0, 1) for view in ("top", "wrist", "wrist.depth")]
    assert _probe(tmp_path / "videos" / "ep0_wrist_depth.mp4")[0] == 3
    assert _probe(tmp_path / "videos" / "ep1_top.mp4") == (2, "h264", "yuv420p", 64, 48)
