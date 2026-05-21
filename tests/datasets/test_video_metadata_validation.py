import re
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, DEFAULT_VIDEO_PATH, load_episodes
from lerobot.datasets.video_utils import get_video_duration_in_s
from lerobot.datasets.video_validation import validate_video_metadata
from tests.fixtures.constants import DUMMY_REPO_ID

VIDEO_KEY = "observation.images.test"
FPS = 5
IMAGE_SIZE = 16
COLORS = [
    (220, 20, 20),
    (20, 220, 20),
    (20, 20, 220),
    (220, 220, 20),
    (220, 20, 220),
]


def _features() -> dict:
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        },
        "state": {"dtype": "float32", "shape": (1,), "names": None},
    }


def _solid_frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _record_color_dataset(
    root: Path,
    batch_encoding_size: int,
    episode_lengths: list[int],
    fps: int = FPS,
) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        root=root,
        fps=fps,
        features=_features(),
        batch_encoding_size=batch_encoding_size,
        vcodec="h264",
    )
    for ep_idx, length in enumerate(episode_lengths):
        for _ in range(length):
            dataset.add_frame(
                {
                    VIDEO_KEY: _solid_frame(COLORS[ep_idx % len(COLORS)]),
                    "state": np.array([ep_idx], dtype=np.float32),
                    "task": "dummy task",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    return dataset


def _decode_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx == frame_index:
                return np.asarray(frame.to_image())
    raise AssertionError(f"Frame {frame_index} not found in {video_path}")


def _write_solid_video(video_path: Path, num_frames: int, fps: int, color=(120, 40, 200)) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    image = _solid_frame(color)
    with av.open(str(video_path), mode="w") as output:
        stream = output.add_stream("h264", fps)
        stream.width = IMAGE_SIZE
        stream.height = IMAGE_SIZE
        stream.pix_fmt = "yuv420p"
        for _ in range(num_frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _episodes_dataframe(num_episodes: int, length: int, fps: int, to_timestamp: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_index": list(range(num_episodes)),
            "length": [length] * num_episodes,
            f"videos/{VIDEO_KEY}/chunk_index": [0] * num_episodes,
            f"videos/{VIDEO_KEY}/file_index": [0] * num_episodes,
            f"videos/{VIDEO_KEY}/from_timestamp": [0.0] * num_episodes,
            f"videos/{VIDEO_KEY}/to_timestamp": [to_timestamp] * num_episodes,
        }
    )


def _corrupt_with_repeated_zero_video_metadata(dataset: LeRobotDataset) -> None:
    episodes_path = dataset.root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes = pd.read_parquet(episodes_path)
    one_episode_duration = episodes.loc[0, "length"] / dataset.fps
    episodes[f"videos/{VIDEO_KEY}/chunk_index"] = 0
    episodes[f"videos/{VIDEO_KEY}/file_index"] = 0
    episodes[f"videos/{VIDEO_KEY}/from_timestamp"] = 0.0
    episodes[f"videos/{VIDEO_KEY}/to_timestamp"] = one_episode_duration
    episodes.to_parquet(episodes_path, index=False)
    dataset.meta.episodes = load_episodes(dataset.root)


def test_batched_encoding_creates_cumulative_timestamps(tmp_path):
    lengths = [4, 4, 4]
    dataset = _record_color_dataset(tmp_path / "batched-cumulative", 3, lengths)
    dataset.finalize()

    episodes = dataset.meta.episodes.to_pandas().sort_values("episode_index")
    from_col = f"videos/{VIDEO_KEY}/from_timestamp"
    to_col = f"videos/{VIDEO_KEY}/to_timestamp"
    file_col = f"videos/{VIDEO_KEY}/file_index"

    assert episodes[file_col].nunique() == 1
    assert episodes.iloc[0][from_col] == pytest.approx(0.0, abs=0.05)
    assert episodes.iloc[1][from_col] == pytest.approx(episodes.iloc[0][to_col], abs=0.08)
    assert episodes.iloc[2][from_col] == pytest.approx(episodes.iloc[1][to_col], abs=0.08)

    video_path = dataset.root / dataset.meta.get_video_file_path(0, VIDEO_KEY)
    assert get_video_duration_in_s(video_path) == pytest.approx(sum(lengths) / FPS, abs=0.1)


def test_batched_encoding_preserves_playback_identity(tmp_path):
    lengths = [4, 4, 4]
    dataset = _record_color_dataset(tmp_path / "batched-identity", 3, lengths)
    dataset.finalize()

    episodes = dataset.meta.episodes.to_pandas().sort_values("episode_index")
    from_col = f"videos/{VIDEO_KEY}/from_timestamp"

    decoded_means = []
    for _ep_idx, row in episodes.iterrows():
        video_path = dataset.root / dataset.meta.get_video_file_path(int(row["episode_index"]), VIDEO_KEY)
        frame_index = int(round(float(row[from_col]) * FPS))
        decoded_means.append(_decode_video_frame(video_path, frame_index).mean(axis=(0, 1)))

    assert np.linalg.norm(decoded_means[0] - decoded_means[1]) > 50
    assert np.linalg.norm(decoded_means[1] - decoded_means[2]) > 50
    assert decoded_means[0].argmax() == 0
    assert decoded_means[1].argmax() == 1
    assert decoded_means[2].argmax() == 2


def test_sanity_checker_rejects_repeated_zero_timestamps(tmp_path):
    root = tmp_path / "bad-repeated-zero"
    video_path = root / DEFAULT_VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
    _write_solid_video(video_path, num_frames=4, fps=FPS)
    episodes = _episodes_dataframe(num_episodes=3, length=4, fps=FPS, to_timestamp=4 / FPS)

    with pytest.raises(ValueError, match="failed batched video encoding"):
        validate_video_metadata(root, episodes, [VIDEO_KEY], DEFAULT_VIDEO_PATH, FPS)


def test_sanity_checker_passes_valid_immediate_encoding(tmp_path):
    dataset = _record_color_dataset(tmp_path / "immediate-valid", 1, [3, 3, 3])
    dataset.finalize()

    dataset.validate_video_metadata()


def test_sanity_checker_runs_before_push(tmp_path):
    dataset = _record_color_dataset(tmp_path / "push-invalid", 1, [3, 3])
    dataset.finalize()
    _corrupt_with_repeated_zero_video_metadata(dataset)

    with (
        patch("lerobot.datasets.lerobot_dataset.HfApi") as mock_hf_api,
        pytest.raises(ValueError, match="failed batched video encoding"),
    ):
        dataset.push_to_hub()
    mock_hf_api.assert_not_called()


def test_finalize_encodes_remainder_episodes(tmp_path):
    dataset = _record_color_dataset(tmp_path / "batched-remainder", 4, [3, 3, 3, 3, 3])
    dataset.finalize()

    episodes = dataset.meta.episodes.to_pandas()
    assert len(episodes) == 5
    for ep_idx in range(5):
        video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, VIDEO_KEY)
        assert video_path.exists()
    dataset.validate_video_metadata()


def test_regression_rejects_twenty_episodes_one_short_video_all_zero(tmp_path):
    root = tmp_path / "bad-e20-pattern"
    fps = 30
    episode_length = 402
    video_path = root / DEFAULT_VIDEO_PATH.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
    _write_solid_video(video_path, num_frames=episode_length, fps=fps)
    mp4_duration = get_video_duration_in_s(video_path)
    episodes = _episodes_dataframe(
        num_episodes=20,
        length=episode_length,
        fps=fps,
        to_timestamp=episode_length / fps,
    )

    expected_group_duration = 20 * episode_length / fps
    match = re.escape(
        f"20 episodes reference the same file, but all from_timestamp values are 0 and "
        f"the MP4 duration is {mp4_duration:.1f}s while expected grouped duration is "
        f"{expected_group_duration:.0f}s"
    )
    with pytest.raises(ValueError, match=match):
        validate_video_metadata(root, episodes, [VIDEO_KEY], DEFAULT_VIDEO_PATH, fps)
