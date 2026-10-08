"""Frame log written while recording, and the recovery of an episode a crash never saved."""

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.recover_episodes import recover
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
    "observation.images.top": {"dtype": "image", "shape": (24, 32, 3), "names": ["h", "w", "c"]},
}


def _frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "task": "put the clipper in the case",
            "observation.state": rng.normal(size=7).astype(np.float32),
            "action": rng.normal(size=7).astype(np.float32),
            "observation.images.top": rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8),
        }
        for _ in range(n)
    ]


def test_saved_and_discarded_episodes_leave_no_log(tmp_path):
    root = tmp_path / "ds"
    ds = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=FEATURES, root=root)
    for f in _frames(4):
        ds.add_frame(f)
    assert (root / "recovery/episode-000000.pkl").is_file()
    ds.save_episode()
    for f in _frames(3):
        ds.add_frame(f)
    ds.clear_episode_buffer()
    ds.finalize()
    assert not (root / "recovery").exists()


def test_crashed_episode_is_recovered(tmp_path):
    root = tmp_path / "crashed"
    ds = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=FEATURES, root=root)
    saved, lost = _frames(5, seed=1), _frames(6, seed=2)
    for f in saved:
        ds.add_frame(dict(f))
    ds.save_episode()
    for f in lost:
        ds.add_frame(dict(f))
    ds.finalize()  # the record script's `finally` after a crash: the buffer is never saved
    (root / "depth/wrist.depth/episode-000001").mkdir(parents=True)
    for i in range(6):
        (root / f"depth/wrist.depth/episode-000001/frame-{i:06d}.png").write_bytes(b"png")

    recover(root, tmp_path / "recovered")

    out = LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "recovered")
    assert out.num_episodes == 1 and out.num_frames == 6
    for i, f in enumerate(lost):
        item = out[i]
        np.testing.assert_allclose(item["observation.state"].numpy(), f["observation.state"])
        np.testing.assert_allclose(item["action"].numpy(), f["action"])
        img = (item["observation.images.top"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        np.testing.assert_array_equal(img, f["observation.images.top"])
        assert item["task"] == "put the clipper in the case"
    assert len(list((tmp_path / "recovered/depth/wrist.depth/episode-000000").iterdir())) == 6


def test_recovery_stops_at_first_missing_image(tmp_path):
    root = tmp_path / "crashed"
    ds = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=FEATURES, root=root)
    for f in _frames(6):
        ds.add_frame(f)
    ds.finalize()
    (root / "images/observation.images.top/episode-000000/frame-000004.png").unlink()

    recover(root, tmp_path / "recovered")

    assert LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "recovered").num_frames == 4
