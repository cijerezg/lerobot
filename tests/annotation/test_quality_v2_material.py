from pathlib import Path

import numpy as np
import pytest

from lerobot.annotation.paths import WORKSPACE

# material imports the strip renderers, which read measured approach angles from the results folders at import.
pytestmark = pytest.mark.skipif(
    not (WORKSPACE / "migration/grasp_strategy_2026-09-24/grasps.jsonl").exists(),
    reason="annotation results folders not on this machine",
)


def test_diverse_episode_render_metadata_for_derived_only_molmoact(tmp_path, monkeypatch):
    monkeypatch.chdir(Path.cwd())  # material changes to the workspace at import; restore afterwards
    from lerobot.annotation.quality_v2 import material as pilot_build

    episode_dir = tmp_path / "episodes/molmoact__household__ep000002"
    episode_dir.mkdir(parents=True)
    np.save(episode_dir / "state.npy", np.array([
        [0, 0, 0, 0, 0, 0, 2.0],
        [0, 0, 0, 0, 0, 0, 3.0],
        [0, 0, 0, 0, 0, 0, 4.0],
    ], dtype=float))
    episode = {
        "episode_id": "molmoact__household__ep000002",
        "source": "molmoact",
        "native_rate_hz": 15.0,
        "cameras": ["primary", "secondary", "wrist"],
    }

    metadata = pilot_build.diverse_episode_render_metadata("corpus", episode, episode_dir)

    assert metadata["_video"]["top"] == (str(episode_dir / "videos/primary.mp4"), 0.0)
    assert metadata["_video"]["wrist"] == (str(episode_dir / "videos/wrist.mp4"), 0.0)
    assert metadata["_fps"] == 15.0
    assert metadata["_off"] == 0
    np.testing.assert_allclose(metadata["_g"], [0.0, 50.0, 100.0])
