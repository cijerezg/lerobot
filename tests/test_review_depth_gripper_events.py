from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.data_processing.annotate.review_depth_gripper_events import (
    ReviewEpisode,
    build_srt,
    format_countdown,
    pick_event_rich,
)


def _review(root: Path, episode: int, close: int, open_: int) -> ReviewEpisode:
    return ReviewEpisode(root, episode, 0, 30, 30.0, close, open_)


def test_countdown_distinguishes_cutoff_from_observed_event() -> None:
    assert format_countdown(-1, 0.0, 30.0) == ">5s / none   target 0.000"
    assert format_countdown(0, 1.0, 30.0) == " 0.0s (  0f)   target 1.000"
    assert format_countdown(30, 0.5, 30.0) == " 1.0s ( 30f)   target 0.500"


def test_event_rich_picker_is_deterministic_and_balances_event_types() -> None:
    root = Path("outputs/rebot_socks_basket-annotated-v2")
    rows = [
        _review(root, 0, 4, 6),
        _review(root, 1, 5, 5),
        _review(root, 2, 6, 5),
    ]

    selected = pick_event_rich(rows, 2)

    assert [row.episode_index for row in selected] == [2, 1]


def test_subtitles_display_materialized_values_and_event_flash() -> None:
    review = ReviewEpisode(Path("dataset"), 2, 100, 106, 30.0, 1, 1)
    labels = pd.DataFrame(
        {
            "depth_gripper_close_delta": np.array([2, 1, 0, -1, -1, -1], dtype=np.int16),
            "depth_gripper_open_delta": np.array([-1, -1, -1, 2, 1, 0], dtype=np.int16),
            "depth_gripper_close_target": np.array([0.9, 0.95, 1, 0, 0, 0], dtype=np.float32),
            "depth_gripper_open_target": np.array([0, 0, 0, 0.9, 0.95, 1], dtype=np.float32),
        }
    )
    events = pd.DataFrame(
        {
            "episode_index": [2, 2],
            "event_type": ["close", "open"],
            "frame_index": [2, 5],
        }
    )

    subtitle = build_srt(review, np.arange(6, dtype=np.float32), labels, events, step_frames=3)

    assert "global 100" in subtitle
    assert "next CLOSE" in subtitle
    assert "next OPEN" in subtitle
    assert "*** CLOSE EVENT ***" in subtitle
    assert "*** OPEN EVENT ***" in subtitle
