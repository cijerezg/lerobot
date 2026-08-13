from __future__ import annotations

import numpy as np
import pandas as pd

from lerobot.data_processing.annotate.depth_gripper_event_annotate import (
    CLOSE_THRESHOLD,
    CUTOFF_FRAMES,
    HALF_LIFE_FRAMES,
    OPEN_THRESHOLD,
    EpisodeBounds,
    build_events,
    build_labels,
    closed_intervals,
    future_event_targets,
    validate_materialization,
)


def test_closed_intervals_use_locked_hysteresis_boundaries() -> None:
    gripper = np.array(
        [
            OPEN_THRESHOLD - 1,
            CLOSE_THRESHOLD,
            CLOSE_THRESHOLD + 1,
            CLOSE_THRESHOLD,
            OPEN_THRESHOLD,
            OPEN_THRESHOLD - 1,
        ],
        dtype=np.float32,
    )

    assert closed_intervals(gripper, min_frames=3) == [(2, 5)]


def test_short_closed_interval_contributes_neither_event() -> None:
    gripper = np.full(40, -100.0, dtype=np.float32)
    gripper[5:19] = 0.0

    events, intervals = build_events(gripper, [EpisodeBounds(0, 0, len(gripper))])

    assert intervals == {0: []}
    assert events.empty


def test_episode_boundary_rules_do_not_invent_events() -> None:
    gripper = np.full(70, -100.0, dtype=np.float32)
    gripper[:15] = 0.0  # Initial interval: open event only.
    gripper[30:50] = 0.0  # Interior interval: close and open.
    gripper[55:] = 0.0  # Terminal interval: close event only.
    bounds = [EpisodeBounds(7, 0, len(gripper))]

    events, intervals = build_events(gripper, bounds)

    assert intervals == {7: [(0, 15), (30, 50), (55, 70)]}
    assert list(events[["event_type", "frame_index"]].itertuples(index=False, name=None)) == [
        ("open", 15),
        ("close", 30),
        ("open", 50),
        ("close", 55),
    ]


def test_future_targets_use_next_event_and_inclusive_cutoff() -> None:
    event = CUTOFF_FRAMES + 2
    delta, target = future_event_targets(event + 2, [event])

    assert delta[1] == -1
    assert target[1] == 0
    assert delta[2] == CUTOFF_FRAMES
    assert target[2] == np.float32(2.0 ** (-CUTOFF_FRAMES / HALF_LIFE_FRAMES))
    assert delta[event] == 0
    assert target[event] == 1
    assert delta[event + 1] == -1
    assert target[event + 1] == 0


def test_labels_are_episode_local_and_pass_acceptance_checks() -> None:
    first = np.full(220, -100.0, dtype=np.float32)
    first[180:200] = 0.0
    second = np.full(220, -100.0, dtype=np.float32)
    second[20:40] = 0.0
    gripper = np.concatenate([first, second])
    bounds = [EpisodeBounds(0, 0, 220), EpisodeBounds(1, 220, 440)]
    source = pd.DataFrame(
        {
            "episode_index": np.repeat([0, 1], 220).astype(np.int64),
            "frame_index": np.tile(np.arange(220), 2).astype(np.int64),
            "index": np.arange(440, dtype=np.int64),
        }
    )

    events, intervals = build_events(gripper, bounds)
    labels = build_labels(source, events, bounds)
    validate_materialization(source, events, labels, bounds, intervals)

    # Episode 1's close event is globally nearby but must not supervise episode 0's tail.
    assert labels.loc[219, "depth_gripper_close_delta"] == -1
    assert labels.loc[219, "depth_gripper_close_target"] == 0
    assert labels.loc[220, "depth_gripper_close_delta"] == 20
