"""Counterfactual construction for the depth-modality probe."""

import pytest
import torch

from lerobot.probes.depth_modality_probe import (
    _drop_depth,
    _match_foreign_depth_donors,
    _paired_stats,
    _replace_depth_window,
    _stale_depth_index,
)

CURRENT = "observation.depth.wrist"
HISTORY = "history.depth.wrist.depth"
RGB = "observation.images.wrist"


def _observation() -> dict:
    return {
        CURRENT: torch.zeros(1, 1, 2, 2),
        HISTORY: torch.zeros(1, 3, 2, 2),
        RGB: torch.ones(1, 3, 2, 2),
        "observation.state": torch.arange(3, dtype=torch.float32)[None],
    }


def test_foreign_treatment_swaps_current_and_historical_depth_only():
    obs = _observation()
    donor = {
        CURRENT: torch.full_like(obs[CURRENT], 4),
        HISTORY: torch.full_like(obs[HISTORY], 7),
    }

    treated = _replace_depth_window(obs, donor, depth_obs_key=CURRENT)

    assert bool(treated[CURRENT].eq(4).all())
    assert bool(treated[HISTORY].eq(7).all())
    assert treated[RGB] is obs[RGB]
    assert treated["observation.state"] is obs["observation.state"]


def test_foreign_treatment_rejects_a_partial_depth_window():
    with pytest.raises(KeyError, match="history.depth"):
        _replace_depth_window(
            _observation(),
            {CURRENT: torch.ones(1, 1, 2, 2)},
            depth_obs_key=CURRENT,
        )


def test_null_stress_drops_current_and_historical_depth_only():
    obs = _observation()

    treated = _drop_depth(obs, depth_obs_key=CURRENT)

    assert CURRENT not in treated
    assert HISTORY not in treated
    assert torch.equal(treated[RGB], obs[RGB])
    assert "observation.state" in treated


class _Dataset:
    def __init__(self):
        self.hf_dataset = []
        specs = [
            # Anchor episode.
            (0, 0, 0, 7, 0.0),
            (0, 1, 0, 7, 0.0),
            (0, 2, 0, 7, 0.0),
            # Same task/subtask: frame 1 is closest in state and progress.
            (1, 0, 0, 7, 5.0),
            (1, 1, 0, 7, 0.2),
            (1, 2, 0, 7, 5.0),
            # Closer state but wrong task, so it must lose to the stronger tier.
            (2, 0, 1, 7, 0.0),
            (2, 1, 1, 7, 0.0),
            (2, 2, 1, 7, 0.0),
        ]
        for episode, frame, task, subtask, state in specs:
            self.hf_dataset.append(
                {
                    "episode_index": torch.tensor(episode),
                    "frame_index": torch.tensor(frame),
                    "task_index": torch.tensor(task),
                    "subtask_index": torch.tensor(subtask),
                    "observation.state": torch.tensor([state]),
                }
            )

    def __len__(self):
        return len(self.hf_dataset)


def test_foreign_matching_prefers_task_subtask_then_state_and_progress():
    matches = _match_foreign_depth_donors(_Dataset(), anchors=[1], stride=1)

    assert matches[1]["episode_idx"] == 1
    assert matches[1]["frame_idx"] == 1
    assert matches[1]["global_idx"] == 4
    assert matches[1]["tier"] == "same_task_subtask"


def test_stale_index_is_same_episode_stride_aligned():
    # Episode starts at global 100; frame 70 with a requested 30-frame lag lands at 40.
    assert _stale_depth_index(170, 70, stale_frames=30, stride=10) == (140, 30)
    assert _stale_depth_index(100, 0, stale_frames=30, stride=10) is None


def _row(global_idx: int, deployment: float, stale: float | None) -> dict:
    trajectory = {"deployment": {"path_mse": deployment}}
    if stale is not None:
        trajectory["stale_depth"] = {"path_mse": stale}
    return {"global_idx": global_idx, "trajectory": trajectory}


def test_penalty_pairs_per_frame_instead_of_differencing_two_means():
    """The 2026-09-20 artifact: an episode-start frame stale_depth cannot run on.

    Its deployment error is 9x the others, so a difference of per-condition means reads
    as a large stale "improvement" while every shared frame says stale is slightly worse.
    """
    rows = [_row(0, 0.30, None), _row(3, 0.03, 0.031), _row(6, 0.04, 0.041)]

    unpaired = sum(r["trajectory"]["stale_depth"]["path_mse"] for r in rows[1:]) / 2 - sum(
        r["trajectory"]["deployment"]["path_mse"] for r in rows
    ) / 3
    stats = _paired_stats(rows, "stale_depth", "path_mse")

    assert unpaired < -0.08  # the artifact
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.001)
    assert stats["frac_worse"] == 1.0
