"""Objective probe sample-fit reductions (probes/objective.py: ``sample_fit``) on synthetic chunks."""

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")

from lerobot.probes.objective import sample_fit  # noqa: E402

T, D, K = 30, 8, 8
VALID = np.array([True] * 7 + [False])


def _chunks(seed: int = 0):
    rng = np.random.RandomState(seed)
    state = rng.normal(size=D)
    hold = np.repeat(state[None], T, axis=0)
    # A moving chunk: a ramp plus a wiggle, offset from the state.
    ramp = np.linspace(0, 1, T)[:, None] * rng.normal(size=D)[None]
    wiggle = np.sin(np.linspace(0, 6, T))[:, None] * 0.2
    gt = hold + 0.5 + ramp + wiggle
    # The padded dim must not matter: give it garbage everywhere.
    gt[:, 7] = 100.0
    hold[:, 7] = -100.0
    return torch.tensor(gt), torch.tensor(hold)


def _samples(chunk: torch.Tensor, k: int = K) -> torch.Tensor:
    return chunk.unsqueeze(0).repeat(k, 1, 1)


def test_exact_samples_score_zero():
    gt, hold = _chunks()
    out = sample_fit(_samples(gt), gt, hold, VALID, T)
    assert out["offset"] == pytest.approx(0.0, abs=1e-12)
    assert out["shape"] == pytest.approx(0.0, abs=1e-12)
    assert out["collapse"] == pytest.approx(0.0, abs=1e-12)


def test_hold_samples_score_one_on_both_ratios():
    gt, hold = _chunks()
    out = sample_fit(_samples(hold), gt, hold, VALID, T)
    assert out["offset"] == pytest.approx(1.0)
    assert out["shape"] == pytest.approx(1.0)


def test_motionless_chunk_at_the_right_level_scores_shape_one_offset_zero():
    gt, hold = _chunks()
    level = gt.mean(dim=0, keepdim=True).repeat(T, 1)  # right DC, no motion
    out = sample_fit(_samples(level), gt, hold, VALID, T)
    assert out["offset"] == pytest.approx(0.0, abs=1e-12)
    assert out["shape"] == pytest.approx(1.0)


def test_collapse_is_the_unbiased_std_over_samples():
    gt, hold = _chunks()
    rng = np.random.RandomState(1)
    spread = torch.tensor(rng.normal(size=(K, T, D)))
    samples = gt.unsqueeze(0) + spread
    out = sample_fit(samples, gt, hold, VALID, T)
    expected = spread[:, :, VALID].std(dim=0).mean()
    assert out["collapse"] == pytest.approx(float(expected))


def test_padded_steps_and_dims_are_excluded():
    gt, hold = _chunks()
    samples = _samples(gt)
    samples[:, 20:, :] = 1e6  # padded horizon
    samples[:, :, 7] = -1e6  # padded dim
    out = sample_fit(samples, gt, hold, VALID, 20)
    assert out["offset"] == pytest.approx(0.0, abs=1e-12)
    assert out["shape"] == pytest.approx(0.0, abs=1e-12)
    assert out["collapse"] == pytest.approx(0.0, abs=1e-12)


def test_one_step_chunk_has_no_shape():
    gt, hold = _chunks()
    out = sample_fit(_samples(gt), gt, hold, VALID, 1)
    assert out["shape"] is None
    assert out["offset"] == pytest.approx(0.0, abs=1e-12)


def test_pooled_ratio_weighs_frames_by_energy_not_by_ratio():
    from lerobot.probes.objective import pooled_ratio

    # One frame with a big denominator and a modest ratio, one near-motionless frame
    # whose own ratio is huge: the pooled headline must stay near the big frame's ratio.
    val = [
        {"offset_num": 0.5, "offset_den": 1.0, "offset": 0.5},
        {"offset_num": 0.1, "offset_den": 0.001, "offset": 100.0},
    ]
    train = [{"offset_num": 0.25, "offset_den": 1.0, "offset": 0.25}]
    entry = pooled_ratio(val, train, "offset", n_boot=50)
    assert entry["val"] == pytest.approx(0.6 / 1.001)
    assert entry["train"] == pytest.approx(0.25)
    assert entry["val_n"] == 2 and entry["train_n"] == 1
    assert entry["gap"] == pytest.approx(entry["val"] - entry["train"])
    assert entry["train_sem"] == 0.0 and entry["val_sem"] > 0.0
