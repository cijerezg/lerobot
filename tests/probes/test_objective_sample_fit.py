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


def test_hold_samples_score_the_hold_reference_and_sum_to_the_chunk_mse():
    gt, hold = _chunks()
    out = sample_fit(_samples(hold), gt, hold, VALID, T)
    assert out["offset"] == pytest.approx(out["hold_offset"])
    assert out["shape"] == pytest.approx(out["hold_shape"])
    chunk_mse = float(((hold - gt)[:, VALID] ** 2).mean())
    assert out["offset"] + out["shape"] == pytest.approx(chunk_mse)


def test_motionless_chunk_at_the_right_level_has_zero_offset_and_the_gt_motion_as_shape():
    gt, hold = _chunks()
    level = gt.mean(dim=0, keepdim=True).repeat(T, 1)  # right DC, no motion
    out = sample_fit(_samples(level), gt, hold, VALID, T)
    assert out["offset"] == pytest.approx(0.0, abs=1e-12)
    assert out["shape"] == pytest.approx(out["hold_shape"])
    assert out["shape"] == pytest.approx(float(((gt - level)[:, VALID] ** 2).mean()))


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


def test_one_step_chunk_has_zero_shape():
    gt, hold = _chunks()
    out = sample_fit(_samples(gt), gt, hold, VALID, 1)
    assert out["shape"] == pytest.approx(0.0, abs=1e-12)
    assert out["offset"] == pytest.approx(0.0, abs=1e-12)
