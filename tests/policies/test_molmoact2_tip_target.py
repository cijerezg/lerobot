import torch

from lerobot.policies.molmoact2.tip_target import GRID, tip_target_loss, tip_vector


def _points():
    points = torch.zeros(1, GRID, GRID, 4)
    points[0, 3, 5] = torch.tensor([0.01, -0.02, 0.10, 1.0])
    points[0, 3, 6] = torch.tensor([0.03, -0.02, 0.12, 1.0])
    return points


def test_tip_vector_is_the_weighted_mean_of_the_patches_with_a_reading():
    scores = torch.full((1, GRID * GRID + 1), -20.0)
    scores[0, 3 * GRID + 5] = scores[0, 3 * GRID + 6] = 5.0
    scores[0, 0] = 5.0  # a patch without a reading: left out of the mean
    none, vector = tip_vector(scores, _points())
    assert none.item() < 1e-6
    assert torch.allclose(vector, torch.tensor([[0.02, -0.02, 0.11]]), atol=1e-5)


def test_tip_vector_reports_no_target():
    scores = torch.full((1, GRID * GRID + 1), -20.0)
    scores[0, -1] = 5.0
    assert tip_vector(scores, _points())[0].item() > 0.99


def test_loss_metric_uses_the_same_vector():
    scores = torch.full((1, GRID * GRID + 1), -20.0)
    scores[0, 3 * GRID + 5] = 5.0
    target = torch.tensor([[1.0, 0.01, -0.02, 0.10, 5.5 * 640 / GRID, 3.5 * 480 / GRID]])
    _, metrics = tip_target_loss(scores, target, _points(), "mean")
    assert metrics["tip_vector_error_cm"] < 1e-3
    assert metrics["tip_peak_on_patch"] == 1.0
