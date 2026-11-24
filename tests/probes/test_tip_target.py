"""Tip target probe (probes/tip_target.py): split summary, Aim scalars, timeline figure, and the
label the ceiling is read through."""

import json
import os

import pytest
import torch

pytest.importorskip("matplotlib")

from lerobot.policies.molmoact2.tip_target import GRID, tip_label, tip_vector  # noqa: E402
from lerobot.probes import tip_target  # noqa: E402


def _row(seconds, in_view, error=None):
    row = dict.fromkeys(
        ("peak_on_patch", "peak_within_one", "error_cm", "floor_cm", "ceiling_cm", "distance_cm", "u", "v")
    )
    row.update(seconds=seconds, in_view=in_view, loss=1.0, none_accuracy=1.0, p_none=0.9, subtask="s")
    if in_view:
        row.update(
            peak_on_patch=0.0, peak_within_one=1.0, error_cm=error, floor_cm=6.0, ceiling_cm=0.5,
            distance_cm=10.0, u=320.0, v=240.0, p_none=0.1,
        )
    return row


def test_summary_takes_the_in_view_numbers_over_the_in_view_frames():
    rows = [_row(0.0, True, 2.0), _row(0.1, True, 4.0), _row(0.2, True, 9.0), _row(0.3, False)]
    summary = tip_target.split_summary(rows)
    assert summary["n_frames"] == 4 and summary["n_in_view"] == 3
    assert summary["error_cm"] == pytest.approx(5.0) and summary["error_cm_median"] == pytest.approx(4.0)
    assert summary["floor_cm"] == pytest.approx(6.0) and summary["none_accuracy_majority"] == pytest.approx(0.75)
    json.dumps(summary)
    scalars = tip_target.aim_scalars({"val": summary})
    assert scalars["tip_probe_val_error_cm"] == pytest.approx(5.0)


def test_summary_of_a_split_without_in_view_frames():
    summary = tip_target.split_summary([_row(0.0, False)])
    assert summary["error_cm"] is None and summary["none_accuracy"] == 1.0
    assert tip_target.split_summary([])["n_frames"] == 0
    assert "tip_probe_val_error_cm" not in tip_target.aim_scalars({"val": summary})


def test_timeline_renders(tmp_path):
    rows = [_row(i / 10, i % 4 != 0, 1.0 + i) for i in range(20)]
    path = str(tmp_path / "timeline.png")
    tip_target.render_timeline(rows, [(0.5, "grasp"), (1.5, "release")], path, "episode")
    assert os.path.getsize(path) > 0


def test_vector_through_the_label_is_the_depth_point_at_the_target():
    points = torch.zeros(1, GRID, GRID, 4)
    points[..., 2], points[..., 3] = 0.1, 1.0
    points[0, 3, 5, :3] = torch.tensor([0.01, -0.02, 0.10])
    target = torch.tensor([[1.0, 0.01, -0.02, 0.10, 5.5 * 640 / GRID, 3.5 * 480 / GRID]])
    label = tip_label(target)
    assert label.sum().item() == pytest.approx(1.0) and label[0, -1] == 0
    vector = tip_vector(label.clamp_min(1e-30).log(), points)[1]
    assert (vector[0] - target[0, 1:4]).norm().item() < 0.02
    assert tip_label(torch.full((1, 6), float("nan")))[0, -1] == 1
