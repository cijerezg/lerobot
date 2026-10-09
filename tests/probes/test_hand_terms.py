"""Hand terms probe (probes/hand_terms.py): gap, percentile bands, per-robot grouping,
summary and the histogram figure, on synthetic rows."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from lerobot.probes import hand_terms  # noqa: E402


def _rows(n: int, robot: str, split: str, seed: int, pose_offset: float = 0.0) -> list[dict]:
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        joint = float(rng.uniform(0.1, 0.3))
        rows.append(
            {
                "loss_hand_joint": joint,
                "loss_hand_pose": joint + pose_offset + float(rng.normal(0, 0.05)),
                "loss_hand_fk": float(rng.uniform(0.0, 0.2)),
                "loss_flow": joint,
                "robot": robot,
                "layout_id": 6 if robot == "ReBot" else 0,
                "kind": "rebot" if robot == "ReBot" else "diverse",
                "split": split,
                "source": "val" if split == "val" else robot.lower(),
                "episode": f"{robot}/{i}",
                "episode_idx": i,
                "frame_idx": i * 3,
                "global_idx": 100 + i,
                "index": i,
                "subtask": "fold",
            }
        )
    return hand_terms.with_gap(rows)


def test_row_drops_frames_without_hand_terms_and_names_the_robot():
    assert hand_terms._row({"loss_flow": 0.2}, kind="rebot") is None
    row = hand_terms._row(
        {"loss_hand_joint": 0.2, "loss_hand_pose": 0.3, "loss_hand_fk": None, "layout_id": 6}, kind="rebot"
    )
    assert row["loss_hand_fk"] is None and row["robot"] == hand_terms.robot_name(6)
    assert hand_terms.robot_name(None) == "unknown"
    assert hand_terms.robot_name(999) == "layout 999"


def test_gap_is_hand_minus_joint_and_bands_sit_at_the_absolute_gap_percentiles():
    rows = _rows(100, "ReBot", "val", seed=0, pose_offset=0.1)
    assert all(abs(r["gap"] - (r["loss_hand_pose"] - r["loss_hand_joint"])) < 1e-12 for r in rows)
    bands = hand_terms.gap_bands(rows, n_per_band=3)
    assert [b["percentile"] for b in bands] == [5, 50, 95]
    magnitude = np.abs(np.array([r["gap"] for r in rows]))
    for band in bands:
        assert band["target"] == pytest.approx(float(np.percentile(magnitude, band["percentile"])))
        assert len(band["frames"]) == 3
        # nearest to the percentile value, listed by signed gap
        picked = [abs(r["gap"]) for r in band["frames"]]
        assert max(abs(v - band["target"]) for v in picked) <= np.sort(np.abs(magnitude - band["target"]))[2]
        assert [r["gap"] for r in band["frames"]] == sorted(r["gap"] for r in band["frames"])
    assert hand_terms.gap_bands([], 3) == []


def test_summary_groups_by_robot_and_split():
    val = _rows(40, "ReBot", "val", seed=1)
    train = _rows(40, "ReBot", "train", seed=2) + _rows(30, "Franka", "train", seed=3, pose_offset=0.2)

    class Cfg:
        class policy:  # noqa: N801
            class hand:  # noqa: N801
                joint_weight = 0.1
                fk_weight = 1.0

    summary = hand_terms.build_summary(val, train, Cfg)
    assert summary["n_val_frames"] == 40 and summary["n_train_frames"] == 70
    assert summary["n_train_diverse_frames"] == 30
    assert summary["robots"] == ["Franka", "ReBot"]
    assert summary["terms"]["loss_hand_joint"]["val"]["n"] == 40
    assert "z" in summary["terms"]["gap"]
    assert summary["by_robot"]["Franka"]["gap"]["val"] == {"n": 0}
    assert summary["by_robot"]["Franka"]["gap"]["train"]["mean"] == pytest.approx(0.2, abs=0.03)
    assert summary["joint_weight"] == 0.1
    json.dumps(summary)  # serializable
    scalars = hand_terms.aim_scalars(summary)
    assert set(scalars) == {
        "hand_terms_val_joint",
        "hand_terms_val_pose",
        "hand_terms_val_fk",
        "hand_terms_val_gap_median",
    }


def test_histograms_render_one_column_per_robot(tmp_path):
    val = _rows(20, "ReBot", "val", seed=4)
    train = _rows(20, "ReBot", "train", seed=5) + _rows(20, "Franka", "train", seed=6)
    path = hand_terms.render_histograms(val, train, str(tmp_path))
    assert path is not None and os.path.exists(path) and os.path.getsize(path) > 0
    assert hand_terms.render_histograms([], [], str(tmp_path)) is None
