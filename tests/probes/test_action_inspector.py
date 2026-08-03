"""Numerical and presentation contracts for the interactive Action Inspector."""

from types import SimpleNamespace

import numpy as np
import torch

from lerobot.probes.action_decoder_comparison import (
    FLOW_COMPARE_COUNT,
    build_figure as build_comparison_figure,
    write_html as write_comparison_html,
)
from lerobot.probes.action_trace_probe import (
    _analyse,
    _figure,
    _rotation_distance_deg,
    _write_dashboard_html,
)
from lerobot.probes.utils import action_inspector_sample_seed, sample_action_inspector_frames
from lerobot.robots.rebot_b601_follower.kinematics import LINK_NAMES


class FakeKinematics:
    """Tiny deterministic FK model: position from q0..2, tool yaw from wrist roll."""

    def frames(self, q_deg):
        q = np.atleast_2d(np.asarray(q_deg, dtype=np.float64))
        frames = np.zeros((len(q), len(LINK_NAMES), 4, 4), dtype=np.float64)
        frames[..., 3, 3] = 1.0
        frames[..., :3, :3] = np.eye(3)
        for timestep, joints in enumerate(q):
            for link_idx in range(len(LINK_NAMES)):
                frames[timestep, link_idx, :3, 3] = joints[:3] / 100.0 * link_idx / (len(LINK_NAMES) - 1)
            angle = np.deg2rad(joints[5])
            frames[timestep, -1, :3, :3] = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        return frames

    def min_heights_by_link(self, frames):
        heights = np.full(frames.shape[:2], 0.20, dtype=np.float64)
        heights[:, 1] = 0.05
        heights[:, -1] = 0.15 + frames[:, -1, 2, 3]
        return heights


def make_record():
    state = np.zeros(7)
    target = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 30.0, -100.0])
    gt = np.repeat(target[None], 4, axis=0)
    samples = [gt.copy() for _ in range(3)]
    record = _analyse(
        FakeKinematics(),
        state,
        gt,
        samples,
        table_z=0.0,
        fps=30.0,
        motor_speeds_deg_s=np.array([60.0] * 6 + [180.0]),
        joint_lower=np.array([-145, -170, -200, -80, -90, -90, -270.0]),
        joint_upper=np.array([145, 1, 1, 90, 90, 90, 0.0]),
    )
    record.update(episode=2, frame=90, global_idx=190, subtask="close the gripper", cameras=[])
    return record


def test_rotation_distance_is_geodesic_degrees():
    identity = np.eye(3)
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.isclose(_rotation_distance_deg(identity, identity), 0.0)
    assert np.isclose(_rotation_distance_deg(identity, quarter_turn), 90.0)


def test_analysis_separates_initial_target_fit_and_tool_clearance():
    metrics = make_record()["metrics"]
    assert np.isclose(metrics["initial_gap_gt"], 0.10)
    assert np.isclose(metrics["initial_gap_pred_max"], 0.10)
    assert np.isclose(metrics["initial_travel_gt_s"], 100.0 / 180.0)
    assert np.isclose(metrics["initial_orientation_gap_gt_deg"], 30.0)
    assert np.isclose(metrics["initial_orientation_gap_pred_max_deg"], 30.0)
    assert np.isclose(metrics["ee_err_best"], 0.0)
    assert np.isclose(metrics["orientation_err_best_deg"], 0.0)
    assert np.isclose(metrics["gripper_err_best_deg"], 0.0)
    assert np.isclose(metrics["clearance_pred"], 0.05)
    assert np.isclose(metrics["clearance_tool_pred"], 0.15)


def test_figure_and_dashboard_make_target_gap_conspicuous(tmp_path):
    record = make_record()
    probe_cfg = SimpleNamespace(trace_clearance_warn_m=0.01, trace_table_z=0.0)
    figure = _figure([record], probe_cfg, fps=30.0)
    names = [trace.name for trace in figure.data]
    assert any(name.startswith("GT INITIAL TARGET GAP") for name in names)
    assert "GT wrist roll" in names
    assert "GT initial wrist-roll gap" in names
    assert "GT gripper" in names
    assert "GT initial gripper gap" in names
    assert len(figure.frames) == 1
    assert len(figure.frames[0].data) == len(figure.data) - 1  # table is static
    assert any(getattr(trace, "xaxis", None) == "x2" for trace in figure.frames[0].data)

    output = tmp_path / "action_trace.html"
    _write_dashboard_html(figure, [record], str(output))
    document = output.read_text()
    assert "Action Inspector" in document
    assert "INITIAL TARGET GAP" in document
    assert "POSSIBLE FOLLOWER LAG" in document
    assert "not an interpolated trajectory timestep" in document
    assert "close the gripper" in document


def make_comparison_record(*, fast: bool, table_clearance: float = 0.30):
    """An inspector record lifted clear of the table, with or without a FAST decode."""
    record = make_record()
    for key in ("gt_ee", "sample_ee", "start_ee"):
        record[key] = record[key] + np.array([0.0, 0.0, table_clearance])
    if fast:
        record.update(fast_error=None, fast_chunk=record["gt_chunk"].copy(), fast_ee=record["gt_ee"].copy())
    else:
        record.update(fast_error="RuntimeError: no decodable action tokens", fast_chunk=None, fast_ee=None)
    return record


def test_comparison_scenes_keep_the_table_visible_and_the_scale_undistorted():
    # Commanded poses sit 30 cm above the table, and explicit axis ranges clip: without
    # the plane in the extent the operator loses the one reference the clearance means
    # anything against.
    figure = build_comparison_figure([make_comparison_record(fast=True)], 0.0, fps=30.0)
    scene = figure.layout.scene
    assert scene.zaxis.range[0] <= 0.0 <= scene.zaxis.range[1]
    spans = [axis.range[1] - axis.range[0] for axis in (scene.xaxis, scene.yaxis, scene.zaxis)]
    assert scene.aspectmode == "cube"
    assert np.allclose(spans, spans[0])  # a cube aspect on a non-cube range distorts shape
    assert figure.layout.scene2.xaxis.range == scene.xaxis.range


def test_comparison_survives_a_failed_fast_decode(tmp_path):
    records = [make_comparison_record(fast=True), make_comparison_record(fast=False)]
    figure = build_comparison_figure(records, 0.0, fps=30.0)

    # Two static table planes; every other trace is swapped per anchor, so a frame that
    # dropped the unavailable FAST path would desynchronise the whole animation.
    assert len(figure.frames) == len(records)
    for frame in figure.frames:
        assert len(frame.data) == len(figure.data) - 2
        assert list(frame.traces) == list(range(2, len(figure.data)))
    assert {trace.scene for trace in figure.frames[0].data} == {"scene", "scene2"}
    assert (
        sum(name.startswith("flow · seed") for name in [trace.name or "" for trace in figure.frames[0].data])
        == FLOW_COMPARE_COUNT
    )

    output = tmp_path / "decoder_comparison.html"
    write_comparison_html(figure, str(output))
    document = output.read_text()
    assert "FAST unavailable" in document  # the reason reaches the legend
    assert "scene2.camera" in document  # both scenes orbit together


class FakeDataset:
    fps = 30

    def __init__(self):
        self.hf_dataset = []
        for episode_idx in range(2):
            for frame_idx in range(20):
                self.hf_dataset.append(
                    {
                        "episode_index": torch.tensor(episode_idx),
                        "frame_index": torch.tensor(frame_idx),
                    }
                )

    def __len__(self):
        return len(self.hf_dataset)


def test_shared_frame_sampler_and_seed_contract():
    probe_cfg = SimpleNamespace(
        trace_episodes="1", trace_anchor_stride_s=0.3, trace_max_anchors_per_episode=3
    )
    assert sample_action_inspector_frames(FakeDataset(), probe_cfg, stride=3) == [
        (1, 0, 20),
        (1, 9, 29),
        (1, 18, 38),
    ]
    assert action_inspector_sample_seed(42, 123, 0) == 165
    assert action_inspector_sample_seed(42, 123, 1) != action_inspector_sample_seed(42, 123, 0)
