"""Numerical and presentation contracts for the interactive Action Inspector."""

from types import SimpleNamespace

import numpy as np
import torch

from lerobot.probes.action_trace_probe import (
    _analyse,
    _figure,
    _fit_metrics,
    _rotation_distance_deg,
    _write_dashboard_html,
)
from lerobot.probes.utils import (
    action_inspector_sample_seed,
    sample_action_inspector_frames,
    trajectory_error_components,
)
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
    record["norm"] = _norm_chunks(torch.zeros(4, 7), torch.zeros(4, 7))
    record["metrics"]["mse_norm"] = 0.0
    return record


def _norm_chunks(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """Normalized-space chunks as ``run()`` attaches them; hold-still is the zero chunk."""
    return {"pred": pred, "gt": gt, "hold": torch.zeros_like(gt)}


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


PROBE_CFG = SimpleNamespace(trace_clearance_warn_m=0.01, trace_table_z=0.0)


def test_figure_and_dashboard_make_target_gap_conspicuous(tmp_path):
    record = make_record()
    figure = _figure([record], PROBE_CFG, fps=30.0)
    names = [trace.name for trace in figure.data]
    assert any(name.startswith("GT initial target gap") for name in names)
    assert "GT wrist roll" in names
    assert "GT initial wrist-roll gap" in names
    assert "GT gripper" in names
    assert "GT initial gripper gap" in names
    assert len(figure.frames) == 1
    assert len(figure.frames[0].data) == len(figure.data) - 1  # only the table plane is static

    output = tmp_path / "action_trace.html"
    _write_dashboard_html(figure, [record], str(output))
    document = output.read_text()
    assert "Action Inspector" in document
    assert "POSSIBLE FOLLOWER LAG" in document
    assert "not an interpolated timestep" in document
    assert "close the gripper" in document
    assert "Trajectory fit · sample 0" in document
    assert "Final direction loss" in document

def test_renderer_accepts_objective_exemplar_labels(tmp_path):
    record = make_record()
    record.update(
        trace_label="p95 worst fit · flow loss 0.1234 · ep 2 frame 90",
        sample_names=["generated action"],
    )
    figure = _figure([record], PROBE_CFG, fps=30.0)
    assert figure.layout.title.text == "<b>p95 worst fit · flow loss 0.1234 · ep 2 frame 90</b>"
    assert any((trace.name or "").startswith("generated action ·") for trace in figure.data)

    output = tmp_path / "action_exemplars.html"
    _write_dashboard_html(
        figure,
        [record],
        str(output),
        page_title="Flow-loss action exemplars",
        subtitle="Generated action versus the demonstrated command.",
        legend_note="Black = demonstrated action · blue = generated action.",
    )
    document = output.read_text()
    assert "Flow-loss action exemplars" in document
    assert "Generated action versus the demonstrated command." in document
    assert "Black = demonstrated action · blue = generated action." in document
    assert record["trace_label"] in document


def test_arm_pose_moves_out_of_the_trace_scene_onto_fixed_cube_axes():
    # The link chain reaches back to the base: left in the trace scene its extent buries
    # the centimetre-scale motion the scene exists to show.
    figure = _figure([make_record()], PROBE_CFG, fps=30.0)
    scenes = {trace.name: trace.scene for trace in figure.data if getattr(trace, "scene", None)}
    assert scenes["ground truth targets"] == "scene"
    assert scenes["arm now — measured follower"] == "scene2"

    # The table plane goes with it: a plane an arm-length below a two-centimetre chunk
    # sets the extent under aspectmode="data", and the motion becomes a dot.
    assert [trace.scene for trace in figure.data if trace.type == "mesh3d"] == ["scene2"]

    pose = figure.layout.scene2
    assert pose.aspectmode == "cube"
    spans = [axis.range[1] - axis.range[0] for axis in (pose.xaxis, pose.yaxis, pose.zaxis)]
    assert np.allclose(spans, spans[0])  # a cube aspect on a non-cube range distorts shape
    assert pose.zaxis.range[0] <= 0.0 <= pose.zaxis.range[1]  # the table plane must not clip


def test_frame_traces_stay_bound_to_their_own_subplot():
    # Frame traces bypass add_trace, so an unbound one silently falls back onto the first
    # scene and the first pair of axes — the gripper timeline landing on the wrist axes.
    figure = _figure([make_record(), make_record()], PROBE_CFG, fps=30.0)
    frame = figure.frames[0]
    assert {trace.scene for trace in frame.data if getattr(trace, "scene", None)} == {"scene", "scene2"}
    assert {trace.xaxis for trace in frame.data if getattr(trace, "xaxis", None)} == {"x", "x2"}


def make_fit_records(pred_a: torch.Tensor, pred_b: torch.Tensor) -> list[dict]:
    """Two anchors with opposite demonstrated chunks.

    Both constant predictors are then wrong by exactly 1.0 per element — holding still
    is the zero chunk, and the dataset mean of $+1$ and $-1$ is also zero — so every
    ``skill_vs_*`` is read directly against a baseline MSE of 1.
    """
    gt_a = torch.ones(4, 7)
    return [
        {"norm": _norm_chunks(pred_a, gt_a)},
        {"norm": _norm_chunks(pred_b, -gt_a)},
    ]


def test_fit_metrics_score_sample_zero_against_both_constant_predictors():
    perfect = _fit_metrics(make_fit_records(torch.ones(4, 7), -torch.ones(4, 7)))
    assert np.isclose(perfect["mse_norm"], 0.0)
    assert np.isclose(perfect["path_mse"], 0.0)
    assert np.isclose(perfect["shape_mse"], 0.0)
    assert np.isclose(perfect["terminal_mse"], 0.0)
    assert np.isclose(perfect["terminal_direction_loss"], 0.0)
    assert np.isclose(perfect["trajectory"]["terminal_direction_loss"]["p75"], 0.0)
    assert np.isclose(perfect["baseline_hold"], 1.0)
    assert np.isclose(perfect["baseline_dataset_mean"], 1.0)
    assert np.isclose(perfect["skill_vs_hold"], 1.0)
    assert np.isclose(perfect["skill_vs_mean"], 1.0)

    # A policy that just repeats the measured pose removes none of the hold baseline.
    inert = _fit_metrics(make_fit_records(torch.zeros(4, 7), torch.zeros(4, 7)))
    assert np.isclose(inert["mse_norm"], 1.0)
    assert np.isclose(inert["skill_vs_hold"], 0.0)
    assert np.isclose(inert["terminal_direction_loss"], 1.0)


def test_shape_error_distinguishes_oscillation_from_constant_offset():
    target = torch.zeros(4, 1)
    hold = torch.zeros_like(target)
    oscillating = torch.tensor([[-1.0], [1.0], [-1.0], [1.0]])
    offset = torch.ones_like(target)

    oscillating_metrics = trajectory_error_components(oscillating, target, hold)
    offset_metrics = trajectory_error_components(offset, target, hold)

    assert np.isclose(oscillating_metrics["path_mse"], 1.0)
    assert np.isclose(offset_metrics["path_mse"], 1.0)
    assert np.isclose(oscillating_metrics["terminal_mse"], 1.0)
    assert np.isclose(offset_metrics["terminal_mse"], 1.0)
    assert np.isclose(oscillating_metrics["shape_mse"], 4.0)
    assert np.isclose(offset_metrics["shape_mse"], 0.0)


RELATIVE_KEYS = ("path_relative", "shape_relative", "terminal_relative")
# Tests state their own floors so they never move when the corpus constants are
# regenerated; unfloored isolates the ratio itself.
UNFLOORED = {"path": 1e-12, "shape": 1e-12, "terminal": 1e-12}


def test_relative_errors_are_invariant_to_the_scale_of_the_motion():
    # The same trajectory, the same relative error, at two very different amplitudes.
    # The raw MSEs move by the square of the amplitude ratio; the relative ones do not.
    hold = torch.zeros(4, 2)
    small_target = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [0.03, 0.0]])
    large_target = small_target * 100.0

    small = trajectory_error_components(small_target * 1.5, small_target, hold, scale_floors=UNFLOORED)
    large = trajectory_error_components(large_target * 1.5, large_target, hold, scale_floors=UNFLOORED)

    assert large["path_mse"] > 100.0 * small["path_mse"]
    for key in RELATIVE_KEYS:
        assert np.isclose(small[key], large[key], rtol=1e-6)

    # A half-amplitude prediction commits a quarter of the hold predictor's error.
    half = trajectory_error_components(large_target * 0.5, large_target, hold, scale_floors=UNFLOORED)
    assert np.isclose(half["path_relative"], 0.25, rtol=1e-6)
    assert np.isclose(half["terminal_relative"], 0.25, rtol=1e-6)


def test_relative_errors_read_against_the_hold_baseline():
    hold = torch.zeros(4, 2)
    target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    exact = trajectory_error_components(target, target, hold)
    inert = trajectory_error_components(hold, target, hold)
    for key in RELATIVE_KEYS:
        assert np.isclose(exact[key], 0.0)
        assert np.isclose(inert[key], 1.0, rtol=1e-6)

    # Moving when the demonstration does not stays finite, and reads worse than freezing.
    stationary_target = hold.clone()
    moving = trajectory_error_components(target, stationary_target, hold)
    for key in RELATIVE_KEYS:
        assert torch.isfinite(moving[key])
        assert moving[key] > 1.0


def test_scale_floor_caps_the_weight_of_a_barely_moving_chunk():
    # A floor, unlike an added epsilon, is exactly neutral above itself: the chunk whose
    # denominator clears the floor reports the same ratio floored or not.
    hold = torch.zeros(4, 2)
    floors = {"path": 1e-2, "shape": 1e-2, "terminal": 1e-2}

    above = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    floored = trajectory_error_components(above * 1.5, above, hold, scale_floors=floors)
    unfloored = trajectory_error_components(above * 1.5, above, hold, scale_floors=UNFLOORED)
    for key in RELATIVE_KEYS:
        assert np.isclose(floored[key], unfloored[key], rtol=1e-6)

    # Below the floor the ratio is capped, so a near-stationary chunk cannot dominate a
    # batch: its relative error can no longer exceed numerator / floor.
    below = above * 1e-3
    capped = trajectory_error_components(below * 1.5, below, hold, scale_floors=floors)
    uncapped = trajectory_error_components(below * 1.5, below, hold, scale_floors=UNFLOORED)
    for key, raw_key in zip(RELATIVE_KEYS, ("path_mse", "shape_mse", "terminal_mse"), strict=True):
        assert capped[key] < uncapped[key]
        assert np.isclose(capped[key], capped[raw_key] / 1e-2, rtol=1e-6)


def test_terminal_direction_loss_ignores_length_and_normalizer_offset():
    # The hold vector removes affine normalizer offsets before taking the cosine.
    hold = torch.full((3, 2), 5.0)
    target = hold.clone()
    target[-1] = torch.tensor([6.0, 5.0])

    aligned = hold.clone()
    aligned[-1] = torch.tensor([15.0, 5.0])
    opposite = hold.clone()
    opposite[-1] = torch.tensor([3.0, 5.0])
    orthogonal = hold.clone()
    orthogonal[-1] = torch.tensor([5.0, 9.0])
    stationary = hold.clone()

    assert np.isclose(
        trajectory_error_components(aligned, target, hold)["terminal_direction_loss"], 0.0
    )
    assert np.isclose(
        trajectory_error_components(opposite, target, hold)["terminal_direction_loss"], 2.0
    )
    assert np.isclose(
        trajectory_error_components(orthogonal, target, hold)["terminal_direction_loss"], 1.0
    )
    assert np.isclose(
        trajectory_error_components(stationary, target, hold)["terminal_direction_loss"], 1.0
    )

    stationary_target = hold.clone()
    assert torch.isnan(
        trajectory_error_components(aligned, stationary_target, hold)["terminal_direction_loss"]
    )


def test_fit_metrics_name_the_worst_joint():
    pred = torch.ones(4, 7)
    pred[:, 6] = -1.0  # gripper misses by 2.0, every other joint is exact
    metrics = _fit_metrics(make_fit_records(pred, -torch.ones(4, 7)))
    assert metrics["worst_joint"] == "gripper"
    assert np.isclose(metrics["worst_joint_mse_norm"], 2.0)  # (2^2 on one of two anchors)
    assert np.isclose(metrics["mse_norm_by_joint"]["shoulder_pan"], 0.0)


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


def test_fast_decode_shares_the_flow_scene(tmp_path):
    # Both decoders in one set of axes: the difference between them only reads as a
    # distance when it is drawn as one, and amber is reserved so it cannot be read as a
    # flow draw.
    figure = _figure([make_comparison_record(fast=True)], PROBE_CFG, fps=30.0)
    fast = next(trace for trace in figure.data if (trace.name or "").startswith("FAST · greedy"))
    assert fast.scene == "scene"
    assert fast.line.color == "#D97706"
    assert "#D97706" not in [trace.line.color for trace in figure.data if "sample" in (trace.name or "")]

    output = tmp_path / "action_trace.html"
    _write_dashboard_html(figure, [make_comparison_record(fast=True)], str(output))
    assert "FAST · greedy" in output.read_text()


def test_a_failed_fast_decode_leaves_the_animation_aligned(tmp_path):
    # Frames address traces by index, so an anchor that dropped its unavailable FAST path
    # would shift every trace after it onto the wrong data.
    records = [make_comparison_record(fast=True), make_comparison_record(fast=False)]
    figure = _figure(records, PROBE_CFG, fps=30.0)

    assert len(figure.frames) == len(records)
    for frame in figure.frames:
        assert len(frame.data) == len(figure.data) - 1  # only the table plane is static
        assert list(frame.traces) == list(range(1, len(figure.data)))
    names = [trace.name or "" for trace in figure.frames[1].data]
    assert any(name.startswith("FAST unavailable") for name in names)  # the reason reaches the legend

    output = tmp_path / "action_trace.html"
    _write_dashboard_html(figure, records, str(output))
    assert "FAST unavailable" in output.read_text()


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
