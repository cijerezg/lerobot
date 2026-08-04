"""Interactive Action Inspector: where the policy would send the arm from demo states.

At anchor frames spaced through the validation episodes, draws ``n_samples`` flow samples
of the predicted action chunk and the ground-truth chunk in Cartesian space, via the reBot
URDF (``robots.rebot_b601_follower.kinematics``), and reports table clearance from the
full link geometry.

Four questions it answers, in decreasing order of what it can prove:

  1. **Clearance.** Would any link go through the table during the next chunk? Computed
     from per-link convex hulls, not link origins, so the gripper body and elbow count.
     This is a genuine pre-flight check: it runs before the arm ever moves.
  2. **Multimodality.** ``n_samples`` independent flow draws from one observation, drawn
     as a fan. A wide fan at a decision point means the policy is torn (which sock
     first); a tight bundle means it committed. No other probe exposes this.
  3. **Baseline-relative fit.** Is the checkpoint predicting at all? Normalized-space MSE
     of flow sample 0 against the demonstrated chunk, next to two constant predictors it
     has to beat — holding the measured pose, and the mean demonstrated chunk. Reported
     as ``skill_vs_hold`` / ``skill_vs_mean`` with a per-joint split in
     ``action_metrics.json``. This is the suite's headline "is it any good" number, and
     it lives here because the anchors and sample-0 seed are this probe's (it absorbed
     the former ``offline_inference`` probe, deleted 2026-08-02).
  4. **Cartesian fit.** Task-space error against GT, reported both as the mean over
     samples and the best sample, because they mean different things (see below).

COMMANDED vs MEASURED. Every action — GT or predicted — is a commanded (leader) pose,
while the skeleton and the ``measured now`` marker are the measured follower pose. The
segment between them is labelled ``initial target gap``: it is a controller target
displacement, not an interpolated trace step. Its cause is intentionally left open. It
can reflect servo tracking, timestamp alignment, action/state calibration, or a genuine
command discontinuity; the inspector reports both distance and an actuator-speed lower
bound so an operator can judge its physical plausibility.

Wrist roll and gripper commands are shown in synchronized plots beside the 3D trace,
while the 3D endpoint triads expose terminal tool orientation.

OPEN-LOOP. Every anchor restarts the policy from a demo state, so this measures *intended*
motion, never closed-loop behaviour: compounding error and recovery from drift are
invisible by construction. Divergence from GT is also not automatically error — the task
is multimodal, and taking the other sock is a valid rollout that scores as a large
Cartesian distance. Read the fan, and prefer ``ee_err_best`` over ``ee_err_mean`` when
judging fit; use real rollouts for the closed-loop question.

TWO DECODERS. On an ``action_mode=both`` checkpoint run with
``inference_action_mode=continuous``, a second panel puts the greedy FAST reconstruction
beside the flow fan in two scale-locked scenes. It stays a picture: FAST is greedy and
quantized, and depth reaches only the flow path, so the two sides are not comparable
enough to scalarise into a decoder score.

Runs inside rl_offline's validation loop when ``probe_parameters.enable_action_trace`` is
set, or standalone:

    python -m lerobot.probes.action_trace_probe --config config_rl.yaml \
        --probe_parameters.trace_anchor_stride_s 2.0 --probe_parameters.trace_n_samples 5
"""

import base64
import csv
import io
import json
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.action_decoder_comparison import (
    FLOW_COMPARE_COUNT,
    build_figure as build_decoder_comparison_figure,
    write_html as write_decoder_comparison_html,
)
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    action_inspector_sample_seed,
    joint_names_for_dim,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    sample_action_inspector_frames,
)
from lerobot.robots.rebot_b601_follower.kinematics import LINK_NAMES, RebotKinematics
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ActionTraceProbeConfig(TrainRLServerPipelineConfig):
    """Tunables live under ``cfg.probe_parameters`` (the ``trace_*`` fields)."""


# One colour per flow draw, so a sample can be followed through the fan and matched to
# its clearance in the legend. Red is deliberately absent: it stays the table-breach
# signal, and reds/greens are kept apart for colour-blind readers.
SAMPLE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#393b79",
    "#637939",
    "#8c6d31",
)


def _observation(dataset, cfg, global_idx: int):
    """One frame's obs (with depth + short-term history attached), GT chunk, and context."""
    frame = probe_frame_inputs(dataset, cfg, global_idx, int(cfg.policy.chunk_size))
    return (
        frame["obs"],
        frame["gt_actions"],
        frame["subtask"],
        frame["task"],
        frame["metadata"],
        frame["frame_idx"],
    )


def _pairwise_spread(points: np.ndarray) -> np.ndarray:
    """Mean pairwise Euclidean distance between samples at each timestep."""
    if len(points) < 2:
        return np.zeros(points.shape[1])
    deltas = points[:, None] - points[None, :]
    distances = np.linalg.norm(deltas, axis=-1)
    k = len(points)
    return distances.sum(axis=(0, 1)) / (k * (k - 1))


def _rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) distance in degrees for matching ``(..., 3, 3)`` arrays."""
    relative = np.einsum("...ji,...jk->...ik", a, b)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _pairwise_rotation_spread(rotations: np.ndarray) -> np.ndarray:
    """Mean pairwise SO(3) distance for ``(K, T, 3, 3)`` samples."""
    k, horizon = rotations.shape[:2]
    if k < 2:
        return np.zeros(horizon)
    total = np.zeros(horizon)
    for i in range(k):
        for j in range(i + 1, k):
            total += _rotation_distance_deg(rotations[i], rotations[j])
    return total * 2.0 / (k * (k - 1))


def _wrapped_abs_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest absolute angular difference, robust to the ±180° seam."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _command_slew_ratio(chunk: np.ndarray, speeds_deg_s: np.ndarray, fps: float) -> float:
    """Largest commanded per-step change divided by the configured motor speed."""
    if len(chunk) < 2:
        return 0.0
    dim = min(chunk.shape[-1], len(speeds_deg_s))
    ratios = np.abs(np.diff(chunk[:, :dim], axis=0)) * float(fps) / speeds_deg_s[:dim]
    return float(ratios.max(initial=0.0))


def _initial_travel_time(state: np.ndarray, target: np.ndarray, speeds_deg_s: np.ndarray) -> float:
    """Optimistic time for every joint to reach ``target`` at its configured speed."""
    dim = min(len(state), len(target), len(speeds_deg_s))
    return float((np.abs(target[:dim] - state[:dim]) / speeds_deg_s[:dim]).max(initial=0.0))


def _joint_limit_margin(chunk: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Minimum distance to a soft joint limit; negative means a commanded violation."""
    dim = min(chunk.shape[-1], len(lower), len(upper))
    margins = np.minimum(chunk[:, :dim] - lower[:dim], upper[:dim] - chunk[:, :dim])
    finite = margins[np.isfinite(margins)]
    return float(finite.min()) if finite.size else float("nan")


def _trajectory_geometry(kin: RebotKinematics, chunk: np.ndarray) -> dict:
    """One FK pass per chunk, reused by every task-space and clearance metric."""
    frames = kin.frames(chunk)
    end_idx = LINK_NAMES.index("end_link")
    heights = kin.min_heights_by_link(frames)
    moving_heights = heights[:, 1:]
    owner = moving_heights.argmin(axis=1) + 1
    return {
        "ee": frames[:, end_idx, :3, 3],
        "rotation": frames[:, end_idx, :3, :3],
        "whole_z": moving_heights.min(axis=1),
        "owner": owner,
        "tool_z": heights[:, end_idx],
        "link_origins": frames[:, :, :3, 3],
    }


def _normalized_chunks(adapter, pred_norm, gt_actions, state):
    """Sample 0, the demonstrated chunk, and a hold-still chunk, in normalized model space.

    The hold chunk repeats the measured state for the whole horizon. Under anchor/delta
    encoding that is the zero vector before normalization, and it is the floor any policy
    that predicts anything at all has to beat.
    """
    hold = state[: gt_actions.shape[-1]].unsqueeze(0).repeat(gt_actions.shape[0], 1)
    return {
        "pred": pred_norm.float(),
        "gt": adapter.normalize_gt_actions(gt_actions, state).float(),
        "hold": adapter.normalize_gt_actions(hold, state).float(),
    }


def _fit_metrics(records: list[dict]) -> dict:
    r"""Normalized-space action error against the two constants worth beating.

    Everything is measured in the normalized model space the policy trains in, not in
    degrees: on the 7-DOF rebot the gripper's spread ($\sigma\approx76^\circ$) is an order
    of magnitude above wrist_yaw's ($\approx11^\circ$), so a raw $\mathrm{deg}^2$ mean is
    mostly gripper, in units that move whenever the normalization stats do. Each figure
    sits next to two constant predictors:

      hold          — repeat the measured state for the whole chunk (the arm holds still)
      dataset mean  — the mean demonstrated chunk over the sampled anchors

    ``skill_vs_hold`` and ``skill_vs_mean`` are the fraction of each baseline's error the
    policy removes; a policy that does not beat both is not predicting.

    Only flow sample 0 enters this, never the best of the fan: best-of-$K$ against a
    multimodal target flatters the model, and the fan already has its own Cartesian
    read-out in ``ee_err_best``.
    """
    pred = torch.stack([record["norm"]["pred"] for record in records])
    gt = torch.stack([record["norm"]["gt"] for record in records])
    hold = torch.stack([record["norm"]["hold"] for record in records])
    dataset_mean = gt.mean(dim=0, keepdim=True).expand_as(gt)

    def mse(a, b):
        return float((a - b).pow(2).mean())

    by_joint = dict(
        zip(joint_names_for_dim(gt.shape[-1]), (pred - gt).pow(2).mean(dim=(0, 1)).tolist())
    )
    worst_joint, worst_joint_mse = max(by_joint.items(), key=lambda item: item[1])
    metrics = {
        "n_frames": len(records),
        "space": "normalized",
        "mse_norm": mse(pred, gt),
        "mse_norm_by_joint": by_joint,
        "worst_joint": worst_joint,
        "worst_joint_mse_norm": worst_joint_mse,
        "baseline_hold": mse(hold, gt),
        "baseline_dataset_mean": mse(dataset_mean, gt),
    }
    metrics["skill_vs_hold"] = 1.0 - metrics["mse_norm"] / max(metrics["baseline_hold"], 1e-12)
    metrics["skill_vs_mean"] = 1.0 - metrics["mse_norm"] / max(
        metrics["baseline_dataset_mean"], 1e-12
    )
    return metrics


def _analyse(
    kin: RebotKinematics,
    state: np.ndarray,
    gt_chunk: np.ndarray,
    samples: list[np.ndarray],
    table_z: float,
    *,
    fps: float = 30.0,
    motor_speeds_deg_s: np.ndarray | None = None,
    joint_lower: np.ndarray | None = None,
    joint_upper: np.ndarray | None = None,
):
    """Reduce raw command chunks into spatial, orientation, actuator, and safety readouts.

    The measured follower state and first commanded target are deliberately kept as
    distinct quantities. Their gap is not an interpolated trajectory step, but it is
    also not dismissed as harmless: it is the target displacement the controller sees
    at the start of the chunk, and may reflect servo tracking, timestamp alignment, or
    a genuinely discontinuous prediction.
    """
    if not samples:
        raise ValueError("Action Inspector requires at least one policy sample.")
    horizon = min(len(gt_chunk), min(len(sample) for sample in samples))
    if horizon < 1:
        raise ValueError("Action Inspector received an empty action chunk.")
    gt_chunk = np.asarray(gt_chunk[:horizon], dtype=np.float64)
    sample_chunks = np.stack([np.asarray(sample[:horizon], dtype=np.float64) for sample in samples])
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    action_dim = gt_chunk.shape[-1]

    if motor_speeds_deg_s is None:
        motor_speeds_deg_s = np.array([60.0] * 6 + [180.0], dtype=np.float64)
    motor_speeds_deg_s = np.asarray(motor_speeds_deg_s, dtype=np.float64)
    motor_speeds_deg_s = np.maximum(motor_speeds_deg_s, 1e-6)
    if joint_lower is None:
        joint_lower = np.full(action_dim, -np.inf)
    if joint_upper is None:
        joint_upper = np.full(action_dim, np.inf)
    joint_lower = np.asarray(joint_lower, dtype=np.float64)
    joint_upper = np.asarray(joint_upper, dtype=np.float64)

    gt_geo = _trajectory_geometry(kin, gt_chunk)
    sample_geo = [_trajectory_geometry(kin, chunk) for chunk in sample_chunks]
    state_geo = _trajectory_geometry(kin, state[None])

    gt_ee = gt_geo["ee"]
    sample_ee = np.stack([geo["ee"] for geo in sample_geo])
    gt_rot = gt_geo["rotation"]
    sample_rot = np.stack([geo["rotation"] for geo in sample_geo])
    sample_z = np.stack([geo["whole_z"] for geo in sample_geo])
    sample_tool_z = np.stack([geo["tool_z"] for geo in sample_geo])
    sample_owner = [geo["owner"] for geo in sample_geo]

    position_error_t = np.linalg.norm(sample_ee - gt_ee[None], axis=-1)
    position_errors = position_error_t.mean(axis=1)
    orientation_error_t = np.stack([_rotation_distance_deg(rotation, gt_rot) for rotation in sample_rot])
    orientation_errors = orientation_error_t.mean(axis=1)
    position_spread = _pairwise_spread(sample_ee)
    orientation_spread = _pairwise_rotation_spread(sample_rot)

    start_ee = state_geo["ee"][0]
    initial_gap_gt = float(np.linalg.norm(gt_ee[0] - start_ee))
    initial_gap_pred = np.linalg.norm(sample_ee[:, 0] - start_ee[None], axis=-1)
    start_rotation = state_geo["rotation"][0]
    initial_orientation_gap_gt = float(_rotation_distance_deg(gt_rot[0], start_rotation))
    initial_orientation_gap_pred = _rotation_distance_deg(sample_rot[:, 0], start_rotation)
    initial_joint_gap_pred = np.abs(sample_chunks[:, 0, :action_dim] - state[None, :action_dim])
    initial_travel_pred = np.array(
        [_initial_travel_time(state, chunk[0], motor_speeds_deg_s) for chunk in sample_chunks]
    )
    slew_pred = np.array([_command_slew_ratio(chunk, motor_speeds_deg_s, fps) for chunk in sample_chunks])
    limit_margin_pred = np.array(
        [_joint_limit_margin(chunk, joint_lower, joint_upper) for chunk in sample_chunks]
    )

    worst_sample = int(sample_z.min(axis=1).argmin())
    worst_step = int(sample_z[worst_sample].argmin())
    worst_owner = int(sample_owner[worst_sample][worst_step])

    metrics = {
        "ee_err_mean": float(position_errors.mean()),
        "ee_err_best": float(position_errors.min()),
        "ee_err_terminal_best": float(position_error_t[:, -1].min()),
        "orientation_err_mean_deg": float(orientation_errors.mean()),
        "orientation_err_best_deg": float(orientation_errors.min()),
        "orientation_err_terminal_best_deg": float(orientation_error_t[:, -1].min()),
        "spread_mean": float(position_spread.mean()),
        "spread_terminal": float(position_spread[-1]),
        "orientation_spread_terminal_deg": float(orientation_spread[-1]),
        "clearance_gt": float(gt_geo["whole_z"].min() - table_z),
        "clearance_pred": float(sample_z.min() - table_z),
        "clearance_tool_gt": float(gt_geo["tool_z"].min() - table_z),
        "clearance_tool_pred": float(sample_tool_z.min() - table_z),
        "clearance_link": LINK_NAMES[worst_owner],
        "initial_gap_gt": initial_gap_gt,
        "initial_gap_pred_mean": float(initial_gap_pred.mean()),
        "initial_gap_pred_max": float(initial_gap_pred.max()),
        "initial_orientation_gap_gt_deg": initial_orientation_gap_gt,
        "initial_orientation_gap_pred_mean_deg": float(initial_orientation_gap_pred.mean()),
        "initial_orientation_gap_pred_max_deg": float(initial_orientation_gap_pred.max()),
        "initial_joint_gap_gt_max_deg": float(np.abs(gt_chunk[0, :action_dim] - state[:action_dim]).max()),
        "initial_joint_gap_pred_max_deg": float(initial_joint_gap_pred.max()),
        "initial_travel_gt_s": _initial_travel_time(state, gt_chunk[0], motor_speeds_deg_s),
        "initial_travel_pred_mean_s": float(initial_travel_pred.mean()),
        "initial_travel_pred_max_s": float(initial_travel_pred.max()),
        "command_slew_gt_ratio": _command_slew_ratio(gt_chunk, motor_speeds_deg_s, fps),
        "command_slew_pred_max_ratio": float(slew_pred.max()),
        "joint_limit_margin_gt_deg": _joint_limit_margin(gt_chunk, joint_lower, joint_upper),
        "joint_limit_margin_pred_min_deg": float(limit_margin_pred.min()),
        # Backward-compatible CSV key; the UI now uses the less presumptive name.
        "follower_lag": initial_gap_gt,
    }

    wrist_roll_idx = 5
    if action_dim > wrist_roll_idx:
        roll_error = _wrapped_abs_deg(
            sample_chunks[:, :, wrist_roll_idx], gt_chunk[None, :, wrist_roll_idx]
        ).mean(axis=1)
        metrics.update(
            wrist_roll_err_mean_deg=float(roll_error.mean()),
            wrist_roll_err_best_deg=float(roll_error.min()),
            wrist_roll_spread_terminal_deg=float(
                _pairwise_spread(sample_chunks[:, :, wrist_roll_idx : wrist_roll_idx + 1])[-1]
            ),
        )

    gripper_idx = 6
    if action_dim > gripper_idx:
        gripper_error = np.abs(sample_chunks[:, :, gripper_idx] - gt_chunk[None, :, gripper_idx]).mean(axis=1)
        metrics.update(
            gripper_err_mean_deg=float(gripper_error.mean()),
            gripper_err_best_deg=float(gripper_error.min()),
            gripper_spread_terminal_deg=float(
                _pairwise_spread(sample_chunks[:, :, gripper_idx : gripper_idx + 1])[-1]
            ),
        )

    return {
        "state": state,
        "start_ee": start_ee,
        "start_rotation": start_rotation,
        "gt_chunk": gt_chunk,
        "sample_chunks": sample_chunks,
        "gt_ee": gt_ee,
        "gt_rotation": gt_rot,
        "sample_ee": sample_ee,
        "sample_rotation": sample_rot,
        "sample_clearance": sample_z.min(axis=1) - table_z,
        "sample_tool_clearance": sample_tool_z.min(axis=1) - table_z,
        "sample_worst_link": [
            LINK_NAMES[int(owner[int(z.argmin())])] for owner, z in zip(sample_owner, sample_z, strict=True)
        ],
        "sample_initial_gap": initial_gap_pred,
        "sample_initial_orientation_gap_deg": initial_orientation_gap_pred,
        "sample_initial_travel_s": initial_travel_pred,
        "anchor_skeleton": state_geo["link_origins"][0],
        "worst_pose": sample_chunks[worst_sample, worst_step],
        "metrics": metrics,
    }


def _figure(records: list[dict], p, fps: float = 30.0) -> "go.Figure":  # noqa: F821
    """Interactive 3-D action inspector plus wrist-roll and gripper timelines."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_points = np.concatenate(
        [record["gt_ee"] for record in records]
        + [record["sample_ee"].reshape(-1, 3) for record in records]
        + [record["start_ee"][None] for record in records]
    )
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    pad = 0.05
    table_x = [lo[0] - pad, hi[0] + pad]
    table_y = [lo[1] - pad, hi[1] + pad]

    def path_hover(label: str, chunk: np.ndarray) -> np.ndarray:
        step = np.arange(len(chunk))
        time_ms = step / float(fps) * 1000.0
        roll = chunk[:, 5] if chunk.shape[1] > 5 else np.full(len(chunk), np.nan)
        grip = chunk[:, 6] if chunk.shape[1] > 6 else np.full(len(chunk), np.nan)
        return np.column_stack([step, time_ms, roll, grip])

    def endpoint_trace(point, *, color, symbol, size, label, hover, showlegend=False):
        return go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers",
            marker=dict(size=size, color=color, symbol=symbol, line=dict(color="#ffffff", width=1)),
            name=label,
            showlegend=showlegend,
            hovertemplate=hover + "<extra></extra>",
        )

    def orientation_traces(point, rotation, *, prefix: str, opacity: float, dash: str):
        traces = []
        for axis, color in zip(range(3), ("#E53935", "#43A047", "#1E88E5"), strict=True):
            tip = point + 0.025 * rotation[:, axis]
            traces.append(
                go.Scatter3d(
                    x=[point[0], tip[0]],
                    y=[point[1], tip[1]],
                    z=[point[2], tip[2]],
                    mode="lines",
                    line=dict(color=color, width=5, dash=dash),
                    opacity=opacity,
                    showlegend=False,
                    name=f"{prefix} tool axis {'xyz'[axis]}",
                    hovertemplate=f"{prefix} terminal tool {'xyz'[axis]} axis<extra></extra>",
                )
            )
        return traces

    def record_traces(record):
        metrics = record["metrics"]
        traces: list[tuple[object, int, int]] = []
        gt_ee = record["gt_ee"]
        gt_chunk = record["gt_chunk"]
        gt_custom = path_hover("GT", gt_chunk)
        traces.append(
            (
                go.Scatter3d(
                    x=gt_ee[:, 0],
                    y=gt_ee[:, 1],
                    z=gt_ee[:, 2],
                    mode="lines+markers",
                    line=dict(color="#111111", width=9),
                    marker=dict(size=3, color="#111111"),
                    customdata=gt_custom,
                    name="ground truth targets",
                    hovertemplate=(
                        "<b>GT command</b><br>step %{customdata[0]:.0f} · %{customdata[1]:.0f} ms"
                        "<br>x %{x:.3f} m · y %{y:.3f} m · z %{z:.3f} m"
                        "<br>wrist roll %{customdata[2]:.1f}° · gripper %{customdata[3]:.1f}°"
                        "<extra></extra>"
                    ),
                ),
                1,
                1,
            )
        )
        traces.extend(
            [
                (
                    endpoint_trace(
                        gt_ee[0],
                        color="#111111",
                        symbol="circle",
                        size=8,
                        label="GT chunk start",
                        hover="<b>GT first target</b>",
                    ),
                    1,
                    1,
                ),
                (
                    endpoint_trace(
                        gt_ee[-1],
                        color="#111111",
                        symbol="square",
                        size=9,
                        label="GT chunk end",
                        hover="<b>GT terminal target</b>",
                        showlegend=True,
                    ),
                    1,
                    1,
                ),
            ]
        )

        skeleton = record["anchor_skeleton"]
        traces.append(
            (
                go.Scatter3d(
                    x=skeleton[:, 0],
                    y=skeleton[:, 1],
                    z=skeleton[:, 2],
                    mode="lines+markers",
                    line=dict(color="#9A9A9A", width=8),
                    marker=dict(size=4, color="#666666"),
                    name="arm now — measured follower",
                    hovertemplate="measured arm link<extra></extra>",
                ),
                1,
                1,
            )
        )
        traces.append(
            (
                endpoint_trace(
                    record["start_ee"],
                    color="#D81B60",
                    symbol="diamond",
                    size=9,
                    label="gripper now — measured",
                    hover="<b>Measured gripper pose now</b>",
                    showlegend=True,
                ),
                1,
                1,
            )
        )
        traces.append(
            (
                go.Scatter3d(
                    x=[record["start_ee"][0], gt_ee[0, 0]],
                    y=[record["start_ee"][1], gt_ee[0, 1]],
                    z=[record["start_ee"][2], gt_ee[0, 2]],
                    mode="lines+markers",
                    line=dict(color="#D81B60", width=9, dash="dash"),
                    marker=dict(size=[7, 7], color="#D81B60"),
                    name=(
                        f"GT INITIAL TARGET GAP / POSSIBLE FOLLOWER LAG — "
                        f"{metrics['initial_gap_gt'] * 1000:.0f} mm · "
                        f"{metrics['initial_orientation_gap_gt_deg']:.1f}° · "
                        f"≥{metrics['initial_travel_gt_s'] * 1000:.0f} ms"
                    ),
                    hovertemplate=(
                        "<b>Measured pose → first GT command</b>"
                        f"<br>{metrics['initial_gap_gt'] * 1000:.1f} mm translation"
                        f"<br>{metrics['initial_orientation_gap_gt_deg']:.1f}° tool orientation"
                        f"<br>{metrics['initial_joint_gap_gt_max_deg']:.1f}° largest joint gap"
                        f"<br>≥{metrics['initial_travel_gt_s'] * 1000:.0f} ms at configured motor speeds"
                        "<br><i>Target displacement, not an interpolated timestep.</i><extra></extra>"
                    ),
                ),
                1,
                1,
            )
        )

        for axis_trace in orientation_traces(
            gt_ee[-1], record["gt_rotation"][-1], prefix="GT", opacity=1.0, dash="solid"
        ):
            traces.append((axis_trace, 1, 1))

        for sample_idx, (
            ee,
            chunk,
            clearance,
            tool_clearance,
            link,
            gap,
            orientation_gap,
            travel,
        ) in enumerate(
            zip(
                record["sample_ee"],
                record["sample_chunks"],
                record["sample_clearance"],
                record["sample_tool_clearance"],
                record["sample_worst_link"],
                record["sample_initial_gap"],
                record["sample_initial_orientation_gap_deg"],
                record["sample_initial_travel_s"],
                strict=True,
            )
        ):
            color = SAMPLE_COLORS[sample_idx % len(SAMPLE_COLORS)]
            unsafe = clearance < p.trace_clearance_warn_m
            custom = path_hover(f"sample {sample_idx}", chunk)
            traces.append(
                (
                    go.Scatter3d(
                        x=ee[:, 0],
                        y=ee[:, 1],
                        z=ee[:, 2],
                        mode="lines+markers",
                        line=dict(color=color, width=6 if unsafe else 3, dash="dash" if unsafe else "solid"),
                        marker=dict(size=2.5, color=color),
                        opacity=1.0 if unsafe else 0.72,
                        customdata=custom,
                        name=(
                            f"sample {sample_idx} · gap {gap * 1000:.0f} mm · "
                            f"tool {tool_clearance * 1000:+.0f} mm" + (" · ⚠ TABLE" if unsafe else "")
                        ),
                        hovertemplate=(
                            f"<b>sample {sample_idx}</b><br>step %{{customdata[0]:.0f}} · "
                            "%{customdata[1]:.0f} ms"
                            "<br>x %{x:.3f} m · y %{y:.3f} m · z %{z:.3f} m"
                            "<br>wrist roll %{customdata[2]:.1f}° · gripper %{customdata[3]:.1f}°"
                            f"<br>whole-arm clearance {clearance * 1000:+.1f} mm ({link})"
                            f"<br>tool clearance {tool_clearance * 1000:+.1f} mm<extra></extra>"
                        ),
                    ),
                    1,
                    1,
                )
            )
            traces.extend(
                [
                    (
                        endpoint_trace(
                            ee[0],
                            color=color,
                            symbol="circle",
                            size=7,
                            label=f"sample {sample_idx} start",
                            hover=f"<b>sample {sample_idx} first target</b>",
                        ),
                        1,
                        1,
                    ),
                    (
                        endpoint_trace(
                            ee[-1],
                            color=color,
                            symbol="square",
                            size=7,
                            label=f"sample {sample_idx} end",
                            hover=f"<b>sample {sample_idx} terminal target</b>",
                        ),
                        1,
                        1,
                    ),
                    (
                        go.Scatter3d(
                            x=[record["start_ee"][0], ee[0, 0]],
                            y=[record["start_ee"][1], ee[0, 1]],
                            z=[record["start_ee"][2], ee[0, 2]],
                            mode="lines",
                            line=dict(color=color, width=4, dash="dot"),
                            opacity=0.75,
                            showlegend=False,
                            name=f"sample {sample_idx} initial target gap",
                            hovertemplate=(
                                f"<b>sample {sample_idx} initial target gap</b>"
                                f"<br>{gap * 1000:.1f} mm translation · {orientation_gap:.1f}° orientation"
                                f"<br>≥{travel * 1000:.0f} ms all-actuator lower bound"
                                "<br><i>Target displacement, not an interpolated timestep.</i><extra></extra>"
                            ),
                        ),
                        1,
                        1,
                    ),
                ]
            )

        for axis_trace in orientation_traces(
            record["sample_ee"][0, -1],
            record["sample_rotation"][0, -1],
            prefix="sample 0",
            opacity=0.65,
            dash="dot",
        ):
            traces.append((axis_trace, 1, 1))

        steps = np.arange(len(gt_chunk))
        if gt_chunk.shape[1] > 5:
            traces.append(
                (
                    go.Scatter(
                        x=[-1],
                        y=[record["state"][5]],
                        mode="markers",
                        marker=dict(color="#D81B60", size=9, symbol="diamond"),
                        name="measured wrist roll now",
                        showlegend=False,
                        hovertemplate="measured wrist roll now: %{y:.1f}°<extra></extra>",
                    ),
                    1,
                    2,
                )
            )
            traces.append(
                (
                    go.Scatter(
                        x=steps,
                        y=gt_chunk[:, 5],
                        mode="lines+markers",
                        line=dict(color="#111111", width=4),
                        marker=dict(size=4),
                        name="GT wrist roll",
                        showlegend=False,
                        hovertemplate="GT wrist roll<br>step %{x}: %{y:.1f}°<extra></extra>",
                    ),
                    1,
                    2,
                )
            )
            traces.append(
                (
                    go.Scatter(
                        x=[-1, 0],
                        y=[record["state"][5], gt_chunk[0, 5]],
                        mode="lines",
                        line=dict(color="#D81B60", width=5, dash="dash"),
                        name="GT initial wrist-roll gap",
                        showlegend=False,
                        hovertemplate="GT initial wrist-roll gap: %{y:.1f}°<extra></extra>",
                    ),
                    1,
                    2,
                )
            )
            for sample_idx, chunk in enumerate(record["sample_chunks"]):
                color = SAMPLE_COLORS[sample_idx % len(SAMPLE_COLORS)]
                traces.append(
                    (
                        go.Scatter(
                            x=[-1, 0],
                            y=[record["state"][5], chunk[0, 5]],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dot"),
                            opacity=0.7,
                            name=f"sample {sample_idx} initial wrist-roll gap",
                            showlegend=False,
                            hovertemplate=f"sample {sample_idx} initial wrist-roll gap: %{{y:.1f}}°<extra></extra>",
                        ),
                        1,
                        2,
                    )
                )
                traces.append(
                    (
                        go.Scatter(
                            x=steps,
                            y=chunk[:, 5],
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=0.8,
                            name=f"sample {sample_idx} wrist roll",
                            showlegend=False,
                            hovertemplate=f"sample {sample_idx} wrist roll<br>step %{{x}}: %{{y:.1f}}°<extra></extra>",
                        ),
                        1,
                        2,
                    )
                )

        if gt_chunk.shape[1] > 6:
            traces.append(
                (
                    go.Scatter(
                        x=[-1],
                        y=[record["state"][6]],
                        mode="markers",
                        marker=dict(color="#D81B60", size=9, symbol="diamond"),
                        name="measured gripper now",
                        showlegend=False,
                        hovertemplate="measured gripper now: %{y:.1f}°<extra></extra>",
                    ),
                    2,
                    2,
                )
            )
            traces.append(
                (
                    go.Scatter(
                        x=steps,
                        y=gt_chunk[:, 6],
                        mode="lines+markers",
                        line=dict(color="#111111", width=4),
                        marker=dict(size=4),
                        name="GT gripper",
                        showlegend=False,
                        hovertemplate="GT gripper<br>step %{x}: %{y:.1f}°<extra></extra>",
                    ),
                    2,
                    2,
                )
            )
            traces.append(
                (
                    go.Scatter(
                        x=[-1, 0],
                        y=[record["state"][6], gt_chunk[0, 6]],
                        mode="lines",
                        line=dict(color="#D81B60", width=5, dash="dash"),
                        name="GT initial gripper gap",
                        showlegend=False,
                        hovertemplate="GT initial gripper gap: %{y:.1f}°<extra></extra>",
                    ),
                    2,
                    2,
                )
            )
            for sample_idx, chunk in enumerate(record["sample_chunks"]):
                color = SAMPLE_COLORS[sample_idx % len(SAMPLE_COLORS)]
                traces.append(
                    (
                        go.Scatter(
                            x=[-1, 0],
                            y=[record["state"][6], chunk[0, 6]],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dot"),
                            opacity=0.7,
                            name=f"sample {sample_idx} initial gripper gap",
                            showlegend=False,
                            hovertemplate=f"sample {sample_idx} initial gripper gap: %{{y:.1f}}°<extra></extra>",
                        ),
                        2,
                        2,
                    )
                )
                traces.append(
                    (
                        go.Scatter(
                            x=steps,
                            y=chunk[:, 6],
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=0.8,
                            name=f"sample {sample_idx} gripper",
                            showlegend=False,
                            hovertemplate=f"sample {sample_idx} gripper<br>step %{{x}}: %{{y:.1f}}°<extra></extra>",
                        ),
                        2,
                        2,
                    )
                )
        return traces

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]],
        column_widths=[0.72, 0.28],
        row_heights=[0.5, 0.5],
        horizontal_spacing=0.04,
        vertical_spacing=0.14,
        subplot_titles=("Commanded trajectory", "Wrist roll", "Gripper command"),
    )
    table = go.Mesh3d(
        x=[table_x[0], table_x[1], table_x[1], table_x[0]],
        y=[table_y[0], table_y[0], table_y[1], table_y[1]],
        z=[p.trace_table_z] * 4,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="#C8B89A",
        opacity=0.35,
        name="table plane",
        showlegend=True,
        hoverinfo="skip",
    )
    fig.add_trace(table, row=1, col=1)
    first_traces = record_traces(records[0])
    for trace, row, col in first_traces:
        fig.add_trace(trace, row=row, col=col)

    updated = list(range(1, len(fig.data)))
    frames = []
    for frame_idx, record in enumerate(records):
        dynamic = []
        for trace, row, col in record_traces(record):
            # add_trace(row=..., col=...) binds the initial traces to their
            # subplot, but frame traces bypass add_trace. Bind them explicitly
            # so gripper data cannot fall back onto the wrist axes during
            # animation in Plotly versions that replace rather than merge.
            if col == 1:
                trace.scene = "scene"
            elif row == 1:
                trace.xaxis = "x"
                trace.yaxis = "y"
            else:
                trace.xaxis = "x2"
                trace.yaxis = "y2"
            dynamic.append(trace)
        frames.append(
            go.Frame(
                data=dynamic,
                traces=updated,
                name=str(frame_idx),
                layout=dict(title=dict(text=_title(record))),
            )
        )
    fig.frames = frames

    fig.update_layout(
        title=dict(text=_title(records[0]), font=dict(size=15), x=0.01, xanchor="left"),
        scene=dict(
            xaxis=dict(title="x (m)"),
            yaxis=dict(title="y (m)"),
            zaxis=dict(title="z (m)"),
            aspectmode="data",
            bgcolor="#FBFBFC",
            uirevision="action-inspector-camera",
        ),
        xaxis=dict(
            title="measured now (−1) → chunk step",
            range=[-1, len(records[0]["gt_chunk"]) - 1],
        ),
        yaxis=dict(title="degrees", zeroline=False),
        xaxis2=dict(
            title="measured now (−1) → chunk step",
            range=[-1, len(records[0]["gt_chunk"]) - 1],
        ),
        yaxis2=dict(title="degrees", zeroline=False),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFB",
        height=860,
        margin=dict(l=10, r=10, b=35, t=115),
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=10),
            orientation="v",
            x=0.01,
            y=0.99,
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
        sliders=[
            dict(
                active=0,
                y=-0.02,
                x=0.04,
                len=0.92,
                pad=dict(t=45),
                currentvalue=dict(prefix="anchor ", font=dict(size=12)),
                steps=[
                    dict(
                        method="animate",
                        label=f"ep{record['episode']}:{record['frame']}",
                        value=str(index),
                        args=[
                            [str(index)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for index, record in enumerate(records)
                ],
            )
        ],
    )
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#333333")
    return fig


def _title(record: dict) -> str:
    m = record["metrics"]
    return (
        f"<b>episode {record['episode']} · frame {record['frame']}</b>  |  "
        f"<b>INITIAL TARGET GAP / POSSIBLE FOLLOWER LAG</b> — GT "
        f"{m['initial_gap_gt'] * 1000:.0f} mm / {m['initial_orientation_gap_gt_deg']:.1f}° / "
        f"≥{m['initial_travel_gt_s'] * 1000:.0f} ms, pred "
        f"{m['initial_gap_pred_mean'] * 1000:.0f} mean / {m['initial_gap_pred_max'] * 1000:.0f} max mm  |  "
        f"tool clearance {m['clearance_tool_pred'] * 1000:+.0f} mm  |  "
        f"best fit {m['ee_err_best'] * 1000:.0f} mm / {m['orientation_err_best_deg']:.1f}°"
    )


def _resolve_actuator_config(cfg, action_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Configured motor speeds and soft limits in dataset action order."""
    names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    )
    default_speeds = np.array([60.0] * 6 + [180.0], dtype=np.float64)
    default_limits = {
        "shoulder_pan": (-145.0, 145.0),
        "shoulder_lift": (-170.0, 1.0),
        "elbow_flex": (-200.0, 1.0),
        "wrist_flex": (-80.0, 90.0),
        "wrist_yaw": (-90.0, 90.0),
        "wrist_roll": (-90.0, 90.0),
        "gripper": (-270.0, 0.0),
    }
    env_cfg = getattr(cfg, "env", None)
    robot_cfg = getattr(env_cfg, "robot", None)
    configured_speeds = getattr(robot_cfg, "pos_vel_velocity", default_speeds.tolist())
    if np.isscalar(configured_speeds):
        speeds = np.repeat(float(configured_speeds), action_dim)
    else:
        speeds = np.asarray(configured_speeds, dtype=np.float64).reshape(-1)
        if len(speeds) < action_dim:
            speeds = np.pad(speeds, (0, action_dim - len(speeds)), constant_values=60.0)
    limits = dict(default_limits)
    limits.update(getattr(robot_cfg, "joint_limits", {}) or {})
    lower = np.full(action_dim, -np.inf)
    upper = np.full(action_dim, np.inf)
    for idx, name in enumerate(names[:action_dim]):
        if name in limits:
            lower[idx], upper[idx] = limits[name]
    return speeds[:action_dim], lower, upper


def _image_data_uri(image_tensor, max_size: tuple[int, int] = (420, 315)) -> str:
    """Compress one CHW/BCHW camera tensor for the self-contained HTML context rail."""
    from PIL import Image

    image = image_tensor.detach().cpu().squeeze(0)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.float().numpy()
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    pil_image = Image.fromarray(array).convert("RGB")
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _camera_context(obs: dict) -> list[dict[str, str]]:
    cameras = []
    for key in sorted(k for k in obs if k.startswith("observation.images.")):
        cameras.append({"label": key.rsplit(".", 1)[-1], "src": _image_data_uri(obs[key])})
    return cameras


def _dashboard_context(record: dict) -> dict:
    m = record["metrics"]
    return {
        "episode": int(record["episode"]),
        "frame": int(record["frame"]),
        "subtask": record.get("subtask") or "(no subtask clause)",
        "cameras": record.get("cameras", []),
        "metrics": {
            "gt_gap": (
                f"{m['initial_gap_gt'] * 1000:.0f} mm · "
                f"{m['initial_orientation_gap_gt_deg']:.1f}° · "
                f"≥{m['initial_travel_gt_s'] * 1000:.0f} ms"
            ),
            "pred_gap": (
                f"{m['initial_gap_pred_mean'] * 1000:.0f}/{m['initial_gap_pred_max'] * 1000:.0f} mm mean/max · "
                f"{m['initial_orientation_gap_pred_max_deg']:.1f}° max"
            ),
            "clearance": (
                f"tool {m['clearance_tool_pred'] * 1000:+.0f} mm · "
                f"whole arm {m['clearance_pred'] * 1000:+.0f} mm"
            ),
            "fit": (
                f"{m['ee_err_best'] * 1000:.0f} mm · {m['orientation_err_best_deg']:.1f}° best sample · "
                f"MSE {m['mse_norm']:.3f} sample 0"
            ),
            "fan": (
                f"{m['spread_terminal'] * 1000:.0f} mm · {m['orientation_spread_terminal_deg']:.1f}° terminal"
            ),
            "gripper": (
                f"{m.get('gripper_err_best_deg', float('nan')):.1f}° best MAE · "
                f"{m.get('gripper_spread_terminal_deg', float('nan')):.1f}° spread"
            ),
        },
    }


def _write_dashboard_html(fig, records: list[dict], html_path: str) -> None:
    """Write a responsive action-inspector shell around the Plotly figure."""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        auto_play=False,
        div_id="action-inspector-plot",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    payload = json.dumps([_dashboard_context(record) for record in records]).replace("</", "<\\/")
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Action Inspector</title>
<style>
:root { color-scheme: light; --ink:#18181b; --muted:#6b7280; --line:#e4e4e7; --panel:#fff; --bg:#f4f4f5; --accent:#d81b60; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.shell { max-width:1900px; margin:0 auto; padding:18px; }
.topbar { display:flex; align-items:flex-start; justify-content:space-between; gap:20px; margin-bottom:14px; }
h1 { margin:0; font-size:22px; letter-spacing:-.02em; }
.subtitle { color:var(--muted); margin-top:3px; max-width:900px; }
.anchor-pill { background:#18181b; color:white; border-radius:999px; padding:7px 12px; font-weight:700; white-space:nowrap; }
.layout { display:grid; grid-template-columns:minmax(700px, 1fr) 350px; gap:14px; align-items:start; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.plot-card { overflow:hidden; min-width:0; }
.context { padding:14px; position:sticky; top:12px; }
.warning { border:2px solid rgba(216,27,96,.42); background:#fff5f8; border-radius:12px; padding:12px; margin-bottom:12px; }
.warning strong { color:var(--accent); display:block; font-size:13px; letter-spacing:.04em; }
.warning p { margin:4px 0 0; font-size:12px; color:#6f1d3d; }
.controls { display:flex; gap:8px; margin-bottom:12px; }
button { flex:1; border:1px solid var(--line); background:#fafafa; border-radius:9px; padding:8px; font-weight:700; cursor:pointer; }
button:hover { background:#f0f0f1; }
.section-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; margin:14px 0 7px; }
.subtask { border-left:3px solid #111; padding:4px 0 4px 10px; font-weight:650; }
.metrics { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.metric { background:#fafafa; border:1px solid #ececef; border-radius:10px; padding:9px; min-height:66px; }
.metric .label { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.055em; font-weight:800; }
.metric .value { margin-top:4px; font-weight:750; font-size:12px; }
.metric.gap { background:#fff5f8; border-color:#f4bfd2; }
.metric.gap .value { color:#9d174d; }
.cameras { display:grid; gap:9px; }
.camera { margin:0; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#111; }
.camera img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }
.camera figcaption { background:white; padding:6px 8px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.legend-note { color:var(--muted); font-size:11px; margin-top:10px; }
@media (max-width:1100px) { .layout { grid-template-columns:1fr; } .context { position:static; } .cameras { grid-template-columns:1fr 1fr; } }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div><h1>Action Inspector</h1><div class="subtitle">One observation, one set of deterministic policy draws, viewed in task space and joint space. Orbit the arm, hover any command, then use the same colors in wrist roll and gripper.</div></div>
    <div class="anchor-pill" id="anchor-pill"></div>
  </div>
  <div class="layout">
    <main class="card plot-card">__PLOT__</main>
    <aside class="card context">
      <div class="warning"><strong>POSSIBLE FOLLOWER LAG / INITIAL TARGET GAP</strong><p>The thick magenta segment is measured follower pose → first demonstrated target. Colored dotted spokes are the equivalent policy gaps. This may be tracking lag, timestamp alignment, calibration, or a real discontinuity; it is not an interpolated trajectory timestep.</p></div>
      <div class="controls"><button id="prev">← Previous</button><button id="next">Next →</button></div>
      <div class="section-label">Conditioning</div><div class="subtask" id="subtask"></div>
      <div class="section-label">Physical readout</div><div class="metrics" id="metrics"></div>
      <div class="section-label">Observation</div><div class="cameras" id="cameras"></div>
      <div class="legend-note">Circle = first target · square = terminal target · solid RGB axes = GT terminal tool orientation · dotted RGB axes = sample 0.</div>
    </aside>
  </div>
</div>
<script id="action-context" type="application/json">__CONTEXTS__</script>
<script>
(() => {
  const contexts = JSON.parse(document.getElementById('action-context').textContent);
  const plot = document.getElementById('action-inspector-plot');
  let active = 0;
  const labels = {gt_gap:'GT initial target', pred_gap:'Pred initial targets', clearance:'Clearance', fit:'Task-space fit', fan:'Fan spread', gripper:'Gripper'};
  function render(index) {
    active = Math.max(0, Math.min(contexts.length - 1, Number(index)));
    const context = contexts[active];
    document.getElementById('anchor-pill').textContent = `episode ${context.episode} · frame ${context.frame}`;
    document.getElementById('subtask').textContent = context.subtask;
    const metricRoot = document.getElementById('metrics'); metricRoot.replaceChildren();
    Object.entries(context.metrics).forEach(([key, value]) => {
      const node = document.createElement('div'); node.className = 'metric' + (key.includes('gap') ? ' gap' : '');
      const label = document.createElement('div'); label.className='label'; label.textContent=labels[key];
      const number = document.createElement('div'); number.className='value'; number.textContent=value;
      node.append(label, number); metricRoot.append(node);
    });
    const cameras = document.getElementById('cameras'); cameras.replaceChildren();
    context.cameras.forEach(camera => {
      const figure=document.createElement('figure'); figure.className='camera';
      const image=document.createElement('img'); image.src=camera.src; image.alt=`${camera.label} camera at selected anchor`;
      const caption=document.createElement('figcaption'); caption.textContent=camera.label;
      figure.append(image,caption); cameras.append(figure);
    });
  }
  function go(index) {
    const bounded=(index+contexts.length)%contexts.length;
    Plotly.animate(plot,[String(bounded)],{mode:'immediate',frame:{duration:0,redraw:true},transition:{duration:0}});
    Plotly.relayout(plot, {'sliders[0].active':bounded});
    render(bounded);
  }
  plot.on('plotly_sliderchange', event => render(event.step.value ?? event.step.args[0][0]));
  document.getElementById('prev').addEventListener('click', () => go(active-1));
  document.getElementById('next').addEventListener('click', () => go(active+1));
  document.addEventListener('keydown', event => { if(event.key==='ArrowLeft') go(active-1); if(event.key==='ArrowRight') go(active+1); });
  render(0);
})();
</script>
</body>
</html>"""
    document = template.replace("__PLOT__", plot_html).replace("__CONTEXTS__", payload)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(document)


def _write_manifest(
    records: list[dict], output_dir: str, fit: dict, *, has_decoder_comparison: bool = False
) -> dict:
    """Expose action-inspector headline metrics to the manifest-driven viewer."""

    def values(key: str) -> np.ndarray:
        return np.asarray([record["metrics"][key] for record in records], dtype=np.float64)

    summary = {
        "fit": fit,
        "anchors": len(records),
        "initial_gap_pred_p90_mm": float(np.percentile(values("initial_gap_pred_max"), 90) * 1000),
        "initial_travel_pred_p90_ms": float(np.percentile(values("initial_travel_pred_max_s"), 90) * 1000),
        "initial_orientation_gap_pred_p90_deg": float(
            np.percentile(values("initial_orientation_gap_pred_max_deg"), 90)
        ),
        "tool_clearance_min_mm": float(values("clearance_tool_pred").min() * 1000),
        "whole_clearance_min_mm": float(values("clearance_pred").min() * 1000),
        "position_error_best_mean_mm": float(values("ee_err_best").mean() * 1000),
        "orientation_error_best_mean_deg": float(values("orientation_err_best_deg").mean()),
        "gripper_error_best_mean_deg": float(values("gripper_err_best_deg").mean()),
        "command_slew_max_ratio": float(values("command_slew_pred_max_ratio").max()),
    }
    panels = [
        Panel(
            "action_trace.html",
            "Interactive task-space and joint-space action inspector",
            how="Magenta marks the measured-state → first demonstrated target gap (possible follower lag); colored dotted spokes are policy gaps. Step −1 is measured now, step 0 is the first target. Colors stay matched across 3-D, wrist roll, and gripper.",
            primary=True,
        )
    ]
    if has_decoder_comparison:
        panels.append(
            Panel(
                "decoder_comparison.html",
                "FAST and flow reconstructions in one scene",
                how=(
                    "One 3-D scene per anchor holding both decoders against the same ground "
                    "truth: the greedy FAST reconstruction in amber, "
                    f"{FLOW_COMPARE_COUNT} fixed-seed flow draws in blue-green, demonstrated "
                    "motion in black, and the measured arm in grey. Dotted spokes are each "
                    "prediction's measured-pose → first-target gap. The slider steps through "
                    "anchors; the camera holds across anchors, so a shape that moves between "
                    "two steps really moved.\n\n"
                    "Read it as a picture, not a score — nothing here is scalarised, and the "
                    "two decoders are not equivalent. FAST is greedy, so it has no fan and it "
                    "carries the action tokenizer's quantization error; flow does not. The "
                    "discrete path never runs the action expert, so with point-map depth "
                    "configured the depth stream reaches the flow draws and structurally "
                    "cannot reach FAST. A FAST decode that failed at an anchor says so in the "
                    "legend instead of drawing a trace."
                ),
            )
        )
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Action Inspector",
        group="Actions",
        claim="Where would the policy command the arm, wrist, and gripper from this measured state?",
        summary=summary,
        metrics=[
            Metric(
                "fit.skill_vs_hold",
                "Skill recovered vs hold-still",
                good="high",
                fmt=3,
                baseline=0.0,
                bad=0.0,
                primary=True,
                note="Fraction of the hold-still baseline's error the policy removes, from flow sample 0. At or below zero it is not predicting: repeating the measured pose scores as well.",
            ),
            Metric(
                "fit.skill_vs_mean",
                "Skill recovered vs dataset mean",
                good="high",
                fmt=3,
                baseline=0.0,
                bad=0.0,
                note="Same, against the best constant chunk over these anchors.",
            ),
            Metric(
                "fit.mse_norm",
                "Normalized action MSE",
                good="low",
                fmt=4,
                primary=True,
                note="Sample 0 against the demonstrated chunk in normalized model space. Absolute scale is meaningless on its own — read it against the two baseline errors below.",
            ),
            Metric("fit.baseline_hold", "Hold-still baseline MSE", good="none", fmt=4),
            Metric("fit.baseline_dataset_mean", "Dataset-mean baseline MSE", good="none", fmt=4),
            Metric(
                "fit.worst_joint_mse_norm",
                "Worst joint MSE",
                good="low",
                fmt=4,
                note=f"{fit['worst_joint']}. Per-joint breakdown for every joint is in ``action_metrics.json``.",
            ),
            Metric(
                "initial_gap_pred_p90_mm",
                "Initial target gap p90",
                good="low",
                fmt=0,
                primary=True,
                note="P90 across anchors of the largest policy sample gap, in mm. Read with actuator travel time; this is a controller target displacement, not one rendered timestep.",
            ),
            Metric(
                "initial_travel_pred_p90_ms",
                "Initial travel lower bound p90",
                good="low",
                fmt=0,
                note="Optimistic time in ms for all joints to reach the first target at configured motor speeds.",
            ),
            Metric(
                "initial_orientation_gap_pred_p90_deg",
                "Initial orientation gap p90",
                good="low",
                fmt=1,
                note="P90 across anchors of the largest sample's measured-to-first-target tool rotation.",
            ),
            Metric(
                "tool_clearance_min_mm",
                "Worst tool clearance",
                good="high",
                fmt=0,
                warn=10.0,
                bad=0.0,
                primary=True,
                note="Minimum end-link hull clearance, separated from the proximal link that often owns whole-arm minimum.",
            ),
            Metric(
                "whole_clearance_min_mm",
                "Worst whole-arm clearance",
                good="high",
                fmt=0,
                warn=10.0,
                bad=0.0,
                primary=True,
            ),
            Metric(
                "position_error_best_mean_mm",
                "Best-of-fan position error",
                good="low",
                fmt=0,
                note="Mean across anchors of the best sample's path-mean end-effector position error.",
            ),
            Metric("orientation_error_best_mean_deg", "Best-of-fan orientation error", good="low", fmt=1),
            Metric("gripper_error_best_mean_deg", "Best-of-fan gripper error", good="low", fmt=1),
            Metric(
                "command_slew_max_ratio",
                "Largest command slew / motor speed",
                good="low",
                fmt=2,
                baseline=1.0,
                note="Above 1 means adjacent targets change faster than the configured motor speed can physically track.",
            ),
            Metric("anchors", "Anchors inspected", good="none", fmt=0),
        ],
        panels=panels,
        see_also=["actions", "subtask_sweep"],
    )


def run(adapter, dataset, cfg, output_dir):
    """Probe entry point shared by the CLI and rl_offline's validation loop."""
    p = cfg.probe_parameters
    makedirs(output_dir)
    kin = RebotKinematics()
    fps = float(dataset.fps)

    decoder_predict = getattr(adapter, "predict_action_chunk_in_mode", None)
    # Only a "both" checkpoint holds two decoders, and the comparison is only a
    # comparison when the fan itself is the flow one: under
    # inference_action_mode=discrete every draw is the same greedy FAST decode, and the
    # right-hand scene would claim to show flow while showing FAST three times.
    has_decoder_comparison = bool(
        callable(decoder_predict)
        and getattr(cfg.policy, "action_mode", None) == "both"
        and getattr(cfg.policy, "inference_action_mode", None) == "continuous"
    )
    flow_sample_count = max(int(p.trace_n_samples), FLOW_COMPARE_COUNT if has_decoder_comparison else 1)

    anchors = sample_action_inspector_frames(dataset, p, stride=int(getattr(cfg.policy, "image_stride", 1)))
    if not anchors:
        raise ValueError(f"No anchors selected (trace_episodes={p.trace_episodes!r}).")
    logging.info(
        f"[action_trace] {len(anchors)} anchors x {flow_sample_count} deterministic flow samples"
        + (" + 1 greedy FAST reconstruction per anchor" if has_decoder_comparison else "")
    )

    # A replayed CUDA graph can capture one noise tensor and collapse the fan. Every
    # sample instead receives an independent deterministic seed: sample 0 is the draw the
    # normalized fit metrics score, while samples 1..K make the fan reproducible across
    # checkpoints and validation runs.
    adapter._set_probe_cuda_graph_enabled(False)
    records = []
    try:
        actuator_config = None
        for anchor_idx, (episode_idx, expected_frame_idx, global_idx) in enumerate(anchors):
            obs, gt_actions, subtask, task_str, metadata, frame_idx = _observation(dataset, cfg, global_idx)
            if int(frame_idx) != int(expected_frame_idx):
                raise RuntimeError(
                    f"Action Inspector sampler resolved frame {expected_frame_idx}, "
                    f"but dataset lookup returned {frame_idx}."
                )
            state_tensor = obs[OBS_STATE].reshape(-1).float().cpu()
            if actuator_config is None:
                actuator_config = _resolve_actuator_config(cfg, int(gt_actions.shape[-1]))
            motor_speeds, joint_lower, joint_upper = actuator_config

            samples = []
            pred_norm = None
            for sample_idx in range(flow_sample_count):
                generator = torch.Generator(device=adapter.device)
                generator.manual_seed(action_inspector_sample_seed(p.random_seed, global_idx, sample_idx))
                prediction, prediction_norm, _ = adapter.predict_action_chunk(
                    obs,
                    task_str,
                    state=state_tensor,
                    subtask=subtask,
                    metadata=metadata,
                    generator=generator,
                )
                samples.append(prediction.numpy())
                if sample_idx == 0:
                    pred_norm = prediction_norm

            fast_prediction = None
            fast_error = None
            if has_decoder_comparison:
                try:
                    fast_prediction = decoder_predict(
                        obs,
                        task_str,
                        inference_action_mode="discrete",
                        state=state_tensor,
                        subtask=subtask,
                        metadata=metadata,
                    )[0].numpy()
                # The comparison scene is a secondary view: a checkpoint that generates
                # no decodable action tokens, or a machine that cannot fetch the FAST
                # tokenizer, must not take the inspector down with it. The failure is
                # logged and shown in the scene's own legend.
                except Exception as exc:
                    fast_error = f"{type(exc).__name__}: {exc}"[:240]
                    logging.warning(
                        f"[action_trace] FAST decode failed at ep{episode_idx}:{frame_idx}: {fast_error}"
                    )

            record = _analyse(
                kin,
                state_tensor.numpy(),
                gt_actions.numpy(),
                samples,
                p.trace_table_z,
                fps=fps,
                motor_speeds_deg_s=motor_speeds,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
            )
            record.update(
                episode=int(episode_idx),
                frame=int(frame_idx),
                global_idx=int(global_idx),
                subtask=subtask,
                cameras=_camera_context(obs),
                norm=_normalized_chunks(adapter, pred_norm, gt_actions, state_tensor),
            )
            record["metrics"]["mse_norm"] = float(
                (record["norm"]["pred"] - record["norm"]["gt"]).pow(2).mean()
            )
            if has_decoder_comparison:
                record.update(fast_error=fast_error, fast_chunk=None, fast_ee=None)
                if fast_prediction is not None:
                    fast_chunk = np.asarray(
                        fast_prediction[: len(record["gt_chunk"])], dtype=np.float64
                    )
                    if len(fast_chunk) < 1:
                        raise RuntimeError("FAST reconstruction returned an empty action chunk.")
                    fast_geometry = _trajectory_geometry(kin, fast_chunk)
                    record.update(fast_chunk=fast_chunk, fast_ee=fast_geometry["ee"])
            records.append(record)
            logging.info(f"[{anchor_idx + 1}/{len(anchors)}] {_title(record)}")
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    fig = _figure(records, p, fps=fps)
    html_path = os.path.join(output_dir, "action_trace.html")
    _write_dashboard_html(fig, records, html_path)

    comparison_html_path = None
    if has_decoder_comparison:
        comparison_fig = build_decoder_comparison_figure(records, p.trace_table_z, fps=fps)
        comparison_html_path = os.path.join(output_dir, "decoder_comparison.html")
        write_decoder_comparison_html(comparison_fig, comparison_html_path)

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        columns = ["episode", "frame", "global_idx", *records[0]["metrics"]]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, record["metrics"].get(key)) for key in columns})

    fit = _fit_metrics(records)
    with open(os.path.join(output_dir, "action_metrics.json"), "w") as handle:
        json.dump(fit, handle, indent=2)

    _write_manifest(records, output_dir, fit, has_decoder_comparison=has_decoder_comparison)

    logging.info(
        f"[action_trace] mse_norm={fit['mse_norm']:.4f}  hold={fit['baseline_hold']:.4f}  "
        f"dataset-mean={fit['baseline_dataset_mean']:.4f}  "
        f"skill_vs_hold={fit['skill_vs_hold']:+.3f}  skill_vs_mean={fit['skill_vs_mean']:+.3f}  "
        f"worst joint: {fit['worst_joint']} {fit['worst_joint_mse_norm']:.4f}"
    )

    logging.info("── largest initial policy target gaps ──")
    for record in sorted(records, key=lambda item: -item["metrics"]["initial_gap_pred_max"])[:10]:
        metrics = record["metrics"]
        logging.info(
            f"  ep{record['episode']} frame {record['frame']:5d}  pred max "
            f"{metrics['initial_gap_pred_max'] * 1000:6.1f} mm / "
            f"≥{metrics['initial_travel_pred_max_s'] * 1000:5.0f} ms   "
            f"GT {metrics['initial_gap_gt'] * 1000:6.1f} mm"
        )
    logging.info("── worst predicted tool / whole-arm clearance ──")
    for record in sorted(records, key=lambda item: item["metrics"]["clearance_tool_pred"])[:10]:
        metrics = record["metrics"]
        logging.info(
            f"  ep{record['episode']} frame {record['frame']:5d}  tool "
            f"{metrics['clearance_tool_pred'] * 1000:+7.1f} mm   whole "
            f"{metrics['clearance_pred'] * 1000:+7.1f} mm ({metrics['clearance_link']})"
        )
    logging.info("── widest terminal fan ──")
    for record in sorted(records, key=lambda item: -item["metrics"]["spread_terminal"])[:10]:
        metrics = record["metrics"]
        logging.info(
            f"  ep{record['episode']} frame {record['frame']:5d}  "
            f"{metrics['spread_terminal'] * 1000:6.1f} mm / "
            f"{metrics['orientation_spread_terminal_deg']:5.1f} deg orientation   "
            f"best fit {metrics['ee_err_best'] * 1000:6.1f} mm"
        )

    breaches = [
        record for record in records if record["metrics"]["clearance_pred"] < p.trace_clearance_warn_m
    ]
    log = logging.warning if breaches else logging.info
    log(
        f"[action_trace] {len(breaches)}/{len(records)} anchors predict a pass within "
        f"{p.trace_clearance_warn_m * 1000:.0f} mm of the table."
    )
    logging.info(f"wrote {html_path}, {csv_path}, and {os.path.join(output_dir, 'index.json')}")


@parser.wrap()
def cli(cfg: ActionTraceProbeConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    output_dir = os.path.join(cfg.probe_parameters.output_dir, "action_trace")
    run(adapter, dataset, cfg, output_dir)
    logging.info(f"Done. Output in {output_dir}/")


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401 — registers robot configs
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401 — registers teleop configs

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    main()
