"""Interactive Action Inspector: where the policy would send the arm from demo states.

At anchor frames through the validation episodes it draws ``trace_n_samples`` flow samples
of the action chunk, puts them and the demonstrated chunk through the reBot URDF, and
answers four questions:

  1. **Fit.** ``skill_vs_hold`` / ``skill_vs_mean``: the fraction of a constant
     predictor's error the policy removes, from flow sample 0 in normalized space. Four
     separate diagnostics keep the failure mode visible: path MSE, temporal-shape MSE,
     final-position MSE, and pure final-direction alignment. The last is a cosine from the
     hold position, so it ignores how long the final displacement is. At or below zero
     skill, repeating the measured pose scores as well. These are the only numbers here
     scored on the **unfiltered** model output: they live in the space the flow loss and
     the trajectory aux loss are defined in, and a low-pass would credit the filter for
     removing exactly the high-frequency content ``shape_mse`` measures. Everything in
     task space is scored on the filtered command instead.
  2. **Clearance.** Would any link go through the table during the next chunk? From
     per-link convex hulls, not link origins, so the gripper body and elbow count. A
     genuine pre-flight check: it runs before the arm ever moves.
  3. **Multimodality.** Independent flow draws from one observation, drawn as a fan. Wide
     at a decision point means the policy is torn (which sock first); tight means it
     committed. No other probe exposes this.
  4. **Discontinuity.** ``initial target gap``: how far the first commanded target sits
     from the measured pose. This is a controller target displacement, not an interpolated
     trace step, and its cause is left open — servo tracking, timestamp alignment,
     calibration, or a genuinely discontinuous prediction all produce it.

Three things change how every number here reads. Every action, GT or predicted, is a
**commanded** leader pose, while the arm and the ``measured now`` marker are the
**measured** follower pose. Every *predicted* chunk is the **filtered** command: the flow
fan and the FAST decode pass through the same zero-phase Butterworth low-pass the deployed
runtimes put between the policy and the controller (applied in ``_analyse``, so the paths
and the numbers printed on them describe one trajectory), which makes the traces what the
arm would be told to do rather than the raw decode. GT is not filtered — it is a recorded
demonstration, not something this pipeline would command — and neither is the normalized
``pred`` behind ``fit.*``, see item 1. The safety clamp is not applied either, so
``joint_limit_margin`` still reports commanded violations. And it is **open-loop**: each
anchor restarts the policy from a demo state, so compounding error and recovery are
invisible by construction, and divergence from GT is not automatically error on a
multimodal task — read the fan, and prefer ``ee_err_best`` over ``ee_err_mean``.

Per-anchor numbers are in ``metrics.csv``; the per-joint split, baseline MSEs, and
across-anchor trajectory-metric distributions are in ``action_metrics.json``. Two
static diagnostics test whether target arm intricacy predicts error after accounting
for action energy: ``intricacy_vs_mse.png`` and
``energy_intricacy_error_heatmaps.png``.
"""

import base64
import csv
import html
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
from lerobot.probes.action_spectrum import dct_coefficients
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    TRAJECTORY_ERROR_KEYS,
    TRAJECTORY_RELATIVE_KEYS,
    action_inspector_sample_seed,
    joint_names_for_dim,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    register_config_choices,
    sample_action_inspector_frames,
    trajectory_error_components,
)
from lerobot.robots.rebot_b601_follower.kinematics import LINK_NAMES, RebotKinematics
from lerobot.utils.action_smoothing import apply_butterworth_filter
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ActionTraceProbeConfig(TrainRLServerPipelineConfig):
    """Tunables live under ``cfg.probe_parameters`` (the ``trace_*`` fields)."""


# One colour per flow draw, so a sample can be followed through the fan and matched to
# its clearance in the legend. Two colours are deliberately absent: red stays the
# table-breach signal, and amber belongs to the FAST reconstruction sharing the scene.
# Reds and greens are kept apart for colour-blind readers.
SAMPLE_COLORS = (
    "#1f77b4",
    "#9467bd",
    "#2ca02c",
    "#17becf",
    "#e377c2",
    "#8c564b",
    "#7f7f7f",
    "#bcbd22",
    "#393b79",
    "#637939",
)

# The greedy FAST reconstruction, drawn in the same axes as the flow fan.
FAST_COLOR = "#D97706"


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


def _trajectory_metrics(norm: dict[str, torch.Tensor]) -> dict[str, float | int | bool | None]:
    """JSON-safe trajectory metrics for one anchor and flow sample 0."""
    components = trajectory_error_components(norm["pred"], norm["gt"], norm["hold"])
    metrics = {
        key: float(value) if bool(torch.isfinite(value)) else None for key, value in components.items()
    }
    metrics.update(_target_spectral_metrics(norm))
    return metrics


def _target_spectral_metrics(norm: dict[str, torch.Tensor]) -> dict[str, float | int | bool | None]:
    r"""Target arm energy and intricacy in normalized model-motion space.

    The target is measured relative to the normalized hold chunk. This removes the
    normalizer's affine offset and makes DC describe commanded displacement from the
    measured pose. Intricacy is the fraction of trusted non-DC arm energy in k=4..20:

        C_arm = sum_{d in arm, k=4..20} |a_tilde[k,d]|^2
                / sum_{d in arm, k=1..20} |a_tilde[k,d]|^2.

    A numerically constant chunk has no defined intricacy; it is counted separately
    instead of receiving a denominator floor.
    """
    target = norm["gt"]
    hold = norm["hold"]
    if target.ndim != 2 or target.shape != hold.shape:
        raise ValueError(
            "Target intricacy requires matching [T,D] gt and hold chunks, got "
            f"{tuple(target.shape)} and {tuple(hold.shape)}."
        )
    horizon, action_dim = target.shape
    names = joint_names_for_dim(action_dim)
    arm_indices = [index for index, name in enumerate(names) if "gripper" not in name.lower()]
    if not arm_indices:
        arm_indices = list(range(action_dim))

    target_cpu = target.detach().to(device="cpu", dtype=torch.float64)
    hold_cpu = hold.detach().to(device="cpu", dtype=torch.float64)
    motion = target_cpu - hold_cpu
    coefficients = dct_coefficients(motion[None, :, arm_indices])[0]
    power = coefficients.square()
    trusted_stop = min(horizon, 21)
    signal = power[:trusted_stop]
    non_dc = power[1:trusted_stop]
    detail = power[4:trusted_stop]

    signal_sum = float(signal.sum())
    non_dc_sum = float(non_dc.sum())
    detail_sum = float(detail.sum())
    numerical_tolerance = torch.finfo(power.dtype).eps * max(signal_sum, 1.0) * max(int(power.numel()), 1)
    constant = non_dc_sum <= numerical_tolerance
    metrics = {
        "target_signal_energy_arm": float(signal.mean()) if signal.numel() else 0.0,
        "target_non_dc_energy_arm": float(non_dc.mean()) if non_dc.numel() else 0.0,
        "target_detail_energy_arm": float(detail.mean()) if detail.numel() else 0.0,
        "target_intricacy_arm": None if constant else detail_sum / non_dc_sum,
        "target_arm_is_constant": constant,
        "target_arm_dimensions": len(arm_indices),
    }
    if "pred" in norm:
        prediction = norm["pred"].detach().to(device="cpu", dtype=torch.float64)
        metrics["arm_path_mse"] = float(
            (prediction[:, arm_indices] - target_cpu[:, arm_indices]).square().mean()
        )
    return metrics


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
    trajectory_components = trajectory_error_components(pred, gt, hold)

    def mse(a, b):
        return float((a - b).pow(2).mean())

    trajectory = {}
    trajectory_means = {}
    for key, values in trajectory_components.items():
        finite = values[torch.isfinite(values)]
        trajectory_means[key] = float(finite.mean()) if finite.numel() else None
        trajectory[key] = {
            "mean": trajectory_means[key],
            "median": float(finite.median()) if finite.numel() else None,
            "p10": float(torch.quantile(finite, 0.10)) if finite.numel() else None,
            "p75": float(torch.quantile(finite, 0.75)) if finite.numel() else None,
            "p90": float(torch.quantile(finite, 0.90)) if finite.numel() else None,
            "valid": int(finite.numel()),
            "values": [float(value) if bool(torch.isfinite(value)) else None for value in values],
        }

    by_joint = dict(
        zip(joint_names_for_dim(gt.shape[-1]), (pred - gt).pow(2).mean(dim=(0, 1)).tolist())
    )
    worst_joint, worst_joint_mse = max(by_joint.items(), key=lambda item: item[1])
    metrics = {
        "n_frames": len(records),
        "space": "normalized",
        "mse_norm": trajectory_means["path_mse"],
        **trajectory_means,
        "trajectory": trajectory,
        "terminal_direction_valid_fraction": (
            trajectory["terminal_direction_loss"]["valid"] / len(records)
        ),
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


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Finite Spearman rank correlation, or None when the sample cannot define one."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return None
    from scipy.stats import spearmanr

    correlation = float(spearmanr(x[valid], y[valid]).statistic)
    return correlation if np.isfinite(correlation) else None


def _quantile_bins(values: np.ndarray, requested: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Quantile bins with duplicate edges collapsed instead of splitting tied values."""
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Quantile binning requires a non-empty finite rank-one array.")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, requested + 1)))
    if edges.size == 1:
        return np.zeros(values.size, dtype=np.int64), edges
    return np.searchsorted(edges[1:-1], values, side="right"), edges


def _bin_labels(count: int) -> list[str]:
    if count == 1:
        return ["all"]
    if count == 2:
        return ["low", "high"]
    if count == 3:
        return ["low", "medium", "high"]
    return [f"bin {index + 1}" for index in range(count)]


def _intricacy_diagnostics(records: list[dict], output_dir: str) -> dict:
    """Write target-intricacy/error plots and return their machine-readable summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intricacy = np.asarray(
        [
            np.nan
            if record["metrics"].get("target_intricacy_arm") is None
            else record["metrics"]["target_intricacy_arm"]
            for record in records
        ],
        dtype=np.float64,
    )
    energy = np.asarray(
        [record["metrics"]["target_signal_energy_arm"] for record in records],
        dtype=np.float64,
    )
    action_mse = np.asarray(
        [
            (
                record["metrics"]["arm_path_mse"]
                if "arm_path_mse" in record["metrics"]
                else record["metrics"]["path_mse"]
            )
            for record in records
        ],
        dtype=np.float64,
    )
    ee_error = np.asarray([record["metrics"]["ee_err_sample0"] for record in records], dtype=np.float64)
    gripper_transition = np.asarray(
        [bool(record["metrics"].get("gripper_transition", False)) for record in records],
        dtype=bool,
    )
    constant_count = int(np.count_nonzero(~np.isfinite(intricacy)))
    valid_scatter = np.isfinite(intricacy) & np.isfinite(action_mse)
    valid_heatmap = valid_scatter & np.isfinite(energy) & np.isfinite(ee_error)

    positive_mse = action_mse[valid_scatter & (action_mse > 0)]
    display_floor = max(float(positive_mse.min()) * 0.1, 1e-12) if positive_mse.size else 1e-12
    log_mse = np.log10(np.maximum(action_mse, display_floor))

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    scatter_count = int(valid_scatter.sum())
    if scatter_count:
        x = intricacy[valid_scatter]
        y = log_mse[valid_scatter]
        if scatter_count >= 4:
            gridsize = max(6, min(20, int(round(np.sqrt(scatter_count) * 2))))
            density = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, bins="log", cmap="viridis")
            fig.colorbar(density, ax=ax, label="chunk count (log colour scale)")
        else:
            ax.scatter(x, y, color="#457B9D", s=45, label="chunk")

        switched = valid_scatter & gripper_transition
        if switched.any():
            ax.scatter(
                intricacy[switched],
                log_mse[switched],
                marker="D",
                s=38,
                facecolors="none",
                edgecolors="#D97706",
                linewidths=1.2,
                label="gripper transition",
            )
            ax.legend(loc="best")

        rho = _spearman(intricacy[valid_scatter], action_mse[valid_scatter])
        rho_text = "undefined" if rho is None else f"{rho:+.3f}"
        ax.text(
            0.02,
            0.98,
            f"Spearman rho = {rho_text}\nn = {scatter_count}; constant = {constant_count}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
    else:
        rho = None
        ax.text(0.5, 0.5, "No non-constant target chunks", ha="center", va="center")

    ax.set(
        xlabel="target arm intricacy  C = energy(k=4..20) / energy(k=1..20)",
        ylabel=f"log10 normalized arm generated-chunk MSE (display floor {display_floor:.1e})",
        title="Do spectrally intricate demonstrated motions have larger prediction error?",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "intricacy_vs_mse.png"), dpi=180)
    plt.close(fig)

    cells = []
    energy_edges = np.asarray([], dtype=np.float64)
    intricacy_edges = np.asarray([], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    if valid_heatmap.any():
        valid_energy = energy[valid_heatmap]
        valid_intricacy = intricacy[valid_heatmap]
        energy_bin, energy_edges = _quantile_bins(valid_energy)
        intricacy_bin, intricacy_edges = _quantile_bins(valid_intricacy)
        energy_count = int(energy_bin.max()) + 1
        intricacy_count = int(intricacy_bin.max()) + 1
        counts = np.zeros((energy_count, intricacy_count), dtype=np.int64)
        action_grid = np.full((energy_count, intricacy_count), np.nan)
        ee_grid_mm = np.full((energy_count, intricacy_count), np.nan)
        switch_grid = np.full((energy_count, intricacy_count), np.nan)
        valid_action = action_mse[valid_heatmap]
        valid_ee_mm = ee_error[valid_heatmap] * 1000.0
        valid_switch = gripper_transition[valid_heatmap]

        for energy_idx in range(energy_count):
            for intricacy_idx in range(intricacy_count):
                member = (energy_bin == energy_idx) & (intricacy_bin == intricacy_idx)
                count = int(member.sum())
                counts[energy_idx, intricacy_idx] = count
                if count:
                    action_grid[energy_idx, intricacy_idx] = float(np.median(valid_action[member]))
                    ee_grid_mm[energy_idx, intricacy_idx] = float(np.median(valid_ee_mm[member]))
                    switch_grid[energy_idx, intricacy_idx] = float(valid_switch[member].mean())
                cells.append(
                    {
                        "energy_bin": energy_idx,
                        "intricacy_bin": intricacy_idx,
                        "count": count,
                        "arm_action_mse_median": (
                            None if not count else float(action_grid[energy_idx, intricacy_idx])
                        ),
                        "ee_error_sample0_median_m": (
                            None if not count else float(ee_grid_mm[energy_idx, intricacy_idx] / 1000.0)
                        ),
                        "gripper_transition_fraction": (
                            None if not count else float(switch_grid[energy_idx, intricacy_idx])
                        ),
                    }
                )

        xlabels = _bin_labels(intricacy_count)
        ylabels = _bin_labels(energy_count)
        for ax, grid, title, unit in (
            (axes[0], action_grid, "Median normalized arm action MSE", ""),
            (axes[1], ee_grid_mm, "Median sample-0 end-effector trace error", " mm"),
        ):
            image = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap="magma")
            if np.isfinite(grid).any():
                fig.colorbar(image, ax=ax, label=title + unit)
            for energy_idx in range(energy_count):
                for intricacy_idx in range(intricacy_count):
                    count = counts[energy_idx, intricacy_idx]
                    value = grid[energy_idx, intricacy_idx]
                    rendered = "—" if not np.isfinite(value) else f"{value:.3g}{unit}"
                    ax.text(
                        intricacy_idx,
                        energy_idx,
                        f"{rendered}\nn={count}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none"},
                    )
            ax.set_xticks(range(intricacy_count), xlabels)
            ax.set_yticks(range(energy_count), ylabels)
            ax.set_xlabel("target arm intricacy quantile")
            ax.set_ylabel("target trusted-energy quantile")
            ax.set_title(title)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No complete non-constant records", ha="center", va="center")
            ax.set_axis_off()

    fig.suptitle(
        "Prediction error after separating action magnitude from temporal intricacy"
        f"  (constant chunks excluded: {constant_count})"
    )
    fig.savefig(os.path.join(output_dir, "energy_intricacy_error_heatmaps.png"), dpi=180)
    plt.close(fig)

    horizon = int(records[0]["norm"]["gt"].shape[0]) if records else 0
    trusted_last = min(20, horizon - 1)
    return {
        "definition": (
            "arm-only trusted detail-energy fraction: sum power(k=4..20) / "
            "sum power(k=1..20), in normalized target-minus-hold model space"
        ),
        "trusted_indices": [0, trusted_last] if horizon else [],
        "detail_indices": [4, trusted_last] if trusted_last >= 4 else [],
        "anchors": len(records),
        "valid_intricacy": scatter_count,
        "constant_chunks": constant_count,
        "gripper_transition_chunks": int(gripper_transition.sum()),
        "spearman_intricacy_arm_action_mse": rho,
        "spearman_intricacy_ee_error_sample0": _spearman(intricacy, ee_error),
        "mse_log_display_floor": display_floor,
        "energy_quantile_edges": energy_edges.tolist(),
        "intricacy_quantile_edges": intricacy_edges.tolist(),
        "cells": cells,
    }


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

    Predicted chunks are low-passed on entry (``utils.action_smoothing``), so every
    readout below — clearance especially, which is a pre-flight check on a trajectory the
    robot is about to be commanded through — describes the filtered command rather than
    the raw decode. ``samples`` therefore arrives raw and leaves smoothed in
    ``sample_chunks``; callers that draw the returned chunks get the deployed one.

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
    # Predicted chunks are smoothed here and nowhere earlier: this is the one reduction
    # every drawn path, hover, legend and task-space metric comes out of, so the trace
    # and the numbers printed on it cannot disagree about which trajectory they describe.
    # GT is left raw — it is the demonstrated leader command, which never passed through
    # this filter — and so is the normalized ``pred``, which scores the model itself.
    sample_chunks = np.stack(
        [apply_butterworth_filter(np.asarray(sample[:horizon], dtype=np.float64)) for sample in samples]
    )
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
        "ee_err_sample0": float(position_errors[0]),
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
    """Interactive action inspector: trace scene, arm-pose panel, wrist roll, gripper.

    The trace scene holds commanded paths only. The arm itself is drawn once, small, in
    its own panel on the right: the link chain reaches back to the base, and carrying it
    in the main scene stretched the extent far past the centimetre-scale motion the
    scene exists to show.

    On an ``action_mode=both`` checkpoint the greedy FAST reconstruction shares these
    axes with the flow fan rather than taking a page of its own — the comparison is a
    distance the eye can read directly, and it only reads as a distance in one scene.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_fast = any("fast_chunk" in record for record in records)

    trace_points = np.concatenate(
        [record["gt_ee"] for record in records]
        + [record["sample_ee"].reshape(-1, 3) for record in records]
        + [record["start_ee"][None] for record in records]
    )
    pose_points = np.concatenate(
        [trace_points]
        + [record["anchor_skeleton"] for record in records]
        + [record["fast_ee"] for record in records if record.get("fast_ee") is not None]
    )
    pose_lo, pose_hi = pose_points.min(axis=0), pose_points.max(axis=0)
    # Explicit ranges clip: the table plane has to sit inside the extent or the one
    # reference the clearance means anything against silently disappears.
    pose_lo[2] = min(pose_lo[2], p.trace_table_z)
    pose_hi[2] = max(pose_hi[2], p.trace_table_z)
    # aspectmode="cube" gives every axis the same screen length, so the range has to be a
    # cube too or the arm is read in a stretched space. Fixing it also holds the panel
    # still while the slider steps through anchors, so a link that moves really moved.
    pose_centre = (pose_lo + pose_hi) / 2.0
    pose_half = max((pose_hi - pose_lo).max() / 2.0, 0.05) * 1.05
    pose_lo, pose_hi = pose_centre - pose_half, pose_centre + pose_half

    def path_hover(chunk: np.ndarray) -> np.ndarray:
        step = np.arange(len(chunk))
        time_ms = step / float(fps) * 1000.0
        roll = chunk[:, 5] if chunk.shape[1] > 5 else np.full(len(chunk), np.nan)
        grip = chunk[:, 6] if chunk.shape[1] > 6 else np.full(len(chunk), np.nan)
        return np.column_stack([step, time_ms, roll, grip])

    def path_trace(
        ee,
        chunk,
        *,
        color,
        name,
        width,
        opacity=1.0,
        dash="solid",
        marker_size=2.5,
        showlegend=True,
        title=None,
        detail="",
    ):
        """One commanded path in a 3-D scene; ``ee=None`` leaves a legend-only stub.

        The stub keeps the trace count identical across anchors, which the animation
        needs: frames address traces by index, so an anchor whose FAST decode failed
        would otherwise shift every trace after it onto the wrong data.
        """
        if ee is None:
            return go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="lines",
                line=dict(color=color, width=width),
                name=name,
                showlegend=showlegend,
                hoverinfo="skip",
            )
        return go.Scatter3d(
            x=ee[:, 0],
            y=ee[:, 1],
            z=ee[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=width, dash=dash),
            marker=dict(size=marker_size, color=color),
            opacity=opacity,
            customdata=path_hover(chunk),
            name=name,
            showlegend=showlegend,
            hovertemplate=(
                f"<b>{title or name}</b>"
                "<br>step %{customdata[0]:.0f} · %{customdata[1]:.0f} ms"
                "<br>x %{x:.3f} m · y %{y:.3f} m · z %{z:.3f} m"
                "<br>wrist roll %{customdata[2]:.1f}° · gripper %{customdata[3]:.1f}°"
                f"{detail}<extra></extra>"
            ),
        )

    def gap_trace(start, ee, *, color, name, width, dash="dot", opacity=0.75, hover=""):
        """Measured pose → first commanded target. A displacement, not a rendered step."""
        if ee is None:
            return go.Scatter3d(
                x=[None], y=[None], z=[None], mode="lines", line=dict(color=color, width=width, dash=dash),
                name=name, showlegend=False, hoverinfo="skip",
            )
        return go.Scatter3d(
            x=[start[0], ee[0, 0]],
            y=[start[1], ee[0, 1]],
            z=[start[2], ee[0, 2]],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            name=name,
            showlegend=False,
            hovertemplate=(
                f"<b>{name}</b>{hover}"
                "<br><i>Target displacement, not an interpolated timestep.</i><extra></extra>"
            ),
        )

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

    def context_path(ee, *, color, name, width=3, opacity=0.75):
        """A path in the pose panel: shape only, no hover, no second legend entry."""
        if ee is None:
            return go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", name=name, showlegend=False)
        return go.Scatter3d(
            x=ee[:, 0],
            y=ee[:, 1],
            z=ee[:, 2],
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )

    def sample_label(record, sample_idx: int) -> str:
        names = record.get("sample_names") or []
        if sample_idx < len(names):
            return str(names[sample_idx])
        return f"sample {sample_idx}"

    def timeline_traces(record, joint_idx: int, *, label: str, slug: str, row: int):
        """Measured value, GT, FAST, and the fan for one joint, on a shared step axis.

        Step −1 is the measured value now and step 0 the first commanded target, so the
        dashed leader between them is the same initial gap the 3-D scene draws.
        """
        gt_chunk = record["gt_chunk"]
        steps = np.arange(len(gt_chunk))
        traces = [
            (
                go.Scatter(
                    x=[-1],
                    y=[record["state"][joint_idx]],
                    mode="markers",
                    marker=dict(color="#D81B60", size=9, symbol="diamond"),
                    name=f"measured {label} now",
                    showlegend=False,
                    hovertemplate=f"measured {label} now: %{{y:.1f}}°<extra></extra>",
                ),
                row,
                2,
            ),
            (
                go.Scatter(
                    x=steps,
                    y=gt_chunk[:, joint_idx],
                    mode="lines+markers",
                    line=dict(color="#111111", width=4),
                    marker=dict(size=4),
                    name=f"GT {label}",
                    showlegend=False,
                    hovertemplate=f"GT {label}<br>step %{{x}}: %{{y:.1f}}°<extra></extra>",
                ),
                row,
                2,
            ),
            (
                go.Scatter(
                    x=[-1, 0],
                    y=[record["state"][joint_idx], gt_chunk[0, joint_idx]],
                    mode="lines",
                    line=dict(color="#D81B60", width=5, dash="dash"),
                    name=f"GT initial {slug} gap",
                    showlegend=False,
                    hovertemplate=f"GT initial {slug} gap: %{{y:.1f}}°<extra></extra>",
                ),
                row,
                2,
            ),
        ]
        if has_fast:
            fast_chunk = record.get("fast_chunk")
            fast_y = fast_chunk[:, joint_idx] if fast_chunk is not None else None
            traces.extend(
                [
                    (
                        go.Scatter(
                            x=steps if fast_y is not None else [],
                            y=fast_y if fast_y is not None else [],
                            mode="lines",
                            line=dict(color=FAST_COLOR, width=3),
                            opacity=0.9,
                            name=f"FAST {label}",
                            showlegend=False,
                            hovertemplate=f"FAST {label}<br>step %{{x}}: %{{y:.1f}}°<extra></extra>",
                        ),
                        row,
                        2,
                    ),
                    (
                        go.Scatter(
                            x=[-1, 0] if fast_y is not None else [],
                            y=[record["state"][joint_idx], fast_chunk[0, joint_idx]] if fast_y is not None else [],
                            mode="lines",
                            line=dict(color=FAST_COLOR, width=2, dash="dot"),
                            opacity=0.75,
                            name=f"FAST initial {slug} gap",
                            showlegend=False,
                            hovertemplate=f"FAST initial {slug} gap: %{{y:.1f}}°<extra></extra>",
                        ),
                        row,
                        2,
                    ),
                ]
            )
        for sample_idx, chunk in enumerate(record["sample_chunks"]):
            color = SAMPLE_COLORS[sample_idx % len(SAMPLE_COLORS)]
            sample_name = sample_label(record, sample_idx)
            traces.extend(
                [
                    (
                        go.Scatter(
                            x=[-1, 0],
                            y=[record["state"][joint_idx], chunk[0, joint_idx]],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dot"),
                            opacity=0.7,
                            name=f"{sample_name} initial {slug} gap",
                            showlegend=False,
                            hovertemplate=(
                                f"{sample_name} initial {slug} gap: %{{y:.1f}}°<extra></extra>"
                            ),
                        ),
                        row,
                        2,
                    ),
                    (
                        go.Scatter(
                            x=steps,
                            y=chunk[:, joint_idx],
                            mode="lines",
                            line=dict(color=color, width=2),
                            opacity=0.8,
                            name=f"{sample_name} {label}",
                            showlegend=False,
                            hovertemplate=(
                                f"{sample_name} {label}<br>step %{{x}}: %{{y:.1f}}°<extra></extra>"
                            ),
                        ),
                        row,
                        2,
                    ),
                ]
            )
        return traces

    def record_traces(record):
        metrics = record["metrics"]
        gt_ee = record["gt_ee"]
        gt_chunk = record["gt_chunk"]
        start_ee = record["start_ee"]
        fast_ee = record.get("fast_ee")
        fast_chunk = record.get("fast_chunk")

        # ---- Trace scene ------------------------------------------------------
        traces: list[tuple[object, int, int]] = [
            (
                path_trace(
                    gt_ee,
                    gt_chunk,
                    color="#111111",
                    name="ground truth targets",
                    title="GT command",
                    width=9,
                    marker_size=3,
                ),
                1,
                1,
            ),
            (
                endpoint_trace(
                    start_ee,
                    color="#D81B60",
                    symbol="diamond",
                    size=9,
                    label="gripper now — measured",
                    hover="<b>Measured gripper pose now</b>",
                    showlegend=True,
                ),
                1,
                1,
            ),
            (
                gap_trace(
                    start_ee,
                    gt_ee,
                    color="#D81B60",
                    # The numbers behind this segment live in the hover, metrics.csv and
                    # the run log; the legend only has to name it.
                    name="GT initial target gap / possible follower lag",
                    width=9,
                    dash="dash",
                    opacity=1.0,
                    hover=(
                        f"<br>{metrics['initial_gap_gt'] * 1000:.1f} mm translation"
                        f"<br>{metrics['initial_orientation_gap_gt_deg']:.1f}° tool orientation"
                        f"<br>{metrics['initial_joint_gap_gt_max_deg']:.1f}° largest joint gap"
                        f"<br>≥{metrics['initial_travel_gt_s'] * 1000:.0f} ms at configured motor speeds"
                    ),
                ),
                1,
                1,
            ),
        ]
        for axis_trace in orientation_traces(
            gt_ee[-1], record["gt_rotation"][-1], prefix="GT", opacity=1.0, dash="solid"
        ):
            traces.append((axis_trace, 1, 1))

        if has_fast:
            fast_error = record.get("fast_error")
            traces.extend(
                [
                    (
                        path_trace(
                            fast_ee,
                            fast_chunk,
                            color=FAST_COLOR,
                            name=(
                                "FAST · greedy discrete decode"
                                if not fast_error
                                else f"FAST unavailable · {fast_error}"
                            ),
                            title="FAST command",
                            width=6,
                            opacity=0.95,
                            marker_size=3,
                        ),
                        1,
                        1,
                    ),
                    (
                        gap_trace(
                            start_ee,
                            fast_ee,
                            color=FAST_COLOR,
                            name="FAST initial target gap",
                            width=4,
                            opacity=0.8,
                        ),
                        1,
                        1,
                    ),
                ]
            )

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
            sample_name = sample_label(record, sample_idx)
            unsafe = clearance < p.trace_clearance_warn_m
            traces.extend(
                [
                    (
                        path_trace(
                            ee,
                            chunk,
                            color=color,
                            name=(
                                f"{sample_name} · gap {gap * 1000:.0f} mm · "
                                f"tool {tool_clearance * 1000:+.0f} mm"
                                + (" · ⚠ TABLE" if unsafe else "")
                            ),
                            title=sample_name,
                            width=6 if unsafe else 3,
                            dash="dash" if unsafe else "solid",
                            opacity=1.0 if unsafe else 0.72,
                            detail=(
                                f"<br>whole-arm clearance {clearance * 1000:+.1f} mm ({link})"
                                f"<br>tool clearance {tool_clearance * 1000:+.1f} mm"
                            ),
                        ),
                        1,
                        1,
                    ),
                    (
                        gap_trace(
                            start_ee,
                            ee,
                            color=color,
                            name=f"{sample_name} initial target gap",
                            width=4,
                            hover=(
                                f"<br>{gap * 1000:.1f} mm translation · {orientation_gap:.1f}° orientation"
                                f"<br>≥{travel * 1000:.0f} ms all-actuator lower bound"
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
            prefix=sample_label(record, 0),
            opacity=0.65,
            dash="dot",
        ):
            traces.append((axis_trace, 1, 1))

        # ---- Arm-pose panel ---------------------------------------------------
        skeleton = record["anchor_skeleton"]
        traces.extend(
            [
                (
                    go.Scatter3d(
                        x=skeleton[:, 0],
                        y=skeleton[:, 1],
                        z=skeleton[:, 2],
                        mode="lines+markers",
                        line=dict(color="#9A9A9A", width=8),
                        marker=dict(size=4, color="#666666"),
                        name="arm now — measured follower",
                        showlegend=False,
                        hovertemplate="measured arm link<extra></extra>",
                    ),
                    1,
                    2,
                ),
                (
                    endpoint_trace(
                        start_ee,
                        color="#D81B60",
                        symbol="diamond",
                        size=7,
                        label="gripper now — measured",
                        hover="<b>Measured gripper pose now</b>",
                    ),
                    1,
                    2,
                ),
                (context_path(gt_ee, color="#111111", name="GT path (pose panel)", width=5), 1, 2),
            ]
        )
        if has_fast:
            traces.append(
                (context_path(fast_ee, color=FAST_COLOR, name="FAST path (pose panel)", width=4), 1, 2)
            )
        for sample_idx, ee in enumerate(record["sample_ee"]):
            traces.append(
                (
                    context_path(
                        ee,
                        color=SAMPLE_COLORS[sample_idx % len(SAMPLE_COLORS)],
                        name=f"{sample_label(record, sample_idx)} path (pose panel)",
                        opacity=0.6,
                    ),
                    1,
                    2,
                )
            )

        # ---- Joint timelines --------------------------------------------------
        if gt_chunk.shape[1] > 5:
            traces.extend(timeline_traces(record, 5, label="wrist roll", slug="wrist-roll", row=2))
        if gt_chunk.shape[1] > 6:
            traces.extend(timeline_traces(record, 6, label="gripper", slug="gripper", row=3))
        return traces

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "scene"}],
            [None, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        column_widths=[0.72, 0.28],
        row_heights=[0.40, 0.30, 0.30],
        horizontal_spacing=0.04,
        vertical_spacing=0.11,
        subplot_titles=("Commanded trajectory", "Arm pose now", "Wrist roll", "Gripper command"),
    )

    # The table plane belongs to the pose panel alone. In the trace scene it sat a whole
    # arm-length below a chunk that moves a couple of centimetres, and aspectmode="data"
    # sizes the box to the widest thing in it — so the plane, not the motion, set the
    # zoom. Nothing is lost by dropping it: the pose panel draws it against the arm, every
    # sample's clearance is in its own legend entry and hover, and a sample that breaches
    # is still redrawn thick and dashed.
    fig.add_trace(
        go.Mesh3d(
            x=[pose_lo[0], pose_hi[0], pose_hi[0], pose_lo[0]],
            y=[pose_lo[1], pose_lo[1], pose_hi[1], pose_hi[1]],
            z=[p.trace_table_z] * 4,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#C8B89A",
            opacity=0.30,
            name="table plane",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    # The plane is static; every trace after it is swapped anchor by anchor.
    static_count = len(fig.data)
    for trace, row, col in record_traces(records[0]):
        fig.add_trace(trace, row=row, col=col)

    # add_trace(row=, col=) binds a trace to its subplot; frame traces bypass add_trace and
    # would fall back onto the first scene and the first pair of axes. Read the bindings
    # back off the initial traces rather than deriving them, so the two scenes and two
    # timelines cannot drift apart from the grid.
    bindings = [
        (getattr(trace, "scene", None), getattr(trace, "xaxis", None), getattr(trace, "yaxis", None))
        for trace in fig.data[static_count:]
    ]
    updated = list(range(static_count, len(fig.data)))
    frames = []
    for frame_idx, record in enumerate(records):
        dynamic = []
        for (trace, _, _), (scene, xaxis, yaxis) in zip(record_traces(record), bindings, strict=True):
            if scene is not None:
                trace.scene = scene
            else:
                trace.xaxis, trace.yaxis = xaxis, yaxis
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

    horizon = len(records[0]["gt_chunk"])
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
        scene2=dict(
            xaxis=dict(title="", range=[float(pose_lo[0]), float(pose_hi[0])], showticklabels=False),
            yaxis=dict(title="", range=[float(pose_lo[1]), float(pose_hi[1])], showticklabels=False),
            zaxis=dict(title="", range=[float(pose_lo[2]), float(pose_hi[2])], showticklabels=False),
            aspectmode="cube",
            bgcolor="#FBFBFC",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.05), up=dict(x=0, y=0, z=1)),
            uirevision="action-inspector-pose-camera",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFB",
        height=880,
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
    for row in (2, 3):
        fig.update_xaxes(title="measured now (−1) → chunk step", range=[-1, horizon - 1], row=row, col=2)
        fig.update_yaxes(title="degrees", zeroline=False, row=row, col=2)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#333333")
    return fig


def _title(record: dict) -> str:
    label = record.get("trace_label") or f"episode {record['episode']} · frame {record['frame']}"
    return f"<b>{html.escape(str(label))}</b>"


def _log_line(record: dict) -> str:
    """The per-anchor numbers, kept in the run log instead of on the figure."""
    m = record["metrics"]
    return (
        f"episode {record['episode']} · frame {record['frame']}  |  initial gap GT "
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
    return {
        "episode": int(record["episode"]),
        "frame": int(record["frame"]),
        "label": record.get("trace_label")
        or f"episode {record['episode']} · frame {record['frame']}",
        "subtask": record.get("subtask") or "(no subtask clause)",
        "cameras": record.get("cameras", []),
        "trajectory": {
            key: record.get("metrics", {}).get(key)
            for key in TRAJECTORY_ERROR_KEYS + TRAJECTORY_RELATIVE_KEYS
        },
    }


def _write_dashboard_html(
    fig,
    records: list[dict],
    html_path: str,
    *,
    page_title: str = "Action Inspector",
    subtitle: str = (
        "One observation, one set of deterministic policy draws, viewed in task space "
        "and joint space. Orbit the commanded paths, check where the arm actually is "
        "in the pose panel, then follow the same colors through wrist roll and gripper."
    ),
    legend_note: str = (
        "Solid RGB axes = GT terminal tool orientation · dotted RGB axes = sample 0. "
        "Amber, when present, is the greedy FAST decode. "
        "The pose panel holds fixed axes across anchors."
    ),
) -> None:
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
<title>__PAGE_TITLE__</title>
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
.warning { border:1px solid rgba(216,27,96,.32); background:#fff5f8; border-radius:9px; padding:8px 9px; margin-bottom:12px; }
.warning strong { color:var(--accent); display:block; font-size:11px; letter-spacing:.04em; }
.warning p { margin:3px 0 0; font-size:11px; line-height:1.35; color:#6f1d3d; }
.controls { display:flex; gap:8px; margin-bottom:12px; }
button { flex:1; border:1px solid var(--line); background:#fafafa; border-radius:9px; padding:8px; font-weight:700; cursor:pointer; }
button:hover { background:#f0f0f1; }
.section-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; margin:14px 0 7px; }
.subtask { border-left:3px solid #111; padding:4px 0 4px 10px; font-weight:650; }
.cameras { display:grid; gap:9px; }
.camera { margin:0; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#111; }
.camera img { width:100%; display:block; aspect-ratio:4/3; object-fit:cover; }
.camera figcaption { background:white; padding:6px 8px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.metric-list { display:grid; gap:9px; }
.metric-head { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
.metric-name { font-size:12px; font-weight:700; }
.metric-value { font:700 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
.metric-track { height:8px; border-radius:99px; background:#e4e4e7; position:relative; margin:5px 4px 3px; }
.metric-dot { width:5px; height:5px; border-radius:99px; background:#71717a; position:absolute; top:1.5px; transform:translateX(-50%); opacity:.5; }
.metric-dot.active { width:9px; height:9px; top:-.5px; background:var(--accent); opacity:1; border:2px solid white; box-shadow:0 0 0 1px var(--accent); z-index:2; }
.metric-range { display:flex; justify-content:space-between; color:var(--muted); font:9px ui-monospace, SFMono-Regular, Menlo, monospace; }
.metric-help { color:var(--muted); font-size:10px; line-height:1.35; margin-top:7px; }
.legend-note { color:var(--muted); font-size:11px; margin-top:10px; }
@media (max-width:1100px) { .layout { grid-template-columns:1fr; } .context { position:static; } .cameras { grid-template-columns:1fr 1fr; } }
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div><h1>__PAGE_TITLE__</h1><div class="subtitle">__SUBTITLE__</div></div>
    <div class="anchor-pill" id="anchor-pill"></div>
  </div>
  <div class="layout">
    <main class="card plot-card">__PLOT__</main>
    <aside class="card context">
      <div class="warning"><strong>POSSIBLE FOLLOWER LAG</strong><p>Magenta: measured pose → first demonstrated target. Dotted spokes: the same gap per sample. Not an interpolated timestep.</p></div>
      <div class="controls"><button id="prev">← Previous</button><button id="next">Next →</button></div>
      <div class="section-label">Conditioning</div><div class="subtask" id="subtask"></div>
      <div class="section-label">Trajectory fit · sample 0</div><div class="metric-list" id="trajectory-metrics"></div>
      <div class="metric-help">Dots are the distribution across anchors; magenta is the action currently drawn. Every value is lower-is-better and scale-free, so anchors are comparable to each other however far the demonstration travels. The first three divide by the error of holding the arm still: 1 means the prediction is worth no more than freezing, above 1 means worth less. Final direction loss is 0 aligned, 1 perpendicular or no predicted displacement, and 2 opposite; displacement length is ignored. The auxiliary loss still gates on the raw MSEs, whose p75s are in action_metrics.json.</div>
      <div class="section-label">Observation</div><div class="cameras" id="cameras"></div>
      <div class="legend-note">__LEGEND_NOTE__</div>
    </aside>
  </div>
</div>
<script id="action-context" type="application/json">__CONTEXTS__</script>
<script>
(() => {
  const contexts = JSON.parse(document.getElementById('action-context').textContent);
  const plot = document.getElementById('action-inspector-plot');
  // Every row is relative to the hold predictor, so one scale reads for every anchor:
  // 1 is the freeze-the-arm baseline, and a small careful motion is not flattered by
  // having had little distance to travel. The raw MSEs stay in action_metrics.json and
  // metrics.csv for threshold-setting.
  const metricSpecs = [
    {key:'path_relative', label:'Path error / hold', fixed:null},
    {key:'shape_relative', label:'Temporal shape error / hold', fixed:null},
    {key:'terminal_relative', label:'Final-position error / hold', fixed:null},
    {key:'terminal_direction_loss', label:'Final direction loss', fixed:[0,2]}
  ];
  let active = 0;
  // Temporal shape MSE lives two to three decades below the other three: it is measured
  // on adjacent-target differences, which are ~1/T of the excursion the other metrics
  // see. Three decimals would print its value, its p75 and both ends of its range as
  // 0.000, so anything under the last decimal place switches to scientific notation.
  function metricText(value) {
    if(!Number.isFinite(value)) return 'undefined';
    if(Math.abs(value) >= 100) return value.toFixed(1);
    return value !== 0 && Math.abs(value) < 1e-3 ? value.toExponential(2) : value.toFixed(3);
  }
  function renderMetrics(index) {
    const host = document.getElementById('trajectory-metrics');
    host.replaceChildren();
    metricSpecs.forEach(spec => {
      const values = contexts.map(c => c.trajectory && c.trajectory[spec.key]);
      const finite = values.filter(Number.isFinite);
      const row = document.createElement('div');
      const current = values[index];
      if(!finite.length) {
        row.innerHTML = `<div class="metric-head"><span class="metric-name">${spec.label}</span><span class="metric-value">undefined</span></div>`;
        host.append(row); return;
      }
      const lo = spec.fixed ? spec.fixed[0] : Math.min(...finite);
      const hi = spec.fixed ? spec.fixed[1] : Math.max(...finite);
      const span = hi - lo || 1;
      const position = value => Math.max(0, Math.min(100, 100 * (value - lo) / span));
      const sorted = [...finite].sort((a,b) => a-b);
      const q = .75 * (sorted.length - 1);
      const q0 = Math.floor(q), q1 = Math.ceil(q);
      const p75 = sorted[q0] + (sorted[q1] - sorted[q0]) * (q - q0);
      const dots = values.map((value, dotIndex) => Number.isFinite(value)
        ? `<span class="metric-dot ${dotIndex===index?'active':''}" style="left:${position(value)}%"></span>`
        : '').join('');
      row.innerHTML = `<div class="metric-head"><span class="metric-name">${spec.label}</span><span class="metric-value">${metricText(current)}</span></div>
        <div class="metric-track">${dots}</div><div class="metric-range"><span>${metricText(lo)}</span><span>p75 ${metricText(p75)}</span><span>${metricText(hi)}</span></div>`;
      host.append(row);
    });
  }
  function render(index) {
    active = Math.max(0, Math.min(contexts.length - 1, Number(index)));
    const context = contexts[active];
    document.getElementById('anchor-pill').textContent = context.label;
    document.getElementById('subtask').textContent = context.subtask;
    renderMetrics(active);
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
  window.probeInspectorGo = go;
  plot.on('plotly_sliderchange', event => render(event.step.value ?? event.step.args[0][0]));
  document.getElementById('prev').addEventListener('click', () => go(active-1));
  document.getElementById('next').addEventListener('click', () => go(active+1));
  document.addEventListener('keydown', event => { if(event.key==='ArrowLeft') go(active-1); if(event.key==='ArrowRight') go(active+1); });
  render(0);
})();
</script>
</body>
</html>"""
    document = (
        template.replace("__PAGE_TITLE__", html.escape(page_title))
        .replace("__SUBTITLE__", html.escape(subtitle))
        .replace("__PLOT__", plot_html)
        .replace("__LEGEND_NOTE__", html.escape(legend_note))
        .replace("__CONTEXTS__", payload)
    )
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
    how = (
        "Four panels, one anchor at a time. The large scene holds commanded paths only: "
        "demonstrated motion in black, the flow fan in colour, magenta for the "
        "measured-state → first demonstrated target gap (possible follower lag) and "
        "colour-matched dotted spokes for the policy's own. It frames the chunk and only "
        "the chunk, so the motion fills the box — the table plane and the arm live in the "
        "small scene beside it, on fixed cube axes, where a link that moves between two "
        "slider steps really moved. Clearance is still per-sample in the trace legend, and "
        "a sample that breaches is redrawn thick and dashed. Below the pose panel wrist "
        "roll and gripper share the same colours, with step −1 the measured value now and "
        "step 0 the first target. The sidebar's compact trajectory-fit tracks show the "
        "distribution over anchors, with the currently drawn action highlighted."
    )
    if has_decoder_comparison:
        how += (
            "\n\nOn this ``action_mode=both`` checkpoint the greedy FAST reconstruction is "
            "drawn in amber in the same axes as the flow fan, so the gap between decoders "
            "is a distance the eye reads directly. It stays a picture, not a score: FAST is "
            "greedy, so it has no fan and it carries the action tokenizer's quantization "
            "error, which flow does not; and the discrete path never runs the action expert, "
            "so with point-map depth configured the depth stream reaches the flow draws and "
            "structurally cannot reach FAST. A FAST decode that failed at an anchor says so "
            "in the legend instead of drawing a trace."
        )
    panels = [
        Panel(
            "action_trace.html",
            "Interactive task-space and joint-space action inspector",
            how=how,
            primary=True,
        ),
        Panel(
            "energy_intricacy_error_heatmaps.png",
            "Generated-action error by target energy and arm intricacy",
            how="Rows and columns are quantile bins over non-constant demonstrated chunks; tied boundaries are collapsed. Each cell shows the median error and sample count. If error rises from left to right within a row, intricacy predicts difficulty after approximately controlling for action magnitude.",
            primary=True,
        ),
        Panel(
            "intricacy_vs_mse.png",
            "Target arm intricacy against arm-only generated-chunk MSE",
            how="Hexagon colour is chunk density. Intricacy is the fraction of trusted non-DC arm energy in k=4..20; orange diamonds contain a gripper transition but the gripper never enters the intricacy calculation. Correlation is descriptive and does not establish task importance.",
        ),
    ]
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
                note=f"Same, against the best constant chunk over these anchors. Worst joint is {fit['worst_joint']}.",
            ),
            Metric(
                "tool_clearance_min_mm",
                "Worst tool clearance",
                good="high",
                fmt=0,
                warn=10.0,
                bad=0.0,
                primary=True,
                note="Minimum end-link hull clearance, separated from the proximal link that often owns the whole-arm minimum.",
            ),
            Metric(
                "whole_clearance_min_mm",
                "Worst whole-arm clearance",
                good="high",
                fmt=0,
                warn=10.0,
                bad=0.0,
            ),
            Metric(
                "initial_gap_pred_p90_mm",
                "Initial target gap p90",
                good="low",
                fmt=0,
                primary=True,
                note="P90 across anchors of the largest policy sample gap, in mm. A controller target displacement, not one rendered timestep — no threshold, because where the line sits depends on the controller.",
            ),
            Metric(
                "command_slew_max_ratio",
                "Largest command slew / motor speed",
                good="low",
                fmt=2,
                baseline=1.0,
                warn=1.0,
                note="Above 1 means adjacent targets change faster than the configured motor speed can physically track.",
            ),
            Metric(
                "fit.path_relative",
                "Path error / hold",
                good="low",
                fmt=3,
                baseline=1.0,
                warn=1.0,
                note="Path MSE over the MSE of holding the arm still, per anchor, then averaged. Scale-free, so a short careful motion is not flattered by having little distance to travel. Above 1 means the prediction is worth less than freezing the arm.",
            ),
            Metric(
                "fit.shape_relative",
                "Temporal shape error / hold",
                good="low",
                fmt=3,
                baseline=1.0,
                warn=1.0,
                note="Adjacent-target-change MSE over the demonstration's own per-step motion, per anchor. Penalizes oscillation and mistimed motion even where a constant offset leaves path error intact.",
            ),
            Metric(
                "fit.terminal_relative",
                "Final-position error / hold",
                good="low",
                fmt=3,
                baseline=1.0,
                warn=1.0,
                note="Error at target 30 over the final displacement the demonstration actually asked for.",
            ),
            Metric(
                "fit.path_mse",
                "Path MSE (raw)",
                good="low",
                fmt=3,
                note="Mean normalized action error over all 30 targets and joints, using flow sample 0. Scale-dependent — it rises with how far the chunk travels, so read fit.path_relative to compare anchors. Kept as a raw diagnostic; the auxiliary loss gates on fit.path_relative.",
            ),
            Metric(
                "fit.shape_mse",
                "Temporal shape MSE (raw)",
                good="low",
                fmt=3,
                note="MSE between adjacent-target changes. Two to three decades below the other raw terms because increments are ~1/T of the excursion.",
            ),
            Metric(
                "fit.terminal_mse",
                "Final-position MSE (raw)",
                good="low",
                fmt=3,
                note="Normalized error at target 30 only.",
            ),
            Metric(
                "fit.terminal_direction_loss",
                "Final direction loss",
                good="low",
                fmt=3,
                note=f"1 − cosine similarity of final displacement from hold, independent of length. Undefined hold targets are excluded ({fit['trajectory']['terminal_direction_loss']['valid']}/{fit['n_frames']} valid).",
            ),
        ],
        panels=panels,
        extra={
            "viewer": {
                "metric_groups": [
                    {
                        "title": "Trajectory fit · flow sample 0",
                        "keys": [
                            "fit.path_relative",
                            "fit.shape_relative",
                            "fit.terminal_relative",
                            "fit.terminal_direction_loss",
                        ],
                    }
                ]
            }
        },
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
    # scene would claim to show flow while showing FAST once per sample.
    has_decoder_comparison = bool(
        callable(decoder_predict)
        and getattr(cfg.policy, "action_mode", None) == "both"
        and getattr(cfg.policy, "inference_action_mode", None) == "continuous"
    )
    flow_sample_count = max(int(p.trace_n_samples), 1)

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
            record["metrics"].update(_trajectory_metrics(record["norm"]))
            record["metrics"]["mse_norm"] = record["metrics"]["path_mse"]
            joint_names = joint_names_for_dim(int(gt_actions.shape[-1]))
            gripper_indices = [index for index, name in enumerate(joint_names) if "gripper" in name.lower()]
            record["metrics"]["gripper_transition"] = any(
                float(gt_actions[:, index].max() - gt_actions[:, index].min()) > 1e-6
                for index in gripper_indices
            )
            if has_decoder_comparison:
                record.update(fast_error=fast_error, fast_chunk=None, fast_ee=None)
                if fast_prediction is not None:
                    # Smoothed like the flow fan: the two share one set of axes, and the
                    # comparison is only a comparison if both are the deployed command.
                    fast_chunk = apply_butterworth_filter(
                        np.asarray(fast_prediction[: len(record["gt_chunk"])], dtype=np.float64)
                    )
                    if len(fast_chunk) < 1:
                        raise RuntimeError("FAST reconstruction returned an empty action chunk.")
                    fast_geometry = _trajectory_geometry(kin, fast_chunk)
                    record.update(fast_chunk=fast_chunk, fast_ee=fast_geometry["ee"])
            records.append(record)
            logging.info(f"[{anchor_idx + 1}/{len(anchors)}] {_log_line(record)}")
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    fig = _figure(records, p, fps=fps)
    html_path = os.path.join(output_dir, "action_trace.html")
    _write_dashboard_html(fig, records, html_path)

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        columns = ["episode", "frame", "global_idx", *records[0]["metrics"]]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, record["metrics"].get(key)) for key in columns})

    fit = _fit_metrics(records)
    fit["intricacy"] = _intricacy_diagnostics(records, output_dir)
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


# Runs inside rl_offline's validation loop when probe_parameters.enable_action_trace is
# set, or standalone:
#
#     python -m lerobot.probes.action_trace_probe --config config_rl.yaml \
#         --probe_parameters.trace_anchor_stride_s 30 --probe_parameters.trace_n_samples 4
#
# Standalone has no val set to fall back on and loads dataset.sources[0], so pin
# trace_episodes unless you want every episode of it.
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
    register_config_choices()
    main()
