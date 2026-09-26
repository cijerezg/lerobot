#!/usr/bin/env python
"""
Generic critic probe — visualise the distributional critic's value estimates.

Policy-agnostic: works with any policy whose ``ProbablePolicy`` adapter
implements ``predict_value`` and ``predict_value_and_probs``. Adapters that
also implement ``value_gradient_magnitude`` get the gradient-based plots and
percentile-exemplar frames; otherwise those sections are skipped.

Every critic forward carries the frame's own reviewed labels (quality / mistake /
speed / precision / contact) and subtask, because the training critic reads the
actor's prompt (MolmoAct2Trainer._critic_batches). The TD error is built with the
training sampler's reward rule (ReplayBuffer.sample): in subtask mode -1 per step,
0 on the step whose chunk holds a segment or episode end, minus the mistake penalty
at a mistake onset, over the normalization constant.

Outputs (under ``probe_parameters.output_dir/critic/``):
  predicted_distributions.png   per-frame P(V) curves with E[V] overlay
  advantage_dist.png            advantage (TD-error) histogram with the percentile cuts,
                                its CDF, the V(s) histogram, by-subtask spread
  advantage_weights.png         the AWR weights the actor is trained with
                                (lerobot.rl.advantage): standardized advantage, weight
                                histogram, beta sweep
  advantage_weight_where.png    mean weight by seconds to the segment end, by mistake
                                flag and by quality label
  advantage_percentiles.png     camera frames nearest the p5/p10/p50/p90/p95 advantage
                                cuts (two per cut, different segments)
  value_vs_time_to_end.png      V(s) against seconds to the segment end, with the
                                duration-only ideal V* the reward rule implies
  value_by_label.png            mean V - V* by quality label and by mistake flag
  value_outliers.png            camera frames of the 2 most pessimistic + 2 most
                                optimistic frames (tagged P1/P2/O1/O2 on the scatter)
  gradient_magnitudes.png       (if adapter supports it)
  frame_p{XX}.png               (if adapter supports it) percentile exemplars
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import textwrap
import warnings
from dataclasses import dataclass

warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.rl.advantage import advantage_weights
from lerobot.probes.utils import (
    build_episode_index,
    canonical_camera_obs,
    frame_metadata_lookup,
    get_frame_data,
    load_probe_dataset,
    probe_frame_inputs,
    register_config_choices,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeCriticConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters`` (ProbeConfig)."""


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def get_random_valid_samples(
    dataset,
    n_frames: int,
    seed: int,
    val_ep_indices: set[int] | None = None,
    lookahead_frames: int = 1,
    stride: int = 1,
) -> list[int]:
    """Pick *n_frames* indices whose [idx, idx + lookahead_frames] window stays in one episode.

    ``stride`` snaps anchors onto the image/depth grid (``policy.image_stride``), the
    same contract as ``probes.utils.sample_episodes_evenly``. Off-grid frames have no
    depth PNG in the sidecar, so any caller feeding these indices to
    ``probe_frame_inputs`` with depth enabled must pass the stride.
    """
    rng = random.Random(seed)

    if val_ep_indices is not None:
        ep_to_indices = build_episode_index(dataset)
        candidates = [g for ep, idxs in ep_to_indices.items() if ep in val_ep_indices for g in idxs]
    else:
        candidates = list(range(len(dataset)))
    rng.shuffle(candidates)

    samples: list[int] = []
    for idx in candidates:
        if len(samples) >= n_frames:
            break
        end = idx + lookahead_frames
        if end >= len(dataset):
            continue
        item = dataset.hf_dataset[idx]
        if stride > 1 and int(item["frame_index"].item()) % stride != 0:
            continue
        end_item = dataset.hf_dataset[end]
        if item["episode_index"].item() != end_item["episode_index"].item():
            continue
        samples.append(idx)
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    sns.despine(ax=ax, offset=10, trim=True)


def _render_percentile_frame(obs, ep_idx, fr_idx, subtask, mag, p, output_dir):
    camera_keys = sorted(k for k in obs if "images" in k)
    n_cameras = len(camera_keys)

    fig = plt.figure(figsize=(12 + 3, 5))
    gs = GridSpec(1, n_cameras + 1, figure=fig,
                  width_ratios=[1] * n_cameras + [0.8])
    for i, key in enumerate(camera_keys):
        ax = fig.add_subplot(gs[0, i])
        img = obs[key].squeeze(0).cpu()
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.float().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(key.split(".")[-1], fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")

    ax_info = fig.add_subplot(gs[0, n_cameras])
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    wrapped = "\n  ".join(textwrap.wrap(subtask or "None", width=25))
    info = (
        f"Gradient percentile:\n  p{p}\n\n"
        f"Magnitude:\n  {mag:.4f}\n\n"
        f"Episode:\n  {ep_idx}\n\n"
        f"Frame:\n  {fr_idx}\n\n"
        f"Subtask:\n  {wrapped}"
    )
    ax_info.text(
        0.1, 0.8, info, transform=ax_info.transAxes,
        fontsize=13, va="top", ha="left", color="#333",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ccc",
                  boxstyle="round,pad=1"),
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"frame_p{p:02d}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Probe sections
# ──────────────────────────────────────────────────────────────────────────────

def run_episode_critic_traces(
    adapter: ProbablePolicy, val_dataset, val_ep_indices,
    cfg, output_dir: str,
):
    """For each selected episode: run the critic at a fixed stride, save the V(s)
    curve with the segment ends (subtask boundaries after the release fold, the
    episode end) and the duration ideal V* + JSON, and optionally per-frame PNGs
    and a critic-overlay video.

    The V(s) curve is sub-sampled (``probe_parameters.critic_trace_stride_frames``,
    default 30 = one V per second at 30 fps, rounded up onto the image stride
    grid). The per-frame PNG dump and the overlay video (native fps) are behind
    ``probe_parameters.critic_trace_video``: decoding every frame one by one costs
    about 8 minutes per episode, the curve alone about 40 seconds. Reuses
    :func:`lerobot.rl.utils.save_video_with_critic_overlay`, which is
    policy-agnostic (reads PNGs from disk + plots the supplied curve).

    Outputs (one sub-directory per episode):
        {output_dir}/ep{NNNN}/critic_values.json
        {output_dir}/ep{NNNN}/critic_plot.png
        {output_dir}/ep{NNNN}/episode_video.mp4

    TODO(future): gradient-magnitude variant (``run_episode_gradient_traces`` in
    the reference) — overlays L2 norm of dV/d(vision) onto the same video. Needs
    ``adapter.value_gradient_magnitude`` and is currently pi05-only because
    molmoact2 hasn't plumbed requires_grad through forward_critic yet.
    """
    from lerobot.rl.utils import save_video_with_critic_overlay
    from PIL import Image as PILImage

    p = cfg.probe_parameters
    chunk_size = adapter.chunk_size
    subsample = max(1, int(getattr(p, "critic_trace_stride_frames", 30)))
    with_video = bool(getattr(p, "critic_trace_video", False))
    video_stride = max(1, int(getattr(p, "critic_trace_video_frame_stride", 5)))
    seed = int(getattr(p, "random_seed", 42))
    max_episodes = getattr(p, "max_episodes", None)
    video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])
    fps = cfg.env.fps

    targets = _training_targets(val_dataset, cfg, chunk_size)
    labels = targets["labels"]
    ideal_all = _ideal_duration_value(
        targets["frames_to_end"], chunk_size,
        float(getattr(cfg.policy, "discount", 0.97)),
        float(getattr(cfg.policy, "reward_normalization_constant", 1.0)),
        float(getattr(cfg.policy, "value_support_min", -2.0)),
    )
    boundary_all = targets["episode_end"] if targets["terminals"] is None else (targets["terminals"] | targets["episode_end"])
    ep_to_indices = build_episode_index(val_dataset)
    if val_ep_indices is not None:
        ep_to_indices = {k: v for k, v in ep_to_indices.items() if k in val_ep_indices}
    selected_eps = sorted(ep_to_indices.keys())
    if max_episodes:
        rng = random.Random(seed)
        rng.shuffle(selected_eps)
        selected_eps = sorted(selected_eps[: int(max_episodes)])

    if not selected_eps:
        logging.warning("[CRITIC] episode_traces: no episodes")
        return None

    os.makedirs(output_dir, exist_ok=True)

    for ep_idx in selected_eps:
        indices = ep_to_indices[ep_idx]
        ep_dir = os.path.join(output_dir, f"ep{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        # ── 1. Video frames: every video_stride-th frame, half size, + subtask text ─
        # Decoding is the cost (random access into the episode's video), so the video
        # runs at fps / video_stride; the renderer resizes every frame to 448 px anyway.
        subtask_texts: list[str] = []
        video_frames = indices[::video_stride] if with_video else []
        for step_idx, global_idx in enumerate(video_frames):
            obs, _, _, gt_subtask, _, _, _ = get_frame_data(
                val_dataset, global_idx, chunk_size,
            )
            subtask_texts.append(gt_subtask or "")
            for key, val in obs.items():
                if "image" not in key:
                    continue
                cam_name = key.split(".")[-1]
                if cam_name not in video_logging_cameras:
                    continue
                img_tensor = val[0] if val.ndim == 4 else val
                if img_tensor.dtype == torch.uint8:
                    img_np = img_tensor.numpy().transpose(1, 2, 0)
                else:
                    v_max = img_tensor.max().item()
                    # Heuristic: small max => [0,1] float; otherwise [0,255].
                    if v_max <= 5.0:
                        img_np = (img_tensor.float().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = img_tensor.float().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
                image = PILImage.fromarray(img_np)
                image = image.resize((image.width // 2, image.height // 2))
                image.save(os.path.join(ep_dir, f"step_{step_idx:06d}_{cam_name}.png"))

        # ── 2. Subsampled V(s) via adapter ───────────────────────────────────
        # Must use probe_frame_inputs, not get_frame_data: the training critic
        # reads the encoder token sequence, which carries the state/RGB/depth
        # history windows (buffer.py puts them in batch_state, and
        # history_dropout is 0.0, so it sees them on every sample). Feeding a
        # history-free frame here is an out-of-distribution prompt. Depth is
        # included for parity, so anchors must stay on the image_stride grid.
        # The training critic reads the actor's prompt, metadata clause included
        # (_critic_batches forwards the sampler's label columns), so every frame
        # carries its own reviewed labels here.
        critic_values: list[float] = []
        stride = int(getattr(cfg.policy, "image_stride", 1))
        step = subsample + (-subsample % stride)
        critic_indices = list(range(0, len(indices), step))
        for ci in critic_indices:
            frame = probe_frame_inputs(
                val_dataset, cfg, indices[ci], chunk_size, metadata=labels.get(indices[ci]),
            )
            obs, gt_subtask, task_str = frame["obs"], frame["subtask"], frame["task"]
            try:
                v = adapter.predict_value(obs, task_str, gt_subtask, metadata=frame["metadata"])
            except Exception as exc:
                logging.warning(f"[CRITIC] ep{ep_idx} step{ci}: V(s) failed: {exc}")
                v = 0.0
            critic_values.append(v)

        # ── 3. Save critic JSON + plot ───────────────────────────────────────
        trace_seconds = [(indices[ci] - indices[0]) / float(fps) for ci in critic_indices]
        # The critic's segments: subtask boundaries (release folded into its predecessor)
        # and the episode end, exactly where the training reward rule places terminals.
        trace_ideal = [float(ideal_all[indices[ci]]) for ci in critic_indices]
        boundary_seconds = [(g - indices[0]) / float(fps) for g in indices if boundary_all[g]]
        with open(os.path.join(ep_dir, "critic_values.json"), "w") as f:
            json.dump(
                {
                    "seconds": trace_seconds,
                    "values": critic_values,
                    "ideal_values": trace_ideal if targets["mode"] == "subtask" else None,
                    "segment_end_seconds": boundary_seconds,
                    "stride_frames": step,
                },
                f,
            )
        if critic_values:
            plt.figure(figsize=(12, 5))
            plt.plot(trace_seconds, critic_values, marker=".", color="steelblue", label="V(s)")
            if targets["mode"] == "subtask":
                plt.plot(trace_seconds, trace_ideal, color="black", linewidth=1.5, label="duration-only ideal V*")
            for k, b in enumerate(boundary_seconds):
                plt.axvline(b, color="gray", linestyle="--", linewidth=1, label="segment end" if k == 0 else None)
            plt.title(f"Critic values along episode {ep_idx} (segments = subtasks, release folded)")
            plt.xlabel("seconds into the episode")
            plt.ylabel("V(s)")
            plt.grid(True, alpha=0.4)
            plt.legend(loc="lower left", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(ep_dir, "critic_plot.png"), dpi=150)
            plt.close()

        if not with_video:
            continue

        # ── 4. Overlay video ─────────────────────────────────────────────────
        try:
            # Critic point j sits at source frame j*step = video frame j*step/video_stride.
            save_video_with_critic_overlay(
                ep_dir, critic_values,
                camera_names=video_logging_cameras,
                fps=max(1.0, float(fps) / video_stride),
                subtask_texts=subtask_texts,
                subsample=max(1, round(step / video_stride)),
            )
        except Exception as exc:
            logging.warning(
                f"[CRITIC] ep{ep_idx}: overlay video failed: {exc}", exc_info=True,
            )

    return {"episodes": selected_eps, "subsample": subsample}


def run_predicted_distributions(
    adapter: ProbablePolicy, val_dataset, val_ep_indices,
    cfg, output_dir: str,
):
    """Plot P(V) curves for a handful of random frames, with E[V] overlay."""
    p = cfg.probe_parameters
    chunk_size = adapter.chunk_size
    n_frames = int(getattr(p, "critic_dist_frames", 9))
    seed = int(getattr(p, "random_seed", 42)) + 7

    indices = get_random_valid_samples(
        val_dataset, n_frames, seed,
        val_ep_indices=val_ep_indices, lookahead_frames=0,
        stride=int(getattr(cfg.policy, "image_stride", 1)),
    )
    if not indices:
        logging.warning("[CRITIC] predicted_distributions: no samples")
        return None

    # Time to the end of the critic's segment (subtask boundary with the release fold, or
    # the episode end), the horizon the reward rule actually scores, not the episode end.
    targets = _training_targets(val_dataset, cfg, chunk_size)
    labels = targets["labels"]
    fps = float(cfg.env.fps)
    ideal_all = _ideal_duration_value(
        targets["frames_to_end"], chunk_size,
        float(getattr(cfg.policy, "discount", 0.97)),
        float(getattr(cfg.policy, "reward_normalization_constant", 1.0)),
        float(getattr(cfg.policy, "value_support_min", -2.0)),
    )

    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

    for i, idx in enumerate(indices):
        frame = probe_frame_inputs(val_dataset, cfg, idx, chunk_size, metadata=labels.get(idx))
        obs, gt_subtask, task_str = frame["obs"], frame["subtask"], frame["task"]
        ep_idx, fr_idx = frame["episode_idx"], frame["frame_idx"]
        v, probs, bin_centers = adapter.predict_value_and_probs(
            obs, task_str, gt_subtask, metadata=frame["metadata"]
        )
        seconds_to_end = targets["frames_to_end"][idx] / fps

        ax = axes[i // n_cols, i % n_cols]
        ax.plot(bin_centers, probs, color="steelblue", linewidth=2)
        ax.fill_between(bin_centers, probs, alpha=0.2, color="steelblue")
        ax.axvline(v, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"E[V] = {v:.3f}")
        if targets["mode"] == "subtask":
            ax.axvline(float(ideal_all[idx]), color="black", linestyle=":", linewidth=1.5,
                       label=f"V* = {float(ideal_all[idx]):.3f}")
        title = f"ep{ep_idx} f{fr_idx}  ({seconds_to_end:.1f} s to segment end)"
        if gt_subtask:
            title += f"\n{gt_subtask[:40]}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("V")
        ax.set_ylabel("P(V)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(indices), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    plt.tight_layout(pad=2.0)
    out_path = os.path.join(output_dir, "predicted_distributions.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logging.info(f"[CRITIC] saved {out_path}")
    return {"indices": indices}


_ADVANTAGE_PERCENTILES = (5, 10, 50, 90, 95)
_BETA_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0)


def _ess_frac(w: np.ndarray) -> float:
    """ESS / N = (Σw)² / (N Σw²), as the trainer logs adv_weight_ess_frac."""
    return float(w.sum() ** 2 / (w.size * (w**2).sum()))


def _advantage_plots(advantages, values, subtasks, cuts: dict[int, float], output_dir):
    """advantage_dist.png: A histogram with the percentile cuts, A CDF, V(s) histogram, A by subtask."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    sns.histplot(advantages, bins=50, kde=True, ax=axes[0, 0], color="coral", edgecolor="white")
    top = axes[0, 0].get_ylim()[1]
    for k, (pct, cut) in enumerate(cuts.items()):
        axes[0, 0].axvline(cut, color="black", linestyle=":", linewidth=1)
        axes[0, 0].annotate(f"p{pct}", (cut, top * (1.0 if k % 2 == 0 else 0.96)), ha="center", va="bottom", fontsize=9)
    _style(axes[0, 0], "Advantage (TD-error) Histogram", "A = r + γ V(s') − V(s)", "Count")
    sns.ecdfplot(advantages, ax=axes[0, 1], color="coral", linewidth=3)
    _style(axes[0, 1], "Advantage CDF", "A", "Cumulative Probability")
    axes[0, 1].margins(y=0.05)
    sns.histplot(values, bins=50, kde=True, ax=axes[1, 0], color="steelblue", edgecolor="white")
    _style(axes[1, 0], "V(s) Histogram", "V(s)", "Count")
    data = {"advantage": advantages, "subtask": subtasks}
    sns.boxplot(x="advantage", y="subtask", hue="subtask", legend=False, data=data,
                ax=axes[1, 1], palette="pastel", fliersize=0)
    sns.stripplot(x="advantage", y="subtask", data=data, ax=axes[1, 1], color=".3", size=3, alpha=0.5, jitter=True)
    _style(axes[1, 1], "Advantage by Subtask", "A", "Subtask")
    axes[1, 1].tick_params(axis="y", labelsize=8)
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "advantage_dist.png")
    plt.savefig(out, dpi=200); plt.close()
    logging.info(f"[CRITIC] saved {out}")


def _weight_plots(advantages, a_hat, weights, beta: float, clip: float, lam: float, batch_rows: int, seed: int, output_dir) -> dict:
    """advantage_weights.png: Â and w histograms plus the beta sweep; returns the numbers behind them.

    ESS/N and KL(target ‖ BC) = mean(w log w) are the trainer's adv_weight_ess_frac /
    adv_weight_kl over the pooled probe frames. The ``_batch`` numbers are medians over
    200 resampled batches of ``batch_rows`` frames (the trainer's effective batch, the
    set it actually standardizes over), so they are what one optimizer step sees.
    """
    a = torch.as_tensor(np.asarray(advantages), dtype=torch.float32)
    w = np.asarray(weights, dtype=np.float64)
    ah = np.asarray(a_hat, dtype=np.float64)
    top = max(1, int(round(0.05 * w.size)))
    summary = {
        "advantage_beta": beta,
        "advantage_clip": clip,
        "advantage_lambda": lam,
        "adv_weight_ess_frac": _ess_frac(w),
        "adv_weight_kl": float((w * np.log(np.clip(w, 1e-12, None))).mean()),
        "adv_weight_max": float(w.max()),
        "adv_clip_frac": float((np.abs(ah) >= clip - 1e-6).mean()),
        "adv_top5_weight_share": float(np.sort(w)[::-1][:top].sum() / w.sum()),
        "adv_batch_rows": int(batch_rows),
    }
    rng = np.random.default_rng(seed)
    ess_batch, max_batch = [], []
    for _ in range(200):
        wb, _ = advantage_weights(a[rng.integers(0, a.numel(), size=batch_rows)], beta, clip, lam)
        wb = wb.numpy().astype(np.float64)
        ess_batch.append(_ess_frac(wb))
        max_batch.append(float(wb.max()))
    summary["adv_weight_ess_frac_batch"] = float(np.median(ess_batch))
    summary["adv_weight_max_batch"] = float(np.median(max_batch))

    sweep_ess, sweep_top = [], []
    for b in _BETA_SWEEP:
        wb, _ = advantage_weights(a, b, clip, lam)
        wb = wb.numpy().astype(np.float64)
        sweep_ess.append(_ess_frac(wb))
        sweep_top.append(float(np.sort(wb)[::-1][:top].sum() / wb.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(ah, bins=50, ax=axes[0], color="seagreen", edgecolor="white")
    _style(axes[0], "Standardized advantage", f"Â = clip((A − mean A) / std A, ±{clip:g})", "Count")
    sns.histplot(w, bins=np.logspace(np.log10(w.min()), np.log10(w.max()), 50), ax=axes[1], color="seagreen", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="black", linestyle=":", linewidth=1)
    _style(axes[1], f"AWR weight (β = {beta:g}, λ = {lam:g})", "w = exp(Â / β) / mean w", "Count")
    axes[2].plot(_BETA_SWEEP, sweep_ess, marker="o", color="steelblue", label="ESS / N")
    axes[2].plot(_BETA_SWEEP, sweep_top, marker="s", color="crimson", label="weight share of the top 5 % frames")
    axes[2].axvline(beta, color="black", linestyle=":", linewidth=1, label=f"configured β = {beta:g}")
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1)
    _style(axes[2], "Beta sweep on the same advantages", "advantage_beta", "fraction")
    axes[2].legend(fontsize=11)
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "advantage_weights.png")
    plt.savefig(out, dpi=200); plt.close()
    logging.info(f"[CRITIC] saved {out}")
    return summary


def _weight_where_plots(weights, seconds_to_end, mistake, quality, output_dir) -> dict:
    """advantage_weight_where.png: mean w by seconds to the segment end, by mistake flag, by quality label."""
    w = np.asarray(weights, dtype=np.float64)
    s = np.asarray(seconds_to_end, dtype=np.float64)
    mistake = np.asarray(mistake, dtype=bool)
    quality = np.asarray(quality)
    edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, np.inf]
    bin_labels = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+" for lo, hi in zip(edges[:-1], edges[1:])]
    bin_masks = [(s >= lo) & (s < hi) for lo, hi in zip(edges[:-1], edges[1:])]

    def annotate_bars(ax, names, masks):
        means = [float(w[m].mean()) if m.any() else float("nan") for m in masks]
        ax.bar(names, [0.0 if np.isnan(v) else v for v in means], color="steelblue")
        for x, (m, v) in enumerate(zip(masks, means)):
            ax.annotate(f"n={int(m.sum())}", (x, 0.0 if np.isnan(v) else v), ha="center", va="bottom", fontsize=9)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        return means

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    annotate_bars(axes[0], bin_labels, bin_masks)
    _style(axes[0], "Mean weight by time to the segment end", "seconds until the segment ends", "mean w")
    axes[0].tick_params(axis="x", labelsize=9)
    gmeans = annotate_bars(axes[1], ["clean", "mistake"], [~mistake, mistake])
    axes[1].patches[1].set_color("crimson")
    _style(axes[1], "Mean weight by mistake flag", "frame label", "mean w")
    levels = sorted(int(q) for q in set(quality.tolist()) if q >= 1)
    qmeans = annotate_bars(axes[2], [str(q) for q in levels], [quality == q for q in levels])
    _style(axes[2], "Mean weight by quality label", "quality (1-5)", "mean w")
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "advantage_weight_where.png")
    plt.savefig(out, dpi=200); plt.close()
    logging.info(f"[CRITIC] saved {out}")

    near = s < 2.0
    summary = {
        "adv_weight_end_ratio": float(w[near].mean() / w[~near].mean()) if near.any() and (~near).any() else float("nan"),
        "adv_weight_mistake_ratio": gmeans[1] / gmeans[0] if mistake.any() and (~mistake).any() else float("nan"),
    }
    for q, m in zip(levels, qmeans):
        summary[f"adv_weight_mean_quality_{q}"] = m
    if len(levels) >= 2:
        summary["adv_weight_quality_slope"] = float(np.polyfit(levels, qmeans, 1)[0])
    return summary


def _pick_percentile_frames(advantages, segment_ids, percentiles=_ADVANTAGE_PERCENTILES, per_cut: int = 2) -> tuple[dict[int, float], list[tuple[str, int]]]:
    """The frames nearest each percentile cut of the advantage, ``per_cut`` per cut, at most one
    per segment and none twice: (cut value per percentile, [(tag, sample index)])."""
    a = np.asarray(advantages, dtype=np.float64)
    cuts = {pct: float(np.percentile(a, pct)) for pct in percentiles}
    used_frames: set[int] = set()
    used_segments: set = set()
    tagged: list[tuple[str, int]] = []
    for pct, cut in cuts.items():
        k = 0
        for i in np.argsort(np.abs(a - cut)):
            i = int(i)
            if i in used_frames or segment_ids[i] in used_segments:
                continue
            k += 1
            tagged.append((f"p{pct:02d}-{k}", i))
            used_frames.add(i)
            used_segments.add(segment_ids[i])
            if k == per_cut:
                break
    return cuts, tagged


def _training_targets(dataset, cfg, chunk_size: int) -> dict:
    """The per-frame markers the training sampler derives rewards and terminals from.

    Mirrors ReplayBuffer.sample: in subtask mode a transition from t is terminal when a
    reviewed segment boundary (offline_dataset_utils._subtask_terminals_from_windows,
    a release folded into its predecessor) or the episode end falls in [t, t + chunk);
    a mistake onset inside that window costs critic_mistake_penalty once. Episode mode
    keeps only the episode ends. ``frames_to_end`` counts frames to the next such
    marker at or after each frame.
    """
    n = len(dataset)
    episode_end = np.zeros(n, dtype=bool)
    for idxs in build_episode_index(dataset).values():
        episode_end[max(idxs)] = True
    labels = frame_metadata_lookup(dataset)
    mistake = np.zeros(n, dtype=bool)
    for idx, row in labels.items():
        mistake[idx] = bool(row.get("mistake", False))
    onset = mistake.copy()
    onset[1:] &= ~mistake[:-1]
    terminals = None
    mode = str(getattr(cfg.policy, "critic_reward_mode", "episode"))
    if mode == "subtask":
        from lerobot.rl.offline_dataset_utils import _subtask_terminals_from_windows

        markers = _subtask_terminals_from_windows(dataset, n)
        if markers is None:
            logging.warning("[CRITIC] subtask reward mode but the dataset has no subtask_windows.json; episode ends only")
            mode = "episode"
        else:
            terminals = markers.numpy().astype(bool)
    boundary = episode_end if terminals is None else (terminals | episode_end)
    next_boundary = np.empty(n, dtype=np.int64)
    last = n - 1
    for i in range(n - 1, -1, -1):
        if boundary[i]:
            last = i
        next_boundary[i] = last
    return {
        "mode": mode,
        "labels": labels,
        "episode_end": episode_end,
        "mistake_onset": onset,
        "terminals": terminals,
        "frames_to_end": next_boundary - np.arange(n),
    }


def _transition_target(targets: dict, dataset, idx: int, chunk_size: int, penalty: float, norm: float, cfg):
    """(normalized reward, done) for the TD step from frame idx, as ReplayBuffer.sample builds it."""
    stop = min(idx + chunk_size, len(dataset))
    episode_end = bool(targets["episode_end"][idx:stop].any())
    if targets["mode"] == "subtask":
        done = episode_end or bool(targets["terminals"][idx:stop].any())
        reward = 0.0 if done else -1.0
        if penalty > 0 and targets["mistake_onset"][idx:stop].any():
            reward -= penalty
        return reward / norm, done
    reward = -1.0
    if episode_end:
        end = idx + int(np.flatnonzero(targets["episode_end"][idx:stop])[0])
        r = dataset.hf_dataset[end].get("next.reward", 0.0)
        r = float(r.item() if isinstance(r, torch.Tensor) else r)
        reward = 0.0 if r > 0.5 else float(getattr(cfg.policy, "terminal_failure_reward", -1.0))
    return reward / norm, episode_end


def _ideal_duration_value(frames_to_end, chunk_size: int, discount: float, norm: float, v_min: float):
    """V the subtask reward rule implies for a frame with no mistake ahead: -1/N on each of
    the m = frames_to_end // chunk non-terminal steps, 0 on the terminal one, discounted,
    floored at the support minimum the TD target is clamped to."""
    steps = np.asarray(frames_to_end, dtype=np.int64) // chunk_size
    value = -(1.0 - discount**steps) / ((1.0 - discount) * norm)
    return np.maximum(value, v_min)


def _pick_outliers(values, ideal, segment_ids, per_side: int = 2) -> list[tuple[str, int]]:
    """The frames furthest below (P) and above (O) the duration ideal, as (tag, sample index),
    at most one per segment so the panel shows different situations, not one segment twice."""
    residual = values - ideal

    def pick(order, sign):
        chosen, seen = [], set()
        for i in order:
            i = int(i)
            if sign * residual[i] <= 0 or segment_ids[i] in seen:
                continue
            chosen.append(i)
            seen.add(segment_ids[i])
            if len(chosen) == per_side:
                break
        return chosen

    below = pick(np.argsort(residual), -1)
    above = pick(np.argsort(residual)[::-1], +1)
    return [(f"P{k + 1}", i) for k, i in enumerate(below)] + [(f"O{k + 1}", i) for k, i in enumerate(above)]


def _frame_to_uint8(value) -> np.ndarray:
    """Dataset image tensor (C,H,W or 1,C,H,W; uint8 or float) -> H,W,C uint8 for imshow."""
    tensor = value[0] if value.ndim == 4 else value
    if tensor.dtype == torch.uint8:
        return tensor.numpy().transpose(1, 2, 0)
    array = tensor.float().numpy().transpose(1, 2, 0)
    if array.max() <= 5.0:
        array = array * 255.0
    return array.clip(0, 255).astype(np.uint8)


def _duration_plots(values, frames_to_end, ideal, quality, mistake, fps: float, output_dir: str, segment_ids=None) -> tuple[dict, list]:
    """value_vs_time_to_end.png and value_by_label.png; returns (numbers behind them, tagged outliers)."""
    seconds = frames_to_end / fps
    summary: dict = {"n_mistake": int(mistake.sum()), "n_clean": int((~mistake).sum())}
    residual = values - ideal if ideal is not None else values
    outliers: list[tuple[str, int]] = []

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(seconds[~mistake], values[~mistake], s=18, alpha=0.6, color="steelblue", label="clean frame")
    if mistake.any():
        ax.scatter(seconds[mistake], values[mistake], s=22, alpha=0.8, color="crimson", label="mistake window")
    if ideal is not None:
        order = np.argsort(seconds)
        ax.plot(seconds[order], ideal[order], color="black", linewidth=2, label="duration-only ideal $V^*$")
        summary["value_fit_abs_err"] = float(np.abs(values - ideal).mean())
        summary["value_fit_corr"] = (
            float(np.corrcoef(values, ideal)[0, 1]) if values.std() > 0 and ideal.std() > 0 else float("nan")
        )
        # Tag the frames furthest from the ideal on both sides; value_outliers.png shows them.
        ids = np.arange(len(values)) if segment_ids is None else np.asarray(segment_ids)
        outliers = _pick_outliers(values, ideal, ids)
        for k, (tag, i) in enumerate(outliers):
            ax.scatter([seconds[i]], [values[i]], s=160, facecolors="none", edgecolors="black", linewidths=1.8)
            ax.annotate(tag, (seconds[i], values[i]), textcoords="offset points",
                        xytext=(8, 8 if k % 2 == 0 else -16), fontsize=11, fontweight="bold")
    _style(ax, "V(s) against time to the segment end", "seconds until the segment (or episode) ends", "V(s)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = os.path.join(output_dir, "value_vs_time_to_end.png")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    logging.info(f"[CRITIC] saved {out}")

    ylabel = "V - V*" if ideal is not None else "V"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    levels = sorted(int(q) for q in set(quality.tolist()) if q >= 1)
    means = [float(residual[quality == q].mean()) for q in levels]
    counts = [int((quality == q).sum()) for q in levels]
    axes[0].bar([str(q) for q in levels], means, color="steelblue")
    for x, (m, c) in enumerate(zip(means, counts)):
        axes[0].annotate(f"n={c}", (x, m), ha="center", va="bottom" if m >= 0 else "top", fontsize=10)
    _style(axes[0], f"mean {ylabel} by quality label", "quality (1-5)", f"mean {ylabel}")
    for q, m in zip(levels, means):
        summary[f"residual_mean_quality_{q}"] = m
    if len(levels) >= 2:
        summary["residual_quality_slope"] = float(np.polyfit(levels, means, 1)[0])
    groups = [("clean", ~mistake), ("mistake", mistake)]
    gmeans = [float(residual[mask].mean()) if mask.any() else float("nan") for _, mask in groups]
    axes[1].bar([name for name, _ in groups], gmeans, color=["steelblue", "crimson"])
    for x, ((_name, mask), m) in enumerate(zip(groups, gmeans)):
        y = 0.0 if np.isnan(m) else m
        axes[1].annotate(f"n={int(mask.sum())}", (x, y), ha="center", va="bottom" if y >= 0 else "top", fontsize=10)
    _style(axes[1], f"mean {ylabel} by mistake flag", "frame label", f"mean {ylabel}")
    summary["residual_mean_clean"], summary["residual_mean_mistake"] = gmeans
    summary["residual_mistake_gap"] = gmeans[1] - gmeans[0]
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "value_by_label.png")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    logging.info(f"[CRITIC] saved {out}")
    return summary, outliers


def _render_tagged_frames(dataset, cfg, chunk_size: int, tagged, meta: list[dict], text_fn, out_path: str):
    """One row per tagged frame: its cameras plus what ``text_fn(tag, i)`` says about it."""
    rows = []
    for tag, i in tagged:
        obs, _, _, _, _, _, _ = get_frame_data(dataset, meta[i]["global_idx"], chunk_size)
        obs = canonical_camera_obs(obs, cfg)
        images = [(key.split(".")[-1], _frame_to_uint8(obs[key])) for key in sorted(obs) if "images" in key]
        rows.append((tag, i, images))
    if not rows:
        return
    n_cams = max(len(r[2]) for r in rows)
    fig, axes = plt.subplots(len(rows), n_cams + 1, figsize=(4.2 * (n_cams + 1), 3.3 * len(rows)), squeeze=False)
    for r, (tag, i, images) in enumerate(rows):
        for c in range(n_cams):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(images):
                ax.imshow(images[c][1])
                ax.set_title(f"{tag}  {images[c][0]}" if c == 0 else images[c][0], fontsize=11, fontweight="bold" if c == 0 else None)
        ax = axes[r, n_cams]
        ax.axis("off")
        ax.text(0.0, 0.95, text_fn(tag, i), transform=ax.transAxes, fontsize=11, va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="#ccc", boxstyle="round,pad=0.6"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"[CRITIC] saved {out_path}")


def _gradient_plots(grad_mags, subtasks, episodes, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.boxplot(x="grad_mag", y="subtask", hue="subtask", legend=False,
                data={"grad_mag": grad_mags, "subtask": subtasks},
                ax=axes[0], palette="pastel", fliersize=0)
    sns.stripplot(x="grad_mag", y="subtask",
                  data={"grad_mag": grad_mags, "subtask": subtasks},
                  ax=axes[0], color=".3", size=4, alpha=0.5, jitter=True)
    _style(axes[0], "Gradient Magnitude by Subtask", "Magnitude (L2 Norm)", "Subtask")
    sns.scatterplot(x=range(len(grad_mags)), y=grad_mags, hue=episodes,
                    palette="viridis", s=80, alpha=0.8,
                    edgecolor="white", ax=axes[1])
    _style(axes[1], "Gradient Magnitude per Frame (colored by Episode)",
           "Sample Index", "Magnitude (L2 Norm)")
    axes[1].legend(title="Episode Index", bbox_to_anchor=(1.05, 1),
                   loc="upper left", frameon=True)
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "gradient_magnitudes.png")
    plt.savefig(out, dpi=200); plt.close()
    logging.info(f"[CRITIC] saved {out}")


_PERCENTILE_EXEMPLARS = (1, 10, 25, 50, 75, 90, 99)


def _write_manifest(output_dir: str, raw: dict, extra: dict | None = None) -> dict:
    """Describe the critic's value, advantage and weight distributions to the viewer."""
    td = raw["td_errors"].float()
    summary = {
        "n_frames": int(td.numel()),
        "td_error_mean": float(td.mean()),
        "td_error_abs_mean": float(td.abs().mean()),
        "td_error_std": float(td.std()),
    }
    if "grad_mags" in raw:
        grads = raw["grad_mags"].float()
        summary["grad_mag_mean"] = float(grads.mean())
        summary["grad_mag_max"] = float(grads.max())
    summary.update(extra or {})

    panels = [
        Panel(
            "predicted_distributions.png",
            "Per-frame $P(V)$ curves with $\\mathbb{E}[V]$ overlaid",
            how="One curve per sampled frame over the value support. Spiky under-fit distributions mean the HL-Gauss $\\sigma$ is too narrow; a flat $\\mathbb{E}[V]$ across frames early in training is by design, not collapse.",
            primary=True,
        ),
        Panel(
            "advantage_dist.png",
            "Advantage (TD error) histogram with the percentile cuts, its CDF, the $V(s)$ histogram, and the by-subtask spread",
            how="The advantage is $r + \\gamma V(s') (1-d) - V(s)$ with the training sampler's own reward rule (subtask mode: $-1/N$ per step, $0$ once a segment or episode end falls inside the chunk, the mistake penalty at an onset) and $V(s')$ from the same critic, as the actor run computes it. A mean far from zero is a systematic value bias, not noise. The dotted cuts are the percentiles shown in advantage_percentiles.png.",
            primary=True,
        ),
        Panel(
            "advantage_weights.png",
            "The AWR weights the actor is trained with: standardized advantage, weight histogram, beta sweep",
            how="Same formula as the trainer (lerobot.rl.advantage): $\\hat A = \\mathrm{clip}((A - \\bar A)/\\sigma_A, \\pm c)$, $w = e^{\\hat A/\\beta} / \\bar w$, then the $\\lambda$ BC mix, with the statistics pooled over every sampled frame. A weight histogram with a long right tail on the log axis means a few frames carry the update; the sweep shows how ESS and the top-5 % share move with $\\beta$ on these same advantages.",
            primary=True,
        ),
        Panel(
            "advantage_weight_where.png",
            "Mean weight by seconds to the segment end, by mistake flag, by quality label",
            how="Where the actor's gradient goes under the weighting. Mass in the last 2 s of the segments means AWR is up-weighting the terminal steps the critic under-predicts (a duration effect), not good behaviour; the mistake bar should sit below the clean bar and the quality bars should rise.",
            primary=True,
        ),
        Panel(
            "advantage_percentiles.png",
            "Camera frames nearest the p5 / p10 / p50 / p90 / p95 advantage cuts, two per cut from different segments",
            how="What a low-, median- and high-advantage frame looks like, away from the extremes so they are typical of their tail. Each row states $A$, $\\hat A$, the weight $w$, $V$ and $V^*$.",
            primary=True,
        ),
        Panel(
            "value_vs_time_to_end.png",
            "$V(s)$ against seconds until the segment ends, with the duration-only ideal $V^*$",
            how="Under the subtask reward rule the critic is a clock: $V^*(s) = -(1-\\gamma^m)/((1-\\gamma)N)$ for the $m$ chunks left in the segment. Points hugging the black curve mean the critic has learnt the duration; red points (mistake windows) should sit below it by about the mistake penalty.",
            primary=True,
        ),
        Panel(
            "value_outliers.png",
            "The frames furthest from the duration ideal: P = most pessimistic, O = most optimistic",
            how="Tagged on the scatter with the same letters. A pessimistic frame mid-segment with the object plainly in hand says the critic is not reading progress off the image; an optimistic one right after a segment start says it took the scene for a finished step.",
        ),
        Panel(
            "value_by_label.png",
            "Mean $V - V^*$ by quality label and by mistake flag",
            how="Duration is removed, so what is left is what the critic reads off the labels. A negative mistake gap and a rising trend over quality mean the critic agrees with the reviewers; flat bars mean it ignores the clause.",
        ),
        Panel(
            "gradient_magnitudes.png",
            "$\\|\\partial V / \\partial \\text{vision}\\|$ across frames, by subtask and episode",
            how="Which observations move the value estimate. Near-zero everywhere means the critic is reading the state vector and ignoring the cameras.",
        ),
    ]
    panels += [
        Panel(
            f"frame_p{percentile:02d}.png",
            f"p{percentile} gradient-magnitude exemplar frame",
            how="The scene at this percentile of value sensitivity. Compare the p01 and p99 frames: if they are indistinguishable, the gradient ranking is noise.",
        )
        for percentile in _PERCENTILE_EXEMPLARS
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Critic Values",
        group="Critic",
        claim="What values does the critic assign, and where do its advantage weights put the actor's gradient?",
        summary=summary,
        metrics=[
            Metric(
                "td_error_abs_mean",
                "Mean |TD error|",
                good="low",
                fmt=4,
                primary=True,
                note="Averaged over sampled frames. No threshold: the scale depends on the reward normalization constant and the value support, so read it against its own history across checkpoints.",
            ),
            Metric(
                "td_error_mean",
                "TD error mean (signed)",
                good="none",
                fmt=4,
                baseline=0.0,
                primary=True,
                note="Away from zero the critic is systematically optimistic or pessimistic, which the absolute mean above cannot show.",
            ),
            Metric("td_error_std", "TD error spread", good="none", fmt=4),
            Metric(
                "adv_weight_ess_frac",
                "ESS / N of the AWR weights",
                good="high",
                fmt=3,
                primary=True,
                note="Pooled over the sampled frames with the configured beta/clip/lambda. A Gaussian standardized advantage at beta 1 gives about 0.37; far below that a heavy tail is carrying the update.",
            ),
            Metric(
                "adv_top5_weight_share",
                "Weight share of the top 5 % frames",
                good="low",
                fmt=3,
                primary=True,
                note="Uniform weights give 0.05; a Gaussian standardized advantage at beta 1 gives about 0.26. Higher means the update is a handful of frames.",
            ),
            Metric(
                "adv_weight_end_ratio",
                "Mean w within 2 s of the segment end / elsewhere",
                good="none",
                fmt=3,
                baseline=1.0,
                primary=True,
                note="Above 1 the weighting favours the terminal steps, which is the critic's duration error, not behaviour quality.",
            ),
            Metric(
                "adv_weight_mistake_ratio",
                "Mean w on mistake windows / clean frames",
                good="low",
                fmt=3,
                baseline=1.0,
                primary=True,
                note="The weighting should put this below 1: mistake frames get less of the actor's gradient.",
            ),
            Metric("adv_weight_quality_slope", "Slope of mean w over quality 1..5", good="high", fmt=4, baseline=0.0),
            Metric("adv_weight_kl", "KL(weighted ‖ BC) = mean w log w", good="none", fmt=3),
            Metric("adv_weight_max", "Largest weight", good="none", fmt=2),
            Metric(
                "adv_clip_frac",
                "Fraction of frames at the clip",
                good="low",
                fmt=3,
                note="Share with |Â| at advantage_clip. A Gaussian at clip 3 gives 0.003; more means the tail is wider than the clip assumes.",
            ),
            Metric(
                "adv_weight_ess_frac_batch",
                "ESS / N per effective batch (median)",
                good="high",
                fmt=3,
                note="Median over 200 resampled batches of adv_batch_rows frames, the set the trainer standardizes over, so it matches the adv_weight_ess_frac the console prints.",
            ),
            Metric("adv_weight_max_batch", "Largest weight per effective batch (median)", good="none", fmt=2),
            Metric(
                "value_fit_abs_err",
                "Mean |V − V*| (duration fit)",
                good="low",
                fmt=4,
                primary=True,
                note="Subtask reward mode only: the gap between the critic and the value its own reward rule implies for a mistake-free segment.",
            ),
            Metric("value_fit_corr", "corr(V, V*)", good="high", fmt=3, warn=0.5),
            Metric(
                "residual_mistake_gap",
                "Mean (V − V*): mistake − clean",
                good="none",
                fmt=4,
                baseline=0.0,
                note="Negative when the critic values mistake windows below what duration alone predicts.",
            ),
            Metric("residual_quality_slope", "Slope of mean (V − V*) over quality 1..5", good="none", fmt=4, baseline=0.0),
            Metric("grad_mag_mean", "Mean value-gradient magnitude", good="none", fmt=5),
            Metric("n_frames", "Frames sampled", good="none", fmt=0),
        ],
        panels=panels,
        see_also=["action_trace"],
    )


def run_critic_values_distribution(
    adapter: ProbablePolicy, val_dataset, val_ep_indices,
    cfg, output_dir: str,
):
    """Advantage / AWR-weight distributions, percentile frames + (optional) gradient exemplars."""
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(output_dir, exist_ok=True)

    p = cfg.probe_parameters
    chunk_size = adapter.chunk_size
    n_adv = int(getattr(p, "critic_adv_frames", 1000))
    n_grad = int(getattr(p, "critic_grad_frames", 200))
    seed = int(getattr(p, "random_seed", 42))
    discount = float(getattr(adapter.policy.config, "discount", 0.99))
    beta = float(getattr(cfg.policy, "advantage_beta", 1.0))
    clip = float(getattr(cfg.policy, "advantage_clip", 3.0))
    lam = float(getattr(cfg.policy, "advantage_lambda", 1.0))
    batch_rows = int(cfg.batch_size) * int(getattr(cfg.policy, "gradient_accumulation_steps", 1))

    # ── Part 0: per-episode V(s) traces + overlay video ──────────────────────
    try:
        run_episode_critic_traces(
            adapter, val_dataset, val_ep_indices, cfg,
            output_dir=os.path.join(output_dir, "episode_traces"),
        )
    except Exception as exc:
        logging.warning(f"[CRITIC] episode_traces failed: {exc}", exc_info=True)

    # ── Part 0.5: predicted value distributions for random frames ────────────
    try:
        run_predicted_distributions(adapter, val_dataset, val_ep_indices, cfg, output_dir)
    except Exception as exc:
        logging.warning(f"[CRITIC] predicted_distributions failed: {exc}", exc_info=True)

    # ── Part 1: TD-error under the training sampler's reward rule ────────────
    # s' = s_{t + chunk_size}; done when a segment boundary (subtask mode) or the
    # episode end falls in [t, t + chunk); reward -1 per step, 0 on the terminal
    # step, minus critic_mistake_penalty at a mistake onset, over the normalization
    # constant; target clamped into the value support. The transition
    # ReplayBuffer.sample builds for update_critic.
    logging.info(f"[CRITIC] sampling {n_adv} frames for the advantage")
    targets = _training_targets(val_dataset, cfg, chunk_size)
    labels = targets["labels"]
    norm = float(getattr(cfg.policy, "reward_normalization_constant", 1.0))
    penalty = float(getattr(cfg.policy, "critic_mistake_penalty", 0.0))
    v_min = float(getattr(cfg.policy, "value_support_min", -2.0))
    v_max = float(getattr(cfg.policy, "value_support_max", 0.0))
    adv_indices = get_random_valid_samples(
        val_dataset, n_adv, seed,
        val_ep_indices=val_ep_indices, lookahead_frames=chunk_size,
        stride=int(getattr(cfg.policy, "image_stride", 1)),
    )
    if not adv_indices:
        logging.warning("[CRITIC] no advantage samples")
        return None

    td_errors: list[float] = []
    adv_subtasks: list[str] = []
    values: list[float] = []
    frames_to_end: list[int] = []
    qualities: list[int] = []
    mistakes: list[bool] = []
    sample_meta: list[dict] = []
    for idx in adv_indices:
        metadata = labels.get(idx)
        fr_c = probe_frame_inputs(val_dataset, cfg, idx, chunk_size, metadata=metadata)
        # s' rides s's subtask and labels, as _critic_batches does: the sampler has no
        # next-state columns, and a boundary inside the chunk makes the step terminal.
        fr_n = probe_frame_inputs(val_dataset, cfg, idx + chunk_size, chunk_size, metadata=metadata)
        obs, gt_subtask, task_str = fr_c["obs"], fr_c["subtask"], fr_c["task"]
        v_curr = adapter.predict_value(obs, task_str, gt_subtask, metadata=fr_c["metadata"])
        v_next = adapter.predict_value(fr_n["obs"], task_str, gt_subtask, metadata=fr_c["metadata"])
        reward, done = _transition_target(targets, val_dataset, idx, chunk_size, penalty, norm, cfg)
        target_v = float(np.clip(reward + discount * v_next * (1.0 - float(done)), v_min, v_max))
        td = target_v - v_curr
        td_errors.append(td)
        adv_subtasks.append(gt_subtask or "None")
        values.append(v_curr)
        frames_to_end.append(int(targets["frames_to_end"][idx]))
        qualities.append(int(metadata["quality"]) if metadata and "quality" in metadata else -1)
        mistakes.append(bool(metadata["mistake"]) if metadata and "mistake" in metadata else False)
        sample_meta.append(
            {
                "global_idx": int(idx),
                "episode_idx": int(fr_c["episode_idx"]),
                "frame_idx": int(fr_c["frame_idx"]),
                "subtask": gt_subtask or "",
                "quality": qualities[-1],
                "mistake": mistakes[-1],
            }
        )

    # ── Part 1a: advantage distribution, the AWR weights it gives, percentile frames ──
    td_arr = np.asarray(td_errors, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    fte = np.asarray(frames_to_end, dtype=np.int64)
    seconds = fte / float(cfg.env.fps)
    qualities_arr = np.asarray(qualities)
    mistakes_arr = np.asarray(mistakes, dtype=bool)
    # One frame per critic segment: the segment is identified by its terminal frame.
    segment_ids = np.asarray(adv_indices, dtype=np.int64) + fte
    cuts, percentile_frames = _pick_percentile_frames(td_arr, segment_ids)
    _advantage_plots(td_errors, values, adv_subtasks, cuts, output_dir)
    w_t, a_hat_t = advantage_weights(torch.tensor(td_errors, dtype=torch.float32), beta, clip, lam)
    weights = w_t.numpy()
    a_hat = a_hat_t.numpy()
    weight_summary = _weight_plots(td_arr, a_hat, weights, beta, clip, lam, batch_rows, seed, output_dir)
    weight_summary.update(_weight_where_plots(weights, seconds, mistakes_arr, qualities_arr, output_dir))

    # ── Part 1b: the duration critic against what its own reward rule implies ──
    ideal = (
        _ideal_duration_value(fte, chunk_size, discount, norm, v_min) if targets["mode"] == "subtask" else None
    )
    duration_summary, outliers = _duration_plots(
        values_arr, fte, ideal, qualities_arr, mistakes_arr, float(cfg.env.fps), output_dir,
        segment_ids=segment_ids,
    )
    duration_summary["critic_reward_mode"] = targets["mode"]
    duration_summary.update(weight_summary)

    def frame_text(i: int) -> str:
        m = sample_meta[i]
        return (
            f"episode {m['episode_idx']}, frame {m['frame_idx']}\n"
            f"{textwrap.fill(m['subtask'] or '(no subtask)', 30)}\n"
            f"{seconds[i]:.1f} s to segment end\n"
            f"quality {m['quality'] if m['quality'] >= 0 else '-'}, "
            f"{'mistake window' if m['mistake'] else 'clean'}"
        )

    def value_line(i: int) -> str:
        return f"V = {values_arr[i]:.2f}" + ("" if ideal is None else f"    V* = {ideal[i]:.2f}")

    if outliers:
        _render_tagged_frames(
            val_dataset, cfg, chunk_size, outliers, sample_meta,
            lambda tag, i: f"{tag}: {'pessimistic' if tag.startswith('P') else 'optimistic'}\n{frame_text(i)}\n{value_line(i)}",
            os.path.join(output_dir, "value_outliers.png"),
        )
    _render_tagged_frames(
        val_dataset, cfg, chunk_size, percentile_frames, sample_meta,
        lambda tag, i: (
            f"{tag}: advantage percentile {int(tag[1:3])} (cut {cuts[int(tag[1:3])]:+.3f})\n"
            f"A = {td_arr[i]:+.3f}   Â = {a_hat[i]:+.2f}   w = {weights[i]:.2f}\n"
            f"{value_line(i)}\n{frame_text(i)}"
        ),
        os.path.join(output_dir, "advantage_percentiles.png"),
    )

    # ── Part 2: gradient magnitudes (skip if adapter doesn't support it) ─────
    try:
        _probe_value_grad = adapter.value_gradient_magnitude  # type: ignore[attr-defined]
    except AttributeError:
        _probe_value_grad = None

    # Touch-test that gradient mag works on this adapter before sampling many.
    grad_supported = True
    if _probe_value_grad is not None and adv_indices:
        try:
            test_obs, _, _, _, test_task, _, _ = get_frame_data(val_dataset, adv_indices[0], chunk_size)
            adapter.value_gradient_magnitude(test_obs, test_task)
        except NotImplementedError:
            grad_supported = False
            logging.info("[CRITIC] adapter does not support value_gradient_magnitude; skipping.")
        except Exception as exc:
            grad_supported = False
            logging.warning(f"[CRITIC] gradient probe touch-test failed: {exc}", exc_info=True)

    raw: dict = {
        "td_errors": torch.tensor(td_errors),
        "advantages_standardized": torch.tensor(a_hat),
        "advantage_weights": torch.tensor(weights),
        "adv_subtasks": adv_subtasks,
        "values": torch.tensor(values),
        "frames_to_end": torch.tensor(frames_to_end),
        "ideal_values": None if ideal is None else torch.tensor(ideal),
        "qualities": torch.tensor(qualities),
        "mistakes": torch.tensor(mistakes),
    }

    if grad_supported and _probe_value_grad is not None:
        logging.info(f"[CRITIC] sampling {n_grad} frames for gradients")
        grad_indices = get_random_valid_samples(
            val_dataset, n_grad, seed + 1,
            val_ep_indices=val_ep_indices,
            stride=int(getattr(cfg.policy, "image_stride", 1)),
        )
        grad_mags, episodes, subtasks, frames = [], [], [], []
        frame_cache: dict[int, dict] = {}
        for idx in grad_indices:
            _fr = probe_frame_inputs(val_dataset, cfg, idx, chunk_size, metadata=None)
            obs, gt_subtask, task_str = _fr["obs"], _fr["subtask"], _fr["task"]
            ep_idx, fr_idx = _fr["episode_idx"], _fr["frame_idx"]
            frame_cache[idx] = {k: v.clone() for k, v in obs.items() if "image" in k}
            try:
                mag = adapter.value_gradient_magnitude(obs, task_str)
            except Exception as exc:
                logging.warning(f"[CRITIC] grad mag failed at idx={idx}: {exc}")
                mag = 0.0
            grad_mags.append(mag)
            episodes.append(ep_idx); subtasks.append(gt_subtask or "None"); frames.append(fr_idx)

        if grad_mags:
            _gradient_plots(grad_mags, subtasks, episodes, output_dir)

            # Percentile exemplar frames
            percentiles = [1, 10, 25, 50, 75, 90, 99]
            grad_arr = np.array(grad_mags)
            for pct in percentiles:
                val = float(np.percentile(grad_arr, pct))
                closest = int(np.argmin(np.abs(grad_arr - val)))
                ds_idx = grad_indices[closest]
                if ds_idx in frame_cache:
                    _render_percentile_frame(
                        frame_cache[ds_idx], episodes[closest], frames[closest],
                        subtasks[closest], float(grad_arr[closest]), pct, output_dir,
                    )

            raw.update({
                "grad_mags": torch.tensor(grad_mags),
                "grad_episodes": torch.tensor(episodes),
                "grad_frames": torch.tensor(frames),
                "grad_subtasks": subtasks,
            })

    _write_manifest(output_dir, raw, duration_summary)
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────────────────

def run(adapter, dataset, cfg, output_dir, val_ep_indices=None):
    """Uniform run() entry — thin wrapper over :func:`run_critic_values_distribution`
    matching the other probes' signature for rl_offline dispatch."""
    if adapter is None or dataset is None:
        return
    if not hasattr(adapter.policy, "critic"):
        logging.warning(
            f"[CRITIC] policy {type(adapter.policy).__name__} has no .critic; skipping."
        )
        return
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(output_dir, exist_ok=True)
    run_critic_values_distribution(
        adapter, val_dataset=dataset, val_ep_indices=val_ep_indices,
        cfg=cfg, output_dir=output_dir,
    )


@parser.wrap()
def probe_cli(cfg: ProbeCriticConfig):
    init_logging()
    sns.set_theme(style="whitegrid", palette="muted")

    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "critic")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    dataset = load_probe_dataset(cfg)

    logging.info("Loading policy adapter …")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    if not hasattr(adapter.policy, "critic"):
        raise ValueError(
            f"Policy of type {type(adapter.policy).__name__} has no .critic attribute. "
            "Did you load an RL policy variant (e.g. pi05_rl, molmoact2_rl)?"
        )

    run_critic_values_distribution(
        adapter, val_dataset=dataset, val_ep_indices=None,
        cfg=cfg, output_dir=output_dir,
    )


if __name__ == "__main__":
    register_config_choices()
    probe_cli()
