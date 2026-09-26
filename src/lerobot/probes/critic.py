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
  advantage_dist.png            TD-error histogram + CDF + by-subtask boxplot
  advantage_squashed_dist.png   tanh(TD-error / scaling) version of the above
  value_vs_time_to_end.png      V(s) against seconds to the segment end, with the
                                duration-only ideal V* the reward rule implies
  value_by_label.png            mean V - V* by quality label and by mistake flag
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
from lerobot.probes.utils import (
    build_episode_index,
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
    """For each selected episode: save per-frame PNGs, run critic at a fixed
    stride, save the critic curve + JSON, and render a critic-overlay video.

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
    seed = int(getattr(p, "random_seed", 42))
    max_episodes = getattr(p, "max_episodes", None)
    video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])
    fps = cfg.env.fps

    labels = frame_metadata_lookup(val_dataset)
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

        # ── 1. Save per-frame PNGs + collect subtask labels for overlay ──────
        subtask_texts: list[str] = []
        for step_idx, global_idx in enumerate(indices if with_video else ()):
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
                PILImage.fromarray(img_np).save(
                    os.path.join(ep_dir, f"step_{step_idx:06d}_{cam_name}.png")
                )

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
        trace_seconds = [indices[ci] - indices[0] for ci in critic_indices]
        trace_seconds = [frames / float(fps) for frames in trace_seconds]
        with open(os.path.join(ep_dir, "critic_values.json"), "w") as f:
            json.dump({"seconds": trace_seconds, "values": critic_values, "stride_frames": step}, f)
        if critic_values:
            plt.figure(figsize=(10, 5))
            plt.plot(trace_seconds, critic_values, marker=".")
            plt.title(f"Critic Values - Episode {ep_idx}")
            plt.xlabel("seconds into the episode")
            plt.ylabel("V(s)")
            plt.grid(True)
            plt.savefig(os.path.join(ep_dir, "critic_plot.png"))
            plt.close()

        if not with_video:
            continue

        # ── 4. Overlay video ─────────────────────────────────────────────────
        try:
            save_video_with_critic_overlay(
                ep_dir, critic_values,
                camera_names=video_logging_cameras,
                fps=fps,
                subtask_texts=subtask_texts,
                subsample=step,
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

    labels = frame_metadata_lookup(val_dataset)
    ep_to_indices = build_episode_index(val_dataset)
    ep_last_frame = {
        ep: max(val_dataset.hf_dataset[i]["frame_index"].item() for i in idxs)
        for ep, idxs in ep_to_indices.items()
    }

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
        frames_to_end = ep_last_frame[ep_idx] - fr_idx

        ax = axes[i // n_cols, i % n_cols]
        ax.plot(bin_centers, probs, color="steelblue", linewidth=2)
        ax.fill_between(bin_centers, probs, alpha=0.2, color="steelblue")
        ax.axvline(v, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"E[V] = {v:.3f}")
        title = f"ep{ep_idx} f{fr_idx}  ({frames_to_end} to end)"
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


def _td_error_plots(td_errors, squashed, subtasks, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(td_errors, bins=50, kde=True, ax=axes[0],
                 color="coral", edgecolor="white")
    _style(axes[0], "TD-Error (Advantage) Histogram", "TD-Error", "Count")
    sns.ecdfplot(td_errors, ax=axes[1], color="coral", linewidth=3)
    _style(axes[1], "TD-Error (Advantage) CDF", "TD-Error", "Cumulative Probability")
    axes[1].margins(y=0.05)
    sns.boxplot(x="td_error", y="subtask", hue="subtask", legend=False,
                data={"td_error": td_errors, "subtask": subtasks},
                ax=axes[2], palette="pastel", fliersize=0)
    sns.stripplot(x="td_error", y="subtask",
                  data={"td_error": td_errors, "subtask": subtasks},
                  ax=axes[2], color=".3", size=3, alpha=0.5, jitter=True)
    _style(axes[2], "TD-Error by Subtask", "TD-Error", "Subtask")
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "advantage_dist.png")
    plt.savefig(out, dpi=200); plt.close()
    logging.info(f"[CRITIC] saved {out}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(squashed, bins=50, kde=True, ax=axes[0],
                 color="seagreen", edgecolor="white")
    _style(axes[0], "Squashed Advantage Histogram", "tanh(adv / scale)", "Count")
    sns.ecdfplot(squashed, ax=axes[1], color="seagreen", linewidth=3)
    _style(axes[1], "Squashed Advantage CDF", "tanh(adv / scale)", "Cumulative Probability")
    axes[1].margins(y=0.05)
    sns.boxplot(x="squashed", y="subtask", hue="subtask", legend=False,
                data={"squashed": squashed, "subtask": subtasks},
                ax=axes[2], palette="pastel", fliersize=0)
    sns.stripplot(x="squashed", y="subtask",
                  data={"squashed": squashed, "subtask": subtasks},
                  ax=axes[2], color=".3", size=3, alpha=0.5, jitter=True)
    _style(axes[2], "Squashed Advantage by Subtask", "tanh(adv / scale)", "Subtask")
    plt.tight_layout(pad=3.0)
    out = os.path.join(output_dir, "advantage_squashed_dist.png")
    plt.savefig(out, dpi=200); plt.close()
    logging.info(f"[CRITIC] saved {out}")


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


def _duration_plots(values, frames_to_end, ideal, quality, mistake, fps: float, output_dir: str) -> dict:
    """value_vs_time_to_end.png and value_by_label.png; returns the numbers behind them."""
    seconds = frames_to_end / fps
    summary: dict = {"n_mistake": int(mistake.sum()), "n_clean": int((~mistake).sum())}
    residual = values - ideal if ideal is not None else values

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
    return summary


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


def _write_manifest(output_dir: str, raw: dict, advantage_scaling: float, extra: dict | None = None) -> dict:
    """Describe the critic's value and TD-error distributions to the viewer."""
    td = raw["td_errors"].float()
    squashed = raw["squashed_advantages"].float()
    summary = {
        "n_frames": int(td.numel()),
        "td_error_mean": float(td.mean()),
        "td_error_abs_mean": float(td.abs().mean()),
        "td_error_std": float(td.std()),
        "advantage_scaling": advantage_scaling,
        "squashed_abs_mean": float(squashed.abs().mean()),
        "squashed_saturated_fraction": float((squashed.abs() > 0.99).float().mean()),
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
            "TD-error histogram, CDF, and by-subtask spread",
            how="The TD error is $r + \\gamma V(s') (1-d) - V(s)$ with the training sampler's own reward rule (subtask mode: $-1/N$ per step, $0$ once a segment or episode end falls inside the chunk, the mistake penalty at an onset). A mean far from zero is a systematic value bias, not noise.",
            primary=True,
        ),
        Panel(
            "value_vs_time_to_end.png",
            "$V(s)$ against seconds until the segment ends, with the duration-only ideal $V^*$",
            how="Under the subtask reward rule the critic is a clock: $V^*(s) = -(1-\\gamma^m)/((1-\\gamma)N)$ for the $m$ chunks left in the segment. Points hugging the black curve mean the critic has learnt the duration; red points (mistake windows) should sit below it by about the mistake penalty.",
            primary=True,
        ),
        Panel(
            "value_by_label.png",
            "Mean $V - V^*$ by quality label and by mistake flag",
            how="Duration is removed, so what is left is what the critic reads off the labels. A negative mistake gap and a rising trend over quality mean the critic agrees with the reviewers; flat bars mean it ignores the clause.",
        ),
        Panel(
            "advantage_squashed_dist.png",
            "The same after $\\tanh(\\delta / \\text{advantage\\_scaling})$",
            how="This is the form the policy is actually conditioned on. Mass piled at $\\pm 1$ means the scaling is too small and the conditioning has collapsed to a sign bit.",
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
        claim="What values does the critic assign, and is its TD error centred and unsaturated?",
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
            Metric(
                "squashed_saturated_fraction",
                "Fraction of advantages at $\\pm 1$",
                good="low",
                fmt=3,
                warn=0.1,
                note="Share of squashed advantages with $|\\tanh| > 0.99$. High means ``advantage_scaling`` is too small for this TD-error scale and the conditioning signal has collapsed to a sign.",
            ),
            Metric("td_error_std", "TD error spread", good="none", fmt=4),
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
    """TD-error / advantage distributions + (optional) gradient exemplars."""
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(output_dir, exist_ok=True)

    p = cfg.probe_parameters
    chunk_size = adapter.chunk_size
    n_adv = int(getattr(p, "critic_adv_frames", 1000))
    n_grad = int(getattr(p, "critic_grad_frames", 200))
    seed = int(getattr(p, "random_seed", 42))
    discount = float(getattr(adapter.policy.config, "discount", 0.99))
    advantage_scaling = float(getattr(cfg.policy, "advantage_scaling", 1.0))

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
    logging.info(f"[CRITIC] sampling {n_adv} frames for TD-error")
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
    squashed: list[float] = []
    adv_subtasks: list[str] = []
    values: list[float] = []
    frames_to_end: list[int] = []
    qualities: list[int] = []
    mistakes: list[bool] = []
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
        squashed.append(float(np.tanh(td / advantage_scaling)))
        adv_subtasks.append(gt_subtask or "None")
        values.append(v_curr)
        frames_to_end.append(int(targets["frames_to_end"][idx]))
        qualities.append(int(metadata["quality"]) if metadata and "quality" in metadata else -1)
        mistakes.append(bool(metadata["mistake"]) if metadata and "mistake" in metadata else False)

    _td_error_plots(td_errors, squashed, adv_subtasks, output_dir)

    # ── Part 1b: the duration critic against what its own reward rule implies ──
    values_arr = np.asarray(values, dtype=np.float64)
    fte = np.asarray(frames_to_end, dtype=np.int64)
    ideal = (
        _ideal_duration_value(fte, chunk_size, discount, norm, v_min) if targets["mode"] == "subtask" else None
    )
    duration_summary = _duration_plots(
        values_arr, fte, ideal, np.asarray(qualities), np.asarray(mistakes, dtype=bool), float(cfg.env.fps), output_dir
    )
    duration_summary["critic_reward_mode"] = targets["mode"]

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
        "squashed_advantages": torch.tensor(squashed),
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

    _write_manifest(output_dir, raw, advantage_scaling, duration_summary)
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
