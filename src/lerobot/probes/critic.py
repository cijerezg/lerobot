#!/usr/bin/env python
"""
Generic critic probe — visualise the distributional critic's value estimates.

Policy-agnostic: works with any policy whose ``ProbablePolicy`` adapter
implements ``predict_value`` and ``predict_value_and_probs``. Adapters that
also implement ``value_gradient_magnitude`` get the gradient-based plots and
percentile-exemplar frames; otherwise those sections are skipped.

Outputs (under ``probe_parameters.output_dir/critic/``):
  predicted_distributions.png   per-frame P(V) curves with E[V] overlay
  advantage_dist.png            TD-error histogram + CDF + by-subtask boxplot
  advantage_squashed_dist.png   tanh(TD-error / scaling) version of the above
  gradient_magnitudes.png       (if adapter supports it)
  frame_p{XX}.png               (if adapter supports it) percentile exemplars
"""

from __future__ import annotations

import json
import logging
import os
import random
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
from lerobot.probes.utils import build_episode_index, get_frame_data
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
) -> list[int]:
    """Pick *n_frames* indices whose [idx, idx + lookahead_frames] window stays in one episode."""
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

    The video plays at native fps; only the V(s) curve is sub-sampled
    (stride = ``probe_parameters.attn_eval_subsample``, default 2). Reuses
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
    subsample = max(1, int(getattr(p, "attn_eval_subsample", 2)))
    seed = int(getattr(p, "random_seed", 42))
    max_episodes = getattr(p, "max_episodes", None)
    video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])
    fps = cfg.env.fps

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
        for step_idx, global_idx in enumerate(indices):
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
        critic_values: list[float] = []
        critic_indices = list(range(0, len(indices), subsample))
        for ci in critic_indices:
            obs, _, _, _, task_str, _, _ = get_frame_data(
                val_dataset, indices[ci], chunk_size,
            )
            try:
                v = adapter.predict_value(obs, task_str)
            except Exception as exc:
                logging.warning(f"[CRITIC] ep{ep_idx} step{ci}: V(s) failed: {exc}")
                v = 0.0
            critic_values.append(v)

        # ── 3. Save critic JSON + plot ───────────────────────────────────────
        with open(os.path.join(ep_dir, "critic_values.json"), "w") as f:
            json.dump(critic_values, f)
        if critic_values:
            plt.figure(figsize=(10, 5))
            plt.plot(critic_values)
            plt.title(f"Critic Values - Episode {ep_idx}")
            plt.xlabel("Step")
            plt.ylabel("V(s)")
            plt.grid(True)
            plt.savefig(os.path.join(ep_dir, "critic_plot.png"))
            plt.close()

        # ── 4. Overlay video ─────────────────────────────────────────────────
        try:
            save_video_with_critic_overlay(
                ep_dir, critic_values,
                camera_names=video_logging_cameras,
                fps=fps,
                subtask_texts=subtask_texts,
                subsample=subsample,
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
    )
    if not indices:
        logging.warning("[CRITIC] predicted_distributions: no samples")
        return None

    ep_to_indices = build_episode_index(val_dataset)
    ep_last_frame = {
        ep: max(val_dataset.hf_dataset[i]["frame_index"].item() for i in idxs)
        for ep, idxs in ep_to_indices.items()
    }

    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

    for i, idx in enumerate(indices):
        obs, _, _, gt_subtask, task_str, ep_idx, fr_idx = get_frame_data(
            val_dataset, idx, chunk_size,
        )
        v, probs, bin_centers = adapter.predict_value_and_probs(obs, task_str)
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

    # ── Part 1: TD-error / squashed advantage ────────────────────────────────
    # Mirrors training buffer: next_state = s_{t + chunk_size}; reward = max over
    # [t + chunk, t + 2*chunk); done = any over the same window.
    logging.info(f"[CRITIC] sampling {n_adv} frames for TD-error")
    adv_indices = get_random_valid_samples(
        val_dataset, n_adv, seed,
        val_ep_indices=val_ep_indices, lookahead_frames=2 * chunk_size - 1,
    )
    if not adv_indices:
        logging.warning("[CRITIC] no advantage samples")
        return None

    td_errors: list[float] = []
    squashed: list[float] = []
    adv_subtasks: list[str] = []
    for idx in adv_indices:
        obs, _, _, gt_subtask, task_str, _, _ = get_frame_data(val_dataset, idx, chunk_size)
        next_obs, _, _, _, next_task_str, _, _ = get_frame_data(
            val_dataset, idx + chunk_size, chunk_size,
        )
        v_curr = adapter.predict_value(obs, task_str)
        v_next = adapter.predict_value(next_obs, next_task_str)

        rewards, dones = [], []
        for off in range(chunk_size):
            w_idx = min(idx + chunk_size + off, len(val_dataset) - 1)
            f = val_dataset.hf_dataset[w_idx]
            r = f.get("reward", 0.0); d = f.get("next.done", False)
            if isinstance(r, torch.Tensor): r = r.item()
            if isinstance(d, torch.Tensor): d = d.item()
            rewards.append(float(r)); dones.append(bool(d))
        reward = max(rewards); done = any(dones)

        target_v = reward + discount * v_next * (1.0 - float(done))
        td = target_v - v_curr
        td_errors.append(td)
        squashed.append(float(np.tanh(td / advantage_scaling)))
        adv_subtasks.append(gt_subtask or "None")

    _td_error_plots(td_errors, squashed, adv_subtasks, output_dir)

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
    }

    if grad_supported and _probe_value_grad is not None:
        logging.info(f"[CRITIC] sampling {n_grad} frames for gradients")
        grad_indices = get_random_valid_samples(
            val_dataset, n_grad, seed + 1,
            val_ep_indices=val_ep_indices,
        )
        grad_mags, episodes, subtasks, frames = [], [], [], []
        frame_cache: dict[int, dict] = {}
        for idx in grad_indices:
            obs, _, _, gt_subtask, task_str, ep_idx, fr_idx = get_frame_data(
                val_dataset, idx, chunk_size,
            )
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

    from lerobot.datasets.factory import make_dataset
    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None

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
    probe_cli()
