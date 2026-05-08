#!/usr/bin/env python
import os
import random
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Set

warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import torch
import textwrap

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.inference_utils import _finalize_episode_log
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

from lerobot.probes.offline_inference_pi05 import (
    _build_episode_index,
    _load_policy_and_processors,
    get_frame_data,
)


@dataclass
class ProbeCriticConfig(TrainRLServerPipelineConfig):
    pass


def get_random_valid_samples(
    dataset,
    n_frames: int,
    seed: int,
    val_ep_indices: Optional[Set[int]] = None,
    lookahead_frames: int = 1,
) -> List[int]:
    """Pick *n_frames* dataset indices whose [idx, idx + lookahead_frames] window
    lies entirely in the same episode.

    If *val_ep_indices* is provided, the candidate pool is restricted to those
    episodes only.
    """
    rng = random.Random(seed)

    if val_ep_indices is not None:
        ep_to_indices = _build_episode_index(dataset)
        candidate_indices: List[int] = []
        for ep, idxs in ep_to_indices.items():
            if ep not in val_ep_indices:
                continue
            candidate_indices.extend(idxs)
    else:
        candidate_indices = list(range(len(dataset)))

    rng.shuffle(candidate_indices)

    samples: List[int] = []
    for idx in candidate_indices:
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


@torch.no_grad()
def get_v(policy, preprocessor, obs, task_str, device):
    batch_size = 1
    dummy_action = torch.zeros(batch_size, 1, 6, device=device)
    complementary_data = {
        "task": [task_str],
        "subtask": [""],
        "advantage": torch.tensor([[1.0]], device=device),
    }

    batch_for_proc = {
        TransitionKey.ACTION: dummy_action,
        **{k: v.to(device) for k, v in obs.items()},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }
    processed = preprocessor(batch_for_proc)

    actor_tokens = processed[OBS_LANGUAGE_TOKENS]
    actor_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]

    critic_tokens = processed.get("critic_tokens", actor_tokens)
    critic_token_masks = processed.get("critic_pad_mask", actor_masks)

    actor_embed_layer = policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    critic_text_embs = actor_embed_layer(critic_tokens).detach()

    images, img_masks = policy._preprocess_images({k: v.to(device) for k, v in obs.items()})

    vision_features = []
    vision_pad_masks = []

    encoder = policy.critic
    for img, img_mask in zip(images, img_masks):
        feat = encoder.embed_image(img)
        vision_features.append(feat)
        B, N, _ = feat.shape
        mask = img_mask[:, None].expand(B, N)
        vision_pad_masks.append(mask)

    vision_features = torch.cat(vision_features, dim=1)
    vision_pad_masks = torch.cat(vision_pad_masks, dim=1)

    out = policy.critic(vision_features, critic_text_embs, critic_token_masks)
    return out["value"].item()


@torch.no_grad()
def get_v_and_probs(policy, preprocessor, obs, task_str, device):
    """Like `get_v` but also returns the full predicted distribution over value bins."""
    batch_size = 1
    dummy_action = torch.zeros(batch_size, 1, 6, device=device)
    complementary_data = {
        "task": [task_str],
        "subtask": [""],
        "advantage": torch.tensor([[1.0]], device=device),
    }

    batch_for_proc = {
        TransitionKey.ACTION: dummy_action,
        **{k: v.to(device) for k, v in obs.items()},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }
    processed = preprocessor(batch_for_proc)

    actor_tokens = processed[OBS_LANGUAGE_TOKENS]
    actor_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]

    critic_tokens = processed.get("critic_tokens", actor_tokens)
    critic_token_masks = processed.get("critic_pad_mask", actor_masks)

    actor_embed_layer = policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    critic_text_embs = actor_embed_layer(critic_tokens).detach()

    images, img_masks = policy._preprocess_images({k: v.to(device) for k, v in obs.items()})

    vision_features = []
    encoder = policy.critic
    for img, img_mask in zip(images, img_masks):
        feat = encoder.embed_image(img)
        vision_features.append(feat)
    vision_features = torch.cat(vision_features, dim=1)

    out = policy.critic(vision_features, critic_text_embs, critic_token_masks)
    v = out["value"].item()
    probs = out["probs"].squeeze(0).float().cpu().numpy()
    return v, probs


def compute_gradient_magnitude(policy, preprocessor, obs, task_str, device):
    batch_size = 1
    dummy_action = torch.zeros(batch_size, 1, 6, device=device)
    complementary_data = {
        "task": [task_str],
        "subtask": [""],
        "advantage": torch.tensor([[1.0]], device=device),
    }

    batch_for_proc = {
        TransitionKey.ACTION: dummy_action,
        **{k: v.to(device) for k, v in obs.items()},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }
    processed = preprocessor(batch_for_proc)

    actor_tokens = processed[OBS_LANGUAGE_TOKENS]
    actor_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]

    critic_tokens = processed.get("critic_tokens", actor_tokens)
    critic_token_masks = processed.get("critic_pad_mask", actor_masks)

    actor_embed_layer = policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    critic_text_embs = actor_embed_layer(critic_tokens).detach()

    images, img_masks = policy._preprocess_images({k: v.to(device) for k, v in obs.items()})

    vision_features = []
    vision_pad_masks = []

    encoder = policy.critic
    for img, img_mask in zip(images, img_masks):
        feat = encoder.embed_image(img)
        vision_features.append(feat)
        B, N, _ = feat.shape
        mask = img_mask[:, None].expand(B, N)
        vision_pad_masks.append(mask)

    vision_features = torch.cat(vision_features, dim=1)
    vision_pad_masks = torch.cat(vision_pad_masks, dim=1)

    # Make them leaf tensors to receive gradients
    vision_features = vision_features.detach().requires_grad_(True)
    critic_text_embs = critic_text_embs.detach().requires_grad_(True)

    out = policy.critic(vision_features, critic_text_embs, critic_token_masks)
    v = out["value"].sum()

    policy.critic.zero_grad()
    v.backward()

    grad_mag = vision_features.grad.norm().item()
    return grad_mag


def render_percentile_frame(obs, episode_idx, frame_idx, subtask, mag_val, p, output_dir):
    camera_keys = sorted(k for k in obs if "images" in k)
    n_cameras = len(camera_keys)

    fig = plt.figure(figsize=(12 + 3, 5))
    gs = GridSpec(1, n_cameras + 1, figure=fig, width_ratios=[1]*n_cameras + [0.8])

    # Draw cameras
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

    # Draw Info panel
    ax_info = fig.add_subplot(gs[0, n_cameras])
    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)

    info_text = (
        f"Gradient Percentile:\n  p{p}\n\n"
        f"Magnitude:\n  {mag_val:.4f}\n\n"
        f"Episode:\n  {episode_idx}\n\n"
        f"Frame:\n  {frame_idx}\n\n"
        f"Subtask:\n"
    )

    # Wrap subtask text nicely
    wrapped_subtask = "\\n  ".join(textwrap.wrap(subtask or "None", width=25))
    info_text += f"  {wrapped_subtask}"

    ax_info.text(0.1, 0.8, info_text, transform=ax_info.transAxes,
                 fontsize=13, va="top", ha="left", color="#333",
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ccc', boxstyle='round,pad=1'))

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"frame_p{p:02d}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    sns.despine(ax=ax, offset=10, trim=True)


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode V(s) traces (mirrors online _finalize_episode_log → critic-overlay video)
# ──────────────────────────────────────────────────────────────────────────────

def run_episode_critic_traces(
    policy,
    preprocessor,
    val_dataset,
    val_ep_indices,
    cfg,
    output_dir: str,
    device,
):
    """For each selected episode, build the same per-frame buffer that the
    online rollout produces, then hand it to `_finalize_episode_log` — the
    function that already saves PNGs, runs critic inference, dumps the JSON +
    plot, and renders the critic-overlay video.

    The video plays at native fps; only the critic forward is sub-sampled
    (stride = ``probe_parameters.attn_eval_subsample``, default 2).

    Outputs (one sub-directory per episode):
        {output_dir}/ep{NNNN}/critic_values.json
        {output_dir}/ep{NNNN}/critic_plot.png
        {output_dir}/ep{NNNN}/episode_video.mp4
    """
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    subsample = max(1, int(getattr(p, "attn_eval_subsample", 2)))
    seed = int(getattr(p, "random_seed", 42))
    max_episodes = getattr(p, "max_episodes", None)
    video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])

    # `_finalize_episode_log` reads cfg.policy.preprocessor for the critic batch.
    if not hasattr(policy, "preprocessor") or policy.preprocessor is None:
        policy.preprocessor = preprocessor

    ep_to_indices = _build_episode_index(val_dataset)
    if val_ep_indices is not None:
        ep_to_indices = {k: v for k, v in ep_to_indices.items() if k in val_ep_indices}
    selected_eps = sorted(ep_to_indices.keys())
    if max_episodes:
        rng = random.Random(seed)
        rng.shuffle(selected_eps)
        selected_eps = sorted(selected_eps[: int(max_episodes)])

    if not selected_eps:
        logging.warning("[VAL] critic_episode_traces: no episodes to process")
        return None

    os.makedirs(output_dir, exist_ok=True)

    for ep_idx in selected_eps:
        indices = ep_to_indices[ep_idx]
        ep_dir = os.path.join(output_dir, f"ep{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        episode_log_buffer = []
        for global_idx in indices:
            obs, _, _, gt_subtask, _, _, _ = get_frame_data(
                val_dataset, global_idx, chunk_size,
            )
            episode_log_buffer.append({
                "obs": obs,
                "subtask_text": gt_subtask or "",
            })

        try:
            _finalize_episode_log(
                episode_log_buffer=episode_log_buffer,
                policy=policy,
                cfg=cfg,
                log_dir=ep_dir,
                episode_counter=ep_idx,
                video_logging_cameras=video_logging_cameras,
                critic_subsample=subsample,
            )
        except Exception as exc:
            logging.warning(
                f"[VAL] critic_episode_traces ep{ep_idx}: failed: {exc}",
                exc_info=True,
            )

    return {"episodes": selected_eps, "subsample": subsample}


def _rescale_to_critic_range(values: list[float]) -> list[float]:
    """Linearly rescale *values* to the critic's display range [-2.1, 0.1].

    Uses the 2nd/98th percentiles as the input extent so a single spike
    doesn't flatten the rest of the curve.
    """
    arr = np.array(values, dtype=np.float64)
    p_lo = float(np.percentile(arr, 2))
    p_hi = float(np.percentile(arr, 98))
    if p_hi == p_lo:
        return [-1.0] * len(values)
    c_min, c_max = -2.1, 0.1
    scaled = (arr - p_lo) / (p_hi - p_lo) * (c_max - c_min) + c_min
    return scaled.tolist()


def run_episode_gradient_traces(
    policy,
    preprocessor,
    val_dataset,
    val_ep_indices,
    cfg,
    output_dir: str,
    device,
):
    """Same structure as run_episode_critic_traces but overlays gradient magnitude.

    For each selected episode:
      1. Saves per-frame PNGs.
      2. Computes gradient magnitude at every *subsample*-th frame.
      3. Rescales the magnitudes into the critic display range [-2.1, 0.1] so
         the existing save_video_with_critic_overlay is reused unchanged.

    Output per episode:
        {output_dir}/ep{NNNN}/episode_video.mp4
    """
    from lerobot.rl.utils import save_video_with_critic_overlay
    from PIL import Image as PILImage

    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    subsample = max(1, int(getattr(p, "attn_eval_subsample", 2)))
    seed = int(getattr(p, "random_seed", 42))
    max_episodes = getattr(p, "max_episodes", None)
    video_logging_cameras = getattr(cfg, "video_logging_cameras", ["top", "side"])

    ep_to_indices = _build_episode_index(val_dataset)
    if val_ep_indices is not None:
        ep_to_indices = {k: v for k, v in ep_to_indices.items() if k in val_ep_indices}
    selected_eps = sorted(ep_to_indices.keys())
    if max_episodes:
        rng = random.Random(seed)
        rng.shuffle(selected_eps)
        selected_eps = sorted(selected_eps[: int(max_episodes)])

    if not selected_eps:
        logging.warning("[VAL] gradient_episode_traces: no episodes to process")
        return None

    os.makedirs(output_dir, exist_ok=True)

    for ep_idx in selected_eps:
        indices = ep_to_indices[ep_idx]
        ep_dir = os.path.join(output_dir, f"ep{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        # ── 1. Save per-frame PNGs ───────────────────────────────────────────
        subtask_texts = []
        for step_idx, global_idx in enumerate(indices):
            obs, _, _, gt_subtask, task_str, _, _ = get_frame_data(
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
                    if v_max <= 5.0:
                        img_np = (img_tensor.float().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        img_np = img_tensor.float().numpy().transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(img_np).save(
                    os.path.join(ep_dir, f"step_{step_idx:06d}_{cam_name}.png")
                )

        # ── 2. Compute gradient magnitudes (subsampled) ──────────────────────
        grad_mags: list[float] = []
        critic_indices = list(range(0, len(indices), subsample))
        for ci in critic_indices:
            obs, _, _, _, task_str, _, _ = get_frame_data(
                val_dataset, indices[ci], chunk_size,
            )
            try:
                mag = compute_gradient_magnitude(policy, preprocessor, obs, task_str, device)
            except Exception as exc:
                logging.warning(f"[VAL] gradient_traces ep{ep_idx} step{ci}: {exc}")
                mag = 0.0
            grad_mags.append(mag)

        # ── 3. Rescale and render video ──────────────────────────────────────
        if not grad_mags:
            continue
        scaled = _rescale_to_critic_range(grad_mags)
        try:
            save_video_with_critic_overlay(
                ep_dir,
                scaled,
                camera_names=video_logging_cameras,
                fps=cfg.env.fps,
                subtask_texts=subtask_texts,
                subsample=subsample,
            )
        except Exception as exc:
            logging.warning(
                f"[VAL] gradient_episode_traces ep{ep_idx}: video failed: {exc}",
                exc_info=True,
            )

    return {"episodes": selected_eps, "subsample": subsample}


def run_predicted_distributions(
    policy,
    preprocessor,
    val_dataset,
    val_ep_indices,
    cfg,
    output_dir: str,
    device,
):
    """Plot the critic's predicted distribution over the value support for a
    handful of random frames. One subplot per frame, in a grid; x-axis is the
    bin centers (V_min..V_max), y-axis is probability mass. The expected V is
    overlaid as a vertical dashed line.

    Output:
        {output_dir}/predicted_distributions.png
    """
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    n_frames = int(getattr(p, "critic_dist_frames", 9))
    seed = int(getattr(p, "random_seed", 42)) + 7

    indices = get_random_valid_samples(
        val_dataset, n_frames, seed, val_ep_indices=val_ep_indices,
        lookahead_frames=0,
    )
    if not indices:
        logging.warning("[VAL] predicted_distributions: no samples")
        return None

    bin_centers = policy.critic.bin_centers.detach().float().cpu().numpy()

    # Per-episode last frame index, used to label each subplot with how many
    # frames remain until episode end (a critic value is much easier to judge
    # when you know the frame is e.g. 5 steps from termination vs 200).
    ep_to_indices = _build_episode_index(val_dataset)
    ep_last_frame = {
        ep: max(val_dataset.hf_dataset[i]["frame_index"].item() for i in idxs)
        for ep, idxs in ep_to_indices.items()
    }

    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

    for i, idx in enumerate(indices):
        obs, _, _, gt_subtask, task_str, ep_idx, frame_idx = get_frame_data(
            val_dataset, idx, chunk_size,
        )
        v, probs = get_v_and_probs(policy, preprocessor, obs, task_str, device)

        frames_to_end = ep_last_frame[ep_idx] - frame_idx

        ax = axes[i // n_cols, i % n_cols]
        ax.plot(bin_centers, probs, color="steelblue", linewidth=2)
        ax.fill_between(bin_centers, probs, alpha=0.2, color="steelblue")
        ax.axvline(v, color="crimson", linestyle="--", linewidth=1.5, label=f"E[V] = {v:.3f}")
        title = f"ep{ep_idx} f{frame_idx}  ({frames_to_end} to end)"
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
    logging.info(f"[VAL] predicted_distributions: saved → {out_path}")

    return {
        "indices": indices,
        "bin_centers": torch.tensor(bin_centers),
    }


def run_critic_values_distribution(
    policy,
    preprocessor,
    val_dataset,
    val_ep_indices,
    cfg,
    output_dir: str,
    device,
):
    """Compute V(s) / TD-error distribution and gradient-magnitude exemplars
    for the critic. Writes:

        {output_dir}/advantage_dist.png
        {output_dir}/gradient_magnitudes.png
        {output_dir}/frame_p{XX}.png   (one per percentile)

    Returns a raw-data dict (or None if no samples / no critic).
    """
    if not hasattr(policy, "critic"):
        logging.warning(
            "[VAL] critic_values_distribution: policy has no .critic attribute; skipping."
        )
        return None

    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(output_dir, exist_ok=True)

    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    n_adv = int(getattr(p, "critic_adv_frames", 1000))
    n_grad = int(getattr(p, "critic_grad_frames", 200))
    seed = int(getattr(p, "random_seed", 42))
    discount = getattr(policy.config, "discount", 0.99)
    advantage_scaling = float(getattr(cfg.policy, "advantage_scaling", 1.0))

    # ── Part 0: per-episode V(s) traces ─────────────────────────────────────
    # Walk full episodes at native fps; sub-sample only the critic forward.
    # Reuses inference_utils._finalize_episode_log so the rendered video is
    # identical to the one produced by the online rollout logger.
    try:
        run_episode_critic_traces(
            policy, preprocessor,
            val_dataset, val_ep_indices,
            cfg,
            output_dir=os.path.join(output_dir, "episode_traces"),
            device=device,
        )
    except Exception as exc:
        logging.warning(
            f"[VAL] critic_values_distribution: episode traces failed: {exc}",
            exc_info=True,
        )

    # ── Part 0.25: per-episode gradient magnitude traces ────────────────────
    try:
        run_episode_gradient_traces(
            policy, preprocessor,
            val_dataset, val_ep_indices,
            cfg,
            output_dir=os.path.join(output_dir, "episode_gradient_traces"),
            device=device,
        )
    except Exception as exc:
        logging.warning(
            f"[VAL] critic_values_distribution: gradient traces failed: {exc}",
            exc_info=True,
        )

    # ── Part 0.5: predicted value distributions for random frames ───────────
    try:
        run_predicted_distributions(
            policy, preprocessor,
            val_dataset, val_ep_indices,
            cfg,
            output_dir=output_dir,
            device=device,
        )
    except Exception as exc:
        logging.warning(
            f"[VAL] critic_values_distribution: predicted distributions failed: {exc}",
            exc_info=True,
        )

    # ── Part 1: advantage / TD-error ────────────────────────────────────────
    # Mirrors training (buffer.py:309 + buffer.py:330-334):
    #   next_state = s_{t + chunk_size}
    #   reward     = max over [t + chunk, t + 2*chunk)
    #   done       = any over [t + chunk, t + 2*chunk)
    # Need the full window [idx, idx + 2*chunk - 1] inside the same episode.
    logging.info(f"[VAL] critic_values_distribution: sampling {n_adv} frames for TD-error")
    adv_indices = get_random_valid_samples(
        val_dataset, n_adv, seed, val_ep_indices=val_ep_indices,
        lookahead_frames=2 * chunk_size - 1,
    )
    if not adv_indices:
        logging.warning("[VAL] critic_values_distribution: no advantage samples")
        return None

    td_errors: list[float] = []
    squashed_advantages: list[float] = []
    adv_subtasks: list[str] = []

    for i, idx in enumerate(adv_indices):
        obs, _, _, gt_subtask, task_str, _, _ = get_frame_data(val_dataset, idx, chunk_size)
        next_obs, _, _, _, next_task_str, _, _ = get_frame_data(val_dataset, idx + chunk_size, chunk_size)

        v_current = get_v(policy, preprocessor, obs, task_str, device)
        v_next = get_v(policy, preprocessor, next_obs, next_task_str, device)

        # Aggregate reward (max) and done (any) over the lookahead chunk window.
        rewards_window: list[float] = []
        dones_window: list[bool] = []
        for off in range(chunk_size):
            w_idx = min(idx + chunk_size + off, len(val_dataset) - 1)
            f = val_dataset.hf_dataset[w_idx]
            r = f.get("reward", 0.0)
            d = f.get("next.done", False)
            if isinstance(r, torch.Tensor): r = r.item()
            if isinstance(d, torch.Tensor): d = d.item()
            rewards_window.append(float(r))
            dones_window.append(bool(d))
        reward = max(rewards_window)
        done = any(dones_window)

        target_v = reward + discount * v_next * (1.0 - float(done))
        td_error = target_v - v_current
        squashed = float(np.tanh(td_error / advantage_scaling))

        td_errors.append(td_error)
        squashed_advantages.append(squashed)
        adv_subtasks.append(gt_subtask or "None")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(td_errors, bins=50, kde=True, ax=axes[0], color="coral", edgecolor="white")
    style_plot(axes[0], "TD-Error (Advantage) Histogram", "TD-Error", "Count")

    sns.ecdfplot(td_errors, ax=axes[1], color="coral", linewidth=3)
    style_plot(axes[1], "TD-Error (Advantage) CDF", "TD-Error", "Cumulative Probability")
    axes[1].margins(y=0.05)

    sns.boxplot(x="td_error", y="subtask", hue="subtask", legend=False,
                data={"td_error": td_errors, "subtask": adv_subtasks},
                ax=axes[2], palette="pastel", fliersize=0)
    sns.stripplot(x="td_error", y="subtask", data={"td_error": td_errors, "subtask": adv_subtasks},
                  ax=axes[2], color=".3", size=3, alpha=0.5, jitter=True)
    style_plot(axes[2], "TD-Error by Subtask", "TD-Error", "Subtask")
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)

    plt.tight_layout(pad=3.0)
    adv_plot_path = os.path.join(output_dir, "advantage_dist.png")
    plt.savefig(adv_plot_path, dpi=200)
    plt.close()
    logging.info(f"[VAL] critic_values_distribution: saved advantage plot → {adv_plot_path}")

    # Squashed advantage = tanh(raw / advantage_scaling), matching
    # pi05_train_utils.py:720 (the value the actor actually conditions on).
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(squashed_advantages, bins=50, kde=True, ax=axes[0],
                 color="seagreen", edgecolor="white")
    style_plot(axes[0], "Squashed Advantage Histogram", "tanh(adv / scale)", "Count")

    sns.ecdfplot(squashed_advantages, ax=axes[1], color="seagreen", linewidth=3)
    style_plot(axes[1], "Squashed Advantage CDF", "tanh(adv / scale)", "Cumulative Probability")
    axes[1].margins(y=0.05)

    sns.boxplot(x="squashed", y="subtask", hue="subtask", legend=False,
                data={"squashed": squashed_advantages, "subtask": adv_subtasks},
                ax=axes[2], palette="pastel", fliersize=0)
    sns.stripplot(x="squashed", y="subtask",
                  data={"squashed": squashed_advantages, "subtask": adv_subtasks},
                  ax=axes[2], color=".3", size=3, alpha=0.5, jitter=True)
    style_plot(axes[2], "Squashed Advantage by Subtask", "tanh(adv / scale)", "Subtask")
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)

    plt.tight_layout(pad=3.0)
    sq_plot_path = os.path.join(output_dir, "advantage_squashed_dist.png")
    plt.savefig(sq_plot_path, dpi=200)
    plt.close()
    logging.info(f"[VAL] critic_values_distribution: saved squashed advantage plot → {sq_plot_path}")

    # ── Part 2: gradient magnitudes ─────────────────────────────────────────
    logging.info(f"[VAL] critic_values_distribution: sampling {n_grad} frames for gradients")
    grad_indices = get_random_valid_samples(
        val_dataset, n_grad, seed + 1, val_ep_indices=val_ep_indices,
    )
    if not grad_indices:
        logging.warning("[VAL] critic_values_distribution: no gradient samples")
        return {
            "td_errors": torch.tensor(td_errors),
            "squashed_advantages": torch.tensor(squashed_advantages),
            "adv_subtasks": adv_subtasks,
        }

    grad_mags: list[float] = []
    episodes: list[int] = []
    subtasks: list[str] = []
    frames: list[int] = []
    frame_cache: dict[int, dict] = {}

    for i, idx in enumerate(grad_indices):
        obs, _, _, gt_subtask, task_str, ep_idx, frame_idx = get_frame_data(
            val_dataset, idx, chunk_size,
        )
        frame_cache[idx] = {k: v.clone() for k, v in obs.items() if "image" in k}

        grad_mag = compute_gradient_magnitude(policy, preprocessor, obs, task_str, device)

        grad_mags.append(grad_mag)
        episodes.append(ep_idx)
        subtasks.append(gt_subtask or "None")
        frames.append(frame_idx)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.boxplot(x="grad_mag", y="subtask", hue="subtask", legend=False,
                data={"grad_mag": grad_mags, "subtask": subtasks},
                ax=axes[0], palette="pastel", fliersize=0)
    sns.stripplot(x="grad_mag", y="subtask", data={"grad_mag": grad_mags, "subtask": subtasks},
                  ax=axes[0], color=".3", size=4, alpha=0.5, jitter=True)
    style_plot(axes[0], "Gradient Magnitude by Subtask", "Magnitude (L2 Norm)", "Subtask")

    sns.scatterplot(x=range(len(grad_mags)), y=grad_mags, hue=episodes, palette="viridis",
                    s=80, alpha=0.8, edgecolor="white", ax=axes[1])
    style_plot(axes[1], "Gradient Magnitude per Frame (colored by Episode)", "Sample Index", "Magnitude (L2 Norm)")
    axes[1].legend(title="Episode Index", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.tight_layout(pad=3.0)
    grad_plot_path = os.path.join(output_dir, "gradient_magnitudes.png")
    plt.savefig(grad_plot_path, dpi=200)
    plt.close()
    logging.info(f"[VAL] critic_values_distribution: saved gradient plot → {grad_plot_path}")

    # ── Part 3: percentile exemplar frames ──────────────────────────────────
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    grad_mags_np = np.array(grad_mags)
    percentile_values = np.percentile(grad_mags_np, percentiles)
    percentile_records = []

    for pct, val in zip(percentiles, percentile_values):
        closest = int(np.argmin(np.abs(grad_mags_np - val)))
        ds_idx = grad_indices[closest]
        mag_val = float(grad_mags_np[closest])
        ep_idx = episodes[closest]
        frame_idx = frames[closest]
        subtask = subtasks[closest]

        if ds_idx in frame_cache:
            render_percentile_frame(
                frame_cache[ds_idx], ep_idx, frame_idx, subtask, mag_val, pct, output_dir,
            )
        percentile_records.append({
            "percentile": pct, "magnitude": mag_val,
            "episode_idx": ep_idx, "frame_idx": frame_idx, "subtask": subtask,
        })

    raw = {
        "td_errors":   torch.tensor(td_errors),
        "squashed_advantages": torch.tensor(squashed_advantages),
        "grad_mags":   torch.tensor(grad_mags),
        "episodes":    torch.tensor(episodes),
        "frames":      torch.tensor(frames),
        "adv_subtasks": adv_subtasks,
        "grad_subtasks": subtasks,
        "percentiles": percentile_records,
    }
    return raw


@parser.wrap()
def probe_cli(cfg: ProbeCriticConfig):
    init_logging()
    sns.set_theme(style="whitegrid", palette="muted")

    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "critic")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    logging.info("Loading policy and dataset...")
    policy, preprocessor, _postprocessor, dataset = _load_policy_and_processors(cfg, device)

    if not hasattr(policy, "critic"):
        raise ValueError("Policy does not have a critic. Are you sure you are loading PI05RLPolicy?")

    policy.eval()
    run_critic_values_distribution(
        policy, preprocessor,
        val_dataset=dataset,
        val_ep_indices=None,
        cfg=cfg,
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    probe_cli()
