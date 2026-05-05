#!/usr/bin/env python
import os
import random
import logging
from dataclasses import dataclass
from typing import List, Optional, Set

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
) -> List[int]:
    """Pick *n_frames* dataset indices whose next index lies in the same episode.

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
        if idx + 1 >= len(dataset):
            continue

        item = dataset.hf_dataset[idx]
        next_item = dataset.hf_dataset[idx + 1]
        if item["episode_index"].item() != next_item["episode_index"].item():
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

    v = policy.critic(vision_features, critic_text_embs, critic_token_masks)
    return v.item()


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

    v = policy.critic(vision_features, critic_text_embs, critic_token_masks)

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
    logging.info(f"  Saved p{p} frame to {out_path}")


def style_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=12)
    sns.despine(ax=ax, offset=10, trim=True)


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

    # ── Part 1: advantage / TD-error ────────────────────────────────────────
    logging.info(f"[VAL] critic_values_distribution: sampling {n_adv} frames for TD-error")
    adv_indices = get_random_valid_samples(
        val_dataset, n_adv, seed, val_ep_indices=val_ep_indices,
    )
    if not adv_indices:
        logging.warning("[VAL] critic_values_distribution: no advantage samples")
        return None

    td_errors: list[float] = []
    adv_subtasks: list[str] = []

    for i, idx in enumerate(adv_indices):
        obs, _, _, gt_subtask, task_str, _, _ = get_frame_data(val_dataset, idx, chunk_size)
        next_obs, _, _, _, next_task_str, _, _ = get_frame_data(val_dataset, idx + 1, chunk_size)

        v_current = get_v(policy, preprocessor, obs, task_str, device)
        v_next = get_v(policy, preprocessor, next_obs, next_task_str, device)

        item = val_dataset.hf_dataset[idx]
        reward = item.get("reward", 0.0)
        done = item.get("next.done", False)

        if isinstance(reward, torch.Tensor): reward = reward.item()
        if isinstance(done, torch.Tensor): done = done.item()

        target_v = reward + discount * v_next * (1.0 - float(done))
        td_error = target_v - v_current

        td_errors.append(td_error)
        adv_subtasks.append(gt_subtask or "None")

        if (i + 1) % 50 == 0:
            logging.info(f"  Processed {i+1}/{len(adv_indices)} advantage frames")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    sns.histplot(td_errors, bins=50, kde=True, ax=axes[0], color="coral", edgecolor="white")
    style_plot(axes[0], "TD-Error (Advantage) Histogram", "TD-Error", "Count")

    sns.ecdfplot(td_errors, ax=axes[1], color="coral", linewidth=3)
    style_plot(axes[1], "TD-Error (Advantage) CDF", "TD-Error", "Cumulative Probability")
    axes[1].margins(y=0.05)

    sns.boxplot(x="td_error", y="subtask", data={"td_error": td_errors, "subtask": adv_subtasks},
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

    # ── Part 2: gradient magnitudes ─────────────────────────────────────────
    logging.info(f"[VAL] critic_values_distribution: sampling {n_grad} frames for gradients")
    grad_indices = get_random_valid_samples(
        val_dataset, n_grad, seed + 1, val_ep_indices=val_ep_indices,
    )
    if not grad_indices:
        logging.warning("[VAL] critic_values_distribution: no gradient samples")
        return {
            "td_errors": torch.tensor(td_errors),
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

        if (i + 1) % 20 == 0:
            logging.info(f"  Processed {i+1}/{len(grad_indices)} gradient frames")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.boxplot(x="grad_mag", y="subtask", data={"grad_mag": grad_mags, "subtask": subtasks},
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
