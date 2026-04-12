#!/usr/bin/env python
"""
Offline evaluation script for PI05 RL policy.

Samples frames from a dataset, runs inference, and generates comparison plots:
  - Robot camera images at the sampled timestep
  - Predicted vs GT action chunk per joint (full chunk_size horizon)
  - Generated subtask vs GT subtask label

Usage examples
--------------
# Manual: specific (episode, frame) pairs
python eval_offline_pi05.py config.json \
    --eval_episodes "0,1,5" --eval_frames "10,20,30" \
    --eval_output_dir outputs/eval/

# Random: sample N frames from the dataset
python eval_offline_pi05.py config.json \
    --eval_random_n 10 \
    --eval_output_dir outputs/eval/

# Checkpoint comparison (A vs B) — loads each checkpoint sequentially
python eval_offline_pi05.py config.json \
    --eval_random_n 5 \
    --eval_checkpoint_b /path/to/other/checkpoint \
    --eval_output_dir outputs/eval/
"""

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import lerobot.rl.rl_pi05  # noqa: F401 — registers PI05RLConfig

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.processor.core import TransitionKey
from lerobot.rl.pi05_train_utils import (
    hydrate_subtasks,
    make_pi05_full_processors_with_upgrade,
)
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.utils import get_safe_torch_device, init_logging


# SO100 joint names in action-vector order (matches so_follower motor bus definition).
SO100_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass
class EvalOfflineConfig(TrainRLServerPipelineConfig):
    """Extra fields for offline evaluation (all optional, all have defaults)."""

    # Manual frame selection: comma-separated episode indices, e.g. "0,1,5"
    eval_episodes: Optional[str] = None
    # Comma-separated frame indices within each episode, e.g. "10,20,30"
    # If a single value, it is applied to every episode.
    eval_frames: Optional[str] = None
    # Random sampling: sample this many frames at random (in addition to any explicit ones)
    eval_random_n: int = 0
    # Where to save evaluation plots
    eval_output_dir: str = "outputs/eval_offline"
    # Path to a second checkpoint for side-by-side comparison (optional)
    eval_checkpoint_b: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_policy_and_processors(cfg, device, dataset=None):
    """
    Load policy and pre/post-processors.
    Follows the same pattern as offline_learner_pi05.py.

    If `dataset` is not provided, it is loaded from cfg (first call).
    Pass it in on subsequent calls (e.g. checkpoint B) to avoid reloading.
    """
    if dataset is None:
        dataset = make_dataset(cfg)
        dataset.delta_timestamps = None
        dataset.delta_indices = None

    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(
        cfg, dataset=dataset, is_main_process=True
    )

    original_device = cfg.policy.device
    cfg.policy.device = device.type
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    cfg.policy.device = original_device

    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor
    policy.eval()
    policy.to(device)

    return policy, preprocessor, postprocessor, dataset


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def _build_episode_index(dataset):
    """Returns {episode_idx: [global_idx, ...]} sorted by frame_index."""
    ep_to_indices = {}
    for global_idx in range(len(dataset)):
        item = dataset.hf_dataset[global_idx]
        ep_idx = item["episode_index"].item()
        ep_to_indices.setdefault(ep_idx, []).append(global_idx)
    # Each episode list is already in frame order (parquet is sequential)
    return ep_to_indices


def build_sample_list(dataset, episodes_str, frames_str, random_n, chunk_size):
    """
    Returns a list of (episode_idx, frame_idx_in_episode, global_idx).

    If both --episodes and --random-n are given, the explicit pairs come first,
    then the random ones are appended.
    """
    ep_to_indices = _build_episode_index(dataset)
    samples = []

    # ── Explicit pairs ──────────────────────────────────────────────────────
    if episodes_str:
        ep_list = [int(e) for e in episodes_str.split(",")]
        fr_list = [int(f) for f in frames_str.split(",")] if frames_str else [0] * len(ep_list)
        if len(fr_list) == 1:
            fr_list = fr_list * len(ep_list)

        for ep_idx, fr_idx in zip(ep_list, fr_list):
            if ep_idx not in ep_to_indices:
                logging.warning(f"Episode {ep_idx} not found in dataset, skipping.")
                continue
            indices = ep_to_indices[ep_idx]
            if fr_idx >= len(indices):
                logging.warning(
                    f"Frame {fr_idx} out of range for episode {ep_idx} "
                    f"({len(indices)} frames), skipping."
                )
                continue
            global_idx = indices[fr_idx]
            samples.append((ep_idx, fr_idx, global_idx))

    # ── Random sampling ─────────────────────────────────────────────────────
    if random_n:
        all_global = list(range(len(dataset)))
        random.shuffle(all_global)
        added = 0
        existing_globals = {g for _, _, g in samples}
        for global_idx in all_global:
            if added >= random_n:
                break
            if global_idx in existing_globals:
                continue
            item = dataset.hf_dataset[global_idx]
            ep_idx = item["episode_index"].item()
            fr_idx = item["frame_index"].item()
            # Ensure the frame has room for at least a partial chunk
            indices = ep_to_indices[ep_idx]
            remaining = len(indices) - indices.index(global_idx)
            if remaining < 1:
                continue
            samples.append((ep_idx, fr_idx, global_idx))
            existing_globals.add(global_idx)
            added += 1

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Data extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_frame_data(dataset, global_idx, chunk_size):
    """
    Returns:
        obs (dict[str, Tensor]):  observation tensors with batch dim [1, ...]
        gt_actions (Tensor):      [chunk_size, action_dim] — raw (unnormalised)
        state (Tensor | None):    [action_dim] — current joint state (unnormalised), if available
        gt_subtask (str):         subtask label from dataset metadata
        task_str (str):           high-level task string
        episode_idx (int)
        frame_idx (int)
    """
    frame = dataset[global_idx]
    episode_idx = frame["episode_index"].item()
    frame_idx = frame["frame_index"].item()
    task_str = frame.get("task", "")

    # ── GT action chunk ─────────────────────────────────────────────────────
    # Read actions directly from hf_dataset (parquet) to skip image decoding.
    gt_actions = []
    for offset in range(chunk_size):
        candidate_idx = global_idx + offset
        if candidate_idx >= len(dataset):
            break
        f_item = dataset.hf_dataset[candidate_idx]
        if f_item["episode_index"].item() != episode_idx:
            break
        is_pad = f_item.get("action_is_pad", False)
        if isinstance(is_pad, torch.Tensor):
            is_pad = is_pad.item()
        if is_pad:
            break
        gt_actions.append(torch.tensor(f_item["action"]))

    if not gt_actions:
        gt_actions = [torch.zeros_like(frame["action"])]

    # Repeat last valid action to pad to chunk_size
    while len(gt_actions) < chunk_size:
        gt_actions.append(gt_actions[-1].clone())
    gt_actions = torch.stack(gt_actions[:chunk_size])  # [chunk_size, action_dim]

    # ── Observations (add batch dim) ────────────────────────────────────────
    obs = {
        k: v.unsqueeze(0)
        for k, v in frame.items()
        if k.startswith("observation.") and isinstance(v, torch.Tensor)
    }

    # ── Current joint state ──────────────────────────────────────────────────
    state = None
    state_key = "observation.state"
    if state_key in frame:
        state = frame[state_key].float()  # [state_dim]

    # ── GT subtask ───────────────────────────────────────────────────────────
    subtask_idx = -1
    for key in ("subtask_index", "complementary_info.subtask_index"):
        if key in frame:
            val = frame[key]
            subtask_idx = val.item() if isinstance(val, torch.Tensor) else int(val)
            break
    gt_subtask = hydrate_subtasks([subtask_idx], dataset)[0]

    return obs, gt_actions, state, gt_subtask, task_str, episode_idx, frame_idx


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(policy, preprocessor, postprocessor, obs, task_str, device, state=None):
    """
    Run one forward pass.
    Returns:
        pred_actions_unnorm (Tensor):  [chunk_size, action_dim] — unnormalised, float32 CPU
        pred_actions_norm (Tensor):    [chunk_size, action_dim] — normalised (model space), float32 CPU
        pred_subtask (str):            decoded subtask text
    """
    batch_size = 1

    complementary_data = {
        "task": [task_str],
        "subtask": [""],
        "advantage": torch.zeros(batch_size, 1, device=device),
    }
    # Dummy action — only used by the preprocessor for FAST tokenisation,
    # which we don't need for inference.
    dummy_action = torch.zeros(batch_size, 1, 6, device=device)

    batch_for_proc = {
        TransitionKey.ACTION: dummy_action,
        **{k: v.to(device) for k, v in obs.items()},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }
    processed = preprocessor(batch_for_proc)

    # Image preprocessing expects raw [0, 1] normalised tensors
    images, img_masks = policy._preprocess_images({k: v.to(device) for k, v in obs.items()})

    task_tokens = processed[OBS_LANGUAGE_TOKENS]
    task_masks = processed[OBS_LANGUAGE_ATTENTION_MASK]

    # ── Subtask generation ───────────────────────────────────────────────────
    subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
        images, img_masks, task_tokens, task_masks
    )
    tokenizer = policy.model._paligemma_tokenizer
    valid_tokens = subtask_tokens[0][subtask_masks[0]]
    pred_subtask = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()

    # ── Action sampling ──────────────────────────────────────────────────────
    pred_actions = policy.model.sample_actions(
        images, img_masks, task_tokens, task_masks, subtask_tokens, subtask_masks
    )
    # Slice to real action dim (6D), same as _prepare_batch / log_sampled_actions
    action_dim = policy.config.output_features[ACTION].shape[0]
    pred_actions = pred_actions[:, :, :action_dim].squeeze(0)  # [chunk_size, action_dim]
    pred_actions_norm = pred_actions.float().cpu()

    # ── Unnormalise ──────────────────────────────────────────────────────────
    unnorm = postprocessor(pred_actions.unsqueeze(0).float())
    pred_actions_unnorm = unnorm.squeeze(0).float().cpu()

    # Reconstruct absolute actions from offsets
    action_encoding = getattr(policy.config, "action_encoding", "absolute")
    if state is not None and action_encoding in ["anchor", "delta"]:
        anchor_val = state[:action_dim].unsqueeze(0).cpu()  # [1, action_dim]
        if action_encoding == "anchor":
            pred_actions_unnorm = pred_actions_unnorm + anchor_val
        elif action_encoding == "delta":
            pred_actions_unnorm = torch.cumsum(pred_actions_unnorm, dim=0) + anchor_val

    return pred_actions_unnorm, pred_actions_norm, pred_subtask


@torch.no_grad()
def _find_normalizer_step(preprocessor):
    """Return the NormalizerProcessorStep from the preprocessor pipeline.

    Searches by duck-type so that step ordering changes don't break this.
    """
    for step in preprocessor.steps:
        if hasattr(step, "norm_map") and hasattr(step, "_tensor_stats"):
            return step
    raise RuntimeError(
        "Could not find a normalizer step in the preprocessor pipeline. "
        f"Steps: {[type(s).__name__ for s in preprocessor.steps]}"
    )


@torch.no_grad()
def normalize_gt(preprocessor, gt_actions, state, device, action_encoding="absolute"):
    """
    Normalize gt_actions (and state if provided) using the same stats as the model.
    To support delta/anchor properly, it converts actions to relative offsets.

    Calls only the NormalizerProcessorStep directly, bypassing all other pipeline
    steps.

    Returns:
        gt_actions_norm (Tensor):  [chunk_size, action_dim]
        state_norm (Tensor|None):  [action_dim]
    """
    norm_step = _find_normalizer_step(preprocessor)
    
    # Calculate offset representations
    if state is not None and action_encoding in ["anchor", "delta"]:
        anchor_val = state[:gt_actions.shape[-1]].unsqueeze(0).cpu()
        if action_encoding == "anchor":
            processed_gt = gt_actions - anchor_val
        elif action_encoding == "delta":
            d_0 = gt_actions[0:1] - anchor_val
            if gt_actions.shape[0] > 1:
                d_rest = torch.diff(gt_actions, dim=0)
                processed_gt = torch.cat([d_0, d_rest], dim=0)
            else:
                processed_gt = d_0
    else:
        processed_gt = gt_actions

    batch = {TransitionKey.ACTION: processed_gt.unsqueeze(0).to(device)}

    processed = norm_step(batch)
    
    gt_actions_norm = processed[TransitionKey.ACTION].squeeze(0).float().cpu()

    return gt_actions_norm, None


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

_GT_COLOR = "#3A86FF"          # blue
_CKPT_COLORS = ["#FF6B35", "#2EC4B6", "#9B5DE5", "#F15BB5"]   # orange, teal, purple, pink


def smooth_actions(actions, window_size):
    """Applies a centered moving average over the sequence dimension."""
    if actions.shape[0] < window_size:
        return actions
    import torch.nn.functional as F
    pad = window_size // 2
    padded = torch.cat([actions[:1].repeat(pad, 1), actions, actions[-1:].repeat(pad, 1)], dim=0)
    x = padded.t().unsqueeze(1) # [D, 1, T]
    weight = torch.ones(1, 1, window_size, device=actions.device, dtype=actions.dtype) / window_size
    smoothed = F.conv1d(x, weight).squeeze(1).t()
    return smoothed


def render_sample(
    obs,
    gt_actions,
    checkpoints_info,      # list of dict: {"label": "A", "subtask": "...", "color_idx": 0}
    pred_traces,           # list of dict: {"actions": tensor, "label": "A", "color_idx": 0, "kwargs": dict}
    gt_subtask,
    episode_idx,
    frame_idx,
    output_dir,
    joint_names=None,
    checkpoint_paths=None, # dict {label: full_path_str} for display in info panel
    state=None,            # Tensor [action_dim] or [chunk_size, action_dim]
):
    """
    Save one evaluation figure.

    Layout:
      Row 0, cols 0..K-1  — camera images  (K = n_cameras)
      Row 0, col -1       — subtask info box  (when a spare column exists)
      Rows 1-2, 3 cols    — 2×3 joint action traces (all checkpoints overlaid)
    """
    from matplotlib.gridspec import GridSpec

    camera_keys = sorted(k for k in obs if "images" in k)
    n_cameras = len(camera_keys)
    n_joints = gt_actions.shape[-1]
    chunk_size = gt_actions.shape[0]
    steps = np.arange(chunk_size)

    n_cols = 3  # fixed: joints live in a 2×3 grid
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(
        3, n_cols, figure=fig,
        height_ratios=[1.8, 1.0, 1.0],
        hspace=0.50, wspace=0.35,
        top=0.91, bottom=0.07, left=0.07, right=0.97,
    )

    # ── Camera images ────────────────────────────────────────────────────────
    for i, key in enumerate(camera_keys[:n_cols]):
        ax = fig.add_subplot(gs[0, i])
        img = obs[key].squeeze(0)
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
        img = img.float().numpy()
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(key.split(".")[-1], fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")

    # ── Subtask info box (spare top-right column when cameras < n_cols) ──────
    spare_cols = list(range(min(n_cameras, n_cols), n_cols))
    if spare_cols:
        ax_info = fig.add_subplot(gs[0, spare_cols[0]])
        ax_info.axis("off")

        def _info_text(ax, y, text, color, fontsize=7.5, bold=False):
            """Render text with simple word-wrap, return new y after last line."""
            words = text.split()
            line_buf, line_h = [], y
            for word in words:
                line_buf.append(word)
                if len(" ".join(line_buf)) > 30:
                    ax.text(
                        0.06, line_h, " ".join(line_buf[:-1]),
                        transform=ax.transAxes, fontsize=fontsize, va="top",
                        color=color, fontweight="bold" if bold else "normal",
                    )
                    line_buf = [word]
                    line_h -= 0.09
            if line_buf:
                ax.text(
                    0.06, line_h, " ".join(line_buf),
                    transform=ax.transAxes, fontsize=fontsize, va="top",
                    color=color, fontweight="bold" if bold else "normal",
                )
            return line_h - 0.09

        y = 0.97
        # GT subtask
        ax_info.text(
            0.06, y, "GT subtask:", transform=ax_info.transAxes,
            fontsize=8, fontweight="bold", color="#555555", va="top",
        )
        y -= 0.10
        y = _info_text(ax_info, y, gt_subtask or "(none)", "#333333")
        y -= 0.06

        # Per-checkpoint: path + predicted subtask
        for info in checkpoints_info:
            label = info["label"]
            pred_subtask = info["subtask"]
            color = _CKPT_COLORS[info["color_idx"] % len(_CKPT_COLORS)]
            path_str = (checkpoint_paths or {}).get(label, "")
            ax_info.text(
                0.06, y, f"Checkpoint {label}:", transform=ax_info.transAxes,
                fontsize=8, fontweight="bold", color=color, va="top",
            )
            y -= 0.10
            if path_str:
                y = _info_text(ax_info, y, path_str, "#555555", fontsize=6.5)
                y -= 0.04
            ax_info.text(
                0.06, y, "subtask:", transform=ax_info.transAxes,
                fontsize=7, color="#666666", va="top", style="italic",
            )
            y -= 0.09
            y = _info_text(ax_info, y, pred_subtask or "(empty)", "#333333")
            y -= 0.10

    # ── 2×3 joint action traces (all checkpoints overlaid per joint) ─────────
    for j in range(min(n_joints, 6)):
        row = (j // 3) + 1
        col = j % 3
        ax = fig.add_subplot(gs[row, col])

        # Current state — horizontal reference line or curve
        if state is not None and j < state.shape[-1]:
            if state.dim() == 1:
                ax.axhline(
                    state[j].item(), color="#888888", linewidth=1.2,
                    linestyle="-", alpha=0.55, zorder=1,
                    label="state(t)" if j == 0 else "_nolegend_",
                )
            elif state.dim() == 2:
                ax.plot(
                    steps, state[:, j].numpy(),
                    label="state(t)" if j == 0 else "_nolegend_",
                    color="#888888", linewidth=1.2, linestyle="-",
                    alpha=0.55, zorder=1,
                )

        ax.plot(
            steps, gt_actions[:, j].numpy(),
            label="GT", color=_GT_COLOR, linewidth=1.5, zorder=100,
        )
        for t_idx, trace in enumerate(pred_traces):
            color = _CKPT_COLORS[trace["color_idx"] % len(_CKPT_COLORS)]
            ax.plot(
                steps, trace["actions"][:, j].numpy(),
                label=trace["label"] if j == 0 else "_nolegend_",
                color=color,
                zorder=50 - t_idx,
                **trace.get("kwargs", {})
            )

        jname = joint_names[j] if joint_names and j < len(joint_names) else f"joint_{j}"
        ax.set_title(jname, fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25, linewidth=0.5, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if j == 0:
            ax.legend(fontsize=7, framealpha=0.85, loc="best", handlelength=1.8)

    # ── Figure title ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"Episode {episode_idx}  |  Frame {frame_idx}",
        fontsize=11, fontweight="bold", ha="left", x=0.01,
    )

    fname = f"ep{episode_idx:04d}_fr{frame_idx:04d}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def eval_cli(cfg: EvalOfflineConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)

    output_dir = cfg.eval_output_dir
    chunk_size = cfg.policy.n_action_steps
    checkpoint_b = cfg.eval_checkpoint_b
    random_n = cfg.eval_random_n or None

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    # Short labels used in plot legends; full paths shown in info panel
    label_a, label_b = "A", "B"
    path_a = str(cfg.policy.pi05_checkpoint) if cfg.policy.pi05_checkpoint else "unknown"
    path_b = str(checkpoint_b) if checkpoint_b else "unknown"

    # ── Load primary checkpoint ───────────────────────────────────────────────
    logging.info("Loading policy A ...")
    policy_a, pre_a, post_a, dataset = _load_policy_and_processors(cfg, device)

    # ── Build sample list ─────────────────────────────────────────────────────
    samples = build_sample_list(
        dataset,
        episodes_str=cfg.eval_episodes,
        frames_str=cfg.eval_frames,
        random_n=random_n,
        chunk_size=chunk_size,
    )
    if not samples:
        raise ValueError(
            "No samples selected. Set eval_episodes and/or eval_random_n in your config."
        )
    logging.info(f"Evaluating {len(samples)} frames ...")

    # ── Output subdirectories ─────────────────────────────────────────────────
    dir_unnorm = os.path.join(output_dir, "unnormalized_eval")
    dir_norm   = os.path.join(output_dir, "normalized_eval")
    os.makedirs(dir_unnorm, exist_ok=True)
    os.makedirs(dir_norm,   exist_ok=True)

    # ── Run inference with policy A ───────────────────────────────────────────
    # Stored per global_idx:
    #   (obs, gt_actions, gt_actions_norm, state, state_norm,
    #    gt_subtask, task_str, pred_unnorm, pred_norm, pred_subtask)
    results_a = {}
    mse_a_list = []
    for ep_idx, fr_idx, global_idx in samples:
        obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size
        )
        pred_unnorm, pred_norm, pred_subtask = run_inference(
            policy_a, pre_a, post_a, obs, task_str, device, state=state
        )
        action_encoding = getattr(policy_a.config, "action_encoding", "absolute")
        gt_actions_norm, state_norm = normalize_gt(pre_a, gt_actions, state, device, action_encoding=action_encoding)
        mse = torch.nn.functional.mse_loss(pred_unnorm, gt_actions.float()).item()
        mse_a_list.append(mse)
        results_a[global_idx] = (
            obs, gt_actions, gt_actions_norm, state, state_norm,
            gt_subtask, task_str, pred_unnorm, pred_norm, pred_subtask,
        )
        logging.info(
            f"  ep={ep_idx:04d} fr={fr_idx:04d} | mse={mse:.4f} | "
            f"GT: '{gt_subtask}' | pred: '{pred_subtask}'"
        )

    # ── Optional: load checkpoint B and run inference ─────────────────────────
    # Load sequentially to avoid holding two large models in GPU memory.
    results_b = {}
    if checkpoint_b:
        del policy_a
        torch.cuda.empty_cache()

        logging.info("Loading policy B ...")
        cfg.policy.pi05_checkpoint = checkpoint_b
        policy_b, pre_b, post_b, _ = _load_policy_and_processors(cfg, device, dataset=dataset)

        mse_b_list = []
        for ep_idx, fr_idx, global_idx in samples:
            obs, gt_actions, _, state, _, _, task_str, _, _, _ = results_a[global_idx]
            pred_unnorm, pred_norm, pred_subtask = run_inference(
                policy_b, pre_b, post_b, obs, task_str, device, state=state
            )
            mse = torch.nn.functional.mse_loss(pred_unnorm, gt_actions.float()).item()
            mse_b_list.append(mse)
            results_b[global_idx] = (pred_unnorm, pred_norm, pred_subtask)
            logging.info(f"  [B] ep={ep_idx:04d} fr={fr_idx:04d} | mse={mse:.4f} | pred: '{pred_subtask}'")

        del policy_b
        torch.cuda.empty_cache()

    # ── MSE summary ───────────────────────────────────────────────────────────
    mean_mse_a = sum(mse_a_list) / len(mse_a_list)
    summary = f"MSE  A ({path_a}): {mean_mse_a:.4f}"
    if results_b:
        mean_mse_b = sum(mse_b_list) / len(mse_b_list)
        summary += f"  |  B ({path_b}): {mean_mse_b:.4f}"
    logging.info(summary)

    # ── Render plots ──────────────────────────────────────────────────────────
    action_dim = next(iter(results_a.values()))[1].shape[-1]
    joint_names = SO100_JOINT_NAMES[:action_dim]
    checkpoint_paths = {label_a: path_a}
    if checkpoint_b:
        checkpoint_paths[label_b] = path_b

    for ep_idx, fr_idx, global_idx in samples:
        (obs, gt_actions, gt_actions_norm, state, state_norm,
         gt_subtask, task_str, pred_a_unnorm, pred_a_norm, sub_a) = results_a[global_idx]

        common_kwargs = dict(
            obs=obs,
            gt_subtask=gt_subtask,
            episode_idx=ep_idx,
            frame_idx=fr_idx,
            joint_names=joint_names,
            checkpoint_paths=checkpoint_paths,
        )

        checkpoints_info = [
            {"label": label_a, "subtask": sub_a, "color_idx": 0}
        ]
        if global_idx in results_b:
            checkpoints_info.append({"label": label_b, "subtask": results_b[global_idx][2], "color_idx": 1})

        def build_traces(pred_a, pred_b=None):
            traces = []
            # Policy A
            traces.append({"actions": pred_a, "label": f"{label_a} raw", "color_idx": 0,
                           "kwargs": {"linewidth": 1.2, "linestyle": "-", "alpha": 1.0}})
            traces.append({"actions": smooth_actions(pred_a, 5), "label": f"{label_a} (w=5)", "color_idx": 0,
                           "kwargs": {"linewidth": 1.0, "linestyle": "-", "alpha": 0.4}})
            # Policy B
            if pred_b is not None:
                traces.append({"actions": pred_b, "label": f"{label_b} raw", "color_idx": 1,
                               "kwargs": {"linewidth": 1.2, "linestyle": "-", "alpha": 1.0}})
                traces.append({"actions": smooth_actions(pred_b, 5), "label": f"{label_b} (w=5)", "color_idx": 1,
                               "kwargs": {"linewidth": 1.0, "linestyle": "-", "alpha": 0.4}})
            return traces

        pred_b_unnorm = results_b[global_idx][0] if global_idx in results_b else None
        pred_b_norm   = results_b[global_idx][1] if global_idx in results_b else None

        render_sample(
            **common_kwargs,
            gt_actions=gt_actions,
            checkpoints_info=checkpoints_info,
            pred_traces=build_traces(pred_a_unnorm, pred_b_unnorm),
            output_dir=dir_unnorm,
            state=state,
        )
        render_sample(
            **common_kwargs,
            gt_actions=gt_actions_norm,
            checkpoints_info=checkpoints_info,
            pred_traces=build_traces(pred_a_norm, pred_b_norm),
            output_dir=dir_norm,
            state=None,
        )

    logging.info(f"Done. {len(samples)} plots saved to {dir_unnorm}/ and {dir_norm}/")


if __name__ == "__main__":
    eval_cli()
