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
from lerobot.types import TransitionKey
from lerobot.rl.pi05_train_utils import (
    hydrate_subtasks,
    make_pi05_full_processors_with_upgrade,
)
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


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
    # Seed for random frame sampling — set to any int for reproducible picks
    eval_random_seed: Optional[int] = None
    # Where to save evaluation plots
    eval_output_dir: str = "outputs/eval_offline"
    # Path to a second checkpoint for side-by-side comparison (optional)
    eval_checkpoint_b: Optional[str] = None
    # When True: run each checkpoint with both "positive" (1.0) and "negative" (-1.0) advantage
    # overlaid as solid/dashed traces. Mutually exclusive with eval_compare_subtasks.
    eval_compare_advantage: bool = False
    # When True: run each checkpoint with three subtask conditions per frame —
    #   "gen"    — model-generated subtask (solid)
    #   "gt"     — ground-truth subtask label from the dataset (dashed)
    #   "manual" — string from MANUAL_SUBTASKS dict below, skipped if no entry (dotted)
    # Mutually exclusive with eval_compare_advantage.
    eval_compare_subtasks: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_policy_and_processors(cfg, device, dataset=None):
    """
    Load policy and pre/post-processors.
    Follows the same pattern as offline_learner_pi05.py.

    If `dataset` is not provided, it is loaded from cfg.  When
    ``cfg.val_dataset_path`` is set the dataset is loaded from that path
    (same logic the probe scripts use via ``offline_val_pi05``).
    Pass `dataset` in on subsequent calls (e.g. checkpoint B) to avoid reloading.
    """
    if dataset is None:
        val_path = getattr(cfg, "val_dataset_path", None)
        if val_path:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            logging.info(f"Loading eval dataset from val_dataset_path: {val_path}")
            dataset = LeRobotDataset(
                repo_id=cfg.dataset.repo_id,
                root=val_path,
            )
        else:
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


def build_sample_list(dataset, episodes_str, frames_str, random_n, chunk_size, seed=None):
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
        rng = random.Random(seed)
        all_global = list(range(len(dataset)))
        rng.shuffle(all_global)
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
        gt_actions.append(f_item["action"].detach().clone())

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
def run_inference(policy, preprocessor, postprocessor, obs, task_str, device, state=None,
                  advantage: float = 1.0,
                  injected_subtask_tokens=None, injected_subtask_masks=None):
    """
    Run one forward pass.

    Args:
        advantage: scalar passed to the processor. The processor applies tanh + binning:
            ≥ ~0.37 → "Advantage: positive", otherwise → "Advantage: negative".
            Default 1.0 (positive) matches how the model was trained on golden data.
        injected_subtask_tokens: if provided ([1, max_len] long), skip generate_subtask_tokens
            and use these tokens directly for action sampling.
        injected_subtask_masks: required when injected_subtask_tokens is provided ([1, max_len] bool).

    Returns:
        pred_actions_unnorm (Tensor):  [chunk_size, action_dim] — unnormalised, float32 CPU
        pred_actions_norm (Tensor):    [chunk_size, action_dim] — normalised (model space), float32 CPU
        pred_subtask (str):            decoded subtask text (generated or decoded from injection)
    """
    batch_size = 1

    complementary_data = {
        "task": [task_str],
        "subtask": [""],
        "advantage": torch.tensor([[advantage]], device=device),
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

    # ── Subtask generation (or injection) ────────────────────────────────────
    tokenizer = policy.model._paligemma_tokenizer
    if injected_subtask_tokens is not None:
        subtask_tokens = injected_subtask_tokens.to(device)
        subtask_masks  = injected_subtask_masks.to(device)
        valid_tokens   = subtask_tokens[0][subtask_masks[0]]
        pred_subtask   = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
    else:
        subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
            images, img_masks, task_tokens, task_masks
        )
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


def tokenize_subtask(subtask_str: str, tokenizer, max_len: int, device):
    """
    Tokenize a raw subtask label into the exact format that generate_subtask_tokens returns.

    The processor formats subtask labels as "Subtask: {text};\n" before tokenizing during
    training. We replicate that here so injected tokens match the training distribution.
    The tokenizer was loaded with add_eos_token=True, add_bos_token=False, so encoding
    produces [...content tokens..., EOS] — matching what generate_subtask_tokens returns
    (BOS is used as the generation seed but is not included in the returned tensor).

    Returns:
        tokens: [1, max_len] long
        masks:  [1, max_len] bool
    """
    formatted = f"Subtask: {subtask_str.strip().lower()};\n"
    encoded = tokenizer.encode(formatted, add_special_tokens=True)
    n = min(len(encoded), max_len)
    tokens = torch.zeros(1, max_len, dtype=torch.long, device=device)
    masks  = torch.zeros(1, max_len, dtype=torch.bool,  device=device)
    tokens[0, :n] = torch.tensor(encoded[:n], dtype=torch.long, device=device)
    masks[0,  :n] = True
    return tokens, masks


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

_GT_COLOR = "#3A86FF"          # blue
_CKPT_COLORS = ["#FF6B35", "#2EC4B6", "#9B5DE5", "#F15BB5"]   # orange, teal, purple, pink

# Plot style per condition label.
# Info-panel style tags (unicode line characters) mirror the linestyle.
_CONDITION_STYLE: dict[str, dict] = {
    # advantage-sweep conditions
    "pos":    {"linewidth": 1.4, "linestyle": "-",  "alpha": 1.00},
    "neg":    {"linewidth": 1.4, "linestyle": "--", "alpha": 0.80},
    # subtask-sweep conditions
    "gen":    {"linewidth": 1.4, "linestyle": "-",  "alpha": 1.00},
    "gt":     {"linewidth": 1.4, "linestyle": "--", "alpha": 0.80},
    "manual": {"linewidth": 1.4, "linestyle": ":",  "alpha": 0.80},
    # default (single-condition)
    "pred":   {"linewidth": 1.4, "linestyle": "-",  "alpha": 1.00},
}
_CONDITION_TAG: dict[str, str] = {
    "pos": "─", "neg": "╌",
    "gen": "─", "gt":  "╌", "manual": "·",
    "pred": "─",
}

# ── Manual subtask overrides ──────────────────────────────────────────────────
# Fill in entries here when running with eval_compare_subtasks=True.
# Keys are (episode_idx, frame_idx). Frames without an entry skip the "manual" condition.
# The string should be the raw subtask label (e.g. "grasp red truck"), NOT the full
# "Subtask: ...;\n" format — the script applies that formatting automatically.
MANUAL_SUBTASKS: dict[tuple[int, int], str] = {
    # (57, 222): "grasp red truck",
}

# Advantage conditions used when eval_compare_advantage=True.
_ADVANTAGE_CONDITIONS = [("pos", 1.0), ("neg", -1.0)]


def smooth_actions(actions, window_size):
    """Applies a centered moving average over the sequence dimension."""
    if actions.shape[0] < window_size:
        return actions
    import torch.nn.functional as F
    pad = window_size // 2
    padded = torch.cat([actions[:1].repeat(pad, 1), actions, actions[-1:].repeat(pad, 1)], dim=0)
    x = padded.t().unsqueeze(1)
    weight = torch.ones(1, 1, window_size, device=actions.device, dtype=actions.dtype) / window_size
    smoothed = F.conv1d(x, weight).squeeze(1).t()
    return smoothed


def render_sample(
    obs,
    gt_actions,
    checkpoints_info,      # list of dict: {"label": "A", "subtasks": {"pos": "...", "neg": "..."}, "color_idx": 0}
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

    # ── Info panel (spare top-right column when cameras < n_cols) ───────────
    spare_cols = list(range(min(n_cameras, n_cols), n_cols))
    if spare_cols:
        import textwrap

        ax_info = fig.add_subplot(gs[0, spare_cols[0]])
        ax_info.axis("off")
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        n_ckpts = len(checkpoints_info)
        # Two-column layout: A left, B right. Single checkpoint uses full width.
        if n_ckpts >= 2:
            col_x   = [0.02, 0.52]   # left edge of each checkpoint column (axes coords)
            wrap_w  = 20              # chars per column before wrapping
        else:
            col_x   = [0.02]
            wrap_w  = 40

        def _wrap(text, max_lines=3):
            """Wrap at wrap_w chars (breaking long tokens too), cap at max_lines."""
            lines = textwrap.wrap(
                text or "(empty)", width=wrap_w,
                break_long_words=True, break_on_hyphens=False,
            )
            if not lines:
                return ["(empty)"]
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = lines[-1][:wrap_w - 1] + "…"
            return lines

        def _draw_lines(x, y, lines, color, fontsize=7.0, step=0.085):
            for line in lines:
                ax_info.text(x, y, line, transform=ax_info.transAxes,
                             fontsize=fontsize, va="top", color=color, clip_on=True)
                y -= step
            return y

        # GT subtask — full width at top
        y_gt = 0.97
        ax_info.text(0.02, y_gt, "GT subtask:", transform=ax_info.transAxes,
                     fontsize=7.5, fontweight="bold", color="#555555", va="top")
        y_gt -= 0.10
        y_gt = _draw_lines(0.02, y_gt, _wrap(gt_subtask or "(none)", max_lines=2), "#333333")

        # Thin separator line
        sep_y = y_gt - 0.02
        ax_info.plot([0.02, 0.98], [sep_y, sep_y], transform=ax_info.transAxes,
                     color="#cccccc", linewidth=0.6, clip_on=True)

        # Per-checkpoint columns
        for col_idx, info in enumerate(checkpoints_info[:2]):
            x     = col_x[col_idx]
            y     = sep_y - 0.06
            label = info["label"]
            color = _CKPT_COLORS[info["color_idx"] % len(_CKPT_COLORS)]
            subtasks = info["subtasks"]

            # Header
            ax_info.text(x, y, f"Ckpt {label}", transform=ax_info.transAxes,
                         fontsize=7.5, fontweight="bold", color=color, va="top")
            y -= 0.10

            # Path — last 2 path components only, truncated to wrap_w chars
            path_str = (checkpoint_paths or {}).get(label, "")
            if path_str:
                parts     = path_str.replace("\\", "/").split("/")
                short_path = "/".join(parts[-2:]) if len(parts) >= 2 else path_str
                if len(short_path) > wrap_w:
                    short_path = "…" + short_path[-(wrap_w - 1):]
                ax_info.text(x, y, short_path, transform=ax_info.transAxes,
                             fontsize=6.0, color="#888888", va="top", clip_on=True)
                y -= 0.09

            # Predicted subtask per condition
            for cond_label, sub in subtasks.items():
                style_tag = _CONDITION_TAG.get(cond_label, "─")
                ax_info.text(x, y, f"{style_tag} {cond_label}:", transform=ax_info.transAxes,
                             fontsize=6.5, color="#777777", va="top", style="italic")
                y -= 0.09
                y = _draw_lines(x, y, _wrap(sub, max_lines=2), "#333333", fontsize=6.5)
                y -= 0.04

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
    logging.debug(f"  Saved {out_path}")


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
    compare_advantage = cfg.eval_compare_advantage
    compare_subtasks  = cfg.eval_compare_subtasks

    # Determine which advantage conditions to run.
    # Default mode: single "pos" (=positive) inference — fixes the original bug where 0 was passed.
    # Advantage-sweep mode: both "pos" and "neg" overlaid with solid/dashed linestyles.
    adv_conditions = _ADVANTAGE_CONDITIONS if compare_advantage else [("pos", 1.0)]

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
        seed=cfg.eval_random_seed,
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
    # frame_data[global_idx]: per-frame data shared across all conditions.
    # preds_a[global_idx]: {adv_label: (pred_unnorm, pred_norm, pred_subtask)}
    frame_data = {}
    preds_a = {}
    mse_a = {lbl: [] for lbl, _ in adv_conditions}
    action_encoding_a = getattr(policy_a.config, "action_encoding", "absolute")

    for ep_idx, fr_idx, global_idx in samples:
        obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size
        )
        gt_actions_norm, _ = normalize_gt(pre_a, gt_actions, state, device, action_encoding=action_encoding_a)
        frame_data[global_idx] = {
            "obs": obs, "gt_actions": gt_actions, "gt_actions_norm": gt_actions_norm,
            "state": state, "gt_subtask": gt_subtask, "task_str": task_str,
        }

        preds_a[global_idx] = {}
        for adv_label, adv_val in adv_conditions:
            pred_unnorm, pred_norm, pred_subtask = run_inference(
                policy_a, pre_a, post_a, obs, task_str, device,
                state=state, advantage=adv_val,
            )
            mse = torch.nn.functional.mse_loss(pred_unnorm, gt_actions.float()).item()
            mse_a[adv_label].append(mse)
            preds_a[global_idx][adv_label] = (pred_unnorm, pred_norm, pred_subtask)

        mse_str = "  ".join(f"mse_{lbl}={mse_a[lbl][-1]:.4f}" for lbl, _ in adv_conditions)
        pred_sub_log = preds_a[global_idx]["pos"][2]
        logging.info(
            f"  ep={ep_idx:04d} fr={fr_idx:04d} | {mse_str} | "
            f"GT: '{gt_subtask}' | pred(pos): '{pred_sub_log}'"
        )

    # ── Optional: load checkpoint B and run inference ─────────────────────────
    # Load sequentially to avoid holding two large models in GPU memory.
    preds_b = {}
    if checkpoint_b:
        del policy_a
        torch.cuda.empty_cache()

        logging.info("Loading policy B ...")
        cfg.policy.pi05_checkpoint = checkpoint_b
        policy_b, pre_b, post_b, _ = _load_policy_and_processors(cfg, device, dataset=dataset)

        mse_b = {lbl: [] for lbl, _ in adv_conditions}
        for ep_idx, fr_idx, global_idx in samples:
            fd = frame_data[global_idx]
            obs, gt_actions, state, task_str = fd["obs"], fd["gt_actions"], fd["state"], fd["task_str"]
            preds_b[global_idx] = {}
            for adv_label, adv_val in adv_conditions:
                pred_unnorm, pred_norm, pred_subtask = run_inference(
                    policy_b, pre_b, post_b, obs, task_str, device,
                    state=state, advantage=adv_val,
                )
                mse = torch.nn.functional.mse_loss(pred_unnorm, gt_actions.float()).item()
                mse_b[adv_label].append(mse)
                preds_b[global_idx][adv_label] = (pred_unnorm, pred_norm, pred_subtask)
            mse_str = "  ".join(f"mse_{lbl}={mse_b[lbl][-1]:.4f}" for lbl, _ in adv_conditions)
            logging.debug(f"  [B] ep={ep_idx:04d} fr={fr_idx:04d} | {mse_str}")

        del policy_b
        torch.cuda.empty_cache()

    # Shared across both advantage and subtask rendering blocks
    action_dim = frame_data[samples[0][2]]["gt_actions"].shape[-1]
    joint_names = SO100_JOINT_NAMES[:action_dim]
    checkpoint_paths = {label_a: path_a}
    if checkpoint_b:
        checkpoint_paths[label_b] = path_b

    # ── Subtask-sweep inference + rendering ──────────────────────────────────
    # Runs completely independently of the advantage sweep above.
    # Produces plots in separate subfolders so nothing is overwritten.
    if compare_subtasks:
        dir_sub_unnorm = os.path.join(output_dir, "subtask_unnormalized_eval")
        dir_sub_norm   = os.path.join(output_dir, "subtask_normalized_eval")
        os.makedirs(dir_sub_unnorm, exist_ok=True)
        os.makedirs(dir_sub_norm,   exist_ok=True)

        # Policy A must still be in scope (not deleted by ckpt-B path).
        # If checkpoint_b was used we already deleted policy_a above, so reload it.
        if checkpoint_b:
            cfg.policy.pi05_checkpoint = path_a
            policy_sub_a, pre_sub_a, post_sub_a, _ = _load_policy_and_processors(
                cfg, device, dataset=dataset
            )
        else:
            policy_sub_a, pre_sub_a, post_sub_a = policy_a, pre_a, post_a

        tokenizer = policy_sub_a.model._paligemma_tokenizer
        max_sub_len = policy_sub_a.config.tokenizer_max_length

        def _run_subtask_inference_for_policy(policy, pre, post, label):
            """Run gen/gt/manual subtask conditions for all samples. Returns preds dict."""
            preds = {}
            for ep_idx, fr_idx, global_idx in samples:
                fd = frame_data[global_idx]
                obs, gt_actions, state, gt_subtask_str, task_str = (
                    fd["obs"], fd["gt_actions"], fd["state"], fd["gt_subtask"], fd["task_str"]
                )
                preds[global_idx] = {}

                # "gen" — model generates the subtask normally
                pred_unnorm, pred_norm, gen_subtask_text = run_inference(
                    policy, pre, post, obs, task_str, device, state=state, advantage=1.0,
                )
                preds[global_idx]["gen"] = (pred_unnorm, pred_norm, gen_subtask_text)

                # "gt" — inject the ground-truth subtask label
                gt_tokens, gt_masks = tokenize_subtask(
                    gt_subtask_str, tokenizer, max_sub_len, device
                )
                pred_unnorm, pred_norm, _ = run_inference(
                    policy, pre, post, obs, task_str, device, state=state, advantage=1.0,
                    injected_subtask_tokens=gt_tokens, injected_subtask_masks=gt_masks,
                )
                preds[global_idx]["gt"] = (pred_unnorm, pred_norm, gt_subtask_str)

                # "manual" — user-provided string, skip frame if no entry
                manual_str = MANUAL_SUBTASKS.get((ep_idx, fr_idx))
                if manual_str is not None:
                    m_tokens, m_masks = tokenize_subtask(
                        manual_str, tokenizer, max_sub_len, device
                    )
                    pred_unnorm, pred_norm, _ = run_inference(
                        policy, pre, post, obs, task_str, device, state=state, advantage=1.0,
                        injected_subtask_tokens=m_tokens, injected_subtask_masks=m_masks,
                    )
                    preds[global_idx]["manual"] = (pred_unnorm, pred_norm, manual_str)

                logging.debug(
                    f"  [{label}] ep={ep_idx:04d} fr={fr_idx:04d} | "
                    f"gen='{gen_subtask_text}' | gt='{gt_subtask_str}'"
                )
            return preds

        logging.info("Subtask sweep — policy A ...")
        preds_sub_a = _run_subtask_inference_for_policy(
            policy_sub_a, pre_sub_a, post_sub_a, label_a
        )

        preds_sub_b = {}
        if checkpoint_b:
            cfg.policy.pi05_checkpoint = path_b
            policy_sub_b, pre_sub_b, post_sub_b, _ = _load_policy_and_processors(
                cfg, device, dataset=dataset
            )
            logging.info("Subtask sweep — policy B ...")
            preds_sub_b = _run_subtask_inference_for_policy(
                policy_sub_b, pre_sub_b, post_sub_b, label_b
            )
            del policy_sub_b
            torch.cuda.empty_cache()

        if checkpoint_b:
            del policy_sub_a
            torch.cuda.empty_cache()

        def build_subtask_traces(ckpt_preds, color_idx, ckpt_label, val_idx):
            traces = []
            for cond_label, vals in ckpt_preds.items():
                traces.append({
                    "actions": vals[val_idx],
                    "label": f"{ckpt_label} {cond_label}",
                    "color_idx": color_idx,
                    "kwargs": _CONDITION_STYLE.get(cond_label, _CONDITION_STYLE["pred"]),
                })
            return traces

        for ep_idx, fr_idx, global_idx in samples:
            fd = frame_data[global_idx]
            obs, gt_actions, gt_actions_norm, state, gt_subtask = (
                fd["obs"], fd["gt_actions"], fd["gt_actions_norm"], fd["state"], fd["gt_subtask"]
            )

            sub_ckpts_info = [
                {
                    "label": label_a,
                    "subtasks": {k: v[2] for k, v in preds_sub_a[global_idx].items()},
                    "color_idx": 0,
                }
            ]
            if global_idx in preds_sub_b:
                sub_ckpts_info.append({
                    "label": label_b,
                    "subtasks": {k: v[2] for k, v in preds_sub_b[global_idx].items()},
                    "color_idx": 1,
                })

            sub_common = dict(
                obs=obs, gt_subtask=gt_subtask,
                episode_idx=ep_idx, frame_idx=fr_idx,
                joint_names=joint_names, checkpoint_paths=checkpoint_paths,
                checkpoints_info=sub_ckpts_info,
            )

            sub_traces_unnorm = build_subtask_traces(preds_sub_a[global_idx], 0, label_a, 0)
            sub_traces_norm   = build_subtask_traces(preds_sub_a[global_idx], 0, label_a, 1)
            if global_idx in preds_sub_b:
                sub_traces_unnorm += build_subtask_traces(preds_sub_b[global_idx], 1, label_b, 0)
                sub_traces_norm   += build_subtask_traces(preds_sub_b[global_idx], 1, label_b, 1)

            render_sample(
                **sub_common, gt_actions=gt_actions,
                pred_traces=sub_traces_unnorm, output_dir=dir_sub_unnorm, state=state,
            )
            render_sample(
                **sub_common, gt_actions=gt_actions_norm,
                pred_traces=sub_traces_norm, output_dir=dir_sub_norm, state=None,
            )

        logging.debug(
            f"Subtask plots saved to {dir_sub_unnorm}/ and {dir_sub_norm}/"
        )

    # ── MSE summary ───────────────────────────────────────────────────────────
    parts = [f"MSE  A ({path_a}):"]
    for lbl, _ in adv_conditions:
        parts.append(f"  {lbl}={sum(mse_a[lbl]) / len(mse_a[lbl]):.4f}")
    if preds_b:
        parts.append(f"  |  B ({path_b}):")
        for lbl, _ in adv_conditions:
            parts.append(f"  {lbl}={sum(mse_b[lbl]) / len(mse_b[lbl]):.4f}")
    logging.info("".join(parts))

    # ── Render plots ──────────────────────────────────────────────────────────
    def build_traces(ckpt_preds, color_idx, ckpt_label, val_idx):
        """Build trace list for one checkpoint across all advantage conditions.

        ckpt_preds: {adv_label: (unnorm, norm, subtask)}
        val_idx: 0 = unnormalized, 1 = normalized.

        Default mode (single "pos" condition): adds raw + smooth to match original behavior.
        Advantage-sweep mode (multiple conditions): adds one trace per condition, no smooth
        (linestyle already encodes the condition; adding smooth would double the line count).
        """
        traces = []
        for adv_label, vals in ckpt_preds.items():
            actions = vals[val_idx]
            if compare_advantage:
                # Linestyle encodes advantage; no smooth to keep the plot readable.
                label = f"{ckpt_label} {adv_label}"
                traces.append({
                    "actions": actions, "label": label, "color_idx": color_idx,
                    "kwargs": _CONDITION_STYLE.get(adv_label, _CONDITION_STYLE["pred"]),
                })
            else:
                # Single condition: show raw + smooth, same as original behavior.
                traces.append({
                    "actions": actions, "label": f"{ckpt_label} raw", "color_idx": color_idx,
                    "kwargs": {"linewidth": 1.2, "linestyle": "-", "alpha": 1.0},
                })
                traces.append({
                    "actions": smooth_actions(actions, 5), "label": f"{ckpt_label} (w=5)",
                    "color_idx": color_idx,
                    "kwargs": {"linewidth": 1.0, "linestyle": "-", "alpha": 0.4},
                })
        return traces

    for ep_idx, fr_idx, global_idx in samples:
        fd = frame_data[global_idx]
        obs          = fd["obs"]
        gt_actions      = fd["gt_actions"]
        gt_actions_norm = fd["gt_actions_norm"]
        state        = fd["state"]
        gt_subtask   = fd["gt_subtask"]

        checkpoints_info = [
            {
                "label": label_a,
                "subtasks": {lbl: preds_a[global_idx][lbl][2] for lbl, _ in adv_conditions},
                "color_idx": 0,
            }
        ]
        if global_idx in preds_b:
            checkpoints_info.append({
                "label": label_b,
                "subtasks": {lbl: preds_b[global_idx][lbl][2] for lbl, _ in adv_conditions},
                "color_idx": 1,
            })

        common_kwargs = dict(
            obs=obs,
            gt_subtask=gt_subtask,
            episode_idx=ep_idx,
            frame_idx=fr_idx,
            joint_names=joint_names,
            checkpoint_paths=checkpoint_paths,
            checkpoints_info=checkpoints_info,
        )

        traces_unnorm = build_traces(preds_a[global_idx], color_idx=0, ckpt_label=label_a, val_idx=0)
        traces_norm   = build_traces(preds_a[global_idx], color_idx=0, ckpt_label=label_a, val_idx=1)
        if global_idx in preds_b:
            traces_unnorm += build_traces(preds_b[global_idx], color_idx=1, ckpt_label=label_b, val_idx=0)
            traces_norm   += build_traces(preds_b[global_idx], color_idx=1, ckpt_label=label_b, val_idx=1)

        render_sample(
            **common_kwargs, gt_actions=gt_actions,
            pred_traces=traces_unnorm, output_dir=dir_unnorm, state=state,
        )
        render_sample(
            **common_kwargs, gt_actions=gt_actions_norm,
            pred_traces=traces_norm, output_dir=dir_norm, state=None,
        )

    logging.debug(f"Done. {len(samples)} plots saved to {dir_unnorm}/ and {dir_norm}/")


if __name__ == "__main__":
    eval_cli()
