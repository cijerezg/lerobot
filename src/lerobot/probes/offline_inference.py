#!/usr/bin/env python
"""
Generic offline inference probe — pick frames, run inference, save per-frame plots.

Policy-agnostic: works for any policy with a registered
:class:`lerobot.probes.base.ProbablePolicy` adapter (pi05, molmoact2, …).

For each selected frame, saves a figure with:
  - Camera images at the sampled timestep
  - 2×3 grid of predicted vs GT action chunk traces (per joint)
  - Subtask info panel (GT + predicted if the policy generates subtasks)
  - Optional checkpoint A vs B overlay

Two output subdirectories:
  unnormalized_eval/   actions in dataset units
  normalized_eval/     actions in normalised model space (GT via adapter.normalize_gt_actions)

Usage examples:
    # Random sample
    python -m lerobot.probes.offline_inference config.yaml --eval_random_n 10

    # Manual frame selection
    python -m lerobot.probes.offline_inference config.yaml \
        --eval_episodes "0,1,5" --eval_frames "10,20,30"

    # Checkpoint comparison (A vs B, sequential to fit one model in GPU at a time)
    python -m lerobot.probes.offline_inference config.yaml \
        --eval_random_n 5 --eval_checkpoint_b /path/to/other/checkpoint
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import assemble_frame_history, build_sample_list, get_frame_data
from lerobot.rl.inference_utils import apply_butterworth_filter
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


# Known action-vector layouts used by the current probe targets.
SO100_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

REBOT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


def _joint_names_for_dim(action_dim: int) -> list[str]:
    if action_dim == len(REBOT_JOINT_NAMES):
        return REBOT_JOINT_NAMES
    if action_dim == len(SO100_JOINT_NAMES):
        return SO100_JOINT_NAMES
    return [f"joint_{i}" for i in range(action_dim)]


@dataclass
class EvalOfflineConfig(TrainRLServerPipelineConfig):
    """Offline-eval extras (all optional)."""

    eval_episodes: Optional[str] = None        # "0,1,5"
    eval_frames: Optional[str] = None          # "10,20,30" (or single value applied to all eps)
    eval_random_n: int = 0                     # number of random frames in addition
    eval_random_seed: Optional[int] = None
    eval_output_dir: str = "outputs/eval_offline"
    eval_checkpoint_b: Optional[str] = None    # path to second checkpoint for A vs B


# ──────────────────────────────────────────────────────────────────────────────
# Visual style
# ──────────────────────────────────────────────────────────────────────────────

_GT_COLOR = "#3A86FF"
_CKPT_COLORS = ["#FF6B35", "#2EC4B6", "#9B5DE5", "#F15BB5"]
_BUTTER_STYLE = {"linewidth": 1.0, "linestyle": "--", "alpha": 0.4}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def smooth_actions(actions: torch.Tensor, window_size: int) -> torch.Tensor:
    """Centered moving average over the sequence dimension."""
    if actions.shape[0] < window_size:
        return actions
    import torch.nn.functional as F
    pad = window_size // 2
    padded = torch.cat([actions[:1].repeat(pad, 1), actions, actions[-1:].repeat(pad, 1)], dim=0)
    x = padded.t().unsqueeze(1)
    weight = torch.ones(1, 1, window_size, device=actions.device, dtype=actions.dtype) / window_size
    return torch.nn.functional.conv1d(x, weight).squeeze(1).t()


def _policy_checkpoint_path(policy_cfg) -> str:
    for field in ("base_path", "pretrained_path"):
        value = getattr(policy_cfg, field, None)
        if value:
            return str(value)
    return "unknown"


def _set_policy_checkpoint_path(policy_cfg, checkpoint: str) -> bool:
    for field in ("base_path", "pretrained_path"):
        if hasattr(policy_cfg, field):
            setattr(policy_cfg, field, checkpoint)
            return True
    return False


def _load_dataset(cfg):
    """Load dataset, honouring cfg.val_dataset_path if set (works for both policies)."""
    from lerobot.datasets.factory import make_dataset
    val_path = getattr(cfg, "val_dataset_path", None)
    if val_path:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logging.info(f"Loading eval dataset from val_dataset_path: {val_path}")
        dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
    else:
        dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_sample(
    obs,
    gt_actions,
    checkpoints_info,    # list of {"label": "A", "subtask": "...", "color_idx": 0, "summary": ...}
    pred_traces,         # list of {"actions": tensor, "label": "A raw", "color_idx": 0, "kwargs": dict}
    gt_subtask,
    episode_idx,
    frame_idx,
    output_dir,
    joint_names=None,
    checkpoint_paths=None,
    state=None,
    gt_summary=None,
):
    """
    Save one evaluation figure.

    Layout:
      Row 0, cols 0..K-1  — camera images
      Row 0, col -1       — subtask info box (when a spare column exists)
      Remaining rows      — all joint action traces (all checkpoints overlaid)
    """
    from matplotlib.gridspec import GridSpec

    camera_keys = sorted(k for k in obs if k.startswith("observation.images."))
    n_cameras = len(camera_keys)
    n_joints = gt_actions.shape[-1]
    chunk_size = gt_actions.shape[0]
    steps = np.arange(chunk_size)

    n_cols = 4 if n_joints > 6 else 3
    n_action_rows = max(1, (n_joints + n_cols - 1) // n_cols)
    fig = plt.figure(figsize=(4.6 * n_cols, 4.8 + 2.4 * n_action_rows))
    gs = GridSpec(
        1 + n_action_rows, n_cols, figure=fig,
        height_ratios=[1.8] + [1.0] * n_action_rows,
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

    # ── Info panel ───────────────────────────────────────────────────────────
    spare_cols = list(range(min(n_cameras, n_cols), n_cols))
    if spare_cols:
        import textwrap
        ax_info = fig.add_subplot(gs[0, spare_cols[0]])
        ax_info.axis("off")
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)

        n_ckpts = len(checkpoints_info)
        col_x  = [0.02, 0.52] if n_ckpts >= 2 else [0.02]
        wrap_w = 20 if n_ckpts >= 2 else 40

        def _wrap(text, max_lines=3):
            lines = textwrap.wrap(text or "(empty)", width=wrap_w,
                                  break_long_words=True, break_on_hyphens=False)
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

        y_gt = 0.97
        ax_info.text(0.02, y_gt, "GT subtask:", transform=ax_info.transAxes,
                     fontsize=7.5, fontweight="bold", color="#555555", va="top")
        y_gt -= 0.10
        y_gt = _draw_lines(0.02, y_gt, _wrap(gt_subtask or "(none)", max_lines=2), "#333333")
        if gt_summary is not None:
            ax_info.text(0.02, y_gt, "GT memory:", transform=ax_info.transAxes,
                         fontsize=7.5, fontweight="bold", color="#555555", va="top")
            y_gt -= 0.10
            y_gt = _draw_lines(0.02, y_gt, _wrap(gt_summary or "(empty)", max_lines=2),
                               "#333333", fontsize=6.5)
        sep_y = y_gt - 0.02
        ax_info.plot([0.02, 0.98], [sep_y, sep_y], transform=ax_info.transAxes,
                     color="#cccccc", linewidth=0.6, clip_on=True)

        for col_idx, info in enumerate(checkpoints_info[:2]):
            x     = col_x[col_idx]
            y     = sep_y - 0.06
            label = info["label"]
            color = _CKPT_COLORS[info["color_idx"] % len(_CKPT_COLORS)]
            ax_info.text(x, y, f"Ckpt {label}", transform=ax_info.transAxes,
                         fontsize=7.5, fontweight="bold", color=color, va="top")
            y -= 0.10

            path_str = (checkpoint_paths or {}).get(label, "")
            if path_str:
                parts      = path_str.replace("\\", "/").split("/")
                short_path = "/".join(parts[-2:]) if len(parts) >= 2 else path_str
                if len(short_path) > wrap_w:
                    short_path = "…" + short_path[-(wrap_w - 1):]
                ax_info.text(x, y, short_path, transform=ax_info.transAxes,
                             fontsize=6.0, color="#888888", va="top", clip_on=True)
                y -= 0.09

            sub = info.get("subtask")
            if sub:
                ax_info.text(x, y, "─ pred:", transform=ax_info.transAxes,
                             fontsize=6.5, color="#777777", va="top", style="italic")
                y -= 0.09
                y = _draw_lines(x, y, _wrap(sub, max_lines=2), "#333333", fontsize=6.5)
            mem = info.get("summary")
            if mem is not None:
                ax_info.text(x, y, "─ mem:", transform=ax_info.transAxes,
                             fontsize=6.5, color="#777777", va="top", style="italic")
                y -= 0.09
                _draw_lines(x, y, _wrap(mem or "(empty)", max_lines=2), "#333333", fontsize=6.5)

    # ── 2×3 joint action traces ──────────────────────────────────────────────
    for j in range(n_joints):
        row = (j // n_cols) + 1
        col = j % n_cols
        ax = fig.add_subplot(gs[row, col])

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

        ax.plot(steps, gt_actions[:, j].numpy(),
                label="GT", color=_GT_COLOR, linewidth=1.5, zorder=100)

        for t_idx, trace in enumerate(pred_traces):
            color = _CKPT_COLORS[trace["color_idx"] % len(_CKPT_COLORS)]
            ax.plot(
                steps, trace["actions"][:, j].numpy(),
                label=trace["label"] if j == 0 else "_nolegend_",
                color=color, zorder=50 - t_idx,
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

    fig.suptitle(f"Episode {episode_idx}  |  Frame {frame_idx}",
                 fontsize=11, fontweight="bold", ha="left", x=0.01)

    fname = f"ep{episode_idx:04d}_fr{frame_idx:04d}.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.debug(f"  Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference per checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def _summary_lookup(dataset, cfg) -> dict[int, tuple[str, str]]:
    """global frame index -> (conditioning summary, GT target summary), using the
    same hold/update-window label rule as training. Empty when the dataset has no
    meta/summaries.parquet."""
    from lerobot.rl.buffer import summary_label_spans
    from lerobot.rl.offline_dataset_utils import load_summary_segments

    segments, texts = load_summary_segments(dataset.root)
    if not segments:
        return {}
    window = max(1, round(getattr(cfg.policy, "subtask_regeneration_interval", 1.0) * dataset.fps))
    lookup: dict[int, tuple[str, str]] = {}
    for start, stop, prev_row, target_row in summary_label_spans(segments, window):
        pair = ("" if prev_row < 0 else texts[prev_row], "" if target_row < 0 else texts[target_row])
        for i in range(start, stop):
            lookup[i] = pair
    return lookup


def _run_checkpoint(adapter: ProbablePolicy, dataset, samples, frame_data, chunk_size, summary_lookup, cfg):
    """
    Run inference for all samples through *adapter*.

    Populates frame_data[global_idx] with shared per-frame fields the first time
    it sees a global_idx. Returns:
        preds:  {global_idx: (pred_unnorm, pred_norm, pred_subtask, generation)}
                where generation is adapter.generate_subtask's tuple or None
        mse:    list of float, one per sample
    """
    preds: dict[int, tuple] = {}
    mse: list[float] = []
    memory_ablation: list[dict] = []

    adapter.suppress_logs(True)
    try:
        for ep_idx, fr_idx, global_idx in samples:
            if global_idx not in frame_data:
                obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
                    dataset, global_idx, chunk_size
                )
                memory_cfg = getattr(cfg.policy, "memory", None)
                if memory_cfg is not None and memory_cfg.history_keys and memory_cfg.history_num_samples > 0:
                    obs.update(
                        assemble_frame_history(
                            dataset,
                            global_idx,
                            memory_cfg,
                            dataset.fps,
                            list(memory_cfg.history_keys),
                        )
                    )
                gt_actions_norm = adapter.normalize_gt_actions(gt_actions, state)
                prev_summary, target_summary = summary_lookup.get(global_idx, (None, None))
                frame_data[global_idx] = {
                    "obs": obs, "gt_actions": gt_actions, "gt_actions_norm": gt_actions_norm,
                    "state": state, "gt_subtask": gt_subtask, "task_str": task_str,
                    "summary_prev": prev_summary, "gt_summary": target_summary,
                }

            fd = frame_data[global_idx]
            pred_unnorm, pred_norm, pred_subtask = adapter.predict_action_chunk(
                fd["obs"], fd["task_str"], state=fd["state"],
                subtask=fd["gt_subtask"], metadata={"quality": 5, "mistake": False},
            )
            generation = adapter.generate_subtask(fd["obs"], fd["task_str"], summary=fd["summary_prev"])
            generation_empty = None
            if generation is not None and fd["summary_prev"] and fd["gt_subtask"]:
                generation_empty = adapter.generate_subtask(fd["obs"], fd["task_str"], summary=None)
                if generation_empty is not None:
                    memory_ablation.append(
                        {
                            "episode_idx": int(ep_idx),
                            "frame_idx": int(fr_idx),
                            "global_idx": int(global_idx),
                            "gt_subtask": fd["gt_subtask"],
                            "gt_memory": fd["summary_prev"],
                            "gt_memory_pred_subtask": generation[1],
                            "empty_memory_pred_subtask": generation_empty[1],
                            "gt_memory_raw_decode": generation[0],
                            "empty_memory_raw_decode": generation_empty[0],
                            "gt_memory_pred_summary": generation[3],
                            "empty_memory_pred_summary": generation_empty[3],
                        }
                    )
            if generation is not None:
                pred_subtask = generation[1]
            this_mse = torch.nn.functional.mse_loss(pred_unnorm, fd["gt_actions"].float()).item()
            mse.append(this_mse)
            preds[global_idx] = (pred_unnorm, pred_norm, pred_subtask, generation)
            logging.info(
                f"  ep={ep_idx:04d} fr={fr_idx:04d} | mse={this_mse:.4f} | "
                f"GT: '{fd['gt_subtask']}' | pred: '{pred_subtask or ''}'"
                + (f" | mem: '{generation[3]}'" if generation is not None else "")
            )
    finally:
        adapter.suppress_logs(False)

    return preds, mse, memory_ablation


def _write_memory_ablation(rows: list[dict], output_dir: str) -> None:
    if not rows:
        return

    def _correct(pred: str, target: str) -> bool:
        return pred.strip().casefold() == target.strip().casefold()

    gt_correct = [_correct(row["gt_memory_pred_subtask"], row["gt_subtask"]) for row in rows]
    empty_correct = [_correct(row["empty_memory_pred_subtask"], row["gt_subtask"]) for row in rows]
    changed = [
        row["gt_memory_pred_subtask"].strip().casefold()
        != row["empty_memory_pred_subtask"].strip().casefold()
        for row in rows
    ]
    summary = {
        "n_frames_with_nonempty_gt_memory": len(rows),
        "gt_memory_subtask_accuracy": float(np.mean(gt_correct)),
        "empty_memory_subtask_accuracy": float(np.mean(empty_correct)),
        "accuracy_delta": float(np.mean(gt_correct) - np.mean(empty_correct)),
        "decode_changed_fraction": float(np.mean(changed)),
    }
    with open(os.path.join(output_dir, "memory_ablation.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)
    logging.info(
        "[offline_inference] memory ablation "
        f"n={len(rows)}  GT-memory acc={summary['gt_memory_subtask_accuracy']:.3f}  "
        f"empty-memory acc={summary['empty_memory_subtask_accuracy']:.3f}  "
        f"delta={summary['accuracy_delta']:+.3f}"
    )


def _render_memory_ablation(rows: list[dict], frame_data: dict[int, dict], output_dir: str) -> None:
    """Contact sheet: current images and the full GT-memory/empty-memory decodes."""
    if not rows:
        return
    import textwrap

    shown = rows[:12]
    fig, axes = plt.subplots(len(shown), 4, figsize=(19, 3.4 * len(shown)), squeeze=False)

    def _image(tensor: torch.Tensor) -> np.ndarray:
        item = tensor.detach().float().cpu().squeeze()
        if item.ndim == 3 and item.shape[0] in (1, 3):
            item = item.permute(1, 2, 0)
        array = item.numpy()
        if array.max() <= 1.0:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        return array

    def _wrapped(value, width=52, limit=500) -> str:
        text = str(value or "(empty)")
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return textwrap.fill(text, width=width)

    for row_idx, row in enumerate(shown):
        fd = frame_data[row["global_idx"]]
        camera_keys = sorted(k for k in fd["obs"] if k.startswith("observation.images."))
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            if col_idx < len(camera_keys):
                key = camera_keys[col_idx]
                ax.imshow(_image(fd["obs"][key]))
                ax.set_title(f"ep {row['episode_idx']} fr {row['frame_idx']} — {key.split('.')[-1]}")
            ax.axis("off")

        target = row["gt_subtask"].strip().casefold()
        conditions = (
            (
                "GT memory",
                row["gt_memory_pred_subtask"],
                f"INPUT MEMORY:\n{_wrapped(row['gt_memory'])}\n\nRAW DECODE:\n"
                f"{_wrapped(row['gt_memory_raw_decode'])}",
            ),
            (
                "Empty memory",
                row["empty_memory_pred_subtask"],
                f"INPUT MEMORY:\nnone yet.\n\nRAW DECODE:\n{_wrapped(row['empty_memory_raw_decode'])}",
            ),
        )
        for offset, (name, prediction, body) in enumerate(conditions, start=2):
            ax = axes[row_idx, offset]
            correct = prediction.strip().casefold() == target
            ax.set_facecolor("#E8F5E9" if correct else "#FDECEC")
            ax.text(
                0.02,
                0.97,
                f"GT SUBTASK: {row['gt_subtask']}\nPREDICTED: {prediction}\n\n{body}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.2,
                family="monospace",
            )
            ax.set_title(f"{name} — {'CORRECT' if correct else 'WRONG'}", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Language-memory ablation: same observation, GT memory vs empty memory "
        f"(showing {len(shown)}/{len(rows)} frames)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(os.path.join(output_dir, "memory_ablation.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _traces_for_unnorm(unnorm: torch.Tensor, color_idx: int, ckpt_label: str) -> list[dict]:
    """Raw + smooth + butterworth traces for one checkpoint's actions."""
    return [
        {"actions": unnorm, "label": f"{ckpt_label} raw", "color_idx": color_idx,
         "kwargs": {"linewidth": 1.2, "linestyle": "-", "alpha": 1.0}},
        {"actions": smooth_actions(unnorm, 5), "label": f"{ckpt_label} (w=5)",
         "color_idx": color_idx,
         "kwargs": {"linewidth": 1.0, "linestyle": "-", "alpha": 0.4}},
        {"actions": apply_butterworth_filter(unnorm),
         "label": f"{ckpt_label} butter", "color_idx": color_idx,
         "kwargs": _BUTTER_STYLE},
    ]


def run(adapter, dataset, cfg, output_dir, *, path_label="A", path_str=None):
    """Single-checkpoint offline inference probe (used by both CLI and rl_offline).

    For checkpoint A vs B comparison, use :func:`eval_cli` — that wraps this
    with a second model load. ``adapter`` and ``dataset`` may be ``None`` if
    the caller has nothing to evaluate (returns immediately).
    """
    if adapter is None or dataset is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    dir_unnorm = os.path.join(output_dir, "unnormalized_eval")
    dir_norm   = os.path.join(output_dir, "normalized_eval")
    os.makedirs(dir_unnorm, exist_ok=True)
    os.makedirs(dir_norm,   exist_ok=True)

    # The CLI path uses cfg.eval_* fields (EvalOfflineConfig); the rl_offline
    # path supplies a TrainRLServerPipelineConfig and relies on probe_parameters.
    # getattr() falls back to probe_parameters when the eval_* field isn't defined.
    p = cfg.probe_parameters
    random_n = (
        getattr(cfg, "eval_random_n", 0)
        or getattr(p, "offline_inference_n_frames", 5)
    )
    chunk_size = adapter.chunk_size
    path_str = path_str or _policy_checkpoint_path(cfg.policy)

    samples = build_sample_list(
        dataset,
        episodes_str=getattr(cfg, "eval_episodes", None),
        frames_str=getattr(cfg, "eval_frames", None),
        random_n=random_n, chunk_size=chunk_size,
        seed=getattr(cfg, "eval_random_seed", None) or p.random_seed,
    )
    if not samples:
        logging.warning("[offline_inference] no samples selected, skipping.")
        return
    logging.info(f"Evaluating {len(samples)} frames …")

    summary_lookup = _summary_lookup(dataset, cfg)
    if summary_lookup:
        logging.info("Summaries found: conditioning generation on GT memory (hold/update rule).")

    frame_data: dict[int, dict] = {}
    preds, mse, memory_ablation = _run_checkpoint(
        adapter, dataset, samples, frame_data, chunk_size, summary_lookup, cfg
    )
    logging.info(f"MSE  {path_label} ({path_str}): {sum(mse) / len(mse):.4f}")
    _write_memory_ablation(memory_ablation, output_dir)
    _render_memory_ablation(memory_ablation, frame_data, output_dir)

    action_dim = frame_data[samples[0][2]]["gt_actions"].shape[-1]
    joint_names = _joint_names_for_dim(action_dim)
    checkpoint_paths = {path_label: path_str}

    for ep_idx, fr_idx, global_idx in samples:
        fd = frame_data[global_idx]
        pred_unnorm, pred_norm, sub, generation = preds[global_idx]
        ckpts = [{
            "label": path_label, "subtask": sub, "color_idx": 0,
            "summary": generation[3] if generation is not None else None,
        }]
        common = dict(
            obs=fd["obs"], gt_subtask=fd["gt_subtask"], gt_summary=fd["gt_summary"],
            episode_idx=ep_idx, frame_idx=fr_idx,
            joint_names=joint_names, checkpoint_paths=checkpoint_paths,
            checkpoints_info=ckpts,
        )
        render_sample(
            **common, gt_actions=fd["gt_actions"],
            pred_traces=_traces_for_unnorm(pred_unnorm, 0, path_label),
            output_dir=dir_unnorm, state=fd["state"],
        )
        render_sample(
            **common, gt_actions=fd["gt_actions_norm"],
            pred_traces=_traces_for_unnorm(pred_norm, 0, path_label),
            output_dir=dir_norm, state=None,
        )

    logging.debug(f"Done. {len(samples)} plots saved to {dir_unnorm}/ and {dir_norm}/")


@parser.wrap()
def eval_cli(cfg: EvalOfflineConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = cfg.eval_output_dir
    checkpoint_b = cfg.eval_checkpoint_b

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    path_a = _policy_checkpoint_path(cfg.policy)
    dataset = _load_dataset(cfg)

    logging.info("Loading policy A …")
    adapter_a = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    run(adapter_a, dataset, cfg, output_dir, path_label="A", path_str=path_a)

    if checkpoint_b:
        del adapter_a
        torch.cuda.empty_cache()
        if not _set_policy_checkpoint_path(cfg.policy, checkpoint_b):
            raise ValueError(
                "eval_checkpoint_b was set, but this policy config has no known checkpoint field "
                "(base_path or pretrained_path)."
            )
        logging.info("Loading policy B …")
        adapter_b = ProbablePolicy.for_config(cfg, device, dataset=dataset)
        # B writes to a sibling subdir; the original overlay-in-one-plot
        # behaviour was lost in the rl_offline refactor (single-adapter run()).
        run(adapter_b, dataset, cfg, os.path.join(output_dir, "ckpt_B"),
            path_label="B", path_str=str(checkpoint_b))


if __name__ == "__main__":
    eval_cli()
