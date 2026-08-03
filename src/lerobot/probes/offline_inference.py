#!/usr/bin/env python
"""
Generic offline inference probe — pick frames, run inference, save per-frame plots.

Policy-agnostic: works for any policy with a registered
:class:`lerobot.probes.base.ProbablePolicy` adapter (pi05, molmoact2, …).

For each selected frame, saves a figure with:
  - Camera images at the sampled timestep
  - 2×3 grid of predicted vs GT action chunk traces (per joint)
  - The subtask clause the frame was conditioned on
  - Optional checkpoint A vs B overlay

Two output subdirectories:
  unnormalized_eval/   actions in dataset units
  normalized_eval/     actions in normalised model space (GT via adapter.normalize_gt_actions)

Plus, at the top level:
  action_metrics.json  normalized-space MSE, split per joint, against the hold-still
                       and dataset-mean baselines (``skill_vs_*`` is the fraction of
                       each baseline's error the policy removes)

This probe measures actions only. It used to also decode a subtask and ablate the
language memory that conditioned that decode; both were removed 2026-08-01, when
the policy stopped generating anything (``subtask_max_new_tokens: 0``,
``subtask_loss_weight: 0``) and the summary memory was dropped. The subtask is now
an input the deployment prompt carries, so the question "does that clause reach the
actions" belongs to ``subtask_sweep``, which answers it causally.

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
import sys
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
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    action_inspector_sample_seed,
    build_sample_list,
    load_probe_dataset,
    probe_frame_inputs,
    probe_image_stride,
    sample_action_inspector_frames,
)
from lerobot.rl.inference_utils import apply_butterworth_filter
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


# The target embodiment: rebot B601, 7-DOF. Anything else falls back to indices.
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
    val_path = getattr(cfg, "val_dataset_path", None)
    if not val_path:
        return load_probe_dataset(cfg)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    logging.info(f"Loading eval dataset from val_dataset_path: {val_path}")
    dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_sample(
    obs,
    gt_actions,
    checkpoints_info,    # list of {"label": "A", "color_idx": 0}
    pred_traces,         # list of {"actions": tensor, "label": "A raw", "color_idx": 0, "kwargs": dict}
    gt_subtask,
    episode_idx,
    frame_idx,
    output_dir,
    joint_names=None,
    checkpoint_paths=None,
    state=None,
):
    """
    Save one evaluation figure.

    Layout:
      Row 0, cols 0..K-1  — camera images
      Row 0, col -1       — conditioning info box (when a spare column exists)
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
        # The subtask is an input the prompt carried, not something the policy
        # produced — there is no predicted counterpart to compare it against.
        ax_info.text(0.02, y_gt, "Subtask clause:", transform=ax_info.transAxes,
                     fontsize=7.5, fontweight="bold", color="#555555", va="top")
        y_gt -= 0.10
        y_gt = _draw_lines(0.02, y_gt, _wrap(gt_subtask or "(none)", max_lines=2), "#333333")
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

def _run_checkpoint(adapter: ProbablePolicy, dataset, samples, frame_data, chunk_size, cfg):
    """
    Run inference for all samples through *adapter*.

    Populates frame_data[global_idx] with shared per-frame fields the first time
    it sees a global_idx. Returns:
        preds:  {global_idx: (pred_unnorm, pred_norm)}
        mse:    list of float, one per sample
    """
    preds: dict[int, tuple] = {}
    mse: list[float] = []
    seed = int(getattr(cfg.probe_parameters, "random_seed", 0))

    adapter.suppress_logs(True)
    # Match Action Inspector sample 0 exactly. Disabling replayed graphs prevents a
    # captured noise tensor from overriding the per-frame generator.
    adapter._set_probe_cuda_graph_enabled(False)
    try:
        for ep_idx, fr_idx, global_idx in samples:
            if global_idx not in frame_data:
                frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
                state = frame["state"]
                gt_actions = frame["gt_actions"]
                frame_data[global_idx] = {
                    "obs": frame["obs"],
                    "gt_actions": gt_actions,
                    "gt_actions_norm": adapter.normalize_gt_actions(gt_actions, state),
                    # "the arm holds still": the normalized encoding of a chunk that
                    # repeats the measured state. Under anchor/delta this is the zero
                    # vector before normalization, and it is the floor any useful
                    # policy has to beat.
                    "hold_norm": adapter.normalize_gt_actions(
                        state[: gt_actions.shape[-1]].unsqueeze(0).repeat(gt_actions.shape[0], 1),
                        state,
                    ) if state is not None else None,
                    "state": state, "gt_subtask": frame["subtask"], "task_str": frame["task"],
                    "metadata": frame["metadata"],
                }

            fd = frame_data[global_idx]
            # Seeded per frame: flow noise is then identical across checkpoints, so a
            # change in the number is a change in the policy, not in the draw.
            generator = torch.Generator(device=adapter.device)
            generator.manual_seed(action_inspector_sample_seed(seed, global_idx, 0))
            pred_unnorm, pred_norm, _ = adapter.predict_action_chunk(
                fd["obs"], fd["task_str"], state=fd["state"],
                subtask=fd["gt_subtask"], metadata=fd["metadata"], generator=generator,
            )
            this_mse = float((pred_norm - fd["gt_actions_norm"]).pow(2).mean())
            mse.append(this_mse)
            preds[global_idx] = (pred_unnorm, pred_norm)
            logging.info(
                f"  ep={ep_idx:04d} fr={fr_idx:04d} | mse_norm={this_mse:.4f} | "
                f"subtask: '{fd['gt_subtask']}'"
            )
    finally:
        adapter._restore_probe_cuda_graph_enabled()
        adapter.suppress_logs(False)

    return preds, mse


def _write_action_metrics(preds, frame_data, samples, joint_names, output_dir: str) -> dict:
    """Normalized-space action error against the two constants worth beating.

    The old headline was a raw deg^2 MSE over all joints. On the 7-DOF rebot the
    gripper's spread (sigma ~76 deg) is an order of magnitude above wrist_yaw's
    (~11 deg), so that number was mostly gripper, in units that change whenever the
    normalization stats do, and with nothing to compare it to. Everything here is in
    the normalized model space the policy is actually trained in, and every figure
    sits next to:

      hold          — repeat the measured state for the whole chunk (arm holds still)
      dataset mean  — the mean GT chunk over the sampled frames (best constant)

    A policy that does not beat both is not predicting; ``skill_vs_hold`` and
    ``skill_vs_mean`` are the fraction of each baseline's error it removes.
    """
    order = [g for _, _, g in samples if g in preds]
    if not order:
        return {}

    gt = torch.stack([frame_data[g]["gt_actions_norm"] for g in order]).float()
    pred = torch.stack([preds[g][1] for g in order]).float()
    dataset_mean = gt.mean(dim=0, keepdim=True).expand_as(gt)

    def _mse(a, b):
        return float((a - b).pow(2).mean())

    def _per_joint(a, b):
        return (a - b).pow(2).mean(dim=(0, 1)).tolist()

    metrics = {
        "n_frames": len(order),
        "space": "normalized",
        "mse_norm": _mse(pred, gt),
        "mse_norm_by_joint": dict(zip(joint_names, _per_joint(pred, gt))),
        "baseline_dataset_mean": _mse(dataset_mean, gt),
        "mse_unnormalized_deg2": float(
            torch.stack([(preds[g][0] - frame_data[g]["gt_actions"]).pow(2).mean() for g in order]).mean()
        ),
    }

    holds = [frame_data[g]["hold_norm"] for g in order]
    if all(h is not None for h in holds):
        hold = torch.stack(holds).float()
        metrics["baseline_hold"] = _mse(hold, gt)
        metrics["skill_vs_hold"] = 1.0 - metrics["mse_norm"] / max(metrics["baseline_hold"], 1e-12)
    metrics["skill_vs_mean"] = 1.0 - metrics["mse_norm"] / max(metrics["baseline_dataset_mean"], 1e-12)

    with open(os.path.join(output_dir, "action_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    worst = sorted(metrics["mse_norm_by_joint"].items(), key=lambda kv: -kv[1])[:3]
    logging.info(
        f"[offline_inference] mse_norm={metrics['mse_norm']:.4f}  "
        f"hold={metrics.get('baseline_hold', float('nan')):.4f}  "
        f"dataset-mean={metrics['baseline_dataset_mean']:.4f}  "
        f"skill_vs_hold={metrics.get('skill_vs_hold', float('nan')):+.3f}  "
        f"skill_vs_mean={metrics['skill_vs_mean']:+.3f}  "
        f"worst joints: {', '.join(f'{k} {v:.4f}' for k, v in worst)}"
    )
    return metrics


def _write_manifest(output_dir: str, action_metrics: dict, samples) -> dict:
    """Describe the Offline Inference readout without viewer-side special cases."""
    summary = {"action": action_metrics}
    metrics = [
        Metric(
            "action.skill_vs_hold", "Skill recovered vs hold-still", good="high", fmt=3,
            baseline=0.0, primary=True,
            note="Positive means the policy removes error relative to repeating the current pose.",
        ),
        Metric(
            "action.skill_vs_mean", "Skill recovered vs dataset mean", good="high", fmt=3,
            baseline=0.0, primary=True,
            note="Positive means the policy beats the best constant chunk over these sampled frames.",
        ),
        Metric(
            "action.mse_norm", "Normalized action MSE", good="low", fmt=4, primary=True,
            note="Read against both baseline errors below; its absolute scale is model-normalized.",
        ),
        Metric("action.baseline_hold", "Hold-still baseline MSE", good="none", fmt=4),
        Metric("action.baseline_dataset_mean", "Dataset-mean baseline MSE", good="none", fmt=4),
        Metric("action.n_frames", "Frames evaluated", good="none", fmt=0),
    ]

    panels = []
    for index, (episode_idx, frame_idx, _) in enumerate(samples):
        filename = f"ep{episode_idx:04d}_fr{frame_idx:04d}.png"
        panels.append(
            Panel(
                f"normalized_eval/{filename}",
                f"Normalized action trace — episode {episode_idx}, frame {frame_idx}",
                how="Blue is the demonstrated chunk; the coloured traces are the policy prediction. "
                     "Judge whether their shape and timing agree in normalized model space, then use "
                     "the headline baseline-relative metrics for the aggregate verdict.",
                primary=index == 0,
            )
        )
        panels.append(
            Panel(
                f"unnormalized_eval/{filename}",
                f"Robot-space action trace — episode {episode_idx}, frame {frame_idx}",
                how="The same prediction in dataset units. Use this to identify which physical joint "
                     "drives an error; compare checkpoints with the normalized metrics, not this scale.",
            )
        )
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Offline Inference",
        group="Actions",
        claim="Does the checkpoint beat the simple action baselines on held-out frames?",
        summary=summary,
        metrics=metrics,
        panels=panels,
        see_also=["actions", "subtask_sweep", "mem_history_influence"],
    )


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

    explicit_episodes = getattr(cfg, "eval_episodes", None)
    explicit_random_n = getattr(cfg, "eval_random_n", 0)
    if getattr(p, "enable_action_trace", False) and not explicit_episodes and not explicit_random_n:
        samples = sample_action_inspector_frames(
            dataset, p, stride=probe_image_stride(cfg)
        )
        logging.info(
            f"Using {len(samples)} shared Action Inspector anchors so the 2-D and 3-D "
            "views show the same observation and seeded sample 0."
        )
    else:
        samples = build_sample_list(
            dataset,
            episodes_str=explicit_episodes,
            frames_str=getattr(cfg, "eval_frames", None),
            random_n=random_n, chunk_size=chunk_size,
            seed=getattr(cfg, "eval_random_seed", None) or p.random_seed,
            stride=probe_image_stride(cfg),
        )
    if not samples:
        logging.warning("[offline_inference] no samples selected, skipping.")
        return
    logging.info(f"Evaluating {len(samples)} frames …")

    frame_data: dict[int, dict] = {}
    preds, mse = _run_checkpoint(adapter, dataset, samples, frame_data, chunk_size, cfg)
    logging.info(f"mse_norm  {path_label} ({path_str}): {sum(mse) / len(mse):.4f}")

    action_dim = frame_data[samples[0][2]]["gt_actions"].shape[-1]
    joint_names = _joint_names_for_dim(action_dim)
    action_metrics = _write_action_metrics(preds, frame_data, samples, joint_names, output_dir)
    checkpoint_paths = {path_label: path_str}

    for ep_idx, fr_idx, global_idx in samples:
        fd = frame_data[global_idx]
        pred_unnorm, pred_norm = preds[global_idx]
        ckpts = [{"label": path_label, "color_idx": 0}]
        common = dict(
            obs=fd["obs"], gt_subtask=fd["gt_subtask"],
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

    _write_manifest(output_dir, action_metrics, samples)
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
