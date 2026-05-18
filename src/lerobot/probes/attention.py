#!/usr/bin/env python
"""
Generic attention probe — visualise per-layer cross- and self-attention.

Policy-agnostic: works with any policy that has a registered
:class:`lerobot.probes.base.ProbablePolicy` adapter exposing
``capture_attention(...)``.

For each sampled frame and each requested layer, emits:

  overlay_<cam>_summary.mp4   — mean-over-heads heatmaps over each camera
                                (only if the adapter returns ``img*`` segments)
  overlay_<cam>_heads.mp4     — per-head grid for each camera segment
  cross_matrix_mean.mp4       — cross-attn matrix view (rows=actions, cols=encoder),
                                mean over heads, with segment dividers
  cross_matrix_heads.mp4      — per-head 2×4 grid of the same matrix
  self_matrix_mean.mp4        — self-attn matrix view (action ↔ action), mean over heads
  norm_consts.csv             — per-panel vmax for stable colorbars

Usage:
    python -m lerobot.probes.attention config.yaml \\
        --probe_parameters.max_episodes 5 \\
        --probe_parameters.timestep 0.5
"""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import AttentionCaptureResult, ProbablePolicy
from lerobot.probes.utils import build_episode_index, get_frame_data, load_extra_dataset
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeAttentionConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters``."""


# ──────────────────────────────────────────────────────────────────────────────
# Sample selection — per-episode, fixed-stride
# ──────────────────────────────────────────────────────────────────────────────

def build_episode_samples(dataset, episodes_str, random_n, subsample, seed=None):
    """Return ``[(ep_idx, [(fr_idx, global_idx), ...]), ...]``."""
    ep_to_indices = build_episode_index(dataset)
    selected: list[int] = []

    if episodes_str:
        for ep_idx in [int(e) for e in episodes_str.split(",")]:
            if ep_idx in ep_to_indices:
                selected.append(ep_idx)

    if random_n:
        rng = random.Random(seed)
        all_eps = list(ep_to_indices.keys())
        rng.shuffle(all_eps)
        for ep_idx in all_eps:
            if len(selected) >= random_n:
                break
            if ep_idx not in selected:
                selected.append(ep_idx)

    samples = []
    for ep_idx in selected:
        indices = ep_to_indices[ep_idx]
        ep_frames = [(fr_idx, indices[fr_idx]) for fr_idx in range(0, len(indices), subsample)]
        if ep_frames:
            samples.append((ep_idx, ep_frames))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def cv2_overlay(img_np, heatmap, title, alpha=0.55, vmax=None):
    """Blend a heatmap onto a camera image."""
    if vmax is None:
        vmax = float(heatmap.max().item())
    h_norm = heatmap / (vmax + 1e-8)
    h_gray = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    if h_rgb.shape != img_np.shape:
        h_rgb = cv2.resize(h_rgb, (img_np.shape[1], img_np.shape[0]))
    out = cv2.addWeighted(img_np, 1 - alpha, h_rgb, alpha, 0)
    cv2.putText(out, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def cv2_heatmap(heatmap, title, img_h, img_w, vmax=None):
    """Render a standalone heatmap (no camera image)."""
    if vmax is None:
        vmax = float(heatmap.max().item())
    h_norm = heatmap / (vmax + 1e-8)
    h_gray = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    if h_rgb.shape[:2] != (img_h, img_w):
        h_rgb = cv2.resize(h_rgb, (img_w, img_h))
    cv2.putText(h_rgb, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return h_rgb


def _attn_to_image_heatmap(cross_attn_layer, k_start, k_end, n_p, img_h, img_w):
    """
    Mean attention from all action queries to key positions [k_start:k_end].

    Inputs:
        cross_attn_layer: [n_heads, n_action, encoder_seq_len]  (already squeezed to B=1)
        n_p: side length of the patch grid (so n_p*n_p == k_end - k_start)
    Returns:
        per_head: [n_heads, img_h, img_w]
        mean:     [img_h, img_w]
    """
    n_heads = cross_attn_layer.shape[0]
    slc = cross_attn_layer[:, :, k_start:k_end]      # [H, n_action, n_patches]
    per_head = slc.mean(dim=1)                        # mean over actions → [H, n_patches]

    def _up(x):
        grid = x.reshape(n_p, n_p).float()
        up = F.interpolate(grid[None, None], size=(img_h, img_w),
                           mode="bicubic", align_corners=False)
        return up.squeeze().clamp(min=0)

    per_head_up = torch.stack([_up(per_head[h]) for h in range(n_heads)])
    mean_up = _up(per_head.mean(0))
    return per_head_up, mean_up


def render_image_overlays(result: AttentionCaptureResult, layer_idx: int):
    """Render per-camera overlays + per-head grids for a single layer.

    Returns ``(frames, vmax)`` dicts keyed by panel name. Empty if there are no
    ``img*`` segments (e.g. molmoact2 v1 with single 'encoder' segment).
    """
    frames: dict[str, np.ndarray] = {}
    vmax_by_panel: dict[str, float] = {}

    cross_attn = result.cross_attn_by_layer.get(layer_idx)
    if cross_attn is None:
        return frames, vmax_by_panel
    image_segs = [(name, s, e) for name, s, e in result.encoder_segments
                  if name.startswith("img")]
    if not image_segs or result.patches_per_cam <= 0:
        return frames, vmax_by_panel

    attn = torch.nan_to_num(cross_attn[0].float().cpu(), nan=0.0)  # [H, n_act, enc]
    n_heads = attn.shape[0]
    n_p = int(result.patches_per_cam ** 0.5)
    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols

    for cam_idx, (cam_name, cs, ce) in enumerate(image_segs):
        img_np = None
        if cam_idx < len(result.image_tensors):
            img_t = result.image_tensors[cam_idx].squeeze(0).cpu()
            img_t = img_t * 0.5 + 0.5
            img_np = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_h = img_np.shape[0] if img_np is not None else 224
        img_w = img_np.shape[1] if img_np is not None else 224
        if img_np is None:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        per_head, mean_map = _attn_to_image_heatmap(attn, cs, ce, n_p, img_h, img_w)
        vmean = float(mean_map.max().item())
        vheads = float(per_head.max().item())
        vmax_by_panel[f"{cam_name}_mean"] = vmean
        vmax_by_panel[f"{cam_name}_heads"] = vheads

        # Mean-over-heads summary strip
        summary = [img_np.copy()]
        cv2.putText(summary[0], f"{cam_name} (orig)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        summary.append(cv2_overlay(img_np, mean_map, f"mean: action→{cam_name}", vmax=vmean))
        frames[f"overlay_{cam_name}_summary"] = np.hstack(summary)

        # Per-head grid
        rows = []
        for r in range(h_rows):
            row_imgs = []
            for c in range(h_cols):
                idx = r * h_cols + c
                if idx < n_heads:
                    row_imgs.append(cv2_overlay(img_np, per_head[idx],
                                                f"{cam_name} h{idx}", vmax=vheads))
                else:
                    row_imgs.append(np.zeros_like(img_np))
            rows.append(np.hstack(row_imgs))
        frames[f"overlay_{cam_name}_heads"] = np.vstack(rows)

    return frames, vmax_by_panel


def _render_matrix(mat_2d, vmax, title, out_w=1200, out_h=600,
                   col_segments=None):
    """Render a 2D attention matrix as a heatmap with optional segment dividers.

    Args:
        mat_2d:        [R, C] tensor or array (mean over heads).
        col_segments:  optional list of (name, start, end) for column dividers.
    """
    arr = mat_2d.numpy() if hasattr(mat_2d, "numpy") else mat_2d
    norm = (arr / max(vmax, 1e-8)).clip(0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    margin_top, margin_bottom, margin_left, margin_right = 32, 60, 40, 8
    hm_w = out_w - margin_left - margin_right
    hm_h = out_h - margin_top - margin_bottom
    color = cv2.resize(color, (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[margin_top : margin_top + hm_h, margin_left : margin_left + hm_w] = color
    cv2.putText(canvas, title, (margin_left + 4, margin_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if col_segments:
        n_cols = arr.shape[1]
        px = hm_w / max(n_cols, 1)
        annot_y = margin_top + hm_h + 16
        for name, s, e in col_segments:
            mid = int(margin_left + (s + e) / 2 * px)
            (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(canvas, name, (mid - tw // 2, annot_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if s > 0:
                x_div = int(margin_left + s * px)
                cv2.line(canvas, (x_div, margin_top),
                         (x_div, margin_top + hm_h), (255, 255, 255), 1)
    return canvas


def render_cross_matrix(result: AttentionCaptureResult, layer_idx: int):
    """Mean cross-attention heatmap (rows=actions, cols=encoder)."""
    frames: dict[str, np.ndarray] = {}
    vmax_by_panel: dict[str, float] = {}

    cross = result.cross_attn_by_layer.get(layer_idx)
    if cross is None:
        return frames, vmax_by_panel

    attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)   # [H, n_act, enc]
    mean = attn.mean(dim=0)                                     # [n_act, enc]
    vmax = float(mean.max().item())
    vmax_by_panel["cross_matrix_mean"] = vmax
    frames["cross_matrix_mean"] = _render_matrix(
        mean, vmax,
        f"L{layer_idx}: cross-attn action→encoder (mean over heads)",
        col_segments=result.encoder_segments,
    )

    # Per-head 2×4 grid
    n_heads = attn.shape[0]
    h_cols, h_rows = 4, (n_heads + 3) // 4
    panel_w, panel_h = 1200, 600
    grid_w = panel_w * h_cols // 2
    grid_h = panel_h * h_rows // 2
    head_vmax = float(attn.max().item())
    vmax_by_panel["cross_matrix_heads"] = head_vmax
    canvas = np.zeros((grid_h + 36, grid_w, 3), dtype=np.uint8)
    cv2.putText(canvas, f"L{layer_idx}: cross-attn per head", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    sub_w = grid_w // h_cols
    sub_h = grid_h // h_rows
    for h in range(n_heads):
        r, c = divmod(h, h_cols)
        sub = _render_matrix(attn[h], head_vmax, f"head {h}",
                             out_w=sub_w, out_h=sub_h,
                             col_segments=result.encoder_segments)
        canvas[36 + r * sub_h : 36 + (r + 1) * sub_h,
               c * sub_w : (c + 1) * sub_w] = sub
    frames["cross_matrix_heads"] = canvas

    return frames, vmax_by_panel


def render_self_matrix(result: AttentionCaptureResult, layer_idx: int):
    """Mean self-attention heatmap (action ↔ action)."""
    frames: dict[str, np.ndarray] = {}
    vmax_by_panel: dict[str, float] = {}

    selfa = result.self_attn_by_layer.get(layer_idx)
    if selfa is None:
        return frames, vmax_by_panel

    attn = torch.nan_to_num(selfa[0].float().cpu(), nan=0.0)
    mean = attn.mean(dim=0)
    vmax = float(mean.max().item())
    vmax_by_panel["self_matrix_mean"] = vmax
    frames["self_matrix_mean"] = _render_matrix(
        mean, vmax, f"L{layer_idx}: self-attn action↔action (mean over heads)",
        out_w=900, out_h=900,
    )
    return frames, vmax_by_panel


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _probe_dataset(adapter, ds, ds_output_dir, attn_layers, timestep, cfg):
    """Per-dataset attention rendering loop. Used by both standalone CLI and
    the rl_offline validation loop."""
    p = cfg.probe_parameters
    chunk_size = adapter.chunk_size
    os.makedirs(ds_output_dir, exist_ok=True)
    samples = build_episode_samples(
        ds,
        episodes_str=getattr(p, "attn_eval_episodes", None),
        random_n=p.max_episodes,
        subsample=getattr(p, "attn_eval_subsample", 1),
        seed=p.random_seed,
    )
    if not samples:
        logging.warning(f"  No samples in {ds_output_dir}, skipping.")
        return

    fps = getattr(ds, "fps", 30) / max(1, getattr(p, "attn_eval_subsample", 1))

    for ep_idx, ep_frames in samples:
        writers: dict[int, dict[str, Any]] = {l: {} for l in attn_layers}  # noqa: E741
        csv_files: dict[int, Any] = {}

        for fr_idx, global_idx in ep_frames:
            obs, _, state, _, task_str, _, _ = get_frame_data(ds, global_idx, chunk_size)
            result = adapter.capture_attention(
                obs, task_str, state=state, timestep=timestep, layers=attn_layers,
            )

            for layer_idx in attn_layers:
                ep_dir = os.path.join(ds_output_dir, f"ep{ep_idx:04d}_L{layer_idx:02d}")
                os.makedirs(ep_dir, exist_ok=True)

                if layer_idx not in csv_files:
                    csv_path = os.path.join(ep_dir, "norm_consts.csv")
                    f = open(csv_path, "a", newline="")
                    w = csv.writer(f)
                    if os.path.getsize(csv_path) == 0:
                        w.writerow(["ep", "fr", "layer", "panel", "vmax"])
                    csv_files[layer_idx] = (f, w)
                csv_f, csv_w = csv_files[layer_idx]

                panels: dict[str, np.ndarray] = {}
                vmaxes: dict[str, float] = {}
                for renderer in (render_image_overlays, render_cross_matrix, render_self_matrix):
                    p_frames, p_vmax = renderer(result, layer_idx)
                    panels.update(p_frames)
                    vmaxes.update(p_vmax)

                for panel, vmax in vmaxes.items():
                    csv_w.writerow([ep_idx, fr_idx, layer_idx, panel, f"{vmax:.6e}"])

                for key, frame_np in panels.items():
                    if key not in writers[layer_idx]:
                        writers[layer_idx][key] = imageio.get_writer(
                            os.path.join(ep_dir, f"{key}.mp4"),
                            fps=fps, macro_block_size=1,
                        )
                    writers[layer_idx][key].append_data(frame_np)

        for d in writers.values():
            for w in d.values():
                w.close()
        for f, _ in csv_files.values():
            f.close()


def run(adapter, primary_dataset, cfg, output_dir):
    """Run the attention probe end-to-end on the primary dataset (and any
    ``additional_offline_dataset_paths``). Idempotent re-runs overwrite outputs.
    """
    if adapter is None or primary_dataset is None:
        return

    p = cfg.probe_parameters
    attn_layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    timestep = float(getattr(p, "timestep", 0.5))
    logging.info(f"Probing layers: {attn_layers} timestep: {timestep}")
    os.makedirs(output_dir, exist_ok=True)

    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    _probe_dataset(adapter, primary_dataset,
                   os.path.join(output_dir, primary_name), attn_layers, timestep, cfg)

    for extra_root in getattr(cfg.dataset, "additional_offline_dataset_paths", []) or []:
        logging.info(f"Additional dataset: {extra_root}")
        extra_ds = load_extra_dataset(cfg.dataset.repo_id, extra_root)
        _probe_dataset(adapter, extra_ds,
                       os.path.join(output_dir, os.path.basename(os.path.normpath(extra_root))),
                       attn_layers, timestep, cfg)


@parser.wrap()
def probe_cli(cfg: ProbeAttentionConfig):
    init_logging()
    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "attention")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    from lerobot.datasets.factory import make_dataset
    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None

    logging.info("Loading policy adapter …")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    run(adapter, dataset, cfg, output_dir)
    logging.info(f"Done. Output saved to {output_dir}/")


if __name__ == "__main__":
    probe_cli()
