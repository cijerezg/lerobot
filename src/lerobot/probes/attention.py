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
import json
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

def _read_proc_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _read_meminfo_kb() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    out[parts[0].rstrip(":")] = int(parts[1])
    except Exception:
        pass
    return out


def _warn_overcommit_if_risky(probe_name: str) -> None:
    overcommit = _read_proc_int("/proc/sys/vm/overcommit_memory")
    meminfo = _read_meminfo_kb()
    commit_limit = meminfo.get("CommitLimit")
    committed = meminfo.get("Committed_AS")
    if overcommit != 0 or not commit_limit or not committed:
        return
    ratio = committed / max(commit_limit, 1)
    if ratio <= 1.0:
        return
    logging.warning(
        "[%s] Linux vm.overcommit_memory=0 with Committed_AS/CommitLimit=%.2f "
        "(%0.1f/%0.1f GB). imageio/ffmpeg video writers may fail with "
        "[Errno 12] Cannot allocate memory from this large training process. "
        "For ML workstations, consider: sudo sysctl -w vm.overcommit_memory=1",
        probe_name,
        ratio,
        committed / 1024 / 1024,
        commit_limit / 1024 / 1024,
    )


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


def _attn_values_to_image_heatmap(per_head: torch.Tensor, n_p: int, img_h: int, img_w: int):
    """Upsample per-head patch attention from [H, n_p*n_p] to image size."""
    n_heads = per_head.shape[0]

    def _up(x):
        grid = x.reshape(n_p, n_p).float()
        up = F.interpolate(grid[None, None], size=(img_h, img_w),
                           mode="bicubic", align_corners=False)
        return up.squeeze().clamp(min=0)

    per_head_up = torch.stack([_up(per_head[h]) for h in range(n_heads)])
    mean_up = _up(per_head.mean(0))
    return per_head_up, mean_up


def _attn_to_image_heatmap(cross_attn_layer, k_start, k_end, n_p, img_h, img_w):
    """Mean attention from all action queries to contiguous key positions."""
    slc = cross_attn_layer[:, :, k_start:k_end]      # [H, n_action, n_patches]
    per_head = slc.mean(dim=1)                        # mean over actions → [H, n_patches]
    return _attn_values_to_image_heatmap(per_head, n_p, img_h, img_w)


def _attn_indices_to_image_heatmap(cross_attn_layer, indices, n_p, img_h, img_w):
    """Mean attention to explicit, possibly non-contiguous image patch indices."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=cross_attn_layer.device)
    slc = cross_attn_layer.index_select(dim=2, index=idx)
    per_head = slc.mean(dim=1)
    return _attn_values_to_image_heatmap(per_head, n_p, img_h, img_w)


def _attn_indices_to_pooled_patch_heatmap(
    cross_attn_layer,
    indices,
    pooling,
    patch_grid,
    img_h,
    img_w,
):
    """Project attention on pooled image tokens back to crop patch pixels.

    Molmo crop tokens are pooled from one or more ViT patches and are not always
    a square, contiguous token block. The adapter passes the local pooling rows
    for a crop; this renderer scatters each token's attention back to the crop's
    patch grid before upsampling to image size.
    """
    idx = torch.as_tensor(indices, dtype=torch.long, device=cross_attn_layer.device)
    slc = cross_attn_layer.index_select(dim=2, index=idx)
    per_head_token = slc.mean(dim=1).float().cpu()  # [H, n_tokens]

    pooling_t = torch.as_tensor(pooling, dtype=torch.long)
    if pooling_t.ndim == 1:
        pooling_t = pooling_t[:, None]
    if pooling_t.shape[0] != per_head_token.shape[1]:
        raise ValueError(
            f"pooling rows ({pooling_t.shape[0]}) do not match token indices ({per_head_token.shape[1]})"
        )

    n_heads = per_head_token.shape[0]
    n_patches = int(patch_grid) * int(patch_grid)
    patch_values = torch.zeros((n_heads, n_patches), dtype=per_head_token.dtype)
    patch_counts = torch.zeros(n_patches, dtype=per_head_token.dtype)

    for token_idx in range(pooling_t.shape[0]):
        valid = pooling_t[token_idx]
        valid = valid[(valid >= 0) & (valid < n_patches)].unique()
        if valid.numel() == 0:
            continue
        patch_values[:, valid] += per_head_token[:, token_idx : token_idx + 1]
        patch_counts[valid] += 1.0

    patch_values = patch_values / patch_counts.clamp_min(1.0).unsqueeze(0)
    return _attn_values_to_image_heatmap(patch_values, int(patch_grid), img_h, img_w)


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
    overlay_segments = result.extras.get("image_overlay_segments")
    if overlay_segments:
        image_segs = [(str(name), None, None) for name in overlay_segments]
    else:
        image_segs = [(name, s, e) for name, s, e in result.encoder_segments
                      if name.startswith("img")]
    patch_indices = result.extras.get("image_patch_indices_by_segment", {})
    pooling_by_segment = result.extras.get("image_pooling_by_segment", {})
    tensors_by_segment = result.extras.get("image_tensors_by_segment", {})
    if not image_segs or (
        result.patches_per_cam <= 0 and not patch_indices and not pooling_by_segment
    ):
        return frames, vmax_by_panel

    attn = torch.nan_to_num(cross_attn[0].float().cpu(), nan=0.0)  # [H, n_act, enc]
    n_heads = attn.shape[0]
    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols

    for cam_idx, (cam_name, cs, ce) in enumerate(image_segs):
        img_np = None
        img_t = tensors_by_segment.get(cam_name)
        if img_t is None and cam_idx < len(result.image_tensors):
            img_t = result.image_tensors[cam_idx]
        if img_t is not None:
            img_t = img_t.squeeze(0).cpu()
            img_t = img_t * 0.5 + 0.5
            img_np = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_h = img_np.shape[0] if img_np is not None else 224
        img_w = img_np.shape[1] if img_np is not None else 224
        if img_np is None:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        indices = patch_indices.get(cam_name)
        pooling = pooling_by_segment.get(cam_name)
        if indices is not None and pooling is not None:
            patch_grid = int(pooling.get("patch_grid", 0))
            pooling_rows = pooling.get("pooling")
            if patch_grid <= 0 or pooling_rows is None:
                continue
            per_head, mean_map = _attn_indices_to_pooled_patch_heatmap(
                attn, indices, pooling_rows, patch_grid, img_h, img_w
            )
        elif indices is not None:
            n_p = int(len(indices) ** 0.5)
            if n_p * n_p != len(indices):
                continue
            per_head, mean_map = _attn_indices_to_image_heatmap(attn, indices, n_p, img_h, img_w)
        else:
            if cs is None or ce is None:
                continue
            n_p = int(result.patches_per_cam ** 0.5)
            if n_p * n_p != ce - cs:
                continue
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


def _matrix_indices_from_result(result: AttentionCaptureResult, encoder_len: int):
    explicit = result.extras.get("matrix_col_indices_by_segment")
    if not explicit:
        explicit = {}
        for name, indices in result.extras.get("image_patch_indices_by_segment", {}).items():
            explicit[str(name)] = indices
        for name, indices in result.extras.get("text_token_indices_by_segment", {}).items():
            explicit[str(name)] = indices

    selected: list[int] = []
    display_segments: list[tuple[str, int, int]] = []
    seen: set[int] = set()
    for name, raw_indices in explicit.items():
        clean = []
        for idx in raw_indices:
            idx = int(idx)
            if 0 <= idx < encoder_len and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        if not clean:
            continue
        start = len(selected)
        selected.extend(clean)
        display_segments.append((str(name), start, len(selected)))

    if not selected:
        return None, result.encoder_segments
    return torch.as_tensor(selected, dtype=torch.long), display_segments


def _cross_matrix_attn_and_segments(result: AttentionCaptureResult, attn: torch.Tensor):
    idx, segments = _matrix_indices_from_result(result, int(attn.shape[-1]))
    if idx is None:
        return attn, segments
    return attn.index_select(dim=-1, index=idx), segments


def render_cross_matrix(result: AttentionCaptureResult, layer_idx: int):
    """Mean cross-attention heatmap (rows=actions, cols=encoder)."""
    frames: dict[str, np.ndarray] = {}
    vmax_by_panel: dict[str, float] = {}

    cross = result.cross_attn_by_layer.get(layer_idx)
    if cross is None:
        return frames, vmax_by_panel

    attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)   # [H, n_act, enc]
    attn, display_segments = _cross_matrix_attn_and_segments(result, attn)
    mean = attn.mean(dim=0)                                     # [n_act, displayed_enc]
    vmax = float(mean.max().item())
    vmax_by_panel["cross_matrix_mean"] = vmax
    frames["cross_matrix_mean"] = _render_matrix(
        mean, vmax,
        f"L{layer_idx}: cross-attn action→encoder (mean over heads)",
        col_segments=display_segments,
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
                             col_segments=display_segments)
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
# Action → [action | decoded text] focused matrix
# ──────────────────────────────────────────────────────────────────────────────

def _decode_token_label(tokenizer, tid) -> str:
    if tokenizer is None:
        return str(int(tid))
    try:
        text = tokenizer.decode([int(tid)], skip_special_tokens=False)
    except Exception:
        return str(int(tid))
    text = str(text).replace("\n", " ").strip()
    return text.encode("ascii", errors="replace").decode("ascii") if text else ""


def _draw_rotated_text(canvas, text, x_center, y_top, font_scale, color):
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    pad = 2
    sub = np.zeros((th + baseline + pad * 2, tw + pad * 2, 3), dtype=np.uint8)
    cv2.putText(sub, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)
    rotated = cv2.rotate(sub, cv2.ROTATE_90_CLOCKWISE)
    rh, rw = rotated.shape[:2]
    x0 = max(0, x_center - rw // 2)
    y0 = y_top
    x1 = min(canvas.shape[1], x0 + rw)
    y1 = min(canvas.shape[0], y0 + rh)
    if x1 > x0 and y1 > y0:
        roi = canvas[y0:y1, x0:x1]
        np.maximum(roi, rotated[: y1 - y0, : x1 - x0], out=roi)


def _render_action_text_panel(
    mat,
    vmax,
    col_labels,
    groups,
    title,
    out_w=2400,
    out_h=900,
    title_font_scale=0.55,
    label_font_scale=0.35,
    group_font_scale=0.5,
):
    arr = mat.numpy() if hasattr(mat, "numpy") else mat
    n_rows, n_cols = arr.shape[:2]
    norm = (arr / max(float(vmax), 1e-8)).clip(0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    margin_top, margin_bottom, margin_left, margin_right = 32, 140, 42, 8
    hm_w = out_w - margin_left - margin_right
    hm_h = out_h - margin_top - margin_bottom
    color = cv2.resize(color, (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[margin_top : margin_top + hm_h, margin_left : margin_left + hm_w] = color
    cv2.putText(canvas, title, (margin_left + 4, margin_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)

    px_col = hm_w / max(n_cols, 1)
    px_row = hm_h / max(n_rows, 1)
    for row_idx in range(0, n_rows, 5):
        y = int(margin_top + (row_idx + 0.5) * px_row + 4)
        cv2.putText(canvas, str(row_idx), (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 200, 200), 1, cv2.LINE_AA)

    annot_y = margin_top + hm_h + 18
    for group_name, start, end in groups:
        if end <= start:
            continue
        if start > 0:
            x = int(margin_left + start * px_col)
            cv2.line(canvas, (x, margin_top), (x, margin_top + hm_h), (255, 255, 255), 1)
        mid = int(margin_left + ((start + end) / 2) * px_col)
        (tw, _), _ = cv2.getTextSize(group_name, cv2.FONT_HERSHEY_SIMPLEX, group_font_scale, 1)
        cv2.putText(canvas, group_name, (mid - tw // 2, annot_y),
                    cv2.FONT_HERSHEY_SIMPLEX, group_font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    label_y = margin_top + hm_h + 28
    for col_idx, label in enumerate(col_labels):
        if not label:
            continue
        x_center = int(margin_left + (col_idx + 0.5) * px_col)
        _draw_rotated_text(canvas, label, x_center, label_y,
                           label_font_scale, (220, 220, 220))
    return canvas


def _render_action_text_heads(per_head, vmax, col_labels, groups, title):
    n_heads = int(per_head.shape[0])
    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols
    panel_w, panel_h = 900, 520
    title_h = 36
    canvas = np.zeros((title_h + h_rows * panel_h, h_cols * panel_w, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    for head_idx in range(n_heads):
        r, c = divmod(head_idx, h_cols)
        sub = _render_action_text_panel(
            per_head[head_idx], vmax, col_labels, groups, f"head {head_idx}",
            out_w=panel_w, out_h=panel_h,
            title_font_scale=0.45, label_font_scale=0.30, group_font_scale=0.4,
        )
        canvas[title_h + r * panel_h : title_h + (r + 1) * panel_h,
               c * panel_w : (c + 1) * panel_w] = sub
    return canvas


def _token_label_for_position(result: AttentionCaptureResult, segment_name: str, pos: int, seg_start: int | None):
    tokenizer = result.tokenizer
    tokens = result.subtask_tokens if segment_name == "subtask" else result.task_tokens
    if not torch.is_tensor(tokens) or tokens.ndim < 2:
        return ""
    row = tokens[0].detach().cpu()
    if seg_start is None and 0 <= pos < row.shape[0]:
        token_idx = pos
    elif seg_start is not None and 0 <= pos - seg_start < row.shape[0]:
        token_idx = pos - seg_start
    elif 0 <= pos < row.shape[0]:
        token_idx = pos
    else:
        return ""
    return _decode_token_label(tokenizer, row[token_idx].item())


def _text_blocks_for_action_matrix(result: AttentionCaptureResult):
    blocks: list[tuple[str, list[int], int | None]] = []
    segment_lookup = {name: (s, e) for name, s, e in result.encoder_segments}
    for name in ("language", "subtask"):
        if name in segment_lookup:
            s, e = segment_lookup[name]
            blocks.append((name, list(range(s, e)), s))

    explicit = result.extras.get("text_token_indices_by_segment", {})
    for name, indices in explicit.items():
        if name not in {block[0] for block in blocks}:
            blocks.append((str(name), [int(i) for i in indices], None))
    return blocks


def _prompt_group_bounds(labels: list[str]) -> dict[str, tuple[int, int]]:
    """Return compact prompt groups over decoded non-image token labels.

    MolmoAct2 packs task, state, setup/control, advantage, and chat scaffolding
    into one text stream. The grouping is only a visualization aid; it does not
    imply separate model streams.
    """
    low = [label.lower() for label in labels]

    def first(patterns, start=0):
        for idx in range(start, len(low)):
            if any(pattern in low[idx] for pattern in patterns):
                return idx
        return None

    state_start = first(["state_start", "<state_"])
    state_end = None
    if state_start is not None:
        state_end = first(["state_end"], state_start)
        if state_end is None:
            # Six state values plus start/end is the common SO-100 case; fall
            # back conservatively if the tokenizer splits state tokens oddly.
            state_end = min(len(labels) - 1, state_start + 7)

    task_start = first(["task"])
    task_end_candidates = []
    if task_start is not None:
        for pats in (["setup"], ["current"], ["state_start", "<state_"], ["expected"], ["control"], ["advantage"], ["given"]):
            b = first(pats, task_start + 1)
            if b is not None:
                task_end_candidates.append(b)
    task_end = min(task_end_candidates) if task_end_candidates else None

    control_start_candidates = []
    for pats in (["setup"], ["expected"], ["control"], ["advantage"]):
        b = first(pats)
        if b is not None:
            control_start_candidates.append(b)
    control_start = min(control_start_candidates) if control_start_candidates else None
    control_end = first(["given"], control_start + 1) if control_start is not None else None

    bounds: dict[str, tuple[int, int]] = {}
    if task_start is not None:
        bounds["task"] = (task_start, task_end if task_end is not None else len(labels))
    if state_start is not None and state_end is not None:
        bounds["state"] = (state_start, min(len(labels), state_end + 1))
    if control_start is not None:
        bounds["setup/control/adv"] = (control_start, control_end if control_end is not None else len(labels))
    return bounds


def _group_prompt_indices(result: AttentionCaptureResult, text_blocks, encoder_len, encoder_valid):
    prompt_entries: list[tuple[int, str]] = []
    for name, indices, seg_start in text_blocks:
        if name == "subtask":
            for idx in indices:
                if 0 <= idx < encoder_len and (encoder_valid is None or bool(encoder_valid[idx])):
                    prompt_entries.append((idx, "subtask"))
            continue
        for idx in indices:
            if 0 <= idx < encoder_len and (encoder_valid is None or bool(encoder_valid[idx])):
                prompt_entries.append((idx, _token_label_for_position(result, name, idx, seg_start)))

    if not prompt_entries:
        return []

    labels = [label for _, label in prompt_entries]
    bounds = _prompt_group_bounds(labels)
    buckets: dict[str, list[int]] = {
        "task": [],
        "state": [],
        "setup/control/adv": [],
        "subtask": [],
        "other prompt": [],
    }

    for local_pos, (idx, label) in enumerate(prompt_entries):
        assigned = None
        if label == "subtask":
            assigned = "subtask"
        else:
            for group_name in ("state", "task", "setup/control/adv"):
                if group_name in bounds:
                    s, e = bounds[group_name]
                    if s <= local_pos < e:
                        assigned = group_name
                        break
        buckets[assigned or "other prompt"].append(idx)

    order = ["task", "state", "setup/control/adv", "subtask", "other prompt"]
    return [(name, buckets[name]) for name in order if buckets[name]]


def render_action_text_matrix(result: AttentionCaptureResult, layer_idx: int):
    """Compact action-query attention to prompt-token groups.

    MolmoAct2 receives one packed text stream. This view groups the non-image
    prompt tokens for readability; it is not a separate model pathway. The
    rows are normalized across the displayed prompt groups only, so this panel
    compares task/state/setup/etc. within the prompt, not prompt vs image vs
    action self-attention.
    """
    frames: dict[str, np.ndarray] = {}
    vmax_by_panel: dict[str, float] = {}

    cross = result.cross_attn_by_layer.get(layer_idx)
    if cross is None:
        return frames, vmax_by_panel

    text_blocks = _text_blocks_for_action_matrix(result)
    if not text_blocks:
        return frames, vmax_by_panel

    cross_attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)
    encoder_len = int(cross_attn.shape[-1])
    encoder_valid = None
    if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
        encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)

    prompt_groups = _group_prompt_indices(result, text_blocks, encoder_len, encoder_valid)
    if not prompt_groups:
        return frames, vmax_by_panel

    grouped_parts = []
    col_labels = []
    groups: list[tuple[str, int, int]] = []
    for col_pos, (name, indices) in enumerate(prompt_groups):
        idx_t = torch.as_tensor(indices, dtype=torch.long)
        # Sum attention mass over tokens in the group. This answers "how much
        # of the displayed prompt attention goes to this prompt section?"
        group_mass = cross_attn.index_select(dim=-1, index=idx_t).sum(dim=-1, keepdim=True)
        grouped_parts.append(group_mass)
        col_labels.append(f"{name}:{len(indices)}")
        groups.append((name, col_pos, col_pos + 1))

    grouped = torch.cat(grouped_parts, dim=-1)
    grouped = grouped / grouped.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mean = grouped.mean(dim=0)

    fixed_vmax = 1.0
    vmax_by_panel["action_text_matrix_mean"] = fixed_vmax
    vmax_by_panel["action_text_matrix_heads"] = fixed_vmax
    frames["action_text_matrix_mean"] = _render_action_text_panel(
        mean,
        fixed_vmax,
        col_labels,
        groups,
        f"L{layer_idx}: action -> prompt groups (mean, rows sum over prompt groups, scale 0-1)",
    )
    frames["action_text_matrix_heads"] = _render_action_text_heads(
        grouped,
        fixed_vmax,
        col_labels,
        groups,
        f"L{layer_idx}: action -> prompt groups (per-head, rows sum over prompt groups, scale 0-1)",
    )
    return frames, vmax_by_panel


# ──────────────────────────────────────────────────────────────────────────────
# Probe metadata diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _summarize_indices(indices):
    vals = [int(x) for x in indices]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "first": vals[:12],
        "last": vals[-12:],
    }


def _attention_metadata_summary(result: AttentionCaptureResult, layer_idx: int) -> dict[str, Any]:
    cross = result.cross_attn_by_layer.get(layer_idx)
    selfa = result.self_attn_by_layer.get(layer_idx)
    raw_indices: dict[str, Any] = {}
    for key in (
        "image_patch_indices_by_segment",
        "text_token_indices_by_segment",
        "matrix_col_indices_by_segment",
    ):
        value = result.extras.get(key)
        if isinstance(value, dict):
            raw_indices[key] = {str(name): _summarize_indices(indices) for name, indices in value.items()}

    display_segments = None
    if cross is not None:
        attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)
        idx, segments = _matrix_indices_from_result(result, int(attn.shape[-1]))
        display_segments = segments
        if idx is not None:
            raw_indices["cross_matrix_display_indices"] = _summarize_indices(idx.tolist())
        text_blocks = _text_blocks_for_action_matrix(result)
        encoder_valid = None
        if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
            encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)
        prompt_groups = _group_prompt_indices(result, text_blocks, int(attn.shape[-1]), encoder_valid)
        raw_indices["action_prompt_groups"] = {
            name: _summarize_indices(indices) for name, indices in prompt_groups
        }

    return {
        "layer": int(layer_idx),
        "cross_shape": list(cross.shape) if torch.is_tensor(cross) else None,
        "self_shape": list(selfa.shape) if torch.is_tensor(selfa) else None,
        "encoder_segments": [list(seg) for seg in result.encoder_segments],
        "cross_matrix_display_segments": [list(seg) for seg in display_segments or []],
        "patches_per_cam": int(result.patches_per_cam),
        "indices": raw_indices,
        "adapter_debug": result.extras.get("image_attention_debug", {}),
    }


def _append_attention_metadata(result: AttentionCaptureResult, layer_idx: int, ep_idx: int, fr_idx: int, ep_dir: str):
    summary = _attention_metadata_summary(result, layer_idx)
    summary.update({"ep": int(ep_idx), "fr": int(fr_idx)})
    path = os.path.join(ep_dir, "metadata.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")


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
    _warn_overcommit_if_risky("ATTN")

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

                _append_attention_metadata(result, layer_idx, ep_idx, fr_idx, ep_dir)

                panels: dict[str, np.ndarray] = {}
                vmaxes: dict[str, float] = {}
                for renderer in (render_image_overlays, render_cross_matrix, render_self_matrix, render_action_text_matrix):
                    p_frames, p_vmax = renderer(result, layer_idx)
                    panels.update(p_frames)
                    vmaxes.update(p_vmax)

                for panel, vmax in vmaxes.items():
                    csv_w.writerow([ep_idx, fr_idx, layer_idx, panel, f"{vmax:.6e}"])

                for key, frame_np in panels.items():
                    if key not in writers[layer_idx]:
                        out_path = os.path.join(ep_dir, f"{key}.mp4")
                        writers[layer_idx][key] = imageio.get_writer(
                            out_path, fps=fps, macro_block_size=1,
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

    _probe_dataset(adapter, primary_dataset, output_dir, attn_layers, timestep, cfg)

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
