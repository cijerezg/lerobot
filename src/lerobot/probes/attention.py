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
  overlay_depth_summary.mp4   — the same, over the point-map depth columns of the
                                joint softmax, drawn on the depth map (named
                                ``depth_nullbank`` when the frame carried no depth)
  action_to_prompt.png        — episode-aggregated prompt read: clause x head mass,
                                and the individual tokens that carry it
  norm_consts.csv             — per-panel vmax for stable colorbars

Every panel is anchored to something the reader can see — a camera frame, a depth
map, a decoded token. The raw matrix videos (action x 2000 encoder columns, mean
and per-head, plus action↔action self-attention) were removed 2026-08-01: at that
column count the picture is texture rather than signal, and the question they were
meant to answer — how the action queries divide their attention budget over
segments and over the chunk — is answered quantitatively, with compositional
statistics and a noise control, by ``attention_budget``.

Usage:
    python -m lerobot.probes.attention config.yaml \\
        --probe_parameters.max_episodes 5 \\
        --probe_parameters.timestep 0.5
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import AttentionCaptureResult, ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    build_episode_index,
    load_extra_dataset,
    load_probe_dataset,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
)
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


def build_episode_samples(
    dataset, episodes_str, random_n, subsample, seed=None, max_frames=None, grid_stride=1
):
    """Return ``[(ep_idx, [(fr_idx, global_idx), ...]), ...]``.

    ``max_frames`` widens the stride past ``subsample`` so that many frames span the
    whole episode: an unbounded walk of a 4000-frame episode is hours of rendering.
    ``grid_stride`` (``policy.image_stride``) rounds the walk up to a multiple of the
    stored image/depth grid, so every frame is one the model can actually be given
    depth for and one it was trained on.
    """
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
        stride = subsample
        if max_frames:
            stride = max(stride, math.ceil(len(indices) / max_frames))
        if grid_stride > 1:
            stride = max(grid_stride, math.ceil(stride / grid_stride) * grid_stride)
        ep_frames = [(fr_idx, indices[fr_idx]) for fr_idx in range(0, len(indices), stride)]
        if ep_frames:
            samples.append((ep_idx, ep_frames))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def cv2_overlay(img_np, heatmap, title, alpha=0.50, vmax=None, vmin=0.0):
    """Blend a heatmap onto a camera image."""
    if vmax is None:
        vmax = float(heatmap.max().item())
    span = max(float(vmax) - float(vmin), 1e-8)
    h_norm = (heatmap - float(vmin)) / span
    h_gray = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    if h_rgb.shape != img_np.shape:
        h_rgb = cv2.resize(h_rgb, (img_np.shape[1], img_np.shape[0]))
    out = cv2.addWeighted(img_np, 1 - alpha, h_rgb, alpha, 0)
    cv2.putText(out, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def cv2_heatmap(heatmap, title, img_h, img_w, vmax=None, vmin=0.0):
    """Render a standalone heatmap (no camera image)."""
    if vmax is None:
        vmax = float(heatmap.max().item())
    span = max(float(vmax) - float(vmin), 1e-8)
    h_norm = (heatmap - float(vmin)) / span
    h_gray = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    if h_rgb.shape[:2] != (img_h, img_w):
        h_rgb = cv2.resize(h_rgb, (img_w, img_h))
    cv2.putText(h_rgb, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return h_rgb


def _attn_values_to_image_heatmap(per_head: torch.Tensor, grid_hw, img_h: int, img_w: int):
    """Upsample per-head patch attention from [H, rows*cols] to image size.

    ``grid_hw`` is ``(rows, cols)``; camera segments pass a square grid, the
    point-map depth segment is 12x16.
    """
    n_heads = per_head.shape[0]
    rows, cols = (grid_hw, grid_hw) if isinstance(grid_hw, int) else grid_hw

    def _up(x):
        grid = x.reshape(rows, cols).float()
        up = F.interpolate(grid[None, None], size=(img_h, img_w),
                           mode="bicubic", align_corners=False)
        return up.squeeze().clamp(min=0)

    per_head_up = torch.stack([_up(per_head[h]) for h in range(n_heads)])
    mean_up = _up(per_head.mean(0))
    return per_head_up, mean_up


def _attn_to_patch_grid(cross_attn_layer, k_start, k_end):
    """Mean attention from all action queries to contiguous key positions.

    Returns ``per_head`` at patch-grid resolution ``[H, n_patches]``. Upsampling
    to image resolution is deferred to render time so episode-wide vmax stats
    can be aggregated cheaply.
    """
    slc = cross_attn_layer[:, :, k_start:k_end]      # [H, n_action, n_patches]
    return slc.mean(dim=1)                            # [H, n_patches]


def _attn_indices_to_patch_grid(cross_attn_layer, indices):
    """Mean attention to explicit, possibly non-contiguous image patch indices."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=cross_attn_layer.device)
    slc = cross_attn_layer.index_select(dim=2, index=idx)
    return slc.mean(dim=1)


def _attn_indices_to_pooled_patch_grid(cross_attn_layer, indices, pooling, patch_grid):
    """Project attention on pooled image tokens back to crop patch pixels.

    Molmo crop tokens are pooled from one or more ViT patches and are not always
    a square, contiguous token block. The adapter passes the local pooling rows
    for a crop; this scatters each token's attention back to the crop's patch
    grid. Returns ``[H, patch_grid*patch_grid]``.
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

    return patch_values / patch_counts.clamp_min(1.0).unsqueeze(0)


def _extract_overlay_grids(result: AttentionCaptureResult, layer_idx: int):
    """Pass-1 helper for episode-wide vmax: extract per-camera image and per-head
    patch-grid attention. Returns a list of dicts ready to feed ``_render_overlays_from_grids``.

    Each entry: ``{cam_name, img_np, per_head_grid [H, rows*cols], grid_hw}``. The
    grids live at patch resolution so an episode-wide percentile can be computed
    cheaply; upsampling to image size is deferred to the render pass. Camera grids
    are square; the point-map depth block appended at the end is 12x16.
    """
    cross_attn = result.cross_attn_by_layer.get(layer_idx)
    if cross_attn is None:
        return []
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
        return []

    attn = torch.nan_to_num(cross_attn[0].float().cpu(), nan=0.0)  # [H, n_act, enc]
    out: list[dict] = []
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
            per_head_grid = _attn_indices_to_pooled_patch_grid(attn, indices, pooling_rows, patch_grid)
            n_p = patch_grid
        elif indices is not None:
            n_p = int(len(indices) ** 0.5)
            if n_p * n_p != len(indices):
                continue
            per_head_grid = _attn_indices_to_patch_grid(attn, indices)
        else:
            if cs is None or ce is None:
                continue
            n_p = int(result.patches_per_cam ** 0.5)
            if n_p * n_p != ce - cs:
                continue
            per_head_grid = _attn_to_patch_grid(attn, cs, ce)
        out.append({
            "cam_name": cam_name,
            "img_np": img_np,
            "per_head_grid": per_head_grid.detach().cpu().float(),
            "grid_hw": (n_p, n_p),
        })

    depth_grid = _extract_depth_overlay_grid(result, attn)
    if depth_grid is not None:
        out.append(depth_grid)
    return out


def _extract_depth_overlay_grid(result: AttentionCaptureResult, attn: torch.Tensor):
    """Overlay entry for the point-map depth columns of the joint softmax.

    The depth block is a 12x16 row-major patch grid, not a square camera grid, and
    it is rendered over the depth map itself rather than a camera image. Returns
    ``None`` when the policy is depth-free or the grid could not be resolved.
    """
    segment = result.extras.get("depth_segment")
    if not segment or segment.get("grid_hw") is None:
        return None
    indices = [int(i) for i in segment["indices"] if 0 <= int(i) < int(attn.shape[-1])]
    if len(indices) != segment["grid_hw"][0] * segment["grid_hw"][1]:
        return None

    image = segment.get("image")
    if torch.is_tensor(image):
        img_t = image.squeeze(0).cpu() * 0.5 + 0.5
        img_np = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        rows, cols = segment["grid_hw"]
        img_np = np.zeros((rows * 16, cols * 16, 3), dtype=np.uint8)

    name = "depth_nullbank" if segment.get("is_null_bank") else "depth"
    return {
        "cam_name": name,
        "img_np": img_np,
        "per_head_grid": _attn_indices_to_patch_grid(attn, indices).detach().cpu().float(),
        "grid_hw": tuple(segment["grid_hw"]),
    }


def _render_overlays_from_grids(
    grids, vmax_overrides=None, *, mean_title_template="mean: action->{camera}"
):
    """Render image overlays from extracted grids.

    ``vmax_overrides`` (optional) carries fixed scales:
      - ``{cam}_mean`` → scalar vmax for the mean-over-heads panel.
      - ``{cam}_heads`` → sequence of length ``n_heads`` with one vmax per head.
    If absent for a key, the renderer falls back to that frame's local max.
    ``mean_title_template`` is formatted with ``camera=...`` and stays ASCII for
    OpenCV's built-in font. Returns ``(frames, vmax_by_panel)``;
    ``vmax_by_panel`` records the scalar that was actually used (for the heads
    panel, the max across the per-head values).
    """
    frames: dict[str, np.ndarray] = {}
    vmax_by_panel: dict[str, float] = {}
    if not grids:
        return frames, vmax_by_panel
    n_heads = int(grids[0]["per_head_grid"].shape[0])
    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols
    overrides = vmax_overrides or {}

    for g in grids:
        cam_name = g["cam_name"]
        img_np = g["img_np"]
        per_head_grid = g["per_head_grid"]
        img_h, img_w = img_np.shape[:2]

        per_head, mean_map = _attn_values_to_image_heatmap(
            per_head_grid, g["grid_hw"], img_h, img_w
        )

        mean_key = f"{cam_name}_mean"
        if mean_key in overrides:
            vmean = float(overrides[mean_key])
        else:
            vmean = float(mean_map.max().item())
        vmean_min = float(overrides.get(f"{cam_name}_mean_vmin", 0.0))

        heads_key = f"{cam_name}_heads"
        if heads_key in overrides:
            vheads_per = [float(v) for v in overrides[heads_key]]
            if len(vheads_per) != n_heads:
                vheads_per = (vheads_per + [vheads_per[-1]] * n_heads)[:n_heads]
        else:
            v_all = float(per_head.max().item())
            vheads_per = [v_all] * n_heads

        heads_vmin_raw = overrides.get(f"{cam_name}_heads_vmin")
        if heads_vmin_raw is None:
            vheads_min_per = [0.0] * n_heads
        else:
            vheads_min_per = [float(v) for v in heads_vmin_raw]
            if len(vheads_min_per) != n_heads:
                vheads_min_per = (vheads_min_per + [vheads_min_per[-1]] * n_heads)[:n_heads]

        vmax_by_panel[mean_key] = vmean
        vmax_by_panel[heads_key] = max(vheads_per) if vheads_per else 0.0

        summary = [img_np.copy()]
        cv2.putText(summary[0], f"{cam_name} (orig)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        summary.append(
            cv2_overlay(
                img_np,
                mean_map,
                mean_title_template.format(camera=cam_name),
                vmax=vmean,
                vmin=vmean_min,
            )
        )
        frames[f"overlay_{cam_name}_summary"] = np.hstack(summary)

        rows = []
        for r in range(h_rows):
            row_imgs = []
            for c in range(h_cols):
                idx = r * h_cols + c
                if idx < n_heads:
                    row_imgs.append(cv2_overlay(img_np, per_head[idx],
                                                f"{cam_name} h{idx}",
                                                vmax=vheads_per[idx],
                                                vmin=vheads_min_per[idx]))
                else:
                    row_imgs.append(np.zeros_like(img_np))
            rows.append(np.hstack(row_imgs))
        frames[f"overlay_{cam_name}_heads"] = np.vstack(rows)

    return frames, vmax_by_panel


def render_image_overlays(result: AttentionCaptureResult, layer_idx: int):
    """Per-frame vmax image overlays. Returns ``(frames, vmax)`` dicts keyed by
    panel name. Empty if there are no ``img*`` segments. The probe driver uses
    the two-pass path (``_extract_overlay_grids`` + ``_render_overlays_from_grids``
    with episode-wide vmax overrides); this wrapper preserves the original
    single-frame contract for any external callers.
    """
    grids = _extract_overlay_grids(result, layer_idx)
    return _render_overlays_from_grids(grids, vmax_overrides=None)


def _episode_overlay_vmax(buf, percentile: float = 98.0) -> dict:
    """Aggregate per-camera [p(100-percentile), p(percentile)] from buffered grids.

    Returns ``{cam}_mean`` / ``{cam}_mean_vmin`` (scalars) and
    ``{cam}_heads`` / ``{cam}_heads_vmin`` (per-head lists).
    """
    upper = float(percentile)
    lower = max(0.0, 100.0 - upper)

    mean_vals: dict[str, list[torch.Tensor]] = {}
    head_vals: dict[str, list[torch.Tensor]] = {}
    for _fr_idx, grids in buf:
        for g in grids:
            cam_name = g["cam_name"]
            per_head_grid = g["per_head_grid"]  # [H, n_patches]
            mean_vals.setdefault(cam_name, []).append(per_head_grid.mean(dim=0).flatten())
            head_vals.setdefault(cam_name, []).append(per_head_grid)

    out: dict[str, Any] = {}
    for cam_name, parts in mean_vals.items():
        cat = torch.cat(parts).numpy()
        out[f"{cam_name}_mean"] = max(float(np.percentile(cat, upper)), 1e-8)
        out[f"{cam_name}_mean_vmin"] = float(np.percentile(cat, lower))

    for cam_name, parts in head_vals.items():
        # Concat along the patch axis → [H, total_patches_across_frames]
        stacked = torch.cat(parts, dim=1).numpy()
        per_head_upper = np.percentile(stacked, upper, axis=1)
        per_head_lower = np.percentile(stacked, lower, axis=1)
        out[f"{cam_name}_heads"] = np.maximum(per_head_upper, 1e-8).tolist()
        out[f"{cam_name}_heads_vmin"] = per_head_lower.tolist()

    return out


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


# Clause markers for the MolmoAct2 action prompt, in the fixed order
# `_build_robot_text` emits them (processor_molmoact2.py):
#
#   "The task is to {task}." " The depth of the scene is {<extra_1> x N}."
#   " The current step is {subtask}."
#   " The current state of the robot is {<state_*>}."
#   " The recent states of the robot, oldest to newest, are {<extra_0> x T}."
#   " The quality is {n} of 5." " The robot made {a mistake|no mistakes}."
#   " Given these, what action should the robot take to complete the task?"
#
# Each entry is (group, substrings). Matching walks forward through the decoded
# labels and never rewinds, so a marker can only fire after the previous clause —
# that is what keeps "state" off the `<state_*>` values, and the trailing
# "...complete the task?" from being mistaken for the opening task clause. A
# clause that was dropped simply never matches, and the preceding group extends.
# "depth clause" covers only the words: the point-map placeholders themselves are
# dropped from the text segment upstream and carried by `depth_segment`, so the
# name here stays distinct from the "depth" token segment every panel already has.
_PROMPT_CLAUSE_MARKERS = (
    ("task", ("task",)),
    ("depth clause", ("depth",)),
    ("subtask", ("step",)),
    ("state", ("state",)),
    ("state history", ("recent",)),
    ("metadata", ("qual", "mistak")),
    ("question", ("given",)),
)

_PROMPT_GROUP_ORDER = (
    "task",
    "depth clause",
    "subtask",
    "state",
    "state history",
    "metadata",
    "question",
    "other prompt",
)


def _prompt_group_bounds(labels: list[str]) -> dict[str, tuple[int, int]]:
    """Return compact prompt groups over decoded non-image token labels.

    MolmoAct2 packs every clause into one text stream; this grouping is a
    visualization aid and does not imply separate model streams. It exists so the
    panel can answer which clause the action queries actually read — in
    particular the metadata (quality / mistake) steering clause and the proprio
    history placeholders, which older revisions lumped into "other prompt".
    """
    low = [label.lower() for label in labels]

    def clause_start(marker: int) -> int:
        """Back up from the keyword to the start of its sentence.

        Every clause is a sentence, so the tokens between the previous "." (or the
        end of a special token like ``<state_end>``) and the keyword — "The current",
        "The" — belong to this clause, not the previous one. Bounded to three tokens
        so an unexpected tokenization cannot swallow the whole prompt.
        """
        idx = marker
        for _ in range(3):
            if idx == 0 or "." in low[idx - 1] or ">" in low[idx - 1]:
                break
            idx -= 1
        return idx

    starts: list[tuple[str, int]] = []
    cursor = 0
    for name, patterns in _PROMPT_CLAUSE_MARKERS:
        for idx in range(cursor, len(low)):
            if any(pattern in low[idx] for pattern in patterns):
                starts.append((name, clause_start(idx)))
                cursor = idx + 1
                break

    bounds: dict[str, tuple[int, int]] = {}
    for position, (name, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(labels)
        bounds[name] = (start, end)
    return bounds


def _group_prompt_indices(result: AttentionCaptureResult, text_blocks, encoder_len, encoder_valid):
    """Return prompt tokens grouped by section, each entry carrying its decoded label.

    Shape: ``[(group_name, [(token_index, decoded_label), ...]), ...]``. The bounds
    that split task/state/setup-control/adv are computed over language tokens only;
    subtask tokens are assigned to their own group regardless of label content.
    """
    prompt_entries: list[tuple[int, str, str]] = []  # (idx, decoded_label, segment_name)
    for name, indices, seg_start in text_blocks:
        for idx in indices:
            if 0 <= idx < encoder_len and (encoder_valid is None or bool(encoder_valid[idx])):
                label = _token_label_for_position(result, name, idx, seg_start)
                prompt_entries.append((idx, label, name))

    if not prompt_entries:
        return []

    lang_labels = [label for _, label, seg in prompt_entries if seg == "language"]
    bounds = _prompt_group_bounds(lang_labels)
    buckets: dict[str, list[tuple[int, str]]] = {name: [] for name in _PROMPT_GROUP_ORDER}

    lang_pos = 0
    for idx, label, seg in prompt_entries:
        assigned = None
        if seg == "subtask":
            # pi05 carries the subtask in its own token stream; molmoact2 keeps it
            # inline and picks it up from the language bounds below.
            assigned = "subtask"
        else:
            for group_name, (s, e) in bounds.items():
                if s <= lang_pos < e:
                    assigned = group_name
                    break
            lang_pos += 1
        buckets[assigned or "other prompt"].append((idx, label))

    return [(name, buckets[name]) for name in _PROMPT_GROUP_ORDER if buckets[name]]


def _extract_prompt_mass(result: AttentionCaptureResult, layer_idx: int):
    """Pass-1 extract for the prompt panel: how much of each action query's softmax
    row lands on each prompt clause, and on each token inside it.

    Mass is absolute — a share of the whole encoder row, which sums to 1 over
    images, depth, prompt and chat scaffolding together — so a number here is
    directly comparable with ``attention_budget``. The deleted matrix renormalized
    each row within the prompt, which hides the failure that matters: a clause
    whose share *of the prompt* holds steady while the prompt as a whole stops
    being read.

    Returns per-head clause mass ``[H]`` averaged over action queries, and per
    token mass averaged over heads and queries, keyed by ``(clause, label)``.
    """
    cross = result.cross_attn_by_layer.get(layer_idx)
    if cross is None:
        return None
    text_blocks = _text_blocks_for_action_matrix(result)
    if not text_blocks:
        return None
    cross_attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)   # [H, n_act, enc]
    encoder_valid = None
    if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
        encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)
    prompt_groups = _group_prompt_indices(
        result, text_blocks, int(cross_attn.shape[-1]), encoder_valid
    )
    if not prompt_groups:
        return None

    by_head: dict[str, np.ndarray] = {}
    tokens: dict[tuple[str, str], float] = {}
    for name, entries in prompt_groups:
        idx = torch.as_tensor([i for i, _ in entries], dtype=torch.long)
        block = cross_attn.index_select(2, idx)                     # [H, n_act, n_tok]
        by_head[name] = block.sum(dim=2).mean(dim=1).numpy()        # [H]
        per_token = block.mean(dim=1).mean(dim=0).numpy()           # [n_tok]
        for (_, label), mass in zip(entries, per_token, strict=True):
            # A repeated word inside one clause ("the") is one entry carrying the
            # clause's total mass on that word, which is what the reader asks about.
            key = (name, label.strip() or "·")
            tokens[key] = tokens.get(key, 0.0) + float(mass)
    return {"by_head": by_head, "tokens": tokens}


def _aggregate_prompt_mass(buf: list[dict]) -> dict | None:
    """Mean over frames of the per-frame extracts.

    A token is identified by ``(clause, decoded label)`` rather than by a column
    index: the subtask and metadata clauses change length frame to frame, so column
    *k* is a different word in every frame and averaging over it is meaningless. A
    label missing from a frame contributes zero to its mean — the model cannot read
    a word that was not in that frame's prompt, and pretending otherwise would
    reward rare words.
    """
    if not buf:
        return None
    n_frames = len(buf)
    groups = [name for name in _PROMPT_GROUP_ORDER if any(name in d["by_head"] for d in buf)]
    n_heads = len(next(iter(buf[0]["by_head"].values())))

    by_head = np.zeros((n_heads, len(groups)), dtype=np.float64)
    for d in buf:
        for col, name in enumerate(groups):
            if name in d["by_head"]:
                by_head[:, col] += d["by_head"][name]
    by_head /= n_frames

    tokens: dict[tuple[str, str], float] = {}
    for d in buf:
        for key, mass in d["tokens"].items():
            tokens[key] = tokens.get(key, 0.0) + mass / n_frames

    return {"groups": groups, "by_head": by_head, "tokens": tokens, "n_frames": n_frames}


def _render_prompt_panel(
    agg: dict,
    layer_idx: int,
    out_path: str,
    top_k: int = 24,
    units: str = "share of the softmax row",
) -> None:
    """Two views of the same numbers: which clause each head reads, and which words
    carry that clause's mass. Both are episode means, which is the point — the
    per-frame version of this was a video nobody could read.

    ``units`` names what a cell holds. The Jacobian probe passes the same tensors
    through this renderer, and there a value is a causal magnitude rather than a
    probability — it does not sum to 1 over the encoder and must not be read as if
    it did.
    """
    groups = agg["groups"]
    by_head = agg["by_head"]
    n_heads = by_head.shape[0]
    color_of = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(_PROMPT_GROUP_ORDER)}

    fig, (ax_heads, ax_tokens) = plt.subplots(
        1, 2, figsize=(6.2 + 0.42 * len(groups) + 0.10 * top_k, 1.2 + 0.34 * max(n_heads, top_k)),
        gridspec_kw={"width_ratios": [max(len(groups), 4), 7]},
    )

    image = ax_heads.imshow(by_head, aspect="auto", cmap="magma", vmin=0.0)
    ax_heads.set_xticks(range(len(groups)))
    ax_heads.set_xticklabels(
        [f"{name}\n{by_head[:, col].mean():.3f}" for col, name in enumerate(groups)],
        rotation=45, ha="right", fontsize=8,
    )
    ax_heads.set_yticks(range(n_heads))
    ax_heads.set_yticklabels([f"h{h}" for h in range(n_heads)], fontsize=8)
    ax_heads.set_title(f"L{layer_idx}: clause mass per head\n(x-label = mean over heads)", fontsize=9)
    for h in range(n_heads):
        for col in range(len(groups)):
            value = by_head[h, col]
            ax_heads.text(
                col, h, f"{value:.3f}", ha="center", va="center", fontsize=6,
                color="white" if value < by_head.max() * 0.6 else "black",
            )
    fig.colorbar(image, ax=ax_heads, fraction=0.046, label=units)

    top = sorted(agg["tokens"].items(), key=lambda kv: kv[1], reverse=True)[:top_k][::-1]
    positions = np.arange(len(top))
    ax_tokens.barh(
        positions, [mass for _, mass in top],
        color=[color_of.get(clause, "#808080") for (clause, _), _ in top],
    )
    ax_tokens.set_yticks(positions)
    ax_tokens.set_yticklabels([f"{clause} · {label}" for (clause, label), _ in top], fontsize=7)
    ax_tokens.set_xlabel(f"{units} (mean over heads, queries, frames)", fontsize=8)
    ax_tokens.set_title(f"top {len(top)} prompt tokens by mass", fontsize=9)
    ax_tokens.grid(axis="x", alpha=0.25)
    ax_tokens.set_axisbelow(True)

    fig.suptitle(
        f"action queries → prompt, layer {layer_idx}, mean over {agg['n_frames']} frames",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)



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

    if cross is not None:
        encoder_len = int(cross.shape[-1])
        text_blocks = _text_blocks_for_action_matrix(result)
        encoder_valid = None
        if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
            encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)
        prompt_groups = _group_prompt_indices(result, text_blocks, encoder_len, encoder_valid)
        raw_indices["action_prompt_groups"] = {
            name: _summarize_indices([idx for idx, _ in entries]) for name, entries in prompt_groups
        }

    depth_segment = result.extras.get("depth_segment")
    if depth_segment:
        raw_indices["depth_columns"] = _summarize_indices(depth_segment["indices"])
        raw_indices["depth_columns"]["grid_hw"] = depth_segment.get("grid_hw")
        raw_indices["depth_columns"]["is_null_bank"] = bool(depth_segment.get("is_null_bank"))

    return {
        "layer": int(layer_idx),
        "cross_shape": list(cross.shape) if torch.is_tensor(cross) else None,
        "self_shape": list(selfa.shape) if torch.is_tensor(selfa) else None,
        "encoder_segments": [list(seg) for seg in result.encoder_segments],
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
        max_frames=p.n_frames_per_episode,
        grid_stride=probe_image_stride(cfg),
    )
    if not samples:
        logging.warning(f"  No samples in {ds_output_dir}, skipping.")
        return

    stride = samples[0][1][1][0] - samples[0][1][0][0] if len(samples[0][1]) > 1 else 1
    fps = min(10, 4 * getattr(ds, "fps", 30) / stride)  # <=4x real time, <=10 fps display
    logging.info(f"  {len(samples)} episode(s) x {len(samples[0][1])} frames (stride {stride})")
    _warn_overcommit_if_risky("ATTN")

    for ep_idx, ep_frames in samples:
        # Every panel gets an episode-fixed p98 vmax so colors are comparable
        # across frames within the video. Pass 1 extracts and buffers raw data;
        # pass 2 aggregates p98 per panel and renders.
        writers: dict[int, dict[str, Any]] = {l: {} for l in attn_layers}  # noqa: E741
        csv_files: dict[int, Any] = {}
        layer_buf: dict[int, list[dict]] = {l: [] for l in attn_layers}  # noqa: E741

        for fr_idx, global_idx in ep_frames:
            frame = probe_frame_inputs(ds, cfg, global_idx, chunk_size)
            result = adapter.capture_attention(
                frame["obs"], frame["task"], state=frame["state"], timestep=timestep,
                layers=attn_layers, subtask=frame["subtask"], metadata=frame["metadata"],
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

                _append_attention_metadata(result, layer_idx, ep_idx, fr_idx, ep_dir)

                layer_buf[layer_idx].append({
                    "fr_idx": fr_idx,
                    "overlay_grids": _extract_overlay_grids(result, layer_idx),
                    "prompt_mass": _extract_prompt_mass(result, layer_idx),
                })

        for layer_idx, frames_buf in layer_buf.items():
            if not frames_buf:
                continue
            overlay_buf = [(d["fr_idx"], d["overlay_grids"]) for d in frames_buf]
            all_vmax = _episode_overlay_vmax(overlay_buf, percentile=98.0)

            csv_f, csv_w = csv_files[layer_idx]
            ep_dir = os.path.join(ds_output_dir, f"ep{ep_idx:04d}_L{layer_idx:02d}")

            prompt_agg = _aggregate_prompt_mass(
                [d["prompt_mass"] for d in frames_buf if d["prompt_mass"] is not None]
            )
            if prompt_agg is not None:
                _render_prompt_panel(prompt_agg, layer_idx, os.path.join(ep_dir, "action_to_prompt.png"))

            for d in frames_buf:
                fr_idx = d["fr_idx"]
                panels, vmaxes = _render_overlays_from_grids(
                    d["overlay_grids"], vmax_overrides=all_vmax
                )

                for panel, vmax in vmaxes.items():
                    csv_w.writerow([ep_idx, fr_idx, layer_idx, panel, f"{vmax:.6e}"])

                for key, frame_np in panels.items():
                    if key not in writers[layer_idx]:
                        out_path = os.path.join(ep_dir, f"{key}.mp4")
                        writers[layer_idx][key] = imageio.get_writer(
                            out_path, fps=fps, macro_block_size=1,
                            # Panels are megabytes per raw frame; at a low fps ffmpeg
                            # can't estimate the rate within the default 5M probe.
                            input_params=["-probesize", "100M"],
                        )
                    writers[layer_idx][key].append_data(frame_np)

        for d in writers.values():
            for w in d.values():
                w.close()
        for f, _ in csv_files.values():
            f.close()


_PANEL_HOW = {
    "summary": "Mean over heads, drawn on the frame the queries were reading. Bright is where the action tokens spent their attention. Colours are comparable across frames of one video (episode-fixed p98 vmax) but not across videos.",
    "heads": "The same read split per head. Use it when the mean looks uniform: a single head fixating while the rest spread out is invisible in the summary.",
    "depth": "The point-map depth columns of the joint softmax, drawn on the depth map. Named ``depth_nullbank`` when the frame carried no depth, in which case the mass is on the learned null bank rather than on a scene.",
    "prompt": "Episode-aggregated prompt read: which clause the action queries take mass from, per head, and the individual tokens carrying it. Quantitative budget accounting lives in ``attention_budget``.",
    "norm": "Per-panel vmax used for each frame's colorbar, so a claim about a colour change can be checked against the scale it was drawn on.",
}


def _panel_how(filename: str) -> str:
    if filename.startswith("action_to_prompt"):
        return _PANEL_HOW["prompt"]
    if filename.startswith("norm_consts"):
        return _PANEL_HOW["norm"]
    if "depth" in filename:
        return _PANEL_HOW["depth"]
    return _PANEL_HOW["heads"] if filename.endswith("_heads.mp4") else _PANEL_HOW["summary"]


_PANEL_SOURCE = {
    "img_external_0": "first external camera",
    "img_external_1": "second external camera",
    "img_wrist_0": "wrist camera",
    "depth": "depth map",
    "depth_nullbank": "depth null bank",
}


def _panel_caption(filename: str) -> str:
    """What the panel shows, in words — the file name is printed by the viewer anyway."""
    if filename.startswith("action_to_prompt"):
        return "prompt read, per head and per token"
    if filename.startswith("norm_consts"):
        return "colorbar scale behind every frame"
    source, _, split = filename.rsplit(".", 1)[0].removeprefix("overlay_").rpartition("_")
    source = _PANEL_SOURCE.get(source, source.replace("_", " "))
    return f"{source}, {'per head' if split == 'heads' else 'mean over heads'}"


def _write_manifest(output_dir: str, layers: list[int], timestep: float) -> dict:
    """Describe the rendered overlay set to the manifest-driven viewer.

    The panels are discovered from disk rather than declared: which cameras exist, and
    whether a depth panel was drawn at all, is a property of the checkpoint and the
    frames, not of this probe.
    """
    panels = []
    for ep_dir in sorted(os.listdir(output_dir)):
        match = re.fullmatch(r"ep(\d+)_L(\d+)", ep_dir)
        if match is None:
            continue
        episode, layer = int(match.group(1)), int(match.group(2))
        for filename in sorted(os.listdir(os.path.join(output_dir, ep_dir))):
            if filename == "metadata.jsonl":
                continue
            panels.append(
                Panel(
                    f"{ep_dir}/{filename}",
                    f"Episode {episode}, layer {layer} — {_panel_caption(filename)}",
                    how=_panel_how(filename),
                    # One camera at the shallowest layer of the first episode: enough to
                    # see whether anything is being read at all. The rest is fan-out.
                    primary=episode == 0
                    and layer == min(layers)
                    and filename.endswith("_summary.mp4")
                    and "depth" not in filename,
                )
            )
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Attention Maps",
        group="Attention",
        claim="Where on the cameras, the depth map, and the prompt do the action queries look?",
        metrics=[
            Metric("layers", "Layers captured", good="none", fmt=0, value=float(len(layers))),
            Metric("timestep", "Flow timestep", good="none", fmt=2, value=timestep),
        ],
        panels=panels,
        status="info",
        see_also=["attention_budget", "spatial_memorization_attention", "action_drift_jacobian"],
    )


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
    _write_manifest(output_dir, attn_layers, timestep)

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

    dataset = load_probe_dataset(cfg)

    logging.info("Loading policy adapter …")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    run(adapter, dataset, cfg, output_dir)
    logging.info(f"Done. Output saved to {output_dir}/")


if __name__ == "__main__":
    register_config_choices()
    probe_cli()
