#!/usr/bin/env python
"""
Attention probing script for PI05-Full policy.

Loads a dataset, runs a single joint forward pass (no denoising loop) per sample
at one or more diffusion timesteps, captures the layer-0 attention matrix, and
produces two output types per (sample, timestep):

  1. overlay_<ep>_<fr>_t<T>.mp4
       For each camera: mean-head overlays (all / language / subtask / action queries)
       + per-head breakdown for the "all queries" view.

  2. matrix_<ep>_<fr>_t<T>.mp4
       Full [seq × seq] attention matrix (mean across heads) with labeled sections
       and per-head grid.

Usage:
    python attention_pi05.py config.json \\
        --probe_parameters.max_episodes 5 \\
        --probe_parameters.output_dir outputs/probe \\
        --probe_parameters.timestep 0.5
"""

import csv
import logging
import os
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.pi05_full.modeling_pi05 import (
    _PROBING_CAPTURE,
    make_att_2d_masks,
)
from lerobot.types import TransitionKey
from lerobot.probes.offline_inference_pi05 import _build_episode_index, get_frame_data
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

from lerobot.probes.utils_pi05 import (
    load_extra_dataset,
    load_policy_and_processors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeAttentionConfig(TrainRLServerPipelineConfig):
    """Extends the base training config with attention probe parameters.

    All probe tunables live under cfg.probe_parameters (ProbeConfig).
    Relevant fields for this script:
    # ─ Attention / spatial
      validation_batch_size, attn_eval_episodes, attn_eval_subsample,
      max_episodes, random_seed.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Sample selection
# ──────────────────────────────────────────────────────────────────────────────

def build_sample_list(dataset, episodes_str, random_n, subsample, seed=None):
    """
    Build per-episode sample lists.

    Returns [(ep_idx, [(fr_idx, global_idx), ...]), ...].
    Unlike sample_episodes_evenly (which returns a flat list of evenly spaced frames),
    this function subsamples by a fixed stride and optionally picks random episodes.
    """
    ep_to_indices = _build_episode_index(dataset)
    selected_eps  = []

    if episodes_str:
        for ep_idx in [int(e) for e in episodes_str.split(",")]:
            if ep_idx in ep_to_indices:
                selected_eps.append(ep_idx)

    if random_n:
        rng      = random.Random(seed)
        all_eps  = list(ep_to_indices.keys())
        rng.shuffle(all_eps)
        for ep_idx in all_eps:
            if len(selected_eps) >= random_n:
                break
            if ep_idx not in selected_eps:
                selected_eps.append(ep_idx)

    samples = []
    for ep_idx in selected_eps:
        indices   = ep_to_indices[ep_idx]
        ep_frames = [
            (fr_idx, indices[fr_idx])
            for fr_idx in range(0, len(indices), subsample)
        ]
        if ep_frames:
            samples.append((ep_idx, ep_frames))

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Probing forward — no FAST tokens, no loss, one timestep
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_probe_prefix(policy, images, img_masks, task_tokens, task_masks,
                       subtask_tokens, subtask_masks):
    """
    Embed the prefix (images + language + subtask) once per batch.
    Returns a cache dict to be passed to probe_forward for each timestep,
    avoiding redundant vision-encoder passes across timesteps.
    """
    model = policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks, image_len = model.embed_prefix(
        images, img_masks,
        task_tokens, subtask_tokens,
        task_masks, subtask_masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    )
    w_dtype = (model.paligemma_with_expert.paligemma
               .language_model.layers[0].self_attn.q_proj.weight.dtype)
    if w_dtype == torch.bfloat16:
        prefix_embs = prefix_embs.to(torch.bfloat16)

    n_cameras       = len(images)
    patches_per_cam = image_len // n_cameras

    segments_prefix = []
    pos = 0
    for i in range(n_cameras):
        segments_prefix.append((f"img{i + 1}", pos, pos + patches_per_cam))
        pos += patches_per_cam
    segments_prefix.append(("language", pos, pos + task_tokens.shape[1]))
    pos += task_tokens.shape[1]
    segments_prefix.append(("subtask",  pos, pos + subtask_tokens.shape[1]))

    return {
        "model":             model,
        "prefix_embs":       prefix_embs,
        "prefix_pad_masks":  prefix_pad_masks,
        "prefix_att_masks":  prefix_att_masks,
        "w_dtype":           w_dtype,
        "bsize":             task_tokens.shape[0],
        "patches_per_cam":   patches_per_cam,
        "segments_prefix":   segments_prefix,   # without "action" segment
    }


@torch.no_grad()
def probe_forward(prefix_cache, time_val, device):
    """
    Single joint forward (prefix + suffix) without FAST tokens.
    Captures attention from all layers via _PROBING_CAPTURE['attn_weights_by_layer'].

    Returns
    -------
    attn_by_layer   : dict  layer_idx -> Tensor [B, n_heads, seq_len, seq_len]
    segments        : list of (name, start, end)
    pad_masks       : Tensor [B, seq_len]  bool
    patches_per_cam : int
    """
    model             = prefix_cache["model"]
    prefix_embs       = prefix_cache["prefix_embs"]
    prefix_pad_masks  = prefix_cache["prefix_pad_masks"]
    prefix_att_masks  = prefix_cache["prefix_att_masks"]
    w_dtype           = prefix_cache["w_dtype"]
    bsize             = prefix_cache["bsize"]
    patches_per_cam   = prefix_cache["patches_per_cam"]
    segments_prefix   = prefix_cache["segments_prefix"]

    # ── Embed suffix (pure noise at time_val) ─────────────────────────────────
    noise = model.sample_noise(
        (bsize, model.config.chunk_size, model.config.max_action_dim), device
    )
    time_tensor = torch.full((bsize,), time_val, dtype=torch.float32, device=device)

    if w_dtype == torch.bfloat16:
        noise       = noise.to(torch.bfloat16)
        time_tensor = time_tensor.to(torch.bfloat16)

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
        noise, time_tensor
    )

    if w_dtype == torch.bfloat16:
        suffix_embs = suffix_embs.to(torch.bfloat16)

    # ── Combined attention mask ────────────────────────────────────────────────
    prefix_len = prefix_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]
    total_len  = prefix_len + suffix_len

    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    combined = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
    combined[:, :prefix_len, :prefix_len] = prefix_att_masks
    combined[:, prefix_len:, prefix_len:] = suffix_att_2d
    combined[:, prefix_len:, :prefix_len] = True   # suffix sees all prefix

    combined_pad = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    pad_2d       = combined_pad[:, None, :] & combined_pad[:, :, None]
    att_2d       = combined & pad_2d

    position_ids = torch.cumsum(combined_pad, dim=1) - 1
    att_2d_4d    = model._prepare_attention_masks_4d(att_2d)

    # ── Enable capture (all layers) and run ────────────────────────────────────
    _PROBING_CAPTURE["enabled"]               = True
    _PROBING_CAPTURE["all_layers"]            = True
    _PROBING_CAPTURE["attn_weights_by_layer"] = {}

    model.paligemma_with_expert.forward(
        attention_mask=att_2d_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    _PROBING_CAPTURE["enabled"]    = False
    _PROBING_CAPTURE["all_layers"] = False

    attn_by_layer = _PROBING_CAPTURE["attn_weights_by_layer"]

    # ── Append "action" segment ───────────────────────────────────────────────
    action_start = segments_prefix[-1][2]  # end of "subtask"
    segments = segments_prefix + [("action", action_start, action_start + suffix_embs.shape[1])]

    return attn_by_layer, segments, combined_pad, patches_per_cam


# ──────────────────────────────────────────────────────────────────────────────
# Attention → spatial heatmap helpers
# ──────────────────────────────────────────────────────────────────────────────

def _attn_to_heatmap(attn_heads, q_start, q_end, k_start, k_end,
                     pad_masks, n_p, img_h, img_w):
    """
    Average attention from query rows [q_start:q_end] to key cols [k_start:k_end],
    weighted by valid (non-padding) queries. Returns per-head heatmaps and the
    head-mean heatmap, both upsampled to (img_h, img_w).

    attn_heads : [n_heads, seq_len, seq_len]  float32
    pad_masks  : [seq_len]  bool
    Returns    : per_head [n_heads, img_h, img_w], mean [img_h, img_w]
    """
    n_heads = attn_heads.shape[0]
    q_valid = pad_masks[q_start:q_end]
    q_attn  = attn_heads[:, q_start:q_end, k_start:k_end]

    if q_valid.sum() == 0:
        dummy = torch.zeros(img_h, img_w)
        return dummy.unsqueeze(0).expand(n_heads, -1, -1), dummy

    q_attn_valid = q_attn[:, q_valid, :]   # [heads, valid_q, n_patches]
    per_head     = q_attn_valid.mean(dim=1) # [heads, n_patches]

    def _reshape_upsample(x):
        grid    = x.reshape(n_p, n_p)
        grid_4d = grid[None, None].float()
        up      = F.interpolate(grid_4d, size=(img_h, img_w), mode="bicubic",
                                align_corners=False)
        return up.squeeze().clamp(min=0)

    per_head_up = torch.stack([_reshape_upsample(per_head[h]) for h in range(n_heads)])
    mean_up     = _reshape_upsample(per_head.mean(0))
    return per_head_up, mean_up


# ──────────────────────────────────────────────────────────────────────────────
# Figure generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def cv2_overlay(img_np, heatmap, title, alpha=0.55, vmax=None):
    if vmax is None:
        vmax = heatmap.max().item()
    h_norm  = heatmap / (vmax + 1e-8)
    h_gray  = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_color_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)

    if h_color_rgb.shape != img_np.shape:
        h_color_rgb = cv2.resize(h_color_rgb, (img_np.shape[1], img_np.shape[0]))

    overlay = cv2.addWeighted(img_np, 1 - alpha, h_color_rgb, alpha, 0)
    cv2.putText(overlay, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def cv2_heatmap(heatmap, title, img_h, img_w, vmax=None):
    """Render a standalone heatmap (no camera image overlay).

    Used by deeper-layer captures where blending onto the input image is
    misleading because attention has already mixed across positions.
    """
    if vmax is None:
        vmax = heatmap.max().item()
    h_norm = heatmap / (vmax + 1e-8)
    h_gray = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_color_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)

    if h_color_rgb.shape[:2] != (img_h, img_w):
        h_color_rgb = cv2.resize(h_color_rgb, (img_w, img_h))

    cv2.putText(h_color_rgb, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return h_color_rgb


def render_image_overlays(attn_weights, segments, image_tensors, pad_masks,
                          patches_per_cam, overlay=True):
    """
    Render attention in two formats per camera:

    1. **Summary strip**: orig (if overlay) + mean-over-heads per query group
       → key ``{prefix}_{cam}_summary``
    2. **Disaggregated head grid** per (query_group, cam): 2×4 per-head panels
       → key ``{prefix}_{cam}_{qname}_heads``

    When overlay=True (layer 0), heatmaps are blended onto the camera image.
    When overlay=False (deeper layers), standalone heatmaps are rendered.

    Returns
    -------
    frames_out  : dict  key → np.ndarray  (video frames)
    norm_consts : dict  key → float  (raw peak attention value used as vmax per panel)
    """
    frames_out  = {}
    norm_consts = {}
    n_heads     = attn_weights.shape[1]
    n_p         = int(patches_per_cam ** 0.5)

    seg_dict    = {name: (s, e) for name, s, e in segments}
    camera_segs = [(name, s, e) for name, s, e in segments if name.startswith("img")]
    query_groups = [
        ("all",      0,                           segments[-1][2]),
        ("language", *seg_dict["language"]),
        ("subtask",  *seg_dict["subtask"]),
        ("action",   *seg_dict["action"]),
    ]

    attn = torch.nan_to_num(attn_weights[0].float().cpu(), nan=0.0)
    pad  = pad_masks[0].cpu()

    prefix = "overlay" if overlay else "heatmap"
    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols

    for cam_idx, (cam_name, cam_s, cam_e) in enumerate(camera_segs):
        img_np = None
        if cam_idx < len(image_tensors):
            img_t  = image_tensors[cam_idx].squeeze(0).cpu()
            img_t  = img_t * 0.5 + 0.5
            img_np = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        img_h = img_np.shape[0] if img_np is not None else 224
        img_w = img_np.shape[1] if img_np is not None else 224
        if img_np is None:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # ── Summary strip: mean over heads, columns = query groups ─────────
        summary_panels = []
        if overlay:
            summary_panels.append(img_np.copy())
            cv2.putText(summary_panels[0], f"{cam_name} (orig)", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for qname, q_s, q_e in query_groups:
            _, mean_map = _attn_to_heatmap(
                attn, q_s, q_e, cam_s, cam_e, pad, n_p, img_h, img_w
            )
            vmax = mean_map.max().item()
            norm_consts[f"{cam_name}_{qname}_mean"] = vmax
            if overlay:
                summary_panels.append(cv2_overlay(img_np, mean_map, f"mean: {qname}", vmax=vmax))
            else:
                summary_panels.append(cv2_heatmap(mean_map, f"mean: {qname}", img_h, img_w, vmax=vmax))

        frames_out[f"{prefix}_{cam_name}_summary"] = np.hstack(summary_panels)

        # ── Per (query_group, cam) head grids ──────────────────────────────
        for qname, q_s, q_e in query_groups:
            per_head_maps, _ = _attn_to_heatmap(
                attn, q_s, q_e, cam_s, cam_e, pad, n_p, img_h, img_w
            )
            heads_vmax = per_head_maps.max().item()
            norm_consts[f"{cam_name}_{qname}_heads"] = heads_vmax

            head_rows = []
            for r in range(h_rows):
                row_imgs = []
                for c in range(h_cols):
                    idx = r * h_cols + c
                    if idx < n_heads:
                        if overlay:
                            row_imgs.append(cv2_overlay(img_np, per_head_maps[idx],
                                                        f"{qname} h{idx}", vmax=heads_vmax))
                        else:
                            row_imgs.append(cv2_heatmap(per_head_maps[idx],
                                                        f"{qname} h{idx}", img_h, img_w, vmax=heads_vmax))
                    else:
                        row_imgs.append(np.zeros_like(img_np))
                head_rows.append(np.hstack(row_imgs))
            frames_out[f"{prefix}_{cam_name}_{qname}_heads"] = np.vstack(head_rows)


    return frames_out, norm_consts


# ──────────────────────────────────────────────────────────────────────────────
# Action → [action | language | subtask] focused matrix
# ──────────────────────────────────────────────────────────────────────────────

def _decode_token_label(tokenizer, tid):
    """Decode a single token ID to a readable ASCII label.

    Uses tokenizer.decode() (matches the rest of the codebase, e.g. rl_pi05.py
    and offline_inference_pi05.py); convert_ids_to_tokens leaks SentencePiece
    markers (U+2581) which cv2's Hershey font can't render and shows as '?'.
    Empty/whitespace-only results return "" so the label is skipped and the
    column renders as a blank separator (e.g. between digits of a state value).
    """
    text = tokenizer.decode([int(tid)], skip_special_tokens=False)
    text = text.replace("\n", " ").strip()
    if not text:
        return ""
    return text.encode("ascii", errors="replace").decode("ascii")


def _draw_rotated_text(canvas, text, x_center, y_top, font_scale, color):
    """Render `text` rotated 90° CW (reads top-to-bottom) onto `canvas` in-place,
    centered horizontally at x_center, starting at y_top. Uses max-blend so it
    won't erase existing dividers."""
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    pad = 2
    sub = np.zeros((th + baseline + pad * 2, tw + pad * 2, 3), dtype=np.uint8)
    cv2.putText(sub, text, (pad, th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    rotated = cv2.rotate(sub, cv2.ROTATE_90_CLOCKWISE)
    rh, rw = rotated.shape[:2]
    x0 = max(0, x_center - rw // 2)
    y0 = y_top
    x1 = min(canvas.shape[1], x0 + rw)
    y1 = min(canvas.shape[0], y0 + rh)
    if x1 > x0 and y1 > y0:
        roi = canvas[y0:y1, x0:x1]
        np.maximum(roi, rotated[: y1 - y0, : x1 - x0], out=roi)


def _render_matrix_panel(
    mat, vmax, col_labels, n_action, div1, div2, n_lang, n_subtask,
    title, out_w, out_h,
    title_font_scale=0.55, label_font_scale=0.35, group_font_scale=0.5,
):
    """Render a single attention matrix panel using cv2.

    Layout: title strip on top, y-axis index labels on left, heatmap in the
    middle, group annotation strip + rotated column labels on the bottom.
    """
    margin_top    = 32
    margin_left   = 42
    margin_right  = 8
    margin_bottom = 140

    hm_x0 = margin_left
    hm_y0 = margin_top
    hm_w  = out_w - margin_left - margin_right
    hm_h  = out_h - margin_top - margin_bottom

    arr = mat.numpy() if hasattr(mat, "numpy") else mat
    norm = (arr / max(vmax, 1e-8)).clip(0, 1)
    gray = (norm * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb = cv2.resize(color_rgb, (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[hm_y0 : hm_y0 + hm_h, hm_x0 : hm_x0 + hm_w] = color_rgb

    cv2.putText(canvas, title, (hm_x0 + 4, margin_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)

    n_cols     = len(col_labels)
    px_per_col = hm_w / max(n_cols, 1)
    px_per_row = hm_h / max(n_action, 1)

    for ai in range(0, n_action, 5):
        y = int(hm_y0 + (ai + 0.5) * px_per_row + 4)
        cv2.putText(canvas, str(ai), (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    for div in (div1, div2):
        x = int(hm_x0 + div * px_per_col)
        cv2.line(canvas, (x, hm_y0), (x, hm_y0 + hm_h), (255, 255, 255), 1)

    annot_y = hm_y0 + hm_h + 18

    def _label_group(name, x_start_col, x_end_col):
        x_mid = int(hm_x0 + ((x_start_col + x_end_col) / 2) * px_per_col)
        (tw, _th), _ = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, group_font_scale, 1
        )
        cv2.putText(canvas, name, (x_mid - tw // 2, annot_y),
                    cv2.FONT_HERSHEY_SIMPLEX, group_font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    _label_group("action self", 0, div1)
    if n_lang > 0:
        _label_group("language", div1, div1 + n_lang)
    if n_subtask > 0:
        _label_group("subtask", div2, div2 + n_subtask)

    label_y = hm_y0 + hm_h + 28
    for ci, text in enumerate(col_labels):
        if not text:
            continue
        x_center = int(hm_x0 + (ci + 0.5) * px_per_col)
        _draw_rotated_text(
            canvas, text, x_center, label_y, label_font_scale, (220, 220, 220)
        )

    return canvas


def _render_heads_grid(
    per_head_mat, vmax, col_labels, n_action, div1, div2, n_lang, n_subtask,
    out_w, out_h, suptitle,
):
    """Tile per-head panels into a 2×4 grid below a suptitle strip."""
    title_h = 36
    h_cols, h_rows = 4, 2
    grid_h  = out_h - title_h
    panel_w = out_w // h_cols
    panel_h = grid_h // h_rows

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    cv2.putText(canvas, suptitle, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    H = per_head_mat.shape[0]
    for hi in range(H):
        r = hi // h_cols
        c = hi % h_cols
        sub = _render_matrix_panel(
            per_head_mat[hi], vmax, col_labels, n_action, div1, div2,
            n_lang, n_subtask,
            title=f"head {hi}", out_w=panel_w, out_h=panel_h,
            title_font_scale=0.45, label_font_scale=0.30, group_font_scale=0.4,
        )
        canvas[title_h + r * panel_h : title_h + (r + 1) * panel_h,
               c * panel_w : (c + 1) * panel_w] = sub
    return canvas


def render_action_to_prefix_matrix(
    attn_weights, segments, pad_masks,
    task_tokens, subtask_tokens, tokenizer,
    label_prefix="matrix",
    max_lang_cols=48, max_subtask_cols=16,
):
    """
    Attention from action queries to [action | language | subtask] keys.

    Drops image columns (already covered by overlays). Each row is renormalized
    over the displayed columns so values reflect relative weighting among
    {action self, language, subtask} rather than tiny residuals after most
    mass landed on image patches.

    To keep column positions stable across frames (subtask length varies
    frame-to-frame), the language and subtask blocks always take a fixed
    `max_*_cols` budget. Pad positions within that budget render as black
    (their attention is masked to 0, so they contribute 0 to the row sum
    and don't affect renormalization) and have empty labels.

    Parameters
    ----------
    attn_weights     : Tensor [1, H, seq, seq]
    segments         : list of (name, start, end)
    pad_masks        : Tensor [1, seq]
    task_tokens      : Tensor [1, T_lang]   raw token IDs (pre-pad-strip)
    subtask_tokens   : Tensor [1, T_subtask]
    tokenizer        : transformers tokenizer (e.g. policy.model._paligemma_tokenizer)
    label_prefix     : str — prepended to output dict keys
    max_lang_cols    : fixed display budget for language tokens
    max_subtask_cols : fixed display budget for subtask tokens

    Returns
    -------
    frames_out  : dict
        - "{label_prefix}_action_to_prefix_mean"  : [Hp, Wp, 3] uint8
        - "{label_prefix}_action_to_prefix_heads" : [Hp, Wp, 3] uint8 (2×4 grid)
    norm_consts : dict  panel → vmax
    """
    seg_dict = {name: (s, e) for name, s, e in segments}
    a_s, a_e = seg_dict["action"]
    l_s, l_e = seg_dict["language"]
    s_s, s_e = seg_dict["subtask"]

    attn = torch.nan_to_num(attn_weights[0].float().cpu(), nan=0.0)  # [H, seq, seq]
    pad  = pad_masks[0].cpu()
    H    = attn.shape[0]

    # Fixed display budgets so column positions are stable across frames.
    n_lang    = min(l_e - l_s, max_lang_cols)
    n_subtask = min(s_e - s_s, max_subtask_cols)

    action_col_idx  = list(range(a_s, a_e))
    lang_col_idx    = list(range(l_s, l_s + n_lang))
    subtask_col_idx = list(range(s_s, s_s + n_subtask))
    cols            = action_col_idx + lang_col_idx + subtask_col_idx

    n_action = a_e - a_s

    # Labels: blank for pad positions, decoded text for valid ones.
    lang_valid_slice    = pad[l_s : l_s + n_lang]
    subtask_valid_slice = pad[s_s : s_s + n_subtask]

    lang_labels = [
        _decode_token_label(tokenizer, task_tokens[0, i].item()) if v else ""
        for i, v in enumerate(lang_valid_slice.tolist())
    ]
    subtask_labels = [
        _decode_token_label(tokenizer, subtask_tokens[0, i].item()) if v else ""
        for i, v in enumerate(subtask_valid_slice.tolist())
    ]
    action_labels  = [str(i) if (i % 5 == 0) else "" for i in range(n_action)]
    all_col_labels = action_labels + lang_labels + subtask_labels

    # Slice and renormalize each row over displayed cols
    sliced      = attn[:, a_s:a_e][:, :, cols]           # [H, n_action, n_cols]
    row_sum     = sliced.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    sliced_norm = sliced / row_sum

    div1 = n_action               # action ↔ language boundary
    div2 = n_action + n_lang      # language ↔ subtask boundary

    # Fixed output dims (constant across frames so imageio writers stay happy).
    MEAN_W,  MEAN_H  = 2400, 900
    HEADS_W, HEADS_H = 3600, 1350

    frames_out  = {}
    norm_consts = {}

    # ── Mean panel ────────────────────────────────────────────────────────────
    mean_mat  = sliced_norm.mean(0)
    mean_vmax = mean_mat.max().item()
    norm_consts[f"{label_prefix}_action_to_prefix_mean"] = mean_vmax

    frames_out[f"{label_prefix}_action_to_prefix_mean"] = _render_matrix_panel(
        mean_mat, mean_vmax, all_col_labels, n_action, div1, div2,
        n_lang, n_subtask,
        title="action → [action | language | subtask] (mean over heads, row-normalized)",
        out_w=MEAN_W, out_h=MEAN_H,
    )

    # ── Per-head 2×4 grid ─────────────────────────────────────────────────────
    heads_vmax = sliced_norm.max().item()
    norm_consts[f"{label_prefix}_action_to_prefix_heads"] = heads_vmax

    frames_out[f"{label_prefix}_action_to_prefix_heads"] = _render_heads_grid(
        sliced_norm, heads_vmax, all_col_labels, n_action, div1, div2,
        n_lang, n_subtask,
        out_w=HEADS_W, out_h=HEADS_H,
        suptitle="action → [action | language | subtask] (per-head, row-normalized)",
    )

    return frames_out, norm_consts


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: ProbeAttentionConfig):
    init_logging()
    p          = cfg.probe_parameters
    device     = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "attention")
    os.makedirs(output_dir, exist_ok=True)

    attn_layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    logging.info(f"Probing layers: {attn_layers}")
    logging.info(f"Timestep: {p.timestep}")
    logging.info(f"Output dir: {output_dir}")

    logging.info("Loading policy …")
    policy, preprocessor, _, dataset = load_policy_and_processors(cfg, device)
    policy.eval()

    chunk_size = cfg.policy.n_action_steps

    def _probe_dataset(ds, ds_output_dir):
        os.makedirs(ds_output_dir, exist_ok=True)
        samples = build_sample_list(
            ds,
            episodes_str=p.attn_eval_episodes,
            random_n=p.max_episodes,
            subsample=p.attn_eval_subsample,
            seed=p.random_seed,
        )
        if not samples:
            logging.warning(f"  No samples found in {ds_output_dir}, skipping.")
            return

        fps      = getattr(ds, "fps", 30) / p.attn_eval_subsample
        batch_sz = getattr(p, "validation_batch_size", 32)

        for ep_idx, ep_frames in samples:
            # writers[layer_idx][video_key] = imageio writer
            writers = {l: {} for l in attn_layers}

            for i in range(0, len(ep_frames), batch_sz):
                batch_slice = ep_frames[i : i + batch_sz]

                b_obs      = {}
                b_task_str = []
                for fr_idx, global_idx in batch_slice:
                    obs, _, state, gt_subtask, task_str, _, _ = get_frame_data(
                        ds, global_idx, chunk_size
                    )
                    b_task_str.append(task_str)
                    for k, v in obs.items():
                        if k not in b_obs:
                            b_obs[k] = []
                        b_obs[k].append(v)

                logging.debug(
                    f"    ep={ep_idx:04d} frames "
                    f"{batch_slice[0][0]:04d}..{batch_slice[-1][0]:04d} "
                    f"(batch size {len(batch_slice)})"
                )

                for k in b_obs:
                    b_obs[k] = torch.cat(b_obs[k], dim=0).to(device)

                complementary_data = {
                    "task":      b_task_str,
                    "subtask":   [""] * len(batch_slice),
                    "advantage": torch.ones((len(batch_slice), 1), device=device),
                }
                dummy_action   = torch.zeros(len(batch_slice), 1, 6, device=device)
                batch_for_proc = {
                    TransitionKey.ACTION:             dummy_action,
                    **b_obs,
                    TransitionKey.COMPLEMENTARY_DATA: complementary_data,
                }
                processed = preprocessor(batch_for_proc)

                images, img_masks = policy._preprocess_images(b_obs)
                task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
                task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

                subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
                    images, img_masks, task_tokens, task_masks
                )

                prefix_cache = embed_probe_prefix(
                    policy, images, img_masks, task_tokens, task_masks,
                    subtask_tokens, subtask_masks,
                )

                attn_by_layer, segments, pad_masks, patches_per_cam = probe_forward(
                    prefix_cache, p.timestep, device,
                )
                if not attn_by_layer:
                    logging.warning("      No attention captured — skipping.")
                    continue

                for layer_idx in attn_layers:
                    if layer_idx not in attn_by_layer:
                        logging.warning(f"      Layer {layer_idx} not captured — skipping.")
                        continue

                    attn_weights = attn_by_layer[layer_idx]
                    use_overlay = (layer_idx == 0)

                    ep_dir = os.path.join(ds_output_dir, f"ep{ep_idx:04d}_L{layer_idx:02d}")
                    os.makedirs(ep_dir, exist_ok=True)

                    csv_path = os.path.join(ep_dir, "norm_consts.csv")
                    csv_file = open(csv_path, "a", newline="")
                    csv_writer = csv.writer(csv_file)
                    if os.path.getsize(csv_path) == 0:
                        csv_writer.writerow(["ep", "fr", "layer", "panel", "vmax"])

                    for b_idx, (fr_idx, _) in enumerate(batch_slice):
                        a_w = attn_weights[b_idx : b_idx + 1]
                        p_m = pad_masks   [b_idx : b_idx + 1]
                        i_t = [img[b_idx : b_idx + 1] for img in images]
                        t_t = task_tokens   [b_idx : b_idx + 1]
                        s_t = subtask_tokens[b_idx : b_idx + 1]

                        heatmap_frames, norm_consts = render_image_overlays(
                            a_w, segments, i_t, p_m, patches_per_cam,
                            overlay=use_overlay,
                        )
                        frames_out = dict(heatmap_frames)

                        matrix_frames, matrix_norms = render_action_to_prefix_matrix(
                            a_w, segments, p_m,
                            t_t, s_t, policy.model._paligemma_tokenizer,
                        )
                        frames_out.update(matrix_frames)
                        norm_consts.update(matrix_norms)

                        for panel, vmax in norm_consts.items():
                            csv_writer.writerow([
                                ep_idx, fr_idx, layer_idx, panel, f"{vmax:.6e}",
                            ])

                        for k, frame_np in frames_out.items():
                            if k not in writers[layer_idx]:
                                path = os.path.join(ep_dir, f"{k}.mp4")
                                writers[layer_idx][k] = imageio.get_writer(
                                    path, fps=fps, macro_block_size=1
                                )
                            writers[layer_idx][k].append_data(frame_np)

                    csv_file.close()

            for dict_l in writers.values():
                for w in dict_l.values():
                    w.close()

    # ── Primary dataset ────────────────────────────────────────────────────────
    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    logging.info(f"Primary dataset: {cfg.dataset.root}")
    _probe_dataset(dataset, os.path.join(output_dir, primary_name))

    # ── Additional datasets ────────────────────────────────────────────────────
    extra_paths = getattr(cfg.dataset, "additional_offline_dataset_paths", []) or []
    for extra_root in extra_paths:
        logging.info(f"Additional dataset: {extra_root}")
        extra_ds = load_extra_dataset(cfg, extra_root)
        ds_name  = os.path.basename(os.path.normpath(extra_root))
        _probe_dataset(extra_ds, os.path.join(output_dir, ds_name))

    logging.debug(f"Done. Output saved to {output_dir}/")



if __name__ == "__main__":
    probe_cli()