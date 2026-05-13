#!/usr/bin/env python
"""
Action-Drift Jacobian: Causal Attention Analysis for PI05-Full.

Instead of plotting raw softmax attention (which includes attention sinks),
this probe computes the **causal map**:

    causal_map = A * |dA/d(action)|

where A is the softmax attention and the gradient is obtained by
back-propagating torch.norm(action_pred) through the captured attention
weights.  A pixel lights up if and only if:
  1. The model actually looked at it (A > 0).
  2. It actively steered the predicted action (|J| > 0).

Outputs:
  Per (query_group, cam): 2×4 per-head grid  (MP4 frames or PNGs)
  Per cam: summary strip  (mean over heads, columns = query groups)

Usage:
    python action_drift_jacobian.py config.json \\
        --probe_parameters.timestep 0.5
"""

import csv
import logging
import os
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    load_policy_and_processors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reuse from attention_pi05
# ──────────────────────────────────────────────────────────────────────────────

from lerobot.probes.attention_pi05 import (
    embed_probe_prefix,
    build_sample_list,
    render_action_to_prefix_matrix,
)


# ──────────────────────────────────────────────────────────────────────────────
# Jacobian-aware forward pass
# ──────────────────────────────────────────────────────────────────────────────

def jacobian_probe_forward(prefix_cache, time_val, device, policy):
    """
    Forward pass that keeps attention in the autograd graph, then
    back-propagates ``torch.norm(action_pred)`` to obtain the causal map
    ``A * |dA/d(action)|``.

    Returns the same tuple shape as ``probe_forward`` so downstream
    rendering is interchangeable:
        (causal_map, segments, pad_masks, patches_per_cam)

    causal_map : Tensor [B, n_heads, seq, seq]  (cpu, float32)

    Memory: the computational graph is freed by a single ``loss.backward()``
    without ``retain_graph``.  Afterwards we zero grads and delete tensors.
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
    with torch.no_grad():
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

    # ── Enable grad-aware capture and run ──────────────────────────────────────
    _PROBING_CAPTURE["enabled"]       = True
    _PROBING_CAPTURE["attn_weights"]  = None
    _PROBING_CAPTURE["requires_grad"] = True

    with torch.set_grad_enabled(True):
        (prefix_out, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        attn_weights = _PROBING_CAPTURE["attn_weights"]  # [B, H, total, total] on GPU, in graph

        # ── Extract action prediction and compute scalar loss ──────────────
        action_hidden = suffix_out[:, -model.config.chunk_size:]
        action_hidden = action_hidden.to(dtype=model.action_out_proj.weight.dtype)
        action_pred = model.action_out_proj(action_hidden)  # [B, chunk, action_dim]

        loss = torch.norm(action_pred, p=2)
        loss.backward()  # frees graph (no retain_graph)

    # ── Extract causal map: A * |grad(A)| ──────────────────────────────────────
    if attn_weights.grad is not None:
        causal_map = (attn_weights.detach() * attn_weights.grad.abs()).float().cpu()
    else:
        logging.warning("[jacobian] attn_weights.grad is None — falling back to raw attention")
        causal_map = attn_weights.detach().float().cpu()

    # ── Build return values before cleanup ─────────────────────────────────────
    action_start = segments_prefix[-1][2]
    segments = segments_prefix + [("action", action_start, action_start + suffix_len)]

    combined_pad_out = torch.zeros(bsize, total_len, dtype=torch.bool, device="cpu")
    combined_pad_out[:, :prefix_len] = prefix_pad_masks.cpu()
    combined_pad_out[:, prefix_len:] = suffix_pad_masks.cpu()

    # ── Cleanup ────────────────────────────────────────────────────────────────
    _PROBING_CAPTURE["enabled"]       = False
    _PROBING_CAPTURE["requires_grad"] = False
    _PROBING_CAPTURE["attn_weights"]  = None

    policy.zero_grad(set_to_none=True)
    del attn_weights, suffix_out, prefix_out, action_pred, action_hidden, loss
    del suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond
    del att_2d_4d, combined, combined_pad, pad_2d, att_2d, position_ids
    torch.cuda.empty_cache()

    return causal_map, segments, combined_pad_out, patches_per_cam


def jacobian_probe_forward_multilayer(prefix_cache, time_val, device, policy, layers):
    """
    Compute per-layer causal maps by running one forward+backward per layer.

    This avoids keeping all layers' attention tensors in the graph simultaneously.

    Parameters
    ----------
    layers : list[int]  — layer indices to compute Jacobians for

    Returns
    -------
    causal_by_layer : dict[int, Tensor]  — layer_idx → [B, H, seq, seq] (cpu)
    segments        : list of (name, start, end)
    pad_masks       : Tensor [B, seq] (cpu)
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

    # ── Precompute shared inputs (lightweight, reused per layer) ───────────────
    with torch.no_grad():
        noise = model.sample_noise(
            (bsize, model.config.chunk_size, model.config.max_action_dim), device
        )
    time_tensor = torch.full((bsize,), time_val, dtype=torch.float32, device=device)

    if w_dtype == torch.bfloat16:
        noise       = noise.to(torch.bfloat16)
        time_tensor = time_tensor.to(torch.bfloat16)

    suffix_embs_base, suffix_pad_masks, suffix_att_masks, adarms_cond_base = model.embed_suffix(
        noise, time_tensor
    )
    if w_dtype == torch.bfloat16:
        suffix_embs_base = suffix_embs_base.to(torch.bfloat16)

    prefix_len = prefix_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]
    total_len  = prefix_len + suffix_len

    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    combined = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
    combined[:, :prefix_len, :prefix_len] = prefix_att_masks
    combined[:, prefix_len:, prefix_len:] = suffix_att_2d
    combined[:, prefix_len:, :prefix_len] = True

    combined_pad = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    pad_2d       = combined_pad[:, None, :] & combined_pad[:, :, None]
    att_2d       = combined & pad_2d
    position_ids = torch.cumsum(combined_pad, dim=1) - 1
    att_2d_4d    = model._prepare_attention_masks_4d(att_2d)

    # Build output pad/segments once
    combined_pad_out = combined_pad.cpu()
    action_start = segments_prefix[-1][2]
    segments = segments_prefix + [("action", action_start, action_start + suffix_len)]

    causal_by_layer = {}

    for layer_idx in layers:
        # Each iteration: fresh forward+backward for one layer.
        # Detach shared inputs so each iteration builds an independent graph —
        # otherwise the first backward() frees saved tensors and later
        # iterations hit "Trying to backward through the graph a second time".
        prefix_embs_iter = prefix_embs.detach()
        suffix_embs_iter = suffix_embs_base.detach()
        adarms_cond_iter = adarms_cond_base.detach() if adarms_cond_base is not None else None

        _PROBING_CAPTURE["enabled"]       = True
        _PROBING_CAPTURE["requires_grad"] = True
        _PROBING_CAPTURE["all_layers"]    = False
        _PROBING_CAPTURE["attn_weights"]  = None
        # Temporarily override the layer-0-only filter by using a target layer key
        _PROBING_CAPTURE["target_layer"]  = layer_idx

        with torch.set_grad_enabled(True):
            (prefix_out, suffix_out), _ = model.paligemma_with_expert.forward(
                attention_mask=att_2d_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_iter, suffix_embs_iter],
                use_cache=False,
                adarms_cond=[None, adarms_cond_iter],
            )

            attn_weights = _PROBING_CAPTURE["attn_weights"]

            if attn_weights is None:
                logging.warning(f"[jacobian] Layer {layer_idx}: no attention captured, skipping")
                _PROBING_CAPTURE["enabled"] = False
                policy.zero_grad(set_to_none=True)
                del prefix_out, suffix_out
                torch.cuda.empty_cache()
                continue

            action_hidden = suffix_out[:, -model.config.chunk_size:]
            action_hidden = action_hidden.to(dtype=model.action_out_proj.weight.dtype)
            action_pred = model.action_out_proj(action_hidden)

            loss = torch.norm(action_pred, p=2)
            loss.backward()

        if attn_weights.grad is not None:
            causal_by_layer[layer_idx] = (
                attn_weights.detach() * attn_weights.grad.abs()
            ).float().cpu()
        else:
            logging.warning(
                f"[jacobian] Layer {layer_idx}: grad is None, using raw attention"
            )
            causal_by_layer[layer_idx] = attn_weights.detach().float().cpu()

        _PROBING_CAPTURE["enabled"]       = False
        _PROBING_CAPTURE["requires_grad"] = False
        _PROBING_CAPTURE["attn_weights"]  = None
        _PROBING_CAPTURE.pop("target_layer", None)

        policy.zero_grad(set_to_none=True)
        del attn_weights, suffix_out, prefix_out, action_pred, action_hidden, loss
        torch.cuda.empty_cache()

    return causal_by_layer, segments, combined_pad_out, patches_per_cam


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers  (consistent disaggregated presentation)
# ──────────────────────────────────────────────────────────────────────────────

def _upsample_patch_map(patch_vals, n_p, img_h, img_w):
    """Reshape flat patch vector to 2D grid and bicubic-upsample."""
    grid = patch_vals.reshape(n_p, n_p)
    grid_4d = grid[None, None].float()
    up = F.interpolate(grid_4d, size=(img_h, img_w), mode="bicubic", align_corners=False)
    return up.squeeze()


def _cv2_overlay(img_np, heatmap, title, vmax=None, alpha=0.5):
    """Blend a heatmap onto a camera image."""
    if vmax is None or vmax == 0:
        vmax = heatmap.max().item() or 1.0
    normed = (heatmap / vmax).clamp(0, 1)
    gray = (normed * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    h_color_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)

    if h_color_rgb.shape != img_np.shape:
        h_color_rgb = cv2.resize(h_color_rgb, (img_np.shape[1], img_np.shape[0]))

    overlay = cv2.addWeighted(img_np, 1 - alpha, h_color_rgb, alpha, 0)
    cv2.putText(overlay, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _extract_patch_heatmap(attn, q_s, q_e, cam_s, cam_e, pad, n_p, img_h, img_w):
    """
    Extract per-head and mean heatmaps for a (query_group → cam) slice.

    Returns (per_head_maps [H, img_h, img_w], mean_map [img_h, img_w]).
    """
    n_heads = attn.shape[0]
    q_valid = pad[q_s:q_e]
    if q_valid.sum() == 0:
        return None, None

    q_attn = attn[:, q_s:q_e, cam_s:cam_e]           # [H, Q, K]
    q_attn_valid = q_attn[:, q_valid, :]               # [H, valid_Q, K]
    per_head_flat = q_attn_valid.mean(dim=1)           # [H, K]

    per_head_maps = torch.stack([
        _upsample_patch_map(per_head_flat[h], n_p, img_h, img_w)
        for h in range(n_heads)
    ])
    mean_map = per_head_flat.mean(dim=0)
    mean_map = _upsample_patch_map(mean_map, n_p, img_h, img_w)
    return per_head_maps, mean_map


def render_disaggregated(signal_weights, segments, image_tensors, pad_masks,
                         patches_per_cam, label_prefix="causal"):
    """
    Render per-(query_group, cam) head grids and a per-cam summary strip.

    Parameters
    ----------
    signal_weights : Tensor [1, H, seq, seq]  (single sample)
    segments       : list of (name, start, end)
    image_tensors  : list of [1, C, H, W] per camera
    pad_masks      : Tensor [1, seq]
    patches_per_cam: int
    label_prefix   : str — prepended to output keys

    Returns
    -------
    frames_out  : dict  key → np.ndarray
    norm_consts : dict  key → float
    """
    frames_out  = {}
    norm_consts = {}
    n_heads     = signal_weights.shape[1]
    n_p         = int(patches_per_cam ** 0.5)

    seg_dict    = {name: (s, e) for name, s, e in segments}
    camera_segs = [(name, s, e) for name, s, e in segments if name.startswith("img")]
    query_groups = [
        ("all",      0,                           segments[-1][2]),
        ("language", *seg_dict["language"]),
        ("subtask",  *seg_dict["subtask"]),
        ("action",   *seg_dict["action"]),
    ]

    attn = torch.nan_to_num(signal_weights[0].float().cpu(), nan=0.0)
    pad  = pad_masks[0].cpu()

    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols  # 2 rows for 8 heads

    for cam_idx, (cam_name, cam_s, cam_e) in enumerate(camera_segs):
        # Get camera image for overlay
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
        summary_panels = [img_np.copy()]
        cv2.putText(summary_panels[0], f"{cam_name} (orig)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for qname, q_s, q_e in query_groups:
            _, mean_map = _extract_patch_heatmap(
                attn, q_s, q_e, cam_s, cam_e, pad, n_p, img_h, img_w
            )
            if mean_map is None:
                summary_panels.append(np.zeros_like(img_np))
                continue
            vmax = mean_map.max().item()
            norm_consts[f"{cam_name}_{qname}_mean"] = vmax
            summary_panels.append(
                _cv2_overlay(img_np, mean_map, f"{label_prefix} mean: {qname}", vmax=vmax)
            )
        frames_out[f"{label_prefix}_{cam_name}_summary"] = np.hstack(summary_panels)

        # ── Per (query_group, cam) head grids ──────────────────────────────
        for qname, q_s, q_e in query_groups:
            per_head_maps, _ = _extract_patch_heatmap(
                attn, q_s, q_e, cam_s, cam_e, pad, n_p, img_h, img_w
            )
            if per_head_maps is None:
                continue
            heads_vmax = per_head_maps.max().item()
            norm_consts[f"{cam_name}_{qname}_heads"] = heads_vmax

            head_rows = []
            for r in range(h_rows):
                row_imgs = []
                for c in range(h_cols):
                    idx = r * h_cols + c
                    if idx < n_heads:
                        row_imgs.append(
                            _cv2_overlay(img_np, per_head_maps[idx],
                                         f"{qname} h{idx}", vmax=heads_vmax)
                        )
                    else:
                        row_imgs.append(np.zeros_like(img_np))
                head_rows.append(np.hstack(row_imgs))
            frames_out[f"{label_prefix}_{cam_name}_{qname}_heads"] = np.vstack(head_rows)

    return frames_out, norm_consts


# ──────────────────────────────────────────────────────────────────────────────
# Callable entry point (for offline_val_pi05 integration)
# ──────────────────────────────────────────────────────────────────────────────

def run_action_drift_jacobian(
    policy, preprocessor,
    val_dataset, val_ep_indices,
    cfg, output_dir, device,
):
    """
    Run per-frame Jacobian causal maps and write MP4 videos.

    Called from ``_run_probe_action_drift_jacobian`` in offline_val_pi05.py
    or from the standalone CLI below.
    """
    from lerobot.probes.utils_pi05 import makedirs

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    t_val = p.timestep
    # Jacobian needs forward+backward per layer — cap batch to avoid OOM
    batch_sz = min(p.validation_batch_size, 4)
    layers = [int(x.strip()) for x in p.spatial_layers.split(",")]

    # Build sample list (reuse attn probe's logic)
    samples = _build_sample_list_from_val(
        val_dataset, val_ep_indices,
        random_n=p.max_episodes,
        subsample=p.attn_eval_subsample,
        seed=p.random_seed,
    )
    if not samples:
        logging.warning("[jacobian] no samples found")
        return None

    logging.debug(
        f"[jacobian] {len(samples)} episodes × layers {layers} @ t={t_val}, "
        f"batch_size={batch_sz} ..."
    )

    fps = getattr(val_dataset, "fps", 30) / p.attn_eval_subsample

    for ep_idx, ep_frames in samples:
        # writers[layer_idx][key] = imageio writer
        writers = {l: {} for l in layers}

        for batch_start in range(0, len(ep_frames), batch_sz):
            batch_slice = ep_frames[batch_start : batch_start + batch_sz]

            # ── Gather batch observations ──────────────────────────────────
            b_obs = {}
            b_task_str = []
            for fr_idx, global_idx in batch_slice:
                obs, _, _state, _gt_subtask, task_str, _, _ = get_frame_data(
                    val_dataset, global_idx, chunk_size
                )
                b_task_str.append(task_str)
                for k, v in obs.items():
                    b_obs.setdefault(k, []).append(v)

            logging.debug(
                f"    [jacobian] ep={ep_idx:04d} "
                f"frames {batch_slice[0][0]:04d}..{batch_slice[-1][0]:04d} "
                f"(batch {len(batch_slice)})"
            )

            b_obs_batched = {k: torch.cat(v, dim=0).to(device) for k, v in b_obs.items()}

            # ── Preprocessor ────────────────────────────────────────────────
            complementary_data = {
                "task":      b_task_str,
                "subtask":   [""] * len(batch_slice),
                "advantage": torch.ones((len(batch_slice), 1), device=device),
            }
            dummy_action = torch.zeros(len(batch_slice), 1, 6, device=device)
            batch_for_proc = {
                TransitionKey.ACTION: dummy_action,
                **b_obs_batched,
                TransitionKey.COMPLEMENTARY_DATA: complementary_data,
            }
            processed = preprocessor(batch_for_proc)

            # ── Embed prefix (once per batch) ────────────────────────────────
            images, img_masks = policy._preprocess_images(b_obs_batched)
            task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
            task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

            subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
                images, img_masks, task_tokens, task_masks
            )

            prefix_cache = embed_probe_prefix(
                policy, images, img_masks,
                task_tokens, task_masks,
                subtask_tokens, subtask_masks,
            )

            # ── Multi-layer Jacobian forward ─────────────────────────────────
            causal_by_layer, segments, pad_masks, patches_per_cam = (
                jacobian_probe_forward_multilayer(
                    prefix_cache, t_val, device, policy, layers
                )
            )
            if not causal_by_layer:
                logging.warning(f"    [jacobian] No causal maps at t={t_val}; skipping.")
                continue

            t_str = f"{t_val:.2f}".replace(".", "p")

            for layer_idx in layers:
                if layer_idx not in causal_by_layer:
                    continue
                causal_map = causal_by_layer[layer_idx]

                layer_dir = os.path.join(
                    output_dir, f"ep{ep_idx:04d}_t{t_str}", f"L{layer_idx:02d}"
                )
                os.makedirs(layer_dir, exist_ok=True)

                csv_path = os.path.join(layer_dir, "norm_consts.csv")
                csv_file  = open(csv_path, "a", newline="")
                csv_writer = csv.writer(csv_file)
                if os.path.getsize(csv_path) == 0:
                    csv_writer.writerow(["ep", "fr", "layer", "t_val", "panel", "vmax"])

                for b_idx, (fr_idx, _) in enumerate(batch_slice):
                    c_m = causal_map[b_idx : b_idx + 1]
                    p_m = pad_masks[b_idx : b_idx + 1]
                    i_t = [img[b_idx : b_idx + 1] for img in images]
                    t_t = task_tokens   [b_idx : b_idx + 1]
                    s_t = subtask_tokens[b_idx : b_idx + 1]

                    frames_out, norm_consts = render_disaggregated(
                        c_m, segments, i_t, p_m, patches_per_cam,
                        label_prefix=f"causal_L{layer_idx}",
                    )

                    matrix_frames, matrix_norms = render_action_to_prefix_matrix(
                        c_m, segments, p_m,
                        t_t, s_t, policy.model._paligemma_tokenizer,
                        label_prefix=f"causal_L{layer_idx}",
                    )
                    frames_out.update(matrix_frames)
                    norm_consts.update(matrix_norms)

                    for panel, vmax in norm_consts.items():
                        csv_writer.writerow(
                            [ep_idx, fr_idx, layer_idx, t_val, panel, f"{vmax:.6e}"]
                        )

                    for key, frame_np in frames_out.items():
                        if key not in writers[layer_idx]:
                            mp4_path = os.path.join(layer_dir, f"{key}.mp4")
                            writers[layer_idx][key] = imageio.get_writer(
                                mp4_path, fps=fps, macro_block_size=1
                            )
                        writers[layer_idx][key].append_data(frame_np)

                csv_file.close()

            del causal_by_layer, segments, pad_masks, patches_per_cam
            del prefix_cache, b_obs_batched, task_tokens, task_masks
            del subtask_tokens, subtask_masks, images, img_masks
            torch.cuda.empty_cache()

        for layer_writers in writers.values():
            for w in layer_writers.values():
                w.close()

    logging.info(f"[jacobian] Done. Output in {output_dir}/")
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Sample list builder (val-episode aware, mirrors _build_attn_sample_list)
# ──────────────────────────────────────────────────────────────────────────────

def _build_sample_list_from_val(val_dataset, val_ep_indices, random_n, subsample, seed):
    """
    Build per-episode sample lists respecting val_ep_indices.
    Returns [(ep_idx, [(fr_idx, global_idx), ...]), ...].
    """
    ep_to_indices = _build_episode_index(val_dataset)
    if val_ep_indices is not None:
        ep_to_indices = {k: v for k, v in ep_to_indices.items() if k in val_ep_indices}

    all_eps = sorted(ep_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_eps)
    if random_n:
        all_eps = all_eps[:random_n]

    samples = []
    for ep_idx in all_eps:
        indices = ep_to_indices[ep_idx]
        ep_frames = [
            (fr_idx, indices[fr_idx])
            for fr_idx in range(0, len(indices), subsample)
        ]
        if ep_frames:
            samples.append((ep_idx, ep_frames))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: TrainRLServerPipelineConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)

    p = cfg.probe_parameters
    output_dir = os.path.join(p.output_dir, "action_drift_jacobian")
    os.makedirs(output_dir, exist_ok=True)

    layers = [int(x.strip()) for x in p.spatial_layers.split(",")]

    logging.debug("Action-Drift Jacobian probe")
    logging.debug(f"  Timestep: {p.timestep}")
    logging.debug(f"  Layers:   {layers}")
    logging.debug(f"  Output:   {output_dir}")

    policy, preprocessor, _, dataset = load_policy_and_processors(cfg, device)
    policy.eval()

    samples_raw = build_sample_list(
        dataset,
        episodes_str=p.attn_eval_episodes,
        random_n=p.max_episodes,
        subsample=p.attn_eval_subsample,
        seed=p.random_seed,
    )
    if not samples_raw:
        logging.warning("No samples found, exiting.")
        return

    t_val = p.timestep
    # Jacobian needs forward+backward per layer — cap batch to avoid OOM
    batch_sz = min(p.validation_batch_size, 4)
    chunk_size = cfg.policy.n_action_steps
    fps = getattr(dataset, "fps", 30) / p.attn_eval_subsample

    for ep_idx, ep_frames in samples_raw:
        writers = {l: {} for l in layers}

        for batch_start in range(0, len(ep_frames), batch_sz):
            batch_slice = ep_frames[batch_start : batch_start + batch_sz]

            b_obs = {}
            b_task_str = []
            for fr_idx, global_idx in batch_slice:
                obs, _, _state, _gt_subtask, task_str, _, _ = get_frame_data(
                    dataset, global_idx, chunk_size
                )
                b_task_str.append(task_str)
                for k, v in obs.items():
                    b_obs.setdefault(k, []).append(v)

            logging.debug(
                f"  ep={ep_idx:04d} "
                f"frames {batch_slice[0][0]:04d}..{batch_slice[-1][0]:04d}"
            )

            b_obs_batched = {k: torch.cat(v, dim=0).to(device) for k, v in b_obs.items()}

            complementary_data = {
                "task":      b_task_str,
                "subtask":   [""] * len(batch_slice),
                "advantage": torch.ones((len(batch_slice), 1), device=device),
            }
            dummy_action = torch.zeros(len(batch_slice), 1, 6, device=device)
            batch_for_proc = {
                TransitionKey.ACTION: dummy_action,
                **b_obs_batched,
                TransitionKey.COMPLEMENTARY_DATA: complementary_data,
            }
            processed = preprocessor(batch_for_proc)

            images, img_masks = policy._preprocess_images(b_obs_batched)
            task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
            task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

            subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
                images, img_masks, task_tokens, task_masks
            )

            prefix_cache = embed_probe_prefix(
                policy, images, img_masks,
                task_tokens, task_masks,
                subtask_tokens, subtask_masks,
            )

            causal_by_layer, segments, pad_masks, patches_per_cam = (
                jacobian_probe_forward_multilayer(
                    prefix_cache, t_val, device, policy, layers
                )
            )
            if not causal_by_layer:
                continue

            t_str = f"{t_val:.2f}".replace(".", "p")

            for layer_idx in layers:
                if layer_idx not in causal_by_layer:
                    continue
                causal_map = causal_by_layer[layer_idx]

                layer_dir = os.path.join(
                    output_dir, f"ep{ep_idx:04d}_t{t_str}", f"L{layer_idx:02d}"
                )
                os.makedirs(layer_dir, exist_ok=True)

                for b_idx, (fr_idx, _) in enumerate(batch_slice):
                    c_m = causal_map[b_idx : b_idx + 1]
                    p_m = pad_masks[b_idx : b_idx + 1]
                    i_t = [img[b_idx : b_idx + 1] for img in images]
                    t_t = task_tokens   [b_idx : b_idx + 1]
                    s_t = subtask_tokens[b_idx : b_idx + 1]

                    frames_out, _ = render_disaggregated(
                        c_m, segments, i_t, p_m, patches_per_cam,
                        label_prefix=f"causal_L{layer_idx}",
                    )

                    matrix_frames, _ = render_action_to_prefix_matrix(
                        c_m, segments, p_m,
                        t_t, s_t, policy.model._paligemma_tokenizer,
                        label_prefix=f"causal_L{layer_idx}",
                    )
                    frames_out.update(matrix_frames)

                    for key, frame_np in frames_out.items():
                        if key not in writers[layer_idx]:
                            mp4_path = os.path.join(layer_dir, f"{key}.mp4")
                            writers[layer_idx][key] = imageio.get_writer(
                                mp4_path, fps=fps, macro_block_size=1
                            )
                        writers[layer_idx][key].append_data(frame_np)

            del causal_by_layer, segments, pad_masks, patches_per_cam
            del prefix_cache, b_obs_batched, task_tokens, task_masks
            del subtask_tokens, subtask_masks, images, img_masks
            torch.cuda.empty_cache()

        for layer_writers in writers.values():
            for w in layer_writers.values():
                w.close()

    logging.info(f"Done. Output in {output_dir}/")


if __name__ == "__main__":
    probe_cli()
