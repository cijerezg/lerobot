#!/usr/bin/env python
"""
Attention probing script for PI05-Full policy.

Loads a dataset, runs a single joint forward pass (no denoising loop) per sample
at diffusion timestep t=1.0, captures attention matrices from layers 0, 9, and 17,
and produces video outputs per (episode, layer):

  Layer 0:
    - overlay_<cam>_mean.mp4: camera image overlaid with mean-head heatmaps
      for each query group (all / language / subtask / action).
    - overlay_<cam>_heads.mp4: per-head overlays on the camera image.
    - matrix_mean.mp4 / matrix_heads.mp4: full [seq x seq] attention matrix.

  Deeper layers (9, 17):
    - heatmap_<cam>_mean.mp4: standalone heatmaps (no camera overlay) for each
      query group, with bicubic interpolation.
    - heatmap_<cam>_heads.mp4: per-head standalone heatmaps.
    - matrix_mean.mp4 / matrix_heads.mp4: full [seq x seq] attention matrix.

Usage:
    python probe_attention_pi05.py config.json
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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_full.modeling_pi05 import (
    _PROBING_CAPTURE,
    make_att_2d_masks,
)
from lerobot.processor.core import TransitionKey
from lerobot.scripts.probe_offline_inference_pi05 import _build_episode_index, get_frame_data
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.rl.probe_utils_pi05 import (
    load_extra_dataset,
    load_policy_and_processors,
)


# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded settings
# ──────────────────────────────────────────────────────────────────────────────

ATTN_LAYERS = [0, 9, 17]
TIMESTEP = 0.5
MAX_EPISODES = 5
SUBSAMPLE = 2
BATCH_SIZE = 32
OUTPUT_DIR = os.path.join("outputs", "attention")


# ──────────────────────────────────────────────────────────────────────────────
# Sample selection
# ──────────────────────────────────────────────────────────────────────────────

def build_sample_list(dataset, random_n, subsample, seed=None):
    """
    Build per-episode sample lists.

    Returns [(ep_idx, [(fr_idx, global_idx), ...]), ...].
    Subsamples frames by a fixed stride and picks random episodes.
    """
    ep_to_indices = _build_episode_index(dataset)

    rng = random.Random(seed)
    all_eps = list(ep_to_indices.keys())
    rng.shuffle(all_eps)
    selected_eps = all_eps[:random_n]

    samples = []
    for ep_idx in selected_eps:
        indices = ep_to_indices[ep_idx]
        ep_frames = [
            (fr_idx, indices[fr_idx])
            for fr_idx in range(0, len(indices), subsample)
        ]
        if ep_frames:
            samples.append((ep_idx, ep_frames))

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Probing forward — no FAST tokens, no loss, all requested layers
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
        "segments_prefix":   segments_prefix,
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

    action_start = segments_prefix[-1][2]
    segments = segments_prefix + [("action", action_start, action_start + suffix_embs.shape[1])]

    return attn_by_layer, segments, combined_pad, patches_per_cam


# ──────────────────────────────────────────────────────────────────────────────
# Attention -> spatial heatmap helpers
# ──────────────────────────────────────────────────────────────────────────────

def _attn_to_heatmap(attn_heads, q_start, q_end, k_start, k_end,
                     pad_masks, n_p, img_h, img_w):
    """
    Average attention from query rows [q_start:q_end] to key cols [k_start:k_end],
    weighted by valid (non-padding) queries. Returns per-head heatmaps and the
    head-mean heatmap, both upsampled to (img_h, img_w) with bicubic interpolation.

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
    """Blend a heatmap onto a camera image."""
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
    """Render a standalone heatmap (no camera image overlay)."""
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
    frames_out  : dict  key -> np.ndarray  (video frames)
    norm_consts : dict  key -> float  (raw peak attention value used as vmax per panel)
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
# Full attention matrix figure
# ──────────────────────────────────────────────────────────────────────────────

_SEG_COLORS = {
    "img1":     (76,  114, 176),
    "img2":     (85,  168, 104),
    "language": (196, 78,  82),
    "subtask":  (129, 114, 178),
    "action":   (204, 185, 116),
}
_DEFAULT_COLOR = (119, 119, 119)


def cv2_draw_matrix(attn_mat, segments, title):
    mat_np   = attn_mat.numpy() if hasattr(attn_mat, "numpy") else attn_mat
    L        = mat_np.shape[0]
    out_size = 500

    mat_min, mat_max = mat_np.min(), mat_np.max()
    mat_norm  = (mat_np - mat_min) / (mat_max - mat_min + 1e-8)
    mat_gray  = (mat_norm * 255).astype(np.uint8)
    mat_color = cv2.applyColorMap(mat_gray, cv2.COLORMAP_VIRIDIS)
    mat_rgb   = cv2.cvtColor(mat_color, cv2.COLOR_BGR2RGB)
    mat_rgb   = cv2.resize(mat_rgb, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    scale = out_size / L if L > 0 else 1

    for name, s, e in segments:
        c = _SEG_COLORS.get(name, _DEFAULT_COLOR)
        for pos in [s, e]:
            p_scaled     = int(pos * scale)
            line_overlay = mat_rgb.copy()
            cv2.line(line_overlay, (0, p_scaled), (out_size, p_scaled), c, 1)
            cv2.line(line_overlay, (p_scaled, 0), (p_scaled, out_size), c, 1)
            mat_rgb = cv2.addWeighted(mat_rgb, 0.7, line_overlay, 0.3, 0)

    cv2.rectangle(mat_rgb, (0, 0), (out_size, 25), (0, 0, 0), cv2.FILLED)
    cv2.putText(mat_rgb, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return mat_rgb


def render_full_matrix(attn_weights, segments, pad_masks):
    frames_out = {}
    n_heads    = attn_weights.shape[1]
    attn       = torch.nan_to_num(attn_weights[0].float().cpu(), nan=0.0)
    pad        = pad_masks[0].cpu()

    pad_2d      = pad[:, None] & pad[None, :]
    attn_masked = attn * pad_2d[None].float()

    mean_mat = attn_masked.mean(0)
    frames_out["matrix_mean"] = cv2_draw_matrix(mean_mat, segments, "mean across heads")

    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols

    head_rows = []
    for r in range(h_rows):
        row_imgs = []
        for c in range(h_cols):
            idx = r * h_cols + c
            if idx < n_heads:
                row_imgs.append(cv2_draw_matrix(attn_masked[idx], segments, f"head {idx}"))
            else:
                row_imgs.append(np.zeros((500, 500, 3), dtype=np.uint8))
        head_rows.append(np.hstack(row_imgs))

    frames_out["matrix_heads"] = np.vstack(head_rows)
    return frames_out


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: TrainRLServerPipelineConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Probing layers: {ATTN_LAYERS}")
    logging.info(f"Timestep: {TIMESTEP}")
    logging.info(f"Output dir: {OUTPUT_DIR}")

    logging.info("Loading policy …")
    policy, preprocessor, _, dataset = load_policy_and_processors(cfg, device)
    policy.eval()

    chunk_size = cfg.policy.n_action_steps

    samples = build_sample_list(dataset, random_n=MAX_EPISODES,
                                subsample=SUBSAMPLE, seed=42)
    if not samples:
        logging.warning("No samples found, exiting.")
        return

    logging.info(f"{len(samples)} episodes × {len(ATTN_LAYERS)} layers → {OUTPUT_DIR}")

    fps      = getattr(dataset, "fps", 30) / SUBSAMPLE
    batch_sz = BATCH_SIZE

    for ep_idx, ep_frames in samples:
        # writers[layer_idx][video_key] = imageio writer
        writers = {l: {} for l in ATTN_LAYERS}

        for i in range(0, len(ep_frames), batch_sz):
            batch_slice = ep_frames[i : i + batch_sz]

            b_obs      = {}
            b_task_str = []
            for fr_idx, global_idx in batch_slice:
                obs, _, state, gt_subtask, task_str, _, _ = get_frame_data(
                    dataset, global_idx, chunk_size
                )
                b_task_str.append(task_str)
                for k, v in obs.items():
                    if k not in b_obs:
                        b_obs[k] = []
                    b_obs[k].append(v)

            logging.info(
                f"  ep={ep_idx:04d} frames "
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
                prefix_cache, TIMESTEP, device,
            )
            if not attn_by_layer:
                logging.warning("  No attention captured — skipping.")
                continue

            for layer_idx in ATTN_LAYERS:
                if layer_idx not in attn_by_layer:
                    logging.warning(f"  Layer {layer_idx} not captured — skipping.")
                    continue

                attn_weights = attn_by_layer[layer_idx]  # [B, heads, seq, seq]
                use_overlay = (layer_idx == 0)

                ep_dir = os.path.join(OUTPUT_DIR, f"ep{ep_idx:04d}_L{layer_idx:02d}")
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

                    heatmap_frames, norm_consts = render_image_overlays(
                        a_w, segments, i_t, p_m, patches_per_cam,
                        overlay=use_overlay,
                    )
                    frames_out = dict(heatmap_frames)
                    frames_out.update(render_full_matrix(a_w, segments, p_m))

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

    logging.debug(f"Done. Output saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    probe_cli()
