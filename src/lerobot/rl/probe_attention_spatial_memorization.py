#!/usr/bin/env python
"""
Spatial memorization probe for PI05-Full attention maps.

Tests whether attention heads memorize fixed spatial patterns by collecting
attention maps from N frames sampled across unique episodes, then computing:

  1. Mean map:          Mean_over_frames(attn[patch]) for each patch.

  2. Mean / std map:    Mean / (Std + eps) for each patch.
     High ratio = high mean + low variance = memorized spatial position.
     Low ratio  = noisy or rarely attended.

Output: PNG images per (layer, query_group, key_cam) for both mean-head
and per-head breakdowns, plus raw tensors saved as a .pt file.

Usage:
    python probe_attention_spatial_memorization.py config.json
"""

import logging
import os
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
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
from lerobot.types import TransitionKey
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
N_FRAMES = 32          # total frames to sample (1 per unique episode)
BATCH_SIZE = 8         # forward-pass batch size (memory)
OUTPUT_DIR = os.path.join("outputs", "attention_spatial")
STD_EPS = 1e-8         # floor for std in mean/std ratio


# ──────────────────────────────────────────────────────────────────────────────
# Sampling: 1 random frame per unique episode
# ──────────────────────────────────────────────────────────────────────────────

def sample_one_per_episode(dataset, n_frames, seed=42):
    """
    Sample 1 random frame from each of *n_frames* unique episodes.

    Returns [(ep_idx, fr_idx_in_ep, global_idx), ...].
    """
    ep_to_indices = _build_episode_index(dataset)
    all_eps = sorted(ep_to_indices.keys())

    rng = random.Random(seed)
    rng.shuffle(all_eps)
    selected = all_eps[:n_frames]

    samples = []
    for ep_idx in selected:
        indices = ep_to_indices[ep_idx]
        global_idx = rng.choice(indices)
        fr_idx = dataset.hf_dataset[global_idx]["frame_index"].item()
        samples.append((ep_idx, fr_idx, global_idx))

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass (reused from multi-layer probe)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_probe_prefix(policy, images, img_masks, task_tokens, task_masks,
                       subtask_tokens, subtask_masks):
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

    n_cameras = len(images)
    patches_per_cam = image_len // n_cameras

    segments_prefix = []
    pos = 0
    for i in range(n_cameras):
        segments_prefix.append((f"img{i + 1}", pos, pos + patches_per_cam))
        pos += patches_per_cam
    segments_prefix.append(("language", pos, pos + task_tokens.shape[1]))
    pos += task_tokens.shape[1]
    segments_prefix.append(("subtask", pos, pos + subtask_tokens.shape[1]))

    return {
        "model":            model,
        "prefix_embs":      prefix_embs,
        "prefix_pad_masks": prefix_pad_masks,
        "prefix_att_masks": prefix_att_masks,
        "w_dtype":          w_dtype,
        "bsize":            task_tokens.shape[0],
        "patches_per_cam":  patches_per_cam,
        "segments_prefix":  segments_prefix,
    }


@torch.no_grad()
def probe_forward(prefix_cache, time_val, device):
    model            = prefix_cache["model"]
    prefix_embs      = prefix_cache["prefix_embs"]
    prefix_pad_masks = prefix_cache["prefix_pad_masks"]
    prefix_att_masks = prefix_cache["prefix_att_masks"]
    w_dtype          = prefix_cache["w_dtype"]
    bsize            = prefix_cache["bsize"]
    patches_per_cam  = prefix_cache["patches_per_cam"]
    segments_prefix  = prefix_cache["segments_prefix"]

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
# Extract per-head attention vectors for a (query_group → key_cam) slice
# ──────────────────────────────────────────────────────────────────────────────

def extract_qk_attn(attn_heads, q_start, q_end, k_start, k_end, pad_mask):
    """
    Extract mean attention from query rows to key cols, per head.

    attn_heads : [n_heads, seq, seq]
    pad_mask   : [seq] bool
    Returns    : [n_heads, n_key_patches]  or None if no valid queries
    """
    q_valid = pad_mask[q_start:q_end]
    if q_valid.sum() == 0:
        return None

    q_attn = attn_heads[:, q_start:q_end, k_start:k_end]  # [H, Q, K]
    q_attn_valid = q_attn[:, q_valid, :]                   # [H, valid_Q, K]
    return q_attn_valid.mean(dim=1)                        # [H, K]


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation: log-sum, mean, variance across frames
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_maps(all_maps):
    """
    Given a list of [n_heads, n_patches] tensors (one per frame),
    compute mean and mean/std maps.

    Returns dict with keys: mean, mean_over_std — each [n_heads, n_patches].
    """
    stacked = torch.stack(all_maps, dim=0)  # [N, H, K]

    mean_map = stacked.mean(dim=0)          # [H, K]
    std_map  = stacked.std(dim=0)           # [H, K]
    mean_over_std = mean_map / (std_map + STD_EPS)

    return {
        "mean":          mean_map,
        "mean_over_std": mean_over_std,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _upsample_patch_map(patch_vals, n_p, img_h, img_w):
    """Reshape flat patch vector to 2D grid and bicubic-upsample."""
    grid = patch_vals.reshape(n_p, n_p)
    grid_4d = grid[None, None].float()
    up = F.interpolate(grid_4d, size=(img_h, img_w), mode="bicubic",
                       align_corners=False)
    return up.squeeze()


def render_heatmap(values, title, img_h, img_w, colormap=cv2.COLORMAP_JET,
                   vmin=None, vmax=None):
    """Render a single heatmap as an RGB numpy array."""
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()

    normed = (values - vmin) / (vmax - vmin + 1e-8)
    normed = normed.clamp(0, 1)
    gray = (normed * 255).numpy().astype(np.uint8)
    colored = cv2.applyColorMap(gray, colormap)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if rgb.shape[:2] != (img_h, img_w):
        rgb = cv2.resize(rgb, (img_w, img_h))

    cv2.putText(rgb, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def render_stat_panel(stat_map, n_heads, n_p, img_h, img_w, stat_name,
                      query_name, cam_name, colormap=cv2.COLORMAP_JET):
    """
    Render a stat map [n_heads, n_patches] as a per-head grid.
    Each head is normalized independently for maximum contrast.

    Returns a single composited numpy image.
    """
    # Upsample all heads
    head_ups = []
    for h in range(n_heads):
        up = _upsample_patch_map(stat_map[h], n_p, img_h, img_w)
        head_ups.append(up)

    # Per-head grid with per-head normalization
    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols

    head_rows = []
    for r in range(h_rows):
        row_imgs = []
        for c in range(h_cols):
            idx = r * h_cols + c
            if idx < n_heads:
                hmap = head_ups[idx]
                title = f"{stat_name} | {query_name}->{cam_name} | h{idx}"
                row_imgs.append(render_heatmap(hmap, title, img_h, img_w,
                                               colormap=colormap))
            else:
                row_imgs.append(np.zeros((img_h, img_w, 3), dtype=np.uint8))
        head_rows.append(np.hstack(row_imgs))

    return np.vstack(head_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# .pt loading: skip forward passes if raw data already exists
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_results(pt_path):
    """
    Load a previously saved .pt file and reconstruct the raw_results dict.

    Returns (raw_results, n_heads, n_p, img_h, img_w) or None if file missing.
    """
    if not os.path.isfile(pt_path):
        return None

    logging.info(f"Found existing {pt_path}, loading instead of recomputing ...")
    save_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    img_hw = save_dict["img_hw"]
    img_h, img_w = img_hw[0].item(), img_hw[1].item()
    n_p = save_dict["n_p"].item()
    n_frames = save_dict["n_frames"].item()
    layers = save_dict["layers"].tolist()

    logging.info(f"  {n_frames} frames, layers {layers}, img {img_h}x{img_w}, n_p={n_p}")

    # Reconstruct raw_results[(layer, q_name, cam_name)] = {stat: tensor}
    # Keys look like: L0_action_img1_mean
    stat_names = ["mean", "mean_over_std"]
    raw_results = {}
    n_heads = None

    for key, tensor in save_dict.items():
        if key in ("n_frames", "layers", "img_hw", "n_p"):
            continue
        # Parse key: L{layer}_{q_name}_{cam_name}_{stat_name}
        for sn in stat_names:
            if key.endswith(f"_{sn}"):
                remainder = key[: -len(f"_{sn}")]  # e.g. "L0_action_img1"
                # Split off layer
                parts = remainder.split("_", 1)     # ["L0", "action_img1"]
                layer_idx = int(parts[0][1:])
                # Split off cam (last token)
                rest = parts[1]
                last_underscore = rest.rfind("_")
                q_name = rest[:last_underscore]
                cam_name = rest[last_underscore + 1:]

                rkey = (layer_idx, q_name, cam_name)
                if rkey not in raw_results:
                    raw_results[rkey] = {}
                raw_results[rkey][sn] = tensor

                if n_heads is None:
                    n_heads = tensor.shape[0]
                break

    return raw_results, n_heads, n_p, img_h, img_w


# ──────────────────────────────────────────────────────────────────────────────
# Render all results to PNGs
# ──────────────────────────────────────────────────────────────────────────────

def render_all(raw_results, n_heads, n_p, img_h, img_w):
    """Render all stat maps to PNG images."""
    for (layer_idx, q_name, cam_name), stats in raw_results.items():
        layer_dir = os.path.join(OUTPUT_DIR, f"L{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)

        for stat_name in ["mean", "mean_over_std"]:
            stat_map = stats[stat_name]  # [H, K]
            cmap = cv2.COLORMAP_VIRIDIS

            panel = render_stat_panel(
                stat_map, n_heads, n_p, img_h, img_w,
                stat_name, q_name, cam_name, colormap=cmap,
            )

            fname = f"{stat_name}_{q_name}_to_{cam_name}.png"
            path = os.path.join(layer_dir, fname)
            cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.info(f"Spatial memorization probe")
    logging.info(f"  Layers: {ATTN_LAYERS}")
    logging.info(f"  N frames: {N_FRAMES} (1 per episode)")
    logging.info(f"  Output: {OUTPUT_DIR}")

    # ── Try loading from cached .pt ───────────────────────────────────────────
    pt_path = os.path.join(OUTPUT_DIR, "spatial_memorization_raw.pt")
    cached = load_raw_results(pt_path)

    if cached is not None:
        raw_results, n_heads, n_p, img_h, img_w = cached
        logging.info("Re-rendering from cached data (delete .pt to recompute)")
        render_all(raw_results, n_heads, n_p, img_h, img_w)
        logging.info(f"Done. Output in {OUTPUT_DIR}/")
        return

    # ── Full computation path ─────────────────────────────────────────────────
    logging.info("Loading policy ...")
    policy, preprocessor, _, dataset = load_policy_and_processors(cfg, device)
    policy.eval()

    chunk_size = cfg.policy.n_action_steps

    samples = sample_one_per_episode(dataset, n_frames=N_FRAMES, seed=42)
    if not samples:
        logging.warning("No samples found, exiting.")
        return

    logging.info(f"Sampled {len(samples)} frames from {len(samples)} episodes")

    # collected[layer][(q_name, cam_name)] = list of [H, K] tensors
    collected = {l: {} for l in ATTN_LAYERS}
    img_h, img_w = None, None
    n_heads_global = None
    n_p_global = None

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[batch_start : batch_start + BATCH_SIZE]
        bs = len(batch_samples)

        b_obs = {}
        b_task_str = []
        for ep_idx, fr_idx, global_idx in batch_samples:
            obs, _, state, gt_subtask, task_str, _, _ = get_frame_data(
                dataset, global_idx, chunk_size
            )
            b_task_str.append(task_str)
            for k, v in obs.items():
                if k not in b_obs:
                    b_obs[k] = []
                b_obs[k].append(v)

        logging.debug(
            f"  Batch {batch_start // BATCH_SIZE + 1}: "
            f"episodes {[s[0] for s in batch_samples]}"
        )

        for k in b_obs:
            b_obs[k] = torch.cat(b_obs[k], dim=0).to(device)

        complementary_data = {
            "task":      b_task_str,
            "subtask":   [""] * bs,
            "advantage": torch.ones((bs, 1), device=device),
        }
        dummy_action = torch.zeros(bs, 1, 6, device=device)
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
            logging.warning("  No attention captured — skipping batch.")
            continue

        if img_h is None:
            img_t = images[0][0].cpu()
            img_h = img_t.shape[1]
            img_w = img_t.shape[2]
            n_p_global = int(patches_per_cam ** 0.5)

        seg_dict = {name: (s, e) for name, s, e in segments}
        camera_segs = [(name, s, e) for name, s, e in segments if name.startswith("img")]
        query_groups = [
            ("all",      0,                  segments[-1][2]),
            ("language", *seg_dict["language"]),
            ("subtask",  *seg_dict["subtask"]),
            ("action",   *seg_dict["action"]),
        ]

        for layer_idx in ATTN_LAYERS:
            if layer_idx not in attn_by_layer:
                continue

            attn_weights = attn_by_layer[layer_idx]  # [B, H, seq, seq]
            if n_heads_global is None:
                n_heads_global = attn_weights.shape[1]

            for b_idx in range(bs):
                attn = torch.nan_to_num(attn_weights[b_idx].float().cpu(), nan=0.0)
                pad  = pad_masks[b_idx].cpu()

                for q_name, q_s, q_e in query_groups:
                    for cam_name, cam_s, cam_e in camera_segs:
                        key = (q_name, cam_name)
                        vec = extract_qk_attn(attn, q_s, q_e, cam_s, cam_e, pad)
                        if vec is None:
                            continue
                        if key not in collected[layer_idx]:
                            collected[layer_idx][key] = []
                        collected[layer_idx][key].append(vec)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    logging.info("Aggregating statistics ...")

    raw_results = {}
    for layer_idx in ATTN_LAYERS:
        for (q_name, cam_name), maps_list in collected[layer_idx].items():
            n_frames_actual = len(maps_list)
            if n_frames_actual < 2:
                logging.warning(
                    f"  L{layer_idx} {q_name}->{cam_name}: only {n_frames_actual} frames, skipping"
                )
                continue
            raw_results[(layer_idx, q_name, cam_name)] = aggregate_maps(maps_list)

    # ── Save .pt ──────────────────────────────────────────────────────────────
    save_dict = {}
    for (layer_idx, q_name, cam_name), stats in raw_results.items():
        prefix = f"L{layer_idx}_{q_name}_{cam_name}"
        for stat_name, tensor in stats.items():
            save_dict[f"{prefix}_{stat_name}"] = tensor
    save_dict["n_frames"] = torch.tensor(len(samples))
    save_dict["layers"] = torch.tensor(ATTN_LAYERS)
    save_dict["img_hw"] = torch.tensor([img_h, img_w])
    save_dict["n_p"] = torch.tensor(n_p_global)
    torch.save(save_dict, pt_path)
    logging.debug(f"Raw tensors saved to {pt_path}")

    # ── Render ────────────────────────────────────────────────────────────────
    render_all(raw_results, n_heads_global, n_p_global, img_h, img_w)
    logging.info(f"Done. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
