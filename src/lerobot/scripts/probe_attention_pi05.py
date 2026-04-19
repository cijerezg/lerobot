#!/usr/bin/env python
"""
Attention probing script for PI05-Full policy.

Loads a dataset, runs a single joint forward pass (no denoising loop) per sample
at one or more diffusion timesteps, captures the layer-0 attention matrix, and
produces two figures per (sample, timestep):

  1. overlay_<ep>_<fr>_t<T>.png
       For each camera: mean-head overlays (all / language / subtask / action queries)
       + per-head breakdown for the "all queries" view.

  2. matrix_<ep>_<fr>_t<T>.png
       Full [seq × seq] attention matrix (mean across heads) with labeled sections
       and per-head grid.

Usage
-----
python probe_attention_pi05.py config.json \\
    --eval_random_n 5 \\
    --probe_output_dir outputs/probe/ \\
    --probe_timesteps "1.0,0.5,0.25"
"""

import csv
import logging
import os
import random

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from dataclasses import dataclass
from typing import Optional

import imageio

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import lerobot.rl.rl_pi05  # noqa: F401
from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_full.modeling_pi05 import (
    _PROBING_CAPTURE,
    make_att_2d_masks,
)
from lerobot.processor.core import TransitionKey
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging

# Re-use dataset/frame helpers from the eval script
from lerobot.scripts.eval_offline_pi05 import (
    _load_policy_and_processors,
    _build_episode_index,
    get_frame_data,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeAttentionConfig(TrainRLServerPipelineConfig):
    eval_episodes: Optional[str] = None
    eval_subsample: int = 2
    eval_random_n: int = 0
    eval_random_seed: Optional[int] = None
    probe_output_dir: str = "outputs/probe_attention"
    # Comma-separated diffusion timesteps to probe, e.g. "1.0,0.5,0.25"
    probe_timesteps: str = "1.0,0.25"
    probe_batch_size: int = 32


# ──────────────────────────────────────────────────────────────────────────────
# Sample selection (mirrors eval_offline_pi05)
# ──────────────────────────────────────────────────────────────────────────────

def build_sample_list(dataset, episodes_str, random_n, subsample, seed=None):
    ep_to_indices = _build_episode_index(dataset)
    samples = []  # Returns [(ep_idx, [(fr_idx, global_idx), ...]), ...]

    selected_eps = []
    if episodes_str:
        ep_list = [int(e) for e in episodes_str.split(",")]
        for ep_idx in ep_list:
            if ep_idx in ep_to_indices:
                selected_eps.append(ep_idx)

    if random_n:
        rng = random.Random(seed)
        all_eps = list(ep_to_indices.keys())
        rng.shuffle(all_eps)
        for ep_idx in all_eps:
            if len(selected_eps) >= random_n:
                break
            if ep_idx not in selected_eps:
                selected_eps.append(ep_idx)
    
    for ep_idx in selected_eps:
        indices = ep_to_indices[ep_idx]
        ep_frames = []
        for fr_idx in range(0, len(indices), subsample):
            ep_frames.append((fr_idx, indices[fr_idx]))
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
    w_dtype = model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
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
        "model": model,
        "prefix_embs": prefix_embs,
        "prefix_pad_masks": prefix_pad_masks,
        "prefix_att_masks": prefix_att_masks,
        "w_dtype": w_dtype,
        "bsize": task_tokens.shape[0],
        "patches_per_cam": patches_per_cam,
        "segments_prefix": segments_prefix,  # without "action" segment
    }


@torch.no_grad()
def probe_forward(prefix_cache, time_val, device):
    """
    Single joint forward (prefix + suffix) without FAST tokens.
    Populates _PROBING_CAPTURE['attn_weights'] with the layer-0 attention matrix.

    Returns
    -------
    attn_weights : Tensor [B, n_heads, seq_len, seq_len]  (cpu, float32)
    segments     : list of (name, start, end)  — token index ranges, no FAST
    pad_masks    : Tensor [B, seq_len]  bool — valid (non-padding) positions
    patches_per_cam : int
    """
    model = prefix_cache["model"]
    prefix_embs = prefix_cache["prefix_embs"]
    prefix_pad_masks = prefix_cache["prefix_pad_masks"]
    prefix_att_masks = prefix_cache["prefix_att_masks"]
    w_dtype = prefix_cache["w_dtype"]
    bsize = prefix_cache["bsize"]
    patches_per_cam = prefix_cache["patches_per_cam"]
    segments_prefix = prefix_cache["segments_prefix"]

    # ── Embed suffix (pure noise at time_val) ─────────────────────────────────
    noise = model.sample_noise(
        (bsize, model.config.chunk_size, model.config.max_action_dim), device
    )
    time_tensor = torch.full((bsize,), time_val, dtype=torch.float32, device=device)

    if w_dtype == torch.bfloat16:
        noise = noise.to(torch.bfloat16)
        time_tensor = time_tensor.to(torch.bfloat16)

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
        noise, time_tensor
    )

    if w_dtype == torch.bfloat16:
        suffix_embs = suffix_embs.to(torch.bfloat16)

    # ── Combined attention mask (suffix attends to all prefix, no FAST to exclude)
    prefix_len = prefix_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]
    total_len = prefix_len + suffix_len

    suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    combined = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)
    combined[:, :prefix_len, :prefix_len] = prefix_att_masks
    combined[:, prefix_len:, prefix_len:] = suffix_att_2d
    combined[:, prefix_len:, :prefix_len] = True  # suffix sees all prefix

    combined_pad = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    pad_2d = combined_pad[:, None, :] & combined_pad[:, :, None]
    att_2d = combined & pad_2d

    position_ids = torch.cumsum(combined_pad, dim=1) - 1
    att_2d_4d = model._prepare_attention_masks_4d(att_2d)

    # ── Enable capture and run ─────────────────────────────────────────────────
    _PROBING_CAPTURE["enabled"] = True
    _PROBING_CAPTURE["attn_weights"] = None

    model.paligemma_with_expert.forward(
        attention_mask=att_2d_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )

    _PROBING_CAPTURE["enabled"] = False
    attn_weights = _PROBING_CAPTURE["attn_weights"]  # [B, heads, total_len, total_len]

    # ── Append "action" segment ───────────────────────────────────────────────
    action_start = segments_prefix[-1][2]  # end of "subtask"
    segments = segments_prefix + [("action", action_start, action_start + suffix_embs.shape[1])]

    return attn_weights, segments, combined_pad, patches_per_cam


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
    q_valid = pad_masks[q_start:q_end]  # [q_len]
    q_attn = attn_heads[:, q_start:q_end, k_start:k_end]  # [heads, q_len, n_patches]

    if q_valid.sum() == 0:
        dummy = torch.zeros(img_h, img_w)
        return dummy.unsqueeze(0).expand(n_heads, -1, -1), dummy

    q_attn_valid = q_attn[:, q_valid, :]  # [heads, valid_q, n_patches]
    per_head = q_attn_valid.mean(dim=1)   # [heads, n_patches]

    def _reshape_upsample(x):
        # x: [n_patches]  → grid [n_p, n_p] → [img_h, img_w]
        grid = x.reshape(n_p, n_p)
        grid_4d = grid[None, None].float()
        up = F.interpolate(grid_4d, size=(img_h, img_w), mode="bicubic", align_corners=False)
        return up.squeeze().clamp(min=0)

    per_head_up = torch.stack([_reshape_upsample(per_head[h]) for h in range(n_heads)])
    mean_up = _reshape_upsample(per_head.mean(0))
    return per_head_up, mean_up


# ──────────────────────────────────────────────────────────────────────────────
# Figure generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def cv2_overlay(img_np, heatmap, title, alpha=0.55, vmax=None):
    if vmax is None:
        vmax = heatmap.max().item()
    h_norm = heatmap / (vmax + 1e-8)
    h_gray = (h_norm.clamp(0, 1) * 255).numpy().astype(np.uint8)
    h_color = cv2.applyColorMap(h_gray, cv2.COLORMAP_JET)
    h_color_rgb = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    
    if h_color_rgb.shape != img_np.shape:
        h_color_rgb = cv2.resize(h_color_rgb, (img_np.shape[1], img_np.shape[0]))

    overlay = cv2.addWeighted(img_np, 1 - alpha, h_color_rgb, alpha, 0)
    cv2.putText(overlay, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay

def render_image_overlays(attn_weights, segments, image_tensors, pad_masks, patches_per_cam):
    """
    Returns
    -------
    frames_out  : dict  key → np.ndarray  (video frames)
    norm_consts : dict  key → float  (raw peak attention value used as vmax per panel)
    """
    frames_out = {}
    norm_consts = {}
    n_heads = attn_weights.shape[1]
    n_p = int(patches_per_cam ** 0.5)

    seg_dict = {name: (s, e) for name, s, e in segments}
    camera_segs = [(name, s, e) for name, s, e in segments if name.startswith("img")]
    query_groups = [
        ("all",      0,                           segments[-1][2]),
        ("language", *seg_dict["language"]),
        ("subtask",  *seg_dict["subtask"]),
        ("action",   *seg_dict["action"]),
    ]

    attn = torch.nan_to_num(attn_weights[0].float().cpu(), nan=0.0)
    pad = pad_masks[0].cpu()

    for cam_idx, (cam_name, cam_s, cam_e) in enumerate(camera_segs):
        img_np = None
        if cam_idx < len(image_tensors):
            img_t = image_tensors[cam_idx].squeeze(0).cpu()
            img_t = img_t * 0.5 + 0.5
            img_np = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        img_h = img_np.shape[0] if img_np is not None else 224
        img_w = img_np.shape[1] if img_np is not None else 224
        if img_np is None:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        mean_maps = []
        for qname, q_s, q_e in query_groups:
            _, mean_map = _attn_to_heatmap(
                attn, q_s, q_e, cam_s, cam_e, pad, n_p, img_h, img_w
            )
            mean_maps.append((qname, mean_map))

        mean_overlays = [img_np.copy()]
        cv2.putText(mean_overlays[0], f"{cam_name} (orig)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        for qname, mean_map in mean_maps:
            # Each panel normalized independently by its own peak value
            vmax = mean_map.max().item()
            norm_consts[f"{cam_name}_{qname}"] = vmax
            ov = cv2_overlay(img_np, mean_map, f"mean: {qname}", vmax=vmax)
            mean_overlays.append(ov)

        frames_out[f"{cam_name}_mean"] = np.hstack(mean_overlays)

        q_s_all, q_e_all = 0, segments[-1][2]
        per_head_maps, _ = _attn_to_heatmap(
            attn, q_s_all, q_e_all, cam_s, cam_e, pad, n_p, img_h, img_w
        )
        heads_vmax = per_head_maps.max().item()
        norm_consts[f"{cam_name}_heads"] = heads_vmax

        h_cols = 4
        h_rows = (n_heads + h_cols - 1) // h_cols

        head_rows = []
        for r in range(h_rows):
            row_imgs = []
            for c in range(h_cols):
                idx = r * h_cols + c
                if idx < n_heads:
                    row_imgs.append(cv2_overlay(img_np, per_head_maps[idx], f"head {idx}", vmax=heads_vmax))
                else:
                    row_imgs.append(np.zeros_like(img_np))
            head_rows.append(np.hstack(row_imgs))
        frames_out[f"{cam_name}_heads"] = np.vstack(head_rows)

    return frames_out, norm_consts

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: full attention matrix
# ──────────────────────────────────────────────────────────────────────────────

_SEG_COLORS = {
    "img1":     (76, 114, 176),
    "img2":     (85, 168, 104),
    "language": (196, 78, 82),
    "subtask":  (129, 114, 178),
    "action":   (204, 185, 116),
}
_DEFAULT_COLOR = (119, 119, 119)

def cv2_draw_matrix(attn_mat, segments, title):
    mat_np = attn_mat.numpy() if hasattr(attn_mat, 'numpy') else attn_mat
    L = mat_np.shape[0]
    out_size = 500
    
    mat_min, mat_max = mat_np.min(), mat_np.max()
    mat_norm = (mat_np - mat_min) / (mat_max - mat_min + 1e-8)
    mat_gray = (mat_norm * 255).astype(np.uint8)
    mat_color = cv2.applyColorMap(mat_gray, cv2.COLORMAP_VIRIDIS)
    mat_rgb = cv2.cvtColor(mat_color, cv2.COLOR_BGR2RGB)
    mat_rgb = cv2.resize(mat_rgb, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    scale = out_size / L if L > 0 else 1

    for name, s, e in segments:
        c = _SEG_COLORS.get(name, _DEFAULT_COLOR)
        for pos in [s, e]:
            p_scaled = int(pos * scale)
            line_overlay = mat_rgb.copy()
            cv2.line(line_overlay, (0, p_scaled), (out_size, p_scaled), c, 1)
            cv2.line(line_overlay, (p_scaled, 0), (p_scaled, out_size), c, 1)
            mat_rgb = cv2.addWeighted(mat_rgb, 0.7, line_overlay, 0.3, 0)
            
    cv2.rectangle(mat_rgb, (0, 0), (out_size, 25), (0, 0, 0), cv2.FILLED)
    cv2.putText(mat_rgb, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return mat_rgb

def render_full_matrix(attn_weights, segments, pad_masks):
    frames_out = {}
    n_heads = attn_weights.shape[1]
    attn = torch.nan_to_num(attn_weights[0].float().cpu(), nan=0.0)
    pad = pad_masks[0].cpu()

    pad_2d = pad[:, None] & pad[None, :]
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
# Main
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: ProbeAttentionConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)

    output_dir = cfg.probe_output_dir
    os.makedirs(output_dir, exist_ok=True)

    timesteps = [float(t) for t in cfg.probe_timesteps.split(",")]
    logging.info(f"Probing timesteps: {timesteps}")
    logging.info(f"Output dir: {output_dir}")

    # ── Load policy ────────────────────────────────────────────────────────────
    logging.info("Loading policy ...")
    policy, preprocessor, postprocessor, dataset = _load_policy_and_processors(cfg, device)
    policy.eval()

    tokenizer = policy.model._paligemma_tokenizer
    chunk_size = cfg.policy.n_action_steps

    def _probe_dataset(ds, ds_output_dir):
        os.makedirs(ds_output_dir, exist_ok=True)
        samples = build_sample_list(
            ds,
            episodes_str=cfg.eval_episodes,
            random_n=cfg.eval_random_n or None,
            subsample=cfg.eval_subsample,
            seed=cfg.eval_random_seed,
        )
        if not samples:
            logging.warning(f"  No samples found in {ds_output_dir}, skipping.")
            return
        logging.info(f"  {len(samples)} episodes × {len(timesteps)} timesteps → {ds_output_dir}")

        fps = getattr(ds, "fps", 30) / cfg.eval_subsample
        batch_sz = getattr(cfg, "probe_batch_size", 8)

        for ep_idx, ep_frames in samples:
            writers = {t_val: {} for t_val in timesteps}

            for i in range(0, len(ep_frames), batch_sz):
                batch_slice = ep_frames[i : i + batch_sz]

                b_obs = {}
                b_task_str = []
                b_gt_subtask = []
                for fr_idx, global_idx in batch_slice:
                    obs, _, state, gt_subtask, task_str, _, _ = get_frame_data(ds, global_idx, chunk_size)
                    b_task_str.append(task_str)
                    b_gt_subtask.append(gt_subtask)
                    for k, v in obs.items():
                        if k not in b_obs: b_obs[k] = []
                        b_obs[k].append(v)
                
                logging.info(f"    ep={ep_idx:04d} frames {batch_slice[0][0]:04d}..{batch_slice[-1][0]:04d} (batch size {len(batch_slice)})")

                for k in b_obs:
                    b_obs[k] = torch.cat(b_obs[k], dim=0).to(device)

                complementary_data = {
                    "task": b_task_str,
                    "subtask": [""] * len(batch_slice),
                    "advantage": torch.ones((len(batch_slice), 1), device=device),
                }
                dummy_action = torch.zeros(len(batch_slice), 1, 6, device=device)
                batch_for_proc = {
                    TransitionKey.ACTION: dummy_action,
                    **b_obs,
                    TransitionKey.COMPLEMENTARY_DATA: complementary_data,
                }
                processed = preprocessor(batch_for_proc)

                images, img_masks = policy._preprocess_images(b_obs)
                task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
                task_masks = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

                subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
                    images, img_masks, task_tokens, task_masks
                )

                prefix_cache = embed_probe_prefix(
                    policy, images, img_masks, task_tokens, task_masks,
                    subtask_tokens, subtask_masks,
                )

                for t_val in timesteps:
                    attn_weights, segments, pad_masks, patches_per_cam = probe_forward(
                        prefix_cache, t_val, device,
                    )
                    if attn_weights is None:
                        logging.warning(f"      No attention captured at t={t_val} — skipping.")
                        continue
                    
                    t_str = f"{t_val:.2f}".replace(".", "p")
                    ep_dir = os.path.join(ds_output_dir, f"ep{ep_idx:04d}_t{t_str}")
                    os.makedirs(ep_dir, exist_ok=True)

                    csv_path = os.path.join(ep_dir, "norm_consts.csv")
                    csv_file = open(csv_path, "a", newline="")
                    csv_writer = csv.writer(csv_file)
                    if os.path.getsize(csv_path) == 0:
                        csv_writer.writerow(["ep", "fr", "t_val", "panel", "vmax"])

                    for b_idx, (fr_idx, _) in enumerate(batch_slice):
                        a_w = attn_weights[b_idx:b_idx+1]
                        p_m = pad_masks[b_idx:b_idx+1]
                        i_t = [img[b_idx:b_idx+1] for img in images]

                        overlay_frames, norm_consts = render_image_overlays(
                            a_w, segments, i_t, p_m, patches_per_cam
                        )
                        frames_out = dict(overlay_frames)
                        frames_out.update(render_full_matrix(a_w, segments, p_m))

                        for panel, vmax in norm_consts.items():
                            csv_writer.writerow([ep_idx, fr_idx, t_val, panel, f"{vmax:.6e}"])

                        for k, frame_np in frames_out.items():
                            if k not in writers[t_val]:
                                path = os.path.join(ep_dir, f"{k}.mp4")
                                writers[t_val][k] = imageio.get_writer(path, fps=fps, macro_block_size=1)
                            writers[t_val][k].append_data(frame_np)

                    csv_file.close()

            # Close writers for this episode
            for dict_t in writers.values():
                for w in dict_t.values():
                    w.close()

    # ── Primary dataset ────────────────────────────────────────────────────────
    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    logging.info(f"Primary dataset: {cfg.dataset.root}")
    _probe_dataset(dataset, os.path.join(output_dir, primary_name))

    # ── Additional datasets ────────────────────────────────────────────────────
    extra_paths = getattr(cfg.dataset, "additional_offline_dataset_paths", []) or []
    for extra_root in extra_paths:
        logging.info(f"Additional dataset: {extra_root}")
        extra_ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=extra_root)
        extra_ds.delta_timestamps = None
        extra_ds.delta_indices = None
        ds_name = os.path.basename(os.path.normpath(extra_root))
        _probe_dataset(extra_ds, os.path.join(output_dir, ds_name))

    logging.info(f"Done. Output saved to {output_dir}/")


if __name__ == "__main__":
    probe_cli()
