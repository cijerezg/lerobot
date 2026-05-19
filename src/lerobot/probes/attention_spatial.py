#!/usr/bin/env python
"""
Spatial memorization probe — aggregate per-frame attention maps to detect heads
that learned fixed spatial patterns regardless of input.

Policy-agnostic: works with any policy whose ``ProbablePolicy`` adapter
implements ``capture_attention``. Pure consumer of the existing attention
capture — no new adapter methods required.

For N frames (1 per unique episode), captures cross-attention from action
queries to encoder keys, slices to each camera's image-patch segment, and
aggregates per-head statistics across frames:

  mean_map[h, patch]          = E_frames[ attn[h, patch] ]
                                where attn is mean over action queries.
  mean_over_std[h, patch]     = mean / (std + eps)
                                high  = high mean + low variance  → memorized.
                                low   = noisy / rarely attended.

Output (under ``probe_parameters.output_dir/attention_spatial/``):
  spatial_memorization_raw.pt   cached aggregates (delete to recompute)
  L{layer:02d}/
    mean_action_to_{cam}.png            per-head 2×4 grid
    mean_over_std_action_to_{cam}.png   per-head 2×4 grid

NOTE: the original pi05 probe also rendered ``language → cam`` and
``subtask → cam`` query groups by slicing the joint attention matrix on
multiple row ranges. The unified ``capture_attention`` API only exposes
action-query rows (cross_attn). Only ``action -> camera`` heatmaps are
produced; language/subtask query views are dropped.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import build_episode_index, get_frame_data, load_extra_dataset
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeSpatialMemorizationConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters``."""


_STD_EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# Sampling — one random frame per unique episode
# ──────────────────────────────────────────────────────────────────────────────

def sample_one_per_episode(dataset, n_frames: int, seed: int = 42):
    """Return ``[(ep_idx, fr_idx, global_idx), ...]`` (one frame per episode)."""
    ep_to_indices = build_episode_index(dataset)
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
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_maps(maps: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute mean and mean/std maps across a list of ``[H, K]`` tensors."""
    stacked = torch.stack(maps, dim=0)              # [N, H, K]
    mean_map = stacked.mean(dim=0)                   # [H, K]
    std_map  = stacked.std(dim=0)                    # [H, K]
    return {
        "mean":          mean_map,
        "mean_over_std": mean_map / (std_map + _STD_EPS),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def _upsample_patches(patch_vals: torch.Tensor, n_p: int, img_h: int, img_w: int):
    """Reshape ``[K]`` patch vector to ``[n_p, n_p]`` grid and bicubic-upsample."""
    grid = patch_vals.reshape(n_p, n_p).float()
    return F.interpolate(grid[None, None], size=(img_h, img_w),
                         mode="bicubic", align_corners=False).squeeze()


def _render_heatmap(values, title, img_h, img_w, vmin=None, vmax=None,
                    colormap=cv2.COLORMAP_VIRIDIS):
    if vmin is None:
        vmin = float(values.min())
    if vmax is None:
        vmax = float(values.max())
    norm = ((values - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    gray = (norm * 255).numpy().astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, colormap), cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (img_h, img_w):
        rgb = cv2.resize(rgb, (img_w, img_h))
    cv2.putText(rgb, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def render_stat_panel(stat_map: torch.Tensor, n_heads: int, n_p: int,
                      img_h: int, img_w: int, stat_name: str, cam_name: str):
    """Render a ``[H, K]`` stat map as a per-head 2×4 grid (per-head normalized)."""
    head_imgs = []
    for h in range(n_heads):
        up = _upsample_patches(stat_map[h], n_p, img_h, img_w)
        title = f"{stat_name} | action->{cam_name} | h{h}"
        head_imgs.append(_render_heatmap(up, title, img_h, img_w))

    h_cols = 4
    h_rows = (n_heads + h_cols - 1) // h_cols
    rows = []
    for r in range(h_rows):
        row_imgs = []
        for c in range(h_cols):
            idx = r * h_cols + c
            if idx < n_heads:
                row_imgs.append(head_imgs[idx])
            else:
                row_imgs.append(np.zeros((img_h, img_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Save / load cache
# ──────────────────────────────────────────────────────────────────────────────

def _save_cache(pt_path, raw_results, n_frames, layers, img_h, img_w, n_p):
    save_dict: dict = {}
    for (layer_idx, cam_name), stats in raw_results.items():
        prefix = f"L{layer_idx}_action_{cam_name}"
        for stat_name, tensor in stats.items():
            save_dict[f"{prefix}_{stat_name}"] = tensor
    save_dict["n_frames"] = torch.tensor(n_frames)
    save_dict["layers"]   = torch.tensor(layers)
    save_dict["img_hw"]   = torch.tensor([img_h, img_w])
    save_dict["n_p"]      = torch.tensor(n_p)
    torch.save(save_dict, pt_path)


def _load_cache(pt_path):
    if not os.path.isfile(pt_path):
        return None
    logging.info(f"Loading cached aggregates from {pt_path}")
    save_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    img_hw = save_dict["img_hw"]
    img_h, img_w = int(img_hw[0].item()), int(img_hw[1].item())
    n_p = int(save_dict["n_p"].item())

    raw: dict = {}
    n_heads = None
    for key, tensor in save_dict.items():
        if key in ("n_frames", "layers", "img_hw", "n_p"):
            continue
        for sn in ("mean", "mean_over_std"):
            if key.endswith(f"_{sn}"):
                remainder = key[: -len(f"_{sn}")]      # e.g. "L0_action_img1"
                # Format: "L{layer}_action_{cam_name}".
                parts = remainder.split("_", 2)
                layer_idx = int(parts[0][1:])
                cam_name = parts[2]
                rkey = (layer_idx, cam_name)
                raw.setdefault(rkey, {})[sn] = tensor
                if n_heads is None:
                    n_heads = tensor.shape[0]
                break
    return raw, n_heads, n_p, img_h, img_w


def render_all(raw_results, n_heads, n_p, img_h, img_w, output_dir):
    for (layer_idx, cam_name), stats in raw_results.items():
        layer_dir = os.path.join(output_dir, f"L{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        for stat_name in ("mean", "mean_over_std"):
            panel = render_stat_panel(
                stats[stat_name], n_heads, n_p, img_h, img_w,
                stat_name, cam_name,
            )
            fname = f"{stat_name}_action_to_{cam_name}.png"
            cv2.imwrite(os.path.join(layer_dir, fname),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# ──────────────────────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_aggregates(adapter: ProbablePolicy, dataset, samples, layers, timestep):
    """Run capture_attention per frame; aggregate per (layer, camera) head maps.

    Sequential (one frame per call) because ``capture_attention`` is single-batch
    by API. Batched capture would be a future API extension.
    """
    chunk_size = adapter.chunk_size

    # collected[(layer_idx, cam_name)] = list of [H, K] tensors, one per frame
    collected: dict[tuple[int, str], list[torch.Tensor]] = {}
    img_h = img_w = None
    n_heads = n_p = None
    patches_per_cam = None

    for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
        logging.debug(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")
        obs, _, state, _, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size,
        )
        result = adapter.capture_attention(
            obs, task_str, state=state, timestep=timestep, layers=layers,
        )

        if not result.cross_attn_by_layer:
            logging.warning(f"  ep={ep_idx} fr={fr_idx}: no cross-attn captured, skipping.")
            continue

        patch_indices = result.extras.get("image_patch_indices_by_segment", {})
        cam_segs = [(name, s, e) for name, s, e in result.encoder_segments
                    if name.startswith("img")]
        if not cam_segs or (result.patches_per_cam <= 0 and not patch_indices):
            logging.warning(
                f"  ep={ep_idx} fr={fr_idx}: no image patch segments available; "
                "skipping spatial aggregation for this frame."
            )
            continue

        if img_h is None and result.image_tensors:
            img_t = result.image_tensors[0][0].cpu()
            img_h = int(img_t.shape[1])
            img_w = int(img_t.shape[2])
        if patches_per_cam is None:
            if patch_indices:
                first_indices = next(iter(patch_indices.values()))
                patches_per_cam = len(first_indices)
            else:
                patches_per_cam = result.patches_per_cam or (cam_segs[0][2] - cam_segs[0][1])
            n_p = int(patches_per_cam ** 0.5) if patches_per_cam > 0 else 1
            if n_p * n_p != patches_per_cam:
                logging.warning(
                    f"  ep={ep_idx} fr={fr_idx}: patch count {patches_per_cam} is not square; "
                    "skipping spatial aggregation for this frame."
                )
                continue

        for layer_idx, cross in result.cross_attn_by_layer.items():
            # cross: [B=1, H, n_action, encoder_seq_len]
            attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)   # [H, n_action, encoder_seq_len]
            if n_heads is None:
                n_heads = attn.shape[0]

            for cam_name, cs, ce in cam_segs:
                indices = patch_indices.get(cam_name)
                if indices is not None:
                    idx = torch.as_tensor(indices, dtype=torch.long)
                    vec = attn.index_select(dim=2, index=idx).mean(dim=1)
                else:
                    # Mean over action queries → [H, K=ce-cs]
                    vec = attn[:, :, cs:ce].mean(dim=1)
                key = (layer_idx, cam_name)
                collected.setdefault(key, []).append(vec)

    return collected, n_heads, n_p, img_h or 224, img_w or 224


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _probe_one_dataset(adapter, dataset, ds_dir, cfg):
    p = cfg.probe_parameters
    os.makedirs(ds_dir, exist_ok=True)
    pt_path = os.path.join(ds_dir, "spatial_memorization_raw.pt")

    cached = _load_cache(pt_path)
    if cached is not None:
        raw_results, n_heads, n_p, img_h, img_w = cached
        logging.info("Re-rendering from cache (delete .pt to recompute).")
        render_all(raw_results, n_heads, n_p, img_h, img_w, ds_dir)
        return

    layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    timestep = float(getattr(p, "timestep", 0.5))
    n_frames = int(getattr(p, "max_episodes", None) or 32)
    seed = int(getattr(p, "random_seed", 42))

    logging.info(
        f"  layers={layers} timestep={timestep} n_frames={n_frames}"
    )

    samples = sample_one_per_episode(dataset, n_frames=n_frames, seed=seed)
    if not samples:
        logging.warning("  No samples; skipping.")
        return
    logging.info(f"  Sampled {len(samples)} frames from {len(samples)} episodes")

    collected, n_heads, n_p, img_h, img_w = collect_aggregates(
        adapter, dataset, samples, layers, timestep,
    )

    raw_results: dict = {}
    for (layer_idx, cam_name), maps_list in collected.items():
        if len(maps_list) < 2:
            logging.warning(
                f"  L{layer_idx} action->{cam_name}: only {len(maps_list)} frames, skipping"
            )
            continue
        raw_results[(layer_idx, cam_name)] = aggregate_maps(maps_list)

    if not raw_results:
        logging.warning("  No aggregates produced; nothing to render.")
        return

    _save_cache(pt_path, raw_results, len(samples), layers, img_h, img_w, n_p)
    logging.debug(f"  Saved cache → {pt_path}")
    render_all(raw_results, n_heads, n_p, img_h, img_w, ds_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(adapter, primary_dataset, cfg, output_dir):
    """Run the spatial-memorization probe on primary + additional datasets."""
    if adapter is None or primary_dataset is None:
        return
    os.makedirs(output_dir, exist_ok=True)

    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    logging.info(f"=== Dataset: {primary_name} ===")
    _probe_one_dataset(adapter, primary_dataset,
                       os.path.join(output_dir, primary_name), cfg)

    for extra_root in getattr(cfg.dataset, "additional_offline_dataset_paths", None) or []:
        ds_name = os.path.basename(os.path.normpath(extra_root))
        logging.info(f"=== Dataset: {ds_name} ===")
        extra_ds = load_extra_dataset(cfg.dataset.repo_id, extra_root)
        _probe_one_dataset(adapter, extra_ds,
                           os.path.join(output_dir, ds_name), cfg)


@parser.wrap()
def probe_cli(cfg: ProbeSpatialMemorizationConfig):
    init_logging()
    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "attention_spatial")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    from lerobot.datasets.factory import make_dataset
    primary_dataset = make_dataset(cfg)
    primary_dataset.delta_timestamps = None
    primary_dataset.delta_indices = None

    logging.info("Loading policy adapter …")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=primary_dataset)
    run(adapter, primary_dataset, cfg, output_dir)
    logging.info(f"Done. Output in {output_dir}/")


if __name__ == "__main__":
    probe_cli()
