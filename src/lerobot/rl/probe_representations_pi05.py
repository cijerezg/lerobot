#!/usr/bin/env python
"""
PI05 representation probe — visualise what the model has learned.

For each sampled frame we run a forward pass and capture two activation tensors:
  prefix_out  (B, prefix_len, 2048)  — VLM hidden states after image + language tokens
  suffix_out  (B, suffix_len, 1024)  — expert hidden states after noisy action tokens

Each tensor is mean-pooled to one vector per frame, then compressed with PCA (→100 dims)
and UMAP (→2D or 3D) for plotting. Suffix is collected at multiple denoising timesteps t.

For the subtask-injection analysis we run each frame twice — once with the GT subtask
tokens, once with model-generated ones — and embed both sets in the same UMAP space to
see how much the subtask representation drifts.

Output layout (all under probe_parameters.output_dir/representations/):
  {dataset}/activations_cache.pt          reusable tensor cache (skip re-inference with --probe_mode plot)
  {dataset}/episode_thumbnails/           first-frame images so you can identify each episode
  {dataset}/pca_variance/                 scree plots for every PCA fit
  {dataset}/2d/<site>/by_episode.png      per-episode gradient: dark=early frame, light=late
  {dataset}/2d/<site>/by_frame.png        all episodes pooled, coloured by frame index
  {dataset}/2d/<site>/by_subtask.png      all episodes pooled, coloured by subtask
  {dataset}/3d/<site>/by_episode.html     interactive version of by_episode
  {dataset}/3d/<site>/by_subtask.html     interactive version of by_subtask
  {dataset}/3d/<site>/ep{A}_vs_ep{B}.html two-episode comparison coloured by frame index
  {dataset}/subtask_injection/<site>/     GT vs generated subtask UMAP (2d + 3d)
  {dataset}/subtask_injection/generated_subtasks.csv  per-frame GT and model-generated subtask text

Usage:
    python probe_representations_pi05.py config-hiserl.json
    python probe_representations_pi05.py config-hiserl.json --probe_parameters.output_dir outputs/probe
    python probe_representations_pi05.py config-hiserl.json --probe_parameters.mode plot
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.pi05_full.modeling_pi05 import pad_vector
from lerobot.processor.core import TransitionKey
from lerobot.scripts.probe_offline_inference_pi05 import _build_episode_index, get_frame_data
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.rl.probe_utils_pi05 import (
    EP_COLORS,
    SEQ_CMAPS,
    ax_style,
    frame_colors_rgba,
    get_subtask_idx,
    load_extra_dataset,
    load_policy_and_processors,
    makedirs,
    plotly_3d_layout,
    run_pca,
    run_umap,
    sample_episodes_evenly,
)


# ──────────────────────────────────────────────────────────────────────────────
# Probe parameters — defaults, override via config dataclass
# ──────────────────────────────────────────────────────────────────────────────

N_FRAMES_PER_EPISODE = 128
MAX_EPISODES = 5
RANDOM_SEED = 42

PROBE_SITES = "prefix,suffix"
DENOISING_TIMESTEPS = "1.0,0.25"

PCA_DIMS = 100
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SEED = 42

EPISODE_3D_A = 0
EPISODE_3D_B = 1

DO_SUBTASK_INJECTION = True


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeRepresentationsConfig(TrainRLServerPipelineConfig):
    """Extends the base training config with representation probe parameters.

    All probe tunables live under cfg.probe_parameters (ProbeConfig).
    Relevant fields for this script:
      output_dir, mode, max_episodes, n_frames_per_episode, random_seed,
      sites, timesteps, repr_pca_dims, umap_n_neighbors, umap_min_dist, umap_seed,
      ep_3d_a, ep_3d_b, subtask_injection.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Hook utilities
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _capture_hook(module):
    """
    Monkey-patches module.forward to capture its outputs and yields a shared dict.
    After each call the dict contains:
      "prefix_out": (B, prefix_len, hidden_dim)  — float32, on CPU
      "suffix_out": (B, full_seq_len, hidden_dim) — float32, on CPU

    NOTE: register_forward_hook is NOT used because the call chain calls .forward()
    directly (not via __call__), which bypasses PyTorch hooks entirely.
    """
    captured = {}
    original_forward = module.forward

    def patched_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        if result[0][0] is not None:
            captured["prefix_out"] = result[0][0].detach().float()
        if result[0][1] is not None:
            captured["suffix_out"] = result[0][1].detach().float()
        return result

    module.forward = patched_forward
    try:
        yield captured
    finally:
        module.forward = original_forward


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_at_t(policy, images, img_masks, task_tokens, task_masks,
                  subtask_tokens, subtask_masks, action_tokens, action_masks,
                  actions_padded, t_val, device):
    """Single PI05Pytorch.forward() call with an explicit diffusion timestep t_val."""
    noise = torch.randn_like(actions_padded)
    time_tensor = torch.full(
        (actions_padded.shape[0],), t_val, device=device, dtype=actions_padded.dtype
    )
    # FAST action tokens are training-only; pass None to match inference behaviour
    policy.model.forward(
        images, img_masks,
        task_tokens, task_masks,
        subtask_tokens, subtask_masks,
        None, None,
        actions_padded,
        noise=noise,
        time=time_tensor,
    )


def _prepare_inputs(policy, preprocessor, obs, gt_actions, gt_subtask, task_str, device):
    """
    Run preprocessor to get tokenised inputs and padded actions for model forward.

    Returns:
        images, img_masks  — preprocessed image tensors (list)
        task_tokens, task_masks       — (1, seq_len) on device
        subtask_tokens, subtask_masks — (1, seq_len) on device, using gt_subtask
        action_tokens, action_masks   — FAST discrete action tokens (1, n_tokens) on device
        actions_padded  — (1, chunk_size, max_action_dim) normalised + padded on device
    """
    raw_batch = {
        TransitionKey.ACTION: gt_actions.unsqueeze(0).to(device),
        **{k: v.to(device) for k, v in obs.items()},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task":      [task_str],
            "subtask":   [gt_subtask],
            "advantage": torch.tensor([[1.0]], device=device),
        },
    }
    processed = preprocessor(raw_batch)

    images, img_masks = policy._preprocess_images({k: v.to(device) for k, v in obs.items()})

    task_tokens    = processed[OBS_LANGUAGE_TOKENS].to(device)
    task_masks     = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)
    subtask_tokens = processed[OBS_LANGUAGE_SUBTASK_TOKENS].to(device)
    subtask_masks  = processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK].to(device)
    action_tokens  = processed[ACTION_TOKENS].to(device)
    action_masks   = processed[ACTION_TOKEN_MASK].to(device)

    actions_norm   = processed[ACTION]
    actions_padded = pad_vector(actions_norm.to(device), policy.config.max_action_dim)

    return (images, img_masks, task_tokens, task_masks, subtask_tokens, subtask_masks,
            action_tokens, action_masks, actions_padded)


def _mean_pool(tensor, dim=1):
    return tensor.mean(dim=dim).cpu()  # (B, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Activation collection
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_activations(policy, preprocessor, dataset, samples, device, cfg):
    """
    Forward the model over sampled frames and collect prefix / suffix activations.

    Returns dict:
      "prefix":   (N, 2048) mean-pooled prefix_out using GT subtask
      "suffix":   {t_val: (N, 1024)} mean-pooled suffix_out[:, -chunk_size:] at each t
      "metadata": list of N dicts
    """
    sites      = [s.strip() for s in cfg.probe_parameters.sites.split(",")]
    t_values   = [cfg.probe_parameters.timestep]
    chunk_size = cfg.policy.chunk_size

    all_prefix = []
    all_suffix = {t: [] for t in t_values}
    metadata   = []

    with _capture_hook(policy.model.paligemma_with_expert) as captured:
        for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
            if i % 100 == 0:
                logging.debug(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

            obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
                dataset, global_idx, chunk_size
            )
            inputs = _prepare_inputs(
                policy, preprocessor, obs, gt_actions, gt_subtask, task_str, device
            )
            (images, img_masks, task_tokens, task_masks,
             subtask_tokens, subtask_masks, action_tokens, action_masks, actions_padded) = inputs

            for t_idx, t_val in enumerate(t_values):
                _forward_at_t(
                    policy, images, img_masks,
                    task_tokens, task_masks,
                    subtask_tokens, subtask_masks,
                    action_tokens, action_masks,
                    actions_padded, t_val, device,
                )

                # prefix_out is t-independent; collect once on the first pass
                if t_idx == 0 and "prefix" in sites:
                    all_prefix.append(_mean_pool(captured["prefix_out"]))

                if "suffix" in sites:
                    raw    = captured["suffix_out"]         # (1, seq_len, 1024)
                    sliced = raw[:, -chunk_size:, :]        # (1, chunk_size, 1024)
                    all_suffix[t_val].append(_mean_pool(sliced))

            metadata.append({
                "episode_idx": ep_idx,
                "frame_idx":   fr_idx,
                "global_idx":  global_idx,
                "subtask":     gt_subtask,
                "subtask_idx": get_subtask_idx(dataset, global_idx),
                "task":        task_str,
            })

    result = {"metadata": metadata}
    if "prefix" in sites and all_prefix:
        result["prefix"] = torch.cat(all_prefix, dim=0)
    if "suffix" in sites:
        result["suffix"] = {t: torch.cat(v, dim=0) for t, v in all_suffix.items() if v}
    return result


@torch.no_grad()
def collect_subtask_injection(policy, preprocessor, dataset, samples, device, cfg):
    """
    For each frame run two forwards: one with GT subtask tokens, one with
    model-generated subtask tokens. Captures prefix and suffix for both.

    Returns dict with keys:
      "prefix_gt", "prefix_gen":    (N, 2048)
      "suffix_gt", "suffix_gen":    {t_val: (N, 1024)}
      "gen_subtask_texts":          list of N strings
    """
    t_values   = [cfg.probe_parameters.timestep]
    chunk_size = cfg.policy.chunk_size
    tokenizer  = policy.model._paligemma_tokenizer

    prefix_gt, prefix_gen = [], []
    suffix_gt  = {t: [] for t in t_values}
    suffix_gen = {t: [] for t in t_values}
    gen_subtask_texts = []

    policy.model.suppress_debug_log = True
    with _capture_hook(policy.model.paligemma_with_expert) as captured:
        for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
            if i % 100 == 0:
                logging.debug(f"  [injection {i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

            obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
                dataset, global_idx, chunk_size
            )
            inputs = _prepare_inputs(
                policy, preprocessor, obs, gt_actions, gt_subtask, task_str, device
            )
            (images, img_masks, task_tokens, task_masks,
             gt_sub_tokens, gt_sub_masks, action_tokens, action_masks, actions_padded) = inputs

            gen_sub_tokens, gen_sub_masks = policy.model.generate_subtask_tokens(
                images, img_masks, task_tokens, task_masks
            )
            valid = gen_sub_tokens[0][gen_sub_masks[0]]
            gen_subtask_texts.append(tokenizer.decode(valid, skip_special_tokens=True).strip())

            for cond_name, sub_tok, sub_mask in [
                ("gt",  gt_sub_tokens,  gt_sub_masks),
                ("gen", gen_sub_tokens, gen_sub_masks),
            ]:
                for t_idx, t_val in enumerate(t_values):
                    _forward_at_t(
                        policy, images, img_masks,
                        task_tokens, task_masks,
                        sub_tok, sub_mask,
                        action_tokens, action_masks,
                        actions_padded, t_val, device,
                    )

                    if t_idx == 0:
                        pout = _mean_pool(captured["prefix_out"])
                        (prefix_gt if cond_name == "gt" else prefix_gen).append(pout)

                    sout = _mean_pool(captured["suffix_out"][:, -chunk_size:, :])
                    (suffix_gt if cond_name == "gt" else suffix_gen)[t_val].append(sout)

    policy.model.suppress_debug_log = False
    return {
        "prefix_gt":         torch.cat(prefix_gt,  dim=0),
        "prefix_gen":        torch.cat(prefix_gen, dim=0),
        "suffix_gt":         {t: torch.cat(v, dim=0) for t, v in suffix_gt.items()},
        "suffix_gen":        {t: torch.cat(v, dim=0) for t, v in suffix_gen.items()},
        "gen_subtask_texts": gen_subtask_texts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots (matplotlib static PNGs)
# ──────────────────────────────────────────────────────────────────────────────

# Sequential colormaps for representations — includes warm tones (no pred colour conflict here).
_REP_SEQ_CMAPS = ["Blues", "Reds", "Greens", "Oranges", "Purples",
                  "YlOrBr", "PuRd", "BuGn", "GnBu", "OrRd"]


def plot_2d_by_episode(emb, metadata, output_path):
    from matplotlib.patches import Patch

    ep_ids     = np.array([m["episode_idx"] for m in metadata])
    frame_ids  = np.array([m["frame_idx"]   for m in metadata])
    unique_eps = np.unique(ep_ids)

    fig, ax = plt.subplots(figsize=(7, 6))
    legend_handles = []
    for i, ep in enumerate(unique_eps):
        mask      = ep_ids == ep
        ep_frames = frame_ids[mask]
        fmin, fmax = ep_frames.min(), ep_frames.max()
        norm       = 0.9 - (ep_frames - fmin) / max(fmax - fmin, 1) * 0.6
        cmap_ep    = matplotlib.colormaps.get_cmap(_REP_SEQ_CMAPS[i % len(_REP_SEQ_CMAPS)])
        colors     = cmap_ep(norm)
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors, s=22, alpha=0.85, linewidths=0)
        legend_handles.append(matplotlib.patches.Patch(facecolor=cmap_ep(0.6), label=f"ep {ep}"))

    if len(unique_eps) <= 20:
        ax.legend(handles=legend_handles, fontsize=6, ncol=2)
    ax_style(ax, f"By episode ({len(unique_eps)} eps) — dark→light = early→late frame", width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_frame(emb, metadata, output_path):
    frame_ids = np.array([m["frame_idx"] for m in metadata])
    n_eps     = len(np.unique([m["episode_idx"] for m in metadata]))
    fig, ax   = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=frame_ids, cmap="plasma",
                    s=22, alpha=0.80, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Frame index within episode")
    ax_style(ax, f"By frame index (temporal position) — {n_eps} episodes pooled", width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_subtask(emb, metadata, output_path):
    sub_ids      = np.array([m["subtask_idx"] for m in metadata])
    unique_subs  = np.unique(sub_ids)
    subtask_text = {m["subtask_idx"]: m["subtask"] for m in metadata}
    cmap         = matplotlib.colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, sub_id in enumerate(unique_subs):
        mask  = sub_ids == sub_id
        label = subtask_text.get(sub_id, str(sub_id))
        if len(label) > 32:
            label = label[:30] + "…"
        ax.scatter(emb[mask, 0], emb[mask, 1], s=22, alpha=0.80, linewidths=0,
                   color=cmap(i), label=f"[{sub_id}] {label}")
    n_eps = len(np.unique([m["episode_idx"] for m in metadata]))
    ax.legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_style(ax, f"By subtask — {n_eps} episodes pooled", width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_subtask_injection(emb_gt, emb_gen, metadata, output_path):
    """
    GT points (●) and gen points (✕) in the same UMAP space.
    UMAP was fit on the concatenated GT+gen set; caller slices and passes them separately.
    Points are coloured by subtask index.
    """
    from matplotlib.lines import Line2D

    sub_ids     = np.array([m["subtask_idx"] for m in metadata])
    unique_subs = np.unique(sub_ids)
    cmap        = matplotlib.colormaps.get_cmap("tab10")
    sub_to_i    = {s: i for i, s in enumerate(unique_subs)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for j, m in enumerate(metadata):
        c = cmap(sub_to_i[sub_ids[j]])
        ax.scatter(emb_gt[j, 0],  emb_gt[j, 1],  c=[c], s=20, alpha=0.85,
                   marker="o", linewidths=0)
        ax.scatter(emb_gen[j, 0], emb_gen[j, 1], c=[c], s=24, alpha=0.70,
                   marker="x", linewidths=0.9)

    n_eps = len(set(m["episode_idx"] for m in metadata))
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="gray", linestyle="none", markersize=7,
               label="GT subtask"),
        Line2D([0], [0], marker="x", color="gray", linestyle="none", markersize=7,
               markeredgewidth=1.2, label="Generated subtask"),
    ], fontsize=8)
    ax_style(ax, f"Subtask injection — GT (●) vs generated (✕), by subtask · {n_eps} eps",
             width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3D plots (Plotly interactive HTML)
# ──────────────────────────────────────────────────────────────────────────────

def _plotly_scatter3d_single(emb, color_vals, color_label, hover_texts,
                              title, output_path, colorscale="Turbo", traces=None):
    """Save a single- or multi-trace interactive 3D scatter as HTML."""
    import plotly.graph_objects as go

    if traces is None:
        traces = [go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="markers",
            marker=dict(size=6, color=color_vals, colorscale=colorscale,
                        showscale=True, opacity=0.85, line=dict(width=0),
                        colorbar=dict(title=color_label)),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )]

    fig = go.Figure(data=traces)
    fig.update_layout(**plotly_3d_layout(title))
    fig.write_html(output_path)


def plot_3d_by_episode(emb, metadata, output_path):
    import plotly.graph_objects as go

    ep_ids     = [m["episode_idx"] for m in metadata]
    frame_ids  = [m["frame_idx"]   for m in metadata]
    unique_eps = sorted(set(ep_ids))
    n_eps      = len(unique_eps)

    traces = []
    for i, ep in enumerate(unique_eps):
        idx    = [j for j, e in enumerate(ep_ids) if e == ep]
        frames = np.array([frame_ids[j] for j in idx])
        colors = frame_colors_rgba(frames, _REP_SEQ_CMAPS[i % len(_REP_SEQ_CMAPS)], alpha=0.85)
        hover  = [f"ep={ep} fr={metadata[j]['frame_idx']}<br>{metadata[j]['subtask']}"
                  for j in idx]
        traces.append(go.Scatter3d(
            x=[emb[j, 0] for j in idx],
            y=[emb[j, 1] for j in idx],
            z=[emb[j, 2] for j in idx],
            mode="markers",
            name=f"ep {ep}",
            marker=dict(size=6, color=colors, line=dict(width=0)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    _plotly_scatter3d_single(
        emb, None, None, None,
        f"3D — by episode ({n_eps} eps) · dark→pale = early→late frame",
        output_path,
        traces=traces,
    )


def plot_3d_by_frame(emb, metadata, output_path):
    import plotly.graph_objects as go

    frame_ids = [m["frame_idx"] for m in metadata]
    n_eps     = len(set(m["episode_idx"] for m in metadata))
    hover     = [f"ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
                 for m in metadata]

    traces = [go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode="markers",
        marker=dict(size=6, color=frame_ids, colorscale="Plasma",
                    showscale=True, opacity=0.85, line=dict(width=0),
                    colorbar=dict(title="Frame index within episode")),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    )]

    _plotly_scatter3d_single(
        emb, None, None, None,
        f"3D — by frame index (temporal position) — {n_eps} episodes pooled",
        output_path,
        traces=traces,
    )


def plot_3d_by_subtask(emb, metadata, output_path):
    sub_ids = [m["subtask_idx"] for m in metadata]
    n_eps   = len(set(m["episode_idx"] for m in metadata))
    hover   = [f"ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
               for m in metadata]
    _plotly_scatter3d_single(
        emb, sub_ids, "Subtask ID", hover,
        f"3D — by subtask ({n_eps} episodes pooled)",
        output_path, colorscale="Rainbow",
    )


def plot_3d_two_episodes(emb, metadata, ep_a, ep_b, output_path):
    """
    Only points from ep_a and ep_b. ep_a = circles, ep_b = squares.
    Coloured by frame index within episode.
    """
    import plotly.graph_objects as go

    ep_marker = {ep_a: "circle", ep_b: "square"}
    traces = []
    for ep, sym in ep_marker.items():
        idx    = [i for i, m in enumerate(metadata) if m["episode_idx"] == ep]
        if not idx:
            continue
        sub    = emb[idx]
        frames = [metadata[i]["frame_idx"] for i in idx]
        hover  = [f"ep={ep} fr={metadata[i]['frame_idx']}<br>{metadata[i]['subtask']}"
                  for i in idx]
        traces.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
            mode="markers",
            name=f"Episode {ep}",
            marker=dict(size=6, color=frames, colorscale="Plasma",
                        showscale=(ep == ep_a), opacity=0.85, line=dict(width=0),
                        colorbar=dict(title="Frame index"), symbol=sym),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    _plotly_scatter3d_single(
        emb, None, None, None,
        f"3D — episode {ep_a} (●) vs {ep_b} (■) · coloured by frame index",
        output_path,
        traces=traces,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Episode thumbnails
# ──────────────────────────────────────────────────────────────────────────────

def _grid_layout(n):
    """Return (rows, cols) for a near-square grid that fits n cells, wider than tall."""
    import math
    rows = max(1, math.floor(math.sqrt(n)))
    cols = math.ceil(n / rows)
    return rows, cols


def _save_episode_thumbnails(dataset, ep_to_indices, output_dir):
    """Save a montage of first-frame thumbnails for every episode."""
    sample_frame = dataset[0]
    img_keys = sorted(k for k in sample_frame.keys()
                      if k.startswith("observation.image") and
                      isinstance(sample_frame[k], torch.Tensor))
    if not img_keys:
        logging.warning("No image observation keys found; skipping episode thumbnails.")
        return

    episodes  = sorted(ep_to_indices.keys())
    n_eps     = len(episodes)
    n_cams    = len(img_keys)
    cam_labels = [k.replace("observation.images.", "").replace("observation.image.", "")
                  for k in img_keys]

    if n_cams == 1:
        rows, cols = _grid_layout(n_eps)
    else:
        rows, cols = n_eps, n_cams

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    for i, ep_idx in enumerate(episodes):
        frame  = dataset[ep_to_indices[ep_idx][0]]
        r, c_start = (i, 0) if n_cams > 1 else divmod(i, cols)
        for cam_j, (key, label) in enumerate(zip(img_keys, cam_labels)):
            c      = c_start + cam_j if n_cams == 1 else cam_j
            ax     = axes[r][c]
            img    = frame[key]
            img_np = img.float().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8) \
                     if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            ax.imshow(img_np)
            ax.axis("off")
            title = f"ep {ep_idx}" if n_cams == 1 else f"ep {ep_idx} · {label}"
            ax.set_title(title, fontsize=7)

    if n_cams == 1:
        for j in range(n_eps, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].set_visible(False)

    fig.suptitle(f"Episode thumbnails  ({n_eps} episodes)", fontsize=10)
    fig.tight_layout(pad=0.3)
    out_path = os.path.join(output_dir, "episode_thumbnails.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _run_site(tag, X, metadata, cfg, pca_dir, output_dir):
    """PCA → 2D UMAP → 3D UMAP → all standard plots for one activation tensor."""
    logging.info(f"  Site '{tag}'  shape={tuple(X.shape)}")

    d2 = os.path.join(output_dir, "2d", tag)
    d3 = os.path.join(output_dir, "3d", tag)
    makedirs(d2, d3)

    X_pca, _ = run_pca(X, cfg.probe_parameters.repr_pca_dims, tag, pca_dir)

    logging.info(f"    Fitting 2D UMAP …")
    emb2 = run_umap(X_pca, 2, cfg.probe_parameters.umap_n_neighbors, cfg.probe_parameters.umap_min_dist,
                    cfg.probe_parameters.umap_seed)
    plot_2d_by_episode(emb2, metadata, os.path.join(d2, "by_episode.png"))
    plot_2d_by_frame(emb2, metadata,   os.path.join(d2, "by_frame.png"))
    plot_2d_by_subtask(emb2, metadata, os.path.join(d2, "by_subtask.png"))

    logging.info(f"    Fitting 3D UMAP …")
    emb3 = run_umap(X_pca, 3, cfg.probe_parameters.umap_n_neighbors, cfg.probe_parameters.umap_min_dist,
                    cfg.probe_parameters.umap_seed)
    plot_3d_by_episode(emb3, metadata, os.path.join(d3, "by_episode.html"))
    plot_3d_by_frame(emb3, metadata,   os.path.join(d3, "by_frame.html"))
    plot_3d_by_subtask(emb3, metadata, os.path.join(d3, "by_subtask.html"))
    plot_3d_two_episodes(
        emb3, metadata, cfg.probe_parameters.ep_3d_a, cfg.probe_parameters.ep_3d_b,
        os.path.join(d3, f"ep{cfg.probe_parameters.ep_3d_a}_vs_ep{cfg.probe_parameters.ep_3d_b}.html"),
    )


def run_plotting(cache, cfg, output_dir):
    """Load cached activations, run all PCA+UMAP reductions, save plots."""
    metadata = cache["metadata"]
    sites    = [s.strip() for s in cfg.probe_parameters.sites.split(",")]
    t_values = [cfg.probe_parameters.timestep]
    pca_dir  = os.path.join(output_dir, "pca_variance")
    makedirs(pca_dir)

    # ── Standard sites ──────────────────────────────────────────────────────
    if "prefix" in sites and "prefix" in cache:
        _run_site("prefix", cache["prefix"], metadata, cfg, pca_dir, output_dir)

    if "suffix" in sites and "suffix" in cache:
        for t_val in t_values:
            if t_val not in cache["suffix"]:
                continue
            _run_site(f"suffix_t{t_val}", cache["suffix"][t_val], metadata, cfg,
                      pca_dir, output_dir)

    # ── Subtask injection ───────────────────────────────────────────────────
    if "prefix_gt" not in cache or "prefix_gen" not in cache:
        return

    logging.info("  Plotting subtask injection …")
    n = len(metadata)

    # Save generated subtask CSV
    gen_texts_all = cache.get("gen_subtask_texts", [])
    if gen_texts_all:
        import csv
        csv_path = os.path.join(output_dir, "subtask_injection", "generated_subtasks.csv")
        makedirs(os.path.dirname(csv_path))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode_idx", "frame_idx", "global_idx",
                               "gt_subtask", "gen_subtask"]
            )
            writer.writeheader()
            for i, m in enumerate(metadata):
                writer.writerow({
                    "episode_idx": m["episode_idx"],
                    "frame_idx":   m["frame_idx"],
                    "global_idx":  m["global_idx"],
                    "gt_subtask":  m["subtask"],
                    "gen_subtask": gen_texts_all[i] if i < len(gen_texts_all) else "",
                })

    def _injection_plots(tag, X_gt, X_gen):
        inj2 = os.path.join(output_dir, "subtask_injection", tag, "2d")
        inj3 = os.path.join(output_dir, "subtask_injection", tag, "3d")
        makedirs(inj2, inj3)

        X_combined = torch.cat([X_gt, X_gen], dim=0)
        pca_label  = f"{tag}_inj_GT+Gen"
        X_pca, _   = run_pca(X_combined, cfg.probe_parameters.repr_pca_dims, pca_label, pca_dir)

        logging.info(f"    Fitting 2D UMAP for {tag} injection …")
        emb2 = run_umap(X_pca, 2, cfg.probe_parameters.umap_n_neighbors, cfg.probe_parameters.umap_min_dist,
                        cfg.probe_parameters.umap_seed)
        plot_2d_subtask_injection(emb2[:n], emb2[n:], metadata,
                                  os.path.join(inj2, "gen_vs_gt.png"))

        logging.info(f"    Fitting 3D UMAP for {tag} injection …")
        emb3 = run_umap(X_pca, 3, cfg.probe_parameters.umap_n_neighbors, cfg.probe_parameters.umap_min_dist,
                        cfg.probe_parameters.umap_seed)

        gen_texts = cache.get("gen_subtask_texts", [""] * n)
        hover = (
            [f"GT  | ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
             for m in metadata] +
            [f"GEN | ep={m['episode_idx']} fr={m['frame_idx']}<br>{gen_texts[i]}"
             for i, m in enumerate(metadata)]
        )
        condition_colors = [0] * n + [1] * n
        n_eps = len(set(m["episode_idx"] for m in metadata))
        _plotly_scatter3d_single(
            emb3, condition_colors, "0=GT / 1=Gen", hover,
            f"3D — {tag}: GT (blue) vs generated (red) subtask · {n} frames, {n_eps} episodes",
            os.path.join(inj3, "gen_vs_gt.html"),
            colorscale="Bluered",
        )

    if "prefix" in sites:
        _injection_plots("prefix", cache["prefix_gt"], cache["prefix_gen"])

    if "suffix" in sites and "suffix_gt" in cache:
        for t_val in t_values:
            if t_val not in cache["suffix_gt"] or t_val not in cache["suffix_gen"]:
                continue
            _injection_plots(
                f"suffix_t{t_val}",
                cache["suffix_gt"][t_val],
                cache["suffix_gen"][t_val],
            )


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _probe_one_dataset(policy, preprocessor, dataset, ds_dir, cfg, device):
    """
    Run the full sample → collect → plot pipeline for a single dataset.
    policy / preprocessor may be None when probe_mode == "plot".
    """
    p          = cfg.probe_parameters
    cache_path = os.path.join(ds_dir, "activations_cache.pt")
    makedirs(ds_dir)
    cache = None

    if p.mode in ("collect", "all"):
        logging.info("  Building sample list …")
        samples = sample_episodes_evenly(
            dataset,
            n_per_episode=p.n_frames_per_episode,
            max_episodes=p.max_episodes,
            seed=p.random_seed,
        )
        ep_to_indices         = _build_episode_index(dataset)
        sampled_eps           = {ep for ep, _, _ in samples}
        ep_to_indices_sampled = {ep: ep_to_indices[ep]
                                 for ep in sampled_eps if ep in ep_to_indices}
        _save_episode_thumbnails(dataset, ep_to_indices_sampled, ds_dir)

        logging.info(f"  Collecting activations for {len(samples)} frames …")
        cache = collect_activations(policy, preprocessor, dataset, samples, device, cfg)

        if p.subtask_injection:
            logging.info("  Collecting subtask injection activations …")
            inj = collect_subtask_injection(policy, preprocessor, dataset, samples, device, cfg)
            cache.update(inj)

        torch.save(cache, cache_path)
        logging.debug(f"  Activations saved → {cache_path}")

    if p.mode in ("plot", "all"):
        if cache is None:
            logging.info(f"  Loading cached activations from {cache_path} …")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logging.info("  Running PCA + UMAP and saving plots …")
        run_plotting(cache, cfg, ds_dir)
        logging.info(f"  Done. Plots in {ds_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: ProbeRepresentationsConfig):
    init_logging()
    p          = cfg.probe_parameters
    device     = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "representations")
    makedirs(output_dir)

    policy = preprocessor = primary_dataset = None
    if p.mode in ("collect", "all"):
        logging.info("Loading policy and primary dataset …")
        policy, preprocessor, _, primary_dataset = load_policy_and_processors(cfg, device)

    # ── Primary dataset ────────────────────────────────────────────────────────
    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    logging.info(f"=== Dataset: {primary_name} ===")
    _probe_one_dataset(policy, preprocessor, primary_dataset,
                       os.path.join(output_dir, primary_name), cfg, device)

    # ── Additional datasets ────────────────────────────────────────────────────
    extra_paths = getattr(cfg.dataset, "additional_offline_dataset_paths", None) or []
    for extra_root in extra_paths:
        ds_name  = os.path.basename(os.path.normpath(extra_root))
        logging.info(f"=== Dataset: {ds_name} ===")
        extra_ds = None
        if p.mode in ("collect", "all"):
            extra_ds = load_extra_dataset(cfg, extra_root)
        _probe_one_dataset(policy, preprocessor, extra_ds,
                           os.path.join(output_dir, ds_name), cfg, device)

    logging.info("All datasets done.")


if __name__ == "__main__":
    probe_cli()
