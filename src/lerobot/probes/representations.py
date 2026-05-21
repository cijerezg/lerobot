#!/usr/bin/env python
"""
Generic representation probe — visualise what the policy has learned by
mean-pooling per-layer hidden states, then PCA + UMAP for plotting.

Policy-agnostic: works with any policy whose ``ProbablePolicy`` adapter
implements ``capture_representations``. Site names are policy-specific (e.g.
pi05: ``prefix`` / ``suffix``; molmoact2: ``encoder`` / ``action_expert``) —
the probe iterates whatever sites the adapter returns.

Output layout (under ``probe_parameters.output_dir/representations/{dataset}/``):
  activations_cache.pt              reusable tensor cache (mode=plot skips collection)
  episode_thumbnails.png            first-frame thumbnails per episode
  pca_variance/                     PCA scree plots
  2d/<site>/by_episode.png          per-episode gradient (dark=early, light=late)
  2d/<site>/by_frame.png            all episodes pooled, coloured by frame index
  2d/<site>/by_subtask.png          all episodes pooled, coloured by subtask
  3d/<site>/by_episode.html         interactive 3D
  3d/<site>/by_frame.html
  3d/<site>/by_subtask.html
  3d/<site>/ep{A}_vs_ep{B}.html

Usage:
    python -m lerobot.probes.representations config.yaml \\
        --probe_parameters.max_episodes 5
"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import (
    EP_COLORS,
    SEQ_CMAPS,
    ax_style,
    build_episode_index,
    frame_colors_rgba,
    get_frame_data,
    get_subtask_idx,
    load_extra_dataset,
    makedirs,
    plotly_3d_layout,
    run_pca,
    run_umap,
    sample_episodes_evenly,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeRepresentationsConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters`` (ProbeConfig)."""


# Sequential colormaps for representation plots.
_REP_SEQ_CMAPS = ["Blues", "Reds", "Greens", "Oranges", "Purples",
                  "YlOrBr", "PuRd", "BuGn", "GnBu", "OrRd"]


# ──────────────────────────────────────────────────────────────────────────────
# Collection — loop over samples, ask the adapter for per-site vectors
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_activations(adapter: ProbablePolicy, dataset, samples, cfg):
    """For each sampled frame, call ``adapter.capture_representations`` and stack
    the resulting per-site vectors. Returns:

        {"<site_name>": Tensor[N, hidden_dim], ..., "metadata": list[dict]}
    """
    chunk_size = adapter.chunk_size
    timestep = float(getattr(cfg.probe_parameters, "timestep", 1.0))

    per_site: dict[str, list[torch.Tensor]] = {}
    metadata: list[dict] = []

    for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
        if i % 100 == 0:
            logging.debug(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

        obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size,
        )
        reps = adapter.capture_representations(
            obs, task_str, state=state, timestep=timestep,
            gt_actions=gt_actions, gt_subtask=gt_subtask,
        )
        for site, vec in reps.items():
            per_site.setdefault(site, []).append(vec)

        metadata.append({
            "episode_idx": ep_idx,
            "frame_idx":   fr_idx,
            "global_idx":  global_idx,
            "subtask":     gt_subtask,
            "subtask_idx": get_subtask_idx(dataset, global_idx),
            "task":        task_str,
        })

    result: dict = {"metadata": metadata}
    for site, vecs in per_site.items():
        result[site] = torch.stack(vecs, dim=0)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_2d_by_episode(emb, metadata, output_path):
    from matplotlib.patches import Patch

    ep_ids    = np.array([m["episode_idx"] for m in metadata])
    frame_ids = np.array([m["frame_idx"]   for m in metadata])
    unique_eps = np.unique(ep_ids)

    fig, ax = plt.subplots(figsize=(7, 6))
    legend_handles = []
    for i, ep in enumerate(unique_eps):
        mask      = ep_ids == ep
        ep_frames = frame_ids[mask]
        fmin, fmax = ep_frames.min(), ep_frames.max()
        norm  = 0.9 - (ep_frames - fmin) / max(fmax - fmin, 1) * 0.6
        cmap  = matplotlib.colormaps.get_cmap(_REP_SEQ_CMAPS[i % len(_REP_SEQ_CMAPS)])
        ax.scatter(emb[mask, 0], emb[mask, 1], c=cmap(norm),
                   s=22, alpha=0.85, linewidths=0)
        legend_handles.append(Patch(facecolor=cmap(0.6), label=f"ep {ep}"))

    if len(unique_eps) <= 20:
        ax.legend(handles=legend_handles, fontsize=6, ncol=2)
    ax_style(ax, f"By episode ({len(unique_eps)} eps) — dark→light = early→late frame",
             width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_frame(emb, metadata, output_path):
    frame_ids = np.array([m["frame_idx"] for m in metadata])
    n_eps = len(np.unique([m["episode_idx"] for m in metadata]))
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=frame_ids, cmap="plasma",
                    s=22, alpha=0.80, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Frame index within episode")
    ax_style(ax, f"By frame index — {n_eps} episodes pooled", width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_subtask(emb, metadata, output_path):
    sub_ids     = np.array([m["subtask_idx"] for m in metadata])
    unique_subs = np.unique(sub_ids)
    sub_text    = {m["subtask_idx"]: m["subtask"] for m in metadata}
    cmap        = matplotlib.colormaps.get_cmap("tab20")
    n_eps       = len(np.unique([m["episode_idx"] for m in metadata]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, s in enumerate(unique_subs):
        mask = sub_ids == s
        lbl = sub_text.get(s, str(s))
        if len(lbl) > 32:
            lbl = lbl[:30] + "…"
        ax.scatter(emb[mask, 0], emb[mask, 1], s=22, alpha=0.80, linewidths=0,
                   color=cmap(i), label=f"[{s}] {lbl}")
    ax.legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_style(ax, f"By subtask — {n_eps} episodes pooled", width=55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3D plots
# ──────────────────────────────────────────────────────────────────────────────

def _plotly_3d(emb, traces, title, output_path):
    import plotly.graph_objects as go  # noqa: F401  (consumed by caller traces)
    import plotly.graph_objects as go
    fig = go.Figure(data=traces)
    fig.update_layout(**plotly_3d_layout(title))
    fig.write_html(output_path)


def plot_3d_by_episode(emb, metadata, output_path):
    import plotly.graph_objects as go

    ep_ids     = [m["episode_idx"] for m in metadata]
    frame_ids  = [m["frame_idx"]   for m in metadata]
    unique_eps = sorted(set(ep_ids))

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
            mode="markers", name=f"ep {ep}",
            marker=dict(size=6, color=colors, line=dict(width=0)),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ))
    _plotly_3d(emb, traces,
               f"3D — by episode ({len(unique_eps)} eps) · dark→pale = early→late",
               output_path)


def plot_3d_by_frame(emb, metadata, output_path):
    import plotly.graph_objects as go

    frame_ids = [m["frame_idx"] for m in metadata]
    n_eps = len(set(m["episode_idx"] for m in metadata))
    hover = [f"ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
             for m in metadata]

    traces = [go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode="markers",
        marker=dict(size=6, color=frame_ids, colorscale="Plasma",
                    showscale=True, opacity=0.85, line=dict(width=0),
                    colorbar=dict(title="Frame index within episode")),
        text=hover, hovertemplate="%{text}<extra></extra>",
    )]
    _plotly_3d(emb, traces, f"3D — by frame index — {n_eps} episodes pooled", output_path)


def plot_3d_by_subtask(emb, metadata, output_path):
    import plotly.graph_objects as go

    sub_ids = [m["subtask_idx"] for m in metadata]
    n_eps = len(set(m["episode_idx"] for m in metadata))
    hover = [f"ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
             for m in metadata]
    traces = [go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode="markers",
        marker=dict(size=6, color=sub_ids, colorscale="Rainbow",
                    showscale=True, opacity=0.85, line=dict(width=0),
                    colorbar=dict(title="Subtask ID")),
        text=hover, hovertemplate="%{text}<extra></extra>",
    )]
    _plotly_3d(emb, traces, f"3D — by subtask ({n_eps} eps)", output_path)


def plot_3d_two_episodes(emb, metadata, ep_a, ep_b, output_path):
    import plotly.graph_objects as go

    ep_marker = {ep_a: "circle", ep_b: "square"}
    traces = []
    for ep, sym in ep_marker.items():
        idx = [i for i, m in enumerate(metadata) if m["episode_idx"] == ep]
        if not idx:
            continue
        sub = emb[idx]
        frames = [metadata[i]["frame_idx"] for i in idx]
        hover = [f"ep={ep} fr={metadata[i]['frame_idx']}<br>{metadata[i]['subtask']}"
                 for i in idx]
        traces.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
            mode="markers", name=f"Episode {ep}",
            marker=dict(size=6, color=frames, colorscale="Plasma",
                        showscale=(ep == ep_a), opacity=0.85, line=dict(width=0),
                        colorbar=dict(title="Frame index"), symbol=sym),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ))
    _plotly_3d(emb, traces,
               f"3D — ep {ep_a} (●) vs ep {ep_b} (■) · coloured by frame",
               output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Episode thumbnails
# ──────────────────────────────────────────────────────────────────────────────

def _grid_layout(n: int) -> tuple[int, int]:
    rows = max(1, math.floor(math.sqrt(n)))
    cols = math.ceil(n / rows)
    return rows, cols


def _save_episode_thumbnails(dataset, ep_to_indices, output_dir):
    sample_frame = dataset[0]
    img_keys = sorted(
        k for k in sample_frame.keys()
        if k.startswith("observation.image") and isinstance(sample_frame[k], torch.Tensor)
    )
    if not img_keys:
        logging.warning("No image keys found; skipping episode thumbnails.")
        return

    episodes = sorted(ep_to_indices.keys())
    n_eps = len(episodes)
    n_cams = len(img_keys)
    cam_labels = [k.replace("observation.images.", "").replace("observation.image.", "")
                  for k in img_keys]
    rows, cols = (_grid_layout(n_eps) if n_cams == 1 else (n_eps, n_cams))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    for i, ep_idx in enumerate(episodes):
        frame = dataset[ep_to_indices[ep_idx][0]]
        r, c_start = (i, 0) if n_cams > 1 else divmod(i, cols)
        for cam_j, (key, label) in enumerate(zip(img_keys, cam_labels)):
            c = c_start + cam_j if n_cams == 1 else cam_j
            ax = axes[r][c]
            img = frame[key]
            img_np = img.float().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8) \
                     if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            ax.imshow(img_np)
            ax.axis("off")
            ax.set_title(
                f"ep {ep_idx}" if n_cams == 1 else f"ep {ep_idx} · {label}",
                fontsize=7,
            )

    if n_cams == 1:
        for j in range(n_eps, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].set_visible(False)

    fig.suptitle(f"Episode thumbnails ({n_eps} episodes)", fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(output_dir, "episode_thumbnails.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _run_site(tag, X, metadata, cfg, pca_dir, output_dir):
    """PCA → 2D UMAP → 3D UMAP → all standard plots for one activation tensor."""
    p = cfg.probe_parameters
    logging.info(f"  Site '{tag}'  shape={tuple(X.shape)}")

    d2 = os.path.join(output_dir, "2d", tag)
    d3 = os.path.join(output_dir, "3d", tag)
    makedirs(d2, d3)

    X_pca, _ = run_pca(X, p.repr_pca_dims, tag, pca_dir)

    logging.info("    Fitting 2D UMAP …")
    emb2 = run_umap(X_pca, 2, p.umap_n_neighbors, p.umap_min_dist, p.umap_seed)
    plot_2d_by_episode(emb2, metadata, os.path.join(d2, "by_episode.png"))
    plot_2d_by_frame(emb2, metadata,   os.path.join(d2, "by_frame.png"))
    plot_2d_by_subtask(emb2, metadata, os.path.join(d2, "by_subtask.png"))

    logging.info("    Fitting 3D UMAP …")
    emb3 = run_umap(X_pca, 3, p.umap_n_neighbors, p.umap_min_dist, p.umap_seed)
    plot_3d_by_episode(emb3, metadata, os.path.join(d3, "by_episode.html"))
    plot_3d_by_frame(emb3, metadata,   os.path.join(d3, "by_frame.html"))
    plot_3d_by_subtask(emb3, metadata, os.path.join(d3, "by_subtask.html"))
    plot_3d_two_episodes(
        emb3, metadata, p.ep_3d_a, p.ep_3d_b,
        os.path.join(d3, f"ep{p.ep_3d_a}_vs_ep{p.ep_3d_b}.html"),
    )


def run_plotting(cache, cfg, output_dir):
    metadata = cache["metadata"]
    pca_dir = os.path.join(output_dir, "pca_variance")
    makedirs(pca_dir)

    # Iterate every site the adapter produced. Order is whatever the adapter
    # returned (Python dict insertion order).
    for site, X in cache.items():
        if site == "metadata":
            continue
        _run_site(site, X, metadata, cfg, pca_dir, output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _probe_one_dataset(adapter, dataset, ds_dir, cfg):
    """Run sample → collect → plot for one dataset."""
    p = cfg.probe_parameters
    makedirs(ds_dir)
    cache_path = os.path.join(ds_dir, "activations_cache.pt")
    cache = None

    if p.mode in ("collect", "all"):
        if adapter is None or dataset is None:
            raise ValueError("collect mode requires both adapter and dataset.")
        logging.info("  Building sample list …")
        samples = sample_episodes_evenly(
            dataset,
            n_per_episode=p.n_frames_per_episode,
            max_episodes=p.max_episodes,
            seed=p.random_seed,
        )
        ep_to_indices = build_episode_index(dataset)
        sampled_eps = {ep for ep, _, _ in samples}
        _save_episode_thumbnails(
            dataset,
            {ep: ep_to_indices[ep] for ep in sampled_eps if ep in ep_to_indices},
            ds_dir,
        )

        logging.info(f"  Collecting activations for {len(samples)} frames …")
        cache = collect_activations(adapter, dataset, samples, cfg)
        torch.save(cache, cache_path)
        logging.debug(f"  Activations saved → {cache_path}")

    if p.mode in ("plot", "all"):
        if cache is None:
            logging.info(f"  Loading cache from {cache_path} …")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logging.info("  Running PCA + UMAP …")
        run_plotting(cache, cfg, ds_dir)
        logging.info(f"  Done. Plots in {ds_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(adapter, primary_dataset, cfg, output_dir):
    """Run the representations probe on primary + additional datasets.

    ``adapter`` and ``primary_dataset`` may be ``None`` only if
    ``cfg.probe_parameters.mode == "plot"`` (re-plot from cache).
    """
    p = cfg.probe_parameters
    makedirs(output_dir)

    _probe_one_dataset(adapter, primary_dataset, output_dir, cfg)

    for extra_root in getattr(cfg.dataset, "additional_offline_dataset_paths", None) or []:
        ds_name = os.path.basename(os.path.normpath(extra_root))
        logging.info(f"=== Dataset: {ds_name} ===")
        extra_ds = None
        if p.mode in ("collect", "all"):
            extra_ds = load_extra_dataset(cfg.dataset.repo_id, extra_root)
        _probe_one_dataset(adapter, extra_ds,
                           os.path.join(output_dir, ds_name), cfg)

    logging.info("All datasets done.")


@parser.wrap()
def probe_cli(cfg: ProbeRepresentationsConfig):
    init_logging()
    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "representations")

    adapter = primary_dataset = None
    if p.mode in ("collect", "all"):
        from lerobot.datasets.factory import make_dataset
        primary_dataset = make_dataset(cfg)
        primary_dataset.delta_timestamps = None
        primary_dataset.delta_indices = None
        logging.info("Loading policy adapter …")
        adapter = ProbablePolicy.for_config(cfg, device, dataset=primary_dataset)

    run(adapter, primary_dataset, cfg, output_dir)


if __name__ == "__main__":
    probe_cli()
