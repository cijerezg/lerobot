#!/usr/bin/env python
"""
PI05 action probe — project GT and predicted actions onto the GT action manifold.

Two-phase pipeline:

  Phase 1 — Reference manifold (root dataset, GT only, no model inference):
    Sample all frames from root → flatten action chunks → fit PCA + UMAP.
    This defines the action manifold from ground-truth demonstrations.
    Cost: only parquet reads + PCA/UMAP fitting. Fast even for all root frames.

  Phase 2 — Evaluation (all datasets, model inference, ~4-5 episodes each):
    For each dataset (root + additional_offline_dataset_paths), sample a few
    episodes, run inference, project GT and pred through the fitted
    PCA.transform() → UMAP.transform(). The reference GT is shown as a grey
    background in every plot so you can see where each dataset's actions land.

Computational cost:
  Reference GT collection:  ~10 ms/frame — 40 eps × 256 frames = 10240 frames (~2 min).
  PCA fit:                  seconds.
  UMAP fit (N×50):          ~15 s at N=2k. Fit once.
  UMAP.transform():         ~50 ms per batch of 1 k frames.
  Model inference:          ~1 s/frame — the bottleneck; keep probe_max_episodes small.

Output layout (all under probe_parameters.output_dir/actions/):
  actions_cache.pt                    reusable cache (--probe_parameters.mode plot skips collection)
  pca_variance/                       PCA scree plot (gt_reference)
  2d/overview.png                     all datasets' GT overlaid on reference manifold
  3d/overview.html                    interactive version of overview
  2d/{ds}/trajectories.png            GT paths + pred paths + grey ref background
  2d/{ds}/by_frame.png                GT and pred coloured by frame index
  2d/{ds}/by_subtask.png              GT and pred coloured by subtask
  2d/{ds}/episodes/ep{N:04d}.png      per-episode GT vs pred
  3d/{ds}/by_episode.html             interactive per-episode scatter (dark→pale = early→late)
  3d/{ds}/by_frame.html               interactive by frame index
  3d/{ds}/by_subtask.html             interactive by subtask

Usage:
    python probe_actions_pi05.py config-hiserl.json
    python probe_actions_pi05.py config-hiserl.json --probe_parameters.output_dir outputs/probe
    python probe_actions_pi05.py config-hiserl.json --probe_parameters.mode plot
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.scripts.eval_offline_pi05 import get_frame_data, run_inference
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.rl.probe_utils_pi05 import (
    DS_COLORS,
    EP_COLORS,
    SEQ_CMAPS,
    ax_style,
    frame_colors_rgba,
    load_extra_dataset,
    load_policy_and_processors,
    makedirs,
    plotly_3d_layout,
    ref_bg_trace_3d,
    run_pca,
    sample_episodes_evenly,
    get_subtask_idx,
)


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

# Phase 1 — reference manifold from root GT (no inference, cheap).
REF_MAX_EPISODES = 40
REF_N_FRAMES_PER_EPISODE = 256

# Phase 2 — evaluation with model inference.
MAX_EPISODES = 5
N_FRAMES_PER_EPISODE = 128
RANDOM_SEED = 42

PCA_DIMS = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeActionsConfig(TrainRLServerPipelineConfig):
    """Extends the base training config with action probe parameters.

    All probe tunables live under cfg.probe_parameters (ProbeConfig).
    Relevant fields for this script:
      output_dir, mode, max_episodes, n_frames_per_episode, random_seed,
      ref_max_episodes, ref_n_frames_per_episode,
      pca_dims, umap_n_neighbors, umap_min_dist, umap_seed.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — reference GT collection (no model)
# ──────────────────────────────────────────────────────────────────────────────

def collect_gt_reference(dataset, samples, chunk_size):
    """
    Read GT action chunks from the root dataset (no model needed).
    Returns {"gt": (N, chunk_size*action_dim) tensor, "metadata": list}.
    """
    all_vecs = []
    metadata = []

    for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
        if i % 500 == 0:
            logging.info(f"  [{i + 1}/{len(samples)}] ref ep={ep_idx:04d} fr={fr_idx:04d}")

        _, gt_actions, _, gt_subtask, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size
        )
        all_vecs.append(gt_actions.flatten().float())

        metadata.append({
            "episode_idx": ep_idx,
            "frame_idx":   fr_idx,
            "global_idx":  global_idx,
            "subtask":     gt_subtask,
            "subtask_idx": get_subtask_idx(dataset, global_idx),
        })

    return {"gt": torch.stack(all_vecs), "metadata": metadata}


# ──────────────────────────────────────────────────────────────────────────────
# Manifold fitting — PCA then UMAP on root GT
# ──────────────────────────────────────────────────────────────────────────────

def fit_manifold(X_ref, cfg, pca_dir):
    """
    Fit PCA then 2D and 3D UMAP on root GT vectors.

    Uses reducer.fit() so that reducer.transform() is available for out-of-sample
    projection. The exact training-set embedding is taken from reducer.embedding_
    (not transform(), which uses an approximation).

    Returns (pca, reducer2d, reducer3d, ref_emb2, ref_emb3).
    """
    import warnings
    import umap as umap_lib

    p = cfg.probe_parameters
    X_pca, pca = run_pca(X_ref, p.pca_dims, "gt_reference", pca_dir)
    X_np = X_pca.numpy()

    def _fit_umap(n_components, label):
        logging.info(f"  Fitting {label} UMAP on {X_np.shape[0]} reference frames …")
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=p.umap_n_neighbors,
            min_dist=p.umap_min_dist,
            random_state=p.umap_seed,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="n_jobs value.*overridden",
                                    category=UserWarning)
            reducer.fit(X_np)
        return reducer

    reducer2d = _fit_umap(2, "2D")
    reducer3d = _fit_umap(3, "3D")

    return pca, reducer2d, reducer3d, reducer2d.embedding_, reducer3d.embedding_


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — evaluation collection (with model)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_eval_dataset(policy, preprocessor, postprocessor,
                          dataset, samples, pca, reducer2d, reducer3d, device, cfg):
    """
    Run inference and project GT + pred through the fitted manifold.

    For each frame:
      1. Flatten GT and pred action chunks → (1, chunk_size*action_dim) numpy arrays.
      2. pca.transform() → (1, pca_dims).
      3. Batch all frames, then reducer.transform() → (N, 2) and (N, 3).

    Returns dict with "gt_emb2", "pred_emb2", "gt_emb3", "pred_emb3", "metadata".
    """
    import warnings

    chunk_size    = cfg.policy.chunk_size
    gt_pca_rows   = []
    pred_pca_rows = []
    metadata      = []

    policy.model.suppress_debug_log = True
    for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
        if i % 100 == 0:
            logging.info(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

        obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size
        )
        pred_unnorm, _, _ = run_inference(
            policy, preprocessor, postprocessor,
            obs, task_str, device,
            state=state, advantage=1.0,
        )

        gt_pca_rows.append(
            pca.transform(gt_actions.flatten().float().numpy().reshape(1, -1))[0]
        )
        pred_pca_rows.append(
            pca.transform(pred_unnorm.flatten().float().numpy().reshape(1, -1))[0]
        )

        metadata.append({
            "episode_idx": ep_idx,
            "frame_idx":   fr_idx,
            "global_idx":  global_idx,
            "subtask":     gt_subtask,
            "subtask_idx": get_subtask_idx(dataset, global_idx),
            "task":        task_str,
        })

    policy.model.suppress_debug_log = False

    gt_pca   = np.stack(gt_pca_rows)
    pred_pca = np.stack(pred_pca_rows)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden",
                                category=UserWarning)
        gt_emb2   = reducer2d.transform(gt_pca)
        pred_emb2 = reducer2d.transform(pred_pca)
        gt_emb3   = reducer3d.transform(gt_pca)
        pred_emb3 = reducer3d.transform(pred_pca)

    return {
        "gt_emb2":   gt_emb2,
        "pred_emb2": pred_emb2,
        "gt_emb3":   gt_emb3,
        "pred_emb3": pred_emb3,
        "metadata":  metadata,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2D plot helpers (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_ref_bg(ax, ref_emb2):
    """Grey dots for the reference GT manifold (background layer)."""
    ax.scatter(ref_emb2[:, 0], ref_emb2[:, 1],
               s=5, color="#cccccc", alpha=0.35, linewidths=0, zorder=1)


def _gradient_scatter(ax, x, y, cmap_name, s=18, alpha=0.75, marker="o", zorder=2):
    """Fading scatter (no connecting lines): dark=early, light=late."""
    t = np.linspace(0.30, 0.90, len(x))
    ax.scatter(x, y, c=t, cmap=cmap_name, vmin=0.0, vmax=1.0,
               s=s, alpha=alpha, marker=marker, linewidths=0, zorder=zorder)


def _gradient_path(ax, x, y, cmap_name, linewidth=1.5, alpha=0.85, zorder=2):
    """Solid LineCollection with gradient colour (dark=early, light=late)."""
    from matplotlib.collections import LineCollection
    if len(x) < 2:
        ax.scatter(x, y, s=20, zorder=zorder)
        return
    pts  = np.stack([x, y], axis=1)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t    = np.linspace(0.25, 0.85, len(segs))
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    lc   = LineCollection(segs, colors=cmap(t), linewidth=linewidth,
                          alpha=alpha, zorder=zorder)
    ax.add_collection(lc)


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots — per dataset
# ──────────────────────────────────────────────────────────────────────────────

def plot_2d_trajectories(ref_emb2, gt_emb2, pred_emb2, metadata, output_path, ds_name):
    """
    All episodes on the reference manifold.
    GT: thick gradient line (dark→light = early→late).
    Pred: small semi-transparent dots in the same episode colour.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ep_ids     = np.array([m["episode_idx"] for m in metadata])
    fr_ids     = np.array([m["frame_idx"]   for m in metadata])
    unique_eps = np.unique(ep_ids)

    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_ref_bg(ax, ref_emb2)

    legend_handles = []
    for i, ep in enumerate(unique_eps):
        mask  = ep_ids == ep
        idx   = np.where(mask)[0]
        order = np.argsort(fr_ids[idx])
        xs_gt, ys_gt = gt_emb2  [idx[order], 0], gt_emb2  [idx[order], 1]
        xs_pr, ys_pr = pred_emb2[idx[order], 0], pred_emb2[idx[order], 1]
        cmap_name = SEQ_CMAPS[i % len(SEQ_CMAPS)]
        col       = EP_COLORS[i % len(EP_COLORS)]

        _gradient_path(ax, xs_gt, ys_gt, cmap_name, linewidth=2.0, zorder=3)
        ax.scatter(xs_pr, ys_pr, color=col, s=20, alpha=0.75,
                   marker="o", linewidths=0, zorder=4)
        ax.scatter(xs_gt[0],  ys_gt[0],  marker="^", s=70, color=col,
                   zorder=10, linewidths=0)
        ax.scatter(xs_gt[-1], ys_gt[-1], marker="s", s=50, color=col,
                   zorder=10, linewidths=0)
        legend_handles.append(Patch(facecolor=col, label=f"ep {ep}"))

    if len(unique_eps) <= 20:
        enc = [
            Line2D([0], [0], color="#555", linewidth=2.0,
                   label="GT  (gradient line, dark→light)"),
            Line2D([0], [0], color="#555", marker="o", linestyle="None",
                   markersize=5, alpha=0.5, label="Pred  (dots, same colour as GT)"),
        ]
        ax.legend(handles=legend_handles + enc, fontsize=7, ncol=2)
    ax.autoscale_view()
    ax_style(ax, f"{ds_name}  (line=GT · dots=pred · same colour = same episode · ▲start ■end)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_frame(ref_emb2, gt_emb2, pred_emb2, metadata, output_path, ds_name):
    fr_ids = np.array([m["frame_idx"] for m in metadata])
    n_eps  = len(np.unique([m["episode_idx"] for m in metadata]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, label in [(axes[0], gt_emb2, "GT"), (axes[1], pred_emb2, "Predicted")]:
        _draw_ref_bg(ax, ref_emb2)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=fr_ids, cmap="plasma",
                        s=18, alpha=0.85, linewidths=0, zorder=2)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Frame index")
        ax_style(ax, f"{ds_name} — {label} by frame index  ({n_eps} eps)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_subtask(ref_emb2, gt_emb2, pred_emb2, metadata, output_path, ds_name):
    sub_ids     = np.array([m["subtask_idx"] for m in metadata])
    unique_subs = np.unique(sub_ids)
    sub_text    = {m["subtask_idx"]: m["subtask"] for m in metadata}
    cmap        = matplotlib.colormaps.get_cmap("tab20")
    n_eps       = len(np.unique([m["episode_idx"] for m in metadata]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, emb, label in [(axes[0], gt_emb2, "GT"), (axes[1], pred_emb2, "Predicted")]:
        _draw_ref_bg(ax, ref_emb2)
        for i, s in enumerate(unique_subs):
            mask = sub_ids == s
            lbl  = sub_text.get(s, str(s))
            if len(lbl) > 32:
                lbl = lbl[:30] + "…"
            ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.85, linewidths=0,
                       color=cmap(i % 20), label=f"[{s}] {lbl}", zorder=2)
        ax.legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax_style(ax, f"{ds_name} — {label} by subtask  ({n_eps} eps)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_episode(ref_emb2, gt_emb_ep, pred_emb_ep, meta_ep,
                    ep_idx, ds_name, output_path):
    """Single episode: GT (solid blue gradient) + pred (orange dots)."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fr = np.array([m["frame_idx"] for m in meta_ep])

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_ref_bg(ax, ref_emb2)

    x_gt, y_gt = gt_emb_ep[:, 0], gt_emb_ep[:, 1]
    if len(x_gt) >= 2:
        pts  = np.stack([x_gt, y_gt], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        t    = np.linspace(0.20, 0.88, len(segs))
        cmap = matplotlib.colormaps.get_cmap("Blues")
        lc   = LineCollection(segs, colors=cmap(t), linewidth=2.0, alpha=0.90, zorder=3)
        ax.add_collection(lc)
    ax.scatter(x_gt, y_gt, c=np.linspace(0, 1, len(x_gt)), cmap="Blues",
               s=20, alpha=0.8, zorder=4, linewidths=0)

    x_pr, y_pr = pred_emb_ep[:, 0], pred_emb_ep[:, 1]
    _gradient_scatter(ax, x_pr, y_pr, "Oranges", s=28, alpha=0.85, marker="o", zorder=5)

    ax.scatter(x_gt[0],  y_gt[0],  marker="^", s=120, color="#2ca02c",
               zorder=8, edgecolors="white", linewidths=0.8)
    ax.scatter(x_gt[-1], y_gt[-1], marker="s", s=80,  color="#d62728",
               zorder=8, edgecolors="white", linewidths=0.8)
    ax.annotate(f"fr{fr[0]}",  (x_gt[0],  y_gt[0]),  textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#2ca02c")
    ax.annotate(f"fr{fr[-1]}", (x_gt[-1], y_gt[-1]), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#d62728")

    legend_handles = [
        Line2D([0], [0], color=matplotlib.colormaps.get_cmap("Blues")(0.6),
               linewidth=2, label="GT (dark→light)"),
        Line2D([0], [0], color=matplotlib.colormaps.get_cmap("Oranges")(0.7),
               marker="o", linestyle="None", markersize=7,
               label="Predicted (dark→light)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.85)
    ax.autoscale_view()
    ax_style(ax, f"{ds_name} — ep {ep_idx}  ({len(fr)} frames)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"    ep {ep_idx:04d}  {len(fr)} frames → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2D overview — all datasets on the reference manifold
# ──────────────────────────────────────────────────────────────────────────────

def plot_2d_overview(ref_emb2, datasets_cache, output_path):
    """All datasets' GT scatter overlaid on reference manifold."""
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_ref_bg(ax, ref_emb2)

    legend_handles = [Patch(facecolor="#cccccc", label="ref GT")]
    for i, (ds_name, ds_data) in enumerate(datasets_cache.items()):
        col = DS_COLORS[i % len(DS_COLORS)]
        emb = ds_data["gt_emb2"]
        ax.scatter(emb[:, 0], emb[:, 1], s=14, color=col,
                   alpha=0.75, linewidths=0, zorder=2 + i)
        legend_handles.append(Patch(facecolor=col, label=ds_name))

    ax.legend(handles=legend_handles, fontsize=7)
    ax.autoscale_view()
    ax_style(ax, "Overview — all datasets GT on reference manifold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  2D overview → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3D plots — per dataset
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_by_episode(ref_emb3, gt_emb3, pred_emb3, metadata, output_path, ds_name):
    """Per-episode scatter: pred circles, GT squares; sequential colormap per episode (dark=early, pale=late)."""
    import plotly.graph_objects as go

    ep_ids     = [m["episode_idx"] for m in metadata]
    fr_ids     = [m["frame_idx"]   for m in metadata]
    unique_eps = sorted(set(ep_ids))

    traces = [ref_bg_trace_3d(ref_emb3)]
    for i, ep in enumerate(unique_eps):
        idx    = [j for j, e in enumerate(ep_ids) if e == ep]
        frames = np.array([fr_ids[j] for j in idx])
        colors = frame_colors_rgba(frames, SEQ_CMAPS[i % len(SEQ_CMAPS)], alpha=0.85)

        hover_gt   = [f"GT   ep={ep} fr={fr_ids[j]}<br>{metadata[j]['subtask']}" for j in idx]
        hover_pred = [f"pred ep={ep} fr={fr_ids[j]}<br>{metadata[j]['subtask']}" for j in idx]

        # GT → square, smaller
        traces.append(go.Scatter3d(
            x=[gt_emb3[j, 0] for j in idx],
            y=[gt_emb3[j, 1] for j in idx],
            z=[gt_emb3[j, 2] for j in idx],
            mode="markers", name=f"ep {ep} GT",
            legendgroup=f"ep{ep}", showlegend=False,
            marker=dict(size=3, symbol="square", color=colors,
                        opacity=0.85, line=dict(width=0)),
            text=hover_gt, hovertemplate="%{text}<extra></extra>",
        ))
        # Predicted → circle, main focus
        traces.append(go.Scatter3d(
            x=[pred_emb3[j, 0] for j in idx],
            y=[pred_emb3[j, 1] for j in idx],
            z=[pred_emb3[j, 2] for j in idx],
            mode="markers", name=f"ep {ep}",
            legendgroup=f"ep{ep}", showlegend=True,
            marker=dict(size=6, symbol="circle", color=colors, line=dict(width=0)),
            text=hover_pred, hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**plotly_3d_layout(
        f"{ds_name} — by episode  ●=pred  ■=GT  dark→pale=early→late"
    ))
    fig.write_html(output_path)
    logging.info(f"    3D by episode → {output_path}")


def plot_3d_by_frame(ref_emb3, gt_emb3, pred_emb3, metadata, output_path, ds_name):
    """All frames coloured by frame index (Plasma); GT=circle, pred=cross."""
    import plotly.graph_objects as go

    fr_ids     = [m["frame_idx"] for m in metadata]
    n_eps      = len(set(m["episode_idx"] for m in metadata))
    hover_gt   = [f"GT   ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
                  for m in metadata]
    hover_pred = [f"pred ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
                  for m in metadata]

    traces = [
        ref_bg_trace_3d(ref_emb3),
        go.Scatter3d(
            x=gt_emb3[:, 0], y=gt_emb3[:, 1], z=gt_emb3[:, 2],
            mode="markers", name="GT",
            marker=dict(size=6, symbol="circle", color=fr_ids, colorscale="Plasma",
                        showscale=True, opacity=0.85,
                        colorbar=dict(title="Frame index", thickness=16, len=0.65),
                        line=dict(width=0)),
            text=hover_gt, hovertemplate="%{text}<extra></extra>",
        ),
        go.Scatter3d(
            x=pred_emb3[:, 0], y=pred_emb3[:, 1], z=pred_emb3[:, 2],
            mode="markers", name="Predicted",
            marker=dict(size=4, symbol="cross", color=fr_ids, colorscale="Plasma",
                        showscale=False, opacity=0.75, line=dict(width=0)),
            text=hover_pred, hovertemplate="%{text}<extra></extra>",
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(**plotly_3d_layout(
        f"{ds_name} — by frame index  (●=GT  ✕=pred)  {n_eps} eps"
    ))
    fig.write_html(output_path)
    logging.info(f"    3D by frame → {output_path}")


def plot_3d_by_subtask(ref_emb3, gt_emb3, pred_emb3, metadata, output_path, ds_name):
    """Categorical colours per subtask; GT=circle (legend), pred=cross (same group)."""
    import plotly.graph_objects as go

    sub_ids     = [m["subtask_idx"] for m in metadata]
    sub_text    = {m["subtask_idx"]: m["subtask"] for m in metadata}
    unique_subs = sorted(set(sub_ids))
    n_eps       = len(set(m["episode_idx"] for m in metadata))

    traces = [ref_bg_trace_3d(ref_emb3)]
    for i, s in enumerate(unique_subs):
        mask = [j for j, sid in enumerate(sub_ids) if sid == s]
        col  = EP_COLORS[i % len(EP_COLORS)]
        lbl  = sub_text.get(s, str(s))
        if len(lbl) > 35:
            lbl = lbl[:33] + "…"

        hover_gt   = [f"GT   ep={metadata[j]['episode_idx']} fr={metadata[j]['frame_idx']}<br>{lbl}"
                      for j in mask]
        hover_pred = [f"pred ep={metadata[j]['episode_idx']} fr={metadata[j]['frame_idx']}<br>{lbl}"
                      for j in mask]

        traces.append(go.Scatter3d(
            x=[gt_emb3[j, 0] for j in mask],
            y=[gt_emb3[j, 1] for j in mask],
            z=[gt_emb3[j, 2] for j in mask],
            mode="markers", name=f"[{s}] {lbl}",
            legendgroup=f"sub{s}", showlegend=True,
            marker=dict(size=6, symbol="circle", color=col, opacity=0.85, line=dict(width=0)),
            text=hover_gt, hovertemplate="%{text}<extra></extra>",
        ))
        traces.append(go.Scatter3d(
            x=[pred_emb3[j, 0] for j in mask],
            y=[pred_emb3[j, 1] for j in mask],
            z=[pred_emb3[j, 2] for j in mask],
            mode="markers", name=f"[{s}] {lbl} pred",
            legendgroup=f"sub{s}", showlegend=False,
            marker=dict(size=4, symbol="cross", color=col, opacity=0.70, line=dict(width=0)),
            text=hover_pred, hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**plotly_3d_layout(
        f"{ds_name} — by subtask  (●=GT  ✕=pred)  {n_eps} eps"
    ))
    fig.write_html(output_path)
    logging.info(f"    3D by subtask → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3D overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_overview(ref_emb3, datasets_cache, output_path):
    import plotly.graph_objects as go

    traces = [ref_bg_trace_3d(ref_emb3)]
    for i, (ds_name, ds_data) in enumerate(datasets_cache.items()):
        col  = DS_COLORS[i % len(DS_COLORS)]
        emb3 = ds_data["gt_emb3"]
        meta = ds_data["metadata"]
        hover = [f"{ds_name} | ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
                 for m in meta]
        traces.append(go.Scatter3d(
            x=emb3[:, 0], y=emb3[:, 1], z=emb3[:, 2],
            mode="markers", name=ds_name,
            marker=dict(size=3, color=col, opacity=0.80, line=dict(width=0)),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**plotly_3d_layout("Overview — all datasets GT on reference manifold"))
    fig.write_html(output_path)
    logging.info(f"  3D overview → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# NN-distance metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_nn_distances(query_emb, ref_emb):
    """L2 distance from each query point to its nearest neighbor in ref_emb (2D UMAP)."""
    from scipy.spatial import cKDTree
    tree  = cKDTree(ref_emb)
    dists, _ = tree.query(query_emb, k=1, workers=-1)
    return dists


def plot_nn_distances(ref_emb2, gt_emb2, pred_emb2, metadata, output_dir, ds_name):
    """
    Per-episode NN-distance histograms (GT blue vs Pred orange, overlaid).
    Each subplot = one episode. Also saves a CSV summary.
    """
    import csv
    from matplotlib.patches import Patch

    ep_ids     = np.array([m["episode_idx"] for m in metadata])
    unique_eps = sorted(np.unique(ep_ids).tolist())

    gt_nn   = compute_nn_distances(gt_emb2,   ref_emb2)
    pred_nn = compute_nn_distances(pred_emb2, ref_emb2)

    ep_gt_dists   = {ep: gt_nn  [ep_ids == ep] for ep in unique_eps}
    ep_pred_dists = {ep: pred_nn[ep_ids == ep] for ep in unique_eps}

    n_eps  = len(unique_eps)
    ncols  = min(n_eps, 5)
    nrows  = (n_eps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.0),
                             squeeze=False)

    all_vals = np.concatenate(list(ep_gt_dists.values()) + list(ep_pred_dists.values()))
    x_max    = float(np.percentile(all_vals, 98))
    bins     = np.linspace(0, x_max, 30)

    for idx, ep in enumerate(unique_eps):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        ax.hist(ep_gt_dists  [ep], bins=bins, color="#4477aa", alpha=0.55,
                density=True, label="GT")
        ax.hist(ep_pred_dists[ep], bins=bins, color="#dd7733", alpha=0.55,
                density=True, label="Pred")
        med_gt   = np.median(ep_gt_dists  [ep])
        med_pred = np.median(ep_pred_dists[ep])
        ax.axvline(med_gt,   color="#223366", linewidth=1.2, linestyle="--")
        ax.axvline(med_pred, color="#883300", linewidth=1.2, linestyle="--")
        ax.set_title(f"ep {ep}   med GT={med_gt:.3f}  pred={med_pred:.3f}", fontsize=8)
        ax.set_xlabel("NN dist (2D UMAP)", fontsize=7)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("Density", fontsize=7)
        if idx == 0:
            ax.legend(handles=[
                Patch(facecolor="#4477aa", alpha=0.7, label="GT"),
                Patch(facecolor="#dd7733", alpha=0.7, label="Pred"),
            ], fontsize=7)

    for idx in range(n_eps, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{ds_name} — NN distance to reference GT manifold  (dashed = median)",
                 fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(output_dir, "nn_distances.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"    NN distances → {out_png}")

    out_csv = os.path.join(output_dir, "nn_distances.csv")
    rows = []
    for ep in unique_eps:
        for label, dists in [("gt", ep_gt_dists[ep]), ("pred", ep_pred_dists[ep])]:
            rows.append({
                "dataset": ds_name, "episode": ep, "type": label,
                "n":       len(dists),
                "mean":    float(np.mean(dists)),
                "median":  float(np.median(dists)),
                "std":     float(np.std(dists)),
                "p25":     float(np.percentile(dists, 25)),
                "p75":     float(np.percentile(dists, 75)),
                "p95":     float(np.percentile(dists, 95)),
                "max":     float(np.max(dists)),
            })
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"    NN distances CSV → {out_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_plotting(cache, cfg, output_dir):
    ref_emb2 = cache["ref_emb2"]
    ref_emb3 = cache["ref_emb3"]

    d2 = os.path.join(output_dir, "2d")
    d3 = os.path.join(output_dir, "3d")
    makedirs(d2, d3)

    for ds_name, ds_data in cache["datasets"].items():
        logging.info(f"  Plotting '{ds_name}' …")
        meta      = ds_data["metadata"]
        gt_emb2   = ds_data["gt_emb2"]
        pred_emb2 = ds_data["pred_emb2"]
        gt_emb3   = ds_data["gt_emb3"]
        pred_emb3 = ds_data["pred_emb3"]

        ds_dir2 = os.path.join(d2, ds_name)
        ds_dir3 = os.path.join(d3, ds_name)
        ep_dir  = os.path.join(ds_dir2, "episodes")
        makedirs(ds_dir2, ds_dir3, ep_dir)

        plot_2d_trajectories(ref_emb2, gt_emb2, pred_emb2, meta,
                             os.path.join(ds_dir2, "trajectories.png"), ds_name)
        plot_2d_by_frame(ref_emb2, gt_emb2, pred_emb2, meta,
                         os.path.join(ds_dir2, "by_frame.png"), ds_name)
        plot_2d_by_subtask(ref_emb2, gt_emb2, pred_emb2, meta,
                           os.path.join(ds_dir2, "by_subtask.png"), ds_name)

        ep_ids = np.array([m["episode_idx"] for m in meta])
        fr_ids = np.array([m["frame_idx"]   for m in meta])
        for ep in np.unique(ep_ids):
            idx      = np.where(ep_ids == ep)[0]
            meta_ep  = [meta[i] for i in idx]
            fr_order = np.argsort([m["frame_idx"] for m in meta_ep])
            plot_2d_episode(
                ref_emb2,
                gt_emb2  [idx[fr_order]],
                pred_emb2[idx[fr_order]],
                [meta_ep[i] for i in fr_order],
                ep, ds_name,
                os.path.join(ep_dir, f"ep{ep:04d}.png"),
            )

        plot_3d_by_episode(ref_emb3, gt_emb3, pred_emb3, meta,
                           os.path.join(ds_dir3, "by_episode.html"), ds_name)
        plot_3d_by_frame(ref_emb3, gt_emb3, pred_emb3, meta,
                         os.path.join(ds_dir3, "by_frame.html"), ds_name)
        plot_3d_by_subtask(ref_emb3, gt_emb3, pred_emb3, meta,
                           os.path.join(ds_dir3, "by_subtask.html"), ds_name)

        plot_nn_distances(ref_emb2, gt_emb2, pred_emb2, meta, ds_dir2, ds_name)

    plot_2d_overview(ref_emb2, cache["datasets"], os.path.join(d2, "overview.png"))
    plot_3d_overview(ref_emb3, cache["datasets"], os.path.join(d3, "overview.html"))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: ProbeActionsConfig):
    init_logging()
    p = cfg.probe_parameters
    output_dir = os.path.join(p.output_dir, "actions")
    makedirs(output_dir)
    cache_path = os.path.join(output_dir, "actions_cache.pt")
    cache = None

    if p.mode in ("collect", "all"):
        device  = get_safe_torch_device(try_device=cfg.policy.device)
        pca_dir = os.path.join(output_dir, "pca_variance")
        makedirs(pca_dir)

        # ── Phase 1: reference manifold from root GT (no model) ───────────────
        logging.info("Loading root dataset …")
        from lerobot.datasets.factory import make_dataset
        root_dataset = make_dataset(cfg)
        root_dataset.delta_timestamps = None
        root_dataset.delta_indices    = None
        root_name = os.path.basename(os.path.normpath(cfg.dataset.root))

        ref_samples = sample_episodes_evenly(
            root_dataset,
            n_per_episode=p.ref_n_frames_per_episode,
            max_episodes=p.ref_max_episodes,
            seed=p.random_seed,
        )
        logging.info(f"Collecting reference GT: {len(ref_samples)} frames from '{root_name}' …")
        ref_data = collect_gt_reference(root_dataset, ref_samples, cfg.policy.chunk_size)

        logging.info("Fitting PCA + UMAP manifold …")
        pca, reducer2d, reducer3d, ref_emb2, ref_emb3 = fit_manifold(
            ref_data["gt"], cfg, pca_dir
        )

        cache = {
            "pca":          pca,
            "reducer2d":    reducer2d,
            "reducer3d":    reducer3d,
            "ref_emb2":     ref_emb2,
            "ref_emb3":     ref_emb3,
            "ref_metadata": ref_data["metadata"],
            "datasets":     {},
        }

        # ── Phase 2: evaluate all datasets with model ─────────────────────────
        logging.info("Loading policy …")
        policy, preprocessor, postprocessor, _ = load_policy_and_processors(
            cfg, device, dataset=root_dataset
        )

        extra_paths = getattr(cfg.dataset, "additional_offline_dataset_paths", None) or []
        all_datasets = [(root_name, root_dataset)] + [
            (os.path.basename(os.path.normpath(ep)), load_extra_dataset(cfg, ep))
            for ep in extra_paths
        ]

        for ds_name, dataset in all_datasets:
            logging.info(f"=== Evaluating '{ds_name}' ===")
            eval_samples = sample_episodes_evenly(
                dataset,
                n_per_episode=p.n_frames_per_episode,
                max_episodes=p.max_episodes,
                seed=p.random_seed,
            )
            logging.info(f"  {len(eval_samples)} frames")
            cache["datasets"][ds_name] = collect_eval_dataset(
                policy, preprocessor, postprocessor,
                dataset, eval_samples,
                pca, reducer2d, reducer3d, device, cfg,
            )

        torch.save(cache, cache_path)
        logging.info(f"Cache saved → {cache_path}")

    if p.mode in ("plot", "all"):
        if cache is None:
            logging.info(f"Loading cache from {cache_path} …")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logging.info("Generating plots …")
        run_plotting(cache, cfg, output_dir)
        logging.info(f"Done. Plots in {output_dir}/")


if __name__ == "__main__":
    probe_cli()
