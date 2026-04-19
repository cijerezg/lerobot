#!/usr/bin/env python
"""
PI05 action probe — visualise predicted and GT action trajectories.

For each sampled frame we run model inference and collect the predicted action
chunk (chunk_size × 6), plus the corresponding ground-truth chunk.  Both are
flattened to (chunk_size × 6)-dim vectors, embedded jointly in PCA → UMAP, and
plotted as connected episode trajectories.

The joint embedding lets you compare the manifold structure of model predictions
against GT: do the predicted paths follow the same regions? Do they track episode
structure?

Actions are unnormalised (actual joint positions) using the same postprocessor
path as eval_offline_pi05.py.

Output layout (all under --probe_output_dir):
  actions_cache.pt              reusable cache (--probe_mode plot skips inference)
  pca_variance/                 PCA scree plots
  2d/all_trajectories.png       all-episode overlay — GT (solid) and pred (dashed)
  2d/by_frame.png               all frames, coloured by frame index
  2d/by_subtask.png             all frames, coloured by subtask
  2d/episodes/ep{N:04d}.png     per-episode GT vs pred trajectory
  3d/by_episode.html            interactive, per-episode colours, dark→light = early→late
  3d/by_frame.html              interactive, coloured by frame index
  3d/by_subtask.html            interactive, coloured by subtask (categorical)

Usage:
    python probe_actions_pi05.py config-hiserl.json
    python probe_actions_pi05.py config-hiserl.json --probe_output_dir outputs/probe_actions
    python probe_actions_pi05.py config-hiserl.json --probe_mode plot
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

import lerobot.rl.rl_pi05  # noqa: F401 — registers PI05RLConfig

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.scripts.eval_offline_pi05 import (
    _build_episode_index,
    get_frame_data,
    run_inference,
)


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

MAX_EPISODES = 8
N_FRAMES_PER_EPISODE = 2000   # clips to episode length → effectively all frames
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
    """Extends the base training config with action probe parameters."""

    probe_output_dir: str = "outputs/probe_actions"
    # "collect" → run inference and save cache
    # "plot"    → load existing cache and generate all plots
    # "all"     → collect then plot
    probe_mode: str = "all"

    # Data sampling
    probe_max_episodes: Optional[int] = MAX_EPISODES
    probe_n_frames_per_episode: int = N_FRAMES_PER_EPISODE
    probe_random_seed: int = RANDOM_SEED

    # Dimensionality reduction
    probe_pca_dims: int = PCA_DIMS
    probe_umap_n_neighbors: int = UMAP_N_NEIGHBORS
    probe_umap_min_dist: float = UMAP_MIN_DIST
    probe_umap_seed: int = UMAP_SEED


# ──────────────────────────────────────────────────────────────────────────────
# Policy / dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_policy_and_processors(cfg, device, dataset=None):
    if dataset is None:
        dataset = make_dataset(cfg)
        dataset.delta_timestamps = None
        dataset.delta_indices = None

    preprocessor, postprocessor = make_pi05_full_processors_with_upgrade(
        cfg, dataset=dataset, is_main_process=True
    )

    original_device = cfg.policy.device
    cfg.policy.device = device.type
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    cfg.policy.device = original_device

    policy.preprocessor = preprocessor
    policy.postprocessor = postprocessor
    policy.eval()
    policy.to(device)

    return policy, preprocessor, postprocessor, dataset


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_episodes(dataset, n_per_episode, max_episodes, seed):
    """
    Sample up to n_per_episode evenly spaced frames per episode.
    With n_per_episode=2000 this returns all frames for typical episode lengths.
    Returns list of (episode_idx, frame_idx_in_ep, global_idx).
    """
    ep_to_indices = _build_episode_index(dataset)
    episodes = sorted(ep_to_indices.keys())
    if max_episodes is not None:
        rng = np.random.RandomState(seed)
        episodes = sorted(
            rng.choice(episodes, size=min(max_episodes, len(episodes)), replace=False).tolist()
        )

    samples = []
    for ep_idx in episodes:
        indices = ep_to_indices[ep_idx]
        n = min(n_per_episode, len(indices))
        chosen = np.linspace(0, len(indices) - 1, n, dtype=int)
        for pos in chosen:
            global_idx = indices[pos]
            fr_idx = dataset.hf_dataset[global_idx]["frame_index"].item()
            samples.append((ep_idx, fr_idx, global_idx))

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_actions(policy, preprocessor, postprocessor, dataset, samples, device, cfg):
    """
    Run inference on every sample and collect predicted + GT action chunks.

    Both pred and GT are unnormalised absolute joint positions (same space as
    the robot), obtained via run_inference / get_frame_data respectively.

    Returns dict:
      "pred":     (N, chunk_size * action_dim) float32
      "gt":       (N, chunk_size * action_dim) float32
      "metadata": list of N dicts
    """
    chunk_size = cfg.policy.chunk_size

    all_pred = []
    all_gt   = []
    metadata = []

    policy.model.suppress_debug_log = True
    for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
        if i % 100 == 0:
            logging.info(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

        obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
            dataset, global_idx, chunk_size
        )

        # Predicted actions — unnormalised, absolute (run_inference handles anchor/delta)
        pred_unnorm, _, _ = run_inference(
            policy, preprocessor, postprocessor,
            obs, task_str, device,
            state=state, advantage=1.0,
        )
        all_pred.append(pred_unnorm.flatten().float().cpu())

        # GT actions — get_frame_data already returns unnormalised absolute positions
        all_gt.append(gt_actions.flatten().float().cpu())

        subtask_idx = -1
        frame_row = dataset.hf_dataset[global_idx]
        for key in ("subtask_index", "complementary_info.subtask_index"):
            if key in frame_row:
                val = frame_row[key]
                subtask_idx = val.item() if isinstance(val, torch.Tensor) else int(val)
                break

        metadata.append({
            "episode_idx": ep_idx,
            "frame_idx":   fr_idx,
            "global_idx":  global_idx,
            "subtask":     gt_subtask,
            "subtask_idx": subtask_idx,
            "task":        task_str,
        })

    policy.model.suppress_debug_log = False
    return {
        "pred":     torch.stack(all_pred),   # (N, D)
        "gt":       torch.stack(all_gt),     # (N, D)
        "metadata": metadata,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PCA + UMAP
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(X, n_components, label, pca_dir):
    from sklearn.decomposition import PCA

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca   = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X.numpy())

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    comp90 = int(np.searchsorted(cumvar, 0.90)) + 1
    comp95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_,
                color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"{label} — per-component variance")

    axes[1].plot(range(1, n_components + 1), cumvar, color="steelblue", linewidth=1.5)
    axes[1].axhline(0.90, color="#888", linestyle="--", linewidth=0.9, label=f"90% @ {comp90}")
    axes[1].axhline(0.95, color="orange", linestyle="--", linewidth=0.9, label=f"95% @ {comp95}")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title(f"{label} — cumulative variance")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.02)

    fig.suptitle(
        f"PCA scree — {label}  ({X.shape[0]} samples, {X.shape[1]} dims → top {n_components})",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(pca_dir, f"{label}_pca_scree.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"    PCA scree → {out}  (90% @ {comp90}, 95% @ {comp95})")

    return torch.from_numpy(X_pca.astype(np.float32)), pca


def run_umap(X_pca, n_components, n_neighbors, min_dist, seed):
    import warnings
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)
        return reducer.fit_transform(X_pca.numpy())


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots
# ──────────────────────────────────────────────────────────────────────────────

_EP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#a9a9a9", "#ffd8b1", "#fabed4", "#fffac8", "#e0e0e0",
]
_SEQ_CMAPS = ["Blues", "Reds", "Greens", "Oranges", "Purples",
              "YlOrBr", "PuRd", "BuGn", "GnBu", "OrRd"]


def _ax_style(ax, title):
    import textwrap
    ax.set_title(textwrap.fill(title, width=60), fontsize=9)
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


def _gradient_path(ax, x, y, cmap_name, linewidth=1.5, alpha=0.85, zorder=2):
    """Solid LineCollection with gradient colour along the path (dark=early, light=late)."""
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


def plot_2d_all_trajectories(emb2_gt, emb2_pred, metadata, output_path):
    """
    All episodes overlaid.  GT = solid gradient path, pred = dashed gradient path.
    Same colour family per episode; ▲=start, ■=end on GT path.
    """
    from matplotlib.patches import Patch

    ep_ids     = np.array([m["episode_idx"] for m in metadata])
    fr_ids     = np.array([m["frame_idx"]   for m in metadata])
    unique_eps = np.unique(ep_ids)

    fig, ax = plt.subplots(figsize=(9, 8))
    legend_handles = []
    for i, ep in enumerate(unique_eps):
        mask  = ep_ids == ep
        idx   = np.where(mask)[0]
        order = np.argsort(fr_ids[idx])
        xs_gt, ys_gt = emb2_gt  [idx[order], 0], emb2_gt  [idx[order], 1]
        xs_pr, ys_pr = emb2_pred[idx[order], 0], emb2_pred[idx[order], 1]
        cmap_name = _SEQ_CMAPS[i % len(_SEQ_CMAPS)]
        col       = _EP_COLORS [i % len(_EP_COLORS)]

        # GT: gradient path (dark=early, light=late); pred: flat dashed line in episode colour
        _gradient_path(ax, xs_gt, ys_gt, cmap_name, linewidth=1.8, zorder=3)
        ax.plot(xs_pr, ys_pr, color=col, linewidth=1.0, linestyle="--", alpha=0.60, zorder=2)
        ax.scatter(xs_gt[0],  ys_gt[0],  marker="^", s=70, color=col, zorder=10, linewidths=0)
        ax.scatter(xs_gt[-1], ys_gt[-1], marker="s", s=50, color=col, zorder=10, linewidths=0)
        legend_handles.append(Patch(facecolor=col, label=f"ep {ep}"))

    if len(unique_eps) <= 20:
        ax.legend(handles=legend_handles, fontsize=7, ncol=2)
    ax.autoscale_view()
    _ax_style(ax, f"Action trajectories — {len(unique_eps)} eps  (solid=GT, dashed=pred, ▲=start, ■=end)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_frame(emb2_gt, emb2_pred, metadata, output_path):
    """GT (●) and pred (×) coloured by frame index."""
    fr_ids = np.array([m["frame_idx"] for m in metadata])
    n_eps  = len(np.unique([m["episode_idx"] for m in metadata]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, label in [(axes[0], emb2_gt, "GT"), (axes[1], emb2_pred, "Predicted")]:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=fr_ids, cmap="plasma",
                        s=18, alpha=0.80, linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Frame index")
        _ax_style(ax, f"{label} — coloured by frame index  ({n_eps} eps)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_subtask(emb2_gt, emb2_pred, metadata, output_path):
    """GT (●) and pred (×) coloured by subtask index."""
    sub_ids     = np.array([m["subtask_idx"] for m in metadata])
    unique_subs = np.unique(sub_ids)
    sub_text    = {m["subtask_idx"]: m["subtask"] for m in metadata}
    cmap        = matplotlib.colormaps.get_cmap("tab20")
    n_eps       = len(np.unique([m["episode_idx"] for m in metadata]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, emb, label in [(axes[0], emb2_gt, "GT"), (axes[1], emb2_pred, "Predicted")]:
        for i, s in enumerate(unique_subs):
            mask  = sub_ids == s
            lbl   = sub_text.get(s, str(s))
            if len(lbl) > 32:
                lbl = lbl[:30] + "…"
            ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.80, linewidths=0,
                       color=cmap(i % 20), label=f"[{s}] {lbl}")
        ax.legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        _ax_style(ax, f"{label} — coloured by subtask  ({n_eps} eps)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_episode(emb_gt, emb_pred, meta_ep, ep_idx, output_path):
    """
    Single episode: GT (solid blue gradient) vs pred (flat dashed orange).
    Both sorted by frame index.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    fr = np.array([m["frame_idx"] for m in meta_ep])

    fig, ax = plt.subplots(figsize=(7, 7))

    # GT: gradient LineCollection (solid blue, dark=early → light=late)
    x_gt, y_gt = emb_gt[:, 0], emb_gt[:, 1]
    if len(x_gt) >= 2:
        pts  = np.stack([x_gt, y_gt], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        t    = np.linspace(0.20, 0.88, len(segs))
        cmap = matplotlib.colormaps.get_cmap("Blues")
        lc   = LineCollection(segs, colors=cmap(t), linewidth=2.0, alpha=0.90, zorder=3)
        ax.add_collection(lc)
    ax.scatter(x_gt, y_gt, c=np.linspace(0, 1, len(x_gt)), cmap="Blues",
               s=20, alpha=0.8, zorder=4, linewidths=0)

    # Pred: flat dashed line (orange), plain ax.plot so dashes render properly
    x_pr, y_pr = emb_pred[:, 0], emb_pred[:, 1]
    ax.plot(x_pr, y_pr, color="#f58231", linewidth=1.4, linestyle="--", alpha=0.75, zorder=2)
    ax.scatter(x_pr, y_pr, color="#f58231", s=16, alpha=0.65, zorder=2, linewidths=0)

    # Start / end markers on GT path
    x0, y0 = x_gt[0], y_gt[0]
    xn, yn = x_gt[-1], y_gt[-1]
    ax.scatter(x0, y0, marker="^", s=120, color="#2ca02c", zorder=8,
               edgecolors="white", linewidths=0.8)
    ax.scatter(xn, yn, marker="s", s=80,  color="#d62728", zorder=8,
               edgecolors="white", linewidths=0.8)
    ax.annotate(f"fr{fr[0]}",  (x0, y0), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#2ca02c")
    ax.annotate(f"fr{fr[-1]}", (xn, yn), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#d62728")

    legend_handles = [
        Line2D([0], [0], color=matplotlib.colormaps.get_cmap("Blues")(0.6),
               linewidth=2, label="GT"),
        Line2D([0], [0], color=matplotlib.colormaps.get_cmap("Oranges")(0.6),
               linewidth=1.4, linestyle="--", label="Predicted"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.85)
    ax.autoscale_view()
    _ax_style(ax, f"Episode {ep_idx} — GT vs predicted actions  ({len(fr)} frames)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"    ep {ep_idx:04d}  {len(fr)} frames → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3D plots (Plotly interactive HTML)
# ──────────────────────────────────────────────────────────────────────────────

def _episode_colorscale(hex_col):
    """Plotly [[0, dark], [1, light]] colorscale for a given base hex colour."""
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    dark  = f"rgb({max(r - 100, 0)},{max(g - 100, 0)},{max(b - 100, 0)})"
    light = f"rgb({min(r + 80, 255)},{min(g + 80, 255)},{min(b + 80, 255)})"
    return [[0.0, dark], [1.0, light]]


def _3d_layout(title, n_eps=None):
    """Base Plotly layout for all 3D plots."""
    full_title = title if n_eps is None else f"{title}  ({n_eps} eps)"
    return dict(
        title=dict(text=full_title, font=dict(size=14)),
        scene=dict(
            xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            zaxis=dict(title="UMAP-3", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            bgcolor="white",
            aspectmode="auto",
        ),
        paper_bgcolor="white",
        legend=dict(
            font=dict(size=12),
            itemsizing="constant",
            tracegroupgap=2,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        margin=dict(l=0, r=160, b=0, t=55),
        height=720,
    )


def plot_3d_by_episode(emb3_gt, emb3_pred, metadata, output_path):
    """
    3D: episodes as connected traces, GT solid / pred dashed.
    Each episode gets a distinct colour; markers shade dark→light (early→late).
    Legend shows one entry per episode; GT/pred distinguished by line style + marker symbol.
    """
    import plotly.graph_objects as go

    ep_ids     = [m["episode_idx"] for m in metadata]
    fr_ids     = [m["frame_idx"]   for m in metadata]
    unique_eps = sorted(set(ep_ids))

    traces = []
    for i, ep in enumerate(unique_eps):
        idx = sorted(
            [j for j, e in enumerate(ep_ids) if e == ep],
            key=lambda j: fr_ids[j],
        )
        col = _EP_COLORS[i % len(_EP_COLORS)]
        cs  = _episode_colorscale(col)

        fr_arr  = np.array([fr_ids[j] for j in idx], dtype=float)
        fr_norm = (fr_arr - fr_arr.min()) / max(fr_arr.max() - fr_arr.min(), 1.0)

        for emb3, suffix, dash, lw, sym, mksize, show_leg in [
            (emb3_gt,   "GT",   "solid", 4, "circle", 5, True),
            (emb3_pred, "pred", "dash",  2, "cross",  4, False),
        ]:
            hover = [
                f"ep={ep} fr={fr_ids[j]} [{suffix}]<br>{metadata[j]['subtask']}"
                for j in idx
            ]
            traces.append(go.Scatter3d(
                x=[emb3[j, 0] for j in idx],
                y=[emb3[j, 1] for j in idx],
                z=[emb3[j, 2] for j in idx],
                mode="lines+markers",
                name=f"ep {ep}",
                legendgroup=f"ep{ep}",
                showlegend=show_leg,
                line=dict(color=col, width=lw, dash=dash),
                marker=dict(
                    size=mksize,
                    symbol=sym,
                    color=fr_norm,
                    colorscale=cs,
                    showscale=False,
                    line=dict(width=0),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_3d_layout(
        "3D action trajectories — solid=GT  dashed=pred  dark→light=early→late",
        n_eps=len(unique_eps),
    ))
    fig.write_html(output_path)
    logging.info(f"    3D by episode → {output_path}")


def plot_3d_by_frame(emb3_gt, emb3_pred, metadata, output_path):
    """3D: all frames coloured by frame index (Plasma), GT=circle, pred=cross."""
    import plotly.graph_objects as go

    fr_ids = [m["frame_idx"] for m in metadata]
    n_eps  = len(set(m["episode_idx"] for m in metadata))
    hover_gt   = [f"GT   ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
                  for m in metadata]
    hover_pred = [f"pred ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"
                  for m in metadata]

    traces = [
        go.Scatter3d(
            x=emb3_gt[:, 0], y=emb3_gt[:, 1], z=emb3_gt[:, 2],
            mode="markers", name="GT",
            marker=dict(
                size=5, symbol="circle",
                color=fr_ids, colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Frame index", thickness=16, len=0.65, x=1.02),
                line=dict(width=0),
            ),
            text=hover_gt, hovertemplate="%{text}<extra></extra>",
        ),
        go.Scatter3d(
            x=emb3_pred[:, 0], y=emb3_pred[:, 1], z=emb3_pred[:, 2],
            mode="markers", name="Predicted",
            marker=dict(
                size=4, symbol="cross",
                color=fr_ids, colorscale="Plasma",
                showscale=False,
                line=dict(width=0),
            ),
            text=hover_pred, hovertemplate="%{text}<extra></extra>",
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(**_3d_layout(
        "3D — coloured by frame index  (●=GT, ✕=pred)",
        n_eps=n_eps,
    ))
    fig.write_html(output_path)
    logging.info(f"    3D by frame → {output_path}")


def plot_3d_by_subtask(emb3_gt, emb3_pred, metadata, output_path):
    """3D: categorical colours per subtask (GT=circle, pred=cross), named legend entries."""
    import plotly.graph_objects as go

    sub_ids     = [m["subtask_idx"] for m in metadata]
    sub_text    = {m["subtask_idx"]: m["subtask"] for m in metadata}
    unique_subs = sorted(set(sub_ids))
    n_eps       = len(set(m["episode_idx"] for m in metadata))

    traces = []
    for i, s in enumerate(unique_subs):
        mask = [j for j, sid in enumerate(sub_ids) if sid == s]
        col  = _EP_COLORS[i % len(_EP_COLORS)]
        lbl  = sub_text.get(s, str(s))
        if len(lbl) > 35:
            lbl = lbl[:33] + "…"

        hover_gt   = [f"GT   ep={metadata[j]['episode_idx']} fr={metadata[j]['frame_idx']}<br>{lbl}"
                      for j in mask]
        hover_pred = [f"pred ep={metadata[j]['episode_idx']} fr={metadata[j]['frame_idx']}<br>{lbl}"
                      for j in mask]

        # GT circles — shown in legend
        traces.append(go.Scatter3d(
            x=[emb3_gt[j, 0] for j in mask],
            y=[emb3_gt[j, 1] for j in mask],
            z=[emb3_gt[j, 2] for j in mask],
            mode="markers",
            name=f"[{s}] {lbl}",
            legendgroup=f"sub{s}",
            showlegend=True,
            marker=dict(size=5, symbol="circle", color=col, line=dict(width=0)),
            text=hover_gt, hovertemplate="%{text}<extra></extra>",
        ))
        # Pred crosses — same group, hidden from legend
        traces.append(go.Scatter3d(
            x=[emb3_pred[j, 0] for j in mask],
            y=[emb3_pred[j, 1] for j in mask],
            z=[emb3_pred[j, 2] for j in mask],
            mode="markers",
            name=f"[{s}] {lbl}",
            legendgroup=f"sub{s}",
            showlegend=False,
            marker=dict(size=4, symbol="cross", color=col, line=dict(width=0)),
            text=hover_pred, hovertemplate="%{text}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(**_3d_layout(
        "3D — coloured by subtask  (●=GT, ✕=pred)",
        n_eps=n_eps,
    ))
    fig.write_html(output_path)
    logging.info(f"    3D by subtask → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_plotting(cache, cfg, output_dir):
    """PCA → joint UMAP embedding for GT and pred, then all plots."""
    metadata = cache["metadata"]
    X_gt   = cache["gt"]    # (N, D)
    X_pred = cache["pred"]  # (N, D)
    N = len(metadata)

    pca_dir = os.path.join(output_dir, "pca_variance")
    d2      = os.path.join(output_dir, "2d")
    d3      = os.path.join(output_dir, "3d")
    d2_eps  = os.path.join(d2, "episodes")
    for p in (pca_dir, d2, d3, d2_eps):
        os.makedirs(p, exist_ok=True)

    n_eps = len(set(m["episode_idx"] for m in metadata))
    logging.info(f"  GT/pred action vectors: {tuple(X_gt.shape)}  ({n_eps} episodes)")

    # ── PCA: fit on GT+pred combined so both live in the same low-dim space ───
    X_combined = torch.cat([X_gt, X_pred], dim=0)  # (2N, D)
    X_pca, _   = run_pca(X_combined, cfg.probe_pca_dims, "actions_gt+pred", pca_dir)

    # ── 2D UMAP ───────────────────────────────────────────────────────────────
    logging.info("  Fitting 2D UMAP …")
    emb2      = run_umap(X_pca, 2, cfg.probe_umap_n_neighbors, cfg.probe_umap_min_dist, cfg.probe_umap_seed)
    emb2_gt   = emb2[:N]
    emb2_pred = emb2[N:]

    plot_2d_all_trajectories(emb2_gt, emb2_pred, metadata,
                             os.path.join(d2, "all_trajectories.png"))
    plot_2d_by_frame(emb2_gt, emb2_pred, metadata, os.path.join(d2, "by_frame.png"))
    plot_2d_by_subtask(emb2_gt, emb2_pred, metadata, os.path.join(d2, "by_subtask.png"))

    ep_ids = np.array([m["episode_idx"] for m in metadata])
    for ep in np.unique(ep_ids):
        idx      = np.where(ep_ids == ep)[0]
        meta_ep  = [metadata[i] for i in idx]
        fr_order = np.argsort([m["frame_idx"] for m in meta_ep])
        meta_sorted = [meta_ep[i] for i in fr_order]
        emb_gt_ep   = emb2_gt  [idx[fr_order]]
        emb_pred_ep = emb2_pred[idx[fr_order]]
        plot_2d_episode(emb_gt_ep, emb_pred_ep, meta_sorted, ep,
                        os.path.join(d2_eps, f"ep{ep:04d}.png"))
    logging.info(f"  Per-episode plots → {d2_eps}/")

    # ── 3D UMAP ───────────────────────────────────────────────────────────────
    logging.info("  Fitting 3D UMAP …")
    emb3      = run_umap(X_pca, 3, cfg.probe_umap_n_neighbors, cfg.probe_umap_min_dist, cfg.probe_umap_seed)
    emb3_gt   = emb3[:N]
    emb3_pred = emb3[N:]

    plot_3d_by_episode(emb3_gt, emb3_pred, metadata,
                       os.path.join(d3, "by_episode.html"))
    plot_3d_by_frame(emb3_gt, emb3_pred, metadata,
                     os.path.join(d3, "by_frame.html"))
    plot_3d_by_subtask(emb3_gt, emb3_pred, metadata,
                       os.path.join(d3, "by_subtask.html"))


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _probe_one_dataset(policy, preprocessor, postprocessor, dataset, ds_dir, cfg, device):
    """
    Run inference → collect → plot for one dataset.
    policy/preprocessor/postprocessor may be None when probe_mode == "plot".
    """
    cache_path = os.path.join(ds_dir, "actions_cache.pt")
    os.makedirs(ds_dir, exist_ok=True)
    cache = None

    if cfg.probe_mode in ("collect", "all"):
        logging.info("  Building sample list …")
        samples = _sample_episodes(
            dataset,
            n_per_episode=cfg.probe_n_frames_per_episode,
            max_episodes=cfg.probe_max_episodes,
            seed=cfg.probe_random_seed,
        )
        logging.info(f"  Running inference for {len(samples)} frames …")
        cache = collect_actions(
            policy, preprocessor, postprocessor, dataset, samples, device, cfg
        )
        torch.save(cache, cache_path)
        logging.info(f"  Cache saved → {cache_path}")

    if cfg.probe_mode in ("plot", "all"):
        if cache is None:
            logging.info(f"  Loading cache from {cache_path} …")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logging.info("  Running PCA + UMAP and saving plots …")
        run_plotting(cache, cfg, ds_dir)
        logging.info(f"  Done. Plots in {ds_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def probe_cli(cfg: ProbeActionsConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    init_logging()
    output_dir = cfg.probe_output_dir
    os.makedirs(output_dir, exist_ok=True)

    policy = preprocessor = postprocessor = primary_dataset = None
    if cfg.probe_mode in ("collect", "all"):
        device = get_safe_torch_device(try_device=cfg.policy.device)
        logging.info("Loading policy and primary dataset …")
        policy, preprocessor, postprocessor, primary_dataset = _load_policy_and_processors(
            cfg, device
        )
    else:
        device = torch.device("cpu")

    # ── Primary dataset ────────────────────────────────────────────────────────
    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    logging.info(f"=== Dataset: {primary_name} ===")
    _probe_one_dataset(
        policy, preprocessor, postprocessor, primary_dataset,
        os.path.join(output_dir, primary_name),
        cfg, device,
    )

    # ── Additional datasets ────────────────────────────────────────────────────
    extra_paths = getattr(cfg.dataset, "additional_offline_dataset_paths", None) or []
    for extra_root in extra_paths:
        ds_name = os.path.basename(os.path.normpath(extra_root))
        logging.info(f"=== Dataset: {ds_name} ===")
        extra_ds = None
        if cfg.probe_mode in ("collect", "all"):
            extra_ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=extra_root)
            extra_ds.delta_timestamps = None
            extra_ds.delta_indices    = None
        _probe_one_dataset(
            policy, preprocessor, postprocessor, extra_ds,
            os.path.join(output_dir, ds_name),
            cfg, device,
        )

    logging.info("All datasets done.")


if __name__ == "__main__":
    probe_cli()
