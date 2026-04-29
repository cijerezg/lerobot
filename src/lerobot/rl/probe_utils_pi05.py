"""
Shared utilities for PI05 probe scripts (probe_actions, probe_representations, probe_attention).

Provides:
  - Policy / dataset loading helpers
  - Episode sampling
  - Subtask index lookup
  - PCA + UMAP wrappers
  - Matplotlib 2D style helpers
  - Plotly 3D layout helpers
  - Colour palettes
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Side-effect imports — register configs and hardware backends.
import lerobot.rl.rl_pi05  # noqa: F401 — registers PI05RLConfig
from lerobot.cameras import opencv  # noqa: F401
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: F401 — re-exported
from lerobot.utils.utils import init_logging  # noqa: F401 — re-exported


# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

EP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#a9a9a9", "#ffd8b1", "#fabed4", "#fffac8", "#e0e0e0",
]

# Sequential colormaps cycled per episode — maximally distinct hues (up to 10).
SEQ_CMAPS = ["Reds",   "Blues",   "Greens",  "Oranges", "Purples",
             "copper", "cool",    "spring",  "winter",  "autumn"]

DS_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
             "#42d4f4", "#f032e6", "#bfef45", "#469990"]


# ──────────────────────────────────────────────────────────────────────────────
# Policy / dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_policy_and_processors(cfg, device, dataset=None):
    """
    Load policy + pre/post-processors.

    If *dataset* is None it is loaded from *cfg* (first call). Pass an already-loaded
    dataset on subsequent calls to avoid redundant disk reads.

    Returns (policy, preprocessor, postprocessor, dataset).
    """
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


def load_extra_dataset(cfg, root):
    """Load an additional LeRobot dataset from a local *root* directory."""
    ds = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=root)
    ds.delta_timestamps = None
    ds.delta_indices = None
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Episode / frame sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_episodes_evenly(dataset, n_per_episode, max_episodes, seed):
    """
    Sample *n_per_episode* evenly-spaced frames from each episode.

    If *max_episodes* is not None, a reproducible random subset of episodes is
    drawn using *seed*.

    Returns list of (episode_idx, frame_idx_in_episode, global_idx).
    """
    from lerobot.scripts.probe_offline_inference_pi05 import _build_episode_index

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


def get_subtask_idx(dataset, global_idx):
    """
    Read the subtask index from a dataset frame.
    Checks both top-level and nested key variants. Returns -1 if not present.
    """
    frame_row = dataset.hf_dataset[global_idx]
    for key in ("subtask_index", "complementary_info.subtask_index"):
        if key in frame_row:
            val = frame_row[key]
            return val.item() if isinstance(val, torch.Tensor) else int(val)
    return -1


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem
# ──────────────────────────────────────────────────────────────────────────────

def makedirs(*paths):
    """Create each path in *paths* (and its parents) if it does not already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(X, n_components, label, pca_dir):
    """
    Fit PCA on *X* (N, D). Saves a two-panel scree plot to *pca_dir*.

    *n_components* is clamped to min(n_components, N, D).

    Returns (X_pca tensor, pca_object).
    """
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
    out_path = os.path.join(pca_dir, f"{label}_pca_scree.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return torch.from_numpy(X_pca.astype(np.float32)), pca


def run_umap(X_pca, n_components, n_neighbors, min_dist, seed):
    """
    Fit UMAP (in-sample, fit_transform) and return the embedding array (N, n_components).

    Use this for representation / attention probes where no out-of-sample projection
    is needed.  For action-manifold probing (where new points must be projected onto
    a fixed reference) use umap_lib.UMAP.fit() + .transform() directly.
    """
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
# 2D matplotlib helpers
# ──────────────────────────────────────────────────────────────────────────────

def ax_style(ax, title, width=60):
    """Apply consistent axis styling to a 2D UMAP scatter / trajectory axes."""
    import textwrap
    ax.set_title(textwrap.fill(title, width=width), fontsize=9)
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


# ──────────────────────────────────────────────────────────────────────────────
# Plotly 3D helpers
# ──────────────────────────────────────────────────────────────────────────────

def plotly_3d_layout(title):
    """Return a Plotly layout dict for a clean 3D scatter plot."""
    return dict(
        title=dict(text=title, font=dict(size=13)),
        scene=dict(
            xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            zaxis=dict(title="UMAP-3", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            bgcolor="white",
            aspectmode="auto",
        ),
        paper_bgcolor="white",
        legend=dict(font=dict(size=12), itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=55),
        height=720,
    )


def ref_bg_trace_3d(ref_emb3):
    """Grey background scatter trace showing the reference GT manifold in 3D plots."""
    import plotly.graph_objects as go
    return go.Scatter3d(
        x=ref_emb3[:, 0], y=ref_emb3[:, 1], z=ref_emb3[:, 2],
        mode="markers", name="ref GT",
        marker=dict(size=2, color="#cccccc", opacity=0.30, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=True,
    )


def frame_colors_rgba(frames, cmap_name, alpha=0.85):
    """
    Map an array of frame indices to per-point rgba color strings using a sequential
    matplotlib colormap.  Convention: dark = early frame, pale = late frame.

    Returns a list of "rgba(r,g,b,a)" strings, one per frame.
    """
    frames = np.asarray(frames, dtype=float)
    fmin, fmax = frames.min(), frames.max()
    norm = 0.9 - (frames - fmin) / max(fmax - fmin, 1.0) * 0.6
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(norm)
    return [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})" for r, g, b, _ in rgba]
