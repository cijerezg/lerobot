"""
Generic, policy-agnostic helpers for probe scripts.

Covers dataset frame reading, episode sampling, PCA/UMAP fitting, and matplotlib
/ Plotly style helpers used across probes. Anything that depends on a specific
policy lives in the per-policy adapter (``probes.adapters.<policy>``).
"""

from __future__ import annotations

import logging
import os
import random
import textwrap
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

EP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#a9a9a9", "#ffd8b1", "#fabed4", "#fffac8", "#e0e0e0",
]

SEQ_CMAPS = ["Reds",   "Blues",   "Greens",  "Oranges", "Purples",
             "copper", "cool",    "spring",  "winter",  "autumn"]

DS_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
             "#42d4f4", "#f032e6", "#bfef45", "#469990"]


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem
# ──────────────────────────────────────────────────────────────────────────────

def makedirs(*paths: str) -> None:
    """Create each path (and parents) if missing."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset frame reading (no policy involvement)
# ──────────────────────────────────────────────────────────────────────────────

def build_episode_index(dataset) -> dict[int, list[int]]:
    """Map episode_index → sorted list of global frame indices."""
    ep_to_indices: dict[int, list[int]] = {}
    for global_idx in range(len(dataset)):
        ep_idx = dataset.hf_dataset[global_idx]["episode_index"].item()
        ep_to_indices.setdefault(ep_idx, []).append(global_idx)
    for ep_idx in ep_to_indices:
        ep_to_indices[ep_idx].sort()
    return ep_to_indices


def load_extra_dataset(repo_id: str, root: str) -> LeRobotDataset:
    """Load an additional LeRobot dataset from a local *root* directory."""
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    ds.delta_timestamps = None
    ds.delta_indices = None
    return ds


def dataset_display_name(dataset, fallback_root: str | os.PathLike | None = None) -> str:
    """Return a stable short name for a dataset object or fallback root."""
    root = getattr(dataset, "root", None) or fallback_root
    if root is None:
        return "dataset"
    return os.path.basename(os.path.normpath(os.fspath(root)))


def get_subtask_idx(dataset, global_idx: int) -> int:
    """Read the subtask index from a dataset frame; returns -1 if not present."""
    frame_row = dataset.hf_dataset[global_idx]
    for key in ("subtask_index", "complementary_info.subtask_index"):
        if key in frame_row:
            val = frame_row[key]
            return val.item() if isinstance(val, torch.Tensor) else int(val)
    return -1


def get_subtask_str(dataset, subtask_idx: int) -> str:
    """Look up a subtask description string by index; returns "" if unavailable."""
    if subtask_idx < 0:
        return ""
    meta = getattr(dataset, "meta", None)
    subtasks_df = getattr(meta, "subtasks", None) if meta is not None else None
    if subtasks_df is None:
        return ""
    try:
        if hasattr(subtasks_df, "columns") and "subtask_index" in subtasks_df.columns:
            rows = subtasks_df[subtasks_df["subtask_index"] == subtask_idx]
            if not rows.empty:
                return str(rows.iloc[0].name)
        if subtask_idx in subtasks_df.index:
            return str(subtasks_df.loc[subtask_idx, "subtask"])
    except Exception:
        return ""
    return ""


def get_frame_data(dataset, global_idx: int, chunk_size: int):
    """
    Pull a single frame + its GT action chunk from a LeRobot dataset.

    Returns:
        obs:         dict[str, Tensor] with batch dim 1, keys starting with "observation."
        gt_actions:  Tensor [chunk_size, action_dim] — raw (unnormalised), pad with last action
        state:       Tensor [state_dim] or None — current joint state if available
        gt_subtask:  str — subtask description from metadata ("" if absent)
        task_str:    str — high-level task
        episode_idx: int
        frame_idx:   int
    """
    frame = dataset[global_idx]
    episode_idx = frame["episode_index"].item()
    frame_idx = frame["frame_index"].item()
    task_str = frame.get("task", "")

    gt_actions = []
    for offset in range(chunk_size):
        candidate_idx = global_idx + offset
        if candidate_idx >= len(dataset):
            break
        f_item = dataset.hf_dataset[candidate_idx]
        if f_item["episode_index"].item() != episode_idx:
            break
        is_pad = f_item.get("action_is_pad", False)
        if isinstance(is_pad, torch.Tensor):
            is_pad = is_pad.item()
        if is_pad:
            break
        gt_actions.append(f_item["action"].detach().clone())

    if not gt_actions:
        gt_actions = [torch.zeros_like(frame["action"])]
    while len(gt_actions) < chunk_size:
        gt_actions.append(gt_actions[-1].clone())
    gt_actions = torch.stack(gt_actions[:chunk_size])

    obs = {
        k: v.unsqueeze(0)
        for k, v in frame.items()
        if k.startswith("observation.") and isinstance(v, torch.Tensor)
    }

    state = None
    if "observation.state" in frame:
        state = frame["observation.state"].float()

    subtask_idx = get_subtask_idx(dataset, global_idx)
    gt_subtask = get_subtask_str(dataset, subtask_idx)

    return obs, gt_actions, state, gt_subtask, task_str, episode_idx, frame_idx


# ──────────────────────────────────────────────────────────────────────────────
# Episode / frame sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_episodes_evenly(
    dataset,
    n_per_episode: int,
    max_episodes: Optional[int],
    seed: int,
) -> list[tuple[int, int, int]]:
    """
    Sample *n_per_episode* evenly-spaced frames from each episode.

    If *max_episodes* is set, draw a reproducible random subset of episodes.
    Returns list of (episode_idx, frame_idx_in_episode, global_idx).
    """
    ep_to_indices = build_episode_index(dataset)
    episodes = sorted(ep_to_indices.keys())
    if max_episodes is not None:
        rng = np.random.RandomState(seed)
        episodes = sorted(
            rng.choice(episodes, size=min(max_episodes, len(episodes)), replace=False).tolist()
        )

    samples: list[tuple[int, int, int]] = []
    for ep_idx in episodes:
        indices = ep_to_indices[ep_idx]
        n = min(n_per_episode, len(indices))
        chosen = np.linspace(0, len(indices) - 1, n, dtype=int)
        for pos in chosen:
            global_idx = indices[pos]
            fr_idx = dataset.hf_dataset[global_idx]["frame_index"].item()
            samples.append((ep_idx, fr_idx, global_idx))

    return samples


def build_sample_list(
    dataset,
    episodes_str: Optional[str],
    frames_str: Optional[str],
    random_n: Optional[int],
    chunk_size: int,
    seed: Optional[int] = None,
) -> list[tuple[int, int, int]]:
    """
    Build a list of ``(episode_idx, frame_idx_in_episode, global_idx)`` from
    optional explicit pairs and optional random sampling.

    Used by the offline_inference probe for manual / random frame selection.
    """
    ep_to_indices = build_episode_index(dataset)
    samples: list[tuple[int, int, int]] = []

    if episodes_str:
        ep_list = [int(e) for e in episodes_str.split(",")]
        fr_list = [int(f) for f in frames_str.split(",")] if frames_str else [0] * len(ep_list)
        if len(fr_list) == 1:
            fr_list = fr_list * len(ep_list)
        for ep_idx, fr_idx in zip(ep_list, fr_list):
            if ep_idx not in ep_to_indices:
                logging.warning(f"Episode {ep_idx} not found in dataset, skipping.")
                continue
            indices = ep_to_indices[ep_idx]
            if fr_idx >= len(indices):
                logging.warning(
                    f"Frame {fr_idx} out of range for episode {ep_idx} "
                    f"({len(indices)} frames), skipping."
                )
                continue
            samples.append((ep_idx, fr_idx, indices[fr_idx]))

    if random_n:
        rng = random.Random(seed)
        all_global = list(range(len(dataset)))
        rng.shuffle(all_global)
        existing = {g for _, _, g in samples}
        added = 0
        for global_idx in all_global:
            if added >= random_n:
                break
            if global_idx in existing:
                continue
            item = dataset.hf_dataset[global_idx]
            ep_idx = item["episode_index"].item()
            fr_idx = item["frame_index"].item()
            indices = ep_to_indices[ep_idx]
            if (len(indices) - indices.index(global_idx)) < 1:
                continue
            samples.append((ep_idx, fr_idx, global_idx))
            existing.add(global_idx)
            added += 1

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessor introspection
# ──────────────────────────────────────────────────────────────────────────────

def find_normalizer_step(preprocessor):
    """
    Return the NormalizerProcessorStep from a preprocessor pipeline.

    Duck-typed (``norm_map`` + ``_tensor_stats``) so the same lookup works for
    pi05's ``NormalizerProcessorStep`` and molmoact2's
    ``MolmoAct2MaskedNormalizerProcessorStep`` (both subclass the base
    NormalizerProcessorStep).
    """
    for step in preprocessor.steps:
        if hasattr(step, "norm_map") and hasattr(step, "_tensor_stats"):
            return step
    raise RuntimeError(
        f"No normalizer step found in preprocessor pipeline. "
        f"Steps: {[type(s).__name__ for s in preprocessor.steps]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(X: torch.Tensor, n_components: int, label: str, pca_dir: str):
    """
    Fit PCA on *X* (N, D). Saves a two-panel scree plot.

    Returns (X_pca tensor float32, fitted sklearn PCA).
    """
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    with threadpool_limits(limits=1, user_api="blas"):
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


def run_umap(X_pca: torch.Tensor, n_components: int, n_neighbors: int,
             min_dist: float, seed: int) -> np.ndarray:
    """
    In-sample UMAP fit_transform. Returns (N, n_components) array.

    For out-of-sample projection (action probe), use umap.UMAP().fit() then
    reducer.transform() directly.
    """
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

def ax_style(ax, title: str, width: int = 60) -> None:
    """Consistent axis styling for 2D UMAP scatter / trajectory plots."""
    ax.set_title(textwrap.fill(title, width=width), fontsize=9)
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


# ──────────────────────────────────────────────────────────────────────────────
# Plotly 3D helpers
# ──────────────────────────────────────────────────────────────────────────────

def plotly_3d_layout(title: str) -> dict:
    """Layout dict for a clean Plotly 3D scatter."""
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
    """Grey background scatter trace for the reference GT manifold."""
    import plotly.graph_objects as go
    return go.Scatter3d(
        x=ref_emb3[:, 0], y=ref_emb3[:, 1], z=ref_emb3[:, 2],
        mode="markers", name="ref GT",
        marker=dict(size=2, color="#cccccc", opacity=0.30, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=True,
    )


def frame_colors_rgba(frames, cmap_name: str, alpha: float = 0.85) -> list[str]:
    """
    Map frame indices to rgba color strings via a sequential colormap.
    Convention: dark = early frame, pale = late frame.
    """
    frames = np.asarray(frames, dtype=float)
    fmin, fmax = frames.min(), frames.max()
    norm = 0.9 - (frames - fmin) / max(fmax - fmin, 1.0) * 0.6
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(norm)
    return [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})" for r, g, b, _ in rgba]
