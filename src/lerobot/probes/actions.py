#!/usr/bin/env python
"""
Action manifold probe — is the policy's motion inside the demonstrated distribution?

Every other probe asks whether some input reaches the actions. This one asks whether
the actions that come out are ones a demonstrator ever produced. It builds a reference
manifold from training GT and measures how far the policy's chunks land from it, with
held-out GT as the null: a real, unseen demonstration is also "far" from the training
set by some amount, and that amount is the only meaningful yardstick.

**The measurement space is the normalised action space, not raw joint targets.** Under
``action_encoding: anchor`` a chunk is stored as ``a_t - s`` (motion relative to the
current pose) and normalised with per-(step, joint) quantiles. That matters three
times over: the shared conditioning pose is subtracted out, so the distance reflects
what the policy *decided* rather than where the arm happened to be; joints with wide
travel stop dominating; and late chunk steps stop outweighing early ones.

**Distances are measured in PCA space, never in the UMAP picture.** UMAP's
``transform`` places an out-of-sample point at a weighted mean of its nearest training
neighbours, so an off-manifold prediction is pulled onto the manifold by construction —
a distance read there understates deviation by design. The plots are UMAP because it
draws better; every number comes from the 50-dim PCA space.

Three distances per chunk ``x`` (reference set ``{z_i}``, PCA basis ``mu, W``):

  off-span residual   r(x) = ||(x-mu) - W W^T (x-mu)||
                      motion no combination of training motions can express
  global NN           d(x) = min_i ||PCA(x) - z_i||
                      is this motion ever performed?
  state-conditioned   d|s(x) = min over the K reference frames nearest in state space
                      is this motion performed *from here*? — catches a plausible
                      reach executed from the wrong pose, which global NN cannot see

Each is reported as ``median(pred) / median(held-out GT)``, so 1.0 means
indistinguishable from a real demonstration. ``motion_magnitude_ratio`` guards the
failure the distances structurally cannot catch: a policy collapsing to near-zero
motion lands at the origin of anchor space, which is densely populated because the arm
pauses constantly, and would otherwise score a perfect ratio.

The reference manifold is cached (``manifold.pt``) and keyed by a fingerprint of the
reference data and the fitting parameters. Refitting moves the coordinate system, so
distances either side of a refit are not comparable — the manifold id is recorded in
``index.json`` to make that visible rather than something you have to remember.

Output layout (under ``probe_parameters.output_dir/actions/``):
  manifold.pt                   cached reference manifold (PCA + UMAP + reference set)
  actions_cache.pt              per-run eval results (mode=plot re-reads this)
  actions.json                  the summary dict behind index.json
  nn_distances.csv              per-episode distance summary, GT vs pred
  pca_variance/                 PCA scree plot
  2d/manifold.png               GT paths + pred dots on the grey reference manifold
  2d/distances.png              the metric panel: distance distributions + per-episode
  2d/per_subtask.png            which phase of the task drifts off-manifold
  2d/drift_over_time.png        does deviation grow across the episode
  2d/by_frame.png               GT vs pred, coloured by frame index
  2d/by_subtask.png             GT vs pred, coloured by subtask
  2d/episodes/ep{N:04d}.png     per-episode GT vs pred
  2d/overview.png               all eval datasets' GT on the reference manifold

Usage:
    python -m lerobot.probes.actions config.yaml
    python -m lerobot.probes.actions config.yaml --probe_parameters.mode plot
"""

import csv
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    DS_COLORS,
    EP_COLORS,
    SEQ_CMAPS,
    ax_style,
    dataset_display_name,
    get_action_chunk_lowdim,
    get_subtask_idx,
    get_subtask_str,
    load_extra_dataset,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    run_pca,
    sample_episodes_evenly,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeActionsConfig(TrainRLServerPipelineConfig):
    """Probe tunables live under ``cfg.probe_parameters`` (ProbeConfig)."""


# ──────────────────────────────────────────────────────────────────────────────
# Measurement space
# ──────────────────────────────────────────────────────────────────────────────

def chunk_vector(adapter: ProbablePolicy, actions: torch.Tensor, state: torch.Tensor | None):
    """One point on the manifold: a GT action chunk in normalised model space, flattened.

    Mirrors what the policy is trained to emit, so GT and prediction are directly
    comparable — ``predict_action_chunk`` returns its own normalised chunk as
    ``pred_norm`` and needs no conversion.
    """
    return adapter.normalize_gt_actions(actions, state).flatten().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Reference manifold — fit once, cache, fingerprint
# ──────────────────────────────────────────────────────────────────────────────

def reference_datasets(cfg, root_dataset):
    """Every training source, not just the one rl_offline happens to hand the probe.

    ``load_offline_dataset`` returns the normalization source alone, so a manifold fit
    on it would represent one collection out of seven while the policy trains on all of
    them — every motion learned from the other six would score as novel. The reference
    is the demonstrated distribution, so it spans the same sources training does.
    """
    from lerobot.rl.offline_dataset_utils import get_offline_dataset_sources

    root_name = dataset_display_name(root_dataset, cfg.dataset.root)
    datasets = [(root_name, root_dataset)]
    for source in get_offline_dataset_sources(cfg):
        name = os.path.basename(os.path.normpath(source.root)) if source.root else source.name
        if source.root is None or name == root_name:
            continue
        datasets.append((name, load_extra_dataset(source.repo_id, source.root)))
    return datasets


def manifold_fingerprint(cfg, datasets) -> dict:
    """What the cached manifold depends on. Any change here moves the coordinates.

    Deliberately built from the sampling *parameters* rather than the realised sample
    list: the list is a deterministic function of them, and computing it means indexing
    every episode of every training source, which a cache hit should not have to pay.
    """
    p = cfg.probe_parameters
    return {
        "reference": [name for name, _ in datasets],
        "n_dataset_frames": [len(ds) for _, ds in datasets],
        "ref_episodes": p.ref_max_episodes,
        "ref_frames_per_episode": p.ref_n_frames_per_episode,
        "chunk_size": int(cfg.policy.chunk_size),
        "action_encoding": str(getattr(cfg.policy, "action_encoding", "absolute")),
        "action_stats": str(getattr(cfg.policy, "action_encoding_stats_path", "")),
        "pca_dims": int(p.action_pca_dims),
        "umap": [int(p.umap_n_neighbors), float(p.umap_min_dist), int(p.umap_seed)],
        "seed": int(p.random_seed),
    }


def fingerprint_id(fingerprint: dict) -> str:
    return hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:8]


def collect_reference(adapter, datasets, cfg):
    """Normalised GT chunks, states and subtask labels across every reference source.

    No model forward and no video decode: the manifold is a property of the data.
    """
    p = cfg.probe_parameters
    vectors, states, subtasks = [], [], []

    for name, dataset in datasets:
        samples = sample_episodes_evenly(
            dataset,
            n_per_episode=p.ref_n_frames_per_episode,
            max_episodes=p.ref_max_episodes,
            seed=p.random_seed,
        )
        logging.info(f"  {name}: {len(samples)} reference frames")
        for _, _, global_idx in samples:
            actions, state, _, _ = get_action_chunk_lowdim(dataset, global_idx, adapter.chunk_size)
            vectors.append(chunk_vector(adapter, actions, state))
            states.append(state.numpy())
            subtasks.append(get_subtask_str(dataset, get_subtask_idx(dataset, global_idx)))

    return np.stack(vectors), np.stack(states), subtasks


def fit_manifold(adapter, datasets, cfg, pca_dir, fingerprint):
    """Fit PCA (the metric space) and UMAP (the picture) on reference GT.

    States are z-scored by the reference set's own statistics rather than the policy's
    normaliser: the state-conditioned neighbourhood only needs a consistent scale, and
    deriving it here keeps the probe independent of which normaliser the policy carries.
    """
    import warnings
    import umap as umap_lib

    p = cfg.probe_parameters
    vectors, states, subtasks = collect_reference(adapter, datasets, cfg)

    X_pca, pca = run_pca(torch.from_numpy(vectors), p.action_pca_dims, "gt_reference", pca_dir)
    ref_pca = X_pca.numpy()

    state_mean = states.mean(axis=0)
    state_std = np.maximum(states.std(axis=0), 1e-6)

    logging.info(f"  Fitting UMAP on {len(ref_pca)} reference frames …")
    reducer2d = umap_lib.UMAP(
        n_components=2,
        n_neighbors=p.umap_n_neighbors,
        min_dist=p.umap_min_dist,
        random_state=p.umap_seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)
        reducer2d.fit(ref_pca)

    return {
        "fingerprint": fingerprint,
        "id": fingerprint_id(fingerprint),
        "pca": pca,
        "ref_pca": ref_pca,
        "ref_states_z": (states - state_mean) / state_std,
        "ref_subtasks": subtasks,
        "state_mean": state_mean,
        "state_std": state_std,
        "reducer2d": reducer2d,
        "ref_emb2": reducer2d.embedding_,
    }


def default_manifold_cache(output_dir: str) -> str:
    """A path every validation step of a run shares.

    Validation writes each step to ``validation/step_00001000/actions``, so a cache
    inside the probe's own output would be refit at every step and never reused. Step
    directories are skipped, putting the manifold in ``validation/``; the standalone
    CLI has no step directory and lands in ``outputs/probe/``.
    """
    parent = os.path.dirname(os.path.normpath(output_dir))
    if os.path.basename(parent).startswith("step_"):
        parent = os.path.dirname(parent)
    return os.path.join(parent, "action_manifold.pt")


def load_or_build_manifold(adapter, root_dataset, cfg, output_dir, pca_dir):
    """Reuse the cached manifold when its fingerprint matches, else refit and say so."""
    p = cfg.probe_parameters
    cache_path = p.action_manifold_cache or default_manifold_cache(output_dir)
    datasets = reference_datasets(cfg, root_dataset)
    fingerprint = manifold_fingerprint(cfg, datasets)

    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") == fingerprint:
            logging.info(f"Reusing reference manifold {cached['id']} from {cache_path}")
            return cached
        logging.warning(
            "Reference manifold refit: the cached fingerprint no longer matches "
            f"({cached.get('fingerprint')} -> {fingerprint}). Distances from this run "
            "are in a new coordinate system and are NOT comparable to earlier runs."
        )

    logging.info(f"Fitting reference manifold over {len(datasets)} training source(s) …")
    manifold = fit_manifold(adapter, datasets, cfg, pca_dir, fingerprint)
    makedirs(os.path.dirname(cache_path) or ".")
    torch.save(manifold, cache_path)
    logging.info(f"Reference manifold {manifold['id']} saved → {cache_path}")
    return manifold


# ──────────────────────────────────────────────────────────────────────────────
# Distances — all in PCA space, never in the UMAP picture
# ──────────────────────────────────────────────────────────────────────────────

def project(manifold, X):
    """PCA coordinates and the off-span residual left behind.

    ``residual`` is the norm of the part of the chunk outside the PCA span — motion no
    combination of the reference motions can express.
    """
    pca = manifold["pca"]
    centered = X - pca.mean_
    coords = centered @ pca.components_.T
    residual = np.linalg.norm(centered - coords @ pca.components_, axis=1)
    return coords, residual


def state_neighbourhoods(manifold, states, k: int):
    """Indices of the K reference frames nearest each query in z-scored state space."""
    from scipy.spatial import cKDTree

    states_z = (states - manifold["state_mean"]) / manifold["state_std"]
    tree = cKDTree(manifold["ref_states_z"])
    k = min(k, len(manifold["ref_pca"]))
    _, idx = tree.query(states_z, k=k, workers=-1)
    return np.atleast_2d(idx)


def manifold_distances(manifold, X, states, subtasks, k: int) -> dict:
    """Off-span residual, global NN distance, and state-conditioned NN distance.

    The state-conditioned neighbour also answers a semantic question for free: does the
    nearest pose-matched reference motion come from the same phase of the task?
    """
    from scipy.spatial import cKDTree

    coords, residual = project(manifold, X)
    ref_pca = manifold["ref_pca"]

    nn_global, _ = cKDTree(ref_pca).query(coords, k=1, workers=-1)

    neighbourhood = state_neighbourhoods(manifold, states, k)
    deltas = ref_pca[neighbourhood] - coords[:, None, :]      # [N, K, pca_dims]
    local = np.linalg.norm(deltas, axis=2)                    # [N, K]
    nearest = local.argmin(axis=1)
    nn_state = local[np.arange(len(local)), nearest]

    ref_subtasks = np.asarray(manifold["ref_subtasks"], dtype=object)
    nn_subtask = ref_subtasks[neighbourhood[np.arange(len(nearest)), nearest]]
    # Chance is measured inside the neighbourhood, not over the whole reference set:
    # pose-matched frames are already phase-correlated, so the global label frequency
    # would make agreement look far more impressive than it is.
    labels = np.asarray(subtasks, dtype=object)
    agreement = np.array([nn_subtask[i] == labels[i] for i in range(len(labels))])
    chance = np.array([
        float(np.mean(ref_subtasks[neighbourhood[i]] == labels[i])) for i in range(len(labels))
    ])

    return {
        "coords": coords,
        "residual": residual,
        "nn_global": nn_global,
        "nn_state": nn_state,
        "magnitude": np.linalg.norm(X, axis=1),
        "subtask_agreement": agreement,
        "subtask_chance": chance,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — evaluation collection
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_eval_dataset(adapter: ProbablePolicy, dataset, samples, manifold, cfg):
    """Run inference, measure both GT and prediction against the reference manifold.

    Inference runs in the deployment regime (subtask + metadata clauses, short-term
    history, wrist depth) so the projected point is the one a rollout would produce.
    """
    import warnings

    chunk_size = adapter.chunk_size
    seed = int(cfg.probe_parameters.random_seed)
    gt_vecs, pred_vecs, states, subtasks, metadata = [], [], [], [], []

    adapter.suppress_logs(True)
    try:
        for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
            if i % 100 == 0:
                logging.debug(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            gt_subtask, task_str, state = frame["subtask"], frame["task"], frame["state"]
            # Seeded per frame so the projected point moves only when the policy does.
            generator = torch.Generator(device=adapter.device)
            generator.manual_seed(seed + int(global_idx))
            _, pred_norm, _ = adapter.predict_action_chunk(
                frame["obs"], task_str, state=state,
                subtask=gt_subtask, metadata=frame["metadata"], generator=generator,
            )

            gt_vecs.append(chunk_vector(adapter, frame["gt_actions"], state))
            pred_vecs.append(pred_norm.flatten().float().numpy())
            states.append(state.numpy())
            subtasks.append(gt_subtask)
            metadata.append({
                "episode_idx": ep_idx,
                "frame_idx":   fr_idx,
                "global_idx":  global_idx,
                "subtask":     gt_subtask,
                "subtask_idx": get_subtask_idx(dataset, global_idx),
                "task":        task_str,
            })
    finally:
        adapter.suppress_logs(False)

    gt_vec = np.stack(gt_vecs)
    pred_vec = np.stack(pred_vecs)
    states = np.stack(states)
    k = int(cfg.probe_parameters.action_nn_state_k)

    gt_dist = manifold_distances(manifold, gt_vec, states, subtasks, k)
    pred_dist = manifold_distances(manifold, pred_vec, states, subtasks, k)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden",
                                category=UserWarning)
        embeddings = {
            "gt_emb2":   manifold["reducer2d"].transform(gt_dist["coords"]),
            "pred_emb2": manifold["reducer2d"].transform(pred_dist["coords"]),
        }

    return {"metadata": metadata, "gt": gt_dist, "pred": pred_dist, **embeddings}


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def _group_medians(key, gt, pred, groups):
    """median(pred)/median(gt) of ``key`` within each group, ordered worst first."""
    rows = []
    for label in dict.fromkeys(groups):
        mask = np.array([g == label for g in groups])
        gt_med = float(np.median(gt[key][mask]))
        pred_med = float(np.median(pred[key][mask]))
        rows.append({
            "label": str(label),
            "n": int(mask.sum()),
            "gt_median": gt_med,
            "pred_median": pred_med,
            "ratio": pred_med / gt_med,
        })
    return sorted(rows, key=lambda r: r["ratio"], reverse=True)


def summarize(ds_name, ds_data, manifold) -> dict:
    gt, pred, meta = ds_data["gt"], ds_data["pred"], ds_data["metadata"]

    def side(dist):
        return {
            "nn_state_median":  float(np.median(dist["nn_state"])),
            "nn_state_p95":     float(np.percentile(dist["nn_state"], 95)),
            "nn_global_median": float(np.median(dist["nn_global"])),
            "residual_median":  float(np.median(dist["residual"])),
            "magnitude_median": float(np.median(dist["magnitude"])),
        }

    gt_side, pred_side = side(gt), side(pred)
    off_manifold = float(np.mean(pred["nn_state"] > np.percentile(gt["nn_state"], 99)))

    return {
        "dataset": ds_name,
        "manifold_id": manifold["id"],
        "n_frames": len(meta),
        "n_ref_frames": len(manifold["ref_pca"]),
        "dist_ratio_state_cond": pred_side["nn_state_median"] / gt_side["nn_state_median"],
        "dist_ratio_global": pred_side["nn_global_median"] / gt_side["nn_global_median"],
        "residual_ratio": pred_side["residual_median"] / gt_side["residual_median"],
        "off_manifold_rate": off_manifold,
        "nn_subtask_agreement": float(np.mean(pred["subtask_agreement"])),
        "nn_subtask_chance": float(np.mean(pred["subtask_chance"])),
        "nn_subtask_agreement_gt": float(np.mean(gt["subtask_agreement"])),
        "motion_magnitude_ratio": pred_side["magnitude_median"] / gt_side["magnitude_median"],
        "gt": gt_side,
        "pred": pred_side,
        "per_subtask": _group_medians("nn_state", gt, pred, [m["subtask"] for m in meta]),
        "per_episode": _group_medians("nn_state", gt, pred, [m["episode_idx"] for m in meta]),
        "verdict_note": (
            "Ratios are against held-out GT measured against the same reference manifold, "
            "so 1.0 means the policy's motion is as close to the training distribution as "
            "an unseen real demonstration is. motion_magnitude_ratio << 1 with good "
            "distance ratios is hold-still collapse: the origin of anchor space is densely "
            "populated, so pausing scores well on distance alone."
        ),
    }


def write_nn_csv(summaries, output_dir):
    rows = []
    for summary in summaries:
        for row in summary["per_episode"]:
            rows.append({
                "dataset": summary["dataset"],
                "episode": row["label"],
                "n": row["n"],
                "gt_median_nn_state": row["gt_median"],
                "pred_median_nn_state": row["pred_median"],
                "ratio": row["ratio"],
            })
    with open(os.path.join(output_dir, "nn_distances.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2D plot helpers (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_ref_bg(ax, ref_emb2):
    """Grey dots for the reference GT manifold (background layer)."""
    ax.scatter(ref_emb2[:, 0], ref_emb2[:, 1],
               s=5, color="#cccccc", alpha=0.35, linewidths=0, zorder=1)


def _gradient_scatter(ax, x, y, cmap_name, s=18, alpha=0.75, marker="o", zorder=2):
    """Fading scatter: dark=early, light=late."""
    t = np.linspace(0.30, 0.90, len(x))
    ax.scatter(x, y, c=t, cmap=cmap_name, vmin=0.0, vmax=1.0,
               s=s, alpha=alpha, marker=marker, linewidths=0, zorder=zorder)


# ──────────────────────────────────────────────────────────────────────────────
# The metric panels
# ──────────────────────────────────────────────────────────────────────────────

def plot_distances(ds_data, summary, output_path, ds_name):
    """The panel the metrics come from: pred vs held-out GT, in PCA space."""
    from matplotlib.patches import Patch

    gt, pred = ds_data["gt"], ds_data["pred"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for ax, key, title in [
        (axes[0], "nn_state", "State-conditioned NN distance"),
        (axes[1], "residual", "Off-span residual"),
    ]:
        values = np.concatenate([gt[key], pred[key]])
        bins = np.linspace(0, float(np.percentile(values, 98)), 40)
        ax.hist(gt[key],   bins=bins, color="#4477aa", alpha=0.55, density=True, label="GT (held out)")
        ax.hist(pred[key], bins=bins, color="#dd7733", alpha=0.55, density=True, label="Predicted")
        ax.axvline(np.median(gt[key]),   color="#223366", linewidth=1.3, linestyle="--")
        ax.axvline(np.median(pred[key]), color="#883300", linewidth=1.3, linestyle="--")
        ratio = np.median(pred[key]) / np.median(gt[key])
        ax.set_title(f"{title}\nratio = {ratio:.2f}  (1.0 = as close as real data)", fontsize=9)
        ax.set_xlabel("distance in PCA space", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    ax = axes[2]
    rows = summary["per_episode"]
    x = np.arange(len(rows))
    ax.scatter(x, [r["gt_median"] for r in rows], color="#4477aa", s=40, label="GT (held out)")
    ax.scatter(x, [r["pred_median"] for r in rows], color="#dd7733", s=40, label="Predicted")
    for i, row in enumerate(rows):
        ax.plot([i, i], [row["gt_median"], row["pred_median"]], color="#999", linewidth=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep {r['label']}" for r in rows], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("median state-conditioned NN distance", fontsize=8)
    ax.set_title("Per episode", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(handles=[
        Patch(facecolor="#4477aa", label="GT (held out)"),
        Patch(facecolor="#dd7733", label="Predicted"),
    ], fontsize=7)

    fig.suptitle(f"{ds_name} — distance to the training action manifold", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_subtask(summary, output_path, ds_name):
    """Which phase of the task leaves the manifold."""
    rows = [r for r in summary["per_subtask"] if r["label"]]
    if not rows:
        return
    labels = [r["label"][:38] + "…" if len(r["label"]) > 40 else r["label"] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.42 * len(rows) + 1.2)))
    colors = ["#cc4125" if r["ratio"] > 3.0 else "#dd9933" if r["ratio"] > 1.5 else "#4c9a6a"
              for r in rows]
    ax.barh(y, [r["ratio"] for r in rows], color=colors, alpha=0.85)
    ax.axvline(1.0, color="#333", linewidth=1.2, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lbl}  (n={r['n']})" for lbl, r in zip(labels, rows)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("median pred distance / median GT distance   (dashed 1.0 = as close as real data)",
                  fontsize=8)
    ax.set_title(f"{ds_name} — off-manifold ratio by subtask", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drift_over_time(ds_data, output_path, ds_name, n_bins=20):
    """Does the policy drift further off-manifold as the episode goes on?"""
    meta = ds_data["metadata"]
    ep_ids = np.array([m["episode_idx"] for m in meta])
    frames = np.array([m["frame_idx"] for m in meta], dtype=float)

    progress = np.zeros(len(frames))
    for ep in np.unique(ep_ids):
        mask = ep_ids == ep
        span = max(frames[mask].max(), 1.0)
        progress[mask] = frames[mask] / span

    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    slots = np.clip(np.digitize(progress, edges) - 1, 0, n_bins - 1)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for key, color, label in [("gt", "#4477aa", "GT (held out)"), ("pred", "#dd7733", "Predicted")]:
        values = ds_data[key]["nn_state"]
        med = np.array([np.median(values[slots == b]) if np.any(slots == b) else np.nan
                        for b in range(n_bins)])
        lo = np.array([np.percentile(values[slots == b], 25) if np.any(slots == b) else np.nan
                       for b in range(n_bins)])
        hi = np.array([np.percentile(values[slots == b], 75) if np.any(slots == b) else np.nan
                       for b in range(n_bins)])
        ax.plot(centers, med, color=color, linewidth=1.8, label=label)
        ax.fill_between(centers, lo, hi, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel("progress through episode", fontsize=8)
    ax.set_ylabel("state-conditioned NN distance", fontsize=8)
    ax.set_title(f"{ds_name} — deviation over the episode  (band = p25–p75)", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots — the UMAP picture
# ──────────────────────────────────────────────────────────────────────────────

def _pair_segments(ax, gt_emb, pred_emb, alpha=0.35):
    """Join each GT point to the prediction made at that same frame."""
    from matplotlib.collections import LineCollection

    segs = np.stack([gt_emb, pred_emb], axis=1)
    ax.add_collection(LineCollection(segs, colors="#777777", linewidth=0.7,
                                     alpha=alpha, zorder=2))


def plot_2d_manifold(ref_emb2, gt_emb2, pred_emb2, metadata, output_path, ds_name):
    """Held-out GT and the policy's prediction for the same frame, paired.

    A point here is a *motion*, not a pose: consecutive sampled frames are seconds
    apart and their chunks are unrelated, so joining them into a trajectory would draw
    a path that does not exist. What is real is the pairing — each GT point and the
    prediction made at that frame, joined by a segment whose length is the
    disagreement, over the grey cloud of demonstrated motion.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ep_ids     = np.array([m["episode_idx"] for m in metadata])
    fr_ids     = np.array([m["frame_idx"]   for m in metadata])
    unique_eps = np.unique(ep_ids)

    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_ref_bg(ax, ref_emb2)
    _pair_segments(ax, gt_emb2, pred_emb2)

    legend_handles = []
    for i, ep in enumerate(unique_eps):
        idx  = np.where(ep_ids == ep)[0]
        col  = EP_COLORS[i % len(EP_COLORS)]
        cmap = matplotlib.colormaps.get_cmap(SEQ_CMAPS[i % len(SEQ_CMAPS)])
        frames = fr_ids[idx].astype(float)
        shade = 0.85 - (frames - frames.min()) / max(float(np.ptp(frames)), 1.0) * 0.55

        ax.scatter(gt_emb2[idx, 0], gt_emb2[idx, 1], color=cmap(shade),
                   s=26, marker="o", linewidths=0, zorder=3)
        ax.scatter(pred_emb2[idx, 0], pred_emb2[idx, 1], color=col,
                   s=34, marker="x", linewidths=1.2, alpha=0.9, zorder=4)
        legend_handles.append(Patch(facecolor=col, label=f"ep {ep}"))

    if len(unique_eps) <= 20:
        enc = [
            Line2D([0], [0], color="#555", marker="o", linestyle="None", markersize=5,
                   label="GT  (dark→light = early→late frame)"),
            Line2D([0], [0], color="#555", marker="x", linestyle="None", markersize=6,
                   label="Pred  (same frame, same colour)"),
            Line2D([0], [0], color="#777", linewidth=0.9, label="pairing (length = disagreement)"),
        ]
        ax.legend(handles=legend_handles + enc, fontsize=7, ncol=2)
    ax.autoscale_view()
    ax_style(ax, f"{ds_name} — predicted vs demonstrated motion, paired per frame")
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
    """One episode, same pairing as the overview: GT dots, pred crosses, joined."""
    from matplotlib.lines import Line2D

    fr = np.array([m["frame_idx"] for m in meta_ep])

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_ref_bg(ax, ref_emb2)
    _pair_segments(ax, gt_emb_ep, pred_emb_ep, alpha=0.5)

    x_gt, y_gt = gt_emb_ep[:, 0], gt_emb_ep[:, 1]
    ax.scatter(x_gt, y_gt, c=np.linspace(0.85, 0.30, len(x_gt)), cmap="Blues",
               vmin=0.0, vmax=1.0, s=30, zorder=4, linewidths=0)
    _gradient_scatter(ax, pred_emb_ep[:, 0], pred_emb_ep[:, 1], "Oranges",
                      s=38, alpha=0.9, marker="x", zorder=5)

    ax.annotate(f"fr{fr[0]}",  (x_gt[0],  y_gt[0]),  textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#2ca02c")
    ax.annotate(f"fr{fr[-1]}", (x_gt[-1], y_gt[-1]), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#d62728")

    legend_handles = [
        Line2D([0], [0], color=matplotlib.colormaps.get_cmap("Blues")(0.6),
               marker="o", linestyle="None", markersize=6, label="GT (dark→light)"),
        Line2D([0], [0], color=matplotlib.colormaps.get_cmap("Oranges")(0.7),
               marker="x", linestyle="None", markersize=7, label="Predicted"),
        Line2D([0], [0], color="#777", linewidth=0.9, label="pairing (length = disagreement)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.85)
    ax.autoscale_view()
    ax_style(ax, f"{ds_name} — ep {ep_idx}  ({len(fr)} frames)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_overview(ref_emb2, datasets_cache, output_path):
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


# ──────────────────────────────────────────────────────────────────────────────
# Plotting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_plotting(cache, output_dir):
    ref_emb2 = cache["ref_emb2"]

    d2 = os.path.join(output_dir, "2d")
    makedirs(d2)

    for ds_name, ds_data in cache["datasets"].items():
        meta      = ds_data["metadata"]
        gt_emb2   = ds_data["gt_emb2"]
        pred_emb2 = ds_data["pred_emb2"]
        summary   = cache["summaries"][ds_name]

        plot_distances(ds_data, summary, os.path.join(d2, "distances.png"), ds_name)
        plot_per_subtask(summary, os.path.join(d2, "per_subtask.png"), ds_name)
        plot_drift_over_time(ds_data, os.path.join(d2, "drift_over_time.png"), ds_name)

        plot_2d_manifold(ref_emb2, gt_emb2, pred_emb2, meta,
                         os.path.join(d2, "manifold.png"), ds_name)
        plot_2d_by_frame(ref_emb2, gt_emb2, pred_emb2, meta,
                         os.path.join(d2, "by_frame.png"), ds_name)
        plot_2d_by_subtask(ref_emb2, gt_emb2, pred_emb2, meta,
                           os.path.join(d2, "by_subtask.png"), ds_name)

        ep_dir = os.path.join(d2, "episodes")
        makedirs(ep_dir)
        ep_ids = np.array([m["episode_idx"] for m in meta])
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

    plot_2d_overview(ref_emb2, cache["datasets"], os.path.join(d2, "overview.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────

_RATIO_NOTE = ("1.0 = the policy's motion sits as close to the training manifold as an "
               "unseen real demonstration does. Thresholds are provisional.")


def write_manifest(output_dir, summary):
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Action Manifold",
        group="Actions",
        claim="Does the policy's motion stay inside the distribution the demonstrations cover?",
        summary=summary,
        metrics=[
            Metric("dist_ratio_state_cond", "Novelty vs unseen GT (pose-matched)",
                   good="low", fmt=2, baseline=1.0, warn=1.5, bad=3.0, primary=True,
                   note="The headline. Nearest demonstrated motion among reference frames "
                        "in a similar pose — catches a plausible motion done from the wrong "
                        f"place. {_RATIO_NOTE}"),
            Metric("dist_ratio_global", "Novelty vs unseen GT (any pose)",
                   good="low", fmt=2, baseline=1.0, warn=1.5, bad=3.0, primary=True,
                   note=f"Is this motion ever performed at all? {_RATIO_NOTE}"),
            Metric("residual_ratio", "Off-span residual ratio",
                   good="low", fmt=2, baseline=1.0,
                   note="Motion outside the span of the reference PCA basis — shape no "
                        "combination of demonstrated motions can express. Deliberately "
                        "has no threshold: held-out GT sits almost exactly in the span, so "
                        "the denominator is tiny and the ratio runs high (10x) even for a "
                        "policy whose distances are perfect. Track it against this "
                        "checkpoint's own history, and set a line once a trusted "
                        "checkpoint has been measured."),
            Metric("off_manifold_rate", "Frames past the GT p99",
                   good="low", fmt=3, baseline=0.01, warn=0.10, bad=0.25, primary=True,
                   note="Fraction of predictions further out than 99% of held-out GT."),
            Metric("motion_magnitude_ratio", "Motion magnitude vs GT",
                   good="none", fmt=2, baseline=1.0, primary=True,
                   note="Descriptive, and the guard on everything above: << 1 with good "
                        "distance ratios is hold-still collapse, which scores well on "
                        "distance because the arm pauses often in the reference data."),
            Metric("nn_subtask_agreement", "Nearest motion is same-phase",
                   good="high", fmt=2,
                   note="Does the closest pose-matched reference motion come from the same "
                        "subtask? Only read it next to nn_subtask_chance: the labels are "
                        "per-dataset strings, so a val task whose vocabulary the reference "
                        "does not cover scores near zero however good the policy is."),
            Metric("nn_subtask_chance", "…chance rate in that neighbourhood",
                   good="none", fmt=2),
            Metric("n_frames", "Frames evaluated", good="none", fmt=0),
            Metric("n_ref_frames", "Reference frames", good="none", fmt=0),
        ],
        panels=[
            Panel("2d/distances.png",
                  "Distance to the training manifold: predictions vs held-out GT",
                  how="Blue is held-out GT, orange is the policy. The blue distribution is "
                      "the null — a real demonstration is also some distance from the "
                      "training set. Orange sitting on top of blue is the good outcome; "
                      "orange shifted right is novel motion. Distances are PCA-space, not "
                      "UMAP."),
            Panel("2d/per_subtask.png", "Off-manifold ratio broken down by subtask",
                  how="Which phase of the task drifts. Green under 1.5, amber to 3, red "
                      "beyond. Read n= before trusting a bar."),
            Panel("2d/drift_over_time.png", "Deviation across the episode",
                  how="A rising orange line means error accumulates with task progress; a "
                      "flat one means the failure is uniform. Both lines rising together is "
                      "the data getting more varied late, not the policy degrading."),
            Panel("2d/manifold.png", "Predicted vs demonstrated motion, paired per frame",
                  how="Grey is the demonstrated motion. Each dot is a held-out GT chunk, "
                      "each cross the policy's chunk at that same frame, joined by a "
                      "segment whose length is the disagreement. Long segments pointing "
                      "off the grey cloud are the bad case. UMAP is a picture only — judge "
                      "distance from the metrics, not by eye here."),
            Panel("2d/by_subtask.png", "GT vs predicted, coloured by subtask"),
            Panel("2d/by_frame.png", "GT vs predicted, coloured by frame index"),
            Panel("2d/overview.png", "Every evaluated dataset on one manifold"),
            Panel("nn_distances.csv", "Per-episode distance table"),
        ],
        extra={"manifold_id": summary["manifold_id"], "datasets": summary.get("datasets", {})},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(adapter, root_dataset, cfg, output_dir, *, eval_dataset=None):
    """Run the actions probe end-to-end (collect + plot).

    Called by both :func:`probe_cli` (standalone) and the rl_offline validation loop.
    ``adapter`` and ``root_dataset`` may be ``None`` when
    ``cfg.probe_parameters.mode == "plot"`` (cache-only re-plot).

    ``root_dataset`` seeds the reference: the manifold is fit on it plus every other
    source in ``dataset.sources`` (see :func:`reference_datasets`), because that is what
    the policy was trained on. ``eval_dataset`` is the held-out set the policy is
    measured on; it must not be part of the reference, or the GT null collapses to zero.
    """
    p = cfg.probe_parameters
    makedirs(output_dir)
    cache_path = os.path.join(output_dir, "actions_cache.pt")
    cache = None

    if p.mode in ("collect", "all"):
        if adapter is None or root_dataset is None:
            raise ValueError("collect mode requires adapter and root_dataset.")
        pca_dir = os.path.join(output_dir, "pca_variance")
        makedirs(pca_dir)

        manifold = load_or_build_manifold(adapter, root_dataset, cfg, output_dir, pca_dir)

        if eval_dataset is None:
            logging.warning(
                "No eval_dataset: evaluating the reference dataset against itself. Frames "
                "shared with the reference set sit at distance zero, so the GT null "
                "collapses and every ratio is inflated. Set val_dataset_path."
            )
        primary_eval_dataset = eval_dataset if eval_dataset is not None else root_dataset
        eval_name = dataset_display_name(
            primary_eval_dataset,
            getattr(cfg, "val_dataset_path", None) if eval_dataset is not None else cfg.dataset.root,
        )
        # Only the held-out set is evaluated. Every other collection is training data and
        # is already in the reference manifold; measuring the policy against a manifold
        # its own evaluation frames helped build would compare it to itself.
        all_datasets = [(eval_name, primary_eval_dataset)]

        cache = {
            "manifold_id": manifold["id"],
            "ref_emb2": manifold["ref_emb2"],
            "datasets": {},
            "summaries": {},
        }
        for ds_name, dataset in all_datasets:
            logging.info(f"=== Evaluating '{ds_name}' ===")
            eval_samples = sample_episodes_evenly(
                dataset,
                n_per_episode=p.n_frames_per_episode,
                max_episodes=p.max_episodes,
                seed=p.random_seed,
                stride=probe_image_stride(cfg),
            )
            logging.info(f"  {len(eval_samples)} frames")
            ds_data = collect_eval_dataset(adapter, dataset, eval_samples, manifold, cfg)
            cache["datasets"][ds_name] = ds_data
            cache["summaries"][ds_name] = summarize(ds_name, ds_data, manifold)

        torch.save(cache, cache_path)
        logging.debug(f"Cache saved → {cache_path}")

    if p.mode in ("plot", "all"):
        if cache is None:
            logging.info(f"Loading cache from {cache_path} …")
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        logging.info("Generating plots …")
        run_plotting(cache, output_dir)

        summaries = list(cache["summaries"].values())
        summary = dict(summaries[0])
        if len(summaries) > 1:
            summary["datasets"] = {s["dataset"]: s for s in summaries[1:]}
        with open(os.path.join(output_dir, "actions.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        write_nn_csv(summaries, output_dir)
        write_manifest(output_dir, summary)

        logging.info(
            f"Done. ratio(state-cond)={summary['dist_ratio_state_cond']:.2f} "
            f"ratio(global)={summary['dist_ratio_global']:.2f} "
            f"off-manifold={summary['off_manifold_rate']:.3f} "
            f"magnitude={summary['motion_magnitude_ratio']:.2f} → {output_dir}/"
        )


@parser.wrap()
def probe_cli(cfg: ProbeActionsConfig):
    init_logging()
    p = cfg.probe_parameters
    output_dir = os.path.join(p.output_dir, "actions")

    adapter = root_dataset = eval_dataset = None
    if p.mode in ("collect", "all"):
        device = get_safe_torch_device(try_device=cfg.policy.device)
        root_dataset = load_probe_dataset(cfg)
        # Mirror rl_offline: the manifold is fit on training data and the policy is
        # measured on the held-out set, or the GT null it is compared against includes
        # the very frames the manifold was built from.
        val_path = getattr(cfg, "val_dataset_path", None)
        if val_path:
            logging.info(f"Evaluating held-out dataset {val_path}")
            eval_dataset = load_extra_dataset(cfg.dataset.repo_id, val_path)
        logging.info("Loading policy adapter …")
        adapter = ProbablePolicy.for_config(cfg, device, dataset=root_dataset)

    run(adapter, root_dataset, cfg, output_dir, eval_dataset=eval_dataset)


if __name__ == "__main__":
    probe_cli()
