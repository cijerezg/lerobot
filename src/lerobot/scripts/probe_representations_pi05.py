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

Output layout (all under --probe_output_dir):
  activations_cache.pt          reusable tensor cache (skip re-inference with --probe_mode plot)
  episode_thumbnails/           first-frame images so you can identify each episode
  pca_variance/                 scree plots for every PCA fit
  2d/<site>/by_episode.png      per-episode gradient: dark=early frame, light=late
  2d/<site>/by_frame.png        all episodes pooled, coloured by frame index
  2d/<site>/by_subtask.png      all episodes pooled, coloured by subtask
  3d/<site>/by_episode.html     interactive version of by_episode
  3d/<site>/by_subtask.html     interactive version of by_subtask
  3d/<site>/ep{A}_vs_ep{B}.html two-episode comparison coloured by frame index
  subtask_injection/<site>/     GT vs generated subtask UMAP (2d + 3d)
  subtask_injection/generated_subtasks.csv  per-frame GT and model-generated subtask text

Usage:
    python probe_representations_pi05.py config-hiserl.json
    python probe_representations_pi05.py config-hiserl.json --probe_output_dir outputs/my_probe
    python probe_representations_pi05.py config-hiserl.json --probe_mode plot --probe_cache outputs/probe_representations/activations_cache.pt
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

# Optional dependencies — install before running:
#   pip install umap-learn plotly scikit-learn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import lerobot.rl.rl_pi05  # noqa: F401 — registers PI05RLConfig

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.pi05_full.modeling_pi05 import pad_vector
from lerobot.processor.core import TransitionKey
from lerobot.rl.pi05_train_utils import make_pi05_full_processors_with_upgrade
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_USER_PROMPT_TOKENS,
    OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.scripts.eval_offline_pi05 import (
    _build_episode_index,
    get_frame_data,
)

# ──────────────────────────────────────────────────────────────────────────────
# Probe parameters — edit these or override via the config dataclass below
# ──────────────────────────────────────────────────────────────────────────────

N_FRAMES_PER_EPISODE = 128          # evenly spaced frames sampled per episode
MAX_EPISODES = 5                    # set to None to use all episodes in dataset
RANDOM_SEED = 42

PROBE_SITES = "prefix,suffix"      # "prefix", "suffix", or "prefix,suffix"
DENOISING_TIMESTEPS = "1.0,0.25"  # t values for suffix probing (comma-separated)

PCA_DIMS = 100                     # pre-UMAP reduction dimension
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_SEED = 42

EPISODE_3D_A = 0                   # episode A for two-episode 3D plot
EPISODE_3D_B = 1                   # episode B for two-episode 3D plot

DO_SUBTASK_INJECTION = True        # run gen vs GT subtask injection analysis


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProbeRepresentationsConfig(TrainRLServerPipelineConfig):
    """Extends the base training config with representation probe parameters."""

    probe_output_dir: str = "outputs/probe_representations"
    # "collect" → only collect activations and save cache
    # "plot"    → load existing cache and generate all plots
    # "all"     → collect then plot
    probe_mode: str = "all"

    # Data sampling
    probe_n_frames_per_episode: int = N_FRAMES_PER_EPISODE
    probe_max_episodes: Optional[int] = MAX_EPISODES
    probe_random_seed: int = RANDOM_SEED

    # Activation sites and denoising timesteps
    probe_sites: str = PROBE_SITES
    probe_denoising_timesteps: str = DENOISING_TIMESTEPS

    # Dimensionality reduction
    probe_pca_dims: int = PCA_DIMS
    probe_umap_n_neighbors: int = UMAP_N_NEIGHBORS
    probe_umap_min_dist: float = UMAP_MIN_DIST
    probe_umap_seed: int = UMAP_SEED

    # 3D two-episode plot
    probe_ep_3d_a: int = EPISODE_3D_A
    probe_ep_3d_b: int = EPISODE_3D_B

    # Subtask injection
    probe_subtask_injection: bool = DO_SUBTASK_INJECTION


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

    return policy, preprocessor, dataset


# ──────────────────────────────────────────────────────────────────────────────
# Hook utilities
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _capture_hook(module):
    """
    Monkey-patches module.forward to capture its outputs and yields a shared dict.
    After each call, the dict contains:
      "prefix_out": (B, prefix_len, hidden_dim)  — float32, on CPU
      "suffix_out": (B, full_seq_len, hidden_dim) — float32, on CPU

    NOTE: register_forward_hook is NOT used because the call chain calls .forward()
    directly (not via __call__), which bypasses PyTorch hooks entirely.
    """
    captured = {}
    original_forward = module.forward

    def patched_forward(*args, **kwargs):
        # PaliGemmaWithExpertModel.forward() returns ([prefix_out, suffix_out], past_key_values)
        # Either output may be None (e.g. prefix-only call from generate_subtask_tokens)
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
    """Single PI05Pytorch.forward() call with an explicit timestep t_val."""
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
        task_tokens, task_masks  — (1, seq_len) on device
        subtask_tokens, subtask_masks  — (1, seq_len) on device, using gt_subtask
        action_tokens, action_masks  — FAST discrete action tokens (1, n_tokens) on device
        actions_padded  — (1, chunk_size, max_action_dim) normalised + padded on device
    """
    raw_batch = {
        TransitionKey.ACTION: gt_actions.unsqueeze(0).to(device),
        **{k: v.to(device) for k, v in obs.items()},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": [task_str],
            "subtask": [gt_subtask],
            "advantage": torch.tensor([[1.0]], device=device),
        },
    }
    processed = preprocessor(raw_batch)

    images, img_masks = policy._preprocess_images({k: v.to(device) for k, v in obs.items()})

    task_tokens    = processed[OBS_LANGUAGE_TOKENS].to(device)
    task_masks     = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)
    pure_task_tokens = processed[OBS_LANGUAGE_USER_PROMPT_TOKENS].to(device)
    pure_task_masks = processed[OBS_LANGUAGE_USER_PROMPT_ATTENTION_MASK].to(device)
    subtask_tokens = processed[OBS_LANGUAGE_SUBTASK_TOKENS].to(device)
    subtask_masks  = processed[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK].to(device)
    action_tokens  = processed[ACTION_TOKENS].to(device)
    action_masks   = processed[ACTION_TOKEN_MASK].to(device)

    # Normalise + pad actions: (chunk_size, action_dim) → (1, chunk_size, max_action_dim)
    # processed[ACTION] is already normalised by the NormalizerProcessorStep
    actions_norm = processed[ACTION]  # (1, chunk_size, action_dim) or (1, 1, action_dim)
    # If preprocessor only normalised a single-step action, fall back to normalising the chunk
    if actions_norm.shape[1] != gt_actions.shape[0]:
        # Preprocessor got our full chunk; shape should already match — just in case
        actions_norm = processed[ACTION]
    actions_padded = pad_vector(actions_norm.to(device), policy.config.max_action_dim)

    return (images, img_masks, task_tokens, task_masks, pure_task_tokens, pure_task_masks,
            subtask_tokens, subtask_masks, action_tokens, action_masks, actions_padded)


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
    sites    = [s.strip() for s in cfg.probe_sites.split(",")]
    t_values = [float(t.strip()) for t in cfg.probe_denoising_timesteps.split(",")]
    chunk_size = cfg.policy.chunk_size

    all_prefix = []
    all_suffix = {t: [] for t in t_values}
    metadata   = []

    # Hook on PaliGemmaWithExpertModel (policy.model is PI05Pytorch,
    # policy.model.paligemma_with_expert is PaliGemmaWithExpertModel)
    with _capture_hook(policy.model.paligemma_with_expert) as captured:
        for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
            if i % 100 == 0:
                logging.info(f"  [{i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

            obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
                dataset, global_idx, chunk_size
            )

            inputs = _prepare_inputs(
                policy, preprocessor, obs, gt_actions, gt_subtask, task_str, device
            )
            (images, img_masks, task_tokens, task_masks, pure_task_tokens, pure_task_masks,
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
                    all_prefix.append(_mean_pool(captured["prefix_out"]))  # (1, 2048)

                if "suffix" in sites:
                    raw = captured["suffix_out"]            # (1, seq_len, 1024)
                    sliced = raw[:, -chunk_size:, :]        # (1, chunk_size, 1024)
                    all_suffix[t_val].append(_mean_pool(sliced))  # (1, 1024)

            # Retrieve subtask index for coloring
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

    result = {"metadata": metadata}
    if "prefix" in sites and all_prefix:
        result["prefix"] = torch.cat(all_prefix, dim=0)  # (N, 2048)
    if "suffix" in sites:
        result["suffix"] = {t: torch.cat(v, dim=0) for t, v in all_suffix.items() if v}
    return result


@torch.no_grad()
def collect_subtask_injection(policy, preprocessor, dataset, samples, device, cfg):
    """
    For each frame run two forwards: one with GT subtask tokens, one with
    model-generated subtask tokens. Captures prefix and suffix for both.

    Returns dict with keys:
      "prefix_gt", "prefix_gen":             (N, 2048)
      "suffix_gt", "suffix_gen":             {t_val: (N, 1024)}
      "gen_subtask_texts":                   list of N strings
    """
    t_values   = [float(t.strip()) for t in cfg.probe_denoising_timesteps.split(",")]
    chunk_size = cfg.policy.chunk_size
    tokenizer  = policy.model._paligemma_tokenizer

    prefix_gt, prefix_gen     = [], []
    suffix_gt  = {t: [] for t in t_values}
    suffix_gen = {t: [] for t in t_values}
    gen_subtask_texts = []

    policy.model.suppress_debug_log = True
    with _capture_hook(policy.model.paligemma_with_expert) as captured:
        for i, (ep_idx, fr_idx, global_idx) in enumerate(samples):
            if i % 100 == 0:
                logging.info(f"  [injection {i + 1}/{len(samples)}] ep={ep_idx:04d} fr={fr_idx:04d}")

            obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
                dataset, global_idx, chunk_size
            )
            inputs = _prepare_inputs(
                policy, preprocessor, obs, gt_actions, gt_subtask, task_str, device
            )
            (images, img_masks, task_tokens, task_masks, pure_task_tokens, pure_task_masks,
             gt_sub_tokens, gt_sub_masks, action_tokens, action_masks, actions_padded) = inputs

            # Generate subtask tokens with the model using the pure task prompt
            gen_sub_tokens, gen_sub_masks = policy.model.generate_subtask_tokens(
                images, img_masks, pure_task_tokens, pure_task_masks
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
                        pout = _mean_pool(captured["prefix_out"])  # (1, 2048)
                        (prefix_gt if cond_name == "gt" else prefix_gen).append(pout)

                    sout = _mean_pool(captured["suffix_out"][:, -chunk_size:, :])  # (1, 1024)
                    (suffix_gt if cond_name == "gt" else suffix_gen)[t_val].append(sout)

    policy.model.suppress_debug_log = False
    return {
        "prefix_gt":          torch.cat(prefix_gt,  dim=0),
        "prefix_gen":         torch.cat(prefix_gen, dim=0),
        "suffix_gt":          {t: torch.cat(v, dim=0) for t, v in suffix_gt.items()},
        "suffix_gen":         {t: torch.cat(v, dim=0) for t, v in suffix_gen.items()},
        "gen_subtask_texts":  gen_subtask_texts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_evenly(dataset, n_per_episode, max_episodes, seed):
    """
    Sample n_per_episode evenly spaced frames from each episode.
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
        n       = min(n_per_episode, len(indices))
        chosen  = np.linspace(0, len(indices) - 1, n, dtype=int)
        for pos in chosen:
            global_idx = indices[pos]
            fr_idx = dataset.hf_dataset[global_idx]["frame_index"].item()
            samples.append((ep_idx, fr_idx, global_idx))

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# PCA + UMAP
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(X, n_components, label, pca_dir):
    """
    Fit PCA on X (N, D). Saves a scree plot and returns (X_pca, pca_object).
    n_components is clamped to min(n_components, N, D).
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca    = PCA(n_components=n_components, random_state=0)
    X_pca  = pca.fit_transform(X.numpy())

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
    logging.info(f"    PCA scree → {out_path}  (90% @ {comp90} components, 95% @ {comp95})")

    return torch.from_numpy(X_pca.astype(np.float32)), pca


def run_umap(X_pca, n_components, n_neighbors, min_dist, seed):
    """Fit UMAP and return embedding array (N, n_components)."""
    import warnings
    import umap as umap_lib
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    # n_jobs is silently forced to 1 when random_state is set — suppress the noise
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)
        return reducer.fit_transform(X_pca.numpy())


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots (matplotlib static PNGs)
# ──────────────────────────────────────────────────────────────────────────────

def _ax_style(ax, title):
    import textwrap
    ax.set_title(textwrap.fill(title, width=55), fontsize=9)
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


def plot_2d_by_episode(emb, metadata, output_path):
    from matplotlib.patches import Patch

    ep_ids     = np.array([m["episode_idx"] for m in metadata])
    frame_ids  = np.array([m["frame_idx"]   for m in metadata])
    unique_eps = np.unique(ep_ids)

    # One sequential colormap per episode: dark = early frame, light = late frame
    seq_cmaps = ["Blues", "Reds", "Greens", "Oranges", "Purples",
                 "YlOrBr", "PuRd", "BuGn", "GnBu", "OrRd"]

    fig, ax = plt.subplots(figsize=(7, 6))
    legend_handles = []
    for i, ep in enumerate(unique_eps):
        mask      = ep_ids == ep
        ep_frames = frame_ids[mask]
        fmin, fmax = ep_frames.min(), ep_frames.max()
        # Normalise to [0.9, 0.3]: early frames → 0.9 (dark end), late → 0.3 (pale end)
        # Sequential colormaps go light→dark as value increases, so we invert
        norm     = 0.9 - (ep_frames - fmin) / max(fmax - fmin, 1) * 0.6
        cmap_ep  = matplotlib.colormaps.get_cmap(seq_cmaps[i % len(seq_cmaps)])
        colors   = cmap_ep(norm)
        ax.scatter(emb[mask, 0], emb[mask, 1], c=colors, s=22, alpha=0.85, linewidths=0)
        legend_handles.append(Patch(facecolor=cmap_ep(0.6), label=f"ep {ep}"))

    if len(unique_eps) <= 20:
        ax.legend(handles=legend_handles, fontsize=6, ncol=2)
    n_eps = len(unique_eps)
    _ax_style(ax, f"By episode ({n_eps} eps) — dark→light = early→late frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_frame(emb, metadata, output_path):
    frame_ids  = np.array([m["frame_idx"]   for m in metadata])
    n_eps = len(np.unique([m["episode_idx"] for m in metadata]))
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=frame_ids, cmap="plasma", s=22, alpha=0.80, linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Frame index within episode")
    _ax_style(ax, f"By frame index (temporal position) — {n_eps} episodes pooled")
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
    _ax_style(ax, f"By subtask — {n_eps} episodes pooled")
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
        ax.scatter(emb_gt[j, 0],  emb_gt[j, 1],  c=[c], s=20, alpha=0.85, marker="o", linewidths=0)
        ax.scatter(emb_gen[j, 0], emb_gen[j, 1], c=[c], s=24, alpha=0.70, marker="x", linewidths=0.9)

    legend_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="none", markersize=7, label="GT subtask"),
        Line2D([0], [0], marker="x", color="gray", linestyle="none", markersize=7,
               markeredgewidth=1.2, label="Generated subtask"),
    ]
    n_eps = len(set(m["episode_idx"] for m in metadata))
    ax.legend(handles=legend_handles, fontsize=8)
    _ax_style(ax, f"Subtask injection — GT (●) vs generated (✕), by subtask · {n_eps} episodes pooled")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3D plots (Plotly interactive HTML)
# ──────────────────────────────────────────────────────────────────────────────

def _plotly_scatter3d(emb, color_vals, color_label, hover_texts, title, output_path,
                      colorscale="Turbo", traces=None):
    """
    Save a single-trace (or multi-trace) interactive 3D scatter as HTML.
    Pass `traces` to override the default single-trace construction.
    """
    import plotly.graph_objects as go

    if traces is None:
        traces = [go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="markers",
            marker=dict(
                size=6,
                color=color_vals,
                colorscale=colorscale,
                showscale=True,
                opacity=0.85,
                line=dict(width=0),
                colorbar=dict(title=color_label),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(itemsizing='constant', font=dict(size=14)),
    )
    fig.write_html(output_path)
    logging.info(f"    3D plot → {output_path}")


def plot_3d_by_episode(emb, metadata, output_path):
    import plotly.graph_objects as go
    import numpy as np
    import matplotlib

    ep_ids    = [m["episode_idx"] for m in metadata]
    frame_ids = [m["frame_idx"]   for m in metadata]
    unique_eps = sorted(set(ep_ids))
    n_eps = len(unique_eps)

    seq_cmaps = ["Blues", "Reds", "Greens", "Oranges", "Purples",
                 "YlOrBr", "PuRd", "BuGn", "GnBu", "OrRd"]

    traces = []
    for i, ep in enumerate(unique_eps):
        idx    = [j for j, e in enumerate(ep_ids) if e == ep]
        frames = np.array([frame_ids[j] for j in idx])
        hover  = [f"ep={ep} fr={metadata[j]['frame_idx']}<br>{metadata[j]['subtask']}"
                  for j in idx]
        
        fmin, fmax = frames.min(), frames.max()
        norm = 0.9 - (frames - fmin) / max(fmax - fmin, 1) * 0.6
        cmap_ep = matplotlib.colormaps.get_cmap(seq_cmaps[i % len(seq_cmaps)])
        
        rgba_colors = cmap_ep(norm)
        alpha = 0.85
        plotly_colors = [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})" for r, g, b, _ in rgba_colors]
        
        traces.append(go.Scatter3d(
            x=[emb[j, 0] for j in idx],
            y=[emb[j, 1] for j in idx],
            z=[emb[j, 2] for j in idx],
            mode="markers",
            name=f"ep {ep}",
            marker=dict(
                size=6,
                color=plotly_colors,
                line=dict(width=0),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    _plotly_scatter3d(
        emb, None, None, None,
        f"3D — by episode ({n_eps} eps) · dark→pale = early→late frame",
        output_path,
        traces=traces,
    )


def plot_3d_by_frame(emb, metadata, output_path):
    import plotly.graph_objects as go

    frame_ids = [m["frame_idx"] for m in metadata]
    n_eps = len(set(m["episode_idx"] for m in metadata))
    hover = [f"ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}" for m in metadata]
    
    traces = [go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode="markers",
        marker=dict(
            size=6,
            color=frame_ids,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Frame index within episode"),
            opacity=0.85,
            line=dict(width=0),
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    )]
    
    _plotly_scatter3d(
        emb, None, None, None,
        f"3D — by frame index (temporal position) — {n_eps} episodes pooled",
        output_path,
        traces=traces,
    )


def plot_3d_by_subtask(emb, metadata, output_path):
    sub_ids = [m["subtask_idx"] for m in metadata]
    n_eps   = len(set(m["episode_idx"] for m in metadata))
    hover   = [f"ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}" for m in metadata]
    _plotly_scatter3d(emb, sub_ids, "Subtask ID", hover,
                      f"3D — by subtask ({n_eps} episodes pooled)",
                      output_path, colorscale="Rainbow")


def plot_3d_two_episodes(emb, metadata, ep_a, ep_b, output_path):
    """
    Only points from ep_a and ep_b. ep_a = circles, ep_b = squares.
    Coloured by frame index within episode. Each episode is a separate trace.
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
            marker=dict(
                size=6,
                color=frames,
                colorscale="Plasma",
                showscale=(ep == ep_a),
                opacity=0.85,
                line=dict(width=0),
                colorbar=dict(title="Frame index"),
                symbol=sym,
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    _plotly_scatter3d(
        emb, None, None, None,
        f"3D — episode {ep_a} (●) vs {ep_b} (■) · coloured by frame index",
        output_path,
        traces=traces,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plotting pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _makedirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _grid_layout(n):
    """Return (rows, cols) for a near-square grid that fits n cells, wider than tall."""
    import math
    rows = max(1, math.floor(math.sqrt(n)))
    cols = math.ceil(n / rows)
    return rows, cols


def _save_episode_thumbnails(dataset, ep_to_indices, output_dir):
    """Save a single montage image with first-frame thumbnails for every episode."""
    # Find all image observation keys
    sample_frame = dataset[0]
    img_keys = sorted(k for k in sample_frame.keys()
                      if k.startswith("observation.image") and isinstance(sample_frame[k], torch.Tensor))
    if not img_keys:
        logging.warning("No image observation keys found; skipping episode thumbnails.")
        return

    episodes = sorted(ep_to_indices.keys())
    n_eps  = len(episodes)
    n_cams = len(img_keys)
    cam_labels = [k.replace("observation.images.", "").replace("observation.image.", "")
                  for k in img_keys]

    # Layout: one row per episode, one column per camera.
    # When there is only a single camera use a square-ish grid instead.
    if n_cams == 1:
        rows, cols = _grid_layout(n_eps)
    else:
        rows, cols = n_eps, n_cams

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)

    for i, ep_idx in enumerate(episodes):
        frame = dataset[ep_to_indices[ep_idx][0]]
        r, c_start = (i, 0) if n_cams > 1 else divmod(i, cols)
        for cam_j, (key, label) in enumerate(zip(img_keys, cam_labels)):
            c = c_start + cam_j if n_cams == 1 else cam_j
            ax = axes[r][c]
            img    = frame[key]  # (C, H, W)
            img_np = img.float().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            ax.imshow(img_np)
            ax.axis("off")
            title = f"ep {ep_idx}" if n_cams == 1 else (f"ep {ep_idx} · {label}" if i == 0 or True else label)
            ax.set_title(title, fontsize=7)

    # Hide any unused cells (single-camera case only)
    if n_cams == 1:
        for j in range(n_eps, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].set_visible(False)

    fig.suptitle(f"Episode thumbnails  ({n_eps} episodes)", fontsize=10)
    fig.tight_layout(pad=0.3)
    out_path = os.path.join(output_dir, "episode_thumbnails.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Episode thumbnails ({n_eps} episodes) → {out_path}")


def _run_site(tag, X, metadata, cfg, pca_dir, output_dir):
    """PCA → 2D UMAP → 3D UMAP → all standard plots for one activation tensor."""
    logging.info(f"  Site '{tag}'  shape={tuple(X.shape)}")

    d2 = os.path.join(output_dir, "2d", tag)
    d3 = os.path.join(output_dir, "3d", tag)
    _makedirs(d2, d3)

    X_pca, _ = run_pca(X, cfg.probe_pca_dims, tag, pca_dir)

    logging.info(f"    Fitting 2D UMAP …")
    emb2 = run_umap(X_pca, 2, cfg.probe_umap_n_neighbors, cfg.probe_umap_min_dist, cfg.probe_umap_seed)
    plot_2d_by_episode(emb2, metadata, os.path.join(d2, "by_episode.png"))
    plot_2d_by_frame(emb2, metadata,   os.path.join(d2, "by_frame.png"))
    plot_2d_by_subtask(emb2, metadata, os.path.join(d2, "by_subtask.png"))

    logging.info(f"    Fitting 3D UMAP …")
    emb3 = run_umap(X_pca, 3, cfg.probe_umap_n_neighbors, cfg.probe_umap_min_dist, cfg.probe_umap_seed)
    plot_3d_by_episode(emb3, metadata, os.path.join(d3, "by_episode.html"))
    plot_3d_by_frame(emb3, metadata,   os.path.join(d3, "by_frame.html"))
    plot_3d_by_subtask(emb3, metadata, os.path.join(d3, "by_subtask.html"))
    plot_3d_two_episodes(
        emb3, metadata, cfg.probe_ep_3d_a, cfg.probe_ep_3d_b,
        os.path.join(d3, f"ep{cfg.probe_ep_3d_a}_vs_ep{cfg.probe_ep_3d_b}.html"),
    )


def run_plotting(cache, cfg, output_dir):
    """Load cached activations, run all PCA+UMAP reductions, save plots."""
    metadata = cache["metadata"]
    sites    = [s.strip() for s in cfg.probe_sites.split(",")]
    t_values = [float(t.strip()) for t in cfg.probe_denoising_timesteps.split(",")]
    pca_dir  = os.path.join(output_dir, "pca_variance")
    _makedirs(pca_dir)

    # ── Standard sites ──────────────────────────────────────────────────────
    if "prefix" in sites and "prefix" in cache:
        _run_site("prefix", cache["prefix"], metadata, cfg, pca_dir, output_dir)

    if "suffix" in sites and "suffix" in cache:
        for t_val in t_values:
            if t_val not in cache["suffix"]:
                continue
            _run_site(f"suffix_t{t_val}", cache["suffix"][t_val], metadata, cfg, pca_dir, output_dir)

    # ── Subtask injection ───────────────────────────────────────────────────
    has_injection = "prefix_gt" in cache and "prefix_gen" in cache
    if not has_injection:
        return

    logging.info("  Plotting subtask injection …")
    n = len(metadata)

    # ── Save generated subtask CSV ───────────────────────────────────────────
    gen_texts_all = cache.get("gen_subtask_texts", [])
    if gen_texts_all:
        import csv
        csv_path = os.path.join(output_dir, "subtask_injection", "generated_subtasks.csv")
        _makedirs(os.path.dirname(csv_path))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode_idx", "frame_idx", "global_idx", "gt_subtask", "gen_subtask"]
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
        logging.info(f"  Generated subtasks CSV → {csv_path}")

    def _injection_plots(tag, X_gt, X_gen):
        inj2 = os.path.join(output_dir, "subtask_injection", tag, "2d")
        inj3 = os.path.join(output_dir, "subtask_injection", tag, "3d")
        _makedirs(inj2, inj3)

        X_combined = torch.cat([X_gt, X_gen], dim=0)
        # Label clarifies this PCA is fit on GT+Gen combined (2×N samples)
        pca_label  = f"{tag}_inj_GT+Gen"
        X_pca, _   = run_pca(X_combined, cfg.probe_pca_dims, pca_label, pca_dir)

        logging.info(f"    Fitting 2D UMAP for {tag} injection …")
        emb2 = run_umap(X_pca, 2, cfg.probe_umap_n_neighbors, cfg.probe_umap_min_dist, cfg.probe_umap_seed)
        plot_2d_subtask_injection(emb2[:n], emb2[n:], metadata, os.path.join(inj2, "gen_vs_gt.png"))

        logging.info(f"    Fitting 3D UMAP for {tag} injection …")
        emb3 = run_umap(X_pca, 3, cfg.probe_umap_n_neighbors, cfg.probe_umap_min_dist, cfg.probe_umap_seed)

        gen_texts = cache.get("gen_subtask_texts", [""] * n)
        hover = (
            [f"GT  | ep={m['episode_idx']} fr={m['frame_idx']}<br>{m['subtask']}"   for m in metadata] +
            [f"GEN | ep={m['episode_idx']} fr={m['frame_idx']}<br>{gen_texts[i]}" for i, m in enumerate(metadata)]
        )
        condition_colors = [0] * n + [1] * n
        _plotly_scatter3d(
            emb3, condition_colors, "0=GT / 1=Gen", hover,
            f"3D — {tag}: GT (blue) vs generated (red) subtask · {n} frames, {len(set(m['episode_idx'] for m in metadata))} episodes",
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
    Outputs land in ds_dir (a subdirectory named after the dataset root).
    policy / preprocessor may be None when probe_mode == "plot".
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # local to avoid circular imports

    cache_path = os.path.join(ds_dir, "activations_cache.pt")
    _makedirs(ds_dir)
    cache = None

    if cfg.probe_mode in ("collect", "all"):
        logging.info("  Building sample list …")
        samples = _sample_evenly(
            dataset,
            n_per_episode=cfg.probe_n_frames_per_episode,
            max_episodes=cfg.probe_max_episodes,
            seed=cfg.probe_random_seed,
        )
        ep_to_indices = _build_episode_index(dataset)
        sampled_eps   = {ep for ep, _, _ in samples}
        ep_to_indices_sampled = {ep: ep_to_indices[ep] for ep in sampled_eps if ep in ep_to_indices}
        _save_episode_thumbnails(dataset, ep_to_indices_sampled, ds_dir)

        logging.info(f"  Collecting activations for {len(samples)} frames …")
        cache = collect_activations(policy, preprocessor, dataset, samples, device, cfg)

        if cfg.probe_subtask_injection:
            logging.info("  Collecting subtask injection activations …")
            inj = collect_subtask_injection(policy, preprocessor, dataset, samples, device, cfg)
            cache.update(inj)

        torch.save(cache, cache_path)
        logging.info(f"  Activations saved → {cache_path}")

    if cfg.probe_mode in ("plot", "all"):
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
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    init_logging()
    device     = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = cfg.probe_output_dir
    _makedirs(output_dir)

    # Load policy once; reuse across all datasets.
    policy = preprocessor = primary_dataset = None
    if cfg.probe_mode in ("collect", "all"):
        logging.info("Loading policy and primary dataset …")
        policy, preprocessor, primary_dataset = _load_policy_and_processors(cfg, device)

    # ── Primary dataset ────────────────────────────────────────────────────────
    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    logging.info(f"=== Dataset: {primary_name} ===")
    _probe_one_dataset(policy, preprocessor, primary_dataset,
                       os.path.join(output_dir, primary_name), cfg, device)

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
        _probe_one_dataset(policy, preprocessor, extra_ds,
                           os.path.join(output_dir, ds_name), cfg, device)

    logging.info("All datasets done.")


if __name__ == "__main__":
    probe_cli()
