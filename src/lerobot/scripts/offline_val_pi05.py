#!/usr/bin/env python
"""
Periodic validation pipeline for offline Pi05 RL training.

This module is designed to be imported from offline_learner_pi05.py (or any
other training script) without modifying that file. It exposes three public
functions that together implement periodic validation using eight probe
scripts (actions, representations, attention, offline_inference, spatial_memorization,
action_drift_jacobian, spatial_memorization_jacobian, critic_values_distribution).

Each probe can be individually enabled/disabled via ProbeConfig flags
(enable_actions, enable_representations, etc.). Raw probe data is saved
as a .pt file at each validation step for post-hoc analysis.

────────────────────────────────────────────────────────────────────────────────
Architecture
────────────────────────────────────────────────────────────────────────────────

  load_val_dataset(cfg, main_dataset=None)
      → (val_dataset, val_ep_indices) or (None, None)
        • val_dataset_path set  → load a separate LeRobotDataset
        • val_split > 0         → reuse main_dataset; val_ep_indices is a set
                                   of the last N episode indices
        • neither               → (None, None) — validation disabled

  init_action_manifold(val_dataset, val_ep_indices, cfg, device, output_dir)
      → manifold_cache dict or None
        Phase 1 (run ONCE at startup, no model):
          sample GT frames → collect_gt_reference() → fit_manifold()
          (PCA + UMAP fit on reference GT actions — NEVER re-fit)

  run_validation(policy, preprocessor, postprocessor,
                 val_dataset, val_ep_indices, manifold_cache,
                 cfg, step, output_dir, wandb_logger, device)
        Call from rank-0 only (caller guards with accelerator.is_main_process).
        Calls policy.eval() → runs three probes in isolation → policy.train().
        Each probe is wrapped in its own try/except so one failure does not
        skip the others.

────────────────────────────────────────────────────────────────────────────────
Call-graph of run_validation()
────────────────────────────────────────────────────────────────────────────────

  run_validation()
    ├─ policy.eval()
    ├─ _run_probe_actions()                [if enable_actions]
    │    ├─ _sample_val_episodes()
    │    ├─ collect_eval_dataset()          [actions_pi05, @no_grad]
    │    ├─ run_plotting()                  [actions_pi05]
    │    └─ compute_nn_distances()         → WandB scalar + raw embeddings
    ├─ _run_probe_representations()        [if enable_representations]
    │    ├─ _sample_val_episodes()
    │    ├─ collect_activations()           [representations_pi05, @no_grad]
    │    ├─ collect_subtask_injection()     [optional, @no_grad]
    │    └─ run_plotting()                  [representations_pi05]
    ├─ _run_probe_attention()              [if enable_attention]
    │    ├─ _build_attn_sample_list()
    │    └─ per-episode/batch loop → MP4
    ├─ _run_probe_offline_inference()      [if enable_offline_inference]
    │    └─ per-frame inference → render + raw pred/GT
    ├─ _run_probe_spatial_memorization()   [if enable_spatial_memorization]
    │    └─ 1-per-episode sampling → multi-layer attn → aggregate stats → PNG
    ├─ _run_probe_action_drift_jacobian()  [if enable_action_drift_jacobian]
    │    └─ per-frame A*|dA/d(action)| causal maps → MP4
    ├─ _run_probe_spatial_memorization_jacobian()  [if enable_spatial_memorization_jacobian]
    │    └─ 1-per-episode → multi-layer Jacobian → aggregate causal stats → PNG
    ├─ _run_probe_critic_values_distribution()     [if enable_critic_values_distribution]
    │    └─ V(s)/TD-error histograms + ||dV/dvision|| percentile frames → PNGs
    ├─ torch.save(raw_data, "probe_raw_data.pt")
    ├─ log_dict() to WandB
    └─ finally: policy.train()

────────────────────────────────────────────────────────────────────────────────
manifold_cache lifecycle
────────────────────────────────────────────────────────────────────────────────
  Built once at startup by init_action_manifold().
  Passed unchanged into every run_validation() call.
  NEVER re-fit — pca.transform() / reducer.transform() are stable references.

  UMAP for probe_representations IS re-fit at every val step inside
  run_plotting() (via run_umap() which calls fit_transform). This is
  intentional: we want to track representation drift honestly.

────────────────────────────────────────────────────────────────────────────────
DDP / Accelerate note
────────────────────────────────────────────────────────────────────────────────
  run_validation() must be called only from rank-0.
  The caller should guard with:

      if accelerator.is_main_process and val_dataset is not None:
          if step % val_freq == 0:
              run_validation(...)

  No barrier synchronisation is performed inside this module.

────────────────────────────────────────────────────────────────────────────────
Wiring into offline_learner_pi05.py (suggested, not required now)
────────────────────────────────────────────────────────────────────────────────
  After the dataset is loaded and before the training loop:

      from lerobot.scripts.offline_val_pi05 import (
          load_val_dataset, init_action_manifold, run_validation
      )
      val_dataset, val_ep_indices = load_val_dataset(cfg, main_dataset=offline_dataset)
      manifold_cache = None
      if is_main_process and val_dataset is not None:
          manifold_cache = init_action_manifold(
              val_dataset, val_ep_indices, cfg, device, output_dir
          )

  Inside the training loop, after pi05_update_step:

      val_freq = getattr(cfg, 'val_freq', 1000)
      if is_main_process and val_dataset is not None and optimization_step % val_freq == 0:
          run_validation(
              policy=accelerator.unwrap_model(policy),
              preprocessor=preprocessor,
              postprocessor=postprocessor,
              val_dataset=val_dataset,
              val_ep_indices=val_ep_indices,
              manifold_cache=manifold_cache,
              cfg=cfg,
              step=optimization_step,
              output_dir=output_dir,
              wandb_logger=wandb_logger,
              device=device,
          )

  Suggested config fields to add to OfflineTrainRLServerPipelineConfig:

      val_dataset_path: str | None = None
      val_split: float = 0.0
      val_freq: int = 1000
"""

import csv
import gc
import logging
import os
import random as random_mod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from lerobot.probes.offline_inference_pi05 import (
    _build_episode_index,
    get_frame_data,
    normalize_gt,
    render_sample,
    run_inference,
    SO100_JOINT_NAMES,
)
from lerobot.probes.utils_pi05 import makedirs
from lerobot.probes.representations_pi05 import _save_episode_thumbnails

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Standalone config dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OfflineValConfig:
    """
    Validation configuration for offline Pi05 training.

    These fields are intended to be added to OfflineTrainRLServerPipelineConfig
    (in offline_learner_pi05.py) or accessed via getattr with defaults:

        val_freq      = getattr(cfg, 'val_freq',      1000)
        val_split     = getattr(cfg, 'val_split',     0.0)
        val_dataset_path = getattr(cfg, 'val_dataset_path', None)

    Probe-specific parameters (max_episodes, n_frames_per_episode, pca_dims, etc.)
    live on cfg.probe_parameters (ProbeConfig), which is already part of
    TrainRLServerPipelineConfig — no duplication needed here.
    """
    val_dataset_path: Optional[str] = None  # path to a separate validation LeRobotDataset
    val_split: float = 0.0                  # fraction of main-dataset episodes to hold out
    val_freq: int = 1000                    # optimization steps between validation runs


# ──────────────────────────────────────────────────────────────────────────────
# Helper: read val config fields from a training cfg (with safe defaults)
# ──────────────────────────────────────────────────────────────────────────────

def _val_cfg(cfg) -> OfflineValConfig:
    """Extract OfflineValConfig fields from a training cfg via getattr with defaults."""
    return OfflineValConfig(
        val_dataset_path=getattr(cfg, "val_dataset_path", None),
        val_split=float(getattr(cfg, "val_split", 0.0)),
        val_freq=int(getattr(cfg, "val_freq", 1000)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_val_dataset(cfg, main_dataset=None):
    """
    Load (or identify) the validation dataset.

    Returns
    -------
    val_dataset    : LeRobotDataset or None
    val_ep_indices : set of episode indices to use (None = use all episodes)

    Cases:
      - cfg.val_dataset_path is set → load a separate dataset; val_ep_indices=None
      - cfg.val_split > 0           → use main_dataset; val_ep_indices = last N episodes
      - neither                     → (None, None) — validation disabled
    """
    vc = _val_cfg(cfg)

    if vc.val_dataset_path:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logging.info(f"[VAL] Loading separate val dataset from {vc.val_dataset_path}")
        val_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=vc.val_dataset_path,
        )
        val_dataset.delta_timestamps = None
        val_dataset.delta_indices = None
        return val_dataset, None

    if vc.val_split > 0:
        if main_dataset is None:
            logging.warning(
                "[VAL] val_split > 0 but main_dataset was not provided; "
                "skipping validation."
            )
            return None, None
        ep_to_indices = _build_episode_index(main_dataset)
        all_episodes = sorted(ep_to_indices.keys())
        n_val = max(1, int(vc.val_split * len(all_episodes)))
        val_ep_indices = set(all_episodes[-n_val:])
        logging.info(
            f"[VAL] Using last {n_val}/{len(all_episodes)} episodes as val split "
            f"(val_split={vc.val_split:.2f})"
        )
        return main_dataset, val_ep_indices

    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_val_episodes(val_dataset, val_ep_indices,
                          n_per_episode, max_episodes, seed):
    """
    Sample *n_per_episode* evenly-spaced frames from each val episode.

    If *val_ep_indices* is not None, only those episodes are included.
    If *max_episodes* is not None, a random subset of up to that many episodes
    is drawn using *seed* for reproducibility.

    Returns list of (episode_idx, frame_idx_in_episode, global_idx).
    """
    ep_to_indices = _build_episode_index(val_dataset)

    if val_ep_indices is not None:
        episodes = sorted(e for e in ep_to_indices.keys() if e in val_ep_indices)
    else:
        episodes = sorted(ep_to_indices.keys())

    if not episodes:
        return []

    if max_episodes is not None:
        rng = np.random.RandomState(seed)
        episodes = sorted(
            rng.choice(
                episodes, size=min(max_episodes, len(episodes)), replace=False
            ).tolist()
        )

    samples = []
    for ep_idx in episodes:
        indices = ep_to_indices[ep_idx]
        n = min(n_per_episode, len(indices))
        chosen = np.linspace(0, len(indices) - 1, n, dtype=int)
        for pos in chosen:
            global_idx = indices[pos]
            fr_idx = val_dataset.hf_dataset[global_idx]["frame_index"].item()
            samples.append((ep_idx, fr_idx, global_idx))

    return samples


def _build_attn_sample_list(val_dataset, val_ep_indices, random_n, subsample, seed):
    """
    Build per-episode sample lists for the attention probe.

    Similar to build_sample_list in attention_pi05 but respects
    val_ep_indices and does not require an external import of that function.

    Returns [(ep_idx, [(fr_idx, global_idx), ...]), ...].
    """
    ep_to_indices = _build_episode_index(val_dataset)

    if val_ep_indices is not None:
        all_eps = sorted(e for e in ep_to_indices.keys() if e in val_ep_indices)
    else:
        all_eps = sorted(ep_to_indices.keys())

    if random_n:
        rng = random_mod.Random(seed)
        shuffled = list(all_eps)
        rng.shuffle(shuffled)
        all_eps = shuffled[:random_n]

    samples = []
    for ep_idx in sorted(all_eps):
        indices = ep_to_indices[ep_idx]
        ep_frames = [
            (fr_idx, indices[fr_idx])
            for fr_idx in range(0, len(indices), subsample)
        ]
        if ep_frames:
            samples.append((ep_idx, ep_frames))

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — reference action manifold (run once at startup, no model)
# ──────────────────────────────────────────────────────────────────────────────

def init_action_manifold(val_dataset, val_ep_indices, cfg, device, output_dir):
    """
    Fit PCA + UMAP on reference GT action chunks from the val dataset.

    Run this ONCE at training startup, before the main loop.
    The returned manifold_cache is passed unchanged to every run_validation() call.
    The PCA and UMAP reducers are NEVER re-fit.

    Parameters
    ----------
    val_dataset    : LeRobotDataset or None
    val_ep_indices : set[int] or None
    cfg            : training cfg — must have cfg.probe_parameters, cfg.policy.chunk_size
    device         : torch.device (unused here — no model forward)
    output_dir     : base output directory for training (manifold saved under
                     {output_dir}/validation/manifold/)

    Returns
    -------
    dict with keys: pca, reducer2d, reducer3d, ref_emb2, ref_emb3, ref_metadata
    or None if val_dataset is None or no samples are found.
    """
    if val_dataset is None:
        return None

    from lerobot.probes.actions_pi05 import collect_gt_reference, fit_manifold

    p = cfg.probe_parameters
    manifold_dir = os.path.join(output_dir, "validation", "manifold")
    pca_dir = os.path.join(manifold_dir, "pca_variance")
    makedirs(manifold_dir, pca_dir)

    ref_samples = _sample_val_episodes(
        val_dataset, val_ep_indices,
        n_per_episode=p.ref_n_frames_per_episode,
        max_episodes=p.ref_max_episodes,
        seed=p.random_seed,
    )

    if not ref_samples:
        logging.warning(
            "[VAL] init_action_manifold: no reference samples found; "
            "action probe will be skipped at every val step."
        )
        return None

    ref_data = collect_gt_reference(val_dataset, ref_samples, cfg.policy.chunk_size)

    pca, reducer2d, reducer3d, ref_emb2, ref_emb3 = fit_manifold(
        ref_data["gt"], cfg, pca_dir
    )
    return {
        "pca":          pca,
        "reducer2d":    reducer2d,
        "reducer3d":    reducer3d,
        "ref_emb2":     ref_emb2,
        "ref_emb3":     ref_emb3,
        "ref_metadata": ref_data["metadata"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Individual probe runners (each in its own try/except at the call site)
# ──────────────────────────────────────────────────────────────────────────────

def _run_probe_actions(policy, preprocessor, postprocessor,
                        val_dataset, val_ep_indices, manifold_cache,
                        cfg, output_dir, device):
    """
    Run probe_actions at a single validation step.

    Phase 2 only (Phase 1 manifold was already fit at startup).
    Projects GT and predicted action chunks onto the fixed reference manifold
    via pca.transform() → reducer.transform(). Writes 2D/3D plots and CSV.

    Returns median_pred_nn / median_gt_nn ratio (float) or None on failure.
    """
    from lerobot.probes.actions_pi05 import (
        collect_eval_dataset,
        run_plotting,
        compute_nn_distances,
    )

    makedirs(output_dir)
    p = cfg.probe_parameters

    eval_samples = _sample_val_episodes(
        val_dataset, val_ep_indices,
        n_per_episode=p.n_frames_per_episode,
        max_episodes=p.max_episodes,
        seed=p.random_seed,
    )
    if not eval_samples:
        logging.warning("[VAL] probe_actions: no eval samples found")
        return None

    ds_cache = collect_eval_dataset(
        policy, preprocessor, postprocessor,
        val_dataset, eval_samples,
        manifold_cache["pca"],
        manifold_cache["reducer2d"],
        manifold_cache["reducer3d"],
        device, cfg,
    )

    # Build the full cache structure expected by run_plotting
    full_cache = {
        "pca":          manifold_cache["pca"],
        "reducer2d":    manifold_cache["reducer2d"],
        "reducer3d":    manifold_cache["reducer3d"],
        "ref_emb2":     manifold_cache["ref_emb2"],
        "ref_emb3":     manifold_cache["ref_emb3"],
        "ref_metadata": manifold_cache["ref_metadata"],
        "datasets":     {"val": ds_cache},
    }

    run_plotting(full_cache, cfg, output_dir)

    # Compute NN-distance ratio scalar directly from embeddings (avoids CSV read)
    gt_nn   = compute_nn_distances(ds_cache["gt_emb2"],   manifold_cache["ref_emb2"])
    pred_nn = compute_nn_distances(ds_cache["pred_emb2"], manifold_cache["ref_emb2"])
    ratio   = float(np.median(pred_nn) / (np.median(gt_nn) + 1e-8))

    raw = {
        "gt_emb2":   torch.as_tensor(ds_cache["gt_emb2"]),
        "pred_emb2": torch.as_tensor(ds_cache["pred_emb2"]),
        "gt_emb3":   torch.as_tensor(ds_cache["gt_emb3"]),
        "pred_emb3": torch.as_tensor(ds_cache["pred_emb3"]),
        "nn_distances_gt":   torch.as_tensor(gt_nn),
        "nn_distances_pred": torch.as_tensor(pred_nn),
        "metadata": ds_cache["metadata"],
    }
    return ratio, raw


def _run_probe_representations(policy, preprocessor,
                                 val_dataset, val_ep_indices,
                                 cfg, output_dir, device):
    """
    Run probe_representations at a single validation step.

    Collects prefix (VLM hidden states) and suffix (expert hidden states) at
    multiple denoising timesteps via forward hooks, then PCA + UMAP + plots.
    UMAP is re-fit here at every val step (intentional — tracks drift).
    """
    from lerobot.probes.representations_pi05 import (
        collect_activations,
        collect_subtask_injection,
        run_plotting,
    )

    makedirs(output_dir)
    p = cfg.probe_parameters

    samples = _sample_val_episodes(
        val_dataset, val_ep_indices,
        n_per_episode=p.n_frames_per_episode,
        max_episodes=p.max_episodes,
        seed=p.random_seed,
    )
    if not samples:
        logging.warning("[VAL] probe_representations: no samples found")
        return

    cache = collect_activations(policy, preprocessor, val_dataset, samples, device, cfg)

    if p.subtask_injection:
        inj = collect_subtask_injection(
            policy, preprocessor, val_dataset, samples, device, cfg
        )
        cache.update(inj)

    run_plotting(cache, cfg, output_dir)

    # Build raw data dict for .pt saving
    raw = {"metadata": cache.get("metadata")}
    for key in ("prefix", "prefix_gt", "prefix_gen"):
        if key in cache and cache[key] is not None:
            raw[key] = cache[key] if isinstance(cache[key], torch.Tensor) else torch.as_tensor(cache[key])
    for key in ("suffix", "suffix_gt", "suffix_gen"):
        if key in cache and isinstance(cache[key], dict):
            raw[key] = {t: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v))
                        for t, v in cache[key].items()}
    if "gen_subtask_texts" in cache:
        raw["gen_subtask_texts"] = cache["gen_subtask_texts"]
    return raw


@torch.no_grad()
def _run_probe_attention(policy, preprocessor,
                          val_dataset, val_ep_indices,
                          cfg, output_dir, device):
    """
    Run probe_attention at a single validation step.

    Captures attention weights from multiple layers at the configured timestep
    and writes MP4 videos of attention overlays (layer 0) or standalone
    heatmaps (deeper layers) plus full attention matrices.

    This function replicates the _probe_dataset inner loop from
    attention_pi05.probe_cli, filtered to val_ep_indices.

    The @torch.no_grad() decorator ensures all forward passes inside are safe
    even though the individual called functions also manage this.
    """
    import imageio
    from lerobot.probes.attention_pi05 import (
        embed_probe_prefix,
        probe_forward,
        render_image_overlays,
        render_action_to_prefix_matrix,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    from lerobot.types import TransitionKey

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    attn_layers = [int(x.strip()) for x in p.spatial_layers.split(",")]

    samples = _build_attn_sample_list(
        val_dataset,
        val_ep_indices,
        random_n=p.max_episodes,
        subsample=p.attn_eval_subsample,
        seed=p.random_seed,
    )
    if not samples:
        logging.warning("[VAL] probe_attention: no samples found")
        return

    fps      = getattr(val_dataset, "fps", 30) / p.attn_eval_subsample
    batch_sz = p.validation_batch_size

    for ep_idx, ep_frames in samples:
        # writers[layer_idx][video_key] = imageio writer
        writers = {l: {} for l in attn_layers}

        for batch_start in range(0, len(ep_frames), batch_sz):
            batch_slice = ep_frames[batch_start : batch_start + batch_sz]

            # ── Gather batch observations ──────────────────────────────────
            b_obs: dict[str, list] = {}
            b_task_str: list[str]  = []
            for fr_idx, global_idx in batch_slice:
                obs, _, _state, _gt_subtask, task_str, _, _ = get_frame_data(
                    val_dataset, global_idx, chunk_size
                )
                b_task_str.append(task_str)
                for k, v in obs.items():
                    b_obs.setdefault(k, []).append(v)

            # Stack tensors and move to device
            b_obs_batched = {k: torch.cat(v, dim=0).to(device) for k, v in b_obs.items()}

            # ── Preprocessor ────────────────────────────────────────────────
            complementary_data = {
                "task":      b_task_str,
                "subtask":   [""] * len(batch_slice),
                "advantage": torch.ones((len(batch_slice), 1), device=device),
            }
            dummy_action   = torch.zeros(len(batch_slice), 1, 6, device=device)
            batch_for_proc = {
                TransitionKey.ACTION:             dummy_action,
                **b_obs_batched,
                TransitionKey.COMPLEMENTARY_DATA: complementary_data,
            }
            processed = preprocessor(batch_for_proc)

            # ── Embed prefix (once per batch) ────────────────────────────────
            images, img_masks = policy._preprocess_images(b_obs_batched)
            task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
            task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

            subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
                images, img_masks, task_tokens, task_masks
            )

            prefix_cache = embed_probe_prefix(
                policy, images, img_masks,
                task_tokens, task_masks,
                subtask_tokens, subtask_masks,
            )

            # ── Forward + per-layer render ────────────────────────────────────
            attn_by_layer, segments, pad_masks, patches_per_cam = probe_forward(
                prefix_cache, p.timestep, device
            )
            if not attn_by_layer:
                logging.warning(
                    f"    [attn] No attention captured at t={p.timestep}; skipping."
                )
                continue

            for layer_idx in attn_layers:
                if layer_idx not in attn_by_layer:
                    continue

                attn_weights = attn_by_layer[layer_idx]
                use_overlay = (layer_idx == 0)

                ep_dir = os.path.join(output_dir, f"ep{ep_idx:04d}_L{layer_idx:02d}")
                makedirs(ep_dir)

                csv_path = os.path.join(ep_dir, "norm_consts.csv")
                csv_file  = open(csv_path, "a", newline="")
                csv_writer = csv.writer(csv_file)
                if os.path.getsize(csv_path) == 0:
                    csv_writer.writerow(["ep", "fr", "layer", "panel", "vmax"])

                for b_idx, (fr_idx, _) in enumerate(batch_slice):
                    a_w = attn_weights[b_idx : b_idx + 1]
                    p_m = pad_masks[b_idx : b_idx + 1]
                    i_t = [img[b_idx : b_idx + 1] for img in images]
                    t_t = task_tokens   [b_idx : b_idx + 1]
                    s_t = subtask_tokens[b_idx : b_idx + 1]

                    heatmap_frames, norm_consts = render_image_overlays(
                        a_w, segments, i_t, p_m, patches_per_cam,
                        overlay=use_overlay,
                    )
                    frames_out = dict(heatmap_frames)

                    matrix_frames, matrix_norms = render_action_to_prefix_matrix(
                        a_w, segments, p_m,
                        t_t, s_t, policy.model._paligemma_tokenizer,
                    )
                    frames_out.update(matrix_frames)
                    norm_consts.update(matrix_norms)

                    for panel, vmax in norm_consts.items():
                        csv_writer.writerow(
                            [ep_idx, fr_idx, layer_idx, panel, f"{vmax:.6e}"]
                        )

                    for key, frame_np in frames_out.items():
                        if key not in writers[layer_idx]:
                            mp4_path = os.path.join(ep_dir, f"{key}.mp4")
                            writers[layer_idx][key] = imageio.get_writer(
                                mp4_path, fps=fps, macro_block_size=1
                            )
                        writers[layer_idx][key].append_data(frame_np)

                csv_file.close()

            del attn_by_layer, segments, pad_masks, patches_per_cam
            del prefix_cache, b_obs_batched, task_tokens, task_masks
            del subtask_tokens, subtask_masks, images, img_masks
            torch.cuda.empty_cache()

        for layer_writers in writers.values():
            for w in layer_writers.values():
                w.close()

# ──────────────────────────────────────────────────────────────────────────────
# Probe: offline inference (per-frame GT vs predicted action traces)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_probe_offline_inference(
    policy, preprocessor, postprocessor,
    val_dataset, val_ep_indices,
    cfg, output_dir, device,
) -> float | None:
    """
    Run per-frame offline evaluation: inference → render GT vs predicted action
    traces per joint.  Returns mean MSE (unnormalised) for WandB logging.

    Produces two subdirectories under *output_dir*:
        unnormalized/ep{ep:04d}_fr{fr:04d}.png
        normalized/ep{ep:04d}_fr{fr:04d}.png
    """
    probe_cfg = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    action_encoding = getattr(cfg.policy, "action_encoding", "absolute")

    samples = _sample_val_episodes(
        val_dataset, val_ep_indices,
        n_per_episode=getattr(probe_cfg, "offline_inference_n_frames", 5),
        max_episodes=getattr(probe_cfg, "max_episodes", 6),
        seed=getattr(probe_cfg, "seed", 42),
    )
    if not samples:
        logging.warning("[VAL] offline_inference: no samples selected.")
        return None

    dir_unnorm = os.path.join(output_dir, "unnormalized")
    dir_norm = os.path.join(output_dir, "normalized")
    makedirs(dir_unnorm)
    makedirs(dir_norm)

    action_dim = None
    mse_values: list[float] = []
    all_pred_unnorm: list[torch.Tensor] = []
    all_gt_actions: list[torch.Tensor] = []
    all_metadata: list[dict] = []

    for ep_idx, fr_idx, global_idx in samples:
        obs, gt_actions, state, gt_subtask, task_str, _, _ = get_frame_data(
            val_dataset, global_idx, chunk_size
        )
        gt_actions_norm, _ = normalize_gt(
            preprocessor, gt_actions, state, device, action_encoding=action_encoding,
        )

        pred_unnorm, pred_norm, pred_subtask = run_inference(
            policy, preprocessor, postprocessor, obs, task_str, device,
            state=state, advantage=1.0,
        )

        mse = torch.nn.functional.mse_loss(pred_unnorm, gt_actions.float()).item()
        mse_values.append(mse)
        all_pred_unnorm.append(pred_unnorm.cpu())
        all_gt_actions.append(gt_actions.cpu())
        all_metadata.append({
            "episode_idx": ep_idx, "frame_idx": fr_idx,
            "gt_subtask": gt_subtask, "pred_subtask": pred_subtask,
        })

        if action_dim is None:
            action_dim = gt_actions.shape[-1]

        joint_names = SO100_JOINT_NAMES[:action_dim]

        checkpoints_info = [{
            "label": "pred",
            "subtasks": {"pred": pred_subtask},
            "color_idx": 0,
        }]
        pred_traces = [{
            "actions": pred_unnorm,
            "label": "pred raw",
            "color_idx": 0,
            "kwargs": {"linewidth": 1.2, "linestyle": "-", "alpha": 1.0},
        }]
        pred_traces_norm = [{
            "actions": pred_norm,
            "label": "pred raw",
            "color_idx": 0,
            "kwargs": {"linewidth": 1.2, "linestyle": "-", "alpha": 1.0},
        }]

        common = dict(
            obs=obs,
            gt_subtask=gt_subtask,
            episode_idx=ep_idx,
            frame_idx=fr_idx,
            joint_names=joint_names,
            checkpoints_info=checkpoints_info,
        )

        render_sample(
            **common, gt_actions=gt_actions,
            pred_traces=pred_traces, output_dir=dir_unnorm, state=state,
        )
        render_sample(
            **common, gt_actions=gt_actions_norm,
            pred_traces=pred_traces_norm, output_dir=dir_norm, state=None,
        )

        logging.debug(
            f"  [offline_inference] ep={ep_idx:04d} fr={fr_idx:04d} | "
            f"mse={mse:.4f} | GT: '{gt_subtask}' | pred: '{pred_subtask}'"
        )

    mean_mse = sum(mse_values) / len(mse_values) if mse_values else None
    if mean_mse is not None:
        logging.debug(f"  [offline_inference] mean MSE = {mean_mse:.4f}")

    raw = {
        "pred_unnorm": torch.stack(all_pred_unnorm) if all_pred_unnorm else None,
        "gt_actions":  torch.stack(all_gt_actions) if all_gt_actions else None,
        "mse_per_frame": torch.tensor(mse_values) if mse_values else None,
        "metadata": all_metadata,
    }
    return mean_mse, raw


# ──────────────────────────────────────────────────────────────────────────────
# Probe: spatial memorization of attention heads
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_probe_spatial_memorization(
    policy, preprocessor,
    val_dataset, val_ep_indices,
    cfg, output_dir, device,
):
    """
    Test whether attention heads memorize fixed spatial patterns.

    Collects attention maps from N frames (1 per unique episode), then computes
    log-sum, mean, and variance maps per (layer, query_group, key_camera).
    Renders per-head heatmap PNGs and returns raw stat tensors for .pt saving.
    """
    from lerobot.probes.attention_spatial_memorization import (
        embed_probe_prefix,
        probe_forward,
        extract_qk_attn,
        aggregate_maps,
        render_all,
        sample_one_per_episode,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    from lerobot.types import TransitionKey

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps

    attn_layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    timestep = p.timestep
    n_frames = p.spatial_n_frames
    batch_size = p.validation_batch_size

    # Sample 1 random frame per episode (respecting val_ep_indices)
    if val_ep_indices is not None:
        # Filter to val episodes, then use sample_one_per_episode's logic inline
        ep_to_indices = _build_episode_index(val_dataset)
        all_eps = sorted(e for e in ep_to_indices.keys() if e in val_ep_indices)
        import random as random_mod_local
        rng = random_mod_local.Random(p.random_seed)
        rng.shuffle(all_eps)
        selected = all_eps[:n_frames]
        samples = []
        for ep_idx in selected:
            indices = ep_to_indices[ep_idx]
            global_idx = rng.choice(indices)
            fr_idx = val_dataset.hf_dataset[global_idx]["frame_index"].item()
            samples.append((ep_idx, fr_idx, global_idx))
    else:
        samples = sample_one_per_episode(val_dataset, n_frames=n_frames, seed=p.random_seed)

    if not samples:
        logging.warning("[VAL] spatial_memorization: no samples found")
        return None

    logging.debug(
        f"[VAL] spatial_memorization: {len(samples)} frames × "
        f"layers {attn_layers} @ t={timestep} ..."
    )

    collected = {l: {} for l in attn_layers}
    img_h, img_w = None, None
    n_heads_global = None
    n_p_global = None

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        bs = len(batch_samples)

        b_obs = {}
        b_task_str = []
        for ep_idx, fr_idx, global_idx in batch_samples:
            obs, _, state, gt_subtask, task_str, _, _ = get_frame_data(
                val_dataset, global_idx, chunk_size
            )
            b_task_str.append(task_str)
            for k, v in obs.items():
                b_obs.setdefault(k, []).append(v)

        for k in b_obs:
            b_obs[k] = torch.cat(b_obs[k], dim=0).to(device)

        complementary_data = {
            "task":      b_task_str,
            "subtask":   [""] * bs,
            "advantage": torch.ones((bs, 1), device=device),
        }
        dummy_action = torch.zeros(bs, 1, 6, device=device)
        batch_for_proc = {
            TransitionKey.ACTION: dummy_action,
            **b_obs,
            TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        }
        processed = preprocessor(batch_for_proc)

        images, img_masks = policy._preprocess_images(b_obs)
        task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
        task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

        subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
            images, img_masks, task_tokens, task_masks
        )

        prefix_cache = embed_probe_prefix(
            policy, images, img_masks, task_tokens, task_masks,
            subtask_tokens, subtask_masks,
        )

        attn_by_layer, segments, pad_masks, patches_per_cam = probe_forward(
            prefix_cache, timestep, device,
        )
        if not attn_by_layer:
            logging.warning("  [spatial] No attention captured — skipping batch.")
            continue

        if img_h is None:
            img_t = images[0][0].cpu()
            img_h = img_t.shape[1]
            img_w = img_t.shape[2]
            n_p_global = int(patches_per_cam ** 0.5)

        seg_dict = {name: (s, e) for name, s, e in segments}
        camera_segs = [(name, s, e) for name, s, e in segments if name.startswith("img")]
        query_groups = [
            ("all",      0,                  segments[-1][2]),
            ("language", *seg_dict["language"]),
            ("subtask",  *seg_dict["subtask"]),
            ("action",   *seg_dict["action"]),
        ]

        for layer_idx in attn_layers:
            if layer_idx not in attn_by_layer:
                continue
            attn_weights = attn_by_layer[layer_idx]
            if n_heads_global is None:
                n_heads_global = attn_weights.shape[1]
            for b_idx in range(bs):
                attn = torch.nan_to_num(attn_weights[b_idx].float().cpu(), nan=0.0)
                pad  = pad_masks[b_idx].cpu()
                for q_name, q_s, q_e in query_groups:
                    for cam_name, cam_s, cam_e in camera_segs:
                        key = (q_name, cam_name)
                        vec = extract_qk_attn(attn, q_s, q_e, cam_s, cam_e, pad)
                        if vec is None:
                            continue
                        collected[layer_idx].setdefault(key, []).append(vec)

        del prefix_cache, b_obs, task_tokens, task_masks
        del subtask_tokens, subtask_masks, images, img_masks
        torch.cuda.empty_cache()

    # Aggregate stats
    raw_results = {}
    for layer_idx in attn_layers:
        for (q_name, cam_name), maps_list in collected[layer_idx].items():
            if len(maps_list) < 2:
                continue
            raw_results[(layer_idx, q_name, cam_name)] = aggregate_maps(maps_list)

    if not raw_results:
        logging.warning("[VAL] spatial_memorization: no results to aggregate")
        return None

    # Render PNGs (reuse standalone render_all but with our output_dir)
    import lerobot.probes.attention_spatial_memorization as _spatial_mod
    orig_output_dir = _spatial_mod.OUTPUT_DIR
    _spatial_mod.OUTPUT_DIR = output_dir
    try:
        render_all(raw_results, n_heads_global, n_p_global, img_h, img_w)
    finally:
        _spatial_mod.OUTPUT_DIR = orig_output_dir

    # Build raw data for .pt saving
    raw = {}
    for (layer_idx, q_name, cam_name), stats in raw_results.items():
        prefix = f"L{layer_idx}_{q_name}_{cam_name}"
        for stat_name, tensor in stats.items():
            raw[f"{prefix}_{stat_name}"] = tensor
    raw["_layers"] = torch.tensor(attn_layers)
    raw["_n_frames"] = torch.tensor(len(samples))
    raw["_img_hw"] = torch.tensor([img_h, img_w])
    raw["_n_p"] = torch.tensor(n_p_global)

    logging.info(f"[VAL] spatial_memorization: done ({len(raw_results)} combos)")
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Probe: action-drift Jacobian (per-frame causal A*J maps → MP4)
# ──────────────────────────────────────────────────────────────────────────────

def _run_probe_action_drift_jacobian(
    policy, preprocessor,
    val_dataset, val_ep_indices,
    cfg, output_dir, device,
):
    """Per-frame causal maps via A * |dA/d(action)|.  Outputs MP4 videos."""
    from lerobot.probes.action_drift_jacobian import run_action_drift_jacobian

    return run_action_drift_jacobian(
        policy, preprocessor,
        val_dataset, val_ep_indices,
        cfg, output_dir, device,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Probe: spatial memorization with Jacobian signal (aggregated causal stats)
# ──────────────────────────────────────────────────────────────────────────────

def _run_probe_spatial_memorization_jacobian(
    policy, preprocessor,
    val_dataset, val_ep_indices,
    cfg, output_dir, device,
):
    """
    Same as _run_probe_spatial_memorization but uses A*|grad(A)| as the signal.
    One forward+backward per (layer, batch) — slower but reveals causal patterns.
    """
    from lerobot.probes.action_drift_jacobian import (
        jacobian_probe_forward_multilayer,
    )
    from lerobot.probes.attention_pi05 import embed_probe_prefix
    from lerobot.probes.attention_spatial_memorization import (
        extract_qk_attn,
        aggregate_maps,
        render_all,
        sample_one_per_episode,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    from lerobot.types import TransitionKey

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps

    attn_layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    timestep = p.timestep
    n_frames = p.spatial_n_frames
    batch_size = p.validation_batch_size  # smaller batches for backward pass

    # Sample 1 random frame per episode (respecting val_ep_indices)
    if val_ep_indices is not None:
        ep_to_indices = _build_episode_index(val_dataset)
        all_eps = sorted(e for e in ep_to_indices.keys() if e in val_ep_indices)
        import random as random_mod_local
        rng = random_mod_local.Random(p.random_seed)
        rng.shuffle(all_eps)
        selected = all_eps[:n_frames]
        samples = []
        for ep_idx in selected:
            indices = ep_to_indices[ep_idx]
            global_idx = rng.choice(indices)
            fr_idx = val_dataset.hf_dataset[global_idx]["frame_index"].item()
            samples.append((ep_idx, fr_idx, global_idx))
    else:
        samples = sample_one_per_episode(val_dataset, n_frames=n_frames, seed=p.random_seed)

    if not samples:
        logging.warning("[VAL] spatial_memorization_jacobian: no samples found")
        return None

    collected = {l: {} for l in attn_layers}
    img_h, img_w = None, None
    n_heads_global = None
    n_p_global = None

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        bs = len(batch_samples)

        b_obs = {}
        b_task_str = []
        for ep_idx, fr_idx, global_idx in batch_samples:
            obs, _, state, gt_subtask, task_str, _, _ = get_frame_data(
                val_dataset, global_idx, chunk_size
            )
            b_task_str.append(task_str)
            for k, v in obs.items():
                b_obs.setdefault(k, []).append(v)

        for k in b_obs:
            b_obs[k] = torch.cat(b_obs[k], dim=0).to(device)

        complementary_data = {
            "task":      b_task_str,
            "subtask":   [""] * bs,
            "advantage": torch.ones((bs, 1), device=device),
        }
        dummy_action = torch.zeros(bs, 1, 6, device=device)
        batch_for_proc = {
            TransitionKey.ACTION: dummy_action,
            **b_obs,
            TransitionKey.COMPLEMENTARY_DATA: complementary_data,
        }
        processed = preprocessor(batch_for_proc)

        images, img_masks = policy._preprocess_images(b_obs)
        task_tokens = processed[OBS_LANGUAGE_TOKENS].to(device)
        task_masks  = processed[OBS_LANGUAGE_ATTENTION_MASK].to(device)

        subtask_tokens, subtask_masks = policy.model.generate_subtask_tokens(
            images, img_masks, task_tokens, task_masks
        )

        prefix_cache = embed_probe_prefix(
            policy, images, img_masks, task_tokens, task_masks,
            subtask_tokens, subtask_masks,
        )

        # Jacobian multilayer forward — one forward+backward per layer
        causal_by_layer, segments, pad_masks, patches_per_cam = (
            jacobian_probe_forward_multilayer(
                prefix_cache, timestep, device, policy, attn_layers
            )
        )
        if not causal_by_layer:
            logging.warning("  [spatial_jac] No causal maps — skipping batch.")
            continue

        if img_h is None:
            img_t = images[0][0].cpu()
            img_h = img_t.shape[1]
            img_w = img_t.shape[2]
            n_p_global = int(patches_per_cam ** 0.5)

        seg_dict = {name: (s, e) for name, s, e in segments}
        camera_segs = [(name, s, e) for name, s, e in segments if name.startswith("img")]
        query_groups = [
            ("all",      0,                  segments[-1][2]),
            ("language", *seg_dict["language"]),
            ("subtask",  *seg_dict["subtask"]),
            ("action",   *seg_dict["action"]),
        ]

        for layer_idx in attn_layers:
            if layer_idx not in causal_by_layer:
                continue
            causal_weights = causal_by_layer[layer_idx]
            if n_heads_global is None:
                n_heads_global = causal_weights.shape[1]
            for b_idx in range(bs):
                attn = torch.nan_to_num(causal_weights[b_idx].float().cpu(), nan=0.0)
                pad  = pad_masks[b_idx].cpu()
                for q_name, q_s, q_e in query_groups:
                    for cam_name, cam_s, cam_e in camera_segs:
                        key = (q_name, cam_name)
                        vec = extract_qk_attn(attn, q_s, q_e, cam_s, cam_e, pad)
                        if vec is None:
                            continue
                        collected[layer_idx].setdefault(key, []).append(vec)

        del prefix_cache, b_obs, task_tokens, task_masks
        del subtask_tokens, subtask_masks, images, img_masks
        del causal_by_layer
        torch.cuda.empty_cache()

    # Aggregate stats
    raw_results = {}
    for layer_idx in attn_layers:
        for (q_name, cam_name), maps_list in collected[layer_idx].items():
            if len(maps_list) < 2:
                continue
            raw_results[(layer_idx, q_name, cam_name)] = aggregate_maps(maps_list)

    if not raw_results:
        logging.warning("[VAL] spatial_memorization_jacobian: no results to aggregate")
        return None
    # Render PNGs
    import lerobot.probes.attention_spatial_memorization as _spatial_mod
    orig_output_dir = _spatial_mod.OUTPUT_DIR
    _spatial_mod.OUTPUT_DIR = output_dir
    try:
        render_all(raw_results, n_heads_global, n_p_global, img_h, img_w)
    finally:
        _spatial_mod.OUTPUT_DIR = orig_output_dir

    # Build raw data for .pt saving
    raw = {}
    for (layer_idx, q_name, cam_name), stats in raw_results.items():
        prefix = f"L{layer_idx}_{q_name}_{cam_name}"
        for stat_name, tensor in stats.items():
            raw[f"{prefix}_{stat_name}"] = tensor
    raw["_layers"] = torch.tensor(attn_layers)
    raw["_n_frames"] = torch.tensor(len(samples))
    raw["_img_hw"] = torch.tensor([img_h, img_w])
    raw["_n_p"] = torch.tensor(n_p_global)

    logging.info(f"[VAL] spatial_memorization_jacobian: done ({len(raw_results)} combos)")
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Probe: critic values distribution (TD-error + ||dV/dvision|| percentiles)
# ──────────────────────────────────────────────────────────────────────────────

def _run_probe_critic_values_distribution(
    policy, preprocessor,
    val_dataset, val_ep_indices,
    cfg, output_dir, device,
):
    """V(s) / TD-error histograms + critic gradient-magnitude exemplars."""
    from lerobot.probes.critic_values_distribution import run_critic_values_distribution

    return run_critic_values_distribution(
        policy, preprocessor,
        val_dataset, val_ep_indices,
        cfg, output_dir, device,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point — called from the training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(
    policy,
    preprocessor,
    postprocessor,
    val_dataset,
    val_ep_indices,
    manifold_cache,
    cfg,
    step: int,
    output_dir: str,
    wandb_logger,
    device,
):
    """
    Run all three probes at a single validation step.

    IMPORTANT: Must be called from rank-0 only.
    The caller is responsible for guarding with:
        if accelerator.is_main_process and val_dataset is not None:
            if step % val_freq == 0:
                run_validation(...)

    Parameters
    ----------
    policy          : unwrapped (accelerator.unwrap_model) PI05RLPolicy
    preprocessor    : pre-processor pipeline
    postprocessor   : post-processor pipeline
    val_dataset     : LeRobotDataset (or None → early return)
    val_ep_indices  : set[int] or None (None = use all episodes)
    manifold_cache  : dict from init_action_manifold() or None
    cfg             : training config — must have cfg.probe_parameters
    step            : current optimization step number
    output_dir      : base training output directory
    wandb_logger    : WandBLogger instance or None
    device          : torch.device

    Policy state
    ------------
    policy.eval() is called at the start and policy.train() is guaranteed to be
    called in the finally block — even if an exception propagates out of a probe.
    """
    if val_dataset is None:
        return

    p = cfg.probe_parameters
    step_dir = os.path.join(output_dir, "validation", f"step_{step:08d}")
    makedirs(step_dir)

    logging.info(f"[VAL] ═══ Validation at step {step} → {step_dir} ═══")

    # ── Episode thumbnails (no model needed) ────────────────────────────────
    try:
        ep_to_indices = _build_episode_index(val_dataset)
        if val_ep_indices is not None:
            ep_to_indices = {k: v for k, v in ep_to_indices.items() if k in val_ep_indices}
        _save_episode_thumbnails(val_dataset, ep_to_indices, step_dir)
    except Exception as exc:
        logging.warning(f"[VAL] Episode thumbnails failed at step {step}: {exc}", exc_info=True)

    policy.eval()
    try:
        wandb_scalars: dict[str, float] = {}
        raw_data: dict[str, dict] = {}  # probe_name → raw tensors for .pt

        # ── probe_actions ────────────────────────────────────────────────────
        if p.enable_actions:
            logging.info("[VAL] Actions analysis started")
            if manifold_cache is not None:
                try:
                    ratio, raw = _run_probe_actions(
                        policy, preprocessor, postprocessor,
                        val_dataset, val_ep_indices, manifold_cache,
                        cfg,
                        output_dir=os.path.join(step_dir, "actions"),
                        device=device,
                    )
                    if ratio is not None:
                        wandb_scalars["action_nn_distance_ratio"] = ratio
                    if raw is not None:
                        raw_data["actions"] = raw
                except Exception as exc:
                    logging.warning(
                        f"[VAL] probe_actions failed at step {step}: {exc}",
                        exc_info=True,
                    )
            else:
                logging.warning(
                    "[VAL] Skipping probe_actions: manifold_cache is None "
                    "(likely init_action_manifold() was not called or found no samples)."
                )
            logging.info("[VAL] Actions analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_representations ────────────────────────────────────────────
        if p.enable_representations:
            logging.info("[VAL] Representations analysis started")
            try:
                raw = _run_probe_representations(
                    policy, preprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "representations"),
                    device=device,
                )
                if raw is not None:
                    raw_data["representations"] = raw
            except Exception as exc:
                logging.warning(
                    f"[VAL] probe_representations failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Representations analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_attention ──────────────────────────────────────────────────
        if p.enable_attention:
            logging.info("[VAL] Attention analysis started")
            try:
                _run_probe_attention(
                    policy, preprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "attention"),
                    device=device,
                )
            except Exception as exc:
                logging.warning(
                    f"[VAL] probe_attention failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Attention analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_offline_inference ─────────────────────────────────────────
        if p.enable_offline_inference:
            logging.info("[VAL] Offline inference analysis started")
            try:
                mean_mse, raw = _run_probe_offline_inference(
                    policy, preprocessor, postprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "offline_inference"),
                    device=device,
                )
                if mean_mse is not None:
                    wandb_scalars["offline_inference_mse"] = mean_mse
                if raw is not None:
                    raw_data["offline_inference"] = raw
            except Exception as exc:
                logging.warning(
                    f"[VAL] probe_offline_inference failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Offline inference analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_spatial_memorization ───────────────────────────────────────
        if p.enable_spatial_memorization:
            logging.info("[VAL] Spatial memorization analysis started")
            try:
                raw = _run_probe_spatial_memorization(
                    policy, preprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "spatial_memorization"),
                    device=device,
                )
                if raw is not None:
                    raw_data["spatial_memorization"] = raw
            except Exception as exc:
                logging.warning(
                    f"[VAL] probe_spatial_memorization failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Spatial memorization analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── action_drift_jacobian ────────────────────────────────────────
        if p.enable_action_drift_jacobian:
            logging.info("[VAL] Action drift jacobian analysis started")
            try:
                _run_probe_action_drift_jacobian(
                    policy, preprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "action_drift_jacobian"),
                    device=device,
                )
            except Exception as exc:
                logging.warning(
                    f"[VAL] action_drift_jacobian failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Action drift jacobian analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_spatial_memorization_jacobian ─────────────────────────────
        if p.enable_spatial_memorization_jacobian:
            logging.info("[VAL] Spatial memorization jacobian analysis started")
            try:
                raw = _run_probe_spatial_memorization_jacobian(
                    policy, preprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "spatial_memorization_jacobian"),
                    device=device,
                )
                if raw is not None:
                    raw_data["spatial_memorization_jacobian"] = raw
            except Exception as exc:
                logging.warning(
                    f"[VAL] probe_spatial_memorization_jacobian failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Spatial memorization jacobian analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_critic_values_distribution ────────────────────────────────
        if p.enable_critic_values_distribution:
            logging.info("[VAL] Critic values distribution analysis started")
            try:
                raw = _run_probe_critic_values_distribution(
                    policy, preprocessor,
                    val_dataset, val_ep_indices,
                    cfg,
                    output_dir=os.path.join(step_dir, "critic_values_distribution"),
                    device=device,
                )
                if raw is not None:
                    raw_data["critic_values_distribution"] = raw
            except Exception as exc:
                logging.warning(
                    f"[VAL] probe_critic_values_distribution failed at step {step}: {exc}",
                    exc_info=True,
                )
            logging.info("[VAL] Critic values distribution analysis completed")
        gc.collect()
        torch.cuda.empty_cache()

        # ── Save raw probe data as .pt ───────────────────────────────────────
        if raw_data:
            pt_path = os.path.join(step_dir, "probe_raw_data.pt")
            try:
                torch.save(raw_data, pt_path)
                logging.debug(
                    f"[VAL] Raw probe data saved: {pt_path} "
                    f"(probes: {list(raw_data.keys())})"
                )
            except Exception as exc:
                logging.warning(f"[VAL] Failed to save raw probe data: {exc}")

        # ── WandB scalar logging ─────────────────────────────────────────────
        if wandb_logger is not None and wandb_scalars:
            try:
                wandb_logger.log_dict(wandb_scalars, step=step, mode="eval")
                logging.info(f"[VAL] WandB scalars logged: {wandb_scalars}")
            except Exception as exc:
                logging.warning(
                    f"[VAL] WandB logging failed at step {step}: {exc}"
                )

    finally:
        policy.train()
        gc.collect()
        torch.cuda.empty_cache()

    logging.info(f"[VAL] ═══ Validation done at step {step} ═══")
