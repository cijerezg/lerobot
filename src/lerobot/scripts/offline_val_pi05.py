#!/usr/bin/env python
"""
Periodic validation pipeline for offline Pi05 RL training.

This module is designed to be imported from offline_learner_pi05.py (or any
other training script) without modifying that file. It exposes three public
functions that together implement periodic validation using the three probe
scripts (actions, representations, attention).

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
    ├─ _run_probe_actions()          [try/except — does NOT re-fit manifold]
    │    ├─ _sample_val_episodes()
    │    ├─ collect_eval_dataset()   [probe_actions_pi05, @no_grad]
    │    ├─ run_plotting()           [probe_actions_pi05]
    │    └─ compute_nn_distances()  → WandB scalar
    ├─ _run_probe_representations()  [try/except — re-fits UMAP each call]
    │    ├─ _sample_val_episodes()
    │    ├─ collect_activations()   [probe_representations_pi05, @no_grad]
    │    ├─ collect_subtask_injection() [optional, @no_grad]
    │    └─ run_plotting()           [probe_representations_pi05]
    ├─ _run_probe_attention()        [try/except — @no_grad wrapper]
    │    ├─ _build_attn_sample_list()
    │    └─ per-episode/batch loop:
    │         preprocessor → embed_probe_prefix → probe_forward
    │         → render_image_overlays / render_full_matrix → MP4
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

from lerobot.scripts.eval_offline_pi05 import (
    _build_episode_index,
    get_frame_data,
    normalize_gt,
    render_sample,
    run_inference,
    SO100_JOINT_NAMES,
)
from lerobot.rl.probe_utils_pi05 import makedirs
from lerobot.rl.probe_representations_pi05 import _save_episode_thumbnails


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

    Similar to build_sample_list in probe_attention_pi05 but respects
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

    from lerobot.rl.probe_actions_pi05 import collect_gt_reference, fit_manifold

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

    logging.info(
        f"[VAL] Collecting {len(ref_samples)} reference GT frames "
        f"for action manifold (no model) ..."
    )
    ref_data = collect_gt_reference(val_dataset, ref_samples, cfg.policy.chunk_size)

    logging.info("[VAL] Fitting PCA + UMAP reference manifold ...")
    pca, reducer2d, reducer3d, ref_emb2, ref_emb3 = fit_manifold(
        ref_data["gt"], cfg, pca_dir
    )

    logging.info("[VAL] Reference manifold ready.")
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
    from lerobot.rl.probe_actions_pi05 import (
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

    logging.info(f"[VAL] probe_actions: collecting {len(eval_samples)} frames ...")

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

    logging.info("[VAL] probe_actions: generating plots ...")
    run_plotting(full_cache, cfg, output_dir)

    # Compute NN-distance ratio scalar directly from embeddings (avoids CSV read)
    gt_nn   = compute_nn_distances(ds_cache["gt_emb2"],   manifold_cache["ref_emb2"])
    pred_nn = compute_nn_distances(ds_cache["pred_emb2"], manifold_cache["ref_emb2"])
    ratio   = float(np.median(pred_nn) / (np.median(gt_nn) + 1e-8))
    logging.info(f"[VAL] probe_actions: median pred/GT NN ratio = {ratio:.4f}")
    return ratio


def _run_probe_representations(policy, preprocessor,
                                 val_dataset, val_ep_indices,
                                 cfg, output_dir, device):
    """
    Run probe_representations at a single validation step.

    Collects prefix (VLM hidden states) and suffix (expert hidden states) at
    multiple denoising timesteps via forward hooks, then PCA + UMAP + plots.
    UMAP is re-fit here at every val step (intentional — tracks drift).
    """
    from lerobot.rl.probe_representations_pi05 import (
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

    logging.info(f"[VAL] probe_representations: collecting {len(samples)} frames ...")

    cache = collect_activations(policy, preprocessor, val_dataset, samples, device, cfg)

    if p.subtask_injection:
        logging.info("[VAL] probe_representations: collecting subtask injection activations ...")
        inj = collect_subtask_injection(
            policy, preprocessor, val_dataset, samples, device, cfg
        )
        cache.update(inj)

    logging.info("[VAL] probe_representations: running PCA + UMAP + plots ...")
    run_plotting(cache, cfg, output_dir)


@torch.no_grad()
def _run_probe_attention(policy, preprocessor,
                          val_dataset, val_ep_indices,
                          cfg, output_dir, device):
    """
    Run probe_attention at a single validation step.

    Captures layer-0 attention weights at one or more diffusion timesteps and
    writes MP4 videos of attention overlays and full attention matrices.

    This function replicates the _probe_dataset inner loop from
    probe_attention_pi05.probe_cli, filtered to val_ep_indices.

    The @torch.no_grad() decorator ensures all forward passes inside are safe
    even though the individual called functions also manage this.
    """
    import imageio
    from lerobot.rl.probe_attention_pi05 import (
        embed_probe_prefix,
        probe_forward,
        render_image_overlays,
        render_full_matrix,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    from lerobot.processor.core import TransitionKey

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = cfg.policy.n_action_steps
    timesteps  = [float(t.strip()) for t in p.timesteps.split(",")]

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

    logging.info(
        f"[VAL] probe_attention: {len(samples)} episodes × "
        f"{len(timesteps)} timesteps ..."
    )

    fps      = getattr(val_dataset, "fps", 30) / p.attn_eval_subsample
    batch_sz = p.attn_batch_size

    for ep_idx, ep_frames in samples:
        writers: dict[float, dict] = {t_val: {} for t_val in timesteps}

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

            logging.info(
                f"    [attn] ep={ep_idx:04d} "
                f"frames {batch_slice[0][0]:04d}..{batch_slice[-1][0]:04d} "
                f"(batch {len(batch_slice)})"
            )

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

            # ── Per-timestep forward + render ────────────────────────────────
            for t_val in timesteps:
                attn_weights, segments, pad_masks, patches_per_cam = probe_forward(
                    prefix_cache, t_val, device
                )
                if attn_weights is None:
                    logging.warning(
                        f"    [attn] No attention captured at t={t_val}; skipping."
                    )
                    continue

                t_str  = f"{t_val:.2f}".replace(".", "p")
                ep_dir = os.path.join(output_dir, f"ep{ep_idx:04d}_t{t_str}")
                makedirs(ep_dir)

                csv_path = os.path.join(ep_dir, "norm_consts.csv")
                csv_file  = open(csv_path, "a", newline="")
                csv_writer = csv.writer(csv_file)
                if os.path.getsize(csv_path) == 0:
                    csv_writer.writerow(["ep", "fr", "t_val", "panel", "vmax"])

                for b_idx, (fr_idx, _) in enumerate(batch_slice):
                    a_w = attn_weights[b_idx : b_idx + 1]
                    p_m = pad_masks[b_idx : b_idx + 1]
                    i_t = [img[b_idx : b_idx + 1] for img in images]

                    overlay_frames, norm_consts = render_image_overlays(
                        a_w, segments, i_t, p_m, patches_per_cam
                    )
                    frames_out = dict(overlay_frames)
                    frames_out.update(render_full_matrix(a_w, segments, p_m))

                    for panel, vmax in norm_consts.items():
                        csv_writer.writerow(
                            [ep_idx, fr_idx, t_val, panel, f"{vmax:.6e}"]
                        )

                    for key, frame_np in frames_out.items():
                        if key not in writers[t_val]:
                            mp4_path = os.path.join(ep_dir, f"{key}.mp4")
                            writers[t_val][key] = imageio.get_writer(
                                mp4_path, fps=fps, macro_block_size=1
                            )
                        writers[t_val][key].append_data(frame_np)

                csv_file.close()
                del attn_weights, segments, pad_masks, patches_per_cam

            del prefix_cache, b_obs_batched, task_tokens, task_masks
            del subtask_tokens, subtask_masks, images, img_masks
            torch.cuda.empty_cache()

        for writers_t in writers.values():
            for w in writers_t.values():
                w.close()


# ──────────────────────────────────────────────────────────────────────────────
# Probe: offline eval (per-frame GT vs predicted action traces)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_probe_offline_eval(
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
        n_per_episode=getattr(probe_cfg, "n_frames_per_episode", 4),
        max_episodes=getattr(probe_cfg, "max_episodes", 6),
        seed=getattr(probe_cfg, "seed", 42),
    )
    if not samples:
        logging.warning("[VAL] offline_eval: no samples selected.")
        return None

    dir_unnorm = os.path.join(output_dir, "unnormalized")
    dir_norm = os.path.join(output_dir, "normalized")
    makedirs(dir_unnorm)
    makedirs(dir_norm)

    action_dim = None
    mse_values: list[float] = []

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

        logging.info(
            f"  [offline_eval] ep={ep_idx:04d} fr={fr_idx:04d} | "
            f"mse={mse:.4f} | GT: '{gt_subtask}' | pred: '{pred_subtask}'"
        )

    mean_mse = sum(mse_values) / len(mse_values) if mse_values else None
    if mean_mse is not None:
        logging.info(f"  [offline_eval] mean MSE = {mean_mse:.4f}")
    return mean_mse


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

        # ── probe_actions ────────────────────────────────────────────────────
        if manifold_cache is not None:
            try:
                ratio = _run_probe_actions(
                    policy, preprocessor, postprocessor,
                    val_dataset, val_ep_indices, manifold_cache,
                    cfg,
                    output_dir=os.path.join(step_dir, "actions"),
                    device=device,
                )
                if ratio is not None:
                    wandb_scalars["action_nn_distance_ratio"] = ratio
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
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_representations ────────────────────────────────────────────
        try:
            _run_probe_representations(
                policy, preprocessor,
                val_dataset, val_ep_indices,
                cfg,
                output_dir=os.path.join(step_dir, "representations"),
                device=device,
            )
        except Exception as exc:
            logging.warning(
                f"[VAL] probe_representations failed at step {step}: {exc}",
                exc_info=True,
            )
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_attention ──────────────────────────────────────────────────
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
        gc.collect()
        torch.cuda.empty_cache()

        # ── probe_offline_eval ──────────────────────────────────────────────
        try:
            mean_mse = _run_probe_offline_eval(
                policy, preprocessor, postprocessor,
                val_dataset, val_ep_indices,
                cfg,
                output_dir=os.path.join(step_dir, "offline_eval"),
                device=device,
            )
            if mean_mse is not None:
                wandb_scalars["offline_eval_mse"] = mean_mse
        except Exception as exc:
            logging.warning(
                f"[VAL] probe_offline_eval failed at step {step}: {exc}",
                exc_info=True,
            )

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
