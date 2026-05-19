#!/usr/bin/env python
"""
Action-Drift Jacobian probe — causal attention analysis.

Policy-agnostic: works for any policy whose ``ProbablePolicy`` adapter
implements ``capture_attention(requires_grad=True)``.

Instead of plotting raw softmax attention (which includes attention sinks),
this probe computes the **causal map** per layer::

    causal_map = A * |dA/dL|

where ``A`` is the softmax attention and ``L`` is a scalar loss derived from
the predicted action (pi05: ``norm(action_pred)``; molmoact2: the flow-matching
loss). A patch lights up if and only if:

  1. The model actually looked at it (``A > 0``).
  2. It actively steered the predicted action (``|J| > 0``).

The adapter returns the causal maps already packed into ``cross_attn_by_layer``
and ``self_attn_by_layer`` of an :class:`AttentionCaptureResult`, so the
visualisation reuses :mod:`lerobot.probes.attention` rendering verbatim.

Output (under ``probe_parameters.output_dir/action_drift_jacobian/``):
  same layout as the attention probe, just with causal maps instead of raw
  softmax attention.

Usage:
    python -m lerobot.probes.action_drift_jacobian config.yaml \\
        --probe_parameters.timestep 0.5
"""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.attention import (
    _warn_overcommit_if_risky,
    build_episode_samples,
    render_cross_matrix,
    render_image_overlays,
    render_self_matrix,
)
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import get_frame_data, load_extra_dataset
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeJacobianConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters``."""


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _probe_dataset(adapter, ds, ds_output_dir, layers, timestep, cfg):
    """Per-dataset Jacobian-rendered loop. Used by both standalone CLI and
    the rl_offline validation loop."""
    p = cfg.probe_parameters
    chunk_size = adapter.chunk_size
    os.makedirs(ds_output_dir, exist_ok=True)
    samples = build_episode_samples(
        ds,
        episodes_str=getattr(p, "attn_eval_episodes", None),
        random_n=p.max_episodes,
        subsample=getattr(p, "attn_eval_subsample", 1),
        seed=p.random_seed,
    )
    if not samples:
        logging.warning(f"  No samples in {ds_output_dir}, skipping.")
        return

    fps = getattr(ds, "fps", 30) / max(1, getattr(p, "attn_eval_subsample", 1))
    _warn_overcommit_if_risky("JAC")
    t_str = f"{timestep:.2f}".replace(".", "p")

    for ep_idx, ep_frames in samples:
        writers: dict[int, dict[str, "imageio.core.format.Writer"]] = {
            l: {} for l in layers  # noqa: E741
        }
        csv_files: dict[int, tuple] = {}

        for fr_idx, global_idx in ep_frames:
            obs, _, state, _, task_str, _, _ = get_frame_data(ds, global_idx, chunk_size)

            # Causal maps come back packed into cross_attn_by_layer /
            # self_attn_by_layer in the same shape as the regular attention
            # probe, so the renderers from attention.py work unchanged.
            result = adapter.capture_attention(
                obs, task_str, state=state, timestep=timestep, layers=layers,
                requires_grad=True,
            )

            for layer_idx in layers:
                ep_dir = os.path.join(
                    ds_output_dir, f"ep{ep_idx:04d}_t{t_str}", f"L{layer_idx:02d}",
                )
                os.makedirs(ep_dir, exist_ok=True)

                if layer_idx not in csv_files:
                    csv_path = os.path.join(ep_dir, "norm_consts.csv")
                    f = open(csv_path, "a", newline="")
                    w = csv.writer(f)
                    if os.path.getsize(csv_path) == 0:
                        w.writerow(["ep", "fr", "layer", "panel", "vmax"])
                    csv_files[layer_idx] = (f, w)
                csv_f, csv_w = csv_files[layer_idx]

                panels: dict[str, np.ndarray] = {}
                vmaxes: dict[str, float] = {}
                for renderer in (render_image_overlays, render_cross_matrix, render_self_matrix):
                    p_frames, p_vmax = renderer(result, layer_idx)
                    # Prefix output keys with "causal_" so they don't collide
                    # if someone runs both probes into the same dir.
                    panels.update({f"causal_{k}": v for k, v in p_frames.items()})
                    vmaxes.update({f"causal_{k}": v for k, v in p_vmax.items()})

                for panel, vmax in vmaxes.items():
                    csv_w.writerow([ep_idx, fr_idx, layer_idx, panel, f"{vmax:.6e}"])

                for key, frame_np in panels.items():
                    if key not in writers[layer_idx]:
                        out_path = os.path.join(ep_dir, f"{key}.mp4")
                        writers[layer_idx][key] = imageio.get_writer(
                            out_path, fps=fps, macro_block_size=1,
                        )
                    writers[layer_idx][key].append_data(frame_np)

        for d in writers.values():
            for w in d.values():
                w.close()
        for f, _ in csv_files.values():
            f.close()


def run(adapter, primary_dataset, cfg, output_dir):
    """Run the Jacobian probe on the primary dataset (and any extras)."""
    if adapter is None or primary_dataset is None:
        return

    p = cfg.probe_parameters
    layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    timestep = float(getattr(p, "timestep", 0.5))
    logging.info(f"Jacobian layers: {layers} timestep: {timestep}")
    os.makedirs(output_dir, exist_ok=True)

    primary_name = os.path.basename(os.path.normpath(cfg.dataset.root))
    _probe_dataset(adapter, primary_dataset,
                   os.path.join(output_dir, primary_name), layers, timestep, cfg)

    for extra_root in getattr(cfg.dataset, "additional_offline_dataset_paths", []) or []:
        logging.info(f"Additional dataset: {extra_root}")
        extra_ds = load_extra_dataset(cfg.dataset.repo_id, extra_root)
        _probe_dataset(adapter, extra_ds,
                       os.path.join(output_dir, os.path.basename(os.path.normpath(extra_root))),
                       layers, timestep, cfg)


@parser.wrap()
def probe_cli(cfg: ProbeJacobianConfig):
    init_logging()
    p = cfg.probe_parameters
    device = get_safe_torch_device(try_device=cfg.policy.device)
    output_dir = os.path.join(p.output_dir, "action_drift_jacobian")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output dir: {output_dir}")

    from lerobot.datasets.factory import make_dataset
    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None

    logging.info("Loading policy adapter …")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    run(adapter, dataset, cfg, output_dir)
    logging.info(f"Done. Output saved to {output_dir}/")


if __name__ == "__main__":
    probe_cli()
