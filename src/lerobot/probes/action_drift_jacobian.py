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
from dataclasses import dataclass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio
import numpy as np

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.attention import (
    _episode_matrix_vmax,
    _episode_overlay_vmax,
    _extract_action_text_data,
    _extract_cross_matrix_data,
    _extract_overlay_grids,
    _extract_self_matrix_data,
    _render_action_text_from_data,
    _render_cross_matrix_from_data,
    _render_overlays_from_grids,
    _render_self_matrix_from_data,
    _warn_overcommit_if_risky,
    build_episode_samples,
)
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.utils import (
    load_extra_dataset,
    load_probe_dataset,
    probe_frame_inputs,
    probe_image_stride,
)
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
        max_frames=p.n_frames_per_episode,
        grid_stride=probe_image_stride(cfg),
    )
    if not samples:
        logging.warning(f"  No samples in {ds_output_dir}, skipping.")
        return

    stride = samples[0][1][1][0] - samples[0][1][0][0] if len(samples[0][1]) > 1 else 1
    fps = min(10, 4 * getattr(ds, "fps", 30) / stride)  # <=4x real time, <=10 fps display
    logging.info(f"  {len(samples)} episode(s) x {len(samples[0][1])} frames (stride {stride})")
    _warn_overcommit_if_risky("JAC")
    t_str = f"{timestep:.2f}".replace(".", "p")

    for ep_idx, ep_frames in samples:
        # Every panel uses an episode-fixed p98 vmax — see attention.py for
        # the rationale. Pass 1 buffers raw extracts; pass 2 aggregates and renders.
        writers: dict[int, dict[str, "imageio.core.format.Writer"]] = {
            l: {} for l in layers  # noqa: E741
        }
        csv_files: dict[int, tuple] = {}
        layer_buf: dict[int, list[dict]] = {l: [] for l in layers}  # noqa: E741

        for fr_idx, global_idx in ep_frames:
            frame = probe_frame_inputs(ds, cfg, global_idx, chunk_size)

            # Causal maps come back packed into cross_attn_by_layer /
            # self_attn_by_layer in the same shape as the regular attention
            # probe, so the renderers from attention.py work unchanged.
            result = adapter.capture_attention(
                frame["obs"], frame["task"], state=frame["state"], timestep=timestep,
                layers=layers, requires_grad=True, gt_actions=frame["gt_actions"],
                subtask=frame["subtask"], metadata=frame["metadata"],
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

                layer_buf[layer_idx].append({
                    "fr_idx": fr_idx,
                    "overlay_grids": _extract_overlay_grids(result, layer_idx),
                    "cross": _extract_cross_matrix_data(result, layer_idx),
                    "self": _extract_self_matrix_data(result, layer_idx),
                    "action_text": _extract_action_text_data(result, layer_idx),
                })

        for layer_idx, frames_buf in layer_buf.items():
            if not frames_buf:
                continue
            overlay_buf = [(d["fr_idx"], d["overlay_grids"]) for d in frames_buf]
            ep_overlay_vmax = _episode_overlay_vmax(overlay_buf, percentile=98.0)
            ep_matrix_vmax = _episode_matrix_vmax(
                [d["cross"] for d in frames_buf if d["cross"] is not None],
                [d["self"] for d in frames_buf if d["self"] is not None],
                [d["action_text"] for d in frames_buf if d["action_text"] is not None],
                percentile=98.0,
            )
            all_vmax = {**ep_overlay_vmax, **ep_matrix_vmax}

            csv_f, csv_w = csv_files[layer_idx]
            ep_dir = os.path.join(
                ds_output_dir, f"ep{ep_idx:04d}_t{t_str}", f"L{layer_idx:02d}",
            )
            for d in frames_buf:
                fr_idx = d["fr_idx"]
                panels: dict[str, np.ndarray] = {}
                vmaxes: dict[str, float] = {}
                for p_frames, p_vmax in (
                    _render_overlays_from_grids(d["overlay_grids"], vmax_overrides=all_vmax),
                    _render_cross_matrix_from_data(d["cross"], layer_idx, vmax_overrides=all_vmax),
                    _render_self_matrix_from_data(d["self"], layer_idx, vmax_overrides=all_vmax),
                    _render_action_text_from_data(d["action_text"], layer_idx, vmax_overrides=all_vmax),
                ):
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
                            # Panels are megabytes per raw frame; at a low fps ffmpeg
                            # can't estimate the rate within the default 5M probe.
                            input_params=["-probesize", "100M"],
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

    _probe_dataset(adapter, primary_dataset, output_dir, layers, timestep, cfg)

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

    dataset = load_probe_dataset(cfg)

    logging.info("Loading policy adapter …")
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    run(adapter, dataset, cfg, output_dir)
    logging.info(f"Done. Output saved to {output_dir}/")


if __name__ == "__main__":
    probe_cli()
