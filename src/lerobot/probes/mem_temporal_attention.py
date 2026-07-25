"""MEM temporal-attention probe (04_memory.md §2.4).

Measures, per temporal ViT layer, how much of the current frame's attention mass
lands on PAST frames (vs its own frame's patches) in the union softmax. This is the
most direct read on whether the video encoder's temporal path is actually engaging:
~0 means the current frame ignores history at that layer; higher means it reads the
past. Tracked over training, it shows the mechanism waking up (and at which depth).

For each sampled frame we assemble the lookback window, run one prefix forward with
the encoder capture hook on (modeling_molmoact2._MEM_TEMPORAL_CAPTURE), and read one
scalar per temporal layer. Aggregated mean±std per layer over frames.

Registered probe: enable with ``probe_parameters.enable_mem_temporal_attention``.
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.policies.molmoact2.modeling_molmoact2 import _MEM_TEMPORAL_CAPTURE
from lerobot.probes.utils import assemble_frame_history, get_frame_data, makedirs, sample_episodes_evenly
from lerobot.utils.constants import OBS_STATE


def _temporal_layer_indices(policy) -> list[int]:
    resblocks = policy._backbone().vision_backbone.image_vit.transformer.resblocks
    stride = max(int(policy.config.temporal_layer_stride), 1)
    return [i for i in range(len(resblocks)) if (i + 1) % stride == 0]


def run(adapter, dataset, cfg, output_dir: str) -> None:
    memory_cfg = getattr(cfg.policy, "memory", None)
    has_images = memory_cfg is not None and any("images" in k for k in (memory_cfg.history_keys or []))
    if not has_images or memory_cfg.history_num_samples <= 0:
        logging.info("[mem_temporal_attention] no image history configured — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    fps = cfg.env.fps
    keys = [k for k in memory_cfg.history_keys if k == OBS_STATE or "images" in k]
    layers = _temporal_layer_indices(adapter.policy)

    adapter._set_probe_cuda_graph_enabled(False)
    samples = sample_episodes_evenly(dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed)

    per_frame: list[list[float]] = []
    for _ep, _fr, global_idx in samples:
        obs, _gt, _state, _subtask, task_str, _ep_i, _fr_i = get_frame_data(
            dataset, global_idx, int(cfg.policy.chunk_size)
        )
        obs.update(assemble_frame_history(dataset, global_idx, memory_cfg, fps, keys))
        batch = adapter._make_batch(obs, task_str)
        if "history_images" not in batch:
            continue

        model_inputs = adapter.policy._model_inputs(batch)  # stashes the history window
        _MEM_TEMPORAL_CAPTURE["records"].clear()
        _MEM_TEMPORAL_CAPTURE["enabled"] = True
        try:
            with torch.no_grad():
                adapter.policy._run_prefix_backbone(model_inputs)
        finally:
            _MEM_TEMPORAL_CAPTURE["enabled"] = False
        records = list(_MEM_TEMPORAL_CAPTURE["records"])
        if len(records) == len(layers):
            per_frame.append(records)
        else:
            logging.warning(
                f"[mem_temporal_attention] expected {len(layers)} temporal records, got {len(records)} — skipping."
            )

    if not per_frame:
        logging.warning("[mem_temporal_attention] no frames produced measurements.")
        return

    arr = np.asarray(per_frame)  # (n_frames, n_layers)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    summary = {
        "layers": layers,
        "temporal_mass_mean": mean.tolist(),
        "temporal_mass_std": std.tolist(),
        "n_frames": len(per_frame),
    }
    with open(os.path.join(output_dir, "temporal_attention.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(layer) for layer in layers], mean, yerr=std, capsize=3)
    ax.set_xlabel("ViT temporal layer index")
    ax.set_ylabel("current-frame attention mass on past frames")
    ax.set_ylim(0, max(0.05, float(mean.max()) * 1.3))
    ax.set_title(f"MEM temporal attention (n={len(per_frame)})")
    fig.savefig(os.path.join(output_dir, "temporal_attention.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)
    logging.info(
        f"[mem_temporal_attention] n={len(per_frame)}  per-layer past-mass mean="
        f"{[round(float(m), 4) for m in mean]}"
    )
