"""MEM history-influence probe (04_memory.md §2.4).

Measures how much the short-term memory actually moves the policy's action chunk.
For each sampled frame we assemble the 5-frame lookback window and run
``predict_action_chunk`` under four history conditions, all with identical fixed-seed
flow noise so the ONLY difference is what memory the model sees:

  full   : image temporal attention ON  + continuous state tokens present
  none   : image attention masked OFF   + state tokens absent   (the memoryless baseline)
  images : image attention ON           + state tokens absent
  states : image attention masked OFF   + state tokens present

Influence = ||action(cond) - action(none)|| on the normalized action chunk. `full`
is total memory influence; `images` / `states` isolate each channel. A dead feature
reads ~0; the number should GROW over training as the model learns to use memory
(the causal-confusion substrate — run against the no-history baseline offline eval).

Registered probe: enable with ``probe_parameters.enable_mem_history_influence``.
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.utils import assemble_frame_history, get_frame_data, makedirs, sample_episodes_evenly
from lerobot.utils.constants import OBS_STATE


def _variant_batch(batch: dict, images_on: bool, states_on: bool) -> dict:
    """Toggle the memory channels on a packed batch without repacking (identical
    input_ids/seq-length across conditions): the image mask gates temporal attention,
    dropping history_state_values skips the state-token scatter (placeholders keep
    their base embedding)."""
    out = dict(batch)
    mask = batch["history_images_mask"]
    out["history_images_mask"] = torch.ones_like(mask) if images_on else torch.zeros_like(mask)
    if not states_on:
        out.pop("history_state_values", None)
        out.pop("state_history_token_id", None)
    return out


def run(adapter, dataset, cfg, output_dir: str) -> None:
    memory_cfg = getattr(cfg.policy, "memory", None)
    if memory_cfg is None or not memory_cfg.history_keys or memory_cfg.history_num_samples <= 0:
        logging.info("[mem_history_influence] no short-term memory configured — skipping.")
        return
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[mem_history_influence] needs continuous flow actions — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    device = adapter.device
    fps = cfg.env.fps
    # MEM channels only: state + camera history (depth rides the separate pointmap path).
    keys = [k for k in memory_cfg.history_keys if k == OBS_STATE or "images" in k]

    adapter._set_probe_cuda_graph_enabled(False)  # varying mask per condition; keep eager
    samples = sample_episodes_evenly(dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed)

    conditions = {"full": (True, True), "none": (False, False), "images": (True, False), "states": (False, True)}
    rows: list[dict] = []
    for _ep, _fr, global_idx in samples:
        obs, _gt, _state, _subtask, task_str, _ep_i, _fr_i = get_frame_data(
            dataset, global_idx, int(cfg.policy.chunk_size)
        )
        obs.update(assemble_frame_history(dataset, global_idx, memory_cfg, fps, keys))
        batch = adapter._make_batch(obs, task_str)
        if "history_images_mask" not in batch:
            logging.warning("[mem_history_influence] batch carries no history tensors — skipping frame.")
            continue

        acts: dict[str, torch.Tensor] = {}
        for name, (img_on, st_on) in conditions.items():
            gen = torch.Generator(device=device)
            gen.manual_seed(0)  # same flow noise across conditions
            acts[name] = (
                adapter.policy.predict_action_chunk(
                    _variant_batch(batch, img_on, st_on),
                    inference_action_mode="continuous",
                    generator=gen,
                )
                .float()
                .cpu()
            )
        base = acts["none"]
        rows.append(
            {
                "global_idx": int(global_idx),
                "full_maxabs": float((acts["full"] - base).abs().max()),
                "full_rmse": float((acts["full"] - base).pow(2).mean().sqrt()),
                "images_maxabs": float((acts["images"] - base).abs().max()),
                "images_rmse": float((acts["images"] - base).pow(2).mean().sqrt()),
                "states_maxabs": float((acts["states"] - base).abs().max()),
                "states_rmse": float((acts["states"] - base).pow(2).mean().sqrt()),
            }
        )

    if not rows:
        logging.warning("[mem_history_influence] no frames produced measurements.")
        return

    summary = {
        metric: float(np.mean([r[metric] for r in rows]))
        for metric in ("full_maxabs", "full_rmse", "images_maxabs", "images_rmse", "states_maxabs", "states_rmse")
    }
    summary["n_frames"] = len(rows)
    with open(os.path.join(output_dir, "influence.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    channels = ["full", "images", "states"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(channels))
    ax.bar(x - 0.2, [summary[f"{c}_rmse"] for c in channels], width=0.4, label="RMSE")
    ax.bar(x + 0.2, [summary[f"{c}_maxabs"] for c in channels], width=0.4, label="max|Δ|")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("normalized action Δ vs no-memory")
    ax.set_title(f"MEM history influence on the action chunk (n={len(rows)})")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "influence.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)
    logging.info(
        f"[mem_history_influence] n={len(rows)}  full RMSE={summary['full_rmse']:.4f}  "
        f"images RMSE={summary['images_rmse']:.4f}  states RMSE={summary['states_rmse']:.4f}"
    )
