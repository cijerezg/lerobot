"""MEM history-influence probe (04_memory.md §2.4).

Measures whether short-term memory moves the policy's action chunk and whether that
movement is useful on the demonstrated action.
For each sampled frame we assemble the 5-frame lookback window and run
``predict_action_chunk`` under four history conditions, all with identical fixed-seed
flow noise so the ONLY difference is what memory the model sees:

  full   : image temporal attention ON  + continuous state tokens present
  none   : image history absent         + state tokens absent   (the memoryless baseline)
  images : image attention ON           + state tokens absent
  states : image history absent         + state tokens present

All conditions are packed independently, so ``none`` is the real training-time
no-history prompt rather than a full prompt with unfilled placeholder tokens.

The probe reports both influence ``||action(cond) - action(none)||`` and GT MSE
improvement ``MSE(none, GT) - MSE(cond, GT)`` in normalized action space. Positive
improvement means memory moved the prediction in the useful direction; influence by
itself only establishes sensitivity.

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


def _variant_observation(obs: dict, images_on: bool, states_on: bool) -> dict:
    """Remove disabled history channels before packing the prompt/model inputs."""
    out = dict(obs)
    for key in list(out):
        if not key.startswith("history."):
            continue
        remove_state = key == f"history.{OBS_STATE}" and not states_on
        remove_images = "images" in key and not images_on
        if remove_state or remove_images:
            out.pop(key)
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
        obs, gt_actions, state, subtask, task_str, _ep_i, _fr_i = get_frame_data(
            dataset, global_idx, int(cfg.policy.chunk_size)
        )
        obs.update(assemble_frame_history(dataset, global_idx, memory_cfg, fps, keys))
        full_batch = adapter._make_batch(
            obs,
            task_str,
            subtask=subtask,
            metadata={"quality": 5, "mistake": False},
        )
        if "history_images_mask" not in full_batch and "history_state_values" not in full_batch:
            logging.warning("[mem_history_influence] batch carries no history tensors — skipping frame.")
            continue

        gt_norm = adapter.normalize_gt_actions(gt_actions, state)
        acts: dict[str, torch.Tensor] = {}
        for name, (img_on, st_on) in conditions.items():
            batch = adapter._make_batch(
                _variant_observation(obs, img_on, st_on),
                task_str,
                subtask=subtask,
                metadata={"quality": 5, "mistake": False},
            )
            gen = torch.Generator(device=device)
            gen.manual_seed(0)  # same flow noise across conditions
            acts[name] = (
                adapter.policy.predict_action_chunk(
                    batch,
                    inference_action_mode="continuous",
                    generator=gen,
                )
                .squeeze(0)
                .float()
                .cpu()
            )
        base = acts["none"]
        gt_mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
        row = {
            "global_idx": int(global_idx),
            "full_maxabs": float((acts["full"] - base).abs().max()),
            "full_rmse": float((acts["full"] - base).pow(2).mean().sqrt()),
            "images_maxabs": float((acts["images"] - base).abs().max()),
            "images_rmse": float((acts["images"] - base).pow(2).mean().sqrt()),
            "states_maxabs": float((acts["states"] - base).abs().max()),
            "states_rmse": float((acts["states"] - base).pow(2).mean().sqrt()),
        }
        for name in conditions:
            row[f"{name}_gt_mse"] = gt_mse[name]
            if name != "none":
                row[f"{name}_gt_mse_improvement"] = gt_mse["none"] - gt_mse[name]
        rows.append(row)

    adapter._restore_probe_cuda_graph_enabled()
    if not rows:
        logging.warning("[mem_history_influence] no frames produced measurements.")
        return

    metrics = [
        "full_maxabs", "full_rmse", "images_maxabs", "images_rmse", "states_maxabs", "states_rmse",
        "full_gt_mse", "images_gt_mse", "states_gt_mse", "none_gt_mse",
        "full_gt_mse_improvement", "images_gt_mse_improvement", "states_gt_mse_improvement",
    ]
    summary = {metric: float(np.mean([r[metric] for r in rows])) for metric in metrics}
    summary["n_frames"] = len(rows)
    with open(os.path.join(output_dir, "influence.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    channels = ["full", "images", "states"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(channels))
    axes[0].bar(x - 0.2, [summary[f"{c}_rmse"] for c in channels], width=0.4, label="RMSE")
    axes[0].bar(x + 0.2, [summary[f"{c}_maxabs"] for c in channels], width=0.4, label="max|Δ|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels)
    axes[0].set_ylabel("normalized action Δ vs no-history")
    axes[0].set_title("Influence")
    axes[0].legend()
    axes[1].bar(x, [summary[f"{c}_gt_mse_improvement"] for c in channels])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels)
    axes[1].set_ylabel("GT MSE improvement vs no-history")
    axes[1].set_title("Usefulness (positive is better)")
    fig.suptitle(f"MEM history effect on the action chunk (n={len(rows)})")
    fig.savefig(os.path.join(output_dir, "influence.png"), bbox_inches="tight", dpi=100)
    plt.close(fig)
    logging.info(
        f"[mem_history_influence] n={len(rows)}  full RMSE={summary['full_rmse']:.4f}  "
        f"full GT-MSE improvement={summary['full_gt_mse_improvement']:+.4f}  "
        f"images={summary['images_gt_mse_improvement']:+.4f}  "
        f"states={summary['states_gt_mse_improvement']:+.4f}"
    )
