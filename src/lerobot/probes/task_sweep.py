r"""Task-string intervention against the flow-seed floor.

For each held-out frame, keep the full deployment context fixed (images, depth,
state/history, true subtask and metadata), sweep every task string in the dataset
under one identical flow seed, and compare the resulting action spread with the
same true-task prompt decoded under different seeds.

The headline is S_task = mean pairwise task RMSE / mean pairwise reseed RMSE.
S_task below one means flow noise selects more of the chunk than the task string.
The true task's MSE rank against the demonstration is secondary and should only
be read after the task effect clears that floor.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.subtask_sweep import _caption, _pairwise_rmse
from lerobot.probes.utils import (
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)


def _vocabulary(dataset, limit: int) -> list[str]:
    table = getattr(getattr(dataset, "meta", None), "tasks", None)
    if table is None or not hasattr(table, "index"):
        return []
    tasks = [str(name) for name in table.index]
    return tasks[:limit] if limit > 0 else tasks


def _render(rows: list[dict], tasks: list[str], example: dict, output_dir: str) -> None:
    fig = plt.figure(figsize=(17, 6.2))
    grid = fig.add_gridspec(1, 3, wspace=0.28, left=0.045, right=0.985, top=0.80, bottom=0.30)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
    separation = float(np.median([row["separation"] for row in rows]))

    axes[0].boxplot(
        [
            [row["task_spread_mean"] for row in rows],
            [row["seed_floor_mean"] for row in rows],
        ],
        tick_labels=["task vocabulary", "flow seed"],
    )
    axes[0].set_ylabel("pairwise RMSE (normalized actions)")
    axes[0].set_title(
        "Does the task move the chunk\nmore than noise does?\n"
        f"median separation = {separation:.2f}x"
    )
    _caption(axes[0], [
        f"One frame gives one point to each box: pairwise RMSE over {len(tasks)} task-conditioned",
        "chunks on the left and reseeded true-task chunks on the right. Every other input is fixed.",
        r"Read the ratio $S_{task}$ in the title; neither raw box has a verdict of its own.",
    ])

    ranks = [row["gt_rank"] for row in rows if row["gt_rank"] is not None]
    if ranks:
        n_tasks = len(tasks)
        uniform_mean = (n_tasks + 1) / 2
        uniform_se = np.sqrt((n_tasks**2 - 1) / (12 * len(ranks)))
        axes[1].hist(
            ranks,
            bins=np.arange(0.5, n_tasks + 1.5),
            color="#457B9D",
            edgecolor="white",
        )
        axes[1].axhline(
            len(ranks) / n_tasks,
            color="#E63946",
            linestyle="--",
            label="uniform",
        )
        axes[1].set_xlabel(f"rank of true task among {n_tasks} tasks (1 = best)")
        axes[1].set_ylabel("frames")
        axes[1].set_title(
            f"Is it used correctly?\nmean rank {np.mean(ranks):.1f} vs "
            f"{uniform_mean:.1f} ± {uniform_se:.1f} if uniform  ·  "
            f"top-1 {np.mean([rank == 1 for rank in ranks]):.0%}"
        )
        axes[1].legend(fontsize=8)
        _caption(axes[1], [
            "Score every task-conditioned chunk against the demonstration and rank the true task.",
            r"Rank 1 is desirable, but ranks are sampler noise until $S_{task}$ clears its floor.",
        ])
    else:
        axes[1].text(0.5, 0.5, "no true task on sampled frames", ha="center", va="center")
        axes[1].axis("off")

    stacked = torch.stack(list(example["acts"].values())).float()
    joint = int(stacked.std(dim=0).mean(dim=0).argmax())
    steps = np.arange(example["gt"].shape[0])
    for task, chunk in example["acts"].items():
        axes[2].plot(steps, chunk[:, joint], linewidth=1.1, alpha=0.8, label=task)
    axes[2].plot(steps, example["gt"][:, joint], color="black", linewidth=2.3, label="GT")
    if example["gt_task"] in example["acts"]:
        axes[2].plot(
            steps,
            example["acts"][example["gt_task"]][:, joint],
            color="#E63946",
            linestyle="--",
            linewidth=1.8,
            label="true-task chunk",
        )
    axes[2].set_title(
        f"Task fan — ep {example['episode_idx']} fr {example['frame_idx']}\n"
        f"most task-sensitive joint {joint}"
    )
    axes[2].set_xlabel("chunk step")
    axes[2].set_ylabel("normalized action")
    axes[2].grid(True, alpha=0.25, linestyle=":")
    axes[2].legend(fontsize=7)
    _caption(axes[2], [
        "One frame and joint: every coloured trajectory changes only the task string.",
        "Black is the demonstration and dashed red is the chunk produced under the true task.",
    ])

    fig.suptitle(
        f"Task clause → action transfer (n={len(rows)} frames, {len(tasks)} tasks)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(os.path.join(output_dir, "task_sweep.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[task_sweep] needs continuous flow actions — skipping.")
        return

    p = cfg.probe_parameters
    tasks = _vocabulary(dataset, int(getattr(p, "task_sweep_max_labels", 16)))
    if len(tasks) < 2:
        logging.info("[task_sweep] dataset has fewer than two task strings — skipping.")
        return

    makedirs(output_dir)
    device = adapter.device
    chunk_size = int(cfg.policy.chunk_size)
    n_seeds = max(int(getattr(p, "task_sweep_n_seeds", 3)), 2)
    n_frames = int(getattr(p, "task_sweep_n_frames", 8))
    samples = sample_episodes_evenly(
        dataset, n_frames, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    logging.info(
        f"[task_sweep] {len(samples)} frames x ({len(tasks)} tasks + {n_seeds} seeds) = "
        f"{len(samples) * (len(tasks) + n_seeds)} forward passes"
    )

    adapter._set_probe_cuda_graph_enabled(False)
    rows: list[dict] = []
    examples: list[dict] = []
    try:
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])

            def predict(task: str, seed: int, frame: dict = frame) -> torch.Tensor:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                return adapter.predict_action_chunk(
                    frame["obs"],
                    task,
                    state=frame["state"],
                    subtask=frame["subtask"],
                    metadata=frame["metadata"],
                    generator=generator,
                )[1]

            acts = {task: predict(task, 0) for task in tasks}
            seed_draws = [predict(frame["task"], seed) for seed in range(1, n_seeds + 1)]
            task_mean, task_max = _pairwise_rmse(list(acts.values()))
            seed_mean, seed_max = _pairwise_rmse(seed_draws)
            mse = {task: float((chunk - gt_norm).pow(2).mean()) for task, chunk in acts.items()}
            ordered = sorted(mse, key=mse.get)
            gt_rank = ordered.index(frame["task"]) + 1 if frame["task"] in mse else None
            rows.append({
                "episode_idx": int(ep_idx),
                "frame_idx": int(fr_idx),
                "global_idx": int(global_idx),
                "gt_task": frame["task"],
                "task_spread_mean": task_mean,
                "task_spread_max": task_max,
                "seed_floor_mean": seed_mean,
                "seed_floor_max": seed_max,
                "separation": task_mean / max(seed_mean, 1e-9),
                "gt_rank": gt_rank,
                "best_task": ordered[0],
                "gt_mse_by_task": mse,
            })
            examples.append({
                "acts": acts,
                "gt": gt_norm,
                "episode_idx": int(ep_idx),
                "frame_idx": int(fr_idx),
                "gt_task": frame["task"],
            })
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[task_sweep] no frames produced measurements.")
        return

    ranks = [row["gt_rank"] for row in rows if row["gt_rank"] is not None]
    separations = [row["separation"] for row in rows]
    summary = {
        "n_frames": len(rows),
        "vocabulary": tasks,
        "n_tasks": len(tasks),
        "task_spread_mean": float(np.mean([row["task_spread_mean"] for row in rows])),
        "seed_floor_mean": float(np.mean([row["seed_floor_mean"] for row in rows])),
        "separation_mean": float(np.mean(separations)),
        "separation_median": float(np.median(separations)),
        "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
        "gt_rank_uniform_expectation": (len(tasks) + 1) / 2,
        "gt_rank_top1_fraction": float(np.mean([rank == 1 for rank in ranks])) if ranks else None,
        "per_frame": rows,
    }
    with open(os.path.join(output_dir, "task_sweep.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    _render(rows, tasks, examples[0], output_dir)

    write_index(
        output_dir,
        sys.modules[__name__],
        title="Task Sweep",
        group="Steering",
        claim="Does swapping the task string move the chunk more than flow noise does?",
        summary=summary,
        see_also=["subtask_sweep", "metadata_steering", "attention_budget"],
        metrics=[
            Metric(
                "separation_median",
                "task / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                primary=True,
                note="S_task=1 means task replacement and flow reseeding move the chunk equally far.",
            ),
            Metric("gt_rank_mean", "mean rank of true task", good="low", fmt=2),
        ],
        panels=[
            Panel(
                "task_sweep.png",
                "Task spread against flow noise, true-task rank, and one action fan",
                "First require task replacement to clear the flow-seed floor; then ask whether "
                "the true task ranks near one. Every non-task input stays at deployment value.",
                primary=True,
                refs=["subtask_sweep", "metadata_steering"],
            )
        ],
    )
    logging.info(
        f"[task_sweep] n={len(rows)}  task spread={summary['task_spread_mean']:.4f}  "
        f"seed floor={summary['seed_floor_mean']:.4f}  "
        f"separation={summary['separation_median']:.2f}x"
    )
