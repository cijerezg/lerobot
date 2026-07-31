"""Does the subtask clause reach the actions at all? (the old plan's P3)

Hop two of the memory chain. The summary memory never enters the action prompt — it
reaches behaviour only through the decoded subtask — so if the subtask clause does
not move the action chunk, the entire annotate → summarize → decode → act chain is
severed at the last step and every upstream annotation is decorative *for control*,
however good its CE looks. That makes this the probe that gates the interpretation
of the rest of the memory diagnostics, and the one that justifies deleting things.

Method: fix a frame, sweep every subtask in the dataset vocabulary through the
action prompt, identical fixed-seed flow noise on every pass, and ask two questions.

**1. Does the clause move the output more than noise does?** Vocabulary spread is
compared against a *seed floor* — the same prompt re-sampled under different flow
seeds. Both are RMSE in normalized action space, so

    separation = mean pairwise spread across labels / spread across seeds

is the honest statistic. A raw spread of 0.05 means nothing on its own; 0.05 against
a seed floor of 0.05 means the clause does nothing, and against a floor of 0.005 it
means the clause dominates. This is the number to read first.

**2. Is the clause used *correctly*?** For each frame, rank the vocabulary by MSE
against the demonstrated chunk and record where the ground-truth label lands. If the
model conditions on the clause meaningfully the GT label should rank near the top;
a rank distribution indistinguishable from uniform says the clause is read but not
understood — which looks identical to "not read" in any single-number summary.

Cost is ``n_frames x (|vocab| + n_seeds)`` forwards, so both are small knobs.

Registered probe: enable with ``probe_parameters.enable_subtask_sweep``.
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.utils import (
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)

_REBOT_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper",
]


def _vocabulary(dataset, limit: int) -> list[str]:
    """Subtask label texts from meta/subtasks.parquet (the text is the frame index)."""
    table = getattr(getattr(dataset, "meta", None), "subtasks", None)
    if table is None or not hasattr(table, "index"):
        return []
    labels = [str(name) for name in table.index]
    return labels[:limit] if limit and limit > 0 else labels


def _pairwise_rmse(chunks: list[torch.Tensor]) -> tuple[float, float]:
    """Mean and max pairwise RMSE between action chunks."""
    if len(chunks) < 2:
        return 0.0, 0.0
    values = [
        float((chunks[i] - chunks[j]).pow(2).mean().sqrt())
        for i in range(len(chunks))
        for j in range(i + 1, len(chunks))
    ]
    return float(np.mean(values)), float(np.max(values))


def _render(rows: list[dict], vocabulary: list[str], example: dict | None, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))

    spread = [row["vocab_spread_mean"] for row in rows]
    floor = [row["seed_floor_mean"] for row in rows]
    axes[0].boxplot([spread, floor], tick_labels=["subtask vocab", "flow seed"])
    axes[0].set_ylabel("pairwise RMSE (normalized actions)")
    axes[0].set_title(
        "Does the clause move the chunk\nmore than noise does?\n"
        f"median separation = {np.median([r['separation'] for r in rows]):.2f}x"
    )

    ranks = [row["gt_rank"] for row in rows if row["gt_rank"] is not None]
    if ranks:
        n_labels = len(vocabulary)
        axes[1].hist(ranks, bins=np.arange(0.5, n_labels + 1.5, 1.0), color="#457B9D",
                     edgecolor="white")
        axes[1].axhline(len(ranks) / max(n_labels, 1), color="#E63946", linestyle="--",
                        linewidth=1.2, label="uniform (clause not understood)")
        axes[1].set_xlabel(f"rank of the GT subtask among {n_labels} labels (1 = best)")
        axes[1].set_ylabel("frames")
        axes[1].set_title(
            f"Is it used correctly?\nmean rank {np.mean(ranks):.1f} vs "
            f"{(n_labels + 1) / 2:.1f} if uniform"
        )
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "no GT subtask on the sampled frames", ha="center", va="center")
        axes[1].axis("off")

    if example is not None:
        joint = example["joint"]
        steps = np.arange(example["gt"].shape[0])
        for label, chunk in example["acts"].items():
            axes[2].plot(steps, chunk[:, joint], linewidth=1.0, alpha=0.75, label=label)
        axes[2].plot(steps, example["gt"][:, joint], color="black", linewidth=2.2, label="GT")
        axes[2].set_title(
            f"Fan across the vocabulary — ep {example['episode_idx']} fr {example['frame_idx']}\n"
            f"joint: {_REBOT_JOINT_NAMES[joint] if joint < len(_REBOT_JOINT_NAMES) else joint}"
        )
        axes[2].set_xlabel("chunk step")
        axes[2].grid(True, alpha=0.25, linestyle=":")
        axes[2].legend(fontsize=5, ncol=2)

    fig.suptitle(
        f"Subtask clause → action transfer (n={len(rows)} frames, {len(vocabulary)} labels)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(output_dir, "subtask_sweep.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[subtask_sweep] needs continuous flow actions — skipping.")
        return

    p = cfg.probe_parameters
    vocabulary = _vocabulary(dataset, int(getattr(p, "subtask_sweep_max_labels", 16)))
    if len(vocabulary) < 2:
        logging.info("[subtask_sweep] dataset has no subtask vocabulary — skipping.")
        return

    makedirs(output_dir)
    device = adapter.device
    chunk_size = int(cfg.policy.chunk_size)
    n_seeds = max(int(getattr(p, "subtask_sweep_n_seeds", 3)), 2)
    n_frames = int(getattr(p, "subtask_sweep_n_frames", 8))

    samples = sample_episodes_evenly(
        dataset, n_frames, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    logging.info(
        f"[subtask_sweep] {len(samples)} frames x ({len(vocabulary)} labels + {n_seeds} seeds) = "
        f"{len(samples) * (len(vocabulary) + n_seeds)} forward passes"
    )

    adapter._set_probe_cuda_graph_enabled(False)
    rows: list[dict] = []
    example: dict | None = None
    try:
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])

            def predict(subtask: str, seed: int) -> torch.Tensor:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                return adapter.predict_action_chunk(
                    frame["obs"], frame["task"], state=frame["state"], subtask=subtask,
                    metadata=frame["metadata"], generator=generator,
                )[1]

            acts = {label: predict(label, 0) for label in vocabulary}
            # Seed floor: same clause, different noise. Without it the vocabulary
            # spread has no scale and any number looks like evidence.
            gt_subtask = frame["subtask"] or vocabulary[0]
            seed_draws = [predict(gt_subtask, seed) for seed in range(1, n_seeds + 1)]

            vocab_mean, vocab_max = _pairwise_rmse(list(acts.values()))
            seed_mean, seed_max = _pairwise_rmse(seed_draws)
            mse = {label: float((chunk - gt_norm).pow(2).mean()) for label, chunk in acts.items()}
            ordered = sorted(mse, key=lambda label: mse[label])

            gt_rank = ordered.index(frame["subtask"]) + 1 if frame["subtask"] in mse else None
            rows.append(
                {
                    "episode_idx": int(ep_idx), "frame_idx": int(fr_idx), "global_idx": int(global_idx),
                    "gt_subtask": frame["subtask"],
                    "vocab_spread_mean": vocab_mean, "vocab_spread_max": vocab_max,
                    "seed_floor_mean": seed_mean, "seed_floor_max": seed_max,
                    "separation": vocab_mean / max(seed_mean, 1e-9),
                    "gt_rank": gt_rank,
                    "best_label": ordered[0],
                    "gt_mse_by_label": mse,
                }
            )
            if example is None:
                example = {
                    "acts": {label: chunk for label, chunk in acts.items()},
                    "gt": gt_norm, "joint": 0,
                    "episode_idx": int(ep_idx), "frame_idx": int(fr_idx),
                }
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[subtask_sweep] no frames produced measurements.")
        return

    ranks = [row["gt_rank"] for row in rows if row["gt_rank"] is not None]
    separations = [row["separation"] for row in rows]
    summary = {
        "n_frames": len(rows),
        "vocabulary": vocabulary,
        "n_labels": len(vocabulary),
        "vocab_spread_mean": float(np.mean([r["vocab_spread_mean"] for r in rows])),
        "seed_floor_mean": float(np.mean([r["seed_floor_mean"] for r in rows])),
        "separation_mean": float(np.mean(separations)),
        "separation_median": float(np.median(separations)),
        "gt_rank_mean": float(np.mean(ranks)) if ranks else None,
        "gt_rank_uniform_expectation": (len(vocabulary) + 1) / 2,
        "gt_rank_top1_fraction": float(np.mean([r == 1 for r in ranks])) if ranks else None,
        "verdict_note": (
            "separation ~1 => the clause moves the chunk no more than flow noise does "
            "(hop two is dead). separation >> 1 with gt_rank_mean ~ uniform => the clause "
            "is read but not understood."
        ),
        "per_frame": rows,
    }
    with open(os.path.join(output_dir, "subtask_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _render(rows, vocabulary, example, output_dir)

    logging.info(
        f"[subtask_sweep] n={len(rows)}  vocab spread={summary['vocab_spread_mean']:.4f}  "
        f"seed floor={summary['seed_floor_mean']:.4f}  separation={summary['separation_median']:.2f}x  "
        + (
            f"GT rank {summary['gt_rank_mean']:.1f}/{len(vocabulary)} "
            f"(uniform {summary['gt_rank_uniform_expectation']:.1f}, "
            f"top-1 {summary['gt_rank_top1_fraction']:.0%})"
            if ranks else "no GT subtask on sampled frames"
        )
    )
