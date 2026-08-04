"""Metadata-steering probe: does the quality / mistake clause reach the actions?

The action prompt carries a steering clause built by `_build_robot_text`
(processor_molmoact2.py) — "The quality is N of 5." and "The robot made a
mistake."/"...no mistakes." — trained with `metadata_dropout` so the model sees the
clause on ~85% of samples. At rollout every probe and every deployment prompt asks
for the same thing: quality 5, no mistakes. Nothing checked that asking changes
anything.

For each sampled frame we run ``predict_action_chunk`` under four metadata
conditions with IDENTICAL fixed-seed flow noise, so the only difference is the
clause:

  none : no metadata clause at all (the dropout regime)
  good : quality 5, mistake False  (what every rollout asks for)
  bad  : quality 1, mistake True   (the opposite pole)
  gt   : the frame's own dataset labels, from meta/episode_metadata.parquet +
         meta/mistakes.parquet via the same spans training uses

Three readouts, in decreasing order of what they prove:

1. **separation** = ``||a(good) - a(bad)||`` — the steering range the policy
   exposes. Zero means the clause is decorative and conditions 2-3 are moot.
2. **usefulness** = ``MSE(none, GT) - MSE(cond, GT)`` in normalized action space.
   `gt` should be the best of the four: conditioning on what actually happened is
   the easiest version of the prediction problem. If `good` beats `gt` on frames
   the dataset marks as bad, the clause is being ignored rather than obeyed.
3. **mistake split** — the same numbers restricted to frames whose GT mistake flag
   is set. That is the only subset where `good` and `gt` disagree by construction,
   so it is where steering has to show up if it works at all.

Registered probe: enable with ``probe_parameters.enable_metadata_steering``.
"""

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    frame_metadata_lookup,
    joint_names_for_dim,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)

_CONDITION_STYLE = {
    "none": ("#E63946", "--"),
    "good": ("#2A9D8F", "-"),
    "bad": ("#F4A261", "-."),
    "gt": ("#457B9D", ":"),
}

_STEERED = {
    "none": None,
    "good": {"quality": 5, "mistake": False},
    "bad": {"quality": 1, "mistake": True},
}

def _mean(rows: list[dict], metric: str) -> float:
    values = [row[metric] for row in rows if row.get(metric) is not None]
    return float(np.mean(values)) if values else float("nan")


def _render_summary(rows: list[dict], conditions: list[str], output_path: str) -> None:
    """Influence, usefulness, and the good-vs-bad separation split by GT mistake."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    influence = [c for c in conditions if c != "none"]
    x = np.arange(len(influence))

    axes[0].bar(x - 0.2, [_mean(rows, f"{c}_rmse") for c in influence], width=0.4, label="RMSE")
    axes[0].bar(x + 0.2, [_mean(rows, f"{c}_maxabs") for c in influence], width=0.4, label="max|Δ|")
    axes[0].set_xticks(x, influence)
    axes[0].set_ylabel("normalized action Δ vs no-clause")
    axes[0].set_title("Influence of the metadata clause")
    axes[0].legend()

    axes[1].bar(x, [_mean(rows, f"{c}_gt_mse_improvement") for c in influence])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, influence)
    axes[1].set_ylabel("GT MSE improvement vs no-clause")
    axes[1].set_title("Usefulness (positive is better)")

    clean = [row["good_vs_bad_rmse"] for row in rows if not row["gt_mistake"]]
    flagged = [row["good_vs_bad_rmse"] for row in rows if row["gt_mistake"]]
    groups = [values for values in (clean, flagged) if values]
    labels = [name for name, values in (("no mistake", clean), ("GT mistake", flagged)) if values]
    if groups:
        axes[2].boxplot(groups, tick_labels=labels)
    axes[2].set_ylabel("||a(good) - a(bad)|| (RMSE)")
    axes[2].set_title("Steering range, split by GT mistake flag")

    fig.suptitle(f"Metadata steering (n={len(rows)})")
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def _render_example(diagnostic: dict, output_path: str) -> None:
    """Per-joint chunk under each metadata condition, GT overlaid."""
    gt = diagnostic["gt"]
    action_dim = gt.shape[-1]
    names = joint_names_for_dim(action_dim)
    steps = np.arange(gt.shape[0])

    n_cols = 4
    n_rows = (action_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.0 * n_rows), squeeze=False)
    for joint in range(action_dim):
        ax = axes[joint // n_cols][joint % n_cols]
        ax.plot(steps, gt[:, joint], color="black", linewidth=2.0, label="GT")
        for condition, actions in diagnostic["acts"].items():
            color, linestyle = _CONDITION_STYLE[condition]
            ax.plot(steps, actions[:, joint], color=color, linestyle=linestyle,
                    linewidth=1.25, label=condition)
        ax.set_title(f"{names[joint]} (normalized)")
        ax.grid(True, alpha=0.25, linestyle=":")
        if joint == 0:
            ax.legend(fontsize=8, ncol=2)
    for unused in range(action_dim, n_rows * n_cols):
        axes[unused // n_cols][unused % n_cols].axis("off")

    row = diagnostic["row"]
    fig.suptitle(
        f"episode {row['episode_idx']}, frame {row['frame_idx']}  |  "
        f"GT quality={row['gt_quality']} mistake={row['gt_mistake']}  |  "
        f"good-vs-bad RMSE={row['good_vs_bad_rmse']:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str) -> None:
    memory_cfg = getattr(cfg.policy, "memory", None)
    if memory_cfg is None or not getattr(memory_cfg, "metadata_enabled", False):
        logging.info("[metadata_steering] metadata is disabled in the policy config — skipping.")
        return
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[metadata_steering] needs continuous flow actions — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    device = adapter.device
    chunk_size = int(cfg.policy.chunk_size)

    gt_metadata = frame_metadata_lookup(dataset)
    if not gt_metadata:
        logging.warning(
            "[metadata_steering] no meta/episode_metadata.parquet + mistakes.parquet on "
            f"{dataset.root} — the 'gt' condition is unavailable; running the steered pair only."
        )

    adapter._set_probe_cuda_graph_enabled(False)  # prompt changes per condition; keep eager
    samples = sample_episodes_evenly(
        dataset, p.n_frames_per_episode, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    try:
        for _ep, _fr, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size, metadata=None)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])
            labels = gt_metadata.get(global_idx)

            conditions = dict(_STEERED)
            if labels is not None:
                conditions["gt"] = labels

            acts: dict[str, torch.Tensor] = {}
            for name, metadata in conditions.items():
                generator = torch.Generator(device=device)
                generator.manual_seed(0)  # same flow noise across conditions
                acts[name] = adapter.predict_action_chunk(
                    frame["obs"], frame["task"], state=frame["state"],
                    subtask=frame["subtask"], metadata=metadata, generator=generator,
                )[1]

            base = acts["none"]
            gt_mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
            row = {
                "global_idx": int(global_idx),
                "episode_idx": int(_ep),
                "frame_idx": int(_fr),
                "gt_quality": None if labels is None else labels["quality"],
                "gt_mistake": bool(labels["mistake"]) if labels is not None else False,
                "good_vs_bad_rmse": float((acts["good"] - acts["bad"]).pow(2).mean().sqrt()),
                "good_vs_bad_maxabs": float((acts["good"] - acts["bad"]).abs().max()),
            }
            for name, act in acts.items():
                row[f"{name}_gt_mse"] = gt_mse[name]
                if name != "none":
                    row[f"{name}_rmse"] = float((act - base).pow(2).mean().sqrt())
                    row[f"{name}_maxabs"] = float((act - base).abs().max())
                    row[f"{name}_gt_mse_improvement"] = gt_mse["none"] - gt_mse[name]
            rows.append(row)
            diagnostics.append({"gt": gt_norm, "acts": acts, "row": row})
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[metadata_steering] no frames produced measurements.")
        return

    present = sorted({name for row in rows for name in ("good", "bad", "gt") if f"{name}_gt_mse" in row})
    conditions = ["none"] + present
    flagged = [row for row in rows if row["gt_mistake"]]

    summary = {
        "n_frames": len(rows),
        "n_frames_gt_mistake": len(flagged),
        "conditions": conditions,
        "good_vs_bad_rmse": _mean(rows, "good_vs_bad_rmse"),
        "good_vs_bad_maxabs": _mean(rows, "good_vs_bad_maxabs"),
        "good_vs_bad_rmse_on_gt_mistake": _mean(flagged, "good_vs_bad_rmse") if flagged else None,
    }
    for name in present:
        summary[f"{name}_rmse"] = _mean(rows, f"{name}_rmse")
        summary[f"{name}_gt_mse_improvement"] = _mean(rows, f"{name}_gt_mse_improvement")
        if flagged:
            summary[f"{name}_gt_mse_improvement_on_gt_mistake"] = _mean(
                flagged, f"{name}_gt_mse_improvement"
            )
    for name in conditions:
        summary[f"{name}_gt_mse"] = _mean(rows, f"{name}_gt_mse")

    with open(os.path.join(output_dir, "metadata_steering.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    _render_summary(rows, conditions, os.path.join(output_dir, "metadata_steering.png"))

    examples_dir = os.path.join(output_dir, "examples")
    makedirs(examples_dir)
    ranked = sorted(range(len(rows)), key=lambda i: -rows[i]["good_vs_bad_rmse"])
    examples = []
    for index in ranked[:3]:
        row = rows[index]
        name = f"examples/sep_ep{row['episode_idx']:04d}_fr{row['frame_idx']:06d}.png"
        _render_example(diagnostics[index], os.path.join(output_dir, name))
        examples.append(
            Panel(
                name,
                f"Largest separation #{len(examples) + 1} — episode {row['episode_idx']}, "
                f"frame {row['frame_idx']}",
                "The predicted chunk per joint under each metadata condition — ``none``, "
                "``good`` (quality 5, no mistake), ``bad`` (quality 1, mistake) and ``gt`` "
                "(the frame's own labels) — against the demonstrated chunk in black. This is "
                "one of the three frames where the clause moved the chunk most, so if the "
                "four coloured lines still sit on top of each other here, the clause is "
                "decorative everywhere.",
            )
        )

    write_index(
        output_dir,
        sys.modules[__name__],
        title="Metadata Steering",
        group="Steering",
        claim="Does the quality / mistake clause reach the actions, or is it decorative?",
        summary=summary,
        see_also=["subtask_sweep", "mem_history_influence"],
        metrics=[
            Metric(
                "good_vs_bad_rmse", "Steering range (good vs bad)", good="high", fmt=4, primary=True,
                note=(
                    "Compare against the flow-noise floor from Subtask Sweep before believing a "
                    "small non-zero value."
                ),
                refs=["subtask_sweep"],
            ),
            Metric("good_vs_bad_maxabs", "Steering range, max |Δ|", good="high", fmt=4),
            Metric(
                "gt_gt_mse_improvement", "Usefulness of GT labels", good="high", fmt=4, baseline=0.0, primary=True,
                note="Negative means conditioning on what actually happened made the prediction worse.",
            ),
            Metric("good_gt_mse_improvement", "Usefulness of quality-5 / no-mistake",
                   good="high", fmt=4, baseline=0.0, primary=True),
            Metric(
                "n_frames_gt_mistake", "Frames flagged as mistakes", good="none", fmt=0, primary=True,
                note=(
                    "Zero here empties the only split where 'good' and 'gt' disagree by "
                    "construction, so the mistake-split numbers below it mean nothing."
                ),
            ),
            Metric("n_frames", "Frames measured", good="none", fmt=0),
        ],
        panels=[
            Panel(
                "metadata_steering.png",
                "Influence, usefulness, and steering range split by the GT mistake flag",
                "The bars are named after the metadata clause the prompt carried, not after "
                "the result: ``good`` = quality 5 and no mistake (what every rollout asks "
                "for), ``bad`` = quality 1 and mistake True (the opposite pole), ``gt`` = the "
                "frame's own dataset labels. All three are measured against ``none``, no "
                "clause at all, at identical flow noise.\n\n"
                "**Left** — how far each clause moves the chunk away from the no-clause "
                "prediction, as RMSE and as $\\max|\\Delta a|$ in normalised action space. This "
                "is influence only: a tall bar says the clause was read, not that reading it "
                "helped.\n\n"
                "**Middle** — how much GT MSE that movement removes, again against ``none``. "
                "Positive is better and a bar below zero means the clause actively hurt. "
                "``gt`` should be the tallest of the three: conditioning on what actually "
                "happened is the easiest version of the prediction problem.\n\n"
                "**Right** — the same steering range $\\|a(good) - a(bad)\\|$ as a box plot, "
                "split by whether the frame is flagged as a mistake in the dataset. Frames "
                "with the flag are the only ones where ``good`` and ``gt`` disagree by "
                "construction, so that is where a working clause has to separate.",
                primary=True,
                refs=["subtask_sweep"],
            ),
            *examples,
        ],
    )

    logging.info(
        f"[metadata_steering] n={len(rows)} ({len(flagged)} GT-mistake)  "
        f"good-vs-bad RMSE={summary['good_vs_bad_rmse']:.4f}  "
        + "  ".join(f"{name} improvement={summary[f'{name}_gt_mse_improvement']:+.4f}" for name in present)
    )
