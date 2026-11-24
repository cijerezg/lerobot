#!/usr/bin/env python
r"""The tip adapter on held-out frames (docs/notes/tip_target_adapter.md).

The adapter reads the wrist and top image tokens, the wrist depth points and the subtask text, and
gives one distribution over the 27 x 27 wrist patches plus "no target". Its vector (the mean of the
patches' depth points weighted by their probability) fills the prompt's target clause at inference.
This probe runs the adapter alone (vision tower, no prompt forward) on frames spread over every
validation episode and on a sample of the training roots.

Three readouts:

- **Numbers** on val and train: the loss, how often "target in view or not" is right, how often the
  peak patch is the target's patch or next to it, and the vector error in cm. Two references make
  the error readable. The **floor** is the error of always answering the mean offset of the training
  roots: an adapter that is not clearly under it has learned nothing. The **ceiling** is the error
  of the vector read through the true label instead of the adapter's scores: what the depth readout
  alone costs, so the adapter cannot do better.
- **Timeline per validation episode**: the three errors and the distance to the target against
  time, the probability of "no target" below, the gripper events as vertical lines.
- **Heatmaps**: the adapter's distribution over the wrist image on frames spread over each episode,
  with the true target pixel and the peak patch.

The vector error, the floor and the ceiling are means over the frames whose target is in the wrist
view. Registered probe: enable with ``probe_parameters.enable_tip_target``.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from lerobot.policies.molmoact2.tip_target import (  # noqa: E402
    GRID,
    TIP_WRIST_CAMERA,
    tip_label,
    tip_target_loss,
    tip_vector,
)
from lerobot.probes.manifest import Metric, Panel, write_index  # noqa: E402
from lerobot.probes.objective import TRAIN_COLOR, VAL_COLOR, training_datasets  # noqa: E402
from lerobot.probes.utils import (  # noqa: E402
    as_image,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)
from lerobot.utils.tip_target import TIP_DEPTH_KEY, TIP_TARGET_KEY, TIP_TARGET_STORE, tip_target_rows  # noqa: E402

FLOOR_COLOR = "#6C757D"
CEILING_COLOR = "#2A9D8F"
NUMBERS = ("loss", "none_accuracy", "peak_on_patch", "peak_within_one", "error_cm", "floor_cm", "ceiling_cm")


def frame_inputs(dataset, cfg, global_idx: int) -> dict:
    """One frame as the adapter needs it: current cameras only, the adapter reads no history."""
    return probe_frame_inputs(
        dataset, cfg, global_idx, 1, with_depth=False, with_history=False, with_gripper_event_targets=False
    )


def training_mean_offset(cfg) -> np.ndarray:
    """The mean offset [3] (m, tool frame) over the training roots' frames with the target in the wrist view."""
    rows = [tip_target_rows(source.root) for source in cfg.dataset.sources]
    rows = np.concatenate([r for r in rows if r is not None])
    return rows[rows[:, 0] > 0, 1:4].mean(0)


def measure(adapter, name: str, dataset, cfg, split: str, n_per_episode: int, mean_offset: np.ndarray):
    """One adapter forward per sampled frame. Returns the rows and the distributions [frames, 730]."""
    from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png

    if tip_target_rows(dataset.root) is None:
        logging.warning(f"[tip_target] {split}/{name}: no targets.parquet for this root, skipped.")
        return [], np.zeros((0, GRID * GRID + 1), dtype=np.float32)
    table = pd.read_parquet(f"{TIP_TARGET_STORE}/{os.path.basename(str(dataset.root).rstrip('/'))}/targets.parquet")
    table = table.set_index("index")
    samples = sample_episodes_evenly(
        dataset, n_per_episode, None, cfg.probe_parameters.random_seed, probe_image_stride(cfg)
    )
    rows, probabilities = [], []
    for episode_idx, frame_idx, global_idx in samples:
        frame = frame_inputs(dataset, cfg, global_idx)
        batch = adapter._make_batch(
            frame["obs"], frame["task"], subtask=frame["subtask"], metadata=frame["metadata"]
        )
        depth = load_depth_png(dataset.root, "wrist.depth", episode_idx, frame_idx)
        batch[TIP_DEPTH_KEY] = torch.from_numpy(depth.astype(np.int32))[None].to(adapter._device)
        scores, points = adapter.policy.tip_scores(batch)
        target = frame["obs"][f"probe_complementary.{TIP_TARGET_KEY}"].float().to(scores.device)
        _, metrics = tip_target_loss(scores, target, points, "mean")
        p_none, vector = tip_vector(scores, points)
        label_vector = tip_vector(tip_label(target).clamp_min(1e-30).log(), points)[1]
        offset = target[0, 1:4]
        in_view = bool(metrics["tip_target_frames"])
        label = table.loc[global_idx]
        row = {
            "split": split,
            "source": name,
            "episode_idx": int(episode_idx),
            "frame_idx": int(frame_idx),
            "global_idx": int(global_idx),
            "seconds": frame_idx / dataset.fps,
            "subtask": frame["subtask"],
            "kind": label["kind"],
            "in_view": in_view,
            "loss": metrics["tip_loss"],
            "none_accuracy": metrics["tip_none_accuracy"],
            "p_none": p_none.item(),
            "peak_on_patch": None,
            "peak_within_one": None,
            "error_cm": None,
            "floor_cm": None,
            "ceiling_cm": None,
            "distance_cm": None,
            "u": None,
            "v": None,
        }
        if in_view:
            row.update(
                peak_on_patch=metrics["tip_peak_on_patch"],
                peak_within_one=metrics["tip_peak_within_one"],
                error_cm=metrics["tip_vector_error_cm"],
                floor_cm=100 * float(np.linalg.norm(mean_offset - offset.cpu().numpy())),
                ceiling_cm=100 * (label_vector[0] - offset).norm().item(),
                distance_cm=100 * offset.norm().item(),
                u=target[0, 4].item(),
                v=target[0, 5].item(),
            )
        rows.append(row)
        probabilities.append(scores.softmax(-1)[0].cpu().numpy())
    logging.info(f"[tip_target] {split}/{name}: {len(rows)} frames measured")
    return rows, np.stack(probabilities)


def events(dataset) -> dict[int, list[tuple[float, str]]]:
    """Per episode, the gripper events of the root's targets as (seconds into the episode, kind)."""
    table = pd.read_parquet(f"{TIP_TARGET_STORE}/{os.path.basename(str(dataset.root).rstrip('/'))}/targets.parquet")
    table = table[table.kind != "none"]
    result = {}
    for episode_idx, episode in table.groupby("episode_index"):
        first = episode["index"].iloc[0] - episode["frame_index"].iloc[0]
        pairs = episode.drop_duplicates("event_index")
        result[int(episode_idx)] = [
            ((event - first) / dataset.fps, kind) for event, kind in zip(pairs.event_index, pairs.kind)
        ]
    return result


def split_summary(rows: list[dict]) -> dict:
    """Means over the split's frames; the in-view numbers over the frames with the target in the wrist view."""
    summary = {"n_frames": len(rows), "n_in_view": sum(r["in_view"] for r in rows)}
    for key in NUMBERS:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    errors = [r["error_cm"] for r in rows if r["in_view"]]
    summary["error_cm_median"] = float(np.median(errors)) if errors else None
    if rows:
        share = summary["n_in_view"] / len(rows)
        summary["none_accuracy_majority"] = max(share, 1 - share)
    return summary


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=float)


def render_timeline(rows: list[dict], episode_events: list[tuple[float, str]], path: str, title: str) -> None:
    """The errors and the distance to the target against time, the probability of "no target" below."""
    seconds = column(rows, "seconds")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, height_ratios=(2, 1))
    top.plot(seconds, column(rows, "distance_cm"), color="#CED4DA", linewidth=1.2, label="distance to the target")
    top.plot(seconds, column(rows, "floor_cm"), color=FLOOR_COLOR, linewidth=1.2, linestyle=":", label="floor (mean training offset)")
    top.plot(seconds, column(rows, "ceiling_cm"), color=CEILING_COLOR, linewidth=1.2, linestyle="--", label="ceiling (true label)")
    top.plot(seconds, column(rows, "error_cm"), color=VAL_COLOR, linewidth=1.6, label="adapter")
    top.set_yscale("log")
    top.set_ylabel("cm")
    top.set_title(title, fontsize=11)
    top.legend(fontsize=8, ncol=4, loc="upper right")
    out_of_view = np.array([not r["in_view"] for r in rows])
    bottom.fill_between(seconds, 0, 1, where=out_of_view, step="mid", color="#DEE2E6", label="truth: no target in the wrist view")
    bottom.plot(seconds, column(rows, "p_none"), color=VAL_COLOR, linewidth=1.4, label='adapter: probability of "no target"')
    bottom.axhline(0.5, color="black", linewidth=0.6, linestyle=":")
    bottom.set_ylim(0, 1)
    bottom.set_xlabel("seconds into the episode")
    bottom.legend(fontsize=8, ncol=2, loc="upper right")
    for ax in (top, bottom):
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        for event_seconds, _ in episode_events:
            ax.axvline(event_seconds, color="black", linewidth=0.9)
    for event_seconds, kind in episode_events:
        top.annotate(kind, (event_seconds, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                     textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_heatmaps(rows: list[dict], probabilities: np.ndarray, dataset, cfg, path: str, title: str) -> None:
    """The adapter's distribution over the wrist image, each map scaled by its own peak."""
    cols = 4
    fig, axes = plt.subplots(-(-len(rows) // cols), cols, figsize=(4.2 * cols, 3.7 * -(-len(rows) // cols)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row, p in zip(axes.flat, rows, probabilities):
        image = as_image(frame_inputs(dataset, cfg, row["global_idx"])["obs"][TIP_WRIST_CAMERA])
        height, width = image.shape[:2]
        patches = p[:-1].reshape(GRID, GRID)
        ax.imshow(image, extent=(0, width, height, 0))
        ax.imshow(patches, extent=(0, width, height, 0), cmap="magma", alpha=0.75 * patches / patches.max(), interpolation="nearest")
        peak = int(patches.argmax())
        ax.plot((peak % GRID + 0.5) * width / GRID, (peak // GRID + 0.5) * height / GRID, "x", color="white", markersize=9, markeredgewidth=2)
        if row["in_view"]:
            ax.plot(row["u"] * width / 640, row["v"] * height / 480, "+", color="#00E5FF", markersize=13, markeredgewidth=2)
        error = f"error {row['error_cm']:.1f} cm of {row['distance_cm']:.1f}" if row["in_view"] else "truth: no target in view"
        ax.set_title(f"{row['seconds']:.1f} s  p(no target) {row['p_none']:.2f}  {error}\n{row['subtask']}", fontsize=8)
    fig.suptitle(f"{title}  (x = peak patch, + = true target pixel)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(adapter, dataset, cfg, output_dir: str, train_dataset=None) -> dict | None:
    """Measure the adapter on val and on the training roots; write the numbers, timelines and heatmaps.

    Returns the summary so rl_offline can push the headline scalars to Aim.
    """
    if getattr(adapter.policy, "tip_adapter", None) is None:
        logging.warning("[tip_target] policy.tip_target is off: nothing to measure.")
        return None
    makedirs(output_dir)
    p = cfg.probe_parameters
    mean_offset = training_mean_offset(cfg)

    val_rows, val_probabilities = measure(
        adapter, "val", dataset, cfg, "val", p.tip_target_n_frames_per_episode, mean_offset
    )
    train_rows = []
    if train_dataset is not None:
        for name, source in training_datasets(cfg, train_dataset):
            train_rows += measure(
                adapter, name, source, cfg, "train", p.tip_target_n_train_frames_per_episode, mean_offset
            )[0]
    if not val_rows:
        return None

    summary = {"val": split_summary(val_rows), "train": split_summary(train_rows), "mean_offset_m": mean_offset.tolist()}
    with open(os.path.join(output_dir, "tip_target.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "frames.json"), "w") as f:
        json.dump(val_rows + train_rows, f, indent=1)
    np.savez_compressed(os.path.join(output_dir, "val_probabilities.npz"), probabilities=val_probabilities)

    episode_events = events(dataset)
    panels = []
    for episode_idx in sorted({r["episode_idx"] for r in val_rows}):
        picks = [i for i, r in enumerate(val_rows) if r["episode_idx"] == episode_idx]
        rows = [val_rows[i] for i in picks]
        render_timeline(
            rows, episode_events.get(episode_idx, []),
            os.path.join(output_dir, f"timeline_ep{episode_idx:04d}.png"), f"Validation episode {episode_idx}",
        )
        shown = [picks[i] for i in np.linspace(0, len(picks) - 1, min(p.tip_target_n_heatmaps_per_episode, len(picks)), dtype=int)]
        render_heatmaps(
            [val_rows[i] for i in shown], val_probabilities[shown], dataset, cfg,
            os.path.join(output_dir, f"heatmaps_ep{episode_idx:04d}.png"), f"Validation episode {episode_idx}",
        )
        panels += [
            Panel(
                f"timeline_ep{episode_idx:04d}.png",
                f"Timeline, validation episode {episode_idx}",
                "Top, log scale: the adapter's vector error against the floor (always answer the mean "
                "training offset), the ceiling (the vector read through the true label) and the distance "
                "to the target. Lines break where the target is out of the wrist view. Bottom: the "
                'probability of "no target" over the grey spans where that is the truth. Vertical lines '
                "are the gripper events.",
                primary=True,
            ),
            Panel(
                f"heatmaps_ep{episode_idx:04d}.png",
                f"Heatmaps, validation episode {episode_idx}",
                "The adapter's distribution over the wrist patches, each map scaled by its own peak, so "
                'read the probability of "no target" in the title first. x is the peak patch, + the true '
                "target pixel (absent when the target is out of view).",
                primary=True,
            ),
        ]
    _write_index(summary, panels, output_dir)

    val = summary["val"]
    logging.info(
        f"[tip_target] val: error {val['error_cm']:.2f} cm (floor {val['floor_cm']:.2f}, ceiling "
        f"{val['ceiling_cm']:.2f}), none accuracy {val['none_accuracy']:.3f}, loss {val['loss']:.3f}"
        if val["n_in_view"]
        else f"[tip_target] val: no frame with the target in view, none accuracy {val['none_accuracy']:.3f}"
    )
    return summary


def aim_scalars(summary: dict) -> dict:
    """The held-out numbers on the training run's own axes."""
    val = summary["val"]
    return {
        f"tip_probe_val_{key}": val[key]
        for key in ("loss", "none_accuracy", "peak_within_one", "error_cm", "floor_cm", "ceiling_cm")
        if val[key] is not None
    }


def _write_index(summary: dict, panels: list[Panel], output_dir: str) -> None:
    val, train = summary["val"], summary["train"]
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Tip adapter",
        group="Depth",
        claim="Does the tip adapter find the target of the next gripper event, and how far off is its vector?",
        summary=summary,
        see_also=["input_swap"],
        metrics=[
            Metric(
                "val.error_cm", "Vector error, cm (val)", good="low", fmt=2, primary=True, trend=True,
                baseline=val["floor_cm"],
                note="Adapter vector against the true offset, mean over the frames with the target in the "
                "wrist view. The baseline is the floor: always answering the mean training offset.",
            ),
            Metric("val.error_cm_median", "Vector error, median cm (val)", good="low", fmt=2),
            Metric(
                "val.ceiling_cm", "Ceiling: error through the true label, cm (val)", good="none", fmt=2,
                note="What the depth readout alone costs; the adapter cannot do better.",
            ),
            Metric("val.floor_cm", "Floor: error of the mean training offset, cm (val)", good="none", fmt=2),
            Metric(
                "val.none_accuracy", 'Right on "target in view or not" (val)', good="high", fmt=3, primary=True,
                trend=True, baseline=val.get("none_accuracy_majority"),
                note="The baseline is always answering the more common of the two.",
            ),
            Metric(
                "val.peak_within_one", "Peak patch on or next to the target's patch (val)", good="high", fmt=3,
                primary=True, trend=True,
            ),
            Metric("val.peak_on_patch", "Peak patch = the target's patch (val)", good="high", fmt=3),
            Metric("val.loss", "Loss (val)", good="low", fmt=3, trend=True),
            Metric("train.error_cm", "Vector error, cm (train)", good="low", fmt=2, baseline=train.get("floor_cm")),
            Metric("train.none_accuracy", 'Right on "target in view or not" (train)', good="high", fmt=3),
            Metric("train.peak_within_one", "Peak patch on or next to the target's patch (train)", good="high", fmt=3),
            Metric("train.loss", "Loss (train)", good="low", fmt=3),
            Metric("val.n_frames", "Val frames", good="none", fmt=0),
            Metric("val.n_in_view", "Val frames with the target in view", good="none", fmt=0),
            Metric("train.n_frames", "Train frames", good="none", fmt=0),
        ],
        panels=panels,
    )
