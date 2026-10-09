#!/usr/bin/env python
r"""The EE mixture loss, term by term and frame by frame (docs/ee_mixture_loss/TODO.md Phase 4).

The training loss with the hand block on is three terms, each a block mean at the same
chance level: the **joint block** flow loss, the **hand block** flow loss (position,
rotation, aperture, averaged) and the **FK term** (fingertip pose of the implied clean
joint sample against the demonstrated fingertip). This probe measures the three on single
frames with the pinned flow timesteps and noise of the objective probe, on held-out ReBot
episodes, on the ReBot training sources, and on anchors of the diverse corpus with their
own layout ids, so every robot in the mixture is represented.

Two readouts:

- **Histograms by robot** of each term and of the gap $\mathcal{L}_{hand} -
  \mathcal{L}_{joint}$, train against val where a robot has both.
- **Frames at the gap percentiles.** Held-out frames nearest $p_5$, $p_{50}$ and $p_{95}$
  of $|\mathcal{L}_{hand} - \mathcal{L}_{joint}|$: where the two heads agree in how well they
  fit, where they are typical, and where they disagree most. The sign of the gap on each
  rendered frame says which head is the worse one there.

Absolute levels are not readable (see the objective probe); compare val to train and
robots to each other. Registered probe: enable with ``probe_parameters.enable_hand_terms``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.probes.manifest import Metric, Panel, write_index  # noqa: E402
from lerobot.probes.objective import (  # noqa: E402
    TRAIN_COLOR,
    VAL_COLOR,
    _style,
    flow_timestep_grid,
    sem,
    training_datasets,
)
from lerobot.probes.utils import (  # noqa: E402
    build_episode_index,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)

TERMS = (
    ("loss_hand_joint", "joint block"),
    ("loss_hand_pose", "hand block"),
    ("loss_hand_fk", "FK term"),
)
GAP_KEY = "gap"
GAP_LABEL = "hand - joint"
BANDS = ((5, "smallest gap"), (50, "typical gap"), (95, "largest gap"))
MAX_CAMERAS = 3


def robot_name(layout_id: int | None) -> str:
    """The embodiment name of an action layout id, the histogram's column label."""
    if layout_id is None:
        return "unknown"
    from lerobot.datasets.diverse_actor_selection import ACTION_LAYOUTS

    if 0 <= int(layout_id) < len(ACTION_LAYOUTS):
        layout = ACTION_LAYOUTS[int(layout_id)]
        return str(getattr(layout, "embodiment", None) or layout.name)
    return f"layout {int(layout_id)}"


# ──────────────────────────────────────────────────────────────────────────────
# Measurement
# ──────────────────────────────────────────────────────────────────────────────


def _row(losses: dict, **meta) -> dict | None:
    """One measured frame; None when the forward produced no hand terms (flag off)."""
    if losses.get("loss_hand_joint") is None or losses.get("loss_hand_pose") is None:
        return None
    return {
        "loss_hand_joint": float(losses["loss_hand_joint"]),
        "loss_hand_pose": float(losses["loss_hand_pose"]),
        "loss_hand_fk": None if losses.get("loss_hand_fk") is None else float(losses["loss_hand_fk"]),
        "loss_flow": None if losses.get("loss_flow") is None else float(losses["loss_flow"]),
        "robot": robot_name(losses.get("layout_id")),
        "layout_id": losses.get("layout_id"),
        **meta,
    }


def measure_rebot(adapter, datasets, cfg, split: str, timesteps: np.ndarray) -> list[dict]:
    """One ``training_losses`` forward per sampled frame of every ReBot dataset in ``datasets``."""
    p = cfg.probe_parameters
    chunk_size = int(cfg.policy.chunk_size)
    stride = probe_image_stride(cfg)
    n_frames = int(getattr(p, "objective_n_frames_per_episode", None) or p.n_frames_per_episode)
    max_episodes = getattr(p, "objective_max_episodes", None) or p.max_episodes
    grid = torch.from_numpy(timesteps).float()
    per_source = None if max_episodes is None else max(int(max_episodes) // max(len(datasets), 1), 1)

    rows: list[dict] = []
    for name, dataset in datasets:
        episode_lengths = {ep: len(idx) for ep, idx in build_episode_index(dataset).items()}
        samples = sample_episodes_evenly(dataset, n_frames, per_source, p.random_seed, stride)
        skipped = 0
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            losses = adapter.training_losses(
                frame,
                flow_timesteps=grid,
                flow_noise_seed=int(p.random_seed) + int(global_idx),
                dropout=False,
            )
            row = _row(
                losses,
                kind="rebot",
                split=split,
                source=name,
                episode=f"{name}/{int(ep_idx)}",
                episode_idx=int(ep_idx),
                frame_idx=int(fr_idx),
                global_idx=int(global_idx),
                subtask=frame["subtask"],
                progress=fr_idx / max(episode_lengths.get(ep_idx, 1) - 1, 1),
            )
            if row is None:
                skipped += 1
                continue
            rows.append(row)
        logging.info(f"[hand_terms] {split}/{name}: {len(samples) - skipped} frames measured")
        if skipped:
            logging.warning(f"[hand_terms] {split}/{name}: {skipped} frames carried no hand terms")
    return rows


def measure_diverse(adapter, cfg, timesteps: np.ndarray):
    """Anchors of the diverse corpus cache, each with its own layout id, as training rows.

    Returns ``(rows, buffer)``; the buffer re-reads the frames the exemplar pages render.
    An absent or unbuilt cache is a warning and an empty list, not a failure.
    """
    p = cfg.probe_parameters
    n_total = int(getattr(p, "hand_terms_n_diverse_frames", 0) or 0)
    diverse_cfg = getattr(cfg, "diverse", None)
    if n_total <= 0 or diverse_cfg is None or not getattr(diverse_cfg, "enabled", False):
        return [], None
    from lerobot.probes.domain_representations import _diverse_inputs, _diverse_samples, _open_diverse

    try:
        buffer = _open_diverse(cfg)
    except Exception as exc:  # noqa: BLE001 - the cache is optional for this probe
        logging.warning(f"[hand_terms] diverse corpus skipped: {exc}")
        return [], None
    max_episodes = getattr(p, "objective_max_episodes", None) or p.max_episodes
    samples = _diverse_samples(buffer, n_total, max_episodes, int(p.random_seed))
    grid = torch.from_numpy(timesteps).float()
    rows: list[dict] = []
    for sample in samples:
        inputs = _diverse_inputs(buffer, cfg, sample)
        losses = adapter.training_losses(
            inputs,
            flow_timesteps=grid,
            flow_noise_seed=int(p.random_seed) + int(sample["index"]),
            dropout=False,
            extra_complementary=inputs["extra"],
        )
        row = _row(
            losses,
            kind="diverse",
            split="train",
            source=str(sample["source"]),
            episode=str(sample["episode"]),
            frame_idx=float(sample["frame"]),
            index=int(sample["index"]),
            subtask=inputs.get("subtask"),
        )
        if row is not None:
            rows.append(row)
    logging.info(f"[hand_terms] train/diverse: {len(rows)} frames measured over {len(samples)} anchors")
    return rows, buffer


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────


def with_gap(rows: list[dict]) -> list[dict]:
    """Add the signed gap ``hand - joint`` to every row (same chance level, so a plain difference)."""
    for row in rows:
        row[GAP_KEY] = float(row["loss_hand_pose"]) - float(row["loss_hand_joint"])
    return rows


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows if r.get(key) is not None], dtype=np.float64)


def by_robot(rows: list[dict], key: str) -> dict[str, np.ndarray]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            buckets[row["robot"]].append(float(row[key]))
    return {robot: np.array(values) for robot, values in buckets.items()}


def gap_bands(rows: list[dict], n_per_band: int) -> list[dict]:
    """The frames nearest $p_5$, $p_{50}$ and $p_{95}$ of the absolute gap ``|hand - joint|``.

    Within a band the frames are ordered by their signed gap, so the page reads from
    "hand fits worse" to "joints fit worse".
    """
    scored = [r for r in rows if r.get(GAP_KEY) is not None]
    if not scored:
        return []
    magnitude = np.abs(np.array([r[GAP_KEY] for r in scored]))
    bands = []
    for percentile, label in BANDS:
        target = float(np.percentile(magnitude, percentile))
        nearest = np.argsort(np.abs(magnitude - target))[: max(int(n_per_band), 0)]
        bands.append(
            {
                "percentile": percentile,
                "label": label,
                "target": target,
                "frames": [scored[i] for i in sorted(nearest, key=lambda i: scored[i][GAP_KEY])],
            }
        )
    return bands


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "mean": float(values.mean()),
        "sem": sem(values),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "n": int(values.size),
    }


def build_summary(val_rows: list[dict], train_rows: list[dict], cfg) -> dict:
    keys = [key for key, _ in TERMS] + [GAP_KEY]
    robots = sorted({r["robot"] for r in val_rows + train_rows})
    summary = {
        "n_val_frames": len(val_rows),
        "n_train_frames": len(train_rows),
        "n_train_diverse_frames": sum(1 for r in train_rows if r.get("kind") == "diverse"),
        "robots": robots,
        "terms": {},
        "by_robot": {},
        "flow_timesteps_pinned": True,
        "joint_weight": float(getattr(getattr(cfg.policy, "hand", None), "joint_weight", 1.0)),
        "fk_weight": float(getattr(getattr(cfg.policy, "hand", None), "fk_weight", 1.0)),
    }
    for key in keys:
        entry = {"val": _stats(column(val_rows, key)), "train": _stats(column(train_rows, key))}
        if entry["val"].get("n") and entry["train"].get("n"):
            entry["z"] = float(
                (entry["val"]["mean"] - entry["train"]["mean"])
                / max(np.hypot(entry["val"]["sem"], entry["train"]["sem"]), 1e-12)
            )
        summary["terms"][key] = entry
    for robot in robots:
        summary["by_robot"][robot] = {
            key: {
                "val": _stats(by_robot(val_rows, key).get(robot, np.array([]))),
                "train": _stats(by_robot(train_rows, key).get(robot, np.array([]))),
            }
            for key in keys
        }
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────


def render_histograms(val_rows: list[dict], train_rows: list[dict], output_dir: str) -> str | None:
    """One row per term (and the gap), one column per robot; train and val overlaid."""
    rows_spec = list(TERMS) + [(GAP_KEY, GAP_LABEL)]
    robots = sorted({r["robot"] for r in val_rows + train_rows})
    if not robots:
        return None
    # ReBot (the val robot) first, then the rest alphabetically.
    val_robots = {r["robot"] for r in val_rows}
    robots = sorted(robots, key=lambda name: (name not in val_robots, name))
    fig, axes = plt.subplots(
        len(rows_spec), len(robots), figsize=(4.2 * len(robots), 3.2 * len(rows_spec)), squeeze=False
    )
    for i, (key, label) in enumerate(rows_spec):
        everything = column(val_rows + train_rows, key)
        bins = np.histogram_bin_edges(everything, bins=30) if everything.size else 30
        val_by, train_by = by_robot(val_rows, key), by_robot(train_rows, key)
        for j, robot in enumerate(robots):
            ax = axes[i][j]
            train, val = train_by.get(robot, np.array([])), val_by.get(robot, np.array([]))
            if train.size == 0 and val.size == 0:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center")
                ax.axis("off")
                continue
            if train.size:
                ax.hist(
                    train,
                    bins=bins,
                    color=TRAIN_COLOR,
                    alpha=0.55,
                    density=True,
                    label=f"train n={train.size}",
                )
            if val.size:
                ax.hist(val, bins=bins, color=VAL_COLOR, alpha=0.55, density=True, label=f"val n={val.size}")
            if key == GAP_KEY:
                ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlabel(label)
            ax.set_ylabel("density")
            ax.legend(fontsize=7)
            _style(ax, f"{robot}: {label}")
    fig.suptitle("EE mixture loss terms per frame, by robot (train vs held-out val)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(output_dir, "hist_by_robot.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _image_array(image) -> np.ndarray:
    """One camera tensor (CHW, BCHW, or HWC; float in [0, 1] or uint8) as HWC uint8."""
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.float().numpy()
    if array.max(initial=0.0) <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _frame_images(row: dict, loaders: dict, cfg) -> list[tuple[str, np.ndarray]]:
    """Re-read a measured frame's cameras; nothing is held in memory between measure and render."""
    if row["kind"] == "rebot":
        dataset = loaders["rebot"].get(row["source"])
        if dataset is None:
            return []
        frame = probe_frame_inputs(dataset, cfg, int(row["global_idx"]), int(cfg.policy.chunk_size))
        obs = frame["obs"]
    else:
        buffer = loaders.get("diverse")
        if buffer is None:
            return []
        from lerobot.probes.domain_representations import _diverse_inputs

        obs = _diverse_inputs(buffer, cfg, {"index": int(row["index"]), "source": row["source"]})["obs"]
    cameras = sorted(k for k in obs if str(k).startswith("observation.images."))[:MAX_CAMERAS]
    return [(key.rsplit(".", 1)[-1], _image_array(obs[key])) for key in cameras]


def _frame_title(row: dict) -> str:
    fk = row.get("loss_hand_fk")
    where = (
        f"ep {row['episode_idx']} fr {row['frame_idx']}"
        if row["kind"] == "rebot"
        else f"{row['episode']} @ {row['frame_idx']:.1f}s"
    )
    return (
        f"{row['robot']} · {where}\n"
        f"joint {row['loss_hand_joint']:.3f} · hand {row['loss_hand_pose']:.3f} · "
        f"FK {'n/a' if fk is None else f'{fk:.3f}'} · gap {row[GAP_KEY]:+.3f}"
    )


def render_exemplars(bands: list[dict], loaders: dict, cfg, output_dir: str) -> list[str]:
    """One page per band: the frames' cameras side by side, the three terms and the gap above."""
    written = []
    for band in bands:
        frames = band["frames"]
        if not frames:
            continue
        images = [_frame_images(row, loaders, cfg) for row in frames]
        n_cols = max((len(cams) for cams in images), default=0)
        if n_cols == 0:
            continue
        fig, axes = plt.subplots(
            len(frames), n_cols, figsize=(4.0 * n_cols, 3.4 * len(frames)), squeeze=False
        )
        for i, (row, cams) in enumerate(zip(frames, images, strict=True)):
            for j in range(n_cols):
                ax = axes[i][j]
                ax.axis("off")
                if j < len(cams):
                    label, array = cams[j]
                    ax.imshow(array)
                    ax.set_title(label if j else f"{_frame_title(row)}\n{label}", fontsize=8, loc="left")
        fig.suptitle(
            f"p{band['percentile']} {band['label']}: |hand - joint| near {band['target']:.3f} "
            "(gap > 0: the hand block fits worse; gap < 0: the joints fit worse)",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = os.path.join(output_dir, f"exemplars_p{band['percentile']:02d}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(path)
    return written


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────


def run(adapter, dataset, cfg, output_dir: str, train_dataset=None) -> dict | None:
    """Measure the three terms on val, on the ReBot training sources and on the diverse
    corpus; write the histograms, the gap-percentile frames and ``hand_terms.json``.

    Returns the summary so rl_offline can push the headline scalars to Aim.
    """
    if not bool(getattr(cfg.policy, "hand_block", False)):
        logging.warning("[hand_terms] policy.hand_block is off: nothing to measure.")
        return None
    makedirs(output_dir)
    p = cfg.probe_parameters
    n_timesteps = max(1, int(getattr(cfg.policy, "num_flow_timesteps", 1)))
    timesteps = flow_timestep_grid(cfg.policy, n_timesteps)

    loaders: dict = {"rebot": {"val": dataset}, "diverse": None}
    adapter._set_probe_cuda_graph_enabled(False)
    try:
        val_rows = measure_rebot(adapter, [("val", dataset)], cfg, "val", timesteps)
        train_rows: list[dict] = []
        if train_dataset is not None:
            sources = training_datasets(cfg, train_dataset)
            loaders["rebot"].update(dict(sources))
            train_rows += measure_rebot(adapter, sources, cfg, "train", timesteps)
        diverse_rows, buffer = measure_diverse(adapter, cfg, timesteps)
        loaders["diverse"] = buffer
        train_rows += diverse_rows
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not val_rows and not train_rows:
        logging.warning("[hand_terms] no frame produced hand terms.")
        return None
    with_gap(val_rows)
    with_gap(train_rows)

    summary = build_summary(val_rows, train_rows, cfg)
    bands = gap_bands(val_rows or train_rows, int(getattr(p, "hand_terms_exemplars_per_band", 4)))
    summary["exemplars_split"] = "val" if val_rows else "train"
    summary["exemplars"] = [
        {
            "percentile": band["percentile"],
            "label": band["label"],
            "target": band["target"],
            "frames": [
                {
                    key: row.get(key)
                    for key in (
                        "robot",
                        "source",
                        "episode",
                        "episode_idx",
                        "frame_idx",
                        "global_idx",
                        "index",
                        "subtask",
                        "loss_hand_joint",
                        "loss_hand_pose",
                        "loss_hand_fk",
                        GAP_KEY,
                    )
                }
                for row in band["frames"]
            ],
        }
        for band in bands
    ]
    with open(os.path.join(output_dir, "hand_terms.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "frames.json"), "w") as f:
        json.dump(val_rows + train_rows, f, indent=1)

    render_histograms(val_rows, train_rows, output_dir)
    render_exemplars(bands, loaders, cfg, output_dir)
    _write_index(summary, output_dir)

    parts = []
    for key, label in list(TERMS) + [(GAP_KEY, GAP_LABEL)]:
        entry = summary["terms"][key]
        if entry["val"].get("n"):
            parts.append(
                f"{label} val={entry['val']['mean']:.4f}"
                + (f" train={entry['train']['mean']:.4f}" if entry["train"].get("n") else "")
            )
    logging.info("[hand_terms] " + "  |  ".join(parts))
    return summary


def aim_scalars(summary: dict) -> dict:
    """The three held-out terms and the median gap, on the training run's own axes."""
    scalars: dict[str, float] = {}
    for key, _ in TERMS:
        entry = (summary.get("terms") or {}).get(key) or {}
        if entry.get("val", {}).get("n"):
            scalars[f"hand_terms_val_{key.removeprefix('loss_hand_')}"] = float(entry["val"]["mean"])
    gap = ((summary.get("terms") or {}).get(GAP_KEY) or {}).get("val") or {}
    if gap.get("n"):
        scalars["hand_terms_val_gap_median"] = float(gap["median"])
    return scalars


def _write_index(summary: dict, output_dir: str) -> None:
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Hand terms",
        group="Objective",
        claim="How do the joint block, the hand block and the FK term fit, per frame and per robot?",
        summary=summary,
        see_also=["objective"],
        metrics=[
            Metric(
                "terms.loss_hand_joint.val.mean",
                "Joint block loss (val)",
                good="low",
                fmt=4,
                primary=True,
                trend=True,
                note="Block mean of the flow loss over the joint slots on held-out frames.",
            ),
            Metric(
                "terms.loss_hand_pose.val.mean",
                "Hand block loss (val)",
                good="low",
                fmt=4,
                primary=True,
                trend=True,
                note="Mean of the position, rotation and aperture block means on held-out frames.",
            ),
            Metric(
                "terms.loss_hand_fk.val.mean",
                "FK term (val)",
                good="low",
                fmt=4,
                primary=True,
                trend=True,
                note="Fingertip of the implied clean joint sample against the demonstrated "
                "fingertip, in the hand block's normalized units.",
            ),
            Metric(
                "terms.gap.val.median",
                "Median gap hand - joint (val)",
                good="none",
                fmt=4,
                baseline=0.0,
                note="Positive: the hand block fits worse than the joints on the typical frame.",
            ),
            Metric("n_val_frames", "Val frames", good="none", fmt=0),
            Metric("n_train_diverse_frames", "Diverse train frames", good="none", fmt=0),
        ],
        panels=[
            Panel(
                "hist_by_robot.png",
                "Loss terms by robot",
                "One column per robot, one row per term and the gap. Train in one colour, "
                "held-out val in the other where the robot has both. The dotted line on the gap "
                "row is zero: right of it the hand block fits worse, left of it the joints do.",
                primary=True,
            ),
            Panel(
                "exemplars_p05.png",
                "Frames where the hand block and the joints fit alike",
                "Held-out frames nearest $p_5$ of $|\\mathcal{L}_{hand} - \\mathcal{L}_{joint}|$, "
                "cameras side by side, the three terms and the signed gap above each.",
            ),
            Panel(
                "exemplars_p50.png",
                "Typical frames",
                "Held-out frames nearest the median absolute gap.",
            ),
            Panel(
                "exemplars_p95.png",
                "Frames where the two heads disagree most",
                "Held-out frames nearest $p_{95}$ of the absolute gap, ordered by the signed gap: "
                "from the joints fitting worse to the hand block fitting worse.",
                primary=True,
            ),
        ],
    )
