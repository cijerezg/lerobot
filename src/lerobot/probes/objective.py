#!/usr/bin/env python
r"""The training objective on held-out data — is it fitting or memorising?

The only probe that scores the quantity the optimiser minimises, and the only one that
reads training data at all, so the only one that can report a generalisation gap: the
same flow-matching loss and FAST cross-entropy the trainer backwards, measured on
held-out episodes and on a matched sample of training episodes under one regime.

Absolute levels are not readable. $\mathcal{L}_{flow}$ regresses a velocity
$a - \varepsilon$ from $x_t = (1-t)\varepsilon + t a$, and at small $t$ the conditioning
barely determines the target, so a perfect model still scores far from zero. Every
headline is therefore a val/train pair, and the gap carries

    $$z = \frac{\bar{\mathcal{L}}_{val} - \bar{\mathcal{L}}_{train}}
                {\sqrt{\mathrm{SEM}_{val}^2 + \mathrm{SEM}_{train}^2}}$$

Two choices make that pair comparable. Flow timesteps sit on the stratified quantile
grid $t_k = t_0 + (t_{\max}-t_0)F^{-1}\!\big((k+\tfrac12)/K\big)$ of the training
distribution instead of being drawn — equiprobable strata, so the mean estimates the
same $\mathbb{E}_t[\mathcal{L}(t)]$ with the sampler's variance removed and the grid
identical at every checkpoint. And the pack step's dropout is suppressed on both sides,
which is why these numbers are **not** the Aim ``train/loss_flow`` curve: that one is
measured with dropout armed and reads higher. Compare val to the train column here.

Registered probe: enable with ``probe_parameters.enable_objective``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, NullFormatter

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    action_inspector_sample_seed,
    build_episode_index,
    joint_names_for_dim,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
    sample_episodes_evenly,
)
from lerobot.utils.depth_gripper_events import (
    DEPTH_GRIPPER_CLOSE_TARGET,
    DEPTH_GRIPPER_OPEN_TARGET,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

VAL_COLOR = "#E63946"
TRAIN_COLOR = "#457B9D"

AUXILIARY_LOSS_KEYS = (
    "loss_action_aux",
    "loss_discrete_aux",
    "loss_depth_event",
    "depth_event_close_bce",
    "depth_event_open_bce",
    "action_aux_tempered_mse",
    "action_aux_band_dc_mse",
    "action_aux_band_k1_mse",
    "action_aux_band_k2_mse",
    "action_aux_band_k3_mse",
    "action_aux_band_detail_mse",
    "action_aux_band_high_mse",
    "discrete_aux_ordinal_ce",
    "discrete_aux_band_dc_ordinal_ce",
    "discrete_aux_band_k1_ordinal_ce",
    "discrete_aux_band_k2_ordinal_ce",
    "discrete_aux_band_k3_ordinal_ce",
    "discrete_aux_band_detail_ordinal_ce",
    "discrete_aux_band_high_ordinal_ce",
)

# Soft-target BCE cannot be read on its own: it has an irreducible floor E[H(t)] and
# a no-information ceiling H(E[t]) only ~0.155 nats apart on these labels, and neither
# is drawn on a raw loss curve. Carry the targets through so both ends are reported
# next to the measured value. See DEPTH_EVENT_OPEN_HEAD.md.
DEPTH_EVENT_TARGET_BY_HEAD = {
    "close": DEPTH_GRIPPER_CLOSE_TARGET,
    "open": DEPTH_GRIPPER_OPEN_TARGET,
}


# ──────────────────────────────────────────────────────────────────────────────
# Measurement setup
# ──────────────────────────────────────────────────────────────────────────────


def flow_timestep_grid(policy_cfg, n_timesteps: int) -> np.ndarray:
    """Stratified quantile grid of the training timestep distribution.

    Equiprobable strata, so the unweighted mean over the grid estimates the same
    expectation the randomly-drawn training loss does — with the draw's variance gone.
    """
    from scipy.stats import beta as beta_dist

    offset = float(policy_cfg.flow_matching_time_offset)
    upper = min(
        float(policy_cfg.flow_matching_cutoff),
        offset + float(policy_cfg.flow_matching_time_scale),
    )
    quantiles = (np.arange(n_timesteps) + 0.5) / n_timesteps
    unit = beta_dist.ppf(
        quantiles,
        float(policy_cfg.flow_matching_beta_alpha),
        float(policy_cfg.flow_matching_beta_beta),
    )
    return offset + (upper - offset) * unit


def training_datasets(cfg, fallback):
    """Every training source, so the train column spans what the policy actually saw.

    rl_offline hands the probe only the normalization source; a gap measured against
    one collection out of three would call the other two "held out".
    """
    from lerobot.probes.actions import reference_datasets

    return reference_datasets(cfg, fallback)


def sem(values: np.ndarray) -> float:
    """Standard error of the mean. One sample has no spread to report."""
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def gap_z(val: np.ndarray, train: np.ndarray) -> float:
    """How many standard errors separate the two means."""
    spread = np.hypot(sem(val), sem(train))
    return float((val.mean() - train.mean()) / spread) if spread > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────────────────────


def measure_split(adapter, datasets, cfg, split: str, timesteps: np.ndarray) -> list[dict]:
    """One ``training_losses`` forward per sampled frame, over every dataset in the split."""
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
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            losses = adapter.training_losses(
                frame,
                flow_timesteps=grid,
                # Same frame -> same noise at every checkpoint, so a difference between
                # two checkpoints is the model changing and nothing else.
                flow_noise_seed=int(p.random_seed) + int(global_idx),
                dropout=False,
            )
            rows.append(
                {
                    **losses,
                    # Kept per frame, not pooled from the sidecar, so the floor and
                    # ceiling describe exactly the frames the BCE was measured on.
                    **{
                        f"depth_event_{head}_target": (
                            float(frame[key]) if frame.get(key) is not None else None
                        )
                        for head, key in DEPTH_EVENT_TARGET_BY_HEAD.items()
                    },
                    "split": split,
                    "source": name,
                    "source_root": str(getattr(dataset, "root", "")),
                    "episode_idx": int(ep_idx),
                    "frame_idx": int(fr_idx),
                    "global_idx": int(global_idx),
                    "subtask": frame["subtask"],
                    "progress": fr_idx / max(episode_lengths.get(ep_idx, 1) - 1, 1),
                }
            )
        logging.info(f"[objective] {split}/{name}: {len(samples)} frames measured")
    return rows


def column(rows: list[dict], key: str) -> np.ndarray:
    """Scalar metric over frames, dropping the frames where the policy has no such head."""
    return np.array([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)


def stacked(rows: list[dict], key: str) -> np.ndarray | None:
    """Per-frame vectors of equal length, stacked into ``[n_frames, ...]``."""
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values or len({np.asarray(v).shape for v in values}) != 1:
        return None
    return np.stack([np.asarray(v) for v in values])


def by_position(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a ragged per-token array against token index, plus the count behind each."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values = row.get(key)
        if values is None:
            continue
        for position, value in enumerate(np.asarray(values)):
            buckets[position].append(float(value))
    if not buckets:
        return np.zeros(0), np.zeros(0)
    positions = sorted(buckets)
    return (
        np.array([np.mean(buckets[i]) for i in positions]),
        np.array([len(buckets[i]) for i in positions]),
    )


def split_provenance(rows: list[dict]) -> dict:
    """Which episodes of which datasets this split's numbers came from.

    Derived from the rows rather than from config, so it describes what was actually
    measured after episode-budget division and stride snapping dropped whatever they
    dropped.
    """
    sources: dict[str, dict] = {}
    for row in rows:
        entry = sources.setdefault(
            row["source"], {"root": row.get("source_root", ""), "episodes": set(), "n_frames": 0}
        )
        entry["episodes"].add(int(row["episode_idx"]))
        entry["n_frames"] += 1
    return {
        "n_frames": len(rows),
        "n_episodes": sum(len(e["episodes"]) for e in sources.values()),
        "sources": [
            {
                "name": name,
                "root": entry["root"],
                "episodes": sorted(entry["episodes"]),
                "n_episodes": len(entry["episodes"]),
                "n_frames": entry["n_frames"],
            }
            for name, entry in sorted(sources.items())
        ],
    }


def provenance(val_rows: list[dict], train_rows: list[dict], cfg, timesteps: np.ndarray) -> dict:
    """Everything a reader needs to know about where these numbers came from.

    Worth stating explicitly because the sampling is not what a training step does: the
    probe runs **one frame per forward**, not a batch, and covers whole episodes evenly
    rather than drawing uniformly from the buffer.
    """
    p = cfg.probe_parameters
    return {
        "val": split_provenance(val_rows),
        "train": split_provenance(train_rows),
        "frames_per_episode": int(
            getattr(p, "objective_n_frames_per_episode", None) or p.n_frames_per_episode
        ),
        "episode_budget": getattr(p, "objective_max_episodes", None) or p.max_episodes,
        "image_stride": probe_image_stride(cfg),
        "chunk_size": int(cfg.policy.chunk_size),
        "batch_size": 1,
        "forwards": len(val_rows) + len(train_rows),
        "timesteps_per_forward": int(timesteps.size),
        "sampling": (
            "Frames evenly spaced across each episode, snapped onto the image/depth "
            "stride grid; episodes drawn by a seeded subset when the budget is smaller "
            "than the split. The train episode budget is divided across the sources, so "
            "adding a source shrinks the episodes taken from each."
        ),
        "regime": "one frame per forward, batch size 1; dropout suppressed; per-frame seeded noise",
    }


def group_mean(rows: list[dict], group_key: str, value_key: str) -> dict:
    """Mean of one metric per group, with the group's frame count."""
    buckets: dict = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            buckets[row[group_key]].append(float(row[value_key]))
    return {
        str(name): {"mean": float(np.mean(values)), "n": len(values)}
        for name, values in sorted(buckets.items(), key=lambda item: str(item[0]))
    }


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────


def compare(val_rows: list[dict], train_rows: list[dict], key: str) -> dict:
    """Val and train means for one loss, with the standard errors and the gap."""
    val, train = column(val_rows, key), column(train_rows, key)
    if val.size == 0:
        return {}
    entry = {
        "val": float(val.mean()),
        "val_sem": sem(val),
        "val_n": int(val.size),
    }
    if train.size == 0:
        return entry
    return {
        **entry,
        "train": float(train.mean()),
        "train_sem": sem(train),
        "train_n": int(train.size),
        "gap": float(val.mean() - train.mean()),
        "ratio": float(val.mean() / train.mean()) if train.mean() != 0 else None,
        "z": gap_z(val, train),
    }


def binary_entropy(p: np.ndarray | float) -> np.ndarray:
    """Entropy in nats of a Bernoulli with soft parameter ``p``."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def _paired_depth_event(rows: list[dict], head: str) -> tuple[np.ndarray, np.ndarray]:
    """Targets and measured BCE over the frames that carry both."""
    targets, losses = [], []
    for row in rows:
        target = row.get(f"depth_event_{head}_target")
        loss = row.get(f"depth_event_{head}_bce")
        if target is None or loss is None:
            continue
        targets.append(float(target))
        losses.append(float(loss))
    return np.asarray(targets, dtype=np.float64), np.asarray(losses, dtype=np.float64)


def depth_event_reference(rows: list[dict], head: str) -> dict:
    """One head's BCE placed between the two predictors that bracket it.

    ``floor`` is E[H(t)], what a predictor that outputs the target exactly still pays
    because the targets are soft. ``constant`` is H(E[t]), what a predictor that
    ignores its input entirely gets for free from the base rate. ``captured`` maps the
    measured BCE onto that interval: 1.0 is a perfect predictor, 0.0 is one that has
    learned nothing about the depth observation, and negative is worse than the base
    rate. The interval is narrow (~0.155 nats here), which is why the raw curve looks
    static whatever the head is doing.
    """
    targets, losses = _paired_depth_event(rows, head)
    if targets.size == 0:
        return {}
    base_rate = float(targets.mean())
    floor = float(binary_entropy(targets).mean())
    constant = float(binary_entropy(base_rate))
    bce = float(losses.mean())
    learnable = constant - floor
    entry = {
        "n": int(targets.size),
        "base_rate": base_rate,
        "floor": floor,
        "constant": constant,
        "learnable_range": learnable,
        "bce": bce,
        "bce_sem": sem(losses),
    }
    if learnable > 0:
        entry["captured"] = (constant - bce) / learnable
        entry["captured_sem"] = sem(losses) / learnable
    return entry


def build_summary(val_rows, train_rows, timesteps, action_dim, n_action_tokens, cfg) -> dict:
    summary: dict = {
        "n_val_frames": len(val_rows),
        "n_train_frames": len(train_rows),
        "flow_timesteps": timesteps.tolist(),
        "data": provenance(val_rows, train_rows, cfg, timesteps),
        "regime": "prompt/modality dropout suppressed on both splits; flow timesteps on "
        "the stratified quantile grid; flow noise seeded per frame.",
        "loss_flow": compare(val_rows, train_rows, "loss_flow"),
        "loss_discrete_ce": compare(val_rows, train_rows, "loss_discrete_ce"),
        "loss_discrete_z": compare(val_rows, train_rows, "loss_discrete_z"),
        "loss_total": compare(val_rows, train_rows, "loss_total"),
    }
    summary.update({key: compare(val_rows, train_rows, key) for key in AUXILIARY_LOSS_KEYS})

    splits = (("val", val_rows), ("train", train_rows))

    depth_event_reference_summary = {
        head: {split: entry for split, rows in splits if (entry := depth_event_reference(rows, head))}
        for head in DEPTH_EVENT_TARGET_BY_HEAD
    }
    if any(depth_event_reference_summary.values()):
        summary["depth_event_reference"] = depth_event_reference_summary

    for split, rows in splits:
        top1 = [
            np.asarray(r["discrete_token_top1"]) for r in rows if r.get("discrete_token_top1") is not None
        ]
        if not top1:
            continue
        top5 = [
            np.asarray(r["discrete_token_top5"]) for r in rows if r.get("discrete_token_top5") is not None
        ]
        summary.setdefault("discrete", {})[split] = {
            "top1": float(np.concatenate(top1).mean()),
            "top5": float(np.concatenate(top5).mean()),
            "perplexity": float(np.exp(column(rows, "loss_discrete_ce").mean())),
            "tokens_per_chunk": float(column(rows, "n_action_tokens").mean()),
        }

    if n_action_tokens:
        summary["discrete_chance_top1"] = 1.0 / float(n_action_tokens)

    def curves(key: str, names: list[str] | None = None) -> dict:
        """Frame-averaged per-split curve, as a list or as a name -> value mapping."""
        out = {}
        for split, rows in splits:
            values = stacked(rows, key)
            if values is None:
                continue
            mean = values.mean(axis=0).tolist()
            out[split] = mean if names is None else dict(zip(names, mean, strict=False))
        return out

    summary["loss_flow_by_timestep"] = curves("loss_flow_by_timestep")
    summary["loss_flow_by_action_step"] = curves("loss_flow_by_action_step")
    summary["loss_flow_by_joint"] = curves("loss_flow_by_dim", joint_names_for_dim(action_dim))
    # FAST spans are variable length, so the far positions are averaged over only the
    # frames whose span reached that far. Keep the counts: without them the thin tail
    # reads as a real drop in cross-entropy.
    summary["discrete_ce_by_position"] = {}
    summary["discrete_ce_by_position_n"] = {}
    for split, rows in splits:
        means, counts = by_position(rows, "discrete_token_ce")
        if means.size:
            summary["discrete_ce_by_position"][split] = means.tolist()
            summary["discrete_ce_by_position_n"][split] = counts.tolist()
    summary["val_by_episode"] = group_mean(val_rows, "episode_idx", "loss_flow")
    summary["val_by_subtask"] = group_mean(val_rows, "subtask", "loss_flow")
    summary["per_frame"] = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, np.ndarray) and not key.startswith("loss_flow_by")
        }
        for row in val_rows + train_rows
    ]
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────


def _style(ax, title: str) -> None:
    """Wrapped title and quiet ticks. Not ``utils.ax_style`` — that one is for UMAP."""
    ax.set_title(textwrap.fill(title, width=48), fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _pair_points(ax, entry: dict, title: str, ylabel: str) -> None:
    """Val and train means as points with SEM whiskers, gap and z in the title.

    Points rather than bars, and the axis is not forced to zero: this loss has a
    large irreducible floor, so a zero-based bar chart puts the two means within a
    pixel of each other and hides the only thing the panel is for. The whiskers carry
    the scale that makes the separation readable.
    """
    if "train" not in entry:
        ax.text(0.5, 0.5, "no training split measured", ha="center", va="center")
        ax.axis("off")
        return
    means = [entry["train"], entry["val"]]
    errors = [entry["train_sem"], entry["val_sem"]]
    ax.errorbar([0, 1], means, yerr=errors, fmt="none", ecolor="#333", capsize=6, linewidth=1.4)
    ax.scatter([0, 1], means, s=140, c=[TRAIN_COLOR, VAL_COLOR], zorder=3, edgecolor="white")
    for x, value in enumerate(means):
        ax.annotate(
            f"{value:.4f}", (x, value), textcoords="offset points", xytext=(14, 0), va="center", fontsize=9
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["train", "val"])
    ax.set_xlim(-0.6, 1.7)
    ax.set_ylabel(ylabel)
    _style(ax, f"{title} — gap {entry['gap']:+.4f} ({entry['ratio']:.3f}x, z = {entry['z']:+.1f})")


def _overlaid_hist(ax, val_rows, train_rows, key: str, title: str, xlabel: str) -> None:
    val, train = column(val_rows, key), column(train_rows, key)
    if val.size == 0:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
        ax.axis("off")
        return
    bins = np.histogram_bin_edges(np.concatenate([val, train]) if train.size else val, bins=30)
    if train.size:
        ax.hist(train, bins=bins, color=TRAIN_COLOR, alpha=0.55, label="train", density=True)
    ax.hist(val, bins=bins, color=VAL_COLOR, alpha=0.55, label="val", density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    _style(ax, title)


def _trimmed_gap(val: np.ndarray, train: np.ndarray, fraction: float) -> tuple[float, float]:
    """Mean ratio and absolute gap after dropping the worst ``fraction`` of BOTH splits.

    The tail test. If the gap is a handful of blown frames, dropping them collapses it;
    if the whole distribution moved, the ratio barely budges.
    """
    keep_v = np.sort(val)[: max(int(val.size * (1.0 - fraction)), 1)]
    keep_t = np.sort(train)[: max(int(train.size * (1.0 - fraction)), 1)]
    return float(keep_v.mean() / keep_t.mean()), float(keep_v.mean() - keep_t.mean())


def _ecdf_panel(ax, val_rows, train_rows, key: str, title: str, xlabel: str) -> None:
    r"""Empirical CDFs of the per-frame loss, val against train.

    A mean gap has two very different causes that the mean cannot tell apart: every frame
    got a little worse, or a few frames blew up. The histogram beside it shows shape but
    the eye cannot integrate it; stepping the sorted values against $i/n$ can be read
    directly. Curves that coincide low and separate only in the last few percent are a
    tail; curves offset across their whole range are a distribution shift, and the
    distinction decides whether to go looking at individual frames or at the training
    distribution as a whole.

    The trimmed-mean box states the answer numerically, because two step curves a few
    percent apart look identical at figure scale.
    """
    val, train = column(val_rows, key), column(train_rows, key)
    if val.size == 0:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
        ax.axis("off")
        return

    for sample, color, label in ((train, TRAIN_COLOR, "train"), (val, VAL_COLOR, "val")):
        if sample.size == 0:
            continue
        ordered = np.sort(sample)
        # Step from 1/n to 1: the largest observation is the 100th percentile, not 1-1/n.
        ax.step(
            ordered,
            np.arange(1, ordered.size + 1) / ordered.size,
            where="post",
            color=color,
            linewidth=1.8,
            label=label,
            zorder=3,
        )

    # Log x, because the question is about a RATIO. A loss that is uniformly k times worse
    # is a curve displaced horizontally by a constant log k at every height, so a shift
    # reads as two parallel curves and a tail reads as curves that meet low and split high.
    # A linear axis instead lets the two largest frames stretch the range and squeezes the
    # decade everything actually lives in into the left fifth of the panel.
    ax.set_xscale("log")
    for quantile in (0.5, 0.9):
        ax.axhline(quantile, color="#bbbbbb", linewidth=0.6, linestyle=":", zorder=0)
        ax.annotate(
            f"p{int(quantile * 100)}",
            (0.005, quantile),
            xycoords=("axes fraction", "data"),
            fontsize=6.5,
            color="#777777",
            va="bottom",
            ha="left",
            zorder=1,
        )

    if train.size:
        full = val.mean() / train.mean()
        trimmed, _ = _trimmed_gap(val, train, 0.05)
        # Ratios, so the reader can see whether trimming moved it, not just where it landed.
        ax.text(
            0.97,
            0.06,
            f"mean ratio  {full:.3f}\ntop 5% trimmed  {trimmed:.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of frames $\\leq$ x")
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.30))
    _style(ax, title)


def render_objective(summary, val_rows, train_rows, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
    _pair_points(axes[0], summary["loss_flow"], "Flow-matching loss", "$\\mathcal{L}_{flow}$")
    if summary["loss_discrete_ce"]:
        _pair_points(axes[1], summary["loss_discrete_ce"], "FAST token cross-entropy", "CE (nats)")
    else:
        axes[1].text(0.5, 0.5, "no discrete head", ha="center", va="center")
        axes[1].axis("off")
    _overlaid_hist(
        axes[2], val_rows, train_rows, "loss_flow", "Per-frame flow loss", "$\\mathcal{L}_{flow}$ per frame"
    )
    _ecdf_panel(
        axes[3],
        val_rows,
        train_rows,
        "loss_flow",
        "Cumulative distribution — tail or shift?",
        "$\\mathcal{L}_{flow}$ per frame",
    )
    fig.suptitle(
        f"Training objective on held-out data — {summary['n_val_frames']} val / "
        f"{summary['n_train_frames']} train frames, dropout suppressed on both",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(output_dir, "objective.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


def render_flow(summary, output_dir: str) -> None:
    if not summary["loss_flow_by_timestep"]:
        return  # action_mode "discrete": there is no flow loss to break down.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    timesteps = np.array(summary["flow_timesteps"])

    for split, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
        curve = summary["loss_flow_by_timestep"].get(split)
        if curve:
            axes[0].plot(timesteps, curve, "o-", color=color, label=split, linewidth=1.6)
        steps = summary["loss_flow_by_action_step"].get(split)
        if steps:
            axes[1].plot(np.arange(len(steps)), steps, color=color, label=split, linewidth=1.6)
        joints = summary["loss_flow_by_joint"].get(split)
        if joints:
            offset = -0.2 if split == "train" else 0.2
            axes[2].barh(
                np.arange(len(joints)) + offset, list(joints.values()), height=0.38, color=color, label=split
            )

    axes[0].set_xlabel("flow timestep $t$  (0 = pure noise, 1 = data)")
    axes[0].set_ylabel("$\\mathcal{L}_{flow}(t)$")
    axes[0].legend(fontsize=8)
    _style(axes[0], "Loss against the timestep grid")

    axes[1].set_xlabel("action step within the chunk")
    axes[1].set_ylabel("$\\mathcal{L}_{flow}$")
    axes[1].legend(fontsize=8)
    _style(axes[1], "Loss against position in the chunk")

    joints = summary["loss_flow_by_joint"].get("val") or {}
    axes[2].set_yticks(np.arange(len(joints)))
    axes[2].set_yticklabels(list(joints), fontsize=8)
    axes[2].set_xlabel("$\\mathcal{L}_{flow}$ (real dims only — read the shape, not the level)")
    axes[2].legend(fontsize=8)
    _style(axes[2], "Loss per joint")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "flow.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


def render_discrete(summary, val_rows, train_rows, output_dir: str) -> None:
    discrete = summary.get("discrete")
    if not discrete:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    labels, width = ["top-1", "top-5"], 0.36
    for offset, split, color in ((-width / 2, "train", TRAIN_COLOR), (width / 2, "val", VAL_COLOR)):
        entry = discrete.get(split)
        if entry is None:
            continue
        axes[0].bar(
            np.arange(2) + offset,
            [entry["top1"], entry["top5"]],
            width=width,
            color=color,
            label=split,
            edgecolor="white",
        )
    chance = summary.get("discrete_chance_top1")
    if chance:
        axes[0].axhline(chance, color="#666", linestyle="--", linewidth=1.2, label=f"chance ({chance:.1e})")
    axes[0].set_xticks(np.arange(2))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("fraction of action tokens correct")
    axes[0].legend(fontsize=8)
    perplexity = (discrete.get("val") or {}).get("perplexity")
    _style(
        axes[0],
        "FAST token accuracy" + (f" — val perplexity {perplexity:.1f}" if perplexity is not None else ""),
    )

    # Only positions at least half the frames reach: beyond that the mean is a handful
    # of unusually long spans, and it plots as a cross-entropy that falls off a cliff.
    covered = 0
    for split, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
        curve = summary["discrete_ce_by_position"].get(split)
        counts = summary.get("discrete_ce_by_position_n", {}).get(split)
        if not curve:
            continue
        support = np.array(counts) / max(counts) if counts else np.ones(len(curve))
        keep = int(np.searchsorted(-support, -0.5, side="right")) if counts else len(curve)
        keep = max(keep, 1)
        covered = max(covered, keep)
        axes[1].plot(np.arange(keep), curve[:keep], color=color, label=split, linewidth=1.6)
    axes[1].set_xlabel("position $j$ in the action-token span (coarse $\\to$ fine)")
    axes[1].set_ylabel("mean CE (nats) at position $j$")
    axes[1].legend(fontsize=8)
    _style(
        axes[1],
        f"Cross-entropy along the span — positions ≥50% of frames reach (0–{covered - 1})",
    )

    _overlaid_hist(
        axes[2],
        val_rows,
        train_rows,
        "loss_discrete_ce",
        "Per-frame FAST cross-entropy",
        "CE per chunk (nats)",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "discrete.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


def exemplar_bands(val_rows: list[dict], n_per_band: int) -> list[dict]:
    """The frames sitting nearest each loss percentile.

    A loss is a number until you see the frame that produced it. Sampling the
    distribution at $p_5$, $p_{50}$ and $p_{95}$ rather than only its tails shows what
    an ordinary frame looks like next to a bad one, which is what says whether the right
    tail is a distinct failure mode or just the top of one continuum.
    """
    scored = [r for r in val_rows if r.get("loss_flow") is not None]
    if not scored:
        return []
    losses = np.array([r["loss_flow"] for r in scored])
    bands = []
    for percentile, label in ((5, "best fit"), (50, "typical"), (95, "worst fit")):
        target = float(np.percentile(losses, percentile))
        nearest = np.argsort(np.abs(losses - target))[:n_per_band]
        bands.append(
            {
                "percentile": percentile,
                "label": label,
                "target": target,
                "frames": [scored[i] for i in sorted(nearest, key=lambda i: losses[i])],
            }
        )
    return bands


def render_action_exemplars(bands, adapter, dataset, cfg, output_dir: str) -> bool:
    """Generate one deterministic action per selected frame and reuse the 3-D inspector.

    Only the p5/p50/p95 exemplars get an inference forward. The much larger loss sample
    remains unchanged, while each visual exemplar now pairs the demonstrated command
    with what the checkpoint would actually generate from that observation.
    """
    if not bands or not any(band["frames"] for band in bands):
        return False

    from lerobot.probes.action_trace_probe import (
        _analyse,
        _camera_context,
        _figure,
        _normalized_chunks,
        _resolve_actuator_config,
        _trajectory_metrics,
        _write_dashboard_html,
    )
    from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
    from lerobot.utils.constants import OBS_STATE

    chunk_size = int(cfg.policy.chunk_size)
    fps = float(dataset.fps)
    kin = RebotKinematics()
    records = []
    actuator_config = None

    adapter._set_probe_cuda_graph_enabled(False)
    try:
        for band in bands:
            for source in band["frames"]:
                frame = probe_frame_inputs(dataset, cfg, source["global_idx"], chunk_size)
                state = frame["obs"][OBS_STATE].reshape(-1).float().cpu()
                if state.numel() == 0:
                    raise ValueError("A 3-D action exemplar requires observation.state.")

                generator = torch.Generator(device=adapter.device)
                generator.manual_seed(
                    action_inspector_sample_seed(cfg.probe_parameters.random_seed, source["global_idx"])
                )
                prediction, prediction_norm, _ = adapter.predict_action_chunk(
                    frame["obs"],
                    frame["task"],
                    state=state,
                    subtask=frame["subtask"],
                    metadata=frame["metadata"],
                    generator=generator,
                )

                if actuator_config is None:
                    actuator_config = _resolve_actuator_config(cfg, int(frame["gt_actions"].shape[-1]))
                motor_speeds, joint_lower, joint_upper = actuator_config

                record = _analyse(
                    kin,
                    state.numpy(),
                    frame["gt_actions"].numpy(),
                    [prediction.numpy()],
                    cfg.probe_parameters.trace_table_z,
                    fps=fps,
                    motor_speeds_deg_s=motor_speeds,
                    joint_lower=joint_lower,
                    joint_upper=joint_upper,
                )
                record.update(
                    episode=int(source["episode_idx"]),
                    frame=int(source["frame_idx"]),
                    global_idx=int(source["global_idx"]),
                    subtask=frame["subtask"],
                    cameras=_camera_context(frame["obs"]),
                    state=state.numpy(),
                    trace_label=(
                        f"p{band['percentile']} {band['label']} · "
                        f"flow loss {source['loss_flow']:.4f} · "
                        f"ep {source['episode_idx']} frame {source['frame_idx']}"
                    ),
                    sample_names=["generated action"],
                    norm=_normalized_chunks(adapter, prediction_norm, frame["gt_actions"], state),
                )
                record["metrics"].update(_trajectory_metrics(record["norm"]))
                records.append(record)
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not records:
        return False
    fig = _figure(records, cfg.probe_parameters, fps=fps)
    html_path = os.path.join(output_dir, "action_exemplars.html")
    _write_dashboard_html(
        fig,
        records,
        html_path,
        page_title="Flow-loss action exemplars",
        subtitle=(
            "Generated action versus the demonstrated command for held-out frames near "
            "p5, p50, and p95 of flow loss. Orbit the 3-D traces or step through frames."
        ),
        legend_note=(
            "Black = demonstrated action · blue = generated action. Solid RGB axes are "
            "the GT terminal orientation; dotted RGB axes are the generated orientation."
        ),
    )
    logging.info(f"[objective] wrote {len(records)} generated-action exemplars → {html_path}")
    return True


def render_breakdown(summary, val_rows, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    train_mean = summary["loss_flow"].get("train")

    for ax, group, xlabel in (
        (axes[0], summary["val_by_episode"], "held-out episode"),
        (axes[1], summary["val_by_subtask"], "subtask"),
    ):
        if not group:
            ax.text(0.5, 0.5, "no groups", ha="center", va="center")
            ax.axis("off")
            continue
        names = list(group)
        values = [group[name]["mean"] for name in names]
        # Points, not bars, and no forced zero — same reason as the headline panel: the
        # spread between groups is small next to the loss's floor, and a zero-based bar
        # chart renders every group identical. Groups stay in name order rather than
        # sorted by value, so the same bar means the same episode at every checkpoint.
        ax.scatter(np.arange(len(names)), values, s=90, color=VAL_COLOR, zorder=3, edgecolor="white")
        if train_mean is not None:
            ax.axhline(train_mean, color=TRAIN_COLOR, linestyle="--", linewidth=1.4, label="train mean")
            ax.legend(fontsize=8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(
            [f"{name} (n={group[name]['n']})" for name in names], rotation=35, ha="right", fontsize=7
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$\\mathcal{L}_{flow}$")
        _style(ax, f"Val flow loss by {xlabel}")

    # Loss against position in the episode. A rising trend means the late phase of the
    # task — where the scene is most disturbed and the demonstrations most varied — is
    # what the policy fits worst, which the per-episode means average away.
    progress = np.array([r["progress"] for r in val_rows if r.get("loss_flow") is not None])
    losses = column(val_rows, "loss_flow")
    if progress.size == losses.size and progress.size:
        axes[2].scatter(progress, losses, s=14, color=VAL_COLOR, alpha=0.45, edgecolor="none")
        edges = np.linspace(0, 1, 11)
        centres, means = [], []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            inside = (progress >= low) & (progress < high if high < 1 else progress <= 1)
            if inside.any():
                centres.append((low + high) / 2)
                means.append(losses[inside].mean())
        axes[2].plot(centres, means, "o-", color="#1D3557", linewidth=1.8, label="decile mean")
        if train_mean is not None:
            axes[2].axhline(train_mean, color=TRAIN_COLOR, linestyle="--", linewidth=1.4, label="train mean")
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel("position in the episode (0 = start, 1 = end)")
    axes[2].set_ylabel("$\\mathcal{L}_{flow}$")
    _style(axes[2], "Val flow loss across the episode")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "breakdown.png"), bbox_inches="tight", dpi=125)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def run(adapter, dataset, cfg, output_dir: str, train_dataset=None) -> dict | None:
    """Measure the training objective on ``dataset`` (val) and on the training sources.

    Returns the summary dict so the caller can push the headline scalars onto the same
    Aim axes as the training curve.
    """
    makedirs(output_dir)
    p = cfg.probe_parameters
    n_timesteps = max(1, int(getattr(cfg.policy, "num_flow_timesteps", 1)))
    timesteps = flow_timestep_grid(cfg.policy, n_timesteps)

    adapter._set_probe_cuda_graph_enabled(False)
    try:
        val_rows = measure_split(adapter, [("val", dataset)], cfg, "val", timesteps)
        train_rows = (
            measure_split(adapter, training_datasets(cfg, train_dataset), cfg, "train", timesteps)
            if train_dataset is not None
            else []
        )
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not val_rows:
        logging.warning("[objective] no val frames produced measurements.")
        return None

    summary = build_summary(
        val_rows, train_rows, timesteps, adapter.action_dim, adapter.action_token_vocab_size, cfg
    )
    bands = exemplar_bands(val_rows, int(getattr(p, "objective_exemplars_per_band", 3)))
    summary["exemplars"] = [
        {
            "percentile": band["percentile"],
            "label": band["label"],
            "target": band["target"],
            "frames": [
                {
                    "global_idx": row["global_idx"],
                    "episode_idx": row["episode_idx"],
                    "frame_idx": row["frame_idx"],
                    "loss_flow": row["loss_flow"],
                    "subtask": row["subtask"],
                }
                for row in band["frames"]
            ],
        }
        for band in bands
    ]
    with open(os.path.join(output_dir, "objective.json"), "w") as f:
        json.dump(summary, f, indent=2)

    render_objective(summary, val_rows, train_rows, output_dir)
    render_flow(summary, output_dir)
    render_discrete(summary, val_rows, train_rows, output_dir)
    render_breakdown(summary, val_rows, output_dir)
    render_action_exemplars(bands, adapter, dataset, cfg, output_dir)
    _write_index(summary, output_dir)

    parts = []
    for label, key in (
        ("flow", "loss_flow"),
        ("flow aux", "loss_action_aux"),
        ("FAST CE", "loss_discrete_ce"),
        ("FAST aux", "loss_discrete_aux"),
        ("depth event", "loss_depth_event"),
    ):
        entry = summary[key]
        if not entry:
            continue
        parts.append(
            f"{label} val={entry['val']:.5f}"
            + (f" train={entry['train']:.5f} (z={entry['z']:+.1f})" if "train" in entry else "")
        )
    logging.info("[objective] " + "  |  ".join(parts))

    # The raw BCE above is unreadable without its floor and ceiling; print the
    # normalised form next to it so a flat curve can be told from a dead head.
    for head, splits in (summary.get("depth_event_reference") or {}).items():
        for split, entry in splits.items():
            if "captured" not in entry:
                continue
            logging.info(
                f"[objective] depth_event/{head} {split}: bce={entry['bce']:.4f} "
                f"in [{entry['floor']:.4f} floor, {entry['constant']:.4f} base-rate] "
                f"-> captured={entry['captured']:+.1%} +/- {entry['captured_sem']:.1%} "
                f"(n={entry['n']}, base rate {entry['base_rate']:.2%})"
            )
    return summary


def aim_scalars(summary: dict) -> dict:
    """Headline numbers for the training run's own Aim panel.

    Detailed val/train pairs stay in the probe artifact; Aim gets only generalization
    z-scores and held-out FAST top-1.
    """
    scalars: dict[str, float] = {}
    headline_keys = (
        ("flow", "loss_flow"),
        ("action_aux", "loss_action_aux"),
        ("discrete_ce", "loss_discrete_ce"),
        ("discrete_aux", "loss_discrete_aux"),
    )
    for name, key in headline_keys:
        entry = summary.get(key) or {}
        if entry.get("z") is not None:
            scalars[f"objective_z_{name}"] = float(entry["z"])
    for split, entry in (summary.get("discrete") or {}).items():
        if split == "val":
            scalars["objective_val_fast_top1"] = entry["top1"]
    # Fraction of the learnable BCE range each depth head holds — the raw BCE moves
    # too little against its own sampling noise to be worth a Aim panel.
    for head, splits in (summary.get("depth_event_reference") or {}).items():
        captured = (splits.get("val") or {}).get("captured")
        if captured is not None:
            scalars[f"objective_val_depth_event_{head}_captured"] = float(captured)
    return scalars


def _write_index(summary: dict, output_dir: str) -> None:
    provisional = " Thresholds mark a gap several SEM wide, not a calibrated level."
    chance = summary.get("discrete_chance_top1")
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Objective",
        group="Objective",
        claim="Does the loss the trainer minimises hold up on held-out episodes?",
        summary=summary,
        see_also=["action_trace", "actions", "subtask_sweep"],
        metrics=[
            Metric(
                "loss_flow.val",
                "Flow loss (val)",
                good="low",
                fmt=5,
                primary=True,
                note="$\\mathbb{E}_{t,\\varepsilon}\\|v_\\theta(x_t) - (a - \\varepsilon)\\|^2$ "
                "on held-out frames. Large irreducible floor, so read the ratio "
                "and $z$, not this level.",
                trend=True,
            ),
            Metric(
                "loss_flow.z",
                "Flow gap (standard errors)",
                good="low",
                fmt=1,
                baseline=0.0,
                warn=2.0,
                bad=5.0,
                primary=True,
                note="$(\\bar{\\mathcal{L}}_{val} - \\bar{\\mathcal{L}}_{train}) / "
                "\\sqrt{\\mathrm{SEM}^2_{val} + \\mathrm{SEM}^2_{train}}$." + provisional,
                trend=True,
            ),
            Metric(
                "loss_flow.ratio",
                "Flow val/train ratio",
                good="low",
                fmt=3,
                baseline=1.0,
                primary=True,
                note="$\\bar{\\mathcal{L}}_{val} / \\bar{\\mathcal{L}}_{train}$. The floor "
                "compresses this toward 1, so a small excess is a large effect.",
            ),
            Metric(
                "loss_action_aux.val",
                "Flow auxiliary loss (val)",
                good="low",
                fmt=5,
                primary=True,
                note="Weighted, threshold-gated trajectory auxiliary loss on held-out frames. "
                "Absent when the continuous auxiliary family is disabled.",
            ),
            Metric(
                "loss_action_aux.z",
                "Flow auxiliary gap (standard errors)",
                good="low",
                fmt=1,
                baseline=0.0,
                warn=2.0,
                bad=5.0,
                note="Held-out minus training auxiliary loss in pooled SEM units." + provisional,
            ),
            Metric(
                "loss_discrete_aux.val",
                "FAST auxiliary loss (val)",
                good="low",
                fmt=5,
                primary=True,
                note="Weighted ordinal/path/shape auxiliary loss on held-out FAST logits.",
            ),
            Metric(
                "loss_discrete_aux.z",
                "FAST auxiliary gap (standard errors)",
                good="low",
                fmt=1,
                baseline=0.0,
                warn=2.0,
                bad=5.0,
                note="Held-out minus training FAST auxiliary loss in pooled SEM units." + provisional,
            ),
            Metric(
                "loss_discrete_ce.val",
                "FAST cross-entropy (val)",
                good="low",
                fmt=4,
                primary=True,
                note="$-\\frac{1}{N}\\sum_j \\log p(y_j \\mid y_{<j}, c)$ over the action-token "
                "span, token-weighted as the trainer weights it.",
                trend=True,
            ),
            Metric(
                "loss_discrete_ce.z",
                "FAST CE gap (standard errors)",
                good="low",
                fmt=1,
                baseline=0.0,
                warn=2.0,
                bad=5.0,
                primary=True,
                note="As above for the CE. Deterministic given the frame, so this $z$ is "
                "frame-to-frame spread alone." + provisional,
            ),
            Metric(
                "discrete.val.top1",
                "FAST top-1 accuracy (val)",
                good="high",
                fmt=3,
                baseline=chance,
                primary=True,
                note=(
                    f"Fraction of action tokens whose $\\arg\\max$ is correct under "
                    f"teacher forcing. Chance $= 1/V = {chance:.1e}$."
                    if chance
                    else "No action-token vocabulary size in config."
                ),
                trend=True,
            ),
            Metric(
                "discrete.val.perplexity",
                "FAST perplexity (val)",
                good="low",
                fmt=1,
                note="$e^{CE}$ — the effective number of action tokens the head is "
                "choosing between at each position.",
            ),
            *[
                Metric(
                    f"depth_event_reference.{head}.val.captured",
                    label,
                    good="high",
                    fmt=3,
                    baseline=0.0,
                    primary=True,
                    note="Fraction of the learnable BCE range this depth head holds: "
                    "$1 - (\\mathcal{L} - \\mathbb{E}[H(t)]) / (H(\\mathbb{E}[t]) - "
                    "\\mathbb{E}[H(t)])$. Soft targets give BCE an irreducible floor "
                    "$\\mathbb{E}[H(t)]$, and the base rate alone buys "
                    "$H(\\mathbb{E}[t])$; the two are ~0.155 nats apart here, so the "
                    "raw loss cannot be read on its own. $1$ is a perfect predictor, "
                    "$0$ ignores the depth observation, negative is worse than the "
                    "base rate.",
                )
                for head, label in (
                    ("close", "Depth close-event range captured (val)"),
                    ("open", "Depth open-event range captured (val)"),
                )
            ],
            Metric("n_val_frames", "Val frames measured", good="none", fmt=0),
        ],
        extra={
            "provenance": summary.get("data", {}),
            "viewer": {
                "show_supporting_metrics": False,
                "show_documentation": False,
            },
        },
        panels=[
            Panel(
                "objective.png",
                "Generalisation gap",
                "**Left, middle** — $\\bar{\\mathcal{L}}$ per split, one point per split, "
                "whiskers $\\pm\\,\\mathrm{SEM}$; titles carry the gap and its $z$. "
                "**Right** — the distribution those means average: one observation per "
                "frame, area-normalised so the two splits compare despite unequal $n$.",
                primary=True,
                refs=["action_trace"],
            ),
            Panel(
                "action_exemplars.html",
                "GT and generated actions at low, typical, and high flow loss",
                "Held-out frames nearest $p_5$, $p_{50}$, and $p_{95}$ of the val flow-loss "
                "distribution. Black is the demonstrated action chunk; blue is one "
                "deterministic generated chunk from the same observation. Orbit the 3-D "
                "trace, hover a target, and use the slider or arrow keys to move between "
                "the sampled frames; the camera views and subtask update with it.",
                primary=True,
            ),
            Panel(
                "flow.png",
                "Where the flow loss sits",
                "The same $\\mathcal{L}_{flow}$ resolved along three axes it is otherwise "
                "summed over. **Left** — $\\mathcal{L}(t_k)$ at each grid timestep: the "
                "target is nearly unconditioned as $t \\to 0$, so every model rises there. "
                "**Middle** — $\\mathcal{L}$ at each step of the 30-step chunk, averaged "
                "over $t$ and dims. **Right** — $\\mathcal{L}$ per joint, averaged over $t$ "
                "and chunk steps; the seven bars average back to $\\mathcal{L}_{flow}$.",
                primary=True,
            ),
            Panel(
                "discrete.png",
                "FAST head fidelity",
                "**Left** — fraction of action tokens whose true id is the $\\arg\\max$ "
                "(top-1) or within the 5 highest logits (top-5), teacher-forced, against "
                "chance $1/V$. **Middle** — mean CE at position $j$ of the action-token "
                "span, $\\mathrm{CE}_j = -\\frac{1}{N}\\sum_n \\log p(y^{(n)}_j \\mid "
                "y^{(n)}_{<j}, c^{(n)})$: the span is a BPE coding of DCT coefficients "
                "ordered low $\\to$ high frequency, so $j$ runs coarse to fine. Spans vary "
                "in length, so only positions at least half the frames reach are drawn. "
                "**Right** — distribution over frames of the whole-chunk CE.",
                primary=True,
                refs=["action_trace"],
            ),
            Panel(
                "breakdown.png",
                "Val loss by episode, subtask, and episode position",
                "$\\bar{\\mathcal{L}}_{flow}$ over the val frames grouped three ways, "
                "against $\\bar{\\mathcal{L}}_{train}$ (dashed). **Right** — position is "
                "the frame's index within its episode, rescaled to $[0, 1]$; the line is "
                "the decile mean of the scatter.",
            ),
        ],
    )


@parser.wrap()
def cli(cfg: TrainRLServerPipelineConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    val_path = getattr(cfg, "val_dataset_path", None)
    if not val_path:
        raise SystemExit("val_dataset_path is unset — there is no held-out split to measure.")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    val_dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
    val_dataset.delta_timestamps = None
    val_dataset.delta_indices = None
    run(
        adapter,
        val_dataset,
        cfg,
        os.path.join(cfg.probe_parameters.output_dir, "objective"),
        train_dataset=dataset,
    )


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
    import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — registers MolmoAct2RLConfig
    import lerobot.rl.pi05.rl_pi05  # noqa: F401 — registers PI05RLConfig
    from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401 — registers robot configs
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml
    from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401 — registers teleop configs

    config_path, remaining_args = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining_args, f"--config_path={_preprocess_config_yaml(config_path)}"]
    cli()


if __name__ == "__main__":
    register_config_choices()
    main()
