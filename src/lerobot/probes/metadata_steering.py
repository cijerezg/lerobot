r"""Metadata-steering probe: does the quality / mistake / speed clause reach the actions?

The action prompt carries a steering clause built by ``_build_robot_text``
(processor_molmoact2.py) — "The quality is $N$ of 5.", "The robot made a
mistake."/"The robot made no mistakes." and "The speed is $N$ of 5." At rollout every
deployment prompt asks for quality 5, no mistakes, speed 5. This probe tests whether
asking does anything.

Every condition is one ``predict_action_chunk`` on the same frame under the same seeded
flow noise, so the only thing that varies is the clause:

  ``none``          no metadata clause at all
  ``q1`` … ``q5``   quality $q$, no mistake, speed 5
  ``q1m``, ``q5m``  the quality poles with the mistake sentence on
  ``s1`` … ``s5``   speed $k$ at quality 5 with no mistake. ``s5`` is
                    character-for-character the ``q5`` prompt, so it is aliased rather
                    than forwarded twice
  ``gt``            the frame's own labels, from ``meta/episode_metadata.parquet``
                    (quality, per segment) + ``meta/mistakes.parquet`` (per 4 s window)
                    + the speed table (``REBOT_SPEED_TABLE``, frame-level runs since
                    2026-10-26), the same spans training broadcasts
  seed floor        ``q5`` re-drawn under ``n_seeds - 1`` other flow seeds

Write $a^{(c)}$ for the normalized chunk under condition $c$, $T$ and $D$ for chunk steps
and joints, and

$$\lVert x \rVert = \sqrt{\frac{1}{TD}\sum_{t,d} x_{t,d}^{2}}$$

for the RMSE norm every distance here uses.

**1. Does it clear the floor?** (``steering_floor.png``) Four clause contrasts per frame,
each read against the seed floor $F=\operatorname{mean}_{s\neq s'}\lVert a^{(s)}-a^{(s')}\rVert$
over the reseeds of ``q5``:

  quality      $\lVert a^{(q_5)}-a^{(q_1)}\rVert$
  mistake      $\frac{1}{2}\sum_{q\in\{1,5\}}\lVert a^{(q,\mathrm{T})}-a^{(q,\mathrm{F})}\rVert$
  speed        $\lVert a^{(s_5)}-a^{(s_1)}\rVert$
  both poles   $\lVert a^{(q_5)}-a^{(q_1,\mathrm{T})}\rVert$

The separation $S = \text{contrast} / F$ per frame, reported as a median; $S \approx 1$ means
the clause did nothing the sampler was not already doing.

**2. Is the mistake sentence read in context?** (``mistake_context.png``) The mistake
contrast above, split by whether the frame sits inside a flagged window of
``meta/mistakes.parquet``. A model that reads the sentence responds more where the
observation is consistent with it; equal boxes mean a constant offset. The ratio of the
two medians is ``mistake_context_ratio``.

**3. Examples.** The three frames with the largest quality separation, every condition
drawn per joint against the demonstration. The title's $\tau$ is Kendall's over the five
$(q, \pi(q))$ pairs, $\pi(q)=\langle a^{(q)}-\bar{a},u\rangle/\lVert u\rVert$ with
$u=a^{(q_5)}-a^{(q_1)}$ and $\bar{a}$ the five-level mean: $+1$ = the levels land in order.

**Sampling.** Mistake windows are about 6% of the held-out frames (``rebot_val_2026-10-14-v1``:
11 windows, ~1,070 of 17,022 frames), so an even sample alone leaves question 2 a handful of
frames. Two sets: ``metadata_steering_n_frames`` evenly spaced frames per episode
(``sample_episodes_evenly``), plus ``metadata_steering_mistake_frames_per_window`` evenly
spaced frames inside every flagged window (``sample_mistake_windows``, rows marked
``targeted``). Figure 1, the summary medians and the trend metrics use the even set only, so
the mistake frames are not over-weighted there; figure 2 and the examples use both.

Cost is ``frames x (12 + n_seeds)`` forwards — 12 clause rows plus ``gt`` in one batched
call, ``n_seeds - 1`` reseeds in another. The precision ramp and contact sweep (20 rows per
frame, 56% of the old cost) were removed 2026-10-26; the ``precision_contact_sweep`` probe
covers those clauses.

Registered probe: enable with ``probe_parameters.enable_metadata_steering``.
"""

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
from lerobot.probes.utils import (
    frame_metadata_lookup,
    joint_names_for_dim,
    makedirs,
    panel_caption as _caption,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
    sample_mistake_windows,
)

_QUALITY_LEVELS = (1, 2, 3, 4, 5)
_MISTAKE_POLES = (1, 5)  # where the mistake sentence is flipped
_SPEED_LEVELS = (1, 2, 3, 4, 5)
# The rollout clause asks for speed 5, so every quality / mistake cell holds speed at
# 5 and the speed ramp holds quality at 5 with no mistake: one axis moves per cell.
_DEPLOYED_SPEED = 5

# The quality levels get a sequential ramp so a monotone response is visible as a colour
# order; everything that is not a quality level stays off that ramp.
_QUALITY_COLORS = {q: plt.get_cmap("viridis")(i / 4) for i, q in enumerate(_QUALITY_LEVELS)}
_SPEED_COLORS = {k: plt.get_cmap("plasma")(i / 4) for i, k in enumerate(_SPEED_LEVELS)}
_CONDITION_STYLE = {
    "none": ("#E63946", "--"),
    "gt": ("#B5179E", ":"),
    **{f"q{q}": (_QUALITY_COLORS[q], "-") for q in _QUALITY_LEVELS},
    **{f"q{q}m": (_QUALITY_COLORS[q], "-.") for q in _MISTAKE_POLES},
    **{f"s{k}": (_SPEED_COLORS[k], (0, (5, 2))) for k in _SPEED_LEVELS},
}

_STEERED = {f"q{q}": {"quality": q, "mistake": False, "speed": _DEPLOYED_SPEED} for q in _QUALITY_LEVELS}
_STEERED |= {f"q{q}m": {"quality": q, "mistake": True, "speed": _DEPLOYED_SPEED} for q in _MISTAKE_POLES}
# s5 is the same prompt as q5 (quality 5, no mistake, speed 5); it is aliased below
# rather than forwarded twice.
_STEERED |= {f"s{k}": {"quality": 5, "mistake": False, "speed": k} for k in _SPEED_LEVELS if k != _DEPLOYED_SPEED}

# The clause the rollout prompt carries, and the one the old probe called "bad".
_ROLLOUT = "q5"
_OPPOSITE = "q1m"


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def _step_rms(chunk: torch.Tensor) -> float:
    """RMS per-step travel — how much the chunk moves, as opposed to where it sits."""
    return float(torch.diff(chunk, dim=-2).pow(2).mean().sqrt())


def _pairwise_rmse(chunks: list[torch.Tensor]) -> tuple[float, float]:
    """Mean and max pairwise RMSE — the seed floor when the chunks differ only in noise."""
    if len(chunks) < 2:
        return 0.0, 0.0
    values = [
        _rmse(chunks[i], chunks[j]) for i in range(len(chunks)) for j in range(i + 1, len(chunks))
    ]
    return float(np.mean(values)), float(np.max(values))


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([row[key] for row in rows if row.get(key) is not None], dtype=float)


def _mean(rows: list[dict], key: str) -> float:
    values = _column(rows, key)
    return float(values.mean()) if values.size else float("nan")


def _median(rows: list[dict], key: str) -> float:
    values = _column(rows, key)
    return float(np.median(values)) if values.size else float("nan")


def _median_se(rows: list[dict], key: str, n_boot: int = 512) -> float:
    """Bootstrap standard error of the median.

    The relative displacements are a ratio with a heavy right tail, so the buckets are
    summarized by their median; this is the matching spread. Seeded per call so the
    figure is reproducible.
    """
    values = _column(rows, key)
    if values.size < 2:
        return 0.0
    rng = np.random.default_rng(0)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    return float(np.median(draws, axis=1).std())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 1e-12 else 0.0


def _loo_alignment(vectors: list[np.ndarray]) -> list[float]:
    r"""Each displacement against the mean of every *other* frame's displacement.

    Leave-one-out because including $f$ in its own reference is positively biased by
    $1/N$ of the reference's own norm, which at these sample sizes is the whole effect
    being looked for.
    """
    if len(vectors) < 2:
        return []
    stack = np.stack(vectors)
    total = stack.sum(axis=0)
    return [_cosine(v, (total - v) / (len(vectors) - 1)) for v in stack]


def _level_projection(acts: dict[str, torch.Tensor], prefix: str) -> tuple[dict[int, float], float]:
    r"""Where each level sits on the axis its own two poles define, and Kendall's $\tau$.

    $\pi(q) = \langle a^{(q)} - \bar{a}, u\rangle / \lVert u \rVert$ with $u = a^{(q_5)} -
    a^{(q_1)}$, so $\pi(5) - \pi(1) = \lVert u \rVert$ and the interior levels are placed
    between the poles in the same units as every other distance here. $\tau$ over the five
    $(q, \pi(q))$ pairs is $+1$ for a monotone staircase and $0$ for levels in no order at
    all — the difference between a read scale and five unrelated strings. ``prefix``
    ``"q"`` is the quality ramp, ``"s"`` the speed ramp.
    """
    stack = torch.stack([acts[f"{prefix}{q}"] for q in _QUALITY_LEVELS])
    axis = stack[-1] - stack[0]
    norm = float(axis.pow(2).mean().sqrt())
    if norm < 1e-12:
        return dict.fromkeys(_QUALITY_LEVELS, 0.0), 0.0
    centred = stack - stack.mean(dim=0, keepdim=True)
    values = [float((chunk * axis).mean() / norm) for chunk in centred]
    concordant = sum(
        np.sign(values[j] - values[i])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return dict(zip(_QUALITY_LEVELS, values, strict=True)), float(concordant) / 10.0


def _quality_projection(acts: dict[str, torch.Tensor]) -> tuple[dict[int, float], float]:
    return _level_projection(acts, "q")


def _provenance(rows: list[dict], dataset, cfg, conditions: list[str], n_seeds: int) -> dict:
    """Which frames of which dataset the figures average, and what one forward was.

    Derived from the measured rows rather than from config, so it describes the frames
    that actually produced numbers after stride snapping and de-duplication.
    """
    p = cfg.probe_parameters
    episodes = sorted({row["episode_idx"] for row in rows})
    targeted = sum(row["targeted"] for row in rows)
    labelled = [row for row in rows if row["gt_quality"] is not None]
    mix = {q: sum(row["gt_quality"] == q for row in labelled) for q in _QUALITY_LEVELS}
    flagged = sum(row["gt_mistake"] for row in rows)
    forwards_per_frame = len(conditions) + n_seeds - 1

    label_line = (
        ", ".join(f"quality {q}: {mix[q]} frames" for q in _QUALITY_LEVELS if mix[q])
        + f"; {flagged} of {len(rows)} frames carry the mistake flag"
        if labelled
        else "no quality / mistake labels on this split — no ``gt`` row and no mistake split"
    )

    return {
        "val": {
            "n_frames": len(rows),
            "n_episodes": len(episodes),
            "sources": [
                {
                    "name": str(getattr(dataset, "repo_id", "val")),
                    "root": str(getattr(dataset, "root", "")),
                    "episodes": episodes,
                    "n_episodes": len(episodes),
                    "n_frames": len(rows),
                }
            ],
        },
        "frames_per_episode": int(getattr(p, "metadata_steering_n_frames", None) or p.n_frames_per_episode),
        "mistake_frames_per_window": int(p.metadata_steering_mistake_frames_per_window),
        "episode_budget": p.max_episodes,
        "image_stride": probe_image_stride(cfg),
        "chunk_size": int(cfg.policy.chunk_size),
        "batch_size": 1,
        "forwards": forwards_per_frame * len(rows),
        "details": [
            [
                "Per frame",
                f"{forwards_per_frame} forwards — "
                + ", ".join(f"``{name}``" for name in conditions)
                + f", plus {n_seeds - 1} reseed(s) of ``{_ROLLOUT}`` for the floor",
            ],
            [
                "The clause",
                "``The quality is $N$ of 5.``, ``The robot made a mistake.`` / "
                "``The robot made no mistakes.`` and ``The speed is $N$ of 5.``, appended to "
                "the action prompt by ``_build_robot_text``. ``none`` omits all of them.",
            ],
            [
                "Where the labels come from",
                "quality per *segment* from ``meta/episode_metadata.parquet``, mistake per 4 s "
                "window from ``meta/mistakes.parquet``, speed from the frame-level speed table "
                "— the spans ``ReplayBuffer.materialize_metadata`` broadcasts at training, read "
                "here by ``frame_metadata_lookup``",
            ],
            ["Label mix of the sampled frames", label_line],
            [
                "Frame sets",
                f"{len(rows) - targeted} evenly spaced frames (figure 1, summary, trends) + "
                f"{targeted} frames from inside flagged mistake windows (figure 2 and the "
                "examples only)",
            ],
        ],
        "sampling": (
            "Frames evenly spaced across each episode, snapped onto the image/depth stride "
            f"grid; episodes drawn by a seeded subset when the budget is smaller than the "
            f"split (seed {int(p.random_seed)}). On top, evenly spaced frames inside every "
            "flagged mistake window, so the mistake split has enough flagged frames; those "
            "are kept out of every pooled number. Every distance is a within-frame difference "
            "between two conditions on the same observation."
        ),
        "regime": (
            "one frame per forward, batch size 1; "
            f"{int(getattr(cfg.policy, 'num_inference_steps', 0))} flow denoising steps, seed 0 "
            "in every condition so the clause is the only difference; the batch carries no "
            "action target, so training-time prompt dropout is not armed"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — does the clause clear the floor?
# ──────────────────────────────────────────────────────────────────────────────

def _render_floor(rows: list[dict], summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_gridspec(1, 1, left=0.13, right=0.97, top=0.78, bottom=0.32).subplots()

    contrasts = [
        ("quality\nq5 vs q1", "quality_range_rmse"),
        ("mistake\nsentence", "mistake_flip_rmse"),
        ("speed\ns5 vs s1", "speed_range_rmse"),
        ("both poles\n(old good/bad)", "pole_range_rmse"),
        ("flow seed\n(floor)", "seed_floor_mean"),
    ]
    ax.boxplot([_column(rows, key) for _label, key in contrasts],
               tick_labels=[label for label, _key in contrasts])
    ax.set_ylabel(r"$\|a^{(c)} - a^{(c')}\|$  (normalized actions)")
    ax.set_title(
        "Does the clause move the chunk\nmore than noise does?\n"
        f"median separation $S$: quality {summary['separation_median']:.2f}x   ·   "
        f"mistake {summary['mistake_separation_median']:.2f}x   ·   "
        f"speed {summary['speed_separation_median']:.2f}x"
    )
    _caption(ax, [
        r"One evenly sampled frame gives one point to each box; the mistake-window frames are left",
        r"out. The first four are clause contrasts on that frame, the fifth is the same clause",
        r"($q_5$, no mistake, speed 5) re-drawn under " + str(summary["n_seeds"]) + r" flow seeds. Only the ratio $S$",
        r"to the fifth means anything; $S \approx 1$ says the clause did nothing the sampler was",
        r"not already doing. 'speed' holds quality at 5 and no mistake and moves the speed number",
        r"1 to 5. 'both poles' moves quality and the mistake sentence at once.",
    ])

    fig.suptitle(
        f"Metadata steering — clause vs flow-seed floor (n={summary['n_frames_even']} frames, "
        f"{summary['n_episodes']} held-out episodes)",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — is the mistake sentence read in context?
# ──────────────────────────────────────────────────────────────────────────────

def _render_mistake_context(rows: list[dict], summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_gridspec(1, 1, left=0.13, right=0.97, top=0.82, bottom=0.32).subplots()

    clean = [row["mistake_flip_rmse"] for row in rows if not row["gt_mistake"]]
    flagged = [row["mistake_flip_rmse"] for row in rows if row["gt_mistake"]]
    groups = [values for values in (clean, flagged) if values]
    labels = [
        f"{name} (n={len(values)})"
        for name, values in (("no mistake", clean), ("GT mistake", flagged)) if values
    ]
    if groups:
        ax.boxplot(groups, tick_labels=labels)
    ax.axhline(summary["seed_floor_mean"], color="#E63946", linestyle="--", linewidth=1.2,
               label="flow-seed floor")
    ax.set_ylabel(r"$\|a^{(q, T)} - a^{(q, F)}\|$")
    ax.set_title(
        "Does the mistake sentence land harder\non frames that really are mistakes?\n"
        f"median ratio {summary['mistake_context_ratio']:.2f}x"
    )
    ax.legend(fontsize=8)
    _caption(ax, [
        r"The mistake-sentence effect $\frac{1}{2}\sum_q \|a^{(q,T)} - a^{(q,F)}\|$, split by whether the frame",
        r"sits inside a flagged 4 s window in meta/mistakes.parquet. Both frame sets: the GT-mistake",
        r"box is mostly frames sampled from inside the windows on purpose. A model that reads the",
        r"sentence in context responds more where the observation is consistent with it; equal",
        r"boxes (ratio ~1) mean the sentence is a constant offset regardless of the frame.",
    ])

    fig.suptitle(
        f"Metadata steering — mistake sentence by true label (n={summary['n_frames']})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Examples
# ──────────────────────────────────────────────────────────────────────────────

def _render_example(diagnostic: dict, output_path: str) -> None:
    """Per-joint chunk under every metadata condition, demonstration overlaid."""
    gt = diagnostic["gt"]
    action_dim = gt.shape[-1]
    names = joint_names_for_dim(action_dim)
    steps = np.arange(gt.shape[0])

    n_cols = 4
    n_rows = (action_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.0 * n_rows), squeeze=False)
    for joint in range(action_dim):
        ax = axes[joint // n_cols][joint % n_cols]
        ax.plot(steps, gt[:, joint], color="black", linewidth=2.0, label="demonstration")
        for condition, actions in diagnostic["acts"].items():
            color, linestyle = _CONDITION_STYLE[condition]
            ax.plot(steps, actions[:, joint], color=color, linestyle=linestyle,
                    linewidth=1.25, label=condition)
        ax.set_title(f"{names[joint]} (normalized)")
        ax.grid(True, alpha=0.25, linestyle=":")
        if joint == 0:
            ax.legend(fontsize=7, ncol=3)
    for unused in range(action_dim, n_rows * n_cols):
        axes[unused // n_cols][unused % n_cols].axis("off")

    row = diagnostic["row"]
    fig.suptitle(
        f"episode {row['episode_idx']}, frame {row['frame_idx']}  |  "
        f"true quality={row['gt_quality']} mistake={row['gt_mistake']}  |  "
        f"quality range={row['quality_range_rmse']:.4f} "
        f"({row['quality_range_rmse'] / max(row['seed_floor_mean'], 1e-9):.1f}x floor)  |  "
        rf"$\tau$={row['kendall_tau']:+.1f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


_FLOOR_HOW = r"""Every number on this page is a within-frame contrast: the same
observation, the same seeded flow noise, one clause changed. The frames themselves are
listed in *Data behind these numbers*.

Four clause contrasts against the one contrast that is not a clause at all: the rollout
prompt re-drawn under different flow seeds. Only the ratio $S$ between them means
anything, and $S \approx 1$ means the clause is decorative. "both poles" moves the quality
number *and* the mistake sentence together, so it cannot attribute an effect to either.
Evenly sampled frames only."""

_MISTAKE_HOW = r"""The mistake-sentence effect, $\frac{1}{2}\sum_{q\in\{1,5\}}
\lVert a^{(q,T)}-a^{(q,F)}\rVert$, split by whether the frame is inside a flagged mistake
window. Responding more where the observation is consistent with the sentence is the
difference between reading it and applying it as a constant offset. The flagged frames
are mostly sampled from inside the windows on purpose
(``metadata_steering_mistake_frames_per_window`` per window), since an even sample of a
held-out split holds only a handful."""


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
    chunk_size = int(cfg.policy.chunk_size)
    n_seeds = max(int(getattr(p, "metadata_steering_n_seeds", None) or p.n_seeds), 2)
    n_frames = int(getattr(p, "metadata_steering_n_frames", None) or p.n_frames_per_episode)

    gt_metadata = frame_metadata_lookup(dataset)
    if not gt_metadata:
        logging.warning(
            "[metadata_steering] no meta/episode_metadata.parquet + mistakes.parquet on "
            f"{dataset.root} — no ``gt`` condition and no mistake-window frames; the floor "
            "figure still runs."
        )

    adapter._set_probe_cuda_graph_enabled(False)  # prompt changes per condition; keep eager
    stride = probe_image_stride(cfg)
    samples = sample_episodes_evenly(dataset, n_frames, p.max_episodes, p.random_seed, stride)
    targeted = sample_mistake_windows(
        dataset, gt_metadata, int(p.metadata_steering_mistake_frames_per_window),
        sorted({episode_idx for episode_idx, _, _ in samples}), stride,
        {global_idx for _, _, global_idx in samples},
    )
    targeted_idx = {global_idx for _, _, global_idx in targeted}
    samples = samples + targeted
    logging.info(
        f"[metadata_steering] {len(samples) - len(targeted)} even + {len(targeted)} mistake-window "
        f"frames x ({len(_STEERED) + 1} clauses + gt + {n_seeds - 1} reseeds) = "
        f"~{len(samples) * (len(_STEERED) + 1 + n_seeds)} forward passes"
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    try:
        for episode_idx, frame_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size, metadata=None)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])
            labels = gt_metadata.get(global_idx)

            # Every clause in one forward, all sharing seed 0's noise draw, so the
            # contrast between two rows is the clause alone.
            condition_names: list[str] = ["none"]
            condition_metadata: list[dict | None] = [None]
            for name, steered_metadata in _STEERED.items():
                condition_names.append(name)
                condition_metadata.append(steered_metadata)
            if labels is not None:
                condition_names.append("gt")
                condition_metadata.append(labels)
            _, condition_chunks = adapter.predict_action_chunk_batch(
                frame["obs"], frame["task"], [frame["subtask"]] * len(condition_names),
                metadatas=condition_metadata,
                noise=adapter.flow_noise_like(len(condition_names), 0),
            )
            acts = {name: condition_chunks[i] for i, name in enumerate(condition_names)}
            acts[f"s{_DEPLOYED_SPEED}"] = acts[_ROLLOUT]  # same prompt, not forwarded twice

            # Seed floor: the rollout clause re-drawn under different noise. Without it
            # every distance above is a number with no scale. One row per reseed.
            floor_draws = [acts[_ROLLOUT]]
            floor_noise = torch.cat(
                [adapter.flow_noise_like(1, seed) for seed in range(1, n_seeds)], dim=0
            )
            _, floor_chunks = adapter.predict_action_chunk_batch(
                frame["obs"], frame["task"], [frame["subtask"]] * (n_seeds - 1),
                metadatas=[_STEERED[_ROLLOUT]] * (n_seeds - 1),
                noise=floor_noise,
            )
            floor_draws += [floor_chunks[i] for i in range(n_seeds - 1)]
            floor_mean, floor_max = _pairwise_rmse(floor_draws)

            projection, tau = _quality_projection(acts)
            mistake_effect = [_rmse(acts[f"q{q}m"], acts[f"q{q}"]) for q in _MISTAKE_POLES]
            row = {
                "global_idx": int(global_idx),
                "episode_idx": int(episode_idx),
                "frame_idx": int(frame_idx),
                "targeted": global_idx in targeted_idx,
                "gt_quality": None if labels is None else int(labels["quality"]),
                "gt_mistake": bool(labels["mistake"]) if labels is not None else False,
                "gt_speed": None if labels is None else int(labels["speed"]),
                "seed_floor_mean": floor_mean,
                "seed_floor_max": floor_max,
                "quality_range_rmse": _rmse(acts["q5"], acts["q1"]),
                "mistake_flip_rmse": float(np.mean(mistake_effect)),
                "speed_range_rmse": _rmse(acts["s5"], acts["s1"]),
                "pole_range_rmse": _rmse(acts[_ROLLOUT], acts[_OPPOSITE]),
                "kendall_tau": tau,
                **{f"proj_q{q}": value for q, value in projection.items()},
                # The physical reading of the speed number: does asking for 5 move the arm more per step than 1?
                "speed_step_motion_ratio": _step_rms(acts["s5"]) / max(_step_rms(acts["s1"]), 1e-9),
            }
            row["separation"] = row["quality_range_rmse"] / max(floor_mean, 1e-9)
            row["mistake_separation"] = row["mistake_flip_rmse"] / max(floor_mean, 1e-9)
            row["speed_separation"] = row["speed_range_rmse"] / max(floor_mean, 1e-9)
            rows.append(row)
            diagnostics.append({"gt": gt_norm, "acts": acts, "row": row})
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[metadata_steering] no frames produced measurements.")
        return

    conditions = ["none"] + list(_STEERED) + (["gt"] if any(row["gt_quality"] is not None for row in rows) else [])
    even = [row for row in rows if not row["targeted"]]
    flagged = [row for row in rows if row["gt_mistake"]]
    clean = [row for row in rows if not row["gt_mistake"]]
    mistake_clean = _median(clean, "mistake_flip_rmse")
    mistake_flagged = _median(flagged, "mistake_flip_rmse")

    summary = {
        "n_frames": len(rows),
        "n_frames_even": len(even),
        "n_frames_mistake_window": len(rows) - len(even),
        "n_episodes": len({row["episode_idx"] for row in rows}),
        "n_frames_labelled": sum(row["gt_quality"] is not None for row in rows),
        "n_frames_gt_mistake": len(flagged),
        "n_seeds": n_seeds,
        "conditions": conditions,
        "seed_floor_mean": _mean(even, "seed_floor_mean"),
        "quality_range_rmse": _mean(even, "quality_range_rmse"),
        "mistake_flip_rmse": _mean(even, "mistake_flip_rmse"),
        "speed_range_rmse": _mean(even, "speed_range_rmse"),
        "pole_range_rmse": _mean(even, "pole_range_rmse"),
        "separation_median": _median(even, "separation"),
        "mistake_separation_median": _median(even, "mistake_separation"),
        "speed_separation_median": _median(even, "speed_separation"),
        "speed_step_motion_ratio": _median(even, "speed_step_motion_ratio"),
        "kendall_tau_mean": _mean(even, "kendall_tau"),
        "mistake_flip_rmse_clean": mistake_clean,
        "mistake_flip_rmse_gt_mistake": mistake_flagged,
        "mistake_context_ratio": mistake_flagged / max(mistake_clean, 1e-12) if flagged else float("nan"),
        "verdict_note": (
            "separation ~1 => the clause moves the chunk no more than flow noise does. "
            "mistake_context_ratio ~1 => the mistake sentence is one offset applied to every "
            "frame; > 1 => it lands harder where the frame really is a mistake."
        ),
    }
    summary["data"] = _provenance(rows, dataset, cfg, conditions, n_seeds)

    with open(os.path.join(output_dir, "metadata_steering.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    _render_floor(even, summary, os.path.join(output_dir, "steering_floor.png"))
    _render_mistake_context(rows, summary, os.path.join(output_dir, "mistake_context.png"))

    examples_dir = os.path.join(output_dir, "examples")
    makedirs(examples_dir)
    ranked = sorted(range(len(rows)), key=lambda i: -rows[i]["separation"])
    examples = []
    for index in ranked[:3]:
        row = rows[index]
        name = f"examples/sep_ep{row['episode_idx']:04d}_fr{row['frame_idx']:06d}.png"
        _render_example(diagnostics[index], os.path.join(output_dir, name))
        examples.append(
            Panel(
                name,
                f"Largest steering range #{len(examples) + 1} — episode {row['episode_idx']}, "
                f"frame {row['frame_idx']} ({row['separation']:.1f}x its own floor)",
                "The predicted chunk per joint under every metadata condition against the "
                "demonstration in black. The five quality levels run dark-to-light on the "
                "viridis ramp, ``q1m``/``q5m`` are the same colours dash-dotted with the "
                "mistake sentence on, ``s1``-``s4`` are dashed on the plasma ramp, ``none`` is "
                "the red dashed origin and ``gt`` the frame's own labels. Ranked by steering "
                "range in units of the frame's own seed floor, so this is where the clause did "
                "the most it ever does: if the ramp is not ordered here, it is ordered nowhere.",
            )
        )

    panels = [
        Panel("steering_floor.png",
              "Floor — does each clause move the chunk more than flow noise does?",
              how=_FLOOR_HOW, primary=True, refs=["subtask_sweep"]),
        Panel("mistake_context.png",
              "Mistake sentence — does it land harder on frames that really are mistakes?",
              how=_MISTAKE_HOW, primary=True),
        *examples,
    ]
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Metadata Steering",
        group="Steering",
        claim="Does the quality / mistake / speed clause reach the actions, and is the mistake sentence read in context?",
        summary=summary,
        see_also=["subtask_sweep", "precision_contact_sweep", "objective"],
        metrics=[
            Metric(
                "separation_median",
                "quality / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                primary=True,
                trend=True,
                note="Quality-range RMSE (q1 to q5) over the seed floor. 1 means the quality clause moves the chunk no further than reseeding does.",
            ),
            Metric(
                "mistake_separation_median",
                "mistake / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                trend=True,
                note="Mistake-sentence RMSE (on vs off, averaged over quality 1 and 5) over the seed floor.",
            ),
            Metric(
                "speed_separation_median",
                "speed / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                trend=True,
                note="Speed-range RMSE (speed 1 to 5 at quality 5, no mistake) over the seed floor.",
            ),
            Metric(
                "mistake_context_ratio",
                "mistake sentence, GT-mistake over clean frames",
                good="high",
                fmt=2,
                baseline=1.0,
                primary=True,
                trend=True,
                note="Median mistake-sentence effect on frames inside a flagged mistake window over the median on clean frames. 1 means the sentence is the same offset on every frame.",
            ),
            Metric(
                "speed_step_motion_ratio",
                "per-step motion, speed 5 over speed 1",
                fmt=2,
                baseline=1.0,
                note="RMS per-step travel of the speed-5 chunk over the speed-1 chunk. Above 1 means 'fast' reads as moving more per step.",
            ),
        ],
        panels=panels,
        extra={"provenance": summary["data"]},
    )

    logging.info(
        f"[metadata_steering] n={len(rows)} ({len(even)} even, {len(rows) - len(even)} mistake-window, "
        f"{len(flagged)} GT-mistake)  "
        f"separation: quality {summary['separation_median']:.2f}x  "
        f"mistake {summary['mistake_separation_median']:.2f}x  "
        f"speed {summary['speed_separation_median']:.2f}x  "
        f"(floor {summary['seed_floor_mean']:.4f})  "
        f"mistake context {summary['mistake_context_ratio']:.2f}x  "
        f"speed step motion x{summary['speed_step_motion_ratio']:.2f}  "
        f"tau={summary['kendall_tau_mean']:+.2f}"
    )
