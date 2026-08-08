r"""Metadata-steering probe: does the quality / mistake clause reach the actions?

The action prompt carries a steering clause built by ``_build_robot_text``
(processor_molmoact2.py) — "The quality is $N$ of 5." and "The robot made a
mistake."/"The robot made no mistakes." — trained with ``metadata_dropout`` so the
model sees it on ~85% of samples. At rollout every deployment prompt asks for the
same thing: quality 5, no mistakes. This probe is the whole test of whether asking
does anything.

Two clauses, one of them a five-level scale, so the probe is a small factorial rather
than a two-pole contrast. Every condition is one ``predict_action_chunk`` on the same
frame under the same seeded flow noise, so the only thing that varies is the clause:

  ``none``          no metadata clause at all (the dropout regime, and the origin
                    every displacement below is measured from)
  ``q1`` … ``q5``   quality $q$, no mistake — the dose axis
  ``q1m``, ``q5m``  the same poles with the mistake sentence flipped on
  ``gt``            the frame's own labels, from ``meta/episode_metadata.parquet``
                    (per segment) + ``meta/mistakes.parquet`` (per 4 s window), the
                    same spans training broadcasts
  seed floor        ``q5`` re-drawn under different flow seeds

Write $a^{(c)}$ for the normalized chunk under condition $c$, $a^{\star}$ for the
demonstrated chunk, $T$ and $D$ for chunk steps and joints, and

$$\lVert x \rVert = \sqrt{\frac{1}{TD}\sum_{t,d} x_{t,d}^{2}}$$

for the RMSE norm every distance here uses.

**1. Does it clear the floor?** The steering range $\lVert a^{(q_5)}-a^{(q_1)}\rVert$
means nothing in absolute terms. Against the seed floor — the same clause re-sampled
under different noise — it becomes

$$S=\frac{\lVert a^{(q_5)}-a^{(q_1)}\rVert}{\operatorname{mean}_{s\neq s'}\lVert a^{(s)}-a^{(s')}\rVert}$$

and $S \approx 1$ means the clause did nothing the sampler was not already doing.
Read this before anything else; the rest of the probe is conditional on it.

**2. Is the scale ordered, or a switch?** Five levels are only a scale if the model
places them in order. Per frame, take the axis the two poles define,
$u = a^{(q_5)}-a^{(q_1)}$, and project each level onto it,

$$\pi(q)=\frac{\left\langle a^{(q)}-\bar{a},\,u\right\rangle}{\lVert u \rVert},
\qquad \bar{a}=\frac{1}{5}\sum_{q} a^{(q)},
\qquad \left\langle x,y\right\rangle=\frac{1}{TD}\sum_{t,d}x_{t,d}\,y_{t,d}$$

so that $\pi(5)-\pi(1)=\lVert u \rVert$ exactly and the three interior levels have to
fall somewhere between. A monotone staircase means "$N$ of 5" is read as an ordered
quantity; two clumps at the ends mean the model binarized it; interior levels off the
axis mean they are their own thing rather than a point on a scale.

**3. Is it obeyed?** Moving the chunk is not the same as moving it somewhere sensible.
Quality is annotated per segment, so a frame has a *true* quality and can be asked a
different one. With

$$\Delta\mathrm{MSE}(c)=\frac{1}{TD}\sum_{t,d}\left(a^{(\mathrm{none})}_{t,d}-a^{\star}_{t,d}\right)^{2}
-\frac{1}{TD}\sum_{t,d}\left(a^{(c)}_{t,d}-a^{\star}_{t,d}\right)^{2}$$

the obedience matrix averages $\Delta\mathrm{MSE}(q_{\text{asked}})$ over the frames
whose true quality is $q_{\text{true}}$. A model that understands the clause has the
diagonal winning each column: told what actually happened, it predicts what actually
happened. A model that reads the clause as "try harder" has row $q_5$ winning every
column, including the columns where the demonstration was mediocre.

**4. Which clause carries it?** The old poles bundled both — quality 5 with no mistake
against quality 1 with a mistake — so a difference could not be attributed. The $2\times2$
separates them: the quality effect $\lVert a^{(q_5,m)}-a^{(q_1,m)}\rVert$ averaged over
the mistake flag, the mistake effect $\lVert a^{(q,\mathrm{T})}-a^{(q,\mathrm{F})}\rVert$
averaged over the pole, and the interaction $\lVert d_{\mathrm{T}}-d_{\mathrm{F}}\rVert$
with $d_m=a^{(q_5,m)}-a^{(q_1,m)}$ — all three against the same seed floor.

Two facts about the labels that decide what these figures can show. Quality is per
*segment*, not per episode, so it varies within an episode and a frame's true level is
well defined. And the levels are not uniform in training: the annotated corpus runs
6.6% at quality 1, 9.4% at 2, 25.0% at 3, 24.9% at 4 and 34.1% at 5 (165,740 frames,
2026-08-02 merge), so asking for quality 1 is a rare prompt and a weak response at the
low end is as much a data statement as a model one. Which levels the *held-out* frames
actually cover is in the provenance box, and a level missing there has no column in the
obedience matrix.

Cost is ``n_frames x (9 + n_seeds - 1)`` forwards.

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

from lerobot.probes.manifest import Panel, write_index
from lerobot.probes.utils import (
    frame_metadata_lookup,
    joint_names_for_dim,
    makedirs,
    panel_caption as _caption,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)

_QUALITY_LEVELS = (1, 2, 3, 4, 5)
_MISTAKE_POLES = (1, 5)  # where the mistake sentence is flipped, giving the 2x2

# The dose axis gets a sequential ramp so a monotone response is visible as a colour
# order; everything that is not a quality level stays off that ramp.
_QUALITY_COLORS = {q: plt.get_cmap("viridis")(i / 4) for i, q in enumerate(_QUALITY_LEVELS)}
_CONDITION_STYLE = {
    "none": ("#E63946", "--"),
    "gt": ("#B5179E", ":"),
    **{f"q{q}": (_QUALITY_COLORS[q], "-") for q in _QUALITY_LEVELS},
    **{f"q{q}m": (_QUALITY_COLORS[q], "-.") for q in _MISTAKE_POLES},
}

_STEERED = {f"q{q}": {"quality": q, "mistake": False} for q in _QUALITY_LEVELS}
_STEERED |= {f"q{q}m": {"quality": q, "mistake": True} for q in _MISTAKE_POLES}

# The clause the rollout prompt carries, and the one the old probe called "bad".
_ROLLOUT = "q5"
_OPPOSITE = "q1m"


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


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


def _sem(rows: list[dict], key: str) -> float:
    values = _column(rows, key)
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _quality_projection(acts: dict[str, torch.Tensor]) -> tuple[dict[int, float], float]:
    r"""Where each level sits on the axis its own two poles define, and Kendall's $\tau$.

    $\pi(q) = \langle a^{(q)} - \bar{a}, u\rangle / \lVert u \rVert$ with $u = a^{(q_5)} -
    a^{(q_1)}$, so $\pi(5) - \pi(1) = \lVert u \rVert$ and the interior levels are placed
    between the poles in the same units as every other distance here. $\tau$ over the five
    $(q, \pi(q))$ pairs is $+1$ for a monotone staircase and $0$ for levels in no order at
    all — the difference between a read scale and five unrelated strings.
    """
    stack = torch.stack([acts[f"q{q}"] for q in _QUALITY_LEVELS])
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


def _provenance(rows: list[dict], dataset, cfg, conditions: list[str], n_seeds: int) -> dict:
    """Which frames of which dataset the figures average, and what one forward was.

    Derived from the measured rows rather than from config, so it describes the frames
    that actually produced numbers after stride snapping and the episode budget dropped
    whatever they dropped. The label mix is the part that decides what the obedience
    matrix can say: a quality level with no frames in this split has no column, and one
    with three frames has a column that is three frames wide.
    """
    p = cfg.probe_parameters
    episodes = sorted({row["episode_idx"] for row in rows})
    labelled = [row for row in rows if row["gt_quality"] is not None]
    mix = {q: sum(row["gt_quality"] == q for row in labelled) for q in _QUALITY_LEVELS}
    missing = [q for q in _QUALITY_LEVELS if not mix[q]]
    flagged = sum(row["gt_mistake"] for row in rows)
    forwards_per_frame = len(conditions) + n_seeds - 1

    label_line = (
        ", ".join(f"quality {q}: {mix[q]} frames" for q in _QUALITY_LEVELS if mix[q])
        + f"; {flagged} of {len(rows)} frames carry the mistake flag"
        if labelled
        else "no quality / mistake labels on this split — ``gt`` and the obedience matrix are unavailable"
    )
    if missing and labelled:
        label_line += (
            "; no frames at quality "
            + "/".join(str(q) for q in missing)
            + f", so {'that level is' if len(missing) == 1 else 'those levels are'} asked for "
            "but never true here — it gets a row in the obedience matrix and no column"
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
                "``The quality is $N$ of 5.``  and  ``The robot made a mistake.`` / "
                "``The robot made no mistakes.``, appended to the action prompt by "
                "``_build_robot_text``. ``none`` omits both sentences, which is what "
                "``metadata_dropout`` shows the model on ~15% of training samples.",
            ],
            [
                "Where the labels come from",
                "quality per *segment* from ``meta/episode_metadata.parquet`` "
                "(``from_index``/``to_index``), mistake per 4 s window from "
                "``meta/mistakes.parquet`` — the spans "
                "``ReplayBuffer.materialize_metadata`` broadcasts at training, read here "
                "by ``frame_metadata_lookup``",
            ],
            ["Label mix of the sampled frames", label_line],
            [
                "Demonstration",
                f"$a^{{\\star}}$ is the recorded action at the sampled frame and the "
                f"{int(cfg.policy.chunk_size) - 1} that follow it, normalized like the prediction",
            ],
        ],
        "sampling": (
            "Frames evenly spaced across each episode, snapped onto the image/depth stride "
            f"grid; episodes drawn by a seeded subset when the budget is smaller than the "
            f"split (seed {int(p.random_seed)}). Every distance is a within-frame difference "
            "between two conditions on the same observation, so uneven sampling cannot "
            "manufacture an effect — but the obedience matrix *does* compare across frames, "
            "and its columns inherit whatever the label mix above is. A chunk that would run "
            "past the end of its episode is repeat-padded with the last recorded action, "
            f"which makes the demonstration partly constant within {int(cfg.policy.chunk_size)} "
            "frames of the end and shrinks what any clause can improve there."
        ),
        "regime": (
            "one frame per forward, batch size 1; "
            f"{int(getattr(cfg.policy, 'num_inference_steps', 0))} flow denoising steps, seed 0 "
            "in every condition so the clause is the only difference; the batch carries no "
            "action target, so training-time prompt dropout is not armed"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — does it clear the floor, and is the scale ordered?
# ──────────────────────────────────────────────────────────────────────────────

def _render_floor(rows: list[dict], summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(17, 6.4))
    grid = fig.add_gridspec(1, 3, wspace=0.26, left=0.05, right=0.985, top=0.78, bottom=0.32)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    contrasts = [
        ("quality\nq5 vs q1", "quality_range_rmse"),
        ("mistake\nsentence", "mistake_flip_rmse"),
        ("both poles\n(old good/bad)", "pole_range_rmse"),
        ("flow seed\n(floor)", "seed_floor_mean"),
    ]
    axes[0].boxplot([_column(rows, key) for _label, key in contrasts],
                    tick_labels=[label for label, _key in contrasts])
    axes[0].set_ylabel(r"$\|a^{(c)} - a^{(c')}\|$  (normalized actions)")
    axes[0].set_title(
        "Does the clause move the chunk\nmore than noise does?\n"
        f"median quality separation $S$ = {summary['separation_median']:.2f}x   ·   "
        f"mistake {summary['mistake_separation_median']:.2f}x"
    )
    _caption(axes[0], [
        r"One frame gives one point to each box. The first three are clause contrasts on that frame,",
        r"the fourth is the same clause ($q_5$, no mistake) re-drawn under " + str(summary["n_seeds"]) + r" flow seeds.",
        r"None of the first three has a scale of its own — only the ratio $S$ to the fourth does, and",
        r"$S \approx 1$ says the clause did nothing the sampler was not already doing anyway.",
        r"'both poles' is the contrast the previous version of this probe reported alone: it moves",
        r"two clauses at once, so a tall box there attributes to neither.",
    ])

    x = np.arange(len(_QUALITY_LEVELS))
    means = [_mean(rows, f"q{q}_rmse") for q in _QUALITY_LEVELS]
    errors = [_sem(rows, f"q{q}_rmse") for q in _QUALITY_LEVELS]
    axes[1].bar(x, means, yerr=errors, capsize=3,
                color=[_QUALITY_COLORS[q] for q in _QUALITY_LEVELS])
    axes[1].axhline(summary["seed_floor_mean"], color="#E63946", linestyle="--", linewidth=1.2,
                    label="flow-seed floor")
    axes[1].set_xticks(x, [f"quality {q}" for q in _QUALITY_LEVELS])
    axes[1].set_ylabel(r"$\|a^{(q)} - a^{(none)}\|$")
    axes[1].set_title("Dose: how far each level moves\nthe chunk off the no-clause prediction")
    axes[1].legend(fontsize=8)
    _caption(axes[1], [
        r"Mistake sentence held at 'no mistakes', so this axis is the number alone. Bar = mean over",
        r"frames, whisker = standard error of that mean. Distance from $a^{(none)}$, never from each",
        r"other: a level can sit far from no-clause and still be indistinguishable from its neighbour.",
        r"A U shape (both poles high, middle low) is the signature of $q_3$ being the clause the model",
        r"treats as least informative, not of $q_3$ being ignored — the right panel separates those.",
    ])

    for row in rows:
        axes[2].plot(_QUALITY_LEVELS, [row[f"proj_q{q}"] for q in _QUALITY_LEVELS],
                     color="#457B9D", alpha=0.12, linewidth=0.8)
    mean_curve = [_mean(rows, f"proj_q{q}") for q in _QUALITY_LEVELS]
    sem_curve = [_sem(rows, f"proj_q{q}") for q in _QUALITY_LEVELS]
    axes[2].errorbar(_QUALITY_LEVELS, mean_curve, yerr=sem_curve, color="#1D3557",
                     linewidth=2.2, marker="o", capsize=3, label="mean over frames")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(list(_QUALITY_LEVELS))
    axes[2].set_xlabel("quality asked for")
    axes[2].set_ylabel(r"$\pi(q)$ — position on the $q_1 \rightarrow q_5$ axis")
    axes[2].set_title(
        "Is it a scale or a switch?\n"
        f"monotone on {summary['monotone_fraction']:.0%} of frames  ·  "
        rf"mean $\tau$ = {summary['kendall_tau_mean']:+.2f}"
    )
    axes[2].legend(fontsize=8)
    _caption(axes[2], [
        r"Per frame: $u = a^{(q_5)} - a^{(q_1)}$, and each level is projected onto it,",
        r"$\pi(q) = \langle a^{(q)} - \bar{a},\, u \rangle / \|u\|$, centred on the five-level mean $\bar{a}$.",
        r"By construction $\pi(5) - \pi(1) = \|u\|$, so a frame's line spans its own steering range and",
        r"the interior levels have to land somewhere inside it. Evenly spaced rising line = an ordered",
        r"scale. Flat middle with the ends split = binarized. Zig-zag = the five strings are unrelated",
        r"prompts that happen to differ. $\tau$ is Kendall's over the five $(q, \pi(q))$ pairs, $+1$ = sorted.",
    ])

    fig.suptitle(
        f"Metadata steering — floor, dose and ordering (n={summary['n_frames']} frames, "
        f"{summary['n_episodes']} held-out episodes)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — is the clause obeyed?
# ──────────────────────────────────────────────────────────────────────────────

def _render_obedience(rows: list[dict], summary: dict, output_path: str) -> None:
    labelled = [row for row in rows if row["gt_quality"] is not None]
    columns = [q for q in _QUALITY_LEVELS if any(row["gt_quality"] == q for row in labelled)]

    fig = plt.figure(figsize=(18, 6.6))
    grid = fig.add_gridspec(1, 3, wspace=0.42, left=0.05, right=0.985, top=0.78, bottom=0.32)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    asked = [f"q{q}" for q in _QUALITY_LEVELS] + ["gt"]
    matrix = np.full((len(asked), len(columns)), np.nan)
    for col, true_q in enumerate(columns):
        bucket = [row for row in labelled if row["gt_quality"] == true_q]
        for line, name in enumerate(asked):
            matrix[line, col] = _mean(bucket, f"{name}_gt_mse_improvement")

    scale = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    image = axes[0].imshow(1e3 * matrix, cmap="RdBu", vmin=-1e3 * scale, vmax=1e3 * scale,
                           aspect="auto")
    axes[0].set_xticks(range(len(columns)),
                       [f"{q}\n(n={sum(r['gt_quality'] == q for r in labelled)})" for q in columns])
    axes[0].set_yticks(range(len(asked)),
                       [f"asked {q}" for q in _QUALITY_LEVELS] + ["gt (+mistake)"])
    axes[0].set_xlabel("quality the segment actually is")
    for col, true_q in enumerate(columns):
        for line, name in enumerate(asked):
            if not np.isfinite(matrix[line, col]):
                continue
            diagonal = name == f"q{true_q}"
            axes[0].text(col, line, f"{1e3 * matrix[line, col]:+.2f}", ha="center", va="center",
                         fontsize=8, fontweight="bold" if diagonal else "normal")
            if diagonal:
                axes[0].add_patch(plt.Rectangle((col - 0.5, line - 0.5), 1, 1, fill=False,
                                                edgecolor="black", linewidth=2.0))
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03).set_label(
        r"$\Delta$MSE $\times 10^{3}$"
    )
    axes[0].set_title("Obedience: does asking the truth\nbeat asking for quality 5?")
    # Two-line column ticks plus an axis label sit under this panel, so its caption
    # starts lower than the default.
    _caption(axes[0], y=-0.30, lines=[
        r"Cell = mean $\Delta\mathrm{MSE}$ over the frames in that column, $\times 10^{3}$: how much squared",
        r"error against the demonstration the clause removes relative to no clause at all. Blue helped,",
        r"red hurt, white did nothing. Boxed = the honest cell, where the asked quality is the true one.",
        r"Understood clause: the boxed cell is the best of its column. 'Try harder' clause: the $q_5$ row",
        r"wins everywhere, including columns where the demonstration was mediocre. The bottom row adds",
        r"the true mistake flag on top of the true quality, so it differs from the boxed cell only on",
        r"flagged frames. Columns are independent samples of frames — read down them, not across.",
    ])

    width = 0.26
    x = np.arange(len(columns))
    buckets = {q: [row for row in labelled if row["gt_quality"] == q] for q in columns}
    # `None` means "ask for whatever this column's true quality is" — the diagonal.
    series = [("asked = truth", None, "#2A9D8F"), ("asked = 5 (rollout)", 5, "#457B9D"),
              ("asked = 1", 1, "#F4A261")]
    for offset, (label, asked_q, color) in enumerate(series):
        keys = [f"q{asked_q or q}_gt_mse_improvement" for q in columns]
        axes[1].bar(
            x + (offset - 1) * width,
            [_mean(buckets[q], key) for q, key in zip(columns, keys, strict=True)],
            width=width,
            yerr=[_sem(buckets[q], key) for q, key in zip(columns, keys, strict=True)],
            capsize=2, color=color, label=label,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, [f"true quality {q}" for q in columns])
    axes[1].set_ylabel(r"$\Delta$MSE vs no clause (positive = helped)")
    axes[1].set_title("The same three rows, with their\nstandard errors")
    axes[1].legend(fontsize=8)
    _caption(axes[1], [
        r"Three rows of the matrix drawn with the uncertainty the colours hide: the diagonal, the clause",
        r"every rollout issues, and the opposite pole. Whisker = standard error of the column mean, so",
        r"overlapping whiskers mean the matrix's colour difference is not resolved at this sample size.",
        r"The comparison that matters is green vs blue inside a column: green above blue on the low-quality",
        r"columns is the clause being obeyed; blue on top everywhere is quality 5 acting as a global prior.",
    ])

    order = [f"q{q}" for q in _QUALITY_LEVELS] + [f"q{q}m" for q in _MISTAKE_POLES] + ["gt"]
    order = [name for name in order if f"{name}_gt_mse_improvement" in rows[0]]
    values = [_mean(rows, f"{name}_gt_mse_improvement") for name in order]
    errors = [_sem(rows, f"{name}_gt_mse_improvement") for name in order]
    colors = [_CONDITION_STYLE[name][0] for name in order]
    # q1m/q5m carry their pole's colour, so the mistake sentence has to be the hatch.
    axes[2].bar(np.arange(len(order)), values, yerr=errors, capsize=3, color=colors,
                hatch=["//" if name.endswith("m") else "" for name in order],
                edgecolor="white")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(np.arange(len(order)), order, rotation=45, ha="right")
    axes[2].set_ylabel(r"$\Delta$MSE vs no clause")
    axes[2].set_title("Usefulness of every condition,\nover all frames")
    _caption(axes[2], [
        r"$\Delta\mathrm{MSE}(c) = \frac{1}{TD}\sum (a^{(none)} - a^{\star})^2 - \frac{1}{TD}\sum (a^{(c)} - a^{\star})^2$,",
        r"averaged over every sampled frame regardless of its labels. Positive means the clause moved the",
        r"chunk toward the demonstration; below zero it moved it away, which is worse than not reading it.",
        r"gt should be the tallest bar here — conditioning on what actually happened is the easiest",
        r"version of the prediction problem. If q5 matches or beats gt, the model is answering a",
        r"prior rather than the clause. Squared normalized action units, not comparable to the RMSE panels.",
    ])

    fig.suptitle(
        f"Metadata steering — is the clause obeyed? ({len(labelled)} labelled frames)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — which of the two clauses carries the effect
# ──────────────────────────────────────────────────────────────────────────────

def _render_factorial(rows: list[dict], summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(17, 6.4))
    grid = fig.add_gridspec(1, 3, wspace=0.28, left=0.05, right=0.985, top=0.78, bottom=0.32)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    for mistake, marker, color in ((False, "o", "#2A9D8F"), (True, "s", "#E63946")):
        keys = [f"q{q}{'m' if mistake else ''}_gt_mse_improvement" for q in _MISTAKE_POLES]
        axes[0].errorbar(
            _MISTAKE_POLES, [_mean(rows, key) for key in keys],
            yerr=[_sem(rows, key) for key in keys],
            marker=marker, color=color, linewidth=2.0, capsize=3,
            label="mistake sentence on" if mistake else "no mistakes",
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(list(_MISTAKE_POLES), [f"quality {q}" for q in _MISTAKE_POLES])
    axes[0].set_ylabel(r"$\Delta$MSE vs no clause")
    axes[0].set_title("The 2x2, as an interaction plot")
    axes[0].legend(fontsize=8)
    _caption(axes[0], [
        r"Four cells: quality $\in \{1, 5\}$ crossed with the mistake sentence on/off. Parallel lines mean",
        r"the two clauses act independently — whatever the mistake sentence does, it does the same at",
        r"either pole. Crossing lines mean the model reads them jointly (a mistake at quality 5 is a",
        r"different situation than a mistake at quality 1), which is what the annotation actually means.",
        r"A vertical gap between the lines is the mistake sentence's effect; the slope is quality's.",
    ])

    effects = [
        ("quality\n(number)", "quality_effect_rmse", "#457B9D"),
        ("mistake\n(sentence)", "mistake_flip_rmse", "#F4A261"),
        ("interaction", "interaction_rmse", "#9B5DE5"),
    ]
    means = [_mean(rows, key) for _label, key, _color in effects]
    errors = [_sem(rows, key) for _label, key, _color in effects]
    axes[1].bar(np.arange(len(effects)), means, yerr=errors, capsize=3,
                color=[color for _label, _key, color in effects])
    axes[1].axhline(summary["seed_floor_mean"], color="#E63946", linestyle="--", linewidth=1.2,
                    label="flow-seed floor")
    axes[1].set_xticks(np.arange(len(effects)), [label for label, _key, _color in effects])
    axes[1].set_ylabel("distance in normalized action space")
    axes[1].set_title("Which clause carries the movement?")
    axes[1].legend(fontsize=8)
    _caption(axes[1], [
        r"Main effects as distances, each averaged over the other factor, with $d_m = a^{(q_5, m)} - a^{(q_1, m)}$:",
        r"quality $= \frac{1}{2}(\|d_F\| + \|d_T\|)$,   mistake $= \frac{1}{2}\sum_q \|a^{(q, T)} - a^{(q, F)}\|$,",
        r"interaction $= \|d_T - d_F\|$, the part of the quality response that depends on the mistake flag.",
        r"Everything is a distance, so nothing here can be negative and none of it says the movement helped —",
        r"that is figure 2's job. Bars below the dashed floor are indistinguishable from re-rolling the noise.",
    ])

    clean = [row["mistake_flip_rmse"] for row in rows if not row["gt_mistake"]]
    flagged = [row["mistake_flip_rmse"] for row in rows if row["gt_mistake"]]
    groups = [values for values in (clean, flagged) if values]
    labels = [
        f"{name} (n={len(values)})"
        for name, values in (("no mistake", clean), ("GT mistake", flagged)) if values
    ]
    if groups:
        axes[2].boxplot(groups, tick_labels=labels)
    axes[2].axhline(summary["seed_floor_mean"], color="#E63946", linestyle="--", linewidth=1.2,
                    label="flow-seed floor")
    axes[2].set_ylabel(r"$\|a^{(q, T)} - a^{(q, F)}\|$")
    axes[2].set_title("Does the mistake sentence land harder\non frames that really are mistakes?")
    axes[2].legend(fontsize=8)
    _caption(axes[2], [
        r"The mistake-sentence effect from the middle panel, split by whether the frame sits inside a",
        r"flagged 4 s window in meta/mistakes.parquet. A model that reads the sentence in context",
        r"should respond more where the observation is consistent with it; equal boxes mean the sentence",
        r"is a constant offset applied regardless of what the frame shows. Flagged frames are rare — the",
        r"counts are in the tick labels, and a handful of them supports no conclusion either way.",
    ])

    fig.suptitle(
        f"Metadata steering — quality number vs mistake sentence (n={summary['n_frames']})",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


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

**Left — the floor.** Three clause contrasts against the one contrast that is not a
clause at all: the same prompt re-drawn under different flow seeds. Only the ratio
$S$ between them means anything, and $S \approx 1$ means the clause is decorative. The
third box, "both poles", is what the previous version of this probe reported on its
own — it moves the quality number *and* the mistake sentence together, so it cannot
attribute an effect to either.

**Middle — the dose.** $\lVert a^{(q)} - a^{(\mathrm{none})}\rVert$ for each of the five
levels, mistake sentence held off, against the same floor. This is displacement from
the no-clause prediction, not from the neighbouring level: five tall bars are
consistent with five levels the model cannot tell apart.

**Right — the ordering.** Each level projected onto the axis its own poles define,
$\pi(q)=\langle a^{(q)}-\bar a,\,u\rangle/\lVert u\rVert$ with $u=a^{(q_5)}-a^{(q_1)}$.
The endpoints are pinned by construction ($\pi(5)-\pi(1)=\lVert u\rVert$); what is
being measured is where 2, 3 and 4 land. A rising staircase is a scale the model
reads as ordered. A flat middle is a switch wearing five labels."""

_OBEDIENCE_HOW = r"""Quality is annotated per segment, so every held-out frame has a
true level and can be asked a different one. That is what makes obedience measurable
here and not in the previous version, which only ever split on the mistake flag (five
windows in this split).

**Left — the matrix.** Rows are what the prompt asked for, columns what the segment
actually was, cells the mean $\Delta\mathrm{MSE}$ against the demonstration relative to
no clause, in units of $10^{-3}$. Blue helped, red hurt. The boxed diagonal is the
honest cell. **Read down a column, not across a row**: each column is its own sample of
frames, and their sizes are in the tick labels. A clause that is understood puts the
boxed cell at the top of its column; a clause read as "try harder" puts the $q_5$ row on
top everywhere, including where the demonstration was mediocre.

**Middle — the same rows with error bars.** Diagonal, rollout clause ($q_5$) and opposite
pole ($q_1$) per column, with the standard error the heat map's colours hide. Overlapping
whiskers mean the colour difference is not resolved at this sample size.

**Right — usefulness overall.** Every condition against ``none`` over all frames.
``gt`` should be the tallest: conditioning on what actually happened is the easiest
version of the prediction problem. ``q5`` matching ``gt`` means the model is answering a
prior instead of the clause, and any bar below zero is a clause that is read and points
the wrong way."""

_FACTORIAL_HOW = r"""The old ``good``/``bad`` contrast moved both clauses at once. This
page separates them, with $d_m = a^{(q_5,m)} - a^{(q_1,m)}$ the quality response at a
fixed mistake flag $m$.

**Left — interaction plot.** The four cells of the $2\times2$ as usefulness. Parallel
lines mean the two clauses act independently; crossing lines mean the model reads them
jointly, which is what the annotation means (a mistake inside a quality-5 segment is a
different situation from a mistake inside a quality-1 one).

**Middle — main effects as distances.** Quality
$=\frac{1}{2}(\lVert d_F\rVert+\lVert d_T\rVert)$, mistake
$=\frac{1}{2}\sum_q \lVert a^{(q,T)}-a^{(q,F)}\rVert$, interaction
$=\lVert d_T-d_F\rVert$, all against the flow-seed floor. Distances only — a tall bar
says the clause was read, never that reading it helped.

**Right — is the mistake sentence contextual?** The same mistake effect split by whether
the frame is inside a flagged window. Responding more where the observation is
consistent with the sentence is the difference between reading it and applying it as a
constant offset. Flagged frames are rare in a held-out split; the counts are on the
axis, and a handful supports no conclusion."""


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
    n_seeds = max(int(getattr(p, "metadata_steering_n_seeds", 3)), 2)
    n_frames = int(getattr(p, "metadata_steering_n_frames", None) or p.n_frames_per_episode)

    gt_metadata = frame_metadata_lookup(dataset)
    if not gt_metadata:
        logging.warning(
            "[metadata_steering] no meta/episode_metadata.parquet + mistakes.parquet on "
            f"{dataset.root} — no ``gt`` condition and no obedience matrix; the steering "
            "range and the factorial still run."
        )

    adapter._set_probe_cuda_graph_enabled(False)  # prompt changes per condition; keep eager
    samples = sample_episodes_evenly(
        dataset, n_frames, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    logging.info(
        f"[metadata_steering] {len(samples)} frames x ({len(_STEERED) + 1} clauses + "
        f"{n_seeds - 1} reseeds + gt) = ~{len(samples) * (len(_STEERED) + n_seeds + 1)} forward passes"
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    try:
        for episode_idx, frame_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size, metadata=None)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])
            labels = gt_metadata.get(global_idx)

            def predict(metadata: dict | None, seed: int = 0) -> torch.Tensor:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                return adapter.predict_action_chunk(
                    frame["obs"], frame["task"], state=frame["state"],
                    subtask=frame["subtask"], metadata=metadata, generator=generator,
                )[1]

            acts = {"none": predict(None)}
            acts |= {name: predict(metadata) for name, metadata in _STEERED.items()}
            if labels is not None:
                acts["gt"] = predict(labels)
            # Seed floor: the rollout clause re-drawn under different noise. Without it
            # every distance above is a number with no scale.
            floor_draws = [acts[_ROLLOUT]] + [predict(_STEERED[_ROLLOUT], seed) for seed in range(1, n_seeds)]
            floor_mean, floor_max = _pairwise_rmse(floor_draws)

            projection, tau = _quality_projection(acts)
            base = acts["none"]
            gt_mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
            quality_effect = [_rmse(acts[f"q5{m}"], acts[f"q1{m}"]) for m in ("", "m")]
            mistake_effect = [_rmse(acts[f"q{q}m"], acts[f"q{q}"]) for q in _MISTAKE_POLES]

            row = {
                "global_idx": int(global_idx),
                "episode_idx": int(episode_idx),
                "frame_idx": int(frame_idx),
                "gt_quality": None if labels is None else int(labels["quality"]),
                "gt_mistake": bool(labels["mistake"]) if labels is not None else False,
                "seed_floor_mean": floor_mean,
                "seed_floor_max": floor_max,
                "quality_range_rmse": _rmse(acts["q5"], acts["q1"]),
                "pole_range_rmse": _rmse(acts[_ROLLOUT], acts[_OPPOSITE]),
                "quality_effect_rmse": float(np.mean(quality_effect)),
                "mistake_flip_rmse": float(np.mean(mistake_effect)),
                "interaction_rmse": _rmse(acts["q5m"] - acts["q1m"], acts["q5"] - acts["q1"]),
                "kendall_tau": tau,
                "monotone": abs(tau) == 1.0,
                **{f"proj_q{q}": value for q, value in projection.items()},
            }
            row["separation"] = row["quality_range_rmse"] / max(floor_mean, 1e-9)
            row["mistake_separation"] = row["mistake_flip_rmse"] / max(floor_mean, 1e-9)
            for name, act in acts.items():
                row[f"{name}_gt_mse"] = gt_mse[name]
                if name != "none":
                    row[f"{name}_rmse"] = _rmse(act, base)
                    row[f"{name}_maxabs"] = float((act - base).abs().max())
                    row[f"{name}_gt_mse_improvement"] = gt_mse["none"] - gt_mse[name]
            rows.append(row)
            diagnostics.append({"gt": gt_norm, "acts": acts, "row": row})
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[metadata_steering] no frames produced measurements.")
        return

    conditions = ["none"] + list(_STEERED) + (["gt"] if "gt_gt_mse" in rows[0] else [])
    labelled = [row for row in rows if row["gt_quality"] is not None]
    flagged = [row for row in rows if row["gt_mistake"]]

    summary = {
        "n_frames": len(rows),
        "n_episodes": len({row["episode_idx"] for row in rows}),
        "n_frames_labelled": len(labelled),
        "n_frames_gt_mistake": len(flagged),
        "n_seeds": n_seeds,
        "conditions": conditions,
        "seed_floor_mean": _mean(rows, "seed_floor_mean"),
        "quality_range_rmse": _mean(rows, "quality_range_rmse"),
        "pole_range_rmse": _mean(rows, "pole_range_rmse"),
        "quality_effect_rmse": _mean(rows, "quality_effect_rmse"),
        "mistake_flip_rmse": _mean(rows, "mistake_flip_rmse"),
        "interaction_rmse": _mean(rows, "interaction_rmse"),
        "separation_mean": _mean(rows, "separation"),
        "separation_median": float(np.median(_column(rows, "separation"))),
        "mistake_separation_median": float(np.median(_column(rows, "mistake_separation"))),
        "kendall_tau_mean": _mean(rows, "kendall_tau"),
        "monotone_fraction": float(np.mean([row["monotone"] for row in rows])),
        "quality_mix": {
            str(q): sum(row["gt_quality"] == q for row in labelled) for q in _QUALITY_LEVELS
        },
        "verdict_note": (
            "separation ~1 => the quality clause moves the chunk no more than flow noise does, "
            "and the obedience matrix is then ranking noise. separation >> 1 with the q5 row "
            "winning every column of the matrix => the clause is read as a global prior rather "
            "than as a description of the frame."
        ),
    }
    for name in conditions:
        summary[f"{name}_gt_mse"] = _mean(rows, f"{name}_gt_mse")
        if name != "none":
            summary[f"{name}_rmse"] = _mean(rows, f"{name}_rmse")
            summary[f"{name}_gt_mse_improvement"] = _mean(rows, f"{name}_gt_mse_improvement")
            if flagged:
                summary[f"{name}_gt_mse_improvement_on_gt_mistake"] = _mean(
                    flagged, f"{name}_gt_mse_improvement"
                )
    summary["data"] = _provenance(rows, dataset, cfg, conditions, n_seeds)

    with open(os.path.join(output_dir, "metadata_steering.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    _render_floor(rows, summary, os.path.join(output_dir, "steering_floor.png"))
    _render_factorial(rows, summary, os.path.join(output_dir, "factorial.png"))
    if labelled:
        _render_obedience(rows, summary, os.path.join(output_dir, "obedience.png"))

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
                "mistake sentence on, ``none`` is the red dashed origin and ``gt`` the frame's "
                "own labels. Ranked by steering range in units of the frame's own seed floor, "
                "so this is where the clause did the most it ever does: if the ramp is not "
                "ordered here, it is ordered nowhere.",
            )
        )

    panels = [
        Panel("steering_floor.png",
              "Floor, dose and ordering — is the clause read, and is it read as a scale?",
              how=_FLOOR_HOW, primary=True, refs=["subtask_sweep"]),
        Panel("obedience.png",
              "Asked quality against true quality — is the clause obeyed or is it a prior?",
              how=_OBEDIENCE_HOW, primary=True),
        Panel("factorial.png",
              "The 2x2 — does the number or the mistake sentence carry the effect?",
              how=_FACTORIAL_HOW),
        *examples,
    ]
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Metadata Steering",
        group="Steering",
        claim="Does the quality / mistake clause reach the actions, is it ordered, and is it obeyed?",
        summary=summary,
        see_also=["subtask_sweep", "mem_history_influence", "objective"],
        # Each readout is titled onto its own figure next to the null it has to beat —
        # the seed floor, the uniform column, zero improvement. The values stay in
        # metadata_steering.json.
        metrics=[],
        panels=panels,
        extra={"provenance": summary["data"]},
    )

    logging.info(
        f"[metadata_steering] n={len(rows)} ({len(flagged)} GT-mistake)  "
        f"quality range={summary['quality_range_rmse']:.4f} "
        f"({summary['separation_median']:.2f}x floor)  "
        f"mistake={summary['mistake_flip_rmse']:.4f}  "
        f"tau={summary['kendall_tau_mean']:+.2f}  "
        f"gt improvement={summary.get('gt_gt_mse_improvement', float('nan')):+.5f}  "
        f"q5 improvement={summary['q5_gt_mse_improvement']:+.5f}"
    )
