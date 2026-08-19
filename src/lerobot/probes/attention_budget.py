r"""Where the action tokens spend their cross-attention, and how that shifts over training.

Every action query runs one softmax over the whole encoder key axis: both camera
token blocks, the point-map depth columns, each prompt clause, and the chat
scaffolding. That row sums to 1, so partitioning the columns gives a genuine
attention *budget*: does depth take a bigger share when the gripper is close to
something, does the top camera take over during transport, does the subtask clause get
read at all, and does the split move as the policy trains.

## What is measured

For one frame, one layer and one action query, the cross-attention row is a
probability distribution over every encoder column. Partitioning those columns by
what they *are* and summing within each part gives $m_S(t)$: the fraction of that
row landing on segment $S$. Averaged over heads and over the action chunk, that is
the number every panel is built from. It is a **share**, not an activation — one
frame's shares sum to 1 across segments by construction.

## The one thing that will mislead you

$m_S$ is dominated by how many columns $S$ brought. The two cameras bring 196 encoder
columns each and the depth point map 192, against 8-20 for a prompt clause, out of 697
attendable columns in the current MolmoAct2 layout. At a layer that attends near-uniformly
the widest segment therefore takes the largest share *by arithmetic*, and at layer 0 of a
trained run that is measurably all that is happening: every segment sits within $\pm 0.3$
doublings of its share of the columns, normalized row entropy is $0.976$ of the uniform
bound, and the composition is $0.055$ in total variation from the column histogram. Read
as preference, that panel says "the model attends to pixels and depth". What it says is
"pixels and depth brought the most columns".

So every level is reported twice, and the pair is what carries the content:

* $m_S$ — what the model's attention compute is *spent* on. Sums to 1 over segments.
* $m_S/n_S$ — how hard a *column* of $S$ is read, where $n_S$ is its column count. The
  only comparison between two different segments that the counts do not decide in advance.

Neither is the corrected version of the other. A clause read hard but nine tokens long is
cheap; a camera read weakly across 196 patches is expensive.

## How few columns hold it

Dividing by $n_S$ treats a segment's columns as interchangeable, and for a camera they are
not — most patches are background, while all 18 state columns carry a joint. So the spread
*inside* each segment is measured too. Sort $S$'s columns by mass, descending, giving
$s_1 \ge s_2 \ge \dots \ge s_{n_S}$ with $\sum_i s_i = m_S$, and take

$$n_{90} = \min\{k : \textstyle\sum_{i \le k} s_i \ge 0.9\,m_S\}$$

the number of columns carrying 90% of what that segment received — a count, read directly
against $n_S$: "90% of what the top camera gets lands on 12 of its 196 patches". Nothing is
divided out and there is no index to interpret. The running sum it comes from is plotted in
full in ``concentration.png``, where $n_{90}$ is one marked point on the curve.

## Why the training axis is where this gets easy

The cross-segment comparison is the hard one and it is not the interesting one. $n_S$ is
fixed by the prompt layout and the camera geometry, so it is identical at step 400 and step
1600: every count effect is a constant offset on that segment's series, and constants cancel
when a segment is compared with itself over training. Nothing needs normalizing to ask "is
the model learning to lean on state" — which is what ``trajectory.png`` plots, assembled
from every checkpoint already on disk, no GPU and no rerun. What does *not* cancel is the
frame set: measure different scenes and every share moves for reasons unrelated to the
weights, so that is checked and reported in the figure's title rather than assumed.

## Why it is read in logs, and centred

Three properties of a share force the transforms:

1. **A share is only meaningful as a ratio.** The depth clause moving $0.10\%$ to
   $0.20\%$ and the top camera moving $30\%$ to $60\%$ are the same event — the model
   doubled what it spent there — but on a linear axis the first is invisible and the
   second owns the panel. Logs make "doubled" a constant distance at every level.
   Base 2 throughout, so every ratio in this probe is a count of doublings.
2. **Shares cannot move independently.** The row sums to 1, so when image mass rises
   every clause falls *mechanically*, with no change in what the model prefers.
   Segment levels are therefore mutually confounded and a story about one segment's
   level is not safe on its own.
3. **In log space that constraint is an additive offset**, which is why it can be
   removed by a subtraction. Two subtractions are used:

* **centered log-ratio**, the standard statistic for simplex data,
  $\text{clr}(m)_S = \log_2 m_S - \frac{1}{n}\sum_{S'} \log_2 m_{S'}$ over the $n$
  segments that hold mass — i.e. $S$'s share divided by the geometric mean share,
  in logs. A move in clr is a move in the *composition*: $S$ gained on the field, and
  no other segment can manufacture it. Segments below $10^{-6}$ are excluded from the
  $n$: an empty segment sits on the numerical clamp and its log would move the
  geometric mean that every other segment is measured against.
* **pairwise log-ratio** $\log_2(m_A / m_B)$ for a specific hypothesis. It is
  invariant to whatever every other segment does, which makes
  $\log_2(m_\text{depth} / m_\text{wrist})$ the honest test of "depth matters more
  up close".

The primary budget figure intentionally uses ordinary linear percentages: a 100%
stacked composition over frames, a mean-share bar chart with the frame range, and
percentage heatmaps over layers and action-chunk position. The log-ratio statistics
remain in the JSON and training-history analysis, not in the primary PNG.

Two controls ship alongside, because both failure modes are real:

* **Row entropy** per layer per frame. The whole distribution can sharpen or flatten
  across frames or checkpoints, moving every mass without any change in preference.
  Flat entropy + moving composition ⇒ the composition move is real.
* **Column counts per frame**, recorded rather than assumed. ``subtask`` and ``metadata``
  change length frame to frame ("grasp the grey shirt" vs "return to home"), so for those
  two a raw series confounds "attends more" with "clause got longer". Every per-token
  number divides by the median count and ``budget.json`` carries the min and max beside
  it, so a segment whose width moved is visible instead of silently rescaling its row.

The primary PNG has fixed dimensions and contains no off-axis captions or filmstrip.
That keeps the artifact cheap and safe to render even when a segment has tiny mass.

## Two things this probe cannot tell you

1. **Mass is not contribution.** A segment can be attended and its values ignored.
   ``--probe_parameters.budget_fd_sensitivity`` adds the causal complement: a
   finite-difference ``||Δactions||`` per frame for depth and wrist RGB, which is a
   contribution series rather than a mass series. Off by default — it costs two
   extra forwards per frame.
2. **Correlation with distance is mediated.** Approaching an object changes task
   phase, gripper state and image content at once, so a depth-mass/distance
   correlation is suggestive, not causal.

Also note the capture is at a single flow timestep (``probe_parameters.timestep``,
last denoise step), so this is one point on the flow trajectory.

The clause partition here is the same one ``attention`` uses for its
``action_to_prompt`` figure, and the two are complementary: this probe resolves the
budget over *frames* and over the action chunk, that one resolves it over *heads*
and over the individual decoded tokens.

## What it writes

* ``budget.png`` — four fixed-size, linear percentage views of the budget.
* ``concentration.png`` — the within-segment distributions and $n_{90}$, one row per
  probed layer on a shared order and a shared scale.
* ``trajectory.png`` — every checkpoint of the run against training step. Only appears
  once two exist. Also buildable on its own, against a run that has already trained:
  ``python -m lerobot.probes.attention_budget <run>/validation/step_<n>/attention_budget``.
* ``budget_data.npz`` — per-frame arrays, including the cumulative-share curves on a
  log-spaced rank grid, so a new statistic can be computed later without a GPU rerun.

Registered probe: enable with ``probe_parameters.enable_attention_budget``.
"""

import glob
import json
import logging
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import torch

from lerobot.probes.attention import _group_prompt_indices, _text_blocks_for_action_matrix
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    makedirs,
    panel_caption as _caption,
    probe_frame_inputs,
    probe_image_stride,
    sample_episodes_evenly,
)

_EPS = 1e-12
# Below this a segment holds no mass at all. Its log sits on the clamp and would drag
# the geometric mean that every clr is measured against, shifting all n series at once.
_MASS_FLOOR = 1e-6
# How many segments the compositional-shift panel names; the rest stay grey.
_N_MOVERS = 6

# Stable colours so a segment keeps its identity across panels and across runs.
#
# The compositional-shift panel draws whichever six segments moved most, so *any* pair
# of these can end up crossing on one white background — which makes the requirement
# all-pairs separation, not the usual adjacent-pairs one. This set was searched under
# that constraint over an OKLCH grid (lightness 0.44-0.76, chroma >= 0.10, contrast
# >= 2.6 on white) and every one of its 55 coloured pairs clears both gates: worst
# OKLab dE 8.5 under simulated protanopia/deuteranopia (target 8) and 17.7 under
# normal vision (floor 15). The predecessor palette failed both — img_top vs state at
# 3.2 under deuteranopia, and "depth clause" #fabed4 sat at contrast 1.57, i.e. barely
# distinguishable from the page.
#
# ``residual`` is deliberately outside that set: near-ink, no hue. It is chat
# scaffolding rather than a modality, and reading as "not one of the coloured streams"
# is the point.
_SEGMENT_COLORS = {
    "img_top": "#2d3cc1",
    "img_wrist": "#1ab192",
    "depth": "#7f3e00",
    # Mutually exclusive with ``depth`` — one frame yields one or the other — so the two
    # never share a panel and can share a colour.
    "depth_nullbank": "#7f3e00",
    "depth clause": "#c09910",
    "task": "#840390",
    "subtask": "#2ba5fe",
    "state": "#cb2570",
    "state history": "#f069d4",
    "metadata": "#8b5dec",
    "question": "#017a9e",
    "other prompt": "#5b7c07",
    "residual": "#33332f",
}
_INK = "#0b0b0b"
# Column count varies per frame only for these: the clause text itself changes length.
_VARIABLE_LENGTH_SEGMENTS = ("subtask", "metadata")


def _color(name: str) -> str:
    return _SEGMENT_COLORS.get(name, "#808080")


def _row_labels(ax, ordered: list[str], labels: list[str] | None = None) -> None:
    """Segment names as dark text with a colour chip beside them.

    The chip carries the identity and the text stays legible: a pale hue is a fine
    mark and a bad 7pt glyph, and three of these hues sit under 3:1 against white.
    """
    ax.set_yticks(range(len(ordered)), labels or ordered, fontsize=7)
    for tick in ax.get_yticklabels():
        tick.set_color(_INK)
    ax.scatter([-0.72] * len(ordered), range(len(ordered)), marker="s", s=34,
               c=[_color(n) for n in ordered], clip_on=False, zorder=5)


def _segment_columns(result, encoder_len: int) -> dict[str, list[int]]:
    """Partition every *attendable* encoder column into named segments, residual included.

    A budget that drops unlabelled columns does not sum to 1, and the columns most
    likely to be dropped are exactly the chat scaffolding that attracts attention
    sinks — a large, near-constant share that would silently rescale every trend.
    So unlabelled columns are kept, as ``residual``.

    Padded columns are the one exception: the pad mask removes them from the softmax,
    so their mass is identically zero and the partition sums to 1 without them. Left in,
    they only add a permanently empty ``residual`` row to every segment-indexed panel.
    Camera crops are merged into their parent camera: they are the same physical
    view at a different resolution, and splitting them fragments the series.
    """
    segments: dict[str, list[int]] = {}

    for name, indices in (result.extras.get("image_patch_indices_by_segment") or {}).items():
        parent = str(name).split("_crop")[0]
        segments.setdefault(parent, []).extend(int(i) for i in indices)

    depth_segment = result.extras.get("depth_segment")
    if depth_segment:
        name = "depth_nullbank" if depth_segment.get("is_null_bank") else "depth"
        segments[name] = [int(i) for i in depth_segment["indices"]]

    encoder_valid = None
    if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
        encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)
    text_blocks = _text_blocks_for_action_matrix(result)
    if text_blocks:
        for group, entries in _group_prompt_indices(result, text_blocks, encoder_len, encoder_valid):
            if entries:
                segments[group] = [int(idx) for idx, _label in entries]

    claimed: set[int] = set()
    clean: dict[str, list[int]] = {}
    for name, indices in segments.items():
        unique = sorted({i for i in indices if 0 <= i < encoder_len and i not in claimed})
        if unique:
            clean[name] = unique
            claimed.update(unique)

    residual = sorted(i for i in set(range(encoder_len)) - claimed
                      if encoder_valid is None or bool(encoder_valid[i]))
    if residual:
        clean["residual"] = residual
    return clean


def _frame_budget(attn: torch.Tensor, columns: dict[str, list[int]]):
    """One layer's budget for one frame.

    ``attn`` is ``[H, Q, N]`` after softmax. Returns per-segment mass averaged over
    heads and action queries, the same masses resolved per action query, the row
    entropy normalized to [0, 1], and the per-column mass averaged over heads and
    queries — which sums to the segment mass over any segment's columns, so every
    within-segment number below is exactly consistent with the budget above it.
    """
    per_column = attn.mean(dim=(0, 1)).numpy()  # [N]
    mass: dict[str, float] = {}
    by_query: dict[str, np.ndarray] = {}
    for name, indices in columns.items():
        block = attn.index_select(2, torch.as_tensor(indices, dtype=torch.long)).sum(dim=2)  # [H, Q]
        mass[name] = float(block.mean())
        by_query[name] = block.mean(dim=0).numpy()  # [Q]

    probs = attn.clamp_min(0)
    entropy = -(probs * torch.log(probs.clamp_min(_EPS))).sum(dim=2)  # [H, Q]
    normalized = float(entropy.mean()) / float(np.log(max(attn.shape[2], 2)))
    return mass, by_query, normalized, per_column


def _rank_grid(max_columns: int) -> np.ndarray:
    """Log-spaced column ranks, 1 first and ``max_columns`` last.

    Log-spaced because the interesting part of every within-segment distribution is
    its head: whether a 196-patch camera is really being read as twelve patches is
    settled between rank 1 and rank 30, and a linear grid spends 85% of its points
    past the point where every curve has already flattened.
    """
    return np.unique(np.geomspace(1, max(max_columns, 2), 48).round().astype(int))


def _concentration(per_column: np.ndarray, columns: dict[str, list[int]], ranks: np.ndarray):
    """How few columns hold a segment's mass: the sorted cumulative share, and the 90% count.

    Sort segment $S$'s columns by mass, descending, giving $s_1 \\ge s_2 \\ge \\dots \\ge s_{n_S}$
    with $\\sum_i s_i = m_S$. The curve is the running share of the segment's own mass,
    $c_k = \\frac{1}{m_S}\\sum_{i \\le k} s_i$, sampled at ``ranks`` and held at 1 past
    $n_S$ so segments of different width stack on one axis. The scalar is
    $n_{90} = \\min\\{k : c_k \\ge 0.9\\}$, a count of columns directly comparable to
    $n_S$: "90% of what the top camera gets lands on 12 of its 196 patches".

    Both are computed on the head-averaged row, like every other number in this probe.
    That makes $n_{90}$ the count of columns *the layer* reads; a single head can only
    read fewer, never more, so it is an upper bound per head.
    """
    curves: dict[str, np.ndarray] = {}
    n90: dict[str, int] = {}
    for name, indices in columns.items():
        descending = np.sort(per_column[indices])[::-1]
        total = descending.sum()
        if total <= 0:
            curves[name], n90[name] = np.ones(len(ranks)), 0
            continue
        cumulative = np.cumsum(descending) / total
        curves[name] = cumulative[np.minimum(ranks, len(cumulative)) - 1]
        n90[name] = int(np.searchsorted(cumulative, 0.9) + 1)
    return curves, n90


def _clr(mass: np.ndarray) -> np.ndarray:
    """Centered log-ratio over the last axis. Composition in, unconstrained out.

    Base 2, so every log-ratio in this probe is in doublings: a shift of +1 is a
    segment taking twice its usual share relative to the geometric mean of the rest.
    """
    logs = np.log2(np.clip(mass, _EPS, None))
    return logs - logs.mean(axis=-1, keepdims=True)


def _clr_at_focus(focus_mass: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """clr over the segments that carry mass, and the names it was computed over.

    An empty segment sits on the ``_EPS`` clamp; keeping it moves the geometric mean by
    a constant that lands on every other segment, which is exactly the confound clr
    exists to remove.
    """
    keep = np.flatnonzero(focus_mass.mean(axis=0) > _MASS_FLOOR)
    return _clr(focus_mass[:, keep]), [names[i] for i in keep]


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _median_valid_depth_mm(obs: dict, pointmap_config) -> float:
    """Median depth over valid pixels, in mm — the scene covariate.

    Zero is the invalid marker in the raw sidecar, so it is excluded rather than
    dragging the median toward the camera.
    """
    depth = obs.get(f"observation.depth.{pointmap_config.depth_key}")
    if not torch.is_tensor(depth):
        return float("nan")
    values = depth.detach().float().reshape(-1)
    valid = values[values > 0]
    if valid.numel() == 0:
        return float("nan")
    return float(valid.median()) * float(pointmap_config.depth_units_mm)


def _thumbnails(result, obs: dict, pointmap_config, max_width: int = 160) -> dict[str, np.ndarray]:
    """Small previews keyed by the same segment names the budget uses.

    RGB comes from the model-view crops, not the raw observation, so the picture is
    what the attended tokens were actually built from. Depth stays in mm and keeps
    its zeros, which the renderer masks off rather than painting as near-camera.
    """
    thumbs: dict[str, np.ndarray] = {}
    for name, tensor in (result.extras.get("image_tensors_by_segment") or {}).items():
        if "_crop" in str(name) or not torch.is_tensor(tensor):
            continue
        image = (tensor.squeeze(0).detach().float().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0)
        array = image.numpy()
        step = max(1, array.shape[1] // max_width)
        thumbs[str(name)] = (array[::step, ::step] * 255).astype(np.uint8)

    depth = obs.get(f"observation.depth.{pointmap_config.depth_key}") if pointmap_config else None
    if torch.is_tensor(depth):
        array = depth.detach().float().cpu().squeeze().numpy() * float(pointmap_config.depth_units_mm)
        step = max(1, array.shape[1] // max_width)
        thumbs["depth"] = array[::step, ::step]
    return thumbs


def _filmstrip(fig, grid, keys: list[str], picks: np.ndarray, thumbnails: list[dict], frame_meta: list[dict]) -> None:
    """One column per picked frame, one row per modality, aligned to the frame axis."""
    valid = np.concatenate(
        [thumbnails[f]["depth"][thumbnails[f]["depth"] > 0].ravel() for f in picks if "depth" in thumbnails[f]]
        or [np.zeros(1)]
    )
    low, high = np.percentile(valid, [5, 95]) if valid.size > 1 else (0.0, 1.0)

    for row, key in enumerate(keys):
        for col, frame in enumerate(picks):
            ax = fig.add_subplot(grid[row, col])
            thumb = thumbnails[frame].get(key)
            if thumb is not None and key == "depth":
                ax.imshow(np.where(thumb > 0, thumb, np.nan), cmap="turbo", vmin=low, vmax=high)
            elif thumb is not None:
                ax.imshow(thumb)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                meta = frame_meta[frame]
                ax.set_title(f"f{frame}  ep{meta['episode_idx']}:{meta['frame_idx']}", fontsize=7, pad=2)
            if col == 0:
                label = "depth (mm)" if key == "depth" else key
                ax.set_ylabel(label, fontsize=8, color=_INK, fontweight="bold")
            if row == len(keys) - 1:
                ax.set_xlabel(textwrap.fill(frame_meta[frame].get("subtask") or "", 18), fontsize=5.5)


def _episode_rules(ax, frame_meta: list[dict]) -> None:
    """Mark where the frame axis crosses into another episode.

    The frame axis is a concatenation, not a trajectory: a step across a boundary is a
    cut to a different scene, and reading it as a within-episode dynamic is the easiest
    mistake this figure invites.
    """
    for index in range(1, len(frame_meta)):
        if frame_meta[index]["episode_idx"] != frame_meta[index - 1]["episode_idx"]:
            ax.axvline(index - 0.5, color="#999999", linewidth=0.6, linestyle=":", zorder=0)


def _declutter(positions: dict[str, float], min_gap: float) -> list[tuple[str, float]]:
    """Push overlapping end-of-line labels apart, lowest first, keeping their order."""
    placed: list[tuple[str, float]] = []
    for name, y in sorted(positions.items(), key=lambda item: item[1]):
        placed.append((name, y if not placed else max(y, placed[-1][1] + min_gap)))
    return placed


def _levels_panel(ax, ax_per_token, ordered: list[str], focus_mass: np.ndarray, order: np.ndarray,
                  token_counts: dict[str, int], layer: int) -> None:
    r"""The same budget twice: as it is, and per column of the segment it landed on.

    The left half is the true softmax partition, $\sum_S m_S = 1$ — what the model's
    attention compute is actually spent on. Read alone it says the cameras and the depth
    point map dominate, which is close to arithmetic: they bring 196, 196 and 192 of the
    697 columns, so at a layer that attends near-uniformly they take a large share by
    existing. The right half divides by that column count. Nothing else changes, and the
    ranking usually inverts.

    Neither half is the "correct" one. A segment's total is what it costs the model; its
    per-column share is how hard the model reads a column of it. The pair is the point,
    which is why they sit side by side sharing a row order rather than one replacing the
    other.
    """
    means, lows, highs = focus_mass.mean(axis=0)[order], focus_mass.min(axis=0)[order], focus_mass.max(axis=0)[order]
    counts = np.asarray([max(token_counts.get(n, 1), 1) for n in ordered], dtype=float)
    y = np.arange(len(ordered))[::-1]
    left = 1e-4

    ax.barh(y, means, color=[_color(n) for n in ordered], alpha=0.9, height=0.72)
    ax.hlines(y, np.clip(lows, left, None), np.clip(highs, left, None), color="#222222", linewidth=1.1)
    for row, mean, high, name in zip(y, means, highs, ordered):
        ax.text(max(mean, high, left) * 1.35, row, f"{100 * mean:.2f}%  {token_counts.get(name, 0)} tok",
                va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(left, float(highs.max()) * 30)
    ax.set_yticks(y, ordered, fontsize=7.5)
    for tick in ax.get_yticklabels():
        tick.set_color(_INK)
    ax.set_xlabel("share of the budget (log)")
    ax.set_title(f"Total, layer {layer}", fontsize=10)

    per_token = means / counts
    ax_per_token.barh(y, per_token, color=[_color(n) for n in ordered], alpha=0.9, height=0.72)
    ax_per_token.hlines(y, lows / counts, highs / counts, color="#222222", linewidth=1.1)
    for row, value in zip(y, per_token):
        ax_per_token.text(value * 1.35, row, f"{100 * value:.3f}%", va="center", fontsize=7)
    ax_per_token.set_xscale("log")
    ax_per_token.set_xlim(float(per_token.min()) / 3, float(per_token.max()) * 30)
    ax_per_token.tick_params(labelleft=False)
    ax_per_token.set_xlabel("share per column (log)")
    ax_per_token.set_title("Per column of the segment", fontsize=10)

    _caption(ax, [
        r"$m_S$ = softmax mass on segment $S$'s encoder columns, averaged over heads, over the action chunk, and over all",
        r"sampled frames. Bar = mean, rule = min..max across frames. Left: $m_S$, which sums to 1 across segments within",
        r"one frame. Right: $m_S/n_S$, the same number divided by how many columns $S$ brought — this is the only",
        r"comparison between two different segments that the column counts do not decide in advance.",
        r"Both axes are log: the segments span four decades. Segment colours are fixed across every panel of this figure.",
    ])


def _shift_panel(ax, focus_mass: np.ndarray, names: list[str], frame_meta: list[dict], layer: int) -> list[str]:
    r"""Does the model re-prioritize its inputs from frame to frame, and toward what?

    Three transforms, each removing something that would otherwise be mistaken for an
    answer:

    1. **log.** A share is a ratio, so only ratios of it mean anything: the depth clause
       going 0.10% → 0.20% is the same event as the top camera going 30% → 60%, and on a
       linear axis the first is invisible while the second owns the panel. Taking the log
       makes "doubled its share" the same vertical distance at every level. Base 2, so
       one unit on the y-axis is exactly one doubling.
    2. **minus the mean log** (this is what makes it a *centered log-ratio*). The budget
       sums to 1, so raw masses cannot move independently — if the images take more,
       every clause falls whether or not the model changed its mind about clauses. In
       log space that constraint is an additive offset shared by all segments, and
       subtracting the across-segment mean log deletes it. What survives is
       $\log_2$ of segment $S$'s share divided by the geometric mean share, which no
       other segment can push around: a rise here is $S$ gaining *relative to the field*.
    3. **minus each series' own mean over frames.** Levels are already in the left panel
       and they span four decades; centring puts all six series on one readable y-axis so
       the question becomes "when did it move", not "how big is it".

    So a point at $+0.4$ reads: at this frame that segment took $2^{0.4}\approx 1.3\times$
    its own usual share of the attention budget, relative to the rest of the field.
    """
    clr, kept = _clr_at_focus(focus_mass, names)
    delta = clr - clr.mean(axis=0, keepdims=True)
    steps = np.arange(delta.shape[0])
    rank = np.argsort(-delta.std(axis=0))
    movers = [kept[i] for i in rank[:_N_MOVERS]]

    # The quiet segments go in as an envelope, not as lines: they are here to show that
    # nothing outside the named few moved, which a bundle of extra lines cannot say.
    quiet = delta[:, rank[_N_MOVERS:]]
    if quiet.shape[1]:
        ax.fill_between(steps, quiet.min(axis=1), quiet.max(axis=1), color="#d8d8d4", alpha=0.75,
                        linewidth=0, zorder=1, label=f"other {quiet.shape[1]} segments (range)")
        ax.legend(fontsize=7, loc="lower left")
    # A surface-coloured halo under each line: six series crossing this often lose their
    # identity at every intersection otherwise, whatever the colours are.
    halo = [path_effects.withStroke(linewidth=3.4, foreground="white")]
    for name in movers:
        ax.plot(steps, delta[:, kept.index(name)], color=_color(name), linewidth=1.7, zorder=3,
                solid_capstyle="round", path_effects=halo)
    ax.axhline(0.0, color="black", linewidth=0.7)
    _episode_rules(ax, frame_meta)
    span = max(0.05, float(np.percentile(np.abs(delta), 99.5)) * 1.2)
    ax.set_ylim(-span, span)
    for name, y in _declutter({n: float(delta[-1, kept.index(n)]) for n in movers}, 0.075 * 2 * span):
        ax.annotate(name, (steps[-1], y), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=_color(name), fontweight="bold", va="center",
                    annotation_clip=False, path_effects=halo)
    ax.set_xlim(-0.5, len(steps) - 1 + 0.16 * len(steps))
    ax.set_xlabel("sampled frame")
    ax.set_ylabel("share vs own average (log$_2$, doublings)")
    ax.set_title(f"Does the model re-prioritize its inputs? Layer {layer}\n"
                 "(each segment against its own average share — no segment can move another)")
    _caption(ax, [
        r"Measured: $m_S(t)$ = the share of each action query's softmax row that lands on segment $S$'s",
        r"encoder columns, averaged over heads and over the action chunk. Plotted: that share in $\log_2$,",
        r"minus the mean $\log_2$ over the $n$ segments ($\mathrm{clr}$), minus each series' own mean over frames.",
        r"Log, because a share is only meaningful as a ratio — $0.1\%\to0.2\%$ and $30\%\to60\%$ are the same",
        r"move, and base 2 prices it at exactly $+1$. Minus the segment mean, because the budget sums to 1:",
        r"raw masses fall whenever the images rise, with no change of mind. In logs that coupling is one",
        r"shared offset, and removing it leaves $S$ against the geometric mean of the field — a rise no",
        r"other segment can manufacture. $+0.4$ = this frame gave $S$ $2^{0.4}\approx1.3\times$ its usual share.",
        r"Drawn = the " + str(_N_MOVERS) + r" largest std, the rest are one grey band; $n$ counts only segments above $10^{-6}$.",
        r"Dotted rules are episode cuts: the frame axis is a concatenation, not a trajectory.",
    ])
    return kept


def _layer_panel(fig, ax, mass: np.ndarray, ordered: list[str], order: np.ndarray,
                 layers: list[int], token_counts: dict[str, int]) -> None:
    r"""Per-column share by layer — the one grid where segments are compared to each other.

    Raw mass here was actively misleading. Cells were read across rows as "the cameras and
    the depth map get the most attention", when at the early layers that is the token
    histogram and nothing else: at layer 0 of a trained MolmoAct2 run every segment sits
    within $\pm 0.3$ doublings of its share of the columns. Dividing by $n_S$ costs nothing
    (the count is constant down a row, so each row's shape over layers is untouched) and
    makes reading across rows legitimate. The totals live in the levels panel.
    """
    from matplotlib.colors import LogNorm

    counts = np.asarray([max(token_counts.get(n, 1), 1) for n in ordered], dtype=float)
    grid = mass.mean(axis=0).T[order] / counts[:, None]  # [segments, layers]
    norm = LogNorm(vmin=max(float(grid[grid > 0].min()), _MASS_FLOOR / 100), vmax=float(grid.max()))
    image = ax.imshow(np.clip(grid, norm.vmin, None), aspect="auto", cmap="viridis", norm=norm)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            ax.text(col, row, f"{100 * grid[row, col]:.3f}", ha="center", va="center", fontsize=6.5,
                    color="white" if norm(max(grid[row, col], norm.vmin)) < 0.62 else "black")
    ax.set_xticks(range(len(layers)), [str(layer) for layer in layers])
    _row_labels(ax, ordered, [f"{n}  ({token_counts.get(n, 0)})" for n in ordered])
    ax.set_xlabel("action-expert layer")
    ax.set_title("Share per column, by layer")
    fig.colorbar(image, ax=ax, fraction=0.046).set_label("share per column (log)", fontsize=7)
    _caption(ax, [
        r"Cell = $m_S/n_S$ at that layer, averaged over frames, heads and the action chunk, printed as a percentage.",
        r"$n_S$, in brackets after each name, is that segment's column count. Dividing by it is what makes a",
        r"comparison between two rows mean something: the cameras bring 196 columns each and the clauses bring 8-20,",
        r"so in raw mass the widest segment wins by arithmetic. Columns no longer sum to 100% — the totals are in the",
        r"levels panel. A row's shape across layers is identical either way, since $n_S$ is constant along a row.",
    ])


def _distance_panel(ax, focus_mass: np.ndarray, names: list[str], depth_mm: np.ndarray,
                    picks: np.ndarray, layer: int) -> None:
    """The one pairwise log-ratio with a hypothesis behind it, against scene distance."""
    wrist = next((n for n in names if n.startswith("img_") and "wrist" in n), None)
    if "depth" not in names or wrist is None:
        ax.text(0.5, 0.5, "no depth segment", ha="center", va="center")
        ax.axis("off")
        return

    ratio = np.log2(
        np.clip(focus_mass[:, names.index("depth")], _EPS, None)
        / np.clip(focus_mass[:, names.index(wrist)], _EPS, None)
    )
    usable = np.isfinite(depth_mm) & (depth_mm > 0)
    if usable.sum() >= 3:
        rho, p = _spearman(depth_mm[usable], ratio[usable])
        ax.scatter(depth_mm[usable], ratio[usable], s=26, alpha=0.75, color=_color("depth"))
        for frame in picks:
            if usable[frame]:
                ax.annotate(f"f{frame}", (depth_mm[frame], ratio[frame]), fontsize=6,
                            alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_title(f"log2(depth / {wrist}) vs scene distance\nSpearman rho={rho:+.2f} (p={p:.3g})")
    else:
        ax.set_title(f"log2(depth / {wrist}) vs scene distance — no valid depth")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("median valid wrist depth (mm, log)")
    ax.set_ylabel("log2 mass ratio")
    _caption(ax, [
        r"$x$ = median of the wrist depth map over pixels $> 0$, in mm, one point per frame.",
        r"$y=\log_2(m_\mathrm{depth}/m_\mathrm{wrist})$ at layer " + str(layer) + r"; $0$ = equal share, $+1$ = depth takes double.",
        r"Invariant to every other segment, so no other clause can manufacture this trend.",
        r"rho is Spearman on ranks — no line is fitted, and the log $x$ axis is legibility only.",
    ])


def _entropy_panel(ax, entropy: np.ndarray, layers: list[int], frame_meta: list[dict], clr_std: float) -> None:
    """The sharpening control, zoomed to the data so a flat line is visibly flat."""
    steps = np.arange(entropy.shape[0])
    _episode_rules(ax, frame_meta)
    # Layers are ordered, not categorical, so they take one ramp light-to-dark rather
    # than unrelated hues — depth in the stack reads off the line without the legend.
    ramp = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    for index, layer in enumerate(layers):
        series = entropy[:, index]
        line, = ax.plot(steps, series, linewidth=1.3,
                        color=ramp[round(index * (len(ramp) - 1) / max(len(layers) - 1, 1))],
                        label=f"layer {layer}   mean {series.mean():.3f}   std {series.std():.4f}")
        ax.axhline(series.mean(), color=line.get_color(), linewidth=0.6, linestyle="--", alpha=0.6)
    low, high = float(entropy.min()), float(entropy.max())
    pad = max(0.01, 0.18 * (high - low))
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("sampled frame")
    ax.set_ylabel("normalized row entropy")
    ax.set_title("Sharpening control\n(flat here ⇒ composition moves are real)")
    ax.legend(fontsize=7)
    _caption(ax, [
        r"$H(t)=-\sum_j a_j\log a_j$ over the whole key axis, averaged over heads and the action",
        r"chunk, divided by $\log N_\mathrm{cols}$: $1$ = uniform over every encoder column, $0$ = one column.",
        r"Movement here is the whole row sharpening, which moves every mass without any",
        r"change in preference. Its std above vs the largest clr std, " + f"{clr_std:.3f}" + r", is the comparison.",
        r"y is zoomed to the data; the absolute level is on the axis.",
    ])


def _chunk_panel(fig, ax, by_query: np.ndarray, ordered: list[str], order: np.ndarray, layer: int) -> None:
    """Budget across the action chunk, each row against its own mean.

    Raw mass here reproduces the layer panel's problem in a second dimension — the
    image rows saturate the scale and the question ("does the split drift along the
    chunk?") is about each row's shape, not its level.
    """
    query_mass = by_query.mean(axis=0)[order]  # [segments, Q]
    fold = np.log2(np.clip(query_mass, _EPS, None) / np.clip(query_mass.mean(axis=1, keepdims=True), _EPS, None))
    span = max(0.05, float(np.percentile(np.abs(fold), 99)))

    image = ax.imshow(fold, aspect="auto", cmap="RdBu_r", vmin=-span, vmax=span)
    _row_labels(ax, ordered)
    ax.set_xlabel("action-chunk position")
    ax.set_title(f"Budget drift along the chunk, layer {layer}")
    fig.colorbar(image, ax=ax, fraction=0.046).set_label("doublings vs row mean", fontsize=7)
    _caption(ax, [
        r"Row $S$ = $\log_2\left(m_S(q)\,/\,\overline{m_S}\right)$ where $q$ indexes the action chunk and $\overline{m_S}$ is that",
        r"row's mean over $q$; averaged over frames and heads. Red = this position gives $S$ more",
        r"than its usual share, blue = less. Each row is normalized on itself, so a row's shape",
        r"is readable next to any other row regardless of how much mass it holds.",
    ])


def _series_panel(ax, ordered: list[str], pct: np.ndarray, frame_meta: list[dict], layer: int) -> None:
    """One line per segment against the frame axis, every line on the same baseline.

    This panel used to be a stacked area, which cannot be read: a band's thickness is its
    share but its position is the running total of the bands beneath it, so one segment
    moving displaces every series above it and no single share can be traced or compared.
    Sharing one baseline costs the "sums to 100%" cue — which the mean-share panel beside
    it carries anyway — and buys the only thing anyone asks this panel: what fraction of
    the budget went to depth, and did it move.

    Shares span two decades (depth ~35%, residual ~0.3%), so the axis is logarithmic;
    on a linear one the bottom half of the segments lie on the axis in a single smear.
    Each line is named twice, in the legend with its mean and again at its right end, so
    a reader tracing a line never has to match a hue against a swatch far away.
    """
    halo = [path_effects.withStroke(linewidth=3.0, foreground="white")]
    steps = np.arange(pct.shape[0])
    # A segment with no mass has no line. Clamping zero onto the log floor would draw it
    # as a flat series at whatever the floor happens to be, which reads as a real share.
    drawn = [i for i, _ in enumerate(ordered) if pct[:, i].max() > 0]
    silent = [ordered[i] for i in range(len(ordered)) if i not in drawn]
    positive = pct[pct > 0]
    floor = float(positive.min()) * 0.7 if positive.size else 1e-3

    for column in drawn:
        series = np.where(pct[:, column] > 0, pct[:, column], np.nan)
        ax.plot(steps, series, color=_color(ordered[column]), linewidth=1.6, marker="o",
                markersize=2.6, zorder=3, path_effects=halo,
                label=f"{ordered[column]} — {pct[:, column].mean():.1f}%")

    _episode_rules(ax, frame_meta)
    ax.set_yscale("log")
    ax.set_xlim(-0.5, len(steps) - 0.5 + 0.20 * max(len(steps) - 1, 1))
    ax.set_ylim(floor, float(pct.max()) * 1.6)
    # Percentages, not scientific notation, and ticked at 1-2-5 within each decade: a
    # plain log axis over this range labels only "10", which is one tick to read 13 series
    # against. This panel is read for "about what percent", so the axis has to answer that.
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=15))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.5, zorder=0)

    # Direct end-labels as well as the legend: 13 hues is past what a swatch key resolves.
    span = abs(np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0]))
    ends = {ordered[i]: np.log10(max(pct[-1, i], floor)) for i in drawn if pct[-1, i] > 0}
    for name, y in _declutter(ends, 0.055 * span):
        ax.annotate(name, (steps[-1], 10.0 ** y), xytext=(5, 0), textcoords="offset points",
                    fontsize=6.5, color=_color(name), fontweight="bold", va="center",
                    annotation_clip=False, path_effects=halo)

    ax.set_xlabel("sampled frame")
    ax.set_ylabel("share of attention budget (%)")
    ax.set_title(f"Share per segment, layer {layer} (log axis, shared baseline)"
                 + (f" — no mass: {', '.join(silent)}" if silent else ""),
                 fontsize=10)
    # Below the axes, not over them: the lines run the full width of the panel and any
    # in-axes anchor sits on top of the segments with the smallest shares.
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, handlelength=1.6, columnspacing=1.4)


def _render(
    names: list[str],
    mass: np.ndarray,          # [frames, layers, segments]
    by_query: np.ndarray,      # [frames, layers, segments, Q]
    entropy: np.ndarray,       # [frames, layers]
    layers: list[int],
    frame_meta: list[dict],
    depth_mm: np.ndarray,
    token_counts: dict[str, int],
    thumbnails: list[dict],
    output_path: str,
) -> None:
    """Render the primary budget artifact as plain percentages on linear axes."""
    del entropy, depth_mm, thumbnails

    focus = len(layers) // 2
    focus_mass = mass[:, focus, :]
    focus_pct = 100.0 * focus_mass
    steps = np.arange(len(frame_meta))
    order = np.argsort(-focus_mass.mean(axis=0))
    ordered = [names[index] for index in order]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    _series_panel(axes[0, 0], ordered, focus_pct[:, order], frame_meta, layers[focus])

    ax = axes[0, 1]
    means = focus_pct.mean(axis=0)[order]
    lows = focus_pct.min(axis=0)[order]
    highs = focus_pct.max(axis=0)[order]
    y = np.arange(len(ordered))[::-1]
    ax.barh(y, means, color=[_color(name) for name in ordered], alpha=0.9, height=0.72)
    ax.hlines(y, lows, highs, color=_INK, linewidth=1.1)
    pad = max(0.35, float(highs.max()) * 0.015)
    for row, mean, high in zip(y, means, highs):
        ax.text(high + pad, row, f"{mean:.2f}%", va="center", fontsize=8)
    ax.set_xlim(0, min(100.0, float(highs.max()) + max(5.0, 6 * pad)))
    ax.set_yticks(
        y, [f"{name}  ({token_counts.get(name, 0)} cols)" for name in ordered],
        fontsize=8,
    )
    ax.set_xlabel("mean share of attention budget (%)")
    ax.set_title(f"Mean share, layer {layers[focus]} (rule = frame range)")

    ax = axes[1, 0]
    layer_grid = 100.0 * mass.mean(axis=0).T[order]
    image = ax.imshow(
        layer_grid, aspect="auto", cmap="viridis", vmin=0.0,
        vmax=max(float(layer_grid.max()), 1.0),
    )
    for row in range(layer_grid.shape[0]):
        for col in range(layer_grid.shape[1]):
            value = layer_grid[row, col]
            ax.text(
                col, row, f"{value:.1f}%", ha="center", va="center", fontsize=7,
                color="white" if value < 0.62 * image.norm.vmax else "black",
            )
    ax.set_xticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_yticks(range(len(ordered)), ordered, fontsize=8)
    ax.set_xlabel("action-expert layer")
    ax.set_title("Mean share by layer (%)")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of attention budget (%)")

    ax = axes[1, 1]
    query_grid = 100.0 * by_query[:, focus, :, :].mean(axis=0)[order]
    image = ax.imshow(
        query_grid, aspect="auto", cmap="viridis", vmin=0.0,
        vmax=max(float(query_grid.max()), 1.0),
    )
    ax.set_yticks(range(len(ordered)), ordered, fontsize=8)
    ax.set_xlabel("action-chunk position")
    ax.set_title(f"Mean share across the action chunk, layer {layers[focus]} (%)")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of attention budget (%)")

    fig.suptitle(
        f"Action-token attention budget — {len(frame_meta)} frames, layers {layers}, "
        f"focus layer {layers[focus]}",
        fontsize=14, fontweight="bold",
    )
    # Fixed output dimensions: never let a transformed artist expand the canvas.
    fig.savefig(output_path, dpi=125)
    plt.close(fig)


def _curve_panel(ax, ordered: list[str], curves: np.ndarray, ranks: np.ndarray,
                 n90: np.ndarray, *, title: bool, caption: bool) -> None:
    r"""How few columns hold each segment's mass.

    This is the panel that answers "the cameras bring 196 columns, but how many of them
    is the model actually reading". An elbow far to the left is a segment read through a
    handful of its columns; a straight ramp to $n_S$ is a segment read whole.
    """
    for row, name in enumerate(ordered):
        ax.plot(ranks, curves[row], color=_color(name), linewidth=1.6, zorder=3,
                path_effects=[path_effects.withStroke(linewidth=3.2, foreground="white")])
        count = n90[row]
        ax.scatter([count], [0.9], s=30, color=_color(name), zorder=4, edgecolor="white", linewidth=0.7)
    ax.axhline(0.9, color="black", linewidth=0.7, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(1, float(ranks.max()))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("columns of the segment, hottest first (log)")
    ax.set_ylabel("share of that segment's own mass")
    if title:
        ax.set_title("How concentrated is each segment?")
    if caption:
        _caption(ax, [
            r"Sort segment $S$'s columns by mass, descending: $s_1 \geq s_2 \geq \ldots \geq s_{n_S}$, summing to $m_S$.",
            r"Curve = $\frac{1}{m_S}\sum_{i \leq k} s_i$ against $k$, so it starts at the hottest column's share of its",
            r"own segment and reaches 1 at $k=n_S$; held at 1 past $n_S$ so segments of different width share the axis.",
            r"Dot = where the curve crosses 0.9, which is the count in the middle panel — that panel's labelled rows",
            r"are this figure's legend. Averaged over frames, heads and the action chunk.",
            r"A curve says nothing about how much mass $S$ got, only how it spread what it got.",
        ])


def _n90_panel(ax, ordered: list[str], n90_mean: np.ndarray, n90_low: np.ndarray,
               n90_high: np.ndarray, token_counts: dict[str, int], *,
               title: bool, caption: bool) -> None:
    """The same thing as one count per segment, against the columns it was given."""
    y = np.arange(len(ordered))[::-1]
    counts = np.asarray([token_counts.get(n, 0) for n in ordered], dtype=float)

    ax.barh(y, counts, color="#e2e2dd", height=0.74, zorder=1)
    ax.barh(y, n90_mean, color=[_color(n) for n in ordered], height=0.74, zorder=2)
    ax.hlines(y, n90_low, n90_high, color=_INK, linewidth=1.1, zorder=3)
    for row, mean, total in zip(y, n90_mean, counts):
        ax.text(total * 1.12, row, f"{mean:.0f} of {total:.0f}", va="center", fontsize=7)
    # Log x, because the counts being compared run from 8 to 196: on a linear axis every
    # clause is a sliver against the cameras and the panel only reads for two segments.
    ax.set_xscale("log")
    ax.set_xlim(1, float(counts.max()) * 2.6)
    ax.set_yticks(y, ordered, fontsize=7.5)
    for tick in ax.get_yticklabels():
        tick.set_color(_INK)
    ax.set_xlabel("columns")
    if title:
        ax.set_title("Columns holding 90% of the segment")
    if not caption:
        return
    _caption(ax, [
        r"Coloured bar = $n_{90} = \min\{k : \sum_{i \leq k} s_i \geq 0.9\,m_S\}$, the number of the segment's columns",
        r"that carry 90% of the mass it received. Grey bar behind = $n_S$, every column it brought.",
        r"Rule = min..max across frames; the number is the mean, rounded.",
        r"A count, not an index: read it as \"90% of what this segment got landed on this many of its columns\".",
        r"Computed on the head-averaged row, so it counts what the layer reads — one head can only read fewer.",
    ])


def _per_column_mass(curve: np.ndarray, ranks: np.ndarray, total: float, n_columns: int) -> np.ndarray:
    r"""One segment's per-rank share *of the whole budget*, NaN past its last column.

    The stored curve is held at 1 past the segment's last column, so those ranks carry
    no mass and would otherwise draw a line along the bottom of the axis.
    """
    per_column = np.diff(curve, prepend=0.0) * total / np.diff(ranks, prepend=0)
    per_column[ranks > n_columns] = np.nan
    return per_column


def _column_mass_panel(ax, ordered: list[str], per_column: np.ndarray, ranks: np.ndarray,
                       ylim: tuple[float, float], *, title: bool, caption: bool) -> None:
    r"""The distribution itself, in budget units, so segments can be compared column to column."""
    for row, name in enumerate(ordered):
        ax.plot(ranks, per_column[row], color=_color(name), linewidth=1.5, zorder=3,
                path_effects=[path_effects.withStroke(linewidth=3.0, foreground="white")])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, float(ranks.max()))
    ax.set_ylim(*ylim)
    ax.set_xlabel("columns of the segment, hottest first (log)")
    ax.set_ylabel("share of the budget on that column (log)")
    if title:
        ax.set_title("Mass distribution within each segment")
    if not caption:
        return
    _caption(ax, [
        r"Same sort as the left panel, but in absolute units: $y$ is the share of the whole budget that lands on",
        r"one column, so heights are directly comparable between segments and each line ends at that segment's",
        r"$n_S$. Rank 1 is exact; past it each point is the mean over the ranks since the previous point, because",
        r"the curve is stored on a log-spaced grid. Colours are the middle panel's rows. Both axes are log and",
        r"the $y$-range is shared by every layer row, so a height is comparable down the figure as well as across it.",
        r"This is the panel that says whether the hottest camera patch is read as hard as a state column.",
        r"A flat line is a segment read evenly across its columns; a steep one is a segment with a few that matter.",
    ])


def _render_concentration(names, mass, curves, ranks, n90, token_counts, layers, output_path) -> None:
    """Three views of the within-segment distribution, one row per probed layer.

    Rows are layers because concentration is the half of the budget story that is *not*
    fixed by the prompt layout: $n_S$ is the same at every layer, so anything that moves
    down a column of this figure is the model changing how it reads a segment rather than
    the token histogram reasserting itself. Two things are therefore held constant across
    the rows — the segment order and the right column's $y$-range — since a per-row fit
    would rescale each layer onto its own axes and hide the drift the rows exist to show.
    """
    focus = len(layers) // 2
    mass_mean = mass.mean(axis=0)                              # [layers, segments]
    curve_mean = curves.mean(axis=0)                           # [layers, segments, ranks]
    # Ordered by the hottest column's share of its own segment: the concentration story,
    # not the mass story, so the rows of this figure sort on what it is about. Taken at
    # the focus layer and reused down the figure, so one middle-panel row is one segment.
    order = np.argsort(-curve_mean[focus, :, 0])
    ordered = [names[i] for i in order]

    per_column = np.stack([
        [_per_column_mass(curve_mean[index, segment], ranks, mass_mean[index, segment],
                          token_counts.get(names[segment], len(ranks)))
         for segment in order]
        for index in range(len(layers))
    ])                                                         # [layers, segments, ranks]
    positive = per_column[np.isfinite(per_column) & (per_column > 0)]
    ylim = (float(positive.min()) / 2.5, float(positive.max()) * 1.6)

    fig = plt.figure(figsize=(20, 17))
    grid = fig.add_gridspec(len(layers), 3, wspace=0.26, hspace=0.30,
                            left=0.055, right=0.985, top=0.945, bottom=0.155)
    for row, layer in enumerate(layers):
        axes = [fig.add_subplot(grid[row, column]) for column in range(3)]
        # Titles on the top row and captions under the bottom one: every row plots the
        # same three quantities, and repeating either three times only costs the space
        # the panels need.
        head, foot = row == 0, row == len(layers) - 1
        _curve_panel(axes[0], ordered, curve_mean[row][order], ranks,
                     np.round(n90[:, row, :].mean(axis=0))[order], title=head, caption=foot)
        _n90_panel(axes[1], ordered, n90[:, row, :].mean(axis=0)[order],
                   n90[:, row, :].min(axis=0)[order], n90[:, row, :].max(axis=0)[order],
                   token_counts, title=head, caption=foot)
        _column_mass_panel(axes[2], ordered, per_column[row], ranks, ylim,
                           title=head, caption=foot)
        box = axes[0].get_position()
        fig.text(box.x0 - 0.043, (box.y0 + box.y1) / 2, f"layer {layer}", rotation=90,
                 va="center", ha="center", fontsize=14, fontweight="bold")

    fig.suptitle(
        f"Within-segment concentration — layers {', '.join(str(x) for x in layers)} down the rows, "
        f"{mass.shape[0]} frames.  The budget says what each segment got; this says over how many "
        "of its columns.",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.text(0.5, 0.963, "Every row shares the segment order (sorted at layer "
             f"{layers[focus]}) and the right column's scale, so what differs between rows is the read itself.",
             ha="center", fontsize=10.5, color="#333333")
    fig.savefig(output_path, bbox_inches="tight", dpi=125)
    plt.close(fig)


def _load_checkpoints(step_dir: str) -> list[dict]:
    """Every checkpoint of this run that measured the same thing, oldest first.

    ``step_dir`` is this probe's own output directory,
    ``<run>/validation/step_<n>/attention_budget``; its siblings are the earlier
    checkpoints. Files written by an older version of this probe are read for whatever
    they do have — mass has been in the archive from the start, the concentration arrays
    only from this version — so a history assembled today starts its mass series at the
    first checkpoint ever run and its $n_{90}$ series at this one.
    """
    validation_dir = os.path.dirname(os.path.dirname(step_dir))
    pattern = os.path.join(validation_dir, "step_*", os.path.basename(step_dir), "budget_data.npz")
    loaded = []
    for path in sorted(glob.glob(pattern)):
        step_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        try:
            data = np.load(path, allow_pickle=True)
            entry = {
                "step": int(step_name.removeprefix("step_")),
                "names": [str(n) for n in data["segment_names"]],
                "layers": [int(x) for x in data["layers"]],
                "mass": data["mass"],
                "entropy": data["entropy"],
                "token_counts": np.asarray(data["token_counts"], dtype=float),
                "frame_idx": np.asarray(data["frame_idx"]),
                "n90": data["tokens_for_90"] if "tokens_for_90" in data.files else None,
            }
        except (ValueError, KeyError, OSError) as error:
            logging.warning(f"[attention_budget] history: skipping {path} ({error}).")
            continue
        # Segment-indexed arrays from a foreign file are only interpretable through that
        # file's own segment names, so a width that disagrees with them is unreadable
        # rather than merely old.
        if entry["n90"] is not None and entry["n90"].shape[-1] != len(entry["names"]):
            logging.warning(
                f"[attention_budget] history: {path} has {entry['n90'].shape[-1]} concentration "
                f"columns for {len(entry['names'])} segments; ignoring its concentration data."
            )
            entry["n90"] = None
        loaded.append(entry)
    return loaded


def _history_series(checkpoints: list[dict], focus_layer: int):
    """Align the checkpoints onto one segment set and one layer, or explain why not.

    Three things have to match before two checkpoints can be put on one axis: the segment
    set, the layer, and the frames. The first two are enforced here by intersecting and
    by looking the layer up by value rather than by position. The third cannot be fixed
    after the fact — different frames means a different scene mix, which moves every
    share on its own — so it is only detected, and reported in the figure's title.
    """
    usable = [c for c in checkpoints if focus_layer in c["layers"]]
    if len(usable) < 2:
        return None
    common = [n for n in usable[-1]["names"] if all(n in c["names"] for c in usable)]
    if not common:
        return None
    # Segments with no mass at all are excluded before the clr, not after: an empty
    # segment sits on the log clamp at $2^{-40}$ and drags the geometric mean that every
    # other series is measured against, which shows up as every segment drifting together.
    # Archives written before pad columns were dropped from the partition carry exactly
    # such a segment, so a mixed-version history hits this.
    empty = [n for n in common
             if min(c["mass"][:, c["layers"].index(focus_layer), c["names"].index(n)].mean()
                    for c in usable) <= _MASS_FLOOR]
    common = [n for n in common if n not in empty]
    if not common:
        return None

    steps = np.asarray([c["step"] for c in usable])
    mass = np.zeros((len(usable), len(common)))
    n90 = np.full((len(usable), len(common)), np.nan)
    entropy = np.zeros(len(usable))
    for row, c in enumerate(usable):
        layer = c["layers"].index(focus_layer)
        index = [c["names"].index(n) for n in common]
        mass[row] = c["mass"][:, layer, :].mean(axis=0)[index]
        entropy[row] = c["entropy"][:, layer].mean()
        if c["n90"] is not None:
            n90[row] = c["n90"][:, layer, :].mean(axis=0)[index]

    reference = usable[-1]
    counts = reference["token_counts"][[reference["names"].index(n) for n in common]]
    same_frames = all(
        c["frame_idx"].shape == reference["frame_idx"].shape and bool((c["frame_idx"] == reference["frame_idx"]).all())
        for c in usable
    )
    same_counts = all(
        bool((c["token_counts"][[c["names"].index(n) for n in common]] == counts).all()) for c in usable
    )
    return {"steps": steps, "names": common, "mass": mass, "n90": n90, "entropy": entropy,
            "token_counts": counts, "same_frames": same_frames, "same_counts": same_counts,
            "empty": empty,
            "dropped": [n for n in reference["names"] if n not in common and n not in empty]}


def _trend_panel(ax, steps, names, values, title, ylabel, log_y: bool) -> None:
    """One line per segment against training step, labelled at the right edge."""
    halo = [path_effects.withStroke(linewidth=3.2, foreground="white")]
    for col, name in enumerate(names):
        series = values[:, col]
        if not np.isfinite(series).any():
            continue
        ax.plot(steps, series, color=_color(name), linewidth=1.7, marker="o",
                markersize=3.4, zorder=3, path_effects=halo)
    if log_y:
        ax.set_yscale("log")
    span = ax.get_ylim()
    to_data = (lambda v: 10.0 ** v) if log_y else (lambda v: v)
    from_data = (lambda v: np.log10(max(v, _EPS))) if log_y else (lambda v: v)
    gap = 0.055 * abs(from_data(span[1]) - from_data(span[0]))
    ends = {n: from_data(float(values[-1, i])) for i, n in enumerate(names) if np.isfinite(values[-1, i])}
    for name, y in _declutter(ends, gap):
        ax.annotate(name, (steps[-1], to_data(y)), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=_color(name), fontweight="bold", va="center",
                    annotation_clip=False, path_effects=halo)
    spread = max(int(np.ptp(steps)), 1)
    ax.set_xlim(steps.min() - 0.02 * spread, steps.max() + 0.22 * spread)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def build_history(step_dir: str, output_path: str) -> dict:
    r"""The budget against training step, assembled from every checkpoint on disk.

    This is the probe's answer to the question the per-checkpoint figures cannot reach:
    not "what is the budget" but "where is it going". It needs no GPU and no rerun — it
    reads the archives the probe has already written, so it fills in retroactively and
    grows by one column every validation pass.

    Why no normalization is needed here, and this is the crux of the whole probe: a
    segment's column count $n_S$ is fixed by the prompt layout and the camera geometry,
    so it is the same at step 400 and step 1600. Every count effect is therefore a
    constant offset on that segment's series, and constants cancel in a comparison of a
    segment with itself. The size argument that makes the cross-segment bars misleading
    has no purchase on a trend. What does *not* cancel is the frame set — measure a
    different set of scenes and every share moves for reasons that have nothing to do
    with the weights — so that is checked and reported rather than assumed.

    Returns a summary dict, empty when there is nothing to compare yet.
    """
    aligned = _history_series(_load_checkpoints(step_dir), _focus_layer_of(step_dir))
    if aligned is None:
        return {}

    steps, names = aligned["steps"], aligned["names"]
    mass, n90, entropy = aligned["mass"], aligned["n90"], aligned["entropy"]
    clr = _clr(mass)
    order = np.argsort(-mass[-1])
    ordered = [names[i] for i in order]

    fig = plt.figure(figsize=(17, 12.5))
    grid = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.22, left=0.06, right=0.97, top=0.91, bottom=0.07)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]

    _trend_panel(axes[0], steps, ordered, 100 * mass[:, order],
                 "Share of the budget", "share of the budget (%, log)", log_y=True)
    _caption(axes[0], [
        r"$m_S$ at the focus layer, averaged over frames, heads and the action chunk, per checkpoint.",
        r"Log $y$: a share is a ratio, so equal vertical distances are equal factors.",
        r"These are the raw levels and they are coupled — the budget sums to 1, so one segment cannot",
        r"rise without the others falling. The panel to the right removes that coupling.",
    ])

    _trend_panel(axes[1], steps, ordered, clr[:, order],
                 "Share relative to the field (this is the trend to read)",
                 r"$\mathrm{clr}$ (doublings vs the geometric mean)", log_y=False)
    _caption(axes[1], [
        r"$\log_2 m_S$ minus the mean $\log_2 m$ over the " + str(len(names)) + r" segments, per checkpoint.",
        r"Subtracting the across-segment mean deletes the sum-to-1 coupling: a rise here is $S$ gaining on the",
        r"field, which no other segment can manufacture by shrinking. $+1$ = twice the geometric-mean share.",
        r"Absolute height is not meaningful (it carries the segment's column count as a fixed offset); the",
        r"slope is. Column counts are constant across checkpoints, so they cannot bend any of these lines.",
    ])

    if np.isfinite(n90).any():
        _trend_panel(axes[2], steps, ordered, n90[:, order],
                     "Columns holding 90% of the segment", "columns (log)", log_y=True)
        for row, name in enumerate(ordered):
            ax_count = aligned["token_counts"][order][row]
            axes[2].axhline(ax_count, color=_color(name), linewidth=0.5, linestyle=":", alpha=0.45)
    else:
        axes[2].text(0.5, 0.5, "no concentration data in these checkpoints yet\n"
                               "(written from this version of the probe onward)",
                     ha="center", va="center", fontsize=9)
        axes[2].axis("off")
    _caption(axes[2], [
        r"$n_{90}$ at the focus layer, averaged over frames: how many of the segment's columns carry 90% of the",
        r"mass it received. Dotted line = that segment's $n_S$, the ceiling. Falling means the model is reading",
        r"the same segment through fewer of its columns; rising means it is spreading out.",
        r"A count, so it is already comparable across segments and across checkpoints with nothing divided out.",
    ])

    axes[3].plot(steps, entropy, color="#256abf", linewidth=1.8, marker="o", markersize=4)
    axes[3].set_xlabel("training step")
    axes[3].set_ylabel("normalized row entropy")
    axes[3].set_title("Sharpening control\n(a trend here moves every share above at once)")
    low, high = float(entropy.min()), float(entropy.max())
    axes[3].set_ylim(low - max(0.01, 0.25 * (high - low)), high + max(0.01, 0.25 * (high - low)))
    _caption(axes[3], [
        r"$-\sum_j a_j \log a_j$ over the whole key axis divided by $\log N_\mathrm{cols}$, averaged over frames,",
        r"heads and the chunk: $1$ = uniform over every encoder column, $0$ = all mass on one.",
        r"This is the confound for everything above. If the rows are sharpening over training, concentrated",
        r"segments gain share and diffuse ones lose it with no change of preference — so a mass trend is only",
        r"a preference trend to the extent this line is flat. Range here: " + f"{low:.3f} to {high:.3f}.",
    ])

    risers = sorted(range(len(names)), key=lambda i: -(clr[-1, i] - clr[0, i]))
    finite = np.flatnonzero(np.isfinite(n90).any(axis=1))
    first_n90 = int(finite[0]) if finite.size else None
    # Two kinds of note, and only one of them is a problem: a segment that carries no mass
    # or is missing from a checkpoint is excluded and the rest of the figure is unaffected,
    # whereas a changed frame set or column count means the series themselves are not
    # comparable. Colouring the title for the first kind would cry wolf on every run.
    notes = []
    if aligned["empty"]:
        notes.append(f"excluded, no mass: {', '.join(aligned['empty'])}")
    if aligned["dropped"]:
        notes.append(f"not in every checkpoint: {', '.join(aligned['dropped'])}")
    invalid = []
    if not aligned["same_frames"]:
        invalid.append("DIFFERENT FRAMES across checkpoints — trends confound weights with scene mix")
    if not aligned["same_counts"]:
        invalid.append("COLUMN COUNTS CHANGED across checkpoints — per-column numbers shift with them")
    fig.suptitle(
        f"Attention budget over training — steps {steps.min()} to {steps.max()} "
        f"({len(steps)} checkpoints), layer {_focus_layer_of(step_dir)}"
        + (f"  |  {'  |  '.join(invalid + notes)}" if invalid or notes else ""),
        fontsize=13, fontweight="bold",
        color="#8c1c13" if invalid else "black",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=125)
    plt.close(fig)

    return {
        "steps": [int(s) for s in steps],
        "n_checkpoints": len(steps),
        "same_frames": aligned["same_frames"],
        "mass_change": {n: float(mass[-1, i] - mass[0, i]) for i, n in enumerate(names)},
        "clr_change": {n: float(clr[-1, i] - clr[0, i]) for i, n in enumerate(names)},
        # Concentration arrives mid-run for any run that started before this version of
        # the probe, so its span is its own and is recorded with it.
        "n90_change": {n: float(n90[-1, i] - n90[first_n90, i]) for i, n in enumerate(names)
                       if first_n90 is not None and np.isfinite(n90[-1, i])},
        "n90_change_from_step": int(steps[first_n90]) if first_n90 is not None else None,
        "entropy_change": float(entropy[-1] - entropy[0]),
        "top_riser": names[risers[0]],
        "top_faller": names[risers[-1]],
        "clr_change_max": float(clr[-1, risers[0]] - clr[0, risers[0]]),
    }


def _focus_layer_of(step_dir: str) -> int:
    """The focus layer this run's own archive was written with."""
    data = np.load(os.path.join(step_dir, "budget_data.npz"), allow_pickle=True)
    layers = [int(x) for x in data["layers"]]
    return layers[len(layers) // 2]


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[attention_budget] needs continuous flow actions — skipping.")
        return

    makedirs(output_dir)
    p = cfg.probe_parameters
    chunk_size = int(cfg.policy.chunk_size)
    timestep = float(getattr(p, "timestep", 0.5))
    layers = [int(x.strip()) for x in p.spatial_layers.split(",")]
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    want_fd = bool(getattr(p, "budget_fd_sensitivity", False))

    samples = sample_episodes_evenly(
        dataset, int(getattr(p, "budget_n_frames_per_episode", None) or p.n_frames_per_episode),
        p.max_episodes, p.random_seed, probe_image_stride(cfg),
    )
    if not samples:
        logging.warning("[attention_budget] no frames selected.")
        return

    adapter._set_probe_cuda_graph_enabled(False)
    names: list[str] | None = None
    ranks: np.ndarray | None = None
    mass_rows: list[np.ndarray] = []
    query_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    curve_rows: list[np.ndarray] = []
    n90_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    frame_meta: list[dict] = []
    depth_mm: list[float] = []
    fd_rows: list[dict] = []
    thumbnails: list[dict] = []

    try:
        for ep_idx, fr_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size)
            result = adapter.capture_attention(
                frame["obs"], frame["task"], state=frame["state"], timestep=timestep,
                layers=layers, subtask=frame["subtask"], metadata=frame["metadata"],
            )
            if not result.cross_attn_by_layer:
                logging.warning(f"[attention_budget] ep={ep_idx} fr={fr_idx}: no cross-attn; skipping.")
                continue

            encoder_len = int(next(iter(result.cross_attn_by_layer.values())).shape[-1])
            columns = _segment_columns(result, encoder_len)
            if names is None:
                names = list(columns)
                ranks = _rank_grid(max(len(indices) for indices in columns.values()))
            elif list(columns) != names:
                # A clause the packer dropped would silently re-index every array.
                logging.warning(
                    f"[attention_budget] ep={ep_idx} fr={fr_idx}: segment set changed "
                    f"({set(columns) ^ set(names)}); skipping frame."
                )
                continue

            frame_mass, frame_query, frame_entropy, frame_curves, frame_n90 = [], [], [], [], []
            for layer in layers:
                cross = result.cross_attn_by_layer.get(layer)
                if cross is None:
                    break
                attn = torch.nan_to_num(cross[0].float().cpu(), nan=0.0)
                mass, by_query, entropy, per_column = _frame_budget(attn, columns)
                curves, n90 = _concentration(per_column, columns, ranks)
                frame_mass.append([mass[name] for name in names])
                frame_query.append([by_query[name] for name in names])
                frame_entropy.append(entropy)
                frame_curves.append([curves[name] for name in names])
                frame_n90.append([n90[name] for name in names])
            if len(frame_mass) != len(layers):
                continue

            mass_rows.append(np.asarray(frame_mass, dtype=np.float64))
            query_rows.append(np.asarray(frame_query, dtype=np.float32))
            entropy_rows.append(np.asarray(frame_entropy, dtype=np.float64))
            curve_rows.append(np.asarray(frame_curves, dtype=np.float32))
            n90_rows.append(np.asarray(frame_n90, dtype=np.int32))
            token_rows.append(np.asarray([len(columns[name]) for name in names], dtype=np.int32))
            frame_meta.append(
                {"episode_idx": int(ep_idx), "frame_idx": int(fr_idx),
                 "global_idx": int(global_idx), "task": frame["task"],
                 "subtask": frame["subtask"]}
            )
            depth_mm.append(
                _median_valid_depth_mm(frame["obs"], pointmap_config)
                if pointmap_config is not None else float("nan")
            )
            thumbnails.append(_thumbnails(result, frame["obs"], pointmap_config))

            if want_fd:
                fd_rows.append(_fd_sensitivity(adapter, frame, pointmap_config))
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not mass_rows or names is None:
        logging.warning("[attention_budget] no frames produced a budget.")
        return

    mass = np.stack(mass_rows)            # [frames, layers, segments]
    by_query = np.stack(query_rows)       # [frames, layers, segments, Q]
    entropy = np.stack(entropy_rows)      # [frames, layers]
    curves = np.stack(curve_rows)         # [frames, layers, segments, ranks]
    n90 = np.stack(n90_rows)              # [frames, layers, segments]
    tokens = np.stack(token_rows)         # [frames, segments]
    depth_array = np.asarray(depth_mm, dtype=np.float64)
    # Median, not frame 0: the clause segments change length frame to frame, and every
    # per-token number below divides by this.
    token_counts = {name: int(np.median(tokens[:, j])) for j, name in enumerate(names)}

    total = mass.sum(axis=-1)
    if not np.allclose(total, 1.0, atol=2e-2):
        logging.warning(
            f"[attention_budget] budget does not sum to 1 (min={total.min():.4f} "
            f"max={total.max():.4f}) — the column partition is missing mass."
        )

    summary = _summarize(names, mass, entropy, depth_array, token_counts, layers,
                         frame_meta, fd_rows, n90, tokens)
    np.savez_compressed(
        os.path.join(output_dir, "budget_data.npz"),
        segment_names=np.asarray(names), layers=np.asarray(layers),
        mass=mass, mass_by_query=by_query, entropy=entropy,
        median_depth_mm=depth_array,
        token_counts=np.asarray([token_counts[n] for n in names]),
        tokens_per_frame=tokens, cumulative_share=curves, ranks=ranks, tokens_for_90=n90,
        episode_idx=np.asarray([m["episode_idx"] for m in frame_meta]),
        frame_idx=np.asarray([m["frame_idx"] for m in frame_meta]),
    )
    _render(names, mass, by_query, entropy, layers, frame_meta, depth_array,
            token_counts, thumbnails, os.path.join(output_dir, "budget.png"))
    _render_concentration(names, mass, curves, ranks, n90, token_counts, layers,
                          os.path.join(output_dir, "concentration.png"))
    # Reads this run's earlier checkpoints off disk, so it needs the archive above to
    # exist first and it lands in the summary that gets written below.
    summary["history"] = build_history(output_dir, os.path.join(output_dir, "trajectory.png"))
    with open(os.path.join(output_dir, "budget.json"), "w") as f:
        json.dump(summary, f, indent=2)

    focus = len(layers) // 2
    clr_at_focus, kept = _clr_at_focus(mass[:, focus, :], names)
    ranked = sorted(names, key=lambda n: -mass[:, focus, names.index(n)].mean())
    logging.info(
        f"[attention_budget] n={len(frame_meta)} layer {layers[focus]} mean budget: "
        + "  ".join(f"{n}={mass[:, focus, names.index(n)].mean():.3f}" for n in ranked[:6])
    )
    per_token = sorted(names, key=lambda n: -summary["mean_mass_per_token"][n])
    logging.info(
        f"[attention_budget] layer {layers[focus]} per column of the segment: "
        + "  ".join(
            f"{n}={100 * summary['mean_mass_per_token'][n]:.3f}%"
            f"/{token_counts[n]}tok, 90% in {summary['tokens_for_90'][n]:.0f}"
            for n in per_token[:6]
        )
    )
    if summary["history"]:
        h = summary["history"]
        logging.info(
            f"[attention_budget] over {h['n_checkpoints']} checkpoints "
            f"({h['steps'][0]}..{h['steps'][-1]}): {h['top_riser']} gained "
            f"{h['clr_change_max']:+.3f} doublings on the field, {h['top_faller']} lost the most; "
            f"entropy moved {h['entropy_change']:+.4f}"
        )
    volatile = sorted(kept, key=lambda n: -clr_at_focus[:, kept.index(n)].std())
    logging.info(
        "[attention_budget] most variable across frames (clr std, doublings): "
        + "  ".join(f"{n}={clr_at_focus[:, kept.index(n)].std():.3f}" for n in volatile[:5])
    )
    if summary.get("depth_distance_spearman") is not None:
        rho = summary["depth_distance_spearman"]["rho"]
        logging.info(
            f"[attention_budget] log(depth/wrist) vs scene distance: rho={rho:+.3f} "
            f"(p={summary['depth_distance_spearman']['p']:.3g}); negative ⇒ depth share rises up close"
        )

    focus_key = str(layers[focus])
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Attention Budget",
        group="Attention",
        claim="How is the fixed attention budget split across modalities, and what moves it?",
        summary=summary,
        see_also=["attention", "depth_modality", "action_trace"],
        metrics=[
            Metric(
                "budget_sums_to.min", "Budget partition completeness", good="high", fmt=4, primary=True,
                baseline=1.0, warn=0.99, bad=0.98,
                note=(
                    "Every row sums to 1 by construction, so a value below 1 means the column "
                    "partition is missing mass and every share below is diluted by an unknown amount."
                ),
            ),
            Metric(
                "clr_std_max", "Largest compositional shift (clr std, doublings)",
                good="none", fmt=3, primary=True,
                value=float(clr_at_focus.std(axis=0).max()),
                note=f"most variable segment: {volatile[0]}. Base 2, so 1.0 means that segment's "
                     "share swings by a factor of two around its own average.",
            ),
            Metric(
                "entropy_at_focus", "Row entropy at the focus layer", good="none", fmt=3, primary=True,
                value=float(entropy[:, focus].mean()),
                note=(
                    "The sharpening control. If this is flat while the composition moves, the "
                    "composition move is real rather than the whole distribution tightening."
                ),
            ),
            Metric(
                "entropy_std_by_layer." + focus_key, "Entropy spread across frames",
                good="none", fmt=3,
            ),
            Metric(
                "depth_distance_spearman.rho", "log2(depth / wrist) vs scene distance",
                good="low", fmt=3, baseline=0.0, refs=["depth_modality"], primary=True,
                note=(
                    "Negative means depth takes a larger share as the scene gets closer. Mediated, "
                    "not causal: approaching an object changes phase, gripper state and pixels at once."
                ),
            ),
            Metric("depth_distance_spearman.p", "…its p-value", good="low", fmt=4),
            Metric(
                "fd_sensitivity_mean.depth", "Causal sensitivity to depth", good="none", fmt=4,
                note="Only populated with --probe_parameters.budget_fd_sensitivity. Mass says a "
                     "segment was read; this says perturbing it changes the actions.",
            ),
            Metric("fd_sensitivity_mean.rgb_wrist", "Causal sensitivity to wrist RGB",
                   good="none", fmt=4),
            Metric(
                "history.clr_change_max", "Largest gain on the field over training (doublings)",
                good="none", fmt=3, primary=True, baseline=0.0,
                note=(
                    f"Segment: {summary['history'].get('top_riser', 'n/a')}. Change in clr between the "
                    "first and last checkpoint on disk, so it needs at least two. Column counts are "
                    "fixed across checkpoints, so no normalization enters here — but read it next to "
                    "history.entropy_change, which moves every share at once."
                ) if summary["history"] else "Needs two or more checkpoints of this probe.",
            ),
            Metric(
                "history.entropy_change", "Row-entropy drift over training", good="none", fmt=4,
                baseline=0.0,
                note="The confound for every trend in trajectory.png. Near zero means the composition "
                     "trends are preference, not the whole distribution sharpening.",
            ),
            Metric("history.n_checkpoints", "Checkpoints in the trajectory", good="none", fmt=0),
            Metric("focus_layer", "Focus layer", good="none", fmt=0),
            Metric("n_frames", "Frames measured", good="none", fmt=0),
        ],
        panels=[
            Panel(
                "budget.png",
                f"Attention budget across {len(frame_meta)} frames, focus layer {layers[focus]}",
                "Four fixed-size panels using ordinary percentages and linear axes. "
                "**Top left:** every segment's share against the frame axis, one line per segment on a "
                "shared baseline and a log y-axis, so a share can be read off the axis and two "
                "segments compared. Nothing is stacked: on a stacked area a band sits on the "
                "running total beneath it, so one segment moving displaces every series above it. "
                "**Top right:** each segment's mean share, with its min-to-max frame range. "
                "**Bottom left:** mean percentage by probed layer. "
                "**Bottom right:** mean percentage by action-chunk position. "
                "Segment colours are fixed across panels; labels include encoder-column counts.",
                primary=True,
                refs=["attention"],
            ),
            Panel(
                "concentration.png",
                f"Within-segment concentration at layers {', '.join(str(x) for x in layers)}",
                "The budget above says what each segment got. This says over how many of its columns, "
                "which is what makes the totals readable: a camera brings 196 columns and a clause "
                "brings 9, so a bigger total is not by itself a stronger read. One **row per probed "
                "layer**, on a shared segment order and a shared scale, so a change down a column is "
                "the model reading a segment differently and not the token histogram. **Left** sorts each "
                "segment's own columns hottest-first and plots the running share of that segment's "
                "mass — an elbow near the left edge is a segment read through a few of its columns, a "
                "straight ramp is one read whole. **Middle** is the same thing as one count, $n_{90}$, "
                "against the grey bar of every column the segment brought. **Right** drops the "
                "within-segment normalization and plots the columns in budget units, so heights are "
                "comparable between segments: this is where you see whether the hottest camera patch "
                "is read as hard as a state column.",
                primary=True,
            ),
            *([Panel(
                "trajectory.png",
                f"The budget against training step, {summary['history']['n_checkpoints']} checkpoints "
                f"({summary['history']['steps'][0]} to {summary['history']['steps'][-1]})",
                "Assembled from every checkpoint of this run on disk, so it grows by a column each "
                "validation pass and needs no rerun. This is the figure where the column counts stop "
                "mattering: $n_S$ is fixed by the prompt layout and the camera geometry, so it is the "
                "same at every step and contributes a constant offset to a segment's series, which "
                "cancels when that segment is compared with itself. **Top left** is the raw share and "
                "**top right** removes the sum-to-1 coupling, so a rise there is a segment gaining on "
                "the field rather than another one shrinking — that is the panel to read. **Bottom "
                "left** is how many columns of itself each segment reads, and **bottom right** is the "
                "control: if the rows are sharpening, concentrated segments gain share with no change "
                "of preference. A red title means the checkpoints are not comparable — most often "
                "different frames, which moves every share for reasons unrelated to the weights.",
                primary=True,
            )] if summary["history"] else []),
        ],
    )


def rerender_budget(step_dir: str) -> str:
    """Rebuild budget.png from the saved arrays, with no model or dataset load."""
    archive_path = os.path.join(step_dir, "budget_data.npz")
    with np.load(archive_path, allow_pickle=True) as data:
        names = [str(value) for value in data["segment_names"]]
        layers = [int(value) for value in data["layers"]]
        mass = np.asarray(data["mass"])
        by_query = np.asarray(data["mass_by_query"])
        entropy = np.asarray(data["entropy"])
        depth_mm = np.asarray(data["median_depth_mm"])
        counts = [int(value) for value in data["token_counts"]]
        episode_idx = np.asarray(data["episode_idx"])
        frame_idx = (
            np.asarray(data["frame_idx"])
            if "frame_idx" in data.files
            else np.arange(len(episode_idx))
        )

    frame_meta = [
        {"episode_idx": int(episode), "frame_idx": int(frame)}
        for episode, frame in zip(episode_idx, frame_idx)
    ]
    output_path = os.path.join(step_dir, "budget.png")
    _render(
        names, mass, by_query, entropy, layers, frame_meta, depth_mm,
        dict(zip(names, counts)), [], output_path,
    )
    return output_path


if __name__ == "__main__":
    # Re-render the fixed percentage budget from saved arrays, without a GPU:
    #   python -m lerobot.probes.attention_budget <step>/attention_budget --budget-only
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = sys.argv[1].rstrip("/")
    logging.info(f"wrote {rerender_budget(target)}")
    if "--budget-only" not in sys.argv[2:]:
        result = build_history(target, os.path.join(target, "trajectory.png"))
        if not result:
            logging.warning("nothing to compare: need two checkpoints sharing a layer and a segment.")
        else:
            logging.info(json.dumps({k: v for k, v in result.items() if k != "mass_change"}, indent=2))


def _fd_sensitivity(adapter, frame, pointmap_config) -> dict:
    """Per-frame causal complement: ||Δactions|| under a 1%-of-std input perturbation.

    Mass says a segment was read; this says perturbing it changes the output. Two
    extra forwards per frame, hence the flag.
    """
    if pointmap_config is None:
        return {}
    obs = frame["obs"]
    depth_key = f"observation.depth.{pointmap_config.depth_key}"
    rgb_key = f"observation.images.{pointmap_config.depth_key}"

    def predict(observation):
        generator = torch.Generator(device=adapter.device)
        generator.manual_seed(0)
        return adapter.predict_action_chunk(
            observation, frame["task"], state=frame["state"], subtask=frame["subtask"],
            metadata=frame["metadata"], generator=generator,
        )[1]

    base = predict(obs)
    out = {}
    for label, key in (("depth", depth_key), ("rgb_wrist", rgb_key)):
        raw = obs.get(key)
        if not torch.is_tensor(raw):
            continue
        noise = torch.randn_like(raw.float()) * raw.float().std() * 0.01
        out[label] = float((predict({**obs, key: (raw.float() + noise).to(raw.dtype)}) - base).norm())
    return out


def _summarize(names, mass, entropy, depth_mm, token_counts, layers, frame_meta, fd_rows,
               n90, tokens) -> dict:
    focus = len(layers) // 2
    clr, kept = _clr_at_focus(mass[:, focus, :], names)
    summary = {
        "n_frames": len(frame_meta),
        "layers": layers,
        "focus_layer": layers[focus],
        "segments": names,
        "token_counts": token_counts,
        "budget_sums_to": {"min": float(mass.sum(axis=-1).min()), "max": float(mass.sum(axis=-1).max())},
        "mean_mass_by_layer": {
            str(layer): {name: float(mass[:, i, j].mean()) for j, name in enumerate(names)}
            for i, layer in enumerate(layers)
        },
        "clr_std_at_focus_layer": {name: float(clr[:, j].std()) for j, name in enumerate(kept)},
        "entropy_mean_by_layer": {str(layer): float(entropy[:, i].mean()) for i, layer in enumerate(layers)},
        "entropy_std_by_layer": {str(layer): float(entropy[:, i].std()) for i, layer in enumerate(layers)},
        "tasks": sorted({m["task"] for m in frame_meta}),
    }

    # The four numbers this probe is read for, all at the focus layer. Every one is in a
    # unit that appears on the figure: a share, a share, a count of doublings, a count of
    # columns. Two of them divide by the column count, so it is recorded per segment
    # alongside how much it varied across frames.
    summary["mean_mass_per_token"] = {
        name: float(mass[:, focus, j].mean() / max(token_counts.get(name, 1), 1))
        for j, name in enumerate(names)
    }
    summary["tokens_for_90"] = {
        name: float(n90[:, focus, j].mean()) for j, name in enumerate(names)
    }
    summary["token_counts_across_frames"] = {
        name: {"median": int(np.median(tokens[:, j])), "min": int(tokens[:, j].min()),
               "max": int(tokens[:, j].max())}
        for j, name in enumerate(names)
    }

    if "depth" in names:
        wrist = next((n for n in names if n.startswith("img_") and "wrist" in n), None)
        finite = np.isfinite(depth_mm) & (depth_mm > 0)  # same mask the figure plots
        if wrist is not None and finite.sum() >= 3:
            ratio = np.log2(
                np.clip(mass[:, focus, names.index("depth")], _EPS, None)
                / np.clip(mass[:, focus, names.index(wrist)], _EPS, None)
            )
            rho, p = _spearman(depth_mm[finite], ratio[finite])
            summary["depth_distance_spearman"] = {
                "rho": rho, "p": p, "against": wrist,
                "note": "negative rho ⇒ depth's share of the budget rises as the scene gets closer",
            }

    if len(summary["tasks"]) > 1:
        summary["mean_mass_by_task"] = {
            task: {
                name: float(
                    mass[[i for i, m in enumerate(frame_meta) if m["task"] == task], focus, j].mean()
                )
                for j, name in enumerate(names)
            }
            for task in summary["tasks"]
        }

    if fd_rows:
        keys = {k for row in fd_rows for k in row}
        summary["fd_sensitivity_mean"] = {
            key: float(np.mean([row[key] for row in fd_rows if key in row])) for key in keys
        }

    summary["per_frame"] = [
        meta | {"median_depth_mm": float(depth_mm[i]),
                "mass_at_focus_layer": {name: float(mass[i, focus, j]) for j, name in enumerate(names)}}
        for i, meta in enumerate(frame_meta)
    ]
    return summary
