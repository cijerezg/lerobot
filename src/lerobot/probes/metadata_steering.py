r"""Metadata-steering probe: does the quality / mistake / speed clause reach the actions?

The action prompt carries a steering clause built by ``_build_robot_text``
(processor_molmoact2.py) — "The quality is $N$ of 5." and "The robot made a
mistake."/"The robot made no mistakes." The active config keeps it on every
training sample. At rollout every deployment prompt asks for the
same thing: quality 5, no mistakes. This probe is the whole test of whether asking
does anything.

Three clauses, two of them five-level scales, so the probe is a small factorial rather
than a two-pole contrast. Every condition is one ``predict_action_chunk`` on the same
frame under the same seeded flow noise, so the only thing that varies is the clause:

  ``none``          no metadata clause at all (the dropout regime, and the origin
                    every displacement below is measured from)
  ``q1`` … ``q5``   quality $q$, no mistake, speed 5 — the dose axis
  ``q1m``, ``q5m``  the same poles with the mistake sentence flipped on
  ``s1`` … ``s5``   speed $k$ at quality 5 with no mistake — the tempo axis. ``s5`` is
                    character-for-character the ``q5`` prompt, so it is aliased rather
                    than forwarded twice
  ``gt``            the frame's own labels, from ``meta/episode_metadata.parquet``
                    (quality, per segment) + ``meta/mistakes.parquet`` (per 4 s window)
                    + ``meta/speed_hybrid_v1.parquet`` (speed, per segment), the same
                    spans training broadcasts
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

**3. Is the response conditional on the frame, or a constant?** Moving the chunk is not
the same as reading the clause. Write $v^{(c)} = a^{(c)} - a^{(\mathrm{none})}$ for the
displacement a clause causes, $h$ for the chunk that repeats the measured state and never
moves, and

$$\rho(c)=\frac{\frac{1}{TD}\sum_{t,d}\left(v^{(c)}_{t,d}\right)^{2}}
{\max\left(\frac{1}{TD}\sum_{t,d}\left(h_{t,d}-a^{(\mathrm{none})}_{t,d}\right)^{2},\ \phi\right)}$$

for its size in units of the chunk's own motion, $\phi$ being the scale floor from
``action_metrics``. $\rho=1$ means the clause rewrote as much trajectory as the
trajectory has, and — this is the point — every frame contributes on one scale no matter
how far the arm travels in it, so frames may be compared with each other.

Quality is annotated per segment, so a frame has a *true* level and can be asked a
different one. A model that reads the frame moves least when asked the truth and more as
the mismatch grows, giving $\rho(q)$ a dip at $q_{\text{true}} = q$. A clause applied as a
constant prior gives a $\rho(q)$ that does not depend on the true level at all.

**4. One direction, or one per frame?** A prior is literally a shared vector. Within a
frame the two poles should oppose, $\cos(v^{(q_1)},v^{(q_5)})\approx-1$, if the levels
sit on one signed axis. Across frames, $\cos(v^{(q_5)}_f, \bar{v}_{-f})$ against the
leave-one-out mean separates one global style offset ($+1$) from a displacement computed
per frame ($0$), with reseeded flow noise as the null. The shared fraction

$$R=\frac{\lVert\bar{u}\rVert^{2}}{\operatorname{mean}_f\lVert u_f\rVert^{2}},
\qquad u_f=a^{(q_5)}_f-a^{(q_1)}_f$$

is that statement as one number, and $\bar{u}$ itself — per joint, per chunk step — is
what the model thinks "quality 5" means.

Nothing on that page is scored against the demonstration. $a^{\star}$ is one sample from
a multimodal conditional, so "did the clause move the chunk toward it" measures the
sampler's mode choice as much as it measures the clause, and at these bucket sizes the
mode choice wins. The per-condition MSE against $a^{\star}$ stays in
``metadata_steering.json`` and is off the figures.

**5. Which clause carries it?** The old poles bundled both — quality 5 with no mistake
against quality 1 with a mistake — so a difference could not be attributed. The $2\times2$
separates them: the quality effect $\lVert a^{(q_5,m)}-a^{(q_1,m)}\rVert$ averaged over
the mistake flag, the mistake effect $\lVert a^{(q,\mathrm{T})}-a^{(q,\mathrm{F})}\rVert$
averaged over the pole, and the interaction $\lVert d_{\mathrm{T}}-d_{\mathrm{F}}\rVert$
with $d_m=a^{(q_5,m)}-a^{(q_1,m)}$ — all three against the same seed floor.

**6. Is the speed number read as tempo, and is it its own axis?** Quality and mistake are
judgements about a segment; speed is a claim about the arm. It has a physical reading the
other two do not, so it gets a test the other two cannot take. Write

$$\operatorname{step}(x)=\sqrt{\frac{1}{(T-1)D}\sum_{t,d}\left(x_{t+1,d}-x_{t,d}\right)^{2}}$$

for RMS per-step travel — how fast a chunk moves, with no regard for where it goes — and
$m_k=\operatorname{step}\left(a^{(s_k)}\right)$ for the model's tempo under the clause that
asks for speed $k$. The demonstrations give the curve the model would have to reproduce:

$$M_k=\operatorname{median}\left\{\operatorname{step}(a^{\star}_f)\ :\ k_{\mathrm{true}}(f)=k\right\}$$

over the frames the annotation itself calls speed $k$. Both curves are divided by their own
value at $k=3$, the modal label, so a frame's overall pace drops out and what is compared is
the shape. Two numbers come out of that: the shape match

$$r_f=\operatorname{corr}_{k}\left(m_k(f),\ M_k\right),$$

taken per frame and reported as a median, and the endpoint ratio $m_5/m_3$ read against the
demonstrations' own $M_5/M_3$.

Neither is read against zero. $M_k$ is not monotone, so a model that has learned nothing but
"bigger number, faster" already scores $\operatorname{corr}_k(k, M_k)\approx+0.7$ on the ReBot
curve — that, and not $0$, is the line $r$ has to clear before it says the annotation was
learned rather than the digit. And a frame's $r$ is a correlation over five points, so it
scatters by about $0.5$; the median over frames is the only form worth reading.

$M_k$ is not monotone, and that is the reason to measure it rather than assume it. On the
ReBot roots $M_1 > M_2$: the hybrid label calls a thrashing failed attempt slow — which is
exactly why the duration-only labels were rejected — and thrashing carries high per-step
travel with no progress. A model that has learned the annotation therefore has to reproduce
the U, and $m_5/m_1$ can sit comfortably above 1 while the low end of the scale is backwards.
Hence the anchor at $k=3$ and not at $k=1$.

Both five-level clauses are also just numbers in one sentence, and the labels themselves are
correlated — per root, $r\approx+0.26$ to $+0.49$ between the quality and the speed a segment
is given. So the last question is whether the model keeps them apart at all:

$$\cos\left(u^{q}_f,\ u^{s}_f\right),\qquad u^{q}_f=a^{(q_5)}_f-a^{(q_1)}_f,\qquad
u^{s}_f=a^{(s_5)}_f-a^{(s_1)}_f$$

read against the same cosine between two reseeds of one fixed clause taken from a shared
third draw,

$$\cos\left(a^{(\sigma_1)}-a^{(\sigma_0)},\ a^{(\sigma_2)}-a^{(\sigma_0)}\right).$$

The shared draw is deliberate, not sloppiness. The speed ramp holds quality at 5, so
$a^{(s_5)}$ *is* $a^{(q_5)}$ — the same prompt, aliased — and the two clause axes therefore
meet at that point. Two vectors sharing an endpoint are correlated at roughly $+\tfrac{1}{2}$
before any clause is involved, so a null built from disjoint pairs would hand that structural
offset to the measurement and call it steering. The null carries the same shared point and
the offset cancels. It absorbs a second bias for free: the chunks do not fill their
$T\times D$ box, so even unrelated edits do not land at $0$ on their own. Three flow draws,
which is the code floor. At the null, two clauses mean two directions. Near $+1$, one
direction is wearing two names and the speed label buys nothing the quality label had not
already bought.

Three facts about the labels that decide what these figures can show. Quality is per
*segment*, not per episode, so it varies within an episode and a frame's true level is
well defined. And the levels are not uniform in training: the annotated corpus runs
6.6% at quality 1, 9.4% at 2, 25.0% at 3, 24.9% at 4 and 34.1% at 5 (165,740 frames,
2026-08-02 merge), so asking for quality 1 is a rare prompt and a weak response at the
low end is as much a data statement as a model one. Speed is thinner still at the top: over
the five ReBot training roots (230,438 labelled frames, ``speed_hybrid_v1``) the mix is 3.4%
at speed 1, 25.4% at 2, 40.6% at 3, 24.2% at 4 and 6.5% at 5 — and the clause the rollout
prompt actually sends, quality 5 with no mistake AND speed 5 together, covers 3.35% of
training frames (0.25% on ``rebot_rollouts-annotated-v2``). Asking for speed 5 at deployment
is a tail prompt in a way that asking for quality 5 is not. Which levels the *held-out* frames
actually cover is in the provenance box, and a level missing there has no column in the
conditionality panel.

Cost is ``n_frames x (13 + n_seeds - 1)`` forwards — 12 clause rows, plus ``gt`` where the
frame is labelled — issued as two batched calls per frame: every clause in one, the seed
floor in the other. The clause rows share the frame's images and one flow-noise draw, so
what separates them is the clause alone (batched 2026-08-22). The speed ramp is four extra
rows in that existing batch, not four extra forwards (2026-09-09).

**7. Precision and contact (2026-09-25).** Two more clauses after speed, "The precision is
$N$ of 5." and "The contact is <phrase>.", each swept on its own at the rollout clause
(quality 5, no mistake, speed 5): ``p1`` … ``p5`` is a ramp, ``c0`` … ``c14`` a sweep over the
contact vocabulary (``datasets/contact_vocab.py``; 14 is "not applicable"). Each is measured
against the no-clause chunk for its channel, which is ``q5`` — the same prompt with that
sentence absent — and read against the same seed floor: $\lVert a^{(p_k)}-a^{(q_5)}\rVert$ and
$\lVert a^{(c)}-a^{(q_5)}\rVert$ over the floor. The precision ramp also gets the ordering test of
question 2 ($\pi$ and $\tau$ on the $p_1 \rightarrow p_5$ axis); contact is categorical, so
instead of an order it gets a spread, the mean pairwise distance between the fifteen codes over
the floor — whether the codes differ from each other and not only from the absent sentence.
These twenty rows are one extra batched call per frame on the same seed-0 draw.

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

from lerobot.datasets.contact_vocab import CONTACT_VOCAB
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.utils.action_metrics import TRAJECTORY_RELATIVE_KEYS, trajectory_error_components
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
_SPEED_LEVELS = (1, 2, 3, 4, 5)
# The rollout clause asks for speed 5, so every quality / mistake cell holds speed at
# 5 and the speed ramp holds quality at 5 with no mistake: one axis moves per cell.
_DEPLOYED_SPEED = 5
# Tempo is read as a shape, and a shape needs an anchor. Level 3 is the modal label
# (40.6% of training frames) and the level above which the label is unambiguously
# monotone in per-step travel on every root; level 1 is neither.
_TEMPO_ANCHOR = 3

# The dose axis gets a sequential ramp so a monotone response is visible as a colour
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

# Precision and contact: each channel swept on its own at the rollout clause, so the
# channel's sentence is the only difference from ``q5``, the no-clause chunk each row is
# measured against. Contact is categorical: a sweep over the vocabulary, not a ramp.
_PRECISION_LEVELS = (1, 2, 3, 4, 5)
_CONTACT_CODES = tuple(element.code for element in CONTACT_VOCAB)
_CHANNELS = {f"p{k}": {**_STEERED[_ROLLOUT], "precision": k} for k in _PRECISION_LEVELS}
_CHANNELS |= {f"c{code}": {**_STEERED[_ROLLOUT], "contact": code} for code in _CONTACT_CODES}


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


def _sem(rows: list[dict], key: str) -> float:
    values = _column(rows, key)
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


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


def _pearson(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Correlation of two short curves — the cosine of their centred versions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return _cosine(x - x.mean(), y - y.mean())


def _ratio_se(numerator: np.ndarray, denominator: np.ndarray, n_boot: int = 512) -> float:
    """Bootstrap SE of a ratio of medians taken over two *different* groups of frames.

    The demonstration tempo curve is one median per speed label, so its normalization
    divides one group's median by another's and both sides carry sampling error. Seeded
    per call so the figure is reproducible.
    """
    if numerator.size < 2 or denominator.size < 2:
        return 0.0
    rng = np.random.default_rng(0)
    draws = [
        np.median(rng.choice(values, size=(n_boot, values.size), replace=True), axis=1)
        for values in (numerator, denominator)
    ]
    return float((draws[0] / np.maximum(draws[1], 1e-12)).std())


def _chunk_is_unpadded(dataset, global_idx: int, episode_idx: int, chunk_size: int) -> bool:
    """Is the demonstration chunk real to its last step, rather than repeat-padded?

    `get_frame_data` stops at the episode boundary and repeats the final action to
    length. That freezes the tail and so deflates per-step travel — the one quantity the
    tempo panel measures — and `sample_episodes_evenly` puts a sample on the last frame
    of every episode, so the padded frames are not a rare accident. They are dropped from
    the demonstration curve only; every clause contrast is a within-frame difference and
    is unaffected.
    """
    tail = global_idx + chunk_size - 1
    if tail >= len(dataset):
        return False
    row = dataset.hf_dataset[tail]
    if int(row["episode_index"].item()) != episode_idx:
        return False
    is_pad = row.get("action_is_pad", False)
    return not (is_pad.item() if isinstance(is_pad, torch.Tensor) else is_pad)


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


def _channel_measurements(
    channel_acts: dict[str, torch.Tensor], clause_off: torch.Tensor, base: torch.Tensor, floor_mean: float
) -> dict:
    """Per-frame numbers for the precision ramp and the contact sweep.

    ``clause_off`` is the rollout chunk ``q5`` — the same prompt with the channel's sentence
    absent — and every ``*_clause_*`` number is measured from it; ``*_rmse`` is from ``none``
    like every other condition's. Separations divide by the frame's seed floor.
    """
    floor = max(floor_mean, 1e-9)
    row: dict = {}
    for name, act in channel_acts.items():
        row[f"{name}_rmse"] = _rmse(act, base)
        row[f"{name}_clause_rmse"] = _rmse(act, clause_off)
        row[f"{name}_clause_separation"] = row[f"{name}_clause_rmse"] / floor
    projection, tau = _level_projection(channel_acts, "p")
    row["precision_range_rmse"] = _rmse(channel_acts["p5"], channel_acts["p1"])
    row["precision_kendall_tau"] = tau
    row.update({f"proj_p{k}": value for k, value in projection.items()})
    row["precision_clause_rmse"] = float(np.mean([row[f"p{k}_clause_rmse"] for k in _PRECISION_LEVELS]))
    row["contact_clause_rmse"] = float(np.mean([row[f"c{code}_clause_rmse"] for code in _CONTACT_CODES]))
    row["contact_spread_rmse"], _ = _pairwise_rmse([channel_acts[f"c{code}"] for code in _CONTACT_CODES])
    for key in ("precision_range", "precision_clause", "contact_clause", "contact_spread"):
        row[f"{key}_separation"] = row[f"{key}_rmse"] / floor
    return row


def _provenance(rows: list[dict], dataset, cfg, conditions: list[str], n_seeds: int) -> dict:
    """Which frames of which dataset the figures average, and what one forward was.

    Derived from the measured rows rather than from config, so it describes the frames
    that actually produced numbers after stride snapping and the episode budget dropped
    whatever they dropped. The label mix is the part that decides what the conditionality
    panel can say: a quality level with no frames in this split has no column, and one
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
        else "no quality / mistake labels on this split — ``gt`` and the conditionality panel are unavailable"
    )
    if missing and labelled:
        label_line += (
            "; no frames at quality "
            + "/".join(str(q) for q in missing)
            + f", so {'that level is' if len(missing) == 1 else 'those levels are'} asked for "
            "but never true here — it gets a line in the conditionality panel and no column"
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
                "an explicit metadata ablation would show; active dropout is zero.",
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
                "Scale for every displacement",
                "the frozen-arm chunk — the measured state repeated "
                f"{int(cfg.policy.chunk_size)} times and normalized like a prediction — so "
                "$\\rho$ reads as the fraction of the chunk's own motion the clause rewrote, "
                "floored by ``action_metrics.DEFAULT_SCALE_FLOORS``",
            ],
            [
                "Demonstration",
                f"$a^{{\\star}}$ is the recorded action at the sampled frame and the "
                f"{int(cfg.policy.chunk_size) - 1} that follow it, normalized like the "
                "prediction. It scores no figure — it is one sample of a multimodal "
                "conditional, so an error against it moves with the sampler's mode choice. "
                "The per-condition MSE against it is in ``metadata_steering.json``",
            ],
        ],
        "sampling": (
            "Frames evenly spaced across each episode, snapped onto the image/depth stride "
            f"grid; episodes drawn by a seeded subset when the budget is smaller than the "
            f"split (seed {int(p.random_seed)}). Every distance is a within-frame difference "
            "between two conditions on the same observation, so uneven sampling cannot "
            "manufacture an effect — but the conditionality panel *does* compare label groups "
            "across frames, which is why it is drawn in $\\rho$ rather than raw distance: the "
            "per-frame denominator is what makes two frames comparable at all. Its columns "
            "still inherit whatever the label mix above is. Chunks running past the end of an "
            "episode are repeat-padded with the last recorded action; no figure is scored "
            "against the demonstration, so that padding reaches only the $a^{\\star}$ numbers "
            "kept in the JSON."
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
        ("speed\ns5 vs s1", "speed_range_rmse"),
        ("both poles\n(old good/bad)", "pole_range_rmse"),
        ("flow seed\n(floor)", "seed_floor_mean"),
    ]
    axes[0].boxplot([_column(rows, key) for _label, key in contrasts],
                    tick_labels=[label for label, _key in contrasts])
    axes[0].set_ylabel(r"$\|a^{(c)} - a^{(c')}\|$  (normalized actions)")
    axes[0].set_title(
        "Does the clause move the chunk\nmore than noise does?\n"
        f"median separation $S$: quality {summary['separation_median']:.2f}x   ·   "
        f"mistake {summary['mistake_separation_median']:.2f}x   ·   "
        f"speed {summary['speed_separation_median']:.2f}x"
    )
    _caption(axes[0], [
        r"One frame gives one point to each box. The first four are clause contrasts on that frame,",
        r"the fifth is the same clause ($q_5$, no mistake, speed 5) re-drawn under " + str(summary["n_seeds"]) + r" flow seeds.",
        r"None of the first four has a scale of its own — only the ratio $S$ to the fifth does, and",
        r"$S \approx 1$ says the clause did nothing the sampler was not already doing anyway.",
        r"'speed' holds quality at 5 and no mistake and moves the speed number 1 to 5; its per-step",
        r"motion ratio (speed 5 over speed 1) is in the summary: above 1 means 'fast' reads as moving more.",
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
# Figure 2 — is the response conditional on the frame, and is it one direction?
# ──────────────────────────────────────────────────────────────────────────────

def _render_response(rows: list[dict], mean_axis: np.ndarray, summary: dict, output_path: str) -> None:
    labelled = [row for row in rows if row["gt_quality"] is not None]
    columns = [q for q in _QUALITY_LEVELS if any(row["gt_quality"] == q for row in labelled)]
    buckets = {q: [row for row in labelled if row["gt_quality"] == q] for q in columns}

    fig = plt.figure(figsize=(18, 6.6))
    grid = fig.add_gridspec(1, 3, wspace=0.30, left=0.05, right=0.985, top=0.78, bottom=0.32)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    for q in _QUALITY_LEVELS:
        key = f"q{q}_disp_path"
        medians = [_median(buckets[true_q], key) for true_q in columns]
        axes[0].errorbar(
            columns, medians, yerr=[_median_se(buckets[true_q], key) for true_q in columns],
            marker="o", capsize=3, linewidth=1.8, color=_QUALITY_COLORS[q], label=f"asked {q}",
        )
        # The frames this level is the truth about: where a frame-reading model moves least.
        if q in columns:
            axes[0].plot([q], [medians[columns.index(q)]], marker="o", markersize=13,
                         markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5)
    if "gt_disp_path" in rows[0]:
        axes[0].errorbar(
            columns, [_median(buckets[true_q], "gt_disp_path") for true_q in columns],
            yerr=[_median_se(buckets[true_q], "gt_disp_path") for true_q in columns],
            marker="D", linestyle=":", linewidth=1.8, color=_CONDITION_STYLE["gt"][0], label="gt",
        )
    floor = summary["seed_floor_disp_path"]
    axes[0].axhline(floor, color="#E63946", linestyle="--", linewidth=1.2, label="flow-seed floor")
    # Symlog with the floor as the threshold: decades above it, linear below. A clause that
    # moves nothing lands at zero, which a log axis would drop silently — and the diagonal
    # points, the ones this panel is about, are exactly where that is most likely.
    axes[0].set_yscale("symlog", linthresh=max(floor, 1e-12))
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xticks(columns, [f"{q}\n(n={len(buckets[q])})" for q in columns])
    axes[0].set_xlabel("quality the segment actually is")
    axes[0].set_ylabel(r"$\rho$ — displacement from no clause (units of the chunk's own motion)")
    axes[0].set_title(
        "Does the response depend on the frame,\nor is it the same move everywhere?\n"
        rf"conditionality $C$ = {summary['conditionality_ratio']:.2f}x"
    )
    axes[0].legend(fontsize=7, ncol=3, framealpha=0.9)
    _caption(axes[0], y=-0.30, lines=[
        r"$\rho(q) = \|a^{(q)} - a^{(none)}\|^2 / \|h - a^{(none)}\|^2$: how much of the chunk's own motion the clause",
        r"rewrote, $h$ = the frozen-arm chunk. Per-frame ratio, so unlike an MSE these points are",
        r"comparable between columns. Median over the column's frames, whisker = bootstrap SE of it.",
        r"READ ALONG A LINE, NOT DOWN THE COLUMN: one line is one clause, so its shape is the frame's",
        r"doing. Ringed = the column where that line is the truth. A model that reads the frame dips at",
        r"its own ring and rises with mismatch; flat lines are a constant offset applied blind. Below the",
        r"dashed floor the axis turns linear — that band is unresolved, not small. $C$ is the median over",
        r"levels of (off-diagonal $\rho$) / (diagonal $\rho$), both floored at the seed floor so an effect",
        r"nothing can resolve reads as exactly $1$: $1$ = blind, $> 1$ = the clause is read against the frame.",
    ])

    pole = _column(rows, "pole_cosine")
    shared = _column(rows, "shared_cosine")
    null = _column(rows, "noise_shared_cosine")
    groups = [
        (f"$q_1$ vs $q_5$\nsame frame", pole, "#457B9D"),
        (f"$q_5$ vs other frames\nleave-one-out", shared, "#2A9D8F"),
        (f"flow noise\nsame test (null)", null, "#ADB5BD"),
    ]
    boxes = axes[1].boxplot([values for _label, values, _color in groups],
                            tick_labels=[label for label, _values, _color in groups],
                            patch_artist=True, widths=0.55)
    for patch, (_label, _values, color) in zip(boxes["boxes"], groups, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for offset, (_label, values, _color) in enumerate(groups):
        jitter = np.random.default_rng(0).normal(0.0, 0.045, size=len(values))
        axes[1].plot(offset + 1 + jitter, values, ".", color="#1D3557", markersize=3, alpha=0.35)
    for level in (-1.0, 0.0, 1.0):
        axes[1].axhline(level, color="black", linewidth=0.8,
                        linestyle="-" if level == 0.0 else ":")
    axes[1].set_ylim(-1.08, 1.08)
    axes[1].set_ylabel(r"cosine between displacement vectors $v^{(c)} = a^{(c)} - a^{(none)}$")
    axes[1].set_title(
        "Is the clause one shared vector\nor a per-frame response?\n"
        rf"poles {np.median(pole):+.2f}  ·  shared {np.median(shared):+.2f}  ·  null {np.median(null):+.2f}"
    )
    _caption(axes[1], y=-0.30, lines=[
        r"Each displacement is the $T \times D$ chunk difference flattened; one point per frame.",
        r"Left: do the two poles pull against each other on the same frame? $-1$ = one signed quality",
        r"axis, $0$ = two unrelated edits that both happen to move the chunk. Middle: is a frame's $q_5$",
        r"displacement the mean of all the OTHER frames' (leave-one-out, so no frame flatters itself)?",
        r"$+1$ = one global style offset stamped on every observation, which is what a prior is; $0$ = the",
        r"response is computed from the frame. Right: the same leave-one-out test on the displacement",
        r"between two flow seeds, which shares no direction by construction — the zero this panel needs.",
    ])

    scale = float(np.abs(mean_axis).max()) or 1.0
    image = axes[2].imshow(mean_axis.T, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[2].set_yticks(range(mean_axis.shape[1]), joint_names_for_dim(mean_axis.shape[1]), fontsize=8)
    axes[2].set_xlabel("chunk step")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03).set_label(
        r"mean $a^{(q_5)} - a^{(q_1)}$ (normalized)"
    )
    axes[2].set_title(
        "What the model thinks 'quality 5' is\n"
        rf"shared fraction $R$ = {summary['shared_fraction']:.2f}  ·  "
        rf"per-step motion x{summary['step_motion_ratio']:.3f}"
    )
    _caption(axes[2], y=-0.30, lines=[
        r"$\bar{u} = \mathrm{mean}_f\,(a^{(q_5)}_f - a^{(q_1)}_f)$, the average pole-to-pole edit, per joint and chunk step.",
        r"Red = asking for 5 raises the joint, blue = lowers it. A vertical band means the edit is a",
        r"posture offset held for the whole chunk; a ramp toward the right means it changes where the",
        r"chunk ends up. $R = \|\bar{u}\|^2 / \mathrm{mean}_f \|u_f\|^2$ is how much of a typical frame's edit this",
        r"average keeps: near $1$ the map IS the edit, near $0$ the frames disagree and the map is the",
        r"residue of cancellation — read it with the middle panel, not alone. The motion ratio is",
        r"RMS per-step travel under $q_5$ over $q_1$: above 1, asking for quality 5 means moving more.",
    ])

    fig.suptitle(
        f"Metadata steering — what the clause does to the chunk "
        f"({len(labelled)} labelled frames, {summary['n_frames']} total)",
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
        keys = [f"q{q}{'m' if mistake else ''}_disp_path" for q in _MISTAKE_POLES]
        axes[0].errorbar(
            _MISTAKE_POLES, [_median(rows, key) for key in keys],
            yerr=[_median_se(rows, key) for key in keys],
            marker=marker, color=color, linewidth=2.0, capsize=3,
            label="mistake sentence on" if mistake else "no mistakes",
        )
    axes[0].axhline(summary["seed_floor_disp_path"], color="#E63946", linestyle="--",
                    linewidth=1.2, label="flow-seed floor")
    axes[0].set_xticks(list(_MISTAKE_POLES), [f"quality {q}" for q in _MISTAKE_POLES])
    axes[0].set_ylabel(r"$\rho$ — displacement from no clause")
    axes[0].set_title("The 2x2, as an interaction plot")
    axes[0].legend(fontsize=8)
    _caption(axes[0], [
        r"Four cells: quality $\in \{1, 5\}$ crossed with the mistake sentence on/off, each cell the median",
        r"$\rho$ of figure 2 — how much of the chunk's own motion that clause rewrote. Parallel lines mean the",
        r"two clauses act independently: whatever the mistake sentence does, it does the same at either",
        r"pole. Crossing lines mean the model reads them jointly (a mistake at quality 5 is a different",
        r"situation from a mistake at quality 1), which is what the annotation actually means. The vertical",
        r"gap is the mistake sentence's effect, the slope is quality's, and both are sizes, not verdicts.",
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


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — is the speed number read as tempo, and is it its own axis?
# ──────────────────────────────────────────────────────────────────────────────

def _render_speed(rows: list[dict], summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(17.5, 6.6))
    grid = fig.add_gridspec(1, 3, wspace=0.28, left=0.05, right=0.985, top=0.77, bottom=0.34)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    # ── Left: is the ramp ordered in chunk space? (the speed twin of figure 1's right) ──
    for row in rows:
        axes[0].plot(_SPEED_LEVELS, [row[f"proj_s{k}"] for k in _SPEED_LEVELS],
                     color="#7209B7", alpha=0.12, linewidth=0.8)
    axes[0].errorbar(
        _SPEED_LEVELS, [_mean(rows, f"proj_s{k}") for k in _SPEED_LEVELS],
        yerr=[_sem(rows, f"proj_s{k}") for k in _SPEED_LEVELS],
        color="#3A0CA3", linewidth=2.2, marker="o", capsize=3, label="mean over frames",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(list(_SPEED_LEVELS))
    axes[0].set_xlabel("speed asked for")
    axes[0].set_ylabel(r"$\pi(k)$ — position on the $s_1 \rightarrow s_5$ axis")
    axes[0].set_title(
        "Is the speed ramp ordered at all?\n"
        rf"{summary['speed_separation_median']:.2f}x the flow-seed floor"
        "\n"
        rf"mean $\tau$ = {summary['speed_kendall_tau_mean']:+.2f}"
    )
    axes[0].legend(fontsize=8)
    _caption(axes[0], y=-0.32, lines=[
        r"Figure 1's right panel, for speed: quality held at 5, mistake sentence off, so the",
        r"speed number is the only thing moving. Per frame $u = a^{(s_5)} - a^{(s_1)}$, and each",
        r"level projected onto it: $\pi(k) = \langle a^{(s_k)} - \bar{a},\, u \rangle / \|u\|$.",
        r"The ends are pinned ($\pi(5) - \pi(1) = \|u\|$); what is measured is where 2, 3, 4 land.",
        r"Read the separation first — a ramp near 1x sits inside the sampler's own noise and",
        r"the other two panels are moot.",
        r"This says the five strings are ordered IN CHUNK SPACE. It does not say they are",
        r"ordered as TEMPO. That is the middle panel, and the two can disagree.",
    ])

    # ── Middle: does the number mean what the annotation means by it? ──
    levels = sorted(int(k) for k in summary["speed_demo_tempo"])
    if levels:
        axes[1].errorbar(
            levels, [summary["speed_demo_tempo"][str(k)] for k in levels],
            yerr=[summary["speed_demo_tempo_se"][str(k)] for k in levels],
            color="black", marker="D", linewidth=2.2, capsize=3,
            label=rf"demonstration $M_k / M_{_TEMPO_ANCHOR}$",
        )
    axes[1].errorbar(
        _SPEED_LEVELS, [_median(rows, f"tempo_s{k}") for k in _SPEED_LEVELS],
        yerr=[_median_se(rows, f"tempo_s{k}") for k in _SPEED_LEVELS],
        color="#F72585", marker="o", linestyle="--", linewidth=2.2, capsize=3,
        label=rf"model $m_k / m_{_TEMPO_ANCHOR}$",
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    counts = summary["speed_demo_n"]
    axes[1].set_xticks(list(_SPEED_LEVELS),
                       [f"{k}\n(n={counts.get(str(k), 0)})" for k in _SPEED_LEVELS])
    axes[1].set_xlabel("speed level")
    axes[1].set_ylabel(rf"per-step travel, relative to level {_TEMPO_ANCHOR}")
    axes[1].set_title(
        "Does the number mean what the\nannotation means by it?\n"
        rf"shape match $r$ = {summary['speed_tempo_correlation']:+.2f}   "
        rf"(monotone reader {summary['speed_tempo_correlation_monotone']:+.2f})"
        "\n"
        rf"$m_5/m_3$ = {summary['speed_tempo_ratio_5_3']:.2f}   "
        rf"(demonstration {summary['speed_tempo_ratio_5_3_demo']:.2f})"
    )
    axes[1].legend(fontsize=8)
    _caption(axes[1], y=-0.32, lines=[
        r"Per-step travel: the RMS of $x_{t+1} - x_t$ over chunk steps and joints — how fast a",
        r"chunk moves, with no regard for where it goes. Each curve is divided by its own",
        r"value at level 3, so a frame's overall pace cancels and only the SHAPE is compared.",
        r"Pink: five chunks from the SAME frame, one per speed clause; median of $m_k/m_3$.",
        r"Black: the RECORDED chunk on the frames the annotation calls speed $k$, $M_k/M_3$ —",
        r"a different set of frames at every point, the only cross-frame comparison in this",
        r"probe, hence the counts on the axis. Chunks repeat-padded past the end of an episode",
        r"are excluded: their tail is frozen and would read as slow.",
        r"THE BLACK CURVE IS NOT MONOTONE. The hybrid label calls thrashing slow, so $M_1 > M_2$",
        r"on most roots. Matching the annotation means matching that U; a pink curve rising",
        r"straight through is reading 'a bigger number', which is what the monotone-reader",
        r"score in the title is — the line $r$ must clear, well above zero.",
        r"One frame's $r$ is over five points and scatters by ~0.5. Read the median, not a frame.",
    ])

    # ── Right: are quality and speed two directions, or one? ──
    cosines = _column(rows, "quality_speed_cosine")
    null = _column(rows, "quality_speed_null_cosine")
    groups = [(r"$u^q$ vs $u^s$" + "\nquality axis vs speed axis", cosines, "#F72585")]
    if null.size:
        groups.append((r"reseed vs reseed" + "\nshared draw (null)", null, "#ADB5BD"))
    boxes = axes[2].boxplot([values for _label, values, _color in groups],
                            tick_labels=[label for label, _values, _color in groups],
                            patch_artist=True, widths=0.5)
    for patch, (_label, _values, color) in zip(boxes["boxes"], groups, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for offset, (_label, values, _color) in enumerate(groups):
        jitter = np.random.default_rng(0).normal(0.0, 0.045, size=len(values))
        axes[2].plot(offset + 1 + jitter, values, ".", color="#1D3557", markersize=3, alpha=0.35)
    for level in (-1.0, 0.0, 1.0):
        axes[2].axhline(level, color="black", linewidth=0.8,
                        linestyle="-" if level == 0.0 else ":")
    axes[2].set_ylim(-1.08, 1.08)
    axes[2].set_ylabel(r"cosine between the two clause axes")
    axes[2].set_title(
        "Two clauses, or one number twice?\n"
        rf"cosine {summary['quality_speed_cosine_median']:+.2f}   "
        rf"(null {summary['quality_speed_null_cosine_median']:+.2f})"
        "\n"
        rf"labels' own $r$ = {summary['quality_speed_label_correlation']:+.2f}"
    )
    _caption(axes[2], y=-0.32, lines=[
        r"Per frame, the cosine between the quality axis $u^q = a^{(q_5)} - a^{(q_1)}$ and the",
        r"speed axis $u^s = a^{(s_5)} - a^{(s_1)}$, each flattened over chunk step and joint.",
        r"Both contain $a^{(q_5)}$: the speed ramp holds quality at 5, so $s_5$ and $q_5$ are the",
        r"same prompt. Two vectors sharing an endpoint correlate at roughly $+0.5$ before any",
        r"clause is involved, so the null is built to share one the same way — two reseeds of",
        r"one fixed clause, both measured off a shared third draw.",
        r"READ THE GAP, NOT THE HEIGHT.",
        r"At the null: two clauses, two directions, speed doing its own work. Well above it:",
        r"one direction wearing two names, and speed buys nothing quality had not bought.",
        r"The labels' own $r$ in the title is context — a correlation between two integers, not",
        r"a cosine between two chunks, so it is not a line this box is meant to sit on.",
    ])

    fig.suptitle(
        f"Metadata steering — the speed clause (n={summary['n_frames']} frames, "
        f"{sum(summary['speed_demo_n'].values())} with an unpadded demonstration)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — do the precision and contact clauses reach the actions?
# ──────────────────────────────────────────────────────────────────────────────

def _render_channels(rows: list[dict], summary: dict, output_path: str) -> None:
    fig = plt.figure(figsize=(18, 6.4))
    grid = fig.add_gridspec(1, 3, wspace=0.26, left=0.05, right=0.985, top=0.80, bottom=0.30)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]

    # ── Left: every contrast against the same seed floor ──
    contrasts = [
        ("precision\np5 vs p1", "precision_range_rmse"),
        ("precision\nvs no clause", "precision_clause_rmse"),
        ("contact\nvs no clause", "contact_clause_rmse"),
        ("contact\ncode vs code", "contact_spread_rmse"),
        ("flow seed\n(floor)", "seed_floor_mean"),
    ]
    axes[0].boxplot([_column(rows, key) for _label, key in contrasts],
                    tick_labels=[label for label, _key in contrasts])
    axes[0].set_ylabel(r"$\|a^{(c)} - a^{(c')}\|$  (normalized actions)")
    axes[0].set_title(
        "Does either clause move the chunk\nmore than noise does?\n"
        f"precision {summary['precision_separation_median']:.2f}x   ·   "
        f"contact {summary['contact_clause_separation_median']:.2f}x   ·   "
        f"contact spread {summary['contact_spread_separation_median']:.2f}x"
    )
    _caption(axes[0], [
        r"One point per frame per box. 'vs no clause' is the mean over levels (codes) of the distance",
        r"from $a^{(q_5)}$, the rollout prompt with that sentence absent; 'code vs code' is the mean",
        r"pairwise distance between the fifteen contact chunks. Only the ratio to the floor means anything.",
    ])

    # ── Middle: is the precision ramp ordered? ──
    for row in rows:
        axes[1].plot(_PRECISION_LEVELS, [row[f"proj_p{k}"] for k in _PRECISION_LEVELS],
                     color="#457B9D", alpha=0.12, linewidth=0.8)
    axes[1].errorbar(
        _PRECISION_LEVELS, [_mean(rows, f"proj_p{k}") for k in _PRECISION_LEVELS],
        yerr=[_sem(rows, f"proj_p{k}") for k in _PRECISION_LEVELS],
        color="#1D3557", linewidth=2.2, marker="o", capsize=3, label="mean over frames",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(list(_PRECISION_LEVELS))
    axes[1].set_xlabel("precision asked for")
    axes[1].set_ylabel(r"$\pi(p)$ — position on the $p_1 \rightarrow p_5$ axis")
    axes[1].set_title(
        "Is the precision ramp ordered?\n"
        rf"mean $\tau$ = {summary['precision_kendall_tau_mean']:+.2f}"
    )
    axes[1].legend(fontsize=8)
    _caption(axes[1], [
        r"Figure 1's right panel for precision: quality 5, no mistake, speed 5 held, only the",
        r"precision number moving. Read the separation on the left first.",
    ])

    # ── Right: which contact codes move the chunk? ──
    x = np.arange(len(_CONTACT_CODES))
    axes[2].bar(x, [summary["contact_clause_separation"][str(code)] for code in _CONTACT_CODES],
                color="#6D597A")
    axes[2].axhline(1.0, color="#E63946", linestyle="--", linewidth=1.2, label="flow-seed floor")
    axes[2].set_xticks(x, [element.slug for element in CONTACT_VOCAB], rotation=60, ha="right")
    axes[2].set_ylabel(r"median $\|a^{(c)} - a^{(q_5)}\|$ / floor")
    axes[2].set_title("Per contact code: distance from no clause,\nin units of the seed floor")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Metadata steering — precision and contact (n={summary['n_frames']} frames)",
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

_RESPONSE_HOW = r"""Nothing here is scored against the demonstration. $a^{\star}$ is one
draw from a multimodal conditional, so an error against it moves with which mode the
sampler picked, and at these bucket sizes that swamps any clause. What is measurable
without it is the *displacement* the clause causes — its size, whether that size depends
on the frame, and what direction it points.

Everything is expressed as $\rho$: the squared displacement from the no-clause chunk
divided by that chunk's own squared motion away from a frozen arm, floored by the
``action_metrics`` scale floors. $\rho = 1$ means the clause rewrote as much trajectory
as the trajectory had. Being a per-frame ratio, it is comparable *between* frames, which
is what lets the left panel compare label groups at all.

**Left — is the response conditional?** One line per asked level, drawn across the true
level of the segment. Read **along a line**: the clause is fixed there, so any shape is
the frame's doing. The ring on each line marks the column where that level is the truth.
A model that reads the frame moves least at its own ring and more as the mismatch grows;
a clause applied blind gives flat lines and $C \approx 1$. Between-line offsets are the
dose from figure 1, not conditionality.

**Middle — one vector or many?** A prior is a shared vector, so this is the direct test.
$q_1$ against $q_5$ within a frame near $-1$ means the levels sit on one signed axis.
$q_5$ against the leave-one-out mean of every other frame near $+1$ means the identical
edit is stamped on every observation regardless of what it shows; near $0$ means the
response is computed per frame. The flow-noise column is the same statistic on a
displacement with no shared direction by construction — the zero the other two are read
against.

**Right — what that vector is.** The mean $q_1 \rightarrow q_5$ edit per joint and chunk
step. $R$ says how much of a typical frame's edit survives the average, so a low $R$
means the map is cancellation residue and only the middle panel is load-bearing. The
per-step motion ratio is the one-line physical reading: above 1, "quality 5" means the
model moves the arm more."""

_FACTORIAL_HOW = r"""The old ``good``/``bad`` contrast moved both clauses at once. This
page separates them, with $d_m = a^{(q_5,m)} - a^{(q_1,m)}$ the quality response at a
fixed mistake flag $m$.

**Left — interaction plot.** The four cells of the $2\times2$ as displacement from the
no-clause chunk, in the $\rho$ units of figure 2. Parallel lines mean the two clauses act
independently; crossing lines mean the model reads them jointly, which is what the
annotation means (a mistake inside a quality-5 segment is a different situation from a
mistake inside a quality-1 one).

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


_SPEED_HOW = r"""Quality and mistake are judgements about a segment. Speed is a claim
about the arm, so unlike the other two it can be checked against the arm — that is what this
page does, plus one question the other pages cannot ask: whether the two five-level numbers
in the clause are two numbers or one.

**Left — is the ramp ordered?** The speed twin of figure 1's right panel. Each level is
projected onto the axis its own poles define,
$\pi(k)=\langle a^{(s_k)}-\bar a,\,u\rangle/\lVert u\rVert$ with $u=a^{(s_5)}-a^{(s_1)}$ and
quality pinned at 5 throughout, so the only thing moving is the speed number. Read the
separation in the title before the shape: a ramp that sits inside the sampler's own noise
makes the rest of this page moot. And ordering here is ordering *in chunk space* — it says
the five strings are distinguishable and sorted, not that they mean anything about pace.

**Middle — is that order tempo?** Per-step travel
$\operatorname{step}(x)=\sqrt{\frac{1}{(T-1)D}\sum_{t,d}(x_{t+1,d}-x_{t,d})^{2}}$ measures how
fast a chunk moves without caring where it goes. The model gives $m_k$ under each speed
clause; the demonstrations give $M_k$, the median $\operatorname{step}(a^{\star})$ over the
frames the annotation calls speed $k$. Both are divided by their level-3 value so a frame's
overall pace cancels and only the five-point shape is compared, and the headline $r$ is the
correlation between the two shapes, taken per frame and reported as a median.

Read $r$ against the number beside it, not against zero. Because $M_k$ is not monotone, a model
that has learned only "bigger number, faster" already scores the correlation between a straight
ramp and $M_k$ — about $+0.7$ on the ReBot curve. Below that line the clause is not being read
as tempo at all; at it, the model is reading the digit rather than the annotation; only above it
has it learned what this label means. Each frame contributes a correlation over five points, so
one frame's $r$ scatters by roughly $0.5$ and only the median over frames carries information.

Two asymmetries are worth knowing rather than glossing. The model's ratio is formed *within*
a frame and then medianed; the demonstration's can only be a ratio of medians over two
different groups of frames, because no frame has two true speeds — which also means the black
curve carries whatever else differs between speed-1 segments and speed-5 segments, not tempo
alone. And $M_k$ is not monotone: the hybrid label calls a thrashing failed attempt slow, so
$M_1>M_2$ on most roots. A model that has learned this annotation has to reproduce that U.
A pink curve rising straight through is reading "a bigger number", not this label.

**Right — two clauses or one?** The cosine between the quality axis $u^q=a^{(q_5)}-a^{(q_1)}$
and the speed axis $u^s=a^{(s_5)}-a^{(s_1)}$. Both contain $a^{(q_5)}$, because the speed ramp
holds quality at 5 and so $s_5$ and $q_5$ are character-for-character the same prompt; two
vectors sharing an endpoint sit near $+\frac{1}{2}$ before any clause is involved. The null is
therefore built to share an endpoint in the same way — two reseeds of one fixed clause, both
measured off a shared third draw — so that offset lands in both boxes and cancels. Read the
gap between the boxes, never the height of one. The labels' own correlation is printed as
context; it is a correlation between two integers rather than a cosine between two chunks, so
it is not a line the box is meant to sit on."""


_CHANNELS_HOW = r"""Precision and contact are swept one at a time at the rollout clause
(quality 5, no mistake, speed 5), so the only difference between a row and $a^{(q_5)}$ is the one
sentence. That makes $a^{(q_5)}$ the no-clause chunk for both channels, and every distance on the
left is read against the same flow-seed floor as figure 1.

**Left — the floor.** Precision range $\lVert a^{(p_5)}-a^{(p_1)}\rVert$; each channel's mean
distance from the chunk without its sentence; and the contact spread, the mean pairwise distance
between the fifteen code chunks, which separates "any contact sentence moves the chunk" from "the
codes mean different things". A checkpoint trained before these clauses existed should sit at 1.

**Middle — ordering.** The precision ramp projected onto its own $p_1 \rightarrow p_5$ axis, exactly
as figure 1 does for quality.

**Right — per code.** The median distance from no clause for each contact code, over the floor."""


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
    if n_seeds < 3:
        logging.warning(
            f"[metadata_steering] n_seeds={n_seeds}: the quality/speed null cosine needs 3 flow "
            "draws and will be nan; set metadata_steering_n_seeds >= 3."
        )

    gt_metadata = frame_metadata_lookup(dataset)
    if not gt_metadata:
        logging.warning(
            "[metadata_steering] no meta/episode_metadata.parquet + mistakes.parquet on "
            f"{dataset.root} — no ``gt`` condition, and the response figure needs the true "
            "level to draw against, so it is skipped whole; the steering range and the "
            "factorial still run."
        )

    adapter._set_probe_cuda_graph_enabled(False)  # prompt changes per condition; keep eager
    samples = sample_episodes_evenly(
        dataset, n_frames, p.max_episodes, p.random_seed, probe_image_stride(cfg)
    )
    logging.info(
        f"[metadata_steering] {len(samples)} frames x ({len(_STEERED) + 1} clauses + "
        f"{len(_CHANNELS)} precision/contact + {n_seeds - 1} reseeds + gt) = "
        f"~{len(samples) * (len(_STEERED) + len(_CHANNELS) + n_seeds + 1)} forward passes"
    )

    rows: list[dict] = []
    diagnostics: list[dict] = []
    geometry: list[dict] = []
    try:
        for episode_idx, frame_idx, global_idx in samples:
            frame = probe_frame_inputs(dataset, cfg, global_idx, chunk_size, metadata=None)
            gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])
            # The frozen-arm chunk: the scale every displacement below is divided by, so a
            # frame the arm barely moves in cannot dominate an average of ratios.
            hold_raw = frame["state"][: frame["gt_actions"].shape[-1]].unsqueeze(0).repeat(
                frame["gt_actions"].shape[0], 1
            )
            hold_norm = adapter.normalize_gt_actions(hold_raw, frame["state"]).float()
            labels = gt_metadata.get(global_idx)

            # Every clause in one forward, all sharing seed 0's noise draw exactly as
            # the per-clause loop did — the contrast is the clause, so a per-row noise
            # slice would fold the draw into it. Batched 2026-08-22.
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
            if n_seeds > 1:
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

            # Precision ramp and contact sweep: a batch of their own on the same seed-0
            # draw (``flow_noise_like`` replicates one draw across rows), so each row
            # differs from ``q5`` by its channel's sentence alone. Kept out of ``acts``:
            # every figure and gt number above is about the three older clauses.
            _, channel_chunks = adapter.predict_action_chunk_batch(
                frame["obs"], frame["task"], [frame["subtask"]] * len(_CHANNELS),
                metadatas=list(_CHANNELS.values()),
                noise=adapter.flow_noise_like(len(_CHANNELS), 0),
            )
            channel_acts = {name: channel_chunks[i] for i, name in enumerate(_CHANNELS)}

            projection, tau = _quality_projection(acts)
            speed_projection, speed_tau = _level_projection(acts, "s")
            base = acts["none"]

            def displacement(moved: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
                """How far ``moved`` sits from ``reference``, in units of the chunk's own motion.

                ``trajectory_error_components`` with the reference chunk in the target slot:
                the ``_relative`` terms are then the displacement over the frozen-arm
                baseline's own distance from that same reference.
                """
                components = trajectory_error_components(
                    moved.float(), reference.float(), hold_norm
                )
                return {key.removesuffix("_relative"): float(components[key])
                        for key in TRAJECTORY_RELATIVE_KEYS}

            # A repeat-padded chunk has a frozen tail, which is exactly what tempo reads.
            unpadded = _chunk_is_unpadded(dataset, global_idx, episode_idx, chunk_size)
            gt_mse = {name: float((act - gt_norm).pow(2).mean()) for name, act in acts.items()}
            quality_effect = [_rmse(acts[f"q5{m}"], acts[f"q1{m}"]) for m in ("", "m")]
            mistake_effect = [_rmse(acts[f"q{q}m"], acts[f"q{q}"]) for q in _MISTAKE_POLES]

            row = {
                "global_idx": int(global_idx),
                "episode_idx": int(episode_idx),
                "frame_idx": int(frame_idx),
                "gt_quality": None if labels is None else int(labels["quality"]),
                "gt_mistake": bool(labels["mistake"]) if labels is not None else False,
                "gt_speed": None if labels is None else int(labels["speed"]),
                "gt_precision": (
                    None if labels is None or "precision" not in labels else int(labels["precision"])
                ),
                "gt_contact": None if labels is None or "contact" not in labels else int(labels["contact"]),
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
                "speed_range_rmse": _rmse(acts["s5"], acts["s1"]),
                "speed_kendall_tau": speed_tau,
                **{f"proj_s{k}": value for k, value in speed_projection.items()},
                # Tempo: RMS per-step travel under each speed clause, and the
                # demonstration's own, which is the curve those five have to match.
                **{f"step_s{k}": _step_rms(acts[f"s{k}"]) for k in _SPEED_LEVELS},
                "gt_step_rms": _step_rms(gt_norm) if unpadded else None,
            }
            row["separation"] = row["quality_range_rmse"] / max(floor_mean, 1e-9)
            row["mistake_separation"] = row["mistake_flip_rmse"] / max(floor_mean, 1e-9)
            row["speed_separation"] = row["speed_range_rmse"] / max(floor_mean, 1e-9)
            row.update(_channel_measurements(channel_acts, acts[_ROLLOUT], acts["none"], floor_mean))
            row["step_motion_ratio"] = _step_rms(acts["q5"]) / max(_step_rms(acts["q1"]), 1e-9)
            # The physical reading of the speed number: does asking for 5 move the arm more per step than 1?
            row["speed_step_motion_ratio"] = _step_rms(acts["s5"]) / max(_step_rms(acts["s1"]), 1e-9)
            # Two five-level numbers in one sentence: do they move the chunk in different
            # directions? The speed ramp holds quality at 5, so ``s5`` IS ``q5`` and the two
            # axes meet there — hence a null of two reseeds measured from a shared third
            # draw, carrying the same shared endpoint rather than pretending it away.
            quality_axis = (acts["q5"] - acts["q1"]).flatten().float().cpu().numpy()
            speed_axis = (acts["s5"] - acts["s1"]).flatten().float().cpu().numpy()
            row["quality_speed_cosine"] = _cosine(quality_axis, speed_axis)
            row["quality_speed_null_cosine"] = _cosine(
                (floor_draws[1] - floor_draws[0]).flatten().float().cpu().numpy(),
                (floor_draws[2] - floor_draws[0]).flatten().float().cpu().numpy(),
            ) if len(floor_draws) >= 3 else None
            # The floor in the same relative units: the rollout clause against itself under a
            # different seed. Its denominator is that chunk's excursion rather than
            # ``none``'s, which differ by exactly the effect being measured — second order.
            row["seed_floor_disp_path"] = float(np.mean(
                [displacement(draw, acts[_ROLLOUT])["path"] for draw in floor_draws[1:]]
            ))
            for name, act in acts.items():
                if name == f"s{_DEPLOYED_SPEED}":
                    continue  # alias of q5; its numbers are q5's
                row[f"{name}_gt_mse"] = gt_mse[name]
                if name != "none":
                    row[f"{name}_rmse"] = _rmse(act, base)
                    row[f"{name}_maxabs"] = float((act - base).abs().max())
                    row[f"{name}_gt_mse_improvement"] = gt_mse["none"] - gt_mse[name]
                    for term, value in displacement(act, base).items():
                        row[f"{name}_disp_{term}"] = value

            poles = {q: (acts[f"q{q}"] - base).flatten().float().cpu().numpy() for q in (1, 5)}
            row["pole_cosine"] = _cosine(poles[1], poles[5])
            rows.append(row)
            diagnostics.append({"gt": gt_norm, "acts": acts, "row": row})
            geometry.append({
                "q5": poles[5],
                # Same displacement measured across a reseed instead of across a clause:
                # the null the leave-one-out alignment is read against.
                "noise": (floor_draws[1] - floor_draws[0]).flatten().float().cpu().numpy(),
                "axis": (acts["q5"] - acts["q1"]).float().cpu().numpy(),
            })
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    if not rows:
        logging.warning("[metadata_steering] no frames produced measurements.")
        return

    conditions = ["none"] + list(_STEERED) + (["gt"] if "gt_gt_mse" in rows[0] else [])
    speed_labelled = [row for row in rows if row["gt_speed"] is not None]
    labelled = [row for row in rows if row["gt_quality"] is not None]
    flagged = [row for row in rows if row["gt_mistake"]]

    # Direction geometry. Leave-one-out, so a frame is never part of the reference it is
    # scored against, and the same statistic on the reseed displacement as the null.
    # Both alignment lists are either as long as ``rows`` or empty (a single frame has no
    # "every other frame" to be compared against), so the pairing is not strict.
    for row, shared, null in zip(
        rows,
        _loo_alignment([g["q5"] for g in geometry]),
        _loo_alignment([g["noise"] for g in geometry]),
        strict=False,
    ):
        row["shared_cosine"] = shared
        row["noise_shared_cosine"] = null

    axis_stack = np.stack([g["axis"] for g in geometry])
    mean_axis = axis_stack.mean(axis=0)
    shared_fraction = float(
        (mean_axis**2).sum() / max(float((axis_stack**2).sum(axis=(1, 2)).mean()), 1e-12)
    )

    # Conditionality: for each level that some segment truly is, how much more the clause
    # moves the frames it is wrong about than the frames it is right about. Both sides are
    # floored at the seed floor, below which no displacement is resolvable, so a clause with
    # no measurable effect reads as exactly 1 (blind) instead of diverging on a near-zero
    # denominator or — flooring one side only — landing under 1 and inviting the reverse
    # reading. Only differences the sampler's own noise cannot explain move C off 1.
    seed_floor_disp = _median(rows, "seed_floor_disp_path")
    true_levels = [q for q in _QUALITY_LEVELS if any(row["gt_quality"] == q for row in labelled)]
    conditionality = [
        max(_median([row for row in labelled if row["gt_quality"] != q], f"q{q}_disp_path"),
            seed_floor_disp)
        / max(_median([row for row in labelled if row["gt_quality"] == q], f"q{q}_disp_path"),
              seed_floor_disp)
        for q in true_levels
    ]

    # ── Speed as tempo ──────────────────────────────────────────────────────────
    # The demonstration curve: median per-step travel of the RECORDED chunk over the
    # frames the annotation calls speed k. Unlike every clause contrast above, this
    # compares different frames with each other — the only form the demonstration comes
    # in, since a frame has exactly one true speed. Padded chunks are already excluded.
    demo_step = {
        k: _column([row for row in speed_labelled if row["gt_speed"] == k], "gt_step_rms")
        for k in _SPEED_LEVELS
    }
    demo_curve = {k: float(np.median(v)) for k, v in demo_step.items() if v.size}
    tempo_levels = sorted(demo_curve)
    for row in rows:
        for k in _SPEED_LEVELS:
            row[f"tempo_s{k}"] = row[f"step_s{k}"] / max(row[f"step_s{_TEMPO_ANCHOR}"], 1e-12)
        # Shape match: the model's five tempi against the demonstrations' five, over the
        # levels this split actually has. Two points always correlate perfectly, so a
        # split with fewer than three populated levels reports nothing rather than +1.
        row["speed_tempo_r"] = _pearson(
            [row[f"step_s{k}"] for k in tempo_levels], [demo_curve[k] for k in tempo_levels]
        ) if len(tempo_levels) >= 3 else None
    demo_anchored = _TEMPO_ANCHOR in demo_curve
    # What a model that has learned only "bigger number, faster" scores against this split's
    # curve: a straight ramp correlated with it. Because $M_k$ is not monotone that sits well
    # above zero — on the ReBot curve it is about +0.7 — so it, and not 0, is the line the
    # measured correlation has to clear before it means the model read this annotation.
    monotone_reference = _pearson(
        tempo_levels, [demo_curve[k] for k in tempo_levels]
    ) if len(tempo_levels) >= 3 else float("nan")

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
        "speed_range_rmse": _mean(rows, "speed_range_rmse"),
        "speed_separation_median": float(np.median(_column(rows, "speed_separation"))),
        "speed_kendall_tau_mean": _mean(rows, "speed_kendall_tau"),
        "speed_step_motion_ratio": _median(rows, "speed_step_motion_ratio"),
        "speed_mix": {
            str(k): sum(row["gt_speed"] == k for row in speed_labelled) for k in _SPEED_LEVELS
        },
        "speed_tempo_correlation": _median(rows, "speed_tempo_r"),
        "speed_tempo_correlation_monotone": monotone_reference,
        "speed_tempo_ratio_5_3": _median(rows, "tempo_s5"),
        "speed_tempo_ratio_5_3_demo": (
            demo_curve[5] / demo_curve[_TEMPO_ANCHOR]
            if demo_anchored and 5 in demo_curve else float("nan")
        ),
        "speed_demo_step_rms": {str(k): value for k, value in demo_curve.items()},
        "speed_demo_n": {str(k): int(demo_step[k].size) for k in _SPEED_LEVELS},
        "speed_demo_tempo": {
            str(k): demo_curve[k] / demo_curve[_TEMPO_ANCHOR] for k in tempo_levels
        } if demo_anchored else {},
        "speed_demo_tempo_se": {
            str(k): _ratio_se(demo_step[k], demo_step[_TEMPO_ANCHOR]) for k in tempo_levels
        } if demo_anchored else {},
        "quality_speed_cosine_median": _median(rows, "quality_speed_cosine"),
        "quality_speed_null_cosine_median": _median(rows, "quality_speed_null_cosine"),
        # Context for the cosine, not a threshold for it: how correlated the two labels
        # are on these very frames. A model can only separate what the annotation did.
        "quality_speed_label_correlation": _pearson(
            [row["gt_quality"] for row in labelled], [row["gt_speed"] for row in labelled]
        ) if len(labelled) > 1 else float("nan"),
        "kendall_tau_mean": _mean(rows, "kendall_tau"),
        "monotone_fraction": float(np.mean([row["monotone"] for row in rows])),
        "seed_floor_disp_path": seed_floor_disp,
        "conditionality_ratio": float(np.median(conditionality)) if conditionality else float("nan"),
        "shared_fraction": shared_fraction,
        "step_motion_ratio": _median(rows, "step_motion_ratio"),
        "pole_cosine_median": _median(rows, "pole_cosine"),
        "shared_cosine_median": _median(rows, "shared_cosine"),
        "noise_cosine_median": _median(rows, "noise_shared_cosine"),
        "quality_mix": {
            str(q): sum(row["gt_quality"] == q for row in labelled) for q in _QUALITY_LEVELS
        },
        "precision_separation_median": _median(rows, "precision_range_separation"),
        "precision_clause_separation_median": _median(rows, "precision_clause_separation"),
        "precision_kendall_tau_mean": _mean(rows, "precision_kendall_tau"),
        "precision_clause_separation": {
            str(k): _median(rows, f"p{k}_clause_separation") for k in _PRECISION_LEVELS
        },
        "precision_mix": {
            str(k): sum(row["gt_precision"] == k for row in rows) for k in _PRECISION_LEVELS
        },
        "contact_clause_separation_median": _median(rows, "contact_clause_separation"),
        "contact_spread_separation_median": _median(rows, "contact_spread_separation"),
        "contact_clause_separation": {
            str(code): _median(rows, f"c{code}_clause_separation") for code in _CONTACT_CODES
        },
        "contact_mix": {
            str(code): sum(row["gt_contact"] == code for row in rows) for code in _CONTACT_CODES
        },
        "verdict_note": (
            "separation ~1 => the quality clause moves the chunk no more than flow noise does, "
            "and nothing downstream of it means anything. separation >> 1 with "
            "conditionality_ratio ~1 and shared_cosine_median ~+1 => the clause is read, but as "
            "one global offset stamped on every frame rather than as a description of this one."
        ),
    }
    for name in conditions:
        summary[f"{name}_gt_mse"] = _mean(rows, f"{name}_gt_mse")
        if name != "none":
            summary[f"{name}_rmse"] = _mean(rows, f"{name}_rmse")
            summary[f"{name}_disp_path"] = _median(rows, f"{name}_disp_path")
            # Against the demonstration: kept for the record, off the figures. One
            # demonstration is one sample of a multimodal conditional, so this ranks the
            # sampler's mode choice as much as it ranks the clause.
            summary[f"{name}_gt_mse_improvement"] = _mean(rows, f"{name}_gt_mse_improvement")
            if flagged:
                summary[f"{name}_gt_mse_improvement_on_gt_mistake"] = _mean(
                    flagged, f"{name}_gt_mse_improvement"
                )
    summary["data"] = _provenance(rows, dataset, cfg, conditions + list(_CHANNELS), n_seeds)

    with open(os.path.join(output_dir, "metadata_steering.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)

    _render_floor(rows, summary, os.path.join(output_dir, "steering_floor.png"))
    _render_factorial(rows, summary, os.path.join(output_dir, "factorial.png"))
    _render_speed(rows, summary, os.path.join(output_dir, "speed.png"))
    _render_channels(rows, summary, os.path.join(output_dir, "precision_contact.png"))
    if labelled and len(rows) > 1:
        _render_response(rows, mean_axis, summary, os.path.join(output_dir, "response.png"))

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
        Panel("response.png",
              "What the clause does to the chunk — is the move conditional on the frame, "
              "and is it one shared direction?",
              how=_RESPONSE_HOW, primary=True),
        Panel("speed.png",
              "The speed clause — is the number read as tempo, and is it its own axis?",
              how=_SPEED_HOW, primary=True),
        Panel("precision_contact.png",
              "The precision and contact clauses — does either move the chunk past the floor?",
              how=_CHANNELS_HOW),
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
        # metadata_steering.json; only the verdict ratio is declared, so it can be
        # followed across checkpoints.
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
                "speed_separation_median",
                "speed / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                trend=True,
                note="Speed-range RMSE (speed 1 to 5 at quality 5, no mistake) over the seed floor.",
            ),
            Metric(
                "speed_tempo_correlation",
                "speed tempo shape match",
                good="high",
                fmt=2,
                baseline=summary["speed_tempo_correlation_monotone"],
                primary=True,
                trend=True,
                note=(
                    "Median over frames of the correlation between the model's per-step travel across "
                    "the five speed clauses and the demonstrations' per-step travel across the five "
                    "speed labels. The baseline is NOT 0: it is what a model that has learned only "
                    "'bigger number, faster' scores against this split's curve, which is about +0.7 "
                    "because the demonstration curve is not monotone (the hybrid label calls a "
                    "thrashing failed attempt slow, so speed 1 travels further per step than speed 2). "
                    "Below the baseline the number is not read as tempo at all; at it, the model reads "
                    "the digit and not the annotation; only above it has it learned this label. Five "
                    "points per frame, so a single frame's correlation has a spread of about 0.5 and "
                    "only the median over a decent frame count is worth reading."
                ),
            ),
            Metric(
                "speed_tempo_ratio_5_3",
                "per-step motion, speed 5 over speed 3",
                fmt=2,
                baseline=summary["speed_tempo_ratio_5_3_demo"],
                note=(
                    "RMS per-step travel of the speed-5 chunk over the speed-3 chunk. The baseline is "
                    "the same ratio in the demonstrations on this split, so matching it is the target "
                    "and neither high nor low is good on its own. Anchored at 3, not 1: 3 is the modal "
                    "label and 3-to-5 is the stretch of the scale that is monotone in travel on every "
                    "root, whereas speed 1 carries the thrashing segments and sits above speed 2."
                ),
            ),
            Metric(
                "quality_speed_cosine_median",
                "quality axis vs speed axis",
                good="low",
                fmt=2,
                baseline=summary["quality_speed_null_cosine_median"],
                note=(
                    "Cosine between the q1-to-q5 and s1-to-s5 chunk displacements. Both share the q5 "
                    "endpoint (the speed ramp holds quality at 5), which is worth about +0.5 on its own, "
                    "so the baseline is a null built to share an endpoint the same way — read the gap, "
                    "not the value. At the null the two clauses steer in their own directions; well "
                    "above it they are one direction with two names."
                ),
            ),
            Metric(
                "speed_step_motion_ratio",
                "per-step motion, speed 5 over speed 1",
                fmt=2,
                baseline=1.0,
                note="RMS per-step travel of the speed-5 chunk over the speed-1 chunk. Kept for continuity, and weaker than the 5-over-3 ratio above: the demonstrations' own speed-1 segments move MORE per step than their speed-2 and speed-3 ones, so this ratio can sit above 1 while the low end of the scale is backwards.",
            ),
            Metric(
                "precision_separation_median",
                "precision / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                trend=True,
                note="Precision-range RMSE (precision 1 to 5 at quality 5, no mistake, speed 5) over the seed floor.",
            ),
            Metric(
                "contact_clause_separation_median",
                "contact / flow-noise separation",
                good="high",
                fmt=2,
                baseline=1.0,
                trend=True,
                note=(
                    "Mean over the fifteen contact codes of the distance from the rollout chunk without "
                    "the contact sentence, over the seed floor. The spread between codes is "
                    "contact_spread_separation_median in the JSON."
                ),
            ),
        ],
        panels=panels,
        extra={"provenance": summary["data"]},
    )

    logging.info(
        f"[metadata_steering] n={len(rows)} ({len(flagged)} GT-mistake)  "
        f"quality range={summary['quality_range_rmse']:.4f} "
        f"({summary['separation_median']:.2f}x floor)  "
        f"mistake={summary['mistake_flip_rmse']:.4f}  "
        f"speed range={summary['speed_range_rmse']:.4f} ({summary['speed_separation_median']:.2f}x floor, "
        f"tau={summary['speed_kendall_tau_mean']:+.2f}, step motion x{summary['speed_step_motion_ratio']:.2f})  "
        f"speed tempo r={summary['speed_tempo_correlation']:+.2f} "
        f"(monotone {summary['speed_tempo_correlation_monotone']:+.2f}) "
        f"(m5/m3={summary['speed_tempo_ratio_5_3']:.2f} vs demo {summary['speed_tempo_ratio_5_3_demo']:.2f})  "
        f"q/s axis cos={summary['quality_speed_cosine_median']:+.2f} "
        f"(null {summary['quality_speed_null_cosine_median']:+.2f}, labels r={summary['quality_speed_label_correlation']:+.2f})  "
        f"tau={summary['kendall_tau_mean']:+.2f}  "
        f"rho(q5)={summary['q5_disp_path']:.4f} (floor {summary['seed_floor_disp_path']:.4f})  "
        f"conditionality={summary['conditionality_ratio']:.2f}x  "
        f"shared cos={summary['shared_cosine_median']:+.2f} (null {summary['noise_cosine_median']:+.2f})  "
        f"R={summary['shared_fraction']:.2f}  "
        f"precision range {summary['precision_separation_median']:.2f}x floor "
        f"(tau={summary['precision_kendall_tau_mean']:+.2f})  "
        f"contact {summary['contact_clause_separation_median']:.2f}x floor "
        f"(spread {summary['contact_spread_separation_median']:.2f}x)"
    )
