#!/usr/bin/env python
"""Subspace spans: which directions do a robot's representations vary along, and do the robots' spans meet?

**Question.** Take the hidden states of many frames at one layer and one token group,
centred on the robot's own mean. They span a subspace of R^D. How large is it, which frames
define it, and is another robot's span the same subspace, an orthogonal one, or something in
between? The mean is subtracted per robot because the raw mean direction is shared by every
robot at cosine 0.99+ (massive activations), so uncentred it is the first singular direction
of every span, sets the scale of every tolerance and makes any two spans overlap by
construction. Centring leaves the directions of variation, which is where the structure is.
The removed direction is not lost: ``mean_norm`` (per robot) and ``mean_cos`` (per pair, the
cosine between the two robots' raw means) are recorded beside the span measurements. No
per-frame normalisation is applied.

**Input.** A ``conditions_matrix`` cache view: per token group one memmap of shape
(rows, layers, D) holding the pooled hidden state of every captured frame, plus meta.json
(robot, episode, class, phase, text condition, holdout flag) and thumbnails. The analysis
runs no forward pass. New suite/CLI runs default to ``attention_input``: native VLM
attention-normalization outputs and expert cross-attention inputs after native normalization
and time modulation. ``expert_key`` and ``expert_value`` read prepared cross-attention K/V.
These views are float32 means of native tensors, without external normalization.
``block_output`` explicitly selects the historical raw fp16 cache. Layer indices name the
consuming block for native views: attention-input L17 reads block 16's residual output.
In the official suite, ``enable_subspace_spans`` reuses the
conditions-matrix capture, or collects it when that probe is disabled. The standalone
cache command always reads saved representations. Rows are taken under one text condition (``--text``, default real; the image groups
are identical under both). Training rows (holdout = False) define a robot's span; the
robot's holdout rows are tested against it.

**Dimension.** For robot r with centred frame matrix X_r (n_r x D, float64; training rows
minus their mean mu_r, holdout rows minus the same mu_r), singular values
s_1 >= s_2 >= .... The numerical rank at tolerance tau is k_r(tau) = #{i : s_i > tau s_1},
for every tau in ``--tolerances`` (default 0.3, 0.1, 0.05, picked from the centred spectra of the
2026-10-05 step-1200 cache: 0.1 keeps 1 < k < n at nearly every layer and robot; 0.05 runs into the
frame count for state and action_output below ~100 frames). In exact arithmetic n generic
vectors have rank min(n, D), so a tolerance is what makes "the span" a defined object; it is
swept, not chosen. Cache rounding is estimated at its actual storage dtype; directions below the quantisation floor
s_floor = sigma (sqrt(n) + sqrt(D)), sigma^2 = mean(ulp(x)^2) / 12, are rounding noise; the
floor is drawn on the spectra. The training residual sum_{i>k} s_i^2 / sum_i s_i^2 and the
holdout residual 1 - ||X_hold Q_k||_F^2 / ||X_hold||_F^2 say how much energy of seen and
unseen frames lies outside the k-span.

**Which frames.** Column-pivoted QR of X_r^T: the j-th pivot is the frame with the largest
residual outside the span of the first j-1 pivots and |R_jj| is that residual. The sequence
|R_jj| / |R_11| is the frame-anchored counterpart of the singular values; the first pivots
are the frames that define the span, the tail falls inside it.

**Between robots.** With Q_A (D x p) and Q_B (D x q) orthonormal bases of two tau-spans, the
singular values of Q_A^T Q_B are cos(theta_i), the principal angles (Bjorck & Golub 1973).
All near 1: the smaller span lies in the larger; all near 0: orthogonal. Per layer and tau:
overlap = sum_i cos^2(theta_i) / min(p, q), in [0, 1], read against the random-subspace
expectation E[sum_i cos^2(theta_i)] = p q / D, which is max(p, q) / D after the division;
the number of angles with cos > 0.9; and the energy of B's frames outside A's span,
1 - ||X_B Q_A||_F^2 / ||X_B||_F^2 (and A outside B). At the ``--layers`` layers the whole
angle spectrum of every ReBot pair is drawn against the 95th percentile over ``--n_null``
random subspace pairs with the same (p, q, D). Two subspaces of dimensions p and q in R^D intersect in at least
p + q - D dimensions whatever the network did; pairs.csv records p and q for that check.

**Output.** ``explorer.html`` is a wide, offline report with question, token-group,
tolerance, layer, robot and pair selectors. One chart is shown at a time, with exact
values, count tables and explanations. ``report_details.json`` preserves reconstructed
spectra and pointwise random references. ``spans.csv``, ``pairs.csv``, ``pivots.csv`` and
``summary.json`` retain all measurements. ``index.json`` integrates the report with the
regular viewer. Existing legacy PNGs are preserved, but the manifest uses the explorer.
Use ``--report_only`` to rebuild from existing CSVs and caches without recomputing every
layer's measurements. This mode never runs model inference.

    uv run python -m lerobot.probes.subspace_spans \\
        --cache_dir outputs/<run>/validation/step_<n>/conditions_matrix/cache
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict

# The matrices are 150 x 2560: a full BLAS pool just thrashes and starves a training run on
# the same box. Set before numpy is imported; the environment overrides it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.linalg import qr

from lerobot.probes.conditions_matrix import REBOT, ROBOT_ORDER, TEXT_CONDITIONS, _load_cache, _thumb_path
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.representation_views import VIEW_METADATA, resolve_view_cache
from lerobot.probes.utils import makedirs
from lerobot.utils.utils import init_logging

MIN_FRAMES = 8
COS_SHARED = 0.9
NULL_PERCENTILE = 95
PIVOT_CAMERA = "external_0"
COLORS = {r: plt.get_cmap("tab10")(i) for i, r in enumerate(ROBOT_ORDER)}


# ──────────────────────────────────────────────────────────────────────────────
# Linear algebra
# ──────────────────────────────────────────────────────────────────────────────

def numerical_rank(s: np.ndarray, tau: float) -> int:
    return int(np.sum(s > tau * s[0]))


def fp16_floor(x16: np.ndarray) -> float:
    """Estimated cache rounding floor at the input dtype; legacy name kept for callers."""
    ulp = np.spacing(np.abs(x16)).astype(np.float64)
    sigma = np.sqrt(np.mean(ulp ** 2) / 12.0)
    n, d = x16.shape
    return float(sigma * (np.sqrt(n) + np.sqrt(d)))


def centred_matrices(arr16: np.ndarray, ids: dict[str, np.ndarray], layer: int):
    """(cached training rows, centred training rows, centred holdout rows, mean) at one layer.
    Both centrings use the TRAINING mean, so holdout rows are tested against the training span."""
    x16 = arr16[ids["train"], layer, :]
    mu = x16.astype(np.float64).mean(axis=0)
    x = x16.astype(np.float64) - mu
    xh = arr16[ids["holdout"], layer, :].astype(np.float64) - mu
    return x16, x, xh, mu


def pivot_sequence(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-pivoted QR of X^T: pivots are row indices of X, |R_jj| their residual norms."""
    _, r, piv = qr(x.T, mode="economic", pivoting=True)
    return piv, np.abs(np.diag(r))


def principal_cosines(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def residual_outside(x: np.ndarray, q: np.ndarray) -> float:
    return float(1.0 - np.sum((x @ q) ** 2) / np.sum(x ** 2))


def null_cosines(p: int, q: int, d: int, n_null: int, rng) -> np.ndarray:
    """95th percentile of cos(theta_i) between a random p-space and a random q-space in R^d.
    By rotation invariance the q-space is the first q coordinates, so one QR per draw."""
    draws = np.empty((n_null, min(p, q)))
    for t in range(n_null):
        qa = np.linalg.qr(rng.standard_normal((d, p)))[0]
        draws[t] = np.linalg.svd(qa[:q, :], compute_uv=False)
    return np.percentile(draws, NULL_PERCENTILE, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def robot_rows(rows: list[dict], present: np.ndarray, text: str) -> dict[str, dict[str, np.ndarray]]:
    """robot -> {"train": row indices defining the span, "holdout": row indices tested against it}."""
    by_robot: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"train": [], "holdout": []})
    for r in rows:
        if r["text"] != text or not present[r["row"]]:
            continue
        by_robot[r["robot"]]["holdout" if r["holdout"] else "train"].append(r["row"])
    order = [r for r in ROBOT_ORDER if r in by_robot] + sorted(set(by_robot) - set(ROBOT_ORDER))
    return {r: {k: np.array(v, dtype=int) for k, v in by_robot[r].items()} for r in order
            if len(by_robot[r]["train"]) >= MIN_FRAMES}


def analyze_group(group: str, arr16: np.ndarray, sel: dict, tolerances: list[float], layers: list[int],
                  n_null: int, rng, out: dict) -> None:
    """One token group, every layer: spans per robot, angles per pair, pivots at ``layers``."""
    n_layers, d = arr16.shape[1], arr16.shape[2]
    robots = list(sel)
    pairs = [(a, b) for i, a in enumerate(robots) for b in robots[i + 1:]]
    null_cache: dict[tuple, np.ndarray] = {}
    for layer in range(n_layers):
        spans = {}
        for robot in robots:
            x16, x, xh, mu = centred_matrices(arr16, sel[robot], layer)
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            spans[robot] = {"x": x, "s": s, "vt": vt, "floor": fp16_floor(x16), "xh": xh, "mu": mu}
            energy = np.cumsum(s ** 2) / np.sum(s ** 2)
            for tau in tolerances:
                k = numerical_rank(s, tau)
                q = vt[:k].T
                out["spans"].append({
                    "group": group, "layer": layer, "robot": robot, "tau": tau,
                    "n_frames": len(x), "n_holdout": len(spans[robot]["xh"]), "d": d,
                    "s1": s[0], "mean_norm": float(np.linalg.norm(mu)), "fp16_floor": spans[robot]["floor"], "k": k,
                    "train_residual": max(0.0, 1.0 - energy[k - 1]),
                    "holdout_residual": residual_outside(spans[robot]["xh"], q) if len(spans[robot]["xh"]) else float("nan"),
                })
            if layer in layers:
                piv, rdiag = pivot_sequence(x)
                out["pivots"][(group, layer, robot)] = (sel[robot]["train"][piv], rdiag)
                out["spectra"][(group, layer, robot)] = (s, rdiag, spans[robot]["floor"])
        for a, b in pairs:
            for tau in tolerances:
                p, q_ = numerical_rank(spans[a]["s"], tau), numerical_rank(spans[b]["s"], tau)
                qa, qb = spans[a]["vt"][:p].T, spans[b]["vt"][:q_].T
                cos = principal_cosines(qa, qb)
                out["pairs"].append({
                    "group": group, "layer": layer, "a": a, "b": b, "tau": tau, "p": p, "q": q_, "d": d,
                    "overlap": float(np.sum(cos ** 2) / min(p, q_)), "null_overlap": max(p, q_) / d,
                    "shared": int(np.sum(cos > COS_SHARED)), "cos_1": float(cos[0]),
                    "mean_cos": float(np.dot(spans[a]["mu"], spans[b]["mu"])
                                      / (np.linalg.norm(spans[a]["mu"]) * np.linalg.norm(spans[b]["mu"]))),
                    "residual_b_outside_a": residual_outside(spans[b]["x"], qa),
                    "residual_a_outside_b": residual_outside(spans[a]["x"], qb),
                })
                if layer in layers and tau == tolerances[len(tolerances) // 2] and REBOT in (a, b):
                    key = (p, q_, d)
                    if key not in null_cache:
                        null_cache[key] = null_cosines(p, q_, d, n_null, rng)
                    out["angles"][(group, layer, a, b)] = (cos, null_cache[key])
        if layer % 6 == 0 or layer == n_layers - 1:
            logging.info(f"  {group}: layer {layer + 1}/{n_layers}")


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _grid(n_rows: int, n_cols: int, w: float = 3.6, h: float = 2.8):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows), squeeze=False)
    return fig, axes


def plot_spectra(spectra: dict, groups: list[str], layers: list[int], robots: list[str], tolerances: list[float], path: str) -> None:
    fig, axes = _grid(len(groups), len(layers))
    for i, group in enumerate(groups):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            floors = []
            for robot in robots:
                if (group, layer, robot) not in spectra:
                    continue
                s, rdiag, floor = spectra[(group, layer, robot)]
                idx = np.arange(1, len(s) + 1)
                ax.semilogy(idx, s / s[0], color=COLORS.get(robot, "k"), lw=1.2, label=robot)
                ax.semilogy(idx, rdiag / rdiag[0], color=COLORS.get(robot, "k"), lw=0.9, ls=":")
                floors.append(floor / s[0])
            for tau in tolerances:
                ax.axhline(tau, color="0.6", lw=0.6, ls="--")
            if floors:
                ax.axhline(np.median(floors), color="0.3", lw=0.8, ls="-.", label="cache rounding floor")
            ax.set_title(f"{group}  L{layer}", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == len(groups) - 1:
                ax.set_xlabel("index i", fontsize=8)
            if j == 0:
                ax.set_ylabel("s_i / s_1  (dotted: |R_jj| / |R_11|)", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Centred singular-value spectra per robot (solid) and pivoted-QR residuals (dotted); dashed = tolerances", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dimension(spans: list[dict], groups: list[str], robots: list[str], tolerances: list[float], path: str) -> None:
    fig, axes = _grid(len(groups), len(tolerances))
    table = defaultdict(dict)
    n_frames = {}
    for r in spans:
        table[(r["group"], r["robot"], r["tau"])][r["layer"]] = r["k"]
        n_frames[(r["group"], r["robot"])] = r["n_frames"]
    for i, group in enumerate(groups):
        for j, tau in enumerate(tolerances):
            ax = axes[i, j]
            for robot in robots:
                ks = table.get((group, robot, tau))
                if not ks:
                    continue
                layers = sorted(ks)
                ax.plot(layers, [ks[l] for l in layers], color=COLORS.get(robot, "k"), lw=1.2, label=robot)
                if max(ks.values()) == n_frames[(group, robot)]:
                    ax.axhline(n_frames[(group, robot)], color=COLORS.get(robot, "k"), lw=0.5, ls=":")
            ax.set_title(f"{group}  tau = {tau:g}", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == len(groups) - 1:
                ax.set_xlabel("layer", fontsize=8)
            if j == 0:
                ax.set_ylabel("k(tau)", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Numerical rank of each robot's centred span against depth (dotted: a robot's frame count where k reaches it)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _rebot_pairs(pairs_rows: list[dict]) -> list[tuple[str, str]]:
    seen = []
    for r in pairs_rows:
        pair = (r["a"], r["b"])
        if REBOT in pair and pair not in seen:
            seen.append(pair)
    return seen


def _other(pair: tuple[str, str]) -> str:
    return pair[1] if pair[0] == REBOT else pair[0]


def plot_overlap(pairs_rows: list[dict], groups: list[str], tolerances: list[float], path: str) -> None:
    pairs = _rebot_pairs(pairs_rows)
    if not pairs:
        return
    fig, axes = _grid(len(groups), len(tolerances))
    table = defaultdict(dict)
    for r in pairs_rows:
        table[(r["group"], r["a"], r["b"], r["tau"])][r["layer"]] = (r["overlap"], r["null_overlap"])
    for i, group in enumerate(groups):
        for j, tau in enumerate(tolerances):
            ax = axes[i, j]
            for pair in pairs:
                vals = table.get((group, *pair, tau))
                if not vals:
                    continue
                layers = sorted(vals)
                color = COLORS.get(_other(pair), "k")
                ax.plot(layers, [vals[l][0] for l in layers], color=color, lw=1.2, label=f"{pair[0]} vs {pair[1]}")
                ax.plot(layers, [vals[l][1] for l in layers], color=color, lw=0.8, ls="--")
            ax.set_ylim(0, 1)
            ax.set_title(f"{group}  tau = {tau:g}", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == len(groups) - 1:
                ax.set_xlabel("layer", fontsize=8)
            if j == 0:
                ax.set_ylabel("sum cos^2 / min(p,q)", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Overlap of the tau-spans (solid) against the random-subspace expectation max(p,q)/D (dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_angles(angles: dict, groups: list[str], layers: list[int], pairs: list[tuple[str, str]], tau: float, path: str) -> None:
    if not pairs:
        return
    fig, axes = _grid(len(groups), len(layers))
    for i, group in enumerate(groups):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            for pair in pairs:
                if (group, layer, *pair) not in angles:
                    continue
                cos, null = angles[(group, layer, *pair)]
                idx = np.arange(1, len(cos) + 1)
                color = COLORS.get(_other(pair), "k")
                ax.plot(idx, cos, color=color, lw=1.2, marker=".", ms=3, label=f"{pair[0]} vs {pair[1]}")
                ax.plot(idx, null, color=color, lw=0.8, ls="--")
            ax.axhline(COS_SHARED, color="0.6", lw=0.6, ls=":")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{group}  L{layer}", fontsize=9)
            ax.tick_params(labelsize=7)
            if i == len(groups) - 1:
                ax.set_xlabel("principal angle index i", fontsize=8)
            if j == 0:
                ax.set_ylabel("cos theta_i", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Principal angles between tau-spans at tau = {tau:g} (dashed: {NULL_PERCENTILE}th percentile of random subspaces)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pivot_frames(pivots: dict, rows: list[dict], cache_dir: str, group: str, layer: int, robots: list[str],
                      n_pivots: int, path: str) -> bool:
    if not os.path.isdir(os.path.join(cache_dir, "thumbs")):
        return False
    by_row = {r["row"]: r for r in rows}
    fig, axes = plt.subplots(len(robots), n_pivots, figsize=(1.6 * n_pivots, 1.75 * len(robots)), squeeze=False)
    for i, robot in enumerate(robots):
        order, rdiag = pivots[(group, layer, robot)]
        for j in range(n_pivots):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(order):
                continue
            meta = by_row[int(order[j])]
            thumb = _thumb_path(cache_dir, meta["row"] // len(TEXT_CONDITIONS), PIVOT_CAMERA)
            if os.path.exists(thumb):
                with Image.open(thumb) as im:
                    ax.imshow(im)
            ax.set_title(f"#{j + 1}  r={rdiag[j] / rdiag[0]:.2g}\n{meta['object_class']}/{meta['phase']}", fontsize=6.5)
            if j == 0:
                ax.text(-0.05, 0.5, robot, transform=ax.transAxes, fontsize=8, rotation=90, va="center", ha="right")
    fig.suptitle(f"Pivot frames of {group} at L{layer}: the first {n_pivots} frames of the pivoted QR, r = |R_jj| / |R_11|", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _pivot_rows(pivots: dict, rows: list[dict]) -> list[dict]:
    by_row = {r["row"]: r for r in rows}
    out = []
    for (group, layer, robot), (order, rdiag) in sorted(pivots.items()):
        for j, row in enumerate(order):
            m = by_row[int(row)]
            out.append({"group": group, "layer": layer, "robot": robot, "pivot": j + 1, "row": int(row),
                        "residual": rdiag[j], "residual_rel": rdiag[j] / rdiag[0], "episode": m["episode"],
                        "frame": m["frame"], "object_class": m["object_class"], "phase": m["phase"]})
    return out


def _summary(spans, pairs_rows, sel_by_group, available, text, tolerances, layers, headline_layer, tau, headline_group) -> dict:
    dimension = defaultdict(dict)
    for r in spans:
        if r["layer"] == headline_layer and r["tau"] == tau:
            dimension[r["group"]][r["robot"]] = {"k": r["k"], "n_frames": r["n_frames"],
                                                 "holdout_residual": r["holdout_residual"]}
    overlap = defaultdict(dict)
    for r in pairs_rows:
        if r["layer"] == headline_layer and r["tau"] == tau:
            overlap[r["group"]][f"{r['a']}_{r['b']}"] = {k: r[k] for k in (
                "p", "q", "overlap", "null_overlap", "shared", "cos_1", "mean_cos", "residual_b_outside_a", "residual_a_outside_b")}
    return {
        "text": text, "tolerances": tolerances, "layers": layers, "headline_layer": headline_layer,
        "headline_tau": tau, "headline_group": headline_group, "cos_shared": COS_SHARED,
        "robots": {g: {r: int(len(s["train"])) for r, s in sel.items()} for g, sel in sel_by_group.items()},
        # Training episodes the conditions sampler could have drawn per robot and cell: the
        # ceiling of the frame counts above (one frame per episode and cell).
        "available": {r: {"total": int(sum(cells.values())), "cells": cells} for r, cells in available.items()},
        "dimension": dimension, "overlap": overlap,
    }


def _write_manifest(output_dir: str, summary: dict, groups: list[str], pivot_group: str, pivot_written: bool) -> dict:
    layer, tau, g = summary["headline_layer"], summary["headline_tau"], summary["headline_group"]
    metrics = []
    for robot, entry in summary["dimension"].get(g, {}).items():
        ceiling = summary["available"].get(robot, {}).get("total")
        metrics.append(Metric(f"dimension.{g}.{robot}.k", f"{robot}: numerical rank of the centred {g} span, layer {layer}, tau {tau:g}",
                              good="none", fmt=0, primary=(robot == REBOT),
                              note=f"{entry['n_frames']} frames" + (f" of {ceiling} available" if ceiling is not None else "")))
    for pair, entry in summary["overlap"].get(g, {}).items():
        if REBOT not in pair.split("_"):
            continue
        metrics.append(Metric(f"overlap.{g}.{pair}.overlap", f"{pair.replace('_', ' vs ')}: span overlap, {g}, layer {layer}, tau {tau:g}",
                              good="none", fmt=2, primary=True,
                              note=f"random-subspace expectation {entry['null_overlap']:.2f}; {entry['shared']} angles with cos > {COS_SHARED}; "
                                   f"cosine between the removed raw means {entry['mean_cos']:.3f}"))
    panels = [
        Panel("spectra.png", "Centred singular-value spectra per robot with the pivoted-QR residual sequence",
              how="Solid: s_i / s_1 of each robot's frame matrix after subtracting the robot's training mean. "
                  "Dotted: |R_jj| / |R_11| of the column-pivoted QR, "
                  "the residual of the j-th most novel frame. Dashed grey: the tolerances; dash-dot: the cache "
                  "quantisation floor. Directions below the floor are rounding noise.", primary=True),
        Panel("dimension_by_layer.png", "Numerical rank k(tau) of each robot's centred span against depth, one column per tolerance",
              how="k(tau) = number of singular values above tau s_1. Compare robots at the same tau; compare "
                  "columns to see how much the answer depends on the tolerance.", primary=True),
        Panel("overlap_by_layer.png", "ReBot pairs: overlap of the tau-spans against depth",
              how="sum_i cos^2(theta_i) / min(p, q) over the principal angles between the two spans (solid) against "
                  "the random-subspace expectation max(p, q) / D (dashed, same colour). At the dashed line the two "
                  "spans are no more aligned than random subspaces of the same dimensions.", primary=True),
        Panel("angles.png", "ReBot pairs: the full principal-angle spectrum at the chosen layers",
              how="cos(theta_i) for every principal angle (solid) against the 95th percentile of random subspace pairs "
                  "of the same (p, q, D) (dashed). Angles above the dotted line count as shared directions; "
                  "angles at the dashed line are chance alignment.", primary=True),
    ]
    if pivot_written:
        panels.append(Panel("pivot_frames.png", f"The first pivot frames of {pivot_group}: the frames that define each robot's span",
                            how="Left to right, the pivoted-QR order: each frame is the one with the largest residual "
                                "outside the span of the frames before it. r is that residual relative to the first."))
    return write_index(
        output_dir, sys.modules[__name__], title="Subspace spans", group="Representation",
        claim="What subspace do a robot's mean-centred representations span at each layer, which frames define it, "
              "and do the robots' spans coincide, meet at an angle, or stay orthogonal?",
        summary=summary, metrics=metrics, panels=panels, status="info",
        see_also=["conditions_matrix", "domain_representations"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────

def analyze(cache_dir: str, output_dir: str, text: str, tolerances: list[float], layers: list[int],
            n_null: int, n_pivots: int, pivot_group: str, seed: int) -> dict:
    makedirs(output_dir)
    rows, arrays, present = _load_cache(cache_dir)
    cache_meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    available = cache_meta.get("available", {})
    rng = np.random.RandomState(seed)
    out = {"spans": [], "pairs": [], "pivots": {}, "spectra": {}, "angles": {}}
    sel_by_group = {}
    groups = list(arrays)
    for group in groups:
        sel = robot_rows(rows, present[group], text)
        sel_by_group[group] = sel
        logging.info(f"{group}: " + ", ".join(
            f"{r} {len(s['train'])}f of {sum(available[r].values())} available (+{len(s['holdout'])} holdout)"
            if r in available else f"{r} {len(s['train'])}f (+{len(s['holdout'])} holdout)" for r, s in sel.items()))
        arr16 = np.asarray(arrays[group])
        analyze_group(group, arr16, sel, tolerances, layers, n_null, rng, out)
    robots = [r for r in ROBOT_ORDER if any(r in s for s in sel_by_group.values())]
    robots += sorted({r for s in sel_by_group.values() for r in s} - set(robots))
    tau = tolerances[len(tolerances) // 2]
    headline_layer = layers[-1]
    headline_group = pivot_group if pivot_group in groups else groups[0]

    _write_csv(os.path.join(output_dir, "spans.csv"), out["spans"])
    _write_csv(os.path.join(output_dir, "pairs.csv"), out["pairs"])
    _write_csv(os.path.join(output_dir, "pivots.csv"), _pivot_rows(out["pivots"], rows))
    summary = _summary(out["spans"], out["pairs"], sel_by_group, available, text, tolerances, layers, headline_layer, tau, headline_group)
    summary["representation_view"] = cache_meta.get("representation_view", "block_output")
    summary["representation"] = VIEW_METADATA[summary["representation_view"]]
    summary["cache_dtype"] = cache_meta.get("cache_dtype", str(next(iter(arrays.values())).dtype))
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    from lerobot.probes.subspace_report import render
    render(output_dir, cache_dir, n_null=n_null, seed=seed, n_pivots=n_pivots)
    logging.info(f"wrote {output_dir}")
    return summary


def run(adapter, dataset, cfg, output_dir: str) -> dict | None:
    """Official suite entry; share the same-step conditions cache, or collect it alone."""
    from threadpoolctl import threadpool_limits
    from lerobot.probes.conditions_matrix import collect_cache

    p = cfg.probe_parameters
    if p.mode not in ("collect", "plot", "all"):
        raise ValueError(f"Unknown subspace probe mode: {p.mode!r}")
    if p.subspace_view not in VIEW_METADATA:
        raise ValueError(f"Unknown subspace_view: {p.subspace_view!r}")
    tolerances = sorted((float(t) for t in p.subspace_tolerances.split(",")), reverse=True)
    layers = [int(layer) for layer in p.subspace_layers.split(",")]
    if p.subspace_text not in TEXT_CONDITIONS:
        raise ValueError(f"subspace_text must be one of {TEXT_CONDITIONS}")
    if not tolerances or any(not 0 < t < 1 for t in tolerances):
        raise ValueError("subspace_tolerances must be between zero and one")
    if not layers or min(layers) < 0 or p.subspace_n_null < 1 or p.subspace_n_pivots < 1:
        raise ValueError("Subspace layers must be nonnegative and null/pivot counts positive")

    conditions_dir = os.path.join(os.path.dirname(output_dir), "conditions_matrix")
    cache_dir = os.path.join(conditions_dir, "cache")
    if p.mode in ("collect", "all") and not p.enable_conditions_matrix:
        collect_cache(adapter, dataset, cfg, conditions_dir)
    cache_dir = resolve_view_cache(cache_dir, p.subspace_view)
    # Validate even in collect mode: an enabled conditions probe may have failed earlier.
    _, arrays, _ = _load_cache(cache_dir)
    if not arrays or any(max(layers) >= array.shape[1] for array in arrays.values()):
        raise ValueError("subspace_layers must index the captured blocks of every token group")
    if p.mode == "collect":
        return None
    # NumPy is already loaded in the validation process, so environment defaults alone
    # cannot constrain its BLAS pool here.
    with threadpool_limits(limits=4, user_api="blas"):
        return analyze(cache_dir, output_dir, p.subspace_text, tolerances, layers,
                       p.subspace_n_null, p.subspace_n_pivots, p.subspace_pivot_group,
                       int(p.random_seed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache_dir", required=True, help="a conditions_matrix cache directory (meta.json + <group>.npy)")
    ap.add_argument("--output_dir", default=None, help="default: <step>/subspace_spans[_<text>] beside the cache's probe dir")
    ap.add_argument("--view", choices=tuple(VIEW_METADATA), default="attention_input", help="Native capture view; legacy raw caches require explicit block_output")
    ap.add_argument("--report_only", action="store_true", help="rebuild report from saved CSVs and cache; no inference")
    ap.add_argument("--text", default="real", choices=TEXT_CONDITIONS)
    ap.add_argument("--tolerances", default="0.3,0.1,0.05", help="relative singular-value cutoffs tau; the middle one is the headline")
    ap.add_argument("--layers", default="14,15,16,28,32", help="zero-based layers for spectra, angles and pivots; the last one is the headline")
    ap.add_argument("--n_null", type=int, default=100, help="random subspace pairs behind the angle null band")
    ap.add_argument("--n_pivots", type=int, default=12)
    ap.add_argument("--pivot_group", default="img_external_0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    init_logging()
    cache_dir = os.path.normpath(args.cache_dir)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(cache_dir)), "subspace_spans" if args.text == "real" else f"subspace_spans_{args.text}")
    cache_dir = resolve_view_cache(cache_dir, args.view)
    if args.report_only:
        saved_view = json.load(open(os.path.join(output_dir, "summary.json"))).get("representation_view", "block_output")
        if saved_view != args.view:
            raise ValueError(f"Saved analysis uses {saved_view}, requested {args.view}; rerun without --report_only.")
        from lerobot.probes.subspace_report import render
        render(output_dir, cache_dir, n_null=args.n_null, seed=args.seed, n_pivots=args.n_pivots)
        return
    tolerances = sorted((float(t) for t in args.tolerances.split(",")), reverse=True)
    layers = [int(l) for l in args.layers.split(",")]
    analyze(cache_dir, output_dir, args.text, tolerances, layers, args.n_null, args.n_pivots, args.pivot_group, args.seed)


if __name__ == "__main__":
    main()
