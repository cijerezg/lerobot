"""Input swap: which input stream does the action chunk follow?

Take an ANCHOR frame from the validation set and a DONOR frame. Build every observation
in which each of three input streams comes either from the anchor or from the donor:

    S  state          ``observation.state`` — the current joints, which is also the
                      anchor the relative action chunk is decoded against
    I  image          both cameras (``observation.images.*``) and their history windows
    H  state history  ``history.observation.state``

Everything else — wrist depth and its history, the subtask clause, the metadata clause,
the task string — stays the anchor's own in every cell. Three binary switches give
$2^3 = 8$ observations, the cells of a cube. A cell is named by the switches flipped to
the donor: ``...`` is the anchor's own prompt (the reference), ``.I.`` has only the
donor's images, ``S.H`` has the donor's state and state history under the anchor's
images, ``SIH`` is the donor's frame under the anchor's subtask and depth.

All 8 cells run in ONE forward pass with ONE flow-noise draw (``adapter.flow_noise_like``),
so two cells differ only in what was swapped.

**Donors.** Three per anchor, and every figure shows the three side by side:

* ``same_episode`` — a frame of the same episode at least ``SAME_EPISODE_MIN_GAP_S`` s
  away, drawn at random. Same session, lighting and camera placement; another moment of
  the same task.
* ``matched`` — the frame from another episode nearest in state (max over the arm joints
  of $|\\Delta q|$ in degrees, gripper within ``MATCH_GRIPPER_TOL`` units). The arm is
  roughly where it is, so swapping the images changes the scene and little else. The gap
  achieved is recorded per pair; read it before treating a state cell as a scene effect.
* ``random`` — a uniformly drawn frame from another episode. Everything changes.

**Quantities.** In the policy's normalized action space unless marked ``deg``. With
$a_c$ the chunk of cell $c$ and $\\|\\cdot\\|$ the RMS over chunk steps and joints:

* displacement of a cell from the reference, $\\|a_c - a_{...}\\|$, per pair, per joint
  and per chunk step; also from the all-donor corner, $\\|a_c - a_{SIH}\\|$;
* main effect of a switch: mean over the 4 cell pairs that differ only in that switch;
* variance shares: the 7 orthogonal $\\pm1$ contrasts of the cube split
  $\\sum_c \\|a_c - \\bar a\\|^2$ exactly into three main effects, three two-way and one
  three-way interaction, reported as fractions of the total;
* the demonstrations' own gap over the same pair, $\\|a^\\star_{donor} - a^\\star_{anchor}\\|$,
  next to the policy's all-donor displacement: how far apart the humans' next second was
  for the two frames, against how far apart the policy's is;
* ``state_follow``: for the ``S..`` cell in absolute degrees,
  $(A_t^{S..} - A_t^{...})_j / \\Delta s_j$ per step, median over the joints the donor
  moved by more than $0.05\\,\\sigma_j$. Flat at 1: the chunk is a fixed delta riding on
  the state. Falling: it converges on something the images fix;
* ``reseed_distance``: $\\|a_{...}^{(\\mathrm{seed}')} - a_{...}\\|$ for ``n_seeds - 1``
  other flow seeds on the anchor's own input. The sampler's own spread, drawn on the
  figures for scale. Not a floor: every cell shares one seed.

**Figures.** ``input_swap.png``: one box per (cell, donor kind), each box the
distribution over anchors of that cell's displacement. ``variance_shares.png``: which
switch the cube varies with, per donor kind. ``per_joint.png``: where a single switch's
displacement lands, per joint and per chunk step. ``demo_gap.png``: policy displacement
against the demonstrations' gap. ``state_follow.png``. ``anchors.png``: where the
anchors sit in their episodes, joined to their matched donors. ``examples/``: one anchor
per figure — both frames' images and all 8 chunks per joint in degrees plus end-effector
paths — for the anchors at the 10th, 50th and 90th percentile of the cube's spread under
each donor kind.

Runs inside rl_offline's validation loop when ``probe_parameters.enable_input_swap`` is
set, or standalone on the validation set (output lands in
``<output_dir>/validation/step_<ckpt>/input_swap`` so ``view_probes <output_dir>`` shows it):

    .venv/bin/python -m lerobot.probes.input_swap --config config_rl.yaml \\
        --policy.pretrained_path=outputs/<run>/checkpoints/<step>/pretrained_model \\
        --val_dataset_path=outputs/rebot_val-annotated-v3 \\
        --probe_parameters.output_dir=outputs/probe_runs/<name>
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    as_image,
    build_episode_index,
    joint_names_for_dim,
    load_probe_dataset,
    makedirs,
    panel_caption,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
    sample_episodes_evenly,
)
from lerobot.utils.action_smoothing import apply_butterworth_filter
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

FACTORS = ("state", "image", "state_history")
LETTERS = "SIH"
N_CELLS = 1 << len(FACTORS)
ALL_DONOR = N_CELLS - 1
DONOR_KINDS = ("same_episode", "matched", "random")
# A same-episode donor sits at least this far from the anchor in time: closer frames
# share the pose and the scene and swapping them changes nothing worth measuring.
SAME_EPISODE_MIN_GAP_S = 4.0
# Matched donors must hold the gripper within this many raw units: "holding" and "empty"
# are different scenes, and the arm-joint distance alone would happily pair them.
MATCH_GRIPPER_TOL = 25.0
# A joint has to move at least this fraction of its dataset std between anchor and donor
# before its state-follow ratio is read; below it the ratio is a division by noise.
STATE_FOLLOW_MIN_STD = 0.05

FACTOR_COLORS = {"state": "#1f77b4", "image": "#2ca02c", "state_history": "#9467bd"}
REF_COLOR = "#000000"
ALL_DONOR_COLOR = "#d62728"
KIND_COLORS = {"same_episode": "#2a9d8f", "matched": "#e07b00", "random": "#5b2c83"}
KIND_LABELS = {
    "same_episode": f"same episode, ≥{SAME_EPISODE_MIN_GAP_S:.0f} s away",
    "matched": "other episode, nearest state",
    "random": "other episode, random frame",
}


@dataclass
class InputSwapProbeConfig(TrainRLServerPipelineConfig):
    frame_indices: str = ""  # comma-separated global dataset indices; empty = sample evenly


# ── Cells ─────────────────────────────────────────────────────────────────────


def cell_code(mask: int) -> str:
    return "".join(LETTERS[i] if mask >> i & 1 else "." for i in range(len(FACTORS)))


def single_in(factor_idx: int) -> int:
    return 1 << factor_idx


def all_but(factor_idx: int) -> int:
    return ALL_DONOR ^ (1 << factor_idx)


def cell_label(mask: int) -> str:
    if mask == 0:
        return "anchor"
    if mask == ALL_DONOR:
        return "all donor"
    for i in range(len(FACTORS)):
        if mask == single_in(i):
            return f"{LETTERS[i]} only"
        if mask == all_but(i):
            return f"all but {LETTERS[i]}"
    return cell_code(mask)


def _factor_keys(obs: dict) -> tuple[dict[str, list[str]], list[str]]:
    """Every observation key claimed by exactly one switch, or held fixed (depth).

    An unclaimed key would ride along as the anchor's in every cell without anyone
    knowing, and an empty switch would make half the cells silent duplicates, so both
    are errors rather than warnings.
    """
    groups: dict[str, list[str]] = {factor: [] for factor in FACTORS}
    fixed: list[str] = []
    for key in obs:
        name = str(key)
        if name == OBS_STATE:
            groups["state"].append(key)
        elif name == f"history.{OBS_STATE}":
            groups["state_history"].append(key)
        elif name.startswith("observation.images.") or name.startswith("history.observation.images."):
            groups["image"].append(key)
        elif name.startswith("observation.depth.") or name.startswith("history.depth."):
            fixed.append(key)
        else:
            raise KeyError(f"observation key {key!r} belongs to no swap switch")
    empty = [factor for factor, keys in groups.items() if not keys]
    if empty:
        raise KeyError(f"no observation keys for switch(es) {empty}; the cube would have duplicate cells")
    return groups, fixed


def _cell(anchor: dict, donor: dict, mask: int, groups: dict[str, list[str]], fixed: list[str]) -> dict:
    out = {key: anchor[key] for key in fixed}
    for idx, factor in enumerate(FACTORS):
        source = donor if mask >> idx & 1 else anchor
        for key in groups[factor]:
            out[key] = source[key].to(anchor[key].dtype)
    return out


# ── Frames and donors ─────────────────────────────────────────────────────────


def _frame_table(dataset, stride: int) -> dict:
    """Low-dimensional view of every stride-grid frame: the donor pool."""
    # Plain-Python column access: the numpy/torch formatters route through torchvision's
    # video reader on some installs.
    hf = dataset.hf_dataset.with_format(None)
    states = np.asarray(hf["observation.state"], dtype=np.float64)
    episodes = np.asarray(hf["episode_index"], dtype=np.int64).reshape(-1)
    frames = np.asarray(hf["frame_index"], dtype=np.int64).reshape(-1)
    on_grid = frames % stride == 0
    return {
        "global_idx": np.arange(len(states))[on_grid],
        "episode": episodes[on_grid],
        "frame": frames[on_grid],
        "state": states[on_grid],
        "state_std": states.std(axis=0),
    }


def _choose_donors(table: dict, anchor_gidx: int, fps: float, rng: np.random.RandomState) -> dict[str, dict]:
    position = int(np.flatnonzero(table["global_idx"] == anchor_gidx)[0])
    episode = int(table["episode"][position])
    frame = int(table["frame"][position])
    state = table["state"][position]
    same = np.flatnonzero(table["episode"] == episode)
    other = np.flatnonzero(table["episode"] != episode)
    if other.size == 0:
        raise ValueError("input_swap needs at least two episodes for matched/random donors")

    far = same[np.abs(table["frame"][same] - frame) >= SAME_EPISODE_MIN_GAP_S * fps]
    if far.size == 0:
        far = same[same != position]
    same_pos = int(rng.choice(far))

    arm = slice(0, state.shape[0] - 1)
    arm_gap = np.abs(table["state"][other][:, arm] - state[arm]).max(axis=1)
    gripper_gap = np.abs(table["state"][other][:, -1] - state[-1])
    admissible = gripper_gap <= MATCH_GRIPPER_TOL
    if not admissible.any():
        admissible = np.ones_like(admissible)
    matched_pos = int(other[admissible][np.argmin(arm_gap[admissible])])
    random_pos = int(rng.choice(other))

    def describe(pos: int) -> dict:
        donor_state = table["state"][pos]
        donor_episode = int(table["episode"][pos])
        return {
            "global_idx": int(table["global_idx"][pos]),
            "episode": donor_episode,
            "frame": int(table["frame"][pos]),
            "arm_gap_deg": float(np.abs(donor_state[arm] - state[arm]).max()),
            "gripper_gap": float(abs(donor_state[-1] - state[-1])),
            "time_gap_s": float(abs(int(table["frame"][pos]) - frame) / fps) if donor_episode == episode else None,
        }

    return {"same_episode": describe(same_pos), "matched": describe(matched_pos), "random": describe(random_pos)}


# ── Measurements ──────────────────────────────────────────────────────────────


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))


def _factorial_shares(cells: np.ndarray) -> dict[str, float]:
    """Split the cube's total variance into main, two-way and three-way contrasts.

    With $h_S(c) = (-1)^{|c \\wedge S|}$ and $\\beta_S = \\frac{1}{8}\\sum_c h_S(c)\\,a_c$,
    $\\sum_c \\|a_c - \\bar a\\|^2 = 8 \\sum_{S \\neq \\emptyset} \\|\\beta_S\\|^2$; each share
    is $\\|\\beta_S\\|^2$ over that sum. ``spread`` is the RMS deviation of the cells
    around their mean, in the cells' own units.
    """
    x = cells.reshape(N_CELLS, -1)
    masks = np.arange(N_CELLS)
    beta = {}
    for subset in range(1, N_CELLS):
        sign = np.where([bin(m & subset).count("1") % 2 == 0 for m in masks], 1.0, -1.0)
        beta[subset] = float(np.square((sign[:, None] * x).mean(axis=0)).sum())
    total = sum(beta.values())
    scale = total if total > 0 else 1.0
    shares = {FACTORS[i]: beta[1 << i] / scale for i in range(len(FACTORS))}
    shares["two_way"] = sum(v for s, v in beta.items() if bin(s).count("1") == 2) / scale
    shares["three_way"] = beta[ALL_DONOR] / scale
    shares["spread"] = float(np.sqrt(total / x.shape[1]))
    return shares


def _main_effects(cells: np.ndarray) -> dict[str, float]:
    """Mean over the 4 cell pairs differing only in the switch of $\\|a_{c \\cup f} - a_c\\|$."""
    out = {}
    for idx, factor in enumerate(FACTORS):
        bit = 1 << idx
        out[factor] = float(np.mean([_rms(cells[c | bit] - cells[c]) for c in range(N_CELLS) if not c & bit]))
    return out


def _state_follow(unnorm: np.ndarray, anchor_state: np.ndarray, donor_state: np.ndarray, std: np.ndarray):
    """Per-step ratio of the state-only cell's absolute shift to the state shift."""
    delta = donor_state - anchor_state
    moved = np.abs(delta) >= STATE_FOLLOW_MIN_STD * std
    if not moved.any():
        return None
    shift = unnorm[single_in(0)] - unnorm[0]  # [T, D] degrees
    return np.median(shift[:, moved] / delta[moved][None, :], axis=1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 4:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation)


# ── Summary ───────────────────────────────────────────────────────────────────


def _quartiles(values) -> dict:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "n": int(values.size),
    }


def _summarize(rows: list[dict], horizon: int) -> dict:
    by_kind = {kind: [r for r in rows if r["donor_kind"] == kind] for kind in DONOR_KINDS}
    codes = [cell_code(m) for m in range(N_CELLS)]
    summary: dict = {
        "cells": codes,
        "cell_labels": {code: cell_label(m) for m, code in enumerate(codes)},
        "factors": list(FACTORS),
        "donor_kinds": list(DONOR_KINDS),
        "n_anchors": len(by_kind[DONOR_KINDS[0]]),
        "n_pairs": {kind: len(v) for kind, v in by_kind.items()},
        "reseed_distance": _quartiles([r["reseed_distance"] for r in rows if r["reseed_distance"] is not None]),
        "horizon": horizon,
        "kinds": {},
    }
    for kind in DONOR_KINDS:
        rows_k = by_kind[kind]
        gaps = np.array([r["demo_gap"] for r in rows_k])
        policy = np.array([r["rms"][ALL_DONOR] for r in rows_k])
        image_only = np.array([r["rms"][single_in(1)] for r in rows_k])
        curves = [r["state_follow"] for r in rows_k if r["state_follow"] is not None]
        block: dict = {
            "arm_gap_deg": _quartiles([r["donor"]["arm_gap_deg"] for r in rows_k]),
            "gripper_gap": _quartiles([r["donor"]["gripper_gap"] for r in rows_k]),
            "time_gap_s": _quartiles([r["donor"]["time_gap_s"] for r in rows_k]),
            "displacement": {code: _quartiles([r["rms"][c] for r in rows_k]) for c, code in enumerate(codes)},
            "displacement_from_all_donor": {
                code: _quartiles([r["rms_vs_all_donor"][c] for r in rows_k]) for c, code in enumerate(codes)
            },
            "displacement_abs_deg": {code: _quartiles([r["rms_abs_deg"][c] for r in rows_k]) for c, code in enumerate(codes)},
            "main_effect": {factor: _quartiles([r["main_effect"][factor] for r in rows_k]) for factor in FACTORS},
            "variance_share": {
                key: _quartiles([r["shares"][key] for r in rows_k]) for key in (*FACTORS, "two_way", "three_way", "spread")
            },
            "per_joint_single_in": {
                factor: np.median([r["per_joint"][single_in(i)] for r in rows_k], axis=0).tolist()
                for i, factor in enumerate(FACTORS)
            },
            "per_step_single_in": {
                factor: np.median([r["per_step"][single_in(i)] for r in rows_k], axis=0).tolist()
                for i, factor in enumerate(FACTORS)
            },
            "demo_gap": _quartiles(gaps),
            "policy_over_demo_gap": _quartiles(policy / np.maximum(gaps, 1e-9)),
            "spearman_all_donor_vs_demo_gap": _spearman(policy, gaps),
            "spearman_image_only_vs_demo_gap": _spearman(image_only, gaps),
            "state_follow": {
                "n": len(curves),
                "median": np.median(curves, axis=0).tolist() if curves else None,
                "q25": np.percentile(curves, 25, axis=0).tolist() if curves else None,
                "q75": np.percentile(curves, 75, axis=0).tolist() if curves else None,
            },
        }
        # Headline numbers, keyed by switch name: the manifest's lookup follows dots into
        # dicts and the cell codes carry dots of their own.
        two_way, three_way = block["variance_share"]["two_way"]["median"], block["variance_share"]["three_way"]["median"]
        block["headline"] = {
            "variance_share": {factor: block["variance_share"][factor]["median"] for factor in FACTORS},
            "interactions": None if two_way is None else two_way + three_way,
            "single_in_displacement": {
                factor: block["displacement"][cell_code(single_in(i))]["median"] for i, factor in enumerate(FACTORS)
            },
            "all_donor_displacement": block["displacement"][cell_code(ALL_DONOR)]["median"],
            "policy_over_demo_gap": block["policy_over_demo_gap"]["median"],
            "spearman_all_donor_vs_demo_gap": block["spearman_all_donor_vs_demo_gap"],
        }
        summary["kinds"][kind] = block
    return summary


# ── Figures ───────────────────────────────────────────────────────────────────


def _box(ax, position: float, values, color: str, width: float):
    values = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    if not values:
        return
    bp = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(color)


def _kind_legend(ax, summary: dict, loc: str = "upper left") -> None:
    handles = []
    for kind in DONOR_KINDS:
        block = summary["kinds"][kind]
        detail = (
            f"median time gap {block['time_gap_s']['median']:.0f} s"
            if kind == "same_episode"
            else f"median arm gap {block['arm_gap_deg']['median']:.0f} deg"
        )
        handles.append(Patch(facecolor=KIND_COLORS[kind], label=f"{KIND_LABELS[kind]}  ({detail})"))
    ax.legend(handles=handles, fontsize=8, loc=loc, title="donor", title_fontsize=8)


def _render_distributions(rows: list[dict], summary: dict, output_path: str) -> None:
    by_kind = {kind: [r for r in rows if r["donor_kind"] == kind] for kind in DONOR_KINDS}
    cells = [single_in(i) for i in range(len(FACTORS))] + [all_but(i) for i in range(len(FACTORS))] + [ALL_DONOR]
    reseed = summary["reseed_distance"]["median"]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    offsets = np.linspace(-0.27, 0.27, len(DONOR_KINDS))
    for pos, mask in enumerate(cells):
        for offset, kind in zip(offsets, DONOR_KINDS):
            _box(ax, pos + offset, [r["rms"][mask] for r in by_kind[kind]], KIND_COLORS[kind], width=0.22)
    for boundary in (len(FACTORS) - 0.5, 2 * len(FACTORS) - 0.5):
        ax.axvline(boundary, color="#dddddd", linewidth=0.8)
    if reseed:
        ax.axhline(reseed, color="#555555", linestyle="--", linewidth=0.9)
        ax.text(len(cells) - 0.55, reseed, "reseed distance", fontsize=7.5, va="bottom", ha="right", color="#555555")
    ax.set_yscale("log")
    ax.set_ylabel("displacement from the anchor's own chunk  (normalized RMS)", fontsize=9)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([cell_label(m) for m in cells], fontsize=9)
    ax.set_title(
        f"How far each cell's chunk moves from the anchor's own — {summary['n_anchors']} anchors, one flow seed",
        fontsize=10.5,
    )
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.tick_params(labelsize=8)
    _kind_legend(ax, summary)
    panel_caption(
        ax,
        [
            "One box per (cell, donor). The box is the distribution over anchors of that cell's RMS distance from the",
            "anchor's own chunk, all cells at the same flow seed. Left third: one stream taken from the donor, the other two",
            "kept. Middle third: two streams from the donor, one kept. Right: the donor's whole frame under the anchor's",
            "subtask and depth. Dashed line: median distance between two flow seeds on the anchor's own input, for scale.",
        ],
        y=-0.16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_per_joint(summary: dict, joint_names: list[str], output_path: str) -> None:
    fig, axes = plt.subplots(len(DONOR_KINDS), 2, figsize=(13, 3.6 * len(DONOR_KINDS)), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for row, kind in enumerate(DONOR_KINDS):
        block = summary["kinds"][kind]
        matrix = np.array([block["per_joint_single_in"][f] for f in FACTORS])
        ax = axes[row, 0]
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(FACTORS)))
        ax.set_yticklabels([f"{LETTERS[i]} only ({f})" for i, f in enumerate(FACTORS)], fontsize=8)
        ax.set_xticks(range(len(joint_names)))
        ax.set_xticklabels(joint_names, rotation=30, ha="right", fontsize=7.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.6,
                        color="white" if matrix[i, j] < matrix.max() * 0.6 else "black")
        ax.set_title(f"{KIND_LABELS[kind]}: median displacement per joint, one stream from the donor", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax = axes[row, 1]
        for i, f in enumerate(FACTORS):
            ax.plot(block["per_step_single_in"][f], color=FACTOR_COLORS[f], label=f"{LETTERS[i]} only")
        ax.set_xlabel("chunk step", fontsize=8)
        ax.set_ylabel("RMS over joints (normalized)", fontsize=8)
        ax.set_title(f"{KIND_LABELS[kind]}: median displacement per chunk step", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_shares(summary: dict, output_path: str) -> None:
    keys = (*FACTORS, "two_way", "three_way")
    labels = {**{f: f"{f} (main effect)" for f in FACTORS}, "two_way": "two-way interactions", "three_way": "three-way interaction"}
    colors = [*(FACTOR_COLORS[f] for f in FACTORS), "#bbbbbb", "#777777"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for x, kind in enumerate(DONOR_KINDS):
        block = summary["kinds"][kind]["variance_share"]
        values = np.array([block[k]["median"] or 0.0 for k in keys])
        values = values / values.sum() if values.sum() > 0 else values
        bottom = 0.0
        for value, key, color in zip(values, keys, colors):
            ax.bar(x, value, bottom=bottom, color=color, edgecolor="white", width=0.6, label=labels[key] if x == 0 else None)
            if value > 0.04:
                ax.text(x, bottom + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if key in FACTORS else "black")
            bottom += value
        ax.text(x, 1.02, f"spread {block['spread']['median']:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(DONOR_KINDS)))
    ax.set_xticklabels([KIND_LABELS[k].replace(", ", ",\n") for k in DONOR_KINDS], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of the 8 chunks' variance (median over anchors)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Which switch the 8 cells vary with", fontsize=10)
    panel_caption(ax, [
        "For one anchor and one donor the 8 chunks differ; their total variance splits exactly into what each switch",
        "does on its own (colours) and what only appears when switches flip together (greys). Bars are medians over",
        "anchors, renormalized to 1. 'spread' is the RMS deviation of the 8 chunks around their mean, normalized units.",
    ], y=-0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_anchors(rows: list[dict], episode_lengths: dict[int, int], fps: float, output_path: str) -> None:
    matched = [r for r in rows if r["donor_kind"] == "matched"]
    episodes = sorted(episode_lengths)
    fig, ax = plt.subplots(figsize=(12, 0.9 + 0.75 * len(episodes)))
    gaps = np.array([r["donor"]["arm_gap_deg"] for r in matched])
    norm = plt.Normalize(0.0, max(float(gaps.max()) if gaps.size else 1.0, 1.0))
    cmap = plt.get_cmap("viridis")
    for y, episode in enumerate(episodes):
        ax.hlines(y, 0, episode_lengths[episode] / fps, color="#cccccc", linewidth=3, zorder=1)
        rows_e = [r for r in matched if r["anchor"]["episode"] == episode]
        ax.scatter(
            [r["anchor"]["frame"] / fps for r in rows_e], [y] * len(rows_e),
            c=[cmap(norm(r["donor"]["arm_gap_deg"])) for r in rows_e], s=42, zorder=3, edgecolor="black", linewidth=0.4,
        )
        for r in rows_e:
            ax.annotate("", xy=(r["donor"]["frame"] / fps, episodes.index(r["donor"]["episode"])),
                        xytext=(r["anchor"]["frame"] / fps, y),
                        arrowprops={"arrowstyle": "-", "color": "#999999", "linewidth": 0.35, "alpha": 0.6}, zorder=2)
    ax.set_yticks(range(len(episodes)))
    ax.set_yticklabels([f"episode {e}" for e in episodes], fontsize=8.5)
    ax.set_xlabel("time in episode (s)", fontsize=8.5)
    ax.set_title(f"{len(matched)} anchors (dots) joined to their matched donors; colour = arm-joint gap to that donor", fontsize=9.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="max |Δq| over arm joints (deg)")
    ax.tick_params(labelsize=7.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_demo_gap(rows: list[dict], summary: dict, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (mask, title) in zip(axes, ((ALL_DONOR, "all donor: the donor's whole frame"), (single_in(1), "I only: the donor's images"))):
        lo, hi = np.inf, 0.0
        for kind in DONOR_KINDS:
            rows_k = [r for r in rows if r["donor_kind"] == kind]
            gaps = np.array([r["demo_gap"] for r in rows_k])
            policy = np.array([r["rms"][mask] for r in rows_k])
            rho = _spearman(policy, gaps)
            ax.scatter(gaps, policy, s=18, color=KIND_COLORS[kind], alpha=0.8,
                       label=f"{KIND_LABELS[kind]}  (Spearman {rho:.2f})" if rho is not None else KIND_LABELS[kind])
            lo, hi = min(lo, gaps.min(), policy.min()), max(hi, gaps.max(), policy.max())
        lo, hi = max(lo * 0.8, 1e-4), hi * 1.2
        ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance between the two frames' demonstrated chunks", fontsize=8.5)
        ax.set_ylabel("policy displacement in this cell", fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25, which="both")
        ax.tick_params(labelsize=7.5)
    panel_caption(axes[0], [
        "x: how far apart the humans' next second was for the anchor and the donor frame (normalized anchor-delta space,",
        "each chunk with its own anchor). y: how far the donor's inputs move the policy's chunk. On the diagonal the policy",
        "varies between the two frames as much as the humans did; below it, less. Left: whole frame; right: images only.",
    ], y=-0.17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_state_follow(summary: dict, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    drawn = False
    for kind in DONOR_KINDS:
        block = summary["kinds"][kind]["state_follow"]
        if not block["median"]:
            continue
        drawn = True
        steps = np.arange(len(block["median"]))
        ax.plot(steps, block["median"], color=KIND_COLORS[kind], label=f"{KIND_LABELS[kind]} (n={block['n']})")
        ax.fill_between(steps, block["q25"], block["q75"], color=KIND_COLORS[kind], alpha=0.18)
    if not drawn:
        plt.close(fig)
        return
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8)
    ax.set_xlabel("chunk step", fontsize=8.5)
    ax.set_ylabel("shift of the absolute chunk  /  shift of the state", fontsize=9)
    ax.set_title("S only, in degrees: does the absolute chunk move with the state?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)
    panel_caption(ax, [
        "Only the state (and with it the action anchor) comes from the donor; images and state history stay the anchor's.",
        "1 at every step: the chunk is a fixed delta riding on the state. Falling toward 0: the chunk converges on a target",
        "the pictures fix. Median over the joints the donor moved by more than 0.05 std; band = IQR over anchors.",
    ], y=-0.17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _depth_image(tensor) -> np.ndarray:
    depth = tensor.detach().float().cpu().squeeze().numpy()
    return np.where(depth > 0, depth, np.nan)


def _render_example(
    output_path: str,
    kind: str,
    percentile_label: str,
    anchor_frame: dict,
    donor_frame: dict,
    row: dict,
    chunks: dict,
    joint_names: list[str],
    fps: float,
) -> None:
    from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics

    unnorm = chunks["unnorm"]  # [8, T, D] degrees
    gt_anchor, gt_donor = chunks["gt_anchor_deg"], chunks["gt_donor_deg"]
    n_joints = unnorm.shape[-1]
    steps = np.arange(unnorm.shape[1]) / fps

    fig = plt.figure(figsize=(17, 12.5))
    grid = fig.add_gridspec(4, 6, height_ratios=[0.8, 0.95, 0.95, 1.05], hspace=0.42, wspace=0.32)

    def current_keys(obs: dict, prefix: str) -> list[str]:
        return sorted(k for k in obs if str(k).startswith(prefix))

    for col, (label, frame) in enumerate((("anchor", anchor_frame), ("donor", donor_frame))):
        obs = frame["obs"]
        panels = [(obs[k], k.rsplit(".", 1)[-1], False) for k in current_keys(obs, "observation.images.")]
        panels += [(obs[k], "depth (always the anchor's)", True) for k in current_keys(obs, "observation.depth.")]
        for i, (tensor, name, is_depth) in enumerate(panels[:3]):
            ax = fig.add_subplot(grid[0, col * 3 + i])
            if is_depth:
                ax.imshow(_depth_image(tensor), cmap="turbo")
            else:
                ax.imshow(as_image(tensor))
            ax.set_title(f"{label} ep{frame['episode_idx']} fr{frame['frame_idx']} — {name}", fontsize=8)
            ax.axis("off")

    lines = [(0, "anchor's own (...)", REF_COLOR, 2.2, "-"), (ALL_DONOR, "all donor (SIH)", ALL_DONOR_COLOR, 1.8, "-")]
    lines += [(single_in(i), f"{LETTERS[i]} only ({f})", FACTOR_COLORS[f], 1.3, "-") for i, f in enumerate(FACTORS)]
    lines += [(all_but(i), f"all but {LETTERS[i]}", FACTOR_COLORS[f], 0.9, ":") for i, f in enumerate(FACTORS)]

    positions = [(1, c) for c in range(6)] + [(2, c) for c in range(6)]
    for j in range(min(n_joints, len(positions))):
        r, c = positions[j]
        ax = fig.add_subplot(grid[r, c])
        for mask, label, color, width, style in lines:
            ax.plot(steps, unnorm[mask, :, j], color=color, linewidth=width, linestyle=style, label=label if j == 0 else None)
        ax.plot(steps, gt_anchor[:, j], color="#666666", linestyle="--", linewidth=1.0, label="anchor's demonstration" if j == 0 else None)
        ax.plot(steps, gt_donor[:, j], color="#f08080", linestyle="--", linewidth=1.0, label="donor's demonstration" if j == 0 else None)
        ax.set_title(joint_names[j], fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if r == 2:
            ax.set_xlabel("s", fontsize=7.5)
    handles, labels = fig.axes[6].get_legend_handles_labels()

    kin = RebotKinematics()
    ax3 = fig.add_subplot(grid[3, 0:3], projection="3d")
    for mask, label, color, width, style in lines:
        path = kin.ee_path(apply_butterworth_filter(unnorm[mask].astype(np.float64)))
        ax3.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=width, linestyle=style)
    for gt, color in ((gt_anchor, "#666666"), (gt_donor, "#f08080")):
        path = kin.ee_path(gt.astype(np.float64))
        ax3.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linestyle="--", linewidth=1.0)
    start = kin.ee_path(chunks["anchor_state"][None].astype(np.float64))[0]
    ax3.scatter(*start, color=REF_COLOR, s=30, marker="o")
    ax3.set_title("end-effector paths (Butterworth-smoothed, as the runtime sends them)", fontsize=8.5)
    ax3.tick_params(labelsize=6.5)
    ax3.set_xlabel("x (m)", fontsize=7); ax3.set_ylabel("y (m)", fontsize=7); ax3.set_zlabel("z (m)", fontsize=7)

    ax_text = fig.add_subplot(grid[3, 3:6])
    ax_text.axis("off")
    ax_text.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7.8, ncol=3, frameon=False)
    order = np.argsort(row["rms"])[::-1]
    table = "\n".join(
        f"{cell_label(m):>10}  {cell_code(m)}  {row['rms'][m]:.4f}   {row['rms_abs_deg'][m]:6.2f} deg" for m in order if m
    )
    shares = "  ".join(f"{LETTERS[i]} {row['shares'][f]:.2f}" for i, f in enumerate(FACTORS))
    gap = (
        f"time gap {row['donor']['time_gap_s']:.0f} s"
        if row["donor"]["time_gap_s"] is not None
        else f"arm gap {row['donor']['arm_gap_deg']:.1f} deg, gripper gap {row['donor']['gripper_gap']:.0f}"
    )
    ax_text.text(
        0.0, 0.62,
        f"donor: {KIND_LABELS[kind]}  ({gap});  {percentile_label} of the cube's spread ({row['shares']['spread']:.4f})\n"
        f"subtask (kept in every cell): {anchor_frame['subtask']!r}\n"
        f"variance shares  {shares}  interactions {row['shares']['two_way'] + row['shares']['three_way']:.2f}\n"
        f"demonstrations' gap {row['demo_gap']:.4f}   all-donor displacement {row['rms'][ALL_DONOR]:.4f}\n\n"
        f"      cell  code  from anchor   RMS deg\n{table}",
        fontsize=7.2, family="monospace", va="top", transform=ax_text.transAxes,
    )
    fig.suptitle(
        f"input swap — anchor ep{anchor_frame['episode_idx']} fr{anchor_frame['frame_idx']} "
        f"← donor ep{donor_frame['episode_idx']} fr{donor_frame['frame_idx']}   "
        "(joint panels in degrees; every cell at one flow seed; depth, subtask and metadata are the anchor's throughout)",
        fontsize=10.5,
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Manifest ──────────────────────────────────────────────────────────────────


def _write_manifest(output_dir: str, summary: dict, example_files: list[tuple[str, str]]) -> dict:
    reseed = summary["reseed_distance"]["median"]
    metrics = []
    for kind in DONOR_KINDS:
        for i, factor in enumerate(FACTORS):
            metrics.append(
                Metric(
                    f"kinds.{kind}.headline.single_in_displacement.{factor}",
                    f"{kind}: chunk displacement when only the {factor} comes from the donor",
                    fmt=3,
                    baseline=reseed,
                    primary=(kind == "matched"),
                    trend=(kind == "matched" and factor in ("state", "image")),
                    note="Median over anchors of the normalized RMS distance from the anchor's own chunk. The baseline is the median distance between two flow seeds on the same input, for scale.",
                )
            )
        metrics.append(
            Metric(
                f"kinds.{kind}.headline.all_donor_displacement",
                f"{kind}: chunk displacement for the donor's whole frame",
                fmt=3,
                baseline=reseed,
            )
        )
        for factor in FACTORS:
            metrics.append(
                Metric(
                    f"kinds.{kind}.headline.variance_share.{factor}",
                    f"{kind}: share of the cube's variance from the {factor}",
                    fmt=2,
                    primary=(kind == "matched"),
                    note="Fraction of the 8 chunks' total variance carried by this switch's main effect, median over anchors.",
                )
            )
        metrics.append(Metric(f"kinds.{kind}.headline.interactions", f"{kind}: share of the cube's variance from interactions", fmt=2))
        metrics.append(
            Metric(
                f"kinds.{kind}.headline.policy_over_demo_gap",
                f"{kind}: policy displacement over the demonstrations' gap",
                fmt=2,
                baseline=1.0,
                primary=(kind == "matched"),
                note="$\\|a_{SIH}-a_{...}\\| / \\|a^\\star_{donor}-a^\\star_{anchor}\\|$ per pair, median. Below 1: the policy's chunk differs less between the two frames than the humans' did.",
            )
        )
        metrics.append(
            Metric(f"kinds.{kind}.headline.spearman_all_donor_vs_demo_gap", f"{kind}: Spearman(policy displacement, demonstrations' gap)", fmt=2, baseline=0.0)
        )
    panels = [
        Panel(
            "input_swap.png",
            "How far each cell moves the chunk, per donor kind",
            how=(
                "One box per (cell, donor kind), the distribution over anchors of $\\|a_c - a_{...}\\|$ at one shared "
                "flow seed. Left third: one stream from the donor. Middle third: two streams from the donor. Right: the "
                "donor's whole frame under the anchor's subtask and depth. A stream whose box sits near the reseed line "
                "moves the chunk as much as changing the noise does; one near the bottom is barely read."
            ),
            primary=True,
        ),
        Panel("variance_shares.png", "Which switch the cube varies with", how="Stacked main-effect shares per donor kind; the greys are interactions.", primary=True),
        Panel("per_joint.png", "Where a single switch's displacement lands", how="Per joint and per chunk step, median over anchors, one stream from the donor."),
        Panel("demo_gap.png", "Policy displacement against the demonstrations' own gap", how="Diagonal: the policy varies between the two frames as much as the humans did.", primary=True),
        Panel("state_follow.png", "S only, in degrees", how="Ratio of the chunk's absolute shift to the state shift per step: 1 is a delta riding on the state."),
        Panel("anchors.png", "Where the anchors sit and where their matched donors come from", how="Colour is the arm-joint gap to the matched donor; a line joins each anchor to its donor's position."),
        Panel("input_swap.json", "Summary and every pair", how="Per pair: donor provenance and gaps, the 8 cells' displacements (normalized, from the all-donor cell, and in degrees), per-joint and per-step displacement, variance shares, demonstrations' gap, state-follow curve."),
    ]
    for file, caption in example_files:
        panels.append(Panel(
            file, caption,
            how="Top: the anchor's and the donor's frames. Rows 2-3: every cell's chunk in degrees; black is the anchor's own, red the donor's whole frame, solid colours one stream from the donor, dotted two streams. Dashed grey/pink: the two demonstrations.",
            align=f"examples/{os.path.basename(file).rsplit('_ep', 1)[0]}.png",
        ))
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Input Swap",
        group="Sensitivity",
        claim="When another frame's state, images or state history is injected, which one does the action chunk follow?",
        summary=summary,
        metrics=metrics,
        panels=panels,
        see_also=["depth_modality", "mem_history_influence", "subtask_sweep", "action_trace"],
    )


# ── Run ───────────────────────────────────────────────────────────────────────


def run(adapter, dataset, cfg, output_dir: str) -> None:
    makedirs(output_dir, os.path.join(output_dir, "examples"))
    p = cfg.probe_parameters
    stride = probe_image_stride(cfg)
    chunk_size = int(cfg.policy.chunk_size)
    fps = float(cfg.env.fps)
    seed = int(p.random_seed)
    n_reseeds = max(int(p.n_seeds) - 1, 1)
    rng = np.random.RandomState(seed)

    table = _frame_table(dataset, stride)
    explicit = [int(s) for s in str(getattr(cfg, "frame_indices", "") or "").split(",") if s.strip()]
    if explicit:
        # The stride grid is per episode (frame_index % stride), not global.
        grid = table["global_idx"]
        anchors = [int(grid[np.argmin(np.abs(grid - idx))]) for idx in explicit]
    else:
        n_per_episode = int(getattr(p, "input_swap_n_frames_per_episode", None) or p.n_frames_per_episode)
        anchors = [g for _, _, g in sample_episodes_evenly(dataset, n_per_episode, p.max_episodes, seed, stride)]
    if not anchors:
        logging.warning("[input_swap] no anchors selected.")
        return

    episode_lengths = {ep: len(idx) for ep, idx in build_episode_index(dataset).items()}
    logging.info(
        f"[input_swap] {len(anchors)} anchors x {len(DONOR_KINDS)} donors x {N_CELLS} cells (+{n_reseeds} reseeds) = "
        f"{len(anchors) * len(DONOR_KINDS)} stacked forwards; seed {seed}, stride {stride}"
    )

    base_noise = adapter.flow_noise_like(1, seed)
    reseed_noise = torch.cat([adapter.flow_noise_like(1, seed + 1_000_003 * k) for k in range(1, n_reseeds + 1)])
    adapter._set_probe_cuda_graph_enabled(False)

    rows: list[dict] = []
    chunks_by_pair: dict[tuple[int, str], dict] = {}
    consistency = None
    started = time.time()
    try:
        for a_idx, anchor_gidx in enumerate(anchors):
            anchor = probe_frame_inputs(dataset, cfg, anchor_gidx, chunk_size)
            groups, fixed = _factor_keys(anchor["obs"])
            n_joints = int(anchor["gt_actions"].shape[-1])
            anchor_state = anchor["state"].numpy().astype(np.float64)
            gt_anchor_norm = adapter.normalize_gt_actions(anchor["gt_actions"], anchor["state"]).numpy()[:, :n_joints]
            donors = _choose_donors(table, anchor_gidx, fps, rng)

            for k_idx, kind in enumerate(DONOR_KINDS):
                donor_info = donors[kind]
                donor = probe_frame_inputs(dataset, cfg, donor_info["global_idx"], chunk_size)
                cells = [_cell(anchor["obs"], donor["obs"], mask, groups, fixed) for mask in range(N_CELLS)]
                noise = base_noise.expand(N_CELLS, *base_noise.shape[1:])
                if k_idx == 0:
                    # The reseeds ride on the first donor's batch: same anchor input, other noise.
                    cells += [cells[0]] * n_reseeds
                    noise = torch.cat([noise, reseed_noise])
                unnorm_t, norm_t = adapter.predict_action_chunk_stacked(
                    cells,
                    anchor["task"],
                    subtask=anchor["subtask"],
                    metadata=anchor["metadata"],
                    noise=noise.contiguous(),
                    inference_action_mode="continuous",
                )
                horizon = min(int(norm_t.shape[1]), chunk_size)
                norm = norm_t[:, :horizon, :n_joints].numpy().astype(np.float64)
                unnorm = unnorm_t[:, :horizon, :n_joints].numpy().astype(np.float64)
                cube, cube_deg = norm[:N_CELLS], unnorm[:N_CELLS]

                if consistency is None:
                    # The all-anchor row against the one-frame path at the same seed: how much
                    # batching alone moves a chunk. Every contrast stays inside one batch.
                    _, single = adapter.predict_action_chunk_batch(
                        anchor["obs"], anchor["task"], [anchor["subtask"]], metadatas=[anchor["metadata"]],
                        noise=base_noise, inference_action_mode="continuous",
                    )
                    consistency = float(np.abs(single[0, :horizon, :n_joints].numpy() - cube[0]).max())
                    logging.info(f"[input_swap] stacked vs single-frame forward, max |Δ| = {consistency:.2e}")

                gt_donor_norm = adapter.normalize_gt_actions(donor["gt_actions"], donor["state"]).numpy()[:, :n_joints]
                donor_state = donor["state"].numpy().astype(np.float64)
                follow = _state_follow(cube_deg, anchor_state[:n_joints], donor_state[:n_joints], table["state_std"][:n_joints])
                rows.append({
                    "donor_kind": kind,
                    "anchor": {"global_idx": int(anchor_gidx), "episode": int(anchor["episode_idx"]), "frame": int(anchor["frame_idx"])},
                    "donor": donor_info,
                    "subtask": anchor["subtask"],
                    "rms": [float(_rms(cube[m] - cube[0])) for m in range(N_CELLS)],
                    "rms_vs_all_donor": [float(_rms(cube[m] - cube[ALL_DONOR])) for m in range(N_CELLS)],
                    "rms_abs_deg": [float(_rms(cube_deg[m] - cube_deg[0])) for m in range(N_CELLS)],
                    "per_joint": [_rms(cube[m] - cube[0], axis=0).tolist() for m in range(N_CELLS)],
                    "per_step": [_rms(cube[m] - cube[0], axis=1).tolist() for m in range(N_CELLS)],
                    "main_effect": _main_effects(cube),
                    "shares": _factorial_shares(cube),
                    "demo_gap": float(_rms(gt_donor_norm[:horizon] - gt_anchor_norm[:horizon])),
                    "reseed_distance": float(np.mean([_rms(norm[N_CELLS + k] - cube[0]) for k in range(n_reseeds)])) if k_idx == 0 else None,
                    "state_follow": follow.tolist() if follow is not None else None,
                })
                chunks_by_pair[(int(anchor_gidx), kind)] = {
                    "unnorm": cube_deg,
                    "gt_anchor_deg": anchor["gt_actions"].numpy()[:horizon, :n_joints].astype(np.float64),
                    "gt_donor_deg": donor["gt_actions"].numpy()[:horizon, :n_joints].astype(np.float64),
                    "anchor_state": anchor_state[:n_joints],
                }

            matched = rows[-2]
            elapsed = time.time() - started
            logging.info(
                f"[input_swap] anchor {a_idx + 1}/{len(anchors)} ep{anchor['episode_idx']} fr{anchor['frame_idx']}"
                f"  matched (arm gap {donors['matched']['arm_gap_deg']:.1f} deg) S/I/H only = "
                f"{matched['rms'][1]:.3f}/{matched['rms'][2]:.3f}/{matched['rms'][4]:.3f}  all {matched['rms'][ALL_DONOR]:.3f}"
                f"  ({elapsed / (a_idx + 1):.1f} s/anchor)"
            )
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    horizon = len(rows[0]["per_step"][0])
    n_joints = len(rows[0]["per_joint"][0])
    joint_names = joint_names_for_dim(n_joints)
    summary = _summarize(rows, horizon)
    summary["batch_consistency_max_abs"] = consistency
    summary["seed"] = seed
    summary["stride"] = stride
    summary["same_episode_min_gap_s"] = SAME_EPISODE_MIN_GAP_S
    summary["match_gripper_tol"] = MATCH_GRIPPER_TOL

    _render_distributions(rows, summary, os.path.join(output_dir, "input_swap.png"))
    _render_per_joint(summary, joint_names, os.path.join(output_dir, "per_joint.png"))
    _render_shares(summary, os.path.join(output_dir, "variance_shares.png"))
    _render_anchors(rows, episode_lengths, fps, os.path.join(output_dir, "anchors.png"))
    _render_demo_gap(rows, summary, os.path.join(output_dir, "demo_gap.png"))
    _render_state_follow(summary, os.path.join(output_dir, "state_follow.png"))

    # Examples stratified by the cube's spread: concentrated, typical, spread out.
    example_files: list[tuple[str, str]] = []
    for kind in DONOR_KINDS:
        rows_k = sorted((r for r in rows if r["donor_kind"] == kind), key=lambda r: r["shares"]["spread"])
        for pct in (10, 50, 90):
            row = rows_k[min(int(round(pct / 100 * (len(rows_k) - 1))), len(rows_k) - 1)]
            anchor = probe_frame_inputs(dataset, cfg, row["anchor"]["global_idx"], chunk_size)
            donor = probe_frame_inputs(dataset, cfg, row["donor"]["global_idx"], chunk_size)
            file = f"examples/{kind}_p{pct:02d}_ep{row['anchor']['episode']:04d}_fr{row['anchor']['frame']:06d}.png"
            _render_example(
                os.path.join(output_dir, file), kind, f"p{pct}", anchor, donor, row,
                chunks_by_pair[(row["anchor"]["global_idx"], kind)], joint_names, fps,
            )
            example_files.append((file, f"{KIND_LABELS[kind]}, spread p{pct}: anchor ep{row['anchor']['episode']} fr{row['anchor']['frame']}"))

    with open(os.path.join(output_dir, "input_swap.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    _write_manifest(output_dir, summary, example_files)

    for kind in DONOR_KINDS:
        head = summary["kinds"][kind]["headline"]
        shares = "  ".join(f"{LETTERS[i]} {head['variance_share'][f]:.2f}" for i, f in enumerate(FACTORS))
        single = "  ".join(f"{LETTERS[i]} {head['single_in_displacement'][f]:.3f}" for i, f in enumerate(FACTORS))
        logging.info(
            f"[input_swap] {kind}: variance shares {shares}  single-stream displacement {single}  "
            f"all donor {head['all_donor_displacement']:.3f}  policy/demo gap {head['policy_over_demo_gap']:.2f}"
        )
    logging.info(
        f"[input_swap] reseed distance {summary['reseed_distance']['median']:.3f}; wrote {output_dir} in {time.time() - started:.0f} s"
    )


@parser.wrap()
def cli(cfg: InputSwapProbeConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    val_path = getattr(cfg, "val_dataset_path", None)
    if val_path:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
        dataset.delta_timestamps = None
        dataset.delta_indices = None
    else:
        dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    # Same layout rl_offline writes, so ``view_probes <output_dir>`` finds it; the step is
    # the checkpoint directory's number when the path carries one (0 for the untrained init).
    step = 0
    for part in str(getattr(cfg.policy, "pretrained_path", "") or "").split(os.sep):
        if part.isdigit():
            step = int(part)
    run(adapter, dataset, cfg, os.path.join(cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", "input_swap"))


def main() -> None:
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
