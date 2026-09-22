"""Input swap: which input stream does the action chunk follow?

Take an ANCHOR frame from the validation set and a DONOR frame. Build every input in
which each of the switched streams comes either from the anchor or from the donor:

    S  state          ``observation.state`` — the current joints, which is also the
                      anchor the relative action chunk is decoded against
    I  image          both cameras (``observation.images.*``)
    T  subtask text   the prompt's subtask clause ("The current step is ...")

Everything else — wrist depth, the metadata clause, the task string — stays the
anchor's own in every cell. The switches give $2^3 = 8$ inputs, the cells of the S/I/T
cube. A cell is named by the switches flipped to the donor: ``...`` is the
anchor's own prompt (the reference), ``.I.`` has only the donor's images, ``..T`` is
the anchor's frame under the donor's subtask text, ``SIT`` the donor's frame and text
under the anchor's depth.

All cells run in ONE forward pass with ONE flow-noise draw (``adapter.flow_noise_like``),
so two cells differ only in what was swapped; the pack step builds every row's prompt
from its own subtask string, so the T cells share the batch too.

The S and I readouts are the same measurements as in the probe's runs from before the
T switch existed, so they compare across them.

**The T switch is a no-op when both frames carry the same subtask text** (a same-episode
donor inside the same segment, a matched donor at the same phase of the same object):
the T cells then duplicate their partners and every T contrast is exactly zero. Each
pair records ``subtask_differs``, and every T-specific number in the summary — the T
cells' displacement, the T main effect, the T and interaction variance shares — is
taken over the pairs whose texts differ, with its own ``n``. The S and I numbers, the
spread and the all-donor corner stay over every pair: a duplicated T cell leaves them
unchanged.

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
* main effect of a switch: mean over the $2^{k-1}$ cell pairs that differ only in that switch;
* variance shares: the $2^k - 1$ orthogonal $\\pm1$ contrasts of the cube split
  $\\sum_c \\|a_c - \\bar a\\|^2$ exactly into the main effects and the interactions,
  grouped by order, reported as fractions of the total;
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
per figure — both frames' images and every cell's chunk per joint in degrees plus end-effector
paths — for the anchors at the 10th, 50th and 90th percentile of the cube's spread under
each donor kind; ``examples/trace.html`` shows the same nine pairs as an action-inspector
dashboard (below).

**Hand-picked pairs.** ``--pairs=anchor:donor,...`` (global dataset indices, snapped onto
the stride grid) runs the cube on frames you chose, one donor per anchor, and writes
``trace.html``: the action-inspector view of every pair — every cell's end-effector path and both
demonstrations in one orbitable scene, both arms' poses, the wrist-roll and gripper
timelines, and both frames' cameras beside them. Cells whose state comes from the donor
start at the donor's pose, because the chunk is decoded relative to the state it was given.
The population figures are skipped and the output lands in ``input_swap_pairs``, so a full
run in the same ``output_dir`` is kept. A pair worth picking is two frames at nearly the
same pose whose demonstrations then go to different places, so a cell that follows the
images has somewhere else to go: the start of a reach (0.3-2.5 s into a "grasp" segment —
every reach leaves the same container pose while the objects sit elsewhere), an episode
start (the rest pose under two tasks' scenes), or a carry. For the T switch the two frames
must also carry different subtask texts (another object, another container, another verb).
``migration/pick_input_swap_pairs.py`` searches a dataset for them.

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

# The switches in bit order: the state is bit 0 (``_state_follow`` reads the S-only cell
# there), the images bit 1, the subtask text bit 2.
FACTOR_LETTERS = {"state": "S", "image": "I", "subtask": "T"}
INTERACTION_NAMES = {2: "two_way", 3: "three_way"}
# Switches that select observation keys; the subtask switch selects the prompt text.
OBS_FACTORS = ("state", "image")
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

FACTOR_COLORS = {"state": "#1f77b4", "image": "#2ca02c", "subtask": "#ff7f0e"}
FACTOR_LEGEND = {"state": "blue state", "image": "green images", "subtask": "orange subtask text"}
REF_COLOR = "#000000"
ALL_DONOR_COLOR = "#d62728"
KIND_COLORS = {"same_episode": "#2a9d8f", "matched": "#e07b00", "random": "#5b2c83"}
KIND_LABELS = {
    "same_episode": f"same episode, ≥{SAME_EPISODE_MIN_GAP_S:.0f} s away",
    "matched": "other episode, nearest state",
    "random": "other episode, random frame",
    "picked": "hand-picked frame",
}


@dataclass
class InputSwapProbeConfig(TrainRLServerPipelineConfig):
    frame_indices: str = ""  # comma-separated global dataset indices; empty = sample evenly
    pairs: str = ""  # comma-separated anchor:donor global indices; runs only these, one donor each


# ── Cells ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cube:
    """The switches in bit order and the cells they span."""

    factors: tuple[str, ...]

    @property
    def letters(self) -> str:
        return "".join(FACTOR_LETTERS[f] for f in self.factors)

    @property
    def n_cells(self) -> int:
        return 1 << len(self.factors)

    @property
    def all_donor(self) -> int:
        return self.n_cells - 1

    @property
    def interaction_keys(self) -> tuple[str, ...]:
        """Variance-share keys for the interactions, by order: ("two_way", "three_way")."""
        return tuple(INTERACTION_NAMES[order] for order in range(2, len(self.factors) + 1))

    @property
    def share_keys(self) -> tuple[str, ...]:
        return (*self.factors, *self.interaction_keys, "spread")

    @property
    def streams_text(self) -> str:
        names = {"state": "state", "image": "images", "subtask": "subtask text"}
        words = [names[f] for f in self.factors]
        return ", ".join(words[:-1]) + " or " + words[-1]

    def index(self, factor: str) -> int:
        return self.factors.index(factor)

    def single_in(self, factor_idx: int) -> int:
        return 1 << factor_idx

    def all_but(self, factor_idx: int) -> int:
        return self.all_donor ^ (1 << factor_idx)

    def code(self, mask: int) -> str:
        return "".join(self.letters[i] if mask >> i & 1 else "." for i in range(len(self.factors)))

    def label(self, mask: int) -> str:
        if mask == 0:
            return "anchor"
        if mask == self.all_donor:
            return "all donor"
        for i in range(len(self.factors)):
            if mask == self.single_in(i):
                return f"{self.letters[i]} only"
        for i in range(len(self.factors)):
            if mask == self.all_but(i):
                return f"all but {self.letters[i]}"
        return self.code(mask)

    def shown_cells(self) -> list[int]:
        """Single-stream cells, then two-stream cells, then the all-donor corner."""
        n = len(self.factors)
        return [self.single_in(i) for i in range(n)] + [self.all_but(i) for i in range(n)] + [self.all_donor]


CUBE = Cube(("state", "image", "subtask"))


def _factor_keys(obs: dict, cube: Cube) -> tuple[dict[str, list[str]], list[str]]:
    """Every observation key claimed by exactly one switch, or held fixed (depth).

    An unclaimed key would ride along as the anchor's in every cell without anyone
    knowing, and an empty switch would make half the cells silent duplicates, so both
    are errors rather than warnings. The subtask switch owns no observation key: it
    selects the prompt text (``_cell_subtasks``).
    """
    groups: dict[str, list[str]] = {factor: [] for factor in cube.factors if factor in OBS_FACTORS}
    fixed: list[str] = []
    for key in obs:
        name = str(key)
        if name == OBS_STATE:
            groups["state"].append(key)
        elif name.startswith("observation.images."):
            groups["image"].append(key)
        elif name.startswith(("observation.depth.", "probe_complementary.depth.")):
            fixed.append(key)   # the depth frame and its present flag / intrinsics stay the anchor's
        else:
            raise KeyError(f"observation key {key!r} belongs to no swap switch")
    empty = [factor for factor, keys in groups.items() if not keys]
    if empty:
        raise KeyError(f"no observation keys for switch(es) {empty}; the cube would have duplicate cells")
    return groups, fixed


def _cell(anchor: dict, donor: dict, mask: int, groups: dict[str, list[str]], fixed: list[str], cube: Cube) -> dict:
    out = {key: anchor[key] for key in fixed}
    for factor, keys in groups.items():
        source = donor if mask >> cube.index(factor) & 1 else anchor
        for key in keys:
            out[key] = source[key].to(anchor[key].dtype)
    return out


def _cell_subtasks(anchor_subtask: str | None, donor_subtask: str | None, cube: Cube) -> list[str | None]:
    """The prompt's subtask text per cell: the donor's where the T bit is set, else the anchor's."""
    bit = cube.index("subtask")
    return [(donor_subtask if mask >> bit & 1 else anchor_subtask) or None for mask in range(cube.n_cells)]


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


def _snap(grid: np.ndarray, idx: int) -> int:
    """The nearest frame on the stride grid — per episode (frame_index % stride), not global."""
    return int(grid[np.argmin(np.abs(grid - idx))])


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

    return {
        "same_episode": _describe_donor(table, position, same_pos, fps),
        "matched": _describe_donor(table, position, matched_pos, fps),
        "random": _describe_donor(table, position, random_pos, fps),
    }


def _describe_donor(table: dict, anchor_pos: int, donor_pos: int, fps: float) -> dict:
    state, donor_state = table["state"][anchor_pos], table["state"][donor_pos]
    episode, donor_episode = int(table["episode"][anchor_pos]), int(table["episode"][donor_pos])
    frame, donor_frame = int(table["frame"][anchor_pos]), int(table["frame"][donor_pos])
    return {
        "global_idx": int(table["global_idx"][donor_pos]),
        "episode": donor_episode,
        "frame": donor_frame,
        "arm_gap_deg": float(np.abs(donor_state[:-1] - state[:-1]).max()),
        "gripper_gap": float(abs(donor_state[-1] - state[-1])),
        "time_gap_s": float(abs(donor_frame - frame) / fps) if donor_episode == episode else None,
    }


def _picked_donor(table: dict, anchor_gidx: int, donor_gidx: int, fps: float) -> dict[str, dict]:
    anchor_pos = int(np.flatnonzero(table["global_idx"] == anchor_gidx)[0])
    donor_pos = int(np.flatnonzero(table["global_idx"] == donor_gidx)[0])
    return {"picked": _describe_donor(table, anchor_pos, donor_pos, fps)}


# ── Measurements ──────────────────────────────────────────────────────────────


def _rms(x: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))


def _factorial_shares(cells: np.ndarray, cube: Cube) -> dict[str, float]:
    """Split the cube's total variance into main-effect and interaction contrasts.

    With $N$ cells, $h_S(c) = (-1)^{|c \\wedge S|}$ and $\\beta_S = \\frac{1}{N}\\sum_c h_S(c)\\,a_c$,
    $\\sum_c \\|a_c - \\bar a\\|^2 = N \\sum_{S \\neq \\emptyset} \\|\\beta_S\\|^2$; each share
    is $\\|\\beta_S\\|^2$ over that sum, interactions grouped by order (``two_way``,
    ``three_way``). ``spread`` is the RMS deviation of the cells around their mean, in
    the cells' own units.
    """
    x = cells.reshape(cube.n_cells, -1)
    masks = np.arange(cube.n_cells)
    beta = {}
    for subset in range(1, cube.n_cells):
        sign = np.where([bin(m & subset).count("1") % 2 == 0 for m in masks], 1.0, -1.0)
        beta[subset] = float(np.square((sign[:, None] * x).mean(axis=0)).sum())
    total = sum(beta.values())
    scale = total if total > 0 else 1.0
    shares = {factor: beta[1 << i] / scale for i, factor in enumerate(cube.factors)}
    for order, name in INTERACTION_NAMES.items():
        shares[name] = sum(v for s, v in beta.items() if bin(s).count("1") == order) / scale
    shares["spread"] = float(np.sqrt(total / x.shape[1]))
    return shares


def _main_effects(cells: np.ndarray, cube: Cube) -> dict[str, float]:
    """Mean over the cell pairs differing only in the switch of $\\|a_{c \\cup f} - a_c\\|$."""
    out = {}
    for idx, factor in enumerate(cube.factors):
        bit = 1 << idx
        out[factor] = float(np.mean([_rms(cells[c | bit] - cells[c]) for c in range(cube.n_cells) if not c & bit]))
    return out


def _state_follow(unnorm: np.ndarray, anchor_state: np.ndarray, donor_state: np.ndarray, std: np.ndarray):
    """Per-step ratio of the state-only cell's absolute shift to the state shift."""
    delta = donor_state - anchor_state
    moved = np.abs(delta) >= STATE_FOLLOW_MIN_STD * std
    if not moved.any():
        return None
    shift = unnorm[1] - unnorm[0]  # the S-only cell (the state is bit 0), [T, D] degrees
    return np.median(shift[:, moved] / delta[moved][None, :], axis=1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 4:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation)


# ── Summary ───────────────────────────────────────────────────────────────────


def _fmt(value, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _median_curve(curves: list, length: int) -> list:
    """Median over rows of equal-length lists; NaNs when there are no rows."""
    return np.median(curves, axis=0).tolist() if curves else [float("nan")] * length


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


def _summarize(rows: list[dict], horizon: int, cube: Cube) -> dict:
    by_kind = {kind: [r for r in rows if r["donor_kind"] == kind] for kind in DONOR_KINDS}
    codes = [cube.code(m) for m in range(cube.n_cells)]
    n_joints = len(rows[0]["per_joint"][0])
    t_bit = 1 << cube.index("subtask")
    # Numbers that isolate the T switch are read over the pairs whose texts differ: the
    # T cells (other than the all-donor corner, which a duplicated T cell leaves as it
    # was), the T main effect, and the variance shares that carry T contrasts.
    t_share_keys = {"subtask", *cube.interaction_keys}
    summary: dict = {
        "cells": codes,
        "cell_labels": {code: cube.label(m) for m, code in enumerate(codes)},
        "factors": list(cube.factors),
        "donor_kinds": list(DONOR_KINDS),
        "n_anchors": len(by_kind[DONOR_KINDS[0]]),
        "n_pairs": {kind: len(v) for kind, v in by_kind.items()},
        "reseed_distance": _quartiles([r["reseed_distance"] for r in rows if r["reseed_distance"] is not None]),
        "horizon": horizon,
        "over_subtask_differs": {
            "cells": [code for c, code in enumerate(codes) if c & t_bit and c != cube.all_donor],
            "main_effect": ["subtask"],
            "variance_share": sorted(t_share_keys),
        },
        "kinds": {},
    }
    for kind in DONOR_KINDS:
        rows_k = by_kind[kind]
        rows_t = [r for r in rows_k if r["subtask_differs"]]

        def rows_for_cell(mask: int) -> list[dict]:
            return rows_t if mask & t_bit and mask != cube.all_donor else rows_k

        def rows_for_factor(factor: str) -> list[dict]:
            return rows_t if factor == "subtask" else rows_k

        gaps = np.array([r["demo_gap"] for r in rows_k])
        policy = np.array([r["rms"][cube.all_donor] for r in rows_k])
        image_only = np.array([r["rms"][cube.single_in(1)] for r in rows_k])
        curves = [r["state_follow"] for r in rows_k if r["state_follow"] is not None]
        block: dict = {
            "n_pairs": len(rows_k),
            "subtask_differs_n": len(rows_t),
            "arm_gap_deg": _quartiles([r["donor"]["arm_gap_deg"] for r in rows_k]),
            "gripper_gap": _quartiles([r["donor"]["gripper_gap"] for r in rows_k]),
            "time_gap_s": _quartiles([r["donor"]["time_gap_s"] for r in rows_k]),
            "displacement": {code: _quartiles([r["rms"][c] for r in rows_for_cell(c)]) for c, code in enumerate(codes)},
            "displacement_from_all_donor": {
                code: _quartiles([r["rms_vs_all_donor"][c] for r in rows_for_cell(c)]) for c, code in enumerate(codes)
            },
            "displacement_abs_deg": {
                code: _quartiles([r["rms_abs_deg"][c] for r in rows_for_cell(c)]) for c, code in enumerate(codes)
            },
            "main_effect": {
                factor: _quartiles([r["main_effect"][factor] for r in rows_for_factor(factor)]) for factor in cube.factors
            },
            "variance_share": {
                key: _quartiles([r["shares"][key] for r in (rows_t if key in t_share_keys else rows_k)])
                for key in cube.share_keys
            },
            "per_joint_single_in": {
                factor: _median_curve([r["per_joint"][cube.single_in(i)] for r in rows_for_factor(factor)], n_joints)
                for i, factor in enumerate(cube.factors)
            },
            "per_step_single_in": {
                factor: _median_curve([r["per_step"][cube.single_in(i)] for r in rows_for_factor(factor)], horizon)
                for i, factor in enumerate(cube.factors)
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
        interactions = [block["variance_share"][key]["median"] for key in cube.interaction_keys]
        block["headline"] = {
            "variance_share": {factor: block["variance_share"][factor]["median"] for factor in cube.factors},
            "interactions": None if any(v is None for v in interactions) else float(sum(interactions)),
            "single_in_displacement": {
                factor: block["displacement"][cube.code(cube.single_in(i))]["median"] for i, factor in enumerate(cube.factors)
            },
            "all_donor_displacement": block["displacement"][cube.code(cube.all_donor)]["median"],
            "policy_over_demo_gap": block["policy_over_demo_gap"]["median"],
            "spearman_all_donor_vs_demo_gap": block["spearman_all_donor_vs_demo_gap"],
            "subtask_differs_fraction": len(rows_t) / max(len(rows_k), 1),
        }
        summary["kinds"][kind] = block
    return summary


def _summarize_pairs(rows: list[dict], horizon: int, cube: Cube) -> dict:
    """Hand-picked mode: one block per pair, no population statistics."""
    codes = [cube.code(m) for m in range(cube.n_cells)]
    summary: dict = {
        "cells": codes,
        "cell_labels": {code: cube.label(m) for m, code in enumerate(codes)},
        "factors": list(cube.factors),
        "donor_kinds": ["picked"],
        "n_anchors": len(rows),
        "n_pairs": {"picked": len(rows)},
        "reseed_distance": _quartiles([r["reseed_distance"] for r in rows if r["reseed_distance"] is not None]),
        "horizon": horizon,
        "pairs": {},
    }
    for k, row in enumerate(rows, start=1):
        summary["pairs"][f"pair{k:02d}"] = {
            "anchor": row["anchor"],
            "donor": row["donor"],
            "subtask": row["subtask"],
            "donor_subtask": row["donor_subtask"],
            "subtask_differs": row["subtask_differs"],
            "displacement": {code: row["rms"][c] for c, code in enumerate(codes)},
            "displacement_abs_deg": {code: row["rms_abs_deg"][c] for c, code in enumerate(codes)},
            "variance_share": row["shares"],
            "demo_gap": row["demo_gap"],
            "reseed_distance": row["reseed_distance"],
            "headline": {
                "single_in_displacement": {factor: row["rms"][cube.single_in(i)] for i, factor in enumerate(cube.factors)},
                "all_donor_displacement": row["rms"][cube.all_donor],
                "variance_share": {factor: row["shares"][factor] for factor in cube.factors},
            },
        }
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
        detail += f"; T differs on {block['subtask_differs_n']}/{block['n_pairs']}"
        handles.append(Patch(facecolor=KIND_COLORS[kind], label=f"{KIND_LABELS[kind]}  ({detail})"))
    ax.legend(handles=handles, fontsize=8, loc=loc, title="donor", title_fontsize=8)


def _render_distributions(rows: list[dict], summary: dict, output_path: str, cube: Cube) -> None:
    by_kind = {kind: [r for r in rows if r["donor_kind"] == kind] for kind in DONOR_KINDS}
    cells = cube.shown_cells()
    reseed = summary["reseed_distance"]["median"]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    offsets = np.linspace(-0.27, 0.27, len(DONOR_KINDS))
    t_bit = 1 << cube.index("subtask")
    for pos, mask in enumerate(cells):
        for offset, kind in zip(offsets, DONOR_KINDS):
            rows_k = by_kind[kind]
            if mask & t_bit and mask != cube.all_donor:
                rows_k = [r for r in rows_k if r["subtask_differs"]]
            _box(ax, pos + offset, [r["rms"][mask] for r in rows_k], KIND_COLORS[kind], width=0.22)
    # Separators: after the single-stream cells and before the all-donor corner.
    for boundary in (len(cube.factors) - 0.5, len(cells) - 1.5):
        ax.axvline(boundary, color="#dddddd", linewidth=0.8)
    if reseed:
        ax.axhline(reseed, color="#555555", linestyle="--", linewidth=0.9)
        ax.text(len(cells) - 0.55, reseed, "reseed distance", fontsize=7.5, va="bottom", ha="right", color="#555555")
    ax.set_yscale("log")
    ax.set_ylabel("displacement from the anchor's own chunk  (normalized RMS)", fontsize=9)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([cube.label(m) for m in cells], fontsize=9)
    ax.set_title(
        f"How far each cell's chunk moves from the anchor's own — {summary['n_anchors']} anchors, one flow seed",
        fontsize=10.5,
    )
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.tick_params(labelsize=8)
    _kind_legend(ax, summary)
    caption = [
        "One box per (cell, donor). The box is the distribution over anchors of that cell's RMS distance from the",
        "anchor's own chunk, all cells at the same flow seed. Left: one switch flipped to the donor, the rest kept. Middle:",
        "all switches but one flipped. Right: the donor's whole frame and subtask text under the anchor's depth. Cells with",
        "the T switch flipped are read over the pairs whose subtask texts differ (counts in the legend). Dashed line: median",
        "distance between two flow seeds on the anchor's own input, for scale.",
    ]
    panel_caption(ax, caption, y=-0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_per_joint(summary: dict, joint_names: list[str], output_path: str, cube: Cube) -> None:
    fig, axes = plt.subplots(len(DONOR_KINDS), 2, figsize=(13, 3.6 * len(DONOR_KINDS)), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for row, kind in enumerate(DONOR_KINDS):
        block = summary["kinds"][kind]
        matrix = np.array([block["per_joint_single_in"][f] for f in cube.factors], dtype=np.float64)
        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
        ax = axes[row, 0]
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(cube.factors)))
        ax.set_yticklabels(
            [f"{cube.letters[i]} only ({f})" + (f" [n={block['subtask_differs_n']}]" if f == "subtask" else "") for i, f in enumerate(cube.factors)],
            fontsize=8,
        )
        ax.set_xticks(range(len(joint_names)))
        ax.set_xticklabels(joint_names, rotation=30, ha="right", fontsize=7.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6.6,
                            color="white" if matrix[i, j] < vmax * 0.6 else "black")
        ax.set_title(f"{KIND_LABELS[kind]}: median displacement per joint, one switch flipped to the donor", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax = axes[row, 1]
        for i, f in enumerate(cube.factors):
            ax.plot(block["per_step_single_in"][f], color=FACTOR_COLORS[f], label=f"{cube.letters[i]} only")
        ax.set_xlabel("chunk step", fontsize=8)
        ax.set_ylabel("RMS over joints (normalized)", fontsize=8)
        ax.set_title(f"{KIND_LABELS[kind]}: median displacement per chunk step", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_shares(summary: dict, output_path: str, cube: Cube) -> None:
    keys = (*cube.factors, *cube.interaction_keys)
    labels = {**{f: f"{f} (main effect)" for f in cube.factors}, "two_way": "two-way interactions", "three_way": "three-way interaction"}
    colors = [*(FACTOR_COLORS[f] for f in cube.factors), "#bbbbbb", "#777777"]
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
                        color="white" if key in cube.factors else "black")
            bottom += value
        ax.text(x, 1.02, f"spread {_fmt(block['spread']['median'], 3)}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(DONOR_KINDS)))
    ax.set_xticklabels([KIND_LABELS[k].replace(", ", ",\n") for k in DONOR_KINDS], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"share of the {cube.n_cells} chunks' variance (median over anchors)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(f"Which switch the {cube.n_cells} cells vary with", fontsize=10)
    panel_caption(ax, [
        f"For one anchor and one donor the {cube.n_cells} chunks differ; their total variance splits exactly into what each switch",
        "does on its own (colours) and what only appears when switches flip together (greys). Bars are medians over",
        f"anchors, renormalized to 1; the subtask and interaction shares over the pairs whose subtask texts differ. 'spread'",
        f"is the RMS deviation of the {cube.n_cells} chunks around their mean, normalized units.",
    ], y=-0.22)
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


def _render_demo_gap(rows: list[dict], summary: dict, output_path: str, cube: Cube) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (mask, title) in zip(axes, ((cube.all_donor, "all donor: the donor's whole frame"), (cube.single_in(1), "I only: the donor's images"))):
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


def _render_state_follow(summary: dict, output_path: str, cube: Cube) -> None:
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
        "Only the state (and with it the action anchor) comes from the donor; the other streams and the subtask text stay",
        "the anchor's.",
        "1 at every step: the chunk is a fixed delta riding on the state. Falling toward 0: the chunk converges on a target",
        "the pictures fix. Median over the joints the donor moved by more than 0.05 std; band = IQR over anchors.",
    ], y=-0.17)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cell_lines(cube: Cube) -> list[tuple[int, str, str, float, str]]:
    """(mask, label, colour, width, dash) per drawn cell: the reference and the all-donor
    corner heavy, one stream from the donor solid, two streams dotted."""
    lines = [
        (0, f"anchor's own ({cube.code(0)})", REF_COLOR, 2.2, "-"),
        (cube.all_donor, f"all donor ({cube.code(cube.all_donor)})", ALL_DONOR_COLOR, 1.8, "-"),
    ]
    lines += [(cube.single_in(i), f"{cube.letters[i]} only ({f})", FACTOR_COLORS[f], 1.3, "-") for i, f in enumerate(cube.factors)]
    lines += [(cube.all_but(i), f"all but {cube.letters[i]}", FACTOR_COLORS[f], 0.9, ":") for i, f in enumerate(cube.factors)]
    return lines


def _pair_notes(kind: str, label: str, subtask: str | None, row: dict, cube: Cube) -> str:
    """The pair's numbers as one monospace block, for the example figure and the dashboard rail."""
    order = np.argsort(row["rms"])[::-1]
    table = "\n".join(
        f"{cube.label(m):>10}  {cube.code(m)}  {row['rms'][m]:.4f}   {row['rms_abs_deg'][m]:6.2f} deg" for m in order if m
    )
    shares = "  ".join(f"{cube.letters[i]} {row['shares'][f]:.2f}" for i, f in enumerate(cube.factors))
    interactions = sum(row["shares"][key] for key in cube.interaction_keys)
    gap = f"arm gap {row['donor']['arm_gap_deg']:.1f} deg, gripper gap {row['donor']['gripper_gap']:.0f}"
    if row["donor"]["time_gap_s"] is not None:
        gap = f"time gap {row['donor']['time_gap_s']:.0f} s, " + gap
    same_text = "" if row["subtask_differs"] else "   (same text: the T cells duplicate their partners)"
    return (
        f"donor: {KIND_LABELS[kind]}  ({gap});  {label};  cube spread {row['shares']['spread']:.4f}\n"
        f"subtask  anchor: {subtask!r}   donor: {row['donor_subtask']!r}{same_text}\n"
        f"variance shares  {shares}  interactions {interactions:.2f}\n"
        f"demonstrations' gap {row['demo_gap']:.4f}   all-donor displacement {row['rms'][cube.all_donor]:.4f}\n\n"
        f"      cell  code  from anchor   RMS deg\n{table}"
    )


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
    cube: Cube,
) -> None:
    from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics

    unnorm = chunks["unnorm"]  # [n_cells, T, D] degrees
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

    lines = _cell_lines(cube)
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
    ax_text.text(
        0.0, 0.62, _pair_notes(kind, percentile_label, anchor_frame["subtask"], row, cube),
        fontsize=7.2, family="monospace", va="top", transform=ax_text.transAxes,
    )
    fig.suptitle(
        f"input swap — anchor ep{anchor_frame['episode_idx']} fr{anchor_frame['frame_idx']} "
        f"← donor ep{donor_frame['episode_idx']} fr{donor_frame['frame_idx']}   "
        "(joint panels in degrees; every cell at one flow seed; depth and metadata are the anchor's throughout)",
        fontsize=10.5,
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _trace_record(anchor_frame: dict, donor_frame: dict, row: dict, chunks: dict, label: str, cube: Cube) -> dict:
    """What the dashboard needs for one pair: the cube in degrees, both demonstrations
    and states, both frames' cameras, the notes block."""
    from lerobot.probes.action_trace_probe import _image_data_uri

    cameras = []
    for who, frame in (("anchor", anchor_frame), ("donor", donor_frame)):
        for key in sorted(k for k in frame["obs"] if str(k).startswith("observation.images.")):
            cameras.append({"label": f"{who} · {str(key).rsplit('.', 1)[-1]}", "src": _image_data_uri(frame["obs"][key])})
    return {
        "label": (
            f"{label}: anchor ep{anchor_frame['episode_idx']} fr{anchor_frame['frame_idx']} ← donor "
            f"ep{donor_frame['episode_idx']} fr{donor_frame['frame_idx']} ({KIND_LABELS[row['donor_kind']]})"
        ),
        "slider": f"ep{anchor_frame['episode_idx']}:{anchor_frame['frame_idx']}←ep{donor_frame['episode_idx']}:{donor_frame['frame_idx']}",
        "subtask": f"anchor: {anchor_frame['subtask'] or '(no clause)'}   |   donor: {donor_frame['subtask'] or '(no clause)'}",
        "cameras": cameras,
        "notes": _pair_notes(row["donor_kind"], label, anchor_frame["subtask"], row, cube),
        "cells_deg": chunks["unnorm"],
        "gt_anchor_deg": chunks["gt_anchor_deg"],
        "gt_donor_deg": chunks["gt_donor_deg"],
        "anchor_state": chunks["anchor_state"],
        "donor_state": chunks["donor_state"],
    }


def _render_trace(records: list[dict], output_path: str, joint_names: list[str], fps: float, table_z: float, cube: Cube) -> None:
    """Action-inspector dashboard over the example pairs (``action_trace_probe``'s shell).

    Left: every cell's end-effector path and both demonstrations in one orbitable scene.
    Right: both arms as measured, then the wrist-roll and gripper timelines — the two joints
    the scene cannot show. The slider steps through the pairs; the rail beside the plot
    shows both frames' cameras and the pair's numbers. Paths are Butterworth-filtered, as
    the runtime sends them; demonstrations are raw.
    """
    import html as html_lib

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from lerobot.probes.action_trace_probe import _write_dashboard_html
    from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics

    kin = RebotKinematics()
    lines = _cell_lines(cube)
    dashes = {"-": "solid", ":": "dot"}
    for rec in records:
        rec["cells"] = np.stack([apply_butterworth_filter(c.astype(np.float64)) for c in rec["cells_deg"]])
        rec["ee"] = np.stack([kin.ee_path(c) for c in rec["cells"]])
        rec["gt"] = [rec["gt_anchor_deg"].astype(np.float64), rec["gt_donor_deg"].astype(np.float64)]
        rec["gt_ee"] = [kin.ee_path(gt) for gt in rec["gt"]]
        rec["states"] = [rec["anchor_state"].astype(np.float64), rec["donor_state"].astype(np.float64)]
        rec["start"] = [kin.ee_path(s[None])[0] for s in rec["states"]]
        rec["skeleton"] = [kin.link_origins(s[None])[0] for s in rec["states"]]

    points = np.concatenate(
        [rec["ee"].reshape(-1, 3) for rec in records]
        + [np.concatenate(rec["gt_ee"]) for rec in records]
        + [np.concatenate(rec["skeleton"]) for rec in records]
    )
    lo, hi = points.min(axis=0), points.max(axis=0)
    lo[2], hi[2] = min(lo[2], table_z), max(hi[2], table_z)
    centre, half = (lo + hi) / 2.0, max((hi - lo).max() / 2.0, 0.05) * 1.05
    lo, hi = centre - half, centre + half

    horizon = records[0]["cells"].shape[1]
    steps = np.arange(horizon)
    demo_style = [("anchor's demonstration", "#666666"), ("donor's demonstration", "#f08080")]
    who_style = [("anchor", REF_COLOR), ("donor", ALL_DONOR_COLOR)]
    roll, grip = 5, 6

    def path3d(ee, chunk, name, color, width, dash):
        return go.Scatter3d(
            x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines+markers",
            line=dict(color=color, width=width, dash=dash), marker=dict(size=2, color=color), name=name,
            customdata=np.column_stack([steps, steps / fps * 1000.0, chunk[:, roll], chunk[:, grip]]),
            hovertemplate=(
                f"<b>{name}</b><br>step %{{customdata[0]:.0f}} · %{{customdata[1]:.0f}} ms"
                "<br>x %{x:.3f} m · y %{y:.3f} m · z %{z:.3f} m"
                f"<br>{joint_names[roll]} %{{customdata[2]:.1f}}° · {joint_names[grip]} %{{customdata[3]:.1f}}<extra></extra>"
            ),
        )

    def context3d(ee, color, width, dash="solid"):
        return go.Scatter3d(
            x=ee[:, 0], y=ee[:, 1], z=ee[:, 2], mode="lines", line=dict(color=color, width=width, dash=dash),
            opacity=0.7, showlegend=False, hoverinfo="skip",
        )

    def record_traces(rec):
        out = []
        for mask, label, color, width, style in lines:
            out.append((path3d(rec["ee"][mask], rec["cells"][mask], label, color, 2.5 * width + 1.0, dashes[style]), 1, 1))
        for (name, color), ee, gt in zip(demo_style, rec["gt_ee"], rec["gt"]):
            out.append((path3d(ee, gt, name, color, 3.0, "dash"), 1, 1))
        for (who, color), start in zip(who_style, rec["start"]):
            out.append((go.Scatter3d(
                x=[start[0]], y=[start[1]], z=[start[2]], mode="markers", name=f"{who} pose now",
                marker=dict(size=7, color=color, symbol="diamond", line=dict(color="#ffffff", width=1)),
                hovertemplate=f"{who} measured pose<extra></extra>",
            ), 1, 1))
        for (who, color), skeleton in zip(who_style, rec["skeleton"]):
            out.append((go.Scatter3d(
                x=skeleton[:, 0], y=skeleton[:, 1], z=skeleton[:, 2], mode="lines+markers",
                line=dict(color=color, width=6), marker=dict(size=3, color=color), opacity=0.55,
                showlegend=False, hovertemplate=f"{who} arm now<extra></extra>",
            ), 1, 2))
        for mask, label, color, width, style in lines:
            out.append((context3d(rec["ee"][mask], color, 1.5 * width + 1.0, dashes[style]), 1, 2))
        for (name, color), ee in zip(demo_style, rec["gt_ee"]):
            out.append((context3d(ee, color, 2.0, "dash"), 1, 2))
        for row, joint in ((2, roll), (3, grip)):
            for (who, color), state in zip(who_style, rec["states"]):
                out.append((go.Scatter(
                    x=[-1], y=[state[joint]], mode="markers", marker=dict(color=color, size=8, symbol="diamond"),
                    showlegend=False, hovertemplate=f"{who} {joint_names[joint]} now: %{{y:.1f}}<extra></extra>",
                ), row, 2))
            for mask, label, color, width, style in lines:
                out.append((go.Scatter(
                    x=steps, y=rec["cells"][mask][:, joint], mode="lines",
                    line=dict(color=color, width=1.6 * width + 0.5, dash=dashes[style]), showlegend=False,
                    hovertemplate=f"{label}<br>step %{{x}}: %{{y:.1f}}<extra></extra>",
                ), row, 2))
            for (name, color), gt in zip(demo_style, rec["gt"]):
                out.append((go.Scatter(
                    x=steps, y=gt[:, joint], mode="lines", line=dict(color=color, width=2, dash="dash"),
                    showlegend=False, hovertemplate=f"{name}<br>step %{{x}}: %{{y:.1f}}<extra></extra>",
                ), row, 2))
        return out

    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "scene", "rowspan": 3}, {"type": "scene"}], [None, {"type": "xy"}], [None, {"type": "xy"}]],
        column_widths=[0.72, 0.28], row_heights=[0.40, 0.30, 0.30], horizontal_spacing=0.04, vertical_spacing=0.11,
        subplot_titles=(f"End-effector paths of the {cube.n_cells} cells", "Both arms now", joint_names[roll], joint_names[grip]),
    )
    fig.add_trace(go.Mesh3d(
        x=[lo[0], hi[0], hi[0], lo[0]], y=[lo[1], lo[1], hi[1], hi[1]], z=[table_z] * 4, i=[0, 0], j=[1, 2], k=[2, 3],
        color="#C8B89A", opacity=0.30, name="table plane", showlegend=True, hoverinfo="skip",
    ), row=1, col=2)
    static_count = len(fig.data)
    for trace, row, col in record_traces(records[0]):
        fig.add_trace(trace, row=row, col=col)
    # Frame traces bypass add_trace's subplot binding; read it back off the initial traces.
    bindings = [
        (getattr(trace, "scene", None), getattr(trace, "xaxis", None), getattr(trace, "yaxis", None))
        for trace in fig.data[static_count:]
    ]
    updated = list(range(static_count, len(fig.data)))
    frames = []
    for idx, rec in enumerate(records):
        dynamic = []
        for (trace, _, _), (scene, xaxis, yaxis) in zip(record_traces(rec), bindings, strict=True):
            if scene is not None:
                trace.scene = scene
            else:
                trace.xaxis, trace.yaxis = xaxis, yaxis
            dynamic.append(trace)
        frames.append(go.Frame(
            data=dynamic, traces=updated, name=str(idx),
            layout=dict(title=dict(text=f"<b>{html_lib.escape(rec['label'])}</b>")),
        ))
    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"<b>{html_lib.escape(records[0]['label'])}</b>", font=dict(size=15), x=0.01, xanchor="left"),
        scene=dict(
            xaxis=dict(title="x (m)"), yaxis=dict(title="y (m)"), zaxis=dict(title="z (m)"),
            aspectmode="data", bgcolor="#FBFBFC", uirevision="input-swap-camera",
        ),
        scene2=dict(
            xaxis=dict(title="", range=[float(lo[0]), float(hi[0])], showticklabels=False),
            yaxis=dict(title="", range=[float(lo[1]), float(hi[1])], showticklabels=False),
            zaxis=dict(title="", range=[float(lo[2]), float(hi[2])], showticklabels=False),
            aspectmode="cube", bgcolor="#FBFBFC",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.05), up=dict(x=0, y=0, z=1)), uirevision="input-swap-pose-camera",
        ),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFB", height=880, margin=dict(l=10, r=10, b=35, t=115),
        legend=dict(bgcolor="rgba(255,255,255,0.88)", bordercolor="#DDDDDD", borderwidth=1, font=dict(size=10), x=0.01, y=0.99),
        hoverlabel=dict(bgcolor="white", font_size=12),
        sliders=[dict(
            active=0, y=-0.02, x=0.04, len=0.92, pad=dict(t=45), currentvalue=dict(prefix="pair ", font=dict(size=12)),
            steps=[dict(
                method="animate", label=rec["slider"], value=str(idx),
                args=[[str(idx)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            ) for idx, rec in enumerate(records)],
        )],
    )
    for row in (2, 3):
        fig.update_xaxes(title="now (−1) → chunk step", range=[-1, horizon - 1], row=row, col=2)
        fig.update_yaxes(title="dataset units", zeroline=False, row=row, col=2)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color="#333333")

    contexts = [
        {"label": rec["label"], "subtask": rec["subtask"], "cameras": rec["cameras"], "notes": rec["notes"], "trajectory": {}}
        for rec in records
    ]
    _write_dashboard_html(
        fig, [], output_path, contexts=contexts,
        page_title="Input Swap · action inspector",
        subtitle=(
            f"One anchor frame, one donor frame, the {cube.n_cells} chunks of the cube between them at one flow seed. "
            "Orbit the end-effector paths, read which frame each cell follows, then check the two frames on the right."
        ),
        legend_note=(
            "Black: the anchor's own chunk. Red: the donor's whole frame and subtask text. Solid colours: one switch flipped "
            f"to the donor ({', '.join(FACTOR_LEGEND[f] for f in cube.factors)}). Dotted: all but one. Dashed grey / pink: the anchor's / "
            "the donor's demonstration. Diamonds: the anchor's (black) and the donor's (red) measured pose. Paths are "
            "Butterworth-filtered as the runtime sends them; demonstrations are raw."
        ),
        warning=(
            "STATE CELLS START AT THE DONOR",
            "A cell whose state (S) comes from the donor is decoded relative to the donor's pose, so its path starts at "
            "the red diamond. Read those cells against the donor's demonstration, the others against the anchor's.",
        ),
    )


# ── Manifest ──────────────────────────────────────────────────────────────────


def _write_manifest(output_dir: str, summary: dict, example_files: list[tuple[str, str]], cube: Cube) -> dict:
    reseed = summary["reseed_distance"]["median"]
    metrics = []
    for kind in DONOR_KINDS:
        for factor in cube.factors:
            metrics.append(
                Metric(
                    f"kinds.{kind}.headline.single_in_displacement.{factor}",
                    f"{kind}: chunk displacement when only the {factor} comes from the donor",
                    fmt=3,
                    baseline=reseed,
                    primary=(kind == "matched"),
                    trend=(kind == "matched"),
                    note=(
                        "Median over the pairs whose subtask texts differ of the normalized RMS distance from the anchor's own chunk (a shared text makes this cell a duplicate)."
                        if factor == "subtask"
                        else "Median over anchors of the normalized RMS distance from the anchor's own chunk."
                    ) + " The baseline is the median distance between two flow seeds on the same input, for scale.",
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
        for factor in cube.factors:
            metrics.append(
                Metric(
                    f"kinds.{kind}.headline.variance_share.{factor}",
                    f"{kind}: share of the cube's variance from the {factor}",
                    fmt=2,
                    primary=(kind == "matched"),
                    note=f"Fraction of the {cube.n_cells} chunks' total variance carried by this switch's main effect, median over anchors.",
                )
            )
        metrics.append(Metric(f"kinds.{kind}.headline.interactions", f"{kind}: share of the cube's variance from interactions", fmt=2))
        metrics.append(
            Metric(
                f"kinds.{kind}.headline.subtask_differs_fraction",
                f"{kind}: fraction of pairs whose subtask texts differ",
                fmt=2,
                good="none",
                note="The T switch is a no-op on the other pairs; every subtask-specific number is read over these.",
            )
        )
        metrics.append(
            Metric(
                f"kinds.{kind}.headline.policy_over_demo_gap",
                f"{kind}: policy displacement over the demonstrations' gap",
                fmt=2,
                baseline=1.0,
                primary=(kind == "matched"),
                note=f"$\\|a_{{{cube.code(cube.all_donor)}}}-a_{{{cube.code(0)}}}\\| / \\|a^\\star_{{donor}}-a^\\star_{{anchor}}\\|$ per pair, median. Below 1: the policy's chunk differs less between the two frames than the humans' did.",
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
                "flow seed. Left: one switch flipped to the donor. Middle: all but one flipped. Right: the donor's whole "
                "frame and subtask text under the anchor's depth. A switch whose box sits near the reseed line moves the "
                "chunk as much as changing the noise does; one near the bottom is barely read. T cells are read over the "
                "pairs whose subtask texts differ."
            ),
            primary=True,
        ),
        Panel("variance_shares.png", "Which switch the cube varies with", how="Stacked main-effect shares per donor kind; the greys are interactions.", primary=True),
        Panel("per_joint.png", "Where a single switch's displacement lands", how="Per joint and per chunk step, median over anchors, one switch flipped to the donor."),
        Panel("demo_gap.png", "Policy displacement against the demonstrations' own gap", how="Diagonal: the policy varies between the two frames as much as the humans did.", primary=True),
        Panel("state_follow.png", "S only, in degrees", how="Ratio of the chunk's absolute shift to the state shift per step: 1 is a delta riding on the state."),
        Panel("anchors.png", "Where the anchors sit and where their matched donors come from", how="Colour is the arm-joint gap to the matched donor; a line joins each anchor to its donor's position."),
        Panel("input_swap.json", "Summary and every pair", how=f"Per pair: donor provenance and gaps, the {cube.n_cells} cells' displacements (normalized, from the all-donor cell, and in degrees), per-joint and per-step displacement, variance shares, demonstrations' gap, state-follow curve."),
    ]
    panels.append(Panel(
        "examples/trace.html", "Action inspector over the example pairs",
        how=f"Orbit the {cube.n_cells} cells' end-effector paths with both demonstrations; the slider steps through the nine example pairs. Cells with the donor's state start at the donor's pose.",
        primary=True,
    ))
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
        claim=f"When another frame's {cube.streams_text} is injected, which one does the action chunk follow?",
        summary=summary,
        metrics=metrics,
        panels=panels,
        see_also=["depth_modality", "subtask_sweep", "action_trace"],
    )


def _write_pairs_manifest(output_dir: str, summary: dict, example_files: list[tuple[str, str]], cube: Cube) -> dict:
    reseed = summary["reseed_distance"]["median"]
    metrics = []
    for name, block in summary["pairs"].items():
        where = f"ep{block['anchor']['episode']} fr{block['anchor']['frame']} ← ep{block['donor']['episode']} fr{block['donor']['frame']}"
        for factor in cube.factors:
            metrics.append(Metric(
                f"pairs.{name}.headline.single_in_displacement.{factor}",
                f"{name} ({where}): chunk displacement when only the {factor} comes from the donor",
                fmt=3, baseline=reseed, primary=(factor == "image"), trend=True,
                note="Normalized RMS distance from the anchor's own chunk. The baseline is the distance between two flow seeds on the same input, for scale.",
            ))
        metrics.append(Metric(
            f"pairs.{name}.headline.all_donor_displacement", f"{name} ({where}): chunk displacement for the donor's whole frame",
            fmt=3, baseline=reseed,
        ))
        for factor in cube.factors:
            metrics.append(Metric(
                f"pairs.{name}.headline.variance_share.{factor}", f"{name} ({where}): share of the cube's variance from the {factor}", fmt=2,
            ))
    panels = [
        Panel(
            "trace.html", "Action inspector over the hand-picked pairs",
            how=(
                f"Orbit the {cube.n_cells} cells' end-effector paths with both demonstrations; the slider steps through the pairs and "
                "the rail shows both frames. A cell that follows the images heads for the donor's target from the "
                "anchor's pose; a cell whose state comes from the donor starts at the donor's pose."
            ),
            primary=True,
        ),
    ]
    for file, caption in example_files:
        panels.append(Panel(
            file, caption,
            how="Top: the anchor's and the donor's frames. Rows 2-3: every cell's chunk in degrees; black is the anchor's own, red the donor's whole frame, solid colours one stream from the donor, dotted two streams. Dashed grey/pink: the two demonstrations.",
            align=f"examples/{os.path.basename(file).rsplit('_ep', 1)[0]}.png",
            primary=True,
        ))
    panels.append(Panel("input_swap.json", "Every pair", how=f"Per pair: donor provenance and gaps, the {cube.n_cells} cells' displacements, variance shares, demonstrations' gap, state-follow curve."))
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Input Swap — hand-picked pairs",
        group="Sensitivity",
        claim="On frames chosen by hand, which frame's inputs does the action chunk follow?",
        summary=summary,
        metrics=metrics,
        panels=panels,
        see_also=["input_swap", "action_trace"],
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
    cube = CUBE

    table = _frame_table(dataset, stride)
    grid = table["global_idx"]
    pairs: list[tuple[int, int]] = []
    for item in str(getattr(cfg, "pairs", "") or "").split(","):
        if item.strip():
            anchor_s, donor_s = item.split(":")
            pairs.append((_snap(grid, int(anchor_s)), _snap(grid, int(donor_s))))
    explicit = [int(s) for s in str(getattr(cfg, "frame_indices", "") or "").split(",") if s.strip()]
    kinds = ("picked",) if pairs else DONOR_KINDS
    if pairs:
        anchors = [anchor for anchor, _ in pairs]
    elif explicit:
        anchors = [_snap(grid, idx) for idx in explicit]
    else:
        n_per_episode = int(getattr(p, "input_swap_n_frames_per_episode", None) or p.n_frames_per_episode)
        anchors = [g for _, _, g in sample_episodes_evenly(dataset, n_per_episode, p.max_episodes, seed, stride)]
    if not anchors:
        logging.warning("[input_swap] no anchors selected.")
        return

    episode_lengths = {ep: len(idx) for ep, idx in build_episode_index(dataset).items()}
    logging.info(
        f"[input_swap] {len(anchors)} anchors x {len(kinds)} donors x {cube.n_cells} cells ({cube.letters}; +{n_reseeds} reseeds) = "
        f"{len(anchors) * len(kinds)} stacked forwards; seed {seed}, stride {stride}"
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
            groups, fixed = _factor_keys(anchor["obs"], cube)
            n_joints = int(anchor["gt_actions"].shape[-1])
            anchor_state = anchor["state"].numpy().astype(np.float64)
            gt_anchor_norm = adapter.normalize_gt_actions(anchor["gt_actions"], anchor["state"]).numpy()[:, :n_joints]
            if pairs:
                donors = _picked_donor(table, anchor_gidx, pairs[a_idx][1], fps)
            else:
                donors = _choose_donors(table, anchor_gidx, fps, rng)

            for k_idx, kind in enumerate(kinds):
                donor_info = donors[kind]
                donor = probe_frame_inputs(dataset, cfg, donor_info["global_idx"], chunk_size)
                cells = [_cell(anchor["obs"], donor["obs"], mask, groups, fixed, cube) for mask in range(cube.n_cells)]
                subtasks = _cell_subtasks(anchor["subtask"], donor["subtask"], cube)
                noise = base_noise.expand(cube.n_cells, *base_noise.shape[1:])
                if k_idx == 0:
                    # The reseeds ride on the first donor's batch: same anchor input, other noise.
                    cells += [cells[0]] * n_reseeds
                    subtasks += [subtasks[0]] * n_reseeds
                    noise = torch.cat([noise, reseed_noise])
                unnorm_t, norm_t = adapter.predict_action_chunk_stacked(
                    cells,
                    anchor["task"],
                    subtask=subtasks,
                    metadata=anchor["metadata"],
                    noise=noise.contiguous(),
                    inference_action_mode="continuous",
                )
                horizon = min(int(norm_t.shape[1]), chunk_size)
                norm = norm_t[:, :horizon, :n_joints].numpy().astype(np.float64)
                unnorm = unnorm_t[:, :horizon, :n_joints].numpy().astype(np.float64)
                cells_norm, cells_deg = norm[: cube.n_cells], unnorm[: cube.n_cells]

                if consistency is None:
                    # The all-anchor row against the one-frame path at the same seed: how much
                    # batching alone moves a chunk. Every contrast stays inside one batch.
                    _, single = adapter.predict_action_chunk_batch(
                        anchor["obs"], anchor["task"], [anchor["subtask"]], metadatas=[anchor["metadata"]],
                        noise=base_noise, inference_action_mode="continuous",
                    )
                    consistency = float(np.abs(single[0, :horizon, :n_joints].numpy() - cells_norm[0]).max())
                    logging.info(f"[input_swap] stacked vs single-frame forward, max |Δ| = {consistency:.2e}")

                gt_donor_norm = adapter.normalize_gt_actions(donor["gt_actions"], donor["state"]).numpy()[:, :n_joints]
                donor_state = donor["state"].numpy().astype(np.float64)
                follow = _state_follow(cells_deg, anchor_state[:n_joints], donor_state[:n_joints], table["state_std"][:n_joints])
                rows.append({
                    "donor_kind": kind,
                    "anchor": {"global_idx": int(anchor_gidx), "episode": int(anchor["episode_idx"]), "frame": int(anchor["frame_idx"])},
                    "donor": donor_info,
                    "subtask": anchor["subtask"],
                    "donor_subtask": donor["subtask"],
                    "subtask_differs": (anchor["subtask"] or "") != (donor["subtask"] or ""),
                    "rms": [float(_rms(cells_norm[m] - cells_norm[0])) for m in range(cube.n_cells)],
                    "rms_vs_all_donor": [float(_rms(cells_norm[m] - cells_norm[cube.all_donor])) for m in range(cube.n_cells)],
                    "rms_abs_deg": [float(_rms(cells_deg[m] - cells_deg[0])) for m in range(cube.n_cells)],
                    "per_joint": [_rms(cells_norm[m] - cells_norm[0], axis=0).tolist() for m in range(cube.n_cells)],
                    "per_step": [_rms(cells_norm[m] - cells_norm[0], axis=1).tolist() for m in range(cube.n_cells)],
                    "main_effect": _main_effects(cells_norm, cube),
                    "shares": _factorial_shares(cells_norm, cube),
                    "demo_gap": float(_rms(gt_donor_norm[:horizon] - gt_anchor_norm[:horizon])),
                    "reseed_distance": float(np.mean([_rms(norm[cube.n_cells + k] - cells_norm[0]) for k in range(n_reseeds)])) if k_idx == 0 else None,
                    "state_follow": follow.tolist() if follow is not None else None,
                })
                chunks_by_pair[(int(anchor_gidx), int(donor_info["global_idx"]))] = {
                    "unnorm": cells_deg,
                    "gt_anchor_deg": anchor["gt_actions"].numpy()[:horizon, :n_joints].astype(np.float64),
                    "gt_donor_deg": donor["gt_actions"].numpy()[:horizon, :n_joints].astype(np.float64),
                    "anchor_state": anchor_state[:n_joints],
                    "donor_state": donor_state[:n_joints],
                }

            focus_kind = "picked" if pairs else "matched"
            focus = next(r for r in rows[-len(kinds):] if r["donor_kind"] == focus_kind)
            elapsed = time.time() - started
            singles = "/".join(f"{focus['rms'][cube.single_in(i)]:.3f}" for i in range(len(cube.factors)))
            logging.info(
                f"[input_swap] anchor {a_idx + 1}/{len(anchors)} ep{anchor['episode_idx']} fr{anchor['frame_idx']}"
                f"  {focus_kind} (arm gap {focus['donor']['arm_gap_deg']:.1f} deg) {'/'.join(cube.letters)} only = "
                f"{singles}  all {focus['rms'][cube.all_donor]:.3f}"
                f"{'' if focus['subtask_differs'] else '  [same subtask text]'}"
                f"  ({elapsed / (a_idx + 1):.1f} s/anchor)"
            )
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    horizon = len(rows[0]["per_step"][0])
    n_joints = len(rows[0]["per_joint"][0])
    joint_names = joint_names_for_dim(n_joints)
    # (row, file, caption, label) per example figure; the same pairs feed the dashboard.
    examples: list[tuple[dict, str, str, str]] = []
    if pairs:
        summary = _summarize_pairs(rows, horizon, cube)
        for k, row in enumerate(rows, start=1):
            file = f"examples/pair{k:02d}_ep{row['anchor']['episode']:04d}_fr{row['anchor']['frame']:06d}.png"
            caption = (
                f"pair {k}: anchor ep{row['anchor']['episode']} fr{row['anchor']['frame']} "
                f"← donor ep{row['donor']['episode']} fr{row['donor']['frame']}"
            )
            examples.append((row, file, caption, f"hand-picked pair {k}"))
    else:
        summary = _summarize(rows, horizon, cube)
        _render_distributions(rows, summary, os.path.join(output_dir, "input_swap.png"), cube)
        _render_per_joint(summary, joint_names, os.path.join(output_dir, "per_joint.png"), cube)
        _render_shares(summary, os.path.join(output_dir, "variance_shares.png"), cube)
        _render_anchors(rows, episode_lengths, fps, os.path.join(output_dir, "anchors.png"))
        _render_demo_gap(rows, summary, os.path.join(output_dir, "demo_gap.png"), cube)
        _render_state_follow(summary, os.path.join(output_dir, "state_follow.png"), cube)
        # Examples stratified by the cube's spread: concentrated, typical, spread out.
        for kind in DONOR_KINDS:
            rows_k = sorted((r for r in rows if r["donor_kind"] == kind), key=lambda r: r["shares"]["spread"])
            for pct in (10, 50, 90):
                row = rows_k[min(int(round(pct / 100 * (len(rows_k) - 1))), len(rows_k) - 1)]
                file = f"examples/{kind}_p{pct:02d}_ep{row['anchor']['episode']:04d}_fr{row['anchor']['frame']:06d}.png"
                caption = f"{KIND_LABELS[kind]}, spread p{pct}: anchor ep{row['anchor']['episode']} fr{row['anchor']['frame']}"
                examples.append((row, file, caption, f"p{pct} of the cube's spread"))
    summary["batch_consistency_max_abs"] = consistency
    summary["seed"] = seed
    summary["stride"] = stride
    summary["same_episode_min_gap_s"] = SAME_EPISODE_MIN_GAP_S
    summary["match_gripper_tol"] = MATCH_GRIPPER_TOL

    example_files: list[tuple[str, str]] = []
    trace_records: list[dict] = []
    for row, file, caption, label in examples:
        anchor = probe_frame_inputs(dataset, cfg, row["anchor"]["global_idx"], chunk_size)
        donor = probe_frame_inputs(dataset, cfg, row["donor"]["global_idx"], chunk_size)
        chunks = chunks_by_pair[(row["anchor"]["global_idx"], row["donor"]["global_idx"])]
        _render_example(os.path.join(output_dir, file), row["donor_kind"], label, anchor, donor, row, chunks, joint_names, fps, cube)
        example_files.append((file, caption))
        trace_records.append(_trace_record(anchor, donor, row, chunks, label, cube))
    trace_file = "trace.html" if pairs else "examples/trace.html"
    _render_trace(trace_records, os.path.join(output_dir, trace_file), joint_names, fps, float(p.trace_table_z), cube)

    with open(os.path.join(output_dir, "input_swap.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    if pairs:
        _write_pairs_manifest(output_dir, summary, example_files, cube)
        for name, block in summary["pairs"].items():
            head = block["headline"]
            single = "  ".join(f"{cube.letters[i]} {head['single_in_displacement'][f]:.3f}" for i, f in enumerate(cube.factors))
            logging.info(
                f"[input_swap] {name} ep{block['anchor']['episode']} fr{block['anchor']['frame']} ← "
                f"ep{block['donor']['episode']} fr{block['donor']['frame']}: single-stream displacement {single}  "
                f"all donor {head['all_donor_displacement']:.3f}  demo gap {block['demo_gap']:.3f}"
                f"{'' if block['subtask_differs'] else '  [same subtask text: T is a no-op]'}"
            )
    else:
        _write_manifest(output_dir, summary, example_files, cube)
        for kind in DONOR_KINDS:
            head = summary["kinds"][kind]["headline"]
            shares = "  ".join(f"{cube.letters[i]} {_fmt(head['variance_share'][f], 2)}" for i, f in enumerate(cube.factors))
            single = "  ".join(f"{cube.letters[i]} {_fmt(head['single_in_displacement'][f], 3)}" for i, f in enumerate(cube.factors))
            block = summary["kinds"][kind]
            logging.info(
                f"[input_swap] {kind}: variance shares {shares}  single-stream displacement {single}  "
                f"all donor {head['all_donor_displacement']:.3f}  policy/demo gap {head['policy_over_demo_gap']:.2f}  "
                f"(subtask text differs on {block['subtask_differs_n']}/{block['n_pairs']} pairs)"
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
    probe_dir = "input_swap_pairs" if str(getattr(cfg, "pairs", "") or "").strip() else "input_swap"
    run(adapter, dataset, cfg, os.path.join(cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", probe_dir))


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
