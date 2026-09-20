r"""Embodiment-swap probe: does "The robot is a ..." reach the actions, and on which side?

The action prompt opens with an embodiment clause, ``The robot is {a/an} {name}.``
(``_build_robot_text``, processor_molmoact2.py), rendered from the sample's
``embodiment_index`` with no training dropout. Two facts about the training mixture
decide what a swap can mean:

  * Every diverse-corpus row carries its own name (Franka Panda, UR5, ARX5, UR7e, …), so
    for a diverse frame the *training-regime* prompt is the one with its own label.
  * ReBot rows carry NO clause. ``RoleAlignedBuffer`` stamps ``action_layout_id`` only,
    and with ``policy.embodiment`` unset the processor's default index is $-1$, which
    omits the sentence. For a ReBot frame the training-regime prompt is therefore the
    clause-free one, and ``The robot is a Rebot B601.`` is itself a prompt the model has
    never seen on ReBot images. (If ``policy.embodiment`` is set the probe reads that as
    ReBot's home label instead; the provenance box says which.)

So the probe has a *home* prompt per frame — clause-free on ReBot, own label on diverse —
and every other label is a swap measured from it.

The second sentence of the prompt is the control-mode clause, ``The control mode is joint
space.`` / ``... end-effector space.``, rendered on every row from the layout record behind
``action_layout_id`` (``ACTION_LAYOUTS[id].control_mode``; MolmoAct rows are the only
end-effector layout). It is the second axis: the home prompt is the home name at the home
mode — ReBot: no name + joint (``cfg.diverse.rebot_layout``); diverse: own name + own
layout's mode — and the grid is every name x every mode. Every cell is one row of a single
batched ``predict_action_chunk_batch`` on the same frame under the same seeded flow noise
and the same deployment metadata clause, so the only things that vary are the two clauses:

  names                  ``none`` (no embodiment clause) + one per name in
                         ``EMBODIMENT_NAMES`` — Franka Panda, UR5, UR7e, ARX5, Rebot B601,
                         SO-101, ALOHA, YAM — of which the ones present in this run's
                         training selection are *trained* and the rest *unseen* strings
  modes                  ``none`` (no control-mode clause), ``joint``, ``end_effector``
  seed floor             the home prompt re-drawn under different flow seeds

Columns: ``{name}_sep`` is the name axis at the home mode (the pre-clause report shape),
``{name}@{mode}_sep`` the full grid, ``mode_none_sep`` / ``mode_swapped_sep`` the mode axis
at the home name (clause removed / the other space). ``none@none`` on a ReBot frame is the
legacy prompt a pre-clause checkpoint trained on.

Write $a^{(L)}$ for the normalized chunk under label $L$, $a^{(h)}$ for the home chunk,
and $\lVert x \rVert = \sqrt{\frac{1}{TD}\sum_{t,d} x_{t,d}^{2}}$ for the RMSE norm. The
one number everything is read in is the separation

$$S(L)=\frac{\lVert a^{(L)}-a^{(h)}\rVert}{\operatorname{mean}_{s\neq s'}\lVert a^{(s)}-a^{(s')}\rVert}$$

— the swap's displacement over the seed floor, $S\approx1$ meaning the name did nothing
the sampler was not already doing. It is a within-frame ratio, so ReBot frames and diverse
frames sit on one scale with no cross-domain normalization in the way, and the diverse
chunks are normalized with their own row's statistics (the row's identity columns ride
into the pack step) exactly as in training.

**1. ReBot frames under other robots' names.** $S(\text{Franka Panda})$, $S(\text{UR5})$,
… on ReBot images, plus $S(\text{Rebot B601})$ — the never-trained "correct" label — and
the unseen strings as a control for "any new sentence moves the chunk". If the trained
names move the chunk and the unseen ones do not, the name is read as a robot and not as
tokens.

**2. Diverse frames with the label swapped.** $S(\text{none})$ — what removing the clause
the model always saw does — against $S(\text{Rebot B601})$ and the other trained names,
broken down by the frame's own robot. The direct answer to "does the ReBot label carry
ReBot behaviour onto another robot's frames".

**3. One direction, or one per name?** Within a frame, the mean pairwise cosine among the
trained foreign displacements $v^{(L)}=a^{(L)}-a^{(h)}$: near $+1$ every foreign name is
one "not this robot" move; near the null they are name-specific. The null is two reseeds
of the home prompt measured from a shared third draw, carrying the same shared endpoint
(the argument is metadata_steering's). Across frames, the leave-one-out cosine of one
designated swap — Franka Panda on ReBot frames, Rebot B601 on diverse frames — against
every other frame's, with the reseed displacement as its null: $+1$ is a global offset
stamped on every frame, $0$ a displacement computed per frame.

**4. Tempo.** Robots differ in speed, so a read name could show up as per-step travel,
$\operatorname{step}(a^{(L)})/\operatorname{step}(a^{(h)})$ with
$\operatorname{step}(x)=\sqrt{\frac{1}{(T-1)D}\sum_{t,d}(x_{t+1,d}-x_{t,d})^{2}}$.

**5. The control-mode clause.** $S$ of the home name under the other space
(``mode_swapped``) and under no clause (``mode_none``), per domain and, on the diverse
side, per home mode: on a MolmoAct frame ``Franka Panda@joint`` is the DROID/FMB training
prompt on a pose-space image, the direct read of whether the clause and not the image
decides the space. The name x mode grid (heatmaps) says whether the two clauses act
independently: a name's row at a constant offset across modes, or the mode rewriting it.

Nothing is scored against the demonstration. Frames: the held-out ReBot split sampled as
metadata_steering samples it, and the diverse holdout — the episodes of
``<root>/holdout_episodes.json`` (``holdout_actor_selection``), never trained on and never
cached, decoded from video — every episode, anchors evenly spaced.

Cost is ``n_frames x ((1 + len(EMBODIMENT_NAMES)) x 3 + n_seeds - 1)`` forwards, two
batched calls per frame.

Registered probe: enable with ``probe_parameters.enable_embodiment_swap``. Standalone::

    uv run python -m lerobot.probes.embodiment_swap --config config_rl.yaml \\
        --policy.pretrained_path outputs/<run>/checkpoints/<step>/pretrained_model
"""

import json
import logging
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.embodiment import EMBODIMENT_NAMES, canonical_embodiment
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.domain_representations import _diverse_inputs
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.metadata_steering import (
    _column,
    _cosine,
    _loo_alignment,
    _median,
    _median_se,
    _pairwise_rmse,
    _rmse,
    _step_rms,
)
from lerobot.probes.utils import (
    DEPLOYMENT_METADATA,
    load_probe_dataset,
    makedirs,
    panel_caption as _caption,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
    sample_episodes_evenly,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeEmbodimentSwapConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters`` (ProbeConfig)."""


REBOT = "rebot"
DIVERSE = "diverse"
NONE = "none"
REBOT_NAME = "Rebot B601"
_DOMAIN_COLOR = {REBOT: "#1f77b4", DIVERSE: "#ff7f0e"}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


LABELS: dict[str, str] = {_slug(name): name for name in EMBODIMENT_NAMES}
CONDITIONS: list[str] = [NONE, *LABELS]
# Control-mode clause per row: the adapter's override strings ("" = no clause).
JOINT, END_EFFECTOR = "joint", "end_effector"
MODES: tuple[str, ...] = (NONE, JOINT, END_EFFECTOR)
GRID: list[tuple[str, str]] = [(name, mode) for name in CONDITIONS for mode in MODES]


def _cell(name: str, mode: str) -> str:
    return f"{name}@{mode}"


def _name_text(name: str) -> str:
    return "" if name == NONE else LABELS[name]


def _mode_text(mode: str) -> str:
    return "" if mode == NONE else mode


def _swapped(mode: str) -> str:
    return END_EFFECTOR if mode == JOINT else JOINT


# ──────────────────────────────────────────────────────────────────────────────
# Frames
# ──────────────────────────────────────────────────────────────────────────────

def _open_holdout(cfg) -> tuple:
    """The diverse holdout buffer (video-decoded, never cached) and the set of names the
    training selection actually renders."""
    from lerobot.datasets.diverse_actor_selection import (
        holdout_actor_selection,
        open_federated_corpus,
        select_actor_anchors,
    )
    from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer
    from lerobot.rl.data_sources.diverse_integration import sample_spec_from_config

    corpus = open_federated_corpus(cfg.diverse.root)
    trained = {canonical_embodiment(row["embodiment"]) for row in select_actor_anchors(corpus).rows}
    buffer = DiverseActorBuffer(
        holdout_actor_selection(corpus), sample_spec_from_config(cfg),
        render_automatic_quality=cfg.diverse.render_automatic_quality,
    )
    return buffer, trained - {None}


def _rebot_home(cfg) -> str:
    configured = canonical_embodiment(getattr(cfg.policy, "embodiment", None))
    return NONE if configured is None else _slug(configured)


def _rebot_mode(cfg) -> str:
    """The clause every ReBot training row renders: the layout ``identity_columns`` stamps."""
    from lerobot.datasets.diverse_actor_selection import action_layout_by_name

    return action_layout_by_name(cfg.diverse.rebot_layout).control_mode


def _layout_mode(extra: dict) -> str:
    """A diverse row's own clause, from the ``action_layout_id`` it carries into the pack step."""
    from lerobot.datasets.diverse_actor_selection import ACTION_LAYOUTS

    return ACTION_LAYOUTS[int(torch.as_tensor(extra["action_layout_id"]).reshape(-1)[0])].control_mode


def _rebot_samples(dataset, cfg, n_frames: int) -> list[dict]:
    p = cfg.probe_parameters
    samples = sample_episodes_evenly(dataset, n_frames, p.max_episodes, p.random_seed, probe_image_stride(cfg))
    return [
        {"domain": REBOT, "source": REBOT, "own": REBOT_NAME, "episode": f"rebot/{ep}",
         "frame": int(fr), "index": int(gidx)}
        for ep, fr, gidx in samples
    ]


def _diverse_samples(buffer, n_frames: int) -> list[dict]:
    """Every holdout episode, ``n_frames`` anchors evenly spaced over each."""
    by_episode: dict[str, list[int]] = {}
    for row_index, row in enumerate(buffer.rows):
        by_episode.setdefault(str(row["episode_id"]), []).append(row_index)
    samples = []
    for episode_id in sorted(by_episode):
        rows = sorted(by_episode[episode_id], key=lambda i: float(buffer.rows[i]["anchor_s"]))
        for pos in np.unique(np.linspace(0, len(rows) - 1, min(n_frames, len(rows)), dtype=int)):
            row = buffer.rows[rows[int(pos)]]
            samples.append({
                "domain": DIVERSE, "source": str(row["source"]),
                "own": canonical_embodiment(row["embodiment"]),
                "episode": f"{row['source']}/{episode_id}", "frame": float(row["anchor_s"]),
                "index": int(rows[int(pos)]),
            })
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Measurement
# ──────────────────────────────────────────────────────────────────────────────

def _measure_frame(
    adapter, inputs: dict, home: str, home_mode: str, trained_slugs: set[str], n_seeds: int
) -> tuple[dict, dict]:
    """Every (name, mode) cell in one forward, the home prompt reseeded in another;
    per-frame row + the displacement vectors the cross-frame geometry needs."""
    n = len(GRID)
    _, chunks = adapter.predict_action_chunk_batch(
        inputs["obs"], inputs["task"], [inputs["subtask"]] * n,
        metadatas=[dict(DEPLOYMENT_METADATA)] * n,
        noise=adapter.flow_noise_like(n, 0),
        embodiments=[_name_text(name) for name, _ in GRID],
        control_modes=[_mode_text(mode) for _, mode in GRID],
        extra_complementary=inputs["extra"],
    )
    acts = {cell: chunks[i] for i, cell in enumerate(GRID)}
    base = acts[(home, home_mode)]

    floor_draws = [base]
    if n_seeds > 1:
        floor_noise = torch.cat([adapter.flow_noise_like(1, seed) for seed in range(1, n_seeds)], dim=0)
        _, floor_chunks = adapter.predict_action_chunk_batch(
            inputs["obs"], inputs["task"], [inputs["subtask"]] * (n_seeds - 1),
            metadatas=[dict(DEPLOYMENT_METADATA)] * (n_seeds - 1),
            noise=floor_noise,
            embodiments=[_name_text(home)] * (n_seeds - 1),
            control_modes=[_mode_text(home_mode)] * (n_seeds - 1),
            extra_complementary=inputs["extra"],
        )
        floor_draws += [floor_chunks[i] for i in range(n_seeds - 1)]
    floor_mean, floor_max = _pairwise_rmse(floor_draws)
    base_step = max(_step_rms(base), 1e-12)

    row = {"home": home, "home_mode": home_mode, "seed_floor_mean": floor_mean, "seed_floor_max": floor_max}
    vectors: dict[str, np.ndarray] = {}
    for name, mode in GRID:
        if (name, mode) == (home, home_mode):
            continue
        cell = _cell(name, mode)
        row[f"{cell}_rmse"] = _rmse(acts[name, mode], base)
        row[f"{cell}_sep"] = row[f"{cell}_rmse"] / max(floor_mean, 1e-9)
        row[f"{cell}_tempo"] = _step_rms(acts[name, mode]) / base_step
        vectors[cell] = (acts[name, mode] - base).flatten().float().cpu().numpy()
    # The name axis at the home mode keeps the pre-clause column names; the mode axis at
    # the home name is "clause removed" and "the other space".
    aliases = {name: _cell(name, home_mode) for name in CONDITIONS if name != home}
    aliases["mode_none"] = _cell(home, NONE)
    aliases["mode_swapped"] = _cell(home, _swapped(home_mode))
    for alias, cell in aliases.items():
        for suffix in ("rmse", "sep", "tempo"):
            row[f"{alias}_{suffix}"] = row[f"{cell}_{suffix}"]
        vectors[alias] = vectors[cell]
    vectors["noise"] = (floor_draws[1] - floor_draws[0]).flatten().float().cpu().numpy() if n_seeds > 1 else None

    foreign_trained = [s for s in trained_slugs if s != home]
    # The ReBot name has its own column on every frame, so it is never an "unseen" control.
    unseen = [s for s in LABELS if s not in trained_slugs and s not in (home, _slug(REBOT_NAME))]
    row["foreign_trained_sep"] = float(np.mean([row[f"{s}_sep"] for s in foreign_trained])) if foreign_trained else None
    row["foreign_unseen_sep"] = float(np.mean([row[f"{s}_sep"] for s in unseen])) if unseen else None
    row["foreign_trained_tempo"] = float(np.mean([row[f"{s}_tempo"] for s in foreign_trained])) if foreign_trained else None
    pairs = [
        _cosine(vectors[a], vectors[b])
        for i, a in enumerate(foreign_trained) for b in foreign_trained[i + 1:]
    ]
    row["foreign_pair_cosine"] = float(np.mean(pairs)) if pairs else None
    row["null_pair_cosine"] = _cosine(
        (floor_draws[1] - floor_draws[0]).flatten().float().cpu().numpy(),
        (floor_draws[2] - floor_draws[0]).flatten().float().cpu().numpy(),
    ) if n_seeds >= 3 else None
    return row, vectors


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def _tick(slug: str, trained: set[str], homes: dict[str, str]) -> str:
    if slug == NONE:
        return "none"
    tag = "trained" if slug in trained else "unseen"
    if slug == homes[REBOT]:
        tag += ", ReBot home"
    return f"{LABELS[slug]}\n({tag})"


def _render(rows: list[dict], summary: dict, trained: set[str], homes: dict[str, str], output_path: str) -> None:
    by_domain = {d: [r for r in rows if r["domain"] == d] for d in (REBOT, DIVERSE)}
    fig = plt.figure(figsize=(19, 19))
    grid = fig.add_gridspec(3, 2, wspace=0.22, hspace=0.62, left=0.085, right=0.985, top=0.93, bottom=0.11)
    ax_sep, ax_robot, ax_dir, ax_tempo, ax_grid_rebot, ax_grid_diverse = (
        fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)
    )

    def paired_boxes(ax, key_suffix: str, ylabel: str, baseline: float) -> None:
        width, positions, ticks = 0.36, [], []
        for x, name in enumerate(CONDITIONS):
            for k, domain in enumerate((REBOT, DIVERSE)):
                values = _column(by_domain[domain], f"{name}_{key_suffix}")
                if values.size:
                    box = ax.boxplot([values], positions=[x + (k - 0.5) * width * 1.1], widths=width,
                                     patch_artist=True, showfliers=False)
                    box["boxes"][0].set_facecolor(_DOMAIN_COLOR[domain])
                    box["boxes"][0].set_alpha(0.55)
                    for median in box["medians"]:
                        median.set_color("black")
            positions.append(x)
            ticks.append(_tick(name, trained, homes))
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(positions, ticks, fontsize=8)
        ax.set_ylabel(ylabel)
        for domain in (REBOT, DIVERSE):
            ax.plot([], [], color=_DOMAIN_COLOR[domain], linewidth=8, alpha=0.55,
                    label=f"{domain} frames (n={len(by_domain[domain])})")
        ax.legend(fontsize=8, loc="upper right")

    paired_boxes(ax_sep, "sep", r"$S(L)$ — swap displacement / seed floor", 1.0)
    ax_sep.set_title(
        "Does the name move the chunk more than noise does?\n"
        f"ReBot frames: trained foreign {summary['foreign_trained_on_rebot_sep_median']:.2f}x, "
        f"Rebot B601 {summary['rebot_label_on_rebot_sep_median']:.2f}x, "
        f"unseen {summary['foreign_unseen_on_rebot_sep_median']:.2f}x\n"
        f"diverse frames: none {summary['none_on_diverse_sep_median']:.2f}x, "
        f"Rebot B601 {summary['rebot_label_on_diverse_sep_median']:.2f}x, "
        f"trained foreign {summary['foreign_trained_on_diverse_sep_median']:.2f}x",
        fontsize=10,
    )
    _caption(ax_sep, [
        "One frame gives one point per box; each label's displacement from the frame's HOME prompt (ReBot: "
        f"{'no clause' if homes[REBOT] == NONE else LABELS[homes[REBOT]]}; diverse: its own label),",
        "over the home prompt reseeded. A frame's own label is its home, so it has no point in its box. "
        "Dashed line = the floor: at 1 the name did nothing the sampler was not already doing.",
    ], y=-0.22)

    # (b) diverse frames by their own robot
    groups = summary["diverse_by_robot"]
    buckets = [("none", "none_sep"), ("Rebot B601", f"{_slug(REBOT_NAME)}_sep"),
               ("other trained", "foreign_trained_sep"), ("unseen", "foreign_unseen_sep")]
    width = 0.8 / len(buckets)
    for k, (label, key) in enumerate(buckets):
        xs = np.arange(len(groups)) + (k - (len(buckets) - 1) / 2) * width
        ax_robot.bar(xs, [groups[g][key]["median"] for g in groups], width * 0.95,
                     yerr=[groups[g][key]["se"] for g in groups], capsize=2, label=label,
                     color=plt.get_cmap("tab20c")(k / 4 * 0.75 + 0.05))
    ax_robot.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax_robot.set_xticks(np.arange(len(groups)), [f"{g}\n(n={groups[g]['n']})" for g in groups], fontsize=9)
    ax_robot.set_ylabel(r"median $S$ over the frames of that robot")
    ax_robot.set_title("Diverse frames by their own robot — what each swap does", fontsize=10)
    ax_robot.legend(fontsize=8)
    _caption(ax_robot, [
        "Per own-robot group of the holdout: clause removed, Rebot B601 substituted, the other trained names",
        "(mean per frame), the unseen names; medians with a bootstrap SE. 'Rebot B601' above 'other trained'",
        "says the ReBot label is a specific move rather than any different name.",
    ], y=-0.22)

    # (c) direction
    designated = {REBOT: summary["designated_swap"][REBOT], DIVERSE: summary["designated_swap"][DIVERSE]}
    boxes, labels_c, colors_c = [], [], []
    for domain in (REBOT, DIVERSE):
        for key, tag in (("shared_cosine", f"{LABELS[designated[domain]]}\nvs other frames"),
                         ("noise_shared_cosine", "reseed\nvs other frames"),
                         ("foreign_pair_cosine", "foreign names\npairwise"),
                         ("null_pair_cosine", "reseeds\npairwise (null)")):
            values = _column(by_domain[domain], key)
            if values.size:
                boxes.append(values)
                labels_c.append(f"{domain}\n{tag}")
                colors_c.append(_DOMAIN_COLOR[domain])
    if boxes:
        bp = ax_dir.boxplot(boxes, tick_labels=labels_c, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors_c, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")
    ax_dir.axhline(0.0, color="grey", linestyle="--", linewidth=1)
    ax_dir.set_ylabel("cosine")
    ax_dir.tick_params(axis="x", labelsize=7.5)
    ax_dir.set_title("One direction, or one per frame and per name?", fontsize=10)
    _caption(ax_dir, [
        "Leave-one-out: the designated swap's displacement against the mean displacement of every other frame in "
        "that domain, next to the same statistic on a reseed (its null).",
        "Pairwise: within a frame, the mean cosine among the trained foreign names' displacements, next to two "
        "reseeds measured from a shared third draw (shared-endpoint null).",
    ], y=-0.36)

    paired_boxes(ax_tempo, "tempo", "per-step travel, swap / home", 1.0)
    ax_tempo.ticklabel_format(axis="y", useOffset=False)
    ax_tempo.set_title("Does the name change the tempo?", fontsize=10)
    _caption(ax_tempo, [
        "RMS per-step travel of the swapped chunk over the home chunk. Robots differ in speed,",
        "so a read name can show here even when the posture change is small.",
    ], y=-0.22)

    # (e, f) the name x mode grid, median S per cell; the home cell(s) are blank
    mode_ticks = {NONE: "no clause", JOINT: "joint", END_EFFECTOR: "end-effector"}
    for ax, domain in ((ax_grid_rebot, REBOT), (ax_grid_diverse, DIVERSE)):
        cells = np.full((len(CONDITIONS), len(MODES)), np.nan)
        for i, name in enumerate(CONDITIONS):
            for j, mode in enumerate(MODES):
                cells[i, j] = _median(by_domain[domain], f"{_cell(name, mode)}_sep")
        image = ax.imshow(np.log10(cells), cmap="viridis", aspect="auto")
        for i in range(len(CONDITIONS)):
            for j in range(len(MODES)):
                home = np.isnan(cells[i, j])
                ax.text(j, i, "home" if home else f"{cells[i, j]:.2f}",
                        ha="center", va="center", fontsize=8, color="black" if home else "white")
        ax.set_xticks(range(len(MODES)), [mode_ticks[m] for m in MODES], fontsize=9)
        ax.set_yticks(range(len(CONDITIONS)), [_tick(n, trained, homes).replace("\n", " ") for n in CONDITIONS],
                      fontsize=8)
        ax.set_xlabel("control-mode clause")
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=r"$\log_{10}$ median $S$")
        ax.set_title(
            f"{domain} frames (n={len(by_domain[domain])}): name x control mode — "
            f"mode swapped {summary[f'mode_swapped_on_{domain}_sep_median']:.2f}x, "
            f"clause removed {summary[f'mode_none_on_{domain}_sep_median']:.2f}x",
            fontsize=10,
        )
    _caption(ax_grid_rebot, [
        f"Every cell is one prompt on the same frame; home = {'no clause' if homes[REBOT] == NONE else LABELS[homes[REBOT]]} "
        f"+ {mode_ticks[summary['rebot_mode']]} (the ReBot training prompt). 'none / no clause' is the pre-clause legacy prompt.",
        "A name's row constant across modes = the two clauses act independently; the mode rewriting the row = they interact.",
    ], y=-0.2)
    _caption(ax_grid_diverse, [
        "Home = the row's own name + own layout's mode (joint everywhere but MolmoAct), so the diverse rows mix two homes;",
        "'diverse_by_mode' in the JSON splits them. Franka Panda@joint on a MolmoAct frame is the DROID/FMB training prompt.",
    ], y=-0.2)

    fig.suptitle(
        "Embodiment swap — ReBot frames under other robots' names, diverse frames under the ReBot name, "
        "both under the other control-mode clause",
        fontsize=13, y=0.98,
    )
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def _stat(rows: list[dict], key: str) -> dict:
    return {"median": _median(rows, key), "se": _median_se(rows, key), "n": int(_column(rows, key).size)}


def run(adapter, dataset, cfg, output_dir: str) -> dict | None:
    """``dataset`` is the held-out ReBot set; the diverse side is the holdout of ``cfg.diverse.root``."""
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[embodiment_swap] needs continuous flow actions — skipping.")
        return None
    makedirs(output_dir)
    p = cfg.probe_parameters
    n_seeds = max(int(getattr(p, "embodiment_swap_n_seeds", None) or p.n_seeds), 2)
    n_frames = int(getattr(p, "embodiment_swap_n_frames", None) or p.n_frames_per_episode)
    chunk_size = int(cfg.policy.chunk_size)

    rebot = _rebot_samples(dataset, cfg, n_frames)
    diverse_buffer, trained_names = _open_holdout(cfg)
    diverse = _diverse_samples(diverse_buffer, n_frames)
    homes = {REBOT: _rebot_home(cfg)}
    rebot_mode = _rebot_mode(cfg)
    if homes[REBOT] != NONE:
        trained_names.add(LABELS[homes[REBOT]])
    trained = {_slug(name) for name in trained_names}
    designated = {
        REBOT: _slug("Franka Panda") if _slug("Franka Panda") in trained else sorted(trained)[0],
        DIVERSE: _slug(REBOT_NAME),
    }
    forwards = len(GRID) + n_seeds - 1
    logging.info(
        f"[embodiment_swap] {len(rebot)} ReBot + {len(diverse)} diverse frames x {forwards} forwards; "
        f"trained names {sorted(trained_names)}; ReBot home = {homes[REBOT]} + {rebot_mode}"
    )

    adapter._set_probe_cuda_graph_enabled(False)  # prompt changes per row; keep eager
    rows: list[dict] = []
    geometry: list[dict] = []
    try:
        for i, sample in enumerate(rebot + diverse):
            if i % 25 == 0:
                logging.info(f"  [{i + 1}/{len(rebot) + len(diverse)}] {sample['episode']} @ {sample['frame']}")
            if sample["domain"] == REBOT:
                frame = probe_frame_inputs(dataset, cfg, sample["index"], chunk_size)
                inputs = {"obs": frame["obs"], "task": frame["task"], "subtask": frame["subtask"], "extra": None}
                home, home_mode = homes[REBOT], rebot_mode
            else:
                inputs = _diverse_inputs(diverse_buffer, cfg, sample)
                home, home_mode = _slug(sample["own"]), _layout_mode(inputs["extra"])
            row, vectors = _measure_frame(adapter, inputs, home, home_mode, trained, n_seeds)
            rows.append({**sample, **row})
            geometry.append(vectors)
    finally:
        adapter._restore_probe_cuda_graph_enabled()
    if not rows:
        logging.warning("[embodiment_swap] no frames produced measurements.")
        return None

    for domain in (REBOT, DIVERSE):
        idx = [i for i, r in enumerate(rows) if r["domain"] == domain]
        swap = [geometry[i][designated[domain]] for i in idx if designated[domain] in geometry[i]]
        noise = [geometry[i]["noise"] for i in idx if geometry[i]["noise"] is not None]
        for i, shared, null in zip(idx, _loo_alignment(swap), _loo_alignment(noise), strict=False):
            rows[i]["shared_cosine"] = shared
            rows[i]["noise_shared_cosine"] = null

    by_domain = {d: [r for r in rows if r["domain"] == d] for d in (REBOT, DIVERSE)}
    rebot_slug = _slug(REBOT_NAME)
    rebot_joint = _cell(rebot_slug, JOINT)
    summary: dict = {
        "n_rebot_frames": len(by_domain[REBOT]),
        "n_diverse_frames": len(by_domain[DIVERSE]),
        "n_seeds": n_seeds,
        "trained_labels": sorted(trained_names),
        "unseen_labels": [LABELS[s] for s in LABELS if s not in trained],
        "rebot_home": homes[REBOT],
        "rebot_mode": rebot_mode,
        "designated_swap": designated,
        "mode_swapped_on_rebot_sep_median": _median(by_domain[REBOT], "mode_swapped_sep"),
        "mode_none_on_rebot_sep_median": _median(by_domain[REBOT], "mode_none_sep"),
        "mode_swapped_on_diverse_sep_median": _median(by_domain[DIVERSE], "mode_swapped_sep"),
        "mode_none_on_diverse_sep_median": _median(by_domain[DIVERSE], "mode_none_sep"),
        "rebot_label_joint_on_diverse_sep_median": _median(by_domain[DIVERSE], f"{rebot_joint}_sep"),
        "mode_swapped_on_rebot_tempo_median": _median(by_domain[REBOT], "mode_swapped_tempo"),
        "mode_swapped_on_diverse_tempo_median": _median(by_domain[DIVERSE], "mode_swapped_tempo"),
        "rebot_label_on_rebot_sep_median": _median(by_domain[REBOT], f"{rebot_slug}_sep"),
        "foreign_trained_on_rebot_sep_median": _median(by_domain[REBOT], "foreign_trained_sep"),
        "foreign_unseen_on_rebot_sep_median": _median(by_domain[REBOT], "foreign_unseen_sep"),
        "none_on_diverse_sep_median": _median(by_domain[DIVERSE], f"{NONE}_sep"),
        "rebot_label_on_diverse_sep_median": _median(by_domain[DIVERSE], f"{rebot_slug}_sep"),
        "foreign_trained_on_diverse_sep_median": _median(by_domain[DIVERSE], "foreign_trained_sep"),
        "foreign_unseen_on_diverse_sep_median": _median(by_domain[DIVERSE], "foreign_unseen_sep"),
        "rebot_shared_cosine_median": _median(by_domain[REBOT], "shared_cosine"),
        "rebot_noise_shared_cosine_median": _median(by_domain[REBOT], "noise_shared_cosine"),
        "diverse_shared_cosine_median": _median(by_domain[DIVERSE], "shared_cosine"),
        "diverse_noise_shared_cosine_median": _median(by_domain[DIVERSE], "noise_shared_cosine"),
        "rebot_foreign_pair_cosine_median": _median(by_domain[REBOT], "foreign_pair_cosine"),
        "diverse_foreign_pair_cosine_median": _median(by_domain[DIVERSE], "foreign_pair_cosine"),
        "null_pair_cosine_median": _median(rows, "null_pair_cosine"),
        "rebot_seed_floor_mean": _median(by_domain[REBOT], "seed_floor_mean"),
        "diverse_seed_floor_mean": _median(by_domain[DIVERSE], "seed_floor_mean"),
        "per_label": {
            domain: {
                name: {
                    "sep_median": _median(by_domain[domain], f"{name}_sep"),
                    "rmse_mean": float(np.mean(_column(by_domain[domain], f"{name}_rmse")))
                    if _column(by_domain[domain], f"{name}_rmse").size else float("nan"),
                    "tempo_median": _median(by_domain[domain], f"{name}_tempo"),
                }
                for name in CONDITIONS
            }
            for domain in (REBOT, DIVERSE)
        },
        "per_cell": {
            domain: {
                _cell(name, mode): {
                    "sep_median": _median(by_domain[domain], f"{_cell(name, mode)}_sep"),
                    "tempo_median": _median(by_domain[domain], f"{_cell(name, mode)}_tempo"),
                }
                for name, mode in GRID
            }
            for domain in (REBOT, DIVERSE)
        },
        "diverse_by_robot": {
            own: {
                "n": len(group),
                "none_sep": _stat(group, f"{NONE}_sep"),
                f"{rebot_slug}_sep": _stat(group, f"{rebot_slug}_sep"),
                "foreign_trained_sep": _stat(group, "foreign_trained_sep"),
                "foreign_unseen_sep": _stat(group, "foreign_unseen_sep"),
                "mode_swapped_sep": _stat(group, "mode_swapped_sep"),
                "mode_none_sep": _stat(group, "mode_none_sep"),
            }
            for own in sorted({r["own"] for r in by_domain[DIVERSE]})
            for group in [[r for r in by_domain[DIVERSE] if r["own"] == own]]
        },
        # The diverse holdout mixes two homes (joint layouts, the MolmoAct end-effector layout);
        # per home mode is where "the clause, not the image, decides the space" is read.
        "diverse_by_mode": {
            mode: {
                "n": len(group),
                "swapped_to": _swapped(mode),
                "mode_swapped_sep": _stat(group, "mode_swapped_sep"),
                "mode_none_sep": _stat(group, "mode_none_sep"),
                "mode_swapped_tempo": _stat(group, "mode_swapped_tempo"),
                f"{rebot_joint}_sep": _stat(group, f"{rebot_joint}_sep"),
                "none_sep": _stat(group, f"{NONE}_sep"),
            }
            for mode in sorted({r["home_mode"] for r in by_domain[DIVERSE]})
            for group in [[r for r in by_domain[DIVERSE] if r["home_mode"] == mode]]
        },
        "reading": (
            "S ~ 1 on every label => the embodiment clause is not read on that side. Trained foreign "
            "names >> unseen names on ReBot frames => the name is read as a robot, not as new tokens. "
            "On diverse frames, Rebot B601 above the other trained names => the ReBot label is a specific "
            "move; none ~ 1 => dropping the clause costs nothing. shared_cosine at its null => the swap "
            "is a per-frame displacement, not one global offset. mode_swapped >> 1 => the control-mode "
            "clause is read (on the MolmoAct home the joint clause is the DROID/FMB prompt on a pose-space "
            "image); mode_none ~ 1 => the clause carries nothing the image and the name did not."
        ),
    }
    summary["data"] = {
        "rebot": {
            "n_frames": len(by_domain[REBOT]),
            "n_episodes": len({r["episode"] for r in by_domain[REBOT]}),
            "root": str(getattr(dataset, "root", "")),
        },
        "diverse": {
            "n_frames": len(by_domain[DIVERSE]),
            "n_episodes": len({r["episode"] for r in by_domain[DIVERSE]}),
            "episodes": sorted({r["episode"] for r in by_domain[DIVERSE]}),
            "by_robot": {own: g["n"] for own, g in summary["diverse_by_robot"].items()},
            "root": str(cfg.diverse.root),
        },
        "frames_per_episode": n_frames,
        "forwards": forwards * len(rows),
        "details": [
            ["Per frame", f"{forwards} forwards — (``none`` + one row per name in EMBODIMENT_NAMES) x "
                          f"(no clause, joint, end_effector control-mode clause), plus {n_seeds - 1} "
                          "reseed(s) of the home prompt for the floor; deployment metadata clause "
                          "(quality 5, no mistake, speed 5) on every row"],
            ["Home prompt", f"ReBot frames: ``{homes[REBOT]}`` + ``{rebot_mode}``" +
             (" — ReBot training rows carry no embodiment clause (RoleAlignedBuffer stamps action_layout_id "
              "only; policy.embodiment unset), so ``Rebot B601`` is an unseen prompt on ReBot images; the "
              "control-mode clause is rendered from that layout"
              if homes[REBOT] == NONE else " (policy.embodiment; control-mode clause from diverse.rebot_layout)") +
             "; diverse frames: the row's own label + its layout's control mode, as training rendered them"],
            ["Control-mode axis", "``mode_swapped`` = the home name under the other space (joint <-> "
                                  "end_effector), ``mode_none`` = the home name with no control-mode clause; "
                                  "``{name}@{mode}`` columns are the full grid, ``{name}`` columns the name "
                                  "axis at the home mode (pre-clause report shape)"],
            ["Trained / unseen names", f"trained {sorted(trained_names)} (the training selection's rows); "
                                       f"unseen {summary['unseen_labels']}"],
            ["Diverse frames", "the episodes of <root>/holdout_episodes.json (holdout_actor_selection), "
                               "never trained on, decoded from video, every episode with anchors evenly "
                               "spaced; identity columns (action_layout_id, embodiment_index, camera and "
                               "depth presence) ride into the pack step so normalization is the row's own"],
            ["Readout", "normalized flow chunks; the unnormalized output is not used, so the diverse rows' "
                        "unnormalization path is not exercised"],
        ],
    }

    with open(os.path.join(output_dir, "embodiment_swap.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)
    _render(rows, summary, trained, homes, os.path.join(output_dir, "embodiment_swap.png"))

    write_index(
        output_dir,
        sys.modules[__name__],
        title="Embodiment Swap",
        group="Steering",
        claim="Does the embodiment clause reach the actions — on ReBot frames under other names, and on diverse frames under the ReBot name?",
        summary=summary,
        see_also=["metadata_steering", "domain_representations", "subtask_sweep"],
        metrics=[
            Metric("foreign_trained_on_rebot_sep_median", "ReBot frames: trained foreign name / floor",
                   good="none", fmt=2, baseline=1.0, primary=True, trend=True,
                   note="Median over ReBot frames of the mean separation under the trained diverse names. "
                        "1 = the name moves the chunk no further than reseeding."),
            Metric("rebot_label_on_rebot_sep_median", "ReBot frames: 'Rebot B601' / floor",
                   good="none", fmt=2, baseline=1.0, trend=True,
                   note="The never-trained correct label on ReBot images."),
            Metric("foreign_unseen_on_rebot_sep_median", "ReBot frames: unseen name / floor",
                   good="none", fmt=2, baseline=1.0,
                   note="Names no training row carries: the 'any new sentence' control."),
            Metric("rebot_label_on_diverse_sep_median", "diverse frames: 'Rebot B601' / floor",
                   good="none", fmt=2, baseline=summary["foreign_trained_on_diverse_sep_median"], primary=True, trend=True,
                   note="The ReBot label on another robot's frames, read against the other trained names on the same frames."),
            Metric("none_on_diverse_sep_median", "diverse frames: clause removed / floor",
                   good="none", fmt=2, baseline=1.0, trend=True,
                   note="What removing the clause the model always saw on this frame's robot does."),
            Metric("diverse_shared_cosine_median", "diverse frames: 'Rebot B601' shared direction",
                   good="none", fmt=2, baseline=summary["diverse_noise_shared_cosine_median"],
                   note="Leave-one-out cosine of the Rebot B601 displacement against the other frames'; at the reseed null it is computed per frame, near +1 it is one global offset."),
            Metric("mode_swapped_on_rebot_sep_median", "ReBot frames: control mode swapped / floor",
                   good="none", fmt=2, baseline=1.0, trend=True,
                   note=f"The home name under the other space ({_swapped(rebot_mode)} on a {rebot_mode} robot). "
                        "1 = the control-mode clause is not read on ReBot frames."),
            Metric("mode_none_on_rebot_sep_median", "ReBot frames: control-mode clause removed / floor",
                   good="none", fmt=2, baseline=1.0,
                   note="The pre-clause legacy prompt on a ReBot frame."),
            Metric("mode_swapped_on_diverse_sep_median", "diverse frames: control mode swapped / floor",
                   good="none", fmt=2, baseline=1.0, trend=True,
                   note="Own name under the other space; diverse_by_mode splits the joint and end-effector homes."),
        ],
        panels=[Panel("embodiment_swap.png",
                      "Separation by name, the diverse breakdown by robot, direction geometry, tempo, and the "
                      "name x control-mode grid per domain.",
                      how="Every point is a within-frame contrast of two prompts on the same frame under the same flow noise, "
                          "in units of that frame's own seed floor; the two domains share the scale because the ratio "
                          "needs no cross-domain normalization.",
                      primary=True)],
        extra={"provenance": summary["data"]},
    )
    logging.info(
        f"[embodiment_swap] rebot n={len(by_domain[REBOT])}: foreign trained "
        f"{summary['foreign_trained_on_rebot_sep_median']:.2f}x, Rebot B601 "
        f"{summary['rebot_label_on_rebot_sep_median']:.2f}x, unseen {summary['foreign_unseen_on_rebot_sep_median']:.2f}x  |  "
        f"diverse n={len(by_domain[DIVERSE])}: none {summary['none_on_diverse_sep_median']:.2f}x, Rebot B601 "
        f"{summary['rebot_label_on_diverse_sep_median']:.2f}x, foreign trained "
        f"{summary['foreign_trained_on_diverse_sep_median']:.2f}x  |  shared cos rebot "
        f"{summary['rebot_shared_cosine_median']:+.2f} (null {summary['rebot_noise_shared_cosine_median']:+.2f}), "
        f"diverse {summary['diverse_shared_cosine_median']:+.2f} (null {summary['diverse_noise_shared_cosine_median']:+.2f})"
        f"  |  control mode swapped: rebot {summary['mode_swapped_on_rebot_sep_median']:.2f}x, "
        f"diverse {summary['mode_swapped_on_diverse_sep_median']:.2f}x; removed: rebot "
        f"{summary['mode_none_on_rebot_sep_median']:.2f}x, diverse {summary['mode_none_on_diverse_sep_median']:.2f}x"
    )
    return summary


@parser.wrap()
def cli(cfg: ProbeEmbodimentSwapConfig):
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
    step = 0
    for part in str(getattr(cfg.policy, "pretrained_path", "") or "").split(os.sep):
        if part.isdigit():
            step = int(part)
    run(adapter, dataset, cfg, os.path.join(
        cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", "embodiment_swap"))


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
