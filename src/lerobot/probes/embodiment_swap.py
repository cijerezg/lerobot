r"""Bounded embodiment and control-mode sensitivity on held-out frames.

The report shows robot-name sensitivity only, with the original control mode and action
layout held fixed. Tempo, cosine and control-mode comparisons are omitted.

Home prompts use the same dataset identity and action-layout control mode as training,
including Rebot B601 on ReBot frames. Each frame evaluates nine names (omitted plus
all registry names) at the home mode, and the home name under the other two modes.
All eleven variants share flow noise within each of three seeds. This tests separate
name and mode interventions; it does not estimate their interaction.

Separation S is the mean paired action displacement divided by the mean distance
between reseeded home chunks. S=0 means no paired change; S=1 matches reseeding
magnitude. Neither sensitivity, direction cosine, nor tempo establishes useful transfer.
Names absent from this fine-tuning mixture may still have appeared in pretraining.

Already-decoded native actions also give demonstration RMSE and gripper MAE changes
(swap minus home; negative is closer to the recording). These are imitation diagnostics
under fixed deployment metadata, not rollout-success measurements. Native padding is
excluded. Source-specific results retain the original action units.

At most 96 frames are sampled round-robin over held-out episodes, with up to six
candidates per episode and phase coverage on the diverse side. YAM episodes are omitted
because of limited coverage; YAM remains a counterfactual name. Cost is at most
96 x 11 x 3 = 3,168 decoded chunks, in three batched calls per frame.
The holdout guarantee applies to new runs trained with the accompanying ledger.

Standalone::

    uv run python -m lerobot.probes.embodiment_swap --config config_rl.yaml \
        --policy.pretrained_path outputs/<run>/checkpoints/<step>/pretrained_model
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.embodiment import EMBODIMENT_NAMES, canonical_embodiment, embodiment_name
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
    dataset_identity_columns,
    build_episode_index,
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


def _rebot_home(dataset, cfg) -> str:
    name = embodiment_name(dataset_identity_columns(dataset, cfg)["embodiment_index"])
    return NONE if name is None else _slug(name)


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
    ep_index = build_episode_index(dataset)
    return [
        {"domain": REBOT, "source": REBOT, "own": REBOT_NAME, "episode": f"rebot/{ep}",
         "frame": int(fr), "index": int(gidx)}
        for ep, fr, gidx in samples
        if gidx + int(cfg.policy.chunk_size) <= ep_index[ep][-1] + 1
    ]


def _diverse_samples(buffer, n_frames: int) -> list[dict]:
    """Every holdout episode, ``n_frames`` anchors evenly spaced over each."""
    by_episode: dict[str, list[int]] = {}
    for row_index, row in enumerate(buffer.rows):
        if row["source"] == "yam":
            continue
        by_episode.setdefault(str(row["episode_id"]), []).append(row_index)
    samples = []
    for episode_id in sorted(by_episode):
        rows = sorted(by_episode[episode_id], key=lambda i: float(buffer.rows[i]["anchor_s"]))
        phases = defaultdict(list)
        for i in rows:
            phases[str(buffer.rows[i]["subtask"]).split(" ", 1)[0]].append(i)
        ordered = []
        for phase in sorted(phases):
            bucket = phases[phase]
            picks = np.unique(np.linspace(0, len(bucket) - 1, min(n_frames, len(bucket)) + 2, dtype=int)[1:-1])
            ordered.append([bucket[int(k)] for k in picks] or [bucket[len(bucket) // 2]])
        selected = [bucket[k] for k in range(n_frames) for bucket in ordered if k < len(bucket)][:n_frames]
        for index in selected:
            row = buffer.rows[index]
            samples.append({
                "domain": DIVERSE, "source": str(row["source"]),
                "own": canonical_embodiment(row["embodiment"]),
                "episode": f"{row['source']}/{episode_id}", "frame": float(row["anchor_s"]),
                "index": int(index),
            })
    return samples


def _bounded_frames(samples: list[dict], cap: int, seed: int) -> list[dict]:
    """Round-robin episodes under a global frame budget; deterministic across checkpoints."""
    if cap < 1:
        raise ValueError("embodiment_swap_max_frames must be positive")
    groups = defaultdict(list)
    for s in samples:
        groups[s["episode"]].append(s)
    rng = np.random.RandomState(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for bucket in groups.values():
        rng.shuffle(bucket)
    return [groups[key][i] for i in range(max(map(len, groups.values()), default=0))
            for key in keys if i < len(groups[key])][:cap]


def _frame_grid(home: str, mode: str) -> list[tuple[str, str]]:
    """All names at the true mode, plus mode interventions at the true name (11 cells)."""
    return [(name, mode) for name in CONDITIONS] + [(home, other) for other in MODES if other != mode]


# ──────────────────────────────────────────────────────────────────────────────
# Measurement
# ──────────────────────────────────────────────────────────────────────────────

def _measure_frame(
    adapter, inputs: dict, home: str, home_mode: str, trained_slugs: set[str], n_seeds: int
) -> tuple[dict, dict]:
    """Each name/mode intervention paired with its home under every flow seed."""
    grid = _frame_grid(home, home_mode)
    n = len(grid)
    normalized, raw = [], []
    for seed in range(n_seeds):
        unnorm, chunks = adapter.predict_action_chunk_batch(
            inputs["obs"], inputs["task"], [inputs["subtask"]] * n,
            metadatas=[dict(DEPLOYMENT_METADATA)] * n,
            noise=adapter.flow_noise_like(n, seed),
            embodiments=[_name_text(name) for name, _ in grid],
            control_modes=[_mode_text(mode) for _, mode in grid],
            extra_complementary=inputs["extra"], inference_action_mode="continuous",
        )
        normalized.append(chunks)
        if unnorm is not None:
            raw.append(unnorm)
    acts = torch.stack(normalized)  # [seed, condition, time, dim]
    home_index = grid.index((home, home_mode))
    base = acts[:, home_index]
    floor_draws = list(base.unbind())
    floor_mean, floor_max = _pairwise_rmse(floor_draws)
    base_steps = [max(_step_rms(x), 1e-12) for x in base]
    row = {"home": home, "home_mode": home_mode, "seed_floor_mean": floor_mean,
           "seed_floor_max": floor_max, "paired_seeds": n_seeds, "variants": n}
    vectors = {}
    gt = inputs.get("gt_actions")
    raw_actions = torch.stack(raw) if len(raw) == n_seeds else None
    if gt is not None and raw_actions is not None:
        gt = torch.as_tensor(gt).detach().float().cpu()
        if gt.ndim == 3:
            gt = gt[0]
        width = int(inputs.get("native_width", gt.shape[-1]))
        horizon = min(gt.shape[0], raw_actions.shape[-2])
        raw_actions = raw_actions[:, :, :horizon, :width]
        gt = gt[:horizon, :width]
        errors = (raw_actions - gt).square().mean(dim=(-2, -1)).sqrt()
        gripper_errors = (raw_actions[..., -1] - gt[:, -1]).abs().mean(dim=-1)
        row["home_demo_rmse"] = float(errors[:, home_index].mean())
        row["home_gripper_mae"] = float(gripper_errors[:, home_index].mean())
    for i, (name, mode) in enumerate(grid):
        if i == home_index:
            continue
        cell = _cell(name, mode)
        delta = acts[:, i] - base
        row[f"{cell}_rmse"] = float(delta.square().mean(dim=(-2, -1)).sqrt().mean())
        row[f"{cell}_sep"] = row[f"{cell}_rmse"] / max(floor_mean, 1e-9)
        row[f"{cell}_tempo"] = float(np.mean([_step_rms(acts[s, i]) / base_steps[s] for s in range(n_seeds)]))
        vectors[cell] = delta.mean(dim=0).flatten().float().cpu().numpy()
        if gt is not None and raw_actions is not None:
            row[f"{cell}_demo_rmse_delta"] = float((errors[:, i] - errors[:, home_index]).mean())
            row[f"{cell}_gripper_mae_delta"] = float((gripper_errors[:, i] - gripper_errors[:, home_index]).mean())
            row[f"{cell}_gripper_shift"] = float((raw_actions[:, i, :, -1] - raw_actions[:, home_index, :, -1]).mean())
    aliases = {name: _cell(name, home_mode) for name in CONDITIONS if name != home}
    aliases["mode_none"] = _cell(home, NONE)
    aliases["mode_swapped"] = _cell(home, _swapped(home_mode))
    for alias, cell in aliases.items():
        for suffix in ("rmse", "sep", "tempo", "demo_rmse_delta", "gripper_mae_delta", "gripper_shift"):
            if f"{cell}_{suffix}" in row:
                row[f"{alias}_{suffix}"] = row[f"{cell}_{suffix}"]
        vectors[alias] = vectors[cell]
    vectors["noise"] = (base[1] - base[0]).flatten().float().cpu().numpy()

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
    home_note = "\nReBot home" if slug == homes[REBOT] else ""
    return f"{LABELS[slug]}\n({tag}){home_note}"


def _render(rows: list[dict], summary: dict, trained: set[str], homes: dict[str, str], output_path: str) -> None:
    by_domain = {d: [r for r in rows if r["domain"] == d] for d in (REBOT, DIVERSE)}
    fig = plt.figure(figsize=(19, 7))
    grid = fig.add_gridspec(1, 2, wspace=0.25, left=0.065, right=0.985, top=0.79, bottom=0.32)
    ax_sep, ax_robot = (fig.add_subplot(grid[0, j]) for j in range(2))

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
        f"name removed {summary['none_on_rebot_sep_median']:.2f}x, "
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
        "Dashed line = reseeding magnitude (S=1); S=0 means no paired action change.",
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
        "shows a larger displacement for that name; it does not establish ReBot behavior transfer.",
    ], y=-0.22)

    fig.suptitle(
        "Embodiment name swaps — original control mode and action layout held fixed",
        fontsize=13, y=0.97,
    )
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

def _stat(rows: list[dict], key: str) -> dict:
    return {"median": _median(rows, key), "se": _median_se(rows, key), "n": int(_column(rows, key).size)}


def write_report(rows: list[dict], summary: dict, output_dir: str) -> None:
    """Render saved measurements without loading the policy or repeating inference."""
    trained = {_slug(name) for name in summary["trained_labels"]}
    homes = {REBOT: summary["rebot_home"]}
    # Old measurements retain their raw mode diagnostics; keep the report's explanation current.
    details = summary["data"].get("details", [])
    summary["data"]["details"] = [
        ["Displayed comparisons", "Robot-name changes keep the recorded frame, original control-mode clause "
         "and action layout fixed. Changing the text does not convert coordinates."]
        if item[0] == "Control-mode axis" else item
        for item in details
    ]
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
            Metric("none_on_rebot_sep_median", "ReBot frames: name removed / floor",
                   good="none", fmt=2, baseline=1.0, trend=True,
                   note="Removing the embodiment clause supplied by the training dataset."),
            Metric("foreign_unseen_on_rebot_sep_median", "ReBot frames: unseen name / floor",
                   good="none", fmt=2, baseline=1.0,
                   note="Names absent from this fine-tuning selection; they may be familiar from pretraining."),
            Metric("rebot_label_on_diverse_sep_median", "diverse frames: 'Rebot B601' / floor",
                   good="none", fmt=2, baseline=summary["foreign_trained_on_diverse_sep_median"], primary=True, trend=True,
                   note="The ReBot label on another robot's frames, read against the other trained names on the same frames."),
            Metric("none_on_diverse_sep_median", "diverse frames: clause removed / floor",
                   good="none", fmt=2, baseline=1.0, trend=True,
                   note="What removing the clause the model always saw on this frame's robot does."),
        ] + [
            Metric(f"action_quality_by_source.{source}.{alias}.{metric}",
                   f"{source}: {label}, {title} change vs home",
                   good="none", fmt=4,
                   note="Swap minus home, averaged over paired seeds then median over frames. "
                        "Negative is closer to the recorded demonstration; native units, not rollout success.")
            for source in summary["action_quality_by_source"]
            for alias, label in ((NONE, "name removed"),)
            for metric, title in (("demo_rmse_delta", "demonstration RMSE"), ("gripper_mae_delta", "gripper MAE"))
        ],
        panels=[Panel("embodiment_swap.png",
                      "Action sensitivity to robot-name changes, with the original control mode and action layout held fixed; "
                      "overall and grouped by the recorded robot.",
                      how="Every point is a within-frame contrast of two prompts on the same frame under the same flow noise, "
                          "divided by that frame's reseeding variation. S=1 matches reseeding; S=0 means no change. "
                          "This measures prompt sensitivity, not transfer success or joint-versus-EE motion equivalence.",
                      primary=True)],
        extra={"provenance": summary["data"]},
    )

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
    homes = {REBOT: _rebot_home(dataset, cfg)}
    rebot_mode = _rebot_mode(cfg)
    if homes[REBOT] != NONE:
        trained_names.add(LABELS[homes[REBOT]])
    trained = {_slug(name) for name in trained_names}
    designated = {
        REBOT: _slug("Franka Panda") if _slug("Franka Panda") in trained else sorted(trained)[0],
        DIVERSE: _slug(REBOT_NAME),
    }
    samples = _bounded_frames(rebot + diverse, int(p.embodiment_swap_max_frames), int(p.random_seed))
    forwards = (len(CONDITIONS) + len(MODES) - 1) * n_seeds
    logging.info(
        f"[embodiment_swap] {len(samples)} capped frames x {forwards} decoded chunks; "
        f"trained names {sorted(trained_names)}; ReBot home = {homes[REBOT]} + {rebot_mode}"
    )

    adapter._set_probe_cuda_graph_enabled(False)  # prompt changes per row; keep eager
    rows: list[dict] = []
    geometry: list[dict] = []
    try:
        for i, sample in enumerate(samples):
            if i % 25 == 0:
                logging.info(f"  [{i + 1}/{len(samples)}] {sample['episode']} @ {sample['frame']}")
            if sample["domain"] == REBOT:
                frame = probe_frame_inputs(dataset, cfg, sample["index"], chunk_size, with_gripper_event_targets=False)
                inputs = {"obs": frame["obs"], "task": frame["task"], "subtask": frame["subtask"], "extra": None,
                          "gt_actions": frame["gt_actions"]}
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
        "protocol": "bounded_v1_paired_seeds",
        "max_frames": int(p.embodiment_swap_max_frames),
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
        "none_on_rebot_sep_median": _median(by_domain[REBOT], f"{NONE}_sep"),
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
                    "demo_rmse_delta_median": _median(by_domain[domain], f"{name}_demo_rmse_delta"),
                    "gripper_mae_delta_median": _median(by_domain[domain], f"{name}_gripper_mae_delta"),
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
        "action_quality_by_source": {
            source: {alias: {metric: _median(group, f"{alias}_{metric}")
                            for metric in ("demo_rmse_delta", "gripper_mae_delta", "gripper_shift")}
                     for alias in (NONE, "mode_none", "mode_swapped", *LABELS)}
            for source in sorted({r["source"] for r in rows})
            for group in [[r for r in rows if r["source"] == source]]
        },
        "reading": (
            "S near 0 means little action change under paired noise; S near 1 means a change "
            "as large as reseeding, not an ignored clause. Larger S measures sensitivity, not "
            "action quality or successful transfer. Trained/unseen refers only to this fine-tuning "
            "selection. ReBot home uses the dataset's training identity; none removes its name. "
            "The report shows name interventions at the original control mode only. Legacy mode interventions "
            "keep the source action layout fixed and do not compare joint and end-effector motions."
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
            ["Per frame", f"{forwards} decoded chunks: all names at the home control mode plus "
                          f"the home name under the other modes, repeated for {n_seeds} paired seeds. "
                          "Deployment metadata (quality 5, no mistake, speed 5) stays fixed."],
            ["Action quality", "Native-unit demonstration RMSE and gripper MAE changes are swap minus home; "
                               "negative is closer to the recording. Reported per source in native units. "
                               "These are imitation diagnostics under deployment metadata, not rollout success."],
            ["Home prompt", f"ReBot frames: ``{homes[REBOT]}`` + ``{rebot_mode}`` "
             "(identity resolved by the training dataset label loader); diverse frames: the row's "
             "own label + its layout's control mode. Removing the name is an intervention."],
            ["Displayed comparisons", "Robot-name changes keep the recorded frame, original control-mode clause "
                                       "and action layout fixed. Legacy mode interventions remain in the raw data "
                                       "but are omitted from the report: changing the text does not convert coordinates."],
            ["Trained / unseen names", f"trained {sorted(trained_names)} (the training selection's rows); "
                                       f"unseen {summary['unseen_labels']}"],
            ["Diverse frames", "the episodes of <root>/holdout_episodes.json (holdout_actor_selection), "
                               "excluded from new training runs, decoded from video, with episode-balanced "
                               "sampling and phase coverage; identity columns (action_layout_id, embodiment_index, camera and "
                               "depth presence) ride into the pack step so normalization is the row's own"],
            ["Readout", "Normalized chunks measure sensitivity. Native actions measure demonstration "
                        "RMSE and gripper MAE, with padded dimensions excluded."],
        ],
    }

    with open(os.path.join(output_dir, "embodiment_swap.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": rows}, f, indent=2)
    write_report(rows, summary, output_dir)

    logging.info(
        f"[embodiment_swap] rebot n={len(by_domain[REBOT])}: foreign trained "
        f"{summary['foreign_trained_on_rebot_sep_median']:.2f}x, name removed "
        f"{summary['none_on_rebot_sep_median']:.2f}x, unseen {summary['foreign_unseen_on_rebot_sep_median']:.2f}x  |  "
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
