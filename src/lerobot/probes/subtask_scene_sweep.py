r"""Does the subtask clause move the chunk when every label it is offered is plausible?

``subtask_sweep`` sweeps the whole vocabulary through the prompt at evenly sampled frames,
so most of what it contrasts is nonsense for the scene — ``release the red shirt in the
bin`` while the arm is empty over a table of socks — and its ranking came out ordered by
VERB (grasp < release < move < home), never by object. This probe removes both confounds:
the frames are empty-handed (a ``grasp`` reach before its close, or ``return to home``) and
each frame is swept only with labels that make sense there: ``grasp <class>`` for every
class on the table with exactly ONE instance, plus ``return to home`` (always in, the user's
rule). Classes not on the table are swept too, as a separate control population on the
same frame. Frame list, scene inventory and label sets are hand-reviewed on a top-camera
contact sheet (``migration/subtask_scene_sweep_frames.py``) and consumed here as
``probe_parameters.subtask_scene_sweep_frames``; the probe never derives them.

Per frame, one batched forward over all labels under one flow draw (seed 0) and one over
$n_s$ reseeds of the true label. With $a_\ell$ the normalized chunk under label $\ell$,
$V_p$ the plausible set, $V_g \subseteq V_p$ its grasp labels, $V_a$ the absent labels and
$\bar d(V) = \text{mean}_{\ell \ne \ell' \in V} \lVert a_\ell - a_{\ell'} \rVert$ the mean
pairwise RMSE,

    $$S_p = \frac{\bar d(V_p)}{\bar d(\text{seeds})}, \qquad S_g = \frac{\bar d(V_g)}{\bar d(\text{seeds})}, \qquad S_a = \frac{\bar d(V_a)}{\bar d(\text{seeds})}$$

are the three separations. $S_g$ is the object-only contrast the old probe could not
isolate; $S_p$ adds the one verb contrast (home); $S_a$ is what the old probe measured.
Sets with fewer than two labels (a home window with an empty table) give no $S$.

The second readout references the chunk under the TRUE label rather than the demonstration:
for each group $G \in \{\text{present-other}, \text{home}, \text{absent}\}$,

    $$D_G = \frac{\text{mean}_{\ell \in G} \lVert a_\ell - a_{\ell^\star} \rVert}{\bar d(\text{seeds})}$$

is how far the label moves the chunk off the truthful prediction, in seed-floor units. A
model that reads the object shows $D_{\text{present-other}} > 1$; one that only reads the
verb shows $D_{\text{home}} \gg D_{\text{present-other}} \approx 1$; a scene-blind model
shows $D_{\text{absent}} \approx D_{\text{present-other}}$.

Third, the old GT-rank readout on the plausible set, normalized because $|V_p|$ varies:
$\tilde r = (r - 1) / (|V_p| - 1)$ with the uniform null at $0.5$.

Fourth, a task-space readout with no demonstration in it. Every present class $X$ has a
table position $p_X$ = the end-effector at the close event of its next grasp (objects do
not move until grasped). With $e(\cdot)$ the end-effector of the last chunk step and $e_0$
the current one, the approach of a chunk toward $X$ is
$\alpha_\ell(X) = \lVert e_0 - p_X \rVert - \lVert e(a_\ell) - p_X \rVert$ and

    $$\Delta(X) = \alpha_{\text{grasp } X}(X) - \alpha_{\ell^\star}(X), \qquad X \ne \ell^\star$$

is the extra approach toward $X$ that naming $X$ buys, in metres over the chunk. Its null is
the same difference between two reseeds of the true label. Positive $\Delta$ above the null
is the object being read as a target; the home contrast $\alpha_{\ell^\star}(X^\star) -
\alpha_{\text{home}}(X^\star)$ is reported alongside.

Cost is ``n_frames x (|V_p| + |V_a| + n_seeds)`` forwards in two batched calls per frame.
Registered probe: ``probe_parameters.enable_subtask_scene_sweep``.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.subtask_sweep import _pairwise_rmse
from lerobot.probes.utils import (
    REBOT_JOINT_NAMES,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    register_config_choices,
)
from lerobot.robots.rebot_b601_follower.kinematics import RebotKinematics
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

HOME = "return to home"
GROUPS = ("present_other", "home", "absent")


class SceneSweepProbeConfig(TrainRLServerPipelineConfig):
    pass


def _grasp(cls: str) -> str:
    return f"grasp {cls}"


def _median(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def _caption(ax, lines: list[str]) -> None:
    """subtask_sweep's caption, set lower to clear two-line tick labels."""
    ax.text(0.0, -0.30, "\n".join(lines), transform=ax.transAxes, fontsize=7.4,
            va="top", ha="left", color="#333333", linespacing=1.55)


def _object_positions(dataset, kin: RebotKinematics) -> dict[tuple[int, str], list[tuple[int, np.ndarray]]]:
    """(episode, class) -> [(close frame, end-effector position)] over the episode's grasps.

    subtask_windows.json is global-indexed; the events table carries per-episode frame
    indices. The position is FK of the recorded state at the close event, the one moment the
    tool is provably at the object.
    """
    root = str(dataset.root)
    windows = json.load(open(os.path.join(root, "meta", "subtask_windows.json")))["episodes"]
    events = pd.read_parquet(os.path.join(root, "meta", "depth_gripper_events.parquet"))
    closes = events[events.event_type == "close"]
    episodes = pd.read_parquet(os.path.join(root, "meta", "episodes", "chunk-000", "file-000.parquet"))
    ep_start = dict(zip(episodes.episode_index, episodes.dataset_from_index, strict=True))
    states = np.stack(dataset.hf_dataset.with_format(None)["observation.state"])

    positions: dict[tuple[int, str], list[tuple[int, np.ndarray]]] = {}
    for ep_key, segs in windows.items():
        ep, base = int(ep_key), int(ep_start[int(ep_key)])
        ep_closes = sorted(closes[closes.episode_index == ep].frame_index.tolist())
        for seg in segs:
            if not seg["subtask"].startswith("grasp "):
                continue
            start, stop = seg["from_index"] - base, seg["to_index"] - base
            inside = [c for c in ep_closes if start <= c < stop]
            if not inside:
                continue
            cls = seg["subtask"][len("grasp "):]
            pos = kin.ee_path(states[base + inside[0]][None, :])[0]
            positions.setdefault((ep, cls), []).append((inside[0], pos))
    return positions


def _next_position(positions, ep: int, cls: str, frame_idx: int) -> np.ndarray | None:
    later = [pos for close, pos in positions.get((ep, cls), []) if close > frame_idx]
    return later[0] if later else None


def _measure_frame(row: dict, frame: dict, adapter, kin, positions, n_seeds: int) -> dict:
    gt_label = row["gt_subtask"]
    plausible = [_grasp(c) for c in row["present"]] + [HOME]
    absent = [_grasp(c) for c in row["absent"]]
    if gt_label not in plausible:
        raise ValueError(f"frame {row['global_idx']}: GT label {gt_label!r} is not in its plausible set {plausible}")
    labels = plausible + absent

    unnorm, norm = adapter.predict_action_chunk_batch(
        frame["obs"], frame["task"], labels,
        metadatas=[frame["metadata"]] * len(labels),
        noise=adapter.flow_noise_like(len(labels), 0),
        inference_action_mode="continuous",
    )
    acts = {label: norm[i] for i, label in enumerate(labels)}
    raw = {label: unnorm[i] for i, label in enumerate(labels)}

    seed_noise = torch.cat([adapter.flow_noise_like(1, seed) for seed in range(1, n_seeds + 1)], dim=0)
    seed_unnorm, seed_norm = adapter.predict_action_chunk_batch(
        frame["obs"], frame["task"], [gt_label] * n_seeds,
        metadatas=[frame["metadata"]] * n_seeds,
        noise=seed_noise,
        inference_action_mode="continuous",
    )
    seed_draws = [seed_norm[i] for i in range(n_seeds)]
    seed_floor, _ = _pairwise_rmse(seed_draws)
    floor = max(seed_floor, 1e-9)

    grasp_labels = [_grasp(c) for c in row["present"]]
    spreads = {
        "plausible": _pairwise_rmse([acts[l] for l in plausible])[0] if len(plausible) >= 2 else None,
        "grasp": _pairwise_rmse([acts[l] for l in grasp_labels])[0] if len(grasp_labels) >= 2 else None,
        "absent": _pairwise_rmse([acts[l] for l in absent])[0] if len(absent) >= 2 else None,
    }
    separation = {k: (v / floor if v is not None else None) for k, v in spreads.items()}

    groups = {
        "present_other": [l for l in grasp_labels if l != gt_label],
        "home": [HOME] if gt_label != HOME else [],
        "absent": absent,
    }
    displacement = {
        g: (float(np.mean([_rmse(acts[l], acts[gt_label]) for l in ls])) if ls else None)
        for g, ls in groups.items()
    }
    displacement_ratio = {g: (v / floor if v is not None else None) for g, v in displacement.items()}

    gt_norm = adapter.normalize_gt_actions(frame["gt_actions"], frame["state"])
    mse = {l: float((acts[l] - gt_norm).pow(2).mean()) for l in labels}
    rank_p = sorted(plausible, key=mse.get).index(gt_label) + 1
    rank_all = sorted(labels, key=mse.get).index(gt_label) + 1
    gt_rank_normalized = (rank_p - 1) / (len(plausible) - 1) if len(plausible) >= 2 else None

    # Task-space: approach toward each present object with a known position.
    state = frame["state"].reshape(-1).float().cpu().numpy()
    e0 = kin.ee_path(state[None, :])[0]
    ends = {l: kin.ee_path(raw[l].float().cpu().numpy())[-1] for l in labels}
    seed_ends = [kin.ee_path(seed_unnorm[i].float().cpu().numpy())[-1] for i in range(n_seeds)]

    def approach(end: np.ndarray, p: np.ndarray) -> float:
        return float(np.linalg.norm(e0 - p) - np.linalg.norm(end - p))

    direction = []
    for cls in row["present"]:
        p = _next_position(positions, row["episode_idx"], cls, row["frame_idx"])
        if p is None or _grasp(cls) == gt_label:
            continue
        direction.append({
            "class": cls,
            "delta": approach(ends[_grasp(cls)], p) - approach(ends[gt_label], p),
            "null": [approach(seed_ends[i], p) - approach(ends[gt_label], p) for i in range(n_seeds)],
            "distance_m": float(np.linalg.norm(e0 - p)),
        })
    home_contrast = None
    if gt_label != HOME:
        p = _next_position(positions, row["episode_idx"], gt_label[len("grasp "):], row["frame_idx"])
        if p is not None:
            home_contrast = approach(ends[gt_label], p) - approach(ends[HOME], p)

    return {
        **{k: row[k] for k in ("episode_idx", "frame_idx", "global_idx", "window", "offset", "seconds_in", "gt_subtask", "present", "absent")},
        "plausible": plausible,
        "seed_floor_mean": seed_floor,
        "spread": spreads,
        "separation": separation,
        "displacement": displacement,
        "displacement_ratio": displacement_ratio,
        "gt_rank_plausible": rank_p,
        "gt_rank_normalized": gt_rank_normalized,
        "gt_top1_plausible": rank_p == 1,
        "gt_top1_null": 1.0 / len(plausible),
        "gt_rank_all": rank_all,
        "n_all": len(labels),
        "gt_mse_by_label": mse,
        "direction": direction,
        "home_contrast_m": home_contrast,
        "_acts": acts,
        "_gt_norm": gt_norm,
    }


# ── Figures ───────────────────────────────────────────────────────────────────


def _box(ax, columns: list[tuple[str, list[float | None]]], ylabel: str, title: str) -> None:
    data = [[v for v in vals if v is not None] for _, vals in columns]
    ticks = [f"{name}\n(n={len(d)})" for (name, _), d in zip(columns, data, strict=True)]
    ax.boxplot([d if d else [np.nan] for d in data], tick_labels=ticks, widths=0.55)
    for i, d in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(d)), d, s=9, alpha=0.5, color="#457B9D")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.tick_params(labelsize=8)


def _render_summary(rows: list[dict], summary: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))
    _box(
        axes[0],
        [
            ("plausible", [r["spread"]["plausible"] for r in rows]),
            ("grasp only", [r["spread"]["grasp"] for r in rows]),
            ("absent", [r["spread"]["absent"] for r in rows]),
            ("flow seed", [r["seed_floor_mean"] for r in rows]),
        ],
        "mean pairwise RMSE (normalized action)",
        "Label-set spread against the seed floor\n"
        + "  ".join(f"$S_{{{k[0]}}}$ = {_fmt(summary['separation_median'][k])}x" for k in ("plausible", "grasp", "absent")),
    )
    _caption(axes[0], [
        r"One frame gives one point per box: mean RMSE over all pairs of chunks the set produced,",
        r"and over all pairs of reseeds of the true label (right). $S$ = set spread / seed floor, median over frames.",
        r"Sets with one label (a home window over an empty table) contribute nothing to that box.",
    ])
    _box(
        axes[1],
        [
            ("other present\nobject", [r["displacement_ratio"]["present_other"] for r in rows]),
            ("return to home", [r["displacement_ratio"]["home"] for r in rows]),
            ("absent object", [r["displacement_ratio"]["absent"] for r in rows]),
        ],
        "RMSE to the true-label chunk / seed floor",
        "How far a wrong label moves the chunk off the truthful one\n"
        + "  ".join(f"{g}: {_fmt(summary['displacement_ratio_median'][g])}x" for g in GROUPS),
    )
    axes[1].axhline(1.0, color="#E63946", linestyle="--", linewidth=1.2, label="seed floor")
    axes[1].legend(fontsize=8, frameon=False)
    _caption(axes[1], [
        r"$D_G$ per frame: mean RMSE between the chunk under each label of group $G$ and the chunk under",
        r"the TRUE label (same seed), over the seed floor. 1 = the label moves the chunk no more than reseeding.",
        r"Object read: other-present > 1. Verb only: home $\gg$ other-present $\approx$ 1. Scene-blind: absent $\approx$ other-present.",
    ])
    ranks = [r["gt_rank_normalized"] for r in rows if r["gt_rank_normalized"] is not None]
    if ranks:
        axes[2].hist(ranks, bins=np.linspace(-0.05, 1.05, 12), color="#457B9D", edgecolor="white")
        axes[2].axvline(0.5, color="#E63946", linestyle="--", linewidth=1.2, label="uniform null (0.5)")
        axes[2].axvline(float(np.mean(ranks)), color="black", linewidth=1.4, label=f"mean {np.mean(ranks):.2f}")
        axes[2].legend(fontsize=8, frameon=False)
    axes[2].set_xlabel("normalized rank of the true label among the plausible set (0 = best)")
    axes[2].set_ylabel("frames")
    axes[2].set_title(
        f"Where the true label ranks (n={len(ranks)} frames)\n"
        f"top-1 {_fmt(summary['gt_top1_fraction'], pct=True)} vs null {_fmt(summary['gt_top1_null_mean'], pct=True)}",
        fontsize=10,
    )
    axes[2].grid(True, axis="y", alpha=0.25, linestyle=":")
    _caption(axes[2], [
        r"Every plausible label's chunk is scored against the demonstrated chunk and sorted; $\tilde r = (r-1)/(|V_p|-1)$.",
        r"Piled at 0 = the true label drives the policy closest to the demonstration; flat = read but not understood.",
        r"Noise until $S_p$ clears the floor. $|V_p|$ varies per frame, hence the normalization.",
    ])
    fig.suptitle(
        f"Scene-plausible subtask sweep (n={len(rows)} empty-handed frames, {summary['n_windows']} windows)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(os.path.join(output_dir, "subtask_scene_sweep.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def _render_direction(rows: list[dict], summary: dict, output_dir: str) -> None:
    pairs = [(r, d) for r in rows for d in r["direction"]]
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    deltas = [d["delta"] for _, d in pairs]
    nulls = [v for _, d in pairs for v in d["null"]]
    homes = [r["home_contrast_m"] for r in rows if r["home_contrast_m"] is not None]
    _box(
        axes[0],
        [("naming the other\nobject X", deltas), ("reseed null", nulls), ("true label vs\nreturn to home", homes)],
        "extra approach toward the object over the chunk (m)",
        "Task-space: does the named object become the target?\n"
        f"median $\\Delta$ {_fmt(summary['direction']['delta_median'], 3)} m, null |.| {_fmt(summary['direction']['null_abs_median'], 3)} m",
    )
    axes[0].axhline(0.0, color="#E63946", linestyle="--", linewidth=1.2)
    _caption(axes[0], [
        r"$\Delta(X) = \alpha_{\mathrm{grasp}\,X}(X) - \alpha_{\ell^\star}(X)$ with $\alpha_\ell(X)$ = distance to $X$ closed by the chunk",
        r"under label $\ell$ (end-effector FK, last step). $p_X$ = tool position at the close of $X$'s next grasp.",
        r"Null = the same difference between two reseeds of the true label. Right box: true label vs home toward its own object.",
    ])
    colors = plt.cm.tab10(np.arange(10))
    classes = sorted({d["class"] for _, d in pairs})
    for i, cls in enumerate(classes):
        xs = [d["distance_m"] for _, d in pairs if d["class"] == cls]
        ys = [d["delta"] for _, d in pairs if d["class"] == cls]
        axes[1].scatter(xs, ys, s=22, color=colors[i % 10], label=cls, alpha=0.85)
    axes[1].axhline(0.0, color="#E63946", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("current tool distance to the named object (m)")
    axes[1].set_ylabel("$\\Delta$ (m)")
    axes[1].set_title("Per (frame, object) pair", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(True, alpha=0.25, linestyle=":")
    _caption(axes[1], [
        r"A chunk covers a fraction of a reach, so read the sign and the size against the null, not against the distance.",
    ])
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    fig.savefig(os.path.join(output_dir, "subtask_scene_sweep_direction.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def _render_fan(rows: list[dict], grid: int, output_dir: str) -> None:
    """Rows: the joints the plausible set moves most; columns: one frame per window, most present classes first."""
    ordered = sorted(rows, key=lambda r: (-len(r["present"]), r["episode_idx"], r["frame_idx"]))
    columns, seen = [], set()
    for r in ordered:
        key = (r["episode_idx"], tuple(r["window"]))
        if key not in seen and r["offset"] == "25%":
            seen.add(key)
            columns.append(r)
    columns = (columns or ordered)[: max(grid, 1)]
    stds = [torch.stack([r["_acts"][l] for l in r["plausible"]]).float().std(dim=0).mean(dim=0) for r in columns]
    joints = list(np.argsort(-torch.stack(stds).mean(dim=0).numpy())[:grid])

    fig, axes = plt.subplots(len(joints), len(columns), figsize=(4.2 * len(columns), 2.7 * len(joints)),
                             squeeze=False, sharex=True, sharey="row")
    palette = plt.cm.tab10(np.arange(10))
    for col, r in enumerate(columns):
        steps = np.arange(r["_gt_norm"].shape[0])
        for row_i, joint in enumerate(joints):
            ax = axes[row_i][col]
            for i, label in enumerate(r["plausible"] + [_grasp(c) for c in r["absent"]]):
                absent = label not in r["plausible"]
                ax.plot(steps, r["_acts"][label][:, joint], linewidth=0.9, alpha=0.4 if absent else 0.9,
                        linestyle=":" if absent else ("-." if label == HOME else "-"),
                        color="grey" if absent else palette[i % 10], label=label + (" (absent)" if absent else ""))
            ax.plot(steps, r["_gt_norm"][:, joint], color="black", linewidth=2.2, label="demonstrated")
            ax.plot(steps, r["_acts"][r["gt_subtask"]][:, joint], color="black", linestyle="--", linewidth=1.7,
                    label="chunk under the true label")
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.tick_params(labelsize=8)
            if row_i == 0:
                ax.set_title(f"ep {r['episode_idx']} fr {r['frame_idx']} ({r['offset']})  $S_p$ = {_fmt(r['separation']['plausible'])}x\n"
                             f"true: {r['gt_subtask']}", fontsize=8.5)
            if row_i == len(joints) - 1:
                ax.set_xlabel("chunk step", fontsize=9)
            if col == 0:
                name = REBOT_JOINT_NAMES[joint] if joint < len(REBOT_JOINT_NAMES) else joint
                ax.set_ylabel(f"{name}\nnormalized action", fontsize=9)
            if row_i == 0:
                ax.legend(fontsize=6.5, frameon=False, loc="best")
    fig.suptitle("Scene-plausible fan: solid = present object, dash-dot = return to home, dotted grey = absent object",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(output_dir, "subtask_scene_sweep_fan.png"), bbox_inches="tight", dpi=110)
    plt.close(fig)


def _fmt(value, places: int = 2, pct: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.{places}f}"


# ── Entry point ───────────────────────────────────────────────────────────────


def run(adapter, dataset, cfg, output_dir: str) -> None:
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[subtask_scene_sweep] needs continuous flow actions — skipping.")
        return
    p = cfg.probe_parameters
    frames_path = getattr(p, "subtask_scene_sweep_frames", None)
    if not frames_path:
        raise ValueError("probe_parameters.subtask_scene_sweep_frames must point at the reviewed frames.json")
    frame_rows = json.load(open(frames_path))
    n_seeds = max(int(p.subtask_scene_sweep_n_seeds or p.subtask_sweep_n_seeds or p.n_seeds), 2)
    chunk_size = int(cfg.policy.chunk_size)

    makedirs(output_dir)
    kin = RebotKinematics()
    positions = _object_positions(dataset, kin)
    n_forwards = sum(len(r["present"]) + 1 + len(r["absent"]) + n_seeds for r in frame_rows)
    logging.info(f"[subtask_scene_sweep] {len(frame_rows)} frames from {frames_path}: {n_forwards} forward passes")

    adapter._set_probe_cuda_graph_enabled(False)
    rows: list[dict] = []
    try:
        for row in frame_rows:
            frame = probe_frame_inputs(dataset, cfg, int(row["global_idx"]), chunk_size)
            if (frame["episode_idx"], frame["frame_idx"]) != (row["episode_idx"], row["frame_idx"]):
                raise RuntimeError(
                    f"{frames_path} entry global {row['global_idx']} = ep{row['episode_idx']} fr{row['frame_idx']} "
                    f"but {dataset.root} resolves it to ep{frame['episode_idx']} fr{frame['frame_idx']}: frame list built for another root"
                )
            if frame["subtask"] != row["gt_subtask"]:
                logging.warning(f"[subtask_scene_sweep] global {row['global_idx']}: dataset subtask {frame['subtask']!r} != frame list {row['gt_subtask']!r}; sweeping the frame list's")
            rows.append(_measure_frame(row, frame, adapter, kin, positions, n_seeds))
    finally:
        adapter._restore_probe_cuda_graph_enabled()
    if not rows:
        logging.warning("[subtask_scene_sweep] no frames measured.")
        return

    pairs = [d for r in rows for d in r["direction"]]
    ranks = [r["gt_rank_normalized"] for r in rows if r["gt_rank_normalized"] is not None]
    offsets = sorted({r["offset"] for r in rows}, key=[r["offset"] for r in frame_rows].index)
    summary = {
        "n_frames": len(rows),
        "n_windows": len({(r["episode_idx"], tuple(r["window"])) for r in rows}),
        "n_seeds": n_seeds,
        "frames_path": frames_path,
        "seed_floor_median": _median([r["seed_floor_mean"] for r in rows]),
        "separation_median": {k: _median([r["separation"][k] for r in rows]) for k in ("plausible", "grasp", "absent")},
        "separation_n": {k: sum(r["separation"][k] is not None for r in rows) for k in ("plausible", "grasp", "absent")},
        "displacement_ratio_median": {g: _median([r["displacement_ratio"][g] for r in rows]) for g in GROUPS},
        "gt_rank_normalized_mean": float(np.mean(ranks)) if ranks else None,
        "gt_rank_normalized_null": 0.5,
        "gt_top1_fraction": float(np.mean([r["gt_top1_plausible"] for r in rows if r["gt_rank_normalized"] is not None])) if ranks else None,
        "gt_top1_null_mean": float(np.mean([r["gt_top1_null"] for r in rows if r["gt_rank_normalized"] is not None])) if ranks else None,
        "direction": {
            "n_pairs": len(pairs),
            "delta_median": _median([d["delta"] for d in pairs]),
            "delta_positive_fraction": float(np.mean([d["delta"] > 0 for d in pairs])) if pairs else None,
            "null_abs_median": _median([abs(v) for d in pairs for v in d["null"]]),
            "home_contrast_median_m": _median([r["home_contrast_m"] for r in rows]),
        },
        "by_offset": {
            o: {
                "n": sum(r["offset"] == o for r in rows),
                "separation_plausible_median": _median([r["separation"]["plausible"] for r in rows if r["offset"] == o]),
                "displacement_ratio_median": {g: _median([r["displacement_ratio"][g] for r in rows if r["offset"] == o]) for g in GROUPS},
            }
            for o in offsets
        },
        "verdict_note": (
            "S_grasp ~1 and D_present_other ~1 => the object in the clause does not reach the chunk; "
            "D_home >> D_present_other => verb read, object not; D_absent ~ D_present_other => scene-blind. "
            "direction.delta above its null => the named object is steered toward in task space."
        ),
        "per_frame": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    with open(os.path.join(output_dir, "subtask_scene_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _render_summary(rows, summary, output_dir)
    _render_direction(rows, summary, output_dir)
    _render_fan(rows, int(getattr(p, "subtask_sweep_fan_grid", 4)), output_dir)

    write_index(
        output_dir,
        sys.modules[__name__],
        title="Subtask Scene Sweep",
        group="Steering",
        claim="Does the subtask clause move the chunk when every label offered is plausible for the scene?",
        summary=summary,
        see_also=["subtask_sweep", "input_swap", "metadata_steering"],
        metrics=[
            Metric("separation_median.grasp", "object-only separation S_g", good="high", fmt=2, baseline=1.0, primary=True, trend=True,
                   note="Spread over present-object grasp labels / seed floor. Frames with one present object give none."),
            Metric("separation_median.plausible", "plausible-set separation S_p", good="high", fmt=2, baseline=1.0, trend=True),
            Metric("separation_median.absent", "absent-set separation S_a", good="none", fmt=2, baseline=1.0, trend=True,
                   note="What subtask_sweep measured, on these frames."),
            Metric("displacement_ratio_median.present_other", "D other present object", good="high", fmt=2, baseline=1.0, primary=True, trend=True),
            Metric("displacement_ratio_median.home", "D return to home", good="none", fmt=2, baseline=1.0, trend=True),
            Metric("displacement_ratio_median.absent", "D absent object", good="none", fmt=2, baseline=1.0, trend=True),
            Metric("gt_rank_normalized_mean", "normalized GT rank (plausible)", good="low", fmt=2, baseline=0.5, trend=True),
            Metric("direction.delta_median", "target agreement delta (m)", good="high", fmt=3, baseline=0.0, trend=True,
                   note="Read against direction.null_abs_median."),
            Metric("direction.null_abs_median", "target agreement null |delta| (m)", good="none", fmt=3),
        ],
        panels=[
            Panel("subtask_scene_sweep.png", "Spread by label set, displacement off the true label, rank of the true label",
                  "Left: the three sets' spreads against the seed floor; the numbers are the ratios. Middle: how far each "
                  "kind of wrong label pushes the chunk off the truthful prediction, in seed-floor units — the object is "
                  "read only if 'other present object' clears 1. Right: normalized rank of the true label among the "
                  "plausible set, uniform null at 0.5.", primary=True, refs=["subtask_sweep"]),
            Panel("subtask_scene_sweep_direction.png", "Task-space target agreement",
                  "Naming another present object X: does the chunk close more distance to X than the true label's chunk "
                  "does? Metres over the chunk, against the reseed null; right panel per pair vs current distance.",
                  primary=True, refs=["action_trace"]),
            Panel("subtask_scene_sweep_fan.png", "Fan over the frames with the most present objects",
                  "Rows are the joints the plausible set moves most; solid = present-object grasp labels, dash-dot = "
                  "return to home, dotted grey = absent objects, black = demonstration, dashed black = the true label.",
                  refs=["subtask_sweep"]),
        ],
    )
    s, d = summary["separation_median"], summary["displacement_ratio_median"]
    logging.info(
        f"[subtask_scene_sweep] n={len(rows)}  S plausible {_fmt(s['plausible'])}x grasp {_fmt(s['grasp'])}x absent {_fmt(s['absent'])}x  "
        f"D present-other {_fmt(d['present_other'])}x home {_fmt(d['home'])}x absent {_fmt(d['absent'])}x  "
        f"rank {_fmt(summary['gt_rank_normalized_mean'])} (null 0.5)  "
        f"direction delta {_fmt(summary['direction']['delta_median'], 3)} m vs null {_fmt(summary['direction']['null_abs_median'], 3)} m "
        f"({summary['direction']['n_pairs']} pairs)"
    )


@parser.wrap()
def cli(cfg: SceneSweepProbeConfig):
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
    run(adapter, dataset, cfg, os.path.join(cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", "subtask_scene_sweep"))


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
