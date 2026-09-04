"""Depth read at gripper events: does depth content move the chunk where a grasp is
about to happen, and not where nothing is?

The matched-depth counterfactual (``depth_modality_probe``) samples frames evenly and so
measures the depth read averaged over a whole episode, most of which is carrying,
retreating or waiting. This probe picks its frames by the commanded gripper instead and
reports every readout per stratum, so the question becomes a contrast: pre-grasp against
frames far from any gripper event, on the same checkpoint, at identical flow noise.

Frames come from ``meta/depth_gripper_events.parquet`` and the dense
``depth_gripper_event_labels.parquet`` deltas — the locked rubric the auxiliary head is
trained on (commanded ``gripper.pos``, closed above $-60^\\circ$, open below
$-90^\\circ$, 0.5 s persistence). Nothing here reads observation.state or a semantic
annotation to choose a frame.

Strata, every anchor snapped onto the image/depth stride grid:

  ``pre_close_{L}s``  $L$ seconds before a commanded close, and that close is the next one
  ``pre_open_{L}s``   the same before a commanded open
  ``carry``           gripper closed, no open within $L_{max}+1$ s, at least 1 s after the close
  ``free``            gripper open, no close within $L_{max}+1$ s, at least 1 s after the open

Controls are sized per episode to that episode's event frames, so $n$ is matched where
the contrast is read.

Conditions intervene on the wrist depth window (current frame plus history slots); the
top camera and the wrist RGB are present throughout:

  ``deployment``   the rollout prompt
  ``shift_{D}s``   the depth window taken $D$ seconds earlier in the same episode, RGB left
                   at $t$. On an approach the object is farther in depth than in RGB, so
                   the sign of the effect is known before the run.
  ``cross_phase``  the depth window of a same-stratum frame from another episode, nearest
                   by standardized joint state — same phase, different scene
  ``z_offset``     every valid pixel pushed $\\Delta z$ mm farther; the pinhole
                   back-projection scales $X$ and $Y$ with it, and pixels pushed past
                   ``z_max_mm`` drop out of the valid mask
  ``no_depth``     window removed (learned null bank). Untrained shape: depth dropout is
                   0 in training, so its penalty mixes lost information with shift. Kept
                   as the reference the older probe reports, hatched in the figure.

Readouts per frame and condition, all paired against the same frame's deployment chunk
and never against the demonstration:

  - displacement: ``trajectory_error_components(chunk_c, chunk_{dep}, hold)`` — path,
    shape and terminal error of the treated chunk from the deployment chunk, each over the
    deployment chunk's own excursion from ``hold``. 1 means the treatment moved the chunk
    as far as the chunk itself moves.
  - seed floor: the same displacement for the deployment prompt under the other flow
    seeds. An effect that does not clear it is noise.
  - gripper timing, in the dataset's own degrees from the unnormalized chunk: the terminal
    delta $g_T - g_{now}$ (positive is toward closed) and the first step at which the
    labels' hysteresis rule fires from the frame's current state.
  - the auxiliary head: $\\sigma(\\mathrm{logit})$ for close and open, read off the same
    forward. The head sees only the depth tokens, so this is the depth path's own opinion
    of the event, before the action expert.

The claim under test: under ``shift`` and ``z_offset`` the pre-close strata move more
than ``free``, the terminal gripper delta falls (the chunk closes less when the object
reads farther) and the head's $p_{close}$ falls with it. A displacement that is flat
across strata means depth acts as a global bias, not as a grasp cue.

Outputs under ``<output_dir>/``:
  depth_event.json   summary, per-stratum tables, per-frame rows with donor provenance
  depth_event.png    displacement, terminal gripper shift and head $p_{close}$ by stratum

Runs inside rl_offline's validation loop when ``probe_parameters.enable_depth_event`` is
set, or standalone against ``val_dataset_path``:

    uv run python -m lerobot.probes.depth_event_probe --config config_rl.yaml
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.depth_modality_probe import (
    _drop_depth,
    _load_depth_window,
    _replace_depth_window,
    _stale_depth_index,
)
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    build_episode_index,
    load_probe_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
)
from lerobot.utils.action_metrics import trajectory_error_components
from lerobot.utils.depth_gripper_events import (
    DEPTH_GRIPPER_CLOSE_TARGET,
    DEPTH_GRIPPER_EVENT_LABEL_FILENAME,
    DEPTH_GRIPPER_OPEN_TARGET,
    load_depth_gripper_event_targets,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

DEPLOYMENT = "deployment"
CROSS_PHASE = "cross_phase"
Z_OFFSET = "z_offset"
NO_DEPTH = "no_depth"
CARRY = "carry"
FREE = "free"
EVENT_TYPES = ("close", "open")
RELATIVE_KEYS = ("path_relative", "shape_relative", "terminal_relative")
TARGET_KEYS = {"close": DEPTH_GRIPPER_CLOSE_TARGET, "open": DEPTH_GRIPPER_OPEN_TARGET}


def shift_condition(seconds: float) -> str:
    return f"shift_{seconds:g}s"


def event_stratum(event_type: str, lead_s: float) -> str:
    return f"pre_{event_type}_{lead_s:g}s"


def _seconds(text: str) -> list[float]:
    return [float(s) for s in str(text).split(",") if s.strip()]


def _mean_sem(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "sem": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    sem = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else None
    return {"mean": float(array.mean()), "sem": sem, "n": int(array.size)}


# ──────────────────────────────────────────────────────────────────────────────
# Frame selection — the event labels choose, the model never does
# ──────────────────────────────────────────────────────────────────────────────


def load_event_labels(dataset) -> dict:
    """The locked gripper-event sidecars, identity-checked against the dataset rows."""
    load_depth_gripper_event_targets(dataset)  # raises on a row-count or identity mismatch
    meta = Path(dataset.root) / "meta"
    labels = pd.read_parquet(
        meta / DEPTH_GRIPPER_EVENT_LABEL_FILENAME,
        columns=["depth_gripper_close_delta", "depth_gripper_open_delta"],
    )
    with open(meta / "depth_gripper_event_labels_info.json") as f:
        info = json.load(f)
    return {
        "events": pd.read_parquet(meta / "depth_gripper_events.parquet"),
        "close_delta": labels["depth_gripper_close_delta"].to_numpy(dtype=np.int64),
        "open_delta": labels["depth_gripper_open_delta"].to_numpy(dtype=np.int64),
        "gripper_dim": int(info["resolved_gripper_dimension"]),
        "close_threshold": float(info["thresholds_degrees"]["close"]),
        "open_threshold": float(info["thresholds_degrees"]["open"]),
        "label_fps": float(info["fps"]),
        "rubric": str(info["rubric_version"]),
    }


def select_frames(
    dataset,
    labels: dict,
    *,
    leads_s: list[float],
    fps: int,
    stride: int,
    max_episodes: int | None,
    seed: int,
) -> list[dict]:
    """Event-relative anchors plus matched far-from-event controls, per episode.

    A pre-event anchor is kept only when the labels agree that the event it was placed
    before is the next one of its type (``delta == event - frame``): a nearer event, or a
    lead beyond the labels' 5 s cutoff, drops it. Controls need no event within
    ``max(lead) + 1`` s and a 1 s settle after the last transition, so the carry and free
    strata never overlap a pre-event window.
    """
    by_episode = build_episode_index(dataset)
    episodes = sorted(by_episode)
    if max_episodes is not None:
        rng = np.random.RandomState(seed)
        episodes = sorted(
            rng.choice(episodes, size=min(max_episodes, len(episodes)), replace=False).tolist()
        )
    events = labels["events"]
    close_delta, open_delta = labels["close_delta"], labels["open_delta"]
    leads = [(lead_s, int(round(lead_s * fps))) for lead_s in leads_s]
    far = max(frames for _, frames in leads) + fps
    settle = fps

    rows: list[dict] = []
    for episode_idx in episodes:
        indices = by_episode[episode_idx]
        episode_start, length = indices[0], len(indices)
        episode_events = events[events["episode_index"] == episode_idx]
        intervals = sorted(
            {
                (int(start), int(stop))
                for start, stop in zip(
                    episode_events["closed_interval_start"], episode_events["closed_interval_stop"]
                )
            }
        )

        def containing(frame: int) -> tuple[int, int] | None:
            return next(((a, b) for a, b in intervals if a <= frame < b), None)

        counts: Counter = Counter()
        for event in episode_events.itertuples():
            event_type = str(event.event_type)
            event_frame = int(event.frame_index)
            delta = close_delta if event_type == "close" else open_delta
            for lead_s, lead_frames in leads:
                frame = event_frame - lead_frames
                if frame < 0:
                    continue
                frame -= frame % stride
                global_idx = episode_start + frame
                if int(delta[global_idx]) != event_frame - frame:
                    logging.info(
                        f"[depth_event] ep{episode_idx} {event_type}@{event_frame} lead {lead_s:g}s: "
                        f"frame {frame} has delta {int(delta[global_idx])}, not this event — skipped"
                    )
                    continue
                rows.append(
                    {
                        "stratum": event_stratum(event_type, lead_s),
                        "event_type": event_type,
                        "event_frame": event_frame,
                        "lead_frames": event_frame - frame,
                        "episode_idx": int(episode_idx),
                        "frame_idx": int(frame),
                        "global_idx": int(global_idx),
                        "closed_now": containing(frame) is not None,
                    }
                )
                counts[event_type] += 1

        carry: list[int] = []
        free: list[int] = []
        for frame in range(0, length, stride):
            global_idx = episode_start + frame
            interval = containing(frame)
            if interval is not None:
                if frame - interval[0] >= settle and not (0 <= open_delta[global_idx] <= far):
                    carry.append(global_idx)
            else:
                previous_open = max((stop for _, stop in intervals if stop <= frame), default=None)
                settled = previous_open is None or frame - previous_open >= settle
                if settled and not (0 <= close_delta[global_idx] <= far):
                    free.append(global_idx)
        n_control = max(counts.values(), default=0)
        for name, candidates in ((CARRY, carry), (FREE, free)):
            n = min(n_control, len(candidates))
            positions = sorted(set(np.linspace(0, len(candidates) - 1, n, dtype=int).tolist())) if n else []
            for pos in positions:
                global_idx = candidates[pos]
                rows.append(
                    {
                        "stratum": name,
                        "event_type": None,
                        "event_frame": None,
                        "lead_frames": None,
                        "episode_idx": int(episode_idx),
                        "frame_idx": int(global_idx - episode_start),
                        "global_idx": int(global_idx),
                        "closed_now": name == CARRY,
                    }
                )
    return rows


def match_cross_phase_donors(dataset, rows: list[dict]) -> dict[int, dict]:
    """For every anchor, the same-stratum anchor from another episode nearest in
    standardized joint state. Same phase by construction; only the scene differs."""
    states = {
        row["global_idx"]: torch.as_tensor(dataset.hf_dataset[row["global_idx"]]["observation.state"])
        .float()
        .numpy()
        for row in rows
    }
    scale = np.maximum(np.stack(list(states.values())).std(axis=0), 1e-6)
    donors: dict[int, dict] = {}
    for row in rows:
        candidates = [
            other
            for other in rows
            if other["stratum"] == row["stratum"] and other["episode_idx"] != row["episode_idx"]
        ]
        if not candidates:
            continue
        anchor = states[row["global_idx"]]

        def distance(candidate: dict) -> float:
            return float(np.sqrt(np.mean(((states[candidate["global_idx"]] - anchor) / scale) ** 2)))

        donor = min(candidates, key=lambda candidate: (distance(candidate), candidate["global_idx"]))
        donors[row["global_idx"]] = {
            "global_idx": int(donor["global_idx"]),
            "episode_idx": int(donor["episode_idx"]),
            "frame_idx": int(donor["frame_idx"]),
            "state_distance": distance(donor),
        }
    return donors


# ──────────────────────────────────────────────────────────────────────────────
# Interventions and readouts
# ──────────────────────────────────────────────────────────────────────────────


def _offset_depth_window(obs: dict, *, depth_obs_key: str, levels: float) -> dict:
    """Push every valid pixel of the current and history depth farther by ``levels``.
    Zero is the sensor's own invalid value and stays zero."""
    out = dict(obs)
    for key in [depth_obs_key, *(k for k in obs if str(k).startswith("history.depth."))]:
        depth = obs[key]
        out[key] = torch.where(depth > 0, depth + levels, depth)
    return out


def gripper_transitions(
    gripper: np.ndarray, closed: bool, *, close_threshold: float, open_threshold: float
) -> tuple[int | None, int | None]:
    """First chunk step at which the labels' hysteresis rule closes / opens the gripper,
    started from the frame's own state. ``None`` when the chunk never gets there."""
    close_step = open_step = None
    for step, command in enumerate(gripper):
        now_closed = bool(command > close_threshold) if not closed else bool(command >= open_threshold)
        if now_closed and not closed and close_step is None:
            close_step = step
        if closed and not now_closed and open_step is None:
            open_step = step
        closed = now_closed
    return close_step, open_step


def _relative(prediction: torch.Tensor, reference: torch.Tensor, hold: torch.Tensor) -> dict:
    components = trajectory_error_components(prediction, reference, hold)
    return {key: float(components[key]) for key in RELATIVE_KEYS}


# ──────────────────────────────────────────────────────────────────────────────
# Summary, figure, manifest
# ──────────────────────────────────────────────────────────────────────────────


def summarize(per_frame: list[dict], strata: list[str], conditions: list[str]) -> dict:
    treatments = conditions[1:]
    by_stratum: dict[str, dict] = {}
    for name in strata:
        rows = [r for r in per_frame if r["stratum"] == name]
        entry = {
            "n": len(rows),
            "n_by_episode": dict(sorted(Counter(r["episode_idx"] for r in rows).items())),
            "seed_floor": _mean_sem([r["seed_floor"]["path_relative"] for r in rows]),
            "target_close_mean": _mean_sem(
                [r["targets"]["close"] for r in rows if r["targets"]["close"] is not None]
            )["mean"],
            "target_open_mean": _mean_sem(
                [r["targets"]["open"] for r in rows if r["targets"]["open"] is not None]
            )["mean"],
            "conditions": {},
        }
        for condition in conditions:
            present = [r for r in rows if condition in r["gripper"]]
            cond: dict = {"n": len(present)}
            if condition != DEPLOYMENT:
                for key in RELATIVE_KEYS:
                    cond[key] = _mean_sem([r["displacement"][condition][key] for r in present])
                cond["terminal_delta_shift_deg"] = _mean_sem(
                    [
                        r["gripper"][condition]["terminal_delta"] - r["gripper"][DEPLOYMENT]["terminal_delta"]
                        for r in present
                    ]
                )
                paired_heads = [
                    (r["head"][DEPLOYMENT], r["head"][condition])
                    for r in present
                    if r["head"][DEPLOYMENT] is not None and r["head"][condition] is not None
                ]
                cond["p_close_drop"] = _mean_sem([d["p_close"] - c["p_close"] for d, c in paired_heads])
                cond["p_open_drop"] = _mean_sem([d["p_open"] - c["p_open"] for d, c in paired_heads])
            cond["terminal_delta_deg"] = _mean_sem([r["gripper"][condition]["terminal_delta"] for r in present])
            for transition in ("close", "open"):
                steps = [r["gripper"][condition][f"{transition}_step"] for r in present]
                fired = [s for s in steps if s is not None]
                cond[f"{transition}_fraction"] = float(len(fired) / len(steps)) if steps else None
                cond[f"{transition}_step_mean"] = float(np.mean(fired)) if fired else None
            heads = [r["head"][condition] for r in present if r["head"][condition] is not None]
            cond["p_close"] = _mean_sem([h["p_close"] for h in heads])
            cond["p_open"] = _mean_sem([h["p_open"] for h in heads])
            entry["conditions"][condition] = cond
        by_stratum[name] = entry

    pre_close = [name for name in strata if name.startswith("pre_close_")]
    contrast: dict[str, dict] = {}
    for condition in treatments:
        pre_rows = [r for r in per_frame if r["stratum"] in pre_close and condition in r["displacement"]]
        free_rows = [r for r in per_frame if r["stratum"] == FREE and condition in r["displacement"]]

        def block(rows: list[dict]) -> dict:
            displacement = _mean_sem([r["displacement"][condition]["path_relative"] for r in rows])["mean"]
            floor = _mean_sem([r["seed_floor"]["path_relative"] for r in rows])["mean"]
            heads = [
                r["head"][DEPLOYMENT]["p_close"] - r["head"][condition]["p_close"]
                for r in rows
                if r["head"][DEPLOYMENT] is not None and r["head"][condition] is not None
            ]
            return {
                "n": len(rows),
                "path_relative": displacement,
                "seed_floor": floor,
                "over_floor": None if displacement is None or not floor else displacement / floor,
                "terminal_delta_shift_deg": _mean_sem(
                    [
                        r["gripper"][condition]["terminal_delta"] - r["gripper"][DEPLOYMENT]["terminal_delta"]
                        for r in rows
                    ]
                )["mean"],
                "p_close_drop": _mean_sem(heads)["mean"],
            }

        pre, free = block(pre_rows), block(free_rows)
        contrast[condition] = {
            "pre_close": pre,
            "free": free,
            "ratio": (
                None
                if pre["path_relative"] is None or not free["path_relative"]
                else pre["path_relative"] / free["path_relative"]
            ),
        }
    return {"by_stratum": by_stratum, "contrast": contrast}


def _render(summary: dict, output_path: str) -> None:
    strata = summary["strata"]
    conditions = summary["conditions"]
    treatments = conditions[1:]
    by_stratum = summary["by_stratum"]
    palette = ["#E76F51", "#F4A261", "#2A9D8F", "#457B9D", "#9D4EDD", "#6C757D"]
    colors = {condition: palette[i % len(palette)] for i, condition in enumerate(treatments)}
    colors[DEPLOYMENT] = "#264653"

    def series(condition: str, *path: str) -> tuple[np.ndarray, np.ndarray]:
        values, errors = [], []
        for name in strata:
            node = by_stratum[name]["conditions"].get(condition, {})
            for part in path:
                node = node.get(part, {}) if isinstance(node, dict) else {}
            values.append(np.nan if not isinstance(node, dict) or node.get("mean") is None else node["mean"])
            errors.append(0.0 if not isinstance(node, dict) or node.get("sem") is None else node["sem"])
        return np.asarray(values), np.asarray(errors)

    def grouped(ax, members: list[str], *path: str) -> None:
        width = 0.8 / max(len(members), 1)
        x = np.arange(len(strata))
        for i, condition in enumerate(members):
            values, errors = series(condition, *path)
            bars = ax.bar(
                x + (i - (len(members) - 1) / 2) * width, values, width, yerr=errors,
                color=colors[condition], label=condition, error_kw={"lw": 0.8},
            )
            if condition == NO_DEPTH:
                for bar in bars:
                    bar.set_hatch("///")
                    bar.set_alpha(0.55)
        ax.set_xticks(x, strata, rotation=25, ha="right", fontsize=8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(strata))

    grouped(axes[0], treatments, "path_relative")
    floor = [by_stratum[name]["seed_floor"]["mean"] or np.nan for name in strata]
    axes[0].scatter(x, floor, marker="_", s=500, color="black", zorder=3, label="seed floor")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("path displacement / deployment excursion")
    axes[0].set_title("How far the treated chunk moves, per stratum", fontsize=10)
    axes[0].legend(fontsize=7)

    grouped(axes[1], treatments, "terminal_delta_shift_deg")
    deployment_delta = [
        by_stratum[name]["conditions"][DEPLOYMENT]["terminal_delta_deg"]["mean"] or np.nan for name in strata
    ]
    axes[1].scatter(x, deployment_delta, marker="_", s=500, color="black", zorder=3, label="deployment $g_T-g_{now}$")
    axes[1].axhline(0.0, color="#333333", lw=0.8)
    axes[1].set_ylabel("terminal gripper shift vs deployment (deg)")
    axes[1].set_title("Does the chunk close less when depth reads farther?", fontsize=10)
    axes[1].legend(fontsize=7)

    grouped(axes[2], conditions, "p_close")
    targets = [by_stratum[name]["target_close_mean"] for name in strata]
    axes[2].scatter(
        x, [np.nan if t is None else t for t in targets], marker="_", s=500, color="black", zorder=3,
        label="label target",
    )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("head $p_{close}$")
    axes[2].set_title("The depth-only head's own read of the event", fontsize=10)
    handles, labels = axes[2].get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="///", label="untrained shape"))
    axes[2].legend(handles=handles, fontsize=7)

    headline = summary["headline_condition"]
    contrast = summary["contrast"].get(headline, {})
    fig.suptitle(
        f"Depth at gripper events — n={summary['n_frames']} frames, "
        f"{summary['n_events']['close']} closes / {summary['n_events']['open']} opens  |  "
        f"{headline}: pre-close/free displacement {contrast.get('ratio') or float('nan'):.2f}x, "
        f"terminal gripper shift {(contrast.get('pre_close') or {}).get('terminal_delta_shift_deg') or float('nan'):+.1f} deg",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _write_manifest(output_dir: str, summary: dict) -> dict:
    headline = summary["headline_condition"]
    conditions = summary["conditions"]

    def contrast_metrics(condition: str, *, primary: bool, trend: bool) -> list[Metric]:
        tag = " (untrained shape)" if condition == NO_DEPTH else ""
        return [
            Metric(
                f"contrast.{condition}.ratio",
                f"{condition}: pre-close over free displacement{tag}",
                good="none",
                fmt=2,
                baseline=1.0,
                primary=primary,
                trend=trend,
                note=(
                    "Mean path displacement of the treated chunk from the deployment chunk, over "
                    "the deployment chunk's own excursion, pre-close strata pooled, divided by the "
                    "same number on frames far from any gripper event. 1 means depth moves the "
                    "chunk the same everywhere — a global bias, not a grasp cue."
                ),
            ),
            Metric(
                f"contrast.{condition}.pre_close.terminal_delta_shift_deg",
                f"{condition}: pre-close terminal gripper shift (deg){tag}",
                good="none",
                fmt=2,
                baseline=0.0,
                primary=primary,
                trend=trend,
                note=(
                    "$\\left(g_T - g_{now}\\right)_{treated} - \\left(g_T - g_{now}\\right)_{deployment}$ "
                    "on pre-close frames, in the dataset's degrees (closed is near 0, open is "
                    "negative). Negative means the chunk closes less when the object reads farther."
                ),
            ),
            Metric(
                f"contrast.{condition}.pre_close.p_close_drop",
                f"{condition}: pre-close head p_close drop{tag}",
                good="none",
                fmt=3,
                baseline=0.0,
                primary=primary,
                note=(
                    "$p_{close}(\\text{deployment}) - p_{close}(\\text{treated})$ from the depth-only "
                    "auxiliary head on pre-close frames. Positive means the head reads the treated "
                    "depth as farther from a close."
                ),
            ),
            Metric(
                f"contrast.{condition}.pre_close.over_floor",
                f"{condition}: pre-close displacement over seed floor",
                good="none",
                fmt=2,
                baseline=1.0,
                note="Pre-close displacement divided by the reseed displacement on the same frames. Near 1 is noise.",
            ),
            Metric(
                f"contrast.{condition}.free.over_floor",
                f"{condition}: free displacement over seed floor",
                good="none",
                fmt=2,
                baseline=1.0,
            ),
        ]

    metrics = [
        *contrast_metrics(headline, primary=True, trend=True),
        *[m for c in conditions[1:] if c != headline for m in contrast_metrics(c, primary=False, trend=False)],
        Metric("n_frames", "Frames probed", good="none", fmt=0),
        Metric("n_events.close", "Close events in the probed episodes", good="none", fmt=0),
        Metric("n_events.open", "Open events in the probed episodes", good="none", fmt=0),
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Depth at Gripper Events",
        group="Depth",
        claim="Does depth content move the chunk where a grasp is imminent, and not where nothing is?",
        summary=summary,
        metrics=metrics,
        panels=[
            Panel(
                "depth_event.png",
                "Displacement, terminal gripper shift and head $p_{close}$, by stratum and condition",
                how=(
                    "**Left** — path displacement of each treated chunk from the deployment chunk, "
                    "over the deployment chunk's own motion, log scale; the black tick is the "
                    "reseed floor for that stratum. The claim is a pre-close bar above its free "
                    "bar, both above the tick.\n\n"
                    "**Middle** — the terminal gripper command shift against deployment, in degrees; "
                    "the black tick is deployment's own $g_T - g_{now}$. Pre-close bars below zero "
                    "under ``shift`` and ``z_offset`` mean the chunk closes less when the object "
                    "reads farther.\n\n"
                    "**Right** — the depth-only head's $p_{close}$ per condition; the black tick is "
                    "the label target the head was trained toward at that lead."
                ),
                primary=True,
            ),
            Panel(
                "depth_event.json",
                "Per-stratum tables and per-frame rows",
                how="Each row records the stratum, the event and lead it was placed before, donor provenance per condition, the paired displacements, gripper timing and head probabilities.",
            ),
        ],
        see_also=["depth_modality", "objective"],
        extra={
            "viewer": {
                "metric_groups": [
                    {
                        "title": f"{condition}",
                        "keys": [
                            f"contrast.{condition}.ratio",
                            f"contrast.{condition}.pre_close.terminal_delta_shift_deg",
                            f"contrast.{condition}.pre_close.p_close_drop",
                            f"contrast.{condition}.pre_close.over_floor",
                            f"contrast.{condition}.free.over_floor",
                        ],
                    }
                    for condition in conditions[1:]
                ]
            }
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────


def run(adapter, dataset, cfg, output_dir: str) -> dict | None:
    pointmap_config = getattr(cfg.policy, "pointmap_config", None)
    if pointmap_config is None:
        logging.info("[depth_event] policy.pointmap_config is null — skipping.")
        return None
    if getattr(cfg.policy, "action_mode", "") == "discrete":
        logging.info("[depth_event] needs continuous flow actions — skipping.")
        return None
    labels_path = Path(dataset.root) / "meta" / DEPTH_GRIPPER_EVENT_LABEL_FILENAME
    if not labels_path.is_file():
        logging.warning(f"[depth_event] no gripper-event labels at {labels_path} — skipping.")
        return None

    makedirs(output_dir)
    p = cfg.probe_parameters
    policy = adapter.policy
    chunk_size = int(cfg.policy.chunk_size)
    fps = int(round(float(cfg.env.fps)))
    stride = probe_image_stride(cfg)
    labels = load_event_labels(dataset)
    if int(round(labels["label_fps"])) != fps:
        raise ValueError(f"labels were built at {labels['label_fps']} fps, the run is at {fps}.")

    leads_s = _seconds(getattr(p, "depth_event_leads_s", "1.0,2.0"))
    shifts_s = _seconds(getattr(p, "depth_event_shift_s", "1.0,2.0"))
    z_offset_mm = float(getattr(p, "depth_event_z_offset_mm", 30.0))
    z_offset_levels = z_offset_mm / float(pointmap_config.depth_units_mm)
    n_seeds = max(int(p.n_seeds), 2)
    depth_obs_key = f"observation.depth.{pointmap_config.depth_key}"
    gripper_dim = labels["gripper_dim"]

    rows = select_frames(
        dataset, labels, leads_s=leads_s, fps=fps, stride=stride, max_episodes=p.max_episodes, seed=p.random_seed
    )
    if not rows:
        logging.warning("[depth_event] no frames selected.")
        return None
    donors = match_cross_phase_donors(dataset, rows)
    conditions = [DEPLOYMENT, *[shift_condition(s) for s in shifts_s], CROSS_PHASE, Z_OFFSET, NO_DEPTH]
    strata = [event_stratum(t, lead) for t in EVENT_TYPES for lead in leads_s] + [CARRY, FREE]
    episodes = sorted({row["episode_idx"] for row in rows})
    events = labels["events"]
    n_events = {
        event_type: int(((events["event_type"] == event_type) & events["episode_index"].isin(episodes)).sum())
        for event_type in EVENT_TYPES
    }
    for name in strata:
        by_episode = Counter(r["episode_idx"] for r in rows if r["stratum"] == name)
        logging.info(f"[depth_event] {name:>14s}: {sum(by_episode.values())} frames {dict(sorted(by_episode.items()))}")
    logging.info(
        f"[depth_event] {len(rows)} frames x ({len(conditions)} conditions + 1 batched floor of "
        f"{n_seeds - 1} seeds) forwards; cross-phase donors for {len(donors)} frames"
    )

    adapter._set_probe_cuda_graph_enabled(False)
    noise_seed0 = adapter.flow_noise_like(1, 0)
    floor_noise = torch.cat([adapter.flow_noise_like(1, seed) for seed in range(1, n_seeds)], dim=0)

    def predict(obs: dict, frame: dict, noise: torch.Tensor):
        n = int(noise.shape[0])
        unnorm, norm = adapter.predict_action_chunk_batch(
            obs,
            frame["task"],
            [frame["subtask"]] * n,
            metadatas=[frame["metadata"]] * n,
            noise=noise,
            inference_action_mode="continuous",
        )
        logits = policy._depth_gripper_event_logits
        probs = None if logits is None else torch.sigmoid(logits.detach().float()).cpu()
        return unnorm, norm, probs

    per_frame: list[dict] = []
    try:
        for row in rows:
            frame = probe_frame_inputs(dataset, cfg, row["global_idx"], chunk_size)
            obs = frame["obs"]
            width = frame["gt_actions"].shape[-1]
            hold_raw = frame["state"][:width].unsqueeze(0).repeat(chunk_size, 1)
            hold_norm = adapter.normalize_gt_actions(hold_raw, frame["state"]).float()
            g_now = float(frame["gt_actions"][0, gripper_dim])

            condition_obs = {
                DEPLOYMENT: obs,
                Z_OFFSET: _offset_depth_window(obs, depth_obs_key=depth_obs_key, levels=z_offset_levels),
                NO_DEPTH: _drop_depth(obs, depth_obs_key=depth_obs_key),
            }
            provenance: dict[str, dict] = {}
            for seconds in shifts_s:
                shifted = _stale_depth_index(
                    row["global_idx"], row["frame_idx"], stale_frames=int(round(seconds * fps)), stride=stride
                )
                if shifted is None:
                    continue
                donor_idx, lag = shifted
                window = _load_depth_window(dataset, cfg, donor_idx, obs, depth_obs_key=depth_obs_key)
                condition_obs[shift_condition(seconds)] = _replace_depth_window(obs, window, depth_obs_key=depth_obs_key)
                provenance[shift_condition(seconds)] = {"global_idx": int(donor_idx), "lag_frames": int(lag)}
            donor = donors.get(row["global_idx"])
            if donor is not None:
                window = _load_depth_window(dataset, cfg, donor["global_idx"], obs, depth_obs_key=depth_obs_key)
                condition_obs[CROSS_PHASE] = _replace_depth_window(obs, window, depth_obs_key=depth_obs_key)
                provenance[CROSS_PHASE] = donor

            chunks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            probs: dict[str, torch.Tensor | None] = {}
            for condition in conditions:
                if condition not in condition_obs:
                    continue
                unnorm, norm, prob = predict(condition_obs[condition], frame, noise_seed0)
                chunks[condition] = (unnorm[0], norm[0])
                probs[condition] = None if prob is None else prob[0]
            _, floor_norm, _ = predict(obs, frame, floor_noise)

            _, deployment_norm = chunks[DEPLOYMENT]
            record = {
                **row,
                "g_now": g_now,
                "provenance": provenance,
                "displacement": {},
                "gripper": {},
                "head": {},
                "seed_floor": {
                    key: float(np.mean([_relative(floor_norm[i], deployment_norm, hold_norm)[key] for i in range(floor_norm.shape[0])]))
                    for key in RELATIVE_KEYS
                },
                "targets": {
                    head: (float(frame[key]) if frame.get(key) is not None else None)
                    for head, key in TARGET_KEYS.items()
                },
            }
            for condition, (unnorm, norm) in chunks.items():
                gripper = unnorm[:, gripper_dim].numpy()
                close_step, open_step = gripper_transitions(
                    gripper,
                    row["closed_now"],
                    close_threshold=labels["close_threshold"],
                    open_threshold=labels["open_threshold"],
                )
                record["gripper"][condition] = {
                    "terminal_delta": float(gripper[-1] - g_now),
                    "mean_delta": float(gripper.mean() - g_now),
                    "close_step": close_step,
                    "open_step": open_step,
                }
                prob = probs[condition]
                record["head"][condition] = (
                    None if prob is None else {"p_close": float(prob[0]), "p_open": float(prob[1])}
                )
                if condition != DEPLOYMENT:
                    record["displacement"][condition] = _relative(norm, deployment_norm, hold_norm)
            per_frame.append(record)

            shown = " ".join(
                f"{condition}={record['displacement'][condition]['path_relative']:.3f}"
                for condition in conditions[1:]
                if condition in record["displacement"]
            )
            logging.info(
                f"[depth_event] {row['stratum']} ep{row['episode_idx']} fr{row['frame_idx']}: "
                f"floor={record['seed_floor']['path_relative']:.3f} {shown} "
                f"g_T-g_now={record['gripper'][DEPLOYMENT]['terminal_delta']:+.1f}"
            )
    finally:
        adapter._restore_probe_cuda_graph_enabled()

    summary = {
        "n_frames": len(per_frame),
        "n_events": n_events,
        "episodes": episodes,
        "leads_s": leads_s,
        "shifts_s": shifts_s,
        "z_offset_mm": z_offset_mm,
        "n_seeds": n_seeds,
        "rubric": labels["rubric"],
        "gripper_dim": gripper_dim,
        "conditions": conditions,
        "condition_kind": {
            DEPLOYMENT: "deployment",
            **{condition: "counterfactual" for condition in conditions[1:] if condition != NO_DEPTH},
            NO_DEPTH: "untrained shape",
        },
        "headline_condition": conditions[1],
        "strata": strata,
        **summarize(per_frame, strata, conditions),
    }

    with open(os.path.join(output_dir, "depth_event.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)
    # write_index drops panels whose file is not on disk yet, so render first.
    _render(summary, os.path.join(output_dir, "depth_event.png"))
    _write_manifest(output_dir, summary)

    logging.info("── depth_event: pre-close over free, per condition ──")
    for condition, entry in summary["contrast"].items():
        pre, free = entry["pre_close"], entry["free"]
        ratio = entry["ratio"]
        logging.info(
            f"{condition:>12s}: pre-close {pre['path_relative'] if pre['path_relative'] is not None else float('nan'):.3f} "
            f"(x{pre['over_floor'] if pre['over_floor'] is not None else float('nan'):.1f} floor, n={pre['n']})  "
            f"free {free['path_relative'] if free['path_relative'] is not None else float('nan'):.3f} "
            f"(x{free['over_floor'] if free['over_floor'] is not None else float('nan'):.1f} floor, n={free['n']})  "
            f"ratio {ratio if ratio is not None else float('nan'):.2f}  "
            f"gripper shift {pre['terminal_delta_shift_deg'] if pre['terminal_delta_shift_deg'] is not None else float('nan'):+.2f} deg  "
            f"p_close drop {pre['p_close_drop'] if pre['p_close_drop'] is not None else float('nan'):+.3f}"
        )
    logging.info(f"wrote {os.path.join(output_dir, 'depth_event.json')} and .png")
    return summary


@parser.wrap()
def cli(cfg: TrainRLServerPipelineConfig):
    init_logging()
    if getattr(cfg.policy, "pointmap_config", None) is None:
        raise SystemExit("policy.pointmap_config is null in this config — nothing to probe.")
    val_path = getattr(cfg, "val_dataset_path", None)
    if not val_path:
        raise SystemExit("val_dataset_path is unset — the probe measures held-out episodes only.")
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = load_probe_dataset(cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    val_dataset = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=val_path)
    val_dataset.delta_timestamps = None
    val_dataset.delta_indices = None
    run(adapter, val_dataset, cfg, os.path.join(cfg.probe_parameters.output_dir, "depth_event"))


def main() -> None:
    # Same pre-parse machinery as rl_offline: register policy configs, strip inactive-model YAML fields.
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
