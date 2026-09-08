"""Depth read at gripper events: does the wrist depth move the gripper command where a
grasp or a release is about to happen, and nowhere else?

The matched-depth counterfactual (``depth_modality_probe``) samples frames evenly and
scores the whole chunk over every joint, so it averages the depth read over carrying,
retreating and waiting, and a gripper-only change is diluted by the arm. Depth's job in
this task is the last stretch of an approach and the timing of the gripper, so this probe
picks its frames by the commanded gripper and reads only the gripper.

Frame selection
---------------
``meta/depth_gripper_events.parquet`` and the dense ``depth_gripper_event_labels.parquet``
deltas are the locked rubric the auxiliary head is trained on: commanded ``gripper.pos``,
closed once above -60 deg, open once below -90 deg, 0.5 s persistence. Nothing here reads
observation.state or a semantic annotation to choose a frame. Every anchor is snapped onto
the image/depth stride grid.

  pre_close_{L}s   L seconds before a commanded close, and the labels agree that close is
                   the next one
  pre_open_{L}s    the same before a commanded open
  carry            gripper closed, no open within max(L)+1 s, at least 1 s after the close
  free             gripper open, no close within max(L)+1 s, at least 1 s after the open

Controls get two frames per event of the commoner type in each episode, spread evenly
over that episode's candidates.

Conditions
----------
Each condition replaces the wrist depth window (current frame plus history slots) and
leaves the top camera, the wrist RGB, the state and the prompt untouched. Chunks are
decoded with the flow decoder under one fixed noise draw, so two conditions differ only
through depth.

  deployment    the rollout prompt as is
  z_offset      every valid depth pixel (reading > 0) gets +dz mm; zeros stay zero. The
                back-projection scales X and Y with Z, so the whole scene sits dz farther
                from the wrist camera. Headline condition: same frame, only the metric
                scale moves, and the sign is known in advance.
  shift_{D}s    the depth window from D s earlier in the same episode, RGB left at t. On an
                approach the object reads farther than RGB shows.
  cross_phase   the depth window of a same-stratum frame from another episode, nearest by
                standardized joint state: same phase, different scene. The paired null:
                depth that reads the same distance should give the same gripper.
  no_depth      window removed (learned null bank). Depth dropout is 0 in training, so this
                is an untrained input shape; kept as the reference the older probe reports.

Readouts, all on the gripper dimension in the dataset's own degrees
-------------------------------------------------------------------
  g_c[t]          gripper command at chunk step t under condition c, t = 1..T (T = 30 = 1 s)
  g_now           the dataset's commanded gripper at the frame
  terminal delta  g_c[T] - g_now: how far the chunk itself closes (positive) or opens by its
                  last step
  terminal shift  g_c[T] - g_dep[T]: the treated chunk's last gripper command minus the
                  deployment chunk's. g_now cancels. Negative = ends less closed.
  mean shift      (1/T) * sum_t (g_c[t] - g_dep[t]), the same over the whole chunk
  less closed     terminal shift < -5 deg; more closed: > +5 deg; moved: either. On a close
                  approach "less closed" is the chunk delaying or cancelling the close; on an
                  open approach "more closed" is the same for the release. Fractions of
                  frames, so one runaway chunk cannot carry a mean.
  fire step       the first t at which the labels' hysteresis rule (close: g > -60; open:
                  g < -90) trips from the frame's own state; None when the chunk never gets
                  there. Observable only when the event lands inside the chunk (lead < 1 s).
                  step shift = fire step treated - fire step deployment, on frames where
                  both fire.
  p_close, p_open sigmoid of the depth-only auxiliary head's two logits,
                  head(mean over the depth tokens of LayerNorm(token)), read off the same
                  forward. The head sees only the depth tokens, so this is the depth path's
                  own opinion of the event, before the action expert.
  target          2^(-lead / 1 s), the label the head was trained toward at that lead
                  (1 s half-life, zero past 5 s)

What a working depth read looks like: under z_offset the pre-close fraction "less closed"
rises toward the event and sits at the control rate on free and carry; cross_phase stays
at the control rate everywhere; p_close under deployment rises toward the event along the
target curve and drops under z_offset. A shift that is flat across strata means depth acts
as a global bias, not as a grasp cue.

Outputs under ``<output_dir>/``:
  depth_event.json   summary, per-stratum tables, per-frame rows with donor provenance and
                     the full gripper trajectory per condition
  depth_event.png    close row / open row: head p vs lead, terminal shift vs lead, fraction
                     less (more) closed vs lead, mean gripper trajectory at two leads

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
TARGET_KEYS = {"close": DEPTH_GRIPPER_CLOSE_TARGET, "open": DEPTH_GRIPPER_OPEN_TARGET}
DEFAULT_LEADS_S = "0.25,0.5,0.75,1,1.5,2,3,4"
# A terminal shift past this counts as the chunk having moved; the rig's gripper swings
# 100-200 deg between open and closed.
SHIFT_THRESHOLD_DEG = 5.0
# The chunk is 1 s, so this lead puts the commanded event at the chunk's last step.
HEADLINE_LEAD_S = 1.0
# The trajectory panel draws these two leads (nearest configured).
TRAJECTORY_LEADS_S = (0.5, 1.0)


def shift_condition(seconds: float) -> str:
    return f"shift_{seconds:g}s"


def event_stratum(event_type: str, lead_s: float) -> str:
    return f"pre_{event_type}_{lead_s:g}s"


def _seconds(text: str) -> list[float]:
    return [float(s) for s in str(text).split(",") if s.strip()]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "sem": None, "median": None, "q25": None, "q75": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    sem = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else None
    return {
        "mean": float(array.mean()),
        "sem": sem,
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "n": int(array.size),
    }


def _fraction(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


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
    """Event-relative anchors plus far-from-event controls, per episode.

    A pre-event anchor is kept only when the labels agree that the event it was placed
    before is the next one of its type (``delta == event - frame``): a nearer event, or a
    lead beyond the labels' 5 s cutoff, drops it. Controls need no event within
    ``max(lead) + 1`` s and a 1 s settle after the last transition, so the carry and free
    strata never overlap a pre-event window; each gets two frames per event of the
    episode's commoner type.
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
                        "lead_s": lead_s,
                        "lead_frames": event_frame - frame,
                        "episode_idx": int(episode_idx),
                        "frame_idx": int(frame),
                        "global_idx": int(global_idx),
                        "closed_now": containing(frame) is not None,
                    }
                )

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
        n_events = Counter(str(t) for t in episode_events["event_type"])
        n_control = 2 * max(n_events.values(), default=0)
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
                        "lead_s": None,
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


# ──────────────────────────────────────────────────────────────────────────────
# Summary, figure, manifest
# ──────────────────────────────────────────────────────────────────────────────


def summarize(
    per_frame: list[dict], strata: list[str], conditions: list[str], *, leads_s: list[float], fps: int,
    headline_condition: str,
) -> dict:
    by_stratum: dict[str, dict] = {}
    for name in strata:
        rows = [r for r in per_frame if r["stratum"] == name]
        leads = [r["lead_frames"] / fps for r in rows if r["lead_frames"] is not None]
        entry = {
            "n": len(rows),
            "n_by_episode": dict(sorted(Counter(r["episode_idx"] for r in rows).items())),
            "event_type": rows[0]["event_type"] if rows else None,
            "lead_s": rows[0]["lead_s"] if rows else None,
            "lead_s_actual": float(np.mean(leads)) if leads else None,
            "target_close": _stats([r["targets"]["close"] for r in rows if r["targets"]["close"] is not None])["mean"],
            "target_open": _stats([r["targets"]["open"] for r in rows if r["targets"]["open"] is not None])["mean"],
            "conditions": {},
        }
        for condition in conditions:
            present = [r for r in rows if condition in r["conditions"]]
            own = [r["conditions"][condition] for r in present]
            cond: dict = {"n": len(present)}
            for head in EVENT_TYPES:
                cond[f"p_{head}"] = _stats([o[f"p_{head}"] for o in own if o[f"p_{head}"] is not None])
            cond["terminal_delta_deg"] = _stats([o["terminal_delta"] for o in own])
            cond["mean_delta_deg"] = _stats([o["mean_delta"] for o in own])
            for transition in EVENT_TYPES:
                steps = [o[f"{transition}_step"] for o in own]
                fired = [s for s in steps if s is not None]
                cond[f"{transition}_fire_fraction"] = float(len(fired) / len(steps)) if steps else None
                cond[f"{transition}_step_mean"] = float(np.mean(fired)) if fired else None
            cond["trajectory_mean"] = (
                np.mean([np.asarray(o["gripper"]) - r["g_now"] for r, o in zip(present, own)], axis=0).tolist()
                if own
                else None
            )
            if condition != DEPLOYMENT:
                pairs = [(r["conditions"][DEPLOYMENT], o) for r, o in zip(present, own)]
                shifts = [o["gripper"][-1] - d["gripper"][-1] for d, o in pairs]
                cond["terminal_shift_deg"] = _stats(shifts)
                cond["mean_shift_deg"] = _stats(
                    [float(np.mean(o["gripper"]) - np.mean(d["gripper"])) for d, o in pairs]
                )
                cond["fraction_less_closed"] = _fraction([s < -SHIFT_THRESHOLD_DEG for s in shifts])
                cond["fraction_more_closed"] = _fraction([s > SHIFT_THRESHOLD_DEG for s in shifts])
                cond["fraction_moved"] = _fraction([abs(s) > SHIFT_THRESHOLD_DEG for s in shifts])
                for transition in EVENT_TYPES:
                    key = f"{transition}_step"
                    cond[f"{transition}_step_shift"] = _stats(
                        [o[key] - d[key] for d, o in pairs if d[key] is not None and o[key] is not None]
                    )
                for head in EVENT_TYPES:
                    key = f"p_{head}"
                    cond[f"{key}_drop"] = _stats(
                        [d[key] - o[key] for d, o in pairs if d[key] is not None and o[key] is not None]
                    )
            entry["conditions"][condition] = cond
        by_stratum[name] = entry

    def node(name: str, condition: str) -> dict:
        return by_stratum.get(name, {}).get("conditions", {}).get(condition, {})

    lead = min(leads_s, key=lambda value: abs(value - HEADLINE_LEAD_S))
    headline: dict = {"condition": headline_condition, "lead_s": lead}
    for event_type in EVENT_TYPES:
        name = event_stratum(event_type, lead)
        treated, deployment = node(name, headline_condition), node(name, DEPLOYMENT)
        headline[f"pre_{event_type}"] = {
            "stratum": name,
            "n": treated.get("n", 0),
            "terminal_shift_median_deg": (treated.get("terminal_shift_deg") or {}).get("median"),
            "terminal_shift_mean_deg": (treated.get("terminal_shift_deg") or {}).get("mean"),
            "fraction_less_closed": treated.get("fraction_less_closed"),
            "fraction_more_closed": treated.get("fraction_more_closed"),
            f"p_{event_type}": (deployment.get(f"p_{event_type}") or {}).get("mean"),
            f"p_{event_type}_target": by_stratum.get(name, {}).get(f"target_{event_type}"),
            f"p_{event_type}_drop": (treated.get(f"p_{event_type}_drop") or {}).get("mean"),
        }
    for name in (FREE, CARRY):
        treated = node(name, headline_condition)
        headline[name] = {
            "n": treated.get("n", 0),
            "terminal_shift_median_deg": (treated.get("terminal_shift_deg") or {}).get("median"),
            "fraction_moved": treated.get("fraction_moved"),
        }
    return {"by_stratum": by_stratum, "headline": headline}


def _render(summary: dict, output_path: str) -> None:
    conditions = summary["conditions"]
    treatments = conditions[1:]
    by_stratum = summary["by_stratum"]
    leads = summary["leads_s"]
    fps = summary["fps"]
    chunk_size = summary["chunk_size"]
    headline = summary["headline_condition"]
    palette = ["#E76F51", "#F4A261", "#2A9D8F", "#457B9D", "#9D4EDD", "#6C757D"]
    colors = {condition: palette[i % len(palette)] for i, condition in enumerate(treatments)}
    colors[DEPLOYMENT] = "#264653"
    line_style = {condition: {"ls": "-", "alpha": 1.0} for condition in conditions}
    line_style[NO_DEPTH] = {"ls": ":", "alpha": 0.6}
    line_style[DEPLOYMENT] = {"ls": "-", "alpha": 1.0, "lw": 2.0}

    def value(name: str, condition: str, *path: str) -> float:
        node = by_stratum.get(name, {}).get("conditions", {}).get(condition, {})
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        return np.nan if node is None else float(node)

    def per_lead(event_type: str, condition: str, *path: str) -> np.ndarray:
        return np.asarray([value(event_stratum(event_type, lead), condition, *path) for lead in leads])

    def lead_axis(ax, event_type: str) -> None:
        ax.set_xscale("log")
        ax.set_xticks(leads)
        ax.set_xticklabels([f"{lead:g}" for lead in leads], fontsize=8)
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.set_xlabel(f"seconds before the commanded {event_type}")

    fig, axes = plt.subplots(2, 4, figsize=(23, 9.5))
    for row, event_type in enumerate(EVENT_TYPES):
        p_key = f"p_{event_type}"
        fraction_key = "fraction_less_closed" if event_type == "close" else "fraction_more_closed"
        against = "less closed" if event_type == "close" else "more closed"
        delayed = "negative = ends less closed, close delayed" if event_type == "close" else "positive = ends more closed, release delayed"

        ax = axes[row, 0]
        target = [np.nan if (t := by_stratum.get(event_stratum(event_type, lead), {}).get(f"target_{event_type}")) is None else t for lead in leads]
        ax.plot(leads, target, color="black", ls="--", lw=1.0, label="label target 2^(-lead)")
        for condition in conditions:
            ax.errorbar(
                leads, per_lead(event_type, condition, p_key, "mean"), yerr=per_lead(event_type, condition, p_key, "sem"),
                color=colors[condition], marker="o", ms=3, capsize=2, label=condition, **line_style[condition],
            )
        for name, shade in ((CARRY, "#8D99AE"), (FREE, "#CBD5E1")):
            mean, sem = value(name, DEPLOYMENT, p_key, "mean"), value(name, DEPLOYMENT, p_key, "sem")
            if not np.isnan(mean):
                sem = 0.0 if np.isnan(sem) else sem
                ax.axhspan(mean - sem, mean + sem, color=shade, alpha=0.6, label=f"{name}, deployment")
        lead_axis(ax, event_type)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"head p_{event_type}")
        ax.set_title(f"Depth-only head: p_{event_type} approaching a {event_type}", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        for condition in treatments:
            median = per_lead(event_type, condition, "terminal_shift_deg", "median")
            ax.plot(leads, median, color=colors[condition], marker="o", ms=3, label=condition, **line_style[condition])
            ax.fill_between(
                leads, per_lead(event_type, condition, "terminal_shift_deg", "q25"),
                per_lead(event_type, condition, "terminal_shift_deg", "q75"), color=colors[condition], alpha=0.10,
            )
        ax.plot(
            leads, per_lead(event_type, DEPLOYMENT, "terminal_delta_deg", "median"), color="black", ls=":",
            marker="_", ms=10, label="deployment g[T] - g_now",
        )
        ax.axhline(0.0, color="#333333", lw=0.8)
        lead_axis(ax, event_type)
        ax.set_ylabel("last-step gripper shift vs deployment (deg)")
        ax.set_title(f"g_c[T] - g_dep[T], median with IQR; {delayed}", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[row, 2]
        for condition in treatments:
            ax.plot(leads, per_lead(event_type, condition, fraction_key), color=colors[condition], marker="o", ms=3, label=condition, **line_style[condition])
        control = value(FREE, headline, "fraction_moved")
        if not np.isnan(control):
            ax.axhline(control, color="#6C757D", ls="--", lw=1.0, label=f"free, moved either way ({headline})")
        lead_axis(ax, event_type)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"fraction of chunks ending >{SHIFT_THRESHOLD_DEG:g} deg {against}")
        ax.set_title(f"How often the treated chunk ends {against} than deployment", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[row, 3]
        picks = sorted({min(leads, key=lambda lead, want=want: abs(lead - want)) for want in TRAJECTORY_LEADS_S})
        shades = ["#F4A261", "#264653"] if len(picks) > 1 else ["#264653"]
        steps = np.arange(1, chunk_size + 1) / fps
        for lead, shade in zip(picks, shades):
            name = event_stratum(event_type, lead)
            for condition, ls in ((DEPLOYMENT, "-"), (headline, "--")):
                trajectory = by_stratum.get(name, {}).get("conditions", {}).get(condition, {}).get("trajectory_mean")
                if trajectory is not None:
                    ax.plot(steps, trajectory, color=shade, ls=ls, label=f"{condition}, {lead:g} s before")
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_xlabel("chunk step (s)")
        ax.set_ylabel("mean g[t] - g_now (deg)")
        ax.set_title(f"Gripper trajectory over the chunk, deployment vs {headline}", fontsize=10)
        ax.legend(fontsize=7)

    head = summary["headline"]
    pre_close, free = head["pre_close"], head[FREE]
    fig.suptitle(
        f"Depth at gripper events — n={summary['n_frames']} frames, "
        f"{summary['n_events']['close']} closes / {summary['n_events']['open']} opens  |  "
        f"{headline}, {head['lead_s']:g} s before a close: last-step gripper shift median "
        f"{pre_close['terminal_shift_median_deg'] if pre_close['terminal_shift_median_deg'] is not None else float('nan'):+.1f} deg, "
        f"{(pre_close['fraction_less_closed'] or 0.0):.0%} of chunks less closed (free moved {(free['fraction_moved'] or 0.0):.0%})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _write_manifest(output_dir: str, summary: dict) -> dict:
    headline = summary["headline_condition"]
    head = summary["headline"]
    lead = head["lead_s"]
    conditions = summary["conditions"]
    close_name, open_name = head["pre_close"]["stratum"], head["pre_open"]["stratum"]

    def condition_metrics(condition: str) -> list[Metric]:
        tag = " (untrained shape)" if condition == NO_DEPTH else ""
        prefix_close = f"by_stratum.{close_name}.conditions.{condition}"
        prefix_open = f"by_stratum.{open_name}.conditions.{condition}"
        return [
            Metric(
                f"{prefix_close}.terminal_shift_deg.median",
                f"{condition}: pre-close {lead:g} s last-step gripper shift, median (deg){tag}",
                good="none", fmt=1, baseline=0.0,
            ),
            Metric(
                f"{prefix_close}.fraction_less_closed",
                f"{condition}: pre-close {lead:g} s chunks ending less closed{tag}",
                good="none", fmt=2, baseline=0.0,
            ),
            Metric(
                f"{prefix_close}.p_close_drop.mean",
                f"{condition}: pre-close {lead:g} s head p_close drop{tag}",
                good="none", fmt=3, baseline=0.0,
            ),
            Metric(
                f"{prefix_open}.terminal_shift_deg.median",
                f"{condition}: pre-open {lead:g} s last-step gripper shift, median (deg){tag}",
                good="none", fmt=1, baseline=0.0,
            ),
            Metric(
                f"{prefix_open}.fraction_more_closed",
                f"{condition}: pre-open {lead:g} s chunks ending more closed{tag}",
                good="none", fmt=2, baseline=0.0,
            ),
            Metric(
                f"by_stratum.{FREE}.conditions.{condition}.fraction_moved",
                f"{condition}: free chunks moved either way{tag}",
                good="none", fmt=2, baseline=0.0,
            ),
        ]

    metrics = [
        Metric(
            "headline.pre_close.terminal_shift_median_deg",
            f"{headline}: pre-close {lead:g} s last-step gripper shift, median (deg)",
            good="none", fmt=1, baseline=0.0, primary=True, trend=True,
            note=(
                "g_c[T] - g_dep[T]: the treated chunk's last gripper command minus the deployment "
                "chunk's, in the dataset's degrees (closed near 0, open negative), on frames "
                f"{lead:g} s before a commanded close. Negative means the chunk ends less closed "
                "when the scene reads farther. Median over frames."
            ),
        ),
        Metric(
            "headline.pre_close.fraction_less_closed",
            f"{headline}: pre-close {lead:g} s chunks ending less closed",
            good="none", fmt=2, baseline=0.0, primary=True, trend=True,
            note=(
                f"Fraction of pre-close frames whose treated chunk ends more than {SHIFT_THRESHOLD_DEG:g} deg "
                "less closed than the deployment chunk. Read against the free rate below: the "
                "difference is the grasp cue."
            ),
        ),
        Metric(
            "headline.free.fraction_moved",
            f"{headline}: free chunks moved either way",
            good="none", fmt=2, baseline=0.0, primary=True,
            note=(
                f"Fraction of free frames (open gripper, no close within reach) whose last gripper "
                f"command moved more than {SHIFT_THRESHOLD_DEG:g} deg in either direction under the same "
                "treatment. The null rate."
            ),
        ),
        Metric(
            "headline.pre_close.p_close",
            f"deployment head p_close {lead:g} s before a close",
            good="none", fmt=3, baseline=head["pre_close"]["p_close_target"], primary=True, trend=True,
            note=(
                "Sigmoid of the depth-only head's close logit under the real depth, mean over "
                "pre-close frames. The baseline is the label target 2^(-lead) at this lead."
            ),
        ),
        Metric(
            "headline.pre_close.p_close_drop",
            f"{headline}: pre-close {lead:g} s head p_close drop",
            good="none", fmt=3, baseline=0.0, primary=True,
            note="p_close(deployment) - p_close(treated). Positive means the head reads the treated depth as farther from a close.",
        ),
        Metric(
            "headline.pre_open.terminal_shift_median_deg",
            f"{headline}: pre-open {lead:g} s last-step gripper shift, median (deg)",
            good="none", fmt=1, baseline=0.0, primary=True,
            note="The same shift on frames before a commanded open. Positive means the chunk ends more closed, the release delayed.",
        ),
        Metric(
            "headline.pre_open.fraction_more_closed",
            f"{headline}: pre-open {lead:g} s chunks ending more closed",
            good="none", fmt=2, baseline=0.0, primary=True,
        ),
        Metric(
            "headline.pre_open.p_open",
            f"deployment head p_open {lead:g} s before an open",
            good="none", fmt=3, baseline=head["pre_open"]["p_open_target"], primary=True, trend=True,
        ),
        *[m for condition in conditions[1:] if condition != headline for m in condition_metrics(condition)],
        Metric("n_frames", "Frames probed", good="none", fmt=0),
        Metric("n_events.close", "Close events in the probed episodes", good="none", fmt=0),
        Metric("n_events.open", "Open events in the probed episodes", good="none", fmt=0),
    ]
    return write_index(
        output_dir,
        sys.modules[__name__],
        title="Depth at Gripper Events",
        group="Depth",
        claim="Does the wrist depth move the gripper command where a grasp or release is imminent, and nowhere else?",
        summary=summary,
        metrics=metrics,
        panels=[
            Panel(
                "depth_event.png",
                "Top row closes, bottom row opens; x is seconds before the commanded event, the event at the right edge",
                how=(
                    "**Column 1** — the depth-only head's probability under each condition, with the "
                    "label target $2^{-\\text{lead}}$ dashed and the carry / free deployment levels as "
                    "bands. A head that sees the approach rises along the target as the event nears "
                    "and drops under ``z_offset``.\n\n"
                    "**Column 2** — last-step gripper shift $g_c[T] - g_{dep}[T]$ in degrees, median "
                    "with the interquartile band; the dotted black line is deployment's own "
                    "$g[T] - g_{now}$, how far the untouched chunk closes by its end. Below zero on "
                    "the close row means the chunk ends less closed when the scene reads farther.\n\n"
                    "**Column 3** — the fraction of frames whose chunk ends more than "
                    f"{SHIFT_THRESHOLD_DEG:g} deg less closed (close row) or more closed (open row) "
                    "than deployment; the grey dashed line is the same treatment's rate on free "
                    "frames, moved either way. The gap between a line and the grey is the grasp cue; "
                    "``cross_phase`` should sit on the grey everywhere.\n\n"
                    "**Column 4** — the mean gripper trajectory $g[t] - g_{now}$ over the 30 chunk "
                    "steps at two leads, deployment solid against the headline condition dashed, so "
                    "the delay is seen rather than summarised."
                ),
                primary=True,
            ),
            Panel(
                "depth_event.json",
                "Per-stratum tables and per-frame rows",
                how=(
                    "Each row records the stratum, the event and lead it was placed before, donor "
                    "provenance per condition, and per condition the full gripper trajectory, its "
                    "terminal and mean delta from g_now, the first-crossing steps and the head "
                    "probabilities."
                ),
            ),
        ],
        see_also=["depth_modality", "objective"],
        extra={
            "viewer": {
                "metric_groups": [
                    {
                        "title": f"{headline} (headline)",
                        "keys": [
                            "headline.pre_close.terminal_shift_median_deg",
                            "headline.pre_close.fraction_less_closed",
                            "headline.free.fraction_moved",
                            "headline.pre_close.p_close",
                            "headline.pre_close.p_close_drop",
                            "headline.pre_open.terminal_shift_median_deg",
                            "headline.pre_open.fraction_more_closed",
                            "headline.pre_open.p_open",
                        ],
                    },
                    *[
                        {
                            "title": condition,
                            "keys": [
                                f"by_stratum.{close_name}.conditions.{condition}.terminal_shift_deg.median",
                                f"by_stratum.{close_name}.conditions.{condition}.fraction_less_closed",
                                f"by_stratum.{close_name}.conditions.{condition}.p_close_drop.mean",
                                f"by_stratum.{open_name}.conditions.{condition}.terminal_shift_deg.median",
                                f"by_stratum.{open_name}.conditions.{condition}.fraction_more_closed",
                                f"by_stratum.{FREE}.conditions.{condition}.fraction_moved",
                            ],
                        }
                        for condition in conditions[1:]
                        if condition != headline
                    ],
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

    leads_s = _seconds(getattr(p, "depth_event_leads_s", DEFAULT_LEADS_S))
    shifts_s = _seconds(getattr(p, "depth_event_shift_s", "1.0,2.0"))
    z_offset_mm = float(getattr(p, "depth_event_z_offset_mm", 30.0))
    z_offset_levels = z_offset_mm / float(pointmap_config.depth_units_mm)
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
        logging.info(f"[depth_event] {name:>16s}: {sum(by_episode.values())} frames {dict(sorted(by_episode.items()))}")
    logging.info(
        f"[depth_event] {len(rows)} frames x {len(conditions)} conditions forwards; "
        f"cross-phase donors for {len(donors)} frames"
    )

    adapter._set_probe_cuda_graph_enabled(False)
    noise = adapter.flow_noise_like(1, 0)

    def predict(obs: dict, frame: dict):
        unnorm, _ = adapter.predict_action_chunk_batch(
            obs,
            frame["task"],
            [frame["subtask"]],
            metadatas=[frame["metadata"]],
            noise=noise,
            inference_action_mode="continuous",
        )
        logits = policy._depth_gripper_event_logits
        probs = None if logits is None else torch.sigmoid(logits.detach().float()).cpu()[0]
        return unnorm[0], probs

    per_frame: list[dict] = []
    try:
        for row in rows:
            frame = probe_frame_inputs(dataset, cfg, row["global_idx"], chunk_size)
            obs = frame["obs"]
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

            record = {
                **row,
                "g_now": g_now,
                "provenance": provenance,
                "targets": {
                    head: (float(frame[key]) if frame.get(key) is not None else None)
                    for head, key in TARGET_KEYS.items()
                },
                "conditions": {},
            }
            for condition in conditions:
                if condition not in condition_obs:
                    continue
                unnorm, prob = predict(condition_obs[condition], frame)
                gripper = unnorm[:, gripper_dim].numpy()
                close_step, open_step = gripper_transitions(
                    gripper,
                    row["closed_now"],
                    close_threshold=labels["close_threshold"],
                    open_threshold=labels["open_threshold"],
                )
                record["conditions"][condition] = {
                    "gripper": [round(float(g), 2) for g in gripper],
                    "terminal_delta": float(gripper[-1] - g_now),
                    "mean_delta": float(gripper.mean() - g_now),
                    "close_step": close_step,
                    "open_step": open_step,
                    "p_close": None if prob is None else float(prob[0]),
                    "p_open": None if prob is None else float(prob[1]),
                }
            per_frame.append(record)

            deployment = record["conditions"][DEPLOYMENT]
            shown = " ".join(
                f"{condition}={record['conditions'][condition]['gripper'][-1] - deployment['gripper'][-1]:+.1f}"
                for condition in conditions[1:]
                if condition in record["conditions"]
            )
            logging.info(
                f"[depth_event] {row['stratum']} ep{row['episode_idx']} fr{row['frame_idx']}: "
                f"g_T-g_now={deployment['terminal_delta']:+.1f} shift {shown} "
                f"p_close={deployment['p_close'] if deployment['p_close'] is not None else float('nan'):.3f}"
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
        "shift_threshold_deg": SHIFT_THRESHOLD_DEG,
        "fps": fps,
        "chunk_size": chunk_size,
        "rubric": labels["rubric"],
        "gripper_dim": gripper_dim,
        "conditions": conditions,
        "condition_kind": {
            DEPLOYMENT: "deployment",
            **{condition: "counterfactual" for condition in conditions[1:] if condition != NO_DEPTH},
            NO_DEPTH: "untrained shape",
        },
        "headline_condition": Z_OFFSET,
        "strata": strata,
        **summarize(per_frame, strata, conditions, leads_s=leads_s, fps=fps, headline_condition=Z_OFFSET),
    }

    with open(os.path.join(output_dir, "depth_event.json"), "w") as f:
        json.dump({"summary": summary, "per_frame": per_frame}, f, indent=2)
    # write_index drops panels whose file is not on disk yet, so render first.
    _render(summary, os.path.join(output_dir, "depth_event.png"))
    _write_manifest(output_dir, summary)

    head = summary["headline"]
    logging.info(f"── depth_event: {head['condition']} at {head['lead_s']:g} s before the event ──")
    for name in ("pre_close", "pre_open", FREE, CARRY):
        entry = head[name]
        fields = " ".join(
            f"{key}={value:+.3f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in entry.items()
            if key != "stratum"
        )
        logging.info(f"{name:>10s}: {fields}")
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
